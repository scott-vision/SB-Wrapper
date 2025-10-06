from __future__ import annotations

import threading
from typing import List

import pytest

from sbwrapper import MicroscopeConnection
import sbwrapper.connection as connection_module


class DummySocket:
    def __init__(self) -> None:
        self.timeout = None
        self.shutdown_called = False
        self.closed = False
        self.sent: List[bytes] = []

    def settimeout(self, timeout):
        self.timeout = timeout

    def shutdown(self, how):  # pragma: no cover - platform dependent
        self.shutdown_called = True

    def close(self):
        self.closed = True

    def sendall(self, data: bytes):
        self.sent.append(data)


class FakeSBAccess:
    def __init__(self, sock):
        self.sock = sock


@pytest.fixture(autouse=True)
def patch_sbaccess(monkeypatch):
    monkeypatch.setattr(connection_module, "SBAccess", FakeSBAccess)
    yield


def test_context_manager_handles_connection_lifecycle():
    socket_instance = DummySocket()

    def socket_factory(address, timeout):
        assert address == ("localhost", 1234)
        assert timeout == 1.0
        return socket_instance

    conn = MicroscopeConnection(
        "localhost",
        1234,
        connect_timeout=1.0,
        read_timeout=2.0,
        socket_factory=socket_factory,
    )

    with conn as client:
        assert client.sock is socket_instance
        assert socket_instance.timeout == 2.0
    assert socket_instance.closed is True
    assert conn.client is None


def test_reconnect_callback_invoked_on_drop():
    sockets = [DummySocket(), DummySocket()]

    def socket_factory(address, timeout):
        return sockets.pop(0)

    reconnection_event = threading.Event()

    def on_reconnect(client):
        assert isinstance(client, FakeSBAccess)
        reconnection_event.set()

    conn = MicroscopeConnection(
        "localhost",
        1234,
        reconnect_attempts=2,
        reconnect_delay=0.01,
        socket_factory=socket_factory,
        on_reconnect=on_reconnect,
    )

    first_client = conn.connect()
    assert isinstance(first_client, FakeSBAccess)
    # Simulate a connection drop from a worker.
    conn._handle_connection_drop(RuntimeError("boom"))

    assert reconnection_event.wait(1.0), "Reconnection callback not triggered"
    assert conn.client is not None
    conn.close()


def test_keep_alive_sends_message_until_closed():
    socket_instance = DummySocket()
    send_event = threading.Event()

    def socket_factory(address, timeout):
        return socket_instance

    original_sendall = socket_instance.sendall

    def sendall(data):
        original_sendall(data)
        send_event.set()

    socket_instance.sendall = sendall

    conn = MicroscopeConnection(
        "localhost",
        1234,
        socket_factory=socket_factory,
        keep_alive_interval=0.05,
        keep_alive_message=b"ping",
    )

    conn.connect()
    assert send_event.wait(1.0), "Keep-alive message was not sent"
    conn.close()
    assert socket_instance.closed is True
