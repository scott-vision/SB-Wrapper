"""Connection management utilities for SBAccess clients."""
from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Callable, Optional

from .sb_access import SBAccess

LOGGER = logging.getLogger(__name__)


class MicroscopeConnection:
    """High level connection helper for :class:`~sbwrapper.sb_access.SBAccess`.

    Parameters
    ----------
    host:
        Hostname or IP address of the SB server.
    port:
        TCP port of the SB server.
    connect_timeout:
        Socket creation timeout in seconds passed to
        :func:`socket.create_connection`.
    read_timeout:
        Optional per-operation timeout in seconds applied to the socket
        after it is created.
    reconnect_attempts:
        How many times to attempt reconnecting before failing. ``1`` means
        a single attempt without retries.
    reconnect_delay:
        Delay in seconds between reconnect attempts.
    keep_alive_interval:
        If provided, enables a keep-alive worker that periodically invokes
        ``keep_alive_callback`` or sends ``keep_alive_message``.
    keep_alive_message:
        Raw bytes (or UTF-8 string) to transmit for keep-alive when no
        callback is supplied.
    keep_alive_callback:
        Optional callable ``callback(client)`` executed for keep-alive
        checks. Exceptions raised by the callback trigger reconnection
        attempts.
    on_reconnect:
        Optional callable ``callback(client)`` executed after a successful
        reconnection. Exceptions are logged but ignored.
    auto_reconnect:
        Enables automatic reconnects when communication errors are detected.
    socket_factory:
        Custom factory returning a connected socket. Defaults to
        :func:`socket.create_connection`.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_timeout: float | None = 5.0,
        read_timeout: float | None = None,
        reconnect_attempts: int = 3,
        reconnect_delay: float = 1.0,
        keep_alive_interval: float | None = None,
        keep_alive_message: bytes | str | None = None,
        keep_alive_callback: Callable[[SBAccess], None] | None = None,
        on_reconnect: Callable[[SBAccess], None] | None = None,
        auto_reconnect: bool = True,
        socket_factory: Callable[[tuple[str, int], Optional[float]], socket.socket]
        | None = None,
    ) -> None:
        if reconnect_attempts < 1:
            raise ValueError("reconnect_attempts must be >= 1")
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        if keep_alive_interval is not None and keep_alive_interval <= 0:
            raise ValueError("keep_alive_interval must be positive when provided")
        self.keep_alive_interval = keep_alive_interval
        self._keep_alive_message = keep_alive_message
        self._keep_alive_callback = keep_alive_callback
        self._on_reconnect = on_reconnect
        self.auto_reconnect = auto_reconnect
        self._socket_factory = socket_factory or socket.create_connection

        self._lock = threading.RLock()
        self._socket: socket.socket | None = None
        self.client: SBAccess | None = None
        self._keep_alive_thread: threading.Thread | None = None
        self._keep_alive_event: threading.Event | None = None
        self._closing = False

    # ------------------------------------------------------------------
    # context manager helpers
    def __enter__(self) -> SBAccess:
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    def connect(self) -> SBAccess:
        """Establish a socket connection and create an :class:`SBAccess`.

        Returns
        -------
        SBAccess
            Connected SBAccess client instance.

        Raises
        ------
        ConnectionError
            If the connection cannot be established.
        """

        with self._lock:
            if self.client is not None and self._socket is not None:
                return self.client

        attempt_error: Exception | None = None
        for attempt in range(1, self.reconnect_attempts + 1):
            sock: socket.socket | None = None
            try:
                LOGGER.debug(
                    "Connecting to %s:%s (attempt %s)",
                    self.host,
                    self.port,
                    attempt,
                )
                sock = self._socket_factory((self.host, self.port), self.connect_timeout)
                if self.read_timeout is not None:
                    sock.settimeout(self.read_timeout)
                client = SBAccess(sock)
                with self._lock:
                    self._socket = sock
                    self.client = client
                    self._closing = False
                    self._start_keep_alive()
                    return client
            except Exception as exc:  # pragma: no cover - error logging path
                attempt_error = exc
                LOGGER.warning(
                    "Connection attempt %s to %s:%s failed: %s",
                    attempt,
                    self.host,
                    self.port,
                    exc,
                )
                if sock is not None:
                    try:
                        sock.close()
                    except Exception:  # pragma: no cover - best effort cleanup
                        pass
                if attempt < self.reconnect_attempts:
                    time.sleep(self.reconnect_delay)
        message = f"Unable to connect to {self.host}:{self.port}"
        raise ConnectionError(message) from attempt_error

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close the underlying socket and stop keep-alive processing."""

        with self._lock:
            self._closing = True
            event = self._keep_alive_event
            thread = self._keep_alive_thread
            self._keep_alive_event = None
            self._keep_alive_thread = None
        if event is not None:
            event.set()
        if thread is not None:
            thread.join(timeout=1.0)
        with self._lock:
            self._teardown_connection()
            self._closing = False

    # ------------------------------------------------------------------
    def _teardown_connection(self) -> None:
        if self.client is not None:
            LOGGER.debug("Tearing down SBAccess client")
        sock = self._socket
        self.client = None
        self._socket = None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            finally:
                sock.close()

    # ------------------------------------------------------------------
    def _start_keep_alive(self) -> None:
        if self.keep_alive_interval is None:
            return
        with self._lock:
            if self._keep_alive_thread and self._keep_alive_thread.is_alive():
                return
            self._keep_alive_event = threading.Event()
            self._keep_alive_thread = threading.Thread(
                target=self._keep_alive_worker,
                name="MicroscopeConnectionKeepAlive",
                daemon=True,
            )
            self._keep_alive_thread.start()

    # ------------------------------------------------------------------
    def _keep_alive_worker(self) -> None:
        assert self._keep_alive_event is not None
        event = self._keep_alive_event
        while not event.wait(self.keep_alive_interval or 0.0):
            with self._lock:
                client = self.client
                sock = self._socket
            if client is None or sock is None:
                continue
            try:
                if self._keep_alive_callback is not None:
                    self._keep_alive_callback(client)
                elif self._keep_alive_message is not None:
                    message = (
                        self._keep_alive_message.encode("utf-8")
                        if isinstance(self._keep_alive_message, str)
                        else self._keep_alive_message
                    )
                    sock.sendall(message)
            except Exception as exc:  # pragma: no cover - keep alive failure path
                LOGGER.warning("Keep-alive failed: %s", exc)
                self._handle_connection_drop(exc)

    # ------------------------------------------------------------------
    def _handle_connection_drop(self, error: Exception) -> None:
        LOGGER.error("Connection to %s:%s dropped: %s", self.host, self.port, error)
        self._teardown_connection()
        if self._closing or not self.auto_reconnect:
            return
        try:
            client = self.connect()
        except ConnectionError:
            LOGGER.exception("Automatic reconnection failed")
        else:
            if self._on_reconnect is not None:
                try:
                    self._on_reconnect(client)
                except Exception:  # pragma: no cover - callback failure path
                    LOGGER.exception("Reconnection callback failed")

