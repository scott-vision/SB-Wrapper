import importlib
import sys
from typing import Tuple

if "sbs_interface" not in sys.modules:
    sys.modules["sbs_interface"] = importlib.import_module("src")

from sbs_interface.services.microscope import MicroscopeService

Point = Tuple[float, float, float, float]


class _ConnectedFakeClient:
    """Mixin implementing connection lifecycle expected by the service."""

    instances = []
    connect_count = 0

    def __init__(self, *args, **kwargs):
        self.connected = False
        self.sb = None
        type(self).instances.append(self)

    def connect(self):
        self.connected = True
        type(self).connect_count += 1
        self.sb = object()

    def disconnect(self):
        self.connected = False
        self.sb = None

    def is_healthy(self):
        return self.connected


def test_fetch_and_push_points(monkeypatch):
    class FakeClient(_ConnectedFakeClient):
        instances = []
        connect_count = 0
        pushed = None

        def __init__(self, host, port, timeout=5.0):
            super().__init__(host, port, timeout)
            self.points = [(1.0, 2.0, 3.0, 4.0)]

        def fetch_points(self):
            return self.points

        def push_points(self, ordered):
            type(self).pushed = ordered

    monkeypatch.setattr(
        "sbs_interface.services.microscope.MicroscopeClient", FakeClient
    )
    svc = MicroscopeService("h", 1)
    assert svc.fetch_points() == [(1.0, 2.0, 3.0, 4.0)]
    svc.push_points([(5.0, 6.0, 7.0, 8.0)])
    assert FakeClient.pushed == [(5.0, 6.0, 7.0, 8.0)]
    svc.disconnect()


def test_start_capture_and_channels(monkeypatch):
    class FakeClient(_ConnectedFakeClient):
        instances = []
        connect_count = 0
        last_script = None

        def start_capture(self, script="Default"):
            type(self).last_script = script
            return 42

        def fetch_latest_capture_index(self):
            return 0

        def fetch_num_channels(self, capture=0):
            return 7

    monkeypatch.setattr(
        "sbs_interface.services.microscope.MicroscopeClient", FakeClient
    )
    svc = MicroscopeService("h", 1)
    assert svc.start_capture("foo") == 42
    assert FakeClient.last_script == "foo"
    assert svc.fetch_num_channels() == 7
    svc.disconnect()


def test_fetch_display_image(monkeypatch):
    class FakeNP:
        @staticmethod
        def maximum(a, b):
            return [[max(x, y) for x, y in zip(r1, r2)] for r1, r2 in zip(a, b)]

    class FakeClient(_ConnectedFakeClient):
        instances = []
        connect_count = 0
        def is_capturing(self):
            return False

        def fetch_capture_count(self):
            return 1

        def fetch_latest_capture_index(self):
            return 0

        def fetch_image_dims(self, capture=0):
            return [2, 2, 3]

        def fetch_image(self, capture=0, channel=0, z_plane=0):
            return [[z_plane, z_plane], [z_plane, z_plane]]

    monkeypatch.setattr(
        "sbs_interface.services.microscope.MicroscopeClient", FakeClient
    )
    monkeypatch.setattr("sbs_interface.services.microscope.np", FakeNP)
    svc = MicroscopeService("h", 1)
    arr, planes = svc.fetch_display_image(channel=0, z=0, max_project=True)
    assert planes == 3
    assert arr == [[2, 2], [2, 2]]
    svc.disconnect()


def test_push_points_from_pixels(monkeypatch):
    class FakeClient(_ConnectedFakeClient):
        instances = []
        connect_count = 0
        pushed = None

        def fetch_latest_capture_index(self):
            return 0

        def fetch_voxel_size(self, capture_index=0):
            return 0.5, 0.25, 1.0

        def fetch_image_XYZ(self, capture_index=0):
            return [0.0, 0.0, 1.0]

        def fetch_image_dims(self, capture_index=0):
            return [4, 24, 1]

        def push_points(self, pts):
            type(self).pushed = pts

    monkeypatch.setattr(
        "sbs_interface.services.microscope.MicroscopeClient", FakeClient
    )
    svc = MicroscopeService("h", 1)
    uploaded = svc.push_points_from_pixels([(2, 4, 1, 0)])
    assert uploaded == 1
    assert FakeClient.pushed == [(0.0, -2.0, 2.0, 0.0)]
    svc.disconnect()


def test_fetch_stitched_montage(monkeypatch):
    class FakeMontage:
        def __init__(self, mc, capture_index=0, channel=0, z_plane=0, max_project=False):
            pass

        def stitch(self, cross_correlation=False, use_features=False):
            arr = [[1, 2], [3, 4]]
            return arr, (2, 2), (0.0, 0.0)

    class FakeClient(_ConnectedFakeClient):
        instances = []
        connect_count = 0
        def is_capturing(self):
            return False

        def fetch_capture_count(self):
            return 1

        def fetch_latest_capture_index(self):
            return 0

    monkeypatch.setattr(
        "sbs_interface.services.microscope.MicroscopeClient", FakeClient
    )
    monkeypatch.setattr(
        "sbs_interface.services.microscope.MontageUtils", FakeMontage
    )
    monkeypatch.setattr("sbs_interface.services.microscope.np", object())
    svc = MicroscopeService("h", 1)
    arr, dims = svc.fetch_stitched_montage(channel=0, z=0)
    assert dims == (2, 2)
    assert arr == [[1, 2], [3, 4]]
    svc.disconnect()


def test_reuses_client_between_operations(monkeypatch):
    class FakeClient(_ConnectedFakeClient):
        instances = []
        connect_count = 0
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = []

        def fetch_points(self):
            self.calls.append("fetch_points")
            return []

        def fetch_capture_count(self):
            self.calls.append("fetch_capture_count")
            return 0

    monkeypatch.setattr(
        "sbs_interface.services.microscope.MicroscopeClient", FakeClient
    )
    svc = MicroscopeService("h", 1)
    svc.fetch_points()
    svc.fetch_capture_count()
    assert len(FakeClient.instances) == 1
    assert FakeClient.connect_count == 1
    assert FakeClient.instances[0].calls == ["fetch_points", "fetch_capture_count"]
    svc.disconnect()
