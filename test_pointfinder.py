from typing import Tuple

from sbs_interface import MicroscopeClient, PointFinder


class FakeSB:
    """Minimal stub mimicking :class:`SBAccess` for unit tests.

    The real microscope is unavailable during automated testing, but we can
    still exercise :class:`PointFinder` by attaching this stub as the ``sb``
    attribute on a :class:`MicroscopeClient` instance.  This keeps the test
    close to real usage while avoiding network and hardware dependencies.
    """

    def GetVoxelSize(self, capture_index: int):
        assert capture_index == 0
        return 0.5, 0.25, 1.0

    def GetCurrentSlideId(self):
        return 0

    def SetTargetSlide(self, slide_id: int):
        assert slide_id == 0

    def GetXPosition(self, capture_index: int, image_number: int):
        return 0.0

    def GetYPosition(self, capture_index: int, image_number: int):
        return 0.0

    def GetZPosition(self, capture_index: int, image_number: int, z_plane_index: int):
        return 1.0

    def GetNumXColumns(self, capture_index: int):
        return 4

    def GetNumYRows(self, capture_index: int):
        return 24

    def GetNumZPlanes(self, capture_index: int):
        return 1


def make_client() -> MicroscopeClient:
    mc = MicroscopeClient.__new__(MicroscopeClient)
    mc.sb = FakeSB()
    return mc


def test_pixel_to_physical_single():
    finder = PointFinder(make_client())
    assert finder.pixel_to_physical(2, 4, 1) == (0.0, -2.0, 2.0, 0.0)


def test_convert_points_list():
    finder = PointFinder(make_client())
    Point = Tuple[float, float, float, float]
    pts: list[Point] = [(0, 0, 0, 0), (2, 4, 1, 0)]
    expected = [
        finder.pixel_to_physical(x, y, z, auxz)
        for x, y, z, auxz in pts
    ]
    assert finder.convert_points(pts) == expected


def test_stage_direction(monkeypatch):
    """Pixel-to-physical conversion honours stage direction multipliers."""
    # Force stage signs to -1 for both axes to simulate an inverted stage.
    monkeypatch.setattr("sbs_interface.SBPointFinder._stage_signs", lambda: (-1, -1))
    finder = PointFinder(make_client())
    # Use a point away from the image centre so both axes are affected.
    assert finder.pixel_to_physical(0, 4, 1) == (1.0, 2.0, 2.0, 0.0)
