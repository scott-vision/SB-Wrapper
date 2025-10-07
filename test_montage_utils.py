import pytest

np = pytest.importorskip("numpy")

from sbs_interface.microscope_client import MicroscopeClient
from sbs_interface.SBMontageUtils import MontageUtils


class FakeSB:
    """Stub implementing the minimal SBAccess API for montage tests."""

    positions = [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 3), (1, 2), (1, 1), (1, 0),
    ]

    def GetCurrentSlideId(self):
        return 0

    def SetTargetSlide(self, slide_id):
        assert slide_id == 0

    def GetNumPositions(self, capture_index):
        assert capture_index == 0
        return len(self.positions)

    def GetMontageRow(self, capture_index, position_index):
        return self.positions[position_index][0]

    def GetMontageColumn(self, capture_index, position_index):
        return self.positions[position_index][1]

    def GetNumXColumns(self, capture_index):
        return 2

    def GetNumYRows(self, capture_index):
        return 2

    def GetNumZPlanes(self, capture_index):
        return 1

    def GetVoxelSize(self, capture_index):
        return 1.0, 1.0, 1.0

    def GetXPosition(self, capture_index, image_number):
        return self.positions[image_number][1] * 2.0

    def GetYPosition(self, capture_index, image_number):
        return self.positions[image_number][0] * 2.0

    def GetZPosition(self, capture_index, image_number, z_plane_index):
        return 0.0

    def ReadImagePlaneBuf(self, capture_index, position_index, timepoint_index, z_plane_index, channel_index):
        # Each tile is 2x2 with constant value equal to its index
        arr = np.full((2, 2), position_index, dtype=np.uint16)
        return arr.tobytes()



def make_client() -> MicroscopeClient:
    mc = MicroscopeClient.__new__(MicroscopeClient)
    mc.sb = FakeSB()
    return mc


class StageSignSB(FakeSB):
    """Variant where X coordinates decrease when moving right."""

    def GetXPosition(self, capture_index, image_number):
        return -self.positions[image_number][1] * 2.0


def make_sign_client() -> MicroscopeClient:
    mc = MicroscopeClient.__new__(MicroscopeClient)
    mc.sb = StageSignSB()
    return mc


def test_build_map_and_offsets():
    util = MontageUtils(make_client())
    expected_map = np.array([[0, 1, 2, 3], [7, 6, 5, 4]])
    assert np.array_equal(util.build_map(), expected_map)

    offsets = util.compute_offsets()
    assert offsets[0] == (0.0, 0.0)
    assert offsets[4] == (6.0, 2.0)


def test_stage_direction(monkeypatch):
    monkeypatch.setattr("sbs_interface.SBMontageUtils._STAGE_SIGNS", None)
    monkeypatch.setattr(
        "sbs_interface.SBMontageUtils._stage_signs", lambda: (-1, 1)
    )
    util = MontageUtils(make_sign_client())
    offsets = util.compute_offsets()
    assert offsets[1] == (2.0, 0.0)


def test_stitch():
    util = MontageUtils(make_client())
    stitched, pixel_dims, micron_dims = util.stitch()
    assert stitched.shape == (4, 8)
    expected = np.array([
        [0, 0, 1, 1, 2, 2, 3, 3],
        [0, 0, 1, 1, 2, 2, 3, 3],
        [7, 7, 6, 6, 5, 5, 4, 4],
        [7, 7, 6, 6, 5, 5, 4, 4],
    ], dtype=np.uint16)
    assert np.array_equal(stitched, expected)
    assert pixel_dims == (8, 4)
    assert micron_dims == (8.0, 4.0)


class OverlapSB(FakeSB):
    positions = [(0, 0), (0, 1)]

    def GetNumPositions(self, capture_index):
        return len(self.positions)

    def GetNumXColumns(self, capture_index):
        return 4

    def ReadImagePlaneBuf(self, capture_index, position_index, timepoint_index, z_plane_index, channel_index):
        value = 1 if position_index == 0 else 3
        arr = np.full((2, 4), value, dtype=np.uint16)
        return arr.tobytes()

    def GetXPosition(self, capture_index, image_number):
        return [0.0, 3.0][image_number]

    def GetYPosition(self, capture_index, image_number):
        return 0.0


def make_overlap_client() -> MicroscopeClient:
    mc = MicroscopeClient.__new__(MicroscopeClient)
    mc.sb = OverlapSB()
    return mc


def test_stitch_with_overlap():
    util = MontageUtils(make_overlap_client())
    stitched, pixel_dims, micron_dims = util.stitch()
    assert stitched.shape == (2, 7)
    expected = np.array([
        [1, 1, 1, 2, 3, 3, 3],
        [1, 1, 1, 2, 3, 3, 3],
    ], dtype=np.uint16)
    assert np.array_equal(stitched, expected)
    assert pixel_dims == (7, 2)
    assert micron_dims == (7.0, 2.0)


class CorrSB(FakeSB):
    positions = [(0, 0), (0, 1)]

    def GetNumPositions(self, capture_index):
        return len(self.positions)

    def GetNumXColumns(self, capture_index):
        return 4

    def ReadImagePlaneBuf(self, capture_index, position_index, timepoint_index, z_plane_index, channel_index):
        if position_index == 0:
            arr = np.array([
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ], dtype=np.uint16)
        else:
            arr = np.array([
                [3, 4, 5, 6],
                [7, 8, 9, 10],
                [11, 12, 13, 14],
                [15, 16, 17, 18],
            ], dtype=np.uint16)
        return arr.tobytes()

    def GetXPosition(self, capture_index, image_number):
        # Stage reports 1 pixel overlap (3.0), actual data overlaps by 2 pixels
        return [0.0, 3.0][image_number]

    def GetYPosition(self, capture_index, image_number):
        return 0.0


def make_corr_client() -> MicroscopeClient:
    mc = MicroscopeClient.__new__(MicroscopeClient)
    mc.sb = CorrSB()
    return mc


def test_stitch_cross_corr():
    util = MontageUtils(make_corr_client())
    stitched, pixel_dims, micron_dims = util.stitch(cross_correlation=True)
    assert stitched.shape == (4, 6)
    expected = np.array([
        [1, 2, 3, 4, 5, 6],
        [5, 6, 7, 8, 9, 10],
        [9, 10, 11, 12, 13, 14],
        [13, 14, 15, 16, 17, 18],
    ], dtype=np.uint16)
    assert np.array_equal(stitched, expected)
    assert pixel_dims == (6, 4)
    assert micron_dims == (6.0, 4.0)


class FeatureFallbackSB(FakeSB):
    positions = [(0, 0), (0, 1)]

    def GetNumPositions(self, capture_index):
        return len(self.positions)

    def GetNumXColumns(self, capture_index):
        return 4

    def ReadImagePlaneBuf(self, capture_index, position_index, timepoint_index, z_plane_index, channel_index):
        if position_index == 0:
            arr = np.array([
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
            ], dtype=np.uint16)
        else:
            arr = np.array([
                [0, 0, 2, 2],
                [0, 0, 2, 2],
                [0, 0, 2, 2],
                [0, 0, 2, 2],
            ], dtype=np.uint16)
        return arr.tobytes()

    def GetXPosition(self, capture_index, image_number):
        # Stage reports 1 pixel overlap (3.0) but correlation is ambiguous
        return [0.0, 3.0][image_number]

    def GetYPosition(self, capture_index, image_number):
        return 0.0


def make_feature_client() -> MicroscopeClient:
    mc = MicroscopeClient.__new__(MicroscopeClient)
    mc.sb = FeatureFallbackSB()
    return mc


def test_stitch_feature_fallback(monkeypatch):
    util = MontageUtils(make_feature_client())
    # First run with only cross correlation – misregistration yields width 6
    _, dims, _ = util.stitch(cross_correlation=True)
    assert dims[0] == 6

    called = {"count": 0}

    def fake_feature(a, b):
        called["count"] += 1
        return 0, 1  # shift right by one pixel

    monkeypatch.setattr("sbs_interface.SBMontageUtils._feature_offset", fake_feature)
    _, dims_feat, _ = util.stitch(cross_correlation=True, use_features=True)
    assert called["count"] > 0
    assert dims_feat[0] == 7


class SingleTileSB(FakeSB):
    """Montage returning a single tile and zero positions."""

    positions = [(0, 0)]

    def GetNumPositions(self, capture_index):
        # SlideBook can report zero positions for a lone tile
        return 0

    def ReadImagePlaneBuf(self, capture_index, position_index, timepoint_index, z_plane_index, channel_index):
        # Distinct value to verify placement
        arr = np.full((2, 2), 9, dtype=np.uint16)
        return arr.tobytes()


def make_single_client() -> MicroscopeClient:
    mc = MicroscopeClient.__new__(MicroscopeClient)
    mc.sb = SingleTileSB()
    return mc


def test_stitch_single_tile():
    util = MontageUtils(make_single_client())
    stitched, pixel_dims, micron_dims = util.stitch()
    assert stitched.shape == (2, 2)
    assert np.array_equal(stitched, np.full((2, 2), 9, dtype=np.uint16))
    assert pixel_dims == (2, 2)
    assert micron_dims == (2.0, 2.0)


class ReverseOrderSB(FakeSB):
    """Montage starting at the top-right corner."""

    positions = [(0, 1), (0, 0)]

    def GetNumPositions(self, capture_index):
        return len(self.positions)

    def GetMontageRow(self, capture_index, position_index):
        return self.positions[position_index][0]

    def GetMontageColumn(self, capture_index, position_index):
        return self.positions[position_index][1]

    def ReadImagePlaneBuf(self, capture_index, position_index, timepoint_index, z_plane_index, channel_index):
        arr = np.full((2, 2), position_index + 1, dtype=np.uint16)
        return arr.tobytes()

    def GetXPosition(self, capture_index, image_number):
        return [2.0, 0.0][image_number]

    def GetYPosition(self, capture_index, image_number):
        return 0.0


def make_reverse_client() -> MicroscopeClient:
    mc = MicroscopeClient.__new__(MicroscopeClient)
    mc.sb = ReverseOrderSB()
    return mc


def test_stitch_reverse_order():
    util = MontageUtils(make_reverse_client())
    stitched, pixel_dims, micron_dims = util.stitch()
    assert stitched.shape == (2, 4)
    expected = np.array([
        [2, 2, 1, 1],
        [2, 2, 1, 1],
    ], dtype=np.uint16)
    assert np.array_equal(stitched, expected)
    assert pixel_dims == (4, 2)
    assert micron_dims == (4.0, 2.0)


class MaxProjSB(FakeSB):
    def GetNumZPlanes(self, capture_index):
        return 2

    def ReadImagePlaneBuf(
        self,
        capture_index,
        position_index,
        timepoint_index,
        z_plane_index,
        channel_index,
    ):
        value = position_index + z_plane_index * 10
        arr = np.full((2, 2), value, dtype=np.uint16)
        return arr.tobytes()


def make_maxproj_client() -> MicroscopeClient:
    mc = MicroscopeClient.__new__(MicroscopeClient)
    mc.sb = MaxProjSB()
    return mc


def test_fetch_max_projected_tile():
    util = MontageUtils(make_maxproj_client(), max_project=True)
    arr = util._fetch_image_at_position(3)
    expected = np.full((2, 2), 13, dtype=np.uint16)
    assert np.array_equal(arr, expected)

