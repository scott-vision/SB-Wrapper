from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from sbwrapper.client import MicroscopeClient, Point
from sbwrapper.sb_access import SBAccess


def make_mock_sbaccess() -> MagicMock:
    return MagicMock(spec=SBAccess)


def test_add_points_clears_then_batches():
    mock_access = make_mock_sbaccess()
    client = MicroscopeClient(mock_access, chunk_size=2)

    points = [
        Point(1, 2, 3),
        {"x": 4, "y": 5, "z": 6, "aux_z": 7},
        (8, 9, 10, 11, False),
    ]

    client.add_points(points, clear_existing=True)

    assert mock_access.method_calls[0] == call.ClearXYZPoints()
    mock_access.AddXYZPoint.assert_has_calls(
        [
            call(1.0, 2.0, 3.0, 0.0, 0),
            call(4.0, 5.0, 6.0, 7.0, 1),
            call(8.0, 9.0, 10.0, 11.0, 0),
        ]
    )


def test_add_points_accepts_generators_and_override_chunk():
    mock_access = make_mock_sbaccess()
    client = MicroscopeClient(mock_access, chunk_size=10)

    def generator():
        for idx in range(5):
            yield (idx, idx + 0.5, idx + 1.0)

    client.add_points(generator(), chunk_size=2)

    assert mock_access.ClearXYZPoints.call_count == 0
    assert mock_access.AddXYZPoint.call_count == 5
    mock_access.AddXYZPoint.assert_has_calls(
        [
            call(0.0, 0.5, 1.0, 0.0, 0),
            call(1.0, 1.5, 2.0, 0.0, 0),
            call(2.0, 2.5, 3.0, 0.0, 0),
            call(3.0, 3.5, 4.0, 0.0, 0),
            call(4.0, 4.5, 5.0, 0.0, 0),
        ]
    )


def test_add_point_handles_auxiliary_defaults():
    mock_access = make_mock_sbaccess()
    client = MicroscopeClient(mock_access)

    client.add_point(1, 2, 3)
    client.add_point(4, 5, 6, aux_z=7)
    client.add_point(7, 8, 9, aux_z=10, use_aux=False)

    mock_access.AddXYZPoint.assert_has_calls(
        [
            call(1.0, 2.0, 3.0, 0.0, 0),
            call(4.0, 5.0, 6.0, 7.0, 1),
            call(7.0, 8.0, 9.0, 10.0, 0),
        ]
    )


def test_invalid_point_sequence_raises():
    mock_access = make_mock_sbaccess()
    client = MicroscopeClient(mock_access)

    with pytest.raises(ValueError):
        client.add_points([(1, 2)])

