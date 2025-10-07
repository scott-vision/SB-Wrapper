from __future__ import annotations

import logging
import socket
from typing import Any, List, Tuple, Optional

# ``MicroscopeClient`` depends on :mod:`SBAccess` for real microscope control,
# but importing it pulls in heavy optional dependencies.  Defer the import so
# unit tests can run in minimal environments.
try:  # pragma: no cover - tested indirectly
    from .SBAccess import SBAccess
except ImportError:  # pragma: no cover - allow tests to provide a stub
    SBAccess = Any  # type: ignore[misc,assignment]

# NumPy is only required for image transfer; see :func:`fetch_image`.
try:  # pragma: no cover
    import numpy as np
except ImportError:  # pragma: no cover - gracefully degrade when NumPy absent
    np = None  # type: ignore[assignment]

Point = Tuple[float, float, float, float]  # (x, y, z, auxz)


def parse_point_string(point_str: str) -> Point:
    """Convert one SlideBook XYZ-point list entry into a 4-tuple."""
    coord_section = point_str.split(":", 1)[1].strip().lstrip("(").rstrip(")")
    values = [float(v.strip()) for v in coord_section.split(",")]
    if len(values) == 3:
        values.append(0.0)
    if len(values) != 4:
        raise ValueError(f"Unexpected coordinate length in '{point_str}'")
    return tuple(values)


class MicroscopeClient:
    """Context-manager wrapper simplifying SBAccess socket use."""

    def __init__(self, host: str, port: int, timeout: float = 5.0):
        self.host, self.port, self.timeout = host, port, timeout
        self.sock: Optional[socket.socket] = None
        self.sb: Optional[SBAccess] = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_exc) -> None:
        self.disconnect()

    def connect(self) -> None:
        """Open TCP socket and initialise SBAccess."""
        logging.info("Connecting to %s:%d …", self.host, self.port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect((self.host, self.port))
        self.sb = SBAccess(self.sock)
        logging.info("Connection established")

    def disconnect(self) -> None:
        """Close socket and clean up."""
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.sock.close()
        self.sock = None
        self.sb = None
        logging.info("Disconnected from microscope")

    def fetch_points(self) -> List[Point]:
        """Return all current XYZ points from SlideBook."""
        if not self.sb:
            raise RuntimeError("Not connected to SlideBook")
        raw = self.sb.GetXYZPointList()
        if raw == ["Empty"]:
            raise ValueError("SlideBook reports an empty XYZ list")
        points = [
            parse_point_string(s)
            for s in raw
            if s and s.strip() and s != "Empty"
        ]
        logging.info("Fetched %d points", len(points))
        return points

    def push_points(self, ordered: List[Point]) -> None:
        """Upload an ordered list of points back to SlideBook."""
        if not self.sb:
            raise RuntimeError("Not connected to SlideBook")
        self.sb.ClearXYZPoints()
        for x, y, z, aux in ordered:
            self.sb.AddXYZPoint(x, y, z, aux, bool(aux))
        logging.info("Uploaded %d points in optimised order", len(ordered))

    def fetch_image(
        self,
        capture_number: int = 0,
        channel: int = 0,
        z_plane: int = 0,
    ) -> np.ndarray:
        """Grab an image plane from the currently open slide."""
        if np is None:
            raise RuntimeError("NumPy is required for image transfer")
        if not self.sb:
            raise RuntimeError("Not connected to SlideBook")
        slide_id = self.sb.GetCurrentSlideId()
        self.sb.SetTargetSlide(slide_id)
        rows = self.sb.GetNumYRows(capture_number)
        cols = self.sb.GetNumXColumns(capture_number)
        buf = self.sb.ReadImagePlaneBuf(capture_number, 0, 0, z_plane, channel)
        arr = np.frombuffer(buf, dtype=np.uint16).reshape(rows, cols)
        return arr

    def fetch_num_channels(self, capture_number: int = 0) -> int:
        """Return the number of channels available for the current slide."""
        if not self.sb:
            raise RuntimeError("Not connected to SlideBook")
        slide_id = self.sb.GetCurrentSlideId()
        self.sb.SetTargetSlide(slide_id)
        return self.sb.GetNumChannels(capture_number)
    
    def fetch_voxel_size(
        self,
        capture_number: int = 0
    ) -> np.ndarray:
        """Enforce open slide."""
        slide_id = self.sb.GetCurrentSlideId()
        self.sb.SetTargetSlide(slide_id)
        logging.info("Getting Voxel size")
        return self.sb.GetVoxelSize(capture_number)
    
    def fetch_image_XYZ(
        self,
        capture_number: int = 0,
        image_number: int = 0,
        z_plane_index: int = 0
    ) -> np.ndarray:
        """Enforce open slide."""
        slide_id = self.sb.GetCurrentSlideId()
        self.sb.SetTargetSlide(slide_id)
        logging.info("Getting XYZ starting location")
        return [self.sb.GetXPosition(capture_number, image_number), self.sb.GetYPosition(capture_number, image_number), self.sb.GetZPosition(capture_number, image_number, z_plane_index)]

    def fetch_image_dims(
            self,
            capture_number: int = 0
    ) -> np.array:
        """Enforce open slide."""
        slide_id = self.sb.GetCurrentSlideId()
        self.sb.SetTargetSlide(slide_id)
        logging.info("Getting image dimensions")
        return [
            self.sb.GetNumXColumns(capture_number),
            self.sb.GetNumYRows(capture_number),
            self.sb.GetNumZPlanes(capture_number),
        ]

    def fetch_num_positions(
            self,
            capture_number: int = 0,
    ) -> int:
        """Return the number of positions in the current montage."""
        if not self.sb:
            raise RuntimeError("Not connected to SlideBook")
        slide_id = self.sb.GetCurrentSlideId()
        self.sb.SetTargetSlide(slide_id)
        return self.sb.GetNumPositions(capture_number)

    def fetch_montage_position(
            self,
            capture_number: int = 0,
            position: int = 0,
    ) -> Tuple[int, int]:
        """Return the montage row and column for a position."""
        if not self.sb:
            raise RuntimeError("Not connected to SlideBook")
        slide_id = self.sb.GetCurrentSlideId()
        self.sb.SetTargetSlide(slide_id)
        row = self.sb.GetMontageRow(capture_number, position)
        col = self.sb.GetMontageColumn(capture_number, position)
        return row, col

    def fetch_latest_capture_index(self) -> int:
        """Return the index of the most recent capture on the active slide."""
        if not self.sb:
            raise RuntimeError("Not connected to SlideBook")
        slide_id = self.sb.GetCurrentSlideId()
        self.sb.SetTargetSlide(slide_id)
        count = self.sb.GetNumCaptures()
        return max(0, count - 1)

    def fetch_capture_count(self) -> int:
        """Return the number of captures on the active slide."""
        if not self.sb:
            raise RuntimeError("Not connected to SlideBook")
        slide_id = self.sb.GetCurrentSlideId()
        self.sb.SetTargetSlide(slide_id)
        return self.sb.GetNumCaptures()

    def is_capturing(self) -> bool:
        """Return ``True`` while an acquisition is in progress."""
        if not self.sb:
            raise RuntimeError("Not connected to SlideBook")
        return bool(self.sb.IsCapturing())

    def start_capture(self, script: str = "Default") -> int:
        if not self.sb:
            raise RuntimeError("Not connected to SlideBook")
        return self.sb.StartCapture(script)
