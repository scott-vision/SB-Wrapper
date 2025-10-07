from __future__ import annotations

import os
from typing import List, Tuple, Optional, Union
import time

# NumPy is only required for image operations; make optional for light tests.
try:  # pragma: no cover
    import numpy as np
except ImportError:  # pragma: no cover - allow tests to mock
    np = None  # type: ignore[assignment]

from ..microscope_client import MicroscopeClient, Point
from ..SBPointFinder import PointFinder
from ..SBMontageUtils import MontageUtils
from ..objective_manager import ObjectiveManager


class MicroscopeService:
    """High-level wrapper around :class:`MicroscopeClient`.

    Each method opens a short-lived connection to SlideBook and performs
    the requested operation.  This keeps the web layer free from direct
    socket management and enables easy mocking during unit tests.
    """

    def __init__(self, host: str, port: int, timeout: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout

    # ------------------------------------------------------------------
    def _client(self) -> MicroscopeClient:
        return MicroscopeClient(self.host, self.port, self.timeout)

    def _wait_for_capture(self, mc: MicroscopeClient, poll: float = 0.5) -> None:
        """Block until the microscope is no longer capturing and at least one capture exists.

        Parameters
        ----------
        mc:
            Active :class:`MicroscopeClient`.
        poll:
            Delay between status checks.
        """
        while True:
            capturing = mc.is_capturing()
            count = mc.fetch_capture_count()
            if not capturing and count > 0:
                break
            time.sleep(poll)

    # ------------------------------------------------------------------
    def check_connection(self) -> bool:
        """Return ``True`` if a connection to the microscope can be made."""
        try:
            with self._client():
                return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    def fetch_points(self) -> List[Point]:
        with self._client() as mc:
            return mc.fetch_points()

    # ------------------------------------------------------------------
    def push_points(self, points: List[Point]) -> None:
        with self._client() as mc:
            mc.push_points(points)

    # ------------------------------------------------------------------
    def start_capture(self, script: str = "Default") -> int:
        with self._client() as mc:
            return mc.start_capture(script)

    # ------------------------------------------------------------------
    def fetch_num_channels(self) -> int:
        with self._client() as mc:
            capture = mc.fetch_latest_capture_index()
            return mc.fetch_num_channels(capture)

    # ------------------------------------------------------------------
    def list_objectives(self) -> List[Tuple[str, int]]:
        """Return available objectives as ``(name, position)`` pairs."""
        with self._client() as mc:
            if mc.sb is None:
                return []
            mgr = ObjectiveManager(mc.sb)
            return mgr.list_objectives()

    # ------------------------------------------------------------------
    def get_current_objective(self) -> Optional[Tuple[str, int]]:
        """Return the currently selected objective."""
        with self._client() as mc:
            if mc.sb is None:
                return None
            mgr = ObjectiveManager(mc.sb)
            return mgr.get_current_objective()

    # ------------------------------------------------------------------
    def set_objective(self, objective: Union[str, int]) -> bool:
        """Change the microscope objective."""
        with self._client() as mc:
            if mc.sb is None:
                return False
            mgr = ObjectiveManager(mc.sb)
            return mgr.set_objective(objective)

    # ------------------------------------------------------------------
    def fetch_capture_count(self) -> int:
        with self._client() as mc:
            return mc.fetch_capture_count()

    # ------------------------------------------------------------------
    def fetch_display_image(
        self,
        channel: int = 0,
        z: int = 0,
        max_project: bool = False,
        capture_index: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """Return an image plane (optionally a Z max projection)."""
        if np is None:
            raise RuntimeError("NumPy is required for image operations")
        with self._client() as mc:
            self._wait_for_capture(mc)
            if mc.fetch_capture_count() == 0:
                raise ValueError("No captures available")
            capture = (
                mc.fetch_latest_capture_index()
                if capture_index is None
                else capture_index
            )
            dims = mc.fetch_image_dims(capture)
            planes = int(dims[2])
            if planes == 0:
                return mc.fetch_image(capture, channel=channel, z_plane=0), planes
            z = max(0, min(int(z), planes - 1))
            if max_project:
                arr = mc.fetch_image(capture, channel=channel, z_plane=0)
                for idx in range(1, planes):
                    arr = np.maximum(
                        arr,
                        mc.fetch_image(capture, channel=channel, z_plane=idx),
                    )
            else:
                arr = mc.fetch_image(capture, channel=channel, z_plane=z)
            return arr, planes

    # ------------------------------------------------------------------
    def fetch_stitched_montage(
        self,
        channel: int = 0,
        z: int = 0,
        max_project: bool = False,
        cross_corr: bool = False,
        use_features: bool = False,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Return a stitched montage image and its pixel dimensions."""
        if np is None:
            raise RuntimeError("NumPy is required for image operations")
        with self._client() as mc:
            self._wait_for_capture(mc)
            if mc.fetch_capture_count() == 0:
                raise ValueError("No captures available")
            capture = mc.fetch_latest_capture_index()

            util = MontageUtils(
                mc,
                capture_index=capture,
                channel=channel,
                z_plane=z,
                max_project=max_project,
            )
            arr, pixel_dims, _ = util.stitch(
                cross_correlation=cross_corr, use_features=use_features
            )
            return arr, pixel_dims

    # ------------------------------------------------------------------
    def push_points_from_pixels(self, pts: List[Point]) -> int:
        """Convert pixel-space points to physical units and upload them."""
        with self._client() as mc:
            capture = mc.fetch_latest_capture_index()
            finder = PointFinder(mc, capture)
            physical = finder.convert_points(pts)
            mc.push_points(physical)
            return len(physical)


MICROSCOPE_HOST = os.getenv("MICROSCOPE_HOST", "127.0.0.1")
MICROSCOPE_PORT = int(os.getenv("MICROSCOPE_PORT", "65432"))

_service: Optional[MicroscopeService] = None


def get_microscope_service() -> MicroscopeService:
    """FastAPI dependency returning a singleton :class:`MicroscopeService`."""
    global _service
    if _service is None:
        _service = MicroscopeService(MICROSCOPE_HOST, MICROSCOPE_PORT)
    return _service
