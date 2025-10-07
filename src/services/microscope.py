from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Iterator, List, Tuple, Optional, Union
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
from ..configuration import get_setting


class MicroscopeService:
    """High-level wrapper around :class:`MicroscopeClient`.

    The service maintains a long-lived connection guarded by a thread lock so
    repeated operations can reuse the same socket without reconnecting to
    SlideBook for every call.  This still keeps the web layer free from direct
    socket management and enables easy mocking during unit tests.
    """

    def __init__(self, host: str, port: int, timeout: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._client_lock = threading.Lock()
        self._client: Optional[MicroscopeClient] = None
        self._sb: Optional[Any] = None

    # ------------------------------------------------------------------
    def connect(self) -> MicroscopeClient:
        """Create or return the long-lived :class:`MicroscopeClient`."""
        with self._client_lock:
            client = self._client
            if client and client.is_healthy():
                return client

            client = MicroscopeClient(self.host, self.port, self.timeout)
            try:
                client.connect()
            except Exception:
                self._client = None
                self._sb = None
                raise

            self._client = client
            self._sb = client.sb
            return client

    def disconnect(self) -> None:
        """Close and clear any cached client."""
        with self._client_lock:
            client = self._client
            self._client = None
            self._sb = None

        if client:
            try:
                client.disconnect()
            except Exception:
                pass

    def _ensure_client(self) -> MicroscopeClient:
        """Return a healthy client, reconnecting as necessary."""
        client = self.connect()
        if not client.is_healthy():
            self.disconnect()
            client = self.connect()
        self._sb = client.sb
        return client

    def _handle_client_failure(self) -> None:
        self.disconnect()

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


@dataclass(frozen=True)
class MontageTile:
    """Description of a single montage tile returned by the microscope."""

    image: Any
    stage_offset_um: Tuple[float, float]
    pixel_offset: Tuple[float, float]
    size_pixels: Tuple[int, int]

    # ------------------------------------------------------------------
    def check_connection(self) -> bool:
        """Return ``True`` if a connection to the microscope can be made."""
        try:
            self.connect()
            return True
        except Exception:
            self.disconnect()
            return False

    # ------------------------------------------------------------------
    def fetch_points(self) -> List[Point]:
        mc = self._ensure_client()
        try:
            return mc.fetch_points()
        except Exception:
            self._handle_client_failure()
            raise

    # ------------------------------------------------------------------
    def push_points(self, points: List[Point]) -> None:
        mc = self._ensure_client()
        try:
            mc.push_points(points)
        except Exception:
            self._handle_client_failure()
            raise

    # ------------------------------------------------------------------
    def start_capture(self, script: str = "Default") -> int:
        mc = self._ensure_client()
        try:
            return mc.start_capture(script)
        except Exception:
            self._handle_client_failure()
            raise

    # ------------------------------------------------------------------
    def fetch_num_channels(self) -> int:
        mc = self._ensure_client()
        try:
            capture = mc.fetch_latest_capture_index()
            return mc.fetch_num_channels(capture)
        except Exception:
            self._handle_client_failure()
            raise

    # ------------------------------------------------------------------
    def list_objectives(self) -> List[Tuple[str, int]]:
        """Return available objectives as ``(name, position)`` pairs."""
        mc = self._ensure_client()
        try:
            if mc.sb is None:
                return []
            mgr = ObjectiveManager(mc.sb)
            return mgr.list_objectives()
        except Exception:
            self._handle_client_failure()
            raise

    # ------------------------------------------------------------------
    def get_current_objective(self) -> Optional[Tuple[str, int]]:
        """Return the currently selected objective."""
        mc = self._ensure_client()
        try:
            if mc.sb is None:
                return None
            mgr = ObjectiveManager(mc.sb)
            return mgr.get_current_objective()
        except Exception:
            self._handle_client_failure()
            raise

    # ------------------------------------------------------------------
    def set_objective(self, objective: Union[str, int]) -> bool:
        """Change the microscope objective."""
        mc = self._ensure_client()
        try:
            if mc.sb is None:
                return False
            mgr = ObjectiveManager(mc.sb)
            return mgr.set_objective(objective)
        except Exception:
            self._handle_client_failure()
            raise

    # ------------------------------------------------------------------
    def fetch_capture_count(self) -> int:
        mc = self._ensure_client()
        try:
            return mc.fetch_capture_count()
        except Exception:
            self._handle_client_failure()
            raise

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
        mc = self._ensure_client()
        try:
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
        except Exception:
            self._handle_client_failure()
            raise

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
        mc = self._ensure_client()
        try:
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
        except Exception:
            self._handle_client_failure()
            raise

    # ------------------------------------------------------------------
    def push_points_from_pixels(self, pts: List[Point]) -> int:
        """Convert pixel-space points to physical units and upload them."""
        mc = self._ensure_client()
        try:
            capture = mc.fetch_latest_capture_index()
            finder = PointFinder(mc, capture)
            physical = finder.convert_points(pts)
            mc.push_points(physical)
            return len(physical)
        except Exception:
            self._handle_client_failure()
            raise

    # ------------------------------------------------------------------
    def ensure_capture_ready(self, poll: float = 0.5) -> int:
        """Wait until SlideBook finishes capturing and return latest capture index."""

        mc = self._ensure_client()
        self._wait_for_capture(mc, poll)
        count = mc.fetch_capture_count()
        if count <= 0:
            raise ValueError("No captures available")
        return mc.fetch_latest_capture_index()

    # ------------------------------------------------------------------
    def fetch_latest_capture_index(self) -> int:
        """Return the index of the most recent capture."""

        mc = self._ensure_client()
        try:
            return mc.fetch_latest_capture_index()
        except Exception:
            self._handle_client_failure()
            raise

    # ------------------------------------------------------------------
    def iterate_montage_tiles(
        self,
        channel: int = 0,
        z: int = 0,
        max_project: bool = False,
        capture_index: Optional[int] = None,
    ) -> Iterator[MontageTile]:
        """Yield montage tiles with their stage and pixel offsets."""

        if np is None:
            raise RuntimeError("NumPy is required for montage operations")

        mc = self._ensure_client()
        capture = (
            capture_index
            if capture_index is not None
            else mc.fetch_latest_capture_index()
        )

        try:
            util = MontageUtils(
                mc,
                capture_index=capture,
                channel=channel,
                z_plane=z,
                max_project=max_project,
            )
            voxel_x, voxel_y, _ = mc.fetch_voxel_size(capture)
            if not voxel_x or not voxel_y:
                raise ValueError("Microscope reported zero voxel size")

            for arr, (x_um, y_um) in util.fetch_images_with_coords():
                height, width = arr.shape[:2]
                pixel_offset = (
                    float(x_um) / float(voxel_x),
                    float(y_um) / float(voxel_y),
                )
                yield MontageTile(
                    image=arr,
                    stage_offset_um=(float(x_um), float(y_um)),
                    pixel_offset=pixel_offset,
                    size_pixels=(int(width), int(height)),
                )
        except Exception:
            self._handle_client_failure()
            raise

_service: Optional[MicroscopeService] = None


def _configured_connection() -> Tuple[str, int]:
    """Return the configured host/port for the microscope service."""

    host = get_setting("microscope.host", "127.0.0.1")
    port_value = get_setting("microscope.port", 65432)
    try:
        port = int(port_value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError("microscope.port must be an integer") from exc

    return str(host), port


def get_microscope_service() -> MicroscopeService:
    """FastAPI dependency returning a singleton :class:`MicroscopeService`."""

    global _service
    if _service is None:
        host, port = _configured_connection()
        _service = MicroscopeService(host, port)
    return _service
