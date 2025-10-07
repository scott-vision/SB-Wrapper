from __future__ import annotations

"""Utility for converting pixel-space clicks into physical microscope coordinates.

This module defines :class:`PointFinder`, a small helper class that translates
user selected points from an image (pixel units) into physical stage
coordinates in microns.  The conversion is based on the voxel size of the
currently open image in SlideBook, obtained via :meth:`SBAccess.GetVoxelSize`.

Example
-------
>>> with MicroscopeClient(host, port) as mc:
...     finder = PointFinder(mc)
...     stage_point = finder.pixel_to_physical(100, 200)

The code intentionally mirrors the structure of :mod:`SBPointOptimiser` and
provides a simple, object‑oriented abstraction for future expansion.
"""
import logging
from typing import Iterable, List, Tuple, TYPE_CHECKING, Any

# Reuse stage direction logic from montage utilities to ensure consistency
from .SBMontageUtils import _stage_signs

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from .microscope_client import MicroscopeClient, Point
else:  # Fallback definitions to avoid heavy imports at runtime
    MicroscopeClient = Any
    Point = Tuple[float, float, float, float]


class PointFinder:
    """Convert pixel coordinates into SlideBook XYZ points.

    Parameters
    ----------
    mc:
        An active :class:`~microscope_client.MicroscopeClient` connection.
    capture_index:
        Index of the capture whose voxel size should be used.  Defaults to
        ``0`` which is typically the currently open image.
    """

    def __init__(self, mc: MicroscopeClient, capture_index: int = 0) -> None:
        if mc.sb is None:
            raise RuntimeError("MicroscopeClient must be connected before use")
        self._mc = mc
        self._capture = capture_index
        # Fetch voxel size once; values returned are (x, y, z) in microns
        self._vx, self._vy, self._vz = mc.fetch_voxel_size(capture_index)
        logging.info("Voxel size retrieved")

        self.start_x, self.start_y, self.start_z = mc.fetch_image_XYZ(capture_index)
        logging.info("Image location retrieved")
        logging.info(f"{self.start_x}, {self.start_y}, {self.start_z}")

        # Retrieve stage direction multipliers (1 or -1) so pixel coordinates
        # are translated into physical space according to how the microscope
        # stage moves.  This mirrors the behaviour in other utilities such as
        # :mod:`SBMontageUtils`.
        self.stage_x_direction, self.stage_y_direction = _stage_signs()

        self.image_dim_x, self.image_dim_y, self.image_z_planes = mc.fetch_image_dims(capture_index)
        # this will be useful when we scale up to montage
        # self.montage_x, self.montage_y = mc.fetch_montage_position(capture_index, position)
        

    # ------------------------------------------------------------------
    def pixel_to_physical(
        self, x: float, y: float, z: float = 0.0, auxz: float = 0.0
    ) -> Point:
        """Convert a single pixel-space coordinate to microns.

        Parameters
        ----------
        x, y:
            Pixel coordinates relative to the top-left of the image.
        z:
            Z-plane index (0-based) from which the point was selected.
        auxz:
            Auxiliary Z value passed through unchanged.

        Returns
        -------
        Point
            Tuple of ``(x_um, y_um, z_um, auxz)`` suitable for upload to
            SlideBook.
        """
        # start by taking centre of image and subtracting half the w/h,
        # accounting for the direction in which stage coordinates increase
        x_offset = self.start_x - self.image_dim_x/2 * self._vx * self.stage_x_direction
        y_offset = self.start_y - self.image_dim_y/2 * self._vy * self.stage_y_direction

        # calculate location from voxel
        adjusted_x = x_offset + x * self._vx * self.stage_x_direction
        adjusted_y = y_offset + y * self._vy * self.stage_y_direction
        adjusted_z = self.start_z + z * self._vz

        return (adjusted_x, adjusted_y, adjusted_z, auxz)

        

    # ------------------------------------------------------------------
    def convert_points(self, pts: Iterable[Point]) -> List[Point]:
        """Convert an iterable of pixel-space points to microns.

        Each input point is expected to be ``(x, y, z, auxz)`` where ``x`` and
        ``y`` are in pixels.  The returned list contains points expressed in
        microns.
        """

        return [self.pixel_to_physical(*p) for p in pts]
