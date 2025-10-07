from __future__ import annotations

"""Utilities for reading and stitching SlideBook montages.

This module builds upon :class:`~sbs_interface.microscope_client.MicroscopeClient`
by providing helpers to query montage layout, compute positional offsets and
assemble a stitched image.
"""

from typing import Dict, List, Tuple, TYPE_CHECKING, Any, Optional
import logging
import math
import re
from pathlib import Path

try:  # pragma: no cover - optional dependency
    import numpy as np
except ImportError:  # pragma: no cover - allow documentation building without NumPy
    np = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .microscope_client import MicroscopeClient
else:  # Fallback to satisfy static checkers when module unavailable
    MicroscopeClient = Any  # type: ignore[assignment]


# ----------------------------------------------------------------------
_STAGE_SIGNS: Optional[Tuple[int, int]] = None


def _stage_signs() -> Tuple[int, int]:
    """Return (x_sign, y_sign) describing stage coordinate direction.

    The signs are read from SlideBook's ``SlideBookHardwareProperties.dat``
    file if available.  Values of ``-1`` indicate that stage coordinates
    decrease when moving the field right/down.  If the file cannot be read
    default signs of ``(1, 1)`` are returned.
    """

    global _STAGE_SIGNS
    if _STAGE_SIGNS is None:
        right, down = 1, 1
        try:
            path = Path(
                r"C:/ProgramData/Intelligent Imaging Innovations/SlideBook 2025/Global Preferences/SlideBookHardwareProperties.dat"
            )
            text = path.read_text(errors="ignore")
            m = re.search(r"{MoveFieldRightSign}.*?Int\s*([-\d]+)", text, re.S)
            if m:
                right = int(m.group(1)) or 1
            m = re.search(r"{MoveFieldDownSign}.*?Int\s*([-\d]+)", text, re.S)
            if m:
                down = int(m.group(1)) or 1
        except Exception:
            pass
        _STAGE_SIGNS = (1 if right >= 0 else -1, 1 if down >= 0 else -1)
    return _STAGE_SIGNS


# ----------------------------------------------------------------------
def _feature_offset(a: np.ndarray, b: np.ndarray) -> Tuple[int, int]:
    """Return ``(dy, dx)`` shift aligning ``b`` onto ``a`` using features."""
    try:  # Prefer OpenCV if available
        import cv2

        a8 = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        b8 = cv2.normalize(b, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        detector = cv2.ORB_create()
        k1, d1 = detector.detectAndCompute(a8, None)
        k2, d2 = detector.detectAndCompute(b8, None)
        if d1 is None or d2 is None:
            return 0, 0
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = matcher.knnMatch(d1, d2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append((k1[m.queryIdx].pt, k2[m.trainIdx].pt))
        if len(good) < 3:
            return 0, 0
        src = np.float32([p1 for p1, _ in good])
        dst = np.float32([p2 for _, p2 in good])
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
        if M is None:
            return 0, 0
        dx = M[0, 2]
        dy = M[1, 2]
        return int(round(dy)), int(round(dx))
    except Exception:  # pragma: no cover - requires optional deps
        try:
            from skimage.feature import ORB, match_descriptors
            from skimage.measure import ransac
            from skimage.transform import AffineTransform

            a_norm = a.astype(float)
            b_norm = b.astype(float)
            a_norm = (a_norm - a_norm.min()) / (a_norm.ptp() or 1.0)
            b_norm = (b_norm - b_norm.min()) / (b_norm.ptp() or 1.0)

            orb = ORB(n_keypoints=200)
            orb.detect_and_extract(a_norm)
            k1, d1 = orb.keypoints, orb.descriptors
            orb.detect_and_extract(b_norm)
            k2, d2 = orb.keypoints, orb.descriptors
            matches = match_descriptors(d1, d2, max_ratio=0.75)
            if matches.shape[0] < 3:
                return 0, 0
            src = k1[matches[:, 0]][:, ::-1]
            dst = k2[matches[:, 1]][:, ::-1]
            model, _ = ransac(
                (src, dst),
                AffineTransform,
                min_samples=3,
                residual_threshold=2,
                max_trials=1000,
            )
            dx, dy = model.translation
            return int(round(dy)), int(round(dx))
        except Exception:  # pragma: no cover - feature modules missing
            return 0, 0


class MontageUtils:
    """High level helper for SlideBook montages.

    Parameters
    ----------
    mc:
        Active :class:`~sbs_interface.microscope_client.MicroscopeClient`.
    capture_index:
        Index of the capture containing the montage.
    channel:
        Channel index to fetch when retrieving image data.
    z_plane:
        Z plane index to fetch when retrieving image data.
    max_project:
        If ``True`` and the capture contains multiple Z planes, each tile will
        be maximum projected across all planes before stitching.
    """

    def __init__(
        self,
        mc: MicroscopeClient,
        capture_index: int = 0,
        channel: int = 0,
        z_plane: int = 0,
        max_project: bool = False,
    ) -> None:
        if mc.sb is None:
            raise RuntimeError("MicroscopeClient must be connected before use")
        if np is None:
            raise RuntimeError("NumPy is required for montage utilities")

        self._mc = mc
        self.capture = capture_index
        self.channel = channel
        self.z_plane = z_plane
        self.max_project = max_project

        self._map: Optional[np.ndarray] = None
        self._offsets: Optional[Dict[int, Tuple[float, float]]] = None

        cols_px, rows_px, planes = mc.fetch_image_dims(capture_index)
        vx, vy, _ = mc.fetch_voxel_size(capture_index)
        self._tile_cols = int(cols_px)
        self._tile_rows = int(rows_px)
        self._planes = int(planes)
        self._vx = float(vx)
        self._vy = float(vy)
        self._tile_width_um = self._tile_cols * self._vx
        self._tile_height_um = self._tile_rows * self._vy
        self._planes = int(planes)

    # ------------------------------------------------------------------
    def build_map(self) -> np.ndarray:
        """Return a 2D array describing montage tile order."""
        if self._map is None:
            num = self._mc.fetch_num_positions(self.capture)

            # Some SlideBook captures with a single tile report zero montage
            # positions.  In this situation treat the montage as a trivial
            # 1x1 grid containing only tile 0.
            if num <= 1:
                self._map = np.array([[0]], dtype=int)
            else:
                coords = [
                    self._mc.fetch_montage_position(self.capture, idx)
                    for idx in range(num)
                ]
                max_row = max(r for r, _ in coords)
                max_col = max(c for _, c in coords)
                arr = np.full((max_row + 1, max_col + 1), -1, dtype=int)
                for idx, (r, c) in enumerate(coords):
                    arr[r, c] = idx
                self._map = arr
            logging.info("Montage map built: %s", self._map)
        return self._map

    # ------------------------------------------------------------------
    def compute_offsets(self) -> Dict[int, Tuple[float, float]]:
        """Return mapping of position index to (x_um, y_um) offsets."""
        TL_x, TL_y, TL_z = self._mc.fetch_image_XYZ(
            capture_number=self.capture,
            image_number=0,
            z_plane_index=self.z_plane
        ) 

        if self._offsets is None:
            m = self.build_map()
            offsets: Dict[int, Tuple[float, float]] = {}
            x_sign, y_sign = _stage_signs()
            for r in range(m.shape[0]):
                for c in range(m.shape[1]):
                    idx = int(m[r, c])
                    if idx >= 0:
                        Pos_x, Pos_y, Pos_z = self._mc.fetch_image_XYZ(
                            capture_number=self.capture,
                            image_number=idx,
                            z_plane_index=self.z_plane
                        )
                        # Preserve the sign of the stage‑reported offsets so
                        # that montages acquired in orientations other than
                        # top‑left still stitch correctly. Apply stage
                        # direction signs so that offsets reflect the field
                        # orientation regardless of how coordinates change
                        # with stage motion. Final dimensions are normalised
                        # later by ``stitch``.
                        offsets[idx] = (
                            (Pos_x - TL_x) * x_sign,
                            (Pos_y - TL_y) * y_sign,
                        )
            self._offsets = offsets
            logging.info("Offset map computed: %s", offsets)
        return self._offsets

    # ------------------------------------------------------------------
    def _fetch_image_at_position(self, pos: int) -> np.ndarray:
        """Internal helper to retrieve one image tile."""
        sb = self._mc.sb
        if sb is None:
            raise RuntimeError("Not connected to SlideBook")
        slide_id = sb.GetCurrentSlideId()
        sb.SetTargetSlide(slide_id)

        # currently there is an issue in readplane buffer that means we have to use pos in the timepoint arg

        if self.max_project and self._planes > 1:
            buf = sb.ReadImagePlaneBuf(self.capture, 0, pos, 0, self.channel)
            arr = np.frombuffer(buf, dtype=np.uint16).reshape(
                self._tile_rows, self._tile_cols
            )
            for zp in range(1, self._planes):
                buf = sb.ReadImagePlaneBuf(self.capture, 0, pos, zp, self.channel)
                plane = np.frombuffer(buf, dtype=np.uint16).reshape(
                    self._tile_rows, self._tile_cols
                )
                arr = np.maximum(arr, plane)
        else:
            buf = sb.ReadImagePlaneBuf(
                self.capture, 0, pos, self.z_plane, self.channel
            )
            arr = np.frombuffer(buf, dtype=np.uint16).reshape(
                self._tile_rows, self._tile_cols
            )
        return arr

    # ------------------------------------------------------------------
    def fetch_images_with_coords(
        self,
    ) -> List[Tuple[np.ndarray, Tuple[float, float]]]:
        """Return list of ``(image, (x_um, y_um))`` before stitching."""
        offsets = self.compute_offsets()
        images = []
        for idx in sorted(offsets):
            logging.info(f"index {idx}")
            arr = self._fetch_image_at_position(idx)
            images.append((arr, offsets[idx]))
        return images

    # ------------------------------------------------------------------
    def stitch(
        self, cross_correlation: bool = False, use_features: bool = False
    ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[float, float]]:
        """Return a stitched montage and its final size.

        Parameters
        ----------
        cross_correlation:
            When ``True`` an additional phase cross‑correlation step is
            performed on the overlapping regions of neighbouring tiles.  This
            refines the stage‑reported offsets to improve the final alignment.
        use_features:
            If ``True`` and cross‑correlation produces a weak peak or an
            implausibly large shift, attempt a feature based registration as a
            fallback.  This requires either OpenCV or scikit‑image to be
            installed.
        """
        # Ensure map/offets cached
        m = self.build_map()
        offsets = self.compute_offsets().copy()

        # Fetch all tile images keyed by index so that we can perform
        # pair‑wise refinements when requested.
        images: Dict[int, np.ndarray] = {}
        for idx in sorted(offsets):
            images[idx] = self._fetch_image_at_position(idx)

        if cross_correlation:
            MAX_SHIFT_UM = 20.0

            # Helper performing FFT based cross correlation returning the
            # required (y, x) shift to align ``b`` onto ``a`` along with a
            # normalised peak strength.
            def _xcorr_offset(a: np.ndarray, b: np.ndarray) -> Tuple[int, int, float]:
                a = a.astype(np.float32)
                b = b.astype(np.float32)
                a -= a.mean()
                b -= b.mean()
                shape = (
                    a.shape[0] + b.shape[0] - 1,
                    a.shape[1] + b.shape[1] - 1,
                )
                f1 = np.fft.rfftn(a, shape)
                f2 = np.fft.rfftn(b, shape)
                cc = np.fft.irfftn(f1 * np.conj(f2), shape)
                peak = float(cc.max())
                denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
                strength = peak / denom if denom else 0.0
                y, x = np.unravel_index(np.argmax(cc), cc.shape)
                return y - b.shape[0] + 1, x - b.shape[1] + 1, strength


            # Horizontal neighbours
            for r in range(m.shape[0]):
                for c in range(m.shape[1] - 1):
                    left = int(m[r, c])
                    right = int(m[r, c + 1])
                    if left < 0 or right < 0:
                        continue
                    x_l, y_l = offsets[left]
                    x_r, y_r = offsets[right]
                    shift_px = int(round((x_r - x_l) / self._vx))
                    overlap = self._tile_cols - shift_px
                    if overlap <= 0:
                        continue
                    patch_w = min(self._tile_cols, max(overlap * 2, 1))
                    a = images[left][:, -patch_w:]
                    b = images[right][:, :patch_w]
                    dy, dx, strength = _xcorr_offset(a, b)
                    x_new = x_l + (self._tile_cols - patch_w + dx) * self._vx
                    y_new = y_l + dy * self._vy
                    if use_features and (
                        strength < 0.1
                        or abs(x_new - x_r) > MAX_SHIFT_UM
                        or abs(y_new - y_r) > MAX_SHIFT_UM
                    ):
                        dy, dx = _feature_offset(a, b)
                        x_new = x_l + (self._tile_cols - patch_w + dx) * self._vx
                        y_new = y_l + dy * self._vy
                    x_new = max(x_r - MAX_SHIFT_UM, min(x_r + MAX_SHIFT_UM, x_new))
                    y_new = max(y_r - MAX_SHIFT_UM, min(y_r + MAX_SHIFT_UM, y_new))
                    offsets[right] = (x_new, y_new)

            # Vertical neighbours
            for r in range(m.shape[0] - 1):
                for c in range(m.shape[1]):
                    top = int(m[r, c])
                    bottom = int(m[r + 1, c])
                    if top < 0 or bottom < 0:
                        continue
                    x_t, y_t = offsets[top]
                    x_b, y_b = offsets[bottom]
                    shift_px = int(round((y_b - y_t) / self._vy))
                    overlap = self._tile_rows - shift_px
                    if overlap <= 0:
                        continue
                    patch_h = min(self._tile_rows, max(overlap * 2, 1))
                    a = images[top][-patch_h:, :]
                    b = images[bottom][:patch_h, :]
                    dy, dx, strength = _xcorr_offset(a, b)
                    x_new = x_t + dx * self._vx
                    y_new = y_t + (self._tile_rows - patch_h + dy) * self._vy
                    if use_features and (
                        strength < 0.1
                        or abs(x_new - x_b) > MAX_SHIFT_UM
                        or abs(y_new - y_b) > MAX_SHIFT_UM
                    ):
                        dy, dx = _feature_offset(a, b)
                        x_new = x_t + dx * self._vx
                        y_new = y_t + (self._tile_rows - patch_h + dy) * self._vy
                    x_new = max(x_b - MAX_SHIFT_UM, min(x_b + MAX_SHIFT_UM, x_new))
                    y_new = max(y_b - MAX_SHIFT_UM, min(y_b + MAX_SHIFT_UM, y_new))
                    offsets[bottom] = (x_new, y_new)

        # Convert mapping back to list of ``(image, (x_um, y_um))`` for
        # stitching.
        img_offsets = [
            (images[idx], offsets[idx])
            for idx in sorted(offsets)
        ]

        # Determine output size from offsets rather than tile count so that
        # stage overlaps reduce the final dimensions correctly.  Cross
        # correlation can shift tiles beyond the origin so track both the
        # minimum and maximum pixel coordinates to ensure the stitched canvas
        # fully encompasses all tiles.
        x_positions: List[int] = []
        y_positions: List[int] = []
        for _arr, (x_um, y_um) in img_offsets:
            x_positions.append(int(round(x_um / self._vx)))
            y_positions.append(int(round(y_um / self._vy)))

        min_x = min(x_positions)
        min_y = min(y_positions)
        width_px = max(x_positions) - min_x + self._tile_cols
        height_px = max(y_positions) - min_y + self._tile_rows

        # Accumulate weighted tiles and normalise by total weight to taper
        # overlaps towards tile edges.
        stitched = np.zeros((height_px, width_px), dtype=np.float32)
        weights = np.zeros((height_px, width_px), dtype=np.float32)

        # Pre-compute a weighting mask for one tile based on distance from
        # edges.  Values near the edge contribute less to the final stitched
        # image than interior pixels.
        y_idx = np.minimum(np.arange(self._tile_rows), np.arange(self._tile_rows)[::-1])
        x_idx = np.minimum(np.arange(self._tile_cols), np.arange(self._tile_cols)[::-1])
        # Normalise distances and clip to avoid zero weights which would lead
        # to division by zero when normalising.
        wy = y_idx / (self._tile_rows / 2.0)
        wx = x_idx / (self._tile_cols / 2.0)
        mask = np.outer(wy, wx)
        mask = np.clip(mask, 0.01, 1.0)

        for arr, (x_um, y_um) in img_offsets:
            x_px = int(round(x_um / self._vx)) - min_x
            y_px = int(round(y_um / self._vy)) - min_y
            tile = arr.astype(np.float32)
            stitched[y_px : y_px + self._tile_rows, x_px : x_px + self._tile_cols] += tile * mask
            weights[y_px : y_px + self._tile_rows, x_px : x_px + self._tile_cols] += mask

        result = np.zeros_like(stitched)
        np.divide(stitched, weights, out=result, where=weights > 0)
        result = result.round().astype(np.uint16)

        pixel_dims = (width_px, height_px)
        micron_dims = (width_px * self._vx, height_px * self._vy)
        return result, pixel_dims, micron_dims


__all__ = ["MontageUtils"]

