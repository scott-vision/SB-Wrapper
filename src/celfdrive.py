from __future__ import annotations

"""CelFDrive backend helpers for montage-based detection."""

from typing import Dict, List

import numpy as np

from .SBDetectObjects import ObjectDetector
from .SBMontageUtils import MontageUtils
from .services.microscope import MicroscopeService


# ---------------------------------------------------------------------------
def _iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Return intersection over union of two boxes."""
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["width"], ay1 + a["height"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["width"], by1 + b["height"]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / union if union else 0.0


# ---------------------------------------------------------------------------
def _merge_boxes(boxes: List[Dict[str, float]], thresh: float = 0.3) -> List[Dict[str, float]]:
    """Remove overlapping boxes keeping the highest confidence ones."""
    merged: List[Dict[str, float]] = []
    for box in sorted(boxes, key=lambda b: b.get("confidence", 0), reverse=True):
        if any(_iou(box, m) > thresh for m in merged):
            continue
        merged.append(box)
    return merged


# ---------------------------------------------------------------------------
def detect_montage(
    ms: MicroscopeService,
    channel: int = 0,
    z: int = 0,
    max_project: bool = False,
    use_sahi: bool = False,
) -> Dict[str, object]:
    """Detect objects on each tile of a montage and merge duplicates.

    Parameters
    ----------
    ms:
        Active :class:`MicroscopeService` used for image retrieval.
    channel, z:
        Image selection parameters forwarded to SlideBook.
    max_project:
        If ``True`` and the capture contains multiple Z planes, a per-tile
        maximum projection is performed before detection.
    use_sahi:
        Enable :mod:`sahi` sliding-window inference for each tile when ``True``.
    """

    with ms._client() as mc:
        capture = mc.fetch_latest_capture_index()
        util = MontageUtils(
            mc,
            capture_index=capture,
            channel=channel,
            z_plane=z,
            max_project=max_project,
        )
        offsets = util.compute_offsets()
        vx, vy = util._vx, util._vy
        tile_w, tile_h = util._tile_cols, util._tile_rows

        detector = ObjectDetector()
        boxes: List[Dict[str, float]] = []

        for idx in sorted(offsets):
            arr = util._fetch_image_at_position(idx)
            result = detector.detect(arr, use_sahi=use_sahi)
            x_um, y_um = offsets[idx]
            off_x = x_um / vx
            off_y = y_um / vy
            for b in result["boxes"]:
                boxes.append(
                    {
                        # Ensure all numeric values are plain Python floats
                        # to avoid JSON encoding errors when FastAPI attempts
                        # to serialise NumPy scalar types.
                        "x": float(b["x"]) + float(off_x),
                        "y": float(b["y"]) + float(off_y),
                        "width": float(b["width"]),
                        "height": float(b["height"]),
                        "label": b.get("label", ""),
                        "confidence": float(b.get("confidence", 0.0)),
                    }
                )

        width = int(max((off[0] / vx) + tile_w for off in offsets.values()))
        height = int(max((off[1] / vy) + tile_h for off in offsets.values()))

    return {"boxes": _merge_boxes(boxes), "width": width, "height": height}


__all__ = ["detect_montage"]
