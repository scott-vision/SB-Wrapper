from __future__ import annotations

"""Workflow helpers for CelFDrive montage detection and capture orchestration."""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from ..SBDetectObjects import ObjectDetector
from ..SBPointFinder import Point
from .microscope import MicroscopeService, MontageTile


@dataclass(frozen=True)
class MontageDetection:
    """Result of running object detection across a microscope montage."""

    boxes: List[Dict[str, float]]
    width: int
    height: int

    def to_dict(self) -> Dict[str, Union[int, List[Dict[str, float]]]]:
        return {"boxes": self.boxes, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class CelFDriveWorkflowResult:
    """Summary returned after completing the CelFDrive workflow."""

    montage_image: Any
    width: int
    height: int
    boxes: List[Dict[str, float]]
    capture_count: int


def _iou(a: Dict[str, float], b: Dict[str, float]) -> float:
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


def _merge_boxes(boxes: Iterable[Dict[str, float]], thresh: float = 0.3) -> List[Dict[str, float]]:
    merged: List[Dict[str, float]] = []
    for box in sorted(boxes, key=lambda b: b.get("confidence", 0), reverse=True):
        if any(_iou(box, existing) > thresh for existing in merged):
            continue
        merged.append(box)
    return merged


def _tile_canvas_dimensions(tiles: List[MontageTile]) -> Tuple[int, int]:
    if not tiles:
        return 0, 0
    max_x = max(float(tile.pixel_offset[0]) + tile.size_pixels[0] for tile in tiles)
    max_y = max(float(tile.pixel_offset[1]) + tile.size_pixels[1] for tile in tiles)
    return int(max_x), int(max_y)


def detect_montage(
    ms: MicroscopeService,
    channel: int = 0,
    z: int = 0,
    max_project: bool = False,
    *,
    use_sahi: bool = False,
    detector: Optional[ObjectDetector] = None,
) -> MontageDetection:
    """Detect objects on each tile of a montage and merge duplicates."""

    ms.ensure_capture_ready()
    tiles = list(ms.iterate_montage_tiles(channel, z, max_project))
    if not tiles:
        return MontageDetection([], 0, 0)

    detector = detector or ObjectDetector()
    boxes: List[Dict[str, float]] = []
    for tile in tiles:
        result = detector.detect(tile.image, use_sahi=use_sahi)
        for box in result.get("boxes", []):
            boxes.append(
                {
                    "x": float(box["x"]) + float(tile.pixel_offset[0]),
                    "y": float(box["y"]) + float(tile.pixel_offset[1]),
                    "width": float(box["width"]),
                    "height": float(box["height"]),
                    "label": box.get("label", ""),
                    "confidence": float(box.get("confidence", 0.0)),
                }
            )

    width, height = _tile_canvas_dimensions(tiles)
    return MontageDetection(_merge_boxes(boxes), width, height)


def run_celfdrive_workflow(
    ms: MicroscopeService,
    *,
    prescan_script: str,
    highres_script: str,
    class_thresholds: Mapping[str, float],
    offsets: Mapping[str, float],
    simulated: bool,
    max_project: bool,
    objective: Optional[Union[str, int]],
    use_sahi: bool = False,
    detector: Optional[ObjectDetector] = None,
) -> CelFDriveWorkflowResult:
    """Execute the CelFDrive workflow end-to-end."""

    if not simulated:
        ms.start_capture(prescan_script)
        ms.ensure_capture_ready()

    detection = detect_montage(
        ms,
        channel=0,
        z=0,
        max_project=max_project,
        use_sahi=use_sahi,
        detector=detector,
    )

    filtered_boxes: List[Dict[str, float]] = []
    if class_thresholds:
        for box in detection.boxes:
            label = box.get("label", "")
            if label in class_thresholds and box.get("confidence", 0) >= class_thresholds[label]:
                filtered_boxes.append(box)

    x_off = offsets.get("x_offset", offsets.get("x", 0.0))
    y_off = offsets.get("y_offset", offsets.get("y", 0.0))
    z_off = offsets.get("z_offset", offsets.get("z", 0.0))

    points: List[Point] = [
        (
            box["x"] + box["width"] / 2 + x_off,
            box["y"] + box["height"] / 2 + y_off,
            float(z_off),
            float(z_off),
        )
        for box in filtered_boxes
    ]
    if points:
        ms.push_points_from_pixels(points)

    if not simulated:
        if objective is not None:
            ms.set_objective(objective)
        ms.start_capture(highres_script)

    montage_image, (width, height) = ms.fetch_stitched_montage(
        0,
        0,
        max_project,
        False,
        False,
    )
    capture_count = ms.fetch_capture_count()

    return CelFDriveWorkflowResult(
        montage_image=montage_image,
        width=int(width),
        height=int(height),
        boxes=filtered_boxes,
        capture_count=int(capture_count),
    )


__all__ = [
    "CelFDriveWorkflowResult",
    "MontageDetection",
    "detect_montage",
    "run_celfdrive_workflow",
]

