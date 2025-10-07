from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from .SBDetectObjects import ObjectDetector
from .SBPointOptimiser import RouteOptimizer, distance, Point
from .SBPointFinder import PointFinder
from .services.microscope import MicroscopeService
from .celfdrive import detect_montage


class PointModel(BaseModel):
    x: float
    y: float
    z: float = 0.0
    auxz: float = Field(0.0, alias="auxZ")

    def to_tuple(self) -> Point:
        return (self.x, self.y, self.z, self.auxz)


class ParfocalityRequest(BaseModel):
    """Pairs of pixel coordinates from low- and high-resolution captures."""

    lowres: List[PointModel]
    highres: List[PointModel]


class OffsetModel(BaseModel):
    x: float
    y: float
    z: float


class ParfocalityResponse(BaseModel):
    offsets: List[OffsetModel]
    average: OffsetModel


class OptimiseResponse(BaseModel):
    original_points: List[PointModel] = Field(..., alias="originalPoints")
    ordered_points: List[PointModel] = Field(..., alias="orderedPoints")
    original_length: float = Field(..., alias="originalLength")
    optimised_length: float = Field(..., alias="optimisedLength")
    percent_saved: float = Field(..., alias="percentSaved")
    plot_png: str = Field(..., alias="plotPNG")
    compute_ms: float = Field(..., alias="computeMs")


class CaptureRequest(BaseModel):
    script: str = "Default"


class CelFDriveRequest(BaseModel):
    prescan: str
    highres: str
    classes: Dict[str, float] = Field(default_factory=dict)
    simulated: bool = False
    max_project: bool = True
    objective: Optional[Union[str, int]] = None
    offsets: Dict[str, float] = Field(default_factory=dict)


class ObjectiveRequest(BaseModel):
    objective: Union[str, int]


def microscope_status(ms: MicroscopeService) -> dict:
    return {"connected": ms.check_connection()}


def get_microscope_points(ms: MicroscopeService) -> List[PointModel]:
    pts = ms.fetch_points()
    if not pts:
        raise ValueError("No points found on microscope")
    return [PointModel(x=p[0], y=p[1], z=p[2], auxZ=p[3]) for p in pts]


def set_microscope_points(points: List[PointModel], ms: MicroscopeService) -> int:
    ms.push_points([p.to_tuple() for p in points])
    return len(points)


def start_capture(req: CaptureRequest, ms: MicroscopeService) -> int:
    return ms.start_capture(req.script)


def get_microscope_channels(ms: MicroscopeService) -> int:
    return ms.fetch_num_channels()


def list_objectives(ms: MicroscopeService) -> List[Dict[str, Union[str, int]]]:
    objs = ms.list_objectives()
    return [{"name": n, "position": p} for n, p in objs]


def get_current_objective(ms: MicroscopeService) -> Dict[str, Optional[Union[str, int]]]:
    obj = ms.get_current_objective()
    if not obj:
        return {"name": None, "position": None}
    name, pos = obj
    return {"name": name, "position": pos}


def set_microscope_objective(req: ObjectiveRequest, ms: MicroscopeService) -> bool:
    return ms.set_objective(req.objective)


def _encode_image(arr):
    max_val = arr.max()
    if max_val > 255:
        arr = (255 * (arr.astype(float) / max_val)).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def get_microscope_image(
    channel: int,
    z: int,
    max_project: bool,
    ms: MicroscopeService,
    capture: Optional[int] = None,
) -> dict:
    arr, planes = ms.fetch_display_image(channel, z, max_project, capture)
    if arr.size == 0:
        raise ValueError("Empty image returned")
    b64 = _encode_image(arr)
    return {
        "image": b64,
        "width": int(arr.shape[1]),
        "height": int(arr.shape[0]),
        "planes": planes,
    }


def get_microscope_montage(
    channel: int,
    z: int,
    max_project: bool,
    cross_corr: bool,
    use_features: bool,
    ms: MicroscopeService,
) -> dict:
    arr, (width, height) = ms.fetch_stitched_montage(
        channel,
        z,
        max_project,
        cross_corr,
        use_features,
    )
    if arr.size == 0:
        raise ValueError("Empty image returned")
    b64 = _encode_image(arr)
    return {
        "image": b64,
        "width": int(width),
        "height": int(height),
    }


def detect_objects(
    channel: int,
    z: int,
    max_project: bool,
    sahi: bool,
    ms: MicroscopeService,
    montage: bool = False,
):
    if montage:
        # Propagate the SAHI flag so montage detection can toggle between
        # standard and sliding‑window inference.
        return detect_montage(ms, channel, z, max_project, use_sahi=sahi)
    arr, _ = ms.fetch_display_image(channel, z, max_project)
    if arr.size == 0:
        raise ValueError("Empty image returned")
    detector = ObjectDetector()
    return detector.detect(arr, use_sahi=sahi)


def get_capture_count(ms: MicroscopeService) -> int:
    return ms.fetch_capture_count()


def start_celfdrive(req: CelFDriveRequest, ms: MicroscopeService) -> dict:
    if not req.simulated:
        ms.start_capture(req.prescan)
        # Ensure at least one capture exists and acquisition has finished
        with ms._client() as mc:
            ms._wait_for_capture(mc)
    det = detect_montage(ms, 0, 0, req.max_project)
    boxes = [
        b
        for b in det["boxes"]
        if b["label"] in req.classes
        and b["confidence"] >= req.classes[b["label"]]
    ]
    x_off = req.offsets.get("x_offset", req.offsets.get("x", 0.0))
    y_off = req.offsets.get("y_offset", req.offsets.get("y", 0.0))
    z_off = req.offsets.get("z_offset", req.offsets.get("z", 0.0))
    points = [
        PointModel(
            x=b["x"] + b["width"] / 2 + x_off,
            y=b["y"] + b["height"] / 2 + y_off,
            z=z_off,
            auxZ=z_off,
        )
        for b in boxes
    ]
    if points:
        find_microscope_points(points, ms)
    if not req.simulated:
        if req.objective is not None:
            ms.set_objective(req.objective)
        ms.start_capture(req.highres)
    montage = get_microscope_montage(
        0, 0, req.max_project, False, False, ms
    )
    capture_count = get_capture_count(ms)
    return {
        "montage": montage["image"],
        "width": montage["width"],
        "height": montage["height"],
        "boxes": boxes,
        "captureCount": int(capture_count),
    }


def _plot_routes(points: List[Point], order: List[int]) -> str:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    axes[0].plot(xs + [xs[0]], ys + [ys[0]], "o-")
    axes[0].set_title("Original")
    ordered = [points[i] for i in order]
    ox = [p[0] for p in ordered]
    oy = [p[1] for p in ordered]
    axes[1].plot(ox + [ox[0]], oy + [oy[0]], "o-")
    axes[1].set_title("Optimised")
    for ax in axes:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def optimise_points(
    ms: MicroscopeService,
    algorithm: str = "nn2opt",
) -> OptimiseResponse:
    pts = ms.fetch_points()
    if not pts:
        raise ValueError("No points found on microscope")
    if len(pts) < 2:
        raise ValueError("Need at least two points")

    optimiser = RouteOptimizer(pts)
    start = time.perf_counter()
    if algorithm == "anneal":
        order = optimiser.optimise_sa()
    elif algorithm == "stochastic":
        order = optimiser.optimise_stochastic()
    else:
        order = optimiser.optimise()
    compute_ms = (time.perf_counter() - start) * 1000.0
    original_length = sum(
        distance(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))
    )
    optimised_length = optimiser._tour_length(order)
    percent_saved = 100.0 * (1 - optimised_length / original_length)
    plot_data = _plot_routes(pts, order)
    ordered_pts = [
        PointModel(x=pts[i][0], y=pts[i][1], z=pts[i][2], auxZ=pts[i][3])
        for i in order
    ]
    original_pts = [PointModel(x=p[0], y=p[1], z=p[2], auxZ=p[3]) for p in pts]

    return OptimiseResponse(
        originalPoints=original_pts,
        orderedPoints=ordered_pts,
        originalLength=original_length,
        optimisedLength=optimised_length,
        percentSaved=percent_saved,
        plotPNG=plot_data,
        computeMs=compute_ms,
    )


def parfocality_offset(req: ParfocalityRequest, ms: MicroscopeService) -> ParfocalityResponse:
    """Calculate average parfocality offset between two captures."""

    if len(req.lowres) != len(req.highres) or not req.lowres:
        raise ValueError("Point lists must be equal length and non-empty")

    with ms._client() as mc:
        finder_low = PointFinder(mc, 0)
        finder_high = PointFinder(mc, 1)
        offsets: List[OffsetModel] = []
        for p1, p2 in zip(req.lowres, req.highres):
            x1, y1, z1, _ = finder_low.pixel_to_physical(p1.x, p1.y, p1.z, p1.auxz)
            x2, y2, z2, _ = finder_high.pixel_to_physical(p2.x, p2.y, p2.z, p2.auxz)
            offsets.append(OffsetModel(x=x2 - x1, y=y2 - y1, z=z2 - z1))
    avg_x = sum(o.x for o in offsets) / len(offsets)
    avg_y = sum(o.y for o in offsets) / len(offsets)
    avg_z = sum(o.z for o in offsets) / len(offsets)
    return ParfocalityResponse(
        offsets=offsets,
        average=OffsetModel(x=avg_x, y=avg_y, z=avg_z),
    )


def find_microscope_points(points: List[PointModel], ms: MicroscopeService) -> int:
    return ms.push_points_from_pixels([p.to_tuple() for p in points])


def list_capture_scripts() -> List[str]:
    base = Path(
        "C:/ProgramData/Intelligent Imaging Innovations/SlideBook 2025/Users/Default User"
    )
    if not base.exists():
        return []
    return [p.name.replace(".Cap.prefs", "") for p in base.glob("*.Cap.prefs")]


def list_detection_classes() -> List[str]:
    detector = ObjectDetector()
    model = detector._load_model(False)
    names = getattr(model, "names", [])
    if isinstance(names, dict):
        return [names[k] for k in sorted(names)]
    return list(names)
