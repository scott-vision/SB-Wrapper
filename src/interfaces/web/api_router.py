"""FastAPI router exposing the microscope control API."""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException

from ... import api
from ...api import (
    CaptureRequest,
    CelFDriveRequest,
    ObjectiveRequest,
    OptimiseResponse,
    ParfocalityRequest,
    ParfocalityResponse,
    PointModel,
)
from ...services.microscope import MicroscopeService, get_microscope_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/capture/scripts")
async def capture_scripts() -> dict:
    try:
        scripts = api.list_capture_scripts()
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"scripts": scripts}


@router.get("/detect/classes")
async def detect_classes() -> dict:
    try:
        classes = api.list_detection_classes()
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"classes": classes}


@router.get("/microscope/status")
async def microscope_status(
    ms: MicroscopeService = Depends(get_microscope_service),
) -> dict:
    return api.microscope_status(ms)


@router.get("/microscope/points", response_model=List[PointModel])
async def get_microscope_points(
    ms: MicroscopeService = Depends(get_microscope_service),
) -> List[PointModel]:
    try:
        return api.get_microscope_points(ms)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail="Microscope not reachable") from exc


@router.post("/microscope/points")
async def set_microscope_points(
    points: List[PointModel],
    ms: MicroscopeService = Depends(get_microscope_service),
):
    """Upload points to SlideBook."""

    try:
        uploaded = api.set_microscope_points(points, ms)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"uploaded": uploaded}


@router.post("/microscope/start_capture")
async def start_capture(
    req: CaptureRequest, ms: MicroscopeService = Depends(get_microscope_service)
):
    try:
        cid = api.start_capture(req, ms)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    logger.info("capture ID: %s", cid)
    return {"capture": int(cid)}


@router.post("/celfdrive/start")
async def start_celfdrive(
    req: CelFDriveRequest, ms: MicroscopeService = Depends(get_microscope_service)
):
    try:
        return api.start_celfdrive(req, ms)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/microscope/channels")
async def get_microscope_channels(
    ms: MicroscopeService = Depends(get_microscope_service),
):
    try:
        count = api.get_microscope_channels(ms)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail="Microscope not reachable") from exc
    return {"count": int(count)}


@router.get("/microscope/objectives")
async def get_objectives(ms: MicroscopeService = Depends(get_microscope_service)):
    try:
        objs = api.list_objectives(ms)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail="Microscope not reachable") from exc
    return {"objectives": objs}


@router.get("/microscope/objective")
async def get_objective(ms: MicroscopeService = Depends(get_microscope_service)):
    try:
        return api.get_current_objective(ms)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail="Microscope not reachable") from exc


@router.post("/microscope/objective")
async def set_objective(
    req: ObjectiveRequest, ms: MicroscopeService = Depends(get_microscope_service)
):
    try:
        success = api.set_microscope_objective(req, ms)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"success": bool(success)}


@router.get("/microscope/captures")
async def get_capture_count(
    ms: MicroscopeService = Depends(get_microscope_service),
):
    try:
        count = api.get_capture_count(ms)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail="Microscope not reachable") from exc
    return {"count": int(count)}


@router.get("/microscope/image")
async def get_microscope_image(
    channel: int = 0,
    z: int = 0,
    max_project: bool = False,
    capture: Optional[int] = None,
    ms: MicroscopeService = Depends(get_microscope_service),
):
    try:
        return api.get_microscope_image(channel, z, max_project, ms, capture)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail="Microscope not reachable") from exc


@router.get("/microscope/montage")
async def get_microscope_montage(
    channel: int = 0,
    z: int = 0,
    max_project: bool = False,
    cross_corr: bool = False,
    use_features: bool = False,
    ms: MicroscopeService = Depends(get_microscope_service),
):
    try:
        return api.get_microscope_montage(
            channel,
            z,
            max_project,
            cross_corr,
            use_features,
            ms,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail="Microscope not reachable") from exc


@router.get("/microscope/detect")
async def detect_objects(
    channel: int = 0,
    z: int = 0,
    max_project: bool = False,
    sahi: bool = False,
    montage: bool = False,
    ms: MicroscopeService = Depends(get_microscope_service),
):
    try:
        return api.detect_objects(channel, z, max_project, sahi, ms, montage)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail="Microscope not reachable") from exc


@router.get("/points/optimise", response_model=OptimiseResponse)
async def optimise_points(
    algorithm: str = "nn2opt",
    ms: MicroscopeService = Depends(get_microscope_service),
) -> OptimiseResponse:
    try:
        return api.optimise_points(ms, algorithm)
    except ValueError as exc:
        status = 404 if "No points" in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail="Microscope not reachable") from exc


@router.post("/parfocality", response_model=ParfocalityResponse)
async def parfocality(
    req: ParfocalityRequest,
    ms: MicroscopeService = Depends(get_microscope_service),
) -> ParfocalityResponse:
    try:
        return api.parfocality_offset(req, ms)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail="Microscope not reachable") from exc


@router.post("/microscope/findpoints")
async def find_microscope_points(
    points: List[PointModel],
    ms: MicroscopeService = Depends(get_microscope_service),
):
    """Upload points to SlideBook after pixel-to-physical conversion."""

    try:
        uploaded = api.find_microscope_points(points, ms)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"uploaded": uploaded}
