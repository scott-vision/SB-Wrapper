import os
from pathlib import Path
from typing import List, Optional
import logging

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .api import (
    CaptureRequest,
    CelFDriveRequest,
    ObjectiveRequest,
    OptimiseResponse,
    PointModel,
    ParfocalityRequest,
    ParfocalityResponse,
)
from . import api
from .services.microscope import MicroscopeService, get_microscope_service

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="SBSynergy Web")

# Locate the optional frontend directory.  By default this looks for a
# top-level ``frontend`` folder, but an explicit path can be provided via
# the ``SBS_FRONTEND_DIR`` environment variable.  If the directory is not
# present the API can still be used independently by other front ends.
FRONTEND_DIR = Path(
    os.environ.get("SBS_FRONTEND_DIR", Path(__file__).resolve().parents[1] / "frontend")
)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return index_file.read_text(encoding="utf-8")


@app.get("/pointoptimiser", response_class=HTMLResponse)
async def pointoptimiser() -> str:
    po_file = FRONTEND_DIR / "pointoptimiser.html"
    if not po_file.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return po_file.read_text(encoding="utf-8")


@app.get("/pointfinder", response_class=HTMLResponse)
async def pointfinder() -> str:
    pf_file = FRONTEND_DIR / "pointfinder.html"
    if not pf_file.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return pf_file.read_text(encoding="utf-8")


@app.get("/celfdrive", response_class=HTMLResponse)
async def celfdrive() -> str:
    cf_file = FRONTEND_DIR / "celfdrive.html"
    if not cf_file.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return cf_file.read_text(encoding="utf-8")


@app.get("/detect", response_class=HTMLResponse)
async def detect() -> str:
    det_file = FRONTEND_DIR / "detect.html"
    if not det_file.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return det_file.read_text(encoding="utf-8")


@app.get("/guide", response_class=HTMLResponse)
async def guide() -> str:
    guide_file = FRONTEND_DIR / "guide.html"
    if not guide_file.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return guide_file.read_text(encoding="utf-8")


@app.get("/objectivechanger", response_class=HTMLResponse)
async def objective_changer() -> str:
    obj_file = FRONTEND_DIR / "objectivechanger.html"
    if not obj_file.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return obj_file.read_text(encoding="utf-8")


@app.get("/parfocalityoptimiser", response_class=HTMLResponse)
async def parfocality_optimiser() -> str:
    pf_file = FRONTEND_DIR / "parfocalityoptimiser.html"
    if not pf_file.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return pf_file.read_text(encoding="utf-8")


@app.get("/capture/scripts")
async def capture_scripts() -> dict:
    try:
        scripts = api.list_capture_scripts()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"scripts": scripts}


@app.get("/detect/classes")
async def detect_classes() -> dict:
    try:
        classes = api.list_detection_classes()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"classes": classes}


@app.get("/microscope/status")
async def microscope_status(
    ms: MicroscopeService = Depends(get_microscope_service),
) -> dict:
    return api.microscope_status(ms)


@app.get("/microscope/points", response_model=List[PointModel])
async def get_microscope_points(
    ms: MicroscopeService = Depends(get_microscope_service),
) -> List[PointModel]:
    try:
        return api.get_microscope_points(ms)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=503, detail="Microscope not reachable")


@app.post("/microscope/points")
async def set_microscope_points(
    points: List[PointModel],
    ms: MicroscopeService = Depends(get_microscope_service),
):
    """Upload points to SlideBook."""

    try:
        uploaded = api.set_microscope_points(points, ms)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"uploaded": uploaded}


@app.post("/microscope/start_capture")
async def start_capture(
    req: CaptureRequest, ms: MicroscopeService = Depends(get_microscope_service)
):
    try:
        cid = api.start_capture(req, ms)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    logging.info(f"capture ID: {cid}")
    return {"capture": int(cid)}


@app.post("/celfdrive/start")
async def start_celfdrive(
    req: CelFDriveRequest, ms: MicroscopeService = Depends(get_microscope_service)
):
    try:
        return api.start_celfdrive(req, ms)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/microscope/channels")
async def get_microscope_channels(
    ms: MicroscopeService = Depends(get_microscope_service),
):
    try:
        count = api.get_microscope_channels(ms)
    except Exception:
        raise HTTPException(status_code=503, detail="Microscope not reachable")
    return {"count": int(count)}


@app.get("/microscope/objectives")
async def get_objectives(ms: MicroscopeService = Depends(get_microscope_service)):
    try:
        objs = api.list_objectives(ms)
    except Exception:
        raise HTTPException(status_code=503, detail="Microscope not reachable")
    return {"objectives": objs}


@app.get("/microscope/objective")
async def get_objective(ms: MicroscopeService = Depends(get_microscope_service)):
    try:
        return api.get_current_objective(ms)
    except Exception:
        raise HTTPException(status_code=503, detail="Microscope not reachable")


@app.post("/microscope/objective")
async def set_objective(
    req: ObjectiveRequest, ms: MicroscopeService = Depends(get_microscope_service)
):
    try:
        success = api.set_microscope_objective(req, ms)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"success": bool(success)}


@app.get("/microscope/captures")
async def get_capture_count(
    ms: MicroscopeService = Depends(get_microscope_service),
):
    try:
        count = api.get_capture_count(ms)
    except Exception:
        raise HTTPException(status_code=503, detail="Microscope not reachable")
    return {"count": int(count)}


@app.get("/microscope/image")
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
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=503, detail="Microscope not reachable")


@app.get("/microscope/montage")
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
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=503, detail="Microscope not reachable")


@app.get("/microscope/detect")
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
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=503, detail="Microscope not reachable")


@app.get("/points/optimise", response_model=OptimiseResponse)
async def optimise_points(
    algorithm: str = "nn2opt",
    ms: MicroscopeService = Depends(get_microscope_service),
) -> OptimiseResponse:
    try:
        return api.optimise_points(ms, algorithm)
    except ValueError as exc:
        status = 404 if "No points" in str(exc) else 400
        raise HTTPException(status_code=status, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=503, detail="Microscope not reachable")


@app.post("/parfocality", response_model=ParfocalityResponse)
async def parfocality(
    req: ParfocalityRequest,
    ms: MicroscopeService = Depends(get_microscope_service),
) -> ParfocalityResponse:
    try:
        return api.parfocality_offset(req, ms)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=503, detail="Microscope not reachable")


@app.post("/microscope/findpoints")
async def find_microscope_points(
    points: List[PointModel],
    ms: MicroscopeService = Depends(get_microscope_service),
):
    """Upload points to SlideBook after pixel-to-physical conversion."""

    try:
        uploaded = api.find_microscope_points(points, ms)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"uploaded": uploaded}


def main() -> None:
    """Launch the FastAPI application.

    The server address can be customised via ``--host`` and ``--port``
    command-line arguments.  The default port matches the microscope
    connection port so the web interface can proxy requests on the same
    number.
    """

    import argparse
    import uvicorn
    import webbrowser

    parser = argparse.ArgumentParser(description="Launch the SBS interface")
    parser.add_argument(
        "--host", default="127.0.0.1", help="IP address to bind (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", default=8000, type=int, help="Port to bind (default: 8000)"
    )
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    webbrowser.open(url)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
