"""Router and helpers for serving the optional static UI."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

PAGE_ROUTES: Tuple[Tuple[str, str], ...] = (
    ("/", "index.html"),
    ("/pointoptimiser", "pointoptimiser.html"),
    ("/pointfinder", "pointfinder.html"),
    ("/celfdrive", "celfdrive.html"),
    ("/detect", "detect.html"),
    ("/guide", "guide.html"),
    ("/objectivechanger", "objectivechanger.html"),
    ("/parfocalityoptimiser", "parfocalityoptimiser.html"),
)


def _load_html(frontend_dir: Path, filename: str) -> str:
    file_path = frontend_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return file_path.read_text(encoding="utf-8")


def create_static_ui_router(frontend_dir: Path) -> APIRouter:
    """Create a router that serves HTML pages from ``frontend_dir``."""

    router = APIRouter()

    for route, filename in PAGE_ROUTES:
        async def serve_page(filename: str = filename) -> str:
            return _load_html(frontend_dir, filename)

        router.add_api_route(
            route,
            serve_page,
            methods=["GET"],
            response_class=HTMLResponse,
        )

    return router


def include_static_ui(app: FastAPI, frontend_dir: Path) -> None:
    """Register static files and HTML routes on ``app``."""

    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
    app.include_router(create_static_ui_router(frontend_dir))
