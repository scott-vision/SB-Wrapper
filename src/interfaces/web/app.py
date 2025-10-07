"""FastAPI application configuration for the SBS interface."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from fastapi import FastAPI

from .api_router import router as api_router
from .static_ui import include_static_ui

logging.basicConfig(level=logging.INFO)


def _resolve_frontend_dir() -> Path:
    """Return the configured frontend directory path."""

    configured = os.environ.get("SBS_FRONTEND_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "frontend"


def create_app(include_ui: bool = False) -> FastAPI:
    """Create the FastAPI application.

    Parameters
    ----------
    include_ui:
        When ``True`` the optional static UI router is included if the
        frontend directory exists.
    """

    app = FastAPI(title="SBSynergy Web")
    app.include_router(api_router)

    if include_ui:
        frontend_dir = _resolve_frontend_dir()
        if frontend_dir.exists():
            include_static_ui(app, frontend_dir)

    return app


# Default application instance for ``uvicorn src.interfaces.web.app:app``.
app = create_app(include_ui=True)


def main() -> None:
    """Launch the FastAPI application from the command line."""

    parser = argparse.ArgumentParser(description="Launch the SBS interface")
    parser.add_argument(
        "--host", default="127.0.0.1", help="IP address to bind (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", default=8000, type=int, help="Port to bind (default: 8000)"
    )
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    try:
        import webbrowser

        webbrowser.open(url)
    except Exception:  # pragma: no cover - opening a browser is best-effort
        logging.getLogger(__name__).info("Unable to open browser for %s", url)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
