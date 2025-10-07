"""Tests for the FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from src.interfaces.web.app import create_app
from src import configuration


@pytest.fixture(autouse=True)
def _reset_configuration() -> None:
    """Ensure each test sees a clean configuration state."""

    configuration.reset_overrides()
    configuration.reload_settings()


def test_create_app_without_ui() -> None:
    app = create_app(include_ui=False)
    client = TestClient(app)

    response = client.get("/capture/scripts")
    assert response.status_code == 200
    assert response.json() == {"scripts": []}

    missing = client.get("/")
    assert missing.status_code == 404


def test_create_app_with_ui_missing(tmp_path: Path) -> None:
    frontend_dir = tmp_path / "missing"

    with configuration.override_settings({"frontend": {"directory": str(frontend_dir)}}):
        app = create_app(include_ui=True)
        client = TestClient(app)

        response = client.get("/capture/scripts")
        assert response.status_code == 200

        missing = client.get("/")
        assert missing.status_code == 404


def test_create_app_with_ui_available(tmp_path: Path) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<html><body>Hello</body></html>")
    (frontend / "styles.css").write_text("body {color: black;}")
    with configuration.override_settings({"frontend": {"directory": str(frontend)}}):
        app = create_app(include_ui=True)
        client = TestClient(app)

        response = client.get("/")
        assert response.status_code == 200
        assert "Hello" in response.text

        static_response = client.get("/static/styles.css")
        assert static_response.status_code == 200
        assert "color" in static_response.text
