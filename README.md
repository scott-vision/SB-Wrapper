# SBSynergyInterface

This repository provides utilities for SlideBook automation packaged under the
``sbs_interface`` namespace. A small FastAPI application exposes the core
functionality as a JSON API.  A lightweight HTML front end lives in the top-
level `frontend/` directory and is entirely optional – any other client can
communicate with the same backend endpoints.  The core microscope operations
are also exposed as plain Python functions in `sbs_interface.api`, allowing
alternate interfaces (e.g. a Tkinter GUI) to drive the backend directly
without going through HTTP.

## Quick start

1. Create and activate the conda environment (includes `numpy`, `matplotlib`,
   `pyyaml`, `bleak`, `fastapi`, `uvicorn`, and `pydantic`):
   ```bash
   conda env create -f environment.yml
   conda activate SBAccess
   ```
2. Launch the web API (and optional bundled UI):
   ```bash
   python -m src
   ```
 The server looks for static files in `frontend/`.  Set the environment
  variable `SBS_FRONTEND_DIR` to serve a different front-end directory or omit
  the folder entirely to use only the backend API.

## Programmatic use

All API endpoints are thin wrappers around functions in `sbs_interface.api`.
These functions can be imported and called directly from any Python front end:

```python
from sbs_interface.api import get_microscope_points
from sbs_interface.services.microscope import get_microscope_service

ms = get_microscope_service()
points = get_microscope_points(ms)
```

## User Guide

The web interface exposes several tools:

- **Detect Objects** – browse microscope images, adjust channels and Z planes, run object detection, and upload detected points.
- **CelFDrive** – select capture scripts, filter detections, and manage high-resolution workflows.
- **Point Finder** – manually pick points on the microscope image.
- **Point Optimiser** – reorder points to minimise stage travel.

Open http://127.0.0.1:65432/guide for more detailed instructions.  Pass
``--host`` and ``--port`` to ``python -m src`` to bind to a different address.

## API Endpoints
- `GET /microscope/status` – basic connection status.
- `GET /microscope/points` – fetch current points from SlideBook.
- `GET /microscope/montage?channel=...&z=...&cross_corr=0|1&use_features=0|1` –
  retrieve a stitched montage image. Set `cross_corr=1` to refine tile offsets
  via phase correlation and `use_features=1` to enable a feature-based
  fallback when correlation is weak.
- `GET /points/optimise?algorithm=...` – reorder points and report metrics. Use
  `algorithm=nn2opt` (default) for a deterministic tour, `algorithm=stochastic`
  for a quicker stochastic search, or `algorithm=anneal` for the most thorough
  but slow simulated annealing. The JSON response includes `computeMs`, the
  time taken to calculate the optimised order in milliseconds.

### Montage utilities

`sbs_interface.SBMontageUtils.MontageUtils` can assemble SlideBook montages into
a single mosaic. The :py:meth:`stitch` method accepts two optional flags:
``cross_correlation`` refines stage‑reported offsets using phase correlation and
``use_features`` enables a feature‑based fallback (requiring OpenCV or
scikit‑image) when correlation peaks are weak or shifts appear implausible.

## Flutter desktop CelFDrive

The `celfdrive_flutter` directory contains a Flutter reimplementation of the
web-based CelFDrive page. It communicates with the same FastAPI backend and can
be built for Windows, macOS, or Linux.

### Run during development
1. Start the Python backend:
   ```bash
   python -m src
   ```
2. Launch the Flutter UI:
   ```bash
   cd celfdrive_flutter
   flutter create .   # only needed the first time
   flutter pub get
   flutter run
   ```

### Build a standalone executable
1. Freeze the backend with [`pyinstaller`](https://www.pyinstaller.org/):
   ```bash
   pyinstaller -F src/interfaces/web/app.py -n sbs_interface_server
   ```
2. Place the resulting `sbs_interface_server` next to the Flutter project.
3. Compile the Flutter desktop app:
   ```bash
   flutter build windows   # or macos/linux
   ```
4. Copy `sbs_interface_server` into the generated runner directory
   (`build/<platform>/runner/Release`).
5. Optionally use a packaging tool such as
   [`msix`](https://pub.dev/packages/msix) or Inno Setup to bundle the folder
   into a single distributable `.exe` that launches both the backend and the
   Flutter UI.
