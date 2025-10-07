import sys
import types
from pathlib import Path

real_numpy = sys.modules.get("numpy")
sys.modules["numpy"] = types.SimpleNamespace()
sys.modules["sbs_interface.SBDetectObjects"] = types.SimpleNamespace(ObjectDetector=object)

pkg = types.ModuleType("sbs_interface")
pkg.__path__ = [str(Path("src").resolve())]
sys.modules["sbs_interface"] = pkg

from sbs_interface.services import celfdrive_workflow
from sbs_interface.services.microscope import MontageTile

if real_numpy is None:
    sys.modules.pop("numpy", None)
else:
    sys.modules["numpy"] = real_numpy


def test_run_celfdrive_workflow(monkeypatch):
    boxes = [
        {"x": 0, "y": 0, "width": 10, "height": 10, "label": "cell", "confidence": 0.9},
        {"x": 20, "y": 20, "width": 5, "height": 5, "label": "noise", "confidence": 0.4},
    ]

    class FakeDetector:
        def detect(self, image, use_sahi=False):
            return {"boxes": boxes}

    monkeypatch.setattr(celfdrive_workflow, "ObjectDetector", lambda: FakeDetector())

    class FakeService:
        def __init__(self):
            self.started = []
            self.ensured = 0
            self.points = []
            self.objective = None

        def start_capture(self, script):
            self.started.append(script)

        def ensure_capture_ready(self):
            self.ensured += 1
            return 0

        def iterate_montage_tiles(self, channel=0, z=0, max_project=False, capture_index=None):
            yield MontageTile(
                image=object(),
                stage_offset_um=(0.0, 0.0),
                pixel_offset=(0.0, 0.0),
                size_pixels=(10, 10),
            )

        def push_points_from_pixels(self, pts):
            self.points.append(pts)
            return len(pts)

        def set_objective(self, objective):
            self.objective = objective

        def fetch_stitched_montage(self, channel=0, z=0, max_project=False, cross_corr=False, use_features=False):
            return "montage", (120, 80)

        def fetch_capture_count(self):
            return 7

    svc = FakeService()

    result = celfdrive_workflow.run_celfdrive_workflow(
        svc,
        prescan_script="pre",
        highres_script="hi",
        class_thresholds={"cell": 0.5},
        offsets={"x_offset": 1, "y_offset": 2, "z_offset": 3},
        simulated=False,
        max_project=True,
        objective="40x",
    )

    assert svc.started == ["pre", "hi"]
    assert svc.ensured == 2  # run + detect
    assert svc.objective == "40x"
    assert svc.points
    uploaded = svc.points[0]
    assert uploaded[0][0] == 6
    assert uploaded[0][1] == 7
    assert uploaded[0][2] == 3
    assert uploaded[0][3] == 3
    assert result.width == 120
    assert result.height == 80
    assert result.capture_count == 7
    assert len(result.boxes) == 1
