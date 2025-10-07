import sys
import types

real_numpy = sys.modules.get("numpy")
sys.modules["numpy"] = types.SimpleNamespace()
sys.modules["sbs_interface.SBDetectObjects"] = types.SimpleNamespace(ObjectDetector=object)
sys.modules["PIL"] = types.SimpleNamespace(Image=object)
class _BaseModel:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

sys.modules["pydantic"] = types.SimpleNamespace(BaseModel=_BaseModel, Field=lambda *a, **k: None)
sys.modules["yaml"] = types.SimpleNamespace(safe_load=lambda *a, **k: {})

from sbs_interface import api

if real_numpy is None:
    sys.modules.pop("numpy", None)
else:
    sys.modules["numpy"] = real_numpy

def test_start_celfdrive(monkeypatch):
    calls = {"start_capture": []}

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def is_capturing(self):
            return False

        def fetch_capture_count(self):
            return 1

    class FakeMS:
        def start_capture(self, script):
            calls["start_capture"].append(script)

        def _client(self):
            return FakeClient()

        def _wait_for_capture(self, mc, poll=0.5):
            calls["wait"] = calls.get("wait", 0) + 1

        def set_objective(self, objective):
            calls["objective"] = objective

    boxes = [
        {"x": 0, "y": 0, "width": 10, "height": 10, "label": "cell", "confidence": 0.9},
        {"x": 20, "y": 20, "width": 5, "height": 5, "label": "noise", "confidence": 0.4},
    ]

    def fake_detect(ms, channel, z, max_project):
        return {"boxes": boxes, "width": 100, "height": 100}

    monkeypatch.setattr(api, "detect_montage", fake_detect)

    def fake_find(points, ms):
        calls["points"] = points
        return len(points)

    monkeypatch.setattr(api, "find_microscope_points", fake_find)

    def fake_montage(channel, z, max_project, cross_corr, use_features, ms):
        return {"image": "mont", "width": 100, "height": 100}

    monkeypatch.setattr(api, "get_microscope_montage", fake_montage)

    monkeypatch.setattr(api, "get_capture_count", lambda ms: 7)
    api.PointModel = types.SimpleNamespace
    req = types.SimpleNamespace(
        prescan="pre",
        highres="hi",
        classes={"cell": 0.5},
        simulated=False,
        max_project=True,
        objective="40x",
        offsets={"x_offset": 1, "y_offset": 2, "z_offset": 3},
    )

    result = api.start_celfdrive(req, FakeMS())
    assert calls["start_capture"] == ["pre", "hi"]
    assert calls.get("wait") == 1
    assert len(result["boxes"]) == 1
    assert result["captureCount"] == 7
    assert calls["points"][0].x == 6
    assert calls["points"][0].y == 7
    assert calls["points"][0].z == 3
    assert calls["points"][0].auxZ == 3
    assert calls["objective"] == "40x"
