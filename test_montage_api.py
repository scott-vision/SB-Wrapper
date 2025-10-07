import sys
import types
import importlib
from pathlib import Path

_mods_backup = {name: sys.modules.get(name) for name in [
    "numpy",
    "PIL",
    "PIL.Image",
    "pydantic",
    "sbs_interface.SBDetectObjects",
    "sbs_interface.SBPointOptimiser",
]}

sys.modules["numpy"] = types.SimpleNamespace()
sys.modules["PIL"] = types.SimpleNamespace(Image=types.SimpleNamespace())
sys.modules["PIL.Image"] = types.SimpleNamespace()
sys.modules["pydantic"] = types.SimpleNamespace(
    BaseModel=object, Field=lambda *a, **k: None
)
sys.modules["sbs_interface.SBDetectObjects"] = types.SimpleNamespace(
    ObjectDetector=object
)
sys.modules["sbs_interface.SBPointOptimiser"] = types.SimpleNamespace(
    RouteOptimizer=object, distance=lambda *a, **k: 0, Point=tuple
)

pkg = types.ModuleType("sbs_interface")
pkg.__path__ = [str(Path("src").resolve())]
sys.modules["sbs_interface"] = pkg

api = importlib.import_module("sbs_interface.api")

for name, mod in _mods_backup.items():
    if mod is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = mod


class FakeArray:
    size = 4


class FakeService:
    def fetch_display_image(self, channel=0, z=0, max_project=False, capture_index=None):
        return FakeArray(), 1

    def fetch_stitched_montage(
        self,
        channel=0,
        z=0,
        max_project=False,
        cross_corr=False,
        use_features=False,
    ):
        return FakeArray(), (2, 2)


def test_get_microscope_montage(monkeypatch):
    monkeypatch.setattr(api, "_encode_image", lambda arr: "enc")
    svc = FakeService()
    result = api.get_microscope_montage(0, 0, False, False, False, svc)
    assert result["width"] == 2
    assert result["height"] == 2
    assert result["image"] == "enc"


def test_detect_objects_montage(monkeypatch):
    called = {}

    def fake_detect(ms, channel=0, z=0, max_project=False, use_sahi=False):
        called["done"] = True
        return types.SimpleNamespace(to_dict=lambda: {"boxes": []})

    monkeypatch.setattr(api.celfdrive_workflow, "detect_montage", fake_detect)

    result = api.detect_objects(0, 0, False, False, FakeService(), montage=True)
    assert called.get("done")
    assert "boxes" in result
