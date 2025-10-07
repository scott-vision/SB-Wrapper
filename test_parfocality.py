import sys
import types

real_numpy = sys.modules.get("numpy")
real_PIL = sys.modules.get("PIL")
real_pydantic = sys.modules.get("pydantic")
real_ultra = sys.modules.get("ultralytics")
real_yaml = sys.modules.get("yaml")

sys.modules["numpy"] = types.SimpleNamespace()
sys.modules["PIL"] = types.SimpleNamespace(Image=object)
sys.modules["pydantic"] = types.SimpleNamespace(BaseModel=object, Field=lambda *a, **k: None)
sys.modules["ultralytics"] = types.SimpleNamespace(YOLO=object)
sys.modules["yaml"] = types.SimpleNamespace(safe_load=lambda *a, **k: {})

import sbs_interface.api as api
from sbs_interface.services.microscope import MicroscopeService

for name, mod in [
    ("numpy", real_numpy),
    ("PIL", real_PIL),
    ("pydantic", real_pydantic),
    ("ultralytics", real_ultra),
    ("yaml", real_yaml),
]:
    if mod is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = mod


def test_parfocality_offset(monkeypatch):
    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.sb = object()
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass
        def fetch_voxel_size(self, capture_index=0):
            return 1.0, 1.0, 1.0
        def fetch_image_XYZ(self, capture_index=0):
            return [10.0 * capture_index, 20.0 * capture_index, 30.0 * capture_index]
        def fetch_image_dims(self, capture_index=0):
            return [100, 100, 1]
    def fake_client(self):
        return FakeClient()
    monkeypatch.setattr(MicroscopeService, "_client", fake_client)
    svc = MicroscopeService("h", 1)
    req = types.SimpleNamespace(
        lowres=[types.SimpleNamespace(x=0, y=0, z=0, auxz=0.0)],
        highres=[types.SimpleNamespace(x=0, y=0, z=0, auxz=0.0)],
    )
    class DummyOffset:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z
    api.OffsetModel = DummyOffset
    api.ParfocalityResponse = types.SimpleNamespace
    resp = api.parfocality_offset(req, svc)
    assert resp.average.x == 10
    assert resp.average.y == 20
    assert resp.average.z == 30
