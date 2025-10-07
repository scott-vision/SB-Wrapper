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

from sbs_interface import api

if real_numpy is None:
    sys.modules.pop("numpy", None)
else:
    sys.modules["numpy"] = real_numpy
if real_PIL is None:
    sys.modules.pop("PIL", None)
else:
    sys.modules["PIL"] = real_PIL
if real_pydantic is None:
    sys.modules.pop("pydantic", None)
else:
    sys.modules["pydantic"] = real_pydantic
if real_ultra is None:
    sys.modules.pop("ultralytics", None)
else:
    sys.modules["ultralytics"] = real_ultra
if real_yaml is None:
    sys.modules.pop("yaml", None)
else:
    sys.modules["yaml"] = real_yaml


def test_list_capture_scripts(tmp_path, monkeypatch):
    (tmp_path / "foo.Cap.prefs").touch()
    (tmp_path / "bar.Cap.prefs").touch()

    class DummyPath:
        def __init__(self, *args, **kwargs):
            self._path = tmp_path

        def exists(self):
            return True

        def glob(self, pattern):
            return self._path.glob(pattern)

    monkeypatch.setattr(api, "Path", DummyPath)

    scripts = api.list_capture_scripts()
    assert sorted(scripts) == ["bar", "foo"]
