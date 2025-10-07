import types
import sys

# Provide minimal stubs for optional dependencies used during import.
sys.modules.setdefault("yaml", types.ModuleType("yaml"))

class _NP(types.SimpleNamespace):
    def array(self, *args, **kwargs):
        class _Arr:
            def tobytes(self) -> bytes:
                return b""

        return _Arr()

    def frombuffer(self, *args, **kwargs):
        return []


np_stub = _NP()
for _name in ("uint16", "int16", "uint32", "int32", "uint64", "int64", "float32", "float64"):
    setattr(np_stub, _name, _name)
sys.modules.setdefault("numpy", np_stub)

from sbs_interface.objective_manager import ObjectiveManager


class FakeLens:
    def __init__(self, name="20x", pos=2):
        self.mName = name
        self.mTurretPosition = pos


class FakeSB:
    def __init__(self):
        self.selected = None
        self.get_calls = 0

    def FocusWindowScopeSelectObjective(self, name):
        self.selected = name
        return 1

    def GetObjectives(self):
        self.get_calls += 1
        return [FakeLens()]


def test_set_objective_by_name():
    sb = FakeSB()
    mgr = ObjectiveManager(sb)
    assert mgr.set_objective("20x") is True
    assert sb.selected == "20x"


def test_set_objective_by_position():
    sb = FakeSB()
    mgr = ObjectiveManager(sb)
    assert mgr.set_objective(2) is True
    assert sb.selected == "20x"


def test_list_objectives_uses_getobjectives():
    sb = FakeSB()
    mgr = ObjectiveManager(sb)
    objs = mgr.list_objectives()
    assert objs == [("20x", 2)]
    assert sb.get_calls == 1
