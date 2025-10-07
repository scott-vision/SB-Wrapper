import types
import sys

# Temporarily stub heavy dependencies used during import
real_numpy = sys.modules.get("numpy")
sys.modules["numpy"] = types.SimpleNamespace()
sys.modules["sbs_interface.SBDetectObjects"] = types.SimpleNamespace(ObjectDetector=object)

from sbs_interface.celfdrive import _merge_boxes

# Restore numpy for other tests
if real_numpy is None:
    sys.modules.pop("numpy", None)
else:
    sys.modules["numpy"] = real_numpy


def test_merge_boxes_removes_overlap():
    boxes = [
        {"x": 0, "y": 0, "width": 10, "height": 10, "confidence": 0.5, "label": "a"},
        {"x": 2, "y": 2, "width": 10, "height": 10, "confidence": 0.9, "label": "a"},
        {"x": 20, "y": 20, "width": 5, "height": 5, "confidence": 0.6, "label": "b"},
    ]
    merged = _merge_boxes(boxes)
    assert len(merged) == 2
    confidences = sorted(b["confidence"] for b in merged)
    assert confidences == [0.6, 0.9]
