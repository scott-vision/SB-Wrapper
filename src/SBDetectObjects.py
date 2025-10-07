from __future__ import annotations

"""Object detection utilities for microscope images.

This module wraps a YOLO model and provides a small helper class
:class:`ObjectDetector` that converts a numpy array into a list of bounding
boxes.  The heavy lifting is borrowed from ``webapp.detect_objects`` but is
packaged here for reuse across the project.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import os
import numpy as np
import logging
from ultralytics import YOLO

__all__ = ["ObjectDetector"]


class ObjectDetector:
    """Run YOLO object detection on microscope images.

    Parameters
    ----------
    model_path:
        Optional path to a YOLO weight file.  If not provided, the path
        defaults to the bundled ``weights/yolov11l_ALLDATA_UNDERSAMPLE/best.pt``
        model or the value of the ``YOLO_MODEL`` environment variable if set.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        default_model = (
            Path(__file__).parent
            / "weights"
            / "yolov11l_ALLDATA_UNDERSAMPLE"
            / "best.pt"
        )
        self.model_path = model_path or os.getenv("YOLO_MODEL", str(default_model))
        # Cache separate models for SAHI and standard inference so that
        # toggling the ``use_sahi`` flag between calls loads the correct
        # backend each time.
        self._yolo_model = None
        self._sahi_model = None

    # ------------------------------------------------------------------
    def _load_model(self, use_sahi: bool):
        if use_sahi:
            if self._sahi_model is None:
                from sahi import AutoDetectionModel

                self._sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="ultralytics",
                    model_path=self.model_path,
                    confidence_threshold=0.3,
                    device="cuda",  # or 'cuda:0'
                )
            return self._sahi_model

        if self._yolo_model is None:
            self._yolo_model = YOLO(self.model_path)
        return self._yolo_model
    

    def preprocess_image(self, img):
        """
        Preprocesses the input image by applying a percentile-based threshold and normalizing pixel values.

        Parameters:
        - img (array): Input image array, assumed to be 2D or 3D with identical channels.

        Returns:
        - img (array): Preprocessed image normalized to a 0-255 range and converted to uint8 type.
        """

        percentile = 0.03
        percentile_threshold = 100 - percentile  # 100 - 0.3
        threshold_value = np.percentile(img, percentile_threshold)
        # we assume 2D here so if its 3D at this point the 3rd index we assume is channels that are identical as others for saving
        if len(img.shape) == 3:
            h, w, _ = img.shape
            timepoint_data = img[:,:, 0]
        elif len(img.shape) == 2:
            h, w = img.shape
            timepoint_data = img
        else:
            print("Shape Error in predict.preprocess_image")

        timepoint_data_normalized = np.where(timepoint_data > threshold_value, threshold_value, timepoint_data)

        # Normalize the image to the range of 0 to 1

        timepoint_data_normalized2 = (timepoint_data_normalized - timepoint_data_normalized.min()) / (
            timepoint_data_normalized.max() - timepoint_data_normalized.min())
        # plt.imshow(timepoint_data_normalized2)

        img = (timepoint_data_normalized2 * 255).astype(np.uint8)

        return img

    # ------------------------------------------------------------------
    def detect(self, arr: np.ndarray, use_sahi: bool = False) -> Dict[str, Any]:
        """Return bounding boxes for objects in ``arr``.

        Parameters
        ----------
        arr:
            Input image array which may be 2-D or 3-D.
        use_sahi:
            If ``True`` perform sliding window inference using the
            :mod:`sahi` package.  Otherwise run a standard forward pass with
            the ultralytics model.

        The input array is converted to an 8-bit RGB image before being
        passed to the detection model.
        """
        arr = self.preprocess_image(arr)
        if arr.max() > 255:
            arr = (255 * (arr.astype(float) / arr.max())).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)

        model = self._load_model(use_sahi)
        boxes: List[Dict[str, float]] = []

        if use_sahi:
            # Deferred imports so that the dependency is optional at runtime
            from sahi.predict import get_sliced_prediction
            
            
            result = get_sliced_prediction(
                arr,
                model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                # postprocess_type="NMS",
                postprocess_match_threshold=0.25
            )
            for op in result.object_prediction_list:
                bbox = op.bbox
                boxes.append(
                    {
                        # Convert any NumPy scalar values to plain Python
                        # ``float`` objects so that FastAPI's JSON encoder
                        # can serialise the response without raising
                        # ``TypeError: 'numpy.float32' object is not iterable``.
                        "x": float(bbox.minx),
                        "y": float(bbox.miny),
                        "width": float(bbox.maxx - bbox.minx),
                        "height": float(bbox.maxy - bbox.miny),
                        "label": op.category.name,
                        "confidence": float(op.score.value),
                    }
                )
        else:
            results = model(arr, verbose=False)
            r = results[0]
            names = r.names  # mapping of class indices to names
            for (x1, y1, x2, y2), cls, conf in zip(
                r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()
            ):
                boxes.append(
                    {
                        "x": float(x1),
                        "y": float(y1),
                        "width": float(x2 - x1),
                        "height": float(y2 - y1),
                        "label": names[int(cls)],
                        "confidence": float(conf),
                    }
                )
        logging.info(boxes)
        return {"boxes": boxes, "width": int(arr.shape[1]), "height": int(arr.shape[0])}
