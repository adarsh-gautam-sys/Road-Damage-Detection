"""YOLO11 road damage detector wrapper (`U08`)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from rdai.severity.rules import grade_detection_box, load_severity_config
from rdai.train.train import EXPECTED_CLASS_MAP, _default_yolo_loader, load_app_model_config

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_APP_YAML = ROOT_DIR / "configs" / "app.yaml"

CLASS_CODE_TO_NAME: dict[str, str] = {
    "D00": "longitudinal_crack",
    "D10": "transverse_crack",
    "D20": "alligator_crack",
    "D40": "pothole",
}


class RoadDamageDetector:
    """Wrapper around Ultralytics YOLO11 that clamps boxes, filters by conf, and grades severity."""

    def __init__(
        self,
        weights_path: Path | None = None,
        default_conf: float | None = None,
        iou_threshold: float = 0.45,
        app_yaml_path: Path = DEFAULT_APP_YAML,
        yolo_loader: Callable[[str], Any] | None = None,
    ) -> None:
        model_cfg = load_app_model_config(app_yaml_path) if app_yaml_path.is_file() else {}
        self.weights_path = (
            Path(weights_path)
            if weights_path is not None
            else ROOT_DIR
            / str(model_cfg.get("weights_path", "runs/e1_yolo11s_640/weights/best.pt"))
        )
        self.default_conf = (
            float(default_conf)
            if default_conf is not None
            else float(model_cfg.get("default_conf", 0.25))
        )
        self.iou_threshold = float(model_cfg.get("iou_threshold", iou_threshold))
        self.model_name = str(model_cfg.get("name", "yolo11s"))
        self.run_name = str(model_cfg.get("run", "e1_yolo11s_640"))
        self.severity_config = load_severity_config()
        self._yolo_loader = yolo_loader if yolo_loader is not None else _default_yolo_loader
        self._model: Any | None = None

    @property
    def is_loaded(self) -> bool:
        """Return whether the underlying model has been initialized."""
        return self._model is not None

    def load(self) -> Any:
        """Lazily load and cache the YOLO11 model instance."""
        if self._model is None:
            self._model = self._yolo_loader(str(self.weights_path))
        return self._model

    def detect_image(
        self,
        image_source: Any,
        conf: float | None = None,
    ) -> list[dict[str, Any]]:
        """Run detection on an image array or path and return graded detections (`U08`)."""
        threshold = float(conf) if conf is not None else self.default_conf
        model = self.load()
        raw_results = model.predict(
            image_source,
            conf=threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        if not raw_results:
            return []

        pred = raw_results[0]
        boxes = getattr(pred, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        orig_h, orig_w = getattr(pred, "orig_shape", (720, 720))
        xyxy = np.asarray(
            boxes.xyxy.cpu() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy, dtype=float
        )
        confs = np.asarray(
            boxes.conf.cpu() if hasattr(boxes.conf, "cpu") else boxes.conf, dtype=float
        )
        clss = np.asarray(boxes.cls.cpu() if hasattr(boxes.cls, "cpu") else boxes.cls, dtype=int)

        detections: list[dict[str, Any]] = []
        for raw_box, conf_val, cls_id in zip(xyxy, confs, clss, strict=True):
            if float(conf_val) < threshold:
                continue
            cls_code = EXPECTED_CLASS_MAP.get(int(cls_id))
            if cls_code is None:
                continue

            x1 = float(np.clip(raw_box[0], 0.0, float(orig_w)))
            y1 = float(np.clip(raw_box[1], 0.0, float(orig_h)))
            x2 = float(np.clip(raw_box[2], 0.0, float(orig_w)))
            y2 = float(np.clip(raw_box[3], 0.0, float(orig_h)))
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            clamped_box = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            sev = grade_detection_box(
                cls_code,
                clamped_box,
                img_w=int(orig_w),
                img_h=int(orig_h),
                config=self.severity_config,
            )
            detections.append(
                {
                    "class_id": int(cls_id),
                    "class_code": cls_code,
                    "class_name": CLASS_CODE_TO_NAME[cls_code],
                    "confidence": round(float(conf_val), 4),
                    "box": clamped_box,
                    "severity": sev,
                }
            )
        return detections

    def detect(
        self,
        image_source: Any,
        conf: float | None = None,
        conf_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """Alias for `detect_image` accepting either `conf` or `conf_threshold`."""
        effective_conf = conf if conf is not None else conf_threshold
        return self.detect_image(image_source, conf=effective_conf)
