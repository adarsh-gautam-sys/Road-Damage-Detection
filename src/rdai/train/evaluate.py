"""YOLO11 validation F1 threshold calibration, test-split evaluation, and ONNX parity (`FR-M2`..`FR-M5`)."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rdai.train.train import EXPECTED_CLASS_MAP, _default_yolo_loader, load_app_model_config

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_APP_YAML = ROOT_DIR / "configs" / "app.yaml"
DEFAULT_DATA_YAML = ROOT_DIR / "configs" / "data.yaml"
DEFAULT_METRICS_JSON = ROOT_DIR / "reports" / "metrics.json"


def extract_val_f1_threshold(
    f1_curve: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Extract the confidence threshold maximizing mean macro-F1 across classes (`FR-M4`)."""
    arr = np.asarray(f1_curve, dtype=float)
    mean_f1 = arr.mean(axis=0) if arr.ndim == 2 else arr
    if mean_f1.size == 0:
        return 0.25, 0.0

    xs = (
        np.asarray(thresholds, dtype=float)
        if thresholds is not None
        else np.linspace(0.0, 1.0, mean_f1.size)
    )
    best_idx = int(np.argmax(mean_f1))
    best_conf = float(np.clip(xs[best_idx], 0.05, 0.90))
    best_f1 = float(mean_f1[best_idx])
    return round(best_conf, 3), round(best_f1, 4)


def update_app_default_conf(
    new_conf: float,
    app_yaml_path: Path = DEFAULT_APP_YAML,
) -> None:
    """Write calibrated default confidence threshold into `configs/app.yaml` (`FR-M4`)."""
    with app_yaml_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if "model" not in cfg or not isinstance(cfg["model"], dict):
        cfg["model"] = {}
    cfg["model"]["default_conf"] = round(float(new_conf), 3)
    with app_yaml_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)


def format_val_result_to_dict(
    val_result: Any,
    experiment_id: str = "E1",
    split: str = "test",
    params_m: float = 9.4,
    cpu_ms: float | None = None,
) -> dict[str, Any]:
    """Convert an Ultralytics `DetMetrics` validation result object into a serializable dictionary."""
    box = val_result.box
    speed = getattr(val_result, "speed", {}) or {}
    gpu_ms = float(speed.get("inference", 0.0))

    ap_cls_indices = [int(i) for i in getattr(box, "ap_class_index", [])]
    p_arr = np.asarray(getattr(box, "p", []), dtype=float)
    r_arr = np.asarray(getattr(box, "r", []), dtype=float)
    ap50_arr = np.asarray(getattr(box, "ap50", []), dtype=float)
    ap_arr = np.asarray(getattr(box, "ap", []), dtype=float)

    per_class: dict[str, dict[str, float]] = {}
    for cls_id, cls_code in EXPECTED_CLASS_MAP.items():
        if cls_id in ap_cls_indices:
            pos = ap_cls_indices.index(cls_id)
            per_class[cls_code] = {
                "precision": round(float(p_arr[pos]), 4),
                "recall": round(float(r_arr[pos]), 4),
                "ap50": round(float(ap50_arr[pos]), 4),
                "ap50_95": round(float(ap_arr[pos]), 4),
            }
        else:
            per_class[cls_code] = {
                "precision": 0.0,
                "recall": 0.0,
                "ap50": 0.0,
                "ap50_95": 0.0,
            }

    cm_obj = getattr(val_result, "confusion_matrix", None)
    cm_list = (
        np.asarray(cm_obj.matrix, dtype=float).round(4).tolist()
        if cm_obj is not None and hasattr(cm_obj, "matrix")
        else []
    )

    return {
        "experiment_id": experiment_id,
        "split": split,
        "params_m": params_m,
        "gpu_ms": round(gpu_ms, 2),
        "cpu_ms": round(float(cpu_ms), 2) if cpu_ms is not None else None,
        "overall": {
            "precision": round(float(getattr(box, "mp", 0.0)), 4),
            "recall": round(float(getattr(box, "mr", 0.0)), 4),
            "map50": round(float(getattr(box, "map50", 0.0)), 4),
            "map50_95": round(float(getattr(box, "map", 0.0)), 4),
        },
        "per_class": per_class,
        "confusion_matrix": cm_list,
    }


def verify_metric_gate(
    exp_metrics: dict[str, Any],
    min_map50: float = 0.50,
    min_pothole_ap50: float = 0.55,
) -> dict[str, Any]:
    """Verify PRD Section 1 & 12 metric gate (`M03`): macro mAP@50 >= 0.50 and D40 AP@50 >= 0.55."""
    overall_map50 = float(exp_metrics.get("overall", {}).get("map50", 0.0))
    d40_ap50 = float(exp_metrics.get("per_class", {}).get("D40", {}).get("ap50", 0.0))
    passed = overall_map50 >= min_map50 and d40_ap50 >= min_pothole_ap50
    return {
        "passed": passed,
        "map50": overall_map50,
        "min_map50": min_map50,
        "d40_ap50": d40_ap50,
        "min_pothole_ap50": min_pothole_ap50,
    }


def _extract_predictions(pred_result: Any) -> list[dict[str, Any]]:
    """Extract list of `{class_id, class_code, confidence, box}` from a single Ultralytics result."""
    boxes = getattr(pred_result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = np.asarray(boxes.xyxy.cpu() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy, dtype=float)
    confs = np.asarray(boxes.conf.cpu() if hasattr(boxes.conf, "cpu") else boxes.conf, dtype=float)
    clss = np.asarray(boxes.cls.cpu() if hasattr(boxes.cls, "cpu") else boxes.cls, dtype=int)

    h_px, w_px = getattr(pred_result, "orig_shape", (720, 720))
    detections: list[dict[str, Any]] = []
    for box_coords, conf_val, cls_id in zip(xyxy, confs, clss, strict=True):
        x1 = float(np.clip(box_coords[0], 0.0, float(w_px)))
        y1 = float(np.clip(box_coords[1], 0.0, float(h_px)))
        x2 = float(np.clip(box_coords[2], 0.0, float(w_px)))
        y2 = float(np.clip(box_coords[3], 0.0, float(h_px)))
        detections.append(
            {
                "class_id": int(cls_id),
                "class_code": EXPECTED_CLASS_MAP.get(int(cls_id), f"CLS_{cls_id}"),
                "confidence": round(float(conf_val), 4),
                "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            }
        )
    return detections


def run_smoke_inference_on_fixtures(
    weights_path: Path,
    image_paths: list[Path],
    app_yaml_path: Path = DEFAULT_APP_YAML,
    yolo_loader: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Run inference on fixture road images using the default confidence from `app.yaml` (`M02`)."""
    loader = yolo_loader if yolo_loader is not None else _default_yolo_loader
    model_cfg = load_app_model_config(app_yaml_path)
    conf = float(model_cfg.get("default_conf", 0.25))

    model = loader(str(weights_path))
    raw_results = model.predict([str(p) for p in image_paths], conf=conf, verbose=False)

    images_out: list[dict[str, Any]] = []
    for img_path, pred_res in zip(image_paths, raw_results, strict=True):
        h_px, w_px = getattr(pred_res, "orig_shape", (720, 720))
        images_out.append(
            {
                "image": img_path.name,
                "width": int(w_px),
                "height": int(h_px),
                "detections": _extract_predictions(pred_res),
            }
        )
    return {
        "configured_conf": conf,
        "images": images_out,
    }


def check_inference_determinism(
    weights_path: Path,
    image_path: Path,
    conf: float = 0.25,
    yolo_loader: Callable[[str], Any] | None = None,
) -> bool:
    """Verify running inference twice on the same image returns identical boxes (`M04`)."""
    loader = yolo_loader if yolo_loader is not None else _default_yolo_loader
    model = loader(str(weights_path))
    res1 = _extract_predictions(model.predict(str(image_path), conf=conf, verbose=False)[0])
    res2 = _extract_predictions(model.predict(str(image_path), conf=conf, verbose=False)[0])
    return res1 == res2


def count_negative_detections(
    weights_path: Path,
    image_paths: list[Path],
    conf: float = 0.25,
    yolo_loader: Callable[[str], Any] | None = None,
) -> int:
    """Count total detections across synthetic non-road images (`M05`)."""
    loader = yolo_loader if yolo_loader is not None else _default_yolo_loader
    model = loader(str(weights_path))
    results = model.predict([str(p) for p in image_paths], conf=conf, verbose=False)
    return sum(len(_extract_predictions(r)) for r in results)


def compute_box_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute Intersection-over-Union (IoU) between two `[x1, y1, x2, y2]` boxes."""
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    inter = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def verify_onnx_parity(
    pt_weights_path: Path,
    onnx_weights_path: Path,
    image_paths: list[Path],
    conf: float = 0.25,
    min_iou: float = 0.95,
    yolo_loader: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Compare PyTorch `.pt` and `.onnx` predictions on fixture images (`M07`)."""
    loader = yolo_loader if yolo_loader is not None else _default_yolo_loader
    pt_model = loader(str(pt_weights_path))
    onnx_model = loader(str(onnx_weights_path))

    pt_results = pt_model.predict([str(p) for p in image_paths], conf=conf, verbose=False)
    onnx_results = onnx_model.predict([str(p) for p in image_paths], conf=conf, verbose=False)

    ious: list[float] = []
    class_mismatches = 0

    for pt_res, onnx_res in zip(pt_results, onnx_results, strict=True):
        pt_dets = _extract_predictions(pt_res)
        onnx_dets = _extract_predictions(onnx_res)
        if len(pt_dets) != len(onnx_dets):
            class_mismatches += abs(len(pt_dets) - len(onnx_dets))
        for d_pt, d_ox in zip(pt_dets, onnx_dets, strict=False):
            if d_pt["class_code"] != d_ox["class_code"]:
                class_mismatches += 1
            ious.append(compute_box_iou(d_pt["box"], d_ox["box"]))

    min_observed_iou = min(ious) if ious else 1.0
    passed = class_mismatches == 0 and min_observed_iou >= min_iou
    return {
        "passed": passed,
        "min_observed_iou": round(min_observed_iou, 4),
        "required_iou": min_iou,
        "class_mismatches": class_mismatches,
    }


def load_metrics_json(metrics_path: Path = DEFAULT_METRICS_JSON) -> dict[str, Any]:
    """Load `reports/metrics.json` or return an empty structure if not yet created."""
    if metrics_path.is_file():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "status": "awaiting_colab_runs",
        "generated_at": None,
        "optimal_conf": None,
        "experiments": {},
    }


def evaluate_checkpoint_and_update_metrics(
    weights_path: Path,
    experiment_id: str = "E1",
    data_yaml_path: Path = DEFAULT_DATA_YAML,
    app_yaml_path: Path = DEFAULT_APP_YAML,
    metrics_path: Path = DEFAULT_METRICS_JSON,
    export_onnx: bool = True,
    yolo_loader: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Calibrate threshold on `val`, evaluate once on `test`, export ONNX, and update `metrics.json`."""
    loader = yolo_loader if yolo_loader is not None else _default_yolo_loader
    if data_yaml_path.is_file() and yolo_loader is None:
        with data_yaml_path.open("r", encoding="utf-8") as fh:
            d_cfg = yaml.safe_load(fh)
        rel_p = Path(str(d_cfg.get("path", "data/rdd2022_india_yolo")))
        if not rel_p.is_absolute():
            d_cfg["path"] = (ROOT_DIR / rel_p).resolve().as_posix()
            with data_yaml_path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(d_cfg, fh, sort_keys=False)
    model = loader(str(weights_path))

    # 1. Validation split: calibrate optimal F1 confidence threshold
    val_res = model.val(data=str(data_yaml_path), split="val")
    f1_curve = getattr(val_res.box, "f1_curve", np.array([]))
    best_conf, best_f1 = extract_val_f1_threshold(f1_curve)
    if experiment_id == "E1":
        update_app_default_conf(best_conf, app_yaml_path=app_yaml_path)

    # 2. Test split: evaluate once with the calibrated threshold
    test_res = model.val(data=str(data_yaml_path), split="test", conf=best_conf)
    formatted = format_val_result_to_dict(test_res, experiment_id=experiment_id, split="test")
    formatted["optimal_val_conf"] = best_conf
    formatted["optimal_val_f1"] = best_f1
    formatted["metric_gate"] = verify_metric_gate(formatted)

    # 3. Export ONNX for E1 if requested
    if export_onnx and experiment_id == "E1" and hasattr(model, "export"):
        onnx_file = model.export(format="onnx", imgsz=640)
        formatted["onnx_path"] = str(onnx_file)

    # 4. Update reports/metrics.json
    metrics_doc = load_metrics_json(metrics_path)
    metrics_doc["status"] = "populated"
    metrics_doc["generated_at"] = datetime.now(UTC).isoformat()
    if experiment_id == "E1":
        metrics_doc["optimal_conf"] = best_conf
    if "experiments" not in metrics_doc or not isinstance(metrics_doc["experiments"], dict):
        metrics_doc["experiments"] = {}
    metrics_doc["experiments"][experiment_id] = formatted

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_doc, indent=2), encoding="utf-8")
    return formatted


def apply_robustness_perturbation(image: Any, kind: str) -> Any:
    """Apply an M06 robustness perturbation (`brightness_plus_30`, `brightness_minus_30`, `gaussian_blur`, `jpeg_q30`, `half_size`)."""
    import io

    from PIL import Image, ImageEnhance, ImageFilter

    rgb = (
        image.convert("RGB")
        if isinstance(image, Image.Image)
        else Image.fromarray(image).convert("RGB")
    )
    if kind == "brightness_plus_30":
        return ImageEnhance.Brightness(rgb).enhance(1.30)
    if kind == "brightness_minus_30":
        return ImageEnhance.Brightness(rgb).enhance(0.70)
    if kind == "gaussian_blur":
        return rgb.filter(ImageFilter.GaussianBlur(radius=2.0))
    if kind == "jpeg_q30":
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=30)
        buf.seek(0)
        with Image.open(buf) as comp:
            return comp.convert("RGB")
    if kind == "half_size":
        w, h = rgb.size
        return rgb.resize((max(1, w // 2), max(1, h // 2)), Image.Resampling.BILINEAR)
    raise ValueError(f"Unknown perturbation kind: {kind}")


def main() -> None:
    """CLI entrypoint for `make eval`."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO11 checkpoint.")
    parser.add_argument("--exp", default="E1", help="Experiment ID (E1..E7)")
    parser.add_argument("--weights", default=None, help="Path to best.pt checkpoint")
    args = parser.parse_args()

    model_cfg = load_app_model_config()
    weights_path = Path(args.weights) if args.weights else ROOT_DIR / str(model_cfg["weights_path"])
    if not weights_path.is_file():
        logger.warning(
            "No trained checkpoint found at %s yet. Sync runs/ from Google Drive first.",
            weights_path,
        )
        return

    res = evaluate_checkpoint_and_update_metrics(weights_path=weights_path, experiment_id=args.exp)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
