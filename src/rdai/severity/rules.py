"""Rule-based road damage severity scoring and tier assignment (`FR-S1`, `FR-S2`, verified by `U03`)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_SEVERITY_YAML = ROOT_DIR / "configs" / "severity.yaml"

TIER_ORDER: dict[str, int] = {"Low": 0, "Medium": 1, "High": 2}


def load_severity_config(severity_yaml_path: Path = DEFAULT_SEVERITY_YAML) -> dict[str, Any]:
    """Load severity thresholds and class rules from `configs/severity.yaml`."""
    with severity_yaml_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "classes" not in data:
        raise ValueError(f"Invalid severity config in {severity_yaml_path}")
    return data


def compute_box_extent(
    class_code: str,
    box: Sequence[float],
    img_w: int,
    img_h: int,
    config: dict[str, Any] | None = None,
) -> tuple[float, str]:
    """Compute class-specific normalized extent (`area_ratio` or `diag_ratio`)."""
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Image dimensions must be positive, got ({img_w}, {img_h})")

    cfg = config if config is not None else load_severity_config()
    cls_cfg = cfg["classes"].get(class_code)
    if cls_cfg is None:
        raise ValueError(f"Unknown damage class code '{class_code}'")

    measure = str(cls_cfg["measure"])
    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    w_box = max(0.0, min(x2, float(img_w)) - max(0.0, x1))
    h_box = max(0.0, min(y2, float(img_h)) - max(0.0, y1))

    if measure == "area_ratio":
        extent = (w_box * h_box) / float(img_w * img_h)
    elif measure == "diag_ratio":
        box_diag = math.hypot(w_box, h_box)
        img_diag = math.hypot(float(img_w), float(img_h))
        extent = box_diag / img_diag if img_diag > 0.0 else 0.0
    else:
        raise ValueError(f"Unsupported extent measure '{measure}' for {class_code}")

    extent = max(0.0, min(1.0, extent))
    return extent, measure


def compute_severity_from_extent(
    class_code: str,
    extent: float,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute continuous score `S in [0, 100]` and label (`Low`, `Medium`, `High`) from `extent`."""
    cfg = config if config is not None else load_severity_config()
    cls_cfg = cfg["classes"].get(class_code)
    if cls_cfg is None:
        raise ValueError(f"Unknown damage class code '{class_code}'")

    bounds = cfg.get("tier_bounds", {})
    low_upper = float(bounds.get("low_upper", 33.3))
    medium_upper = float(bounds.get("medium_upper", 66.7))
    max_score = float(bounds.get("max_score", 100.0))

    e = max(0.0, min(1.0, float(extent)))
    e_low = float(cls_cfg["e_low"])
    e_high = float(cls_cfg["e_high"])
    measure = str(cls_cfg["measure"])
    min_label = str(cls_cfg.get("min_label", "Low"))

    if e < e_low:
        raw_score = low_upper * (e / e_low) if e_low > 0.0 else 0.0
        raw_label = "Low"
    elif e < e_high:
        span = e_high - e_low
        raw_score = low_upper + low_upper * ((e - e_low) / span) if span > 0.0 else low_upper
        raw_label = "Medium"
    else:
        ratio = min((e - e_high) / e_high, 1.0) if e_high > 0.0 else 1.0
        raw_score = medium_upper + low_upper * ratio
        raw_label = "High"

    score = max(0.0, min(max_score, raw_score))
    final_label = raw_label if TIER_ORDER[raw_label] >= TIER_ORDER[min_label] else min_label

    return {
        "score": round(score, 4),
        "label": final_label,
        "extent": round(e, 6),
        "measure": measure,
    }


def grade_detection_box(
    class_code: str,
    box: Sequence[float],
    img_w: int,
    img_h: int,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute extent, severity score, and tier label for a `[x1, y1, x2, y2]` detection box."""
    cfg = config if config is not None else load_severity_config()
    extent, _ = compute_box_extent(class_code, box, img_w=img_w, img_h=img_h, config=cfg)
    return compute_severity_from_extent(class_code, extent, config=cfg)
