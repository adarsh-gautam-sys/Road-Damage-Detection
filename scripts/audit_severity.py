"""Severity audit sheet generator and agreement/kappa evaluator (`FR-S3`, PRD Section 9)."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from rdai.data.voc_to_yolo import CLASS_MAP, parse_voc_xml
from rdai.severity.rules import grade_detection_box, load_severity_config

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SHEET_CSV = ROOT_DIR / "reports" / "tables" / "severity_audit_sheet.csv"
DEFAULT_KEY_JSON = ROOT_DIR / "reports" / "tables" / "severity_audit_key.json"
DEFAULT_CROPS_DIR = ROOT_DIR / "reports" / "audit_crops"
LABELS_ORDER = ["Low", "Medium", "High"]


def generate_audit_sheet(
    img_dir: Path,
    xml_dir: Path,
    out_csv_path: Path = DEFAULT_SHEET_CSV,
    out_crops_dir: Path = DEFAULT_CROPS_DIR,
    target_samples: int = 100,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Sample `target_samples` detections (~1/3 D40, ~1/3 D20, ~1/3 cracks) for blind hand grading."""
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_crops_dir.mkdir(parents=True, exist_ok=True)

    id_to_code = {v: k for k, v in CLASS_MAP.items()}
    sev_cfg = load_severity_config()
    candidates: list[dict[str, Any]] = []

    for xml_path in sorted(xml_dir.glob("*.xml")):
        img_path = img_dir / f"{xml_path.stem}.jpg"
        w_px, h_px = (720, 720)
        if img_path.is_file():
            with Image.open(img_path) as pil_im:
                w_px, h_px = pil_im.size

        boxes, _, _ = parse_voc_xml(xml_path)
        for box_idx, b in enumerate(boxes):
            cls_code = id_to_code[b.class_id]
            x1 = max(0, int((b.cx - b.w / 2.0) * w_px))
            y1 = max(0, int((b.cy - b.h / 2.0) * h_px))
            x2 = min(w_px, int((b.cx + b.w / 2.0) * w_px))
            y2 = min(h_px, int((b.cy + b.h / 2.0) * h_px))
            sev = grade_detection_box(
                cls_code, [x1, y1, x2, y2], img_w=w_px, img_h=h_px, config=sev_cfg
            )
            candidates.append(
                {
                    "stem": xml_path.stem,
                    "img_path": img_path if img_path.is_file() else None,
                    "box_idx": box_idx,
                    "class_code": cls_code,
                    "box": [x1, y1, x2, y2],
                    "severity": sev,
                }
            )

    if not candidates:
        raise ValueError(f"No valid detections found in {xml_dir}")

    rng = random.Random(seed)
    by_group: dict[str, list[dict[str, Any]]] = {
        "D40": [c for c in candidates if c["class_code"] == "D40"],
        "D20": [c for c in candidates if c["class_code"] == "D20"],
        "cracks": [c for c in candidates if c["class_code"] in ("D00", "D10")],
    }

    sampled: list[dict[str, Any]] = []
    per_group = target_samples // 3
    quotas = {"D40": per_group, "D20": per_group, "cracks": target_samples - 2 * per_group}

    for grp, quota in quotas.items():
        pool = by_group[grp] or candidates
        for i in range(quota):
            item = dict(pool[i % len(pool)])
            sampled.append(item)

    rng.shuffle(sampled)

    blind_rows: list[dict[str, Any]] = []
    hidden_key: list[dict[str, Any]] = []

    for idx, item in enumerate(sampled, start=1):
        crop_id = f"crop_{idx:03d}"
        split_half = "tune_50" if idx <= target_samples // 2 else "eval_50"
        crop_file = out_crops_dir / f"{crop_id}.jpg"

        if item["img_path"] is not None and item["img_path"].is_file():
            with Image.open(item["img_path"]) as pil_im:
                x1, y1, x2, y2 = item["box"]
                pad = 16
                cx1 = max(0, x1 - pad)
                cy1 = max(0, y1 - pad)
                cx2 = min(pil_im.width, max(cx1 + 10, x2 + pad))
                cy2 = min(pil_im.height, max(cy1 + 10, y2 + pad))
                pil_im.crop((cx1, cy1, cx2, cy2)).save(crop_file, format="JPEG", quality=90)

        blind_rows.append(
            {
                "crop_id": crop_id,
                "subset": split_half,
                "image_stem": item["stem"],
                "class_code": item["class_code"],
                "crop_path": str(crop_file.relative_to(ROOT_DIR))
                if crop_file.is_relative_to(ROOT_DIR)
                else str(crop_file),
                "human_label_1": "",
                "human_label_2": "",
            }
        )
        hidden_key.append(
            {
                "crop_id": crop_id,
                "subset": split_half,
                "class_code": item["class_code"],
                "model_score": item["severity"]["score"],
                "model_label": item["severity"]["label"],
                "extent": item["severity"]["extent"],
                "measure": item["severity"]["measure"],
            }
        )

    fieldnames = [
        "crop_id",
        "subset",
        "image_stem",
        "class_code",
        "crop_path",
        "human_label_1",
        "human_label_2",
    ]
    with out_csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(blind_rows)

    key_path = out_csv_path.with_name("severity_audit_key.json")
    key_path.write_text(json.dumps(hidden_key, indent=2), encoding="utf-8")
    return blind_rows


def compute_audit_agreement(
    model_labels: list[str],
    human_labels_1: list[str],
    human_labels_2: list[str] | None = None,
) -> dict[str, Any]:
    """Compute agreement, 3x3 confusion matrix, and optional Cohen's kappa on held-out crops."""
    if not model_labels or len(model_labels) != len(human_labels_1):
        raise ValueError("model_labels and human_labels_1 must be non-empty and equal length")

    matches = sum(1 for m, h in zip(model_labels, human_labels_1, strict=True) if m == h)
    agreement = matches / len(model_labels)
    cm = confusion_matrix(human_labels_1, model_labels, labels=LABELS_ORDER)

    kappa: float | None = None
    if human_labels_2 and len(human_labels_2) == len(human_labels_1):
        raw_k = float(cohen_kappa_score(human_labels_1, human_labels_2, labels=LABELS_ORDER))
        kappa = 1.0 if np.isnan(raw_k) else round(raw_k, 4)

    return {
        "n_evaluated": len(model_labels),
        "agreement": round(agreement, 4),
        "target_met": agreement >= 0.70,
        "cohens_kappa": kappa,
        "labels": LABELS_ORDER,
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    """Generate the 100-crop blind severity audit sheet (`reports/tables/severity_audit_sheet.csv`)."""
    raw_india = ROOT_DIR / "data" / "rdd2022" / "India"
    if (raw_india / "train" / "annotations" / "xmls").is_dir():
        img_dir = raw_india / "train" / "images"
        xml_dir = raw_india / "train" / "annotations" / "xmls"
    else:
        fixtures = ROOT_DIR / "tests" / "fixtures" / "generated"
        img_dir = fixtures / "road_images"
        xml_dir = fixtures / "road_xmls"

    rows = generate_audit_sheet(img_dir=img_dir, xml_dir=xml_dir)
    print(f"Generated {len(rows)} blind audit rows at {DEFAULT_SHEET_CSV}")


if __name__ == "__main__":
    main()
