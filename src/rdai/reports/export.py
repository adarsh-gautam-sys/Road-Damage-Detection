"""CSV, GeoJSON, and Top-10 Segment report exporters (`PRD Section 10, 11, 13`)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

CSV_DETECTION_COLUMNS = [
    "id",
    "image_id",
    "class_code",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "extent",
    "measure",
    "severity_score",
    "severity_label",
    "lat",
    "lng",
    "segment_id",
    "source_filename",
]


def export_detections_to_csv(
    detections: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    """Write detections to an RFC-4180 CSV file with documented columns (`A12`, `AT5`)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(CSV_DETECTION_COLUMNS)
        for d in detections:
            sev = d.get("severity") if isinstance(d.get("severity"), dict) else {}
            box = (
                d.get("box")
                if isinstance(d.get("box"), list) and len(d["box"]) == 4
                else [
                    d.get("x1", 0.0),
                    d.get("y1", 0.0),
                    d.get("x2", 0.0),
                    d.get("y2", 0.0),
                ]
            )
            writer.writerow(
                [
                    d.get("id", ""),
                    d.get("image_id", "") if d.get("image_id") is not None else "",
                    d.get("class_code", ""),
                    d.get("class_name", ""),
                    round(float(d.get("confidence", 0.0)), 4),
                    round(float(box[0]), 2),
                    round(float(box[1]), 2),
                    round(float(box[2]), 2),
                    round(float(box[3]), 2),
                    round(float(d.get("extent", sev.get("extent", 0.0))), 4),
                    d.get("measure", sev.get("measure", "")),
                    round(float(d.get("severity_score", sev.get("score", 0.0))), 2),
                    d.get("severity_label", sev.get("label", "")),
                    d.get("lat", "") if d.get("lat") is not None else "",
                    d.get("lng", "") if d.get("lng") is not None else "",
                    d.get("segment_id", "") if d.get("segment_id") is not None else "",
                    d.get("source_filename", ""),
                ]
            )
    return output_path


def export_to_geojson(
    detections: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    layer: str = "detections",
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Build a GeoJSON `FeatureCollection` in `[lng, lat]` coordinate order, excluding unlocated points (`A07`)."""
    features: list[dict[str, Any]] = []
    if layer == "detections":
        for d in detections:
            if d.get("lat") is None or d.get("lng") is None:
                continue
            sev = d.get("severity") if isinstance(d.get("severity"), dict) else {}
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(d["lng"]), float(d["lat"])],
                    },
                    "properties": {
                        "id": d.get("id"),
                        "image_id": d.get("image_id"),
                        "class_code": d.get("class_code"),
                        "class_name": d.get("class_name"),
                        "confidence": round(float(d.get("confidence", 0.0)), 4),
                        "severity_score": round(
                            float(d.get("severity_score", sev.get("score", 0.0))), 2
                        ),
                        "severity_label": d.get("severity_label", sev.get("label", "")),
                        "segment_id": d.get("segment_id"),
                    },
                }
            )
    else:
        for s in segments:
            if s.get("centroid_lat") is None or s.get("centroid_lng") is None:
                continue
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(s["centroid_lng"]), float(s["centroid_lat"])],
                    },
                    "properties": {
                        "id": s.get("id"),
                        "n_detections": s.get("n_detections"),
                        "max_severity": s.get("max_severity"),
                        "road_class": s.get("road_class"),
                        "priority_score": s.get("priority_score"),
                        "priority_label": s.get("priority_label"),
                        "status": s.get("status", "new"),
                    },
                }
            )

    fc = {"type": "FeatureCollection", "features": features}
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(fc, indent=2) + "\n", encoding="utf-8")
    return fc


def build_top_segments_table(
    segments: list[dict[str, Any]],
    csv_path: Path,
    md_path: Path,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Write the PRD Section 13 Top-10 Segments table (`Priority`, 4 terms, `Label`, `Status`)."""
    ranked = sorted(
        segments,
        key=lambda s: (-float(s.get("priority_score", 0.0)), int(s.get("id", 0))),
    )[:top_k]

    fieldnames = [
        "Rank",
        "Segment ID",
        "Priority Score",
        "Label",
        "S_max",
        "Density (D)",
        "Road Importance (R)",
        "Confidence (C)",
        "Road Class",
        "Detections",
        "Status",
    ]
    rows: list[dict[str, Any]] = []
    for idx, seg in enumerate(ranked, start=1):
        terms = seg.get("terms", {}) or {}
        rows.append(
            {
                "Rank": idx,
                "Segment ID": seg.get("id", idx),
                "Priority Score": round(float(seg.get("priority_score", 0.0)), 2),
                "Label": seg.get("priority_label", "Medium"),
                "S_max": round(float(terms.get("s_max", seg.get("max_severity", 0.0))), 2),
                "Density (D)": round(float(terms.get("density", 10.0)), 2),
                "Road Importance (R)": round(float(terms.get("road_importance", 50.0)), 1),
                "Confidence (C)": round(
                    float(terms.get("confidence", float(seg.get("mean_confidence", 0.0)) * 100.0)),
                    1,
                ),
                "Road Class": seg.get("road_class", "unknown"),
                "Detections": seg.get("n_detections", 1),
                "Status": seg.get("status", "new"),
            }
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    md_lines = [
        "# Top Road Segments by Maintenance Priority",
        "",
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(":---" for _ in fieldnames) + " |",
    ]
    for r in rows:
        md_lines.append("| " + " | ".join(str(r[k]) for k in fieldnames) + " |")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return rows
