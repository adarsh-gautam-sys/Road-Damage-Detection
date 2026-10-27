"""Generate `reports/map.html`, `reports/segments.geojson`, and `reports/tables/top_segments.csv` (`PRD Section 10 & 13`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import select

from rdai.config import load_app_config
from rdai.db.models import DetectionRecord, SegmentRecord
from rdai.db.session import create_db_engine_and_session
from rdai.priority.score import evaluate_weight_sensitivity, recluster_and_rank_segments
from rdai.reports.export import (
    build_top_segments_table,
    export_detections_to_csv,
    export_to_geojson,
)
from rdai.reports.map import build_folium_map

ROOT_DIR = Path(__file__).resolve().parents[1]


def _default_demo_detections() -> list[dict[str, Any]]:
    """Deterministic Indian highway/urban road damage sample for demo map and Top-10 segments table."""
    corridors = [
        # (id, lat, lng, class_code, class_name, conf, sev_score, sev_label, road_class)
        (1, 18.5204, 73.8567, "D40", "pothole", 0.91, 88.4, "High", "motorway"),
        (2, 18.5205, 73.8568, "D20", "alligator_crack", 0.87, 79.2, "High", "motorway"),
        (3, 18.5206, 73.8567, "D40", "pothole", 0.85, 74.0, "High", "motorway"),
        (4, 28.6139, 77.2090, "D40", "pothole", 0.89, 82.5, "High", "trunk"),
        (5, 28.6140, 77.2091, "D00", "longitudinal_crack", 0.82, 61.0, "Medium", "trunk"),
        (6, 19.0760, 72.8777, "D40", "pothole", 0.86, 76.8, "High", "primary"),
        (7, 12.9716, 77.5946, "D20", "alligator_crack", 0.84, 71.5, "High", "primary"),
        (8, 13.0827, 80.2707, "D40", "pothole", 0.80, 60.0, "Medium", "secondary"),
        (9, 17.3850, 78.4867, "D00", "longitudinal_crack", 0.78, 54.5, "Medium", "secondary"),
        (10, 22.5726, 88.3639, "D10", "transverse_crack", 0.76, 48.0, "Medium", "tertiary"),
        (11, 23.0225, 72.5714, "D00", "longitudinal_crack", 0.75, 42.0, "Medium", "tertiary"),
        (12, 26.9124, 75.7873, "D00", "longitudinal_crack", 0.72, 26.5, "Low", "residential"),
        (13, 21.1458, 79.0882, "D10", "transverse_crack", 0.70, 19.0, "Low", "residential"),
    ]
    return [
        {
            "id": cid,
            "image_id": cid,
            "class_code": c_code,
            "class_name": c_name,
            "confidence": conf,
            "x1": 100.0,
            "y1": 180.0,
            "x2": 290.0,
            "y2": 340.0,
            "extent": 0.055 if sev_label == "High" else (0.028 if sev_label == "Medium" else 0.01),
            "measure": "area_ratio" if c_code in {"D20", "D40"} else "diagonal_ratio",
            "severity_score": sev_score,
            "severity_label": sev_label,
            "lat": lat,
            "lng": lng,
            "road_class": r_cls,
            "source_filename": f"india_demo_{cid:02d}.jpg",
        }
        for cid, lat, lng, c_code, c_name, conf, sev_score, sev_label, r_cls in corridors
    ]


def generate_map_and_exports(
    output_html_path: Path = ROOT_DIR / "reports" / "map.html",
    tables_dir: Path = ROOT_DIR / "reports" / "tables",
) -> dict[str, Any]:
    """Load DB rows (or deterministic demo corridor data) and generate Folium map + CSV/GeoJSON/Top-10 tables."""
    cfg = load_app_config()
    _engine, session_factory = create_db_engine_and_session(cfg.database_url)

    with session_factory() as session:
        db_dets = list(session.scalars(select(DetectionRecord).order_by(DetectionRecord.id)).all())
        db_segs = list(
            session.scalars(
                select(SegmentRecord).order_by(SegmentRecord.priority_score.desc())
            ).all()
        )

    if db_dets and db_segs:
        detections = [
            {
                "id": d.id,
                "image_id": d.image_id,
                "class_code": d.class_code,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "x1": d.x1,
                "y1": d.y1,
                "x2": d.x2,
                "y2": d.y2,
                "extent": d.extent,
                "measure": d.measure,
                "severity_score": d.severity_score,
                "severity_label": d.severity_label,
                "lat": d.lat,
                "lng": d.lng,
                "segment_id": d.segment_id,
            }
            for d in db_dets
        ]
        segments = [
            {
                "id": s.id,
                "centroid_lat": s.centroid_lat,
                "centroid_lng": s.centroid_lng,
                "n_detections": s.n_detections,
                "max_severity": s.max_severity,
                "mean_confidence": s.mean_confidence,
                "road_class": s.road_class,
                "priority_score": s.priority_score,
                "priority_label": s.priority_label,
                "terms": json.loads(s.terms_json or "{}"),
                "status": s.status,
            }
            for s in db_segs
        ]
    else:
        raw_demo = _default_demo_detections()
        reclustered = recluster_and_rank_segments(
            detections=raw_demo,
            existing_segments=[],
            osm_lookup_fn=lambda _lat, _lng: {"road_class": "secondary", "road_importance": 70.0},
        )
        detections = reclustered["detections"]
        segments = reclustered["segments"]

    media_root = (ROOT_DIR / cfg.media_dir).resolve()
    build_folium_map(
        detections=detections,
        segments=segments,
        output_html_path=output_html_path,
        media_root=media_root,
    )

    tables_dir.mkdir(parents=True, exist_ok=True)
    export_detections_to_csv(detections, tables_dir / "detections_export.csv")
    export_to_geojson(
        detections,
        segments,
        layer="segments",
        output_path=ROOT_DIR / "reports" / "segments.geojson",
    )
    top_rows = build_top_segments_table(
        segments,
        csv_path=tables_dir / "top_segments.csv",
        md_path=tables_dir / "top_segments.md",
    )

    sensitivity = evaluate_weight_sensitivity(segments, perturbation=0.20)
    (tables_dir / "weight_sensitivity.json").write_text(
        json.dumps(sensitivity, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "map_html": str(output_html_path),
        "n_detections": len(detections),
        "n_segments": len(segments),
        "top_segment_priority": top_rows[0]["Priority Score"] if top_rows else 0.0,
        "min_spearman_rho": sensitivity["min_spearman_rho"],
    }


if __name__ == "__main__":
    info = generate_map_and_exports()
    print(f"Generated map & exports: {info}")
