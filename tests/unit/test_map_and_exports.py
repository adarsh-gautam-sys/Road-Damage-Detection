"""Unit tests for Folium map generation (`src/rdai/reports/map.py`) and CSV/GeoJSON exports (`src/rdai/reports/export.py`)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rdai.reports.export import (
    build_top_segments_table,
    export_detections_to_csv,
    export_to_geojson,
)
from rdai.reports.map import build_folium_map


def test_build_folium_map_writes_self_contained_mobile_html(tmp_path: Path) -> None:
    """Folium map includes OpenStreetMap attribution, severity markers, priority circles, and thumbnail popups."""
    detections = [
        {
            "id": 101,
            "image_id": 1,
            "class_code": "D40",
            "class_name": "pothole",
            "confidence": 0.88,
            "severity_score": 74.2,
            "severity_label": "High",
            "lat": 18.5204,
            "lng": 73.8567,
            "segment_id": 1,
            "thumb_url": "/media/thumbs/1.jpg",
        },
        {
            "id": 102,
            "image_id": 2,
            "class_code": "D00",
            "class_name": "longitudinal_crack",
            "confidence": 0.76,
            "severity_score": 25.0,
            "severity_label": "Low",
            "lat": 18.5250,
            "lng": 73.8600,
            "segment_id": 2,
            "thumb_url": "/media/thumbs/2.jpg",
        },
    ]
    segments = [
        {
            "id": 1,
            "centroid_lat": 18.5204,
            "centroid_lng": 73.8567,
            "n_detections": 1,
            "max_severity": 74.2,
            "mean_confidence": 0.88,
            "road_class": "primary",
            "priority_score": 61.69,
            "priority_label": "Medium",
            "status": "new",
            "terms": {"s_max": 74.2, "density": 10.0, "road_importance": 85.0, "confidence": 88.0},
        },
        {
            "id": 2,
            "centroid_lat": 18.5250,
            "centroid_lng": 73.8600,
            "n_detections": 1,
            "max_severity": 25.0,
            "mean_confidence": 0.76,
            "road_class": "residential",
            "priority_score": 28.35,
            "priority_label": "Low",
            "status": "verified",
            "terms": {"s_max": 25.0, "density": 10.0, "road_importance": 35.0, "confidence": 76.0},
        },
    ]

    out_html = tmp_path / "map.html"
    res_path = build_folium_map(detections=detections, segments=segments, output_html_path=out_html)
    assert res_path == out_html
    assert out_html.is_file()

    html = out_html.read_text(encoding="utf-8")
    assert "OpenStreetMap contributors" in html
    assert "D40" in html
    assert "61.69" in html
    assert "viewport" in html


def test_export_csv_and_geojson_and_top_segments(tmp_path: Path) -> None:
    """Exports write valid CSV, GeoJSON FeatureCollection, and Top 10 segments markdown/CSV."""
    detections = [
        {
            "id": 1,
            "image_id": 1,
            "class_code": "D40",
            "class_name": "pothole",
            "confidence": 0.85,
            "x1": 10.0,
            "y1": 20.0,
            "x2": 100.0,
            "y2": 120.0,
            "extent": 0.05,
            "measure": "area_ratio",
            "severity_score": 68.0,
            "severity_label": "High",
            "lat": 18.5204,
            "lng": 73.8567,
            "segment_id": 1,
            "source_filename": "pune,road.jpg",
        }
    ]
    segments = [
        {
            "id": 1,
            "centroid_lat": 18.5204,
            "centroid_lng": 73.8567,
            "n_detections": 1,
            "max_severity": 68.0,
            "mean_confidence": 0.85,
            "road_class": "primary",
            "priority_score": 58.6,
            "priority_label": "Medium",
            "status": "new",
            "terms": {"s_max": 68.0, "density": 10.0, "road_importance": 85.0, "confidence": 85.0},
        }
    ]

    csv_path = tmp_path / "detections.csv"
    export_detections_to_csv(detections, csv_path)
    rows = list(csv.reader(csv_path.open("r", encoding="utf-8")))
    assert len(rows) == 2
    assert rows[1][2] == "D40"
    assert rows[1][-1] == "pune,road.jpg"

    geojson_path = tmp_path / "detections.geojson"
    fc = export_to_geojson(detections, segments, layer="detections", output_path=geojson_path)
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 1
    loaded_fc = json.loads(geojson_path.read_text(encoding="utf-8"))
    assert loaded_fc["features"][0]["geometry"]["coordinates"] == [73.8567, 18.5204]

    top_csv = tmp_path / "top_segments.csv"
    top_md = tmp_path / "top_segments.md"
    build_top_segments_table(segments, csv_path=top_csv, md_path=top_md)
    assert top_csv.is_file()
    assert top_md.is_file()
    assert "58.6" in top_md.read_text(encoding="utf-8")
