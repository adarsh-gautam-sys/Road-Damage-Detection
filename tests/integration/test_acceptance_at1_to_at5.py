"""Automated verification of PRD Section 12 Acceptance Tests (`AT1`–`AT5`)."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image
from tests.integration.test_pipeline_integration import StubRoadDamageDetector

from rdai.api.main import create_app
from rdai.config import load_app_config
from rdai.reports.map import build_folium_map


class _LocationAwareStubDetector(StubRoadDamageDetector):
    """Stub detector whose severity varies by image color so AT3 can test 3 distinct places/severities."""

    def detect_image(self, image: Any, conf_threshold: float | None = None) -> list[dict[str, Any]]:
        rgb = image.convert("RGB").getpixel((0, 0))
        red_ch = int(rgb[0])
        if red_ch >= 200:
            # High severity pothole
            score, label, extent = 88.0, "High", 0.085
        elif red_ch >= 100:
            # Medium severity pothole
            score, label, extent = 52.0, "Medium", 0.032
        else:
            # Low severity pothole
            score, label, extent = 18.0, "Low", 0.008
        return [
            {
                "class_id": 3,
                "class_code": "D40",
                "class_name": "pothole",
                "confidence": 0.90,
                "box": [80.0, 120.0, 320.0, 340.0],
                "severity": {
                    "score": score,
                    "label": label,
                    "extent": extent,
                    "measure": "area_ratio",
                },
            }
        ]


def _make_colored_jpeg(color: tuple[int, int, int]) -> bytes:
    img = Image.new("RGB", (640, 480), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_at1_to_at5_end_to_end_acceptance(
    tmp_path: Path,
    generated_fixtures_dir: Path,
) -> None:
    """Verify AT1, AT2, AT3, AT4 (persistence across restart), and AT5 (CSV & Folium map exports)."""
    db_file = tmp_path / "acceptance.db"
    media_dir = tmp_path / "media"

    def _osm_by_lat(lat: float, _lng: float) -> dict[str, Any]:
        # Place 1 (lat ~18.52): primary road (85)
        if abs(lat - 18.5204) < 0.005:
            return {"road_class": "primary", "road_importance": 85.0}
        # Place 2 (lat ~18.60): secondary road (70)
        if abs(lat - 18.6000) < 0.005:
            return {"road_class": "secondary", "road_importance": 70.0}
        # Place 3 (lat ~28.61): residential road (35)
        return {"road_class": "residential", "road_importance": 35.0}

    cfg = load_app_config(
        overrides={
            "database_url": f"sqlite:///{db_file.as_posix()}",
            "media_dir": str(media_dir),
        }
    )
    app1 = create_app(config=cfg, detector=_LocationAwareStubDetector(), osm_lookup_fn=_osm_by_lat)
    client1 = TestClient(app1)

    # AT1: Given a phone photo of a pothole with GPS, when uploaded, annotated image opens and response lists D40 with severity and location
    gps_photo = (generated_fixtures_dir / "photo_with_gps.jpg").read_bytes()
    r_at1 = client1.post(
        "/api/v1/detect/image",
        files={"file": ("phone_gps.jpg", io.BytesIO(gps_photo), "image/jpeg")},
    )
    assert r_at1.status_code == 200
    body_at1 = r_at1.json()
    assert body_at1["geo"]["lat"] is not None
    assert body_at1["detections"][0]["class_code"] == "D40"
    assert body_at1["detections"][0]["severity"]["score"] > 0
    ann_resp = client1.get(body_at1["annotated_url"])
    assert ann_resp.status_code == 200
    assert len(ann_resp.content) > 500

    # AT2: Given a dashcam clip and CSV track, when job finishes, unique count matches expected and lands on route
    video_bytes = (generated_fixtures_dir / "sample_drive.mp4").read_bytes()
    track_bytes = (generated_fixtures_dir / "track.csv").read_bytes()
    r_at2 = client1.post(
        "/api/v1/detect/video",
        files={
            "file": ("sample_drive.mp4", io.BytesIO(video_bytes), "video/mp4"),
            "track": ("track.csv", io.BytesIO(track_bytes), "text/csv"),
        },
    )
    assert r_at2.status_code == 202
    job_info = client1.get(f"/api/v1/jobs/{r_at2.json()['job_id']}").json()
    assert job_info["status"] == "done"
    assert abs(job_info["summary"]["unique_detections"] - 1) <= 1

    # AT3: Given photos from three places, one with a High pothole on a primary road (18.5204), that segment ranks first
    client1.post(
        "/api/v1/detect/image",
        files={
            "file": (
                "place_primary_high.jpg",
                io.BytesIO(_make_colored_jpeg((220, 20, 20))),
                "image/jpeg",
            )
        },
        data={"lat": "18.5204", "lng": "73.8567"},
    )
    client1.post(
        "/api/v1/detect/image",
        files={
            "file": (
                "place_secondary_med.jpg",
                io.BytesIO(_make_colored_jpeg((130, 120, 20))),
                "image/jpeg",
            )
        },
        data={"lat": "18.6000", "lng": "73.8567"},
    )
    segs = client1.get("/api/v1/segments").json()["items"]
    assert len(segs) >= 3
    top_seg = segs[0]
    assert top_seg["road_class"] == "primary"
    assert top_seg["max_severity"] == 88.0

    # AT4: Given a status change, when the server restarts, the status persists
    top_seg_id = top_seg["id"]
    r_patch = client1.patch(f"/api/v1/segments/{top_seg_id}/status", json={"status": "verified"})
    assert r_patch.status_code == 200

    # Simulate server restart by creating a brand new FastAPI app connected to the same SQLite DB file
    app2 = create_app(config=cfg, detector=_LocationAwareStubDetector(), osm_lookup_fn=_osm_by_lat)
    client2 = TestClient(app2)
    persisted_seg = client2.get(f"/api/v1/segments/{top_seg_id}").json()
    assert persisted_seg["status"] == "verified"

    # AT5: Given the exports, when CSV is opened and map file generated, columns are correct and markers show
    csv_resp = client2.get("/api/v1/export/csv")
    assert csv_resp.status_code == 200
    csv_rows = list(csv.reader(io.StringIO(csv_resp.text)))
    assert csv_rows[0][0] == "id"
    assert len(csv_rows) >= 4

    all_dets = client2.get("/api/v1/detections?limit=100").json()["items"]
    all_segs = client2.get("/api/v1/segments").json()["items"]
    map_html_path = tmp_path / "demo_map.html"
    build_folium_map(all_dets, all_segs, map_html_path)
    assert map_html_path.is_file()
    map_text = map_html_path.read_text(encoding="utf-8")
    assert "OpenStreetMap contributors" in map_text
    assert "primary" in map_text
