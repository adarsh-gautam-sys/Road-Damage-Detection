"""Tests for the frontend-support routes: live preview, metrics, report figures, segment detail."""

from __future__ import annotations

import io
from pathlib import Path

from tests.api.test_api_routes import _image_bytes, _make_client


def test_preview_is_stateless_and_returns_detections(tmp_path: Path) -> None:
    """`/detect/preview` returns boxes + latency and never writes DB rows."""
    client = _make_client(tmp_path)
    resp = client.post(
        "/api/v1/detect/preview",
        files={"file": ("frame.jpg", io.BytesIO(_image_bytes()), "image/jpeg")},
        data={"conf": "0.193"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["width"] == 640 and body["height"] == 480
    assert body["conf_used"] == 0.193
    assert body["inference_ms"] >= 0.0
    assert isinstance(body["detections"], list)

    stats = client.get("/api/v1/stats").json()
    assert stats["total_images"] == 0
    assert stats["total_detections"] == 0


def test_preview_rejects_bad_conf(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    resp = client.post(
        "/api/v1/detect/preview",
        files={"file": ("frame.jpg", io.BytesIO(_image_bytes()), "image/jpeg")},
        data={"conf": "1.7"},
    )
    assert resp.status_code == 422


def test_metrics_route_serves_metrics_json() -> None:
    """`/api/v1/metrics` mirrors reports/metrics.json (never invented numbers)."""
    import json

    from fastapi.testclient import TestClient

    from rdai.api.main import create_app

    root = Path(__file__).resolve().parents[2]
    expected = json.loads((root / "reports" / "metrics.json").read_text(encoding="utf-8"))
    client = TestClient(create_app())
    body = client.get("/api/v1/metrics").json()
    assert body["experiments"]["E1"]["overall"] == expected["experiments"]["E1"]["overall"]


def test_config_route_exposes_yaml_driven_client_settings(tmp_path: Path) -> None:
    """`/api/v1/config` surfaces thresholds from configs/*.yaml so the UI hard-codes none."""
    client = _make_client(tmp_path)
    body = client.get("/api/v1/config").json()

    weights = body["priority"]["weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert body["priority"]["eps_meters"] > 0
    assert body["priority"]["bands"]["low_max"] < body["priority"]["bands"]["medium_max"]
    assert (
        body["severity"]["tier_bounds"]["low_upper"]
        < body["severity"]["tier_bounds"]["medium_upper"]
    )
    assert set(body["severity"]["classes"]) == {"D00", "D10", "D20", "D40"}

    c = body["client"]
    assert c["auto_patrol_min_distance_m"] > 0
    assert c["auto_patrol_min_interval_s"] > 0
    assert c["preview_fps"] > 0
    assert c["upload_max_edge_px"] == 1280
    assert 0.0 < body["default_conf"] <= 1.0
    assert body["limits"]["max_photo_mb"] > 0 and body["limits"]["max_video_mb"] > 0


def test_report_figure_route_blocks_traversal_and_404s(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    assert client.get("/reports/figures/does_not_exist.png").status_code == 404
    assert client.get("/reports/figures/..%2Fmetrics.json").status_code in (400, 404)
    ok = client.get("/reports/figures/PR_curve.png")
    assert ok.status_code == 200
    assert ok.headers["content-type"] == "image/png"


def test_segment_detail_includes_box_extent_and_thumb(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    up = client.post(
        "/api/v1/detect/image",
        files={"file": ("a.jpg", io.BytesIO(_image_bytes()), "image/jpeg")},
        data={"lat": "28.4595", "lng": "77.0266"},
    )
    seg_id = up.json()["detections"][0]["segment_id"]
    seg = client.get(f"/api/v1/segments/{seg_id}").json()
    det = seg["detections"][0]
    assert {"box", "extent", "measure", "thumb_url", "annotated_url"} <= set(det)
