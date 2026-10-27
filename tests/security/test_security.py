"""Security tests (`S01`–`S04`) for path traversal, decompression bomb, SQL injection, and EXIF privacy."""

from __future__ import annotations

import io
from pathlib import Path

import piexif
from fastapi.testclient import TestClient
from PIL import Image
from tests.integration.test_pipeline_integration import StubRoadDamageDetector

from rdai.api.main import create_app
from rdai.config import load_app_config


def _make_security_client(tmp_path: Path, max_megapixels: float = 40.0) -> tuple[TestClient, Path]:
    db_file = tmp_path / "security.db"
    media_dir = tmp_path / "media"
    cfg = load_app_config(
        overrides={
            "database_url": f"sqlite:///{db_file.as_posix()}",
            "media_dir": str(media_dir),
            "max_megapixels": max_megapixels,
        }
    )
    app = create_app(
        config=cfg,
        detector=StubRoadDamageDetector(),
        osm_lookup_fn=lambda lat, lng: {"road_class": "secondary", "road_importance": 70.0},
    )
    return TestClient(app), media_dir


def test_s01_path_traversal_in_filenames_and_media_route(tmp_path: Path) -> None:
    """S01: Path traversal in uploaded filenames or `/media/...` requests never touches files outside `media_dir`."""
    client, media_dir = _make_security_client(tmp_path)

    img = Image.new("RGB", (320, 240), color=(80, 90, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    resp = client.post(
        "/api/v1/detect/image",
        files={"file": ("../../../outside_escape.jpg", io.BytesIO(buf.getvalue()), "image/jpeg")},
    )
    assert resp.status_code == 200
    assert not (tmp_path / "outside_escape.jpg").exists()

    # Ensure media directory contains only expected subfolders inside media_dir
    for p in media_dir.rglob("*"):
        assert media_dir.resolve() in p.resolve().parents

    # Attempt to read outside media_dir via URL traversal
    r_traversal = client.get("/media/..%2F..%2Fpyproject.toml")
    assert r_traversal.status_code in (400, 404)


def test_s02_decompression_bomb_image_rejected_quickly(tmp_path: Path) -> None:
    """S02: Decompression bomb image exceeding the pixel cap is rejected quickly without exhausting memory."""
    # Cap at 1.0 MP so a 2000x2000 (4 MP) image triggers the decompression bomb / pixel cap guard immediately
    client, _ = _make_security_client(tmp_path, max_megapixels=1.0)

    bomb_img = Image.new("RGB", (2000, 2000), color=(0, 0, 0))
    buf = io.BytesIO()
    bomb_img.save(buf, format="PNG")

    resp = client.post(
        "/api/v1/detect/image",
        files={"file": ("bomb.png", io.BytesIO(buf.getvalue()), "image/png")},
    )
    assert resp.status_code == 422
    assert "error" in resp.json()


def test_s03_sql_injection_strings_in_query_params(tmp_path: Path) -> None:
    """S03: SQL injection strings in query parameters return 422 or an empty list, never 500."""
    client, _ = _make_security_client(tmp_path)

    sqli_payloads = [
        "/api/v1/detections?class=' OR '1'='1",
        "/api/v1/detections?severity=High' UNION SELECT * FROM images--",
        "/api/v1/detections?status=new'; DROP TABLE detections;--",
        "/api/v1/detections?bbox=0,0,10,10' OR 1=1--",
        "/api/v1/detections?since=2026-01-01' OR '1'='1",
        "/api/v1/segments?label=' OR 1=1--",
        "/api/v1/segments?status='; DROP TABLE segments;--",
    ]
    for url in sqli_payloads:
        resp = client.get(url)
        assert resp.status_code in (200, 422), f"Unexpected {resp.status_code} for {url}"
        if resp.status_code == 200:
            assert resp.json()["items"] == []


def test_s04_served_media_images_carry_no_exif_gps(
    tmp_path: Path,
    generated_fixtures_dir: Path,
) -> None:
    """S04: Served `/media/annotated/...` and `/media/thumbs/...` images contain zero EXIF GPS tags."""
    client, _ = _make_security_client(tmp_path)
    gps_photo_bytes = (generated_fixtures_dir / "photo_with_gps.jpg").read_bytes()

    # Confirm input fixture has EXIF GPS
    in_exif = piexif.load(gps_photo_bytes)
    assert in_exif.get("GPS")

    resp = client.post(
        "/api/v1/detect/image",
        files={"file": ("photo_with_gps.jpg", io.BytesIO(gps_photo_bytes), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()

    for media_url in (body["annotated_url"], body["thumb_url"]):
        m_resp = client.get(media_url)
        assert m_resp.status_code == 200
        served_exif = piexif.load(m_resp.content)
        assert served_exif.get("GPS", {}) == {}
        assert served_exif.get("Exif", {}) == {}
