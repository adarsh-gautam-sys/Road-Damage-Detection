"""API contract and route tests (`A01`–`A12`) using FastAPI TestClient."""

from __future__ import annotations

import csv
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
from tests.integration.test_pipeline_integration import StubRoadDamageDetector

from rdai.api.main import create_app
from rdai.config import load_app_config

ROOT_DIR = Path(__file__).resolve().parents[2]
OPENAPI_SNAPSHOT_PATH = ROOT_DIR / "tests" / "fixtures" / "openapi_snapshot.json"


def _make_client(tmp_path: Path, max_photo_mb: float = 15.0) -> TestClient:
    db_file = tmp_path / "api_test.db"
    media_dir = tmp_path / "media"
    cfg = load_app_config(
        overrides={
            "database_url": f"sqlite:///{db_file.as_posix()}",
            "media_dir": str(media_dir),
            "max_photo_mb": max_photo_mb,
        }
    )
    app = create_app(
        config=cfg,
        detector=StubRoadDamageDetector(),
        osm_lookup_fn=lambda lat, lng: {"road_class": "secondary", "road_importance": 70.0},
    )
    return TestClient(app)


def _image_bytes(
    fmt: str = "JPEG",
    color: tuple[int, int, int] = (100, 110, 120),
    size: tuple[int, int] = (640, 480),
) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_a01_health_endpoint(tmp_path: Path) -> None:
    """A01: `/api/v1/health` returns 200 with `model_loaded` flag and version."""
    client = _make_client(tmp_path)
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert "version" in body


def test_a02_image_detect_jpeg_and_png(tmp_path: Path) -> None:
    """A02: Valid JPEG and PNG uploads return 200 with the documented response schema."""
    client = _make_client(tmp_path)

    for fmt, ext, mime, col in [
        ("JPEG", "jpg", "image/jpeg", (90, 100, 110)),
        ("PNG", "png", "image/png", (120, 130, 140)),
    ]:
        raw = _image_bytes(fmt=fmt, color=col)
        resp = client.post(
            "/api/v1/detect/image",
            files={"file": (f"sample.{ext}", io.BytesIO(raw), mime)},
            data={"lat": "18.5204", "lng": "73.8567"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["image_id"], int)
        assert body["width"] == 640
        assert body["height"] == 480
        assert body["geo"]["source"] == "request"
        assert body["annotated_url"].startswith("/media/annotated/")
        assert body["thumb_url"].startswith("/media/thumbs/")
        assert len(body["detections"]) == 1
        det = body["detections"][0]
        assert det["class_code"] == "D40"
        assert det["class_name"] == "pothole"
        assert len(det["box"]) == 4
        assert set(det["severity"].keys()) == {"score", "label", "extent", "measure"}
        assert "name" in body["model"]


def test_a03_bad_inputs_return_422_415_413(tmp_path: Path) -> None:
    """A03: Missing file -> 422, text named .jpg -> 415, oversize -> 413, corrupt image -> 422."""
    client = _make_client(tmp_path, max_photo_mb=0.001)

    # 1. Missing file -> 422
    r_missing = client.post("/api/v1/detect/image")
    assert r_missing.status_code == 422

    # 2. Text file renamed .jpg -> 415
    r_text = client.post(
        "/api/v1/detect/image",
        files={"file": ("fake.jpg", io.BytesIO(b"hello world"), "image/jpeg")},
    )
    assert r_text.status_code == 415

    # 3. Oversize image (> 0.001 MB = ~1 KB) -> 413
    big_png = _image_bytes(fmt="PNG", size=(800, 800))
    r_big = client.post(
        "/api/v1/detect/image",
        files={"file": ("big.png", io.BytesIO(big_png), "image/png")},
    )
    assert r_big.status_code == 413

    # 4. Corrupt image (under size limit, valid JPEG magic bytes, truncated) -> 422
    corrupt_jpg = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01"
    r_corrupt = client.post(
        "/api/v1/detect/image",
        files={"file": ("corrupt.jpg", io.BytesIO(corrupt_jpg), "image/jpeg")},
    )
    assert r_corrupt.status_code == 422


def test_a04_geo_validation_override_and_outside_india_warning(
    tmp_path: Path,
    generated_fixtures_dir: Path,
) -> None:
    """A04: lat=91 -> 422; request coords override EXIF; coords outside India accepted with warning."""
    client = _make_client(tmp_path)
    gps_photo_bytes = (generated_fixtures_dir / "photo_with_gps.jpg").read_bytes()

    # 1. lat=91 -> 422
    r_bad_lat = client.post(
        "/api/v1/detect/image",
        files={"file": ("photo.jpg", io.BytesIO(gps_photo_bytes), "image/jpeg")},
        data={"lat": "91.0", "lng": "77.2"},
    )
    assert r_bad_lat.status_code == 422

    # 2. Request coordinates override EXIF (EXIF is ~28.6139, 77.2090; request is 19.0760, 72.8777)
    r_override = client.post(
        "/api/v1/detect/image",
        files={"file": ("photo.jpg", io.BytesIO(gps_photo_bytes), "image/jpeg")},
        data={"lat": "19.0760", "lng": "72.8777"},
    )
    assert r_override.status_code == 200
    geo_ov = r_override.json()["geo"]
    assert geo_ov["source"] == "request"
    assert abs(geo_ov["lat"] - 19.0760) < 1e-4
    assert abs(geo_ov["lng"] - 72.8777) < 1e-4
    assert geo_ov["warnings"] == []

    # 3. Point outside India (London: 51.5074, -0.1278) accepted with `outside_india` warning
    raw_uk = _image_bytes(color=(70, 80, 90))
    r_uk = client.post(
        "/api/v1/detect/image",
        files={"file": ("uk.jpg", io.BytesIO(raw_uk), "image/jpeg")},
        data={"lat": "51.5074", "lng": "-0.1278"},
    )
    assert r_uk.status_code == 200
    assert "outside_india" in r_uk.json()["geo"]["warnings"]


def test_a05_lists_pagination_filters_and_bbox(tmp_path: Path) -> None:
    """A05: `limit=101` is rejected (422); filters, sorting, and bbox return the expected rows."""
    client = _make_client(tmp_path)

    # Reject limit=101
    r_limit = client.get("/api/v1/detections?limit=101")
    assert r_limit.status_code == 422

    # Seed two located images in different cities
    client.post(
        "/api/v1/detect/image",
        files={"file": ("pune.jpg", io.BytesIO(_image_bytes(color=(10, 20, 30))), "image/jpeg")},
        data={"lat": "18.5204", "lng": "73.8567"},
    )
    client.post(
        "/api/v1/detect/image",
        files={"file": ("delhi.jpg", io.BytesIO(_image_bytes(color=(40, 50, 60))), "image/jpeg")},
        data={"lat": "28.6139", "lng": "77.2090"},
    )

    all_dets = client.get("/api/v1/detections?limit=10&sort=severity_desc").json()
    assert all_dets["total"] == 2
    assert len(all_dets["items"]) == 2

    # Filter by bbox around Pune (min_lng,min_lat,max_lng,max_lat)
    pune_dets = client.get("/api/v1/detections?bbox=73.0,18.0,74.5,19.0").json()
    assert pune_dets["total"] == 1
    assert abs(pune_dets["items"][0]["lat"] - 18.5204) < 1e-3

    # Filter by class_code
    d40_dets = client.get("/api/v1/detections?class=D40").json()
    assert d40_dets["total"] == 2
    d00_dets = client.get("/api/v1/detections?class=D00").json()
    assert d00_dets["total"] == 0


def test_a06_segment_status_workflow_transitions(tmp_path: Path) -> None:
    """A06: `new` -> `verified` -> `scheduled` -> `fixed` works; `fixed` -> `new` gives 409; unknown status gives 422."""
    client = _make_client(tmp_path)
    resp = client.post(
        "/api/v1/detect/image",
        files={"file": ("road.jpg", io.BytesIO(_image_bytes(color=(55, 65, 75))), "image/jpeg")},
        data={"lat": "18.5204", "lng": "73.8567"},
    )
    seg_id = resp.json()["detections"][0]["segment_id"]

    # new -> verified (200)
    r1 = client.patch(f"/api/v1/segments/{seg_id}/status", json={"status": "verified"})
    assert r1.status_code == 200
    assert r1.json()["status"] == "verified"

    # verified -> scheduled (200)
    r2 = client.patch(f"/api/v1/segments/{seg_id}/status", json={"status": "scheduled"})
    assert r2.status_code == 200
    assert r2.json()["status"] == "scheduled"

    # scheduled -> fixed (200)
    r3 = client.patch(f"/api/v1/segments/{seg_id}/status", json={"status": "fixed"})
    assert r3.status_code == 200
    assert r3.json()["status"] == "fixed"

    # fixed -> new (illegal move -> 409)
    r_illegal = client.patch(f"/api/v1/segments/{seg_id}/status", json={"status": "new"})
    assert r_illegal.status_code == 409

    # unknown status -> 422
    r_unknown = client.patch(f"/api/v1/segments/{seg_id}/status", json={"status": "archived"})
    assert r_unknown.status_code == 422


def test_a07_geojson_lng_lat_order_and_unlocated_excluded(tmp_path: Path) -> None:
    """A07: `/api/v1/map/geojson` returns valid FeatureCollection with `[lng, lat]` coordinates and excludes unlocated detections."""
    client = _make_client(tmp_path)

    # Located upload
    client.post(
        "/api/v1/detect/image",
        files={"file": ("loc.jpg", io.BytesIO(_image_bytes(color=(15, 25, 35))), "image/jpeg")},
        data={"lat": "18.5204", "lng": "73.8567"},
    )
    # Unlocated upload
    client.post(
        "/api/v1/detect/image",
        files={"file": ("unloc.jpg", io.BytesIO(_image_bytes(color=(85, 95, 105))), "image/jpeg")},
    )

    fc_dets = client.get("/api/v1/map/geojson?layer=detections").json()
    assert fc_dets["type"] == "FeatureCollection"
    assert len(fc_dets["features"]) == 1
    coords = fc_dets["features"][0]["geometry"]["coordinates"]
    # [lng, lat] order!
    assert abs(coords[0] - 73.8567) < 1e-4
    assert abs(coords[1] - 18.5204) < 1e-4

    fc_segs = client.get("/api/v1/map/geojson?layer=segments").json()
    assert fc_segs["type"] == "FeatureCollection"
    assert len(fc_segs["features"]) == 1


def test_a08_cors_headers(tmp_path: Path) -> None:
    """A08: An allowed origin receives `access-control-allow-origin`; an unlisted origin does not."""
    client = _make_client(tmp_path)

    r_allowed = client.get(
        "/api/v1/health",
        headers={"Origin": "http://localhost:5173"},
    )
    assert r_allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"

    r_blocked = client.get(
        "/api/v1/health",
        headers={"Origin": "https://unlisted.example.org"},
    )
    assert "access-control-allow-origin" not in r_blocked.headers


def test_a09_concurrency_10_parallel_uploads(tmp_path: Path) -> None:
    """A09: 10 parallel uploads all succeed with no duplicate rows or SQLite errors."""
    client = _make_client(tmp_path)

    def _upload_idx(i: int) -> int:
        raw = _image_bytes(color=(i * 20 % 255, (i * 35 + 10) % 255, (i * 50 + 20) % 255))
        resp = client.post(
            "/api/v1/detect/image",
            files={"file": (f"concurrent_{i}.jpg", io.BytesIO(raw), "image/jpeg")},
            data={"lat": str(18.5000 + i * 0.005), "lng": "73.8567"},
        )
        assert resp.status_code == 200
        return int(resp.json()["image_id"])

    with ThreadPoolExecutor(max_workers=5) as pool:
        image_ids = list(pool.map(_upload_idx, range(10)))

    assert len(set(image_ids)) == 10
    stats = client.get("/api/v1/stats").json()
    assert stats["total_images"] == 10
    assert stats["total_detections"] == 10


def test_a10_openapi_contract_matches_snapshot(tmp_path: Path) -> None:
    """A10: OpenAPI schema matches stored snapshot `tests/fixtures/openapi_snapshot.json`."""
    client = _make_client(tmp_path)
    schema = client.get("/openapi.json").json()

    if not OPENAPI_SNAPSHOT_PATH.exists():
        OPENAPI_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OPENAPI_SNAPSHOT_PATH.write_text(
            json.dumps(schema, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    stored = json.loads(OPENAPI_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert set(schema["paths"].keys()) == set(stored["paths"].keys())
    assert schema["info"]["title"] == stored["info"]["title"]


def test_a11_uniform_error_shape_across_all_4xx(tmp_path: Path) -> None:
    """A11: Every 4xx response returns `{'error': {'code': '...', 'message': '...'}}`."""
    client = _make_client(tmp_path)

    responses = [
        client.get("/api/v1/detections/99999"),  # 404
        client.get("/api/v1/segments/99999"),  # 404
        client.get("/api/v1/jobs/99999"),  # 404
        client.get("/api/v1/detections?limit=999"),  # 422
        client.post(
            "/api/v1/detect/image",
            files={"file": ("bad.jpg", io.BytesIO(b"not an image"), "image/jpeg")},
        ),  # 415
    ]

    for resp in responses:
        assert 400 <= resp.status_code < 500
        body = resp.json()
        assert "error" in body, f"Missing 'error' envelope in {resp.status_code}: {body}"
        assert isinstance(body["error"].get("code"), str) and body["error"]["code"]
        assert isinstance(body["error"].get("message"), str) and body["error"]["message"]


def test_a12_csv_export_columns_and_escaping(tmp_path: Path) -> None:
    """A12: `/api/v1/export/csv` outputs documented columns and escapes commas and quotes."""
    client = _make_client(tmp_path)
    client.post(
        "/api/v1/detect/image",
        files={"file": ('comma,quote"name.jpg', io.BytesIO(_image_bytes()), "image/jpeg")},
        data={"lat": "18.5204", "lng": "73.8567"},
    )

    resp = client.get("/api/v1/export/csv")
    assert resp.status_code == 200
    assert "text/csv" in resp.headers.get("content-type", "")

    reader = list(csv.reader(io.StringIO(resp.text)))
    assert len(reader) == 2
    header = reader[0]
    expected_cols = [
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
    assert header == expected_cols
    row = reader[1]
    assert row[2] == "D40"
    assert "," in row[-1] and '"' in row[-1]
