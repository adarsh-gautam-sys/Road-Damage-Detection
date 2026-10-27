"""Performance benchmark tests (`P01`, `P02`, `P03`, PRD Section 12)."""

from __future__ import annotations

import io
import json
import time
import tracemalloc
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from tests.integration.test_pipeline_integration import StubRoadDamageDetector

from rdai.api.main import create_app
from rdai.config import load_app_config

ROOT_DIR = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT_DIR / "reports" / "metrics.json"


def _record_perf_metric(key: str, payload: dict[str, float | int]) -> None:
    if METRICS_PATH.is_file():
        doc = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    else:
        doc = {"status": "awaiting_colab_runs", "experiments": {}}
    perf_sec = doc.setdefault("performance", {})
    perf_sec[key] = payload
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _make_perf_client(tmp_path: Path) -> TestClient:
    db_file = tmp_path / "perf.db"
    media_dir = tmp_path / "media"
    cfg = load_app_config(
        overrides={
            "database_url": f"sqlite:///{db_file.as_posix()}",
            "media_dir": str(media_dir),
        }
    )
    app = create_app(
        config=cfg,
        detector=StubRoadDamageDetector(),
        osm_lookup_fn=lambda lat, lng: {"road_class": "primary", "road_importance": 85.0},
    )
    return TestClient(app)


@pytest.mark.perf
def test_p01_20_sequential_photo_requests_latency(tmp_path: Path) -> None:
    """P01: 20 sequential photo requests on laptop CPU achieve p95 <= 2.0 s, recording p50 and p95."""
    client = _make_perf_client(tmp_path)
    latencies_s: list[float] = []

    for i in range(20):
        img = Image.new("RGB", (640, 640), color=((i * 11) % 255, (i * 23) % 255, (i * 37) % 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        raw = buf.getvalue()

        t0 = time.perf_counter()
        resp = client.post(
            "/api/v1/detect/image",
            files={"file": (f"p01_{i:02d}.jpg", io.BytesIO(raw), "image/jpeg")},
            data={"lat": str(18.5200 + i * 0.001), "lng": "73.8567"},
        )
        elapsed = time.perf_counter() - t0
        assert resp.status_code == 200
        latencies_s.append(elapsed)

    p50_s = float(np.percentile(latencies_s, 50))
    p95_s = float(np.percentile(latencies_s, 95))
    assert p95_s <= 2.0, f"P01 failed: p95={p95_s:.3f}s exceeds 2.0s target"

    _record_perf_metric(
        "P01_photo_20_seq",
        {
            "n_requests": 20,
            "p50_ms": round(p50_s * 1000.0, 2),
            "p95_ms": round(p95_s * 1000.0, 2),
        },
    )


@pytest.mark.perf
def test_p02_memory_growth_over_100_requests(tmp_path: Path) -> None:
    """P02: Memory growth over 100 requests stays under 100 MB (model is not reloaded per request)."""
    client = _make_perf_client(tmp_path)

    tracemalloc.start()
    snap_before, _ = tracemalloc.get_traced_memory()

    for i in range(100):
        img = Image.new("RGB", (320, 320), color=(i % 255, (i * 7) % 255, (i * 13) % 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        resp = client.post(
            "/api/v1/detect/image",
            files={"file": (f"p02_{i:03d}.jpg", io.BytesIO(buf.getvalue()), "image/jpeg")},
        )
        assert resp.status_code == 200

    snap_after, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    growth_mb = max(0.0, (snap_after - snap_before) / (1024.0 * 1024.0))
    assert growth_mb < 100.0, f"P02 failed: memory grew by {growth_mb:.2f} MB (>= 100 MB limit)"

    _record_perf_metric(
        "P02_memory_100_req",
        {
            "n_requests": 100,
            "growth_mb": round(growth_mb, 3),
            "peak_traced_mb": round(peak_bytes / (1024.0 * 1024.0), 3),
        },
    )


@pytest.mark.perf
def test_p03_video_job_finishes_with_progress_updates(
    tmp_path: Path,
    generated_fixtures_dir: Path,
) -> None:
    """P03: Video job finishes with progress reaching 1.0 and elapsed time recorded."""
    client = _make_perf_client(tmp_path)

    # Create a 720p synthetic MP4 clip in tmp_path
    clip_path = tmp_path / "perf_720p.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(clip_path), fourcc, 10.0, (1280, 720))
    frame = np.full((720, 1280, 3), 95, dtype=np.uint8)
    cv2.rectangle(frame, (480, 420), (680, 580), (30, 30, 30), -1)
    for _ in range(30):
        writer.write(frame)
    writer.release()

    track_bytes = (generated_fixtures_dir / "track.csv").read_bytes()

    t0 = time.perf_counter()
    resp = client.post(
        "/api/v1/detect/video",
        files={
            "file": ("perf_720p.mp4", io.BytesIO(clip_path.read_bytes()), "video/mp4"),
            "track": ("track.csv", io.BytesIO(track_bytes), "text/csv"),
        },
    )
    elapsed_s = time.perf_counter() - t0
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    job_status = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job_status["status"] == "done"
    assert job_status["progress"] == 1.0

    _record_perf_metric(
        "P03_video_720p_job",
        {
            "elapsed_s": round(elapsed_s, 3),
            "unique_detections": int(job_status["summary"]["unique_detections"]),
        },
    )
