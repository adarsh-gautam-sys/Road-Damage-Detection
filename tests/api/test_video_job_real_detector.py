"""Video jobs through the real `_LazyDefaultDetector` wrapper (not the test stub), with and without GPS."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.unit.test_detector_and_tracker import _MockModel, _MockResult

from rdai.api.main import create_app
from rdai.config import load_app_config


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"not-a-real-checkpoint")
    mock_res = _MockResult(
        xyxy=[[100.0, 150.0, 220.0, 260.0]], conf=[0.84], cls=[3], orig_shape=(320, 320)
    )
    monkeypatch.setattr(
        "rdai.inference.detector._default_yolo_loader", lambda _path: _MockModel(mock_res)
    )
    cfg = load_app_config(
        overrides={
            "database_url": f"sqlite:///{(tmp_path / 'api.db').as_posix()}",
            "media_dir": str(tmp_path / "media"),
            "weights_path": str(weights),
        }
    )
    app = create_app(
        config=cfg,
        osm_lookup_fn=lambda lat, lng: {"road_class": "secondary", "road_importance": 70.0},
    )
    return TestClient(app)


def test_video_job_completes_without_gps_track(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, generated_fixtures_dir: Path
) -> None:
    """No GPX/CSV: the job still finishes, saves unlocated detections and reports none located."""
    client = _client(tmp_path, monkeypatch)
    video = (generated_fixtures_dir / "sample_drive.mp4").read_bytes()
    accepted = client.post(
        "/api/v1/detect/video",
        files={"file": ("drive.mp4", io.BytesIO(video), "video/mp4")},
    )
    assert accepted.status_code == 202
    job = client.get(f"/api/v1/jobs/{accepted.json()['job_id']}").json()

    assert job["status"] == "done", job.get("error")
    assert job["summary"]["unique_detections"] == 1
    assert job["summary"]["located_detections"] == 0
    stats = client.get("/api/v1/stats").json()
    assert stats["total_detections"] == 1
    assert client.get("/api/v1/segments").json()["total"] == 0
