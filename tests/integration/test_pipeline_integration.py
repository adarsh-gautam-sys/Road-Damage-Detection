"""Integration tests for photo-to-priority (`I01`), video dedupe (`I02`), idempotent upload (`I03`), and failure path (`I06`)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import cv2
from fastapi.testclient import TestClient
from PIL import Image

from rdai.api.main import create_app
from rdai.config import load_app_config


class StubRoadDamageDetector:
    """Deterministic stub detector for fast integration and API tests without GPU weights."""

    model_name = "yolo11s"
    run_name = "e1_yolo11s_640"
    default_conf = 0.25
    is_loaded = True

    def detect_image(self, image: Any, conf_threshold: float | None = None) -> list[dict[str, Any]]:
        thresh = conf_threshold if conf_threshold is not None else self.default_conf
        if thresh > 0.95:
            return []
        return [
            {
                "class_id": 3,
                "class_code": "D40",
                "class_name": "pothole",
                "confidence": 0.85,
                "box": [100.0, 200.0, 280.0, 360.0],
                "severity": {
                    "score": 68.5,
                    "label": "High",
                    "extent": 0.0556,
                    "measure": "area_ratio",
                },
            }
        ]

    def track_video(
        self,
        video_path: Path,
        conf_threshold: float | None = None,
        progress_callback: Any | None = None,
    ) -> list[dict[str, Any]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Corrupt or unreadable video file: {video_path.name}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, _frame = cap.read()
        cap.release()
        if not ok or frame_count <= 0:
            raise ValueError(f"Video file contains no decodable frames: {video_path.name}")

        if progress_callback is not None:
            progress_callback(0.5)
            progress_callback(1.0)

        return [
            {
                "track_id": 1,
                "frame_idx": 0,
                "timestamp_s": 0.5,
                "frames_seen": frame_count,
                "class_id": 3,
                "class_code": "D40",
                "class_name": "pothole",
                "confidence": 0.88,
                "box": [120.0, 220.0, 300.0, 380.0],
                "severity": {
                    "score": 71.2,
                    "label": "High",
                    "extent": 0.0625,
                    "measure": "area_ratio",
                },
            }
        ]


def _build_test_client(tmp_path: Path) -> TestClient:
    db_file = tmp_path / "integration.db"
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


def test_i01_photo_to_priority_located_and_unlocated(
    tmp_path: Path,
    generated_fixtures_dir: Path,
) -> None:
    """I01: Photo with EXIF GPS creates image, detection, and segment rows; photo without GPS stores unlocated detection and no segment."""
    client = _build_test_client(tmp_path)

    # 1. Upload photo WITH EXIF GPS
    gps_photo_path = generated_fixtures_dir / "photo_with_gps.jpg"
    with gps_photo_path.open("rb") as fh:
        resp_located = client.post(
            "/api/v1/detect/image",
            files={"file": ("photo_with_gps.jpg", fh, "image/jpeg")},
        )
    assert resp_located.status_code == 200
    body_loc = resp_located.json()
    assert body_loc["geo"]["source"] == "exif"
    assert body_loc["geo"]["lat"] is not None
    assert len(body_loc["detections"]) == 1
    seg_id = body_loc["detections"][0]["segment_id"]
    assert seg_id is not None

    # Check segments list has 1 segment with a priority score
    segs_resp = client.get("/api/v1/segments")
    assert segs_resp.status_code == 200
    segs = segs_resp.json()["items"]
    assert len(segs) == 1
    assert segs[0]["id"] == seg_id
    assert segs[0]["priority_score"] > 0.0

    # 2. Upload photo WITHOUT GPS
    plain_img = Image.new("RGB", (640, 640), color=(110, 110, 110))
    buf = io.BytesIO()
    plain_img.save(buf, format="JPEG")
    buf.seek(0)

    resp_unloc = client.post(
        "/api/v1/detect/image",
        files={"file": ("no_gps.jpg", buf, "image/jpeg")},
    )
    assert resp_unloc.status_code == 200
    body_unloc = resp_unloc.json()
    assert body_unloc["geo"]["source"] == "none"
    assert body_unloc["geo"]["lat"] is None
    assert body_unloc["detections"][0]["segment_id"] is None

    # Segment count remains 1 (unlocated photo did not create a segment)
    segs_after = client.get("/api/v1/segments").json()["items"]
    assert len(segs_after) == 1


def test_i02_video_dedupe_and_track_interpolation(
    tmp_path: Path,
    generated_fixtures_dir: Path,
) -> None:
    """I02: Video clip with one static pothole across frames gives 1 deduplicated track, updates segments, and reaches progress 1.0."""
    client = _build_test_client(tmp_path)

    video_path = generated_fixtures_dir / "sample_drive.mp4"
    track_path = generated_fixtures_dir / "track.csv"

    with video_path.open("rb") as vf, track_path.open("rb") as tf:
        resp = client.post(
            "/api/v1/detect/video",
            files={
                "file": ("sample_drive.mp4", vf, "video/mp4"),
                "track": ("track.csv", tf, "text/csv"),
            },
            data={"offset_s": "0.0"},
        )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    job_resp = client.get(f"/api/v1/jobs/{job_id}")
    assert job_resp.status_code == 200
    job_data = job_resp.json()
    assert job_data["status"] == "done"
    assert job_data["progress"] == 1.0
    assert job_data["summary"]["unique_detections"] == 1

    segs = client.get("/api/v1/segments").json()["items"]
    assert len(segs) == 1
    assert segs[0]["n_detections"] == 1


def test_i03_idempotent_upload_returns_same_image_id(
    tmp_path: Path,
    generated_fixtures_dir: Path,
) -> None:
    """I03: Uploading the exact same image twice returns the same `image_id` and adds no duplicate rows."""
    client = _build_test_client(tmp_path)
    gps_photo_path = generated_fixtures_dir / "photo_with_gps.jpg"
    raw_bytes = gps_photo_path.read_bytes()

    resp1 = client.post(
        "/api/v1/detect/image",
        files={"file": ("photo_with_gps.jpg", io.BytesIO(raw_bytes), "image/jpeg")},
    )
    resp2 = client.post(
        "/api/v1/detect/image",
        files={"file": ("renamed_copy.jpg", io.BytesIO(raw_bytes), "image/jpeg")},
    )
    assert resp1.status_code == 200
    assert resp2.status_code == 200
    assert resp1.json()["image_id"] == resp2.json()["image_id"]

    stats = client.get("/api/v1/stats").json()
    assert stats["total_images"] == 1
    assert stats["total_detections"] == 1
    assert stats["total_segments"] == 1


def test_i06_corrupt_video_marks_job_failed_and_api_stays_healthy(tmp_path: Path) -> None:
    """I06: A corrupt video marks the background job as `failed` with an error message while the API stays healthy."""
    client = _build_test_client(tmp_path)

    # Craft a file with an MP4 ftyp header followed by corrupt unreadable bytes
    corrupt_mp4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\xff\x00\xab" * 100
    resp = client.post(
        "/api/v1/detect/video",
        files={"file": ("broken_clip.mp4", io.BytesIO(corrupt_mp4), "video/mp4")},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    job_resp = client.get(f"/api/v1/jobs/{job_id}")
    assert job_resp.status_code == 200
    job_body = job_resp.json()
    assert job_body["status"] == "failed"
    assert job_body["error"]

    # Verify API remains healthy
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
