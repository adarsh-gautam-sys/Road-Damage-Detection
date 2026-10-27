"""Unit tests U08: Detector wrapper, ByteTrack deduplication, and severity audit script."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rdai.inference.detector import RoadDamageDetector
from rdai.inference.tracker import deduplicate_tracked_frames, track_video_file


class _MockBoxes:
    def __init__(
        self,
        xyxy: list[list[float]],
        conf: list[float],
        cls: list[int],
        track_id: list[int] | None = None,
    ) -> None:
        self.xyxy = np.array(xyxy, dtype=float)
        self.conf = np.array(conf, dtype=float)
        self.cls = np.array(cls, dtype=float)
        self.id = np.array(track_id, dtype=float) if track_id is not None else None

    def __len__(self) -> int:
        return len(self.conf)


class _MockResult:
    def __init__(
        self,
        xyxy: list[list[float]],
        conf: list[float],
        cls: list[int],
        orig_shape: tuple[int, int] = (720, 720),
    ) -> None:
        self.boxes = _MockBoxes(xyxy, conf, cls)
        self.orig_shape = orig_shape


class _MockModel:
    def __init__(self, result: _MockResult) -> None:
        self._result = result

    def predict(
        self, source: Any, conf: float = 0.0, iou: float = 0.45, verbose: bool = False
    ) -> list[_MockResult]:
        return [self._result]


def test_u08_threshold_filtering_class_mapping_and_box_clamping() -> None:
    """U08: Detections under threshold dropped; class ids map to codes; boxes clamped to image."""
    mock_res = _MockResult(
        xyxy=[
            [-25.0, 100.0, 200.0, 300.0],  # Clamped to [0, 100, 200, 300], conf=0.85 (kept)
            [50.0, 50.0, 120.0, 120.0],  # conf=0.12 < 0.25 (dropped)
            [400.0, 400.0, 780.0, 760.0],  # Clamped to [400, 400, 720, 720], conf=0.60 (kept)
        ],
        conf=[0.85, 0.12, 0.60],
        cls=[0, 1, 3],
        orig_shape=(720, 720),
    )
    detector = RoadDamageDetector(
        weights_path=Path("dummy.pt"),
        default_conf=0.25,
        yolo_loader=lambda _p: _MockModel(mock_res),
    )

    dets = detector.detect_image(np.zeros((720, 720, 3), dtype=np.uint8), conf=0.25)
    assert len(dets) == 2

    assert dets[0]["class_id"] == 0
    assert dets[0]["class_code"] == "D00"
    assert dets[0]["class_name"] == "longitudinal_crack"
    assert dets[0]["box"] == [0.0, 100.0, 200.0, 300.0]
    assert "severity" in dets[0]

    assert dets[1]["class_id"] == 3
    assert dets[1]["class_code"] == "D40"
    assert dets[1]["class_name"] == "pothole"
    assert dets[1]["box"] == [400.0, 400.0, 720.0, 720.0]


def test_u08_empty_result_gives_empty_list() -> None:
    """U08: An empty model prediction returns an empty list."""
    empty_res = _MockResult(xyxy=[], conf=[], cls=[], orig_shape=(720, 720))
    detector = RoadDamageDetector(
        weights_path=Path("dummy.pt"),
        default_conf=0.25,
        yolo_loader=lambda _p: _MockModel(empty_res),
    )
    assert detector.detect_image(np.zeros((720, 720, 3), dtype=np.uint8)) == []


def test_bytetrack_frame_deduplication() -> None:
    """Verify ByteTrack deduplicates the same track_id across video frames into one detection."""
    frame_detections = [
        {
            "frame_idx": 0,
            "track_id": 1,
            "class_code": "D40",
            "confidence": 0.65,
            "box": [100.0, 200.0, 200.0, 300.0],
            "severity": {"score": 45.0, "label": "Medium", "extent": 0.02, "measure": "area_ratio"},
        },
        {
            "frame_idx": 1,
            "track_id": 1,
            "class_code": "D40",
            "confidence": 0.88,
            "box": [100.0, 200.0, 250.0, 350.0],
            "severity": {"score": 68.0, "label": "High", "extent": 0.052, "measure": "area_ratio"},
        },
        {
            "frame_idx": 2,
            "track_id": 2,
            "class_code": "D00",
            "confidence": 0.72,
            "box": [50.0, 100.0, 80.0, 400.0],
            "severity": {"score": 52.0, "label": "Medium", "extent": 0.25, "measure": "diag_ratio"},
        },
    ]
    unique = deduplicate_tracked_frames(frame_detections)
    assert len(unique) == 2
    track1 = next(d for d in unique if d["track_id"] == 1)
    assert track1["frame_idx"] == 1
    assert track1["confidence"] == pytest.approx(0.88)


def test_audit_severity_sheet_generation_and_metrics(project_root: Path, tmp_path: Path) -> None:
    """FR-S3: Generate 100-sample audit sheet and evaluate agreement & Cohen's kappa."""
    from scripts.audit_severity import compute_audit_agreement, generate_audit_sheet

    fixtures_dir = project_root / "tests" / "fixtures" / "generated"
    sheet_csv = tmp_path / "severity_audit_sheet.csv"
    crops_dir = tmp_path / "crops"

    rows = generate_audit_sheet(
        img_dir=fixtures_dir / "road_images",
        xml_dir=fixtures_dir / "road_xmls",
        out_csv_path=sheet_csv,
        out_crops_dir=crops_dir,
        target_samples=100,
        seed=0,
    )
    assert len(rows) == 100
    assert sheet_csv.is_file()
    # Model label must be hidden in the blind grading sheet
    assert "human_label_1" in rows[0]
    assert "model_label_hidden" not in rows[0]

    # Simulate human grading on the 50 held-out test crops (indices 50..99)
    model_labels = ["Low", "Medium", "High"] * 17
    human_1 = ["Low", "Medium", "High"] * 17
    human_2 = ["Low", "Medium", "High"] * 16 + ["Low", "Medium", "Medium"]
    audit_stats = compute_audit_agreement(
        model_labels[:50],
        human_1[:50],
        human_2[:50],
    )
    assert audit_stats["agreement"] == pytest.approx(1.0)
    assert audit_stats["cohens_kappa"] >= 0.90


def test_track_video_file_with_mock_model(generated_fixtures_dir: Path) -> None:
    """Verify `track_video_file` processes a real MP4 fixture and deduplicates tracks."""
    video_path = generated_fixtures_dir / "sample_drive.mp4"
    mock_res = _MockResult(
        xyxy=[[100.0, 150.0, 220.0, 260.0]],
        conf=[0.84],
        cls=[3],
        orig_shape=(320, 320),
    )
    mock_res.boxes.id = np.array([1], dtype=float)
    progress_vals: list[float] = []
    out = track_video_file(
        video_path=video_path,
        model=_MockModel(mock_res),
        conf=0.25,
        progress_callback=progress_vals.append,
    )
    assert out["frames_processed"] == 15
    assert out["raw_detections"] == 15
    assert len(out["unique_detections"]) == 1
    assert progress_vals[-1] == 1.0
