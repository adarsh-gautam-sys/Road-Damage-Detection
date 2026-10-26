"""Fast unit tests for `rdai.train.train` (experiment recipes E1-E7, GPU scaling, and M01 check)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rdai.train.train import (
    EXPERIMENTS,
    build_experiment_spec,
    detect_gpu_training_profile,
    run_training_experiment,
    verify_checkpoint_classes,
)


class _DummyYoloModel:
    """Stub Ultralytics YOLO model for fast unit tests."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.names = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}
        self.train_kwargs: dict[str, Any] = {}

    def info(self, detailed: bool = False, verbose: bool = True) -> tuple[int, int, int, float]:
        return (225, 9414348, 9414332, 21.5)

    def train(self, **kwargs: Any) -> dict[str, Any]:
        self.train_kwargs = kwargs
        return {"status": "ok", "kwargs": kwargs}


def test_experiment_specs_e1_to_e7() -> None:
    """Verify PRD Section 8 experiment recipes E1 through E7."""
    assert set(EXPERIMENTS) == {"E1", "E2", "E3", "E4", "E5", "E6", "E7"}

    e1 = build_experiment_spec("E1")
    assert e1["model"] == "yolo11s.pt"
    assert e1["imgsz"] == 640
    assert e1["epochs"] == 150
    assert e1["patience"] == 50
    assert e1["seed"] == 0
    assert e1["deterministic"] is True
    assert e1["flipud"] == 0.0

    e2 = build_experiment_spec("E2")
    assert e2["model"] == "yolo11n.pt"

    e3 = build_experiment_spec("E3")
    assert e3["model"] == "yolo11m.pt"

    e4 = build_experiment_spec("E4")
    assert e4["model"] == "yolov8s.pt"

    e5 = build_experiment_spec("E5")
    assert "multi" in str(e5["data"]) or e5["name"].startswith("e5_")

    e6 = build_experiment_spec("E6")
    assert e6["imgsz"] == 736

    e7 = build_experiment_spec("E7")
    assert e7["mosaic"] == 0.0
    assert e7["scale"] == 0.0
    assert e7["fliplr"] == 0.0


def test_google_ai_pro_gpu_profile_scaling() -> None:
    """Verify A100/L4 GPUs scale to batch=32 and cache='ram', while T4 uses batch=16."""
    a100_profile = detect_gpu_training_profile("NVIDIA A100-SXM4-40GB")
    assert a100_profile["batch"] == 32
    assert a100_profile["cache"] == "ram"
    assert a100_profile["tier"] == "google_ai_pro_high_vram"

    l4_profile = detect_gpu_training_profile("NVIDIA L4")
    assert l4_profile["batch"] == 32
    assert l4_profile["cache"] == "ram"

    t4_profile = detect_gpu_training_profile("Tesla T4")
    assert t4_profile["batch"] == 16
    assert t4_profile["cache"] is False


def test_run_training_experiment_and_resume(tmp_path: Path) -> None:
    """Verify fresh training and resume=True when last.pt checkpoint exists."""
    created_models: list[_DummyYoloModel] = []

    def _loader(path: str) -> _DummyYoloModel:
        m = _DummyYoloModel(path)
        created_models.append(m)
        return m

    # 1. Fresh run
    res = run_training_experiment(
        experiment_id="E1",
        project_dir=tmp_path / "runs",
        gpu_name="NVIDIA A100-SXM4-40GB",
        yolo_loader=_loader,
    )
    assert res["resumed"] is False
    assert created_models[-1].model_path == "yolo11s.pt"
    assert created_models[-1].train_kwargs["batch"] == 32
    assert created_models[-1].train_kwargs["cache"] == "ram"

    # 2. Simulate interrupted run with existing last.pt
    weights_dir = tmp_path / "runs" / "e1_yolo11s_640" / "weights"
    weights_dir.mkdir(parents=True)
    last_pt = weights_dir / "last.pt"
    last_pt.write_bytes(b"checkpoint")

    res_resume = run_training_experiment(
        experiment_id="E1",
        project_dir=tmp_path / "runs",
        gpu_name="NVIDIA A100-SXM4-40GB",
        yolo_loader=_loader,
    )
    assert res_resume["resumed"] is True
    assert created_models[-1].model_path == str(last_pt)
    assert created_models[-1].train_kwargs == {"resume": True}


def test_verify_checkpoint_classes_valid_and_invalid(tmp_path: Path) -> None:
    """Verify M01 class name check passes on exact RDD order and raises on mismatch."""
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"dummy")

    names = verify_checkpoint_classes(ckpt, yolo_loader=_DummyYoloModel)
    assert names == {0: "D00", 1: "D10", 2: "D20", 3: "D40"}

    class _BadModel(_DummyYoloModel):
        def __init__(self, path: str) -> None:
            super().__init__(path)
            self.names = {0: "pothole", 1: "crack"}

    with pytest.raises(ValueError, match="Expected class map"):
        verify_checkpoint_classes(ckpt, yolo_loader=_BadModel)
