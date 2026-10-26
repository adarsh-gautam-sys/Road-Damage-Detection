"""Fast unit tests for `rdai.train.evaluate` and `scripts/make_results.py` (M3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from rdai.train.evaluate import (
    check_inference_determinism,
    compute_box_iou,
    count_negative_detections,
    extract_val_f1_threshold,
    format_val_result_to_dict,
    run_smoke_inference_on_fixtures,
    update_app_default_conf,
    verify_metric_gate,
    verify_onnx_parity,
)


class _StubBoxes:
    def __init__(self, xyxy: list[list[float]], conf: list[float], cls: list[int]) -> None:
        self.xyxy = np.array(xyxy, dtype=float)
        self.conf = np.array(conf, dtype=float)
        self.cls = np.array(cls, dtype=float)

    def __len__(self) -> int:
        return len(self.conf)


class _StubPredResult:
    def __init__(
        self,
        xyxy: list[list[float]],
        conf: list[float],
        cls: list[int],
        orig_shape: tuple[int, int] = (720, 720),
    ) -> None:
        self.boxes = _StubBoxes(xyxy, conf, cls)
        self.orig_shape = orig_shape


class _StubYoloDetector:
    def __init__(self, boxes_per_call: list[_StubPredResult] | None = None) -> None:
        self._results = boxes_per_call or [
            _StubPredResult([[100.0, 200.0, 300.0, 400.0]], [0.82], [3])
        ]
        self.last_conf: float | None = None

    def predict(
        self, source: Any, conf: float = 0.25, verbose: bool = False
    ) -> list[_StubPredResult]:
        self.last_conf = conf
        if isinstance(source, list):
            return self._results[: len(source)]
        return [self._results[0]]


def test_extract_val_f1_threshold_and_update_config(tmp_path: Path) -> None:
    """FR-M4 & M02: Select optimal confidence threshold from F1 curve and write to app.yaml."""
    # Simulate Ultralytics F1 curve: shape (4 classes, 1000 thresholds from 0..1)
    xs = np.linspace(0.0, 1.0, 101)
    # Peak around 0.34
    f1_per_class = np.vstack([np.exp(-((xs - 0.34) ** 2) / 0.05) for _ in range(4)])

    best_conf, best_f1 = extract_val_f1_threshold(f1_per_class, thresholds=xs)
    assert pytest.approx(best_conf, abs=0.01) == 0.34
    assert pytest.approx(best_f1, abs=0.01) == 1.0

    app_yaml = tmp_path / "app.yaml"
    app_yaml.write_text(
        yaml.safe_dump({"model": {"name": "yolo11s", "default_conf": 0.25}}),
        encoding="utf-8",
    )
    update_app_default_conf(best_conf, app_yaml_path=app_yaml)
    updated = yaml.safe_load(app_yaml.read_text(encoding="utf-8"))
    assert pytest.approx(updated["model"]["default_conf"], abs=1e-4) == round(best_conf, 3)


def test_format_val_result_and_metric_gate() -> None:
    """FR-M2 & M03: Format Ultralytics val metrics and check mAP@50 >= 0.50 and D40 AP@50 >= 0.55."""

    class _FakeBoxMetrics:
        mp = 0.64
        mr = 0.58
        map50 = 0.562
        map = 0.312
        ap_class_index = np.array([0, 1, 2, 3])
        p = np.array([0.62, 0.45, 0.68, 0.81])
        r = np.array([0.55, 0.38, 0.64, 0.75])
        ap50 = np.array([0.54, 0.32, 0.61, 0.778])
        ap = np.array([0.28, 0.16, 0.35, 0.458])

    class _FakeValResult:
        box = _FakeBoxMetrics()
        speed = {"preprocess": 0.8, "inference": 4.2, "postprocess": 1.1}

    formatted = format_val_result_to_dict(_FakeValResult(), experiment_id="E1", split="test")
    assert formatted["overall"]["map50"] == pytest.approx(0.562)
    assert formatted["per_class"]["D40"]["ap50"] == pytest.approx(0.778)

    gate_pass = verify_metric_gate(formatted)
    assert gate_pass["passed"] is True

    # Verify gate fails when mAP@50 < 0.50
    formatted_low = dict(formatted)
    formatted_low["overall"] = dict(formatted["overall"], map50=0.42)
    assert verify_metric_gate(formatted_low)["passed"] is False


def test_smoke_determinism_negatives_and_onnx_parity(project_root: Path) -> None:
    """Verify M02, M04, M05, and M07 helper logic with stubbed models."""
    fixtures_dir = project_root / "tests" / "fixtures" / "generated" / "road_images"
    imgs = sorted(fixtures_dir.glob("*.jpg"))[:5]

    stub = _StubYoloDetector(
        [_StubPredResult([[120.0, 220.0, 280.0, 390.0]], [0.79], [3]) for _ in range(5)]
    )
    smoke = run_smoke_inference_on_fixtures(
        weights_path=Path("dummy.pt"),
        image_paths=imgs,
        app_yaml_path=project_root / "configs" / "app.yaml",
        yolo_loader=lambda _p: stub,
    )
    assert smoke["configured_conf"] == 0.25
    assert len(smoke["images"]) == 5
    assert smoke["images"][0]["detections"][0]["class_code"] == "D40"

    # M04 Determinism
    assert check_inference_determinism(Path("dummy.pt"), imgs[0], yolo_loader=lambda _p: stub)

    # M05 Negatives (0 detections on empty stub)
    empty_stub = _StubYoloDetector([_StubPredResult([], [], []) for _ in range(20)])
    fp_count = count_negative_detections(
        Path("dummy.pt"), imgs, conf=0.25, yolo_loader=lambda _p: empty_stub
    )
    assert fp_count == 0

    # M07 ONNX parity (box shifted by 1 px -> IoU > 0.98 >= 0.95)
    onnx_stub = _StubYoloDetector(
        [_StubPredResult([[121.0, 220.0, 280.0, 390.0]], [0.78], [3]) for _ in range(5)]
    )
    parity = verify_onnx_parity(
        pt_weights_path=Path("dummy.pt"),
        onnx_weights_path=Path("dummy.onnx"),
        image_paths=imgs,
        conf=0.25,
        min_iou=0.95,
        yolo_loader=lambda p: stub if str(p).endswith(".pt") else onnx_stub,
    )
    assert parity["passed"] is True
    assert compute_box_iou([0, 0, 100, 100], [0, 0, 100, 100]) == pytest.approx(1.0)


def test_make_results_builds_tables_and_figures_from_metrics_json(tmp_path: Path) -> None:
    """Section 13: scripts/make_results.py generates tables and figures strictly from metrics.json."""
    from scripts.make_results import build_all_results

    metrics_data = {
        "generated_at": "2026-10-27T03:55:00Z",
        "optimal_conf": 0.28,
        "experiments": {
            "E1": {
                "experiment_id": "E1",
                "split": "test",
                "params_m": 9.4,
                "gpu_ms": 4.2,
                "cpu_ms": 88.5,
                "overall": {
                    "precision": 0.641,
                    "recall": 0.578,
                    "map50": 0.562,
                    "map50_95": 0.312,
                },
                "per_class": {
                    "D00": {"precision": 0.62, "recall": 0.55, "ap50": 0.54, "ap50_95": 0.28},
                    "D10": {"precision": 0.45, "recall": 0.38, "ap50": 0.32, "ap50_95": 0.16},
                    "D20": {"precision": 0.68, "recall": 0.64, "ap50": 0.61, "ap50_95": 0.35},
                    "D40": {"precision": 0.81, "recall": 0.75, "ap50": 0.778, "ap50_95": 0.458},
                },
            },
            "E4": {
                "experiment_id": "E4",
                "split": "test",
                "params_m": 11.2,
                "gpu_ms": 4.9,
                "cpu_ms": 96.0,
                "overall": {
                    "precision": 0.61,
                    "recall": 0.55,
                    "map50": 0.534,
                    "map50_95": 0.295,
                },
                "per_class": {
                    "D00": {"precision": 0.59, "recall": 0.52, "ap50": 0.51, "ap50_95": 0.26},
                    "D10": {"precision": 0.40, "recall": 0.33, "ap50": 0.28, "ap50_95": 0.14},
                    "D20": {"precision": 0.65, "recall": 0.61, "ap50": 0.58, "ap50_95": 0.33},
                    "D40": {"precision": 0.79, "recall": 0.73, "ap50": 0.766, "ap50_95": 0.45},
                },
            },
        },
    }

    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_data, indent=2), encoding="utf-8")

    tables_dir = tmp_path / "tables"
    figures_dir = tmp_path / "figures"

    summary = build_all_results(
        metrics_path=metrics_path,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
    )
    assert summary["status"] == "populated"
    assert (tables_dir / "main_result.csv").is_file()
    assert (tables_dir / "main_result.md").is_file()
    assert (tables_dir / "ablations.csv").is_file()
    assert (tables_dir / "ablations.md").is_file()
    assert (figures_dir / "results_per_class_ap.png").is_file()
    assert (figures_dir / "results_ablations.png").is_file()

    main_csv = (tables_dir / "main_result.csv").read_text(encoding="utf-8")
    assert "0.778" in main_csv
    assert "0.562" in main_csv
