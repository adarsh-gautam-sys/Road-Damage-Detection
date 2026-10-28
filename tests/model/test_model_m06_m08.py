"""Model robustness (`M06`) and inference speed (`M08`) tests (PRD Section 12)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from PIL import Image

from rdai.train.evaluate import apply_robustness_perturbation

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT_DIR / "runs" / "e1_yolo11s_640" / "weights" / "best.pt"
METRICS_PATH = ROOT_DIR / "reports" / "metrics.json"


def test_m06_robustness_perturbation_transforms_unit() -> None:
    """Fast unit check that all 5 M06 perturbations produce valid RGB images."""
    base = Image.new("RGB", (640, 640), color=(120, 130, 140))
    kinds = ("brightness_plus_30", "brightness_minus_30", "gaussian_blur", "jpeg_q30", "half_size")
    for kind in kinds:
        out = apply_robustness_perturbation(base, kind)
        assert isinstance(out, Image.Image)
        assert out.mode == "RGB"
        if kind == "half_size":
            assert out.size == (320, 320)
        else:
            assert out.size == (640, 640)


@pytest.mark.slow
def test_m06_robustness_under_perturbations(generated_fixtures_dir: Path) -> None:
    """M06: Record detection count / confidence stability under +/-30% brightness, blur, JPEG q30, and half-size."""
    if not DEFAULT_WEIGHTS.is_file():
        pytest.skip(f"Trained E1 weights not found at {DEFAULT_WEIGHTS}")

    from rdai.inference.detector import RoadDamageDetector

    detector = RoadDamageDetector(weights_path=DEFAULT_WEIGHTS)
    road_imgs = sorted((generated_fixtures_dir / "road_images").glob("*.jpg"))
    kinds = (
        "baseline",
        "brightness_plus_30",
        "brightness_minus_30",
        "gaussian_blur",
        "jpeg_q30",
        "half_size",
    )

    summary: dict[str, dict[str, float | int]] = {}
    for kind in kinds:
        total_dets = 0
        confs: list[float] = []
        for p in road_imgs:
            with Image.open(p) as im:
                rgb = im.convert("RGB")
                perturbed = rgb if kind == "baseline" else apply_robustness_perturbation(rgb, kind)
                dets = detector.detect(perturbed)
                total_dets += len(dets)
                confs.extend(float(d["confidence"]) for d in dets)
        summary[kind] = {
            "detections": total_dets,
            "mean_conf": round(sum(confs) / len(confs), 4) if confs else 0.0,
        }

    if METRICS_PATH.is_file():
        doc = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        doc["robustness_m06"] = summary
        METRICS_PATH.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


@pytest.mark.slow
def test_m08_inference_speed_benchmark(generated_fixtures_dir: Path) -> None:
    """M08: Median model inference time <= 100 ms on GPU and <= 1000 ms on laptop CPU."""
    if not DEFAULT_WEIGHTS.is_file():
        pytest.skip(f"Trained E1 weights not found at {DEFAULT_WEIGHTS}")

    import numpy as np
    import torch

    from rdai.inference.detector import RoadDamageDetector

    detector = RoadDamageDetector(weights_path=DEFAULT_WEIGHTS)
    sample_img = next((generated_fixtures_dir / "road_images").glob("*.jpg"))
    with Image.open(sample_img) as im:
        rgb = im.convert("RGB")

    # Warmup
    detector.detect(rgb)

    times_ms: list[float] = []
    for _ in range(15):
        t0 = time.perf_counter()
        detector.detect(rgb)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    median_ms = float(np.median(times_ms))
    if not torch.cuda.is_available() and METRICS_PATH.is_file():
        doc = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        if "E1" in doc.get("experiments", {}):
            doc["experiments"]["E1"]["cpu_ms"] = round(median_ms, 2)
            METRICS_PATH.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

    if torch.cuda.is_available():
        assert median_ms <= 100.0, f"GPU median latency {median_ms:.2f} ms > 100 ms"
    else:
        assert median_ms <= 1000.0, f"CPU median latency {median_ms:.2f} ms > 1000 ms"
