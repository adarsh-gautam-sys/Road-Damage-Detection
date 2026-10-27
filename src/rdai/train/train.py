"""YOLO11 training pipeline and experiment runner (`FR-M1`, `FR-M3`, verified by `M01`)."""

from __future__ import annotations

import argparse
import logging
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_TRAIN_YAML = ROOT_DIR / "configs" / "train.yaml"
DEFAULT_APP_YAML = ROOT_DIR / "configs" / "app.yaml"
EXPECTED_CLASS_MAP: dict[int, str] = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}

# Experiment deltas relative to E1 (PRD Section 8 table)
EXPERIMENTS: dict[str, dict[str, Any]] = {
    "E1": {
        "description": "Primary recipe: YOLO11s, 640 px, India train split",
        "overrides": {},
    },
    "E2": {
        "description": "Model size ablation: YOLO11n at 640 px",
        "overrides": {"model": "yolo11n.pt", "name": "e2_yolo11n_640"},
    },
    "E3": {
        "description": "Model size ablation: YOLO11m at 640 px",
        "overrides": {"model": "yolo11m.pt", "name": "e3_yolo11m_640"},
    },
    "E4": {
        "description": "Architecture baseline: YOLOv8s at 640 px",
        "overrides": {"model": "yolov8s.pt", "name": "e4_yolov8s_640"},
    },
    "E5": {
        "description": "Multi-country smartphone transfer: India + Japan + Czech",
        "overrides": {
            "data": "configs/data_multicountry.yaml",
            "name": "e5_yolo11s_multicountry_640",
        },
    },
    "E6": {
        "description": "Resolution ablation: YOLO11s at 736 px",
        "overrides": {"imgsz": 736, "name": "e6_yolo11s_736"},
    },
    "E7": {
        "description": "Augmentation ablation: mosaic, scale, and horizontal flip off",
        "overrides": {
            "mosaic": 0.0,
            "scale": 0.0,
            "fliplr": 0.0,
            "name": "e7_yolo11s_no_aug_640",
        },
    },
}


def load_train_config(train_yaml_path: Path = DEFAULT_TRAIN_YAML) -> dict[str, Any]:
    """Load training defaults from `configs/train.yaml`."""
    with train_yaml_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid training config in {train_yaml_path}")
    return data


def load_app_model_config(app_yaml_path: Path = DEFAULT_APP_YAML) -> dict[str, Any]:
    """Load model checkpoint config section from `configs/app.yaml`."""
    with app_yaml_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return dict(data.get("model", {}))


def _default_yolo_loader(model_path: str) -> Any:
    """Import Ultralytics lazily so fast unit tests run without loading PyTorch."""
    from ultralytics import YOLO

    return YOLO(model_path)


def _probe_nvidia_gpu_name() -> str:
    """Query GPU name via `nvidia-smi` if available."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            timeout=5.0,
        )
        return out.strip().splitlines()[0] if out.strip() else "CPU"
    except Exception:
        return "CPU"


def detect_gpu_training_profile(
    gpu_name: str | None = None,
    train_yaml_path: Path = DEFAULT_TRAIN_YAML,
) -> dict[str, Any]:
    """Select batch size and RAM caching profile based on detected Colab GPU tier."""
    cfg = load_train_config(train_yaml_path)
    detected = gpu_name if gpu_name is not None else _probe_nvidia_gpu_name()
    upper = detected.upper()

    if any(tag in upper for tag in ("A100", "L4", "H100", "V100", "A10")):
        return {
            "gpu_name": detected,
            "tier": "google_ai_pro_high_vram",
            "batch": int(cfg.get("colab_pro_batch", 32)),
            "cache": cfg.get("cache", "ram"),
            "workers": int(cfg.get("workers", 8)),
        }
    return {
        "gpu_name": detected,
        "tier": "standard_t4_or_cpu",
        "batch": int(cfg.get("batch", 16)),
        "cache": False,
        "workers": 2,
    }


def build_experiment_spec(
    experiment_id: str = "E1",
    train_yaml_path: Path = DEFAULT_TRAIN_YAML,
    seed_override: int | None = None,
) -> dict[str, Any]:
    """Build complete training specification for experiment `E1`..`E7`."""
    exp_key = experiment_id.upper()
    if exp_key not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{experiment_id}'. Valid: {sorted(EXPERIMENTS)}")

    cfg = load_train_config(train_yaml_path)
    spec: dict[str, Any] = {
        "model": str(cfg["model"]),
        "data": str(cfg["data"]),
        "epochs": int(cfg["epochs"]),
        "patience": int(cfg["patience"]),
        "imgsz": int(cfg["imgsz"]),
        "batch": int(cfg["batch"]),
        "seed": int(cfg["seed"]),
        "deterministic": bool(cfg["deterministic"]),
        "flipud": float(cfg["flipud"]),
        "fliplr": float(cfg.get("fliplr", 0.5)),
        "mosaic": float(cfg.get("mosaic", 1.0)),
        "project": str(cfg.get("project", "runs")),
        "name": str(cfg.get("name", "e1_yolo11s_640")),
    }
    spec.update(EXPERIMENTS[exp_key]["overrides"])
    if seed_override is not None:
        spec["seed"] = int(seed_override)
        spec["name"] = f"{spec['name']}_seed{seed_override}"
    return spec


def run_training_experiment(
    experiment_id: str = "E1",
    project_dir: Path | None = None,
    gpu_name: str | None = None,
    seed_override: int | None = None,
    yolo_loader: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Run or resume a YOLO11 training experiment (`E1`..`E7`)."""
    loader = yolo_loader if yolo_loader is not None else _default_yolo_loader
    spec = build_experiment_spec(experiment_id, seed_override=seed_override)
    profile = detect_gpu_training_profile(gpu_name=gpu_name)

    # Ensure data YAML has an absolute `path` when running on Colab so Ultralytics DATASETS_DIR is bypassed
    data_yaml_path = Path(str(spec.get("data", "configs/data.yaml")))
    if not data_yaml_path.is_absolute():
        data_yaml_path = ROOT_DIR / data_yaml_path
    if data_yaml_path.is_file() and yolo_loader is None:
        with data_yaml_path.open("r", encoding="utf-8") as fh:
            d_cfg = yaml.safe_load(fh)
        rel_p = Path(str(d_cfg.get("path", "data/rdd2022_india_yolo")))
        if not rel_p.is_absolute():
            d_cfg["path"] = (ROOT_DIR / rel_p).resolve().as_posix()
            with data_yaml_path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(d_cfg, fh, sort_keys=False)

    target_project = Path(project_dir) if project_dir is not None else Path(spec["project"])
    run_name = str(spec["name"])
    last_ckpt = target_project / run_name / "weights" / "last.pt"

    start_t = time.perf_counter()
    if last_ckpt.is_file():
        logger.info("Found existing checkpoint at %s — resuming training", last_ckpt)
        model = loader(str(last_ckpt))
        train_result = model.train(resume=True)
        elapsed_s = time.perf_counter() - start_t
        return {
            "experiment_id": experiment_id,
            "resumed": True,
            "checkpoint": str(last_ckpt),
            "elapsed_seconds": round(elapsed_s, 2),
            "result": train_result,
        }

    model_weights = str(spec.pop("model"))
    spec["project"] = str(target_project)
    spec["batch"] = profile["batch"]
    spec["cache"] = profile["cache"]
    spec["workers"] = profile["workers"]

    model = loader(model_weights)
    model_summary = model.info(verbose=True)
    train_result = model.train(**spec)
    elapsed_s = time.perf_counter() - start_t

    return {
        "experiment_id": experiment_id,
        "resumed": False,
        "gpu_profile": profile,
        "model_summary": model_summary,
        "elapsed_seconds": round(elapsed_s, 2),
        "epoch_seconds_est": round(elapsed_s / max(1, int(spec["epochs"])), 3),
        "result": train_result,
    }


def verify_checkpoint_classes(
    weights_path: Path,
    yolo_loader: Callable[[str], Any] | None = None,
) -> dict[int, str]:
    """Load `weights_path` and verify class names match `{0: D00, 1: D10, 2: D20, 3: D40}` (`M01`)."""
    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {weights_path}")

    loader = yolo_loader if yolo_loader is not None else _default_yolo_loader
    model = loader(str(weights_path))
    raw_names = getattr(model, "names", {})
    names = {int(k): str(v) for k, v in raw_names.items()}
    if names != EXPECTED_CLASS_MAP:
        raise ValueError(f"Expected class map {EXPECTED_CLASS_MAP}, got {names}")
    return names


def main() -> None:
    """CLI entrypoint for `make train` or Colab experiment execution."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Train Road Damage AI YOLO11 experiments.")
    parser.add_argument("--exp", default="E1", choices=sorted(EXPERIMENTS), help="Experiment ID")
    parser.add_argument("--project", default=None, help="Output runs directory (e.g. Google Drive)")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override")
    args = parser.parse_args()

    project_path = Path(args.project) if args.project else None
    summary = run_training_experiment(
        experiment_id=args.exp,
        project_dir=project_path,
        seed_override=args.seed,
    )
    print(f"Training summary: {summary}")


if __name__ == "__main__":
    main()
