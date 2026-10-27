"""Application configuration loader with YAML defaults and environment variable overrides (`U09`)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_APP_YAML = ROOT_DIR / "configs" / "app.yaml"


class ConfigError(ValueError):
    """Raised when a YAML configuration file is missing or malformed (`U09`)."""


@dataclass
class AppConfig:
    """Typed runtime configuration loaded from `configs/app.yaml` and `RDAI_*` env vars."""

    title: str = "Road Damage AI API"
    version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    database_url: str = "sqlite:///./rdai.db"
    media_dir: str = "media"
    model_name: str = "yolo11s"
    model_run: str = "e1_yolo11s_640"
    weights_path: str = "runs/e1_yolo11s_640/weights/best.pt"
    onnx_path: str = "runs/e1_yolo11s_640/weights/best.onnx"
    default_conf: float = 0.25
    iou_threshold: float = 0.45
    max_photo_mb: float = 15.0
    max_video_mb: float = 200.0
    max_megapixels: float = 40.0
    annotated_max_edge_px: int = 1280
    thumb_max_edge_px: int = 320
    allowed_image_formats: list[str] = field(default_factory=lambda: ["JPEG", "PNG", "WEBP"])
    track_tolerance_seconds: float = 5.0
    india_bbox: dict[str, float] = field(
        default_factory=lambda: {
            "lat_min": 6.0,
            "lat_max": 38.0,
            "lng_min": 68.0,
            "lng_max": 98.0,
        }
    )
    default_limit: int = 20
    max_limit: int = 100
    cors_origins: list[str] = field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ]
    )


def load_app_config(
    app_yaml_path: Path = DEFAULT_APP_YAML,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Load `configs/app.yaml` and apply `RDAI_*` environment variable and dictionary overrides (`U09`)."""
    if not app_yaml_path.exists():
        raise ConfigError(f"Configuration file not found: {app_yaml_path}")

    try:
        with app_yaml_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {app_yaml_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"Failed to parse mapping from {app_yaml_path}")

    app_sec = raw.get("app", {}) or {}
    model_sec = raw.get("model", {}) or {}
    uploads_sec = raw.get("uploads", {}) or {}
    geo_sec = raw.get("geo", {}) or {}
    page_sec = raw.get("pagination", {}) or {}
    cors_sec = raw.get("cors", {}) or {}

    cfg = AppConfig(
        title=str(app_sec.get("title", "Road Damage AI API")),
        version=str(app_sec.get("version", "0.1.0")),
        api_prefix=str(app_sec.get("api_prefix", "/api/v1")),
        database_url=str(app_sec.get("database_url", "sqlite:///./rdai.db")),
        media_dir=str(app_sec.get("media_dir", "media")),
        model_name=str(model_sec.get("name", "yolo11s")),
        model_run=str(model_sec.get("run", "e1_yolo11s_640")),
        weights_path=str(model_sec.get("weights_path", "runs/e1_yolo11s_640/weights/best.pt")),
        onnx_path=str(model_sec.get("onnx_path", "runs/e1_yolo11s_640/weights/best.onnx")),
        default_conf=float(model_sec.get("default_conf", 0.25)),
        iou_threshold=float(model_sec.get("iou_threshold", 0.45)),
        max_photo_mb=float(uploads_sec.get("max_photo_mb", 15.0)),
        max_video_mb=float(uploads_sec.get("max_video_mb", 200.0)),
        max_megapixels=float(uploads_sec.get("max_megapixels", 40.0)),
        annotated_max_edge_px=int(uploads_sec.get("annotated_max_edge_px", 1280)),
        thumb_max_edge_px=int(uploads_sec.get("thumb_max_edge_px", 320)),
        allowed_image_formats=list(
            uploads_sec.get("allowed_image_formats", ["JPEG", "PNG", "WEBP"])
        ),
        track_tolerance_seconds=float(geo_sec.get("track_tolerance_seconds", 5.0)),
        india_bbox=dict(
            geo_sec.get(
                "india_bbox",
                {"lat_min": 6.0, "lat_max": 38.0, "lng_min": 68.0, "lng_max": 98.0},
            )
        ),
        default_limit=int(page_sec.get("default_limit", 20)),
        max_limit=int(page_sec.get("max_limit", 100)),
        cors_origins=list(cors_sec.get("allowed_origins", [])),
    )

    # Environment variable overrides
    if "RDAI_DATABASE_URL" in os.environ:
        cfg.database_url = os.environ["RDAI_DATABASE_URL"].strip()
    if "RDAI_MEDIA_DIR" in os.environ:
        cfg.media_dir = os.environ["RDAI_MEDIA_DIR"].strip()
    if "RDAI_WEIGHTS_PATH" in os.environ:
        cfg.weights_path = os.environ["RDAI_WEIGHTS_PATH"].strip()
    if "RDAI_DEFAULT_CONF" in os.environ:
        cfg.default_conf = float(os.environ["RDAI_DEFAULT_CONF"])
    if "RDAI_MAX_PHOTO_MB" in os.environ:
        val = float(os.environ["RDAI_MAX_PHOTO_MB"])
        cfg.max_photo_mb = int(val) if val.is_integer() else val
    if "RDAI_MAX_VIDEO_MB" in os.environ:
        cfg.max_video_mb = float(os.environ["RDAI_MAX_VIDEO_MB"])
    if "RDAI_MAX_MEGAPIXELS" in os.environ:
        cfg.max_megapixels = float(os.environ["RDAI_MAX_MEGAPIXELS"])
    if "RDAI_CORS_ORIGINS" in os.environ:
        raw_origins = os.environ["RDAI_CORS_ORIGINS"]
        cfg.cors_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

    if overrides:
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    return cfg
