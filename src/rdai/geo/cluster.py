"""Haversine distance and scikit-learn DBSCAN road segment clustering (`FR-G3`, `U06`)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.cluster import DBSCAN

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_PRIORITY_YAML = ROOT_DIR / "configs" / "priority.yaml"
EARTH_RADIUS_M = 6371000.0


def load_clustering_config(priority_yaml_path: Path = DEFAULT_PRIORITY_YAML) -> dict[str, float]:
    """Load DBSCAN clustering parameters from `configs/priority.yaml`."""
    with priority_yaml_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    clust = data.get("clustering", {})
    return {
        "eps_meters": float(clust.get("eps_meters", 25.0)),
        "min_samples": int(clust.get("min_samples", 1)),
        "earth_radius_meters": float(clust.get("earth_radius_meters", EARTH_RADIUS_M)),
    }


def haversine_distance_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Compute great-circle Haversine distance in meters between two `(lat, lng)` coordinates."""
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    d_phi = math.radians(float(lat2) - float(lat1))
    d_lam = math.radians(float(lng2) - float(lng1))

    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return EARTH_RADIUS_M * c


def cluster_detections_dbscan(
    located_items: list[dict[str, Any]],
    eps_meters: float | None = None,
    min_samples: int | None = None,
    priority_yaml_path: Path = DEFAULT_PRIORITY_YAML,
) -> list[dict[str, Any]]:
    """Cluster located items (`{'lat': ..., 'lng': ...}`) using Haversine DBSCAN (`U06`)."""
    if not located_items:
        return []

    cfg = load_clustering_config(priority_yaml_path)
    eps_m = float(eps_meters) if eps_meters is not None else cfg["eps_meters"]
    min_s = int(min_samples) if min_samples is not None else int(cfg["min_samples"])
    radius_m = cfg["earth_radius_meters"]

    coords_deg = np.array(
        [[float(it["lat"]), float(it["lng"])] for it in located_items], dtype=float
    )
    coords_rad = np.radians(coords_deg)

    eps_rad = eps_m / radius_m
    clustering = DBSCAN(eps=eps_rad, min_samples=min_s, metric="haversine").fit(coords_rad)
    labels = clustering.labels_.tolist()

    results: list[dict[str, Any]] = []
    for item, lbl in zip(located_items, labels, strict=True):
        enriched = dict(item)
        enriched["cluster_idx"] = int(lbl)
        results.append(enriched)
    return results
