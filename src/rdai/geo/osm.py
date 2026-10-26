"""OpenStreetMap Overpass API road class lookup with caching and offline fallback (`FR-P3`, `U07`)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests
import yaml

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_PRIORITY_YAML = ROOT_DIR / "configs" / "priority.yaml"

_OSM_CACHE: dict[tuple[float, float], dict[str, Any]] = {}


def load_osm_priority_config(priority_yaml_path: Path = DEFAULT_PRIORITY_YAML) -> dict[str, Any]:
    """Load road importance scores and Overpass settings from `configs/priority.yaml`."""
    with priority_yaml_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return {
        "road_importance": dict(data.get("road_importance", {})),
        "overpass": dict(data.get("overpass", {})),
    }


def _default_http_post(url: str, data: dict[str, str], timeout: float) -> dict[str, Any]:
    """Execute HTTP POST against Overpass API and return parsed JSON."""
    resp = requests.post(url, data=data, timeout=timeout)
    resp.raise_for_status()
    return dict(resp.json())


def map_highway_tag_to_score(
    highway_tag: str | None,
    importance_map: dict[str, int | float] | None = None,
) -> tuple[str, float]:
    """Map an OpenStreetMap `highway` tag string to `(normalized_class, importance_score)`."""
    scores = (
        importance_map
        if importance_map is not None
        else load_osm_priority_config()["road_importance"]
    )
    default_score = float(scores.get("unknown", 50))
    if not highway_tag:
        return "unknown", default_score

    normalized = highway_tag.strip().lower().split("_")[0]
    if normalized in scores:
        return normalized, float(scores[normalized])
    return "unknown", default_score


def lookup_road_importance(
    lat: float,
    lng: float,
    priority_yaml_path: Path = DEFAULT_PRIORITY_YAML,
    http_post: Callable[[str, dict[str, str], float], dict[str, Any]] | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Query OpenStreetMap `highway` tag around `(lat, lng)` with offline fallback (`R = 50`)."""
    cache_key = (round(float(lat), 4), round(float(lng), 4))
    if use_cache and cache_key in _OSM_CACHE:
        return dict(_OSM_CACHE[cache_key])

    cfg = load_osm_priority_config(priority_yaml_path)
    scores_map = cfg["road_importance"]
    overpass_cfg = cfg["overpass"]
    url = str(overpass_cfg.get("url", "https://overpass-api.de/api/interpreter"))
    timeout = float(overpass_cfg.get("timeout_seconds", 5.0))
    radius_m = int(overpass_cfg.get("search_radius_meters", 30))

    post_fn = http_post if http_post is not None else _default_http_post
    query = f'[out:json][timeout:{int(timeout)}];way(around:{radius_m},{lat},{lng})["highway"];out tags;'

    try:
        payload = post_fn(url, {"data": query}, timeout)
        elements = payload.get("elements", [])
        best_class = "unknown"
        best_score = -1.0

        for elem in elements:
            tags = elem.get("tags", {}) if isinstance(elem, dict) else {}
            hw = tags.get("highway")
            if hw:
                cls_name, score = map_highway_tag_to_score(str(hw), scores_map)
                if score > best_score:
                    best_class, best_score = cls_name, score

        if best_score < 0.0:
            best_class, best_score = map_highway_tag_to_score(None, scores_map)

        result = {"road_class": best_class, "road_importance": best_score}
    except Exception:
        default_cls, default_val = map_highway_tag_to_score(None, scores_map)
        result = {"road_class": default_cls, "road_importance": default_val}

    if use_cache:
        _OSM_CACHE[cache_key] = dict(result)
    return result
