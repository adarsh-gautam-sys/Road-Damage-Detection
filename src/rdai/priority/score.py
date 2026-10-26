"""Road segment maintenance priority scoring, sensitivity analysis, and reclustering (`FR-P1`..`FR-P4`, `U07`, `I04`, `I05`)."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from rdai.geo.cluster import (
    cluster_detections_dbscan,
    haversine_distance_m,
    load_clustering_config,
)
from rdai.geo.osm import lookup_road_importance, map_highway_tag_to_score

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_PRIORITY_YAML = ROOT_DIR / "configs" / "priority.yaml"


def load_priority_config(priority_yaml_path: Path = DEFAULT_PRIORITY_YAML) -> dict[str, Any]:
    """Load priority weights, bands, saturation count, and road importance table from YAML."""
    with priority_yaml_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    weights = data.get("weights", {})
    bands = data.get("bands", {})
    return {
        "weights": {
            "s_max": float(weights.get("s_max", 0.45)),
            "density": float(weights.get("density", 0.25)),
            "road_importance": float(weights.get("road_importance", 0.20)),
            "confidence": float(weights.get("confidence", 0.10)),
        },
        "density_saturation_count": int(data.get("density_saturation_count", 10)),
        "bands": {
            "low_max": float(bands.get("low_max", 30.0)),
            "medium_max": float(bands.get("medium_max", 70.0)),
        },
        "road_importance": {k: float(v) for k, v in data.get("road_importance", {}).items()},
    }


def classify_priority_band(
    score: float,
    priority_yaml_path: Path = DEFAULT_PRIORITY_YAML,
) -> str:
    """Classify a priority score `P` in `[0, 100]` into `'Low'`, `'Medium'`, or `'High'` (`U07`)."""
    cfg = load_priority_config(priority_yaml_path)
    low_max = cfg["bands"]["low_max"]
    medium_max = cfg["bands"]["medium_max"]
    val = float(score)
    if val <= low_max:
        return "Low"
    if val <= medium_max:
        return "Medium"
    return "High"


def compute_priority_score(
    s_max: float,
    n_detections: int,
    road_class: str | None = None,
    road_importance: float | None = None,
    mean_confidence: float = 0.0,
    weights: dict[str, float] | None = None,
    priority_yaml_path: Path = DEFAULT_PRIORITY_YAML,
) -> dict[str, Any]:
    """Compute segment priority score `P = 0.45*S_max + 0.25*D + 0.20*R + 0.10*C` (`U07`)."""
    cfg = load_priority_config(priority_yaml_path)
    w = weights if weights is not None else cfg["weights"]
    sat_count = max(1, cfg["density_saturation_count"])

    s_term = max(0.0, min(100.0, float(s_max)))
    d_term = 100.0 * min(1.0, max(0, int(n_detections)) / float(sat_count))

    if road_importance is not None:
        resolved_class = road_class or "custom"
        r_term = max(0.0, min(100.0, float(road_importance)))
    else:
        resolved_class, r_val = map_highway_tag_to_score(road_class, cfg["road_importance"])
        r_term = max(0.0, min(100.0, float(r_val)))

    c_term = 100.0 * max(0.0, min(1.0, float(mean_confidence)))

    ws = float(w["s_max"]) * s_term
    wd = float(w["density"]) * d_term
    wr = float(w["road_importance"]) * r_term
    wc = float(w["confidence"]) * c_term

    raw_p = max(0.0, min(100.0, ws + wd + wr + wc))
    score_rounded = round(raw_p, 4)
    label = classify_priority_band(score_rounded, priority_yaml_path=priority_yaml_path)

    return {
        "priority_score": score_rounded,
        "priority_label": label,
        "road_class": resolved_class,
        "terms": {
            "s_max": round(s_term, 4),
            "density": round(d_term, 4),
            "road_importance": round(r_term, 4),
            "confidence": round(c_term, 4),
        },
        "weighted_terms": {
            "s_max": round(ws, 4),
            "density": round(wd, 4),
            "road_importance": round(wr, 4),
            "confidence": round(wc, 4),
        },
    }


def compute_priority_with_osm(
    lat: float,
    lng: float,
    s_max: float,
    n_detections: int,
    mean_confidence: float,
    priority_yaml_path: Path = DEFAULT_PRIORITY_YAML,
    http_post: Callable[[str, dict[str, str], float], dict[str, Any]] | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Look up road importance from OpenStreetMap (with offline fallback) and compute segment priority (`U07`)."""
    osm_info = lookup_road_importance(
        lat=lat,
        lng=lng,
        priority_yaml_path=priority_yaml_path,
        http_post=http_post,
        use_cache=use_cache,
    )
    return compute_priority_score(
        s_max=s_max,
        n_detections=n_detections,
        road_class=str(osm_info["road_class"]),
        road_importance=float(osm_info["road_importance"]),
        mean_confidence=mean_confidence,
        priority_yaml_path=priority_yaml_path,
    )


def recluster_and_rank_segments(
    detections: list[dict[str, Any]],
    existing_segments: list[dict[str, Any]] | None = None,
    priority_yaml_path: Path = DEFAULT_PRIORITY_YAML,
    osm_lookup_fn: Callable[[float, float], dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Recluster all located detections via DBSCAN, preserve matching segment IDs, and rank by priority (`I04`, `I05`)."""
    existing = list(existing_segments or [])
    existing_by_id: dict[int, dict[str, Any]] = {
        int(s["id"]): dict(s) for s in existing if "id" in s
    }

    updated_detections: list[dict[str, Any]] = [dict(d) for d in detections]
    located_indices: list[int] = [
        idx
        for idx, d in enumerate(updated_detections)
        if d.get("lat") is not None and d.get("lng") is not None
    ]

    for idx, d in enumerate(updated_detections):
        if idx not in located_indices:
            d["segment_id"] = None

    if not located_indices:
        return {"segments": [], "detections": updated_detections}

    located_payloads = [updated_detections[idx] for idx in located_indices]
    clustered_payloads = cluster_detections_dbscan(
        located_payloads,
        priority_yaml_path=priority_yaml_path,
    )

    clusters: dict[int, list[int]] = defaultdict(list)
    for det_idx, c_item in zip(located_indices, clustered_payloads, strict=True):
        c_lbl = int(c_item["cluster_idx"])
        clusters[c_lbl].append(det_idx)

    # Match clusters to existing segment IDs by member overlap first, then centroid proximity within eps_meters
    clust_cfg = load_clustering_config(priority_yaml_path)
    eps_meters = clust_cfg["eps_meters"]

    overlap_candidates: list[tuple[int, int, int]] = []
    for c_lbl, member_indices in clusters.items():
        old_id_counts: Counter[int] = Counter(
            int(updated_detections[i]["segment_id"])
            for i in member_indices
            if updated_detections[i].get("segment_id") is not None
        )
        for old_id, count in old_id_counts.items():
            overlap_candidates.append((count, -c_lbl, old_id))

    overlap_candidates.sort(reverse=True)
    assigned_cluster_to_seg_id: dict[int, int] = {}
    claimed_seg_ids: set[int] = set()

    for _count, neg_c_lbl, old_id in overlap_candidates:
        c_lbl = -neg_c_lbl
        if c_lbl not in assigned_cluster_to_seg_id and old_id not in claimed_seg_ids:
            assigned_cluster_to_seg_id[c_lbl] = old_id
            claimed_seg_ids.add(old_id)

    # Fallback: match unclaimed existing segments if their centroid is within eps_meters of a cluster
    for c_lbl, member_indices in sorted(clusters.items()):
        if c_lbl in assigned_cluster_to_seg_id:
            continue
        c_lat = float(np.mean([float(updated_detections[i]["lat"]) for i in member_indices]))
        c_lng = float(np.mean([float(updated_detections[i]["lng"]) for i in member_indices]))
        best_old_id: int | None = None
        best_dist = float("inf")
        for old_id, old_seg in existing_by_id.items():
            if old_id in claimed_seg_ids:
                continue
            if old_seg.get("centroid_lat") is not None and old_seg.get("centroid_lng") is not None:
                dist = haversine_distance_m(
                    c_lat,
                    c_lng,
                    float(old_seg["centroid_lat"]),
                    float(old_seg["centroid_lng"]),
                )
                if dist <= eps_meters and dist < best_dist:
                    best_dist = dist
                    best_old_id = old_id
        if best_old_id is not None:
            assigned_cluster_to_seg_id[c_lbl] = best_old_id
            claimed_seg_ids.add(best_old_id)

    next_seg_id = max(existing_by_id.keys() | claimed_seg_ids, default=0) + 1
    lookup_fn = osm_lookup_fn or (
        lambda lat, lng: lookup_road_importance(lat, lng, priority_yaml_path=priority_yaml_path)
    )

    built_segments: list[dict[str, Any]] = []
    for c_lbl in sorted(clusters.keys()):
        member_indices = clusters[c_lbl]
        if c_lbl in assigned_cluster_to_seg_id:
            seg_id = assigned_cluster_to_seg_id[c_lbl]
        else:
            seg_id = next_seg_id
            next_seg_id += 1

        old_seg = existing_by_id.get(seg_id, {})
        status = str(old_seg.get("status", "new"))

        members = [updated_detections[i] for i in member_indices]
        for m in members:
            m["segment_id"] = seg_id

        centroid_lat = float(np.mean([float(m["lat"]) for m in members]))
        centroid_lng = float(np.mean([float(m["lng"]) for m in members]))
        n_detections = len(members)
        max_severity = float(max(float(m.get("severity_score", 0.0)) for m in members))
        mean_confidence = float(np.mean([float(m.get("confidence", 0.0)) for m in members]))

        # Determine road_class from existing segment, member metadata, or OSM lookup
        road_class = old_seg.get("road_class")
        if not road_class or road_class == "unknown":
            member_classes = [
                str(m["road_class"])
                for m in members
                if m.get("road_class") and str(m["road_class"]) != "unknown"
            ]
            if member_classes:
                road_class = member_classes[0]
            else:
                osm_res = lookup_fn(centroid_lat, centroid_lng)
                road_class = str(osm_res.get("road_class", "unknown"))

        priority_res = compute_priority_score(
            s_max=max_severity,
            n_detections=n_detections,
            road_class=road_class,
            mean_confidence=mean_confidence,
            priority_yaml_path=priority_yaml_path,
        )

        built_segments.append(
            {
                "id": seg_id,
                "centroid_lat": round(centroid_lat, 6),
                "centroid_lng": round(centroid_lng, 6),
                "n_detections": n_detections,
                "max_severity": round(max_severity, 2),
                "mean_confidence": round(mean_confidence, 4),
                "road_class": priority_res["road_class"],
                "priority_score": priority_res["priority_score"],
                "priority_label": priority_res["priority_label"],
                "terms": priority_res["terms"],
                "weighted_terms": priority_res["weighted_terms"],
                "status": status,
                "member_detection_ids": [m.get("id") for m in members if m.get("id") is not None],
            }
        )

    built_segments.sort(key=lambda s: (-float(s["priority_score"]), int(s["id"])))
    return {
        "segments": built_segments,
        "detections": updated_detections,
    }


def _spearman_rank_correlation(ranks_a: list[int], ranks_b: list[int]) -> float:
    """Compute Spearman rank correlation coefficient between two rank permutations."""
    n = len(ranks_a)
    if n <= 1:
        return 1.0
    arr_a = np.asarray(ranks_a, dtype=float)
    arr_b = np.asarray(ranks_b, dtype=float)
    std_a = float(np.std(arr_a))
    std_b = float(np.std(arr_b))
    if std_a == 0.0 or std_b == 0.0:
        return 1.0
    corr = float(np.corrcoef(arr_a, arr_b)[0, 1])
    return round(corr, 4)


def evaluate_weight_sensitivity(
    segments: list[dict[str, Any]],
    perturbation: float = 0.20,
    priority_yaml_path: Path = DEFAULT_PRIORITY_YAML,
) -> dict[str, Any]:
    """Vary each priority weight by `+/- perturbation` (default 20%) and report Spearman rank stability."""
    cfg = load_priority_config(priority_yaml_path)
    base_weights = cfg["weights"]

    if not segments:
        return {"perturbations": [], "min_spearman_rho": 1.0, "mean_spearman_rho": 1.0}

    def _rank_ids_under_weights(w_map: dict[str, float]) -> list[int]:
        scored: list[tuple[float, int]] = []
        for idx, seg in enumerate(segments):
            seg_id = int(seg.get("id", idx + 1))
            res = compute_priority_score(
                s_max=float(seg.get("max_severity", 0.0)),
                n_detections=int(seg.get("n_detections", 1)),
                road_class=str(seg.get("road_class", "unknown")),
                mean_confidence=float(seg.get("mean_confidence", 0.0)),
                weights=w_map,
                priority_yaml_path=priority_yaml_path,
            )
            scored.append((-float(res["priority_score"]), seg_id))
        scored.sort()
        order_by_id = {seg_id: rank for rank, (_neg_p, seg_id) in enumerate(scored, start=1)}
        return [order_by_id[int(seg.get("id", i + 1))] for i, seg in enumerate(segments)]

    baseline_ranks = _rank_ids_under_weights(base_weights)
    perturbations: list[dict[str, Any]] = []

    for key in ("s_max", "density", "road_importance", "confidence"):
        for delta_sign, label in ((1.0 + perturbation, "+20%"), (1.0 - perturbation, "-20%")):
            raw_w = dict(base_weights)
            raw_w[key] = raw_w[key] * delta_sign
            total_w = sum(raw_w.values())
            norm_w = {k: v / total_w for k, v in raw_w.items()}
            pert_ranks = _rank_ids_under_weights(norm_w)
            rho = _spearman_rank_correlation(baseline_ranks, pert_ranks)
            perturbations.append(
                {
                    "weight": key,
                    "change": label,
                    "spearman_rho": rho,
                    "weights": {k: round(v, 4) for k, v in norm_w.items()},
                }
            )

    rhos = [float(p["spearman_rho"]) for p in perturbations]
    return {
        "perturbations": perturbations,
        "min_spearman_rho": round(min(rhos), 4),
        "mean_spearman_rho": round(float(np.mean(rhos)), 4),
    }
