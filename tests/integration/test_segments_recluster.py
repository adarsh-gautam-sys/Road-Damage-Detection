"""Integration tests for road segment reclustering (`I04`) and priority re-ranking (`I05`)."""

from __future__ import annotations

import math
from typing import Any

import pytest

from rdai.priority.score import recluster_and_rank_segments


def _offset_lat(lat: float, meters_north: float) -> float:
    """Offset latitude by `meters_north` meters (~111,195 m per degree)."""
    return lat + (meters_north / 111195.0)


def test_i04_recluster_reuses_segment_id_within_10m_and_splits_at_500m() -> None:
    """I04: A detection 10 m from a segment keeps its id and raises its count; 500 m away creates a new segment."""
    base_lat, base_lng = 18.5204, 73.8567

    # Initial upload: 1 located detection -> creates segment 1
    det1: dict[str, Any] = {
        "id": 1,
        "lat": base_lat,
        "lng": base_lng,
        "severity_score": 55.0,
        "confidence": 0.82,
        "road_class": "secondary",
    }
    res1 = recluster_and_rank_segments(
        detections=[det1],
        existing_segments=[],
        osm_lookup_fn=lambda lat, lng: {"road_class": "secondary", "road_importance": 70.0},
    )
    segments_1 = res1["segments"]
    updated_dets_1 = res1["detections"]

    assert len(segments_1) == 1
    seg1_id = segments_1[0]["id"]
    assert seg1_id == 1
    assert segments_1[0]["n_detections"] == 1
    assert updated_dets_1[0]["segment_id"] == seg1_id

    # Preserve segment status to verify status is also kept on recluster
    segments_1[0]["status"] = "verified"

    # Second upload: detection 10 m north of det1 -> should merge into seg1_id and increment count to 2
    det2: dict[str, Any] = {
        "id": 2,
        "lat": _offset_lat(base_lat, 10.0),
        "lng": base_lng,
        "severity_score": 72.0,
        "confidence": 0.90,
        "road_class": "secondary",
    }
    res2 = recluster_and_rank_segments(
        detections=[updated_dets_1[0], det2],
        existing_segments=segments_1,
        osm_lookup_fn=lambda lat, lng: {"road_class": "secondary", "road_importance": 70.0},
    )
    segments_2 = res2["segments"]
    updated_dets_2 = res2["detections"]

    assert len(segments_2) == 1
    assert segments_2[0]["id"] == seg1_id
    assert segments_2[0]["n_detections"] == 2
    assert segments_2[0]["status"] == "verified"
    assert segments_2[0]["max_severity"] == pytest.approx(72.0)
    assert all(d["segment_id"] == seg1_id for d in updated_dets_2)

    # Third upload: detection 500 m north -> creates a new segment with a distinct ID
    det3: dict[str, Any] = {
        "id": 3,
        "lat": _offset_lat(base_lat, 500.0),
        "lng": base_lng,
        "severity_score": 40.0,
        "confidence": 0.75,
        "road_class": "residential",
    }
    res3 = recluster_and_rank_segments(
        detections=[*updated_dets_2, det3],
        existing_segments=segments_2,
        osm_lookup_fn=lambda lat, lng: {"road_class": "residential", "road_importance": 35.0},
    )
    segments_3 = res3["segments"]
    updated_dets_3 = res3["detections"]

    assert len(segments_3) == 2
    seg_by_id = {s["id"]: s for s in segments_3}
    assert seg1_id in seg_by_id
    assert seg_by_id[seg1_id]["n_detections"] == 2

    det3_updated = next(d for d in updated_dets_3 if d["id"] == 3)
    new_seg_id = det3_updated["segment_id"]
    assert new_seg_id is not None
    assert new_seg_id != seg1_id
    assert seg_by_id[new_seg_id]["n_detections"] == 1
    assert math.isclose(seg_by_id[new_seg_id]["max_severity"], 40.0)


def test_i05_rerank_high_detection_moves_segment_up() -> None:
    """I05: Adding a new High detection to a lower-priority segment moves that segment up in the ranking."""
    lat_a, lng_a = 18.5204, 73.8567  # Place A (initially Low severity)
    lat_b, lng_b = 18.5300, 73.8567  # Place B (~1 km away, Medium severity)

    det_a1: dict[str, Any] = {
        "id": 10,
        "lat": lat_a,
        "lng": lng_a,
        "severity_score": 20.0,
        "confidence": 0.70,
        "road_class": "secondary",
    }
    det_b1: dict[str, Any] = {
        "id": 20,
        "lat": lat_b,
        "lng": lng_b,
        "severity_score": 60.0,
        "confidence": 0.80,
        "road_class": "secondary",
    }

    initial = recluster_and_rank_segments(
        detections=[det_a1, det_b1],
        existing_segments=[],
        osm_lookup_fn=lambda lat, lng: {"road_class": "secondary", "road_importance": 70.0},
    )
    segs_initial = initial["segments"]
    dets_initial = initial["detections"]

    seg_a_id = next(d["segment_id"] for d in dets_initial if d["id"] == 10)
    seg_b_id = next(d["segment_id"] for d in dets_initial if d["id"] == 20)
    assert seg_a_id != seg_b_id

    # Initially Segment B ranks #1 and Segment A ranks #2
    assert [s["id"] for s in segs_initial] == [seg_b_id, seg_a_id]

    # Now add a High severity detection (score=95.0, confidence=0.95) 5 meters from Place A
    det_a2: dict[str, Any] = {
        "id": 11,
        "lat": _offset_lat(lat_a, 5.0),
        "lng": lng_a,
        "severity_score": 95.0,
        "confidence": 0.95,
        "road_class": "secondary",
    }
    after = recluster_and_rank_segments(
        detections=[*dets_initial, det_a2],
        existing_segments=segs_initial,
        osm_lookup_fn=lambda lat, lng: {"road_class": "secondary", "road_importance": 70.0},
    )
    segs_after = after["segments"]

    # Segment A now ranks #1 above Segment B
    assert [s["id"] for s in segs_after] == [seg_a_id, seg_b_id]
    assert segs_after[0]["priority_score"] > segs_after[1]["priority_score"]
    assert segs_after[0]["n_detections"] == 2
