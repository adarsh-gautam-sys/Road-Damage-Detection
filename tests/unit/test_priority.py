"""Unit tests for road segment priority scoring (`U07`) and Hypothesis properties."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rdai.priority.score import (
    classify_priority_band,
    compute_priority_score,
    compute_priority_with_osm,
    evaluate_weight_sensitivity,
    load_priority_config,
)


def test_u07_weights_sum_to_one() -> None:
    """U07: Priority weights in `configs/priority.yaml` must sum to 1.0."""
    cfg = load_priority_config()
    weights = cfg["weights"]
    total = (
        weights["s_max"] + weights["density"] + weights["road_importance"] + weights["confidence"]
    )
    assert total == pytest.approx(1.0, abs=1e-6)


def test_u07_worked_example_gives_60_5() -> None:
    """U07: Worked example (S_max=80, n=1, secondary road R=70, mean_conf=0.8) gives 60.5 (Medium)."""
    result = compute_priority_score(
        s_max=80.0,
        n_detections=1,
        road_class="secondary",
        mean_confidence=0.8,
    )
    assert result["priority_score"] == pytest.approx(60.5, abs=1e-4)
    assert result["priority_label"] == "Medium"
    assert result["terms"]["s_max"] == pytest.approx(80.0)
    assert result["terms"]["density"] == pytest.approx(10.0)
    assert result["terms"]["road_importance"] == pytest.approx(70.0)
    assert result["terms"]["confidence"] == pytest.approx(80.0)
    assert result["weighted_terms"]["s_max"] == pytest.approx(36.0)
    assert result["weighted_terms"]["density"] == pytest.approx(2.5)
    assert result["weighted_terms"]["road_importance"] == pytest.approx(14.0)
    assert result["weighted_terms"]["confidence"] == pytest.approx(8.0)


def test_u07_band_edges_at_30_31_70_71() -> None:
    """U07: Priority band boundaries behave accurately at 30, 31, 70, and 71."""
    assert classify_priority_band(0.0) == "Low"
    assert classify_priority_band(30.0) == "Low"
    assert classify_priority_band(31.0) == "Medium"
    assert classify_priority_band(70.0) == "Medium"
    assert classify_priority_band(71.0) == "High"
    assert classify_priority_band(100.0) == "High"


def test_u07_osm_mocked_offline_defaults_to_50() -> None:
    """U07: When OSM Overpass is mocked offline, R falls back to the default 50 ('unknown')."""

    def failing_post(url: str, data: dict[str, str], timeout: float) -> dict:
        raise ConnectionError("Simulated offline Overpass API")

    result = compute_priority_with_osm(
        lat=18.5204,
        lng=73.8567,
        s_max=80.0,
        n_detections=1,
        mean_confidence=0.8,
        http_post=failing_post,
        use_cache=False,
    )
    assert result["road_class"] == "unknown"
    assert result["terms"]["road_importance"] == pytest.approx(50.0)
    # 0.45*80 (36) + 0.25*10 (2.5) + 0.20*50 (10) + 0.10*80 (8) = 56.5
    assert result["priority_score"] == pytest.approx(56.5, abs=1e-4)
    assert result["priority_label"] == "Medium"


@given(
    s1=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    s2=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    n1=st.integers(min_value=0, max_value=30),
    n2=st.integers(min_value=0, max_value=30),
    r1=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    r2=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    c1=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    c2=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=75, deadline=None)
def test_u07_hypothesis_priority_bounds_and_monotonicity(
    s1: float,
    s2: float,
    n1: int,
    n2: int,
    r1: float,
    r2: float,
    c1: float,
    c2: float,
) -> None:
    """U07 + Hypothesis: P stays within [0, 100] and never decreases when any term rises."""
    s_low, s_high = sorted((s1, s2))
    n_low, n_high = sorted((n1, n2))
    r_low, r_high = sorted((r1, r2))
    c_low, c_high = sorted((c1, c2))

    res_low = compute_priority_score(
        s_max=s_low,
        n_detections=n_low,
        road_importance=r_low,
        mean_confidence=c_low,
    )
    res_high = compute_priority_score(
        s_max=s_high,
        n_detections=n_high,
        road_importance=r_high,
        mean_confidence=c_high,
    )

    assert 0.0 <= res_low["priority_score"] <= 100.0
    assert 0.0 <= res_high["priority_score"] <= 100.0
    assert res_high["priority_score"] >= res_low["priority_score"] - 1e-6


def test_u07_weight_sensitivity_analysis(tmp_path: Path) -> None:
    """PRD Section 10/13: Varying each weight by +/-20% computes Spearman rank stability."""
    sample_segments = [
        {
            "id": i,
            "max_severity": 10.0 * i,
            "n_detections": i,
            "road_class": "primary",
            "mean_confidence": 0.8,
        }
        for i in range(1, 11)
    ]
    report = evaluate_weight_sensitivity(sample_segments, perturbation=0.20)
    assert "perturbations" in report
    assert "min_spearman_rho" in report
    assert len(report["perturbations"]) == 8  # 4 weights x (+20%, -20%)
    assert report["min_spearman_rho"] >= 0.9
