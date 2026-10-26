"""Unit and property-based tests U03: Severity assessment rules (`rdai.severity.rules`)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from rdai.severity.rules import (
    compute_box_extent,
    compute_severity_from_extent,
    grade_detection_box,
    load_severity_config,
)


def test_u03_extent_measures_area_and_diagonal() -> None:
    """U03: D40 and D20 use area_ratio; D00 and D10 use diag_ratio."""
    # 72x72 box in 720x720 image -> area ratio = 0.01, diag ratio = 0.1
    e_area, m_area = compute_box_extent("D40", [0, 0, 72, 72], img_w=720, img_h=720)
    assert m_area == "area_ratio"
    assert pytest.approx(e_area, abs=1e-6) == 0.01

    e_diag, m_diag = compute_box_extent("D00", [0, 0, 72, 72], img_w=720, img_h=720)
    assert m_diag == "diag_ratio"
    assert pytest.approx(e_diag, abs=1e-6) == 0.10


def test_u03_continuity_at_e_low_and_e_high() -> None:
    """U03: S is continuous at e_low and e_high within 0.1 and labels match tiers at boundaries."""
    cfg = load_severity_config()
    e_low = cfg["classes"]["D40"]["e_low"]  # 0.015
    e_high = cfg["classes"]["D40"]["e_high"]  # 0.05

    # Continuity at e_low
    below_low = compute_severity_from_extent("D40", e_low - 1e-7, cfg)
    at_low = compute_severity_from_extent("D40", e_low, cfg)
    assert below_low["label"] == "Low"
    assert at_low["label"] == "Medium"
    assert abs(at_low["score"] - below_low["score"]) <= 0.1001

    # Continuity at e_high
    below_high = compute_severity_from_extent("D40", e_high - 1e-7, cfg)
    at_high = compute_severity_from_extent("D40", e_high, cfg)
    assert below_high["label"] == "Medium"
    assert at_high["label"] == "High"
    assert abs(at_high["score"] - below_high["score"]) <= 0.1001


def test_u03_d20_never_below_medium_and_zero_area_gives_low() -> None:
    """U03: D20 is never below Medium; zero area gives Low for non-D20 classes."""
    cfg = load_severity_config()

    zero_d40 = grade_detection_box("D40", [100, 100, 100, 100], img_w=720, img_h=720, config=cfg)
    assert zero_d40["extent"] == 0.0
    assert zero_d40["score"] == 0.0
    assert zero_d40["label"] == "Low"

    tiny_d20 = grade_detection_box("D20", [100, 100, 110, 110], img_w=720, img_h=720, config=cfg)
    assert tiny_d20["label"] == "Medium"
    assert tiny_d20["score"] >= 0.0


def test_u03_changing_yaml_changes_result(tmp_path: Path) -> None:
    """U03: Changing the YAML thresholds dynamically changes the severity result."""
    custom_yaml = tmp_path / "severity_custom.yaml"
    base_cfg = load_severity_config()
    # Make D40 e_high very small (0.005) so a 1% box becomes High instead of Low
    base_cfg["classes"]["D40"]["e_low"] = 0.002
    base_cfg["classes"]["D40"]["e_high"] = 0.005
    custom_yaml.write_text(yaml.safe_dump(base_cfg), encoding="utf-8")

    default_res = grade_detection_box("D40", [0, 0, 72, 72], img_w=720, img_h=720)
    custom_res = grade_detection_box(
        "D40",
        [0, 0, 72, 72],
        img_w=720,
        img_h=720,
        config=load_severity_config(custom_yaml),
    )

    assert default_res["label"] == "Low"
    assert custom_res["label"] == "High"
    assert custom_res["score"] > default_res["score"]


@settings(max_examples=100)
@given(
    cls_code=st.sampled_from(["D00", "D10", "D20", "D40"]),
    e1=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    e2=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_u03_hypothesis_bounds_and_monotonicity(cls_code: str, e1: float, e2: float) -> None:
    """Hypothesis property test: S stays in [0, 100] and never decreases as extent grows."""
    low_e, high_e = (e1, e2) if e1 <= e2 else (e2, e1)
    r1 = compute_severity_from_extent(cls_code, low_e)
    r2 = compute_severity_from_extent(cls_code, high_e)

    assert 0.0 <= r1["score"] <= 100.0
    assert 0.0 <= r2["score"] <= 100.0
    assert r2["score"] >= r1["score"] - 1e-9
    if cls_code == "D20":
        assert r1["label"] in ("Medium", "High")
        assert r2["label"] in ("Medium", "High")
