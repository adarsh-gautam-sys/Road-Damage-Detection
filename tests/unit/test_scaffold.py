"""Milestone M0 scaffold tests: verify package layout and YAML configurations."""

from __future__ import annotations

from pathlib import Path

import yaml

import rdai

ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT_DIR / "configs"


def test_package_version() -> None:
    """Verify rdai package exposes a semantic version string."""
    assert hasattr(rdai, "__version__")
    assert isinstance(rdai.__version__, str)
    assert len(rdai.__version__.split(".")) == 3


def test_required_configs_exist_and_parse() -> None:
    """Verify all 5 required YAML config files exist and contain valid mappings."""
    expected_files = [
        "data.yaml",
        "train.yaml",
        "severity.yaml",
        "priority.yaml",
        "app.yaml",
    ]
    for filename in expected_files:
        cfg_path = CONFIGS_DIR / filename
        assert cfg_path.is_file(), f"Missing config file: {cfg_path}"
        with cfg_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert isinstance(data, dict), f"Config {filename} must parse to a dict"


def test_data_yaml_classes() -> None:
    """Verify configs/data.yaml defines the 4 RDD2022 classes in exact order."""
    with (CONFIGS_DIR / "data.yaml").open("r", encoding="utf-8") as fh:
        data_cfg = yaml.safe_load(fh)
    assert data_cfg["names"] == {0: "D00", 1: "D10", 2: "D20", 3: "D40"}


def test_severity_and_priority_yaml_structure() -> None:
    """Verify severity and priority configs hold the PRD thresholds and weights."""
    with (CONFIGS_DIR / "severity.yaml").open("r", encoding="utf-8") as fh:
        sev_cfg = yaml.safe_load(fh)
    assert "classes" in sev_cfg
    assert sev_cfg["classes"]["D40"]["e_low"] == 0.015
    assert sev_cfg["classes"]["D40"]["e_high"] == 0.05

    with (CONFIGS_DIR / "priority.yaml").open("r", encoding="utf-8") as fh:
        prio_cfg = yaml.safe_load(fh)
    weights = prio_cfg["weights"]
    total = (
        weights["s_max"] + weights["density"] + weights["road_importance"] + weights["confidence"]
    )
    assert abs(total - 1.0) < 1e-6
