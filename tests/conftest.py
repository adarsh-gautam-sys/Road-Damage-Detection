"""Shared pytest fixtures for Road Damage AI test suites."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def generated_fixtures_dir(project_root: Path) -> Path:
    """Return `tests/fixtures/generated`, generating synthetic fixtures if missing."""
    out_dir = project_root / "tests" / "fixtures" / "generated"
    if not (out_dir / "photo_with_gps.jpg").exists():
        from scripts.make_fixtures import main as generate_fixtures

        generate_fixtures()
    return out_dir
