"""Unit tests for `rdai.data.download` and `rdai.data.eda` (`FR-D1`, `FR-D3`)."""

from __future__ import annotations

from pathlib import Path

from rdai.data.download import RDD2022_URLS, verify_rdd2022_layout
from rdai.data.eda import compute_eda_summary, generate_eda_figures


def test_verify_rdd2022_layout(tmp_path: Path) -> None:
    """Verify RDD2022 India directory layout check detects images and XML annotations."""
    assert "India" in RDD2022_URLS
    img_dir = tmp_path / "India" / "train" / "images"
    xml_dir = tmp_path / "India" / "train" / "annotations" / "xmls"
    img_dir.mkdir(parents=True)
    xml_dir.mkdir(parents=True)

    (img_dir / "India_000001.jpg").write_bytes(b"fake")
    (xml_dir / "India_000001.xml").write_text("<annotation></annotation>", encoding="utf-8")

    layout = verify_rdd2022_layout(tmp_path / "India")
    assert layout["valid"] is True
    assert layout["train_images"] == 1
    assert layout["train_xmls"] == 1


def test_eda_summary_and_figures_on_fixtures(project_root: Path, tmp_path: Path) -> None:
    """FR-D3: Compute EDA summary and generate class count, box size, and sample grid figures."""
    fixtures_dir = project_root / "tests" / "fixtures" / "generated"
    img_dir = fixtures_dir / "road_images"
    xml_dir = fixtures_dir / "road_xmls"

    summary = compute_eda_summary(xml_dir)
    assert summary["total_images"] == 5
    assert summary["total_boxes"] == 5
    assert summary["class_counts"]["D40"] == 2
    assert summary["class_counts"]["D00"] == 1
    assert summary["class_counts"]["D10"] == 1
    assert summary["class_counts"]["D20"] == 1

    out_figs = tmp_path / "figures"
    saved_paths = generate_eda_figures(
        xml_dir=xml_dir,
        img_dir=img_dir,
        out_dir=out_figs,
        summary_override=summary,
    )
    assert (out_figs / "eda_class_counts.png").is_file()
    assert (out_figs / "eda_box_size_distribution.png").is_file()
    assert (out_figs / "eda_sample_grid.png").is_file()
    assert len(saved_paths) == 3
