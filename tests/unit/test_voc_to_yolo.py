"""Unit tests U01: Pascal VOC XML to YOLO label conversion (`rdai.data.voc_to_yolo`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rdai.data.voc_to_yolo import (
    CLASS_MAP,
    convert_box,
    convert_dataset_directory,
    convert_voc_file,
    parse_voc_xml,
)


def test_u01_normalised_box_conversion() -> None:
    """U01: Boxes become normalised centre, width and height within 0 and 1."""
    cx, cy, w, h, clamped = convert_box(180, 360, 540, 540, img_w=720, img_h=720)
    assert not clamped
    assert pytest.approx(cx, abs=1e-6) == 0.5
    assert pytest.approx(cy, abs=1e-6) == 0.625
    assert pytest.approx(w, abs=1e-6) == 0.5
    assert pytest.approx(h, abs=1e-6) == 0.25
    assert 0.0 <= cx <= 1.0
    assert 0.0 <= cy <= 1.0
    assert 0.0 < w <= 1.0
    assert 0.0 < h <= 1.0


def test_u01_boxes_beyond_image_are_clamped() -> None:
    """U01: Boxes extending outside image bounds are clamped to [0, width/height]."""
    cx, cy, w, h, clamped = convert_box(-50, 360, 800, 750, img_w=720, img_h=720)
    assert clamped is True
    assert pytest.approx(cx, abs=1e-6) == 0.5
    assert pytest.approx(cy, abs=1e-6) == 0.75
    assert pytest.approx(w, abs=1e-6) == 1.0
    assert pytest.approx(h, abs=1e-6) == 0.5


def test_u01_drops_unknown_labels_and_counts_them(tmp_path: Path) -> None:
    """U01: Labels outside D00, D10, D20, D40 are dropped and counted."""
    xml_content = """<annotation>
  <size><width>720</width><height>720</height><depth>3</depth></size>
  <object>
    <name>D00</name>
    <bndbox><xmin>100</xmin><ymin>100</ymin><xmax>200</xmax><ymax>300</ymax></bndbox>
  </object>
  <object>
    <name>D43</name>
    <bndbox><xmin>50</xmin><ymin>50</ymin><xmax>150</xmax><ymax>150</ymax></bndbox>
  </object>
  <object>
    <name>D44</name>
    <bndbox><xmin>300</xmin><ymin>300</ymin><xmax>400</xmax><ymax>400</ymax></bndbox>
  </object>
  <object>
    <name>D40</name>
    <bndbox><xmin>-10</xmin><ymin>400</ymin><xmax>170</xmax><ymax>580</ymax></bndbox>
  </object>
</annotation>"""
    xml_path = tmp_path / "India_000001.xml"
    txt_path = tmp_path / "India_000001.txt"
    xml_path.write_text(xml_content, encoding="utf-8")

    boxes, dropped, clamped_count = convert_voc_file(xml_path, txt_path)

    assert len(boxes) == 2
    assert boxes[0].class_id == CLASS_MAP["D00"]
    assert boxes[1].class_id == CLASS_MAP["D40"]
    assert dropped == {"D43": 1, "D44": 1}
    assert clamped_count == 1

    lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("0 ")
    assert lines[1].startswith("3 ")


def test_u01_empty_xml_gives_empty_label_file(tmp_path: Path) -> None:
    """U01: An XML with no objects gives an empty label file."""
    xml_content = """<annotation>
  <size><width>720</width><height>720</height><depth>3</depth></size>
</annotation>"""
    xml_path = tmp_path / "India_000002.xml"
    txt_path = tmp_path / "India_000002.txt"
    xml_path.write_text(xml_content, encoding="utf-8")

    boxes, dropped, clamped_count = convert_voc_file(xml_path, txt_path)

    assert boxes == []
    assert dropped == {}
    assert clamped_count == 0
    assert txt_path.exists()
    assert txt_path.read_text(encoding="utf-8") == ""


def test_u01_directory_conversion_stats(tmp_path: Path) -> None:
    """U01: Directory conversion aggregates kept, dropped, empty, and clamped counts."""
    xml_dir = tmp_path / "xmls"
    out_dir = tmp_path / "labels"
    xml_dir.mkdir()

    (xml_dir / "India_000001.xml").write_text(
        """<annotation>
        <size><width>720</width><height>720</height></size>
        <object><name>D10</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>360</xmax><ymax>360</ymax></bndbox></object>
        <object><name>D50</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>50</ymax></bndbox></object>
        </annotation>""",
        encoding="utf-8",
    )
    (xml_dir / "India_000002.xml").write_text(
        "<annotation><size><width>720</width><height>720</height></size></annotation>",
        encoding="utf-8",
    )

    stats = convert_dataset_directory(xml_dir, out_dir)
    assert stats.total_xmls == 2
    assert stats.empty_xmls == 1
    assert stats.kept_boxes["D10"] == 1
    assert stats.dropped_boxes == {"D50": 1}
    assert (out_dir / "India_000001.txt").is_file()
    assert (out_dir / "India_000002.txt").is_file()
    assert (out_dir / "India_000002.txt").read_text(encoding="utf-8") == ""


def test_u01_parse_voc_xml_handles_float_coordinates(tmp_path: Path) -> None:
    """U01: Handles float strings in VOC bndbox elements cleanly."""
    xml_path = tmp_path / "float_box.xml"
    xml_path.write_text(
        """<annotation>
        <size><width>720</width><height>720</height></size>
        <object><name>D20</name><bndbox><xmin>100.5</xmin><ymin>200.25</ymin><xmax>300.5</xmax><ymax>400.25</ymax></bndbox></object>
        </annotation>""",
        encoding="utf-8",
    )
    boxes, dropped, clamped = parse_voc_xml(xml_path)
    assert len(boxes) == 1
    assert boxes[0].class_id == 2
    assert dropped == {}
    assert clamped == 0
