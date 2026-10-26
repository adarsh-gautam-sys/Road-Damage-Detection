"""Pascal VOC XML to YOLO normalized label conversion (`FR-D1`, verified by `U01`)."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DATA_YAML = ROOT_DIR / "configs" / "data.yaml"


def load_class_map(data_yaml_path: Path = DEFAULT_DATA_YAML) -> dict[str, int]:
    """Load class name -> class ID mapping from `configs/data.yaml`."""
    if data_yaml_path.is_file():
        with data_yaml_path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        names = cfg.get("names", {})
        return {str(code): int(idx) for idx, code in names.items()}
    return {"D00": 0, "D10": 1, "D20": 2, "D40": 3}


CLASS_MAP: dict[str, int] = load_class_map()


@dataclass(frozen=True)
class YoloBox:
    """Normalized YOLO bounding box (`class_id cx cy w h` in `[0, 1]`)."""

    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    def to_yolo_line(self) -> str:
        """Format box as a single YOLO annotation line."""
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


@dataclass
class ConversionStats:
    """Aggregated statistics across a converted Pascal VOC XML directory."""

    total_xmls: int = 0
    empty_xmls: int = 0
    clamped_boxes: int = 0
    kept_boxes: dict[str, int] = field(default_factory=lambda: {k: 0 for k in CLASS_MAP})
    dropped_boxes: dict[str, int] = field(default_factory=dict)


def convert_box(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float, bool]:
    """Convert pixel `(xmin, ymin, xmax, ymax)` to normalized `(cx, cy, w, h)` in `[0, 1]`.

    Coordinates extending outside `[0, img_w]` or `[0, img_h]` are clamped.
    Returns `(cx, cy, w, h, was_clamped)`.
    """
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Image dimensions must be positive, got ({img_w}, {img_h})")

    w_f = float(img_w)
    h_f = float(img_h)

    xmin_c = max(0.0, min(float(xmin), w_f))
    xmax_c = max(0.0, min(float(xmax), w_f))
    ymin_c = max(0.0, min(float(ymin), h_f))
    ymax_c = max(0.0, min(float(ymax), h_f))

    if xmax_c < xmin_c:
        xmin_c, xmax_c = xmax_c, xmin_c
    if ymax_c < ymin_c:
        ymin_c, ymax_c = ymax_c, ymin_c

    was_clamped = (
        xmin_c != float(xmin)
        or xmax_c != float(xmax)
        or ymin_c != float(ymin)
        or ymax_c != float(ymax)
    )

    cx = ((xmin_c + xmax_c) / 2.0) / w_f
    cy = ((ymin_c + ymax_c) / 2.0) / h_f
    w = (xmax_c - xmin_c) / w_f
    h = (ymax_c - ymin_c) / h_f

    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return cx, cy, w, h, was_clamped


def parse_voc_xml(
    xml_path: Path,
    class_map: dict[str, int] | None = None,
    default_size: tuple[int, int] = (720, 720),
) -> tuple[list[YoloBox], dict[str, int], int]:
    """Parse a Pascal VOC XML file into normalized YOLO boxes.

    Labels not present in `class_map` are dropped and counted in `dropped_counts`.
    Returns `(boxes, dropped_counts, clamped_count)`.
    """
    active_map = class_map if class_map is not None else CLASS_MAP
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_elem = root.find("size")
    img_w, img_h = default_size
    if size_elem is not None:
        w_text = size_elem.findtext("width")
        h_text = size_elem.findtext("height")
        if w_text and int(float(w_text)) > 0:
            img_w = int(float(w_text))
        if h_text and int(float(h_text)) > 0:
            img_h = int(float(h_text))

    boxes: list[YoloBox] = []
    dropped: dict[str, int] = {}
    clamped_count = 0

    for obj in root.findall("object"):
        raw_name = obj.findtext("name")
        if not raw_name:
            continue
        cls_name = raw_name.strip()
        if cls_name not in active_map:
            dropped[cls_name] = dropped.get(cls_name, 0) + 1
            continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = float(bndbox.findtext("xmin", "0"))
        ymin = float(bndbox.findtext("ymin", "0"))
        xmax = float(bndbox.findtext("xmax", "0"))
        ymax = float(bndbox.findtext("ymax", "0"))

        cx, cy, w, h, was_clamped = convert_box(xmin, ymin, xmax, ymax, img_w, img_h)
        if was_clamped:
            clamped_count += 1
        if w <= 0.0 or h <= 0.0:
            continue

        boxes.append(YoloBox(class_id=active_map[cls_name], cx=cx, cy=cy, w=w, h=h))

    return boxes, dropped, clamped_count


def convert_voc_file(
    xml_path: Path,
    out_txt_path: Path,
    class_map: dict[str, int] | None = None,
) -> tuple[list[YoloBox], dict[str, int], int]:
    """Convert one Pascal VOC XML file and write the YOLO `.txt` label file."""
    boxes, dropped, clamped_count = parse_voc_xml(xml_path, class_map=class_map)
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    if boxes:
        content = "\n".join(box.to_yolo_line() for box in boxes) + "\n"
    else:
        content = ""
    out_txt_path.write_text(content, encoding="utf-8")
    return boxes, dropped, clamped_count


def convert_dataset_directory(
    xml_dir: Path,
    out_labels_dir: Path,
    class_map: dict[str, int] | None = None,
) -> ConversionStats:
    """Convert all `.xml` files in `xml_dir` to YOLO `.txt` files in `out_labels_dir`."""
    active_map = class_map if class_map is not None else CLASS_MAP
    id_to_code = {v: k for k, v in active_map.items()}
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    stats = ConversionStats(kept_boxes={k: 0 for k in active_map})
    xml_files = sorted(xml_dir.glob("*.xml"))

    for xml_path in xml_files:
        stats.total_xmls += 1
        out_txt = out_labels_dir / f"{xml_path.stem}.txt"
        boxes, dropped, clamped = convert_voc_file(xml_path, out_txt, class_map=active_map)
        stats.clamped_boxes += clamped
        if not boxes:
            stats.empty_xmls += 1
        for box in boxes:
            code = id_to_code[box.class_id]
            stats.kept_boxes[code] = stats.kept_boxes.get(code, 0) + 1
        for d_name, d_count in dropped.items():
            stats.dropped_boxes[d_name] = stats.dropped_boxes.get(d_name, 0) + d_count

    logger.info(
        "Converted %d XMLs (%d empty, %d clamped). Kept: %s | Dropped: %s",
        stats.total_xmls,
        stats.empty_xmls,
        stats.clamped_boxes,
        stats.kept_boxes,
        stats.dropped_boxes,
    )
    return stats
