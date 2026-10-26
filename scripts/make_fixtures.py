"""Generate deterministic synthetic test fixtures for Road Damage AI (`tests/fixtures/generated`).

Per PRD Section 12, this script creates:
- 5 labelled 720x720 road fixture images + Pascal VOC XML annotations
- 1 JPEG photo with synthetic EXIF GPS written via `piexif`
- 1 GPX track and 1 CSV (`t,lat,lng`) track
- 20 synthetic non-road images (solid black, white, random noise, gradients, text)
- 1 short 15-frame synthetic MP4 video made by repeating a road frame with small jitter
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import piexif
from PIL import Image, ImageDraw

ROOT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT_DIR / "tests" / "fixtures" / "generated"


def _deg_to_dms_rational(
    deg_float: float,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Convert decimal degrees to EXIF rational DMS tuple."""
    deg_abs = abs(deg_float)
    degrees = int(deg_abs)
    minutes_float = (deg_abs - degrees) * 60.0
    minutes = int(minutes_float)
    seconds = round((minutes_float - minutes) * 60.0 * 10000)
    return ((degrees, 1), (minutes, 1), (seconds, 10000))


def build_road_fixtures(base_dir: Path) -> None:
    """Create 5 synthetic 720x720 road images and matching Pascal VOC XML files."""
    img_dir = base_dir / "road_images"
    xml_dir = base_dir / "road_xmls"
    img_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)

    classes = ["D00", "D10", "D20", "D40", "D40"]
    for idx, cls_code in enumerate(classes, start=1):
        stem = f"India_{idx:06d}"
        img = Image.new("RGB", (720, 720), color=(90, 92, 95))
        draw = ImageDraw.Draw(img)
        # Draw a dark pothole/crack patch in the lower road region
        x1, y1, x2, y2 = 220 + idx * 10, 420, 360 + idx * 10, 520
        draw.rectangle([x1, y1, x2, y2], fill=(35, 35, 38), outline=(20, 20, 22))
        img_path = img_dir / f"{stem}.jpg"
        img.save(img_path, format="JPEG", quality=90)

        xml_content = f"""<annotation>
  <folder>images</folder>
  <filename>{stem}.jpg</filename>
  <size>
    <width>720</width>
    <height>720</height>
    <depth>3</depth>
  </size>
  <object>
    <name>{cls_code}</name>
    <bndbox>
      <xmin>{x1}</xmin>
      <ymin>{y1}</ymin>
      <xmax>{x2}</xmax>
      <ymax>{y2}</ymax>
    </bndbox>
  </object>
</annotation>
"""
        (xml_dir / f"{stem}.xml").write_text(xml_content, encoding="utf-8")


def build_exif_gps_photo(base_dir: Path) -> None:
    """Create a JPEG containing synthetic EXIF GPS coordinates (Pune, India: 18.5204 N, 73.8567 E)."""
    img = Image.new("RGB", (720, 720), color=(85, 85, 90))
    draw = ImageDraw.Draw(img)
    draw.ellipse([260, 380, 440, 510], fill=(30, 30, 32))

    gps_ifd = {
        piexif.GPSIFD.GPSLatitudeRef: b"N",
        piexif.GPSIFD.GPSLatitude: _deg_to_dms_rational(18.5204),
        piexif.GPSIFD.GPSLongitudeRef: b"E",
        piexif.GPSIFD.GPSLongitude: _deg_to_dms_rational(73.8567),
    }
    exif_bytes = piexif.dump({"GPS": gps_ifd})
    img.save(base_dir / "photo_with_gps.jpg", format="JPEG", exif=exif_bytes, quality=90)


def build_tracks(base_dir: Path) -> None:
    """Create synthetic CSV and GPX GPS tracks."""
    csv_content = "t,lat,lng\n0.0,28.4595,77.0266\n2.0,28.4605,77.0276\n4.0,28.4615,77.0286\n"
    (base_dir / "track.csv").write_text(csv_content, encoding="utf-8")

    gpx_content = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="rdai-fixtures" xmlns="http://www.topografix.com/GPX/1/1">
  <trk>
    <name>Synthetic Gurugram Drive</name>
    <trkseg>
      <trkpt lat="28.4595" lon="77.0266"><time>2026-10-26T10:00:00Z</time></trkpt>
      <trkpt lat="28.4605" lon="77.0276"><time>2026-10-26T10:00:02Z</time></trkpt>
      <trkpt lat="28.4615" lon="77.0286"><time>2026-10-26T10:00:04Z</time></trkpt>
    </trkseg>
  </trk>
</gpx>
"""
    (base_dir / "track.gpx").write_text(gpx_content, encoding="utf-8")


def build_non_road_negatives(base_dir: Path) -> None:
    """Create 20 synthetic non-road images (black, white, noise, gradients, text)."""
    neg_dir = base_dir / "negatives"
    neg_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed=0)

    for idx in range(20):
        kind = idx % 5
        if kind == 0:
            arr = np.zeros((320, 320, 3), dtype=np.uint8)
        elif kind == 1:
            arr = np.full((320, 320, 3), 255, dtype=np.uint8)
        elif kind == 2:
            arr = rng.integers(0, 256, size=(320, 320, 3), dtype=np.uint8)
        elif kind == 3:
            grad = np.linspace(0, 255, 320, dtype=np.uint8)
            arr = np.broadcast_to(grad[:, None, None], (320, 320, 3)).copy()
        else:
            img = Image.new("RGB", (320, 320), color=(240, 240, 245))
            draw = ImageDraw.Draw(img)
            draw.text((40, 140), f"SYNTHETIC TEXT DOC #{idx}", fill=(10, 10, 10))
            img.save(neg_dir / f"neg_{idx:02d}.jpg", format="JPEG")
            continue

        Image.fromarray(arr).save(neg_dir / f"neg_{idx:02d}.jpg", format="JPEG")


def build_synthetic_video(base_dir: Path) -> None:
    """Create a 15-frame synthetic MP4 video with a slightly jittered pothole patch."""
    video_path = base_dir / "sample_drive.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 5.0, (320, 320))
    for frame_idx in range(15):
        frame = np.full((320, 320, 3), 90, dtype=np.uint8)
        dx = frame_idx % 3
        cv2.rectangle(frame, (110 + dx, 180), (190 + dx, 230), (25, 25, 25), -1)
        writer.write(frame)
    writer.release()


def main() -> None:
    """Generate all deterministic test fixtures."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_road_fixtures(OUT_DIR)
    build_exif_gps_photo(OUT_DIR)
    build_tracks(OUT_DIR)
    build_non_road_negatives(OUT_DIR)
    build_synthetic_video(OUT_DIR)
    print(f"Fixtures generated in: {OUT_DIR}")


if __name__ == "__main__":
    main()
