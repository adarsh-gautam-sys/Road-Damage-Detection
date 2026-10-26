"""EXIF GPS extraction, coordinate precedence resolution, and EXIF stripping (`FR-G1`, `NFR-2`, `U04`)."""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import piexif
import yaml
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_APP_YAML = ROOT_DIR / "configs" / "app.yaml"


def load_geo_config(app_yaml_path: Path = DEFAULT_APP_YAML) -> dict[str, Any]:
    """Load `geo` configuration block from `configs/app.yaml`."""
    with app_yaml_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return dict(data.get("geo", {}))


def _rational_to_float(val: Any) -> float:
    """Convert a `(num, den)` rational tuple or number into a float."""
    if isinstance(val, tuple) and len(val) == 2:
        num, den = float(val[0]), float(val[1])
        if den == 0.0:
            raise ValueError("Zero denominator in EXIF rational")
        return num / den
    return float(val)


def dms_rational_to_decimal(dms: Sequence[Any], ref: str | bytes) -> float:
    """Convert `(degrees, minutes, seconds)` rational sequence and hemisphere ref to decimal degrees."""
    if len(dms) != 3:
        raise ValueError(f"Expected 3 DMS elements, got {len(dms)}")

    deg = _rational_to_float(dms[0])
    minutes = _rational_to_float(dms[1])
    seconds = _rational_to_float(dms[2])
    decimal = deg + (minutes / 60.0) + (seconds / 3600.0)

    ref_str = ref.decode("ascii", errors="ignore") if isinstance(ref, bytes) else str(ref)
    ref_upper = ref_str.strip().upper()
    if ref_upper in ("S", "W"):
        decimal = -abs(decimal)
    elif ref_upper in ("N", "E"):
        decimal = abs(decimal)
    else:
        raise ValueError(f"Invalid GPS hemisphere reference: {ref!r}")

    return round(decimal, 6)


def validate_lat_lng(lat: float, lng: float) -> bool:
    """Return True if latitude is within `[-90, 90]` and longitude is within `[-180, 180]`."""
    return -90.0 <= float(lat) <= 90.0 and -180.0 <= float(lng) <= 180.0


def extract_exif_gps(image_source: Path | str | bytes) -> tuple[float, float] | None:
    """Extract `(lat, lng)` from image EXIF metadata, returning `None` if missing or corrupt (`U04`)."""
    try:
        if isinstance(image_source, Path | str):
            raw_bytes = Path(image_source).read_bytes()
        elif isinstance(image_source, bytes):
            raw_bytes = image_source
        else:
            return None

        exif_dict = piexif.load(raw_bytes)
        gps_ifd = exif_dict.get("GPS")
        if not gps_ifd:
            return None

        lat_dms = gps_ifd.get(piexif.GPSIFD.GPSLatitude)
        lat_ref = gps_ifd.get(piexif.GPSIFD.GPSLatitudeRef)
        lng_dms = gps_ifd.get(piexif.GPSIFD.GPSLongitude)
        lng_ref = gps_ifd.get(piexif.GPSIFD.GPSLongitudeRef)

        if not (lat_dms and lat_ref and lng_dms and lng_ref):
            return None

        lat = dms_rational_to_decimal(lat_dms, lat_ref)
        lng = dms_rational_to_decimal(lng_dms, lng_ref)
        if not validate_lat_lng(lat, lng):
            return None
        return (lat, lng)
    except Exception:
        return None


def check_india_bbox_warnings(
    lat: float,
    lng: float,
    app_yaml_path: Path = DEFAULT_APP_YAML,
) -> list[str]:
    """Return `['outside_india']` if `(lat, lng)` falls outside the configured India bounding box."""
    geo_cfg = load_geo_config(app_yaml_path)
    bbox = geo_cfg.get(
        "india_bbox",
        {"lat_min": 6.0, "lat_max": 38.0, "lng_min": 68.0, "lng_max": 98.0},
    )
    in_india = float(bbox["lat_min"]) <= lat <= float(bbox["lat_max"]) and float(
        bbox["lng_min"]
    ) <= lng <= float(bbox["lng_max"])
    return [] if in_india else ["outside_india"]


def resolve_detection_location(
    request_lat: float | None = None,
    request_lng: float | None = None,
    image_source: Path | str | bytes | None = None,
    track_coords: tuple[float, float] | None = None,
    app_yaml_path: Path = DEFAULT_APP_YAML,
) -> dict[str, Any]:
    """Resolve location in PRD Section 10 precedence order: `request > exif > track > none`."""
    if request_lat is not None and request_lng is not None:
        lat_f, lng_f = float(request_lat), float(request_lng)
        if not validate_lat_lng(lat_f, lng_f):
            raise ValueError(f"Coordinates out of valid range: lat={lat_f}, lng={lng_f}")
        return {
            "lat": round(lat_f, 6),
            "lng": round(lng_f, 6),
            "source": "request",
            "warnings": check_india_bbox_warnings(lat_f, lng_f, app_yaml_path),
        }

    if image_source is not None:
        exif_coords = extract_exif_gps(image_source)
        if exif_coords is not None:
            lat_f, lng_f = exif_coords
            return {
                "lat": lat_f,
                "lng": lng_f,
                "source": "exif",
                "warnings": check_india_bbox_warnings(lat_f, lng_f, app_yaml_path),
            }

    if track_coords is not None:
        lat_f, lng_f = float(track_coords[0]), float(track_coords[1])
        if validate_lat_lng(lat_f, lng_f):
            return {
                "lat": round(lat_f, 6),
                "lng": round(lng_f, 6),
                "source": "track",
                "warnings": check_india_bbox_warnings(lat_f, lng_f, app_yaml_path),
            }

    return {
        "lat": None,
        "lng": None,
        "source": "none",
        "warnings": [],
    }


def strip_exif_and_save_image(
    image_source: Path | str | bytes | Image.Image,
    out_path: Path,
    max_edge_px: int = 1280,
    quality: int = 88,
) -> tuple[int, int]:
    """Resize image to at most `max_edge_px` on its long edge and save JPEG with all EXIF stripped (`S04`)."""
    if isinstance(image_source, Image.Image):
        src_img = image_source.convert("RGB")
    elif isinstance(image_source, bytes):
        with Image.open(io.BytesIO(image_source)) as im:
            src_img = im.convert("RGB")
    else:
        with Image.open(image_source) as im:
            src_img = im.convert("RGB")

    w, h = src_img.size
    long_edge = max(w, h)
    if long_edge > max_edge_px and max_edge_px > 0:
        scale = max_edge_px / float(long_edge)
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        src_img = src_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create a fresh pixel-only Image so zero EXIF/ICC/XMP metadata survives
    clean_img = Image.new("RGB", src_img.size)
    clean_img.paste(src_img)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean_img.save(out_path, format="JPEG", quality=quality)
    return clean_img.size
