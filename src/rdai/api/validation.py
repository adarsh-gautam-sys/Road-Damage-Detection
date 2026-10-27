"""Upload content validation, decompression bomb protection, filename sanitization, and EXIF-stripped rendering (`U10`, `S01`, `S02`, `S04`)."""

from __future__ import annotations

import io
import re
from pathlib import Path, PurePath
from typing import Any
from urllib.parse import unquote

from PIL import Image, ImageDraw, UnidentifiedImageError

from rdai.api.errors import APIError
from rdai.geo.exif import strip_exif_and_save_image

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._\-,\"' ]+")


def sanitize_filename(filename: str | None) -> str:
    """Strip directory traversal components (`../`, `..\\`) and return a safe basename (`U10`, `S01`)."""
    if not filename:
        return "upload.bin"
    normalized = unquote(filename).replace("\\", "/")
    base = PurePath(normalized).name.strip()
    while base.startswith("."):
        base = base.lstrip(".")
    cleaned = _SAFE_FILENAME_RE.sub("_", base).strip()
    if not cleaned or cleaned in {".", ".."}:
        return "upload.bin"
    return cleaned[:200]


def sniff_image_format(raw_bytes: bytes) -> str | None:
    """Detect image format (`JPEG`, `PNG`, `WEBP`) strictly from magic bytes (`U10`)."""
    if len(raw_bytes) < 12:
        return None
    if raw_bytes.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"WEBP":
        return "WEBP"
    return None


def validate_and_decode_image(
    raw_bytes: bytes,
    filename: str | None = None,
    max_photo_mb: float = 15.0,
    max_megapixels: float = 40.0,
    allowed_formats: list[str] | None = None,
) -> tuple[Image.Image, str]:
    """Validate photo size, magic bytes, pixel count, and full decode integrity (`U10`, `A03`, `S02`)."""
    if not raw_bytes:
        raise APIError(422, "EMPTY_FILE", "Uploaded image file is empty.")

    max_bytes = int(float(max_photo_mb) * 1024 * 1024)
    if len(raw_bytes) > max_bytes:
        raise APIError(
            413,
            "PAYLOAD_TOO_LARGE",
            f"Image exceeds maximum allowed size of {max_photo_mb} MB.",
        )

    fmt = sniff_image_format(raw_bytes)
    allowed = {f.upper() for f in (allowed_formats or ["JPEG", "PNG", "WEBP"])}
    if fmt is None or fmt not in allowed:
        raise APIError(
            415,
            "UNSUPPORTED_MEDIA_TYPE",
            f"File '{sanitize_filename(filename)}' is not a valid {', '.join(sorted(allowed))} image.",
        )

    max_pixels = int(float(max_megapixels) * 1_000_000)
    prev_max_pixels = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_pixels
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            width, height = img.size
            if width * height > max_pixels:
                raise APIError(
                    422,
                    "PIXEL_LIMIT_EXCEEDED",
                    f"Image resolution ({width}x{height}) exceeds {max_megapixels} megapixel cap.",
                )
            img.load()
            decoded = img.convert("RGB")
    except APIError:
        raise
    except Image.DecompressionBombError as exc:
        raise APIError(
            422,
            "PIXEL_LIMIT_EXCEEDED",
            f"Image exceeds maximum allowed megapixel limit ({max_megapixels} MP).",
        ) from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise APIError(
            422,
            "CORRUPT_IMAGE",
            f"Uploaded image could not be decoded: {exc}",
        ) from exc
    finally:
        Image.MAX_IMAGE_PIXELS = prev_max_pixels

    return decoded, fmt


def validate_video_upload(
    raw_bytes: bytes,
    filename: str | None = None,
    max_video_mb: float = 200.0,
) -> str:
    """Validate video file size and container magic signature (`U10`)."""
    if not raw_bytes:
        raise APIError(422, "EMPTY_VIDEO", "Uploaded video file is empty.")

    max_bytes = int(float(max_video_mb) * 1024 * 1024)
    if len(raw_bytes) > max_bytes:
        raise APIError(
            413,
            "PAYLOAD_TOO_LARGE",
            f"Video exceeds maximum allowed size of {max_video_mb} MB.",
        )

    is_mp4_mov = len(raw_bytes) >= 12 and raw_bytes[4:8] in {b"ftyp", b"moov", b"wide", b"mdat"}
    is_avi = len(raw_bytes) >= 12 and raw_bytes[:4] == b"RIFF" and raw_bytes[8:12] == b"AVI "
    is_mkv_webm = len(raw_bytes) >= 4 and raw_bytes[:4] == b"\x1a\x45\xdf\xa3"

    if not (is_mp4_mov or is_avi or is_mkv_webm):
        raise APIError(
            415,
            "UNSUPPORTED_MEDIA_TYPE",
            f"File '{sanitize_filename(filename)}' is not a supported video container (MP4/MOV/AVI/WEBM).",
        )

    return sanitize_filename(filename)


def _resize_max_edge(image: Image.Image, max_edge_px: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_edge_px:
        return image.copy()
    scale = float(max_edge_px) / float(longest)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def save_annotated_and_thumbnail(
    image: Image.Image,
    detections: list[dict[str, Any]],
    image_id: int,
    media_dir: Path,
    annotated_max_edge_px: int = 1280,
    thumb_max_edge_px: int = 320,
) -> dict[str, Path]:
    """Render annotated bounding boxes, resize to edge caps, and save EXIF-free JPEGs (`U10`, `S04`)."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)

    tier_colors = {
        "High": (220, 38, 38),
        "Medium": (234, 179, 8),
        "Low": (34, 197, 94),
    }

    for det in detections:
        box = det.get("box", [0, 0, 0, 0])
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
        sev = det.get("severity", {}) or {}
        label_tier = str(sev.get("label", "Medium"))
        color = tier_colors.get(label_tier, (59, 130, 246))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        caption = (
            f"{det.get('class_code', 'D40')} {label_tier} ({float(det.get('confidence', 0.0)):.2f})"
        )
        draw.text((max(0.0, x1 + 4), max(0.0, y1 + 4)), caption, fill=color)

    annotated_img = _resize_max_edge(canvas, annotated_max_edge_px)
    thumb_img = _resize_max_edge(canvas, thumb_max_edge_px)

    annotated_path = media_dir / "annotated" / f"{int(image_id)}.jpg"
    thumb_path = media_dir / "thumbs" / f"{int(image_id)}.jpg"

    strip_exif_and_save_image(annotated_img, annotated_path, max_edge_px=annotated_max_edge_px)
    strip_exif_and_save_image(thumb_img, thumb_path, max_edge_px=thumb_max_edge_px)

    return {
        "annotated_path": annotated_path,
        "thumb_path": thumb_path,
    }
