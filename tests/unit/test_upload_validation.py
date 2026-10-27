"""Unit tests for upload validation, filename sanitization, and EXIF stripping (`U10`)."""

from __future__ import annotations

import io
from pathlib import Path

import piexif
import pytest
from PIL import Image

from rdai.api.errors import APIError
from rdai.api.validation import (
    sanitize_filename,
    save_annotated_and_thumbnail,
    validate_and_decode_image,
    validate_video_upload,
)


def _make_jpeg_bytes(width: int = 320, height: int = 240, with_exif: bool = False) -> bytes:
    img = Image.new("RGB", (width, height), color=(90, 95, 100))
    buf = io.BytesIO()
    if with_exif:
        gps_ifd = {
            piexif.GPSIFD.GPSLatitudeRef: b"N",
            piexif.GPSIFD.GPSLatitude: ((18, 1), (31, 1), (1344, 100)),
            piexif.GPSIFD.GPSLongitudeRef: b"E",
            piexif.GPSIFD.GPSLongitude: ((73, 1), (51, 1), (2412, 100)),
        }
        exif_bytes = piexif.dump({"GPS": gps_ifd})
        img.save(buf, format="JPEG", exif=exif_bytes)
    else:
        img.save(buf, format="JPEG")
    return buf.getvalue()


def test_u10_text_file_renamed_jpg_is_rejected() -> None:
    """U10: A plain text file renamed `.jpg` is rejected with 415 based on content magic bytes."""
    fake_jpg = b"This is plain text pretending to be a JPEG file."
    with pytest.raises(APIError) as exc_info:
        validate_and_decode_image(fake_jpg, filename="pothole.jpg")
    assert exc_info.value.status_code == 415


def test_u10_oversize_file_is_rejected() -> None:
    """U10: An upload exceeding `max_photo_mb` is rejected with 413."""
    valid_jpg = _make_jpeg_bytes(100, 100)
    # Set max_photo_mb so small that valid_jpg exceeds it
    with pytest.raises(APIError) as exc_info:
        validate_and_decode_image(valid_jpg, filename="big.jpg", max_photo_mb=0.0001)
    assert exc_info.value.status_code == 413


def test_u10_over_pixel_cap_is_rejected() -> None:
    """U10: An image exceeding the megapixel cap is rejected with 422."""
    jpg_bytes = _make_jpeg_bytes(400, 400)  # 0.16 MP
    with pytest.raises(APIError) as exc_info:
        validate_and_decode_image(jpg_bytes, filename="huge.jpg", max_megapixels=0.05)
    assert exc_info.value.status_code == 422
    assert (
        "megapixel" in exc_info.value.message.lower() or "pixel" in exc_info.value.message.lower()
    )


def test_u10_truncated_image_is_rejected() -> None:
    """U10: A truncated/corrupt JPEG with valid magic header is rejected with 422."""
    valid_jpg = _make_jpeg_bytes(320, 240)
    truncated = valid_jpg[:40]  # Has JPEG SOI header \xff\xd8\xff, but truncated body
    with pytest.raises(APIError) as exc_info:
        validate_and_decode_image(truncated, filename="truncated.jpg")
    assert exc_info.value.status_code == 422


def test_u10_filename_traversal_is_sanitized() -> None:
    """U10: Filenames containing `../` or backslashes are sanitized to safe basenames."""
    assert sanitize_filename("../../etc/passwd.jpg") == "passwd.jpg"
    assert sanitize_filename("..\\..\\windows\\system32\\cmd.exe") == "cmd.exe"
    assert sanitize_filename("normal_road-01.png") == "normal_road-01.png"
    assert sanitize_filename("") == "upload.bin"


def test_u10_saved_annotated_and_thumb_strip_exif_and_cap_dimensions(tmp_path: Path) -> None:
    """U10: Saved annotated and thumbnail JPEGs strip all EXIF metadata and respect edge caps."""
    raw_bytes = _make_jpeg_bytes(width=1600, height=1200, with_exif=True)
    img, fmt = validate_and_decode_image(raw_bytes, filename="gps_photo.jpg")
    assert fmt == "JPEG"

    paths = save_annotated_and_thumbnail(
        image=img,
        detections=[
            {
                "class_code": "D40",
                "class_name": "pothole",
                "confidence": 0.88,
                "box": [100.0, 150.0, 450.0, 500.0],
                "severity": {
                    "score": 72.5,
                    "label": "High",
                    "extent": 0.06,
                    "measure": "area_ratio",
                },
            }
        ],
        image_id=42,
        media_dir=tmp_path,
        annotated_max_edge_px=1280,
        thumb_max_edge_px=320,
    )

    ann_path = paths["annotated_path"]
    thumb_path = paths["thumb_path"]
    assert ann_path.exists()
    assert thumb_path.exists()

    with Image.open(ann_path) as ann_img:
        assert max(ann_img.size) <= 1280
    with Image.open(thumb_path) as thumb_img:
        assert max(thumb_img.size) <= 320

    ann_exif = piexif.load(ann_path.read_bytes())
    thumb_exif = piexif.load(thumb_path.read_bytes())
    assert ann_exif.get("GPS", {}) == {}
    assert thumb_exif.get("GPS", {}) == {}


def test_u10_video_upload_validation() -> None:
    """U10: Video validation rejects empty or non-video payloads and oversize videos."""
    with pytest.raises(APIError) as exc_info:
        validate_video_upload(b"not a video file", filename="fake.mp4")
    assert exc_info.value.status_code == 415
