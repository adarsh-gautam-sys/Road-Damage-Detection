"""Unit and property-based tests U04, U05, U06: EXIF GPS, GPX/CSV track interpolation, DBSCAN clustering, and mocked OSM lookup."""

from __future__ import annotations

from pathlib import Path

import piexif
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image

from rdai.geo.cluster import cluster_detections_dbscan, haversine_distance_m
from rdai.geo.exif import (
    dms_rational_to_decimal,
    extract_exif_gps,
    resolve_detection_location,
    strip_exif_and_save_image,
)
from rdai.geo.osm import lookup_road_importance
from rdai.geo.track import interpolate_track_position, parse_csv_track, parse_gpx_track


def test_u04_dms_to_decimal_and_south_west_negative() -> None:
    """U04: Degrees, minutes, seconds convert to decimal; South and West are negative."""
    # 18 deg 30 min 0 sec = 18.5
    dms = ((18, 1), (30, 1), (0, 1))
    assert dms_rational_to_decimal(dms, "N") == pytest.approx(18.5)
    assert dms_rational_to_decimal(dms, "S") == pytest.approx(-18.5)
    assert dms_rational_to_decimal(dms, "E") == pytest.approx(18.5)
    assert dms_rational_to_decimal(dms, "W") == pytest.approx(-18.5)


def test_u04_exif_extraction_missing_and_corrupt(project_root: Path, tmp_path: Path) -> None:
    """U04: Valid EXIF GPS parses; missing GPS gives None; corrupt EXIF gives None and does not raise."""
    gps_photo = project_root / "tests" / "fixtures" / "generated" / "photo_with_gps.jpg"
    coords = extract_exif_gps(gps_photo)
    assert coords is not None
    assert coords[0] == pytest.approx(18.5204, abs=1e-3)
    assert coords[1] == pytest.approx(73.8567, abs=1e-3)

    # Image without GPS
    plain_img = tmp_path / "no_gps.jpg"
    Image.new("RGB", (64, 64), color=(100, 100, 100)).save(plain_img, format="JPEG")
    assert extract_exif_gps(plain_img) is None

    # Corrupt bytes
    corrupt_img = tmp_path / "corrupt.jpg"
    corrupt_img.write_bytes(b"\xff\xd8\xff\xe1\x00\x10Exif\x00\x00garbage_not_tiff")
    assert extract_exif_gps(corrupt_img) is None

    # Request coordinates override EXIF and outside_india warning is added when outside India bbox
    loc = resolve_detection_location(
        request_lat=51.5074,
        request_lng=-0.1278,
        image_source=gps_photo,
    )
    assert loc["source"] == "request"
    assert loc["lat"] == pytest.approx(51.5074)
    assert "outside_india" in loc["warnings"]

    # Stripping EXIF removes GPS
    stripped_path = tmp_path / "stripped.jpg"
    strip_exif_and_save_image(gps_photo, stripped_path, max_edge_px=320)
    assert extract_exif_gps(stripped_path) is None
    exif_dict = piexif.load(str(stripped_path))
    assert not exif_dict.get("GPS")


@settings(max_examples=80)
@given(raw_bytes=st.binary(min_size=0, max_size=512))
def test_u04_hypothesis_exif_parser_never_raises_on_random_bytes(raw_bytes: bytes) -> None:
    """Hypothesis property test: EXIF parser never raises an unhandled exception on random bytes."""
    res = extract_exif_gps(raw_bytes)
    assert res is None or (isinstance(res, tuple) and len(res) == 2)


def test_u05_csv_and_gpx_track_interpolation(project_root: Path) -> None:
    """U05: CSV and GPX both parse; midpoint gives exact midpoint; >5 s outside track gives None."""
    fixtures_dir = project_root / "tests" / "fixtures" / "generated"
    csv_points = parse_csv_track(fixtures_dir / "track.csv")
    gpx_points = parse_gpx_track(fixtures_dir / "track.gpx")

    assert len(csv_points) == 3
    assert len(gpx_points) == 3

    # Points at t=0.0: (28.4595, 77.0266) and t=2.0: (28.4605, 77.0276)
    # Midpoint at t=1.0 -> (28.4600, 77.0271)
    mid_csv = interpolate_track_position(csv_points, timestamp_s=1.0, tolerance_s=5.0)
    assert mid_csv is not None
    assert mid_csv[0] == pytest.approx(28.4600, abs=1e-6)
    assert mid_csv[1] == pytest.approx(77.0271, abs=1e-6)

    mid_gpx = interpolate_track_position(gpx_points, timestamp_s=1.0, tolerance_s=5.0)
    assert mid_gpx is not None
    assert mid_gpx[0] == pytest.approx(28.4600, abs=1e-6)
    assert mid_gpx[1] == pytest.approx(77.0271, abs=1e-6)

    # More than 5 s outside track (track ends at t=4.0 -> t=10.0 is 6 s outside) gives None
    assert interpolate_track_position(csv_points, timestamp_s=10.0, tolerance_s=5.0) is None
    assert interpolate_track_position(csv_points, timestamp_s=-6.0, tolerance_s=5.0) is None


def test_u06_haversine_and_dbscan_clustering() -> None:
    """U06: 1 deg lat is 111.19 km within 0.5%; 10 m points merge; 100 m points do not; 0 and 1 point work."""
    one_deg_m = haversine_distance_m(18.0, 73.0, 19.0, 73.0)
    assert abs(one_deg_m - 111190.0) / 111190.0 <= 0.005

    # Empty list and single point
    assert cluster_detections_dbscan([]) == []
    single = cluster_detections_dbscan([{"id": 1, "lat": 18.5204, "lng": 73.8567}])
    assert len(single) == 1
    assert single[0]["cluster_idx"] == 0

    # Two points ~10 m apart (~0.00009 deg lat) merge; third point ~100 m away (~0.0009 deg lat) separates
    pts = [
        {"id": 1, "lat": 18.52040, "lng": 73.85670},
        {"id": 2, "lat": 18.52049, "lng": 73.85670},  # ~10 m north
        {"id": 3, "lat": 18.52140, "lng": 73.85670},  # ~111 m north
    ]
    clustered = cluster_detections_dbscan(pts, eps_meters=25.0)
    assert clustered[0]["cluster_idx"] == clustered[1]["cluster_idx"]
    assert clustered[2]["cluster_idx"] != clustered[0]["cluster_idx"]


def test_osm_lookup_with_mocks_and_offline_fallback() -> None:
    """M5 / U07: OSM lookup uses mock getter (no network), maps highway tags, and defaults to 50 offline."""

    def _mock_primary_post(_url: str, _data: dict[str, str], _timeout: float) -> dict[str, object]:
        return {"elements": [{"tags": {"highway": "primary"}}]}

    res_primary = lookup_road_importance(
        18.5204, 73.8567, http_post=_mock_primary_post, use_cache=False
    )
    assert res_primary["road_class"] == "primary"
    assert res_primary["road_importance"] == 85

    def _mock_offline_post(_url: str, _data: dict[str, str], _timeout: float) -> dict[str, object]:
        raise ConnectionError("Simulated offline network")

    res_offline = lookup_road_importance(
        19.0760, 72.8777, http_post=_mock_offline_post, use_cache=False
    )
    assert res_offline["road_class"] == "unknown"
    assert res_offline["road_importance"] == 50
