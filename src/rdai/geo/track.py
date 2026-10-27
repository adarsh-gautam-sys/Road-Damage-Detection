"""GPX and CSV (`t,lat,lng`) GPS track parsing and linear frame-time interpolation (`FR-G2`, `U05`)."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path

import gpxpy


@dataclass(frozen=True)
class TrackPoint:
    """Single GPS track point at relative `t_s` seconds from track start."""

    t_s: float
    lat: float
    lng: float


def parse_csv_track(csv_source: Path | str | bytes) -> list[TrackPoint]:
    """Parse a CSV GPS track with columns `t,lat,lng` (`t` in seconds from start)."""
    if isinstance(csv_source, Path):
        text = csv_source.read_text(encoding="utf-8")
    elif isinstance(csv_source, bytes):
        text = csv_source.decode("utf-8")
    elif "\n" not in str(csv_source) and Path(csv_source).is_file():
        text = Path(csv_source).read_text(encoding="utf-8")
    else:
        text = str(csv_source)

    reader = csv.DictReader(io.StringIO(text.strip()))
    points: list[TrackPoint] = []
    for row in reader:
        t_val = float(row["t"])
        lat_val = float(row["lat"])
        lng_val = float(row["lng"])
        points.append(TrackPoint(t_s=t_val, lat=lat_val, lng=lng_val))

    points.sort(key=lambda p: p.t_s)
    return points


def parse_gpx_track(gpx_source: Path | str | bytes) -> list[TrackPoint]:
    """Parse a GPX track file into relative seconds from the first track point."""
    if isinstance(gpx_source, Path):
        xml_text = gpx_source.read_text(encoding="utf-8")
    elif isinstance(gpx_source, bytes):
        xml_text = gpx_source.decode("utf-8")
    elif "\n" not in str(gpx_source) and "<" not in str(gpx_source) and Path(gpx_source).is_file():
        xml_text = Path(gpx_source).read_text(encoding="utf-8")
    else:
        xml_text = str(gpx_source)

    gpx = gpxpy.parse(xml_text)
    raw_pts: list[tuple[float, float, float]] = []

    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                if pt.time is None:
                    continue
                raw_pts.append((pt.time.timestamp(), float(pt.latitude), float(pt.longitude)))

    if not raw_pts:
        return []

    raw_pts.sort(key=lambda item: item[0])
    t0 = raw_pts[0][0]
    return [TrackPoint(t_s=round(ts - t0, 4), lat=lat, lng=lng) for ts, lat, lng in raw_pts]


def parse_track_auto(track_source: Path | str | bytes, filename: str = "") -> list[TrackPoint]:
    """Auto-detect GPX vs CSV track format and parse into `TrackPoint` list."""
    lower_name = filename.lower()
    if isinstance(track_source, Path):
        lower_name = track_source.name.lower()

    if lower_name.endswith(".gpx"):
        return parse_gpx_track(track_source)
    if lower_name.endswith(".csv"):
        return parse_csv_track(track_source)

    raw = (
        track_source.read_text(encoding="utf-8")
        if isinstance(track_source, Path)
        else (
            track_source.decode("utf-8", errors="ignore")
            if isinstance(track_source, bytes)
            else str(track_source)
        )
    )
    if "<gpx" in raw[:300].lower():
        return parse_gpx_track(raw)
    return parse_csv_track(raw)


def interpolate_track_position(
    points: list[TrackPoint],
    timestamp_s: float,
    offset_s: float = 0.0,
    tolerance_s: float = 5.0,
) -> tuple[float, float] | None:
    """Linearly interpolate `(lat, lng)` at `timestamp_s + offset_s`.

    If query time is more than `tolerance_s` (default 5.0 s) before the first point or
    after the last point, returns `None` (`U05`).
    """
    if not points:
        return None

    sorted_pts = sorted(points, key=lambda p: p.t_s)
    target_t = float(timestamp_s) + float(offset_s)

    first = sorted_pts[0]
    last = sorted_pts[-1]

    if target_t < first.t_s - tolerance_s or target_t > last.t_s + tolerance_s:
        return None

    if target_t <= first.t_s:
        return (round(first.lat, 6), round(first.lng, 6))
    if target_t >= last.t_s:
        return (round(last.lat, 6), round(last.lng, 6))

    for i in range(len(sorted_pts) - 1):
        p0 = sorted_pts[i]
        p1 = sorted_pts[i + 1]
        if p0.t_s <= target_t <= p1.t_s:
            span = p1.t_s - p0.t_s
            if span <= 0.0:
                return (round(p0.lat, 6), round(p0.lng, 6))
            alpha = (target_t - p0.t_s) / span
            lat = p0.lat + alpha * (p1.lat - p0.lat)
            lng = p0.lng + alpha * (p1.lng - p0.lng)
            return (round(lat, 6), round(lng, 6))

    return None
