"""Folium interactive map builder for road damage detections and prioritized segments (`PRD Section 10 & 13`)."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import folium

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MAP_HTML = ROOT_DIR / "reports" / "map.html"

SEVERITY_COLORS: dict[str, str] = {
    "High": "#dc2626",
    "Medium": "#d97706",
    "Low": "#16a34a",
}


def _resolve_thumbnail_html(det: dict[str, Any], media_root: Path | None = None) -> str:
    """Build an `<img>` tag for a detection popup, embedding base64 if local file exists."""
    thumb_url = str(det.get("thumb_url") or "")
    if not thumb_url and det.get("image_id") is not None:
        thumb_url = f"/media/thumbs/{int(det['image_id'])}.jpg"

    if media_root is not None and thumb_url.startswith("/media/"):
        rel = thumb_url.removeprefix("/media/").lstrip("/")
        local_path = (media_root / rel).resolve()
        if local_path.is_file():
            b64 = base64.b64encode(local_path.read_bytes()).decode("ascii")
            return f'<img src="data:image/jpeg;base64,{b64}" style="width:180px;border-radius:6px;margin-top:6px;" alt="Thumbnail"/>'

    if thumb_url:
        return f'<img src="{thumb_url}" style="width:180px;border-radius:6px;margin-top:6px;" alt="Thumbnail"/>'
    return ""


def build_folium_map(
    detections: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    output_html_path: Path = DEFAULT_MAP_HTML,
    media_root: Path | None = None,
) -> Path:
    """Build a mobile-friendly Folium HTML map with severity-colored markers and priority-sized segment circles."""
    located_dets = [d for d in detections if d.get("lat") is not None and d.get("lng") is not None]
    located_segs = [
        s
        for s in segments
        if s.get("centroid_lat") is not None and s.get("centroid_lng") is not None
    ]

    if located_segs:
        center_lat = sum(float(s["centroid_lat"]) for s in located_segs) / len(located_segs)
        center_lng = sum(float(s["centroid_lng"]) for s in located_segs) / len(located_segs)
        zoom = 13
    elif located_dets:
        center_lat = sum(float(d["lat"]) for d in located_dets) / len(located_dets)
        center_lng = sum(float(d["lng"]) for d in located_dets) / len(located_dets)
        zoom = 13
    else:
        # Default center: India (Nagpur / central India)
        center_lat, center_lng, zoom = 20.5937, 78.9629, 5

    fmap = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=zoom,
        tiles="OpenStreetMap",
        attr="© OpenStreetMap contributors",
        control_scale=True,
    )

    seg_group = folium.FeatureGroup(name="Road Segments (Sized by Priority)", show=True).add_to(
        fmap
    )
    det_group = folium.FeatureGroup(
        name="Damage Detections (Coloured by Severity)", show=True
    ).add_to(fmap)

    for seg in located_segs:
        s_lat = float(seg["centroid_lat"])
        s_lng = float(seg["centroid_lng"])
        p_score = float(seg.get("priority_score", 50.0))
        p_label = str(seg.get("priority_label", "Medium"))
        color = SEVERITY_COLORS.get(p_label, "#2563eb")
        radius = max(10.0, min(28.0, 8.0 + 0.18 * p_score))
        terms = seg.get("terms", {}) or {}

        popup_html = (
            f"<div style='font-family:sans-serif;font-size:13px;min-width:210px;'>"
            f"<b>Segment #{seg.get('id')} — {p_label} Priority ({p_score:.2f})</b><br/>"
            f"Road Class: <code>{seg.get('road_class', 'unknown')}</code><br/>"
            f"Status: <b>{seg.get('status', 'new')}</b> | Detections: <b>{seg.get('n_detections', 1)}</b><br/>"
            f"<hr style='margin:5px 0;'/>"
            f"<small>S_max: {float(terms.get('s_max', seg.get('max_severity', 0.0))):.1f} | "
            f"D: {float(terms.get('density', 10.0)):.1f} | "
            f"R: {float(terms.get('road_importance', 50.0)):.0f} | "
            f"C: {float(terms.get('confidence', 80.0)):.1f}</small>"
            f"</div>"
        )
        folium.CircleMarker(
            location=[s_lat, s_lng],
            radius=radius,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.28,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"Segment #{seg.get('id')}: Priority {p_score:.2f} ({p_label})",
        ).add_to(seg_group)

    for det in located_dets:
        d_lat = float(det["lat"])
        d_lng = float(det["lng"])
        sev_obj = det.get("severity") if isinstance(det.get("severity"), dict) else {}
        sev_score = float(det.get("severity_score", sev_obj.get("score", 50.0)))
        sev_label = str(det.get("severity_label", sev_obj.get("label", "Medium")))
        color = SEVERITY_COLORS.get(sev_label, "#d97706")
        cls_code = str(det.get("class_code", "D40"))
        cls_name = str(det.get("class_name", "pothole"))
        conf = float(det.get("confidence", 0.0))
        thumb_tag = _resolve_thumbnail_html(det, media_root=media_root)

        popup_html = (
            f"<div style='font-family:sans-serif;font-size:13px;min-width:190px;'>"
            f"<b>{cls_code} ({cls_name})</b><br/>"
            f"Severity: <b style='color:{color};'>{sev_label} ({sev_score:.1f})</b><br/>"
            f"Confidence: {conf:.2f} | Segment: #{det.get('segment_id', 'N/A')}<br/>"
            f"{thumb_tag}"
            f"</div>"
        )
        folium.CircleMarker(
            location=[d_lat, d_lng],
            radius=6.0,
            color="#111827",
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=f"{cls_code} — {sev_label} ({sev_score:.1f})",
        ).add_to(det_group)

    folium.LayerControl(collapsed=False).add_to(fmap)

    legend_html = """
    <div style="position:fixed;bottom:24px;left:16px;z-index:9999;background:white;
                padding:10px 14px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.2);
                font-family:sans-serif;font-size:12px;line-height:1.5;">
      <b>Road Damage AI — Maintenance Priority</b><br/>
      <span style="color:#dc2626;">&#9679;</span> High (71–100) &nbsp;
      <span style="color:#d97706;">&#9679;</span> Medium (31–70) &nbsp;
      <span style="color:#16a34a;">&#9679;</span> Low (0–30)<br/>
      <small>Map data &copy; OpenStreetMap contributors</small>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output_html_path))
    return output_html_path
