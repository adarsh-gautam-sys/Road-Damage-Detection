# Surface Brief — Road Damage AI (`frontend/index.html`)

## Mode
`operate`

## Approved Comp & Direction
- **Approved Reference Comp:** `.impeccable/mocks/decision/model-pick.png` (`"approved": true` in `.impeccable/mocks/decision/model-pick.png.json`, refined per user direction: *"I prefer impeccable's pick from the 1st set of options image rendered by gemini. but I just want more- tasteful design of it"*).
- **Direction Seed Key:** `de9aa6c7` (`IMPECCABLE'S PICK` — *Refined Leica Optical Rangefinder & Field Controller*).
- **Sampled & Refined Palette Tokens:**
  - Chassis Ground (`--bg-chassis`): `#181B1F` (sampled interior patch from `model-pick.png` anthracite frame)
  - Elevated Instrument Panel (`--bg-panel`): `#21252B`
  - Hairline Bezel Rule (`--border-etch`): `#323842`
  - Transflective LCD Readout Well (`--bg-lcd`): `#C8CEC2` / `#D8DDD2` with deep liquid-crystal ink `#121512` (sampled from `model-pick.png` transflective LCD panel)
  - Primary Foreground (`--text-primary`): `#F3F4F6`
  - Secondary Tinted Foreground (`--text-secondary`): `#B8C0CC` (cool anodized tint, >= 4.5:1 on `#181B1F`)
  - Surveyor Gold Primary Action (`--accent-amber`): `#F2C200` with `#141412` ink (sampled from `model-pick.png` shutter plate, flattened into a precision anodized plate without 3D plastic bevels)
  - High Severity / Pothole (`--sev-high`): `#E02828`
  - Medium Severity / Alligator (`--sev-med`): `#E59400`
  - Low Severity / Linear Crack (`--sev-low`): `#2B8A6E`

## Implementation Fidelity Inventory
| Region / Element | Medium | Commitment |
| :--- | :--- | :--- |
| Instrument Header & Telemetry Bar | Semantic HTML/CSS | Flush anthracite header with compact tabular status pills (`GPS`, `YOLO11s E1`, `CONF 0.193`) — replaces the loud full-width yellow banner from the raw concept. |
| Optical Bright-Line Rangefinder HUD | `<video>` + `<canvas>` + SVG reticle | 1px etched corner brackets and center tick marks; live bounding boxes (`D40`, `D20`, `D00`, `D10`) with `+` corner fiducials; focus-dimming of non-selected boxes to 30% opacity. |
| Transflective LCD Data Register | Semantic HTML/CSS (`tabular-nums`) | Recessed 1px-bordered sage-bone readout (`#D8DDD2`) showing selected defect `area_ratio`/`diag_ratio`, continuous severity `S`, and 4-term segment priority math ($0.45 S_{\max} + 0.25 D + 0.20 R + 0.10 C = P$) with direct-manipulation `conf_threshold` slider (`0.193` default). |
| Primary Shutter / Log Trigger & Auto-Patrol | Semantic `<button>` + Switch | Matte surveyor-gold primary plate (`#F2C200`) + mechanical pill switch for `Auto-Patrol (>=10m / 5s)` + Photo/Video file upload triggers. |
| Interactive Geo-Priority Map | `react-leaflet` + OpenStreetMap | 50m Haversine `DBSCAN` segments & markers colored and shaped by severity/priority, filter bar, draggable location pin, and bottom sheet detail drawer. |
| Ranked Priority Queue & Segment Detail | Semantic Table / List + Detail View | Right-aligned tabular 4-term accounting, work-order status transitions (`open -> scheduled -> repaired`), and CSV/GeoJSON export triggers. |
| Model Card & Audit | Semantic Data View | Bound strictly to `/api/v1/metrics` (`reports/metrics.json`) and evaluation figures (`confusion_matrix_normalized.png`, `PR_curve.png`, `F1_curve.png`). |
