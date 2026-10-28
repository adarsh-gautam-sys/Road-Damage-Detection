---
name: Road Damage AI
description: Refined Leica Optical Total-Station & Rangefinder Field Controller
colors:
  primary: "#f2c200"
  primary-press: "#d9ad00"
  primary-ink: "#141412"
  neutral-bg: "#181b1f"
  surface-panel: "#21252b"
  surface-raised: "#2a2f37"
  border-etch: "#323842"
  border-etch-strong: "#454c58"
  text-ink: "#f3f4f6"
  text-ink-2: "#b8c0cc"
  text-ink-3: "#8b94a3"
  lcd-bg: "#d8ddd2"
  lcd-ink: "#121512"
  sev-high: "#e02828"
  sev-med: "#e59400"
  sev-low: "#23805f"
  sev-high-text: "#ff7a72"
  sev-med-text: "#f5b13d"
  sev-low-text: "#4fc3a0"
  cls-linear: "#38bdf8"
typography:
  display:
    fontFamily: "Barlow Semi Condensed, Barlow, system-ui, sans-serif"
    fontSize: "1.5rem"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "0.08em"
  headline:
    fontFamily: "Barlow Semi Condensed, Barlow, system-ui, sans-serif"
    fontSize: "1.125rem"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "0.04em"
  body:
    fontFamily: "Barlow, system-ui, sans-serif"
    fontSize: "0.9375rem"
    fontWeight: 400
    lineHeight: 1.45
    letterSpacing: "normal"
  label:
    fontFamily: "Barlow Semi Condensed, Barlow, system-ui, sans-serif"
    fontSize: "0.75rem"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "0.08em"
  mono:
    fontFamily: "JetBrains Mono, ui-monospace, monospace"
    fontSize: "0.8125rem"
    fontWeight: 400
    lineHeight: 1.35
    letterSpacing: "normal"
rounded:
  none: "0px"
  sm: "4px"
  md: "6px"
  full: "9999px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.primary-ink}"
    rounded: "{rounded.md}"
    padding: "12px 24px"
  button-primary-hover:
    backgroundColor: "{colors.primary-press}"
---

## Overview

Road Damage AI (`rdai`) is a field-inspection instrumentation interface modeled after precision optical surveying tools (Leica total-stations, industrial rangefinders, and electronic field controllers). It operates directly under high-glare Indian daylight, on bumpy transit mounts, and in maintenance depots.

The interface prioritizes instant legibility, direct manipulation, tactile mechanical feedback, and numerical accountability. Every score shown is backed by explicit ledger math ($P = w_1 S + w_2 D + w_3 R + w_4 C$); every benchmark statistic is fetched strictly from verified evaluation records (`reports/metrics.json`).

## Colors

The palette is strictly calibrated for high visual contrast ($\ge 4.5:1$), glare resistance, and purposeful signaling:

- **Chassis & Panels (`#181B1F`, `#21252B`, `#2A2F37`):** Matte anodized anthracite enclosure. Never pure black, absorbing harsh light and establishing tonal depth without glassmorphism or blur effects.
- **Etched Rules (`#323842`, `#454C58`):** Precision 1px engraved partition lines and instrument bezels.
- **Surveyor Gold (`#F2C200`):** The single accent plate reserved for primary confirmation actions, shutter triggers, and current selections.
- **Recessed Transflective LCD (`#D8DDD2` with `#121512` ink):** Dedicated high-contrast display register for live measurements, physical extents, and numerical tolerances.
- **Redundant Severity Signaling:**
  - High / Critical: `#E02828` fill with white ink, `#FF7A72` chassis text, paired with upward triangle (`▲`).
  - Medium: `#E59400` fill with dark ink, `#F5B13D` chassis text, paired with diamond (`◆`).
  - Low: `#23805F` fill with white ink, `#4FC3A0` chassis text, paired with circle (`●`).
  - Linear cracks (D00/D10): `#38BDF8` sky blue highlight.

## Typography

Three intentional typefaces with strict functional segregation:

1. **Barlow Semi Condensed:** All uppercase instrument labels, section headers, operational toggles, and metadata plates.
2. **Barlow (sans):** Body prose, instructions, and explanatory documentation.
3. **JetBrains Mono:** Solely reserved for measurements, coordinates, hardware timings, parameters, and formula ledger terms. Monospaced tabular numbers (`tnum`) ensure clean vertical alignment across updates.

## Layout

- **Mobile First (360px+ Baseline):** High thumb-reachability with full 44×44px minimum touch targets. Viewport-anchored bottom navigation tabs and full-bleed optical camera reticle HUD.
- **Desktop Command Center (1024px+):** Responsive dual-pane layout. Left instrument rail (Capture / Command / Model), central interactive OpenStreetMap geospatial priority map, and synchronized right-side ranked segment triage ledger.
- **Slide-over Inspection Drawer:** Mobile bottom-sheet and desktop side-drawer for deep dive into 50m road segment clusters, inspection crops, and status workflows.

## Elevation & Depth

No ambient gaussian drop-shadows or skeuomorphic bevels. Depth is conveyed strictly through:
- **Tonal Layering:** Chassis (`#181B1F`) $\rightarrow$ Panel (`#21252B`) $\rightarrow$ Raised card (`#2A2F37`).
- **Engraved Inset Shadows:** Subtle 1px inner rim (`0 1px 0 rgba(255, 255, 255, 0.06) inset`) simulating milled aluminum bezels.
- **Recessed LCD Inset:** 2px inward shadow with low saturation simulating an embedded segmented LCD window.

## Shapes

- Strict mechanical geometry with 4px to 6px radii simulating machined metal edges.
- Circular glyphs and camera shutter buttons (`rounded-full`).
- Rangefinder reticle with 4 corner bright-line brackets and center crosshair.
- Bounding boxes rendered in real-time with optical bright-line corners and severity tags.

## Components

- **Live Camera HUD (`LiveHud`):** Zero-latency canvas overlay, 4 FPS stateless preview loop, active focus reticle, direct-manipulation confidence slider (`0.10`–`0.90`), auto-patrol gate (5s interval, 10m displacement), and manual shutter trigger.
- **Transflective LCD Register (`LcdReading`):** High-contrast measurement readout displaying defect class code, English name, normalized extent, severity score, and detection confidence.
- **Priority Math Ledger (`LedgerTable` / `LedgerLine`):** Right-aligned 4-term breakdown detailing contribution of worst severity ($S$), defect density ($D$), road hierarchy ($R$), and model confidence ($C$).
- **Severity Badge (`SeverityChip`):** Multi-modal redundant representation combining geometric glyph shape, uppercase label, and numeric severity score.
- **Geospatial Map (`MapView`):** Leaflet/OSM vector map rendering 50m road segments colored by priority tier, interactive clustering, and user GPS tracking dot.
- **Model Card & Audit View (`AuditView`):** Interactive transparency report displaying held-out test split metrics, per-class AP@50 and AP@50:95, normalized confusion matrix, inference speed breakdown, and evaluation curve figures.

## Do's and Don'ts

### Do's
- Always pair severity colors with geometric glyph shapes (triangle, diamond, circle) and explicit textual words.
- Keep numbers strictly authentic; fetch every metric from `/api/v1/metrics` and thresholds from `/api/v1/config`.
- Maintain touch target areas $\ge 44 \times 44$ px on all actionable elements.
- Use tabular monospaced numbers for all updating measurements and coordinates.

### Don'ts
- Never use generic AI design clichés: floating rounded cards with glowing multi-colored gradients, purple neon glows, or frosted glassmorphism.
- Never invent metrics, dummy accuracy percentages, or hardcoded threshold constants in the client code.
- Never use colored border-left/border-right accents $> 1$ px.
- Never display kickers or uppercase eyebrows directly above headings.
