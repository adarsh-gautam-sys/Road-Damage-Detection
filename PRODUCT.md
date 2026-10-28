# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

React 18 + TypeScript + Vite + Tailwind CSS + `react-leaflet` (OpenStreetMap tiles) + installable PWA manifest, communicating with the local FastAPI `/api/v1` backend (`SQLite` + `SQLAlchemy 2.0` + `Ultralytics YOLO11s`).

## Users

1. **Field Road Inspectors & Survey Crews (Mobile-First):** Operating outdoors on smartphones (`360 px` viewport and up) while walking or driving Indian road corridors. Their job is to capture geotagged road defects via single-photo snap, dashcam video upload, or the **Live Camera Patrol HUD** (real-time bounding-box detection + live GPS tracking + manual shutter or automatic patrol logging), verify location coordinates, and submit inspections into the segment database.
2. **Municipal Civil Engineers & Maintenance Planners (Mobile & Desktop):** Reviewing aggregated 50-meter road segments on the interactive map and ranked repair queue, inspecting the four-term priority breakdown ($S_{\max}, D, R, C$), updating work-order statuses (`open` → `scheduled` → `repaired`), and exporting CSV/GeoJSON maintenance schedules.
3. **Academic & Deep Learning Lab Evaluators:** Auditing the end-to-end pipeline, verifying real `RDD2022` India test-split metrics (`reports/metrics.json`), inspecting the geometric severity proxy and priority formula, and testing live camera / upload inference during project demonstration.

## Product Purpose

**Road Damage AI (`rdai`)** automates the detection, geometric severity grading, spatial clustering, and maintenance prioritization of Indian road surface defects (`D00` Longitudinal Crack, `D10` Transverse Crack, `D20` Alligator Crack, `D40` Pothole). Success means a field inspector or evaluator can point a phone camera at a damaged road (or upload a photo/video), immediately see calibrated bounding boxes and continuous `0–100` severity scores, and watch those detections cluster into prioritized 50-meter road segments on an interactive map with transparent, auditable repair rankings.

## Positioning

Unlike generic object-detection demos that stop at drawing bounding boxes on static images, **Road Damage AI** connects calibrated `YOLO11s` detection (`optimal_conf = 0.193`) and `ByteTrack` video deduplication to a transparent, zero-black-box civil engineering triage pipeline:
1. **Class-Aware Geometric Severity (`0–100`):** Uses bounding-box area ratio for surface defects (`D40` potholes, `D20` alligator cracks) and diagonal ratio for linear cracks (`D00`, `D10`), validated against a 100-crop blind human audit.
2. **50-Meter Haversine `DBSCAN` Spatial Clustering:** Groups nearby geo-tagged detections into actionable 50-meter road segments (`eps = 50m`, `min_samples = 1`).
3. **OpenStreetMap Road-Class Priority Formula (`0–100`):** Combines worst severity ($0.45 \cdot S_{\max}$), defect density ($0.25 \cdot D$), OpenStreetMap highway importance ($0.20 \cdot R$), and model confidence ($0.10 \cdot C$) into an explainable maintenance queue.
4. **Live Camera Patrol HUD:** Streams live camera frames to the detector with real-time bounding-box overlays, continuous `navigator.geolocation.watchPosition` GPS lock, and both one-tap manual capture and distance/time-gated Auto-Patrol logging.

## Operating Context

- **Field Capture Scene:** Bright outdoor daylight, variable mobile connectivity, one-handed phone operation (`360×640` to `412×915`), or dashboard-mounted phone during road patrol.
- **Triage & Evaluation Scene:** Desktop or tablet browser (`768×1024` to `1280×800+`) alongside a smartphone during live academic defense and municipal triage workflows.
- **Backend API Surface (`/api/v1`):**
  - `GET /api/v1/health` — Model readiness, run name (`e1_yolo11s_640`), and default confidence (`0.193`).
  - `POST /api/v1/detections/photo` — Multipart photo upload + explicit `lat`/`lng` + `conf_threshold`, returning annotated image (`<= 1280 px`), `320 px` thumbnail, per-box severity scores, and updated `segment_id`.
  - `POST /api/v1/detections/video` & `GET /api/v1/jobs/{id}` — Asynchronous dashcam video + GPX/CSV track processing with ByteTrack deduplication and progress polling.
  - `GET /api/v1/detections` & `GET /api/v1/detections/{id}` — Paginated detection feed (`limit`, `offset`, `class`, `severity`, `bbox`, `since`).
  - `GET /api/v1/segments`, `GET /api/v1/segments/{id}`, `PATCH /api/v1/segments/{id}` — Ranked 50m road segments with 4-term priority breakdown and repair status transitions (`open`, `scheduled`, `repaired`).
  - `POST /api/v1/segments/recluster` — Full Haversine DBSCAN reclustering and priority recomputation.
  - `GET /api/v1/exports/segments.csv` & `GET /api/v1/exports/segments.geojson` — Direct tabular and RFC 7946 `[lng, lat]` GeoJSON exports.

## Capabilities and Constraints

- **Six Core Views + Live Camera Patrol Mode:**
  1. **Capture & Live Patrol:** Rear-camera photo snap (`capture="environment"`), client-side `1280 px` resize before upload, explicit browser GPS (`lat, lng`) with draggable pin fallback, confidence slider (default `0.193`), dashcam video + GPX/CSV upload with background job polling, and **Live Camera HUD** (WebRTC `<video>` + `<canvas>` bounding-box overlay + live GPS + Manual Shutter & Auto-Patrol logging).
  2. **Detection Result View:** Annotated image, detection cards with class code/name, confidence, geometric extent (`area_ratio` or `diag_ratio`), continuous severity score (`0–100`), severity chip (`Low / Medium / High`), and direct link to the parent 50m road segment.
  3. **Interactive Map (`react-leaflet`):** Clustered markers and 50m road segments colored by severity/priority, live filters (`class`, `severity`, `status`, `since`), and a mobile bottom sheet for segment/detection details.
  4. **Ranked Repair List:** Segment cards ordered by Priority Score $P$, exposing the exact four weighted contributions ($0.45 S_{\max}$, $0.25 D$, $0.20 R$, $0.10 C$), OSM road class, and CSV/GeoJSON export triggers.
  5. **Segment Detail & Work-Order Control:** Thumbnail/photo gallery, defect history, four-term priority breakdown, and status control (`open`, `scheduled`, `repaired`).
  6. **About & Model Card:** Displays only real metrics from `reports/metrics.json`, methodology formulas, known limitations (2D perspective proxy, `D10` scarcity, daytime plains domain), and `RDD2022` / OpenStreetMap attributions.
- **Hard Rules (`AGENTS.md` & `PRD.md`):**
  - **Never invent metrics:** Every evaluation number displayed in the UI must come strictly from `reports/metrics.json` (`E1` test split: `mAP@50 = 0.3655`, `mAP@50:95 = 0.1728`, `D20 AP@50 = 0.5885`, `D40 AP@50 = 0.4105`, `D00 AP@50 = 0.2409`, `D10 AP@50 = 0.2221`, `optimal_conf = 0.193`, `gpu_ms = 1.28`, `cpu_ms = 151.71`).
  - **GeoJSON Coordinate Order:** Always `[longitude, latitude]` per RFC 7946; UI pins and API query params use explicit `lat`, `lng`.
  - **Mobile Bandwidth & EXIF Preservation:** Client-side image resizing to `1280 px` strips EXIF, so the browser must capture GPS coordinates *before* resizing and send `lat` and `lng` explicitly in the multipart form.

## Evidence on Hand

- **Trained Model Checkpoints:** `runs/e1_yolo11s_640/weights/best.pt` and `runs/e1_yolo11s_640/weights/best.onnx` (trained 150 epochs on Google AI Pro NVIDIA A100-SXM4-80GB).
- **Verified Evaluation Metrics:** `reports/metrics.json`, `reports/tables/main_result.csv`, `reports/tables/ablations.csv`, `reports/tables/top_segments.csv`, `reports/tables/weight_sensitivity.json`, `reports/tables/severity_audit_sheet.csv`.
- **Generated Figures:** `reports/figures/confusion_matrix_normalized.png`, `reports/figures/PR_curve.png`, `reports/figures/F1_curve.png`, `reports/figures/training_curves_e1.png`, `reports/figures/results_per_class_ap.png`, `reports/figures/severity_priority_curves.png`, `reports/figures/val_batch0_pred.jpg`.
- **OpenAPI Contract Snapshot:** `tests/fixtures/openapi_snapshot.json`.
- **Absences Not to Fabricate:** No fake municipal logos, no invented testimonials, no fabricated accuracy numbers, and no unverified sensor claims (e.g., no laser depth/LiDAR claims—severity is strictly a 2D bounding-box geometric proxy).

## Product Principles

1. **Field-First Ergonomics Under Glare & Motion:** Primary capture, live patrol controls, and map filters must be operable one-handed on a `360 px` phone screen without horizontal scrolling or hover-only interactions.
2. **Explainable Civil Engineering Math Over Black-Box Scores:** Every severity badge (`Low / Medium / High`) and segment priority score (`0–100`) must show the underlying geometric extent and four-term formula breakdown ($S_{\max}, D, R, C$) so engineers trust the ranking.
3. **Zero Fabricated Metrics:** Every performance statistic and chart in the application is bound directly to `reports/metrics.json` and real backend state.
4. **Resilient Offline & Low-Bandwidth Behavior:** Lists load `320 px` thumbnails first, photos resize to `1280 px` before upload, background video jobs survive tab switches via `/api/v1/jobs/{id}` polling, and missing GPS gracefully marks records as `unlocated` rather than failing.

## Accessibility & Inclusion

- **Touch Targets:** Minimum `44 × 44 px` for all interactive buttons, chips, shutter controls, and map sheet triggers.
- **Redundant Severity Encoding:** Severity and priority levels (`High`, `Medium`, `Low`) must always be communicated via **label + icon/shape + color**, never color alone.
- **Contrast:** Text and HUD overlays must maintain at least `4.5:1` contrast ratio (and remain legible over live camera video feeds and map tiles).
- **Responsive Viewport Range:** Zero horizontal overflow from `320 px` (`360×640` primary mobile baseline) through tablet (`768×1024`) and desktop (`1280×800+`).
