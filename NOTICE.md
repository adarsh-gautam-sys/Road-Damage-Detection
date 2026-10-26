# NOTICE — Third-Party Attributions & Reused Ideas

Per `docs/PRD.md` Section 2, this file records every open-source repository, dataset, and library whose code patterns or architectural ideas informed this project.

## 1. Permissively Licensed Repositories (MIT)
- **`uzbtrust/smart-road`** (https://github.com/uzbtrust/smart-road)
  - **License:** Code under MIT License; model weights under AGPL-3.0.
  - **What was adapted:** Test suite organization (`unit`, `integration`, `api`, `model`), CI workflow pattern, and the training patience lesson (`patience=50` instead of `30` to avoid premature early stopping).

## 2. Conceptual Ideas Reimplemented from Scratch
The following repositories either lacked an explicit root `LICENSE` file or used CC BY-NC-SA 4.0. No source code was copied from them; only high-level domain ideas were reimplemented cleanly from scratch:
- **`vrishikumar0512/ROADGAURD-AI`** — Inspired the 4-term `0–100` maintenance priority score structure, 5-state repair status workflow (`new`, `verified`, `scheduled`, `fixed`, `rejected`), and strict upload content validation.
- **`saisanthosh-hub/RoadGuard-AI`** — Inspired the modular FastAPI router/service layout, Haversine spatial clustering concept, ByteTrack video deduplication, and 3-notebook workflow (`01_eda`, `02_train_colab`, `03_results`).
- **`MMujtabaX/RoadGuard-AI-Pothole-Detection`** — Provided the starting pothole (`D40`) area-ratio thresholds (`0.015` Low, `0.05` High) in `configs/severity.yaml`.
- **`smlab-niser/iWatchRoad`** (CC BY-NC-SA 4.0) — Inspired the dashcam GPX/CSV track interpolation workflow and mobile-first React + Vite + Leaflet architecture (reimplemented independently).
- **`oracl4/RoadDamageDetection`** — Inspired carving deterministic validation and test splits from the labeled RDD2022 training images (since official test split labels are withheld) and capping empty background images.
- **`vishal122006/roadguard-ai`** — Informed the training/evaluation CLI script and metric reporting layout.

## 3. Datasets & Map Data
- **RDD2022 (Road Damage Dataset 2022):** Arya et al., *"RDD2022: A multi-national image dataset for automatic Road Damage Detection"*, IEEE BigData 2022 (`sekilab/RoadDamageDetector`).
- **OpenStreetMap:** Road importance tags (`highway=*`) and map tiles © OpenStreetMap contributors (ODbL).

## 4. Core Deep Learning Framework
- **Ultralytics YOLO11:** Licensed under GNU Affero General Public License v3.0 (AGPL-3.0). In compliance with AGPL-3.0, this project repository is maintained publicly with full source code accessible to evaluators.
