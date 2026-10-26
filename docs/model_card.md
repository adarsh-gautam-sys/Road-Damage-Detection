# Model Card — Road Damage AI (YOLO11s on RDD2022 India)

## 1. Model Details
- **Architecture:** Ultralytics `YOLO11s` (single-stage detector with C3k2 backbone blocks, SPPF, C2PSA spatial attention neck, and multi-scale detection heads).
- **Classes:**
  - `0: D00` — Longitudinal crack
  - `1: D10` — Transverse crack
  - `2: D20` — Alligator crack
  - `3: D40` — Pothole
- **Compute Environment:** Trained on Google Colab with **Google AI Pro** (NVIDIA A100 / L4 GPUs) from COCO pretrained weights (`yolo11s.pt`) at `640 px` with `seed=0` and `deterministic=True`.
- **License:** AGPL-3.0 (per Ultralytics licensing terms; repository source is public).

## 2. Intended Use
- **Primary Use:** Automated detection, severity triage, and segment-level maintenance prioritization from smartphone or dashcam images of Indian roads.
- **Out-of-Scope Use:** Structural pavement engineering certification, depth/volume estimation for billing without human verification, or safety-critical autonomous vehicle braking.

## 3. Dataset (`RDD2022` India Subset)
- **Source:** Arya et al. (2022), *RDD2022: A multi-national image dataset for automatic Road Damage Detection* (`sekilab/RoadDamageDetector`).
- **Composition:** `7,706` labeled train images (`720 × 720 px`) captured across Delhi, Gurugram, and Haryana.
- **Split Strategy:** Deterministic `70% / 15% / 15%` (`train / val / test`) split in contiguous blocks of 50 file numbers to prevent sequential frame leakage, with empty background images in `train` capped at `10%`.

## 4. Quantitative Metrics
All metrics are populated exclusively from `reports/metrics.json` by `scripts/make_results.py` after Milestone M3 evaluation.

## 5. Known Limitations & Risks
1. **Severe `D10` Class Imbalance:** Only `68` transverse crack (`D10`) boxes exist in the RDD2022 India training set, making `D10` AP noisy; mitigated via training repetition and multi-country (`Japan + Czech`) transfer (`Experiment E5`).
2. **2D Perspective Proxy for Severity:** Bounding-box area/diagonal ratio conflates physical damage size with camera distance. A small pothole close to the lens subtends a larger box than a deep pothole far ahead.
3. **Geographic & Weather Domain Shift:** RDD2022 India images were captured in daytime plains conditions with one car-mounted smartphone geometry.
