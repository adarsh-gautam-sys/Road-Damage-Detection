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

## 4. Quantitative Metrics (`reports/metrics.json` — Experiment `E1`)
Evaluated on the held-out `RDD2022` India test split (`1,156` images across 50-file contiguous blocks, `optimal_conf = 0.193`, validation `F1 = 0.3401`, A100 GPU latency `1.28 ms/image`):

| Class | Description | Precision | Recall | AP@50 | AP@50:95 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `D00` | Longitudinal crack | `0.3223` | `0.2295` | `0.2409` | `0.1243` |
| `D10` | Transverse crack | `0.4316` | `0.1538` | `0.2221` | `0.1046` |
| `D20` | Alligator crack | `0.5443` | `0.6073` | `0.5885` | `0.2921` |
| `D40` | Pothole | `0.4964` | `0.3963` | `0.4105` | `0.1701` |
| **all** | **4-class macro average** | **`0.4487`** | **`0.3467`** | **`0.3655`** | **`0.1728`** |

### Discussion of Test Split Results
- **Strong Surface Texture Detection (`D20` & `D40`):** Alligator cracking (`D20`) achieves **`0.5885` AP@50** (`0.6073` recall) and potholes (`D40`) achieve **`0.4105` AP@50** (`0.4964` precision), demonstrating reliable localization of structural surface failure.
- **Impact of Non-Leaking 50-Block Split:** Unlike random image splits (where adjacent dashcam frames of the exact same road segment appear in both train and test sets, inflating `mAP@50`), our 50-block split enforces strict corridor separation.
- **Extreme `D10` Scarcity & Thin Cracks at `640 px`:** Only `68` transverse crack (`D10`) annotations exist in the India training split (and only `9` ground-truth instances in the test split), yielding `0.2221` AP@50 for `D10` and `0.2409` AP@50 for thin longitudinal cracks (`D00`), which lowers the unweighted 4-class macro `mAP@50` to `0.3655`.

## 5. Known Limitations & Risks
1. **Severe `D10` Class Imbalance:** Only `68` transverse crack (`D10`) boxes exist in the RDD2022 India training set, making `D10` AP noisy; mitigated via training repetition and multi-country (`Japan + Czech`) transfer (`Experiment E5`).
2. **2D Perspective Proxy for Severity:** Bounding-box area/diagonal ratio conflates physical damage size with camera distance. A small pothole close to the lens subtends a larger box than a deep pothole far ahead.
3. **Geographic & Weather Domain Shift:** RDD2022 India images were captured in daytime plains conditions with one car-mounted smartphone geometry.
