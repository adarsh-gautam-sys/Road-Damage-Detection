---
name: rdai-ultralytics
description: YOLO11 training, Google AI Pro Colab GPU configuration, and evaluation conventions from PRD Section 8.
---

# Road Damage AI Ultralytics Skill (`rdai-ultralytics`)

Follow `docs/PRD.md` Section 8 for all YOLO11 training and evaluation tasks:

1. **Primary Recipe (`E1`):**
   - Model: `yolo11s.pt` fine-tuned on `configs/data.yaml` (`0: D00`, `1: D10`, `2: D20`, `3: D40`).
   - Hyperparameters: `epochs=150`, `patience=50`, `imgsz=640`, `seed=0`, `deterministic=True`, `flipud=0.0`.
   - **Google AI Pro Colab GPUs (A100 / L4 / T4):** Auto-scale batch size (`batch=32` or `64` with `cache="ram"` on A100/L4 High-RAM; `batch=16` on T4) and save checkpoints to Google Drive so runs can resume cleanly.
2. **Fair Comparison Rules:**
   - Every ablation (`E1`–`E7`) uses the exact same deterministic train/val/test split.
   - Pick the default confidence threshold from the validation split F1 curve and store it in `configs/app.yaml`.
   - Touch the test split only once per final model.
3. **Architecture Reporting:**
   - Use `model.info(verbose=True)` to output the real layer and parameter table for slides.
