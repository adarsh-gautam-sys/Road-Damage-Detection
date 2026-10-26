# Changelog

- **M5 (2026-10-27):** Add EXIF GPS parser & stripper (`U04` + Hypothesis random-bytes fuzz test), GPX/CSV track interpolation (`U05`), Haversine DBSCAN clustering (`U06`), mocked OSM Overpass road importance lookup, and fix RDD2022 download via official Figshare byte-range slicing.
- **M4 (2026-10-27):** Add rule-based severity assessment module (`src/rdai/severity/rules.py`, `U03` + Hypothesis property tests), YOLO11 detector & ByteTrack wrappers (`src/rdai/inference/detector.py`, `tracker.py`, `U08`), and 100-sample blind severity audit script (`scripts/audit_severity.py`).
- **M3 (2026-10-27):** Add validation F1 threshold calibration, test-split evaluator, ONNX parity verifier (`src/rdai/train/evaluate.py`), automated results generator (`scripts/make_results.py`), and model tests `M02`–`M05`, `M07`.
- **M2 (2026-10-27):** Add YOLO11 training pipeline (`src/rdai/train/train.py`) supporting experiments `E1`–`E7`, Google AI Pro GPU auto-scaling (`A100`/`L4`), checkpoint resume, `M01` class verification test, and `notebooks/02_train_colab.ipynb`.
- **M1 (2026-10-27):** Add RDD2022 download helper, Pascal VOC to YOLO converter (`U01`), deterministic 50-block splitter with background cap and D10 oversampling (`U02`), EDA figure generator, and `notebooks/01_eda.ipynb`.
- **M0 (2026-10-27):** Initialize repository scaffold, pinned dependencies, YAML configs, `Makefile`, `ruff`, `pytest` markers, CI workflow, `NOTICE.md`, and synthetic fixture generator.
