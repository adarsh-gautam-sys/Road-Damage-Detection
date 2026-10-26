# AI-Based Indian Road Damage Detection, Severity Assessment & Geo-Mapped Maintenance Prioritization

End-to-end Deep Learning Lab system that turns Indian road photos or dashcam footage into a ranked maintenance repair list:
1. **Detect** 4 road damage classes (`D00` Longitudinal Crack, `D10` Transverse Crack, `D20` Alligator Crack, `D40` Pothole) using **YOLO11s** (and ByteTrack for video deduplication).
2. **Grade Severity** (`0–100` continuous score and `Low` / `Medium` / `High` tier) from configurable class-specific geometric rules (`configs/severity.yaml`).
3. **Geo-Map & Cluster** detections via photo EXIF, explicit mobile coordinates, or GPX/CSV tracks into `25 m` road segments using Haversine DBSCAN.
4. **Prioritize Repairs** (`0–100` score per segment) combining worst severity ($S_{\max}$), cluster density ($D$), OpenStreetMap road importance ($R$), and model confidence ($C$).

See [docs/PRD.md](docs/PRD.md) for the complete specification, test plan, and rubric mapping.

## Quickstart

```bash
# 1. Create virtualenv, install pinned dependencies, and build test fixtures
make setup

# 2. Run fast unit, integration, API, and security test suites
make test

# 3. Run ruff linter and formatter checks
make lint
```

## Make Targets

| Command | Description |
| :--- | :--- |
| `make setup` | Initialize `.venv`, install dependencies, and generate synthetic test fixtures |
| `make data` | Convert RDD2022 India Pascal VOC XMLs to YOLO format and run deterministic 50-block split |
| `make train` | Train YOLO11s (`E1` recipe) |
| `make eval` | Evaluate trained checkpoint on India test split and build `reports/metrics.json` |
| `make test` | Run fast test suite (`unit`, `integration`, `api`, `security`) |
| `make test-model` | Run real-weights model and ONNX parity tests (`slow`, `gpu`) |
| `make test-perf` | Run latency and memory benchmarks (`perf`) |
| `make lint` | Run `ruff check` and `ruff format --check` |
| `make cov` | Run fast tests enforcing $\ge 80\%$ line coverage on `src/rdai` |
| `make map` | Generate standalone mobile-friendly Folium map (`reports/map.html`) |
| `make run` | Start the FastAPI server on `http://localhost:8000` |

## License & Attributions
- Source code and YOLO11 integration follow **AGPL-3.0** (`LICENSE`).
- Reused architectural ideas and MIT test patterns are documented in [NOTICE.md](NOTICE.md).
- Map data © OpenStreetMap contributors.
