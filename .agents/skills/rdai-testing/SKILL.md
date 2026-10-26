---
name: rdai-testing
description: How to write and run the automated tests in PRD Section 12 for Road Damage AI.
---

# Road Damage AI Testing Skill (`rdai-testing`)

Follow `docs/PRD.md` Section 12 for every test suite:

1. **Test-Driven Workflow:** Write the milestone's tests before the implementation code. Never mark a milestone done while any test fails or coverage drops below 80%.
2. **Test Layers & Commands:**
   - Fast tests (`tests/unit`, `tests/integration`, `tests/api`, `tests/security`): stub the YOLO11 model so `make test` finishes in seconds.
   - Real model tests (`tests/model`, markers `slow`, `gpu`): run via `make test-model`.
   - Performance tests (`tests/perf`, marker `perf`): run via `make test-perf`.
3. **Fixtures (`scripts/make_fixtures.py`):**
   - Generate lightweight synthetic fixtures (5 test images, 1 JPEG with `piexif` GPS, 1 GPX + 1 CSV track, 20 non-road images, 1 short synthetic video) without committing large binaries to Git.
4. **Property-Based Testing (`hypothesis`):**
   - Verify severity monotonicity and `[0, 100]` bounds (`U03`), priority monotonicity and `[0, 100]` bounds (`U07`), and EXIF parser resilience on random bytes (`U04`).
