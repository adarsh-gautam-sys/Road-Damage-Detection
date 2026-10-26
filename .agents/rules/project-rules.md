# Project Rules (Road Damage AI)

- Read `docs/PRD.md` and `AGENTS.md` before starting any task.
- Build one milestone at a time in the order of PRD Section 15 (`M0` -> `M10`, then `M11` later).
- Write tests first (PRD Section 12), then implementation. Run `make test` and `make lint` after every task.
- Use Context7 MCP (`resolve-library-id` / `query-docs`) before writing code with Ultralytics (`/websites/ultralytics`), FastAPI (`/websites/fastapi_tiangolo`), SQLAlchemy 2.0 (`/websites/sqlalchemy_en_20`), scikit-learn, Pydantic, Folium, Pillow, piexif, gpxpy, pytest, or hypothesis.
- Keep all thresholds, weights, and limits in `configs/*.yaml` — no magic numbers in Python code.
- Never invent or hardcode metrics; all reported numbers must come from `reports/metrics.json`.
- Copy code only from repos with an explicit permissive `LICENSE` file (`uzbtrust/smart-road`, MIT) and document in `NOTICE.md`.
- Do not write any frontend code until `M0`–`M9` are complete and the user explicitly requests `M11`.
