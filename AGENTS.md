# AGENTS.md

Project: road damage detection, severity grading and maintenance priority for India.
Read `docs/PRD.md` before any task.

- Work on one milestone at a time, in the order of PRD section 15.
- Write the milestone's tests first (PRD section 12), then the code. Run `make test` and `make lint` after each task.
- Before using Ultralytics, FastAPI, SQLAlchemy, scikit-learn, Folium or Pillow, fetch current docs with Context7.
- Never invent metrics. Every number in docs and slides comes from `reports/metrics.json`.
- Thresholds, weights and limits live in `configs/*.yaml`, never in code.
- Copy code only from repos with an explicit permissive LICENSE file. Record anything reused in `NOTICE.md`.
- Never commit data, weights or secrets.
- Write no frontend code until the user says the backend is done.
