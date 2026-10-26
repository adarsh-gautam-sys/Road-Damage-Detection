---
name: rdai-results
description: Build evaluation tables and figures strictly from reports/metrics.json and run logs per PRD Section 13.
---

# Road Damage AI Results Skill (`rdai-results`)

Follow `docs/PRD.md` Section 13 when generating tables, figures, notebooks, or slide numbers:

1. **Single Source of Truth:**
   - Never invent, estimate, or type metrics by hand.
   - `scripts/make_results.py` must read `reports/metrics.json` and Ultralytics `runs/` outputs to generate all tables (`reports/tables/`) and figures (`reports/figures/`).
2. **Required Outputs:**
   - Tables: Main result (`E1` per-class Precision, Recall, AP@50, AP@50:95), Ablations (`E1`–`E7`), Severity audit (50 held-out crops agreement, Cohen's kappa, confusion matrix), Top 10 priority segments + 20% weight sensitivity.
   - Figures: EDA charts, training/val curves, PR & F1 curves, normalized confusion matrix, per-class AP & ablation bars, 24-image failure gallery (12 FP + 12 FN) with 50-failure cause counts, and Folium map screenshot.
