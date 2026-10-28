"""Build evaluation tables and figures strictly from `reports/metrics.json` and `runs/` (PRD Section 13)."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rdai.severity.rules import compute_severity_from_extent, load_severity_config

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_PATH = ROOT_DIR / "reports" / "metrics.json"
DEFAULT_TABLES_DIR = ROOT_DIR / "reports" / "tables"
DEFAULT_FIGURES_DIR = ROOT_DIR / "reports" / "figures"
DEFAULT_RUNS_DIR = ROOT_DIR / "runs"


def _write_csv_and_md(
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    csv_path: Path,
    md_path: Path,
    title: str,
) -> None:
    """Write a table as both CSV and GitHub-flavored Markdown."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    md_lines = [
        f"# {title}",
        "",
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join(":---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        md_lines.append("| " + " | ".join(str(row.get(k, "")) for k in fieldnames) + " |")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def _plot_per_class_ap(e1_data: dict[str, Any], out_path: Path) -> None:
    """Plot per-class AP@50 and AP@50:95 bar chart for E1."""
    per_cls = e1_data.get("per_class", {})
    classes = ["D00", "D10", "D20", "D40"]
    ap50 = [float(per_cls.get(c, {}).get("ap50", 0.0)) for c in classes]
    ap50_95 = [float(per_cls.get(c, {}).get("ap50_95", 0.0)) for c in classes]

    x = range(len(classes))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    ax.bar([i - width / 2 for i in x], ap50, width=width, label="AP@50", color="#2563eb")
    ax.bar([i + width / 2 for i in x], ap50_95, width=width, label="AP@50:95", color="#059669")
    ax.set_xticks(list(x))
    ax.set_xticklabels(classes)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Average Precision (AP)")
    ax.set_title("Main Result (E1) — Per-Class Test Split AP", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_ablations(experiments: dict[str, dict[str, Any]], out_path: Path) -> None:
    """Plot ablation comparison chart across available experiments in `metrics.json`."""
    exp_ids = sorted(experiments.keys())
    map50 = [float(experiments[e].get("overall", {}).get("map50", 0.0)) for e in exp_ids]
    d40_ap50 = [
        float(experiments[e].get("per_class", {}).get("D40", {}).get("ap50", 0.0)) for e in exp_ids
    ]

    x = range(len(exp_ids))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=150)
    ax.bar(
        [i - width / 2 for i in x], map50, width=width, label="All Classes mAP@50", color="#2563eb"
    )
    ax.bar(
        [i + width / 2 for i in x],
        d40_ap50,
        width=width,
        label="D40 Pothole AP@50",
        color="#d97706",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(exp_ids)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AP@50")
    ax.set_title("Ablation Study Comparison (Test Split)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_training_curves_from_csv(results_csv: Path, out_path: Path) -> None:
    """Plot training and validation loss and mAP curves from Ultralytics `results.csv`."""
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    epochs = df["epoch"] if "epoch" in df.columns else range(1, len(df) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150)
    for col in ("train/box_loss", "val/box_loss", "train/cls_loss", "val/cls_loss"):
        if col in df.columns:
            ax1.plot(epochs, df[col], label=col)
    ax1.set_title("Training & Validation Loss Curves", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.4)

    for col in ("metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        if col in df.columns:
            ax2.plot(epochs, df[col], label=col)
    ax2.set_title("Validation mAP Convergence", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mAP")
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_severity_and_priority_slide_figure(out_path: Path) -> None:
    """Plot Slide 7 figure: piecewise-linear severity curves S(e) and priority term breakdown."""
    sev_cfg = load_severity_config()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=150)

    extents = np.linspace(0.0, 0.50, 250)
    colors = {"D40": "#dc2626", "D20": "#d97706", "D00": "#2563eb"}
    labels = {
        "D40": "D40 Pothole (Area Ratio)",
        "D20": "D20 Alligator Crack (Area Ratio)",
        "D00": "D00/D10 Linear Crack (Diag Ratio)",
    }
    for cls_code in ("D40", "D20", "D00"):
        scores = [
            compute_severity_from_extent(cls_code, float(e), config=sev_cfg)["score"]
            for e in extents
        ]
        ax1.plot(extents, scores, label=labels[cls_code], color=colors[cls_code], linewidth=2.2)

    ax1.axhline(33.3, color="#6b7280", linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.axhline(66.7, color="#6b7280", linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.set_title("Continuous Severity Score S(e) by Class", fontweight="bold")
    ax1.set_xlabel("Normalized Extent (e)")
    ax1.set_ylabel("Severity Score S (0–100)")
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.35)

    # Worked example breakdown (PRD Section 10: 36 + 2.5 + 14 + 8 = 60.5)
    terms = ["0.45 * S_max\n(36.0)", "0.25 * D\n(2.5)", "0.20 * R\n(14.0)", "0.10 * C\n(8.0)"]
    vals = [36.0, 2.5, 14.0, 8.0]
    bar_colors = ["#dc2626", "#f59e0b", "#2563eb", "#059669"]
    ax2.bar(terms, vals, color=bar_colors, width=0.52)
    ax2.set_title("Worked Priority Example (P = 60.5, Medium)", fontweight="bold")
    ax2.set_ylabel("Weighted Contribution to P")
    ax2.set_ylim(0, 45)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def build_all_results(
    metrics_path: Path = DEFAULT_METRICS_PATH,
    tables_dir: Path = DEFAULT_TABLES_DIR,
    figures_dir: Path = DEFAULT_FIGURES_DIR,
    runs_dir: Path = DEFAULT_RUNS_DIR,
) -> dict[str, Any]:
    """Generate PRD Section 13 tables and figures strictly from `reports/metrics.json`."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    _plot_severity_and_priority_slide_figure(figures_dir / "severity_priority_curves.png")

    if not metrics_path.is_file():
        metrics_doc: dict[str, Any] = {"status": "awaiting_colab_runs", "experiments": {}}
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics_doc, indent=2), encoding="utf-8")
    else:
        metrics_doc = json.loads(metrics_path.read_text(encoding="utf-8"))

    experiments: dict[str, dict[str, Any]] = metrics_doc.get("experiments", {})
    main_fields = ["Class", "Precision", "Recall", "AP@50", "AP@50:95"]

    if "E1" in experiments:
        e1 = experiments["E1"]
        per_cls = e1.get("per_class", {})
        main_rows: list[dict[str, Any]] = []
        for cls_code in ("D00", "D10", "D20", "D40"):
            c_metrics = per_cls.get(cls_code, {})
            main_rows.append(
                {
                    "Class": cls_code,
                    "Precision": c_metrics.get("precision", ""),
                    "Recall": c_metrics.get("recall", ""),
                    "AP@50": c_metrics.get("ap50", ""),
                    "AP@50:95": c_metrics.get("ap50_95", ""),
                }
            )
        overall = e1.get("overall", {})
        main_rows.append(
            {
                "Class": "all",
                "Precision": overall.get("precision", ""),
                "Recall": overall.get("recall", ""),
                "AP@50": overall.get("map50", ""),
                "AP@50:95": overall.get("map50_95", ""),
            }
        )
        _write_csv_and_md(
            main_rows,
            main_fields,
            tables_dir / "main_result.csv",
            tables_dir / "main_result.md",
            "Main Result Table (E1 — India Test Split)",
        )
        _plot_per_class_ap(e1, figures_dir / "results_per_class_ap.png")
    else:
        _write_csv_and_md(
            [
                {
                    "Class": "PENDING_COLAB_E1_RUN",
                    "Precision": "N/A",
                    "Recall": "N/A",
                    "AP@50": "N/A",
                    "AP@50:95": "N/A",
                }
            ],
            main_fields,
            tables_dir / "main_result.csv",
            tables_dir / "main_result.md",
            "Main Result Table (Awaiting E1 Colab Evaluation in reports/metrics.json)",
        )

    ablation_fields = [
        "Experiment",
        "mAP@50",
        "mAP@50:95",
        "Pothole AP@50",
        "D10 AP@50",
        "Parameters (M)",
        "GPU ms",
        "CPU ms",
    ]
    if experiments:
        ablation_rows: list[dict[str, Any]] = []
        for exp_id in sorted(experiments.keys()):
            exp = experiments[exp_id]
            ov = exp.get("overall", {})
            pc = exp.get("per_class", {})
            ablation_rows.append(
                {
                    "Experiment": exp_id,
                    "mAP@50": ov.get("map50", ""),
                    "mAP@50:95": ov.get("map50_95", ""),
                    "Pothole AP@50": pc.get("D40", {}).get("ap50", ""),
                    "D10 AP@50": pc.get("D10", {}).get("ap50", ""),
                    "Parameters (M)": exp.get("params_m", ""),
                    "GPU ms": exp.get("gpu_ms", ""),
                    "CPU ms": exp.get("cpu_ms", ""),
                }
            )
        _write_csv_and_md(
            ablation_rows,
            ablation_fields,
            tables_dir / "ablations.csv",
            tables_dir / "ablations.md",
            "Ablations Table (E1–E7)",
        )
        _plot_ablations(experiments, figures_dir / "results_ablations.png")

    e1_run_dir = runs_dir / "e1_yolo11s_640"
    e1_csv = e1_run_dir / "results.csv"
    if e1_csv.is_file():
        _plot_training_curves_from_csv(e1_csv, figures_dir / "training_curves_e1.png")

    for plot_name in (
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "F1_curve.png",
        "val_batch0_pred.jpg",
    ):
        src_plot = e1_run_dir / plot_name
        if src_plot.is_file():
            shutil.copy2(src_plot, figures_dir / plot_name)

    return {
        "status": "populated" if "E1" in experiments else "awaiting_colab_runs",
        "experiments_found": sorted(experiments.keys()),
    }


if __name__ == "__main__":
    summary = build_all_results()
    print(f"Results build summary: {summary}")
