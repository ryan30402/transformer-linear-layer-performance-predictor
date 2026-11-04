"""Plot recorded prediction errors without refitting or selecting models."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=root / "assets/prediction_error.png")
    args = parser.parse_args()
    summary = root / "results/summary"
    with (summary / "model_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    metrics = {(r["session_id"], r["split"], r["model_id"]): float(r["median_ape_pct"]) for r in rows}
    if len(metrics) != len(rows):
        raise ValueError("Model metrics contain duplicate keys")
    with (summary / "selected_pipeline.csv").open(newline="") as handle:
        selected = {r["session_id"]: r["selected_model_id"] for r in csv.DictReader(handle)}

    colors = {"roofline_3p": "#64748b", "shape_log_ridge": "#0086a8", "roofline_residual_krr": "#d96822"}
    labels = {"roofline_3p": "Roofline 3p", "shape_log_ridge": "Ridge", "roofline_residual_krr": "Residual KRR"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "text.color": "#1c3040", "axes.labelcolor": "#334155",
                         "xtick.color": "#475569", "ytick.color": "#475569"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.8), sharey=True)
    fig.subplots_adjust(left=.09, right=.98, top=.80, bottom=.23, hspace=.48, wspace=.18)
    for row, dtype in enumerate(("fp16", "bf16")):
        for col, split in enumerate(("test", "challenge")):
            ax = axes[row, col]
            ax.set_title(f"{dtype.upper()} - {split.title()} ({24 if split == 'test' else 12} shapes)",
                         loc="left", fontsize=12, weight="bold", pad=10)
            sessions = [f"{dtype}_s{i:02d}" for i in (1, 2, 3)]
            for model_id, color in colors.items():
                ax.plot((1, 2, 3), [metrics[s, split, model_id] for s in sessions],
                        color=color, linewidth=2, marker="o", markersize=5)
            for x, session in enumerate(sessions, 1):
                y = metrics[session, split, selected[session]]
                ax.scatter([x], [y], marker="*", s=200, c="#f2c65a", edgecolors="#20384d", linewidths=1.2, zorder=5)
                offset = -21 if session == "fp16_s03" and split == "test" else 10
                ax.annotate(f"{y:.2f}%", (x, y), xytext=(0, offset), textcoords="offset points",
                            ha="center", fontsize=10)
            ax.set(xlim=(.75, 3.25), ylim=(0, 64), xticks=(1, 2, 3), yticks=(0, 15, 30, 45, 60))
            ax.set_xticklabels(("Session 1", "Session 2", "Session 3"))
            ax.grid(axis="y", color="#e4ebf3", linewidth=.9)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("bottom", "left"):
                ax.spines[side].set_color("#b7c4d4")
            if col == 0:
                ax.set_ylabel("Median absolute percentage error (%)")

    fig.text(.09, .945, "Prediction error across three measured rounds", fontsize=20, weight="bold")
    fig.text(.09, .911, "Lower is better. Models are fitted separately for each dtype and session.", fontsize=12)
    handles = [Line2D([0], [0], color=c, marker="o", linewidth=2, label=labels[m]) for m, c in colors.items()]
    handles.append(Line2D([0], [0], color="none", marker="*", markersize=14,
                          markerfacecolor="#f2c65a", markeredgecolor="#20384d",
                          label="Selected using development only"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.08, .12), ncol=4,
               frameon=False, columnspacing=1.2, handlelength=1.5, fontsize=10)
    fig.text(.09, .094, "FP16 selection: Ridge, Ridge, KRR. BF16 selection: Roofline 3p in every session.",
             fontsize=11, weight="bold")
    fig.text(.09, .064, "The same held-out shapes repeat across rounds. Lines describe repeats, not confidence intervals.",
             fontsize=9.5, color="#64748b")
    fig.text(.09, .040, "Three registered candidates are displayed; all five are in the CSV. Candidate scores do not change frozen selections.",
             fontsize=9.5, color="#64748b")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor="white", metadata={"Title": "Recorded prediction error"})
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
