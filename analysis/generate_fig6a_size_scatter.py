"""
Fig (scatter): Organoid size vs prediction surprise — no correlation.

Standalone figure (split from old combined Fig 4 Panel A).
CPU-only — no GPU needed.

Shows that organoid size does not confound prediction surprise,
with Spearman correlation annotation.

Outputs -> paths.FIGURES_DIR / fig_scatter_size.{png,pdf}
"""

import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import paths

OUTPUT_DIR = paths.FIGURES_DIR
CLASSICAL_CSV = paths.CLASSICAL_DIR / "motility_descriptors.csv"
SUMMARY_CSV = paths.INTERMEDIATES_DIR / "per_organoid_summary.csv"
SELECTED_JSON = paths.INTERMEDIATES_DIR / "selected_spatial_organoids.json"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mot = pd.read_csv(CLASSICAL_CSV)
    summary = pd.read_csv(SUMMARY_CSV)
    df = summary.merge(mot[["batch_name", "mean_velocity"]], on="batch_name")

    with open(SELECTED_JSON) as f:
        selected = json.load(f)

    sel_batches = [selected["low"]["batch"], selected["medium"]["batch"],
                   selected["high"]["batch"]]
    sel_labels = ["Low", "Medium", "High"]
    sel_colors = ["#333333", "#b22222", "#ff0000"]  # black-to-red gradient

    # Stats
    rho_size, p_size = stats.spearmanr(df["area_px"], df["mean_motion_loss"])
    print(f"Size vs Prediction Surprise: Spearman rho = {rho_size:.3f}, p = {p_size:.3f}")

    # ── Figure ──
    fig, ax = plt.subplots(figsize=(4.0, 3.5))

    ax.scatter(df["area_px"], df["mean_motion_loss"], c="#1f77b4", s=20,
               alpha=0.5, edgecolors="gray", linewidths=0.3, zorder=2)

    for batch, label, color in zip(sel_batches, sel_labels, sel_colors):
        row = df[df.batch_name == batch]
        if len(row) > 0:
            x_val = row["area_px"].values[0]
            y_val = row["mean_motion_loss"].values[0]
            ax.scatter(x_val, y_val, c=color, s=120, edgecolors="black",
                       linewidths=1.5, zorder=5, label=label)
            ax.annotate(label, (x_val, y_val), fontsize=8, fontweight="bold",
                        color=color, xytext=(8, 0), textcoords="offset points",
                        va="center", zorder=6)

    ax.set_xlabel("Organoid size (px)", fontsize=11)
    ax.set_ylabel(r"Prediction surprise (log$_{10}$)", fontsize=11)
    ax.set_yscale("log")
    ax.legend(fontsize=9, title="Surprise\nlevel", title_fontsize=8,
              loc="upper left", framealpha=0.8)

    sig_str = "n.s." if p_size >= 0.05 else ("*" if p_size >= 0.01 else "**")
    ax.text(0.95, 0.95,
            f"Spearman $\\rho$ = {rho_size:.2f} ({sig_str})",
            transform=ax.transAxes, fontsize=10, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", ec="gray", alpha=0.8))

    ax.tick_params(axis="both", labelsize=9)

    # ── Save ──
    for ext in ["png", "pdf"]:
        out = OUTPUT_DIR / f"fig_scatter_size.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'fig_scatter_size.png'}")

    # Save stats
    with open(OUTPUT_DIR / "fig_scatter_size_results.json", "w") as f:
        json.dump({
            "spearman_rho": round(rho_size, 3),
            "p_value": round(p_size, 4),
            "n_organoids": len(df),
        }, f, indent=2)


if __name__ == "__main__":
    main()
