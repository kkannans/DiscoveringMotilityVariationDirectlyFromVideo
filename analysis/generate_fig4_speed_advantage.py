"""
Fig 3 (Draft 6): Speed-bin crossover plot — model advantage over copy-last.

Standalone figure (split from draft 5 combined Fig 3).
CPU-only — no GPU needed.

Outputs -> Q8_v23-alife-2026/outputs/figures/Fig3/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths

OUTPUT_DIR = paths.FIGURES_DIR / "fig4"
MOTILITY_CSV = paths.CLASSICAL_DIR / "motility_descriptors.csv"
RESIDUALS_DIR = paths.INTERMEDIATES_DIR / "per_organoid_residuals"


def compute_crossover_data():
    """Compute advantage vs velocity decile data."""
    mot = pd.read_csv(MOTILITY_CSV)

    rows = []
    for npz_path in sorted(RESIDUALS_DIR.glob("*.npz")):
        batch = npz_path.stem
        npz = np.load(npz_path)
        raw_mse = npz["raw_mse"]
        rows.append({
            "batch_name": batch,
            "model_mse": raw_mse[:, :, 0].mean(),
            "copy_mse": raw_mse[:, :, 1].mean(),
        })

    df = pd.DataFrame(rows)
    df = df.merge(mot[["batch_name", "mean_velocity"]], on="batch_name")
    df["advantage"] = 1.0 - df["model_mse"] / (df["copy_mse"] + 1e-12)

    df = df.sort_values("mean_velocity").reset_index(drop=True)
    df["decile"] = pd.qcut(df["mean_velocity"], q=10, labels=False)

    decile_stats = df.groupby("decile").agg(
        vel_lo=("mean_velocity", "min"),
        vel_hi=("mean_velocity", "max"),
        vel_median=("mean_velocity", "median"),
        adv_mean=("advantage", "mean"),
        adv_std=("advantage", "std"),
        n_count=("advantage", "count"),
    ).reset_index()

    crossover_vel = None
    crossover_decile = None
    for i in range(len(decile_stats) - 1):
        a0 = decile_stats.iloc[i]["adv_mean"]
        a1 = decile_stats.iloc[i + 1]["adv_mean"]
        if a0 <= 0 and a1 > 0:
            v0 = decile_stats.iloc[i]["vel_median"]
            v1 = decile_stats.iloc[i + 1]["vel_median"]
            crossover_vel = v0 + (0 - a0) / (a1 - a0) * (v1 - v0)
            crossover_decile = i + (0 - a0) / (a1 - a0)
            break

    rho, p = spearmanr(df["mean_velocity"], df["advantage"])
    print(f"n={len(df)} organoids, Spearman rho={rho:.3f}, p={p:.2e}")
    if crossover_vel is not None:
        print(f"Crossover velocity: {crossover_vel:.2f} px/f")

    return df, decile_stats, crossover_vel, crossover_decile


def plot_crossover(ax, df, decile_stats, crossover_vel, crossover_decile):
    """Plot crossover: advantage vs velocity decile."""
    for _, row in df.iterrows():
        ax.scatter(row["decile"], row["advantage"], color="#1f77b4", s=12,
                   alpha=0.2, edgecolors="none", zorder=2)

    ax.errorbar(decile_stats["decile"], decile_stats["adv_mean"],
                yerr=decile_stats["adv_std"], color="#1f77b4",
                marker="o", linewidth=1.8, capsize=3, markersize=5,
                markeredgecolor="black", markeredgewidth=0.4, zorder=4)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, zorder=1)

    ax.text(9.3, 0.02, "copy-last\nequivalent", fontsize=8.5, color="gray",
            va="bottom", ha="right", fontweight="bold")

    if crossover_vel is not None:
        ax.annotate(f"crossover\n~{crossover_vel:.1f} px/f",
                    xy=(crossover_decile, 0),
                    xytext=(crossover_decile + 2.0, -0.3),
                    fontsize=8.5, ha="center",
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow",
                              edgecolor="gray", alpha=0.9))

    ax.set_xticks(range(10))
    x_labels = []
    for _, r in decile_stats.iterrows():
        lo, hi = r["vel_lo"], r["vel_hi"]
        if hi < 0.1:
            x_labels.append(f"{lo:.2f}-{hi:.2f}")
        elif hi < 1.0:
            x_labels.append(f"{lo:.2f}-{hi:.2f}")
        else:
            x_labels.append(f"{lo:.1f}-{hi:.1f}")
    ax.set_xticklabels(x_labels, fontsize=7.5, rotation=45, ha="right")

    ax.set_xlabel("Mean speed bin (px/frame)", fontsize=9.5)
    ax.set_ylabel("Model advantage\nover copy-last", fontsize=9.5)
    ax.tick_params(axis="y", labelsize=8.5)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, decile_stats, crossover_vel, crossover_decile = compute_crossover_data()

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    plot_crossover(ax, df, decile_stats, crossover_vel, crossover_decile)

    for ext in ["png", "pdf"]:
        out = OUTPUT_DIR / f"fig3_speed_bin_crossover.{ext}"
        fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'fig3_speed_bin_crossover.png'}")


if __name__ == "__main__":
    main()
