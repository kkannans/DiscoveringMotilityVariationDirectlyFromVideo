"""
Paper Fig 3: Results — 3 panels.

Panel A: F-statistic bars — mean speed dominates between-organoid variance.
Panel B: Baseline comparison — VideoPredictionModel vs copy-last vs optical flow.
Panel C: Rank scatter — classical feature (mean speed) rank vs prediction surprise rank (ρ = 0.90).

Outputs -> paths.FIGURES_DIR
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import wilcoxon, spearmanr, kendalltau
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = paths.FIGURES_DIR / "fig3"
_CLASSICAL_DIR = paths.CLASSICAL_DIR
_INTERMEDIATES_DIR = paths.INTERMEDIATES_DIR
CENTROID_NPZ = _CLASSICAL_DIR / "centroid_trajectories.npz"
CLASSICAL_CSV = _CLASSICAL_DIR / "motility_descriptors.csv"
SUMMARY_CSV = _INTERMEDIATES_DIR / "per_organoid_summary.csv"
PER_SEED_CSV = _INTERMEDIATES_DIR / "per_seed_mse.csv"

# ── Colors (matched to Figure 1) ─────────────────────────────────────────
# classical pipeline blue: trajectory color in Fig 1 Panel C
# model red:              medium red from Fig 1 Panel F ranking gradient
COLORS = {
    "simvp":     "#e53935",   # model (red)
    "classical": "#2962FF",   # classical (blue, matches Fig 1 trajectory)
    "copylast":  "#b0b0b0",   # light gray, solid
    "optflow":   "#606060",   # dark gray, hatched
    "neutral":   "#a0a0a0",   # neutral gray for non-highlighted bars
}

# Panel A F-statistics are computed by
# classical_pipeline/find_most_reliable_classical_discriminator.py and read here from f_statistics.csv.


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all data
    mot = pd.read_csv(CLASSICAL_CSV)
    summary = pd.read_csv(SUMMARY_CSV)
    seed_df = pd.read_csv(PER_SEED_CSV)

    df = mot.merge(summary[["batch_name", "mean_motion_loss"]], on="batch_name")

    # ================================================================
    # Panel A: F-statistics (computed by the classical module)
    # ================================================================
    metric_labels = {
        "mean_velocity": "Mean speed",
        "acceleration": "Acceleration",
        "path_length": "Path length",
        "total_displacement": "Total displacement",
        "tortuosity": "Tortuosity",
        "outreach_ratio": "Outreach ratio",
        "mean_turning_angle": "Mean turning angle",
        "msd_slope": "MSD slope",
    }

    f_df = pd.read_csv(_CLASSICAL_DIR / "f_statistics.csv")
    f_stats = dict(zip(f_df["feature"], f_df["F_statistic"]))
    metrics = list(metric_labels.keys())
    sorted_metrics = sorted(metrics, key=lambda k: f_stats.get(k, 0), reverse=True)
    for m in sorted_metrics:
        print(f"  {m:25s}: F = {f_stats[m]:8.1f}")

    # ================================================================
    # Compute Panel C stats
    # ================================================================
    df["rank_velocity"] = df["mean_velocity"].rank()
    df["rank_model"] = df["mean_motion_loss"].rank()
    rho, p_rho = spearmanr(df["rank_velocity"], df["rank_model"])
    tau, _ = kendalltau(df["rank_velocity"], df["rank_model"])
    n = len(df)
    print(f"\nSpearman rho = {rho:.3f}, p = {p_rho:.2e}")

    # ================================================================
    # Combined figure: 3 panels
    # ================================================================
    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 3, width_ratios=[1.1, 0.8, 1.0], wspace=0.25)

    # ── Panel A: F-statistic bars ──
    ax_a = fig.add_subplot(gs[0, 0])
    y_pos = np.arange(len(sorted_metrics))
    f_vals = [f_stats[m] for m in sorted_metrics]
    bars = ax_a.barh(y_pos, f_vals, color=COLORS["neutral"], alpha=0.9,
                      edgecolor="gray", linewidth=0.5)

    # Highlight the mean velocity bar in classical blue (explicit index)
    mv_idx = sorted_metrics.index("mean_velocity")
    bars[mv_idx].set_color(COLORS["classical"])
    bars[mv_idx].set_alpha(0.9)

    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([metric_labels[m] for m in sorted_metrics], fontsize=16)
    ax_a.set_xlabel("F-statistic (between / within organoid variance)", fontsize=16)
    ax_a.invert_yaxis()

    for i, (v, m) in enumerate(zip(f_vals, sorted_metrics)):
        ax_a.text(v * 1.15, i, f"{v:.0f}",
                  va="center", fontsize=14,
                  fontweight="bold" if i == mv_idx else "normal")

    ax_a.set_xscale("log")
    ax_a.set_xlim(1, max(f_vals) * 3)

    # ── Panel B: Baseline comparison bars ──
    # Order: SimVP-TAU (red) → Copy-last (light gray solid) → Optical flow (dark gray hatched)
    ax_b = fig.add_subplot(gs[0, 1])
    methods = ["mse_model", "mse_copy", "mse_optical_flow"]
    labels_b = ["Video\nPrediction\nModel", "Copy-\nlast", "Optical\nflow"]
    colors_b = [COLORS["simvp"], COLORS["copylast"], COLORS["optflow"]]
    hatches_b = [None, None, "///"]

    means_b = [seed_df[m].mean() for m in methods]
    stds_b = [seed_df[m].std() for m in methods]
    n_seeds = len(seed_df)

    x_b = np.arange(len(methods))
    bars_b = ax_b.bar(x_b, means_b, yerr=stds_b, color=colors_b, alpha=0.9, capsize=4,
                      edgecolor="black", linewidth=0.8,
                      error_kw=dict(elinewidth=1.2, capthick=1.2))
    for bar, hatch in zip(bars_b, hatches_b):
        if hatch:
            bar.set_hatch(hatch)

    rng = np.random.default_rng(42)
    for i, m in enumerate(methods):
        jitter = rng.uniform(-0.15, 0.15, n_seeds)
        ax_b.scatter(x_b[i] + jitter, seed_df[m].values,
                     color="black", s=15, alpha=0.5, zorder=3)

    ax_b.set_yscale("log")
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(labels_b, fontsize=16)
    ax_b.set_ylabel(r"Mean $\mathcal{L}_{\mathrm{MSE}}$", fontsize=22, labelpad=-2)


    # Wilcoxon brackets
    model_vals = seed_df[methods[0]].values
    n_comparisons = len(methods) - 1
    y_max = max(m + s for m, s in zip(means_b, stds_b))
    bracket_y = y_max * 1.3

    for i in range(1, len(methods)):
        baseline_vals = seed_df[methods[i]].values
        stat, pval = wilcoxon(model_vals, baseline_vals, alternative="less")
        p_corrected = min(pval * n_comparisons, 1.0)
        if p_corrected < 0.001:
            p_str = "***"
        elif p_corrected < 0.01:
            p_str = "**"
        elif p_corrected < 0.05:
            p_str = "*"
        else:
            p_str = "n.s."

        ax_b.plot([0, 0, i, i],
                  [bracket_y, bracket_y * 1.05, bracket_y * 1.05, bracket_y],
                  color="black", linewidth=0.8)
        ax_b.text((0 + i) / 2, bracket_y * 1.07, p_str,
                  ha="center", va="bottom", fontsize=15)
        bracket_y *= 1.25

    # ── Panel C: Rank scatter ──
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.scatter(df["rank_velocity"], df["rank_model"], c="#1f77b4", s=20, alpha=0.6,
                 edgecolors="gray", linewidths=0.3)
    ax_c.plot([0, n + 1], [0, n + 1], "k--", lw=0.8, alpha=0.3)

    ax_c.set_xlabel("Classical feature (mean speed) rank", fontsize=16)
    ax_c.set_ylabel("Prediction surprise rank", fontsize=16)
    ax_c.set_xlim(0, n + 1)
    ax_c.set_ylim(0, n + 1)

    ax_c.text(0.05, 0.95,
              f"Spearman $\\rho$ = {rho:.2f}\nKendall $\\tau$ = {tau:.2f}",
              transform=ax_c.transAxes, fontsize=16, va="top",
              bbox=dict(boxstyle="round,pad=0.3", fc="wheat", ec="gray", alpha=0.8))

    # ── Panel labels: placed via fig coordinates for alignment ──
    fig.subplots_adjust(bottom=0.18)
    for ax, label in [(ax_a, "(A)"), (ax_b, "(B)"), (ax_c, "(C)")]:
        bbox = ax.get_position()
        fig.text(bbox.x0 + bbox.width / 2, 0.03, label,
                 fontsize=18, fontweight="bold", ha="center", va="bottom")

    # ── Save ──
    for ext in ["png", "pdf"]:
        fig.savefig(OUTPUT_DIR / f"fig3_combined.{ext}",
                    dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"\nSaved: {OUTPUT_DIR / 'fig3_combined.png'}")

    # Save stats
    with open(OUTPUT_DIR / "fig3_combined_stats.json", "w") as f:
        json.dump({
            "f_statistics": {m: round(f_stats[m], 1) for m in sorted_metrics},
            "baselines": {
                "model_mse": round(means_b[0], 6),
                "copy_mse": round(means_b[1], 6),
                "optflow_mse": round(means_b[2], 6),
            },
            "rank_agreement": {
                "spearman_rho": round(rho, 3),
                "kendall_tau": round(tau, 3),
                "p_value": float(p_rho),
            },
        }, f, indent=2)


if __name__ == "__main__":
    main()
