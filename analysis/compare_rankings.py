"""
compare_rankings.py — Compare model-based and classical organoid rankings.

Computes rank correlation between the automated (prediction surprise) ranking
and the classical (motility descriptor) ranking, testing whether the video
prediction model captures information beyond what classical centroid tracking
provides.

Usage:
    python compare_rankings.py
    python compare_rankings.py --classical mean_velocity --automated model_mse

Inputs:
    INTERMEDIATES_DIR / per_organoid_summary.csv          Model-based ranking
    CLASSICAL_DIR / motility_descriptors.csv              Classical descriptors
    CLASSICAL_DIR / classical_ranking.csv                 Classical ranking (optional)

Outputs -> FIGURES_DIR /
    ranking_comparison.png           Scatter plot of rankings
    ranking_comparison.csv           Merged rankings
    ranking_comparison_report.json   Statistical summary
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import paths


def main():
    parser = argparse.ArgumentParser(
        description="Compare automated vs classical organoid rankings")
    parser.add_argument("--classical", type=str, default="mean_velocity",
                        help="Classical descriptor to compare (default: mean_velocity)")
    parser.add_argument("--automated", type=str, default="mean_motion_loss",
                        help="Automated metric to compare (default: mean_motion_loss)")
    args = parser.parse_args()

    # Load model-based summary
    model_path = paths.INTERMEDIATES_DIR / "per_organoid_summary.csv"
    if not model_path.exists():
        print(f"ERROR: {model_path} not found.")
        print("Run: python automated_method/rank_by_prediction_surprise.py")
        sys.exit(1)
    df_model = pd.read_csv(model_path)

    # Load classical descriptors
    classical_path = paths.CLASSICAL_DIR / "motility_descriptors.csv"
    if not classical_path.exists():
        print(f"ERROR: {classical_path} not found.")
        print("Run: python classical_pipeline/extract_motility_features.py")
        sys.exit(1)
    df_classical = pd.read_csv(classical_path)

    # Merge on batch_name
    df = df_model.merge(df_classical, on="batch_name", how="inner")
    print(f"Merged: {len(df)} organoids ({len(df_model)} model, {len(df_classical)} classical)")

    if args.automated not in df.columns:
        print(f"ERROR: '{args.automated}' not in model summary. "
              f"Available: {[c for c in df_model.columns if c != 'batch_name']}")
        sys.exit(1)
    if args.classical not in df.columns:
        print(f"ERROR: '{args.classical}' not in classical descriptors. "
              f"Available: {[c for c in df_classical.columns if c != 'batch_name']}")
        sys.exit(1)

    auto_vals = df[args.automated].values
    class_vals = df[args.classical].values

    # Remove NaNs
    valid = np.isfinite(auto_vals) & np.isfinite(class_vals)
    auto_vals = auto_vals[valid]
    class_vals = class_vals[valid]
    df_valid = df[valid].copy()

    # Compute ranks
    auto_ranks = auto_vals.argsort().argsort() + 1
    class_ranks = class_vals.argsort().argsort() + 1

    # Correlations
    rho, p_rho = spearmanr(auto_vals, class_vals)
    tau, p_tau = kendalltau(auto_vals, class_vals)

    print(f"\n{'='*50}")
    print(f"Ranking Comparison")
    print(f"{'='*50}")
    print(f"  Automated: {args.automated}")
    print(f"  Classical: {args.classical}")
    print(f"  N = {len(auto_vals)}")
    print(f"  Spearman ρ = {rho:.3f} (p = {p_rho:.2e})")
    print(f"  Kendall τ = {tau:.3f} (p = {p_tau:.2e})")

    # Identify discordant organoids (large rank difference)
    rank_diff = np.abs(auto_ranks - class_ranks)
    discordant_threshold = len(auto_vals) * 0.4  # top 40% rank difference
    discordant = rank_diff > discordant_threshold
    n_discordant = discordant.sum()
    print(f"  Discordant (rank diff > {discordant_threshold:.0f}): {n_discordant}")

    # Save merged CSV
    output_dir = paths.FIGURES_DIR / "ranking_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    df_valid["auto_rank"] = auto_ranks
    df_valid["classical_rank"] = class_ranks
    df_valid["rank_diff"] = rank_diff
    df_valid[["batch_name", args.automated, args.classical,
              "auto_rank", "classical_rank", "rank_diff"]].to_csv(
        output_dir / "ranking_comparison.csv", index=False)

    # Save report
    report = {
        "automated_metric": args.automated,
        "classical_metric": args.classical,
        "n_organoids": int(len(auto_vals)),
        "spearman_rho": round(float(rho), 4),
        "spearman_p": float(p_rho),
        "kendall_tau": round(float(tau), 4),
        "kendall_p": float(p_tau),
        "n_discordant": int(n_discordant),
        "discordant_threshold": int(discordant_threshold),
    }
    with open(output_dir / "ranking_comparison_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Scatter plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Value scatter
    ax = axes[0]
    ax.scatter(class_vals, auto_vals, s=15, alpha=0.6, edgecolors="k", linewidth=0.3)
    ax.set_xlabel(args.classical.replace("_", " ").title(), fontsize=11)
    ax.set_ylabel(args.automated.replace("_", " ").title(), fontsize=11)
    ax.text(0.02, 0.98, f"Spearman ρ = {rho:.3f}\nKendall τ = {tau:.3f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    # Panel 2: Rank scatter
    ax = axes[1]
    ax.scatter(class_ranks, auto_ranks, s=15, alpha=0.6, c="steelblue",
               edgecolors="k", linewidth=0.3)
    # Highlight discordant
    ax.scatter(class_ranks[discordant], auto_ranks[discordant],
               s=25, alpha=0.8, c="red", edgecolors="k", linewidth=0.3,
               label=f"Discordant ({n_discordant})")
    ax.plot([1, len(auto_vals)], [1, len(auto_vals)], "k--", linewidth=0.8, alpha=0.4)
    ax.set_xlabel(f"Classical rank ({args.classical})", fontsize=11)
    ax.set_ylabel(f"Automated rank ({args.automated})", fontsize=11)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig_path = output_dir / "ranking_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {fig_path}")


if __name__ == "__main__":
    main()
