"""
analyze_test_performance_simVP.py — Analyze SimVP_TAU test performance across seeds.

Aggregates per-organoid test metrics from automated_method/test_model.py across
all 10 seeds, computes:
  - Per-organoid mean MSE and SSIM (averaged across seeds)
  - Model vs baseline comparisons (copy-last, optical flow, mean frame, rolling avg)
  - Cross-seed consistency (Spearman rank correlation between seed pairs)
  - Prediction surprise ranking (model MSE relative to baselines)

Usage:
    python analyze_test_performance_simVP.py
    python analyze_test_performance_simVP.py --seeds 1 2 3 --metric mse

Inputs:
    OUTPUT_ROOT / predictability_analysis / extraction / seed*_per_organoid.json

Outputs -> OUTPUT_ROOT / predictability_analysis /
    simvp_performance_summary.csv     Per-organoid aggregated metrics
    simvp_seed_consistency.json       Cross-seed rank correlations
    simvp_baseline_comparison.png     Model vs baselines bar chart
    simvp_seed_scatter.png            Pairwise seed rank scatter
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import paths

OUTPUT_DIR = paths.OUTPUT_ROOT / "predictability_analysis"


def load_seed_results(seeds, residual=False):
    """Load per-seed extraction results."""
    tag = "_residual" if residual else ""
    extraction_dir = OUTPUT_DIR / "extraction"
    results = {}
    for seed in seeds:
        path = extraction_dir / f"seed{seed}{tag}_per_organoid.json"
        if path.exists():
            with open(path) as f:
                results[seed] = json.load(f)
        else:
            print(f"  Seed {seed}{tag}: not found, skipping")
    return results


def compute_seed_consistency(seed_results, metric="model_mse"):
    """Compute pairwise Spearman rank correlation between seeds."""
    from scipy.stats import spearmanr

    seeds = sorted(seed_results.keys())
    # Get common organoids across all seeds
    all_orgs = set()
    for s in seeds:
        all_orgs.update(seed_results[s]["organoids"].keys())
    common = sorted(all_orgs)

    # Build matrix: (n_organoids, n_seeds)
    matrix = np.full((len(common), len(seeds)), np.nan)
    for si, s in enumerate(seeds):
        for oi, org in enumerate(common):
            if org in seed_results[s]["organoids"]:
                matrix[oi, si] = seed_results[s]["organoids"][org].get(metric, np.nan)

    # Pairwise Spearman
    correlations = {}
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            valid = np.isfinite(matrix[:, i]) & np.isfinite(matrix[:, j])
            if valid.sum() > 5:
                rho, pval = spearmanr(matrix[valid, i], matrix[valid, j])
                correlations[f"seed_{seeds[i]}_vs_{seeds[j]}"] = {
                    "spearman_rho": round(float(rho), 4),
                    "p_value": float(pval),
                    "n_common": int(valid.sum()),
                }

    mean_rho = np.mean([v["spearman_rho"] for v in correlations.values()]) if correlations else 0
    return {"pairwise": correlations, "mean_spearman": round(mean_rho, 4)}


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SimVP_TAU test performance")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--metric", choices=["mse", "ssim"], default="mse")
    parser.add_argument("--residual", action="store_true")
    args = parser.parse_args()

    tag = "_residual" if args.residual else ""
    print(f"Analyzing SimVP_TAU ({tag.strip('_') or 'plain'}) across seeds {args.seeds}")

    seed_results = load_seed_results(args.seeds, residual=args.residual)
    if not seed_results:
        print("ERROR: No seed results found. Run automated_method/test_model.py first.")
        sys.exit(1)

    print(f"  Loaded {len(seed_results)} seeds")

    # Aggregate per organoid across seeds
    organoid_agg = defaultdict(lambda: defaultdict(list))
    metric_keys = ["model_mse", "model_ssim", "copy_mse", "copy_ssim",
                    "optical_flow_mse", "optical_flow_ssim",
                    "mean_frame_mse", "rolling_avg_mse"]

    for seed, data in seed_results.items():
        for org, metrics in data["organoids"].items():
            for key in metric_keys:
                if key in metrics:
                    organoid_agg[org][key].append(metrics[key])
            organoid_agg[org]["seeds"].append(seed)

    rows = []
    for org in sorted(organoid_agg.keys()):
        agg = organoid_agg[org]
        row = {"batch_name": org, "n_seeds": len(agg["model_mse"])}
        for key in metric_keys:
            vals = agg.get(key, [])
            if vals:
                row[f"{key}_mean"] = round(float(np.mean(vals)), 8)
                row[f"{key}_std"] = round(float(np.std(vals)), 8)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Compute prediction surprise: ratio of model MSE to copy-last MSE
    if "model_mse_mean" in df.columns and "copy_mse_mean" in df.columns:
        df["prediction_surprise"] = df["model_mse_mean"] / df["copy_mse_mean"]

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"simvp_performance_summary{tag}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} ({len(df)} organoids)")

    # Seed consistency
    consistency = compute_seed_consistency(seed_results, metric=f"model_{args.metric}")
    consistency_path = OUTPUT_DIR / f"simvp_seed_consistency{tag}.json"
    with open(consistency_path, "w") as f:
        json.dump(consistency, f, indent=2)
    print(f"Saved: {consistency_path}")
    print(f"  Mean cross-seed Spearman: {consistency['mean_spearman']:.3f}")

    # Baseline comparison plot
    baseline_cols = [("model_mse_mean", "Model"),
                     ("copy_mse_mean", "Copy-last"),
                     ("optical_flow_mse_mean", "Optical Flow")]
    available = [(col, label) for col, label in baseline_cols if col in df.columns]

    if len(available) >= 2:
        fig, ax = plt.subplots(figsize=(6, 5))
        means = [df[col].mean() for col, _ in available]
        stds = [df[col].std() for col, _ in available]
        labels = [label for _, label in available]
        colors = ["#2ca02c", "#d62728", "#1f77b4"][:len(available)]

        x = np.arange(len(available))
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.7, capsize=4,
               edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel("Mean MSE", fontsize=11)
        ax.set_yscale("log")

        fig.tight_layout()
        fig_path = OUTPUT_DIR / f"simvp_baseline_comparison{tag}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fig_path}")

    # Summary
    if "model_mse_mean" in df.columns:
        model_mean = df["model_mse_mean"].mean()
        print(f"\n  Model MSE: {model_mean:.6f}")
        for col, label in baseline_cols[1:]:
            if f"{col}" in df.columns:
                bl_mean = df[col].mean()
                improvement = (1 - model_mean / bl_mean) * 100
                print(f"  vs {label}: {bl_mean:.6f} ({improvement:+.1f}%)")


if __name__ == "__main__":
    main()
