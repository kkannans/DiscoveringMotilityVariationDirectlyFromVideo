"""
analyze_test_performance_predRNN.py — Analyze PredRNN test performance across seeds.

Mirrors analyze_test_performance_simVP.py but for PredRNN models. The analysis
pipeline is identical; only the model name and checkpoint paths differ.

Prerequisites:
  1. Train PredRNN models:
     python automated_method/train_model.py --method PredRNN --seed 1 -K 6 -N 6
  2. Extract test predictions:
     python automated_method/test_model.py --method PredRNN --seeds 1 2 3 ...

Usage:
    python analyze_test_performance_predRNN.py
    python analyze_test_performance_predRNN.py --seeds 1 2 3

Inputs:
    OUTPUT_ROOT / predictability_analysis / extraction / seed*_PredRNN_per_organoid.json

Outputs -> OUTPUT_ROOT / predictability_analysis /
    predrnn_performance_summary.csv
    predrnn_seed_consistency.json
    predrnn_baseline_comparison.png
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
METHOD = "PredRNN"


def load_seed_results(seeds):
    """Load per-seed PredRNN extraction results."""
    extraction_dir = OUTPUT_DIR / "extraction"
    results = {}
    for seed in seeds:
        path = extraction_dir / f"seed{seed}_{METHOD}_per_organoid.json"
        if path.exists():
            with open(path) as f:
                results[seed] = json.load(f)
        else:
            print(f"  Seed {seed}: not found at {path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description=f"Analyze {METHOD} test performance")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--metric", choices=["mse", "ssim"], default="mse")
    args = parser.parse_args()

    print(f"Analyzing {METHOD} across seeds {args.seeds}")

    seed_results = load_seed_results(args.seeds)
    if not seed_results:
        print(f"\nERROR: No {METHOD} results found.")
        print(f"To train {METHOD} models:")
        print(f"  python automated_method/train_model.py --method {METHOD} --seed 1 -K 6 -N 6")
        print(f"\nTo extract test predictions:")
        print(f"  python automated_method/test_model.py --method {METHOD} --seeds 1 2 3 ...")
        print(f"\nExpected path pattern:")
        print(f"  {OUTPUT_DIR / 'extraction' / f'seed*_{METHOD}_per_organoid.json'}")
        sys.exit(1)

    print(f"  Loaded {len(seed_results)} seeds")

    # Aggregate per organoid
    organoid_agg = defaultdict(lambda: defaultdict(list))
    metric_keys = ["model_mse", "model_ssim", "copy_mse", "copy_ssim",
                    "optical_flow_mse", "optical_flow_ssim"]

    for seed, data in seed_results.items():
        for org, metrics in data["organoids"].items():
            for key in metric_keys:
                if key in metrics:
                    organoid_agg[org][key].append(metrics[key])

    rows = []
    for org in sorted(organoid_agg.keys()):
        agg = organoid_agg[org]
        row = {"batch_name": org, "n_seeds": len(agg.get("model_mse", []))}
        for key in metric_keys:
            vals = agg.get(key, [])
            if vals:
                row[f"{key}_mean"] = round(float(np.mean(vals)), 8)
                row[f"{key}_std"] = round(float(np.std(vals)), 8)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"predrnn_performance_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} ({len(df)} organoids)")

    # Baseline comparison plot
    if "model_mse_mean" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        cols = [("model_mse_mean", f"{METHOD}"), ("copy_mse_mean", "Copy-last")]
        cols = [(c, l) for c, l in cols if c in df.columns]

        means = [df[c].mean() for c, _ in cols]
        stds = [df[c].std() for c, _ in cols]
        x = np.arange(len(cols))
        ax.bar(x, means, yerr=stds, color=["#ff7f0e", "#d62728"][:len(cols)],
               alpha=0.7, capsize=4, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([l for _, l in cols], fontsize=11)
        ax.set_ylabel("Mean MSE", fontsize=11)
        ax.set_yscale("log")

        fig.tight_layout()
        fig_path = OUTPUT_DIR / "predrnn_baseline_comparison.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fig_path}")

    print(f"\n{METHOD} analysis complete.")


if __name__ == "__main__":
    main()
