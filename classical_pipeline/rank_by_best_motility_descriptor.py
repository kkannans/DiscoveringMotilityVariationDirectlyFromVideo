"""
rank_by_best_motility_descriptor.py — Rank organoids by the best classical descriptor.

Uses the best discriminator identified by find_most_reliable_classical_discriminator.py
(or a user-specified descriptor) to produce a ranked ordering of organoids from
least to most motile/active.

This ranking serves as the classical baseline for comparison against the
model-based prediction surprise ranking from automated_method/.

Usage:
    python rank_by_best_motility_descriptor.py                  # use best auto-detected
    python rank_by_best_motility_descriptor.py --descriptor mean_velocity
    python rank_by_best_motility_descriptor.py --descriptor msd_slope --ascending

Inputs:
    CLASSICAL_DIR / motility_descriptors.csv
    CLASSICAL_DIR / descriptor_reliability_report.json  (optional, for auto-detect)

Outputs -> CLASSICAL_DIR /
    classical_ranking.csv          Ranked organoids with descriptor value
    classical_ranking.png          Ranked bar chart
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import paths


def main():
    parser = argparse.ArgumentParser(
        description="Rank organoids by best classical motility descriptor")
    parser.add_argument("--descriptor", type=str, default=None,
                        help="Descriptor to rank by (default: auto-detect best)")
    parser.add_argument("--ascending", action="store_true",
                        help="Rank ascending instead of descending")
    args = parser.parse_args()

    csv_path = paths.CLASSICAL_DIR / "motility_descriptors.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run extract_motility_features.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Determine which descriptor to use
    descriptor = args.descriptor
    if descriptor is None:
        report_path = paths.CLASSICAL_DIR / "descriptor_reliability_report.json"
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
            descriptor = report.get("best_discriminator", "mean_velocity")
            print(f"Auto-detected best descriptor: {descriptor}")
        else:
            descriptor = "mean_velocity"
            print(f"No reliability report found, defaulting to: {descriptor}")

    if descriptor not in df.columns:
        print(f"ERROR: '{descriptor}' not found in CSV. "
              f"Available: {[c for c in df.columns if c != 'batch_name']}")
        sys.exit(1)

    # Rank organoids
    df_valid = df[["batch_name", descriptor]].dropna()
    df_ranked = df_valid.sort_values(descriptor, ascending=args.ascending).reset_index(drop=True)
    df_ranked["rank"] = range(1, len(df_ranked) + 1)
    df_ranked["percentile"] = df_ranked["rank"] / len(df_ranked) * 100

    # Save ranking
    out_csv = paths.CLASSICAL_DIR / "classical_ranking.csv"
    df_ranked.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv} ({len(df_ranked)} organoids)")

    # Plot ranked bar chart
    fig, ax = plt.subplots(figsize=(14, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_ranked)))
    ax.bar(range(len(df_ranked)), df_ranked[descriptor].values,
           color=colors, edgecolor="none", width=1.0)
    ax.set_xlabel(f"Organoid rank (by {descriptor})", fontsize=11)
    ax.set_ylabel(descriptor.replace("_", " ").title(), fontsize=11)
    ax.set_xlim(-0.5, len(df_ranked) - 0.5)

    fig.tight_layout()
    fig_path = paths.CLASSICAL_DIR / "classical_ranking.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # Print summary
    vals = df_ranked[descriptor].values
    print(f"\nRanking by: {descriptor} ({'ascending' if args.ascending else 'descending'})")
    print(f"  N = {len(df_ranked)}")
    print(f"  Range: [{vals.min():.4f}, {vals.max():.4f}]")
    print(f"  Mean: {vals.mean():.4f}, Std: {vals.std():.4f}")
    print(f"  Top 5: {list(df_ranked['batch_name'].head())}")
    print(f"  Bottom 5: {list(df_ranked['batch_name'].tail())}")


if __name__ == "__main__":
    main()
