"""
visualize_grid_centroids.py — Visualize centroid trajectories for all organoids.

Creates a grid of centroid trajectory plots, one per organoid, showing
the path traced by the organoid centroid over 120 frames.
Useful for quality-checking segmentation and identifying outlier trajectories.

Usage:
    python visualize_grid_centroids.py                          # all organoids
    python visualize_grid_centroids.py --batch batch-000121     # single organoid
    python visualize_grid_centroids.py --color-by velocity      # color by speed

Inputs:
    CLASSICAL_DIR / centroid_trajectories.npz    Centroid data (T, 2) per organoid
    CLASSICAL_DIR / motility_descriptors.csv     Motility metrics (for coloring)

Outputs -> FIGURES_DIR/
    centroid_grid.png              Grid of all centroid trajectories
    centroid_detail_{batch}.png    Detailed single-organoid centroid plot
"""

import argparse
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


def load_all_centroids():
    """Load centroid trajectories for all organoids.

    Returns:
        dict: batch_name -> (T, 2) array of (cx, cy) centroids.
    """
    npz_path = paths.CLASSICAL_DIR / "centroid_trajectories.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Centroid file not found: {npz_path}\n"
            "Run: python classical_pipeline/extract_motility_features.py first."
        )
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def plot_single_centroid(ax, centroids, batch_name, color_by_time=True):
    """Plot a single centroid trajectory on an axes."""
    valid = np.isfinite(centroids[:, 0])
    if valid.sum() < 2:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        return

    cx, cy = centroids[valid, 0], centroids[valid, 1]

    if color_by_time:
        t = np.arange(valid.sum())
        ax.scatter(cx, cy, c=t, cmap="coolwarm", s=2, alpha=0.7)
        ax.plot(cx, cy, "-", color="gray", linewidth=0.3, alpha=0.3)
    else:
        ax.plot(cx, cy, "-", color="steelblue", linewidth=0.5, alpha=0.7)

    # Mark start and end
    ax.scatter(cx[0], cy[0], c="lime", s=20, edgecolors="black",
               linewidth=0.5, zorder=10)
    ax.scatter(cx[-1], cy[-1], c="red", s=20, edgecolors="black",
               linewidth=0.5, zorder=10)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])


def plot_grid(output_dir):
    """Create a grid visualization of all centroid trajectories."""
    centroids = load_all_centroids()
    batches = sorted(centroids.keys())
    n = len(batches)

    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))
    axes = np.atleast_2d(axes)

    for idx in range(nrows * ncols):
        ax = axes[idx // ncols, idx % ncols]
        if idx < n:
            batch = batches[idx]
            plot_single_centroid(ax, centroids[batch], batch)
            ax.set_xlabel(batch.replace("batch-", ""), fontsize=5)
        else:
            ax.axis("off")

    fig.tight_layout()
    out_path = output_dir / "centroid_grid.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path} ({n} organoids)")


def plot_detail(batch_name, output_dir):
    """Create a detailed centroid trajectory plot for one organoid."""
    centroids = load_all_centroids()
    if batch_name not in centroids:
        print(f"ERROR: {batch_name} not found in centroid data")
        return

    c = centroids[batch_name]
    valid = np.isfinite(c[:, 0])
    cx, cy = c[valid, 0], c[valid, 1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Trajectory colored by time
    ax = axes[0]
    t = np.arange(valid.sum())
    sc = ax.scatter(cx, cy, c=t, cmap="coolwarm", s=10, alpha=0.8)
    ax.plot(cx, cy, "-", color="gray", linewidth=0.3, alpha=0.3)
    ax.scatter(cx[0], cy[0], c="lime", s=60, edgecolors="black", linewidth=1, zorder=10)
    ax.scatter(cx[-1], cy[-1], c="red", s=60, edgecolors="black", linewidth=1, zorder=10)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    plt.colorbar(sc, ax=ax, label="Frame")

    # Panel 2: Velocity over time
    ax = axes[1]
    velocities = np.sqrt(np.sum(np.diff(c[valid], axis=0) ** 2, axis=1))
    ax.plot(velocities, color="steelblue", linewidth=0.8)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Velocity (px/frame)")
    ax.axhline(np.mean(velocities), color="red", linestyle="--",
               linewidth=0.8, label=f"Mean: {np.mean(velocities):.2f}")
    ax.legend(fontsize=8)

    # Panel 3: MSD
    ax = axes[2]
    max_lag = min(50, valid.sum() // 2)
    msd = []
    for lag in range(1, max_lag):
        d = c[valid][lag:] - c[valid][:-lag]
        sq = np.sum(d ** 2, axis=1)
        msd.append(np.mean(sq))
    ax.loglog(range(1, max_lag), msd, "o-", markersize=3, color="steelblue")
    ax.set_xlabel("Lag (frames)")
    ax.set_ylabel("MSD (px²)")

    fig.suptitle(batch_name, fontsize=12)
    fig.tight_layout()
    out_path = output_dir / f"centroid_detail_{batch_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize centroid trajectories")
    parser.add_argument("--batch", type=str, default=None,
                        help="Single batch (default: grid of all)")
    args = parser.parse_args()

    output_dir = paths.FIGURES_DIR / "centroid_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch:
        plot_detail(args.batch, output_dir)
    else:
        plot_grid(output_dir)


if __name__ == "__main__":
    main()
