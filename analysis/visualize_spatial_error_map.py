"""
visualize_spatial_error_map.py — Visualize per-pixel prediction error heatmaps.

For selected organoids, overlays the spatial distribution of prediction error
(|Δ_GT - Δ_pred|²) on the video frame to show WHERE the model fails to predict
motion, revealing organoid-specific spatial dynamics patterns.

Usage:
    python visualize_spatial_error_map.py                         # auto-select exemplars
    python visualize_spatial_error_map.py --batches batch-000121 batch-000191
    python visualize_spatial_error_map.py --top-n 6               # top/bottom N

Inputs:
    INTERMEDIATES_DIR / spatial_residual_maps / *.npy    Per-pixel error maps (N, 128, 128)
    INTERMEDIATES_DIR / per_organoid_summary.csv         Organoid rankings
    DATA_ROOT / batch-*.mp4                              Raw videos (for frame overlay)

Outputs -> FIGURES_DIR /
    spatial_error_grid.png           Grid of selected organoid error maps
    spatial_error_{batch}.png        Individual error map overlays
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import paths


def load_frame(batch_name, frame_idx=0, size=128):
    """Load a single video frame as (H, W, 3) float [0,1]."""
    video_path = paths.DATA_ROOT / f"{batch_name}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    for _ in range(frame_idx):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return np.zeros((size, size, 3), dtype=np.float32)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    return frame.astype(np.float32) / 255.0


def plot_error_overlay(batch_name, spatial_map, output_dir, vmin=None, vmax=None):
    """Plot spatial error map overlaid on video frame."""
    frame = load_frame(batch_name)
    mean_map = spatial_map.mean(axis=0)  # average over horizons

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel 1: Raw frame
    axes[0].imshow(frame)
    axes[0].set_xlabel("Raw frame (t=0)")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Panel 2: Error heatmap
    if vmin is None:
        vmin = max(mean_map[mean_map > 0].min(), 1e-8) if (mean_map > 0).any() else 1e-8
    if vmax is None:
        vmax = mean_map.max()
    norm = LogNorm(vmin=vmin, vmax=vmax)
    im = axes[1].imshow(np.where(mean_map > 0, mean_map, vmin),
                        cmap="inferno", norm=norm)
    axes[1].set_xlabel("Spatial error (log scale)")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    # Panel 3: Overlay
    axes[2].imshow(frame, alpha=0.5)
    axes[2].imshow(np.where(mean_map > 0, mean_map, vmin),
                   cmap="inferno", norm=norm, alpha=0.5)
    axes[2].set_xlabel("Overlay")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    fig.suptitle(batch_name, fontsize=11)
    fig.tight_layout()
    out_path = output_dir / f"spatial_error_{batch_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_grid(batches, spatial_maps, output_dir):
    """Plot a grid of error maps for selected organoids."""
    n = len(batches)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    # Shared normalization
    all_vals = np.concatenate([m.mean(axis=0).ravel() for m in spatial_maps])
    vmin = max(all_vals[all_vals > 0].min(), 1e-8) if (all_vals > 0).any() else 1e-8
    vmax = all_vals.max()
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx in range(nrows * ncols):
        ax = axes[idx // ncols, idx % ncols]
        if idx < n:
            frame = load_frame(batches[idx])
            mean_map = spatial_maps[idx].mean(axis=0)
            ax.imshow(frame, alpha=0.5)
            im = ax.imshow(np.where(mean_map > 0, mean_map, vmin),
                           cmap="inferno", norm=norm, alpha=0.5)
            ax.set_xlabel(batches[idx].replace("batch-", ""), fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    out_path = output_dir / "spatial_error_grid.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize spatial prediction error maps")
    parser.add_argument("--batches", nargs="+", default=None,
                        help="Specific batches to visualize")
    parser.add_argument("--top-n", type=int, default=3,
                        help="Show top and bottom N by prediction error")
    args = parser.parse_args()

    spatial_dir = paths.INTERMEDIATES_DIR / "spatial_residual_maps"
    if not spatial_dir.exists():
        print(f"ERROR: {spatial_dir} not found.")
        print("Run: python automated_method/extract_prediction_surprise.py")
        sys.exit(1)

    output_dir = paths.FIGURES_DIR / "spatial_error_maps"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batches:
        batches = args.batches
    else:
        # Auto-select: top and bottom N by prediction error
        summary_path = paths.INTERMEDIATES_DIR / "per_organoid_summary.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            df_sorted = df.sort_values("mean_motion_loss")
            low = list(df_sorted.head(args.top_n)["batch_name"])
            high = list(df_sorted.tail(args.top_n)["batch_name"])
            batches = low + high
            print(f"Auto-selected {len(batches)} organoids "
                  f"({args.top_n} lowest + {args.top_n} highest error)")
        else:
            # Fall back to all available
            batches = sorted([p.stem for p in spatial_dir.glob("*.npy")])[:6]

    # Load spatial maps
    valid_batches = []
    spatial_maps = []
    for b in batches:
        npy_path = spatial_dir / f"{b}.npy"
        if npy_path.exists():
            spatial_maps.append(np.load(npy_path))
            valid_batches.append(b)
        else:
            print(f"  WARNING: {b} not found in spatial maps, skipping")

    if not valid_batches:
        print("ERROR: No valid spatial maps found")
        sys.exit(1)

    # Individual plots
    for b, sm in zip(valid_batches, spatial_maps):
        out = plot_error_overlay(b, sm, output_dir)
        print(f"  Saved: {out}")

    # Grid plot
    if len(valid_batches) >= 2:
        plot_grid(valid_batches, spatial_maps, output_dir)

    print(f"\nDone. {len(valid_batches)} organoids visualized.")


if __name__ == "__main__":
    main()
