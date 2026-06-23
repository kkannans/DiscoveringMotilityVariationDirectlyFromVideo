"""
Fig (spatial error): Size-matched organoids with different prediction surprise.

Standalone figure (split from old combined Fig 4 Panel B).
CPU-only — no GPU needed.

Row 1: Cropped frame with segmentation contour — similar morphology.
Row 2: Spatial prediction surprise heatmap + centroid trajectory — different dynamics.

3 organoids selected from a narrow size band (~200 px) with low/med/high surprise.

Outputs -> paths.FIGURES_DIR / fig_spatial_error.{png,pdf}
"""

import json
import sys
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "classical_pipeline"))
import paths
from autotune_segmentation import segment_frame

OUTPUT_DIR = paths.FIGURES_DIR
CLASSICAL_CSV = paths.CLASSICAL_DIR / "motility_descriptors.csv"
SUMMARY_CSV = paths.INTERMEDIATES_DIR / "per_organoid_summary.csv"
SPATIAL_DIR = paths.INTERMEDIATES_DIR / "spatial_residual_maps"
CENTROID_PATH = paths.CLASSICAL_DIR / "centroid_trajectories.npz"
VIDEO_DIR = paths.DATA_ROOT
SELECTED_JSON = paths.INTERMEDIATES_DIR / "selected_spatial_organoids.json"
AUTOTUNE_DIR = paths.SEGMENTATION_DIR


def load_video_frame(batch_name, frame_idx=2, size=128):
    """Load a single frame (0-indexed) as float32 RGB [0,1], resized."""
    cap = cv2.VideoCapture(str(VIDEO_DIR / f"{batch_name}.mp4"))
    for _ in range(frame_idx + 1):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return np.zeros((size, size, 3), dtype=np.float32)
    cap.release()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    return frame.astype(np.float32) / 255.0


def load_autotune_params(batch_name):
    params_path = AUTOTUNE_DIR / batch_name / "best_params.json"
    with open(params_path) as f:
        return json.load(f)


def get_segmentation_overlay(batch_name, frame_idx=2, size=128):
    """Return (frame_float, mask) with auto-tuned segmentation."""
    frame_float = load_video_frame(batch_name, frame_idx, size)
    frame_uint8 = (frame_float * 255).astype(np.uint8)
    params = load_autotune_params(batch_name)
    mask, _, _, _ = segment_frame(frame_uint8, params)
    mask = (mask > 0).astype(np.uint8)
    return frame_float, mask


def load_centroids(batch_name):
    """Load (T, 2) centroid trajectory."""
    data = np.load(CENTROID_PATH, allow_pickle=True)
    if batch_name in data:
        c = data[batch_name]
        if np.isfinite(c).all():
            return c
    return None


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

    # Pre-load segmentation data
    seg_data = {}
    for batch, label in zip(sel_batches, sel_labels):
        frame_float, mask = get_segmentation_overlay(batch, frame_idx=2)
        seg_data[batch] = (frame_float, mask)
        print(f"{label} ({batch}): mask size = {mask.sum()} px")

    # ── Figure: 2 rows x 3 cols (+ colorbar) ──
    fig = plt.figure(figsize=(8, 5.5))
    gs = GridSpec(2, 4, width_ratios=[1, 1, 1, 0.06], hspace=0.15, wspace=0.08,
                  figure=fig)

    # ── Row 1: Organoid frames with segmentation contour ──
    for col, (batch, label, color) in enumerate(
            zip(sel_batches, sel_labels, sel_colors)):
        ax = fig.add_subplot(gs[0, col])
        frame_float, mask = seg_data[batch]
        ax.imshow(frame_float)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            c = c.squeeze()
            if c.ndim == 2 and len(c) > 2:
                ax.plot(np.append(c[:, 0], c[0, 0]),
                        np.append(c[:, 1], c[0, 1]),
                        color=color, linewidth=2)

        row = df[df.batch_name == batch].iloc[0]
        ax.set_title(f"{label} (Org {selected[label.lower()]['index']})\n"
                     f"Size: {row['area_px']:.0f} px",
                     fontsize=10, fontweight="bold", color=color)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    # ── Row 2: Prediction surprise heatmaps with trajectory ──
    maps = []
    centroids_list = []
    valid_info = []
    for batch, label, color in zip(sel_batches, sel_labels, sel_colors):
        sp = SPATIAL_DIR / f"{batch}.npy"
        if sp.exists():
            maps.append(np.load(sp).mean(axis=0))
            centroids_list.append(load_centroids(batch))
            row = df[df.batch_name == batch].iloc[0]
            valid_info.append((batch, label, color, row["mean_motion_loss"],
                               row["area_px"], row["mean_velocity"]))

    # Shared log-scale normalization
    all_vals = np.concatenate([m.ravel() for m in maps])
    vmin = max(all_vals[all_vals > 0].min(), 1e-8)
    vmax = all_vals.max()
    norm = LogNorm(vmin=vmin, vmax=vmax)

    im = None
    for col, (smap, centroids, (batch, label, color, loss, area, vel)) in enumerate(
            zip(maps, centroids_list, valid_info)):
        ax = fig.add_subplot(gs[1, col])

        # Layer 1: First frame as background
        first_frame = load_video_frame(batch, frame_idx=0)
        ax.imshow(first_frame, alpha=0.6, interpolation="nearest")

        # Layer 2: Residual heatmap
        smap_clipped = np.where(smap > 0, smap, vmin)
        im = ax.imshow(smap_clipped, cmap="inferno", norm=norm, alpha=0.5,
                        interpolation="nearest")

        # Layer 3: Centroid trajectory
        if centroids is not None:
            cx, cy = centroids[:, 0], centroids[:, 1]
            ax.plot(cx, cy, "-", color="cyan", linewidth=1.5, alpha=0.8)
            ax.scatter(cx[0], cy[0], c="white", marker="o", s=45,
                       edgecolors="black", linewidth=1.0, zorder=10)
            ax.scatter(cx[-1], cy[-1], c="black", marker="X", s=50,
                       edgecolors="white", linewidth=0.6, zorder=10)
            n_tail = min(5, len(cx) - 1)
            if n_tail > 0:
                ax.annotate("", xy=(cx[-1], cy[-1]),
                            xytext=(cx[-1 - n_tail], cy[-1 - n_tail]),
                            arrowprops=dict(arrowstyle="-|>", color="cyan",
                                            lw=1.5, mutation_scale=12))

        ax.set_xlabel(f"Surprise: {loss:.2e}", fontsize=9, color=color,
                      fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 127)
        ax.set_ylim(127, 0)

    # Colorbar spanning row 2
    cbar_ax = fig.add_subplot(gs[1, 3])
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.set_ylabel("Prediction surprise", fontsize=10)

    # ── Save ──
    for ext in ["png", "pdf"]:
        out = OUTPUT_DIR / f"fig_spatial_error.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'fig_spatial_error.png'}")

    # Save stats
    with open(OUTPUT_DIR / "fig_spatial_error_results.json", "w") as f:
        json.dump({
            "selected_organoids": {
                label.lower(): {
                    "batch": batch,
                    "size_px": float(area),
                    "mean_velocity": float(vel),
                    "motion_loss": float(loss),
                }
                for batch, label, _, loss, area, vel in valid_info
            },
        }, f, indent=2)


if __name__ == "__main__":
    main()
