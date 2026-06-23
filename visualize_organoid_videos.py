"""
visualize_organoid_videos.py — Visualize raw organoid video data.

Creates grid visualizations of organoid videos showing:
  - Frame montages (sampled frames across the video)
  - Trajectory overlays (centroid path on first/last frame composite)
  - Population overview (thumbnail grid of all 108 organoids)

Usage:
    python visualize_organoid_videos.py                          # all organoids, montage
    python visualize_organoid_videos.py --batch batch-000121     # single organoid
    python visualize_organoid_videos.py --mode trajectory        # centroid overlay
    python visualize_organoid_videos.py --mode grid              # population thumbnail grid

Inputs:
    DATA_ROOT / batch-*.mp4                      Raw organoid videos
    CLASSICAL_DIR / centroid_trajectories.npz    Centroid data (for trajectory mode)

Outputs -> FIGURES_DIR/
    video_montage_{batch}.png       Per-organoid frame montage
    trajectory_overlay_{batch}.png  Centroid trajectory on video
    population_grid.png             Thumbnail grid of all organoids
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import paths


def load_video_frames(video_path, num_frames=120, target_size=128):
    """Load video frames as (T, H, W, 3) uint8 array."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if target_size:
            frame = cv2.resize(frame, (target_size, target_size),
                               interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, target_size, target_size, 3), dtype=np.uint8)


def plot_montage(batch_name, output_dir, n_samples=12, num_frames=120):
    """Create a frame montage for a single organoid video."""
    video_path = paths.DATA_ROOT / f"{batch_name}.mp4"
    if not video_path.exists():
        print(f"  WARNING: {video_path} not found, skipping")
        return

    frames = load_video_frames(video_path, num_frames=num_frames)
    T = len(frames)
    if T == 0:
        return

    # Sample evenly spaced frames
    indices = np.linspace(0, T - 1, n_samples, dtype=int)
    sampled = frames[indices]

    ncols = min(6, n_samples)
    nrows = (n_samples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows))
    axes = np.atleast_2d(axes)

    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        if i < n_samples:
            ax.imshow(sampled[i])
            ax.set_xlabel(f"Frame {indices[i]}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(batch_name, fontsize=12, y=1.01)
    fig.tight_layout()
    out_path = output_dir / f"video_montage_{batch_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_trajectory_overlay(batch_name, output_dir, num_frames=120):
    """Overlay centroid trajectory on first/last frame composite."""
    video_path = paths.DATA_ROOT / f"{batch_name}.mp4"
    if not video_path.exists():
        print(f"  WARNING: {video_path} not found, skipping")
        return

    frames = load_video_frames(video_path, num_frames=num_frames)
    if len(frames) < 2:
        return

    # Composite: first frame blue, last frame red
    first = frames[0].astype(np.float32) / 255.0
    last = frames[-1].astype(np.float32) / 255.0
    gray_first = np.mean(first, axis=2)
    gray_last = np.mean(last, axis=2)
    composite = np.zeros_like(first)
    composite[:, :, 2] = gray_first  # blue = first
    composite[:, :, 0] = gray_last   # red = last
    composite[:, :, 1] = (gray_first + gray_last) * 0.25

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.clip(composite, 0, 1))

    # Load centroids if available
    centroid_path = paths.CLASSICAL_DIR / "centroid_trajectories.npz"
    if centroid_path.exists():
        data = np.load(centroid_path, allow_pickle=True)
        if batch_name in data:
            c = data[batch_name]
            valid = np.isfinite(c[:, 0])
            cx, cy = c[valid, 0], c[valid, 1]
            ax.plot(cx, cy, "-", color="cyan", linewidth=1.0, alpha=0.8)
            ax.scatter(cx[0], cy[0], c="lime", s=50, edgecolors="black",
                       linewidth=1, zorder=10, label="Start")
            ax.scatter(cx[-1], cy[-1], c="red", s=50, edgecolors="black",
                       linewidth=1, zorder=10, label="End")
            ax.legend(fontsize=8, loc="upper right")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(batch_name, fontsize=10)

    out_path = output_dir / f"trajectory_overlay_{batch_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_population_grid(output_dir, num_frames=120, thumb_size=64):
    """Create a thumbnail grid of all organoid videos (first frame)."""
    video_dir = paths.DATA_ROOT
    batches = sorted([p.stem for p in Path(video_dir).glob("batch-*.mp4")])
    if not batches:
        print("No videos found")
        return

    n = len(batches)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    grid = np.ones((nrows * thumb_size, ncols * thumb_size, 3), dtype=np.uint8) * 200

    for idx, batch in enumerate(batches):
        video_path = video_dir / f"{batch}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        cap.release()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
        r, c = idx // ncols, idx % ncols
        grid[r * thumb_size:(r + 1) * thumb_size,
             c * thumb_size:(c + 1) * thumb_size] = frame

    fig, ax = plt.subplots(figsize=(ncols * 0.8, nrows * 0.8))
    ax.imshow(grid)
    ax.set_xticks([])
    ax.set_yticks([])

    out_path = output_dir / "population_grid.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path} ({n} organoids)")


def main():
    parser = argparse.ArgumentParser(description="Visualize organoid videos")
    parser.add_argument("--batch", type=str, default=None,
                        help="Single batch name (default: all)")
    parser.add_argument("--mode", choices=["montage", "trajectory", "grid"],
                        default="montage",
                        help="Visualization mode (default: montage)")
    parser.add_argument("--num-frames", type=int, default=120,
                        help="Number of frames to use (default: 120)")
    args = parser.parse_args()

    output_dir = paths.FIGURES_DIR / "video_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "grid":
        plot_population_grid(output_dir, num_frames=args.num_frames)
        return

    # Get batch list
    if args.batch:
        batches = [args.batch]
    else:
        batches = sorted([p.stem for p in Path(paths.DATA_ROOT).glob("batch-*.mp4")])

    for batch in batches:
        if args.mode == "montage":
            plot_montage(batch, output_dir, num_frames=args.num_frames)
        elif args.mode == "trajectory":
            plot_trajectory_overlay(batch, output_dir, num_frames=args.num_frames)

    print(f"\nDone. {len(batches)} organoids processed.")


if __name__ == "__main__":
    main()
