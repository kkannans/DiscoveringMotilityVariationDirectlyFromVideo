"""
visualize_flow_fields.py — Visualize optical flow and motion fields from organoid videos.

Computes and visualizes optical flow between consecutive frames, showing
the motion patterns the model must learn to predict. Useful for understanding
why certain organoids are harder to predict than others.

Usage:
    python visualize_flow_fields.py --batch batch-000121          # single organoid
    python visualize_flow_fields.py --batch batch-000121 --mode quiver
    python visualize_flow_fields.py --compare batch-000121 batch-000191

Inputs:
    DATA_ROOT / batch-*.mp4    Raw organoid videos

Outputs -> FIGURES_DIR / flow_fields /
    flow_hsv_{batch}.png       HSV-encoded flow visualization
    flow_quiver_{batch}.png    Quiver plot of flow vectors
    flow_comparison.png        Side-by-side comparison of two organoids
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import paths


def load_video_frames(batch_name, num_frames=120, size=128):
    """Load video as (T, H, W, 3) uint8."""
    video_path = paths.DATA_ROOT / f"{batch_name}.mp4"
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    return np.stack(frames) if frames else None


def compute_optical_flow(frame1_bgr, frame2_bgr):
    """Compute dense optical flow between two BGR frames.

    Returns:
        flow: (H, W, 2) flow field (dx, dy)
        magnitude: (H, W) flow magnitude
        angle: (H, W) flow direction in radians
    """
    gray1 = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
    angle = np.arctan2(flow[:, :, 1], flow[:, :, 0])
    return flow, magnitude, angle


def flow_to_hsv(flow, magnitude):
    """Convert flow to HSV visualization."""
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    angle = np.arctan2(flow[:, :, 1], flow[:, :, 0])
    hsv[:, :, 0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    hsv[:, :, 1] = 255
    mag_norm = np.clip(magnitude / (magnitude.max() + 1e-8) * 255, 0, 255)
    hsv[:, :, 2] = mag_norm.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def plot_flow_hsv(batch_name, output_dir, frame_pairs=None, num_frames=120):
    """Create HSV flow visualization for an organoid."""
    frames = load_video_frames(batch_name, num_frames=num_frames)
    if frames is None or len(frames) < 2:
        print(f"  WARNING: Could not load video for {batch_name}")
        return

    if frame_pairs is None:
        # Sample 6 evenly spaced pairs
        T = len(frames)
        indices = np.linspace(0, T - 2, 6, dtype=int)
        frame_pairs = [(i, i + 1) for i in indices]

    n = len(frame_pairs)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))

    for col, (i, j) in enumerate(frame_pairs):
        flow, mag, angle = compute_optical_flow(frames[i], frames[j])
        hsv_img = flow_to_hsv(flow, mag)

        # Top row: original frame
        rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        axes[0, col].imshow(rgb)
        axes[0, col].set_xlabel(f"Frame {i}", fontsize=8)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        # Bottom row: flow HSV
        axes[1, col].imshow(hsv_img)
        axes[1, col].set_xlabel(f"Flow {i}→{j}\nmax={mag.max():.2f}", fontsize=8)
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

    axes[0, 0].set_ylabel("Video frame")
    axes[1, 0].set_ylabel("Optical flow (HSV)")
    fig.suptitle(batch_name, fontsize=11)
    fig.tight_layout()
    out_path = output_dir / f"flow_hsv_{batch_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_flow_quiver(batch_name, output_dir, frame_idx=60, stride=4, num_frames=120):
    """Create quiver plot of flow vectors."""
    frames = load_video_frames(batch_name, num_frames=num_frames)
    if frames is None or len(frames) <= frame_idx + 1:
        return

    flow, mag, _ = compute_optical_flow(frames[frame_idx], frames[frame_idx + 1])
    H, W = flow.shape[:2]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left: frame with quiver overlay
    rgb = cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB)
    axes[0].imshow(rgb)
    y, x = np.mgrid[0:H:stride, 0:W:stride]
    u = flow[::stride, ::stride, 0]
    v = flow[::stride, ::stride, 1]
    m = mag[::stride, ::stride]
    axes[0].quiver(x, y, u, v, m, cmap="hot", scale=50, alpha=0.8, width=0.003)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_xlabel(f"Frame {frame_idx} + flow vectors")

    # Right: magnitude heatmap
    im = axes[1].imshow(mag, cmap="hot")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_xlabel("Flow magnitude")
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    fig.suptitle(batch_name, fontsize=11)
    fig.tight_layout()
    out_path = output_dir / f"flow_quiver_{batch_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_comparison(batch1, batch2, output_dir, frame_idx=60, num_frames=120):
    """Side-by-side flow comparison of two organoids."""
    frames1 = load_video_frames(batch1, num_frames=num_frames)
    frames2 = load_video_frames(batch2, num_frames=num_frames)
    if frames1 is None or frames2 is None:
        return
    if len(frames1) <= frame_idx + 1 or len(frames2) <= frame_idx + 1:
        return

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    for col, (batch, frames) in enumerate([(batch1, frames1), (batch2, frames2)]):
        flow, mag, _ = compute_optical_flow(frames[frame_idx], frames[frame_idx + 1])
        rgb = cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB)
        hsv_img = flow_to_hsv(flow, mag)

        axes[0, col].imshow(rgb)
        axes[0, col].set_xlabel(batch.replace("batch-", ""), fontsize=9)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

        axes[1, col].imshow(hsv_img)
        axes[1, col].set_xlabel(f"max flow = {mag.max():.2f} px", fontsize=9)
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

    axes[0, 0].set_ylabel("Video frame")
    axes[1, 0].set_ylabel("Optical flow")
    fig.tight_layout()
    out_path = output_dir / "flow_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize optical flow fields")
    parser.add_argument("--batch", type=str, default=None,
                        help="Single batch to visualize")
    parser.add_argument("--compare", nargs=2, default=None,
                        help="Two batches to compare side-by-side")
    parser.add_argument("--mode", choices=["hsv", "quiver", "both"],
                        default="both")
    parser.add_argument("--frame", type=int, default=60,
                        help="Frame index for quiver/comparison (default: 60)")
    args = parser.parse_args()

    output_dir = paths.FIGURES_DIR / "flow_fields"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        plot_comparison(args.compare[0], args.compare[1], output_dir,
                        frame_idx=args.frame)
    elif args.batch:
        if args.mode in ("hsv", "both"):
            plot_flow_hsv(args.batch, output_dir)
        if args.mode in ("quiver", "both"):
            plot_flow_quiver(args.batch, output_dir, frame_idx=args.frame)
    else:
        print("Specify --batch or --compare. Examples:")
        print("  python visualize_flow_fields.py --batch batch-000121")
        print("  python visualize_flow_fields.py --compare batch-000121 batch-000191")


if __name__ == "__main__":
    main()
