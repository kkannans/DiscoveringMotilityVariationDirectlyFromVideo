"""
export_frames.py — Export individual video frames as full-resolution PNGs.

Saves frames as {batch}_frame_{idx}.png at native 706x706 resolution
for manual figure assembly.

Usage:
    # Export 4 frames (0, 40, 80, 119) from all exemplar organoids:
    python analysis/export_frames.py

    # Export specific batches:
    python analysis/export_frames.py --batches batch-000121 batch-000191

    # Export specific frame indices:
    python analysis/export_frames.py --frames 0 20 40 60 80 100 119

    # Export from all 108 organoids:
    python analysis/export_frames.py --all

Outputs -> FIGURES_DIR / exported_frames /
    {batch}_frame_{idx}.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths

OUTPUT_DIR = paths.FIGURES_DIR / "exported_frames"
VIDEO_DIR = paths.DATA_ROOT
CLASSICAL_CSV = paths.CLASSICAL_DIR / "motility_descriptors.csv"
CENTROID_NPZ = paths.CLASSICAL_DIR / "centroid_trajectories.npz"


def export_frames(batch, frame_indices, output_dir):
    """Export specific frames from a video as full-res PNGs."""
    video_path = VIDEO_DIR / f"{batch}.mp4"
    if not video_path.exists():
        print(f"  WARNING: {video_path} not found, skipping")
        return 0

    cap = cv2.VideoCapture(str(video_path))
    exported = 0
    idx_set = set(frame_indices)
    n = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if n in idx_set:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_dir = output_dir / batch
            batch_dir.mkdir(parents=True, exist_ok=True)
            out_path = batch_dir / f"frame_{n:03d}.png"
            cv2.imwrite(str(out_path), frame)  # saves as BGR (OpenCV native)
            exported += 1
        if n > max(frame_indices):
            break
        n += 1

    cap.release()
    return exported


def select_exemplars():
    """Select 4 exemplar organoids spanning the velocity range."""
    if not CLASSICAL_CSV.exists():
        return None
    mot = pd.read_csv(CLASSICAL_CSV)
    cdata = np.load(CENTROID_NPZ) if CENTROID_NPZ.exists() else {}

    ms = mot.sort_values("mean_velocity").reset_index(drop=True)
    n = len(ms)
    tgts = [0, int(n * 0.25), int(n * 0.50), n - 1]
    res = []
    for idx in tgts:
        for off in range(15):
            for sg in [0, 1, -1]:
                i = min(max(idx + sg * off, 0), n - 1)
                b = ms.iloc[i]["batch_name"]
                if b in cdata and np.sum(~np.isnan(cdata[b][:120, 0])) > 10:
                    res.append(b)
                    break
            if len(res) == tgts.index(idx) + 1:
                break
    return res


def main():
    parser = argparse.ArgumentParser(
        description="Export video frames as full-resolution PNGs")
    parser.add_argument("--batches", nargs="+", default=None,
                        help="Batch names to export (default: 4 exemplars)")
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 40, 80, 119],
                        help="Frame indices to export (default: 0 40 80 119)")
    parser.add_argument("--all", action="store_true",
                        help="Export from all 108 organoids")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.batches:
        batches = args.batches
    elif args.all:
        batches = sorted([p.stem for p in VIDEO_DIR.glob("batch-*.mp4")])
    else:
        batches = select_exemplars()
        if not batches:
            batches = sorted([p.stem for p in VIDEO_DIR.glob("batch-*.mp4")])[:4]
        print(f"Auto-selected exemplars: {batches}")

    print(f"Exporting frames {args.frames} from {len(batches)} organoids")
    print(f"Output: {OUTPUT_DIR}\n")

    total = 0
    for batch in batches:
        n = export_frames(batch, args.frames, OUTPUT_DIR)
        print(f"  {batch}: {n} frames exported")
        total += n

    print(f"\nDone. {total} PNGs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
