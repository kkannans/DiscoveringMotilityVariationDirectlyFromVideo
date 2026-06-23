"""Assemble 108 organoid videos into grid MP4s (12 cols × 9 rows, 128×128 each)."""

import cv2
import numpy as np
import glob
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths

DATA_DIR = str(paths.DATA_ROOT)              # dataset/batch-*.mp4
SEG_DIR = str(paths.SEGMENTATION_DIR)        # <batch>/overlay.mp4 (from autotune_segmentation --all)
OUT_DIR = str(paths.SUPPLEMENTARY_DIR)       # grid MP4s land here
CELL = 128
COLS, ROWS = 12, 9  # 12×9 = 108
N_FRAMES = 120
FPS = 10.0
# Scale bar: 141 original pixels = 1mm; at 128/706 downsample → ~25.6 grid pixels
SCALE_BAR_PX = round(141 * (CELL / 706))  # 26 px
SCALE_BAR_THICKNESS = 3
SCALE_BAR_MARGIN = 15  # px from bottom-right corner


def get_sorted_batches():
    """Get sorted list of batch names from dataset directory."""
    vids = sorted(glob.glob(os.path.join(DATA_DIR, "batch-*.mp4")))
    return [os.path.basename(p).replace(".mp4", "") for p in vids]


def make_grid_video(output_path, video_paths, n_frames, fps, src_label=""):
    """Read n_frames from each video, resize to 128×128, write grid MP4."""
    n_videos = len(video_paths)
    assert n_videos == COLS * ROWS, f"Expected {COLS*ROWS} videos, got {n_videos}"

    W, H = COLS * CELL, ROWS * CELL
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    # Open all captures
    caps = []
    for p in video_paths:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            print(f"WARNING: Cannot open {p}")
        caps.append(cap)

    for fi in range(n_frames):
        if fi % 20 == 0:
            print(f"  {src_label} frame {fi}/{n_frames}")
        grid = np.zeros((H, W, 3), dtype=np.uint8)
        for idx, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                continue  # black cell if video shorter
            frame = cv2.resize(frame, (CELL, CELL), interpolation=cv2.INTER_AREA)
            r, c = divmod(idx, COLS)
            y, x = r * CELL, c * CELL
            grid[y:y+CELL, x:x+CELL] = frame
        # Draw 1mm scale bar (black) in bottom-right corner
        bar_y = H - SCALE_BAR_MARGIN
        bar_x_end = W - SCALE_BAR_MARGIN
        bar_x_start = bar_x_end - SCALE_BAR_PX
        cv2.line(grid, (bar_x_start, bar_y), (bar_x_end, bar_y),
                 (0, 0, 0), SCALE_BAR_THICKNESS)
        cv2.putText(grid, "1 mm", (bar_x_start, bar_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        writer.write(grid)

    for cap in caps:
        cap.release()
    writer.release()
    print(f"  Wrote {output_path} ({W}x{H}, {n_frames} frames, {fps} fps)")


def main():
    batches = get_sorted_batches()
    print(f"Found {len(batches)} batches")

    os.makedirs(OUT_DIR, exist_ok=True)
    # 1. dataset.mp4 — raw videos from dataset/
    dataset_paths = [os.path.join(DATA_DIR, f"{b}.mp4") for b in batches]
    out1 = os.path.join(OUT_DIR, "dataset.mp4")
    print(f"\n=== Building dataset.mp4 ===")
    make_grid_video(out1, dataset_paths, N_FRAMES, FPS, src_label="dataset")

    # 2. dataset_with_segmentation.mp4 — overlay videos (first 120 frames)
    seg_paths = [os.path.join(SEG_DIR, b, "overlay.mp4") for b in batches]
    # Verify all exist
    missing = [p for p in seg_paths if not os.path.exists(p)]
    if missing:
        print(f"WARNING: {len(missing)} overlay videos missing:")
        for m in missing[:5]:
            print(f"  {m}")
    out2 = os.path.join(OUT_DIR, "dataset_with_segmentation.mp4")
    print(f"\n=== Building dataset_with_segmentation.mp4 ===")
    make_grid_video(out2, seg_paths, N_FRAMES, FPS, src_label="segmentation")


if __name__ == "__main__":
    main()
