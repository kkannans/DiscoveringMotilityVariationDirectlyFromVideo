"""Build a 4x4 grid of scaled panel-A organoid frames for Figure 1.

Layout:
  rows = 4 batches (slow -> fast by velocity)
  cols = 4 frames per batch [0, 40, 80, 119]
  white gutters between cells

Input:  outputs/figures/exported_frames/2x2_scaled/batch-*/frame_*.png
Output: outputs/figures/Fig1/panel_A_4x4.png
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths

INPUT_DIR = paths.FIGURES_DIR / "exported_frames" / "2x2_scaled"
OUTPUT_PATH = paths.FIGURES_DIR / "Fig1" / "panel_A_4x4.png"

# Row order: slowest to fastest (by mean_velocity)
ROWS = [
    "batch-000112",   # vel 0.04
    "batch-000235",   # vel 0.18
    "batch-000193",   # vel 1.44
    "batch-000227",   # vel 19.31
]
COLS = [0, 40, 80, 119]
GUTTER = 20      # px between cells
BG = 255         # white background


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load first image to get cell size
    first = cv2.imread(str(INPUT_DIR / ROWS[0] / f"frame_{COLS[0]:03d}.png"))
    if first is None:
        print(f"ERROR: cannot read {INPUT_DIR / ROWS[0]}")
        sys.exit(1)
    cell_h, cell_w = first.shape[:2]
    print(f"Cell size: {cell_w}x{cell_h}")

    n_rows, n_cols = len(ROWS), len(COLS)
    grid_h = n_rows * cell_h + (n_rows + 1) * GUTTER
    grid_w = n_cols * cell_w + (n_cols + 1) * GUTTER
    grid = np.full((grid_h, grid_w, 3), BG, dtype=np.uint8)

    for ri, batch in enumerate(ROWS):
        for ci, frame_idx in enumerate(COLS):
            fp = INPUT_DIR / batch / f"frame_{frame_idx:03d}.png"
            img = cv2.imread(str(fp))
            if img is None:
                print(f"  WARNING: missing {fp}")
                continue
            y0 = GUTTER + ri * (cell_h + GUTTER)
            x0 = GUTTER + ci * (cell_w + GUTTER)
            grid[y0:y0 + cell_h, x0:x0 + cell_w] = img

    cv2.imwrite(str(OUTPUT_PATH), grid)
    print(f"\nSaved: {OUTPUT_PATH} ({grid_w}x{grid_h})")


if __name__ == "__main__":
    main()
