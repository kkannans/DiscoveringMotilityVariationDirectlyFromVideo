"""Add scale bars to exported panel A frames.

Scale: 141 pixels = 1 mm at native 706x706.
Draws a horizontal white scale bar with "1 mm" label in the bottom-right.

Input:  outputs/figures/exported_frames/2x2/batch-*/frame_*.png
Output: outputs/figures/exported_frames/2x2_scaled/batch-*/frame_*.png
"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths

INPUT_DIR = paths.FIGURES_DIR / "exported_frames" / "2x2"
OUTPUT_DIR = paths.FIGURES_DIR / "exported_frames" / "2x2_scaled"

BAR_PX = 141       # 1 mm
BAR_HEIGHT = 8
MARGIN = 25
LABEL = "1 mm"


def add_scale_bar(img):
    """Draw a white scale bar + label in bottom-right corner."""
    h, w = img.shape[:2]
    x1 = w - MARGIN - BAR_PX
    x2 = w - MARGIN
    y1 = h - MARGIN - BAR_HEIGHT
    y2 = h - MARGIN

    # Black outline for visibility on any background
    cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 0), -1)
    # White bar
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)

    # Label above the bar, centered
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2
    (tw, th), _ = cv2.getTextSize(LABEL, font, scale, thickness)
    tx = x1 + (BAR_PX - tw) // 2
    ty = y1 - 8
    # Black outline
    cv2.putText(img, LABEL, (tx, ty), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    # White text
    cv2.putText(img, LABEL, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return img


def main():
    if not INPUT_DIR.exists():
        print(f"ERROR: {INPUT_DIR} not found")
        sys.exit(1)

    n_processed = 0
    for batch_dir in sorted(INPUT_DIR.glob("batch-*")):
        out_batch = OUTPUT_DIR / batch_dir.name
        out_batch.mkdir(parents=True, exist_ok=True)
        for frame_path in sorted(batch_dir.glob("frame_*.png")):
            img = cv2.imread(str(frame_path))
            if img is None:
                print(f"  WARNING: could not read {frame_path}")
                continue
            img = add_scale_bar(img)
            out_path = out_batch / frame_path.name
            cv2.imwrite(str(out_path), img)
            n_processed += 1
            print(f"  {out_path.relative_to(paths.FIGURES_DIR)}")

    print(f"\nDone. {n_processed} images -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
