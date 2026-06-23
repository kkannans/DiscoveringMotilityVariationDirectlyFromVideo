"""Export tracking overlay (trajectory + current contour + scale bar) for Panel B.

For each frame in [0, 40, 80, 119] of batch-000193:
  - Underlying video frame at native 706x706
  - Segmentation contour for the current frame (from autotune segment_frame)
  - Full centroid trajectory from centroid_trajectories.npz
  - 1 mm scale bar (141 px) placed near the trajectory centroid

Outputs -> FIGURES_DIR / Fig1 / tracking_{batch}_frame_{idx}.png
"""
import cv2, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "classical_pipeline"))
import paths
from autotune_segmentation import segment_frame

REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = paths.DATA_ROOT
AUTOTUNE_DIR = paths.SEGMENTATION_DIR
CENTROID_NPZ = paths.CLASSICAL_DIR / "centroid_trajectories.npz"
OUTPUT_DIR = paths.FIGURES_DIR / "Fig1"

BAR_PX = 141
BAR_HEIGHT = 8
LABEL = "1 mm"


def draw_scale_bar_at(img, cx, cy, offset=(60, 60)):
    """Draw scale bar anchored at (cx+offset_x, cy+offset_y)."""
    h, w = img.shape[:2]
    x1 = int(np.clip(cx + offset[0], 20, w - BAR_PX - 20))
    y1 = int(np.clip(cy + offset[1], 40, h - 30))
    x2 = x1 + BAR_PX
    y2 = y1 + BAR_HEIGHT

    cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 0), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2
    (tw, th), _ = cv2.getTextSize(LABEL, font, scale, thickness)
    tx = x1 + (BAR_PX - tw) // 2
    ty = y1 - 8
    cv2.putText(img, LABEL, (tx, ty), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, LABEL, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img


def main():
    batch = "batch-000193"
    frame_indices = [0, 40, 80, 119]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load auto-tuned segmentation params
    with open(AUTOTUNE_DIR / batch / "best_params.json") as f:
        params = json.load(f)
    print(f"Params: {params}")

    # Load centroid trajectory (native coords — 706x706)
    cdata = np.load(CENTROID_NPZ)
    if batch not in cdata:
        print(f"ERROR: {batch} not in centroid npz")
        sys.exit(1)
    traj = cdata[batch][:120]   # (120, 2) — (cx, cy)

    # ── Choose scale bar anchor: median centroid position over the trajectory ──
    valid = np.isfinite(traj[:, 0])
    if not valid.any():
        print("ERROR: no valid centroids")
        sys.exit(1)
    traj_valid = traj[valid]
    cx_med = float(np.median(traj_valid[:, 0]))
    cy_med = float(np.median(traj_valid[:, 1]))

    # ── Load video and export each frame ──
    cap = cv2.VideoCapture(str(VIDEO_DIR / f"{batch}.mp4"))
    n = 0
    idx_set = set(frame_indices)
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if n in idx_set:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Segment current frame with autotune
            mask, centroid, area, props = segment_frame(frame_rgb, params)

            out = frame_bgr.copy()

            # ── Draw trajectory (blue polyline + green start, red end) ──
            pts = traj_valid.astype(np.int32)
            cv2.polylines(out, [pts], False, (255, 100, 40), 3, cv2.LINE_AA)
            cv2.circle(out, tuple(pts[0]), 10, (0, 255, 0), -1)
            cv2.circle(out, tuple(pts[-1]), 10, (0, 0, 255), -1)

            # ── Draw current-frame contour (green outline) ──
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cnt = max(cnts, key=cv2.contourArea)
                cv2.drawContours(out, [cnt], -1, (0, 255, 0), 3)

            # ── Scale bar anchored near median of trajectory ──
            draw_scale_bar_at(out, cx_med, cy_med, offset=(80, 60))

            h, w = out.shape[:2]
            out_path = OUTPUT_DIR / f"tracking_{batch}_frame_{n:03d}.png"
            cv2.imwrite(str(out_path), out)
            print(f"  {out_path.name} ({w}x{h})")
        if n > max(frame_indices):
            break
        n += 1
    cap.release()
    print(f"\nDone. Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
