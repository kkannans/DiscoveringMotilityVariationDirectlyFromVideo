"""Export segmentation contour overlays at full resolution.

Uses the exact segment_frame() from autotune_segmentation.py with
the auto-tuned best_params.json for each organoid.

Outputs -> FIGURES_DIR / Fig1 / segmentation_{batch}_frame_{idx}.png
"""
import cv2, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "classical_pipeline"))
import paths
from autotune_segmentation import segment_frame

REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = paths.DATA_ROOT
AUTOTUNE_DIR = paths.SEGMENTATION_DIR
OUTPUT_DIR = paths.FIGURES_DIR / "Fig1"


def main():
    batch = "batch-000193"
    frame_indices = [0, 40, 80, 119]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load auto-tuned params
    with open(AUTOTUNE_DIR / batch / "best_params.json") as f:
        params = json.load(f)
    print(f"Params: {params}")

    # Load video and segment each frame
    cap = cv2.VideoCapture(str(VIDEO_DIR / f"{batch}.mp4"))
    n = 0
    idx_set = set(frame_indices)
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if n in idx_set:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mask, centroid, area, props = segment_frame(frame_rgb, params)
            out = frame_bgr.copy()
            # Extract contour from mask
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cnt = max(cnts, key=cv2.contourArea)
                cv2.drawContours(out, [cnt], -1, (0, 255, 0), 3)
            h, w = out.shape[:2]
            out_path = OUTPUT_DIR / f"segmentation_{batch}_frame_{n:03d}.png"
            cv2.imwrite(str(out_path), out)
            print(f"  {out_path.name} ({w}x{h})")
        if n > max(frame_indices):
            break
        n += 1
    cap.release()
    print(f"\nDone. Saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
