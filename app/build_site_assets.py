"""
build_site_assets.py — render the static-site assets for the prediction-surprise explorer.

For each of the 108 organoids it renders a side-by-side clip [ ground truth | model prediction |
prediction-error heatmap ] as a web-friendly H.264 MP4, and writes app/site/data.json with the
per-organoid prediction surprise, rank, percentile, mean speed and size. The frontend
(app/site/index.html) loads these — no GPU or server at view time.

GPU required (model inference). Run via srun on gpu-debug:
    srun -p gpu-debug --gres=gpu:1 --cpus-per-task=4 --mem=64G bash -lc \
        '/users/k/k/kkannans/alife-2026/.venv/bin/python app/build_site_assets.py'

Inputs:  trained checkpoints (paths.MODEL_ROOT), dataset MP4s, test_assignments.json,
         per_organoid_summary.csv (Sₒ, area), motility_descriptors.csv (mean speed).
Outputs: app/site/clips/<batch>.mp4  +  app/site/data.json
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.cm as cm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "automated_method"))
import paths
from extract_prediction_surprise import load_model, load_video_frames_tensor, extract_window_pairs, K, N

SITE = ROOT / "app" / "site"
CLIPS = SITE / "clips"
FFMPEG = "/users/k/k/kkannans/bin/ffmpeg"
FPS = 8
CELL = 128
MAGMA = cm.get_cmap("magma")


def _to_rgb(chw):
    """(3,H,W) float[0,1] BGR tensor → (H,W,3) uint8 RGB."""
    img = chw.permute(1, 2, 0).clamp(0, 1).numpy()      # (H,W,3) BGR
    return (img[:, :, ::-1] * 255).astype(np.uint8)     # → RGB


def _heat(err2d, vmax):
    """(H,W) error → (H,W,3) uint8 magma."""
    n = np.clip(err2d / (vmax + 1e-12), 0, 1)
    return (MAGMA(n)[:, :, :3] * 255).astype(np.uint8)


def predicted_track(model, frames, device):
    """Stitch non-overlapping N-frame predictions into a (T,3,H,W) track (residual reconstruction)."""
    pred = frames.clone()
    for ctx, tgt, start in extract_window_pairs(frames, K, N):
        with torch.no_grad():
            out = model(ctx.unsqueeze(0).to(device), tgt.unsqueeze(0).to(device)).squeeze(0).cpu()
        out = ctx[-1].unsqueeze(0) + torch.cumsum(out, dim=0)   # residual → frames (Eq 1)
        pred[start + K: start + K + N] = out.clamp(0, 1)
    return pred


def write_clip(path, gt, pred, fps=FPS):
    """gt/pred: (T,3,H,W). Writes [GT | Pred | Error] H.264 mp4."""
    err = ((gt - pred) ** 2).mean(dim=1).numpy()        # (T,H,W)
    vmax = float(np.percentile(err, 99))
    H, Wpanel = CELL, CELL * 3
    buf = b"".join(
        np.concatenate([_to_rgb(gt[t]), _to_rgb(pred[t]), _heat(err[t], vmax)], axis=1).tobytes()
        for t in range(gt.shape[0]))
    # ffmpeg writing directly to GPFS races (0-byte files); encode to node-local tmp, then move.
    fd, tmp = tempfile.mkstemp(suffix=".mp4", prefix="kkannans-clip-", dir="/tmp")
    os.close(fd)
    proc = subprocess.Popen(
        [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{Wpanel}x{H}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "30", "-movflags", "+faststart", tmp],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _, errout = proc.communicate(buf)
    if proc.returncode != 0 or os.path.getsize(tmp) == 0:
        os.unlink(tmp)
        raise RuntimeError(f"ffmpeg failed for {path.name}: rc={proc.returncode} {errout.decode()[-300:]}")
    shutil.move(tmp, path)


def main():
    CLIPS.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ta_path = paths.DATAINFO_DIR / "test_assignments.json"
    if not ta_path.exists():
        ta_path = paths.INTERMEDIATES_DIR / "test_assignments.json"
    ta = json.load(open(ta_path))
    assign = {b: s[0] for b, s in ta.items()}
    summ = pd.read_csv(paths.INTERMEDIATES_DIR / "per_organoid_summary.csv").set_index("batch_name")
    mot = pd.read_csv(paths.CLASSICAL_DIR / "motility_descriptors.csv").set_index("batch_name")

    batches = sorted(assign)
    summ["rank"] = summ["mean_motion_loss"].rank().astype(int)
    n = len(batches)

    records = []
    for i, b in enumerate(batches):
        if b not in summ.index:
            continue
        seed = assign[b]
        model = load_model(seed, residual=True, device=device)
        frames = load_video_frames_tensor(b, num_frames=120)
        pred = predicted_track(model, frames, device)
        write_clip(CLIPS / f"{b}.mp4", frames, pred)
        rank = int(summ.loc[b, "rank"])
        records.append({
            "batch": b, "seed": int(seed),
            "surprise": float(summ.loc[b, "mean_motion_loss"]),
            "rank": rank, "percentile": round(100 * rank / n),
            "mean_speed": float(mot.loc[b, "mean_velocity"]) if b in mot.index else None,
            "size_px": float(summ.loc[b, "area_px"]) if "area_px" in summ.columns else None,
            "clip": f"clips/{b}.mp4",
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n} clips", flush=True)

    records.sort(key=lambda r: r["surprise"])
    (SITE / "data.json").write_text(json.dumps({"n": len(records), "organoids": records}, indent=1))
    print(f"Done. {len(records)} clips → {CLIPS}; data.json → {SITE/'data.json'}")


if __name__ == "__main__":
    main()
