#!/usr/bin/env python3
"""
export_fig4_data.py — run ONCE on the cluster (CPU) to cache Fig 4's plot inputs.

Imports analysis/generate_fig4.py's functions (does NOT modify it), runs the
organoid/window selection + model prediction + flows + masks, and saves a small
npz so plotting can be done anywhere with no GPU/torch/videos.

Run:  cd analysis && ../.venv/bin/python export_fig4_data.py
Out:  fig4_plot_data.npz  (next to this script)
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # so `import generate_fig4` works
import generate_fig4 as g4

device = torch.device("cpu")
batch, win, seed = g4.select_organoid_and_window(
    g4.MOTILITY_CSV, g4.CENTROID_NPZ, g4.TEST_ASSIGNMENTS)
frames = g4.load_video_frames_np(batch)
ctx_start = win * g4.N
pred_start = ctx_start + g4.K
pred_frames = g4.get_model_predictions_np(frames, ctx_start, seed, device)
of_flows = g4.get_of_baseline_flows(frames, ctx_start)
seg_params = g4.load_autotune_params(batch)

horizons = (0, 1, 3, 5)
gt_frames, gt_flows, model_flows, of_f, masks = [], [], [], [], []
for fi in horizons:
    t_curr = pred_start + fi
    t_prev = pred_start - 1 if fi == 0 else pred_start + fi - 1
    gt_flow = g4.compute_flow(frames[t_prev], frames[t_curr])
    model_flow = (g4.compute_flow(frames[t_prev], pred_frames[fi]) if fi == 0
                  else g4.compute_flow(pred_frames[fi - 1], pred_frames[fi]))
    gt_frames.append(frames[t_curr]); gt_flows.append(gt_flow)
    model_flows.append(model_flow); of_f.append(of_flows[fi])
    masks.append(g4.compute_organoid_mask(frames[t_curr], seg_params))

meta = {"batch": batch, "window": int(win), "seed": int(seed),
        "horizons": list(horizons), "horizon_labels": ["T=5s", "T=10s", "T=20s", "T=30s"]}
np.savez_compressed(HERE / "fig4_plot_data.npz",
                    gt_frames=np.array(gt_frames), gt_flows=np.array(gt_flows),
                    model_flows=np.array(model_flows), of_flows=np.array(of_f),
                    masks=np.array(masks).astype(np.uint8), meta=json.dumps(meta))
print("SAVED fig4_plot_data.npz", meta)
