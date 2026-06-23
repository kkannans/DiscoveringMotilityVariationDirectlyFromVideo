"""
Fig 4 (Draft 6): Optical flow vector comparison — GT vs model vs OF baseline.

Standalone figure (split from draft 5 combined Fig 3).
Requires GPU (model inference). Run via srun on gpu-debug.

OF baseline flow vectors now match OpticalFlow.predict_sequence exactly:
  - 3-frame context window (K-3, K-2, K-1) with averaged initial flow
  - Sliding window update (cur0, cur1, cur2 = cur1, cur2, pred) each step

Outputs -> Q8_v23-alife-2026/outputs/figures/Fig4/
"""

import json
import sys
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

import torch
import torch.nn.functional as F

# ── Repo imports ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "automated_method"))
sys.path.insert(0, str(REPO_ROOT / "classical_pipeline"))
sys.path.insert(0, str(REPO_ROOT / "OpenSTL"))

import paths
from baseline_control_models_losses import OpticalFlow
from models import create_model
from train_model import get_output_dir

# ── Paths (all via paths.py) ──────────────────────────────────────────────
OUTPUT_DIR = paths.FIGURES_DIR / "Fig4"
MOTILITY_CSV = paths.CLASSICAL_DIR / "motility_descriptors.csv"
RESIDUALS_DIR = paths.INTERMEDIATES_DIR / "per_organoid_residuals"
CENTROID_NPZ = paths.CLASSICAL_DIR / "centroid_trajectories.npz"
TEST_ASSIGNMENTS = paths.INTERMEDIATES_DIR / "test_assignments.json"
# Fig 6 spatial-error overlays look best on the 706x706 originals; defaults to the
# 128x128 dataset. Point paths.DATA_ROOT (or override here) at hi-res videos if available.
VIDEO_DIR = paths.DATA_ROOT
AUTOTUNE_DIR = paths.SEGMENTATION_DIR

K, N = 6, 6


# ═════════════════════════════════════════════════════════════════════════
# Step 1: Select fast organoid with direction change
# ═════════════════════════════════════════════════════════════════════════

def select_organoid_and_window(motility_csv, centroid_npz, test_assignments_path):
    """Find a fast organoid where model outperforms OF (high advantage).

    Uses pre-computed residuals to rank organoids by model advantage,
    then picks the best window (highest turning angle) within top candidates.

    Returns (batch_name, window_index, seed).
    """
    mot = pd.read_csv(motility_csv)
    with open(test_assignments_path) as f:
        assignments = json.load(f)

    rows = []
    for npz_path in sorted(RESIDUALS_DIR.glob("*.npz")):
        batch = npz_path.stem
        if batch not in assignments:
            continue
        npz = np.load(npz_path)
        raw_mse = npz["raw_mse"]  # (n_windows, N, 5)
        model_mse = raw_mse[:, :, 0].mean()
        of_mse = raw_mse[:, :, 2].mean()
        rows.append({
            "batch_name": batch,
            "model_mse": model_mse,
            "of_mse": of_mse,
            "advantage_over_of": 1.0 - model_mse / (of_mse + 1e-12),
        })

    perf = pd.DataFrame(rows)
    perf = perf.merge(mot[["batch_name", "mean_velocity", "mean_turning_angle"]], on="batch_name")

    perf = perf[perf["advantage_over_of"] > 0]
    vel_median = perf["mean_velocity"].median()
    candidates = perf[perf["mean_velocity"] > vel_median].copy()
    candidates = candidates.sort_values("advantage_over_of", ascending=False)

    print(f"Candidates with model > OF and above-median speed: {len(candidates)}")

    centroids = np.load(centroid_npz, allow_pickle=True)

    best_batch = None
    best_window = None
    best_seed = None
    best_score = -1

    for _, row in candidates.head(20).iterrows():
        batch = row["batch_name"]
        if batch not in centroids:
            continue

        traj = centroids[batch]  # (120, 2)
        seed = assignments[batch][0]
        advantage = row["advantage_over_of"]

        n_windows = (120 - K - N) // N + 1
        for w in range(n_windows):
            start = w * N
            end = start + K + N
            if end > len(traj):
                continue

            seg = traj[start:end]
            vels = np.diff(seg, axis=0)
            if len(vels) < 3:
                continue

            angles = []
            for i in range(len(vels) - 1):
                v1, v2 = vels[i], vels[i + 1]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 < 1e-6 or n2 < 1e-6:
                    continue
                cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
                angles.append(np.arccos(cos_a))

            if not angles:
                continue

            max_turn = max(angles)
            mean_speed = np.mean(np.linalg.norm(vels, axis=1))
            score = max_turn * mean_speed * (1 + advantage)

            if score > best_score:
                best_score = score
                best_batch = batch
                best_window = w
                best_seed = seed

    print(f"Selected organoid: {best_batch}")
    print(f"  Window: {best_window} (frames {best_window*N}--{best_window*N+K+N})")
    print(f"  Seed: {best_seed}")
    print(f"  Score: {best_score:.3f}")

    return best_batch, best_window, best_seed


# ═════════════════════════════════════════════════════════════════════════
# Step 2: Load frames and predictions
# ═════════════════════════════════════════════════════════════════════════

def load_video_frames_np(batch_name, num_frames=120, target_size=128):
    """Load video as (T, H, W, 3) float32 [0,1] numpy."""
    cap = cv2.VideoCapture(str(VIDEO_DIR / f"{batch_name}.mp4"))
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_size, target_size))
        frames.append(frame)
    cap.release()
    return np.stack(frames).astype(np.float32) / 255.0


def load_autotune_params(batch_name):
    """Load per-organoid auto-tuned segmentation parameters."""
    with open(AUTOTUNE_DIR / batch_name / "best_params.json") as f:
        return json.load(f)


def compute_organoid_mask(frame_float, params):
    """Apply auto-tuned segmentation to a single frame. Returns (H,W) uint8."""
    from autotune_segmentation import segment_frame
    frame_uint8 = (frame_float * 255).astype(np.uint8)
    mask, _, _, _ = segment_frame(frame_uint8, params)
    return (mask > 0).astype(np.uint8)


def get_model_predictions_np(frames_np, ctx_start, seed, device):
    """Load residual SimVP_TAU, return (N, H, W, 3) predicted frames as numpy."""
    MODEL_DIR = paths.MODEL_ROOT
    ckpt_path = MODEL_DIR / f"SimVP_TAU_K6_N6_seed{seed}_pre_residual" / "best_model.pth"
    print(f"Loading model from {ckpt_path}")

    wrapper, _ = create_model("SimVP_TAU", K, N)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    wrapper.load_state_dict(ckpt["model_state_dict"])
    wrapper.eval().to(device)

    ctx_frames = frames_np[ctx_start:ctx_start + K]
    tgt_frames = frames_np[ctx_start + K:ctx_start + K + N]
    ctx = torch.from_numpy(ctx_frames).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    tgt = torch.from_numpy(tgt_frames).permute(0, 3, 1, 2).unsqueeze(0).to(device)

    with torch.no_grad():
        delta = wrapper(ctx, tgt)
        last_ctx = ctx[:, -1:, :3]
        pred = (last_ctx + torch.cumsum(delta, dim=1)).clamp(0, 1)

    return pred.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()


def get_of_baseline_flows(frames_np, ctx_start):
    """Compute OF baseline flow fields matching OpticalFlow.predict_sequence exactly.

    Uses 3-frame context window with averaged initial flow and sliding window
    update, identical to the actual baseline used during evaluation.

    Returns list of N flow arrays, each (H, W, 2).
    """
    ctx_tensor = torch.from_numpy(
        frames_np[ctx_start:ctx_start + K]
    ).permute(0, 3, 1, 2).float()

    # Match OpticalFlow.predict_sequence: 3-frame context (K-3, K-2, K-1)
    cur0 = ctx_tensor[max(K - 3, 0):max(K - 3, 0) + 1]  # (1, 3, H, W)
    cur1 = ctx_tensor[max(K - 2, 0):max(K - 2, 0) + 1]
    cur2 = ctx_tensor[K - 1:K]

    # Average two flows for initial estimate (same as baseline)
    flow01 = OpticalFlow._compute_flow(cur0, cur1)  # (1, 2, H, W)
    flow12 = OpticalFlow._compute_flow(cur1, cur2)
    flow = 0.5 * (flow01 + flow12)

    flows = []
    for _ in range(N):
        flows.append(flow.squeeze(0).permute(1, 2, 0).numpy())  # (H, W, 2)
        pred = OpticalFlow._warp(cur2, flow).clamp(0, 1)
        new_flow = OpticalFlow._compute_flow(cur2, pred)
        flow = 0.5 * (flow + new_flow)
        # Sliding window update — matches baseline exactly
        cur0, cur1, cur2 = cur1, cur2, pred

    return flows


# ═════════════════════════════════════════════════════════════════════════
# Farneback flow (for GT and model flow visualization)
# ═════════════════════════════════════════════════════════════════════════

def compute_flow(frame_a, frame_b):
    """Farneback flow between two (H,W,3) float32 [0,1] frames. Returns (H,W,2)."""
    gray_a = (np.mean(frame_a, axis=2) * 255).astype(np.uint8)
    gray_b = (np.mean(frame_b, axis=2) * 255).astype(np.uint8)
    return cv2.calcOpticalFlowFarneback(
        gray_a, gray_b, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )


# ═════════════════════════════════════════════════════════════════════════
# Quiver plotting
# ═════════════════════════════════════════════════════════════════════════

def overlay_flow_arrows(flow, mask, ax, step=3, scale=20,
                        color="cyan", label=None):
    """Overlay quiver arrows within organoid mask."""
    u, v = flow[:, :, 0], flow[:, :, 1]
    u_masked = np.where(mask, u, 0)
    v_masked = np.where(mask, v, 0)

    H, W = mask.shape
    Y, X = np.mgrid[0:H:step, 0:W:step]
    U = u_masked[::step, ::step]
    V = v_masked[::step, ::step]
    mask_sub = mask[::step, ::step].astype(bool)

    ax.quiver(X[mask_sub], Y[mask_sub],
              U[mask_sub], V[mask_sub],
              color=color, scale=scale,
              width=0.003, headwidth=4, headlength=5,
              alpha=0.8, zorder=3, label=label)


def plot_quiver_frame(ax, gt_frame_np, gt_flow, model_flow, of_flow, mask):
    """Overlay three quiver fields on one organoid frame."""
    ax.imshow(gt_frame_np)
    overlay_flow_arrows(gt_flow, mask, ax, step=3, scale=20, color="cyan")
    overlay_flow_arrows(of_flow, mask, ax, step=3, scale=20, color="orange")
    overlay_flow_arrows(model_flow, mask, ax, step=3, scale=20, color="red")

    # Scale bar: 141 px = 1 mm (bottom-right corner)
    H, W = gt_frame_np.shape[:2]
    bar_len = 141 * W / 706  # scale to current resolution
    x0 = W - bar_len - 5
    y0 = H - 8
    ax.plot([x0, x0 + bar_len], [y0, y0], color="black", linewidth=2,
            solid_capstyle="butt", zorder=10)
    ax.text(x0 + bar_len / 2, y0 - 3, "1 mm", color="black", fontsize=4,
            ha="center", va="bottom", fontweight="bold", zorder=10)

    ax.set_xticks([])
    ax.set_yticks([])


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Select organoid & window ──
    batch_name, window_idx, seed = select_organoid_and_window(
        MOTILITY_CSV, CENTROID_NPZ, TEST_ASSIGNMENTS
    )

    # ── Load frames ──
    frames = load_video_frames_np(batch_name)
    print(f"Loaded {len(frames)} frames for {batch_name}")

    ctx_start = window_idx * N
    pred_start = ctx_start + K

    pred_frames = get_model_predictions_np(frames, ctx_start, seed, device)
    print(f"Predicted {pred_frames.shape[0]} frames")

    of_flows = get_of_baseline_flows(frames, ctx_start)
    seg_params = load_autotune_params(batch_name)

    # ── Plot: 4 quiver snapshots at T=5s, 10s, 20s, 30s ──
    horizons = (0, 1, 3, 5)
    horizon_labels = ["T=5s", "T=10s", "T=20s", "T=30s"]
    n_cols = len(horizons)

    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2.2, 2.6))

    for col, fi in enumerate(horizons):
        ax = axes[col]
        t_curr = pred_start + fi
        t_prev = pred_start - 1 if fi == 0 else pred_start + fi - 1

        gt_flow = compute_flow(frames[t_prev], frames[t_curr])
        if fi == 0:
            model_flow = compute_flow(frames[t_prev], pred_frames[fi])
        else:
            model_flow = compute_flow(pred_frames[fi - 1], pred_frames[fi])
        of_flow = of_flows[fi]

        mask = compute_organoid_mask(frames[t_curr], seg_params)
        plot_quiver_frame(ax, frames[t_curr], gt_flow, model_flow, of_flow, mask)
        ax.set_title(horizon_labels[col], fontsize=7, pad=3)

    # Shared legend
    legend_elements = [
        Line2D([0], [0], color="cyan", marker=">", linestyle="None",
               markersize=5, label="Ground truth"),
        Line2D([0], [0], color="red", marker=">", linestyle="None",
               markersize=5, label="Model"),
        Line2D([0], [0], color="orange", marker=">", linestyle="None",
               markersize=5, label="OF baseline"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(pad=0.3, w_pad=0.3)
    fig.subplots_adjust(bottom=0.12)

    for ext in ["png", "pdf"]:
        out = OUTPUT_DIR / f"fig4_optical_flow_vectors.{ext}"
        fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'fig4_optical_flow_vectors.png'}")

    # Save metadata
    meta = {
        "batch_name": batch_name,
        "window_index": window_idx,
        "seed": seed,
        "horizons": list(horizons),
        "horizon_labels": horizon_labels,
    }
    with open(OUTPUT_DIR / "fig4_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata: {OUTPUT_DIR / 'fig4_metadata.json'}")


if __name__ == "__main__":
    main()
