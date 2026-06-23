"""
Fig Methods: Two-panel methods figure illustrating prediction task and motion loss.

Panel A: 3 rows x 6 cols — Input frames, Predicted frames, Ground truth frames
Panel B: 2 rows x 6 cols — GT frame differences, Predicted frame differences
         with diverging colormap and shared colorbar

Requires GPU (model inference). Run via srun on gpu-debug.

Outputs -> paths.FIGURES_DIR / Fig_methods / fig_methods.{png,pdf}
"""

import json
import sys
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

import torch

# ── Repo imports ──────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "automated_method"))
sys.path.insert(0, str(REPO_ROOT / "OpenSTL"))
import paths

from models import create_model
from train_model import get_output_dir

# ── Paths ─────────────────────────────────────────────────────────────────
OUTPUT_DIR = paths.FIGURES_DIR / "Fig_methods"
SUMMARY_CSV = paths.INTERMEDIATES_DIR / "per_organoid_summary.csv"
VIDEO_DIR = paths.DATA_ROOT
DATASPLIT = paths.DATAINFO_DIR / "data_split_covering_1.json"

K, N = 6, 6
SEED = 1
FRAME_INTERVAL_S = 5  # seconds between frames


# ═════════════════════════════════════════════════════════════════════════
# Step 1: Select median-surprise organoid from test set
# ═════════════════════════════════════════════════════════════════════════

def select_organoids(n=5):
    """Pick n organoids spread across the surprise distribution from the test set."""
    summary = pd.read_csv(SUMMARY_CSV)
    summary = summary.sort_values("mean_motion_loss").reset_index(drop=True)

    with open(DATASPLIT) as f:
        split = json.load(f)
    test_set = set(split["test"])

    test_df = summary[summary["batch_name"].isin(test_set)].copy()
    test_df = test_df.sort_values("mean_motion_loss").reset_index(drop=True)

    # Pick evenly spaced percentiles: 10th, 30th, 50th, 70th, 90th
    indices = [int(len(test_df) * p) for p in [0.1, 0.3, 0.5, 0.7, 0.9]]
    indices = [min(i, len(test_df) - 1) for i in indices]

    selected = []
    for idx in indices:
        row = test_df.iloc[idx]
        selected.append(row["batch_name"])
        print(f"Selected: {row['batch_name']}  motion_loss={row['mean_motion_loss']:.2e}  rank={idx+1}/{len(test_df)}")

    return selected


# ═════════════════════════════════════════════════════════════════════════
# Step 2: Load video and run model
# ═════════════════════════════════════════════════════════════════════════

def load_video_frames(batch_name, num_frames=120, target_size=128):
    """Load video as (T, H, W, 3) float32 [0,1]."""
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


def select_window(frames_np):
    """Pick a window where organoid is visibly moving (window 3-5)."""
    best_window = 3
    best_motion = -1

    for w in range(3, 6):
        start = w * N
        end = start + K + N
        if end > len(frames_np):
            continue
        chunk = frames_np[start:end]
        diffs = np.abs(np.diff(chunk, axis=0)).mean()
        if diffs > best_motion:
            best_motion = diffs
            best_window = w

    print(f"Selected window: {best_window} (mean frame diff: {best_motion:.6f})")
    return best_window


def run_inference(frames_np, window_idx, device, seed=None):
    """Run residual SimVP_TAU on a single window.

    Returns (input_frames, pred_frames, gt_frames) each as (F, H, W, 3) numpy.
    """
    seed = seed or SEED
    MODEL_DIR = REPO_ROOT / "original" / "models"
    ckpt_path = MODEL_DIR / f"SimVP_TAU_K6_N6_seed{seed}_pre_residual" / "best_model.pth"
    print(f"Loading model from {ckpt_path}")

    wrapper, _ = create_model("SimVP_TAU", K, N)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    wrapper.load_state_dict(ckpt["model_state_dict"])
    wrapper.eval().to(device)

    ctx_start = window_idx * N
    input_frames = frames_np[ctx_start:ctx_start + K]
    gt_frames = frames_np[ctx_start + K:ctx_start + K + N]

    ctx = torch.from_numpy(input_frames).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    tgt = torch.from_numpy(gt_frames).permute(0, 3, 1, 2).unsqueeze(0).to(device)

    with torch.no_grad():
        delta = wrapper(ctx, tgt)
        last_ctx = ctx[:, -1:]
        pred = (last_ctx + torch.cumsum(delta, dim=1)).clamp(0, 1)

    pred_frames = pred.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
    return input_frames, pred_frames, gt_frames


# ═════════════════════════════════════════════════════════════════════════
# Step 3: Build figure
# ═════════════════════════════════════════════════════════════════════════

def to_gray(frame):
    """Convert (H,W,3) float32 RGB to (H,W) float32 grayscale."""
    return np.mean(frame, axis=2)


def save_frame(data, path, black_to_red=False):
    """Save image as raw PNG.

    RGB (H,W,3): saved directly.
    Grayscale (H,W) float [0,1]: if black_to_red, map 0->black, 1->red;
                                  otherwise save as grayscale.
    """
    if data.ndim == 3:
        img = (np.clip(data, 0, 1) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif black_to_red:
        v = np.clip(data, 0, 1)
        r = (v * 255).astype(np.uint8)
        g = np.zeros_like(r)
        b = np.zeros_like(r)
        img = cv2.merge([b, g, r])  # BGR for cv2
    else:
        img = (np.clip(data, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(path), img)


def process_one_batch(batch_name, device, seed=None):
    """Generate all individual frame images for one batch."""
    batch_dir = OUTPUT_DIR / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    frames = load_video_frames(batch_name)
    print(f"\n{'='*60}")
    print(f"Processing {batch_name} ({len(frames)} frames, seed={seed or SEED})")

    window_idx = select_window(frames)
    input_frames, pred_frames, gt_frames = run_inference(frames, window_idx, device, seed=seed)

    # Grayscale for Delta/MSE computation
    input_gray = np.stack([to_gray(f) for f in input_frames])
    pred_gray = np.stack([to_gray(f) for f in pred_frames])
    gt_gray = np.stack([to_gray(f) for f in gt_frames])

    gt_prev = np.concatenate([input_gray[-1:], gt_gray[:-1]], axis=0)
    gt_diff = gt_gray - gt_prev
    pred_prev = np.concatenate([input_gray[-1:], pred_gray[:-1]], axis=0)
    pred_diff = pred_gray - pred_prev

    vmax_diff = max(np.abs(gt_diff).max(), np.abs(pred_diff).max())

    mse_map = (gt_diff - pred_diff) ** 2
    vmax_mse = mse_map.max()

    ctx_start_frame = window_idx * N
    tag = batch_name

    # Input frames (RGB)
    for c in range(K):
        t_sec = (ctx_start_frame + c) * FRAME_INTERVAL_S
        save_frame(input_frames[c],
                   batch_dir / f"input_frame{c}_T{t_sec}s_{tag}.png")

    # GT frames (RGB)
    for c in range(N):
        t_sec = (ctx_start_frame + K + c) * FRAME_INTERVAL_S
        save_frame(gt_frames[c],
                   batch_dir / f"GT_frame{c}_T{t_sec}s_{tag}.png")

    # Prediction frames (RGB)
    for c in range(N):
        t_sec = (ctx_start_frame + K + c) * FRAME_INTERVAL_S
        save_frame(pred_frames[c],
                   batch_dir / f"prediction_frame{c}_T{t_sec}s_{tag}.png")

    # Delta GT — normalize |diff| to [0, 1], black-to-red
    for c in range(N):
        t_sec = (ctx_start_frame + K + c) * FRAME_INTERVAL_S
        normed = np.abs(gt_diff[c]) / (vmax_diff + 1e-12)
        save_frame(normed,
                   batch_dir / f"Delta_GT_frame{c}_T{t_sec}s_{tag}.png",
                   black_to_red=True)

    # Delta prediction
    for c in range(N):
        t_sec = (ctx_start_frame + K + c) * FRAME_INTERVAL_S
        normed = np.abs(pred_diff[c]) / (vmax_diff + 1e-12)
        save_frame(normed,
                   batch_dir / f"Delta_prediction_frame{c}_T{t_sec}s_{tag}.png",
                   black_to_red=True)

    # Prediction error per frame: |ΔI_GT - ΔI_pred| (black-to-red)
    delta_error = np.abs(gt_diff - pred_diff)
    vmax_delta_error = delta_error.max()
    for c in range(N):
        t_sec = (ctx_start_frame + K + c) * FRAME_INTERVAL_S
        normed = delta_error[c] / (vmax_delta_error + 1e-12)
        save_frame(normed,
                   batch_dir / f"Delta_error_frame{c}_T{t_sec}s_{tag}.png",
                   black_to_red=True)

    # Delta average across all prediction frames
    avg_delta_pred = np.abs(pred_diff).mean(axis=0)
    save_frame(avg_delta_pred / (vmax_diff + 1e-12),
               batch_dir / f"Delta_average_{tag}.png",
               black_to_red=True)

    # MSE maps — normalize [0, vmax] to [0, 1], black-to-red
    for c in range(N):
        t_sec = (ctx_start_frame + K + c) * FRAME_INTERVAL_S
        normed = mse_map[c] / (vmax_mse + 1e-12)
        save_frame(normed,
                   batch_dir / f"MSE_frame{c}_T{t_sec}s_{tag}.png",
                   black_to_red=True)

    print(f"  Saved 36 frames -> {batch_dir}")

    # GT vs prediction assembled grid
    row_gt = [cv2.imread(str(batch_dir / f"GT_frame{c}_T{(ctx_start_frame+K+c)*FRAME_INTERVAL_S}s_{tag}.png"))
              for c in range(N)]
    row_pred = [cv2.imread(str(batch_dir / f"prediction_frame{c}_T{(ctx_start_frame+K+c)*FRAME_INTERVAL_S}s_{tag}.png"))
                for c in range(N)]
    grid = np.vstack([np.hstack(row_gt), np.hstack(row_pred)])
    cv2.imwrite(str(batch_dir / f"GT_vs_prediction_{tag}.png"), grid)

    used_seed = seed or SEED
    return {
        "batch_name": batch_name,
        "seed": used_seed,
        "window_index": window_idx,
        "ctx_start_frame": ctx_start_frame,
        "vmax_diff": float(vmax_diff),
        "vmax_mse": float(vmax_mse),
    }


# Extra batches with specific seeds (not in default test set for SEED=1)
EXTRA_BATCHES = [
    ("batch-000072", 5),   # in test set for seed 5
]


def generate_pipeline_figure(batch_name, device, seed=None):
    """Generate pipeline overview figure showing overlapping windows (stride=6).

    Layout (top to bottom):
      Row 0: Timeline bar — 120 frames with windows marked
      Row 1: Bar chart — L_motion per window + S_o annotation
    """
    used_seed = seed or SEED
    frames_np = load_video_frames(batch_name)
    T = len(frames_np)
    STRIDE = N  # 6, overlapping
    n_windows = (T - K - N) // STRIDE + 1  # (120 - 6 - 6) // 6 + 1 = 19

    print(f"\nPipeline figure for {batch_name} (seed={used_seed}, {n_windows} windows)")

    # Load model once
    MODEL_DIR = REPO_ROOT / "original" / "models"
    ckpt_path = MODEL_DIR / f"SimVP_TAU_K6_N6_seed{used_seed}_pre_residual" / "best_model.pth"
    wrapper, _ = create_model("SimVP_TAU", K, N)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    wrapper.load_state_dict(ckpt["model_state_dict"])
    wrapper.eval().to(device)

    # Compute L_motion for each window
    l_motion = []
    for w in range(n_windows):
        ctx_start = w * STRIDE
        inp = frames_np[ctx_start:ctx_start + K]
        gt = frames_np[ctx_start + K:ctx_start + K + N]

        ctx = torch.from_numpy(inp).permute(0, 3, 1, 2).unsqueeze(0).to(device)
        tgt = torch.from_numpy(gt).permute(0, 3, 1, 2).unsqueeze(0).to(device)

        with torch.no_grad():
            delta = wrapper(ctx, tgt)
            last_ctx = ctx[:, -1:]
            pred = (last_ctx + torch.cumsum(delta, dim=1)).clamp(0, 1)

        pred_np = pred.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()

        inp_gray = np.mean(inp, axis=3)
        gt_gray = np.mean(gt, axis=3)
        pred_gray = np.mean(pred_np, axis=3)

        gt_prev = np.concatenate([inp_gray[-1:], gt_gray[:-1]], axis=0)
        gt_diff = gt_gray - gt_prev
        pred_prev = np.concatenate([inp_gray[-1:], pred_gray[:-1]], axis=0)
        pred_diff = pred_gray - pred_prev

        lm = np.mean((gt_diff - pred_diff) ** 2)
        l_motion.append(lm)

    l_motion = np.array(l_motion)
    S_o = l_motion.mean()
    print(f"  L_motion range: [{l_motion.min():.2e}, {l_motion.max():.2e}]")
    print(f"  S_o = {S_o:.2e}")

    batch_dir = OUTPUT_DIR / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    colors_k = "#4a90d9"
    colors_n = "#d94a4a"

    # ── Plot 1: Timeline with staggered overlapping windows ──
    fig1, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(-1, T + 8)
    ax.set_ylim(-1, n_windows + 2)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.text(T / 2, n_windows + 1.2, f"{T} frames, {T * FRAME_INTERVAL_S}s",
            ha="center", va="bottom", fontsize=12)

    bar_h = 0.7
    for w in range(n_windows):
        x0 = w * STRIDE
        y = n_windows - w
        ax.barh(y, K, left=x0, height=bar_h,
                color=colors_k, edgecolor="black", linewidth=0.3, alpha=0.7)
        ax.barh(y, N, left=x0 + K, height=bar_h,
                color=colors_n, edgecolor="black", linewidth=0.3, alpha=0.7)
        ax.text(T + 2, y, f"w{w+1}", ha="left", va="center",
                fontsize=8, color="#555555")

    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor=colors_k, edgecolor="black", linewidth=0.3, label=f"Context (K={K})"),
                    Patch(facecolor=colors_n, edgecolor="black", linewidth=0.3, label=f"Predict (N={N})")]
    ax.legend(handles=legend_items, loc="upper right", fontsize=10,
              frameon=True, framealpha=0.9)

    ax.set_xlabel("Frame index", fontsize=14)
    ax.set_xticks(np.arange(0, T + 1, 12))
    ax.tick_params(axis="x", labelsize=12)

    for ext in ["png", "pdf"]:
        fig1.savefig(batch_dir / f"pipeline_timeline_{batch_name}.{ext}",
                     dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig1)

    # ── Plot 2: L_motion bar chart + S_o ──
    fig2, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(n_windows)
    ax.bar(x, l_motion, color=colors_n, edgecolor="black",
           linewidth=0.5, alpha=0.85, width=0.7)

    ax.axhline(S_o, color="black", linestyle="--", linewidth=1.5, zorder=5)
    ax.text(n_windows - 0.3, S_o * 1.12,
            r"$S_o$ = Prediction surprise",
            fontsize=12, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels([f"w{i+1}" for i in range(n_windows)], fontsize=11)
    ax.set_xlabel("Window", fontsize=14)
    ax.set_ylabel(r"$\mathcal{L}_{\mathrm{motion}}(w)$", fontsize=14)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(12)
    ax.tick_params(axis="y", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for ext in ["png", "pdf"]:
        fig2.savefig(batch_dir / f"pipeline_lmotion_{batch_name}.{ext}",
                     dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig2)

    # ── Plot 3: Combined — bars + heatmap strip below, shared frame x-axis ──
    from matplotlib.colors import LinearSegmentedColormap
    cmap_br = LinearSegmentedColormap.from_list("black_red", ["black", "red"])

    fig3, (ax_bar, ax_heat) = plt.subplots(
        2, 1, figsize=(14, 5), height_ratios=[4, 1],
        sharex=True, gridspec_kw={"hspace": 0.05})

    # Top: L_motion bars
    ymax = l_motion.max() * 1.3
    for w in range(n_windows):
        ctx_start = w * STRIDE
        pred_start = ctx_start + K
        ax_bar.axvspan(ctx_start, ctx_start + K, ymin=0, ymax=0.06,
                       color=colors_k, alpha=0.5)
        ax_bar.bar(pred_start + N / 2, l_motion[w], width=N - 0.3,
                   color=colors_n, edgecolor="black", linewidth=0.5, alpha=0.85)

    ax_bar.axhline(S_o, color="black", linestyle="--", linewidth=1.5, zorder=5)
    ax_bar.text(T - 1, S_o * 1.08,
                r"$S_o$ = Prediction surprise",
                fontsize=12, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.9))

    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=colors_k, alpha=0.5, edgecolor="none", label=f"Context (K={K})"),
        Patch(facecolor=colors_n, edgecolor="black", linewidth=0.5, label=f"Predict (N={N})"),
    ]
    ax_bar.legend(handles=legend_items, loc="upper left", fontsize=11,
                  frameon=True, framealpha=0.9)

    ax_bar.set_xlim(-1, T + 1)
    ax_bar.set_ylim(0, ymax)
    ax_bar.set_ylabel(r"$\mathcal{L}_{\mathrm{motion}}(w)$", fontsize=14)
    ax_bar.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax_bar.yaxis.get_offset_text().set_fontsize(12)
    ax_bar.tick_params(axis="y", labelsize=12)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # Bottom: heatmap strip — L_motion mapped to frame positions (black→red)
    # Build a 1D array of L_motion per frame (prediction frames get the value)
    heat_row = np.full(T, np.nan)
    for w in range(n_windows):
        pred_start = w * STRIDE + K
        heat_row[pred_start:pred_start + N] = l_motion[w]

    # Show as image (1 row x T columns)
    heat_img = heat_row.reshape(1, -1)
    ax_heat.imshow(heat_img, aspect="auto", cmap=cmap_br,
                   vmin=0, vmax=l_motion.max(),
                   extent=[-0.5, T - 0.5, 0, 1],
                   interpolation="nearest")
    # Gray out context-only frames (where heat_row is NaN)
    for i in range(T):
        if np.isnan(heat_row[i]):
            ax_heat.axvspan(i - 0.5, i + 0.5, color="#e0e0e0", zorder=0)

    ax_heat.set_yticks([])
    ax_heat.set_xlabel("Frame index", fontsize=14)
    ax_heat.set_xticks(np.arange(0, T + 1, 12))
    ax_heat.tick_params(axis="x", labelsize=12)
    ax_heat.set_ylabel("Loss", fontsize=11, rotation=0, labelpad=25, va="center")

    # Colorbar for heatmap
    sm = plt.cm.ScalarMappable(cmap=cmap_br,
                                norm=plt.Normalize(0, l_motion.max()))
    sm.set_array([])
    cb = fig3.colorbar(sm, ax=ax_heat, orientation="vertical",
                       fraction=0.015, pad=0.02)
    cb.ax.tick_params(labelsize=9)

    for ext in ["png", "pdf"]:
        fig3.savefig(batch_dir / f"pipeline_combined_{batch_name}.{ext}",
                     dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig3)

    # ── Plot 4: Standalone heatmap strip ──
    fig4, ax_h = plt.subplots(figsize=(14, 1.5))
    ax_h.imshow(heat_img, aspect="auto", cmap=cmap_br,
                vmin=0, vmax=l_motion.max(),
                extent=[-0.5, T - 0.5, 0, 1],
                interpolation="nearest")
    for i in range(T):
        if np.isnan(heat_row[i]):
            ax_h.axvspan(i - 0.5, i + 0.5, color="#e0e0e0", zorder=0)
    ax_h.set_yticks([])
    ax_h.set_xlabel("Frame index", fontsize=14)
    ax_h.set_xticks(np.arange(0, T + 1, 12))
    ax_h.tick_params(axis="x", labelsize=12)
    ax_h.set_ylabel(r"$\mathcal{L}$", fontsize=13, rotation=0, labelpad=15, va="center")
    sm = plt.cm.ScalarMappable(cmap=cmap_br,
                                norm=plt.Normalize(0, l_motion.max()))
    sm.set_array([])
    cb4 = fig4.colorbar(sm, ax=ax_h, orientation="vertical",
                        fraction=0.015, pad=0.02)
    cb4.ax.tick_params(labelsize=9)
    for ext in ["png", "pdf"]:
        fig4.savefig(batch_dir / f"pipeline_heatstrip_{batch_name}.{ext}",
                     dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig4)

    print(f"  Saved pipeline_timeline + pipeline_lmotion + pipeline_combined + pipeline_heatstrip -> {batch_dir}")

    return {"batch_name": batch_name, "seed": used_seed,
            "l_motion": l_motion.tolist(), "S_o": float(S_o)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    batch_names = select_organoids(n=5)

    all_meta = []
    for batch_name in batch_names:
        meta = process_one_batch(batch_name, device)
        meta["frame_interval_s"] = FRAME_INTERVAL_S
        all_meta.append(meta)

    # Process extra batches with their appropriate seeds
    for batch_name, seed in EXTRA_BATCHES:
        meta = process_one_batch(batch_name, device, seed=seed)
        meta["frame_interval_s"] = FRAME_INTERVAL_S
        all_meta.append(meta)

    # Generate pipeline overview figures for all batches
    pipeline_meta = []
    for m in all_meta:
        pm = generate_pipeline_figure(m["batch_name"], device, seed=m["seed"])
        pipeline_meta.append(pm)

    all_meta_combined = {"per_batch": all_meta, "pipeline": pipeline_meta}
    with open(OUTPUT_DIR / "fig_methods_metadata.json", "w") as f:
        json.dump(all_meta_combined, f, indent=2)
    print(f"\nMetadata: {OUTPUT_DIR / 'fig_methods_metadata.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
