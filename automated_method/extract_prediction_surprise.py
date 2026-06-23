"""
extract_prediction_surprise.py — extract predictions, compute raw MSE + delta residuals.

For each organoid (using its assigned test seed from test_assignments.json):
  1. Load the population model for that seed
  2. For each sliding window (K=6 context, N=6 target, advance by N):
     - Model predicts N frames from K context frames
     - Baselines (Copy, LinearInterpolation, OpticalFlow from baseline_control_models_losses.py)
       predict N frames from the same context
     - Raw MSE: ||GT_frame - pred_frame||² per method per horizon
     - Delta residuals: ||Δ_GT - Δ_pred||² per method per horizon
     - Spatial maps: per-pixel delta residual
  3. Save per-organoid residuals and spatial residual maps

Usage:
    python extract_prediction_surprise.py                       # all 108 organoids
    python extract_prediction_surprise.py --batch batch-000121  # single organoid (debug)
    python extract_prediction_surprise.py --dry-run             # print plan, don't run

Inputs:
    paths.DATAINFO_DIR / test_assignments.json
    paths.MODEL_ROOT / SimVP_TAU_K6_N6_seed{s}_pre / best_model.pth

Outputs -> paths.INTERMEDIATES_DIR
    per_organoid_residuals/{batch}.npz   # raw_mse (n_win, N, 4) + delta_residuals (n_win, N, 4)
    spatial_residual_maps/{batch}.npy    # mean spatial delta residual (N, 128, 128)
    raw_mse_comparison.json              # aggregate raw MSE per horizon per method
    raw_mse_per_organoid.csv             # 108 rows × per-horizon raw MSE
    fig1_panel_b_raw_mse.png
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR.parent))
sys.path.insert(0, str(PROJECT_DIR / "openstl_lib"))

from baseline_control_models_losses import OpticalFlow
import paths

K = 6   # context frames
N = 6   # prediction frames
S = 1   # frame stride (consecutive)

OUTPUT_DIR = paths.INTERMEDIATES_DIR

# All baseline names stored in npz (column indices 0-4)
ALL_BASELINE_NAMES = ["model", "copy", "mean_frame", "rolling_avg", "optical_flow"]
RA_W = 6  # rolling average window (same as train_openstl.py)

# Baselines used in R1/R2 reporting (subset of ALL_BASELINE_NAMES)
REPORT_NAMES = ["model", "copy", "optical_flow"]
REPORT_COLS = [ALL_BASELINE_NAMES.index(n) for n in REPORT_NAMES]  # [0, 1, 4]


# ══════════════════════════════════════════════════════════════════════════════
# Video loading
# ══════════════════════════════════════════════════════════════════════════════

def load_video_frames_tensor(batch_name, num_frames=120):
    """Load first num_frames of a video as (T, 3, 128, 128) float32 tensor in [0,1]."""
    from config import PathConfig
    from cache_videos_to_pt import load_video_frames
    from video_dataset import _resize_frames_to_128

    path_config = PathConfig(K=K, N=N, stride=S)
    video_path = Path(path_config.data_path) / f"{batch_name}.mp4"
    frames_np = load_video_frames(video_path, start_frame=0, num_frames=num_frames)
    frames_np = _resize_frames_to_128(frames_np)  # (T, 3, 128, 128) numpy
    return torch.from_numpy(frames_np).float()     # (T, 3, 128, 128)


def extract_window_pairs(frames, K=K, N=N):
    """Extract non-overlapping (context, target) pairs advancing by N frames.

    Window i: ctx = frames[i*N : i*N + K], tgt = frames[i*N + K : i*N + K + N]

    Returns list of (ctx, tgt, start_idx) tuples.
    """
    T = frames.shape[0]
    pairs = []
    window_size = K + N
    for start in range(0, T - window_size + 1, N):
        ctx = frames[start : start + K]          # (K, 3, 128, 128)
        tgt = frames[start + K : start + K + N]  # (N, 3, 128, 128)
        pairs.append((ctx, tgt, start))
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# Delta computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_deltas_with_anchor(anchor_frame, frames):
    """Compute deltas where first delta uses an external anchor.

    Δ[0] = frames[0] - anchor_frame
    Δ[h] = frames[h] - frames[h-1]  for h >= 1

    Returns: (N, 3, H, W)
    """
    full = torch.cat([anchor_frame.unsqueeze(0), frames], dim=0)
    return full[1:] - full[:-1]


# ══════════════════════════════════════════════════════════════════════════════
# Model loading (cached per seed)
# ══════════════════════════════════════════════════════════════════════════════

_model_cache = {}

def load_model(seed, method="SimVP_TAU", residual=True, device="cuda"):
    """Load the population model (`method`) for given seed, with caching."""
    key = (method, seed, residual)
    if key in _model_cache:
        return _model_cache[key]

    from train_model import create_model, get_output_dir
    output_dir = get_output_dir(method, seed, K, N, residual=residual)
    ckpt_path = output_dir / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    wrapper, _ = create_model(method, K, N)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    wrapper.load_state_dict(state_dict)
    wrapper = wrapper.to(device)
    wrapper.eval()
    _model_cache[key] = wrapper
    return wrapper


# ══════════════════════════════════════════════════════════════════════════════
# Per-organoid processing
# ══════════════════════════════════════════════════════════════════════════════

def process_organoid(batch_name, seed, method="SimVP_TAU", residual=True, device="cuda"):
    """Process one organoid: model + baseline predictions, raw MSE + delta residuals.

    Baselines match compute_baselines_on_loader() in train_openstl.py:
    copy, mean_frame, rolling_avg (window=6), optical_flow.

    Returns dict with:
        - raw_mse: (n_windows, N, 5) — [model, copy, mean_frame, rolling_avg, optical_flow]
        - delta_residuals: (n_windows, N, 5) — same order, delta-space
        - spatial_residual_map: (N, 128, 128) — mean |Δ_GT - Δ_pred|² per horizon
        - n_windows: int
    """
    model = load_model(seed, method=method, residual=residual, device=device)
    frames = load_video_frames_tensor(batch_name, num_frames=120)
    pairs = extract_window_pairs(frames, K, N)

    if len(pairs) == 0:
        return None

    all_raw_mse = []
    all_delta_residuals = []
    spatial_accum = None
    n_windows = 0

    for ctx, tgt, start_idx in pairs:
        # Model prediction
        ctx_dev = ctx.unsqueeze(0).to(device)
        tgt_dev = tgt.unsqueeze(0).to(device)
        with torch.no_grad():
            pred_model = model(ctx_dev, tgt_dev).squeeze(0).cpu()  # (N, 3, H, W)

        anchor = ctx[-1]  # last context frame

        if residual:
            # Residual models output per-frame deltas; reconstruct frames = I_K + cumsum(deltas)
            pred_model = anchor.unsqueeze(0) + torch.cumsum(pred_model, dim=0)

        # ── Baselines (same as train_openstl.py::compute_baselines_on_loader) ──

        # Copy: repeat last context frame
        copy_pred = anchor  # (3, H, W)

        # Mean frame: average of all K context frames
        mean_pred = ctx.mean(dim=0)  # (3, H, W)

        # Rolling average: autoregressive, window=RA_W
        if K >= RA_W:
            ra_buffer = [ctx[K - RA_W + i].clone() for i in range(RA_W)]
        else:
            ra_buffer = [ctx[0].clone() for _ in range(RA_W - K)]
            ra_buffer += [ctx[i].clone() for i in range(K)]

        # Optical flow: autoregressive warp
        of_prev = ctx[-2] if K >= 2 else ctx[-1]
        of_cur = ctx[-1].clone()
        of_flow = OpticalFlow._compute_flow(
            of_prev.unsqueeze(0), of_cur.unsqueeze(0)
        ).squeeze(0)  # (2, H, W)

        # Per-horizon predictions + MSE
        raw_mse_h = []
        baseline_frames = []  # collect per-horizon frames for delta computation

        for h in range(N):
            gt_h = tgt[h]

            # Rolling average (autoregressive)
            ra_pred = torch.stack(ra_buffer).mean(dim=0).clamp(0, 1)
            ra_buffer.pop(0)
            ra_buffer.append(ra_pred)

            # Optical flow (autoregressive warp)
            of_pred = OpticalFlow._warp(
                of_cur.unsqueeze(0), of_flow.unsqueeze(0)
            ).squeeze(0).clamp(0, 1)
            new_flow = OpticalFlow._compute_flow(
                of_cur.unsqueeze(0), of_pred.unsqueeze(0)
            ).squeeze(0)
            of_flow = 0.5 * (of_flow + new_flow)
            of_cur = of_pred

            def frame_mse(a, b):
                return float(((a - b) ** 2).mean())

            raw_mse_h.append([
                frame_mse(gt_h, pred_model[h]),
                frame_mse(gt_h, copy_pred),
                frame_mse(gt_h, mean_pred),
                frame_mse(gt_h, ra_pred),
                frame_mse(gt_h, of_pred),
            ])
            baseline_frames.append({
                "copy": copy_pred, "mean_frame": mean_pred,
                "rolling_avg": ra_pred, "optical_flow": of_pred,
            })

        all_raw_mse.append(np.array(raw_mse_h))  # (N, 5)

        # ── Delta residuals ──
        delta_gt = compute_deltas_with_anchor(anchor, tgt)
        delta_model = compute_deltas_with_anchor(anchor, pred_model)

        copy_stack = copy_pred.unsqueeze(0).expand(N, -1, -1, -1)
        mean_stack = mean_pred.unsqueeze(0).expand(N, -1, -1, -1)
        ra_stack = torch.stack([baseline_frames[h]["rolling_avg"] for h in range(N)])
        of_stack = torch.stack([baseline_frames[h]["optical_flow"] for h in range(N)])

        def mse_per_horizon(a, b):
            return ((a - b) ** 2).mean(dim=(1, 2, 3)).numpy()

        delta_res = np.stack([
            mse_per_horizon(delta_gt, delta_model),
            mse_per_horizon(delta_gt, compute_deltas_with_anchor(anchor, copy_stack)),
            mse_per_horizon(delta_gt, compute_deltas_with_anchor(anchor, mean_stack)),
            mse_per_horizon(delta_gt, compute_deltas_with_anchor(anchor, ra_stack)),
            mse_per_horizon(delta_gt, compute_deltas_with_anchor(anchor, of_stack)),
        ], axis=1)  # (N, 5)
        all_delta_residuals.append(delta_res)

        # Spatial residual: |Δ_GT - Δ_pred_model|² per pixel
        spatial_res = ((delta_gt - delta_model) ** 2).mean(dim=1).numpy()
        if spatial_accum is None:
            spatial_accum = spatial_res
        else:
            spatial_accum += spatial_res
        n_windows += 1

    return {
        "raw_mse": np.stack(all_raw_mse, axis=0),                 # (n_windows, N, 5)
        "delta_residuals": np.stack(all_delta_residuals, axis=0),  # (n_windows, N, 5)
        "spatial_residual_map": spatial_accum / n_windows,         # (N, 128, 128)
        "n_windows": n_windows,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def plot_bar_comparison(per_organoid_means, method_labels, colors, ylabel, output_path):
    """Bar plot: mean MSE per method with error bars + Bonferroni-corrected Wilcoxon p-values."""
    from scipy.stats import wilcoxon

    n_methods = len(method_labels)
    n_comparisons = n_methods - 1  # number of model-vs-baseline tests
    means = [per_organoid_means[:, i].mean() for i in range(n_methods)]
    stds = [per_organoid_means[:, i].std() for i in range(n_methods)]

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(n_methods)
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.7, capsize=4,
           edgecolor="black", linewidth=0.5,
           error_kw=dict(elinewidth=1.2, capthick=1.2))

    # Overlay individual data points (one per seed)
    for i in range(n_methods):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, per_organoid_means.shape[0])
        ax.scatter(x[i] + jitter, per_organoid_means[:, i],
                   color="black", s=20, alpha=0.5, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    # Wilcoxon signed-rank with Bonferroni correction
    model_vals = per_organoid_means[:, 0]
    y_max = max(m + s for m, s in zip(means, stds))
    bracket_y = y_max * 1.3
    for i in range(1, n_methods):
        baseline_vals = per_organoid_means[:, i]
        stat, pval = wilcoxon(model_vals, baseline_vals, alternative="less")
        p_corrected = min(pval * n_comparisons, 1.0)  # Bonferroni

        # Format: p = 0.000X < 0.05 *** (or ** or * or n.s.)
        if p_corrected < 0.001:
            sig = "***"
        elif p_corrected < 0.01:
            sig = "**"
        elif p_corrected < 0.05:
            sig = "*"
        else:
            sig = "n.s."
        if p_corrected < 0.001:
            p_str = f"p = {p_corrected:.4f} ***"
        elif p_corrected < 0.01:
            p_str = f"p = {p_corrected:.3f} **"
        elif p_corrected < 0.05:
            p_str = f"p = {p_corrected:.3f} *"
        else:
            p_str = f"p = {p_corrected:.3f} n.s."

        ax.plot([0, 0, i, i], [bracket_y, bracket_y * 1.05, bracket_y * 1.05, bracket_y],
                color="black", linewidth=0.8)
        ax.text((0 + i) / 2, bracket_y * 1.07, p_str, ha="center", va="bottom", fontsize=9)
        bracket_y *= 1.4

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="R1: Compute motion residuals")
    parser.add_argument("--batch", type=str, default=None, help="Single batch (debug)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--method", type=str, default="SimVP_TAU",
                        help="Architecture: SimVP_TAU (default), PredRNN, ...")
    parser.add_argument("--residual", action="store_true",
                        help="Load the residual checkpoint variant (paper uses residual)")
    args = parser.parse_args()

    # Load test assignments
    with open(paths.DATAINFO_DIR / "test_assignments.json") as f:
        test_assignments = json.load(f)
    assignments = {b: seeds[0] for b, seeds in test_assignments.items()}

    if args.batch:
        if args.batch not in assignments:
            print(f"ERROR: {args.batch} not in test_assignments.json")
            sys.exit(1)
        assignments = {args.batch: assignments[args.batch]}

    seed_to_batches = {}
    for batch, seed in assignments.items():
        seed_to_batches.setdefault(seed, []).append(batch)

    if args.dry_run:
        print(f"Would process {len(assignments)} organoids across {len(seed_to_batches)} seeds:")
        for seed in sorted(seed_to_batches):
            print(f"  Seed {seed}: {len(seed_to_batches[seed])} organoids")
        return

    tag = "" if args.method == "SimVP_TAU" else f"{args.method.lower()}_"
    residuals_dir = OUTPUT_DIR / f"{tag}per_organoid_residuals"
    spatial_dir = OUTPUT_DIR / f"{tag}spatial_residual_maps"
    residuals_dir.mkdir(parents=True, exist_ok=True)
    spatial_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    n_processed = 0

    for seed in sorted(seed_to_batches):
        batches = sorted(seed_to_batches[seed])
        print(f"\n── Seed {seed}: {len(batches)} organoids ──")

        for batch_name in tqdm(batches, desc=f"Seed {seed}"):
            npz_path = residuals_dir / f"{batch_name}.npz"
            npy_path = spatial_dir / f"{batch_name}.npy"

            # Check if cached AND has raw_mse (v6 format)
            if npz_path.exists() and npy_path.exists():
                cached = np.load(npz_path)
                if "raw_mse" in cached:
                    all_results[batch_name] = {
                        "raw_mse": cached["raw_mse"],
                        "delta_residuals": cached["delta_residuals"],
                        "n_windows": int(cached["n_windows"]),
                        "seed": seed,
                    }
                    n_processed += 1
                    continue

            result = process_organoid(batch_name, seed, method=args.method,
                                      residual=args.residual, device=args.device)
            if result is None:
                print(f"  WARNING: {batch_name} produced no windows, skipping")
                continue

            np.savez_compressed(npz_path,
                                raw_mse=result["raw_mse"],
                                delta_residuals=result["delta_residuals"],
                                n_windows=np.array(result["n_windows"]))
            np.save(npy_path, result["spatial_residual_map"])

            all_results[batch_name] = {
                "raw_mse": result["raw_mse"],
                "delta_residuals": result["delta_residuals"],
                "n_windows": result["n_windows"],
                "seed": seed,
            }
            n_processed += 1

        if seed in _model_cache:
            del _model_cache[seed]
            torch.cuda.empty_cache()

    print(f"\nProcessed {n_processed}/{len(assignments)} organoids")

    from scipy.stats import wilcoxon

    # ── Per-organoid mean MSE (all methods, for downstream scripts) ──
    sorted_batches = sorted(all_results.keys())
    n_all = len(ALL_BASELINE_NAMES)
    n_report = len(REPORT_NAMES)
    total_windows = sum(r["raw_mse"].shape[0] for r in all_results.values())

    raw_per_org_all = np.array([all_results[b]["raw_mse"].mean(axis=(0, 1)) for b in sorted_batches])
    delta_per_org_all = np.array([all_results[b]["delta_residuals"].mean(axis=(0, 1)) for b in sorted_batches])

    raw_per_org = raw_per_org_all[:, REPORT_COLS]
    delta_per_org = delta_per_org_all[:, REPORT_COLS]

    # ── Per-organoid CSV ──
    rows = []
    for i, b in enumerate(sorted_batches):
        row = {"batch_name": b, "n_windows": all_results[b]["n_windows"],
               "assigned_seed": all_results[b]["seed"]}
        for m_idx, m_name in enumerate(REPORT_NAMES):
            row[f"raw_mse_{m_name}"] = float(raw_per_org[i, m_idx])
            row[f"delta_mse_{m_name}"] = float(delta_per_org[i, m_idx])
        rows.append(row)

    with open(OUTPUT_DIR / "raw_mse_per_organoid.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {OUTPUT_DIR / 'raw_mse_per_organoid.csv'}")

    # ── Delta residuals per organoid CSV (motion loss, for Fig 3) ──
    delta_rows = []
    for i, b in enumerate(sorted_batches):
        delta_rows.append({
            "batch_name": b,
            "mean_motion_loss": float(delta_per_org[i, 0]),
            "n_windows": all_results[b]["n_windows"],
            "assigned_seed": all_results[b]["seed"],
        })
    with open(OUTPUT_DIR / "delta_residuals_per_organoid.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=delta_rows[0].keys())
        writer.writeheader()
        writer.writerows(delta_rows)
    print(f"  Saved: {OUTPUT_DIR / 'delta_residuals_per_organoid.csv'}")

    # ══════════════════════════════════════════════════════════════════════
    # Fig 2 statistics: per-SEED analysis (seed is the independent replicate)
    # ══════════════════════════════════════════════════════════════════════

    # Group per-organoid MSE by seed, compute per-seed means
    seed_to_orgs = {}
    for b in sorted_batches:
        s = all_results[b]["seed"]
        seed_to_orgs.setdefault(s, []).append(b)

    seeds = sorted(seed_to_orgs.keys())
    n_seeds = len(seeds)

    # Per-seed mean raw MSE: (n_seeds, n_report)
    raw_per_seed = np.zeros((n_seeds, n_report))
    seed_csv_rows = []
    for si, s in enumerate(seeds):
        orgs = seed_to_orgs[s]
        org_indices = [sorted_batches.index(b) for b in orgs]
        raw_per_seed[si] = raw_per_org[org_indices].mean(axis=0)
        row = {"seed": s, "n_test_organoids": len(orgs)}
        for m_idx, m_name in enumerate(REPORT_NAMES):
            row[f"mse_{m_name}"] = float(raw_per_seed[si, m_idx])
        seed_csv_rows.append(row)

    with open(OUTPUT_DIR / "per_seed_mse.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=seed_csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(seed_csv_rows)
    print(f"  Saved: {OUTPUT_DIR / 'per_seed_mse.csv'}")

    # Per-seed Wilcoxon (N=10 paired comparisons)
    model_seeds = raw_per_seed[:, 0]
    comparison = {
        "model": {"mean": float(model_seeds.mean()), "std": float(model_seeds.std()),
                   "sem": float(model_seeds.std() / np.sqrt(n_seeds))},
        "n_seeds": n_seeds,
        "n_organoids_total": len(sorted_batches),
        "total_windows": total_windows,
        "test": "wilcoxon_signed_rank_paired_by_seed",
        "bonferroni_comparisons": n_report - 1,
    }
    for m_idx in range(1, n_report):
        m_name = REPORT_NAMES[m_idx]
        baseline_seeds = raw_per_seed[:, m_idx]
        stat, pval = wilcoxon(model_seeds, baseline_seeds, alternative="less")
        p_corrected = min(pval * (n_report - 1), 1.0)
        comparison[m_name] = {
            "mean": float(baseline_seeds.mean()),
            "std": float(baseline_seeds.std()),
            "sem": float(baseline_seeds.std() / np.sqrt(n_seeds)),
            "p_value_raw": float(pval),
            "p_value_bonferroni": float(p_corrected),
            "wilcoxon_stat": float(stat),
            "model_wins": int((model_seeds < baseline_seeds).sum()),
        }

    with open(OUTPUT_DIR / "raw_mse_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"  Saved: {OUTPUT_DIR / 'raw_mse_comparison.json'}")

    # Print summary
    print(f"\n  Raw MSE per-seed analysis (N={n_seeds} seeds):")
    print(f"    Model: mean={model_seeds.mean():.6f}, SEM={model_seeds.std()/np.sqrt(n_seeds):.6f}")
    for m_name in REPORT_NAMES[1:]:
        info = comparison[m_name]
        print(f"    vs {m_name}: p_raw={info['p_value_raw']:.4f}, "
              f"p_bonf={info['p_value_bonferroni']:.4f}, "
              f"wins={info['model_wins']}/{n_seeds}")

    # ── Plot: bars + SEM from 10 seeds + individual seed dots ──
    method_labels = ["Model", "Copy-last", "Optical Flow"]
    colors = ["#2ca02c", "#d62728", "#1f77b4"]

    plot_bar_comparison(raw_per_seed, method_labels, colors,
                        ylabel=r"Mean MSE $\|I_{GT} - I_{pred}\|^2$",
                        output_path=OUTPUT_DIR / "fig2_model_vs_baselines.png")

    print(f"\nR1 complete. Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
