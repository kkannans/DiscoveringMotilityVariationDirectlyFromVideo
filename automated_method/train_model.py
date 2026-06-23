"""
train_model.py — Train OpenSTL models on organoid videos.

Features:
    - Baselines (copy-last, linear interp) computed at start, plotted on loss curves
    - Prediction visualization every --viz-every epochs (GT | Pred | Error grid)
    - MSE + SSIM tracked for both train and val

Usage:
    python automated_method/train_model.py --method SimVP_TAU --seed 1 -K 6 -N 6 --residual
    python automated_method/train_model.py --method PredRNN --seed 1 --epochs 200
    python automated_method/train_model.py --method all --seed 1  # benchmark all models
    python automated_method/train_model.py --debug                 # 3 epochs, small data
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(REPO_ROOT))

import paths
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import create_organoid_dataloaders
from models import (
    SIMVP_STYLE, RECURRENT_STYLE, PHYDNET_STYLE, ALL_METHODS,
    ModelConfigs, UnifiedModelWrapper,
    _build_raw_model, create_model,
)

# Available methods in OpenSTL (this version)
# SimVP variants use model_type parameter: gSTA, tau, convmixer, etc.
METHODS = [
    "ConvLSTM", "PredRNN", "PredRNNpp", "MIM", "E3DLSTM",
    "PhyDNet", "SimVP", "SimVP_TAU", "MAU",
]

EXPERIMENT_NAME = "organoid_motility"


def get_output_dir(method, seed, K, N, exclude_video=None, residual=False):
    base = f"{method}_K{K}_N{N}_seed{seed}"
    if residual:
        base += "_residual"
    if exclude_video:
        base += f"_loocv_{exclude_video}"
    return paths.MODEL_ROOT / base


# ── Baselines (computed once before training) ────────────────────────────

def compute_baselines_on_loader(loader, device):
    """Compute baseline MSE/SSIM on a full dataloader.

    Baselines: copy-last, mean-frame, rolling-average, optical-flow.
    Returns dict with '{name}_mse' and '{name}_ssim' keys.
    """
    from losses import ssim_loss_per_sample
    from baseline_control_models_losses import OpticalFlow

    RA_W = 6  # rolling average window

    keys = ["copy", "mean_frame", "rolling_avg", "optical_flow"]
    accum = {f"{k}_mse": [] for k in keys}
    accum.update({f"{k}_ssim": [] for k in keys})

    with torch.no_grad():
        for batch in loader:
            ctx, tgt = batch[0].to(device), batch[1].to(device)
            B, K, C, H, W = ctx.shape
            N = tgt.shape[1]

            ctx_rgb = ctx

            # Mean frame baseline (average of all context frames)
            mean_pred = ctx_rgb.mean(dim=1)  # (B, 3, H, W)

            # Rolling average buffer: last RA_W context frames
            if K >= RA_W:
                ra_buffer = [ctx_rgb[:, K - RA_W + i].clone() for i in range(RA_W)]
            else:
                ra_buffer = [ctx_rgb[:, 0].clone() for _ in range(RA_W - K)]
                ra_buffer += [ctx_rgb[:, i].clone() for i in range(K)]

            # Optical flow: initial flow from last 2 context frames
            of_prev = ctx_rgb[:, -2] if K >= 2 else ctx_rgb[:, -1]
            of_cur = ctx_rgb[:, -1]
            of_flow = OpticalFlow._compute_flow(of_prev, of_cur)

            for t in range(N):
                gt = tgt[:, t]

                # Copy-last
                copy_pred = ctx_rgb[:, -1]
                accum["copy_mse"].append(F.mse_loss(copy_pred, gt).item())
                accum["copy_ssim"].append(1.0 - ssim_loss_per_sample(copy_pred, gt).mean().item())

                # Mean frame
                accum["mean_frame_mse"].append(F.mse_loss(mean_pred, gt).item())
                accum["mean_frame_ssim"].append(1.0 - ssim_loss_per_sample(mean_pred, gt).mean().item())

                # Rolling average (autoregressive)
                ra_pred = torch.stack(ra_buffer).mean(dim=0).clamp(0, 1)
                accum["rolling_avg_mse"].append(F.mse_loss(ra_pred, gt).item())
                accum["rolling_avg_ssim"].append(1.0 - ssim_loss_per_sample(ra_pred, gt).mean().item())
                ra_buffer.pop(0)
                ra_buffer.append(ra_pred)

                # Optical flow (autoregressive warp)
                of_pred = OpticalFlow._warp(of_cur, of_flow).clamp(0, 1)
                accum["optical_flow_mse"].append(F.mse_loss(of_pred, gt).item())
                accum["optical_flow_ssim"].append(1.0 - ssim_loss_per_sample(of_pred, gt).mean().item())
                new_flow = OpticalFlow._compute_flow(of_cur, of_pred)
                of_flow = 0.5 * (of_flow + new_flow)
                of_cur = of_pred

    return {k: float(np.mean(v)) for k, v in accum.items()}


# ── Training Visualization ───────────────────────────────────────────────

def save_prediction_grid(model, loader, device, epoch, output_dir, n_samples=3,
                         residual=False):
    """
    Save a visualization grid: GT context | GT target | Prediction | Error (×5)
    Pick n_samples from the loader, one row per sample.
    Saved as: output_dir/visualizations/prediction_epoch_{epoch}.png
    """
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    model.eval()
    samples_shown = 0
    rows = []

    with torch.no_grad():
        for batch in loader:
            ctx, tgt = batch[0].to(device), batch[1].to(device)
            pred = model(ctx, tgt)

            # Reconstruct frames from deltas for residual models
            if residual:
                pred = ctx[:, -1:, :3] + torch.cumsum(pred, dim=1)

            for i in range(min(ctx.shape[0], n_samples - samples_shown)):
                # Last context frame, first target frame, first prediction, error
                last_ctx = ctx[i, -1].cpu().clamp(0, 1)  # (3, H, W)
                gt_frame = tgt[i, 0].cpu().clamp(0, 1)       # (3, H, W) first target
                pred_frame = pred[i, 0].cpu().clamp(0, 1)    # (3, H, W) first prediction
                error = (gt_frame - pred_frame).abs() * 5     # amplified error
                error = error.clamp(0, 1)

                rows.append((last_ctx, gt_frame, pred_frame, error))
                samples_shown += 1
                if samples_shown >= n_samples:
                    break
            if samples_shown >= n_samples:
                break

    if not rows:
        return

    fig, axes = plt.subplots(len(rows), 4, figsize=(16, 4 * len(rows)))
    if len(rows) == 1:
        axes = axes[np.newaxis, :]

    titles = ["Last Context", "GT Target (t+1)", "Prediction (t+1)", "Error (×5)"]
    for row_idx, (ctx_f, gt_f, pred_f, err_f) in enumerate(rows):
        for col_idx, (frame, title) in enumerate(zip(
            [ctx_f, gt_f, pred_f, err_f], titles
        )):
            ax = axes[row_idx, col_idx]
            ax.imshow(frame.permute(1, 2, 0).numpy())
            if row_idx == 0:
                ax.set_title(title, fontsize=10)
            ax.axis("off")

        # Annotate MSE for this sample
        sample_mse = F.mse_loss(pred_f, gt_f).item()
        axes[row_idx, 0].set_ylabel(f"MSE: {sample_mse:.6f}", fontsize=9)

    plt.suptitle(f"Predictions — Epoch {epoch}", fontsize=14)
    plt.tight_layout()
    fig.savefig(viz_dir / f"prediction_epoch_{epoch:04d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_loss_curves(train_mses, val_mses, train_ssims, val_ssims,
                     baselines, output_dir):
    """
    Save training curves with baseline reference lines.
    Two panels: MSE (log scale) and SSIM.
    Baselines: copy-last, mean-frame, rolling-average, optical-flow.
    """
    BASELINE_STYLES = [
        ("copy",         "Copy-last",    "orange"),
        ("mean_frame",   "Mean frame",   "gray"),
        ("rolling_avg",  "Rolling avg",  "#bcbd22"),
        ("optical_flow", "Optical flow", "#17becf"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    epochs = range(1, len(train_mses) + 1)

    # MSE panel
    ax1.plot(epochs, train_mses, "b-", linewidth=1.5, alpha=0.8, label="Train MSE")
    if val_mses:
        ax1.plot(epochs, val_mses, "r-", linewidth=1.5, alpha=0.8, label="Val MSE")
    for key, label, color in BASELINE_STYLES:
        mse_key = f"{key}_mse"
        if mse_key in baselines:
            ax1.axhline(baselines[mse_key], color=color, linestyle="--",
                        linewidth=1.5, label=f"{label} ({baselines[mse_key]:.6f})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE")
    ax1.set_yscale("log")
    ax1.set_title("MSE vs Baselines")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # SSIM panel
    ax2.plot(epochs, train_ssims, "b-", linewidth=1.5, alpha=0.8, label="Train SSIM")
    if val_ssims:
        ax2.plot(epochs, val_ssims, "r-", linewidth=1.5, alpha=0.8, label="Val SSIM")
    for key, label, color in BASELINE_STYLES:
        ssim_key = f"{key}_ssim"
        if ssim_key in baselines:
            ax2.axhline(baselines[ssim_key], color=color, linestyle="--",
                        linewidth=1.5, label=f"{label} ({baselines[ssim_key]:.4f})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("SSIM")
    ax2.set_title("SSIM vs Baselines")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── CSV Logging ────────────────────────────────────────────────────────────

def init_csv(path, header):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(header)

def append_csv(path, row):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ── Core Training Loop ────────────────────────────────────────────────────

def train_one_model(method, args):
    """Train a single OpenSTL model with baselines + visualization."""
    exclude_video = getattr(args, 'exclude_video', None)
    residual = getattr(args, 'residual', False)

    output_dir = get_output_dir(method, args.seed, args.K, args.N,
                                exclude_video=exclude_video,
                                residual=residual)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Training {method} | seed={args.seed} | K={args.K} N={args.N} S={args.stride}")
    if exclude_video:
        print(f"LOOCV: excluding {exclude_video}")
    if residual:
        print(f"Residual prediction mode: ON (output = frame deltas)")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # ── Seed ──
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Data ──
    max_vids = 1 if args.debug else None
    exclude_list = [exclude_video] if exclude_video else None
    train_loader, val_loader, test_loader = create_organoid_dataloaders(
        seed=args.seed, K=args.K, N=args.N, stride=args.stride,
        batch_size=args.batch_size, num_workers=args.num_workers,
        max_videos=max_vids,
        exclude_videos=exclude_list,
        datainfo_dir=args.datainfo_dir,
        split_prefix=args.split_prefix,
    )

    # ── Compute or load cached baselines ──
    # Baselines depend only on data split, not on model/residual mode.
    # Cache key: seed, split_prefix, datainfo_dir
    _sp = getattr(args, 'split_prefix', 'data_split')
    _di = getattr(args, 'datainfo_dir', None) or 'default'
    _di_tag = Path(_di).name if _di != 'default' else 'default'
    baseline_cache_dir = paths.BASELINE_CACHE_DIR
    baseline_cache_dir.mkdir(parents=True, exist_ok=True)
    baseline_cache_file = baseline_cache_dir / f"baselines_seed{args.seed}_{_sp}_{_di_tag}.json"

    if baseline_cache_file.exists() and not args.debug:
        print(f"Loading cached baselines from {baseline_cache_file}")
        with open(baseline_cache_file) as f:
            cached = json.load(f)
        train_baselines = cached["train"]
        val_baselines = cached["val"]
    else:
        print("Computing baselines (copy, mean-frame, rolling-avg, optical-flow)...")
        train_baselines = compute_baselines_on_loader(train_loader, device)
        val_baselines = compute_baselines_on_loader(val_loader, device)
        # Save to shared cache
        with open(baseline_cache_file, "w") as f:
            json.dump({"train": train_baselines, "val": val_baselines}, f, indent=2)
        print(f"Baselines cached to {baseline_cache_file}")

    print(f"  Train — copy MSE: {train_baselines['copy_mse']:.6f}, "
          f"optical flow MSE: {train_baselines['optical_flow_mse']:.6f}")
    print(f"  Val   — copy MSE: {val_baselines['copy_mse']:.6f}, "
          f"optical flow MSE: {val_baselines['optical_flow_mse']:.6f}")

    # Also save to run-local logs dir
    with open(logs_dir / "baselines.json", "w") as f:
        json.dump({"train": train_baselines, "val": val_baselines}, f, indent=2)

    # ── Model ──
    model, n_params = create_model(method, args.K, args.N)
    model = model.to(device)
    print(f"Model: {method} ({n_params:,} params)")

    # ── Optimizer ──
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader),
    )

    # ── CSV logging ──
    init_csv(logs_dir / "train_metrics.csv",
             ["epoch", "loss", "ssim", "learning_rate"])
    init_csv(logs_dir / "val_metrics.csv",
             ["epoch", "loss", "ssim", "baseline_copy", "baseline_optical_flow", "is_best_model"])

    # ── Training loop ──
    from losses import ssim_loss_per_sample

    best_val_mse = float("inf")
    train_mses, val_mses = [], []
    train_ssims, val_ssims = [], []

    start_time = time.time()

    for epoch in range(args.epochs):
        # ── Train ──
        model.train()
        epoch_mse, epoch_ssim, n_batches = 0.0, 0.0, 0
        for batch in train_loader:
            ctx, tgt = batch[0].to(device), batch[1].to(device)
            pred = model(ctx, tgt)

            # Loss: residual mode predicts deltas, reconstruct frames via cumsum
            if residual:
                pred_frames = ctx[:, -1:] + torch.cumsum(pred, dim=1)
                loss = F.mse_loss(pred_frames, tgt)
            else:
                pred_frames = pred
                loss = F.mse_loss(pred, tgt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                batch_ssim = 1.0 - ssim_loss_per_sample(
                    pred_frames.reshape(-1, 3, 128, 128),
                    tgt.reshape(-1, 3, 128, 128),
                ).mean().item()

            epoch_mse += loss.item()
            epoch_ssim += batch_ssim
            n_batches += 1

        avg_train_mse = epoch_mse / max(n_batches, 1)
        avg_train_ssim = epoch_ssim / max(n_batches, 1)
        train_mses.append(avg_train_mse)
        train_ssims.append(avg_train_ssim)

        lr = optimizer.param_groups[0]["lr"]
        append_csv(logs_dir / "train_metrics.csv",
                   [epoch, avg_train_mse, avg_train_ssim, lr])

        # ── Validate ──
        model.eval()
        val_mse_sum, val_ssim_sum, val_n = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                ctx, tgt = batch[0].to(device), batch[1].to(device)
                pred = model(ctx, tgt)

                if residual:
                    val_pred_frames = ctx[:, -1:] + torch.cumsum(pred, dim=1)
                else:
                    val_pred_frames = pred
                val_mse_sum += F.mse_loss(val_pred_frames, tgt).item()
                val_ssim_sum += 1.0 - ssim_loss_per_sample(
                    val_pred_frames.reshape(-1, 3, 128, 128),
                    tgt.reshape(-1, 3, 128, 128),
                ).mean().item()
                val_n += 1

        avg_val_mse = val_mse_sum / max(val_n, 1)
        avg_val_ssim = val_ssim_sum / max(val_n, 1)
        val_mses.append(avg_val_mse)
        val_ssims.append(avg_val_ssim)

        is_best = avg_val_mse < best_val_mse
        if is_best:
            best_val_mse = avg_val_mse
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_mse": avg_val_mse,
                "method": method,
                "K": args.K,
                "N": args.N,
                "residual": residual,
            }, output_dir / "best_model.pth")

        append_csv(logs_dir / "val_metrics.csv",
                   [epoch, avg_val_mse, avg_val_ssim,
                    val_baselines["copy_mse"], val_baselines["optical_flow_mse"],
                    is_best])

        # ── Print progress ──
        if (epoch + 1) % 10 == 0 or epoch == 0:
            beat_copy = "YES" if avg_val_mse < val_baselines["copy_mse"] else "no"
            beat_of = "YES" if avg_val_mse < val_baselines["optical_flow_mse"] else "no"
            print(f"Epoch {epoch+1:4d}/{args.epochs} | "
                  f"train={avg_train_mse:.6f} | val={avg_val_mse:.6f} | "
                  f"SSIM={avg_val_ssim:.4f} | "
                  f"vs copy: {beat_copy} | vs optflow: {beat_of}"
                  f"{' *best*' if is_best else ''}")

        # ── Visualization every viz_every epochs ──
        if args.viz_every > 0 and ((epoch + 1) % args.viz_every == 0 or epoch == 0):
            save_prediction_grid(model, val_loader, device, epoch + 1, output_dir,
                                 n_samples=3, residual=residual)

        # ── Update loss curves plot ──
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            save_loss_curves(train_mses, val_mses, train_ssims, val_ssims,
                             val_baselines, output_dir)

    train_time = time.time() - start_time

    # ── Final summary ──
    summary = {
        "method": method,
        "seed": args.seed,
        "K": args.K,
        "N": args.N,
        "stride": args.stride,
        "residual": residual,
        "exclude_video": exclude_video,
        "epochs": args.epochs,
        "train_time_seconds": train_time,
        "best_val_mse": best_val_mse,
        "final_val_mse": val_mses[-1] if val_mses else None,
        "final_val_ssim": val_ssims[-1] if val_ssims else None,
        "baselines_val": val_baselines,
        "baselines_train": train_baselines,
        "beats_copy": best_val_mse < val_baselines["copy_mse"],
        "beats_optical_flow": best_val_mse < val_baselines["optical_flow_mse"],
        "n_params": n_params,
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Finished {method} | best val MSE: {best_val_mse:.6f}")
    print(f"Beats copy-last: {summary['beats_copy']} | Beats optical-flow: {summary['beats_optical_flow']}")
    print(f"Results: {output_dir / 'results.json'}")
    print(f"{'='*60}")

    return summary


def parse_args():
    p = argparse.ArgumentParser(description="OpenSTL benchmark on organoid data")
    p.add_argument("-m", "--method", type=str, default="SimVP",
                   choices=METHODS + ["all"],
                   help="Model architecture (or 'all' for full benchmark)")
    p.add_argument("-s", "--seed", type=int, default=1)
    p.add_argument("-K", type=int, default=10, help="Context frames")
    p.add_argument("-N", type=int, default=2, help="Target frames")
    p.add_argument("-S", "--stride", type=int, default=2)
    p.add_argument("-e", "--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("-b", "--batch-size", type=int, default=16)
    p.add_argument("-w", "--num-workers", type=int, default=4)
    p.add_argument("--viz-every", type=int, default=10,
                   help="Save prediction visualization every N epochs (0=disable)")
    p.add_argument("--debug", action="store_true", help="Quick test: 3 epochs, 10%% data")
    p.add_argument("-r", "--residual", action="store_true",
                   help="Residual prediction: model outputs frame deltas, "
                        "reconstructed via cumsum from last context frame")
    p.add_argument("--exclude-video", type=str, default=None,
                   help="Batch name to exclude from all splits (for LOOCV)")
    p.add_argument("--datainfo-dir", type=str, default=str(REPO_ROOT / "datainfo"),
                   help="Directory for data split JSON files (default: ./datainfo)")
    p.add_argument("--split-prefix", type=str, default="data_split_covering",
                   help="Filename prefix for split files (default: data_split_covering)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.debug:
        args.epochs = 3
        args.viz_every = 1

    methods = METHODS if args.method == "all" else [args.method]

    all_results = []
    for method in methods:
        try:
            result = train_one_model(method, args)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR: {method} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"method": method, "error": str(e)})

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Method':<15} {'Best MSE':>10} {'SSIM':>10} {'vs Copy':>10} {'vs OptFlow':>10} {'Time':>10}")
    print(f"{'='*80}")
    for r in all_results:
        if "error" in r:
            print(f"{r['method']:<15} {'FAILED':>10}")
        else:
            copy_ratio = r["best_val_mse"] / r["baselines_val"]["copy_mse"]
            of_ratio = r["best_val_mse"] / r["baselines_val"]["optical_flow_mse"]
            print(f"{r['method']:<15} {r['best_val_mse']:>10.6f} "
                  f"{r.get('final_val_ssim', 0):>10.4f} "
                  f"{copy_ratio:>9.2f}x {of_ratio:>9.2f}x "
                  f"{r['train_time_seconds']:>9.0f}s")

    # Save combined results
    out_dir = paths.OUTPUT_ROOT / "openstl_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"benchmark_seed{args.seed}_K{args.K}_N{args.N}.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
