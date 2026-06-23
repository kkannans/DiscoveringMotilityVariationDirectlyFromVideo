"""
extract_test_predictions.py — GPU extraction of per-organoid test metrics.

For each trained seed (1-10), loads the SimVP_TAU model and runs inference
on the test split of its covering data split. Computes per-organoid:
  - Model MSE and SSIM (averaged across sequences from that organoid)
  - Baseline MSE and SSIM (copy-last, mean-frame, rolling-avg, optical-flow)

Outputs:
    {OUTPUT_ROOT}/predictability_analysis/extraction/
        seed{N}_per_organoid.json       per-seed results
    {OUTPUT_ROOT}/predictability_analysis/
        per_organoid_test_results.csv   aggregated across seeds

Usage:
    python extract_test_predictions.py                    # all available seeds
    python extract_test_predictions.py --seeds 1 2 3      # specific seeds
    python extract_test_predictions.py --force             # re-extract even if cached
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(REPO_ROOT))

import paths
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

EXPERIMENT_NAME = "organoid_motility"
MODEL_BASE = paths.MODEL_ROOT
DATAINFO_DIR = str(Path(__file__).resolve().parent.parent / "datainfo")
OUTPUT_DIR = paths.OUTPUT_ROOT / "predictability_analysis"
K, N, STRIDE = 6, 6, 2


# ── Dataset that returns batch_name ──────────────────────────────────

class TestDatasetWithNames(Dataset):
    """Test dataset that returns batch_name alongside ctx/tgt."""

    def __init__(self, seed):
        from dataset import OrganoidOpenSTLDataset
        self.ds = OrganoidOpenSTLDataset(
            split='test', seed=seed, K=K, N=N, stride=STRIDE,
            datainfo_dir=DATAINFO_DIR,
            split_prefix='data_split_covering',
        )
        self.index_map = list(range(len(self.ds.inner)))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        real_idx = self.index_map[idx]
        ctx, tgt, batch_name, start = self.ds.inner[real_idx]
        return ctx, tgt, batch_name


def collate_with_names(batch):
    ctxs = torch.stack([b[0] for b in batch])
    tgts = torch.stack([b[1] for b in batch])
    names = [b[2] for b in batch]
    return ctxs, tgts, names


# ── Model loading ────────────────────────────────────────────────────

def load_model(seed, device, method="SimVP_TAU", residual=False):
    """Load trained `method` model for given seed."""
    from train_model import create_model
    suffix = "_residual" if residual else ""
    model_dir = MODEL_BASE / f"{method}_K{K}_N{N}_seed{seed}{suffix}"
    ckpt_path = model_dir / "best_model.pth"
    if not ckpt_path.exists():
        return None

    model, _ = create_model(method, K, N)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


# ── Baseline computation (per-sample) ────────────────────────────────

def compute_per_sample_baselines(ctx, tgt, device):
    """Compute per-sample baseline MSE and SSIM for a batch.

    Args:
        ctx: (B, K, 3, H, W)
        tgt: (B, N, 3, H, W)

    Returns:
        dict[baseline_name] -> {"mse": (B,) ndarray, "ssim_loss": (B,) ndarray}
    """
    from losses import ssim_loss_per_sample
    from baseline_control_models_losses import OpticalFlow

    B, K_ctx, C, H, W = ctx.shape
    N_tgt = tgt.shape[1]
    RA_W = 6

    results = {}

    for bl_name in ["copy", "mean_frame", "rolling_avg", "optical_flow"]:
        if bl_name == "copy":
            pred_all = ctx[:, -1:].expand(-1, N_tgt, -1, -1, -1)

        elif bl_name == "mean_frame":
            mean_pred = ctx.mean(dim=1)  # (B, C, H, W)
            pred_all = mean_pred.unsqueeze(1).expand(-1, N_tgt, -1, -1, -1)

        elif bl_name == "rolling_avg":
            if K_ctx >= RA_W:
                ra_buffer = [ctx[:, K_ctx - RA_W + i].clone() for i in range(RA_W)]
            else:
                ra_buffer = [ctx[:, 0].clone() for _ in range(RA_W - K_ctx)]
                ra_buffer += [ctx[:, i].clone() for i in range(K_ctx)]
            preds = []
            for t in range(N_tgt):
                ra_pred = torch.stack(ra_buffer).mean(dim=0).clamp(0, 1)
                preds.append(ra_pred)
                ra_buffer.pop(0)
                ra_buffer.append(ra_pred)
            pred_all = torch.stack(preds, dim=1)

        elif bl_name == "optical_flow":
            of_prev = ctx[:, -2] if K_ctx >= 2 else ctx[:, -1]
            of_cur = ctx[:, -1]
            of_flow = OpticalFlow._compute_flow(of_prev, of_cur)
            preds = []
            for t in range(N_tgt):
                of_pred = OpticalFlow._warp(of_cur, of_flow).clamp(0, 1)
                preds.append(of_pred)
                new_flow = OpticalFlow._compute_flow(of_cur, of_pred)
                of_flow = 0.5 * (of_flow + new_flow)
                of_cur = of_pred
            pred_all = torch.stack(preds, dim=1)

        # Per-sample MSE: average over N, C, H, W
        mse = F.mse_loss(pred_all, tgt, reduction='none').mean(dim=(1, 2, 3, 4))

        # Per-sample SSIM loss: flatten to (B*N, C, H, W)
        pred_flat = pred_all.reshape(B * N_tgt, C, H, W)
        tgt_flat = tgt.reshape(B * N_tgt, C, H, W)
        ssim_loss_flat = ssim_loss_per_sample(pred_flat, tgt_flat)  # (B*N,)
        ssim_loss_avg = ssim_loss_flat.reshape(B, N_tgt).mean(dim=1)  # (B,)

        results[bl_name] = {
            "mse": mse.cpu().numpy(),
            "ssim_loss": ssim_loss_avg.cpu().numpy(),
        }

    return results


# ── Per-seed extraction ──────────────────────────────────────────────

def extract_seed(seed, device, force=False, residual=False, method="SimVP_TAU"):
    """Extract per-organoid test metrics for one seed."""
    from losses import compute_per_sample_rollout_mse, compute_per_sample_rollout_ssim

    mtag = "" if method == "SimVP_TAU" else f"{method}_"
    tag = "_residual" if residual else ""
    extraction_dir = OUTPUT_DIR / "extraction"
    extraction_dir.mkdir(parents=True, exist_ok=True)
    output_file = extraction_dir / f"{mtag}seed{seed}{tag}_per_organoid.json"

    if output_file.exists() and not force:
        print(f"  Seed {seed}{tag}: cached, skipping (use --force to re-extract)")
        with open(output_file) as f:
            return json.load(f)

    # Load model
    model = load_model(seed, device, method=method, residual=residual)
    if model is None:
        print(f"  Seed {seed}{tag}: no model found, skipping")
        return None

    # Load test data
    ds = TestDatasetWithNames(seed)
    print(f"  Seed {seed}{tag}: {len(ds)} test sequences")
    if len(ds) == 0:
        print(f"  Seed {seed}{tag}: no test sequences, skipping")
        return None

    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4,
                        collate_fn=collate_with_names, pin_memory=True)

    # Accumulate per-organoid results
    metric_keys = ["model_mse", "model_ssim_loss",
                    "copy_mse", "copy_ssim_loss",
                    "mean_frame_mse", "mean_frame_ssim_loss",
                    "rolling_avg_mse", "rolling_avg_ssim_loss",
                    "optical_flow_mse", "optical_flow_ssim_loss"]
    organoid_metrics = defaultdict(lambda: {k: [] for k in metric_keys})

    with torch.no_grad():
        for batch_idx, (ctx, tgt, names) in enumerate(loader):
            ctx, tgt = ctx.to(device), tgt.to(device)

            # Model predictions
            pred = model(ctx, tgt)
            # Residual mode: model outputs deltas, reconstruct frames
            if residual:
                pred = ctx[:, -1:, :3] + torch.cumsum(pred, dim=1)
            model_mse = compute_per_sample_rollout_mse(pred, tgt).cpu().numpy()
            model_ssim_loss = compute_per_sample_rollout_ssim(pred, tgt).cpu().numpy()

            # Baselines
            baselines = compute_per_sample_baselines(ctx, tgt, device)

            # Group by organoid
            for i, name in enumerate(names):
                om = organoid_metrics[name]
                om["model_mse"].append(float(model_mse[i]))
                om["model_ssim_loss"].append(float(model_ssim_loss[i]))
                for bl_name in ["copy", "mean_frame", "rolling_avg", "optical_flow"]:
                    om[f"{bl_name}_mse"].append(float(baselines[bl_name]["mse"][i]))
                    om[f"{bl_name}_ssim_loss"].append(float(baselines[bl_name]["ssim_loss"][i]))

            if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
                print(f"    Batch {batch_idx + 1}/{len(loader)}")

    # Average per organoid
    result = {}
    for name, om in organoid_metrics.items():
        n_seq = len(om["model_mse"])
        result[name] = {
            "n_sequences": n_seq,
            "model_mse": float(np.mean(om["model_mse"])),
            "model_ssim": float(1.0 - np.mean(om["model_ssim_loss"])),
            "copy_mse": float(np.mean(om["copy_mse"])),
            "copy_ssim": float(1.0 - np.mean(om["copy_ssim_loss"])),
            "mean_frame_mse": float(np.mean(om["mean_frame_mse"])),
            "mean_frame_ssim": float(1.0 - np.mean(om["mean_frame_ssim_loss"])),
            "rolling_avg_mse": float(np.mean(om["rolling_avg_mse"])),
            "rolling_avg_ssim": float(1.0 - np.mean(om["rolling_avg_ssim_loss"])),
            "optical_flow_mse": float(np.mean(om["optical_flow_mse"])),
            "optical_flow_ssim": float(1.0 - np.mean(om["optical_flow_ssim_loss"])),
        }

    output_data = {"seed": seed, "residual": residual, "organoids": result}
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"  Seed {seed}{tag}: {len(result)} organoids -> {output_file}")

    return output_data


# ── Aggregation ──────────────────────────────────────────────────────

def aggregate_results(all_seed_results, csv_name="per_organoid_test_results.csv"):
    """Average across seeds for each organoid, save CSV."""
    organoid_agg = defaultdict(lambda: defaultdict(list))

    for seed_result in all_seed_results:
        if seed_result is None:
            continue
        seed = seed_result["seed"]
        for name, metrics in seed_result["organoids"].items():
            for key, val in metrics.items():
                if key != "n_sequences":
                    organoid_agg[name][key].append(val)
            organoid_agg[name]["seeds"].append(seed)

    # Build rows
    metric_cols = ["model_mse", "model_ssim", "copy_mse", "copy_ssim",
                   "mean_frame_mse", "mean_frame_ssim",
                   "rolling_avg_mse", "rolling_avg_ssim",
                   "optical_flow_mse", "optical_flow_ssim"]

    rows = []
    for name in sorted(organoid_agg.keys()):
        agg = organoid_agg[name]
        row = {"batch_name": name, "n_seeds": len(agg.get("model_mse", []))}
        for key in metric_cols:
            vals = agg.get(key, [])
            row[key] = float(np.mean(vals)) if vals else None
            if len(vals) > 1:
                row[f"{key}_std"] = float(np.std(vals))
        row["seeds"] = ",".join(str(s) for s in sorted(agg.get("seeds", [])))
        rows.append(row)

    # Save CSV
    csv_path = OUTPUT_DIR / csv_name
    if rows:
        # Collect all possible keys across all rows
        all_keys = {}
        for r in rows:
            for k in r:
                all_keys[k] = True
        fieldnames = list(all_keys.keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"\nAggregated: {len(rows)} organoids -> {csv_path}")

    return rows


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract per-organoid test metrics from trained models (GPU)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if cached")
    parser.add_argument("--residual", action="store_true",
                        help="Use residual prediction models (frame deltas + cumsum)")
    parser.add_argument("--method", type=str, default="SimVP_TAU",
                        help="Architecture: SimVP_TAU (default), PredRNN, ...")
    args = parser.parse_args()

    tag = "_residual" if args.residual else ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model type: {'residual' if args.residual else 'plain'}")
    print(f"Output: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for seed in args.seeds:
        print(f"\n{'='*50}")
        print(f"Seed {seed} ({'residual' if args.residual else 'plain'})")
        print(f"{'='*50}")
        result = extract_seed(seed, device, force=args.force,
                              residual=args.residual, method=args.method)
        all_results.append(result)

    # Also load any previously cached seeds not in this run
    extraction_dir = OUTPUT_DIR / "extraction"
    for seed in range(1, 11):
        if seed not in args.seeds:
            cached = extraction_dir / f"seed{seed}{tag}_per_organoid.json"
            if cached.exists():
                with open(cached) as f:
                    all_results.append(json.load(f))
                print(f"  Loaded cached seed {seed}{tag}")

    # Use model-type-specific output CSV
    csv_name = f"per_organoid_test_results{tag}.csv"
    aggregate_results(all_results, csv_name=csv_name)
    print("\nDone.")


if __name__ == "__main__":
    main()
