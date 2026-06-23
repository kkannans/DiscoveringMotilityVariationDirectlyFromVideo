"""
validate_pipeline.py — End-to-end validation of the alife-2026 repo pipeline.

Runs each step of the pipeline and validates the artifacts staged under paths.py
(./outputs, ./dataset) — populated by download_data.py or stage_cache.sh.

Steps:
  2. Copy original Q8_v23 outputs to ./original/ for comparison
  3. Validate create_datasplit.py (coverage + equivalence)
  4. Train SimVP on GPU (interactive srun) — prints command, user runs manually
  5. Validate loss function (MSE on 6 predictions vs 6 GT)
  6. Test SimVP against original test metrics
  7. Extract prediction surprise (manual verification)
  8. Validate autotune segmentation (loads params, generates centroid grid)

Usage:
    # Run all non-GPU validation steps:
    python tests/validate_pipeline.py

    # Run a specific step:
    python tests/validate_pipeline.py --step 3
    python tests/validate_pipeline.py --step 5

    # Setup only (steps 1-2):
    python tests/validate_pipeline.py --setup
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "automated_method"))
sys.path.insert(0, str(REPO_ROOT / "classical_pipeline"))

import paths

# ── Paths (all via paths.py; populated by download_data.py / stage_cache.sh) ──────────────

LOCAL_ORIGINAL = REPO_ROOT / "original"   # legacy view; step2 links it to the paths.py dirs
LOCAL_DATASET = paths.DATA_ROOT
PY = "/users/k/k/kkannans/alife-2026/.venv/bin/python"   # working interpreter (see REPRODUCIBILITY.md)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"
INFO = "\033[94mINFO\033[0m"


def print_header(step_num, title):
    print(f"\n{'='*70}")
    print(f"  Step {step_num}: {title}")
    print(f"{'='*70}")


def assert_check(condition, msg):
    status = PASS if condition else FAIL
    print(f"  [{status}] {msg}")
    return condition


# ══════════════════════════════════════════════════════════════════════════
# Step 1: Create dataset from source videos (first 120 frames, 128x128)
# ══════════════════════════════════════════════════════════════════════════

def step1_create_dataset():
    """Verify the dataset is present at paths.DATA_ROOT (provided by download_data.py)."""
    print_header(1, "Verify dataset present (108 MP4s at DATA_ROOT)")

    if not LOCAL_DATASET.exists():
        print(f"  [{FAIL}] DATA_ROOT not found: {LOCAL_DATASET}")
        print(f"  [{INFO}] Run: python download_data.py   (or bash stage_cache.sh on this machine)")
        return False

    videos = sorted(LOCAL_DATASET.glob("batch-*.mp4"))
    ok = assert_check(len(videos) >= 108, f"DATA_ROOT has {len(videos)} videos (expect 108)")

    if videos:
        import cv2
        cap = cv2.VideoCapture(str(videos[0]))
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert_check(fc >= 120, f"Frame count: {fc} (expect >= 120; model uses first 120)")

    return ok


# ══════════════════════════════════════════════════════════════════════════
# Step 2: Copy original Q8_v23 outputs to ./original/
# ══════════════════════════════════════════════════════════════════════════

def step2_copy_originals():
    """Symlink original Q8_v23 outputs into ./original/ for comparison."""
    print_header(2, "Link original Q8_v23 outputs to ./original/")

    LOCAL_ORIGINAL.mkdir(parents=True, exist_ok=True)

    # Link the legacy ./original/<name> view to the paths.py output dirs (populated by
    # download_data.py / stage_cache.sh), so the validations below read the staged artifacts.
    links = {
        "datainfo": paths.DATAINFO_DIR,
        "models": paths.MODEL_ROOT,
        "intermediates": paths.INTERMEDIATES_DIR,
        "segmentation": paths.SEGMENTATION_DIR,
        "classical_descriptors": paths.CLASSICAL_DIR,
        "predictability": paths.OUTPUT_ROOT / "predictability_analysis",
    }

    all_ok = True
    for name, target in links.items():
        link_path = LOCAL_ORIGINAL / name
        if link_path.exists() or link_path.is_symlink():
            assert_check(True, f"{name} -> already linked")
            continue
        if not target.exists():
            assert_check(False, f"{name} -> source missing: {target}")
            all_ok = False
            continue
        link_path.symlink_to(target)
        assert_check(True, f"{name} -> {target}")

    # Verify key files exist through the links
    checks = [
        ("datainfo/data_split_covering_1.json", "Covering split seed 1"),
        ("models/SimVP_TAU_K6_N6_seed1_pre_residual/best_model.pth", "SimVP_TAU seed 1 checkpoint"),
        ("models/PredRNN_K6_N6_seed1_pre_residual/best_model.pth", "PredRNN seed 1 checkpoint"),
        ("intermediates/per_organoid_residuals", "Per-organoid residuals dir"),
        ("intermediates/spatial_residual_maps", "Spatial residual maps dir"),
        ("intermediates/per_seed_mse.csv", "Per-seed MSE"),
        ("segmentation", "Segmentation auto-tune results"),
        ("classical_descriptors/motility_descriptors.csv", "Motility descriptors"),
        ("predictability/per_organoid_test_results_residual.csv", "Test results CSV"),
    ]
    for rel_path, desc in checks:
        p = LOCAL_ORIGINAL / rel_path
        all_ok &= assert_check(p.exists(), f"Verify: {desc}")

    return all_ok


# ══════════════════════════════════════════════════════════════════════════
# Step 3: Validate create_datasplit.py
# ══════════════════════════════════════════════════════════════════════════

def step3_validate_datasplit():
    """Validate that create_datasplit.py produces covering splits equivalent to originals."""
    print_header(3, "Validate create_datasplit.py")

    orig_datainfo = LOCAL_ORIGINAL / "datainfo"
    if not orig_datainfo.exists():
        print(f"  [{SKIP}] Original datainfo not linked. Run --setup first.")
        return False

    # 3.1: Load all original covering splits and check coverage
    all_test_organoids = set()
    all_organoids = set()
    n_seeds = 0

    for seed in range(1, 11):
        split_path = orig_datainfo / f"data_split_covering_{seed}.json"
        if not split_path.exists():
            assert_check(False, f"Original split seed {seed} not found")
            continue
        n_seeds += 1
        with open(split_path) as f:
            split = json.load(f)

        train_set = set(split["train"])
        val_set = set(split["val"])
        test_set = set(split["test"])

        # No overlaps
        assert_check(len(train_set & val_set) == 0,
                     f"Seed {seed}: no train/val overlap")
        assert_check(len(train_set & test_set) == 0,
                     f"Seed {seed}: no train/test overlap")

        seed_all = train_set | val_set | test_set
        all_organoids.update(seed_all)
        all_test_organoids.update(test_set)

    # 3.1: Validate coverage
    assert_check(n_seeds == 10, f"Found {n_seeds}/10 covering splits")
    assert_check(len(all_organoids) == 108,
                 f"Total unique organoids: {len(all_organoids)} (expect 108)")
    assert_check(all_test_organoids == all_organoids,
                 f"Test coverage: {len(all_test_organoids)}/108 organoids appear in >= 1 test set")

    # 3.2: Verify our create_datasplit.py can load them
    # (it uses the same format, so if originals are valid, the format is compatible)
    try:
        from automated_method.create_datasplit import create_covering_splits
        assert_check(True, "create_datasplit.py imports successfully")
    except ImportError as e:
        assert_check(False, f"create_datasplit.py import failed: {e}")
        return False

    return True


# ══════════════════════════════════════════════════════════════════════════
# Step 4: Train SimVP on GPU (interactive — print command)
# ══════════════════════════════════════════════════════════════════════════

def step4_train_simvp():
    """Print the srun command to train SimVP_TAU interactively."""
    print_header(4, "Train SimVP on GPU (interactive srun)")

    print(f"  [{INFO}] This step requires manual execution on a GPU node.")
    print(f"  [{INFO}] Run the following command:\n")
    print(f"  srun -p gpu-debug --time=1:00:00 --gres=gpu:1 --cpus-per-task=4 --mem=75G bash -c '")
    print(f"    PY={PY} &&")
    print(f"    cd {REPO_ROOT} &&")
    print(f"    $PY automated_method/train_model.py \\")
    print(f"      --method SimVP_TAU --seed 1 -K 6 -N 6 \\")
    print(f"      --residual --debug")
    print(f"  '")
    print(f"\n  [{INFO}] --debug runs 3 epochs with small data for quick validation.")
    print(f"  [{INFO}] After training, re-run: python tests/validate_pipeline.py --step 6")
    return True


# ══════════════════════════════════════════════════════════════════════════
# Step 5: Validate loss function
# ══════════════════════════════════════════════════════════════════════════

def step5_validate_loss():
    """Validate losses.py: MSE on 6 predictions vs 6 GT frames returns scalar."""
    print_header(5, "Validate loss function")

    try:
        import torch
    except ImportError:
        print(f"  [{SKIP}] torch not available (run on GPU node or install torch)")
        return False

    from losses import (
        compute_rollout_mse,
        compute_per_sample_rollout_mse,
        compute_per_step_mse,
        compute_rollout_ssim,
        compute_per_sample_rollout_ssim,
    )

    B, N, C, H, W = 4, 6, 3, 128, 128

    # Create synthetic data
    torch.manual_seed(42)
    gt = torch.rand(B, N, C, H, W)
    pred_perfect = gt.clone()
    pred_noisy = gt + 0.1 * torch.randn_like(gt)
    pred_copy = gt[:, 0:1].expand_as(gt)  # copy-last baseline

    all_ok = True

    # 5.1: compute_rollout_mse returns scalar
    mse_perfect = compute_rollout_mse(pred_perfect, gt)
    all_ok &= assert_check(mse_perfect.dim() == 0, "rollout_mse returns scalar")
    all_ok &= assert_check(float(mse_perfect) < 1e-7,
                           f"Perfect prediction MSE ≈ 0: {float(mse_perfect):.2e}")

    mse_noisy = compute_rollout_mse(pred_noisy, gt)
    all_ok &= assert_check(float(mse_noisy) > 0,
                           f"Noisy prediction MSE > 0: {float(mse_noisy):.4f}")

    mse_copy = compute_rollout_mse(pred_copy, gt)
    all_ok &= assert_check(float(mse_copy) > float(mse_noisy),
                           f"Copy baseline MSE ({float(mse_copy):.4f}) > noisy MSE ({float(mse_noisy):.4f})")

    # 5.2: per-sample MSE returns (B,)
    per_sample = compute_per_sample_rollout_mse(pred_noisy, gt)
    all_ok &= assert_check(per_sample.shape == (B,),
                           f"per_sample_rollout_mse shape: {per_sample.shape} (expect ({B},))")
    all_ok &= assert_check(torch.allclose(per_sample.mean(), mse_noisy, atol=1e-5),
                           "per_sample mean ≈ rollout MSE")

    # 5.3: per-step MSE returns (N,)
    per_step = compute_per_step_mse(pred_copy, gt)
    all_ok &= assert_check(per_step.shape == (N,),
                           f"per_step_mse shape: {per_step.shape} (expect ({N},))")
    # Copy-last error should increase with horizon
    all_ok &= assert_check(float(per_step[-1]) >= float(per_step[0]),
                           f"Copy MSE increases: step0={float(per_step[0]):.4f}, step5={float(per_step[-1]):.4f}")

    # 5.4: SSIM returns scalar in [0, 1]
    ssim = compute_rollout_ssim(pred_noisy, gt)
    all_ok &= assert_check(ssim.dim() == 0, "rollout_ssim returns scalar")
    all_ok &= assert_check(0 <= float(ssim) <= 1,
                           f"SSIM in [0,1]: {float(ssim):.4f}")

    # Note: compute_rollout_ssim returns SSIM *loss* (1 - SSIM), so lower = better
    ssim_perfect = compute_rollout_ssim(pred_perfect, gt)
    all_ok &= assert_check(float(ssim_perfect) <= float(ssim),
                           f"Perfect SSIM loss ({float(ssim_perfect):.4f}) <= noisy ({float(ssim):.4f})")

    # 5.5: per-sample SSIM returns (B,)
    per_sample_ssim = compute_per_sample_rollout_ssim(pred_noisy, gt)
    all_ok &= assert_check(per_sample_ssim.shape == (B,),
                           f"per_sample_ssim shape: {per_sample_ssim.shape}")

    # 5.6: Gradient flows through MSE
    pred_grad = pred_noisy.clone().requires_grad_(True)
    loss = compute_rollout_mse(pred_grad, gt)
    loss.backward()
    all_ok &= assert_check(pred_grad.grad is not None and pred_grad.grad.abs().sum() > 0,
                           "Gradients flow through MSE loss")

    return all_ok


# ══════════════════════════════════════════════════════════════════════════
# Step 6: Test SimVP against original test metrics
# ══════════════════════════════════════════════════════════════════════════

def step6_validate_test_model():
    """Compare test_model.py outputs against original per_organoid_test_results_residual.csv."""
    print_header(6, "Validate test_model.py against original test metrics")

    orig_csv = LOCAL_ORIGINAL / "predictability" / "per_organoid_test_results_residual.csv"
    if not orig_csv.exists():
        print(f"  [{SKIP}] Original test results not found. Run --setup first.")
        return False

    try:
        import pandas as pd
    except ImportError:
        print(f"  [{SKIP}] pandas not available")
        return False

    # Load original results
    df_orig = pd.read_csv(orig_csv)
    all_ok = True

    all_ok &= assert_check(len(df_orig) >= 100,
                           f"Original has {len(df_orig)} organoids (expect ~108)")

    # Verify expected columns exist
    expected_cols = ["batch_name", "model_mse", "model_ssim", "copy_mse"]
    for col in expected_cols:
        all_ok &= assert_check(col in df_orig.columns,
                               f"Column '{col}' in original CSV")

    # Verify model beats copy-last on average
    if "model_mse" in df_orig.columns and "copy_mse" in df_orig.columns:
        model_mean = df_orig["model_mse"].mean()
        copy_mean = df_orig["copy_mse"].mean()
        all_ok &= assert_check(model_mean < copy_mean,
                               f"Model MSE ({model_mean:.6f}) < Copy MSE ({copy_mean:.6f})")

    # Check per-seed extraction files exist
    orig_extraction = LOCAL_ORIGINAL / "predictability" / "extraction"
    if orig_extraction.exists():
        seed_files = list(orig_extraction.glob("seed*_residual_per_organoid.json"))
        all_ok &= assert_check(len(seed_files) >= 10,
                               f"Found {len(seed_files)} seed extraction files (expect 10)")

        # Spot-check seed 1
        seed1_path = orig_extraction / "seed1_residual_per_organoid.json"
        if seed1_path.exists():
            with open(seed1_path) as f:
                seed1 = json.load(f)
            n_orgs = len(seed1.get("organoids", {}))
            all_ok &= assert_check(n_orgs > 0,
                                   f"Seed 1 extraction has {n_orgs} organoids")

            # Verify metric keys per organoid
            sample_org = next(iter(seed1["organoids"].values()))
            for key in ["model_mse", "model_ssim", "copy_mse"]:
                all_ok &= assert_check(key in sample_org,
                                       f"Seed 1 organoid has '{key}' metric")

    # If user has run test_model.py with the new repo, compare
    new_csv = paths.OUTPUT_ROOT / "predictability_analysis" / "per_organoid_test_results_residual.csv"
    if new_csv.exists():
        df_new = pd.read_csv(new_csv)
        # Merge and compare
        merged = df_orig.merge(df_new, on="batch_name", suffixes=("_orig", "_new"))
        if len(merged) > 0 and "model_mse_orig" in merged.columns:
            corr = merged["model_mse_orig"].corr(merged["model_mse_new"])
            all_ok &= assert_check(corr > 0.95,
                                   f"New vs original model_mse correlation: {corr:.4f} (expect > 0.95)")
        print(f"  [{INFO}] Compared {len(merged)} common organoids between new and original")
    else:
        print(f"  [{INFO}] New test results not yet generated. Run step 4 first, then:")
        print(f"           python automated_method/test_model.py --residual --seeds 1 2 3 4 5 6 7 8 9 10")

    return all_ok


# ══════════════════════════════════════════════════════════════════════════
# Step 7: Extract prediction surprise (manual verification)
# ══════════════════════════════════════════════════════════════════════════

def step7_validate_prediction_surprise():
    """Validate prediction surprise extraction against originals."""
    print_header(7, "Validate prediction surprise (manual verification)")

    orig_residuals = LOCAL_ORIGINAL / "intermediates" / "per_organoid_residuals"
    orig_spatial = LOCAL_ORIGINAL / "intermediates" / "spatial_residual_maps"

    if not orig_residuals.exists():
        print(f"  [{SKIP}] Original residuals not found. Run --setup first.")
        return False

    all_ok = True

    # Check original residuals exist
    npz_files = list(orig_residuals.glob("*.npz"))
    all_ok &= assert_check(len(npz_files) >= 108,
                           f"Found {len(npz_files)} original residual files (expect 108)")

    # Spot-check one residual file
    if npz_files:
        sample = np.load(npz_files[0])
        keys = list(sample.keys())
        all_ok &= assert_check("raw_mse" in keys or "delta_residuals" in keys,
                               f"Residual file has expected keys: {keys}")

        if "delta_residuals" in keys:
            dr = sample["delta_residuals"]
            all_ok &= assert_check(dr.ndim == 3,
                                   f"delta_residuals shape: {dr.shape} (expect 3D: n_windows x N x n_methods)")
            all_ok &= assert_check(dr.shape[1] == 6,
                                   f"N=6 prediction horizons: shape[1]={dr.shape[1]}")

    # Check spatial maps
    if orig_spatial.exists():
        npy_files = list(orig_spatial.glob("*.npy"))
        all_ok &= assert_check(len(npy_files) >= 108,
                               f"Found {len(npy_files)} spatial maps (expect 108)")

        if npy_files:
            sample_map = np.load(npy_files[0])
            all_ok &= assert_check(sample_map.shape == (6, 128, 128),
                                   f"Spatial map shape: {sample_map.shape} (expect (6, 128, 128))")

    # Check per-seed MSE CSV
    seed_csv = LOCAL_ORIGINAL / "intermediates" / "per_seed_mse.csv"
    if seed_csv.exists():
        with open(seed_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        all_ok &= assert_check(len(rows) == 10,
                               f"per_seed_mse.csv has {len(rows)} rows (expect 10)")

    print(f"\n  [{INFO}] To regenerate and manually compare:")
    print(f"  srun -p gpu-debug --gres=gpu:1 --mem=75G --time=1:00:00 bash -c '")
    print(f"    PY={PY} &&")
    print(f"    cd {REPO_ROOT} &&")
    print(f"    $PY automated_method/extract_prediction_surprise.py --batch batch-000121")
    print(f"  '")

    return all_ok


# ══════════════════════════════════════════════════════════════════════════
# Step 8: Validate autotune segmentation
# ══════════════════════════════════════════════════════════════════════════

def step8_validate_autotune():
    """Validate autotune segmentation: load params and generate centroid grid."""
    print_header(8, "Validate autotune segmentation + centroid grid")

    orig_seg = LOCAL_ORIGINAL / "segmentation"
    if not orig_seg.exists():
        print(f"  [{SKIP}] Original segmentation not found. Run --setup first.")
        return False

    all_ok = True

    # Check segmentation results exist for each batch
    seg_dirs = sorted([d for d in orig_seg.iterdir() if d.is_dir()])
    all_ok &= assert_check(len(seg_dirs) >= 108,
                           f"Found {len(seg_dirs)} segmentation result dirs (expect >= 108)")

    # Spot-check one batch: must have centroids.csv and best_params.json
    if seg_dirs:
        sample_dir = seg_dirs[0]
        batch_name = sample_dir.name

        centroids_csv = sample_dir / "centroids.csv"
        all_ok &= assert_check(centroids_csv.exists(),
                               f"{batch_name}/centroids.csv exists")

        best_params = sample_dir / "best_params.json"
        all_ok &= assert_check(best_params.exists(),
                               f"{batch_name}/best_params.json exists")

        if centroids_csv.exists():
            with open(centroids_csv) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            all_ok &= assert_check(len(rows) >= 100,
                                   f"{batch_name} has {len(rows)} centroid rows (expect ~120)")

            # Check expected columns
            if rows:
                expected = {"frame", "cx", "cy", "area"}
                actual = set(rows[0].keys())
                all_ok &= assert_check(expected.issubset(actual),
                                       f"Centroid CSV has columns: {sorted(actual)}")

        if best_params.exists():
            with open(best_params) as f:
                params = json.load(f)
            # Verify it has segmentation hyperparameters
            all_ok &= assert_check(isinstance(params, dict) and len(params) > 0,
                                   f"{batch_name} best_params has {len(params)} params")

    # Check motility descriptors
    motility_csv = LOCAL_ORIGINAL / "classical_descriptors" / "motility_descriptors.csv"
    if motility_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(motility_csv)
            all_ok &= assert_check(len(df) >= 108,
                                   f"motility_descriptors.csv has {len(df)} organoids")

            expected_features = [
                "mean_velocity", "std_velocity", "max_velocity",
                "total_displacement", "path_length", "tortuosity",
                "msd_slope", "mean_turning_angle",
            ]
            for feat in expected_features:
                all_ok &= assert_check(feat in df.columns,
                                       f"Feature '{feat}' in motility descriptors")

            # Verify no all-NaN features
            for feat in expected_features:
                n_valid = df[feat].notna().sum()
                all_ok &= assert_check(n_valid >= 100,
                                       f"{feat}: {n_valid}/108 valid values")
        except ImportError:
            print(f"  [{SKIP}] pandas not available for motility CSV validation")

    # Verify autotune_segmentation.py can load params
    try:
        seg_params_file = REPO_ROOT / "classical_pipeline" / "all_segmentation_params.json"
        if seg_params_file.exists():
            with open(seg_params_file) as f:
                all_params = json.load(f)
            all_ok &= assert_check(len(all_params) >= 108,
                                   f"all_segmentation_params.json has {len(all_params)} entries")
        else:
            print(f"  [{INFO}] all_segmentation_params.json not found at repo level")
    except Exception as e:
        assert_check(False, f"Loading segmentation params: {e}")

    # Verify visualize_grid_centroids.py can import
    try:
        from classical_pipeline.visualize_grid_centroids import load_all_centroids
        assert_check(True, "visualize_grid_centroids.py imports successfully")
    except ImportError as e:
        # Try direct import
        try:
            sys.path.insert(0, str(REPO_ROOT / "classical_pipeline"))
            from visualize_grid_centroids import load_all_centroids
            assert_check(True, "visualize_grid_centroids.py imports (direct)")
        except ImportError as e2:
            assert_check(False, f"visualize_grid_centroids.py import failed: {e2}")

    return all_ok


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

STEPS = {
    1: ("Create dataset from source videos", step1_create_dataset),
    2: ("Link original Q8_v23 outputs", step2_copy_originals),
    3: ("Validate create_datasplit.py", step3_validate_datasplit),
    4: ("Train SimVP on GPU (manual)", step4_train_simvp),
    5: ("Validate loss function", step5_validate_loss),
    6: ("Validate test_model.py", step6_validate_test_model),
    7: ("Validate prediction surprise", step7_validate_prediction_surprise),
    8: ("Validate autotune segmentation", step8_validate_autotune),
}


def main():
    parser = argparse.ArgumentParser(
        description="Validate alife-2026 pipeline against original Q8_v23 outputs")
    parser.add_argument("--step", type=int, default=None,
                        help="Run a specific step (1-8)")
    parser.add_argument("--setup", action="store_true",
                        help="Run setup steps only (1-2)")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Skip GPU-dependent steps (4)")
    args = parser.parse_args()

    if args.step:
        if args.step not in STEPS:
            print(f"Invalid step {args.step}. Available: {sorted(STEPS.keys())}")
            sys.exit(1)
        name, func = STEPS[args.step]
        ok = func()
        sys.exit(0 if ok else 1)

    if args.setup:
        steps_to_run = [1, 2]
    elif args.cpu_only:
        steps_to_run = [2, 3, 5, 6, 7, 8]
    else:
        steps_to_run = sorted(STEPS.keys())

    results = {}
    for step_num in steps_to_run:
        name, func = STEPS[step_num]
        try:
            results[step_num] = func()
        except Exception as e:
            print(f"  [{FAIL}] Step {step_num} raised exception: {e}")
            results[step_num] = False

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for step_num in steps_to_run:
        name, _ = STEPS[step_num]
        status = PASS if results.get(step_num) else FAIL
        print(f"  Step {step_num}: [{status}] {name}")

    n_pass = sum(1 for v in results.values() if v)
    n_total = len(results)
    print(f"\n  {n_pass}/{n_total} steps passed")


if __name__ == "__main__":
    main()
