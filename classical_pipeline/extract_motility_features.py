"""
Recompute centroid-derived motility descriptors using M3 (auto-tuned) centroids.

Reads pre-computed centroids from auto_tune_segmentation results and computes:
  - Mean velocity (px/frame)
  - MSD slope (diffusion exponent)
  - Tortuosity (path_length / displacement)
  - Plus: total displacement, path length, max velocity, std velocity,
    confinement ratio, direction autocorrelation, turning angle

Usage:
    python compute_centroid_descriptors_m3.py

Inputs:
    {SEGMENTATION_DIR}/<batch>/centroids.csv

Outputs -> {CLASSICAL_DIR}/
    motility_descriptors.csv     — per-organoid descriptors (108 rows)
    centroid_trajectories.npz    — raw centroid arrays for all organoids
    msd_curves.npz               — MSD curves for all organoids
"""

import csv
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import paths

AUTOTUNE_DIR = paths.SEGMENTATION_DIR
OUTPUT_DIR = paths.CLASSICAL_DIR
NUM_PRE_FRAMES = 120  # 120 frames per video


def load_m3_centroids(batch_name):
    """Load centroids from auto-tune results CSV. Returns (T, 2) array."""
    csv_path = AUTOTUNE_DIR / batch_name / "centroids.csv"
    df = pd.read_csv(csv_path)
    # Use all 120 frames
    df = df[df["frame"] < NUM_PRE_FRAMES]
    centroids = np.full((NUM_PRE_FRAMES, 2), np.nan)
    for _, row in df.iterrows():
        t = int(row["frame"])
        if t < NUM_PRE_FRAMES and not (np.isnan(row["cx"]) or np.isnan(row["cy"])):
            centroids[t, 0] = row["cx"]
            centroids[t, 1] = row["cy"]
    return centroids


def compute_msd(centroids, max_lag=50):
    """Mean squared displacement as a function of lag time."""
    T = len(centroids)
    valid = ~np.isnan(centroids[:, 0])
    msd = []
    for lag in range(1, min(max_lag, T)):
        displacements = centroids[lag:] - centroids[:-lag]
        mask = valid[lag:] & valid[:-lag]
        if mask.sum() > 0:
            sq_disp = np.sum(displacements[mask] ** 2, axis=1)
            msd.append(np.mean(sq_disp))
        else:
            msd.append(np.nan)
    return np.array(msd)


def smooth_trajectory(c, window=7, polyorder=2):
    """Savitzky-Golay smooth a (T, 2) centroid trajectory.

    Removes frame-to-frame segmentation noise while preserving directed
    drift. Standard in cell tracking (Metzler et al., Phys Rep 2014).

    Args:
        c: (T, 2) array of centroids (no NaNs)
        window: filter window length (must be odd, >= polyorder+1)
        polyorder: polynomial order for local fit

    Returns:
        (T, 2) smoothed centroids
    """
    from scipy.signal import savgol_filter
    T = len(c)
    if T < window:
        window = max(polyorder + 1, T if T % 2 == 1 else T - 1)
    if window < polyorder + 1:
        return c.copy()
    return np.column_stack([
        savgol_filter(c[:, 0], window, polyorder),
        savgol_filter(c[:, 1], window, polyorder),
    ])


def compute_descriptors(centroids):
    """Compute centroid-derived motility descriptors from (T, 2) centroids.

    All metrics use the raw centroid trajectory (no smoothing). Velocity metrics
    (mean_velocity, std_velocity, max_velocity) reflect the frame-to-frame motion.
    """
    valid = ~np.isnan(centroids[:, 0])
    nan_result = {k: np.nan for k in [
        "total_displacement", "path_length", "tortuosity",
        "mean_velocity", "std_velocity", "max_velocity",
        "velocity_autocorr_lag1", "direction_autocorr_lag1",
        "mean_turning_angle", "confinement_ratio", "msd_slope",
        "valid_frames",
    ]}
    if valid.sum() < 5:
        return nan_result

    c = centroids[valid]
    T = len(c)

    # Raw displacements for velocity metrics (what the model sees)
    raw_disp = c[1:] - c[:-1]
    step_lengths = np.linalg.norm(raw_disp, axis=1)

    mean_velocity = np.mean(step_lengths)
    std_velocity = np.std(step_lengths)
    max_velocity = np.max(step_lengths)

    # Velocity autocorrelation at lag 1 (raw)
    if len(step_lengths) > 2:
        v = step_lengths - np.mean(step_lengths)
        denom = np.sum(v ** 2)
        velocity_autocorr_lag1 = np.sum(v[:-1] * v[1:]) / denom if denom > 1e-12 else 0.0
    else:
        velocity_autocorr_lag1 = np.nan

    # Path-based and directional metrics (raw trajectory)
    cs = c
    smooth_disp = cs[1:] - cs[:-1]
    smooth_step = np.linalg.norm(smooth_disp, axis=1)

    total_displacement = np.linalg.norm(cs[-1] - cs[0])
    path_length = np.sum(smooth_step)
    tortuosity = path_length / (total_displacement + 1e-8)

    # Direction autocorrelation (smoothed)
    angles = np.arctan2(smooth_disp[:, 1], smooth_disp[:, 0])
    if len(angles) > 2:
        angle_diffs = angles[1:] - angles[:-1]
        angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
        direction_autocorr_lag1 = np.mean(np.cos(angle_diffs))
    else:
        direction_autocorr_lag1 = np.nan

    # Mean turning angle (smoothed)
    if len(smooth_disp) > 1:
        turning_angles = []
        for i in range(len(smooth_disp) - 1):
            d1, d2 = smooth_disp[i], smooth_disp[i + 1]
            n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
            if n1 > 1e-8 and n2 > 1e-8:
                cos_a = np.clip(np.dot(d1, d2) / (n1 * n2), -1, 1)
                turning_angles.append(np.arccos(cos_a))
        mean_turning_angle = np.mean(turning_angles) if turning_angles else np.nan
    else:
        mean_turning_angle = np.nan

    # Confinement ratio (smoothed)
    max_dist = np.max(np.linalg.norm(cs - cs[0], axis=1))
    confinement_ratio = max_dist / (path_length + 1e-8)

    # MSD slope (diffusion exponent)
    msd = compute_msd(centroids, max_lag=min(50, T // 2))
    valid_msd = ~np.isnan(msd) & (msd > 0)
    if valid_msd.sum() > 3:
        lags = np.arange(1, len(msd) + 1)[valid_msd]
        msd_slope = np.polyfit(np.log(lags), np.log(msd[valid_msd]), 1)[0]
    else:
        msd_slope = np.nan

    return {
        "total_displacement": float(total_displacement),
        "path_length": float(path_length),
        "tortuosity": float(tortuosity),
        "mean_velocity": float(mean_velocity),
        "std_velocity": float(std_velocity),
        "max_velocity": float(max_velocity),
        "velocity_autocorr_lag1": float(velocity_autocorr_lag1),
        "direction_autocorr_lag1": float(direction_autocorr_lag1),
        "mean_turning_angle": float(mean_turning_angle),
        "confinement_ratio": float(confinement_ratio),
        "msd_slope": float(msd_slope),
        "valid_frames": int(valid.sum()),
    }


def main():
    batches = sorted([d.name for d in AUTOTUNE_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(batches)} batches with auto-tune results")

    all_descriptors = []
    all_centroids = {}
    all_msd = {}

    for batch in batches:
        centroids = load_m3_centroids(batch)
        desc = compute_descriptors(centroids)
        desc["batch_name"] = batch
        all_descriptors.append(desc)
        all_centroids[batch] = centroids
        all_msd[batch] = compute_msd(centroids)

    # Save CSV
    df = pd.DataFrame(all_descriptors)
    cols = ["batch_name"] + [c for c in df.columns if c != "batch_name"]
    df = df[cols]
    csv_path = OUTPUT_DIR / "motility_descriptors.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Save trajectories
    npz_path = OUTPUT_DIR / "centroid_trajectories.npz"
    np.savez_compressed(npz_path, **all_centroids)
    print(f"Saved: {npz_path}")

    # Save MSD curves
    msd_path = OUTPUT_DIR / "msd_curves.npz"
    np.savez_compressed(msd_path, **all_msd)
    print(f"Saved: {msd_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Motility Descriptors Summary (M3 auto-tuned masks)")
    print(f"{'='*60}")
    for col in ["mean_velocity", "msd_slope", "tortuosity", "total_displacement",
                 "confinement_ratio", "direction_autocorr_lag1"]:
        vals = df[col].dropna()
        print(f"  {col:30s}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"range=[{vals.min():.4f}, {vals.max():.4f}]")

    # Print High/Low comparison
    print(f"\n{'='*60}")
    print(f"High (batch-000191) vs Low (batch-000114)")
    print(f"{'='*60}")
    high = df[df["batch_name"] == "batch-000191"].iloc[0]
    low = df[df["batch_name"] == "batch-000114"].iloc[0]
    for col in ["mean_velocity", "msd_slope", "tortuosity", "total_displacement",
                 "path_length", "max_velocity", "confinement_ratio",
                 "direction_autocorr_lag1", "mean_turning_angle", "valid_frames"]:
        print(f"  {col:30s}: High={high[col]:.4f}  Low={low[col]:.4f}")


if __name__ == "__main__":
    main()
