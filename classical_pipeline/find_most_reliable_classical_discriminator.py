"""
find_most_reliable_classical_discriminator.py — Identify the most reliable classical
motility descriptor (paper Fig 3A).

For each of the 8 motility features, each organoid's non-overlapping windows are treated as
repeated measurements and the one-way ANOVA F-statistic is computed:

    F = (between-organoid variance) / (within-organoid variance)

A high F means the feature varies more across organoids than within them. Mean speed has the
highest F, establishing the reference ranking that prediction surprise is evaluated against.

Usage:
    python find_most_reliable_classical_discriminator.py

Inputs:
    CLASSICAL_DIR / centroid_trajectories.npz   (from extract_motility_features.py)

Outputs -> CLASSICAL_DIR /
    window_descriptors.csv              per-organoid, per-window 8-feature table
    f_statistics.csv                    feature, F_statistic (sorted desc)
    descriptor_reliability_report.json  best discriminator + ranking
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import paths

WINDOW_SIZE = 12   # frames per non-overlapping window (= model K+N)
N_FRAMES = 120

METRICS = ["mean_velocity", "acceleration", "path_length", "total_displacement",
           "tortuosity", "outreach_ratio", "mean_turning_angle", "msd_slope"]


def compute_window_descriptors(centroids):
    """8 motility features per non-overlapping WINDOW_SIZE-frame window of a (T,2) track.

    All features are computed from the raw (unsmoothed) centroid trajectory.
    """
    T = centroids.shape[0]
    results = []
    for start in range(0, T - WINDOW_SIZE + 1, WINDOW_SIZE):
        end = start + WINDOW_SIZE
        c = centroids[start:end]
        valid = ~np.isnan(c[:, 0])
        if valid.sum() < 3:
            continue
        cv = c[valid]
        disp = cv[1:] - cv[:-1]
        step_lengths = np.linalg.norm(disp, axis=1)
        mean_vel = np.mean(step_lengths) if len(step_lengths) > 0 else np.nan

        total_disp = np.linalg.norm(cv[-1] - cv[0])
        path_len = np.sum(step_lengths)
        tort = path_len / (total_disp + 1e-8)
        max_disp = np.max(np.linalg.norm(cv - cv[0], axis=1))
        outreach_ratio = max_disp / (path_len + 1e-8)

        if len(disp) > 1:
            turning = []
            for i in range(len(disp) - 1):
                d1, d2 = disp[i], disp[i + 1]
                n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
                if n1 > 1e-8 and n2 > 1e-8:
                    cos_a = np.clip(np.dot(d1, d2) / (n1 * n2), -1, 1)
                    turning.append(np.arccos(cos_a))
            mean_turn = np.mean(turning) if turning else np.nan
        else:
            mean_turn = np.nan

        lags = range(1, min(6, len(cv)))
        msd_vals = [np.mean(np.sum((cv[lag:] - cv[:-lag]) ** 2, axis=1)) for lag in lags]
        if len(msd_vals) > 1:
            log_lags = np.log(list(lags))
            log_msd = np.log(np.array(msd_vals) + 1e-15)
            valid_msd = np.isfinite(log_msd)
            if valid_msd.sum() > 1:
                msd_slope = np.polyfit(log_lags[valid_msd], log_msd[valid_msd], 1)[0]
            else:
                msd_slope = np.nan
        else:
            msd_slope = np.nan

        accel = np.mean(np.abs(np.diff(step_lengths))) if len(step_lengths) > 1 else np.nan

        results.append({
            "mean_velocity": mean_vel, "acceleration": accel,
            "path_length": path_len, "total_displacement": total_disp,
            "tortuosity": tort, "outreach_ratio": outreach_ratio,
            "mean_turning_angle": mean_turn, "msd_slope": msd_slope,
        })
    return results


def compute_f_statistic(per_organoid_windows):
    """One-way ANOVA F = between-organoid / within-organoid variance.

    per_organoid_windows: {batch_name: [values...]} (one value per window).
    """
    groups = [np.array(v) for v in per_organoid_windows.values() if len(v) > 0]
    groups = [g[~np.isnan(g)] for g in groups]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return np.nan
    grand_mean = np.mean(np.concatenate(groups))
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_within = sum(np.sum((g - g.mean()) ** 2) for g in groups)
    df_between = len(groups) - 1
    df_within = sum(len(g) - 1 for g in groups)
    if df_within == 0:
        return np.nan
    return (ss_between / df_between) / (ss_within / df_within + 1e-12)


def main():
    npz_path = paths.CLASSICAL_DIR / "centroid_trajectories.npz"
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found. Run extract_motility_features.py first.")
        sys.exit(1)

    centroids_data = np.load(npz_path)
    batch_names = list(centroids_data.files)
    print(f"Loaded {len(batch_names)} organoid trajectories")

    # Per-organoid window descriptors (long table)
    rows = []
    all_windows = {}
    for b in batch_names:
        w = compute_window_descriptors(centroids_data[b])
        all_windows[b] = w
        for wi, d in enumerate(w):
            rows.append({"batch_name": b, "window": wi, **d})
    win_df = pd.DataFrame(rows)
    win_csv = paths.CLASSICAL_DIR / "window_descriptors.csv"
    win_df.to_csv(win_csv, index=False)
    print(f"Saved: {win_csv} ({len(win_df)} organoid-windows)")

    # F-statistic per feature
    f_stats = {}
    for m in METRICS:
        per_org = {}
        for b, windows in all_windows.items():
            vals = [w[m] for w in windows if not np.isnan(w.get(m, np.nan))]
            if vals:
                per_org[b] = vals
        f_stats[m] = float(compute_f_statistic(per_org))

    sorted_metrics = sorted(METRICS, key=lambda k: f_stats.get(k, 0), reverse=True)
    f_df = pd.DataFrame({"feature": sorted_metrics,
                         "F_statistic": [round(f_stats[m], 3) for m in sorted_metrics]})
    f_csv = paths.CLASSICAL_DIR / "f_statistics.csv"
    f_df.to_csv(f_csv, index=False)
    print(f"Saved: {f_csv}")

    print(f"\n{'Feature':<22s} {'F-stat':>10s}")
    print("-" * 34)
    for m in sorted_metrics:
        print(f"{m:<22s} {f_stats[m]:>10.1f}")

    best = sorted_metrics[0]
    report = {
        "best_discriminator": best,
        "best_F_statistic": round(f_stats[best], 3),
        "ranking": sorted_metrics,
        "n_organoids": len(batch_names),
        "method": "one-way ANOVA between/within organoid variance over non-overlapping windows",
    }
    report_path = paths.CLASSICAL_DIR / "descriptor_reliability_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nBest discriminator: {best} (F={f_stats[best]:.1f})")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
