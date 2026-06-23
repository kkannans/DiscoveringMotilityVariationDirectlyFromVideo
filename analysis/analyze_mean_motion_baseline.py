"""
Falsification test: mean-motion baseline vs learned model.

The mean-motion baseline predicts each future frame by translating the last
context frame by h × mean_displacement_vector (h = 1..6). It "knows" about
motion (average speed + direction) without learning dynamics.

If the learned model's advantage pattern (over copy) matches mean-motion's
pattern, the model just learned average displacement. If the model beats
mean-motion during acceleration and velocity deviations, it genuinely
learned temporal structure.

CPU-only. Loads videos via OpenCV, computes MSE against ground truth.

Outputs -> paths.SUPPLEMENTARY_DIR
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import shift
from scipy.stats import spearmanr
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths

OUTPUT_DIR = paths.SUPPLEMENTARY_DIR
RESIDUALS_DIR = paths.INTERMEDIATES_DIR / "per_organoid_residuals"
CENTROID_NPZ = paths.CLASSICAL_DIR / "centroid_trajectories.npz"
MOTILITY_CSV = paths.CLASSICAL_DIR / "motility_descriptors.csv"
VIDEO_DIR = paths.DATA_ROOT

K, N = 6, 6
FRAME_SIZE = 128
N_FRAMES = 120


def load_video_128(batch_name):
    """Load first 120 frames as (T, 3, H, W) float32 in [0, 1]."""
    path = VIDEO_DIR / f"{batch_name}.mp4"
    cap = cv2.VideoCapture(str(path))
    frames = []
    for _ in range(N_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
        # (H, W, 3) -> (3, H, W) float32 [0, 1]
        frames.append(frame.transpose(2, 0, 1).astype(np.float32) / 255.0)
    cap.release()
    return np.stack(frames)  # (T, 3, H, W)


def translate_frame(frame, dx, dy):
    """Translate a (3, H, W) frame by (dx, dy) pixels using scipy shift.

    Boundary pixels are filled with the nearest edge value (mode='nearest').
    """
    shifted = np.empty_like(frame)
    for c in range(3):
        shifted[c] = shift(frame[c], [dy, dx], order=1, mode="nearest")
    return shifted


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mot = pd.read_csv(MOTILITY_CSV)
    centroids_data = np.load(CENTROID_NPZ)
    batches = sorted([p.stem for p in RESIDUALS_DIR.glob("*.npz")])
    print(f"Processing {len(batches)} organoids")

    all_rows = []

    for idx, batch in enumerate(batches):
        # Load cached model/copy MSE
        npz = np.load(RESIDUALS_DIR / f"{batch}.npz")
        raw_mse = npz["raw_mse"]  # (n_windows, N, 5)
        n_win = raw_mse.shape[0]
        model_mse_win = raw_mse[:, :, 0].mean(axis=1)  # (n_win,)
        copy_mse_win = raw_mse[:, :, 1].mean(axis=1)

        # Compute mean displacement vector from centroids
        if batch not in centroids_data:
            continue
        c = centroids_data[batch]
        valid = ~np.isnan(c[:, 0])
        if valid.sum() < 5:
            continue
        cv = c[valid]
        displacements = cv[1:] - cv[:-1]
        mean_dx = displacements[:, 0].mean()  # mean x displacement per frame
        mean_dy = displacements[:, 1].mean()  # mean y displacement per frame

        # Load video frames
        frames = load_video_128(batch)  # (T, 3, H, W)
        if frames.shape[0] < N_FRAMES:
            print(f"  SKIP {batch}: only {frames.shape[0]} frames")
            continue

        # Compute mean-motion MSE for each window
        mean_motion_mse_win = np.full(n_win, np.nan)

        for w in range(n_win):
            start = w * N
            ctx_end = start + K
            tgt_end = ctx_end + N

            if tgt_end > N_FRAMES:
                break

            last_ctx = frames[ctx_end - 1]  # (3, H, W)
            tgt = frames[ctx_end:tgt_end]    # (N, 3, H, W)

            # Mean-motion prediction: shift last frame by h * mean_displacement
            mm_mse = 0.0
            for h in range(N):
                pred = translate_frame(last_ctx, mean_dx * (h + 1), mean_dy * (h + 1))
                pred = np.clip(pred, 0, 1)
                mm_mse += np.mean((pred - tgt[h]) ** 2)
            mean_motion_mse_win[w] = mm_mse / N

        # Local velocity per window
        for w in range(n_win):
            start = w * N
            end = start + K + N
            if end <= len(c):
                seg = c[start:end]
                v = ~np.isnan(seg[:, 0])
                if v.sum() > 1:
                    sv = seg[v]
                    local_vel = np.linalg.norm(sv[1:] - sv[:-1], axis=1).mean()
                else:
                    local_vel = np.nan
            else:
                local_vel = np.nan

            # Local acceleration
            if end <= len(c):
                seg = c[start:end]
                v = ~np.isnan(seg[:, 0])
                if v.sum() > 2:
                    sv = seg[v]
                    steps = np.linalg.norm(sv[1:] - sv[:-1], axis=1)
                    local_accel = np.mean(np.abs(np.diff(steps))) if len(steps) > 1 else np.nan
                else:
                    local_accel = np.nan
            else:
                local_accel = np.nan

            all_rows.append({
                "batch": batch, "window": w,
                "model_mse": model_mse_win[w],
                "copy_mse": copy_mse_win[w],
                "mean_motion_mse": mean_motion_mse_win[w],
                "local_vel": local_vel,
                "local_accel": local_accel,
                "mean_dx": mean_dx,
                "mean_dy": mean_dy,
            })

        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{len(batches)} done")

    df = pd.DataFrame(all_rows)
    valid = ~np.isnan(df["mean_motion_mse"]) & ~np.isnan(df["local_vel"])
    df = df[valid].reset_index(drop=True)
    print(f"\nTotal valid windows: {len(df)}")

    # ================================================================
    # Key comparisons
    # ================================================================
    print("\n" + "=" * 60)
    print("Falsification test: model vs mean-motion baseline")
    print("=" * 60)

    # Ratios
    df["ratio_model_copy"] = df["model_mse"] / (df["copy_mse"] + 1e-12)
    df["ratio_model_mm"] = df["model_mse"] / (df["mean_motion_mse"] + 1e-12)
    df["ratio_mm_copy"] = df["mean_motion_mse"] / (df["copy_mse"] + 1e-12)

    # Advantages
    df["adv_model_copy"] = df["copy_mse"] - df["model_mse"]
    df["adv_model_mm"] = df["mean_motion_mse"] - df["model_mse"]
    df["adv_mm_copy"] = df["copy_mse"] - df["mean_motion_mse"]

    print(f"\n  Overall MSE (mean across all windows):")
    print(f"    Model:       {df['model_mse'].mean():.6f}")
    print(f"    Copy-last:   {df['copy_mse'].mean():.6f}")
    print(f"    Mean-motion: {df['mean_motion_mse'].mean():.6f}")

    print(f"\n  Model vs copy-last: model wins {(df['adv_model_copy'] > 0).mean()*100:.1f}%")
    print(f"  Model vs mean-motion: model wins {(df['adv_model_mm'] > 0).mean()*100:.1f}%")
    print(f"  Mean-motion vs copy: mean-motion wins {(df['adv_mm_copy'] > 0).mean()*100:.1f}%")

    # By velocity decile
    df["vel_decile"] = pd.qcut(df["local_vel"], q=10, labels=False, duplicates="drop")
    print(f"\n  Ratio MSE_model / MSE_mean_motion by velocity decile:")
    for d in sorted(df["vel_decile"].unique()):
        sub = df[df["vel_decile"] == d]
        print(f"    D{d}: vel=[{sub['local_vel'].min():.2f}-{sub['local_vel'].max():.2f}], "
              f"model/mm={sub['ratio_model_mm'].mean():.4f}, "
              f"model/copy={sub['ratio_model_copy'].mean():.4f}, "
              f"mm/copy={sub['ratio_mm_copy'].mean():.4f}")

    # Does model still beat mean-motion during acceleration?
    df["accel_tercile"] = pd.qcut(df["local_accel"], q=3, labels=["Low", "Med", "High"],
                                   duplicates="drop")
    print(f"\n  Model vs mean-motion by acceleration tercile:")
    for t in ["Low", "Med", "High"]:
        sub = df[df["accel_tercile"] == t]
        model_wins = (sub["adv_model_mm"] > 0).mean() * 100
        mean_ratio = sub["ratio_model_mm"].mean()
        print(f"    {t:4s}: model wins {model_wins:.1f}%, "
              f"ratio model/mm = {mean_ratio:.4f} (n={len(sub)})")

    # Correlation: model advantage over mean-motion vs features
    print(f"\n  Spearman correlations with (MSE_mm - MSE_model):")
    for name, col in [("Local velocity", "local_vel"), ("Local acceleration", "local_accel")]:
        r, p = spearmanr(df[col], df["adv_model_mm"])
        print(f"    {name:20s}: rho = {r:+.4f}, p = {p:.2e}")

    # ================================================================
    # Plot: 3-panel comparison
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: MSE ratio (model/copy and model/mean-motion) vs velocity decile
    ax = axes[0]
    decile_stats = df.groupby("vel_decile").agg(
        vel_mid=("local_vel", "median"),
        ratio_mc=("ratio_model_copy", "mean"),
        ratio_mc_std=("ratio_model_copy", "std"),
        ratio_mm=("ratio_model_mm", "mean"),
        ratio_mm_std=("ratio_model_mm", "std"),
    ).reset_index()

    ax.errorbar(decile_stats["vel_mid"], decile_stats["ratio_mc"],
                yerr=decile_stats["ratio_mc_std"], color="#1f77b4",
                marker="o", linewidth=2, capsize=3, label="Model / Copy")
    ax.errorbar(decile_stats["vel_mid"], decile_stats["ratio_mm"],
                yerr=decile_stats["ratio_mm_std"], color="#d62728",
                marker="s", linewidth=2, capsize=3, label="Model / Mean-motion")
    ax.axhline(1, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Local velocity (px/frame)", fontsize=10)
    ax.set_ylabel("MSE ratio (lower = model better)", fontsize=10)
    ax.set_title("(A) Model compression ratio\nvs velocity", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 2.5)

    # Panel B: Raw advantage over mean-motion vs local velocity
    ax = axes[1]
    ax.scatter(df["local_vel"], df["adv_model_mm"], s=5, alpha=0.2,
               color="#d62728", edgecolors="none", rasterized=True)

    # Binned means
    bins = np.percentile(df["local_vel"], np.arange(0, 101, 10))
    bc, bm, bs = [], [], []
    for i in range(len(bins) - 1):
        mask = (df["local_vel"] >= bins[i]) & (df["local_vel"] < bins[i + 1])
        if mask.sum() > 0:
            bc.append((bins[i] + bins[i + 1]) / 2)
            bm.append(df.loc[mask, "adv_model_mm"].mean())
            bs.append(df.loc[mask, "adv_model_mm"].std())
    ax.errorbar(bc, bm, yerr=bs, color="black", linewidth=2, capsize=3,
                marker="o", markersize=5, zorder=5, label="Decile mean ± std")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Local velocity (px/frame)", fontsize=10)
    ax.set_ylabel("MSE(mean-motion) − MSE(model)", fontsize=10)
    ax.set_title("(B) Model advantage over\nmean-motion baseline", fontsize=11)
    ax.legend(fontsize=8)

    # Panel C: By acceleration tercile — does model beat mean-motion more during accel?
    ax = axes[2]
    accel_labels = ["Low", "Med", "High"]
    accel_colors = ["#2ca02c", "#f39c12", "#d62728"]
    box_data = [df[df["accel_tercile"] == t]["adv_model_mm"].values for t in accel_labels]

    bp = ax.boxplot(box_data, tick_labels=[f"{t}\naccel" for t in accel_labels],
                    patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], accel_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    rng = np.random.default_rng(42)
    for i, data in enumerate(box_data):
        jitter = rng.uniform(-0.1, 0.1, len(data))
        ax.scatter(np.full(len(data), i + 1) + jitter, data,
                   c="black", s=3, alpha=0.15, edgecolors="none", rasterized=True)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("MSE(mean-motion) − MSE(model)", fontsize=10)
    ax.set_title("(C) Model advantage over mean-motion\nby acceleration level", fontsize=11)

    # Annotate win rates
    for i, t in enumerate(accel_labels):
        sub = df[df["accel_tercile"] == t]
        win = (sub["adv_model_mm"] > 0).mean() * 100
        ax.text(i + 1, ax.get_ylim()[1] * 0.9, f"{win:.0f}%\nwin",
                ha="center", fontsize=8, fontweight="bold")

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUTPUT_DIR / f"falsification_mean_motion.{ext}",
                    dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"\nSaved: {OUTPUT_DIR / 'falsification_mean_motion.png'}")

    # Save CSV
    df.to_csv(OUTPUT_DIR / "mean_motion_baseline_per_window.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'mean_motion_baseline_per_window.csv'}")


if __name__ == "__main__":
    main()
