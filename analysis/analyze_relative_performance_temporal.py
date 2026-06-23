"""
Model vs copy-last performance as a function of time and speed.

For each organoid, each of 19 windows:
  - Compute MSE(model) and MSE(copy) averaged over 6 prediction horizons
  - Compute local velocity from centroid trajectory in that window
  - Raw advantage: MSE(copy) - MSE(model) (positive = model wins)

Plots:
  (A) Faceted by velocity tercile: raw advantage temporal profiles on own scale.
  (B) Raw advantage vs velocity rank (rank-normalized, no division).
  (C) Heatmap: organoids (sorted by mean velocity) × window position.

Outputs -> paths.SUPPLEMENTARY_DIR
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths

OUTPUT_DIR = paths.SUPPLEMENTARY_DIR
RESIDUALS_DIR = paths.INTERMEDIATES_DIR / "per_organoid_residuals"
CENTROID_NPZ = paths.CLASSICAL_DIR / "centroid_trajectories.npz"
MOTILITY_CSV = paths.CLASSICAL_DIR / "motility_descriptors.csv"

K, N = 6, 6


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mot = pd.read_csv(MOTILITY_CSV)
    centroids_data = np.load(CENTROID_NPZ)

    batches = sorted([p.stem for p in RESIDUALS_DIR.glob("*.npz")])
    print(f"Found {len(batches)} organoids with cached residuals")

    # Collect per-window data
    all_rows = []
    per_org_profiles = {}

    for batch in batches:
        npz = np.load(RESIDUALS_DIR / f"{batch}.npz")
        raw_mse = npz["raw_mse"]  # (n_windows, N, 5)
        n_windows = raw_mse.shape[0]

        model_mse = raw_mse[:, :, 0].mean(axis=1)  # (n_windows,)
        copy_mse = raw_mse[:, :, 1].mean(axis=1)
        raw_advantage = copy_mse - model_mse

        # Local velocity per window
        if batch in centroids_data:
            c = centroids_data[batch]
            local_vel = np.full(n_windows, np.nan)
            for w in range(n_windows):
                start = w * N
                end = start + K + N
                if end <= len(c):
                    seg = c[start:end]
                    valid = ~np.isnan(seg[:, 0])
                    if valid.sum() > 1:
                        sv = seg[valid]
                        steps = np.linalg.norm(sv[1:] - sv[:-1], axis=1)
                        local_vel[w] = steps.mean()
        else:
            local_vel = np.full(n_windows, np.nan)

        mot_row = mot[mot["batch_name"] == batch]
        mean_vel = mot_row["mean_velocity"].values[0] if len(mot_row) > 0 else np.nan
        conf_ratio = mot_row["confinement_ratio"].values[0] if len(mot_row) > 0 else np.nan

        window_times = np.array([(w * N + K) * 2.5 for w in range(n_windows)])

        per_org_profiles[batch] = {
            "model_mse": model_mse, "copy_mse": copy_mse,
            "raw_advantage": raw_advantage, "local_vel": local_vel,
            "window_times": window_times, "mean_vel": mean_vel,
            "conf_ratio": conf_ratio,
        }

        for w in range(n_windows):
            all_rows.append({
                "batch": batch, "window": w, "time_s": window_times[w],
                "model_mse": model_mse[w], "copy_mse": copy_mse[w],
                "raw_advantage": raw_advantage[w], "local_vel": local_vel[w],
                "mean_vel": mean_vel, "conf_ratio": conf_ratio,
            })

    wdf = pd.DataFrame(all_rows)
    print(f"Total window observations: {len(wdf)}")

    # Velocity terciles
    vel_vals = np.array([per_org_profiles[b]["mean_vel"] for b in batches])
    t1, t2 = np.percentile(vel_vals, [33.3, 66.7])

    def vel_group(v):
        if v < t1: return "Slow"
        elif v <= t2: return "Medium"
        return "Fast"

    wdf["vel_group"] = wdf["mean_vel"].apply(vel_group)

    sorted_batches = sorted(per_org_profiles.keys(),
                            key=lambda b: per_org_profiles[b]["mean_vel"])

    # ================================================================
    # Plot A: Faceted temporal profiles by velocity tercile
    # ================================================================
    fig_a, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    group_colors = {"Slow": "#2ca02c", "Medium": "#1f77b4", "Fast": "#d62728"}

    for ax, group in zip(axes, ["Slow", "Medium", "Fast"]):
        group_batches = [b for b in sorted_batches if vel_group(per_org_profiles[b]["mean_vel"]) == group]

        for batch in group_batches:
            p = per_org_profiles[batch]
            ax.plot(p["window_times"], p["raw_advantage"],
                    color=group_colors[group], alpha=0.25, linewidth=0.8)

        # Mean profile for this group
        profiles = np.array([per_org_profiles[b]["raw_advantage"] for b in group_batches])
        mean_profile = np.nanmean(profiles, axis=0)
        std_profile = np.nanstd(profiles, axis=0)
        times = per_org_profiles[group_batches[0]]["window_times"]

        ax.plot(times, mean_profile, color="black", linewidth=2.5, label="Mean")
        ax.fill_between(times, mean_profile - std_profile, mean_profile + std_profile,
                        color="black", alpha=0.15, label="±1 std")

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_title(f"{group} (n={len(group_batches)})\n"
                     f"v < {t1:.1f}" if group == "Slow" else
                     f"{group} (n={len(group_batches)})\n"
                     f"{t1:.1f} ≤ v ≤ {t2:.1f}" if group == "Medium" else
                     f"{group} (n={len(group_batches)})\n"
                     f"v > {t2:.1f}",
                     fontsize=11)
        ax.legend(fontsize=8)

        # Win rate
        win_rate = (profiles > 0).mean()
        ax.text(0.02, 0.02, f"Model wins: {100*win_rate:.0f}%",
                transform=ax.transAxes, fontsize=9, va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    axes[0].set_ylabel("MSE(copy) − MSE(model)", fontsize=10)

    fig_a.suptitle("Raw model advantage over time, by speed group", fontsize=13, y=1.02)
    fig_a.tight_layout()
    for ext in ["png", "pdf"]:
        fig_a.savefig(OUTPUT_DIR / f"relative_performance_A_temporal.{ext}",
                      dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig_a)
    print(f"Saved: {OUTPUT_DIR / 'relative_performance_A_temporal.png'}")

    # ================================================================
    # Plot B: Raw advantage vs velocity rank
    # ================================================================
    fig_b, axes_b = plt.subplots(1, 2, figsize=(13, 5))

    # Per-organoid mean advantage
    org_adv = wdf.groupby("batch").agg(
        mean_advantage=("raw_advantage", "mean"),
        std_advantage=("raw_advantage", "std"),
        mean_vel=("mean_vel", "first"),
        conf_ratio=("conf_ratio", "first"),
    ).reset_index()
    org_adv["vel_rank"] = org_adv["mean_vel"].rank()
    org_adv["vel_group"] = org_adv["mean_vel"].apply(vel_group)

    # Panel B1: mean advantage vs velocity rank
    ax = axes_b[0]
    for group in ["Slow", "Medium", "Fast"]:
        sub = org_adv[org_adv["vel_group"] == group]
        ax.scatter(sub["vel_rank"], sub["mean_advantage"],
                   c=group_colors[group], s=35, alpha=0.7, edgecolors="black",
                   linewidth=0.3, label=f"{group} (n={len(sub)})", zorder=3)
        ax.errorbar(sub["vel_rank"], sub["mean_advantage"], yerr=sub["std_advantage"],
                    fmt="none", ecolor=group_colors[group], alpha=0.2, elinewidth=0.5)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Velocity rank (1 = slowest)", fontsize=10)
    ax.set_ylabel("Mean MSE(copy) − MSE(model)", fontsize=10)
    ax.set_title("(A) Mean model advantage\nby velocity rank", fontsize=11)
    ax.legend(fontsize=8)

    r, p = spearmanr(org_adv["vel_rank"], org_adv["mean_advantage"])
    ax.text(0.02, 0.98, f"Spearman ρ = {r:.3f}\np = {p:.2e}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    # Panel B2: all windows, raw advantage vs velocity rank
    ax = axes_b[1]
    # Map batch to velocity rank
    rank_map = dict(zip(org_adv["batch"], org_adv["vel_rank"]))
    wdf["vel_rank"] = wdf["batch"].map(rank_map)

    for group in ["Slow", "Medium", "Fast"]:
        sub = wdf[wdf["vel_group"] == group]
        ax.scatter(sub["vel_rank"], sub["raw_advantage"],
                   c=group_colors[group], s=5, alpha=0.15, edgecolors="none",
                   rasterized=True)

    # Binned means (by rank decile)
    rank_bins = np.linspace(0, 108, 11)
    bin_centers, bin_means, bin_stds = [], [], []
    for i in range(len(rank_bins) - 1):
        mask = (wdf["vel_rank"] >= rank_bins[i]) & (wdf["vel_rank"] < rank_bins[i + 1])
        if mask.sum() > 0:
            bin_centers.append((rank_bins[i] + rank_bins[i + 1]) / 2)
            bin_means.append(wdf.loc[mask, "raw_advantage"].mean())
            bin_stds.append(wdf.loc[mask, "raw_advantage"].std())

    ax.errorbar(bin_centers, bin_means, yerr=bin_stds, color="black",
                linewidth=2, capsize=3, marker="o", markersize=5, zorder=5,
                label="Decile mean ± std")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Velocity rank (1 = slowest)", fontsize=10)
    ax.set_ylabel("MSE(copy) − MSE(model)", fontsize=10)
    ax.set_title("(B) All windows: advantage\nvs velocity rank", fontsize=11)
    ax.legend(fontsize=8)

    fig_b.tight_layout()
    for ext in ["png", "pdf"]:
        fig_b.savefig(OUTPUT_DIR / f"relative_performance_B_rank.{ext}",
                      dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig_b)
    print(f"Saved: {OUTPUT_DIR / 'relative_performance_B_rank.png'}")

    # ================================================================
    # Plot C: Heatmap — organoids × window position (raw advantage)
    # ================================================================
    fig_c, ax = plt.subplots(figsize=(10, 8))

    n_org = len(sorted_batches)
    n_win = per_org_profiles[sorted_batches[0]]["raw_advantage"].shape[0]
    heat = np.full((n_org, n_win), np.nan)
    for i, batch in enumerate(sorted_batches):
        heat[i, :] = per_org_profiles[batch]["raw_advantage"]

    # Symmetric colormap centered at 0
    vmax = np.nanpercentile(np.abs(heat), 95)
    im = ax.imshow(heat, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    cbar = fig_c.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("MSE(copy) − MSE(model)\n(blue = model wins)", fontsize=10)

    ax.set_xlabel("Window position (time)", fontsize=11)
    ax.set_ylabel("Organoid (sorted by mean velocity →)", fontsize=11)
    ax.set_title("Model advantage over copy-last: organoid × time", fontsize=12)

    times = per_org_profiles[sorted_batches[0]]["window_times"]
    tick_positions = list(range(0, n_win, 3))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{times[t]:.0f}s" for t in tick_positions], fontsize=8)

    # Y ticks
    y_ticks = list(range(0, n_org, n_org // 8))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(
        [f"{sorted_batches[t].replace('batch-','')}\n"
         f"(v={per_org_profiles[sorted_batches[t]]['mean_vel']:.1f})"
         for t in y_ticks], fontsize=7)

    # Add tercile boundary lines
    slow_count = sum(1 for b in sorted_batches if per_org_profiles[b]["mean_vel"] < t1)
    med_count = sum(1 for b in sorted_batches if per_org_profiles[b]["mean_vel"] <= t2)
    ax.axhline(slow_count - 0.5, color="white", linewidth=1.5, linestyle="--")
    ax.axhline(med_count - 0.5, color="white", linewidth=1.5, linestyle="--")
    ax.text(n_win + 0.3, slow_count / 2, "Slow", fontsize=9, va="center", fontweight="bold")
    ax.text(n_win + 0.3, (slow_count + med_count) / 2, "Med", fontsize=9, va="center", fontweight="bold")
    ax.text(n_win + 0.3, (med_count + n_org) / 2, "Fast", fontsize=9, va="center", fontweight="bold")

    fig_c.tight_layout()
    for ext in ["png", "pdf"]:
        fig_c.savefig(OUTPUT_DIR / f"relative_performance_C_heatmap.{ext}",
                      dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig_c)
    print(f"Saved: {OUTPUT_DIR / 'relative_performance_C_heatmap.png'}")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("Summary: Model vs Copy-Last (raw advantage)")
    print("=" * 60)

    for group in ["Slow", "Medium", "Fast"]:
        sub = wdf[wdf["vel_group"] == group]
        n_org_g = sub["batch"].nunique()
        print(f"\n  {group} (n={n_org_g} organoids, {len(sub)} windows):")
        print(f"    Mean advantage: {sub['raw_advantage'].mean():.6f}")
        print(f"    Model wins: {(sub['raw_advantage'] > 0).sum()}/{len(sub)} "
              f"({100*(sub['raw_advantage'] > 0).mean():.1f}%)")

    r, p = spearmanr(org_adv["vel_rank"], org_adv["mean_advantage"])
    print(f"\n  Spearman(mean_advantage, velocity_rank): rho={r:.4f}, p={p:.2e}")

    # Temporal stability
    r_t, p_t = spearmanr(wdf["window"], wdf["raw_advantage"])
    print(f"  Spearman(advantage, window_position): rho={r_t:.4f}, p={p_t:.4e}")

    # F-stat
    org_means = wdf.groupby("batch")["raw_advantage"].mean()
    between_var = org_means.var()
    within_vars = wdf.groupby("batch")["raw_advantage"].var()
    mean_within = within_vars.mean()
    f_stat = between_var / (mean_within + 1e-12)
    print(f"\n  Variance decomposition (raw advantage):")
    print(f"    Between-organoid: {between_var:.2e}")
    print(f"    Mean within-organoid: {mean_within:.2e}")
    print(f"    F-statistic: {f_stat:.1f}")

    # Organoids where model is consistently worse (mean advantage < 0)
    losers = org_adv[org_adv["mean_advantage"] < 0].sort_values("mean_advantage")
    print(f"\n  Organoids where copy-last beats model (n={len(losers)}):")
    for _, row in losers.head(10).iterrows():
        print(f"    {row['batch']}: advantage={row['mean_advantage']:.6f}, "
              f"vel={row['mean_vel']:.2f}, CR={row['conf_ratio']:.3f}")

    wdf.to_csv(OUTPUT_DIR / "relative_performance_per_window.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'relative_performance_per_window.csv'}")


if __name__ == "__main__":
    main()
