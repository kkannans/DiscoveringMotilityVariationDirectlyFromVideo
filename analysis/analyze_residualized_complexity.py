"""
Behavioral complexity: residualize MSE_model/MSE_copy against velocity.

For each organoid, compute how far its prediction ratio falls above/below
the population trend at that speed. Positive residual = harder to predict
than expected = more complex dynamics.

Outputs -> paths.SUPPLEMENTARY_DIR
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths

OUTPUT_DIR = paths.SUPPLEMENTARY_DIR
WINDOW_CSV = OUTPUT_DIR / "relative_performance_per_window.csv"
MOTILITY_CSV = paths.CLASSICAL_DIR / "motility_descriptors.csv"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    wdf = pd.read_csv(WINDOW_CSV)
    mot = pd.read_csv(MOTILITY_CSV)

    # Per-organoid mean ratio
    org = wdf.groupby("batch").agg(
        model_mse=("model_mse", "mean"),
        copy_mse=("copy_mse", "mean"),
        mean_vel=("mean_vel", "first"),
    ).reset_index()
    org["ratio"] = org["model_mse"] / (org["copy_mse"] + 1e-12)

    # Fit trend: ratio vs velocity rank (quadratic)
    org["vel_rank"] = rankdata(org["mean_vel"])
    coeffs = np.polyfit(org["vel_rank"], org["ratio"], 2)
    org["predicted_ratio"] = np.polyval(coeffs, org["vel_rank"])
    org["residual"] = org["ratio"] - org["predicted_ratio"]

    # Merge classical descriptors
    org = org.merge(mot, left_on="batch", right_on="batch_name", how="left")

    # ================================================================
    # Figure: 3 panels
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Ratio vs velocity rank with fitted trend
    ax = axes[0]
    ax.scatter(org["vel_rank"], org["ratio"], s=25, alpha=0.6, c="#1f77b4",
               edgecolors="black", linewidth=0.3, zorder=3)
    rank_smooth = np.linspace(1, len(org), 200)
    ax.plot(rank_smooth, np.polyval(coeffs, rank_smooth), "r-", linewidth=2,
            label="Quadratic fit")
    ax.axhline(1, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Velocity rank (1 = slowest)", fontsize=10)
    ax.set_ylabel("MSE(model) / MSE(copy)", fontsize=10)
    ax.set_title("(A) Prediction ratio vs velocity\nwith population trend", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.2, max(org["ratio"].max() * 1.1, 2.5))

    # Label extremes
    for _, row in org.nlargest(3, "residual").iterrows():
        ax.annotate(row["batch"].replace("batch-", ""),
                     xy=(row["vel_rank"], row["ratio"]),
                     xytext=(5, 8), textcoords="offset points", fontsize=7,
                     arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

    # Panel B: Residual scatter colored by std_velocity
    ax = axes[1]
    valid = ~np.isnan(org["std_velocity"])
    sc = ax.scatter(org.loc[valid, "vel_rank"], org.loc[valid, "residual"],
                    c=org.loc[valid, "std_velocity"], cmap="magma", s=30, alpha=0.7,
                    edgecolors="black", linewidth=0.3, zorder=3)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("Velocity std (px/frame)", fontsize=9)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Velocity rank (1 = slowest)", fontsize=10)
    ax.set_ylabel("Residual\n(+ = harder than expected)", fontsize=10)
    ax.set_title("(B) Behavioral complexity\nresidual", fontsize=11)

    r_std, p_std = spearmanr(org.loc[valid, "std_velocity"], org.loc[valid, "residual"])
    ax.text(0.02, 0.98, f"rho(residual, vel_std) = {r_std:.3f}\np = {p_std:.4f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    # Panel C: Bar chart of correlations with all descriptors
    ax = axes[2]
    descriptors = [
        ("Velocity std", "std_velocity"),
        ("MSD slope", "msd_slope"),
        ("Confinement ratio", "confinement_ratio"),
        ("Tortuosity", "tortuosity"),
        ("Turning angle", "mean_turning_angle"),
        ("Dir. autocorr", "direction_autocorr_lag1"),
    ]

    names, rhos, pvals = [], [], []
    for label, col in descriptors:
        v = ~np.isnan(org[col])
        r, p = spearmanr(org.loc[v, "residual"], org.loc[v, col])
        names.append(label)
        rhos.append(r)
        pvals.append(p)

    y_pos = np.arange(len(names))
    colors = ["#d62728" if p < 0.05 else "#aaaaaa" for p in pvals]
    ax.barh(y_pos, rhos, color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel(r"Spearman $\rho$ with complexity residual", fontsize=10)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_title("(C) What explains behavioral\ncomplexity?", fontsize=11)

    for i, (r, p) in enumerate(zip(rhos, pvals)):
        sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.text(r + 0.02 * np.sign(r), i, f"{r:.3f}{sig}", va="center", fontsize=8)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#d62728", label="p < 0.05"),
                        Patch(facecolor="#aaaaaa", label="n.s.")],
              fontsize=8, loc="lower right")

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(OUTPUT_DIR / f"behavioral_complexity_residual.{ext}",
                    dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'behavioral_complexity_residual.png'}")

    # Print summary
    print(f"\n{'='*60}")
    print("Behavioral complexity: velocity-residualized prediction ratio")
    print(f"{'='*60}")
    print(f"\nFit: ratio = {coeffs[0]:.6f}*rank^2 + {coeffs[1]:.4f}*rank + {coeffs[2]:.4f}")
    print(f"Residual std: {org['residual'].std():.4f}")

    print(f"\nDescriptor correlations with complexity residual:")
    for name, r, p in zip(names, rhos, pvals):
        sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:20s}: rho = {r:+.4f}, p = {p:.4f} {sig}")

    print(f"\nTop 5 most complex (hardest to predict for their speed):")
    for _, row in org.nlargest(5, "residual").iterrows():
        print(f"  {row['batch']}: residual={row['residual']:+.4f}, vel={row['mean_vel']:.2f}, "
              f"vel_std={row['std_velocity']:.2f}, CR={row['confinement_ratio']:.3f}")

    print(f"\nTop 5 simplest (easiest to predict for their speed):")
    for _, row in org.nsmallest(5, "residual").iterrows():
        print(f"  {row['batch']}: residual={row['residual']:+.4f}, vel={row['mean_vel']:.2f}, "
              f"vel_std={row['std_velocity']:.2f}, CR={row['confinement_ratio']:.3f}")


if __name__ == "__main__":
    main()
