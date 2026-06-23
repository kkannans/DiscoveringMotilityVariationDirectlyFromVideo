"""
compare_architectures.py — SimVP-TAU+R vs PredRNN+R per-organoid ranking agreement.

Robustness check (paper Discussion): if the prediction-surprise ranking is a property of the
organoids and not of a particular network, two different architectures should rank organoids
similarly. The paper reports Spearman rho = 0.938.

IMPORTANT (data hygiene): the paper's 0.938 is computed on RAW per-organoid MSE. The *defined*
prediction surprise S_o (Eq 4) is the delta motion loss; on that metric the agreement is ~0.906.
This script reports BOTH so the distinction is explicit.

Per-organoid metrics:
  - raw_mse   = mean over windows & horizons of model raw MSE      (npz: raw_mse[:, :, 0])
  - delta     = mean over windows & horizons of model delta loss   (npz: delta_residuals[:, :, 0])  == S_o (Eq 4)

Inputs (under paths.INTERMEDIATES_DIR):
  SimVP:   per_organoid_residuals/<batch>.npz       (raw_mse, delta_residuals)
  PredRNN: predrnn_per_organoid_residuals/<batch>.npz   (preferred, if regenerated via
           extract_prediction_surprise.py --method PredRNN), else
           predrnn_per_organoid.csv  (columns: batch_name, predrnn_mse[, seed])  — raw MSE only

Outputs -> paths.FIGURES_DIR / architecture_comparison /
    architecture_comparison_stats.json
    architecture_comparison.csv
    architecture_comparison.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau, pearsonr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import paths


def per_organoid_from_residuals(residuals_dir):
    """Return {batch: (raw_mse, delta)} from a per_organoid_residuals/*.npz directory."""
    out = {}
    for f in sorted(Path(residuals_dir).glob("*.npz")):
        d = np.load(f)
        out[f.stem] = (float(d["raw_mse"][:, :, 0].mean()),
                       float(d["delta_residuals"][:, :, 0].mean()))
    return out


def load_simvp():
    rdir = paths.INTERMEDIATES_DIR / "per_organoid_residuals"
    if not rdir.exists():
        sys.exit(f"ERROR: {rdir} not found (run extract_prediction_surprise.py).")
    d = per_organoid_from_residuals(rdir)
    raw = {b: v[0] for b, v in d.items()}
    delta = {b: v[1] for b, v in d.items()}
    return raw, delta


def load_predrnn():
    """Prefer regenerated PredRNN residual npz; fall back to the cached predrnn_per_organoid.csv."""
    rdir = paths.INTERMEDIATES_DIR / "predrnn_per_organoid_residuals"
    if rdir.exists() and any(rdir.glob("*.npz")):
        d = per_organoid_from_residuals(rdir)
        return {b: v[0] for b, v in d.items()}, {b: v[1] for b, v in d.items()}
    csv = paths.INTERMEDIATES_DIR / "predrnn_per_organoid.csv"
    if not csv.exists():
        sys.exit(f"ERROR: neither {rdir} nor {csv} found. Provide PredRNN per-organoid data.")
    df = pd.read_csv(csv)
    raw = dict(zip(df["batch_name"], df["predrnn_mse"]))
    print(f"  (PredRNN delta-surprise unavailable from CSV; raw MSE only from {csv.name})")
    return raw, None


def _corrs(a, b):
    rho, p = spearmanr(a, b)
    tau, _ = kendalltau(a, b)
    r, _ = pearsonr(a, b)
    return {"spearman": round(float(rho), 4), "kendall": round(float(tau), 4),
            "pearson": round(float(r), 4), "p_spearman": float(p), "n": int(len(a))}


def main():
    out_dir = paths.FIGURES_DIR / "architecture_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    sv_raw, sv_delta = load_simvp()
    pr_raw, pr_delta = load_predrnn()

    batches = sorted(set(sv_raw) & set(pr_raw))
    df = pd.DataFrame({
        "batch_name": batches,
        "simvp_raw_mse": [sv_raw[b] for b in batches],
        "predrnn_raw_mse": [pr_raw[b] for b in batches],
        "simvp_delta": [sv_delta[b] for b in batches],
        "predrnn_delta": [pr_delta[b] if pr_delta else np.nan for b in batches],
    })
    df.to_csv(out_dir / "architecture_comparison.csv", index=False)

    stats = {"n_organoids": len(batches)}
    stats["raw_mse"] = _corrs(df["simvp_raw_mse"], df["predrnn_raw_mse"])
    print(f"SimVP vs PredRNN — RAW MSE   : spearman={stats['raw_mse']['spearman']}  "
          f"kendall={stats['raw_mse']['kendall']}  (paper rho=0.938)")
    if pr_delta is not None:
        stats["delta_surprise"] = _corrs(df["simvp_delta"], df["predrnn_delta"])
        print(f"SimVP vs PredRNN — delta S_o : spearman={stats['delta_surprise']['spearman']}  "
              f"kendall={stats['delta_surprise']['kendall']}")
    else:
        # still report SimVP delta vs PredRNN raw for context
        stats["simvp_delta_vs_predrnn_raw"] = _corrs(df["simvp_delta"], df["predrnn_raw_mse"])
        print(f"SimVP delta S_o vs PredRNN raw: spearman="
              f"{stats['simvp_delta_vs_predrnn_raw']['spearman']}")

    with open(out_dir / "architecture_comparison_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Rank scatter (raw MSE)
    sr = df["simvp_raw_mse"].rank()
    pr = df["predrnn_raw_mse"].rank()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(sr, pr, s=18, alpha=0.6, edgecolors="gray", linewidths=0.3)
    ax.plot([1, len(df)], [1, len(df)], "k--", lw=0.8, alpha=0.4)
    ax.set_xlabel("SimVP-TAU+R rank (raw MSE)")
    ax.set_ylabel("PredRNN+R rank (raw MSE)")
    ax.text(0.05, 0.95, f"Spearman rho = {stats['raw_mse']['spearman']:.3f}",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))
    fig.tight_layout()
    fig.savefig(out_dir / "architecture_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / 'architecture_comparison.png'}")


if __name__ == "__main__":
    main()
