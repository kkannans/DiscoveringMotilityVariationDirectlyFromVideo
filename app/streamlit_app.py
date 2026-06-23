"""
streamlit_app.py — interactive explorer for SimVP-TAU+R prediction surprise.

Pick a held-out organoid and see, from PRECOMPUTED cached artifacts (no GPU at serve time):
  • raw video frames (input context + ground truth)
  • the spatial prediction-error map (where the model is surprised)
  • the per-window motion loss (Fig 2D): SimVP-TAU+R vs the copy-last & optical-flow baselines
  • where the organoid sits on the surprise-vs-mean-speed ranking (Fig 3C), with it highlighted

Run (after `python download_data.py` or `bash stage_cache.sh`):
    streamlit run app/streamlit_app.py
Headless self-check (no streamlit needed):
    python app/streamlit_app.py --selftest

Data is read via paths.py (./outputs, ./dataset). Method column order in the residual npz is
[model, copy, mean_frame, rolling_avg, optical_flow].
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import paths

METHODS = ["model", "copy", "mean_frame", "rolling_avg", "optical_flow"]
SUMMARY = paths.INTERMEDIATES_DIR / "per_organoid_summary.csv"
MOTILITY = paths.CLASSICAL_DIR / "motility_descriptors.csv"
RESIDUALS = paths.INTERMEDIATES_DIR / "per_organoid_residuals"
SPATIAL = paths.INTERMEDIATES_DIR / "spatial_residual_maps"


# ── data layer (pure; importable/testable without streamlit) ──────────────────

def load_table():
    """Per-organoid summary merged with classical mean speed + ranks."""
    s = pd.read_csv(SUMMARY)
    m = pd.read_csv(MOTILITY)[["batch_name", "mean_velocity"]]
    df = s.merge(m, on="batch_name", how="left")
    df["surprise_rank"] = df["mean_motion_loss"].rank().astype(int)
    df["speed_rank"] = df["mean_velocity"].rank().astype(int)
    return df.sort_values("mean_motion_loss").reset_index(drop=True)


def per_window_losses(batch_name):
    """(19,5) per-window motion loss L_motion per method (mean over horizons of delta residuals)."""
    d = np.load(RESIDUALS / f"{batch_name}.npz")
    return d["delta_residuals"][:, :, :].mean(axis=1)  # (n_windows, n_methods)


def spatial_error_map(batch_name):
    """(128,128) mean spatial prediction-error map over the prediction horizon."""
    return np.load(SPATIAL / f"{batch_name}.npy").mean(axis=0)


def load_frames(batch_name, idxs=(0, 5, 6, 11)):
    """Return {idx: HxWx3 RGB float[0,1]} for the requested frame indices."""
    import cv2
    cap = cv2.VideoCapture(str(paths.DATA_ROOT / f"{batch_name}.mp4"))
    out = {}
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if ok:
            out[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    cap.release()
    return out


# ── UI ────────────────────────────────────────────────────────────────────────

def main():
    import streamlit as st
    import matplotlib.pyplot as plt

    st.set_page_config(page_title="Organoid prediction surprise", layout="wide")
    st.title("Discovering organoid motility variation — prediction surprise explorer")

    if not SUMMARY.exists():
        st.error(f"Artifacts not found at {SUMMARY}.\n\n"
                 "Run `python download_data.py` (or `bash stage_cache.sh`) first.")
        st.stop()

    df = load_table()
    n = len(df)

    # sidebar: choose organoid (sorted slowest→fastest surprise)
    st.sidebar.header("Organoid")
    labels = [f"{r.batch_name}  (surprise rank {r.surprise_rank}/{n})" for r in df.itertuples()]
    sel = st.sidebar.selectbox("Select (sorted by surprise)", range(n), format_func=lambda i: labels[i])
    row = df.iloc[sel]
    batch = row["batch_name"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prediction surprise Sₒ", f"{row['mean_motion_loss']:.2e}")
    c2.metric("Surprise rank", f"{int(row['surprise_rank'])}/{n}")
    c3.metric("Mean speed rank", f"{int(row['speed_rank'])}/{n}")
    c4.metric("Organoid size (px)", f"{row['area_px']:.0f}")

    left, right = st.columns(2)

    # raw frames
    with left:
        st.subheader("Raw video (context → ground truth)")
        frames = load_frames(batch)
        if frames:
            fcols = st.columns(len(frames))
            for col, (i, img) in zip(fcols, sorted(frames.items())):
                col.image(img, caption=f"frame {i}", use_container_width=True)

    # spatial error map
    with right:
        st.subheader("Spatial prediction error (where the model is surprised)")
        sm = spatial_error_map(batch)
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(sm, cmap="magma"); ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
        st.pyplot(fig); plt.close(fig)

    # per-window motion loss (Fig 2D): model vs baselines
    st.subheader("Per-window motion loss — SimVP-TAU+R vs baselines (Fig 2D)")
    wl = per_window_losses(batch)  # (19,5)
    widx = np.arange(1, wl.shape[0] + 1)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(widx, wl[:, 0], color="#e53935", alpha=0.85, label="SimVP-TAU+R (model)")
    ax.plot(widx, wl[:, METHODS.index("copy")], "o-", color="#b0b0b0", ms=3, label="copy-last")
    ax.plot(widx, wl[:, METHODS.index("optical_flow")], "s-", color="#606060", ms=3, label="optical flow")
    ax.axhline(wl[:, 0].mean(), ls="--", color="#e53935", lw=1, label="Sₒ (mean)")
    ax.set_xlabel("window"); ax.set_ylabel(r"$\mathcal{L}_{motion}$"); ax.legend(fontsize=8)
    st.pyplot(fig); plt.close(fig)

    # ranking scatter (Fig 3C) with selected highlighted
    st.subheader("Ranking agreement: mean speed vs prediction surprise (Fig 3C)")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df["speed_rank"], df["surprise_rank"], s=18, c="#1f77b4", alpha=0.5, edgecolors="gray", linewidths=0.3)
    ax.scatter([row["speed_rank"]], [row["surprise_rank"]], s=120, c="#e53935", edgecolors="k", zorder=5, label=batch)
    ax.plot([0, n], [0, n], "k--", lw=0.8, alpha=0.3)
    ax.set_xlabel("Mean speed rank"); ax.set_ylabel("Prediction surprise rank"); ax.legend(fontsize=9)
    st.pyplot(fig); plt.close(fig)


def selftest():
    df = load_table()
    assert len(df) == 108, f"expected 108 organoids, got {len(df)}"
    b = df.iloc[len(df) // 2]["batch_name"]
    wl = per_window_losses(b); assert wl.shape[1] == 5, wl.shape
    sm = spatial_error_map(b); assert sm.shape == (128, 128), sm.shape
    fr = load_frames(b); assert len(fr) >= 1, "no frames read"
    print(f"selftest OK: {len(df)} organoids; sample {b}: windows={wl.shape}, "
          f"spatial={sm.shape}, frames={sorted(fr)}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
