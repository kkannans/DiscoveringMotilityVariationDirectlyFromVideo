"""
rank_by_prediction_surprise.py — rank the 108 organoids by prediction surprise.

Per-organoid prediction surprise Sₒ = mean ||Δ_GT - Δ_pred||² (delta motion loss) over all
sliding windows of that organoid's held-out test video.

Usage:
    python rank_by_prediction_surprise.py

Inputs:
    paths.INTERMEDIATES_DIR / per_organoid_residuals / *.npz
    paths.INTERMEDIATES_DIR / spatial_residual_maps / *.npy

Outputs -> paths.INTERMEDIATES_DIR
    per_organoid_summary.csv   (mean_motion_loss = Sₒ, ranked; + area, masked loss)
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths

N = 6
R1_DIR = paths.INTERMEDIATES_DIR
OUTPUT_DIR = paths.INTERMEDIATES_DIR
VIDEO_DIR = paths.DATA_ROOT
CENTROID_PATH = paths.CLASSICAL_DIR / "centroid_trajectories.npz"


def load_first_last_composite(batch_name, num_frames=120, size=128):
    """Load first and last GT frames, return composite (H, W, 3) float [0,1].

    First frame tinted blue, last frame tinted red, blended at 50/50.
    Shows organoid displacement and morphology change.
    """
    cap = cv2.VideoCapture(str(VIDEO_DIR / f"{batch_name}.mp4"))

    # Read first frame
    ret, frame0 = cap.read()
    if not ret:
        cap.release()
        return np.zeros((size, size, 3), dtype=np.float32)
    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    frame0 = cv2.resize(frame0, (size, size), interpolation=cv2.INTER_AREA)
    frame0 = frame0.astype(np.float32) / 255.0

    # Skip to last frame
    for _ in range(num_frames - 2):
        ret, _ = cap.read()
        if not ret:
            break
    ret, frame_last = cap.read()
    cap.release()
    if not ret:
        frame_last = frame0.copy()
    else:
        frame_last = cv2.cvtColor(frame_last, cv2.COLOR_BGR2RGB)
        frame_last = cv2.resize(frame_last, (size, size), interpolation=cv2.INTER_AREA)
        frame_last = frame_last.astype(np.float32) / 255.0

    # Tint first=blue, last=red, blend
    gray0 = np.mean(frame0, axis=2)
    gray_last = np.mean(frame_last, axis=2)

    composite = np.zeros((size, size, 3), dtype=np.float32)
    composite[:, :, 2] = gray0              # blue channel = first frame
    composite[:, :, 0] = gray_last          # red channel = last frame
    composite[:, :, 1] = (gray0 + gray_last) * 0.25  # faint green for overlap

    return np.clip(composite, 0, 1)


def load_centroids(batch_name):
    """Load (T, 2) centroid trajectory. Returns None if not available."""
    data = np.load(CENTROID_PATH, allow_pickle=True)
    if batch_name in data:
        c = data[batch_name]  # (T, 2) — (x, y)
        if np.isfinite(c).all():
            return c
    return None


def compute_morans_i(image):
    """Compute Moran's I for a 2D image using queen contiguity."""
    H, W = image.shape
    x_bar = image.mean()
    x_dev = image - x_bar
    numerator = 0.0
    w_sum = 0.0
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(image, -dy, axis=0), -dx, axis=1)
            mask = np.ones((H, W), dtype=bool)
            if dy == -1: mask[-1, :] = False
            elif dy == 1: mask[0, :] = False
            if dx == -1: mask[:, -1] = False
            elif dx == 1: mask[:, 0] = False
            numerator += ((image - x_bar) * mask * (shifted - x_bar) * mask).sum()
            w_sum += mask.sum()
    denominator = (x_dev ** 2).sum()
    if denominator == 0 or w_sum == 0:
        return 0.0
    return float((H * W / w_sum) * (numerator / denominator))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spatial_out = OUTPUT_DIR / "spatial_residual_maps"
    spatial_out.mkdir(parents=True, exist_ok=True)

    residuals_dir = R1_DIR / "per_organoid_residuals"
    spatial_dir = R1_DIR / "spatial_residual_maps"

    batch_names = sorted([p.stem for p in residuals_dir.glob("*.npz")])
    if not batch_names:
        print("ERROR: No residual files found. Run extract_prediction_surprise.py first.")
        sys.exit(1)
    print(f"Loaded {len(batch_names)} organoids")

    # ── Per-organoid motion loss (delta MSE, model = column 0) ──
    rows = []
    per_organoid_means = []
    per_organoid_window_loss = {}

    for batch_name in batch_names:
        npz = np.load(residuals_dir / f"{batch_name}.npz")
        key = "delta_residuals" if "delta_residuals" in npz else "scalar_residuals"
        dr = npz[key]  # (n_windows, N, 5)
        model_per_window = dr[:, :, 0].mean(axis=1)  # (n_windows,)
        mean_loss = float(model_per_window.mean())
        std_loss = float(model_per_window.std())
        per_organoid_means.append(mean_loss)
        per_organoid_window_loss[batch_name] = model_per_window

    # ── Load organoid areas from auto-tune segmentation (first frame) ──
    print("  Loading organoid areas from auto-tune segmentation...")
    from scipy.stats import spearmanr

    AUTOTUNE_DIR = paths.SEGMENTATION_DIR

    area_dict = {}
    for batch_name in batch_names:
        csv_path = AUTOTUNE_DIR / batch_name / "centroids.csv"
        if csv_path.exists():
            frame_areas = []
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    a = float(row["area"])
                    if a > 0:
                        frame_areas.append(a)
            area_dict[batch_name] = float(np.median(frame_areas)) if frame_areas else 0.0
        else:
            area_dict[batch_name] = 0.0

    # Masked motion loss: use temporal-variance mask for per-pixel loss
    # Inline utilities for per-pixel temporal variance masking
    def load_video_frames_np(batch_name, num_frames=120, target_size=128):
        cap = cv2.VideoCapture(str(VIDEO_DIR / f"{batch_name}.mp4"))
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        cap.release()
        frames_uint8 = np.stack(frames)
        frames_float = frames_uint8.astype(np.float32) / 255.0
        return frames_uint8, frames_float

    def compute_organoid_roi(frames_float):
        from scipy import ndimage
        gray = np.mean(frames_float, axis=3)
        temp_std = np.std(gray, axis=0)
        std_normalized = np.clip(temp_std / (temp_std.max() + 1e-8), 0, 1)
        std_uint8 = (std_normalized * 255).astype(np.uint8)
        _, mask = cv2.threshold(std_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = (mask > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        labeled, n_components = ndimage.label(mask)
        if n_components > 1:
            comp_sizes = ndimage.sum(mask, labeled, range(1, n_components + 1))
            largest_id = np.argmax(comp_sizes) + 1
            mask = (labeled == largest_id).astype(np.uint8)
        return mask

    masked_loss_dict = {}
    for batch_name in batch_names:
        sp = spatial_dir / f"{batch_name}.npy"
        if sp.exists():
            _, frames_float = load_video_frames_np(batch_name, num_frames=120, target_size=128)
            mask = compute_organoid_roi(frames_float)
            mask_bool = mask.astype(bool)
            if mask_bool.sum() > 0:
                spatial_map = np.load(sp).mean(axis=0)
                masked_loss_dict[batch_name] = float(spatial_map[mask_bool].mean())
            else:
                masked_loss_dict[batch_name] = per_organoid_means[batch_names.index(batch_name)]
        else:
            masked_loss_dict[batch_name] = per_organoid_means[batch_names.index(batch_name)]

    # Sort by motion loss → assign organoid indices
    sort_order = np.argsort(per_organoid_means)
    sorted_batches = [batch_names[i] for i in sort_order]
    sorted_means = [per_organoid_means[i] for i in sort_order]

    # Robustness check: area vs motion loss, raw vs normalized ranking
    areas_arr = np.array([area_dict[b] for b in sorted_batches])
    raw_losses = np.array(sorted_means)
    masked_losses = np.array([masked_loss_dict[b] for b in sorted_batches])

    rho_area_loss, p_area_loss = spearmanr(areas_arr, raw_losses)
    raw_ranks = np.argsort(np.argsort(raw_losses))
    norm_ranks = np.argsort(np.argsort(masked_losses))
    rho_raw_norm, p_raw_norm = spearmanr(raw_ranks, norm_ranks)

    robustness = {
        "area_vs_raw_loss": {"spearman_rho": float(rho_area_loss), "p_value": float(p_area_loss)},
        "raw_vs_normalized_ranking": {"spearman_rho": float(rho_raw_norm), "p_value": float(p_raw_norm)},
        "median_area_px": int(np.median(areas_arr)),
        "area_range": [int(areas_arr.min()), int(areas_arr.max())],
    }
    with open(OUTPUT_DIR / "area_robustness_check.json", "w") as f:
        json.dump(robustness, f, indent=2)
    print(f"  Area vs raw loss: ρ = {rho_area_loss:.3f}")
    print(f"  Raw vs normalized ranking: ρ = {rho_raw_norm:.3f}")
    print(f"  Saved: area_robustness_check.json")

    # Build CSV with organoid index + area + masked loss
    for idx, batch_name in enumerate(sorted_batches):
        rows.append({
            "batch_name": batch_name,
            "organoid_index": idx,
            "mean_motion_loss": sorted_means[idx],
            "std_motion_loss": float(per_organoid_window_loss[batch_name].std()),
            "n_windows": len(per_organoid_window_loss[batch_name]),
            "area_px": area_dict[batch_name],
            "masked_motion_loss": masked_loss_dict[batch_name],
        })

    with open(OUTPUT_DIR / "per_organoid_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: per_organoid_summary.csv")

    # ── Variance decomposition ──
    means_arr = np.array(sorted_means)
    between_var = float(np.var(means_arr, ddof=1))
    within_vars = [float(np.var(per_organoid_window_loss[b], ddof=1))
                   for b in sorted_batches if len(per_organoid_window_loss[b]) > 1]
    mean_within_var = float(np.mean(within_vars))
    f_stat = between_var / mean_within_var if mean_within_var > 0 else float("inf")

    with open(OUTPUT_DIR / "variance_decomposition.json", "w") as f:
        json.dump({
            "between_organoid_variance": between_var,
            "mean_within_organoid_variance": mean_within_var,
            "F_statistic": f_stat,
            "n_organoids": len(sorted_batches),
            "metric": "motion_loss",
        }, f, indent=2)
    print(f"  F-statistic: {f_stat:.2f}")

    # ── Moran's I on spatial maps ──
    print("  Computing Moran's I...")
    morans_rows = []
    morans_dict = {}
    for batch_name in sorted_batches:
        sp = spatial_dir / f"{batch_name}.npy"
        if sp.exists():
            spatial_map = np.load(sp)  # (N, 128, 128)
            mean_map = spatial_map.mean(axis=0)
            mi = compute_morans_i(mean_map)
            morans_rows.append({"batch_name": batch_name, "morans_i": mi})
            morans_dict[batch_name] = mi
            # Copy spatial map to output
            np.save(spatial_out / f"{batch_name}.npy", spatial_map)

    with open(OUTPUT_DIR / "morans_i.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["batch_name", "morans_i"])
        writer.writeheader()
        writer.writerows(morans_rows)
    print(f"  Saved: morans_i.csv ({len(morans_rows)} rows)")

    # ── Compute m_copy (GT motion energy) per organoid ──
    mcopy_dict = {}
    for batch_name in sorted_batches:
        npz = np.load(residuals_dir / f"{batch_name}.npz")
        key = "delta_residuals" if "delta_residuals" in npz else "scalar_residuals"
        dr = npz[key]  # (n_windows, N, 5)
        mcopy_dict[batch_name] = float(dr[:, :, 1].mean())  # copy column = GT motion energy

    mcopy_median = np.median(list(mcopy_dict.values()))

    # ── Select organoids for panels B and C ──
    n = len(sorted_batches)

    # Panel B: 5 organoids spanning the range
    sel_indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
    sel_batches = [sorted_batches[i] for i in sel_indices]
    sel_labels = [f"Org {i}" for i in sel_indices]

    # Panel C: Size-controlled selection
    # Filter to organoids within median area ± 1.5 MAD
    median_area = np.median(areas_arr)
    mad_area = np.median(np.abs(areas_arr - median_area))
    size_lo = median_area - 0.75 * mad_area
    size_hi = median_area + 0.75 * mad_area

    size_band = [(i, b) for i, b in enumerate(sorted_batches)
                 if size_lo <= area_dict[b] <= size_hi]
    print(f"  Size band: {size_lo:.0f}–{size_hi:.0f} px², "
          f"{len(size_band)}/{n} organoids qualify")

    if len(size_band) < 3:
        # Fallback: widen to ± 1.5 MAD
        size_lo = median_area - 1.5 * mad_area
        size_hi = median_area + 1.5 * mad_area
        size_band = [(i, b) for i, b in enumerate(sorted_batches)
                     if size_lo <= area_dict[b] <= size_hi]

    # High: among top 10% loss in size band, pick closest area to median
    size_band_sorted = sorted(size_band, key=lambda x: sorted_means[x[0]], reverse=True)
    top_n = max(3, len(size_band_sorted) // 10)
    top_candidates = size_band_sorted[:top_n]
    high_idx, high_batch = min(top_candidates, key=lambda x: abs(area_dict[x[1]] - median_area))

    # Low: lowest motion loss within size band, with above-median GT motion
    active_size = [(i, b) for i, b in size_band if mcopy_dict[b] > mcopy_median]
    if not active_size:
        active_size = size_band
    low_idx, low_batch = min(active_size, key=lambda x: sorted_means[x[0]])

    # Medium: closest to median loss within size band, excluding low and high
    median_loss = np.median(sorted_means)
    med_candidates = [(i, b) for i, b in size_band if b != low_batch and b != high_batch]
    if not med_candidates:
        med_candidates = size_band
    med_idx, med_batch = min(med_candidates, key=lambda x: abs(sorted_means[x[0]] - median_loss))

    c_batches = [low_batch, med_batch, high_batch]
    c_indices = [low_idx, med_idx, high_idx]
    c_labels = [f"Low (Org {low_idx})", f"Medium (Org {med_idx})", f"High (Org {high_idx})"]

    # Save selection rationale
    with open(OUTPUT_DIR / "selected_spatial_organoids.json", "w") as f:
        json.dump({
            "low": {"batch": low_batch, "index": low_idx,
                    "motion_loss": sorted_means[low_idx], "area_px": area_dict[low_batch],
                    "m_copy": mcopy_dict[low_batch]},
            "medium": {"batch": med_batch, "index": med_idx,
                       "motion_loss": sorted_means[med_idx], "area_px": area_dict[med_batch],
                       "m_copy": mcopy_dict[med_batch]},
            "high": {"batch": high_batch, "index": high_idx,
                     "motion_loss": sorted_means[high_idx], "area_px": area_dict[high_batch],
                     "m_copy": mcopy_dict[high_batch]},
            "size_band_px": [float(size_lo), float(size_hi)],
            "median_area_px": float(median_area),
        }, f, indent=2)
    print(f"  Saved: selected_spatial_organoids.json")

    # ══════════════════════════════════════════════════════════════════════
    # Plots — no titles on any panel
    # ══════════════════════════════════════════════════════════════════════

    # ── Fig 3AB: Temporal arrow plot (replaces bar chart + violins) ──
    # Load per-window losses in temporal order
    n_windows = len(per_organoid_window_loss[sorted_batches[0]])
    window_cmap = plt.cm.coolwarm

    # Save per-window losses CSV
    with open(OUTPUT_DIR / "per_window_losses_temporal.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_name", "organoid_index"] +
                         [f"window_{w}" for w in range(n_windows)])
        for idx, b in enumerate(sorted_batches):
            wl = per_organoid_window_loss[b]
            writer.writerow([b, idx] + [f"{v:.8f}" for v in wl])
    print(f"  Saved: per_window_losses_temporal.csv")

    # Vertical orientation: organoid index on x-axis, loss on y-axis
    fig, ax = plt.subplots(figsize=(14, 5))

    for idx, b in enumerate(sorted_batches):
        wl = per_organoid_window_loss[b]  # (n_windows,) in temporal order
        x_pos = idx

        # Arrows connecting consecutive windows
        for w in range(len(wl) - 1):
            color = window_cmap(w / max(len(wl) - 1, 1))
            ax.annotate("", xy=(x_pos, wl[w + 1]), xytext=(x_pos, wl[w]),
                         arrowprops=dict(arrowstyle="-|>", color=color,
                                         lw=0.6, alpha=0.6))

        # Dots colored by temporal order
        ax.scatter([x_pos] * len(wl), wl,
                    c=range(len(wl)), cmap="coolwarm", s=6,
                    edgecolors="none", alpha=0.8, zorder=3,
                    vmin=0, vmax=len(wl) - 1)

        # Mean as black tick
        ax.plot(x_pos, sorted_means[idx], "k_", markersize=8,
                markeredgewidth=1.5, zorder=4)

    ax.set_xlabel("Organoid index (sorted by mean motion loss)", fontsize=11)
    ax.set_ylabel(r"Motion loss $\|\Delta_{GT} - \Delta_{pred}\|^2$", fontsize=11)
    ax.set_xlim(-1, n)

    # Temporal colorbar
    sm = plt.cm.ScalarMappable(cmap="coolwarm",
                                norm=plt.Normalize(0, n_windows - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, aspect=20)
    cbar.set_label("Temporal window", fontsize=10)
    cbar.set_ticks([0, (n_windows - 1) / 2, n_windows - 1])
    cbar.set_ticklabels(["Early", "Mid", "Late"])

    # Variance ratio
    ax.text(0.02, 0.95,
            f"Inter-organoid variance / Intra-organoid variance = {f_stat:.1f}",
            transform=ax.transAxes, fontsize=10, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3ab_temporal_arrows.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig3ab_temporal_arrows.png")

    # ── Fig 3C: Composite visualization (GT base + residual heatmap + centroid trail) ──
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import ImageGrid

    maps = []
    gt_frames = []
    centroids_list = []
    valid_c = []
    for b, lbl in zip(c_batches, c_labels):
        sp = spatial_dir / f"{b}.npy"
        if sp.exists():
            maps.append(np.load(sp).mean(axis=0))  # (128, 128)
            gt_frames.append(load_first_last_composite(b))
            centroids_list.append(load_centroids(b))
            valid_c.append((b, lbl, sorted_means[sorted_batches.index(b)]))

    if len(maps) >= 2:
        # Log-scale normalization shared across panels
        all_vals = np.concatenate([m.ravel() for m in maps])
        vmin = max(all_vals[all_vals > 0].min(), 1e-8)
        vmax = all_vals.max()
        norm = LogNorm(vmin=vmin, vmax=vmax)

        n_panels = len(maps)
        fig = plt.figure(figsize=(5 * n_panels + 1.5, 5))
        grid = ImageGrid(fig, 111, nrows_ncols=(1, n_panels),
                          axes_pad=0.4, cbar_mode="single", cbar_location="right",
                          cbar_pad=0.15, cbar_size="5%")

        for i, (smap, gt, centroids, (b, lbl, loss)) in enumerate(
                zip(maps, gt_frames, centroids_list, valid_c)):
            ax = grid[i]

            # Layer 1: First/last frame composite (blue=first, red=last)
            ax.imshow(gt, alpha=0.6, interpolation="nearest")

            # Layer 2: Residual heatmap (inferno, log scale)
            smap_clipped = np.where(smap > 0, smap, vmin)
            im = ax.imshow(smap_clipped, cmap="inferno", norm=norm, alpha=0.5,
                            interpolation="nearest")

            # Layer 3: Centroid trajectory
            if centroids is not None:
                cx, cy = centroids[:, 0], centroids[:, 1]
                # Scale centroids from original coords to 128x128
                # Centroids are already in 128x128 space from segmentation
                ax.plot(cx, cy, "-", color="cyan", linewidth=0.8, alpha=0.7)
                ax.scatter(cx, cy, c="cyan", s=1, alpha=0.4)
                ax.scatter(cx[0], cy[0], c="lime", s=35, edgecolors="black",
                           linewidth=0.8, zorder=10)
                ax.scatter(cx[-1], cy[-1], c="red", s=35, edgecolors="black",
                           linewidth=0.8, zorder=10)
                # Net displacement arrow
                ax.annotate("", xy=(cx[-1], cy[-1]), xytext=(cx[0], cy[0]),
                            arrowprops=dict(arrowstyle="->", color="cyan", lw=2))
                # Compute total path length
                diffs = np.sqrt(np.diff(cx)**2 + np.diff(cy)**2)
                path_len = diffs.sum()
                displacement_str = f"Path: {path_len:.0f} px"
            else:
                displacement_str = "Path: N/A"

            area = area_dict.get(b, 0)
            ax.set_xlabel(f"{lbl}\n{displacement_str} | Area: {area} px | Loss: {loss:.2e}",
                          fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(0, 127)
            ax.set_ylim(127, 0)

        grid.cbar_axes[0].colorbar(im)
        grid.cbar_axes[0].set_ylabel(r"Mean $|\Delta_{GT} - \Delta_{pred}|^2$", fontsize=11)

        fig.savefig(OUTPUT_DIR / "fig3c_spatial.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: fig3c_spatial.png")

    # ── Area distribution + area vs motion loss scatter ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel 1: Area histogram
    ax = axes[0]
    ax.hist(areas_arr, bins=25, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(median_area, color="red", linestyle="--", linewidth=1.5,
               label=f"Median = {median_area:.0f} px")
    ax.axvspan(size_lo, size_hi, alpha=0.15, color="green", label="Selection band")
    # Mark selected organoids
    for lbl, b, marker in [("Low", low_batch, "v"), ("Med", med_batch, "s"), ("High", high_batch, "^")]:
        ax.axvline(area_dict[b], color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.plot(area_dict[b], ax.get_ylim()[1] * 0.9, marker, color="orange",
                markersize=8, zorder=5)
        ax.annotate(lbl, (area_dict[b], ax.get_ylim()[1] * 0.92),
                     fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("Organoid area (px)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=8)

    # Panel 2: Area vs motion loss scatter
    ax = axes[1]
    ax.scatter(areas_arr, raw_losses, s=15, alpha=0.6, edgecolors="k", linewidth=0.3)
    ax.set_xlabel("Organoid area (px)", fontsize=11)
    ax.set_ylabel(r"Mean motion loss $\|\Delta_{GT} - \Delta_{pred}\|^2$", fontsize=11)
    ax.text(0.02, 0.98, f"Spearman ρ = {rho_area_loss:.3f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
    # Mark selected organoids
    for lbl, b, marker in [("Low", low_batch, "v"), ("Med", med_batch, "s"), ("High", high_batch, "^")]:
        idx_b = sorted_batches.index(b)
        ax.scatter(area_dict[b], sorted_means[idx_b], s=80, marker=marker,
                    color="orange", edgecolors="black", linewidth=1, zorder=5)
        ax.annotate(lbl, (area_dict[b], sorted_means[idx_b]),
                     fontsize=8, xytext=(5, 5), textcoords="offset points")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_area_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: fig_area_distribution.png")

    print(f"\nDone. Outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
