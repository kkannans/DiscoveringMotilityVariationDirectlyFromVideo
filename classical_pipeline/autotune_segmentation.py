#!/usr/bin/env python3
"""
Auto-tune organoid segmentation parameters per batch.

For each organoid video, searches over combinations of color channel selection,
preprocessing, thresholding, morphological cleanup, and area filtering to find
the best binary mask that tracks the organoid across all frames.

Usage:
    # Single batch
    python auto_tune_segmentation.py --batch batch-000084

    # Range of batches (by index into sorted list)
    python auto_tune_segmentation.py --batch-range 0 27

    # All batches
    python auto_tune_segmentation.py --all

    # Custom video directory
    python auto_tune_segmentation.py --all --video-dir /path/to/videos/
"""

import argparse
import csv
import json
import sys
import time
from itertools import product
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import paths

# ─── Default paths ───────────────────────────────────────────────────────────

DEFAULT_VIDEO_DIR = paths.DATA_ROOT
DEFAULT_OUTPUT_DIR = paths.CLASSICAL_DIR / "segmentation"
FRAME_SIZE = 128  # resize to 128x128


# ─── Search spaces ───────────────────────────────────────────────────────────

COARSE_GRID = {
    "color_mode": ["gray", "bg_distance", "lightness_lab", "value_hsv"],
    "blur_size": [0, 11],
    "clahe": [False, True],
    "clahe_clip": [2.0],
    "threshold_method": ["otsu", "adaptive_gaussian", "percentile_5", "bg_relative_30"],
    "morph_kernel": [5],
    "morph_ops": ["close_then_open"],
    "min_area": [50],
    "max_area_frac": [0.3],
}

FINE_GRID_EXTRA = {
    "morph_kernel": [3, 5, 7],
    "morph_ops": ["close_then_open", "open_then_close", "close_only"],
    "min_area": [30, 50, 100],
    "max_area_frac": [0.3, 0.5],
}

# Expanded grid for retry mode: more aggressive options for hard cases
RETRY_COARSE_GRID = {
    "color_mode": ["gray", "gray_inverted", "bg_distance", "lightness_lab",
                    "value_hsv", "saturation_hsv"],
    "blur_size": [0, 5, 11, 21],
    "clahe": [False, True],
    "clahe_clip": [2.0, 4.0],
    "threshold_method": ["otsu", "adaptive_gaussian", "percentile_5",
                         "percentile_10", "bg_relative_20", "bg_relative_30",
                         "bg_relative_40"],
    "morph_kernel": [5],
    "morph_ops": ["close_then_open"],
    "min_area": [50],
    "max_area_frac": [0.3],
}

TOP_K_COARSE = 5
TOP_K_COARSE_RETRY = 10


# ─── Video loading ───────────────────────────────────────────────────────────

def load_video_frames(video_path, target_size=FRAME_SIZE):
    """Load all frames from an MP4 as (T, H, W, 3) uint8 RGB array."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize to target
        if frame.shape[0] != target_size or frame.shape[1] != target_size:
            frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"No frames loaded from: {video_path}")
    return np.stack(frames)  # (T, H, W, 3)


# ─── Core segmentation ──────────────────────────────────────────────────────

def segment_frame(frame, params):
    """Apply segmentation pipeline with given parameters.

    Args:
        frame: (H, W, 3) uint8 RGB
        params: dict with keys from search space

    Returns:
        mask: (H, W) binary uint8 (0 or 255)
        centroid: (x, y) tuple or None
        area: int
        props: dict with eccentricity, orientation, bbox
    """
    H, W = frame.shape[:2]

    # Step 1: Color channel selection
    color_mode = params["color_mode"]
    if color_mode == "gray":
        channel = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        dark_is_organoid = True
    elif color_mode == "gray_inverted":
        channel = 255 - cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        dark_is_organoid = False
    elif color_mode == "value_hsv":
        channel = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)[:, :, 2]
        dark_is_organoid = True
    elif color_mode == "saturation_hsv":
        channel = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)[:, :, 1]
        dark_is_organoid = False
    elif color_mode == "lightness_lab":
        channel = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)[:, :, 0]
        dark_is_organoid = True
    elif color_mode == "a_channel_lab":
        channel = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)[:, :, 1]
        dark_is_organoid = False
    elif color_mode == "bg_distance":
        # Use full border strip (outer 8px) with trimmed mean for robust bg estimation
        border = 8
        top = frame[:border, :].reshape(-1, 3)
        bottom = frame[-border:, :].reshape(-1, 3)
        left = frame[border:-border, :border].reshape(-1, 3)
        right = frame[border:-border, -border:].reshape(-1, 3)
        border_pixels = np.concatenate([top, bottom, left, right]).astype(float)
        # Trimmed mean: remove top/bottom 10% by brightness to exclude artifacts
        brightness = border_pixels.mean(axis=1)
        lo, hi = np.percentile(brightness, [10, 90])
        trim_mask = (brightness >= lo) & (brightness <= hi)
        bg_color = border_pixels[trim_mask].mean(axis=0) if trim_mask.sum() > 0 else border_pixels.mean(axis=0)
        dist = np.sqrt(np.sum((frame.astype(float) - bg_color) ** 2, axis=2))
        channel = np.clip(dist, 0, 255).astype(np.uint8)
        dark_is_organoid = False
    else:
        raise ValueError(f"Unknown color_mode: {color_mode}")

    # Step 2: Preprocessing
    blur = params["blur_size"]
    if blur > 0:
        channel = cv2.GaussianBlur(channel, (blur, blur), 0)

    if params.get("clahe", False):
        clahe = cv2.createCLAHE(
            clipLimit=params.get("clahe_clip", 2.0),
            tileGridSize=(8, 8))
        channel = clahe.apply(channel)

    # Step 3: Thresholding
    method = params["threshold_method"]
    if method == "otsu":
        _, raw_mask = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if dark_is_organoid:
            raw_mask = 255 - raw_mask
    elif method == "adaptive_gaussian":
        raw_mask = cv2.adaptiveThreshold(
            channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV if dark_is_organoid else cv2.THRESH_BINARY,
            blockSize=51, C=10)
    elif method == "adaptive_mean":
        raw_mask = cv2.adaptiveThreshold(
            channel, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV if dark_is_organoid else cv2.THRESH_BINARY,
            blockSize=51, C=10)
    elif method.startswith("percentile_"):
        pct = int(method.split("_")[1])
        # Compute percentile on central 80% of frame to avoid corner artifacts
        border = int(0.1 * H)
        central = channel[border:-border, border:-border] if border > 0 else channel
        thresh = np.percentile(central, pct)
        if dark_is_organoid:
            raw_mask = (channel < thresh).astype(np.uint8) * 255
        else:
            raw_mask = (channel > np.percentile(central, 100 - pct)).astype(np.uint8) * 255
    elif method.startswith("bg_relative_"):
        offset = int(method.split("_")[2])
        corners = np.concatenate([
            channel[:8, :8].ravel(), channel[:8, -8:].ravel(),
            channel[-8:, :8].ravel(), channel[-8:, -8:].ravel()])
        bg_val = np.median(corners)
        if dark_is_organoid:
            raw_mask = (channel < bg_val - offset).astype(np.uint8) * 255
        else:
            raw_mask = (channel > bg_val + offset).astype(np.uint8) * 255
    else:
        raise ValueError(f"Unknown threshold_method: {method}")

    # Step 4: Morphological cleanup
    k = params.get("morph_kernel", 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    ops = params.get("morph_ops", "close_then_open")
    if ops == "close_then_open":
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    elif ops == "open_then_close":
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)
    elif ops == "close_only":
        raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel)

    # Step 5: Largest connected component + area filtering
    contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(raw_mask)
    centroid = None
    area = 0
    props = {}

    if contours:
        min_area = params.get("min_area", 50)
        max_area = params.get("max_area_frac", 0.3) * H * W
        border_margin = 12  # reject contours whose centroid is within 12px of border

        # Filter by area and border exclusion
        valid_contours = []
        for c in contours:
            ca = cv2.contourArea(c)
            if not (min_area < ca < max_area):
                continue
            M_c = cv2.moments(c)
            if M_c["m00"] > 0:
                c_cx = M_c["m10"] / M_c["m00"]
                c_cy = M_c["m01"] / M_c["m00"]
                if (border_margin <= c_cx <= W - border_margin and
                        border_margin <= c_cy <= H - border_margin):
                    valid_contours.append(c)

        if valid_contours:
            # Score each contour by area * center_proximity^2
            # Strongly prefers blobs near frame center over corner artifacts
            frame_cx, frame_cy = W / 2.0, H / 2.0
            max_dist = np.sqrt(frame_cx**2 + frame_cy**2)
            best_contour = None
            best_score = -1
            for c in valid_contours:
                M_c = cv2.moments(c)
                c_cx = M_c["m10"] / M_c["m00"]
                c_cy = M_c["m01"] / M_c["m00"]
                dist = np.sqrt((c_cx - frame_cx)**2 + (c_cy - frame_cy)**2)
                proximity = 1.0 - (dist / max_dist)
                score = cv2.contourArea(c) * (proximity ** 2)
                if score > best_score:
                    best_score = score
                    best_contour = c

            cv2.drawContours(mask, [best_contour], -1, 255, -1)
            area = cv2.contourArea(best_contour)
            M = cv2.moments(best_contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                centroid = (cx, cy)
            if len(best_contour) >= 5:
                ellipse = cv2.fitEllipse(best_contour)
                (ex, ey), (ma, MA), angle = ellipse
                eccentricity = np.sqrt(1 - (min(ma, MA) / (max(ma, MA) + 1e-10))**2)
                props = {"eccentricity": eccentricity, "orientation": angle,
                         "major_axis": max(ma, MA), "minor_axis": min(ma, MA)}

    return mask, centroid, area, props


# ─── Temporal smoothing ──────────────────────────────────────────────

def temporal_smooth_centroids(centroids, areas, jump_threshold=20.0, window=5):
    """Post-process centroid trajectory to fix jumps and gaps.

    - If a frame has no detection (centroid is None), interpolate from neighbors.
    - If a frame's centroid jumps > jump_threshold from the median of its
      window-frame neighborhood, replace with interpolated position.

    Args:
        centroids: list of (cx, cy) tuples or None
        areas: list of int
        jump_threshold: max allowed deviation from local median (pixels)
        window: neighborhood size for median computation

    Returns:
        smoothed_centroids: list of (cx, cy) tuples or None (only None if
            no valid neighbors exist at all)
        smoothed_areas: list of int (unchanged for valid frames, 0 for
            frames that remain unresolved)
    """
    T = len(centroids)
    smoothed_cx = np.full(T, np.nan)
    smoothed_cy = np.full(T, np.nan)
    smoothed_areas = list(areas)

    # Fill in valid centroids
    for i, c in enumerate(centroids):
        if c is not None:
            smoothed_cx[i] = c[0]
            smoothed_cy[i] = c[1]

    # Pass 1: Detect jumps relative to local median and mark as NaN
    for i in range(T):
        if np.isnan(smoothed_cx[i]):
            continue
        half_w = window // 2
        lo = max(0, i - half_w)
        hi = min(T, i + half_w + 1)
        neighbors_cx = []
        neighbors_cy = []
        for j in range(lo, hi):
            if j != i and not np.isnan(smoothed_cx[j]):
                neighbors_cx.append(smoothed_cx[j])
                neighbors_cy.append(smoothed_cy[j])
        if len(neighbors_cx) < 2:
            continue
        med_cx = np.median(neighbors_cx)
        med_cy = np.median(neighbors_cy)
        dist = np.sqrt((smoothed_cx[i] - med_cx)**2 + (smoothed_cy[i] - med_cy)**2)
        if dist > jump_threshold:
            smoothed_cx[i] = np.nan
            smoothed_cy[i] = np.nan

    # Pass 2: Interpolate NaN gaps using linear interpolation
    valid_idx = np.where(~np.isnan(smoothed_cx))[0]
    if len(valid_idx) >= 2:
        smoothed_cx = np.interp(np.arange(T), valid_idx, smoothed_cx[valid_idx])
        smoothed_cy = np.interp(np.arange(T), valid_idx, smoothed_cy[valid_idx])
    elif len(valid_idx) == 1:
        # Only one valid point; fill constant
        smoothed_cx[:] = smoothed_cx[valid_idx[0]]
        smoothed_cy[:] = smoothed_cy[valid_idx[0]]
    else:
        # No valid centroids at all; return original
        return list(centroids), list(areas)

    # Build output
    result_centroids = []
    for i in range(T):
        if np.isnan(smoothed_cx[i]):
            result_centroids.append(None)
        else:
            result_centroids.append((float(smoothed_cx[i]), float(smoothed_cy[i])))
    return result_centroids, smoothed_areas


def temporal_filter_masks(masks, centroids, areas, all_props):
    """Filter masks by replacing false-positive frames with previous valid mask.

    For each frame, compute displacement from the previous frame's centroid.
    If displacement exceeds the median displacement for the batch, replace
    that frame's mask/centroid/area/props with the previous valid frame's values.
    This removes transient false positives (single-frame jumps to wrong object).

    Also replaces empty masks (no detection) with the previous valid mask.

    Args:
        masks: list of (H, W) binary masks
        centroids: list of (cx, cy) or None
        areas: list of int
        all_props: list of dicts

    Returns:
        filtered_masks, filtered_centroids, filtered_areas, filtered_props, n_replaced
    """
    T = len(masks)
    if T < 3:
        return masks, centroids, areas, all_props, 0

    # Compute per-frame displacements
    displacements = []
    for i in range(1, T):
        if centroids[i] is not None and centroids[i - 1] is not None:
            dx = centroids[i][0] - centroids[i - 1][0]
            dy = centroids[i][1] - centroids[i - 1][1]
            displacements.append(np.sqrt(dx**2 + dy**2))

    if len(displacements) < 3:
        return masks, centroids, areas, all_props, 0

    median_disp = np.median(displacements)
    # Threshold: displacements > 3x median are outliers (false positives)
    # Use at least 5px as minimum threshold to avoid over-filtering slow-moving organoids
    threshold = max(3.0 * median_disp, 5.0)

    f_masks = list(masks)
    f_centroids = list(centroids)
    f_areas = list(areas)
    f_props = list(all_props)
    n_replaced = 0

    # Track last valid frame index
    last_valid = 0  # frame 0 is assumed valid (anchor)

    for i in range(1, T):
        replace = False

        if f_centroids[i] is None or f_areas[i] == 0:
            # No detection — replace with last valid
            replace = True
        elif f_centroids[last_valid] is not None:
            dx = f_centroids[i][0] - f_centroids[last_valid][0]
            dy = f_centroids[i][1] - f_centroids[last_valid][1]
            disp = np.sqrt(dx**2 + dy**2)
            if disp > threshold:
                replace = True

        if replace:
            f_masks[i] = f_masks[last_valid].copy()
            f_centroids[i] = f_centroids[last_valid]
            f_areas[i] = f_areas[last_valid]
            f_props[i] = f_props[last_valid]
            n_replaced += 1
        else:
            last_valid = i

    return f_masks, f_centroids, f_areas, f_props, n_replaced


# ─── Quality scoring ─────────────────────────────────────────────────────────

def compute_quality_score(masks, centroids, areas, frame_size=FRAME_SIZE * FRAME_SIZE):
    """Score a segmentation result for one batch."""
    T = len(masks)
    areas = np.array(areas, dtype=float)

    empty_frac = np.sum(areas == 0) / T
    p_empty = -10.0 * empty_frac

    oversized_frac = np.sum(areas > 0.25 * frame_size) / T
    p_oversized = -10.0 * oversized_frac

    valid = (areas > 0) & (areas < 0.25 * frame_size)
    if valid.sum() < T * 0.5:
        return -100.0, {"empty_frac": float(empty_frac), "oversized_frac": float(oversized_frac)}

    valid_areas = areas[valid]

    area_cv = valid_areas.std() / (valid_areas.mean() + 1e-10)
    p_area_cv = -2.0 * area_cv

    valid_centroids = [c for c, v in zip(centroids, valid) if v and c is not None]
    if len(valid_centroids) > 1:
        c_arr = np.array(valid_centroids)
        displacements = np.sqrt(np.sum(np.diff(c_arr, axis=0)**2, axis=1))
        median_disp = np.median(displacements)
        jitter = displacements.std()
        p_jitter = -0.1 * jitter
    else:
        p_jitter = -5.0
        median_disp = 0

    mean_area = valid_areas.mean()
    if mean_area < 30:
        p_small = -3.0
    elif mean_area < 100:
        p_small = -0.5
    else:
        p_small = 0.0

    if 200 <= mean_area <= 5000:
        b_area = 1.0
    elif 100 <= mean_area <= 8000:
        b_area = 0.5
    else:
        b_area = 0.0

    b_valid = 1.0 * (valid.sum() / T)

    # Penalty 6: Off-center centroid (organoids are cropped centered)
    half = FRAME_SIZE / 2.0
    max_dist = np.sqrt(half**2 + half**2)
    p_offcenter = 0.0
    mean_center_dist = 0.0
    if valid_centroids:
        mean_cx = np.mean([c[0] for c in valid_centroids])
        mean_cy = np.mean([c[1] for c in valid_centroids])
        mean_center_dist = np.sqrt((mean_cx - half)**2 + (mean_cy - half)**2)
        frac_from_center = mean_center_dist / max_dist
        if frac_from_center > 0.4:
            p_offcenter = -5.0 * frac_from_center  # heavy penalty for corner blobs
        elif frac_from_center > 0.25:
            p_offcenter = -1.0 * frac_from_center  # mild penalty

    # Penalty 7 (Fix D): Detection gaps surrounded by valid frames
    # A gap is a run of empty frames that has valid frames on BOTH sides
    p_gap = 0.0
    gap_count = 0
    in_gap = False
    has_valid_before = False
    current_gap_len = 0
    for i in range(T):
        if areas[i] > 0:
            if in_gap and has_valid_before:
                # This gap was surrounded by valid frames
                gap_count += current_gap_len
            in_gap = False
            has_valid_before = True
            current_gap_len = 0
        else:
            if has_valid_before:
                in_gap = True
                current_gap_len += 1
    # Penalize: each surrounded gap frame costs -0.5
    p_gap = -0.5 * gap_count

    score = p_empty + p_oversized + p_area_cv + p_jitter + p_small + b_area + b_valid + p_offcenter + p_gap
    details = {
        "score": float(score),
        "empty_frac": float(empty_frac),
        "oversized_frac": float(oversized_frac),
        "area_cv": float(area_cv),
        "mean_area": float(mean_area),
        "centroid_jitter": float(p_jitter),
        "valid_frac": float(valid.sum() / T),
        "median_displacement": float(median_disp),
        "mean_center_distance": float(mean_center_dist),
        "offcenter_penalty": float(p_offcenter),
        "gap_penalty": float(p_gap),
        "gap_frames": gap_count,
    }
    return score, details


# ─── Evaluate one parameter combo on all frames ─────────────────────────────

def evaluate_params(frames, params):
    """Run segmentation with params on all frames, return score + details."""
    masks, centroids, areas, all_props = [], [], [], []
    for frame in frames:
        mask, centroid, area, props = segment_frame(frame, params)
        masks.append(mask)
        centroids.append(centroid)
        areas.append(area)
        all_props.append(props)
    score, details = compute_quality_score(masks, centroids, areas)
    return score, details, masks, centroids, areas, all_props


# ─── Grid generation ─────────────────────────────────────────────────────────

def expand_grid(grid_dict):
    """Expand a dict of lists into a list of dicts (cartesian product)."""
    keys = list(grid_dict.keys())
    combos = []
    for vals in product(*[grid_dict[k] for k in keys]):
        combos.append(dict(zip(keys, vals)))
    return combos


# ─── Overlay video ───────────────────────────────────────────────────────────

def generate_overlay_video(frames, masks, centroids, areas, props_list,
                           output_path, batch_name, params, score, fps=15):
    """Generate overlay MP4: green contour, red centroid, yellow bbox, text."""
    H, W = frames[0].shape[:2]
    # Use 256x256 for visibility
    out_size = 256
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_size, out_size))

    for i, (frame, mask, centroid, area) in enumerate(zip(frames, masks, centroids, areas)):
        # Upscale for visibility
        vis = cv2.resize(frame.copy(), (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        mask_up = cv2.resize(mask, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        # Green contour
        contours, _ = cv2.findContours(mask_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_bgr, contours, -1, (0, 255, 0), 2)

        # Red centroid dot
        if centroid is not None:
            cx_up = int(centroid[0] * out_size / W)
            cy_up = int(centroid[1] * out_size / H)
            cv2.circle(vis_bgr, (cx_up, cy_up), 4, (0, 0, 255), -1)

        # Yellow bounding box
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            cv2.rectangle(vis_bgr, (x, y), (x + w, y + h), (0, 255, 255), 1)

        # Text overlay
        ecc = props_list[i].get("eccentricity", 0) if props_list[i] else 0
        cv2.putText(vis_bgr, f"F{i:03d} A={int(area)} e={ecc:.2f}",
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(vis_bgr, f"{batch_name} S={score:.2f}",
                    (5, out_size - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        writer.write(vis_bgr)
    writer.release()


# ─── Process one batch ───────────────────────────────────────────────────────

def process_one_batch(batch_name, frames, output_dir, retry_mode=False):
    """Auto-tune segmentation for one batch.

    1. Coarse grid search (~64 combos, or expanded grid in retry mode)
    2. Score each on all frames
    3. Pick top K, refine with full morph/area grid
    4. Select best overall
    5. Apply temporal smoothing to centroid trajectory
    6. Generate overlay video
    7. Save best parameters + metrics

    Args:
        retry_mode: If True, uses expanded search space (RETRY_COARSE_GRID)
                    and more fine-tuning candidates (top 10 instead of 5).
    """
    batch_dir = output_dir / "auto_tune_results" / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done (skip check bypassed in retry mode)
    if not retry_mode and (batch_dir / "best_params.json").exists():
        print(f"  [SKIP] {batch_name} already processed")
        with open(batch_dir / "best_params.json") as f:
            params = json.load(f)
        with open(batch_dir / "best_score.json") as f:
            score_info = json.load(f)
        return params, score_info.get("score", 0), score_info

    T = len(frames)
    mode_label = "RETRY" if retry_mode else "normal"
    print(f"  Processing {batch_name}: {T} frames (mode={mode_label})")

    # ── Phase 1: Coarse grid ──
    grid = RETRY_COARSE_GRID if retry_mode else COARSE_GRID
    top_k = TOP_K_COARSE_RETRY if retry_mode else TOP_K_COARSE
    coarse_combos = expand_grid(grid)
    print(f"  Phase 1: {len(coarse_combos)} coarse combinations")

    search_log = []
    coarse_results = []
    t0 = time.time()

    for idx, params in enumerate(coarse_combos):
        score, details, _, _, _, _ = evaluate_params(frames, params)
        search_log.append({"phase": "coarse", "idx": idx, **params, **details})
        coarse_results.append((score, params, details))
        if (idx + 1) % 16 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            print(f"    Coarse {idx+1}/{len(coarse_combos)} "
                  f"({rate:.1f} combos/s, best so far: {max(r[0] for r in coarse_results):.2f})")

    # Sort by score, take top K
    coarse_results.sort(key=lambda x: x[0], reverse=True)
    top_coarse = coarse_results[:top_k]
    print(f"  Top {top_k} coarse scores: {[f'{r[0]:.2f}' for r in top_coarse]}")

    # ── Phase 2: Fine-tune top K ──
    fine_combos = []
    for _, base_params, _ in top_coarse:
        for mk in FINE_GRID_EXTRA["morph_kernel"]:
            for mo in FINE_GRID_EXTRA["morph_ops"]:
                for ma in FINE_GRID_EXTRA["min_area"]:
                    for maf in FINE_GRID_EXTRA["max_area_frac"]:
                        p = dict(base_params)
                        p["morph_kernel"] = mk
                        p["morph_ops"] = mo
                        p["min_area"] = ma
                        p["max_area_frac"] = maf
                        fine_combos.append(p)

    # Deduplicate
    seen = set()
    unique_fine = []
    for p in fine_combos:
        key = tuple(sorted(p.items()))
        if key not in seen:
            seen.add(key)
            unique_fine.append(p)
    fine_combos = unique_fine

    print(f"  Phase 2: {len(fine_combos)} fine-tuning combinations")

    fine_results = []
    t1 = time.time()
    for idx, params in enumerate(fine_combos):
        score, details, _, _, _, _ = evaluate_params(frames, params)
        search_log.append({"phase": "fine", "idx": idx, **params, **details})
        fine_results.append((score, params, details))
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t1
            rate = (idx + 1) / elapsed
            print(f"    Fine {idx+1}/{len(fine_combos)} "
                  f"({rate:.1f} combos/s, best so far: {max(r[0] for r in fine_results):.2f})")

    # ── Select best overall ──
    all_results = coarse_results + fine_results
    all_results.sort(key=lambda x: x[0], reverse=True)
    best_score, best_params, best_details = all_results[0]
    print(f"  Best score: {best_score:.3f}")
    print(f"  Best params: {best_params}")

    # ── Re-run best params to get masks for overlay ──
    _, _, masks, centroids, areas, all_props = evaluate_params(frames, best_params)

    # ── Apply temporal mask filter: replace outlier frames with previous valid mask ──
    masks, centroids, areas, all_props, n_mask_replaced = temporal_filter_masks(
        masks, centroids, areas, all_props)
    if n_mask_replaced > 0:
        print(f"  Temporal mask filter: {n_mask_replaced} frames replaced (displacement > 3x median)")

    # ── Apply temporal smoothing on centroids ──
    raw_centroids = list(centroids)
    centroids, areas = temporal_smooth_centroids(centroids, areas)
    n_smoothed = sum(1 for rc, sc in zip(raw_centroids, centroids)
                     if rc != sc and sc is not None)
    if n_smoothed > 0:
        print(f"  Temporal centroid smoothing: {n_smoothed} frames adjusted")

    # ── Save results ──
    with open(batch_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    with open(batch_dir / "best_score.json", "w") as f:
        json.dump(best_details, f, indent=2)

    # Search log CSV
    if search_log:
        # Collect all possible fieldnames across all entries
        all_keys = set()
        for entry in search_log:
            all_keys.update(entry.keys())
        fieldnames = sorted(all_keys)
        with open(batch_dir / "search_log.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(search_log)

    # Overlay video
    overlay_path = batch_dir / "overlay.mp4"
    generate_overlay_video(frames, masks, centroids, areas, all_props,
                           overlay_path, batch_name, best_params, best_score)
    print(f"  Overlay saved: {overlay_path}")

    # Save centroid trajectory as CSV
    with open(batch_dir / "centroids.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "cx", "cy", "area", "eccentricity", "orientation"])
        for i, (c, a, p) in enumerate(zip(centroids, areas, all_props)):
            cx = c[0] if c else ""
            cy = c[1] if c else ""
            ecc = p.get("eccentricity", "") if p else ""
            ori = p.get("orientation", "") if p else ""
            writer.writerow([i, cx, cy, a, ecc, ori])

    elapsed_total = time.time() - t0
    print(f"  Done in {elapsed_total:.1f}s")

    return best_params, best_score, best_details


# ─── Merge results ───────────────────────────────────────────────────────────

def merge_results(output_dir):
    """Merge all per-batch results into summary files."""
    results_dir = output_dir / "auto_tune_results"
    all_params = {}
    all_scores = []

    batch_dirs = sorted(results_dir.iterdir())
    for bd in batch_dirs:
        if not bd.is_dir():
            continue
        batch_name = bd.name
        params_file = bd / "best_params.json"
        score_file = bd / "best_score.json"
        if not params_file.exists() or not score_file.exists():
            continue
        with open(params_file) as f:
            params = json.load(f)
        with open(score_file) as f:
            score_info = json.load(f)
        all_params[batch_name] = params
        all_scores.append({"batch": batch_name, **score_info})

    # Save merged params
    with open(output_dir / "all_best_params.json", "w") as f:
        json.dump(all_params, f, indent=2)

    # Save merged scores CSV
    if all_scores:
        fieldnames = list(all_scores[0].keys())
        with open(output_dir / "all_scores.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_scores)

    # Identify failures (score < -2.0)
    failed = [s for s in all_scores if s.get("score", -100) < -2.0]
    with open(output_dir / "failed_batches.json", "w") as f:
        json.dump(failed, f, indent=2)

    print(f"\nMerge complete:")
    print(f"  Total batches: {len(all_scores)}")
    print(f"  Failed (score < -2.0): {len(failed)}")
    if all_scores:
        scores = [s["score"] for s in all_scores]
        print(f"  Score range: [{min(scores):.2f}, {max(scores):.2f}]")
        print(f"  Score median: {np.median(scores):.2f}")

    # Generate quality distribution plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        scores = [s["score"] for s in all_scores]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(scores, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(-2.0, color="red", linestyle="--", label="Failure threshold")
        ax.set_xlabel("Quality Score")
        ax.set_ylabel("Count")
        ax.set_title(f"Segmentation Quality Distribution (N={len(scores)})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "quality_distribution.png", dpi=150)
        plt.close()
        print(f"  Saved quality_distribution.png")

        # Method distribution
        color_modes = {}
        thresh_methods = {}
        for batch_name, params in all_params.items():
            cm = params.get("color_mode", "unknown")
            tm = params.get("threshold_method", "unknown")
            color_modes[cm] = color_modes.get(cm, 0) + 1
            thresh_methods[tm] = thresh_methods.get(tm, 0) + 1

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.barh(list(color_modes.keys()), list(color_modes.values()))
        ax1.set_xlabel("Wins")
        ax1.set_title("Color Mode Distribution")
        ax2.barh(list(thresh_methods.keys()), list(thresh_methods.values()))
        ax2.set_xlabel("Wins")
        ax2.set_title("Threshold Method Distribution")
        plt.tight_layout()
        plt.savefig(output_dir / "method_distribution.png", dpi=150)
        plt.close()
        print(f"  Saved method_distribution.png")
    except Exception as e:
        print(f"  Warning: Could not generate plots: {e}")

    return all_params, all_scores, failed


# ─── CLI ─────────────────────────────────────────────────────────────────────

def get_all_batch_names(video_dir):
    """Get sorted list of batch names from video directory."""
    mp4s = sorted(video_dir.glob("batch-*.mp4"))
    return [p.stem for p in mp4s]


def main():
    parser = argparse.ArgumentParser(description="Auto-tune organoid segmentation per batch")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--batch", type=str, help="Single batch name (e.g. batch-000084)")
    group.add_argument("--batch-range", type=int, nargs=2, metavar=("START", "END"),
                       help="Range of batch indices (inclusive)")
    group.add_argument("--all", action="store_true", help="Process all batches")
    group.add_argument("--merge-only", action="store_true", help="Only merge existing results")
    group.add_argument("--retry-batches", type=str, nargs="+", metavar="BATCH",
                       help="Re-run specific batches with expanded search space "
                            "(deletes existing results first)")
    parser.add_argument("--video-dir", type=str, default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        merge_results(output_dir)
        return

    all_batches = get_all_batch_names(video_dir)
    print(f"Found {len(all_batches)} batches in {video_dir}")

    # Select batches and determine mode
    retry_mode = False
    if args.retry_batches:
        retry_mode = True
        batches = args.retry_batches
        # Validate batch names
        for b in batches:
            if b not in all_batches:
                print(f"ERROR: {b} not found. Available: {all_batches[:5]}...")
                sys.exit(1)
        # Delete existing best_params.json for retry batches
        for b in batches:
            batch_dir = output_dir / "auto_tune_results" / b
            for fname in ["best_params.json", "best_score.json", "overlay.mp4",
                          "centroids.csv", "search_log.csv"]:
                fpath = batch_dir / fname
                if fpath.exists():
                    fpath.unlink()
                    print(f"  Deleted {fpath}")
        print(f"\nRetry mode: {len(batches)} batches with expanded search space")
    elif args.batch:
        if args.batch not in all_batches:
            print(f"ERROR: {args.batch} not found. Available: {all_batches[:5]}...")
            sys.exit(1)
        batches = [args.batch]
    elif args.batch_range:
        start, end = args.batch_range
        batches = all_batches[start:end + 1]
    else:
        batches = all_batches

    print(f"Processing {len(batches)} batches: {batches[0]} .. {batches[-1]}")
    print(f"Output: {output_dir}")

    # Process each batch
    results_summary = []
    for i, batch_name in enumerate(batches):
        print(f"\n[{i+1}/{len(batches)}] {batch_name}")
        video_path = video_dir / f"{batch_name}.mp4"
        if not video_path.exists():
            print(f"  [SKIP] Video not found: {video_path}")
            continue

        frames = load_video_frames(video_path)
        best_params, best_score, best_details = process_one_batch(
            batch_name, frames, output_dir, retry_mode=retry_mode)
        results_summary.append({
            "batch": batch_name,
            "score": best_score,
            "mean_area": best_details.get("mean_area", 0),
            "color_mode": best_params.get("color_mode", ""),
            "threshold_method": best_params.get("threshold_method", ""),
        })

    # Print summary
    print(f"\n{'='*60}")
    print(f"Completed {len(results_summary)} batches")
    if results_summary:
        scores = [r["score"] for r in results_summary]
        print(f"Score range: [{min(scores):.2f}, {max(scores):.2f}]")
        failed = [r for r in results_summary if r["score"] < -2.0]
        print(f"Failed: {len(failed)}/{len(results_summary)}")

    # Auto-merge if all batches processed
    if args.all or len(batches) == len(all_batches):
        print("\nMerging all results...")
        merge_results(output_dir)


if __name__ == "__main__":
    main()
