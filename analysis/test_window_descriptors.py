"""Unit tests for compute_window_descriptors (classical_pipeline).

Tests use hand-crafted trajectories where ground truth is analytically known.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "classical_pipeline"))

# Import the function under test + constants
from find_most_reliable_classical_discriminator import compute_window_descriptors, WINDOW_SIZE


def _make_centroids(points):
    """Pad or use points directly as (T, 2) array with T = WINDOW_SIZE."""
    arr = np.array(points, dtype=np.float64)
    assert arr.shape[0] == WINDOW_SIZE, f"Need {WINDOW_SIZE} points, got {arr.shape[0]}"
    return arr


# ── Test 1: Stationary organoid ──────────────────────────────────────────

def test_stationary():
    """An organoid that doesn't move should have zero velocity and displacement."""
    pts = np.tile([50.0, 50.0], (WINDOW_SIZE, 1))
    results = compute_window_descriptors(pts)
    assert len(results) == 1, f"Expected 1 window, got {len(results)}"
    r = results[0]

    assert r["mean_velocity"] == 0.0, f"mean_velocity should be 0, got {r['mean_velocity']}"
    assert r["path_length"] == 0.0 or r["path_length"] < 1e-6, \
        f"path_length should be ~0, got {r['path_length']}"
    assert r["total_displacement"] == 0.0 or r["total_displacement"] < 1e-6, \
        f"total_displacement should be ~0, got {r['total_displacement']}"
    assert r["acceleration"] == 0.0, f"acceleration should be 0, got {r['acceleration']}"
    print("  PASS: stationary organoid")


# ── Test 2: Constant-speed straight line ─────────────────────────────────

def test_straight_line():
    """Straight-line motion: tortuosity ≈ 1, turning angle ≈ 0, displacement = path."""
    step = 2.0
    pts = np.array([[10.0 + i * step, 50.0] for i in range(WINDOW_SIZE)])
    results = compute_window_descriptors(pts)
    assert len(results) == 1
    r = results[0]

    # Mean velocity = step size per frame
    assert abs(r["mean_velocity"] - step) < 1e-6, \
        f"mean_velocity should be {step}, got {r['mean_velocity']}"

    # Acceleration should be 0 (constant speed)
    assert abs(r["acceleration"]) < 1e-6, \
        f"acceleration should be ~0, got {r['acceleration']}"

    # Tortuosity = path_length / displacement ≈ 1 for straight line
    assert abs(r["tortuosity"] - 1.0) < 0.05, \
        f"tortuosity should be ~1.0, got {r['tortuosity']}"

    # Turning angle ≈ 0 for straight line
    assert r["mean_turning_angle"] < 0.05, \
        f"mean_turning_angle should be ~0, got {r['mean_turning_angle']}"

    # Total displacement = path length for straight line
    assert abs(r["total_displacement"] - r["path_length"]) < 0.5, \
        f"total_disp ({r['total_displacement']}) should ≈ path_len ({r['path_length']})"

    # MSD slope for ballistic (straight-line) motion should be ≈ 2
    assert abs(r["msd_slope"] - 2.0) < 0.3, \
        f"msd_slope should be ~2.0 (ballistic), got {r['msd_slope']}"

    print("  PASS: straight line")


# ── Test 3: 90-degree turn ───────────────────────────────────────────────

def test_right_angle_turn():
    """Half the window going right, then half going up → should detect turning."""
    half = WINDOW_SIZE // 2
    pts_right = [[10.0 + i * 2.0, 50.0] for i in range(half)]
    corner_x = 10.0 + (half - 1) * 2.0
    pts_up = [[corner_x, 50.0 + (i + 1) * 2.0] for i in range(WINDOW_SIZE - half)]
    pts = np.array(pts_right + pts_up, dtype=np.float64)

    results = compute_window_descriptors(pts)
    assert len(results) == 1
    r = results[0]

    # Tortuosity > 1 (path is longer than displacement)
    assert r["tortuosity"] > 1.0, \
        f"tortuosity should be > 1.0 for L-shaped path, got {r['tortuosity']}"

    # Mean turning angle should be nonzero (has a bend)
    assert r["mean_turning_angle"] > 0.1, \
        f"mean_turning_angle should be > 0.1 for L-path, got {r['mean_turning_angle']}"

    # Displacement should be less than path length
    assert r["total_displacement"] < r["path_length"], \
        f"displacement ({r['total_displacement']}) should < path_len ({r['path_length']})"

    print("  PASS: right-angle turn")


# ── Test 4: Circular motion ──────────────────────────────────────────────

def test_circular_motion():
    """Points on a circle: high tortuosity, near-constant turning, displacement ≈ 0 if full loop."""
    radius = 20.0
    angles = np.linspace(0, 2 * np.pi, WINDOW_SIZE, endpoint=False)
    pts = np.column_stack([
        50.0 + radius * np.cos(angles),
        50.0 + radius * np.sin(angles),
    ])
    results = compute_window_descriptors(pts)
    assert len(results) == 1
    r = results[0]

    # Velocity should be roughly constant and > 0
    assert r["mean_velocity"] > 0, f"mean_velocity should be > 0, got {r['mean_velocity']}"

    # Tortuosity >> 1 (path wraps around, displacement is small)
    assert r["tortuosity"] > 2.0, \
        f"tortuosity should be >> 1 for circle, got {r['tortuosity']}"

    # MSD slope should be < 2 (sub-ballistic / confined)
    assert r["msd_slope"] < 2.0, \
        f"msd_slope should be < 2 for confined circular motion, got {r['msd_slope']}"

    print("  PASS: circular motion")


# ── Test 5: NaN handling ─────────────────────────────────────────────────

def test_nan_frames():
    """Windows with too many NaNs should be skipped."""
    # All NaN → should return empty
    pts_all_nan = np.full((WINDOW_SIZE, 2), np.nan)
    results = compute_window_descriptors(pts_all_nan)
    assert len(results) == 0, f"All-NaN window should be skipped, got {len(results)} results"

    # Only 2 valid frames (< 3) → should be skipped
    pts_mostly_nan = np.full((WINDOW_SIZE, 2), np.nan)
    pts_mostly_nan[0] = [10.0, 10.0]
    pts_mostly_nan[1] = [12.0, 10.0]
    results = compute_window_descriptors(pts_mostly_nan)
    assert len(results) == 0, f"<3 valid frames should be skipped, got {len(results)} results"

    # 4 valid frames → should produce a result
    pts_some_nan = np.full((WINDOW_SIZE, 2), np.nan)
    pts_some_nan[0] = [10.0, 10.0]
    pts_some_nan[1] = [12.0, 10.0]
    pts_some_nan[2] = [14.0, 10.0]
    pts_some_nan[3] = [16.0, 10.0]
    results = compute_window_descriptors(pts_some_nan)
    assert len(results) == 1, f"4 valid frames should produce 1 result, got {len(results)}"

    print("  PASS: NaN handling")


# ── Test 6: Multiple windows ────────────────────────────────────────────

def test_multiple_windows():
    """A trajectory with 2*WINDOW_SIZE frames should produce 2 windows."""
    step = 1.0
    pts = np.array([[i * step, 0.0] for i in range(2 * WINDOW_SIZE)], dtype=np.float64)
    results = compute_window_descriptors(pts)
    assert len(results) == 2, f"Expected 2 windows from {2*WINDOW_SIZE} frames, got {len(results)}"

    # Both windows should have similar mean velocity
    assert abs(results[0]["mean_velocity"] - results[1]["mean_velocity"]) < 1e-6, \
        "Both windows should have same mean velocity for constant motion"

    print("  PASS: multiple windows")


# ── Test 7: Outreach ratio sanity ────────────────────────────────────────

def test_outreach_ratio():
    """For straight-line motion, outreach_ratio ≈ 1 (max_disp ≈ path_len)."""
    step = 3.0
    pts = np.array([[i * step, 0.0] for i in range(WINDOW_SIZE)], dtype=np.float64)
    results = compute_window_descriptors(pts)
    r = results[0]

    # For straight line, max displacement from start = path length
    assert abs(r["outreach_ratio"] - 1.0) < 0.1, \
        f"outreach_ratio should be ~1.0 for straight line, got {r['outreach_ratio']}"

    print("  PASS: outreach ratio")


# ── Test 8: Acceleration for varying speed ───────────────────────────────

def test_acceleration():
    """Linearly increasing step sizes → nonzero acceleration."""
    # x positions: 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66
    x = [0.0]
    for i in range(1, WINDOW_SIZE):
        x.append(x[-1] + i)
    pts = np.array([[xi, 0.0] for xi in x], dtype=np.float64)
    results = compute_window_descriptors(pts)
    r = results[0]

    # Step lengths: 1, 2, 3, 4, ... → acceleration = mean(|diff|) = 1.0
    assert r["acceleration"] > 0.5, \
        f"acceleration should be > 0.5 for increasing speed, got {r['acceleration']}"
    assert abs(r["acceleration"] - 1.0) < 1e-6, \
        f"acceleration should be ~1.0, got {r['acceleration']}"

    print("  PASS: acceleration")


# ── Run all ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"WINDOW_SIZE = {WINDOW_SIZE}\n")
    test_stationary()
    test_straight_line()
    test_right_angle_turn()
    test_circular_motion()
    test_nan_frames()
    test_multiple_windows()
    test_outreach_ratio()
    test_acceleration()
    print(f"\nAll tests passed.")
