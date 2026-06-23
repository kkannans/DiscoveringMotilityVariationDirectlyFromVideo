#!/usr/bin/env bash
# stage_cache.sh — populate ./outputs and ./dataset from a local NNFM/Q8_v23 cache via symlinks,
# so figures/comparisons regenerate without retraining. For a portable copy of the same layout,
# use download_data.py instead. Idempotent; safe to re-run.
#
# NOTE: some outputs/<sub> entries are now whole-directory symlinks that resolve straight into
# the published cache (e.g. outputs/classical_descriptors -> ../original/... -> $CACHE/...).
# Creating per-file links inside those would either error ("are the same file") or pollute the
# published cache. The link() helper below skips both cases, so this script is safe to re-run
# regardless of how outputs/ has been wired up.
set -euo pipefail
cd "$(dirname "$0")"

CACHE=/users/k/k/kkannans/NNFM/Q8_v23
DRAFT=$CACHE/Alife_draft3
DATASET=/users/k/k/kkannans/alife-2026/dataset
CACHE_REAL=$(readlink -f "$CACHE")

# link SRC DST — create DST as a symlink to SRC. DST ending in "/" means "inside this directory".
# Skips silently when (a) the source is missing, (b) DST's directory resolves into the published
# cache (never write into the cache), or (c) DST already resolves to the same file as SRC.
link() {
  local src=$1 dst=$2
  if [ ! -e "$src" ] && [ ! -L "$src" ]; then
    echo "WARN missing source: $src"
    return 0
  fi
  case "$dst" in
    */) dst="${dst}$(basename "$src")" ;;
  esac
  local parent_real
  parent_real=$(readlink -f "$(dirname "$dst")" 2>/dev/null || true)
  case "${parent_real}/" in
    "$CACHE_REAL"/*)
      echo "skip (cache-backed, already linked): $dst"
      return 0
      ;;
  esac
  if [ -e "$dst" ] || [ -L "$dst" ]; then
    if [ "$(readlink -f "$dst" 2>/dev/null || true)" = "$(readlink -f "$src" 2>/dev/null || true)" ]; then
      return 0
    fi
  fi
  ln -sfn "$src" "$dst"
}

mkdir -p outputs/classical_descriptors outputs/intermediates outputs/datainfo outputs/models

# ── dataset (108 flat MP4s) ──────────────────────────────────────────────
link "$DATASET" dataset

# ── classical artifacts ──────────────────────────────────────────────────
link "$CACHE/classical_descriptors/motility_descriptors.csv"  outputs/classical_descriptors/
link "$CACHE/classical_descriptors/centroid_trajectories.npz" outputs/classical_descriptors/
link "$CACHE/classical_descriptors/segmentation"              outputs/classical_descriptors/segmentation

# ── intermediates (prediction surprise) ──────────────────────────────────
link "$CACHE/intermediates/per_organoid_summary.csv"          outputs/intermediates/
link "$CACHE/intermediates/per_seed_mse.csv"                  outputs/intermediates/
link "$DRAFT/Fig2/per_organoid_residuals"                     outputs/intermediates/per_organoid_residuals
link "$DRAFT/Fig2/spatial_residual_maps"                      outputs/intermediates/spatial_residual_maps
link "$DRAFT/Fig3/selected_spatial_organoids.json"            outputs/intermediates/
link "$DRAFT/Supplementary_Figures/loocv_comparison/predrnn_per_organoid.csv" outputs/intermediates/
link "$DRAFT/test_assignments.json"                          outputs/intermediates/test_assignments.json

# ── datainfo (committed splits + test assignments) ───────────────────────
for f in datainfo/data_split_covering_*.json; do link "$(pwd)/$f" "outputs/datainfo/$(basename "$f")"; done
link "$DRAFT/test_assignments.json"                          outputs/datainfo/test_assignments.json

# ── trained checkpoints (both naming conventions) ────────────────────────
for M in SimVP_TAU PredRNN; do
  for s in $(seq 1 10); do
    src=$CACHE/openstl_training/${M}_K6_N6_seed${s}_pre_residual
    [ -d "$src" ] || { echo "WARN missing $src"; continue; }
    link "$src" outputs/models/${M}_K6_N6_seed${s}_residual      # get_output_dir naming (extract/test)
    link "$src" outputs/models/${M}_K6_N6_seed${s}_pre_residual  # fig scripts' explicit naming
  done
done

echo "staged: $(ls outputs/intermediates/per_organoid_residuals/*.npz 2>/dev/null | wc -l) SimVP residual npz, dataset=$(ls dataset/*.mp4 2>/dev/null | wc -l) mp4s"
echo "OK — now run the figure/comparison commands in REPRODUCIBILITY.md §Runbook."
