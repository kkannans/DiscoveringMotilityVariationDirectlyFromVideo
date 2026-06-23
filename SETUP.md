# Setup & running the pipeline

Setup and end-to-end usage for *Discovering organoid motility variation directly from video*
(ALife 2026). To only regenerate the figures, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Setup

Requires Python 3.11 and CUDA 12.4.

```bash
# 1. Create virtual environment (pick one)
uv venv .venv --python 3.11        # option A: uv (fast)
# python3.11 -m venv .venv         # option B: stdlib venv

# 2. Activate
source .venv/bin/activate

# 3. Install dependencies
uv pip install --index-strategy unsafe-best-match -r requirements.txt   # if using uv
# pip install -r requirements.txt                                       # if using venv

# 4. OpenSTL (SimVP, PredRNN architectures) is a PINNED git submodule (commit eecf8a3)
git submodule update --init          # fetches OpenSTL/ at the exact pinned commit
uv pip install -e OpenSTL            # if using uv
# pip install -e OpenSTL             # if using venv
# (clone with --recurse-submodules to fetch it during git clone)

# 5. Verify
python -c "from openstl.models import SimVP_Model; print('OK')"
```

### Configure paths

Edit `paths.py` to set the three root directories for your system:

```python
# paths.py — edit these defaults to match your local setup

DATA_ROOT   = Path("./dataset")          # 108 MP4 files (706x706)
MODEL_ROOT  = Path("./outputs/models")   # SimVP_TAU / PredRNN checkpoints
OUTPUT_ROOT = Path("./outputs")          # all analysis outputs
```

All other paths (segmentation, intermediates, figures, etc.) are derived from
these three roots automatically. See `paths.py` for the full list.

## Dataset

108 organoid videos (706x706, 120 frames each, 0.2 fps)

## Repository Structure

```
├── visualize_organoid_videos.py           # Visualize raw organoid videos
├── paths.py                               # Centralized path configuration
├── losses.py                              # Loss functions (MSE, SSIM) for training video prediction model
├── video_utils.py                         # Video I/O utilities
├── data_assembly_utils.py                 # Data assembly helpers
│
├── automated_method/                      # ML-based (video) prediction pipeline
│   ├── data_split.py                      # Data split utilities
│   ├── create_datasplit.py                # Create train/val/test splits
│   ├── models.py                          # SimVP_TAU model architecture
│   ├── dataset.py                         # OpenSTL dataset wrapper
│   ├── video_dataset.py                   # Video sequence loader
│   ├── config.py                          # Training & path configuration
│   ├── train_model.py                     # Train models (-m SimVP_TAU/PredRNN, -s seed)
│   ├── test_model.py                      # Extract per-organoid test metrics (GPU)
│   ├── extract_prediction_surprise.py     # Compute motion residuals & spatial maps (GPU)
│   ├── rank_by_prediction_surprise.py     # Rank organoids by prediction error (CPU)
│   ├── cache_videos_to_pt.py              # Cache videos as .pt tensors
│   ├── checkpoint_manager.py              # Save/load model checkpoints
│   ├── metrics_logger.py                  # Training metrics logging
│   ├── baseline_manager.py                # Baseline loss management
│   └── baseline_control_models_losses.py  # Copy-last, optical flow baselines
│
├── classical_pipeline/                    # Classical centroid-tracking pipeline
│   ├── autotune_segmentation.py           # Auto-tune per-organoid segmentation
│   ├── visualize_grid_centroids.py        # Grid of centroid trajectories
│   ├── extract_motility_features.py       # Centroid-derived motility descriptors
│   ├── find_most_reliable_classical_discriminator.py  # Rank descriptors by F-stat
│   ├── rank_by_best_motility_descriptor.py            # Rank organoids by best descriptor
│   └── all_segmentation_params.json       # Tuned segmentation parameters
│
├── analysis/                              # Cross-method analysis & visualization
│   ├── analyze_test_performance_simVP.py  # SimVP_TAU performance across seeds
│   ├── analyze_test_performance_predRNN.py # PredRNN performance across seeds
│   ├── compare_rankings.py                # Compare model vs classical rankings
│   ├── visualize_spatial_error_map.py     # Per-pixel prediction error heatmaps
│   ├── visualize_flow_fields.py           # Optical flow visualization
│   ├── analyze_relative_performance_temporal.py  # Temporal performance profiles
│   ├── analyze_residualized_complexity.py        # Velocity-residualized complexity
│   ├── analyze_mean_motion_baseline.py           # Mean-motion baseline falsification
│   ├── generate_fig_methods.py            # Fig 1/2: method & training schematic panels
│   ├── build_fig1_grid.py                 # Fig 1A: organoid video grid
│   ├── generate_fig3_results.py           # Fig 3: F-stats, baselines, rank scatter (ρ=0.90)
│   ├── generate_fig4_speed_advantage.py   # Fig 4: copy-last advantage vs speed
│   ├── generate_fig5_flow.py              # Fig 5: model vs optical-flow vectors (GPU)
│   ├── generate_fig6a_size_scatter.py     # Fig 6A: size vs surprise (ρ=0.09 n.s.)
│   ├── generate_fig6b_spatial_error.py    # Fig 6B: spatial error maps
│   └── compare_architectures.py           # Suppl: SimVP-TAU vs PredRNN (ρ=0.94/0.98)
│
└── slurm/                                 # HPC job submission
    └── submit_openstl_jobs.py
```

## Running the full pipeline

Download the data first (or `bash stage_cache.sh` on the original machine):
```bash
python download_data.py     # → ./dataset (108 MP4s) + ./outputs (weights, cached artifacts)
```
> To **only regenerate the paper's figures** from the downloaded data, skip to
> [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The steps below run everything from scratch.

GPU steps must run via `srun` (never the login node), e.g.
`srun -p gpu-debug --gres=gpu:1 --cpus-per-task=4 --mem=64G bash -lc '<cmd>'`.

### A. Automated method — video prediction → prediction surprise

| # | Step | Command | GPU |
|---|---|---|---|
| A1 | Data split (10 covering splits; committed under `datainfo/`, regen optional) | `python automated_method/create_datasplit.py --covering --n-seeds 10 --master-seed 42` | no |
| A2 | Train (per seed; both architectures) | `python automated_method/train_model.py --method SimVP_TAU --seed <s> -K 6 -N 6 --residual` &nbsp;(also `--method PredRNN`) | yes |
| A3 | Test / per-organoid metrics + baselines | `python automated_method/test_model.py --residual --seeds 1 … 10 [--method PredRNN]` | yes |
| A4 | Prediction surprise (Eq 3/4) | `python automated_method/extract_prediction_surprise.py --residual [--method PredRNN]` | yes |
| A5 | Rank by surprise → `per_organoid_summary.csv` | `python automated_method/rank_by_prediction_surprise.py` | no |

`--debug` runs a fast smoke (few epochs, 1 video). Smoke-test both architectures first:
`python tests/smoke_train.py`. Batch submission: `python slurm/submit_openstl_jobs.py --seeds 1 … 10`.

### C. Classical method — segmentation → tracking → features → discriminator

| # | Step | Command | GPU |
|---|---|---|---|
| C1 | Auto-tune segmentation + centroids | `python classical_pipeline/autotune_segmentation.py --all` | no |
| C2 | Motility features + `centroid_trajectories.npz` | `python classical_pipeline/extract_motility_features.py` | no |
| C3 | Most-reliable discriminator (window F-stat → mean speed F=576) | `python classical_pipeline/find_most_reliable_classical_discriminator.py` | no |
| C4 | Rank by best descriptor (mean speed) | `python classical_pipeline/rank_by_best_motility_descriptor.py` | no |

### Cross-method analysis (CPU)
```bash
python analysis/compare_rankings.py          # classical vs prediction-surprise ranking (ρ=0.90)
python analysis/compare_architectures.py     # SimVP-TAU+R vs PredRNN+R (ρ=0.94 raw / 0.98 delta)
```

### Figures
See **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)** — regenerates all main (Fig 1–6) and supplementary
figures from the artifacts, with the expected value for each.

## Data Flow

```
Raw videos (108 MP4s)
 ├── classical_pipeline/autotune_segmentation.py ──► segmentation params + centroids
 ├── classical_pipeline/extract_motility_features.py ──► motility_descriptors.csv
 │                                                       centroid_trajectories.npz
 ├── automated_method/train_model.py (×10 seeds) ──► trained models
 ├── automated_method/test_model.py ──► per_organoid_test_results_residual.csv
 └── automated_method/extract_prediction_surprise.py ──► per_organoid_residuals/*.npz
      │                                                   spatial_residual_maps/*.npy
      └── automated_method/rank_by_prediction_surprise.py ──► per_organoid_summary.csv
           ├── analysis/compare_rankings.py ──► ranking comparison report
           └── analysis/generate_fig{3_results,4_speed_advantage,5_flow,6_size_spatial}.py ──► final figures (PNG + PDF)
```

## Interactive app

`app/streamlit_app.py` is an interactive explorer (no GPU at serve time — reads precomputed
artifacts). Pick a held-out organoid to see its raw frames, spatial prediction-error map,
per-window motion loss vs the copy-last/optical-flow baselines (Fig 2D), and its position on the
surprise-vs-mean-speed ranking (Fig 3C).

```bash
python download_data.py            # or: bash stage_cache.sh   (populates ./outputs + ./dataset)
streamlit run app/streamlit_app.py
# headless data-layer check (no streamlit/browser): python app/streamlit_app.py --selftest
```
On a cluster, forward the port: `ssh -L 8501:localhost:8501 <host>` then open `localhost:8501`.

## SLURM

`slurm/` contains HPC job submission scripts. Set `$REPO_DIR` and `$VENV_DIR`
for your cluster.

## License

MIT
