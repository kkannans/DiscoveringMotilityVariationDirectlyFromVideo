# Reproducibility checklist — regenerate the paper's figures

After downloading the data, this regenerates every figure in *Discovering organoid motility
variation directly from video* (ALife 2026) and prints the value to check against the paper.

- To run the **full pipeline from scratch** (datasplit → train → test → analysis, both methods),
  see [SETUP.md](SETUP.md).

Every command below has been run and verified against the cached artifacts.

---

## Setup

```bash
cd <repo>
PY=python      # your Python 3.11 interpreter (venv with requirements.txt + OpenSTL)
GPU(){ srun -p gpu-debug --time=0:30:00 --gres=gpu:1 --cpus-per-task=4 --mem=64G bash -lc "$*"; }
CPU(){ srun -p short     --time=0:10:00              --cpus-per-task=2 --mem=8G  bash -lc "$*"; }

python download_data.py        # dataset + weights + cached artifacts → ./dataset, ./outputs
# (on this machine you can instead use: bash stage_cache.sh)
```
Figures are written to `outputs/figures/`.

---

## 1. Main figures (paper Fig 1–6)

> Figs 1–2 schematics were drawn manually by the authors; the scripts below generate their
> data panels (organoid grid, per-window loss bar).

| Fig | Command | Expected |
|---|---|---|
| **1** | `CPU "$PY analysis/build_fig1_grid.py"` ; `CPU "$PY analysis/generate_fig_methods.py"` | organoid grid + method schematic panels |
| **2** | `CPU "$PY analysis/generate_fig_methods.py"` | per-window motion-loss bar (Eq 3/4) |
| **3** | `CPU "$PY classical_pipeline/find_most_reliable_classical_discriminator.py"` then `CPU "$PY analysis/generate_fig3_results.py"` | **mean speed F=576**, baselines (model beats both), **ρ=0.90, τ=0.71** |
| **4** | `CPU "$PY analysis/generate_fig4_speed_advantage.py"` | model-vs-copy-last crossover **~0.13 px/f** |
| **5** | `GPU "$PY analysis/generate_fig5_flow.py"` | GT vs model vs optical-flow vectors (GPU: model inference) |
| **6** | `CPU "$PY analysis/generate_fig6a_size_scatter.py"` ; `CPU "$PY analysis/generate_fig6b_spatial_error.py"` | **size ρ=0.09 (n.s.)** + spatial error maps |

Fig 3 is the main result. `generate_fig3_results.py` reads `f_statistics.csv` produced by the
classical discriminator step, so run that first.

---

## 2. Supplementary

| Item | Command | Expected |
|---|---|---|
| Architecture robustness | `CPU "$PY analysis/compare_architectures.py"` | SimVP-TAU+R vs PredRNN+R: **raw ρ=0.94, delta-Sₒ ρ=0.98** (paper 0.938) |
| Cross-method ranking detail | `CPU "$PY analysis/compare_rankings.py"` | ρ=0.897, τ=0.709 |
| Supplementary video (108-organoid grid ± segmentation) | `CPU "$PY make_grid_videos.py"` | `outputs/supplementary/dataset.mp4` |
| Temporal performance profile | `CPU "$PY analysis/analyze_relative_performance_temporal.py"` | — |
| Velocity-residualized complexity | `CPU "$PY analysis/analyze_residualized_complexity.py"` | — |
| Mean-motion baseline falsification | `CPU "$PY analysis/analyze_mean_motion_baseline.py"` | — |

---

## Results to check (summary)

| F=576 (mean speed) | model beats copy-last & optical-flow | ρ=0.90 / τ=0.71 | crossover 0.13 | size ρ=0.09 n.s. | arch ρ=0.94/0.98 |
|---|---|---|---|---|---|

Unit tests for the window descriptors: `CPU "$PY analysis/test_window_descriptors.py"`.
