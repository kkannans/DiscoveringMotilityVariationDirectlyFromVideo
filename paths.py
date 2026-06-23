"""paths.py — Centralized path configuration for all scripts.

Edit the three root paths below to match your local setup.
All other paths are derived automatically.
"""
from pathlib import Path

# ── Edit these to match your system ──────────────────────────────────────
DATA_ROOT   = Path("./dataset")          # 108 MP4 files (128x128, 120 frames each)
MODEL_ROOT  = Path("./outputs/models")   # trained model checkpoints
OUTPUT_ROOT = Path("./outputs")          # all analysis outputs

CLASSICAL_DIR = OUTPUT_ROOT / "classical_descriptors"
INTERMEDIATES_DIR = OUTPUT_ROOT / "intermediates"
FIGURES_DIR = OUTPUT_ROOT / "figures"
SUPPLEMENTARY_DIR = OUTPUT_ROOT / "supplementary"
SEGMENTATION_DIR = CLASSICAL_DIR / "segmentation" / "auto_tune_results"
DATAINFO_DIR = OUTPUT_ROOT / "datainfo"
BASELINE_CACHE_DIR = OUTPUT_ROOT / "baseline_cache"
SLURM_LOGS_DIR = OUTPUT_ROOT / "slurm_logs"
