"""
submit_openstl_jobs.py — Submit OpenSTL benchmark jobs to SLURM.

Usage:
    python submit_openstl_jobs.py --seed 1                    # all methods
    python submit_openstl_jobs.py --seed 1 --methods SimVP    # single method
    python submit_openstl_jobs.py --seeds 1 2 3 --methods all # multi-seed
    python submit_openstl_jobs.py --dry-run                   # write scripts only
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
import paths

EXPERIMENT_NAME = "organoid_motility"

METHODS = [
    "ConvLSTM", "PredRNN", "PredRNNpp", "MIM", "E3DLSTM",
    "PhyDNet", "SimVP", "SimVP_TAU", "MAU",
]

# Recurrent models that patch-reshape and need smaller batches to avoid OOM
HEAVY_METHODS = {"PredRNN", "PredRNNpp", "MIM", "E3DLSTM"}
HEAVY_BATCH_SIZE = 8   # for heavy recurrent models
DEFAULT_BATCH_SIZE = 16  # for SimVP, ConvLSTM, MAU, PhyDNet, etc.

NODE_EXCLUDE = os.environ.get("SLURM_EXCLUDE_NODES", "")
SUBMIT_PARTITIONS = [
    (os.environ.get("SLURM_PARTITION", "gpu"), []),
]

SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_file}.out
#SBATCH --error={log_file}.err
#SBATCH --time=8:00:00
#SBATCH --mem=75G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# ── Self-cancel dedup: if another partition already finished this run, exit ──
OUTPUT_DIR="{output_dir}"
if [ -f "$OUTPUT_DIR/summary.json" ]; then
    echo "[DEDUP] $OUTPUT_DIR/summary.json already exists — run completed by another partition."
    echo "[DEDUP] Cancelling this duplicate job (SLURM_JOB_ID=$SLURM_JOB_ID)."
    scancel "$SLURM_JOB_ID"
    exit 0
fi

echo "=========================================="
echo "OpenSTL Benchmark: {method}"
echo "Seed: {seed}, K={K}, N={N}, S={stride}, model_type={model_type}"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

cd {project_dir}
source $VENV_DIR/bin/activate

python automated_method/train_model.py \\
    --method {method} \\
    --seed {seed} \\
    -K {K} -N {N} \\
    --stride {stride} \\
    --epochs {epochs} \\
    --batch-size {batch_size} \\
    --lr {lr} \\
    --model-type {model_type}

echo "=========================================="
echo "Completed: {method} seed={seed} K={K} N={N} S={stride}"
echo "End time: $(date)"
echo "=========================================="
"""


def create_scripts(methods, seeds, K, N, stride, epochs, batch_size, lr,
                   model_type, dry_run):
    logs_dir = paths.SLURM_LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)

    scripts = []
    for seed in seeds:
        for method in methods:
            job_name = f"openstl_{method}_s{seed}_K{K}_N{N}_S{stride}"
            if model_type != "plain":
                job_name += f"_{model_type}"
            log_file = logs_dir / job_name
            script_path = logs_dir / f"{job_name}.sh"

            # Use smaller batch for heavy recurrent models to avoid OOM
            effective_batch = HEAVY_BATCH_SIZE if method in HEAVY_METHODS else batch_size

            # Output dir must match get_output_dir() in train_openstl.py
            base = f"{method}_K{K}_N{N}_seed{seed}"
            if model_type != "plain":
                base += f"_{model_type}"
            output_dir = paths.MODEL_ROOT / base

            content = SCRIPT_TEMPLATE.format(
                job_name=job_name,
                log_file=str(log_file),
                output_dir=str(output_dir),
                method=method,
                seed=seed,
                K=K, N=N, stride=stride,
                epochs=epochs,
                batch_size=effective_batch,
                lr=lr,
                model_type=model_type,
                project_dir=str(PROJECT_DIR),
            )

            with open(script_path, "w") as f:
                f.write(content)
            os.chmod(script_path, 0o755)
            scripts.append(script_path)

            if dry_run:
                print(f"[DRY RUN] {script_path}")
            else:
                print(f"Created: {script_path}")

    return scripts


def submit_scripts(scripts, dry_run, max_jobs=1000):
    submitted = 0
    for script in scripts:
        for partition, extra_args in SUBMIT_PARTITIONS:
            if submitted >= max_jobs:
                print(f"[LIMIT] Reached max_jobs={max_jobs}, stopping submission.")
                return
            if dry_run:
                print(f"[DRY RUN] sbatch --partition {partition} {script}")
                submitted += 1
            else:
                cmd = ["sbatch", "--parsable", "--partition", partition] + extra_args + [str(script)]
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    job_id = result.stdout.strip()
                    print(f"  {script.stem} -> {partition}: job {job_id}")
                    submitted += 1
                except subprocess.CalledProcessError as e:
                    print(f"  Warning: {partition} failed: {e.stderr.strip()}")
    print(f"Total submissions: {submitted}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["all"],
                   choices=METHODS + ["all"])
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("-K", type=int, default=10)
    p.add_argument("-N", type=int, default=2)
    p.add_argument("-S", "--stride", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--max-jobs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--model-type", type=str, default="plain",
                   choices=["plain", "residual"])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    methods = METHODS if "all" in args.methods else args.methods

    scripts = create_scripts(
        methods, args.seeds, args.K, args.N, args.stride,
        args.epochs, args.batch_size, args.lr,
        args.model_type,
        args.dry_run,
    )

    if not args.dry_run:
        submit_scripts(scripts, args.dry_run, max_jobs=args.max_jobs)

    print(f"\nTotal: {len(scripts)} jobs")
    print(f"Monitor: squeue -u $USER | grep openstl")


if __name__ == "__main__":
    main()
