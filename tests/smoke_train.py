"""
smoke_train.py — fast end-to-end training smoke test for both architectures.

Runs a tiny `--debug` training (few epochs, 1 video) for SimVP_TAU and PredRNN with the residual
formulation and asserts each writes a `best_model.pth`. Catches breakage in the data loader, model
factory, residual loss, and checkpointing before launching full multi-hour training jobs.

GPU required. Run via srun on gpu-debug:
    srun -p gpu-debug --time=0:30:00 --gres=gpu:1 --cpus-per-task=4 --mem=64G \
        bash -c '/users/k/k/kkannans/alife-2026/.venv/bin/python tests/smoke_train.py'
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import paths

METHODS = ["SimVP_TAU", "PredRNN"]


def main():
    try:
        import torch
        if not torch.cuda.is_available():
            sys.exit("SKIP: no CUDA. Run on a GPU node (srun -p gpu-debug ...).")
    except ImportError:
        sys.exit("SKIP: torch not importable.")

    failures = []
    for method in METHODS:
        print(f"\n=== smoke: {method} ===", flush=True)
        cmd = [sys.executable, str(ROOT / "automated_method" / "train_model.py"),
               "--method", method, "--seed", "1", "-K", "6", "-N", "6",
               "--residual", "--debug"]
        r = subprocess.run(cmd, cwd=ROOT)
        ckpt = paths.MODEL_ROOT / f"{method}_K6_N6_seed1_residual" / "best_model.pth"
        if r.returncode != 0:
            failures.append(f"{method}: train_model.py exited {r.returncode}")
        elif not ckpt.exists():
            failures.append(f"{method}: no checkpoint at {ckpt}")
        else:
            print(f"  PASS: {method} -> {ckpt}")

    if failures:
        print("\nSMOKE FAILED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
