"""
CheckpointManager: load/save best, latest, and emergency checkpoints.
Used by Trainer; no dependency on global training state.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim


class CheckpointManager:
    """Handles checkpoint paths, loading, and saving (best, latest, emergency). All paths include _epoch{N}."""

    def __init__(
        self,
        checkpoints_folder: Path,
        models_folder: Path,
        dataset_name: str,
        seed: int,
        scheduler_config: Optional[Any] = None,
        num_epochs: Optional[int] = None,
    ) -> None:
        self.checkpoints_folder = Path(checkpoints_folder)
        self.models_folder = Path(models_folder)
        self.dataset_name = dataset_name
        self.seed = seed
        self.scheduler_config = scheduler_config
        self.num_epochs = num_epochs  # Used when recreating CosineAnnealingLR from old checkpoint
        # Single-file paths: only best, latest, and emergency (overwrite; no per-epoch files)
        self._best_path = None
        self._latest_path = None
        self.checkpoint_path = str(self._path_best())
        self.latest_checkpoint_path = str(self._path_latest())

    def _path_best(self) -> Path:
        """Single best checkpoint (overwritten when validation improves)."""
        return self.checkpoints_folder / f"checkpoint_best_{self.dataset_name}_{self.seed}.pth"

    def _path_latest(self) -> Path:
        """Single latest checkpoint (overwritten at end of training for resume)."""
        return self.checkpoints_folder / f"checkpoint_latest_{self.dataset_name}_{self.seed}.pth"

    def _path_emergency(self) -> Path:
        """Single emergency checkpoint (overwritten on interrupt)."""
        return self.checkpoints_folder / f"checkpoint_emergency_{self.dataset_name}_{self.seed}.pth"

    @staticmethod
    def _epoch_from_path(path: Path) -> int:
        """Parse epoch from filename ..._epoch{N}.pth. Returns -1 if not found."""
        stem = path.stem
        if "_epoch" not in stem:
            return -1
        try:
            return int(stem.split("_epoch")[-1])
        except ValueError:
            return -1

    def load_latest(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        device: torch.device,
    ) -> Tuple[int, Optional[Dict[str, Any]], Optional[Any]]:
        """
        Load checkpoint if it exists. Priority: emergency > latest > best.
        Uses single-file paths (overwritten); falls back to legacy _epoch{N} filenames if present.

        Returns:
            (start_epoch, state_dict, new_scheduler_or_None)
            state_dict has: best_val_loss, current_epoch, epochs, train_losses, val_losses,
                train_baseline_losses, val_baseline_losses, learning_rates, epoch_durations, total_training_time.
            If scheduler was recreated (old checkpoint without scheduler_state_dict), new_scheduler is returned.
        """
        checkpoint_path = None
        if self.checkpoints_folder.exists():
            # Priority: emergency > latest > best. Prefer single-file paths (no _epoch in name).
            emergency_single = self._path_emergency()
            if emergency_single.exists():
                checkpoint_path = emergency_single
                print(f"Found emergency checkpoint: {checkpoint_path}", flush=True)
            if checkpoint_path is None:
                latest_single = self._path_latest()
                if latest_single.exists():
                    checkpoint_path = latest_single
                    print(f"Found latest checkpoint: {checkpoint_path}", flush=True)
            if checkpoint_path is None:
                best_single = self._path_best()
                if best_single.exists():
                    checkpoint_path = best_single
                    print(f"Found best checkpoint: {checkpoint_path}", flush=True)
            # Backwards compatibility: old per-epoch filenames
            if checkpoint_path is None:
                emergency_files = list(
                    self.checkpoints_folder.glob(f"checkpoint_emergency_{self.dataset_name}_{self.seed}_epoch*.pth")
                )
                if emergency_files:
                    checkpoint_path = max(emergency_files, key=lambda p: self._epoch_from_path(p))
                    print(f"Found emergency checkpoint (legacy): {checkpoint_path}", flush=True)
            if checkpoint_path is None:
                latest_files = list(
                    self.checkpoints_folder.glob(f"checkpoint_latest_{self.dataset_name}_{self.seed}_epoch*.pth")
                )
                if latest_files:
                    checkpoint_path = max(latest_files, key=lambda p: self._epoch_from_path(p))
                    print(f"Found latest checkpoint (legacy): {checkpoint_path}", flush=True)
            if checkpoint_path is None:
                best_files = list(
                    self.checkpoints_folder.glob(f"checkpoint_best_{self.dataset_name}_{self.seed}_epoch*.pth")
                )
                if best_files:
                    checkpoint_path = max(best_files, key=lambda p: self._epoch_from_path(p))
                    print(f"Found best checkpoint (legacy): {checkpoint_path}", flush=True)
        if checkpoint_path is None:
            print("No checkpoint found. Starting training from scratch (epoch 0).", flush=True)
            return 0, None, None

        if not checkpoint_path.exists():
            return 0, None, None

        print(f"Loading checkpoint from {checkpoint_path}", flush=True)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)

            if "model_state_dict" not in checkpoint:
                print("Warning: Checkpoint does not contain 'model_state_dict'. Attempting to load as model state dict only.", flush=True)
                try:
                    model.load_state_dict(checkpoint)
                    print("Loaded model weights only. Starting from epoch 0.", flush=True)
                    return 0, None, None
                except Exception as e:
                    print(f"Error loading checkpoint: {e}. Starting from epoch 0.", flush=True)
                    return 0, None, None

            saved_epoch = checkpoint.get("epoch", -1)
            print(f"Checkpoint contains epoch: {saved_epoch}", flush=True)

            # Handle checkpoints saved from torch.compile-wrapped models:
            # their state_dict keys are often prefixed with "_orig_mod.".
            state_dict = checkpoint["model_state_dict"]
            if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
                print("Detected _orig_mod.* keys in checkpoint; stripping prefix for load_state_dict.", flush=True)
                state_dict = {
                    k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()
                }

            result = model.load_state_dict(state_dict, strict=False)
            if result.unexpected_keys:
                raise RuntimeError(
                    f"Unexpected keys in checkpoint: {result.unexpected_keys}"
                )
            if result.missing_keys:
                print(
                    f"Note: checkpoint missing keys (newly added params, will use "
                    f"init values): {result.missing_keys}",
                    flush=True,
                )

            if "optimizer_state_dict" not in checkpoint:
                print("Warning: Checkpoint does not contain 'optimizer_state_dict'. Starting with new optimizer.", flush=True)
            else:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            new_scheduler = None
            if "scheduler_state_dict" in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                print(f"Restored scheduler state. Current LR: {scheduler.get_last_lr()[0]:.6e}", flush=True)
            elif scheduler is not None and self.scheduler_config is not None:
                ckpt_epoch = checkpoint["epoch"]
                is_emergency = checkpoint.get("interrupted", False)
                last_completed = ckpt_epoch - 1 if is_emergency else ckpt_epoch
                print(f"Recreating scheduler (old format). Last completed epoch: {last_completed}", flush=True)
                sched_config = self.scheduler_config
                if getattr(sched_config, "scheduler", None) == "cosine_annealing":
                    T_max = self.num_epochs or 600
                    new_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=T_max,
                        eta_min=getattr(sched_config, "lr_min", 1e-6),
                        last_epoch=last_completed,
                    )
                elif getattr(sched_config, "scheduler", None) == "multi_step_lr":
                    new_scheduler = optim.lr_scheduler.MultiStepLR(
                        optimizer,
                        milestones=getattr(sched_config, "lr_schedule", [20, 50, 100]),
                        gamma=getattr(sched_config, "lr_gamma", 0.5),
                        last_epoch=last_completed,
                    )
                elif getattr(sched_config, "scheduler", None) == "cosine_annealing_warm_restarts":
                    # Safer resume: restart peak LR = min(last_lr * multiplier, lr_max); cap at epoch-0 LR (e.g. 1e-3)
                    last_lr = optimizer.param_groups[0]["lr"]
                    mult = getattr(sched_config, "warm_restart_resume_lr_multiplier", 10.0)
                    lr_max = getattr(sched_config, "warm_restart_resume_lr_max", 1e-3)
                    restart_lr = min(last_lr * mult, lr_max)
                    for pg in optimizer.param_groups:
                        pg["lr"] = restart_lr
                    print(
                        f"Warm restarts resume: last_lr={last_lr:.6e} -> restart peak LR={restart_lr:.6e} (mult={mult}, max={lr_max:.0e})",
                        flush=True,
                    )
                    new_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer,
                        T_0=getattr(sched_config, "T_0", 100),
                        T_mult=getattr(sched_config, "T_mult", 2),
                        eta_min=getattr(sched_config, "eta_min", 1e-6),
                        last_epoch=last_completed,
                    )
                    # Step once so the first resumed epoch (start_epoch) runs with restart peak LR, not eta_min
                    new_scheduler.step()
                else:
                    new_scheduler = optim.lr_scheduler.MultiStepLR(
                        optimizer,
                        milestones=getattr(sched_config, "lr_schedule", [20, 50, 100]),
                        gamma=getattr(sched_config, "lr_gamma", 0.5),
                        last_epoch=last_completed,
                    )
                print(f"Current learning rate: {new_scheduler.get_last_lr()[0]:.6e}", flush=True)

            is_emergency = checkpoint.get("interrupted", False)
            ckpt_epoch = checkpoint["epoch"]
            if is_emergency:
                start_epoch = ckpt_epoch
                print(f"Emergency checkpoint - resuming from epoch {start_epoch}", flush=True)
            else:
                start_epoch = ckpt_epoch + 1
                print(f"Regular checkpoint - resuming from epoch {start_epoch}", flush=True)

            state_dict = {
                "best_val_loss": checkpoint.get("loss", float("inf")),
                "current_epoch": ckpt_epoch,
            }
            if "training_history" in checkpoint:
                h = checkpoint["training_history"]
                state_dict["epochs"] = h.get("epochs", [])
                state_dict["train_losses"] = h.get("train_losses", [])
                state_dict["val_losses"] = h.get("val_losses", [])
                state_dict["train_baseline_losses"] = h.get("train_baseline_losses", [])
                state_dict["val_baseline_losses"] = h.get("val_baseline_losses", [])
                state_dict["learning_rates"] = h.get("learning_rates", [])
                state_dict["epoch_durations"] = h.get("epoch_durations", [])
                state_dict["total_training_time"] = h.get("total_training_time", 0.0)
                print(f"Restored training history: {len(state_dict['epochs'])} epochs", flush=True)
            else:
                print("No training history in checkpoint.", flush=True)

            return start_epoch, state_dict, new_scheduler

        except Exception as e:
            print(f"Error loading checkpoint: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return 0, None, None

    def save_best(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        val_loss: float,
        training_history: Dict[str, Any],
    ) -> None:
        """Save best checkpoint and best model weights (single file, overwritten when val improves)."""
        path = self._path_best()
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": val_loss,
            "training_history": training_history,
        }
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(checkpoint, path)
        self.checkpoint_path = str(path)
        model_path = self.models_folder / f"best_model_{self.dataset_name}_{self.seed}.pth"
        torch.save(model.state_dict(), model_path)

    def save_latest(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        val_loss: float,
        training_history: Dict[str, Any],
    ) -> None:
        """Save latest checkpoint (single file, for resume; overwritten at end of training)."""
        path = self._path_latest()
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": val_loss,
            "training_history": training_history,
        }
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(checkpoint, path)
        self.latest_checkpoint_path = str(path)

    def save_emergency(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        best_val_loss: float,
        training_history: Dict[str, Any],
    ) -> Optional[str]:
        """Save emergency checkpoint on interrupt (single file). Returns path if saved."""
        try:
            path = self._path_emergency()
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_val_loss,
                "interrupted": True,
                "training_history": training_history,
            }
            if scheduler is not None:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()
            torch.save(checkpoint, path)
            print(f"Emergency checkpoint saved to: {path}", flush=True)
            return str(path)
        except Exception as e:
            print(f"Error saving emergency checkpoint: {e}", flush=True)
            return None


# Standard checkpoint format (full checkpoint):
#   epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict (optional),
#   loss (val_loss), training_history (dict), model_id (optional), model_name (optional),
#   loss_type (optional), interrupted (optional)
# For inference/testing, use load_model_weights_only() to load only model weights.


def load_model_weights_only(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> nn.Module:
    """
    Load model weights from a checkpoint file.
    Handles both full checkpoint dicts (uses model_state_dict) and raw state_dict files.
    Returns the model (loaded in place).
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)
    return model
