"""

BaselineManager: load/compute/save baseline losses for sequence prediction tasks.
Used by Trainer; compute_all_sequence_baselines is the shared module-level entry point.
"""
import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import ModelConfig, PathConfig
from baseline_control_models_losses import (
    DecodedCopy, DecodedLinearInterpolation, DecodedOpticalFlow,
    compute_sequence_mse_per_sample, compute_all_sequence_baselines,
)

BASELINE_KEYS = ("copy", "black", "linear_interpolation", "optical_flow")
DECODED_BASELINE_KEYS = ("decoded_copy", "decoded_linear_interpolation", "decoded_optical_flow")


def _sequence_motion(sequence: torch.Tensor) -> torch.Tensor:
    if sequence.dim() != 5:
        raise ValueError(f"Expected sequence of shape (B, L, 5, H, W), got {sequence.shape}")
    b, l, c, h, w = sequence.shape
    if l <= 1:
        return torch.zeros(b, device=sequence.device, dtype=sequence.dtype)
    rgb = sequence[:, :, :3, :, :]
    gray = rgb.mean(dim=2)
    diffs = torch.abs(gray[:, 1:, :, :] - gray[:, :-1, :, :])
    return diffs.mean(dim=(1, 2, 3))


def precompute_baseline_losses_and_save(
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    model_config: ModelConfig,
    device: torch.device,
    loss_type: str,
    output_path: Path,
    interrupted_callback: Optional[Callable[[], bool]] = None,
    K: Optional[int] = None,
    N: Optional[int] = None,
    stride: Optional[int] = None,
) -> bool:
    if loss_type != "mse":
        raise ValueError("Sequence baselines only support mse.")

    def _compute_split(loader: DataLoader[Any], desc: str) -> tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        per_sequence: Dict[str, Dict[str, float]] = {}
        totals = {k: 0.0 for k in BASELINE_KEYS}
        count = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                if interrupted_callback and interrupted_callback():
                    return {}, {k: 0.0 for k in BASELINE_KEYS}
                
                # We expect a 6-tuple: (context, target, unused, name, phase, start_idx)
                if len(batch) >= 6 and torch.is_tensor(batch[0]) and torch.is_tensor(batch[1]):
                    context, target = batch[0].to(device), batch[1].to(device)
                    batch_names = batch[3]
                    phases = batch[4]
                    
                    # Use only RGB channels for baselines.
                    c_rgb = context[:, :, :3]
                    t_rgb = target[:, :, :3]
                    
                    # For Sequence-rollout baselines, we want (B, K+N, 5, H, W)
                    # We pad to 5 channels with zeroes so baselines work
                    b, k, _, h, w = context.shape
                    n = target.shape[1]
                    L = k + n
                    
                    seq_rgb = torch.cat([c_rgb, t_rgb], dim=1) # (B, L, 3, H, W)
                    padding = torch.zeros(b, L, 2, h, w, device=device)
                    sequence = torch.cat([seq_rgb, padding], dim=2) # (B, L, 5, H, W)
                    
                    baseline_out = compute_all_sequence_baselines(sequence, model_config, K=k, N=n)
                    motion_vals = _sequence_motion(sequence)
                    
                    for i in range(b):
                        phase = phases[i] if isinstance(phases, (list, tuple)) else phases
                        name = batch_names[i] if isinstance(batch_names, (list, tuple)) else f"{desc}_{count+i}"
                        sid = f"{phase}:{name}"
                        
                        row = {key: float(baseline_out[key][1][i].item()) for key in BASELINE_KEYS}
                        row["motion"] = float(motion_vals[i].item())
                        for key in BASELINE_KEYS:
                            totals[key] += row[key]
                        per_sequence[sid] = row
                        count += 1
                else:
                    if count == 0:
                        print(f"Warning: Unexpected batch format. Length={len(batch)}.", flush=True)

        means = {k: (totals[k] / count if count else 0.0) for k in BASELINE_KEYS}
        return per_sequence, means

    train_per_sequence, train_means = _compute_split(train_loader, "Training baseline losses")
    if interrupted_callback and interrupted_callback():
        return False
    val_per_sequence, val_means = _compute_split(val_loader, "Validation baseline losses")
    if interrupted_callback and interrupted_callback():
        return False

    def _split_pre_post(per_sequence: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        pre_totals = {k: 0.0 for k in BASELINE_KEYS}
        pre_n = 0
        post_totals = {k: 0.0 for k in BASELINE_KEYS}
        post_n = 0
        for sid, bl in per_sequence.items():
            sid_str = str(sid).strip().lower()
            if sid_str.startswith("pre:"):
                for k in BASELINE_KEYS:
                    pre_totals[k] += bl.get(k, 0.0)
                pre_n += 1
            elif sid_str.startswith("post:"):
                for k in BASELINE_KEYS:
                    post_totals[k] += bl.get(k, 0.0)
                post_n += 1
        pre_means = {k: (pre_totals[k] / pre_n if pre_n else 0.0) for k in BASELINE_KEYS}
        post_means = {k: (post_totals[k] / post_n if post_n else 0.0) for k in BASELINE_KEYS}
        return pre_means, post_means

    train_pre, train_post = _split_pre_post(train_per_sequence)
    val_pre, val_post = _split_pre_post(val_per_sequence)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_data = {
        "training_per_sequence": train_per_sequence,
        "validation_per_sequence": val_per_sequence,
        # Legacy keys for backward compatibility with old loaders
        "training_per_quadruplet": train_per_sequence,
        "validation_per_quadruplet": val_per_sequence,
        **{f"training_mean_{k}": np.float32(train_means[k]) for k in BASELINE_KEYS},
        **{f"validation_mean_{k}": np.float32(val_means[k]) for k in BASELINE_KEYS},
        **{f"training_{k}": np.float32(train_means[k]) for k in BASELINE_KEYS},
        **{f"validation_{k}": np.float32(val_means[k]) for k in BASELINE_KEYS},
        **{f"training_mean_{k}_pre": np.float32(train_pre[k]) for k in BASELINE_KEYS},
        **{f"training_mean_{k}_post": np.float32(train_post[k]) for k in BASELINE_KEYS},
        **{f"validation_mean_{k}_pre": np.float32(val_pre[k]) for k in BASELINE_KEYS},
        **{f"validation_mean_{k}_post": np.float32(val_post[k]) for k in BASELINE_KEYS},
        "loss_type": loss_type,
        "K": K,
        "N": N,
        "stride": stride,
        "computed_at": datetime.datetime.now().isoformat(),
        "num_training_sequences": len(train_per_sequence),
        "num_validation_sequences": len(val_per_sequence),
    }
    np.save(output_path, cast(Any, baseline_data))
    print(f"Shared baseline losses saved to {output_path} ({len(train_per_sequence)} train, {len(val_per_sequence)} val)", flush=True)
    return True


class BaselineManager:
    """Load/compute/save baseline losses and compute relative improvements."""

    def __init__(
        self,
        model_config: ModelConfig,
        path_config: PathConfig,
        training_seed: int,
        loss_type: str,
        device: torch.device,
        dir_manager: Any,
        baseline_losses_path: Optional[Path] = None,
        baseline_losses_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_config = model_config
        self.path_config = path_config
        self.training_seed = training_seed
        self.loss_type = loss_type
        self.device = device
        self.dir_manager = dir_manager
        self._baseline_losses_path = Path(baseline_losses_path) if baseline_losses_path else None
        self._baseline_losses_data = baseline_losses_data  # Pre-loaded dict (e.g. from parent process)

        self.training_baseline_losses_per_sequence: Dict[str, Dict[str, float]] = {}
        self.validation_baseline_losses_per_sequence: Dict[str, Dict[str, float]] = {}
        self.training_baseline_losses = {k: 0.0 for k in BASELINE_KEYS}
        self.validation_baseline_losses = {k: 0.0 for k in BASELINE_KEYS}
        self.training_baseline_losses_pre = {k: 0.0 for k in BASELINE_KEYS}
        self.training_baseline_losses_post = {k: 0.0 for k in BASELINE_KEYS}
        self.validation_baseline_losses_pre = {k: 0.0 for k in BASELINE_KEYS}
        self.validation_baseline_losses_post = {k: 0.0 for k in BASELINE_KEYS}
        # Decoded baselines (computed after model is loaded)
        self.training_decoded_baseline_losses: Dict[str, float] = {k: 0.0 for k in DECODED_BASELINE_KEYS}
        self.validation_decoded_baseline_losses: Dict[str, float] = {k: 0.0 for k in DECODED_BASELINE_KEYS}
        self.training_decoded_baseline_losses_pre: Dict[str, float] = {k: 0.0 for k in DECODED_BASELINE_KEYS}
        self.training_decoded_baseline_losses_post: Dict[str, float] = {k: 0.0 for k in DECODED_BASELINE_KEYS}
        self.validation_decoded_baseline_losses_pre: Dict[str, float] = {k: 0.0 for k in DECODED_BASELINE_KEYS}
        self.validation_decoded_baseline_losses_post: Dict[str, float] = {k: 0.0 for k in DECODED_BASELINE_KEYS}

    def ensure_baselines_computed(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        interrupted_callback: Callable[[], bool],
        log_filename: Optional[str] = None,
    ) -> None:
        """Load baselines from file if possible; otherwise compute and save."""
        if self._load_baseline_losses():
            print("Using loaded baseline losses. Skipping computation.", flush=True)
            return
        print("Precomputing per-sequence baseline losses for training and validation datasets...", flush=True)
        shared_path = self.path_config.get_baseline_losses_path(self.training_seed, self.loss_type)
        shared_path.parent.mkdir(parents=True, exist_ok=True)
        completed = precompute_baseline_losses_and_save(
            train_loader,
            val_loader,
            self.model_config,
            self.device,
            self.loss_type,
            shared_path,
            interrupted_callback=interrupted_callback,
            K=getattr(self.path_config, "K", None),
            N=getattr(self.path_config, "N", None),
            stride=getattr(self.path_config, "stride", None),
        )
        if not completed:
            print("Baseline computation interrupted or did not complete.", flush=True)
            return
        self._load_baseline_losses()
        if log_filename:
            with open(log_filename, "a") as f:
                f.write(f"\nper-sequence baseline losses computed and saved to {shared_path}\n")

    def _load_baseline_losses(self) -> bool:
        """Load from pre-loaded data, or from file. Returns True if loaded successfully."""
        try:
            if self._baseline_losses_data is not None:
                data = self._baseline_losses_data
            else:
                shared_path = self.path_config.get_baseline_losses_path(self.training_seed, self.loss_type)
                if self._baseline_losses_path and self._baseline_losses_path.exists():
                    path = self._baseline_losses_path
                elif shared_path.exists():
                    path = shared_path
                elif (self.dir_manager.logs_folder / "baseline_losses.npy").exists():
                    path = self.dir_manager.logs_folder / "baseline_losses.npy"
                else:
                    print(f"Baseline losses file not found at {shared_path}. Will compute new baseline losses.", flush=True)
                    return False
                data = np.load(path, allow_pickle=True).item()
            saved_loss_type = data.get("loss_type", None)
            if saved_loss_type is not None and saved_loss_type != self.loss_type:
                print(f"Warning: Saved loss_type ({saved_loss_type}) differs from current ({self.loss_type}). Will recompute.", flush=True)
                return False

            # Check K/N/stride match (baselines depend on window shape)
            saved_K = data.get("K")
            saved_N = data.get("N")
            saved_stride = data.get("stride")
            expected_K = getattr(self.path_config, "K", None)
            expected_N = getattr(self.path_config, "N", None)
            expected_stride = getattr(self.path_config, "stride", None)

            if saved_K is None and expected_K is not None:
                print("Baseline file missing K/N metadata (old format). Will recompute.", flush=True)
                return False
            if saved_K is not None and expected_K is not None and saved_K != expected_K:
                print(f"Baseline K mismatch (saved={saved_K}, current={expected_K}). Will recompute.", flush=True)
                return False
            if saved_N is not None and expected_N is not None and saved_N != expected_N:
                print(f"Baseline N mismatch (saved={saved_N}, current={expected_N}). Will recompute.", flush=True)
                return False
            if saved_stride is not None and expected_stride is not None and saved_stride != expected_stride:
                print(f"Baseline stride mismatch (saved={saved_stride}, current={expected_stride}). Will recompute.", flush=True)
                return False

            # Support both new ("per_sequence") and legacy ("per_quadruplet") serialized keys
            train_key = "training_per_sequence" if "training_per_sequence" in data else "training_per_quadruplet"
            val_key = "validation_per_sequence" if "validation_per_sequence" in data else "validation_per_quadruplet"
            if train_key in data and val_key in data:
                self.training_baseline_losses_per_sequence = data[train_key]
                self.validation_baseline_losses_per_sequence = data[val_key]
                sample = next(iter(self.training_baseline_losses_per_sequence.values()), {})
                missing = [k for k in BASELINE_KEYS if k not in sample]
                if missing:
                    raise AssertionError(f"Baseline file missing keys: {missing}")
                self.training_baseline_losses = {
                    k: float(data.get(f"training_mean_{k}", data.get(f"training_{k}", 0.0)))
                    for k in BASELINE_KEYS
                }
                self.validation_baseline_losses = {
                    k: float(data.get(f"validation_mean_{k}", data.get(f"validation_{k}", 0.0)))
                    for k in BASELINE_KEYS
                }
                # Load pre/post split (computed by precompute_baseline_losses_and_save)
                has_pre_post = f"validation_mean_{BASELINE_KEYS[0]}_pre" in data
                if has_pre_post:
                    for k in BASELINE_KEYS:
                        self.training_baseline_losses_pre[k] = float(data[f"training_mean_{k}_pre"])
                        self.training_baseline_losses_post[k] = float(data[f"training_mean_{k}_post"])
                        self.validation_baseline_losses_pre[k] = float(data[f"validation_mean_{k}_pre"])
                        self.validation_baseline_losses_post[k] = float(data[f"validation_mean_{k}_post"])
                else:
                    self._compute_pre_post_from_per_quadruplet()
                if self._baseline_losses_data is not None:
                    print("Using baseline losses passed from parent process.", flush=True)
                else:
                    print(f"Loaded baseline losses from {path}", flush=True)
                return True
            print("Legacy baseline format (mean only). Will recompute per-sequence.", flush=True)
            return False
        except Exception as e:
            print(f"Warning: failed to load baseline losses: {e}", flush=True)
            return False

    def _compute_pre_post_from_per_quadruplet(self) -> None:
        """Compute pre/post baseline means from per-sequence dicts (for old files without _pre/_post keys)."""
        for per_quad, pre_dst, post_dst in [
            (self.training_baseline_losses_per_sequence, self.training_baseline_losses_pre, self.training_baseline_losses_post),
            (self.validation_baseline_losses_per_sequence, self.validation_baseline_losses_pre, self.validation_baseline_losses_post),
        ]:
            pre_totals = {k: 0.0 for k in BASELINE_KEYS}
            pre_n = 0
            post_totals = {k: 0.0 for k in BASELINE_KEYS}
            post_n = 0
            for qid, bl in per_quad.items():
                qid_str = str(qid).strip().lower()
                if qid_str.startswith("pre:"):
                    for k in BASELINE_KEYS:
                        pre_totals[k] += bl.get(k, 0.0)
                    pre_n += 1
                elif qid_str.startswith("post:"):
                    for k in BASELINE_KEYS:
                        post_totals[k] += bl.get(k, 0.0)
                    post_n += 1
            for k in BASELINE_KEYS:
                pre_dst[k] = pre_totals[k] / pre_n if pre_n else 0.0
                post_dst[k] = post_totals[k] / post_n if post_n else 0.0

    def save_baseline_losses(self, log_filename: Optional[str] = None) -> None:
        """Save per-sequence and mean baseline losses to dir_manager.logs_folder."""
        try:
            path = self.dir_manager.logs_folder / "baseline_losses.npy"
            data = {
                "training_per_quadruplet": self.training_baseline_losses_per_sequence,
                "validation_per_quadruplet": self.validation_baseline_losses_per_sequence,
                **{f"training_mean_{k}": np.float32(self.training_baseline_losses[k]) for k in BASELINE_KEYS},
                **{f"validation_mean_{k}": np.float32(self.validation_baseline_losses[k]) for k in BASELINE_KEYS},
                **{f"training_{k}": np.float32(self.training_baseline_losses[k]) for k in BASELINE_KEYS},
                **{f"validation_{k}": np.float32(self.validation_baseline_losses[k]) for k in BASELINE_KEYS},
                **{f"training_mean_{k}_pre": np.float32(self.training_baseline_losses_pre[k]) for k in BASELINE_KEYS},
                **{f"training_mean_{k}_post": np.float32(self.training_baseline_losses_post[k]) for k in BASELINE_KEYS},
                **{f"validation_mean_{k}_pre": np.float32(self.validation_baseline_losses_pre[k]) for k in BASELINE_KEYS},
                **{f"validation_mean_{k}_post": np.float32(self.validation_baseline_losses_post[k]) for k in BASELINE_KEYS},
                "loss_type": self.loss_type,
                "K": getattr(self.path_config, "K", None),
                "N": getattr(self.path_config, "N", None),
                "stride": getattr(self.path_config, "stride", None),
                "computed_at": datetime.datetime.now().isoformat(),
                "num_training_sequences": len(self.training_baseline_losses_per_sequence),
                "num_validation_sequences": len(self.validation_baseline_losses_per_sequence),
            }
            np.save(path, cast(Any, data))
            print(f"Baseline losses saved to: {path}", flush=True)
            if log_filename:
                with open(log_filename, "a") as f:
                    f.write(f"\nBaseline losses saved to: {path}\n")
        except Exception as e:
            print(f"Warning: failed to save baseline losses: {e}", flush=True)

    def get_training_baselines(self) -> Dict[str, float]:
        return dict(self.training_baseline_losses)

    def get_validation_baselines(self) -> Dict[str, float]:
        return dict(self.validation_baseline_losses)

    def get_training_baselines_pre(self) -> Dict[str, float]:
        return dict(self.training_baseline_losses_pre)

    def get_training_baselines_post(self) -> Dict[str, float]:
        return dict(self.training_baseline_losses_post)

    def get_validation_baselines_pre(self) -> Dict[str, float]:
        return dict(self.validation_baseline_losses_pre)

    def get_validation_baselines_post(self) -> Dict[str, float]:
        return dict(self.validation_baseline_losses_post)

    def get_validation_decoded_baselines(self) -> Dict[str, float]:
        return dict(self.validation_decoded_baseline_losses)

    def get_validation_decoded_baselines_pre(self) -> Dict[str, float]:
        return dict(self.validation_decoded_baseline_losses_pre)

    def get_validation_decoded_baselines_post(self) -> Dict[str, float]:
        return dict(self.validation_decoded_baseline_losses_post)

    def get_training_decoded_baselines(self) -> Dict[str, float]:
        return dict(self.training_decoded_baseline_losses)

    def get_training_decoded_baselines_pre(self) -> Dict[str, float]:
        return dict(self.training_decoded_baseline_losses_pre)

    def get_training_decoded_baselines_post(self) -> Dict[str, float]:
        return dict(self.training_decoded_baseline_losses_post)

    def _load_decoded_baselines(self) -> bool:
        """Try to load decoded baselines from the per-seed .npy file. Returns True if loaded."""
        try:
            path = self.path_config.get_decoded_baseline_losses_path(self.training_seed, self.loss_type)
            if not path.exists():
                return False
            data = np.load(path, allow_pickle=True).item()
            # Validate K/N/stride match
            if data.get("K") != getattr(self.path_config, "K", None):
                print(f"Decoded baseline K mismatch. Will recompute.", flush=True)
                return False
            if data.get("N") != getattr(self.path_config, "N", None):
                print(f"Decoded baseline N mismatch. Will recompute.", flush=True)
                return False
            if data.get("stride") != getattr(self.path_config, "stride", None):
                print(f"Decoded baseline stride mismatch. Will recompute.", flush=True)
                return False
            self.training_decoded_baseline_losses = data["training_means"]
            self.training_decoded_baseline_losses_pre = data["training_pre"]
            self.training_decoded_baseline_losses_post = data["training_post"]
            self.validation_decoded_baseline_losses = data["validation_means"]
            self.validation_decoded_baseline_losses_pre = data["validation_pre"]
            self.validation_decoded_baseline_losses_post = data["validation_post"]
            print(f"Loaded decoded baselines from {path}", flush=True)
            return True
        except Exception as e:
            print(f"Warning: failed to load decoded baselines: {e}", flush=True)
            return False

    def _save_decoded_baselines(self) -> None:
        """Save decoded baselines to the per-seed .npy file."""
        try:
            path = self.path_config.get_decoded_baseline_losses_path(self.training_seed, self.loss_type)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "training_means": self.training_decoded_baseline_losses,
                "training_pre": self.training_decoded_baseline_losses_pre,
                "training_post": self.training_decoded_baseline_losses_post,
                "validation_means": self.validation_decoded_baseline_losses,
                "validation_pre": self.validation_decoded_baseline_losses_pre,
                "validation_post": self.validation_decoded_baseline_losses_post,
                "loss_type": self.loss_type,
                "K": getattr(self.path_config, "K", None),
                "N": getattr(self.path_config, "N", None),
                "stride": getattr(self.path_config, "stride", None),
                "computed_at": datetime.datetime.now().isoformat(),
            }
            np.save(path, cast(Any, data))
            print(f"Decoded baselines saved to {path}", flush=True)
        except Exception as e:
            print(f"Warning: failed to save decoded baselines: {e}", flush=True)

    def compute_decoded_baselines(
        self,
        model: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """Compute decoded baseline losses using the model's encoder/decoder.

        Loads from per-seed cache if available; otherwise computes and saves.
        Must be called after the model (with frozen encoder/decoder) is loaded.
        """
        if not (hasattr(model, "encoder") and hasattr(model, "decoder")):
            print("Model has no encoder/decoder — skipping decoded baseline computation.", flush=True)
            return

        if self._load_decoded_baselines():
            return

        decoded_baselines = {
            "decoded_copy": DecodedCopy(self.model_config, model),
            "decoded_linear_interpolation": DecodedLinearInterpolation(self.model_config, model),
            "decoded_optical_flow": DecodedOpticalFlow(self.model_config, model),
        }

        def _run_split(loader: DataLoader, desc: str):
            totals = {k: 0.0 for k in DECODED_BASELINE_KEYS}
            pre_totals = {k: 0.0 for k in DECODED_BASELINE_KEYS}
            post_totals = {k: 0.0 for k in DECODED_BASELINE_KEYS}
            count = 0
            pre_n = 0
            post_n = 0
            with torch.no_grad():
                for batch in tqdm(loader, desc=desc):
                    if len(batch) < 6:
                        continue
                    context, target = batch[0].to(self.device), batch[1].to(self.device)
                    phases = batch[4]
                    c_rgb = context[:, :, :3]
                    t_rgb = target[:, :, :3]
                    b, k, _, h, w = context.shape
                    n = target.shape[1]
                    L = k + n
                    seq_rgb = torch.cat([c_rgb, t_rgb], dim=1)
                    padding = torch.zeros(b, L, 2, h, w, device=self.device)
                    sequence = torch.cat([seq_rgb, padding], dim=2)
                    # Decode GT target frames through encoder->decoder so the
                    # comparison is in the same reconstructed space as the ConvLSTM.
                    raw_target = sequence[:, k:k+n, :3]
                    decoded_gt_frames = []
                    for t_idx in range(raw_target.shape[1]):
                        z = model.encoder(raw_target[:, t_idx])
                        decoded_gt_frames.append(model.decoder(z).clamp(0, 1))
                    gt_target = torch.stack(decoded_gt_frames, dim=1)

                    for name, baseline in decoded_baselines.items():
                        try:
                            preds = baseline.predict_sequence(sequence, k, n)
                            losses = compute_sequence_mse_per_sample(preds, gt_target)
                        except Exception as e:
                            print(f"Warning: decoded baseline '{name}' failed: {e}", flush=True)
                            continue
                        for i in range(b):
                            loss_val = float(losses[i].item())
                            totals[name] += loss_val
                            phase = phases[i] if isinstance(phases, (list, tuple)) else phases
                            phase_str = str(phase).strip().lower()
                            if phase_str == "pre":
                                pre_totals[name] += loss_val
                            elif phase_str == "post":
                                post_totals[name] += loss_val
                    for i in range(b):
                        phase = phases[i] if isinstance(phases, (list, tuple)) else phases
                        phase_str = str(phase).strip().lower()
                        count += 1
                        if phase_str == "pre":
                            pre_n += 1
                        elif phase_str == "post":
                            post_n += 1

            means = {k: (totals[k] / count if count else 0.0) for k in DECODED_BASELINE_KEYS}
            pre_means = {k: (pre_totals[k] / pre_n if pre_n else 0.0) for k in DECODED_BASELINE_KEYS}
            post_means = {k: (post_totals[k] / post_n if post_n else 0.0) for k in DECODED_BASELINE_KEYS}
            return means, pre_means, post_means

        print("Computing decoded baselines (train)...", flush=True)
        train_means, train_pre, train_post = _run_split(train_loader, "Decoded baselines (train)")
        print("Computing decoded baselines (val)...", flush=True)
        val_means, val_pre, val_post = _run_split(val_loader, "Decoded baselines (val)")

        self.training_decoded_baseline_losses = train_means
        self.training_decoded_baseline_losses_pre = train_pre
        self.training_decoded_baseline_losses_post = train_post
        self.validation_decoded_baseline_losses = val_means
        self.validation_decoded_baseline_losses_pre = val_pre
        self.validation_decoded_baseline_losses_post = val_post

        self._save_decoded_baselines()
        print(f"Decoded baselines — train: {train_means}, val: {val_means}", flush=True)

    def get_per_sequence_baselines(self, split: str) -> Dict[str, Dict[str, float]]:
        if split == "train":
            return dict(self.training_baseline_losses_per_sequence)
        if split == "val":
            return dict(self.validation_baseline_losses_per_sequence)
        raise ValueError(f"split must be 'train' or 'val', got {split}")

    @staticmethod
    def compute_relative_improvement(model_loss: float, baseline_loss: float, eps: float = 1e-10) -> float:
        """Relative improvement of model over baseline: (baseline - model) / baseline. >0 = model better."""
        if baseline_loss < eps:
            return 0.0 if model_loss < eps else -model_loss
        return (baseline_loss - model_loss) / baseline_loss

    def compute_relative_improvements(
        self,
        filepaths: List[Any],
        model_losses: List[float],
    ) -> Dict[str, Any]:
        """
        Compute per-sequence relative improvements vs copy, linear_interpolation, optical_flow.
        filepaths must be the same ids as saved in validation_per_sequence (e.g. "pre:batch-xxx_idx_y"
        / "post:..." from VideoSequenceDataset).
        Returns dict with keys: motions, relative_improvements (dict of baseline_type -> array), sequence_ids.
        """
        relative_improvements: Dict[str, Any] = {
            "copy": [],
            "linear_interpolation": [],
            "optical_flow": [],
        }
        motions_list: List[float] = []
        sequence_ids: List[Any] = []
        for filepath, model_loss in zip(filepaths, model_losses):
            qid = str(filepath[0] if isinstance(filepath, (list, tuple)) and filepath else filepath)
            if qid not in self.validation_baseline_losses_per_sequence:
                continue
            baseline_data = self.validation_baseline_losses_per_sequence[qid]
            motion = baseline_data.get("motion", 0.0)
            for baseline_type in ["copy", "linear_interpolation", "optical_flow"]:
                bl = baseline_data.get(baseline_type, 0.0)
                rel = self.compute_relative_improvement(model_loss, bl)
                relative_improvements[baseline_type].append(rel)
            motions_list.append(motion)
            sequence_ids.append(filepath)
        motions = np.array(motions_list) if motions_list else np.array([])
        for k in relative_improvements:
            relative_improvements[k] = np.array(relative_improvements[k]) if relative_improvements[k] else np.array([])
        return {
            "motions": motions,
            "relative_improvements": relative_improvements,
            "sequence_ids": sequence_ids,
        }
