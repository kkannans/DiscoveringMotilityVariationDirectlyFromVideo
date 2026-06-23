"""
losses.py

Loss functions for sequence-based frame prediction (teacher-forced rollout).
operates on frame-space tensors (B, N, C, H, W).

Supported loss_type values: "mse", "ssim".

Rollout functions (primary — used by training and test):
    compute_rollout_mse(preds, targets)            (B,N,3,H,W) -> scalar
    compute_per_sample_rollout_mse(preds, targets)  (B,N,3,H,W) -> (B,)
    compute_per_step_mse(preds, targets)            (B,N,3,H,W) -> (N,)
    compute_rollout_ssim(preds, targets)            (B,N,3,H,W) -> scalar
    compute_per_sample_rollout_ssim(preds, targets) (B,N,3,H,W) -> (B,)
    compute_per_step_ssim(preds, targets)           (B,N,3,H,W) -> (N,)

Single-step functions (kept for one-step baselines in test):
    compute_single_step_mse(preds, targets)        (B,3,H,W)   -> scalar
    compute_per_sample_mse(preds, targets)          (B,3,H,W)   -> (B,)
    compute_single_step_ssim(preds, targets)        (B,3,H,W)   -> scalar
    ssim_loss_per_sample(prediction, target)        (B,3,H,W)   -> (B,)

Backward-compatible wrappers (used by existing training / validation code):
    compute_model_batch_loss(...)                   -> scalar
    compute_model_batch_and_per_sample_loss(...)    -> (scalar, (B,))
    MSE_loss(prediction, target)                   -> scalar
"""

import torch
import torch.nn.functional as F
from typing import Any, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# Rollout losses  (B, N, C, H, W)  — primary training losses
# ═══════════════════════════════════════════════════════════════════════════

def compute_rollout_mse(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Batch MSE for teacher-forced rollout. Used as the training loss.

    Args:
        preds   : (B, N, C, H, W)
        targets : (B, N, C, H, W)

    Returns:
        scalar — mean MSE over all batch × step × channel × spatial dimensions
    """
    return F.mse_loss(preds, targets)


def compute_per_sample_rollout_mse(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Per-sample MSE averaged over rollout steps and spatial dims.
    Used for per-organoid error stratification.

    Args:
        preds   : (B, N, C, H, W)
        targets : (B, N, C, H, W)

    Returns:
        (B,) — MSE per sample, averaged over N × C × H × W
    """
    loss = F.mse_loss(preds, targets, reduction="none")  # (B, N, C, H, W)
    return loss.mean(dim=(1, 2, 3, 4))                   # (B,)


def compute_per_step_mse(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Per-step MSE averaged over batch and spatial dims.
    Used for rollout-error-vs-time plots.

    Args:
        preds   : (B, N, C, H, W)
        targets : (B, N, C, H, W)

    Returns:
        (N,) — MSE per rollout step, averaged over B × C × H × W
    """
    loss = F.mse_loss(preds, targets, reduction="none")  # (B, N, C, H, W)
    return loss.mean(dim=(0, 2, 3, 4))                   # (N,)


def compute_rollout_latent_mse(
    pred_latents: torch.Tensor,
    true_latents: torch.Tensor,
) -> torch.Tensor:
    """
    MSE between predicted and ground-truth latents over a full rollout.

    true_latents should be computed with torch.no_grad() by passing target
    frames through the frozen encoder so no gradient flows back through them.

    Args:
        pred_latents : (B, N, C, h, w) — ConvLSTM output-head predictions
        true_latents : (B, N, C, h, w) — encoder(target_frames), detached

    Returns:
        scalar — mean MSE over B × N × C × h × w
    """
    return F.mse_loss(pred_latents, true_latents)


# ═══════════════════════════════════════════════════════════════════════════
# Single-step losses  (B, C, H, W) — kept for one-step baselines
# ═══════════════════════════════════════════════════════════════════════════

def compute_single_step_mse(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Batch MSE for one-step prediction.

    Args:
        preds   : (B, C, H, W)
        targets : (B, C, H, W)

    Returns:
        scalar — mean MSE over the batch and spatial/channel dimensions
    """
    return F.mse_loss(preds, targets)


def compute_per_sample_mse(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Per-sample MSE for each of the B samples.

    Args:
        preds   : (B, C, H, W)
        targets : (B, C, H, W)

    Returns:
        per_sample_mse : (B,) — MSE averaged over (C, H, W) at each sample
    """
    assert preds.shape == targets.shape, f"preds {preds.shape} != targets {targets.shape}"
    loss = F.mse_loss(preds, targets, reduction="none")  # (B, C, H, W)
    return loss.mean(dim=(1, 2, 3))                      # (B,)


# ═══════════════════════════════════════════════════════════════════════════
# Backward-compatible wrappers
# ═══════════════════════════════════════════════════════════════════════════
# These keep the same function names / signatures used by
# training and validation code so that existing call sites don't need
# to change. Extra kwargs (input, config, model) are accepted but ignored.

def MSE_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Simple MSE — kept for check_network2_sensitivity.py compatibility."""
    return F.mse_loss(prediction, target)


def compute_model_batch_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "mse",
    *,
    input: Optional[torch.Tensor] = None,
    config: Any = None,
    model: Any = None,
) -> torch.Tensor:
    """
    Scalar (batch-averaged) loss. Use for training / backprop.

    Dispatches to rollout or single-step based on tensor dimensionality:
        5-D (B, N, C, H, W) -> compute_rollout_mse
        4-D (B, C, H, W)    -> compute_single_step_mse

    The `input`, `config`, and `model` kwargs are accepted for backward
    compatibility but are unused.
    """
    if loss_type == "mse":
        if prediction.dim() == 5:
            return compute_rollout_mse(prediction, target)
        return compute_single_step_mse(prediction, target)
    elif loss_type == "ssim":
        if prediction.dim() == 5:
            return compute_rollout_ssim(prediction, target)
        return compute_single_step_ssim(prediction, target)
    else:
        raise ValueError(
            f"Unsupported loss_type={loss_type!r}. Use 'mse' or 'ssim'."
        )


def compute_model_batch_and_per_sample_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "mse",
    *,
    input: Optional[torch.Tensor] = None,
    config: Any = None,
    model: Any = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return both batch scalar loss and per-sample losses.

    Returns:
        (scalar, (B,))
    """
    if loss_type == "mse":
        if prediction.dim() == 5:
            per_sample = compute_per_sample_rollout_mse(prediction, target)
        else:
            per_sample = compute_per_sample_mse(prediction, target)
    elif loss_type == "ssim":
        if prediction.dim() == 5:
            per_sample = compute_per_sample_rollout_ssim(prediction, target)
        else:
            per_sample = ssim_loss_per_sample(prediction, target)
    else:
        raise ValueError(
            f"Unsupported loss_type={loss_type!r}. Use 'mse' or 'ssim'."
        )
    return per_sample.mean(), per_sample


# ═══════════════════════════════════════════════════════════════════════════
# SSIM helpers
# ═══════════════════════════════════════════════════════════════════════════

def _ssim_kernel_1d(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-x.pow(2) / (2 * sigma ** 2))
    return g / g.sum()


def _ssim_kernel_2d(size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    k1d = _ssim_kernel_1d(size, sigma, device)
    k2d = k1d.unsqueeze(0) * k1d.unsqueeze(1)
    return k2d.expand(channels, 1, size, size).contiguous()


def ssim_loss_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """Per-sample SSIM loss (1 - SSIM). Returns (B,)."""
    C = prediction.shape[1]
    pad = window_size // 2
    kernel = _ssim_kernel_2d(window_size, sigma, C, prediction.device)
    mu_p = F.conv2d(prediction, kernel, padding=pad, groups=C)
    mu_t = F.conv2d(target, kernel, padding=pad, groups=C)
    sigma_p_sq = (F.conv2d(prediction ** 2, kernel, padding=pad, groups=C) - mu_p ** 2).clamp(min=0)
    sigma_t_sq = (F.conv2d(target ** 2, kernel, padding=pad, groups=C) - mu_t ** 2).clamp(min=0)
    sigma_pt = F.conv2d(prediction * target, kernel, padding=pad, groups=C) - mu_p * mu_t
    ssim_map = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / (
        (mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p_sq + sigma_t_sq + C2)
    )
    return 1.0 - ssim_map.reshape(prediction.size(0), -1).mean(dim=1)


# ═══════════════════════════════════════════════════════════════════════════
# Single-step SSIM  (B, C, H, W)
# ═══════════════════════════════════════════════════════════════════════════

def compute_single_step_ssim(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Batch-averaged SSIM loss (1 - SSIM) for single-step prediction.

    Args:
        preds   : (B, C, H, W)
        targets : (B, C, H, W)

    Returns:
        scalar — mean (1 - SSIM) over the batch
    """
    return ssim_loss_per_sample(preds, targets).mean()


# ═══════════════════════════════════════════════════════════════════════════
# Rollout SSIM  (B, N, C, H, W)
# ═══════════════════════════════════════════════════════════════════════════

def compute_rollout_ssim(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Batch SSIM loss (1 - SSIM) for teacher-forced rollout.

    Computes SSIM per frame then averages over all steps and samples.

    Args:
        preds   : (B, N, C, H, W)
        targets : (B, N, C, H, W)

    Returns:
        scalar — mean (1 - SSIM) over B × N
    """
    B, N, C, H, W = preds.shape
    # flatten batch and time -> (B*N, C, H, W)
    preds_flat = preds.reshape(B * N, C, H, W)
    targets_flat = targets.reshape(B * N, C, H, W)
    return ssim_loss_per_sample(preds_flat, targets_flat).mean()


def compute_per_sample_rollout_ssim(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Per-sample SSIM loss (1 - SSIM) averaged over rollout steps.

    Args:
        preds   : (B, N, C, H, W)
        targets : (B, N, C, H, W)

    Returns:
        (B,) — (1 - SSIM) per sample, averaged over N steps
    """
    B, N, C, H, W = preds.shape
    preds_flat = preds.reshape(B * N, C, H, W)
    targets_flat = targets.reshape(B * N, C, H, W)
    per_frame = ssim_loss_per_sample(preds_flat, targets_flat)  # (B*N,)
    return per_frame.reshape(B, N).mean(dim=1)                 # (B,)


def compute_per_step_ssim(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Per-step SSIM loss (1 - SSIM) averaged over the batch.

    Args:
        preds   : (B, N, C, H, W)
        targets : (B, N, C, H, W)

    Returns:
        (N,) — (1 - SSIM) per rollout step, averaged over B
    """
    B, N, C, H, W = preds.shape
    preds_flat = preds.reshape(B * N, C, H, W)
    targets_flat = targets.reshape(B * N, C, H, W)
    per_frame = ssim_loss_per_sample(preds_flat, targets_flat)  # (B*N,)
    return per_frame.reshape(B, N).mean(dim=0)                 # (N,)


# ═══════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    B, N, C, H, W = 4, 37, 3, 64, 64

    # ── MSE rollout losses ──
    preds_r = torch.randn(B, N, C, H, W)
    targets_r = torch.randn(B, N, C, H, W)

    rollout_mse = compute_rollout_mse(preds_r, targets_r)
    print(f"compute_rollout_mse:               {rollout_mse.item():.6f} (scalar)")
    assert rollout_mse.dim() == 0

    per_sample_r = compute_per_sample_rollout_mse(preds_r, targets_r)
    print(f"compute_per_sample_rollout_mse:    {per_sample_r.shape}")  # (B,)
    assert per_sample_r.shape == (B,)

    per_step = compute_per_step_mse(preds_r, targets_r)
    print(f"compute_per_step_mse:              {per_step.shape}")  # (N,)
    assert per_step.shape == (N,)

    # ── MSE single-step losses ──
    preds_s = torch.randn(B, C, H, W)
    targets_s = torch.randn(B, C, H, W)

    single_step_mse = compute_single_step_mse(preds_s, targets_s)
    print(f"compute_single_step_mse:           {single_step_mse.item():.6f} (scalar)")
    assert single_step_mse.dim() == 0

    per_sample_s = compute_per_sample_mse(preds_s, targets_s)
    print(f"compute_per_sample_mse:            {per_sample_s.shape}")  # (B,)
    assert per_sample_s.shape == (B,)

    # ── SSIM rollout losses ──
    rollout_ssim = compute_rollout_ssim(preds_r, targets_r)
    print(f"compute_rollout_ssim:              {rollout_ssim.item():.6f} (scalar)")
    assert rollout_ssim.dim() == 0

    per_sample_r_ssim = compute_per_sample_rollout_ssim(preds_r, targets_r)
    print(f"compute_per_sample_rollout_ssim:   {per_sample_r_ssim.shape}")  # (B,)
    assert per_sample_r_ssim.shape == (B,)

    per_step_ssim = compute_per_step_ssim(preds_r, targets_r)
    print(f"compute_per_step_ssim:             {per_step_ssim.shape}")  # (N,)
    assert per_step_ssim.shape == (N,)

    # ── SSIM single-step losses ──
    single_step_ssim_val = compute_single_step_ssim(preds_s, targets_s)
    print(f"compute_single_step_ssim:          {single_step_ssim_val.item():.6f} (scalar)")
    assert single_step_ssim_val.dim() == 0

    ssim_ps = ssim_loss_per_sample(preds_s, targets_s)
    print(f"ssim_loss_per_sample:              {ssim_ps.shape}")  # (B,)
    assert ssim_ps.shape == (B,)

    # ── Backward-compatible wrappers — MSE 5D ──
    batch_loss_5d = compute_model_batch_loss(
        prediction=preds_r, target=targets_r,
        loss_type="mse", input=torch.empty(0), config=None,
    )
    print(f"compute_model_batch_loss (5D,mse): {batch_loss_5d.item():.6f} (scalar)")
    assert batch_loss_5d.dim() == 0

    batch_and_ps_5d = compute_model_batch_and_per_sample_loss(
        prediction=preds_r, target=targets_r, loss_type="mse",
    )
    print(f"batch_and_per_sample (5D,mse):     scalar={batch_and_ps_5d[0].item():.6f}, per_sample={batch_and_ps_5d[1].shape}")
    assert batch_and_ps_5d[0].dim() == 0
    assert batch_and_ps_5d[1].shape == (B,)

    # ── Backward-compatible wrappers — MSE 4D ──
    batch_loss_4d = compute_model_batch_loss(
        prediction=preds_s, target=targets_s, loss_type="mse",
    )
    print(f"compute_model_batch_loss (4D,mse): {batch_loss_4d.item():.6f} (scalar)")
    assert batch_loss_4d.dim() == 0

    # ── Backward-compatible wrappers — SSIM 5D ──
    batch_loss_5d_ssim = compute_model_batch_loss(
        prediction=preds_r, target=targets_r, loss_type="ssim",
    )
    print(f"compute_model_batch_loss (5D,ssim):{batch_loss_5d_ssim.item():.6f} (scalar)")
    assert batch_loss_5d_ssim.dim() == 0

    batch_and_ps_5d_ssim = compute_model_batch_and_per_sample_loss(
        prediction=preds_r, target=targets_r, loss_type="ssim",
    )
    print(f"batch_and_per_sample (5D,ssim):    scalar={batch_and_ps_5d_ssim[0].item():.6f}, per_sample={batch_and_ps_5d_ssim[1].shape}")
    assert batch_and_ps_5d_ssim[0].dim() == 0
    assert batch_and_ps_5d_ssim[1].shape == (B,)

    # ── Backward-compatible wrappers — SSIM 4D ──
    batch_loss_4d_ssim = compute_model_batch_loss(
        prediction=preds_s, target=targets_s, loss_type="ssim",
    )
    print(f"compute_model_batch_loss (4D,ssim):{batch_loss_4d_ssim.item():.6f} (scalar)")
    assert batch_loss_4d_ssim.dim() == 0

    # MSE_loss
    mse = MSE_loss(preds_s, targets_s)
    print(f"MSE_loss:                          {mse.item():.6f} (scalar)")
    assert mse.dim() == 0

    print("\nAll assertions passed.")