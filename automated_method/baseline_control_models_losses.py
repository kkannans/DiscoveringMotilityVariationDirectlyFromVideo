"""
Baseline predictors for sequence prediction tasks.

For each baseline, the predict_sequence method takes in a sequence of frames and returns a sequence of predicted frames.
The compute_loss_per_sample method takes in a sequence of frames and returns the MSE between the predicted and target frames.
The compute_loss method takes in a sequence of frames and returns the mean of the MSE across all samples.

The baselines are:
- Copy: predicts the last input frame.
- LinearInterpolation: predicts the next frame as the average of the previous three frames.
- OpticalFlow: predicts the next frame as the optical flow of the previous two frames.
- DecodedCopy: predicts the last input frame passed through the encoder-decoder bottleneck.
- DecodedLinearInterpolation: predicts the next frame as the average of the previous three frames passed through the encoder-decoder bottleneck.
- DecodedOpticalFlow: predicts the next frame as the optical flow of the previous two frames passed through the encoder-decoder bottleneck.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config import ModelConfig


def compute_sequence_mse_per_sample(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Unweighted per-sample MSE over sequence/time dimension.

    Args:
        predictions: (B, S, 3, H, W)
        targets:     (B, S, 3, H, W)
    Returns:
        Per-sample MSE, shape (B,)
    """
    if predictions.shape != targets.shape:
        raise ValueError(f"Shape mismatch: {predictions.shape} vs {targets.shape}")
    if predictions.dim() != 5:
        raise ValueError(f"Expected (B,S,3,H,W), got {predictions.shape}")
    b = predictions.shape[0]
    if predictions.shape[1] == 0:
        return torch.zeros(b, device=predictions.device, dtype=predictions.dtype)
    return (predictions - targets).pow(2).reshape(b, -1).mean(dim=1)


class BaselinePredictor(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def predict_sequence(self, sequence: torch.Tensor, K: int, N: int) -> torch.Tensor:
        """
        Args:
            sequence: (B, L, C, H, W) where L >= K + N
            K: number of context frames
            N: number of frames to predict
        Returns:
            preds: (B, N, 3, H, W)
        """

    def compute_loss_per_sample(self, sequence: torch.Tensor, K: int, N: int) -> torch.Tensor:
        preds = self.predict_sequence(sequence, K, N)
        target = sequence[:, K:K+N, :3]
        return compute_sequence_mse_per_sample(preds, target)

    def compute_loss(self, sequence: torch.Tensor, K: int, N: int) -> torch.Tensor:
        return self.compute_loss_per_sample(sequence, K, N).mean()


class Copy(BaselinePredictor):
    def predict_sequence(self, sequence: torch.Tensor, K: int, N: int) -> torch.Tensor:
        b, l, _, h, w = sequence.shape
        if N <= 0:
            return torch.zeros(b, 0, 3, h, w, device=sequence.device, dtype=sequence.dtype)
        cur0 = sequence[:, max(K-3, 0)]
        cur1 = sequence[:, max(K-2, 0)]
        cur2 = sequence[:, K-1]
        preds = []
        for _ in range(N):
            pred = cur2[:, :3]
            preds.append(pred)
            next_frame = torch.cat([pred, cur2[:, 3:5]], dim=1)
            cur0, cur1, cur2 = cur1, cur2, next_frame
        return torch.stack(preds, dim=1)

class LinearInterpolation(BaselinePredictor):
    def predict_sequence(self, sequence: torch.Tensor, K: int, N: int) -> torch.Tensor:
        b, l, _, h, w = sequence.shape
        if N <= 0:
            return torch.zeros(b, 0, 3, h, w, device=sequence.device, dtype=sequence.dtype)
        cur0 = sequence[:, max(K-3, 0)]
        cur1 = sequence[:, max(K-2, 0)]
        cur2 = sequence[:, K-1]
        preds = []
        for _ in range(N):
            pred = (cur0[:, :3] + cur1[:, :3] + cur2[:, :3]) / 3.0
            preds.append(pred)
            next_frame = torch.cat([pred, cur2[:, 3:5]], dim=1)
            cur0, cur1, cur2 = cur1, cur2, next_frame
        return torch.stack(preds, dim=1)

class LinearExtrapolation(BaselinePredictor):
    def predict_sequence(self, sequence: torch.Tensor, K: int, N: int) -> torch.Tensor:
        b, l, _, h, w = sequence.shape
        if N <= 0:
            return torch.zeros(b, 0, 3, h, w, device=sequence.device, dtype=sequence.dtype)
        cur0 = sequence[:, max(K-3, 0)]
        cur1 = sequence[:, max(K-2, 0)]
        cur2 = sequence[:, K-1]
        preds = []
        for _ in range(N):
            pred = (-2/3 * cur0[:, :3] + 1/3 * cur1[:, :3] + 4/3 * cur2[:, :3]).clamp(0, 1)
            preds.append(pred)
            next_frame = torch.cat([pred, cur2[:, 3:5]], dim=1)
            cur0, cur1, cur2 = cur1, cur2, next_frame
        return torch.stack(preds, dim=1)


class Black(BaselinePredictor):
    def predict_sequence(self, sequence: torch.Tensor, K: int, N: int) -> torch.Tensor:
        b, l, _, h, w = sequence.shape
        if N <= 0:
            return torch.zeros(b, 0, 3, h, w, device=sequence.device, dtype=sequence.dtype)
        return torch.zeros(b, N, 3, h, w, device=sequence.device, dtype=sequence.dtype)


class OpticalFlow(BaselinePredictor):
    @staticmethod
    def _to_gray_uint8(frame: np.ndarray) -> np.ndarray:
        if frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = np.mean(frame, axis=2)
        if gray.max() <= 1.0:
            return (np.clip(gray, 0, 1) * 255).astype(np.uint8)
        return np.clip(gray, 0, 255).astype(np.uint8)

    @staticmethod
    def _compute_flow(frame_a: torch.Tensor, frame_b: torch.Tensor) -> torch.Tensor:
        """
        frame_a/frame_b: (B, 3, H, W) -> flow (B,2,H,W)
        """
        b, _, h, w = frame_a.shape
        device = frame_a.device
        a_np = frame_a.detach().cpu().numpy()
        b_np = frame_b.detach().cpu().numpy()
        out = []
        for i in range(b):
            a_img = a_np[i].transpose(1, 2, 0)
            b_img = b_np[i].transpose(1, 2, 0)
            flow = cv2.calcOpticalFlowFarneback(
                OpticalFlow._to_gray_uint8(a_img),
                OpticalFlow._to_gray_uint8(b_img),
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            out.append(flow.transpose(2, 0, 1))
        return torch.from_numpy(np.stack(out, axis=0)).to(device=device, dtype=frame_a.dtype)

    @staticmethod
    def _warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        b, _, h, w = image.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=image.device),
            torch.arange(w, device=image.device),
            indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=0).float().unsqueeze(0).expand(b, -1, -1, -1)
        new_grid = grid - flow
        new_grid[:, 0] = 2.0 * new_grid[:, 0] / max(1, (w - 1)) - 1.0
        new_grid[:, 1] = 2.0 * new_grid[:, 1] / max(1, (h - 1)) - 1.0
        new_grid = new_grid.permute(0, 2, 3, 1)
        return F.grid_sample(image, new_grid, mode="bilinear", padding_mode="border", align_corners=True)

    def predict_sequence(self, sequence: torch.Tensor, K: int, N: int) -> torch.Tensor:
        b, l, _, h, w = sequence.shape
        if N <= 0:
            return torch.zeros(b, 0, 3, h, w, device=sequence.device, dtype=sequence.dtype)

        cur0 = sequence[:, max(K-3, 0), :3]
        cur1 = sequence[:, max(K-2, 0), :3]
        cur2 = sequence[:, K-1, :3]
        flow01 = self._compute_flow(cur0, cur1)
        flow12 = self._compute_flow(cur1, cur2)
        flow = 0.5 * (flow01 + flow12)

        preds = []
        for _ in range(N):
            pred = self._warp(cur2, flow)
            preds.append(pred)
            new_flow = self._compute_flow(cur2, pred)
            flow = 0.5 * (flow + new_flow)
            cur0, cur1, cur2 = cur1, cur2, pred
        return torch.stack(preds, dim=1)


def _decode_context_frames(model, sequence: torch.Tensor, K: int):
    """Encode->decode the last 3 context frames through the model's bottleneck.

    Returns decoded versions of frames K-3, K-2, K-1 as (B, 3, H, W) each.
    All downstream decoded baselines operate on these reconstructed frames
    so the comparison is fair: the ConvLSTM also sees encoder-decoded inputs.
    """
    with torch.no_grad():
        dec = []
        for k in [max(K-3, 0), max(K-2, 0), K-1]:
            z = model.encoder(sequence[:, k, :3])
            dec.append(model.decoder(z).clamp(0, 1))
    return dec[0], dec[1], dec[2]


class DecodedCopy(BaselinePredictor):
    """
    Copy baseline on encoder-decoded context frames.
    Decodes the last context frame and repeats it for all N predicted steps.
    """
    def __init__(self, config: ModelConfig, model):
        super().__init__(config)
        self.model = model

    def predict_sequence(self, sequence: torch.Tensor, K: int, N: int) -> torch.Tensor:
        b, l, _, h, w = sequence.shape
        if N <= 0:
            return torch.zeros(b, 0, 3, h, w, device=sequence.device, dtype=sequence.dtype)
        _, _, dec2 = _decode_context_frames(self.model, sequence, K)
        return dec2.unsqueeze(1).expand(b, N, -1, -1, -1)  # (B, N, 3, H, W)


class DecodedLinearInterpolation(BaselinePredictor):
    """
    Linear combination baseline on encoder-decoded context frames.
    Decodes last 3 context frames, then predicts the next frame as
    the average of the previous three decoded frames.
    """
    def __init__(self, config: ModelConfig, model):
        super().__init__(config)
        self.model = model

    def predict_sequence(self, sequence: torch.Tensor, K: int, N: int) -> torch.Tensor:
        b, l, _, h, w = sequence.shape
        if N <= 0:
            return torch.zeros(b, 0, 3, h, w, device=sequence.device, dtype=sequence.dtype)
        dec0, dec1, dec2 = _decode_context_frames(self.model, sequence, K)
        predicted_next_frame = (dec0 + dec1 + dec2) / 3.0
        return predicted_next_frame.unsqueeze(1).expand(b, N, -1, -1, -1)  # (B, N, 3, H, W)


class DecodedOpticalFlow(BaselinePredictor):
    """
    Optical flow baseline on encoder-decoded context frames.
    Decodes last 3 context frames, computes optical flow on the decoded frames,
    then warps for N steps.
    """
    def __init__(self, config: ModelConfig, model):
        super().__init__(config)
        self.model = model

    def predict_sequence(self, sequence: torch.Tensor, K: int, N: int) -> torch.Tensor:
        b, l, _, h, w = sequence.shape
        if N <= 0:
            return torch.zeros(b, 0, 3, h, w, device=sequence.device, dtype=sequence.dtype)

        dec0, dec1, dec2 = _decode_context_frames(self.model, sequence, K)
        flow01 = OpticalFlow._compute_flow(dec0, dec1)
        flow12 = OpticalFlow._compute_flow(dec1, dec2)
        flow = 0.5 * (flow01 + flow12)

        preds = []
        cur1, cur2 = dec1, dec2
        for _ in range(N):
            pred = OpticalFlow._warp(cur2, flow)
            preds.append(pred)
            new_flow = OpticalFlow._compute_flow(cur2, pred)
            flow = 0.5 * (flow + new_flow)
            cur1, cur2 = cur2, pred
        return torch.stack(preds, dim=1)  # (B, N, 3, H, W)


def compute_all_sequence_baselines(
    sequence: torch.Tensor,
    model_config: ModelConfig,
    model=None,
    K: int = 3,
    N: Optional[int] = None,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Returns baseline name -> (preds, per_sample_mse).

    Args:
        sequence: (B, L, C, H, W) where L >= K + N
        model_config: ModelConfig instance
        model: optional model with .encoder/.decoder for decoded baselines
        K: number of context frames
        N: number of frames to predict (defaults to L - K)
    """
    if N is None:
        N = sequence.shape[1] - K

    baselines: Dict[str, BaselinePredictor] = {
        "copy": Copy(model_config),
        "black": Black(model_config),
        "linear_interpolation": LinearInterpolation(model_config),
        "optical_flow": OpticalFlow(model_config),
    }
    if model is not None and hasattr(model, "encoder") and hasattr(model, "decoder"):
        baselines["decoded_copy"] = DecodedCopy(model_config, model)
        baselines["decoded_linear_interpolation"] = DecodedLinearInterpolation(model_config, model)
        baselines["decoded_optical_flow"] = DecodedOpticalFlow(model_config, model)

    target = sequence[:, K:K+N, :3]
    # For decoded baselines, compare against decoded GT (encoder->decoder) so the
    # comparison is in the same reconstructed space as the ConvLSTM model.
    decoded_target = None
    if model is not None and hasattr(model, "encoder") and hasattr(model, "decoder"):
        with torch.no_grad():
            decoded_frames = []
            for t in range(target.shape[1]):
                z = model.encoder(target[:, t])
                decoded_frames.append(model.decoder(z).clamp(0, 1))
            decoded_target = torch.stack(decoded_frames, dim=1)

    out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for name, baseline in baselines.items():
        try:
            preds  = baseline.predict_sequence(sequence, K, N)
            gt = decoded_target if (name.startswith("decoded_") and decoded_target is not None) else target
            losses = compute_sequence_mse_per_sample(preds, gt)
            out[name] = (preds, losses)
        except Exception as e:
            print(f"Warning: baseline '{name}' failed: {e}", flush=True)
    return out
