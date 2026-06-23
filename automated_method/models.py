"""
models.py — OpenSTL model factory for SimVP_TAU, PredRNN, and other architectures.

Provides:
  - ModelConfigs: configuration object expected by OpenSTL recurrent models
  - UnifiedModelWrapper: wraps all OpenSTL models to a common interface
      forward(ctx, tgt) -> pred_y of shape (B, N, C, H, W)
  - _build_raw_model(): instantiates the raw OpenSTL model
  - create_model(): builds and wraps a model, returns (wrapper, n_params)

Supported methods:
  SimVP-style:  SimVP (gSTA), SimVP_TAU (TAU temporal attention)
  Recurrent:    ConvLSTM, PredRNN, PredRNNpp, MIM, E3DLSTM, MAU

Usage:
    from models import create_model
    wrapper, n_params = create_model("SimVP_TAU", K=6, N=6)
    pred = wrapper(ctx, tgt)  # (B, N, 3, 128, 128)
"""

import torch
import torch.nn as nn

# ── Method categories (determines forward interface) ──

SIMVP_STYLE = {"SimVP", "SimVP_TAU"}
RECURRENT_STYLE = {"ConvLSTM", "PredRNN", "PredRNNpp", "MIM", "E3DLSTM", "MAU"}
PHYDNET_STYLE = {"PhyDNet"}

ALL_METHODS = sorted(SIMVP_STYLE | RECURRENT_STYLE | PHYDNET_STYLE)


class ModelConfigs:
    """Configuration object expected by OpenSTL recurrent models.

    Provides the attributes that PredRNN, ConvLSTM, etc. read from `configs`:
    in_shape, pre_seq_length, aft_seq_length, patch_size, total_length, etc.
    """

    def __init__(self, K, N, C=3, H=128, W=128, patch_size=4,
                 filter_size=5, stride=1, layer_norm=0):
        self.in_shape = (K, C, H, W)
        self.pre_seq_length = K
        self.aft_seq_length = N
        self.patch_size = patch_size
        self.filter_size = filter_size
        self.stride = stride
        self.layer_norm = layer_norm
        self.reverse_scheduled_sampling = 0
        self.scheduled_sampling = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.total_length = K + N
        # MAU-specific
        self.tau = 5
        self.cell_mode = 'normal'
        self.model_mode = 'recall'
        self.sr_size = 2


class UnifiedModelWrapper(nn.Module):
    """Wraps all OpenSTL models to a unified interface.

    forward(ctx, tgt) -> pred_y of shape (B, N, C, H, W)

    Handles the different model interfaces internally:
    - SimVP-style: direct forward, slice output to N frames
    - Recurrent: patch reshape, NHWC layout, mask, concat ctx+tgt
    - PhyDNet: its own forward interface with constraints
    """

    def __init__(self, model, method, configs=None):
        super().__init__()
        self.model = model
        self.method = method
        self.configs = configs

    def forward(self, ctx, tgt=None):
        """
        Args:
            ctx: (B, K, C, H, W) context frames
            tgt: (B, N, C, H, W) target frames (needed by recurrent models during training)

        Returns:
            (B, N, C, H, W) predictions
        """
        if self.method in SIMVP_STYLE:
            return self._forward_simvp(ctx, tgt)
        elif self.method in RECURRENT_STYLE:
            return self._forward_recurrent(ctx, tgt)
        elif self.method in PHYDNET_STYLE:
            return self._forward_phydnet(ctx, tgt)
        else:
            raise ValueError(f"Unknown method style: {self.method}")

    def _forward_simvp(self, ctx, tgt=None):
        """
        Forward pass for SimVP-style models.

        Args:
            ctx: (B, K, C, H, W) context frames
            tgt: (B, N, C, H, W) target frames (needed by recurrent models during training)

        Returns:
            (B, N, C, H, W) predictions
        """
        N = tgt.shape[1] if tgt is not None else self.configs.aft_seq_length
        pred = self.model(ctx)  # (B, K, C, H, W)
        pred = pred[:, :N]  # (B, N, C, H, W)
        if tgt is not None and pred.shape[2] > tgt.shape[2]:
            pred = pred[:, :, :tgt.shape[2]]
        return pred

    def _forward_recurrent(self, ctx, tgt=None):
        """
        Forward pass for recurrent models.

        Args:
            ctx: (B, K, C, H, W) context frames
            tgt: (B, N, C, H, W) target frames (needed by recurrent models during training)

        Returns:
            (B, N, C, H, W) predictions
        """
        from openstl.utils import reshape_patch, reshape_patch_back
        cfg = self.configs
        B, K, C, H, W = ctx.shape
        N = tgt.shape[1] if tgt is not None else cfg.aft_seq_length

        if tgt is not None:
            full_seq = torch.cat([ctx, tgt], dim=1)
        else:
            zeros = torch.zeros(B, N, C, H, W, device=ctx.device, dtype=ctx.dtype)
            full_seq = torch.cat([ctx, zeros], dim=1)

        full_seq_nhwc = full_seq.permute(0, 1, 3, 4, 2).contiguous()
        patched = reshape_patch(full_seq_nhwc, cfg.patch_size)

        mask_input = cfg.pre_seq_length
        real_input_flag = torch.zeros(
            B,
            cfg.total_length - mask_input - 1,
            H // cfg.patch_size,
            W // cfg.patch_size,
            cfg.patch_size ** 2 * C,
            device=ctx.device,
        )

        img_gen, _ = self.model(patched, real_input_flag, return_loss=False)

        if self.method == "MAU":
            img_gen = img_gen.permute(0, 1, 3, 4, 2).contiguous()

        img_gen = reshape_patch_back(img_gen, cfg.patch_size)
        pred_y = img_gen[:, -N:].permute(0, 1, 4, 2, 3).contiguous()
        return pred_y

def _build_raw_model(method, K, N, configs, in_channels=3):
    """Build the raw OpenSTL model (without wrapper).

    Args:
        method: One of ALL_METHODS (e.g., "SimVP_TAU", "PredRNN")
        K: Number of context frames
        N: Number of prediction frames
        configs: ModelConfigs instance
        in_channels: Number of input channels (default: 3 for RGB)

    Returns:
        Raw OpenSTL nn.Module
    """
    in_shape = [K, in_channels, 128, 128]

    if method == "SimVP":
        from openstl.models import SimVP_Model
        return SimVP_Model(in_shape=in_shape, hid_S=64, hid_T=256,
                           N_S=4, N_T=8, model_type='gSTA')
    elif method == "SimVP_TAU":
        from openstl.models import SimVP_Model
        return SimVP_Model(in_shape=in_shape, hid_S=64, hid_T=256,
                           N_S=4, N_T=8, model_type='tau')
    elif method in ("ConvLSTM", "PredRNN", "PredRNNpp", "MIM", "E3DLSTM", "MAU"):
        model_map = {
            "ConvLSTM": "ConvLSTM_Model",
            "PredRNN": "PredRNN_Model",
            "PredRNNpp": "PredRNNpp_Model",
            "MIM": "MIM_Model",
            "E3DLSTM": "E3DLSTM_Model",
            "MAU": "MAU_Model",
        }
        import openstl.models as m
        model_cls = getattr(m, model_map[method])
        num_hidden = [64, 64, 64, 64] # number of hidden units in each layer
        return model_cls(num_layers=4, num_hidden=num_hidden, configs=configs)
    else:
        raise ValueError(f"Unknown method: {method}. Available: {ALL_METHODS}")


def create_model(method, K, N):
    """Create a wrapped OpenSTL model.

    Args:
        method: Model architecture name (e.g., "SimVP_TAU", "PredRNN")
        K: Number of context frames
        N: Number of prediction frames

    Returns:
        (wrapped_model, n_params)
        - wrapped_model: UnifiedModelWrapper with forward(ctx, tgt) -> pred
        - n_params: Total parameter count
    """
    in_channels = 3
    configs = ModelConfigs(K, N, C=in_channels)
    raw_model = _build_raw_model(method, K, N, configs, in_channels=in_channels)
    n_params = sum(p.numel() for p in raw_model.parameters())

    wrapper = UnifiedModelWrapper(raw_model, method, configs)
    return wrapper, n_params
