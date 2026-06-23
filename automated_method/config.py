"""
config.py — Experiment configuration for organoid motility video prediction.

Sections (in order):
    1. Optimizer configs    (LearningAlgorithmConfig → Adam / AdamW / SGD)
    2. Scheduler config     (SchedulerConfig)
    3. Training config      (TrainingConfig)
    4. Path config          (PathConfig)
    5. Model configs        (ModelConfig → E2EConvLSTM variants)
    6. Model registry       (MODEL_REGISTRY: int → ModelConfig)
    7. Directory manager    (DirectoryManager)
    8. Config I/O           (load_experiment_config)
"""

import json
import sys
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, Optional, Tuple, Union

from torch import nn

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import paths


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  1. OPTIMIZER CONFIGURATION                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


@dataclass
class LearningAlgorithmConfig(ABC):
    """
    Base class for optimizer configuration.

    Only fields common to ALL optimizers live here (learning_rate, weight_decay).
    Optimizer-specific parameters (betas, momentum, etc.) belong on subclasses.

    Usage:
        config = AdamWConfig(learning_rate=3e-4, weight_decay=0.01)
        optimizer = config.build_optimizer(model.parameters())
    """

    learning_rate: float = 1e-3
    weight_decay: float = 0.0

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier: 'adam', 'adamw', or 'sgd'."""
        ...

    @abstractmethod
    def build_optimizer(self, params) -> torch.optim.Optimizer:
        """Construct and return the torch.optim.Optimizer."""
        ...

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({fields})"


@dataclass
class AdamConfig(LearningAlgorithmConfig):
    """
    Adam optimizer (Kingma & Ba, 2014).

    Fixed: eps = 1e-8 (numerical stability, no reason to change).
    """

    betas: Tuple[float, float] = (0.9, 0.999)
    amsgrad: bool = False

    # Fixed constant — exposed as field for serialization but not intended to be tuned
    eps: float = field(default=1e-8, repr=False)

    @property
    def name(self) -> str:
        return "adam"

    def build_optimizer(self, params) -> torch.optim.Adam:
        return torch.optim.Adam(
            params,
            lr=self.learning_rate,
            betas=self.betas,
            eps=self.eps,
            amsgrad=self.amsgrad,
            weight_decay=self.weight_decay,
        )


@dataclass
class AdamWConfig(LearningAlgorithmConfig):
    """
    AdamW optimizer (Loshchilov & Hutter, 2017) — decoupled weight decay.

    Default weight_decay = 1e-4 (overrides base class 0.0 since AdamW is
    specifically designed for non-zero decay).

    Fixed: eps = 1e-8.
    """

    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)

    eps: float = field(default=1e-8, repr=False)

    @property
    def name(self) -> str:
        return "adamw"

    def build_optimizer(self, params) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )


@dataclass
class SGDConfig(LearningAlgorithmConfig):
    """
    SGD optimizer with optional momentum and Nesterov acceleration.

    Fixed: dampening = 0.0 (standard default, no use case for changing it).
    """

    momentum: float = 0.9
    nesterov: bool = False

    dampening: float = field(default=0.0, repr=False)

    @property
    def name(self) -> str:
        return "sgd"

    def build_optimizer(self, params) -> torch.optim.SGD:
        return torch.optim.SGD(
            params,
            lr=self.learning_rate,
            momentum=self.momentum,
            dampening=self.dampening,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
        )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  2. SCHEDULER CONFIGURATION                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


@dataclass
class SchedulerConfig:
    """
    Learning rate scheduler configuration.

    Default: cosine_annealing (smooth decay to lr_min over num_epochs). All
    scheduler-specific fields are stored here; only the ones matching
    ``self.scheduler`` are used at build time. Factory classmethods provide
    readable construction for each type.

    Options:
    - cosine_annealing (default): LR decays smoothly to lr_min over T_max=num_epochs.
    - reduce_lr_on_plateau: reduces LR when val metric stalls; use with early_stopping.
    - cosine_annealing_warm_restarts: LR restarts every T_0 epochs; can escape saddles.

    Usage:
        sched_cfg = SchedulerConfig.cosine_annealing(lr_min=1e-6)
        scheduler = sched_cfg.build_scheduler(optimizer, num_epochs=1000)
    """

    scheduler: str = "cosine_annealing"
    # Options: "none", "multi_step_lr", "cosine_annealing", "cosine_annealing_warm_restarts", "reduce_lr_on_plateau"

    # ── MultiStepLR ──
    lr_schedule: list = field(
        default_factory=lambda: [20, 50, 100, 300, 500, 1000, 1500, 2000]
    )
    lr_gamma: float = 0.5

    # ── CosineAnnealingLR ──
    lr_min: float = 1e-6

    # ── CosineAnnealingWarmRestarts ──
    T_0: int = 100  # epochs until first restart
    T_mult: int = 2  # multiply T_0 by T_mult after each restart (e.g. 100, 200, 400, ...)
    eta_min: float = 1e-6  # min LR in each cycle
    # When resuming from checkpoint without scheduler_state_dict: set restart peak LR = min(last_lr * multiplier, warm_restart_resume_lr_max)
    warm_restart_resume_lr_multiplier: float = 10.0
    warm_restart_resume_lr_max: float = 1e-3  # cap at epoch-0 LR; never exceed (e.g. don't go to 1e-2)

    # ── ReduceLROnPlateau ──
    scheduler_patience: int = 30
    scheduler_factor: float = 0.9
    scheduler_threshold: float = 1e-5
    scheduler_min_lr: float = 1e-6
    scheduler_mode: str = "min"
    scheduler_cooldown: int = 5

    # ── Factory classmethods ──

    @classmethod
    def multi_step_lr(cls, lr_schedule: list = None, lr_gamma: float = 0.5):
        if lr_schedule is None:
            lr_schedule = [20, 50, 100, 300, 500, 1000, 1500, 2000]
        return cls(scheduler="multi_step_lr", lr_schedule=lr_schedule, lr_gamma=lr_gamma)

    @classmethod
    def cosine_annealing(cls, lr_min: float = 1e-6):
        return cls(scheduler="cosine_annealing", lr_min=lr_min)

    @classmethod
    def cosine_annealing_warm_restarts(
        cls, T_0: int = 100, T_mult: int = 2, eta_min: float = 1e-6,
        warm_restart_resume_lr_multiplier: float = 10.0,
        warm_restart_resume_lr_max: float = 1e-3,
    ):
        """Cosine annealing with periodic restarts. When resuming, restart peak = min(last_lr * multiplier, warm_restart_resume_lr_max); cap respects epoch-0 LR (e.g. 1e-3)."""
        return cls(
            scheduler="cosine_annealing_warm_restarts",
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
            warm_restart_resume_lr_multiplier=warm_restart_resume_lr_multiplier,
            warm_restart_resume_lr_max=warm_restart_resume_lr_max,
        )

    @classmethod
    def reduce_lr_on_plateau(
        cls,
        patience: int = 30,
        factor: float = 0.9,
        threshold: float = 1e-5,
        min_lr: float = 1e-6,
        mode: str = "min",
        cooldown: int = 5,
    ):
        return cls(
            scheduler="reduce_lr_on_plateau",
            scheduler_patience=patience,
            scheduler_factor=factor,
            scheduler_threshold=threshold,
            scheduler_min_lr=min_lr,
            scheduler_mode=mode,
            scheduler_cooldown=cooldown,
        )

    @classmethod
    def none(cls):
        """Constant learning rate (no scheduling)."""
        return cls(scheduler="none")

    # ── Builder ──

    def build_scheduler(self, optimizer: torch.optim.Optimizer, num_epochs: int = None):
        """Construct the torch LR scheduler. Returns None for scheduler='none'."""
        if self.scheduler == "multi_step_lr":
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.lr_schedule, gamma=self.lr_gamma
            )
        elif self.scheduler == "cosine_annealing":
            if num_epochs is None:
                raise ValueError("CosineAnnealingLR requires num_epochs (T_max)")
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=self.lr_min
            )
        elif self.scheduler == "cosine_annealing_warm_restarts":
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.T_0,
                T_mult=self.T_mult,
                eta_min=self.eta_min,
            )
        elif self.scheduler == "reduce_lr_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.scheduler_mode,
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                threshold=self.scheduler_threshold,
                min_lr=self.scheduler_min_lr,
                cooldown=self.scheduler_cooldown,
            )
        elif self.scheduler == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler!r}")


# Preset for Option B: extend training (e.g. 300->1000) with warm restarts. Use when creating
# TrainingConfig, e.g. TrainingConfig(num_epochs=1000, scheduler_config=SCHEDULER_WARM_RESTARTS_EXTEND).
# CheckpointManager will recreate this scheduler on resume when scheduler_state_dict is missing.
SCHEDULER_WARM_RESTARTS_EXTEND: SchedulerConfig = SchedulerConfig.cosine_annealing_warm_restarts(
    T_0=100, T_mult=2, eta_min=1e-6
)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  3. TRAINING CONFIGURATION                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


@dataclass
class TrainingConfig:
    """
    Training hyperparameters.

    Constants (shared across all runs, not settable per-instance):
        Access via class: TrainingConfig.NUM_WORKERS, or instance: self.NUM_WORKERS.
        These are ClassVar so they're excluded from __init__ and cannot be
        accidentally overridden per-instance.

    Instance fields (settable per run):
        Batch sizes, epochs (default 1000), seed, early stopping, optimizer, scheduler (default cosine_annealing).

    Usage:
        cfg = TrainingConfig(
            num_epochs=1000,
            seed=42,
            learning_algorithm=AdamWConfig(learning_rate=3e-4),
            scheduler_config=SchedulerConfig.cosine_annealing(),
        )
        optimizer = cfg.learning_algorithm.build_optimizer(model.parameters())
        scheduler = cfg.scheduler_config.build_scheduler(optimizer, cfg.num_epochs)
    """

    # ── Constants (ClassVar — not in __init__, not per-instance) ──

    SAVE_FREQUENCY: ClassVar[int] = 1
    KEEP_CHECKPOINTS: ClassVar[int] = 3

    # ── Logging / device transfer ──

    log_frequency: int = 10
    non_blocking: bool = True

    # ── Batch sizes ──

    train_batch_size: int = 512
    val_batch_size: int = 512
    test_batch_size: int = 512

    # ── Data / DataLoader (overridable per run) ──

    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True

    latent_dim: int = 8192
    data_percentage: int = 100

    # ── Sequence / chunking (for VideoSequenceDataset and DataManager) ──
    # overridden by PathConfig while training.
    context_length: int = 10   # K: number of context frames
    rollout_length: int = 2   # N: number of target frames to predict
    stride: int = 2           # step between chunk start positions


    # ── Baseline losses (path to .npy or pre-loaded dict; Trainer reads from here) ──

    baseline_losses_path: Optional[Union[str, Path]] = None
    baseline_losses_data: Optional[dict] = None  # pre-loaded dict, e.g. from multi-model parent

    # ── Output head normalization ──
    # "batchnorm" (default, backward-compatible), "groupnorm", or "none"
    output_norm: str = "batchnorm"

    # ── Motion-weighted pixel loss ──
    # Upweight pixels where motion occurs (organoid region).
    # weight_max=1.0 is equivalent to uniform MSE (disabled).
    # weight_max=20.0 gives organoid ~4% of gradient (was 0.2%).
    # weight_max=100.0 gives organoid ~18% of gradient.
    motion_weight_enabled: bool = False
    motion_weight_max: float = 20.0
    motion_weight_threshold: float = 0.02  # absolute motion floor; bg noise ~0.005-0.01, organoid ~0.03+

    # ── Time-weighted loss ──
    # Linearly increasing step weights from 1.0 to time_weight_alpha.
    # Upweights later prediction steps where copy degrades.
    # Set to 1.0 for uniform weighting.
    time_weight_alpha: float = 1.0

    # ── Warmup prediction loss ──
    # During the K-step context warmup, predict z_{t+1} from the hidden state
    # and compare against GT-encoded z_{t+1}.  Provides a training signal before
    # rollout begins.  Set to 0.0 to disable.
    warmup_weight: float = 0.1

    # ── Training loop ──

    num_epochs: int = 1000  # default 1000; use with cosine_annealing or early stopping
    seed: int = 1

    # ── Curriculum rollout length ──
    # Gradually increase rollout length from curriculum_start_length to curriculum_end_length
    # over curriculum_anneal_epochs. Validation always uses full rollout (curriculum_end_length).
    curriculum_enabled: bool = False
    curriculum_start_length: int = 5
    curriculum_end_length: int = rollout_length
    curriculum_anneal_epochs: int = 50

    # ── Gradient clipping ──
    max_grad_norm: float = 1.0  # clip_grad_norm_ max_norm; 0 disables clipping

    # ── Scheduled sampling ──
    scheduled_sampling_enabled: bool = False
    scheduled_sampling_start_ratio: float = 1.0   # epoch 0: pure teacher forcing
    scheduled_sampling_end_ratio: float = 0.0     # epoch anneal_epochs+: pure autoregressive
    scheduled_sampling_anneal_epochs: int = 50    # linear anneal over this many epochs

    # ── Validation visualization frequency ──
    viz_every: int = 10  # save sequence rollout viz every N epochs (0=disable)

    # ── Early stopping ──

    early_stopping_enabled: bool = True
    early_stopping_patience: int = 100
    early_stopping_min_delta: float = 1e-8

    # ── Optimizer & scheduler (sub-configs) ──

    learning_algorithm: LearningAlgorithmConfig = field(default_factory=AdamWConfig)
    # Default: cosine annealing over num_epochs (smooth decay to lr_min)
    scheduler_config: SchedulerConfig = field(
        default_factory=lambda: SchedulerConfig.cosine_annealing()
    )

    # ── Convenience properties (delegate to learning_algorithm) ──

    @property
    def optimizer(self) -> str:
        """Optimizer name string ('adam', 'adamw', 'sgd')."""
        return self.learning_algorithm.name

    @property
    def learning_rate(self) -> float:
        return self.learning_algorithm.learning_rate

    @property
    def weight_decay(self) -> float:
        return self.learning_algorithm.weight_decay

    def get_curriculum_rollout_length(self, epoch: int) -> int:
        """Compute rollout length for curriculum learning at given epoch.

        Linearly anneals from curriculum_start_length to curriculum_end_length
        over curriculum_anneal_epochs. Returns curriculum_end_length when disabled.
        """
        if not self.curriculum_enabled:
            return self.curriculum_end_length
        progress = min(epoch / max(self.curriculum_anneal_epochs, 1), 1.0)
        T = self.curriculum_start_length + (self.curriculum_end_length - self.curriculum_start_length) * progress
        return min(int(T), self.curriculum_end_length)

    def get_teacher_forcing_ratio(self, epoch: int) -> float:
        """Compute teacher forcing ratio for scheduled sampling at given epoch.

        Linearly anneals from `scheduled_sampling_start_ratio` to
        `scheduled_sampling_end_ratio` over `scheduled_sampling_anneal_epochs`.
        Returns 1.0 (pure teacher forcing) when scheduled sampling is disabled.
        """
        if not self.scheduled_sampling_enabled:
            return 1.0
        start = self.scheduled_sampling_start_ratio
        end = self.scheduled_sampling_end_ratio
        anneal = self.scheduled_sampling_anneal_epochs
        if anneal <= 0:
            return end
        ratio = start - (start - end) * min(epoch / anneal, 1.0)
        return max(end, min(start, ratio))


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  4. PATH CONFIGURATION                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def validate_path_config_inputs(
    K: int,
    N: int,
    stride: int
) -> Tuple[int, int, int]:
    """
    Validate and normalize PathConfig inputs.

    Returns K, N, stride.
    """
    if K <= 0:
        raise ValueError(f"PathConfig.K must be positive, got {K}")
    if N <= 0:
        raise ValueError(f"PathConfig.N must be positive, got {N}")
    if stride <= 0:
        raise ValueError(f"PathConfig.stride must be positive, got {stride}")

    return K, N, stride


class PathConfig:
    """
    File and directory path management.

    Directory tree (all experiment outputs live under exp_folder; raw data does not):

        $OUTPUT_ROOT/
        ├── datainfo/                               data split JSONs
        ├── data_split_analysis/                        split analysis plots
        ├── train_multiple_models_logs/                 SLURM scripts and job .out/.err logs
        ├── training/{K}_{N}_{stride}/
        │   ├── {model}_{loss}_{seed}/                  per-run training outputs
        │   │   ├── logs/
        │   │   ├── checkpoints/
        │   │   ├── models/
        │   │   └── visualizations/
        │   ├── shared_baselines/                       baseline .npy files
        │   └── combined_model_training_results/
        ├── testing/{K}_{N}_{stride}/
        │   └── {model}_{loss}_{seed}/                  per-run test outputs
        └── test_performance/                           cross-run comparisons
    """
    def __init__(self, K: int = 15, N: int = 10, stride: int = 10,
                 sub_experiment: Optional[str] = None):
        # Core sequence/chunking parameters
        self.K = K
        self.N = N
        self.stride = stride
        # Backward-compatible window length (used by some dataset utilities)
        self.K, self.N, self.stride = validate_path_config_inputs(K, N, stride)
        # Sub-experiment label (e.g. "A", "B") → training_A/, testing_A/
        self.sub_experiment = sub_experiment
        self.experiment_name = "organoid_dynamics"
        self.default_loss_type = "mse"
        self.experiment_description = (
            "SimVP_TAU video prediction on organoid videos (120 frames each), "
            "cosine annealing scheduler, loss = MSE"
        )
        self.source_data_path = str(paths.DATA_ROOT)
        self.data_path = self._determine_data_path()
        self.datainfo_path = self.get_datainfo_path()

    # ── Experiment root ──

    def get_exp_folder(self) -> Path:
        return paths.OUTPUT_ROOT

    # ── Data splits ──

    def get_datainfo_path(self) -> Path:
        return paths.DATAINFO_DIR

    @property
    def dataset_name(self) -> str:
        """Experiment + K, N, stride identifier for checkpoint/model filenames and data paths (e.g. Q8_v18_3_37_10)."""
        return f"{self.experiment_name}_{self.K}_{self.N}_{self.stride}"

    def get_data_split_analysis_dir(self) -> Path:
        return self.get_exp_folder() / "data_split_analysis"

    # ── Training ──

    def get_training_base(self) -> Path:
        folder = f"training_{self.sub_experiment}" if self.sub_experiment else "training"
        return self.get_exp_folder() / folder / f"{self.K}_{self.N}_{self.stride}"

    def get_main_folder(self, model_name: str, loss_type: str, seed: int) -> Path:
        """Per-run folder: logs, checkpoints, models, visualizations."""
        return self.get_training_base() / f"{model_name}_{loss_type}_{seed}"

    def get_combined_training_results_dir(self) -> Path:
        return self.get_training_base() / "combined_model_training_results"

    def get_baseline_losses_path(self, seed: int, loss_type: str) -> Path:
        """
        Shared baseline losses for a (seed, loss_type) pair.

        Returns: {training_base}/shared_baselines/seed_{seed}_loss_{loss_type}.npy

        The .npy file contains per-sequence and mean losses for:
        copy, black, linear_interpolation, optical_flow.
        Same file is reused by all model architectures (depends only on data + loss).
        """
        path = (
            self.get_training_base()
            / "shared_baselines"
            / f"seed_{seed}_loss_{loss_type}.npy"
        )
        # Sanity checks — catch path-construction bugs early
        assert "shared_baselines" in str(path)
        return path

    def get_decoded_baseline_losses_path(self, seed: int, loss_type: str) -> Path:
        """
        Shared decoded baseline losses for a (seed, loss_type) pair.

        Returns: {training_base}/shared_baselines/seed_{seed}_loss_{loss_type}_decoded.npy

        Decoded baselines depend on the stage-1 encoder/decoder (determined by seed),
        so they are per-seed but shared across model architectures.
        """
        path = (
            self.get_training_base()
            / "shared_baselines"
            / f"seed_{seed}_loss_{loss_type}_decoded.npy"
        )
        assert "shared_baselines" in str(path)
        return path

    # ── Testing ──

    def get_testing_base(self) -> Path:
        folder = f"testing_{self.sub_experiment}" if self.sub_experiment else "testing"
        return self.get_exp_folder() / folder / f"{self.K}_{self.N}_{self.stride}"

    def get_test_output_folder(self, model_name: str, loss_type: str, seed: int) -> Path:
        return self.get_testing_base() / f"{model_name}_{loss_type}_{seed}"

    def get_test_performance_dir(self) -> Path:
        return self.get_exp_folder() / "test_performance"

    # ── SLURM helpers ──

    def get_train_multiple_models_logs_dir(self) -> str:
        """SLURM scripts and job logs live under the experiment folder."""
        return str(self.get_exp_folder() / "train_multiple_models_logs")

    def get_individual_training_logs_dir(self) -> Path:
        """Directory for individual model training SLURM scripts and logs (same as train_multiple_models_logs for check_job_status compatibility)."""
        return self.get_exp_folder() / "train_multiple_models_logs"

    def get_job_short_prefix(self) -> str:
        """Prefix for SLURM job names and log filenames: the experiment name (e.g. Q8_v1, Q8_v2)."""
        return self.experiment_name.replace(".", "_")

    def get_job_name_prefix(self) -> str:
        """Short prefix for job names and log/script filenames. Same as get_job_short_prefix()."""
        return self.get_job_short_prefix()

    def get_slurm_job_name(self, L: int, seed: int) -> str:
        """Full SLURM job name for train_multiple_models: {prefix}_{K}_{N}_{stride}_seed_{seed}."""
        return f"{self.get_job_short_prefix()}_{self.K}_{self.N}_{self.stride}_seed_{seed}"

    def _determine_data_path(self) -> str:
        """Path to organoid video files."""
        return str(paths.DATA_ROOT)

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict (PathConfig is not a dataclass, so asdict() cannot be used)."""
        d = {
            "K": self.K,
            "N": self.N,
            "stride": self.stride,
            "experiment_name": self.experiment_name,
            "default_loss_type": self.default_loss_type,
            "experiment_description": self.experiment_description,
            "source_data_path": self.source_data_path,
            "data_path": self.data_path,
            "datainfo_path": str(self.datainfo_path),
        }
        if self.sub_experiment:
            d["sub_experiment"] = self.sub_experiment
        return d


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  5. MODEL CONFIGURATION                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


@dataclass
class ModelConfig(ABC):
    """
    Per-network-type configuration: architecture class, channel layout, data transform.

    Input is raw pixel frames (B, K, 3, H, W).  input_channels refers to the RGB
    channels of a single frame (3), output_channels is 3 (RGB prediction).
    """

    model_class: type       # nn.Module subclass
    network_type: str       # used for loss function dispatch
    name: str               # human-readable identifier
    input_channels: int     # channels per frame (3 for RGB)
    output_channels: int    # channels the model predicts
    input_channel_feature_map: Dict[str, list] = field(default_factory=dict)
    output_channel_feature_map: Dict[str, list] = field(default_factory=dict)

    def create_model(self) -> nn.Module:
        """Instantiate the model architecture."""
        return self.model_class()

    @abstractmethod
    def _transform_data_impl(self, data: torch.Tensor) -> torch.Tensor:
        """Subclass hook: select/reorder channels from a (B, C, H, W) tensor."""
        ...


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  6. DIRECTORY MANAGER                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class DirectoryManager:
    """
    Create and expose the per-run directory tree:

        {main_folder}/
        ├── logs/
        ├── checkpoints/
        ├── models/
        └── visualizations/
            ├── relative_improvement_vs_motion/
            ├── validation_viz/
            └── training_viz/
    """

    _SUBDIRS = ("logs", "checkpoints", "models", "visualizations")
    # Only create validation/train viz dirs by default;
    _VIZ_SUBDIRS = ("validation_viz", "training_viz")

    def __init__(self, path_config: PathConfig, model_name: str, loss_type: str, seed: int):
        self.main_folder = path_config.get_main_folder(model_name, loss_type, seed)
        self._create_directories()

    def _create_directories(self):
        self.main_folder.mkdir(parents=True, exist_ok=True)
        for name in self._SUBDIRS:
            (self.main_folder / name).mkdir(exist_ok=True)
        for name in self._VIZ_SUBDIRS:
            (self.main_folder / "visualizations" / name).mkdir(exist_ok=True)

    # ── Convenience properties ──

    @property
    def logs_folder(self) -> Path:
        return self.main_folder / "logs"

    @property
    def checkpoints_folder(self) -> Path:
        return self.main_folder / "checkpoints"

    @property
    def models_folder(self) -> Path:
        return self.main_folder / "models"

    @property
    def viz_folder(self) -> Path:
        return self.main_folder / "visualizations"

    @property
    def viz_validation_losses_folder(self) -> Path:
        return self.viz_folder / "validation_viz"

    @property
    def viz_training_folder(self) -> Path:
        return self.viz_folder / "training_viz"


def load_experiment_config(config_dir: Path) -> dict:
    """
    Load experiment configuration JSONs from a directory.

    Looks for: training_config.json, path_config.json, model_config.json,
    experiment_metadata.json. Returns a dict with keys for whichever files exist.
    """
    configs = {}
    for filename, key in [
        ("training_config.json", "training"),
        ("path_config.json", "path"),
        ("model_config.json", "model"),
        ("experiment_metadata.json", "metadata"),
    ]:
        path = config_dir / filename
        if path.exists():
            with open(path, "r") as f:
                configs[key] = json.load(f)
    return configs