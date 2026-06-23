import logging
import os
import json
import sys
import torch
import numpy as np
import random
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Collection, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
try:
    import psutil
except ImportError:
    psutil = None
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

_PROJECT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PROJECT_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import PathConfig, TrainingConfig, ModelConfig
from cache_videos_to_pt import load_video_frames

logger = logging.getLogger(__name__)

def _resize_frames_to_128(frames):
    """Resize all frames to 128x128. frames: (T, C, H, W) -> (T, C, 128, 128). Accepts np.ndarray or Tensor."""
    t = frames if isinstance(frames, torch.Tensor) else torch.from_numpy(frames)
    t = torch.nn.functional.interpolate(t, size=(128, 128), mode="area")
    return t.numpy()

def _get_batch_names_for_split(path_config: PathConfig, seed: int, flag: str,
                                datainfo_dir: str = None,
                                split_prefix: str = "data_split_covering") -> List[str]:
    """Return batch names for the given split from {split_prefix}_{seed}.json.

    Args:
        datainfo_dir: Override directory for split files. If None, uses ./datainfo/.
        split_prefix: Filename prefix (default "data_split_covering").
    """
    if datainfo_dir is not None:
        base_dir = Path(datainfo_dir).expanduser()
    else:
        base_dir = Path(__file__).resolve().parent.parent / "datainfo"
    split_path = base_dir / f"{split_prefix}_{seed}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Data split not found: {split_path}")
    with open(split_path, "r") as f:
        data = json.load(f)
    part = data.get(flag)
    if not part:
        return []
    return part if isinstance(part, list) else list(part.keys())

# ============================================================================
# VideoSequenceDataset
# ============================================================================


class VideoSequenceDataset(Dataset):
    """
    Sequence-level dataset for teacher-forced rollout training.
    Returns (context_frames, target_frames, info_dict) when
    return_nnfm_format=True, else (context_frames, target_frames,
    batch_name, start).
    """

    def __init__(
        self,
        path_config: PathConfig,
        flag: str,
        seed: int,
        debug: bool = False,
        num_workers: int = 4,
        data_percentage: int = 100,
        context_length: int = 15,
        rollout_length: int = 24,
        stride: int = 10,
        exclude_batches: Optional[Collection[str]] = None,
        batch_names: Optional[List[str]] = None,
        return_nnfm_format: bool = True,
    ):
        self.path_config = path_config
        self.seed = seed
        self.flag = flag
        self.context_length = context_length
        self.rollout_length = rollout_length
        self.stride = stride
        self.exclude_batches: set = set(exclude_batches) if exclude_batches else set()
        self.return_nnfm_format = return_nnfm_format
        self.num_workers = num_workers

        if batch_names is None:
            batch_names = _get_batch_names_for_split(path_config, seed, flag)
        if data_percentage < 100:
            n_keep = max(1, int(len(batch_names) * data_percentage / 100))
            batch_names = random.sample(batch_names, min(n_keep, len(batch_names)))
        if self.exclude_batches:
            batch_names = [b for b in batch_names if b not in self.exclude_batches]
        self.batch_names = batch_names

        self.batch_data: Dict[str, Dict] = {}
        self._load_all_batches()

        self.samples: List[Tuple[str, int]] = []
        self._build_sample_index()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_single_batch(self, batch_name: str) -> Optional[Tuple[str, Dict]]:
        """Worker function to load a single batch of RGB frames."""
        frames = self._load_rgb_frames(batch_name)
        if frames is None:
            return None

        return batch_name, {
            "frames": frames,           # (T, 3, H, W)
        }

    def _load_all_batches(self) -> None:
        """Preload RGB frames for every batch in parallel."""
        # ThreadPoolExecutor requires max_workers >= 1; fall back to 1 when num_workers=0.
        with ThreadPoolExecutor(max_workers=max(1, self.num_workers)) as executor:
            future_to_batch = {
                executor.submit(self._load_single_batch, batch_name): batch_name
                for batch_name in self.batch_names
            }
            for future in tqdm(
                as_completed(future_to_batch),
                total=len(future_to_batch),
                desc=f"[{self.flag}] Loading batches",
                unit="batch",
            ):
                batch_name = future_to_batch[future]
                try:
                    result = future.result()
                    if result is not None:
                        b_name, b_data = result
                        self.batch_data[b_name] = b_data
                except Exception as exc:
                    logger.error(f"Batch {batch_name} generated an exception during loading: {exc}")

    def _load_rgb_frames(self, batch_name: str) -> Optional[torch.Tensor]:
        """
        Load RGB frames for a batch.

        Loads raw mp4 videos in data_path (first 120 frames).
        Returns frames as float tensor of shape
        (T, 3, 128, 128) or None if the source does not exist.
        """
        video_dir = self.path_config.data_path
        video_path = Path(video_dir) / f"{batch_name}.mp4"
        assert video_path.exists(), f"Video path does not exist: {video_path}"
        frames_np = load_video_frames(video_path, start_frame=0, num_frames=120)
        frames_np = _resize_frames_to_128(frames_np)
        return torch.from_numpy(frames_np).float()

    def _build_sample_index(self) -> None:
        K = self.context_length
        N = self.rollout_length
        chunk_size = K + N

        for batch_name, bdata in self.batch_data.items():
            T = bdata["frames"].shape[0]
            max_start = T - chunk_size

            if max_start < 0:
                logger.warning(
                    "Batch %s: only %d frames, need %d (K=%d+N=%d). Skipping.",
                    batch_name, T, chunk_size, K, N,
                )
                continue

            for start in range(0, max_start + 1, self.stride):
                self.samples.append((batch_name, start))

        logger.info(
            "[%s] Built %d sequence chunks (K=%d, N=%d, stride=%d) from %d batches.",
            self.flag, len(self.samples), K, N, self.stride, len(self.batch_data),
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        batch_name, start = self.samples[idx]
        bdata = self.batch_data[batch_name]
        K, N = self.context_length, self.rollout_length
        frames = bdata["frames"]

        context_frames = frames[start : start + K]
        target_frames = frames[start + K : start + K + N]

        if self.return_nnfm_format:
            info_dict = {
                "batch_name": batch_name,
                "start_frame": start,
                "K": K,
                "N": N,
                "stride": self.stride,
            }
            return context_frames, target_frames, info_dict

        return (
            context_frames,
            target_frames,
            batch_name,
            start,
        )


class DummySequenceDataset(Dataset):
    """Synthetic dataset matching VideoSequenceDataset 4-tuple interface."""

    def __init__(
        self,
        n_samples: int = 200,
        context_length: int = 3,
        rollout_length: int = 37,
        seed: int = 0,
        H: int = 128,
        W: int = 128,
    ):
        self.n_samples = n_samples
        self.context_length = context_length
        self.rollout_length = rollout_length
        self.seed = seed
        self.H = H
        self.W = W
        self.samples = self._generate()

    def _generate(self):
        g = torch.Generator().manual_seed(self.seed)
        samples = []
        for i in range(self.n_samples):
            K, N = self.context_length, self.rollout_length
            context = torch.rand(K, 3, self.H, self.W, generator=g)
            targets = torch.rand(N, 3, self.H, self.W, generator=g)
            samples.append((context, targets, f"dummy_{i % 5}", i * 10))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def collate_sequences(batch):
    """Collate VideoSequenceDataset 4-tuple samples into batched tensors."""
    context_frames = torch.stack([b[0] for b in batch])
    target_frames = torch.stack([b[1] for b in batch])
    batch_names = [b[2] for b in batch]
    start_frames = [b[3] for b in batch]
    return context_frames, target_frames, batch_names, start_frames


class DataManager:
    def __init__(
        self,
        path_config: PathConfig,
        training_config: TrainingConfig,
        seed: int,
        debug: bool = False,
        exclude_batches: Optional[Collection[str]] = None,
    ):
        self.path_config = path_config
        self.training_config = training_config
        self.seed = seed
        self.context_length = getattr(training_config, "context_length", 15)
        self.rollout_length = getattr(training_config, "rollout_length", 24)
        self.stride = getattr(training_config, "stride", 10)
        self.debug = debug
        self.exclude_batches = exclude_batches or set()

    def _make_dataset(self, flag: str) -> Dataset:
        if self.debug:
            n = {"train": 500, "val": 100, "test": 100}.get(flag, 100)
            return DummySequenceDataset(
                n_samples=n,
                context_length=self.context_length,
                rollout_length=self.rollout_length,
                seed=self.seed,
            )
        return VideoSequenceDataset(
            path_config=self.path_config,
            flag=flag,
            seed=self.seed,
            num_workers=getattr(self.training_config, "num_workers", 4),
            data_percentage=getattr(self.training_config, "data_percentage", 100),
            context_length=self.context_length,
            rollout_length=self.rollout_length,
            stride=self.stride,
            exclude_batches=self.exclude_batches,
            return_nnfm_format=False,
        )

    def create_train_val_datasets(self):
        train_ds = self._make_dataset("train")
        val_ds = self._make_dataset("val")
        print(f"Train sequences: {len(train_ds)}, Val sequences: {len(val_ds)}", flush=True)
        return train_ds, val_ds

    def create_test_dataset(self):
        test_ds = self._make_dataset("test")
        print(f"Test sequences: {len(test_ds)}", flush=True)
        return test_ds

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        # If the dataset is empty, force non-shuffling so that PyTorch uses a
        # SequentialSampler instead of RandomSampler (which requires num_samples > 0).
        if len(dataset) == 0:
            logger.warning(
                "Requested DataLoader for empty %s split; using non-shuffling loader with 0 samples.",
                getattr(dataset, "flag", "unknown"),
            )
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=self.training_config.train_batch_size,
            shuffle=shuffle,
            num_workers=self.training_config.num_workers,
            collate_fn=collate_sequences,
            drop_last=False,
            pin_memory=self.training_config.pin_memory,
        )

    def create_train_val_dataloaders(self, train_ds, val_ds):
        return (
            self._make_loader(train_ds, shuffle=True),
            self._make_loader(val_ds, shuffle=False),
        )

    def create_test_dataloader(self, test_ds):
        return self._make_loader(test_ds, shuffle=False)


# ============================================================================
# FrameDataset — individual frames for encoder-decoder pre-training
# ============================================================================

class DummyFrameDataset(Dataset):
    """
    Synthetic frame dataset for debugging encoder-decoder pre-training.
    Each sample is a 3-tuple (frame, video_id, frame_idx):
        frame     : (3, 128, 128)
        video_id  : str  — "dummy" for all items
        frame_idx : int  — position within the dummy video
    """

    def __init__(self, size: int = 1000, seed: int = 42):
        g = torch.Generator().manual_seed(seed)
        self.frames = torch.rand(size, 3, 128, 128, generator=g)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int]:
        return self.frames[idx], "dummy", idx


class FrameDataset(Dataset):
    """
    Individual-frame dataset for encoder-decoder pre-training.

    Loads the same videos as VideoSequenceDataset but returns single
    frames (3, 128, 128) instead of sequences.  All frames from every
    video in the split are concatenated into one contiguous tensor so
    that random-access indexing is O(1).

    Usage:
        ds = FrameDataset(path_config, flag="train", seed=1)
        frame = ds[42]   # (3, 128, 128)
    """

    def __init__(
        self,
        path_config: PathConfig,
        flag: str,
        seed: int,
        num_workers: int = 4,
        data_percentage: int = 100,
        datainfo_dir: str = None,
        split_prefix: str = "data_split",
    ):
        self.path_config = path_config
        self.flag = flag

        batch_names = _get_batch_names_for_split(path_config, seed, flag,
                                                  datainfo_dir=datainfo_dir,
                                                  split_prefix=split_prefix)
        if data_percentage < 100:
            n_keep = max(1, int(len(batch_names) * data_percentage / 100))
            batch_names = random.sample(batch_names, min(n_keep, len(batch_names)))
        self.batch_names = batch_names

        # frames: (N_total, 3, 128, 128)
        # _video_ids[i]: name of the source video for flat index i
        # _frame_idxs[i]: position of frame i within its source video
        self.frames, self._video_ids, self._frame_idxs = self._load_all(num_workers)

    def _load_batch(self, batch_name: str) -> Optional[torch.Tensor]:
        video_path = Path(self.path_config.data_path) / f"{batch_name}.mp4"
        if not video_path.exists():
            logger.warning("Video not found, skipping: %s", video_path)
            return None
        frames_np = load_video_frames(video_path, start_frame=0, num_frames=120)
        frames_np = _resize_frames_to_128(frames_np)
        return torch.from_numpy(frames_np).float()  # (T, 3, 128, 128)

    def _load_all(self, num_workers: int) -> Tuple[torch.Tensor, List[str], List[int]]:
        chunks: List[torch.Tensor] = []
        video_id_meta: List[str] = []
        frame_idx_meta: List[int] = []

        with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
            futures = {
                executor.submit(self._load_batch, bn): bn
                for bn in self.batch_names
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"[{self.flag}] Loading frames",
                unit="video",
            ):
                bn = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        T = result.shape[0]
                        chunks.append(result)
                        video_id_meta.extend([bn] * T)
                        frame_idx_meta.extend(range(T))
                except Exception as exc:
                    logger.error("Error loading %s: %s", bn, exc)

        if chunks:
            all_frames = torch.cat(chunks, dim=0)  # (N_total, 3, 128, 128)
        else:
            all_frames = torch.zeros(0, 3, 128, 128)

        logger.info(
            "[%s] Loaded %d frames from %d/%d videos.",
            self.flag, len(all_frames), len(chunks), len(self.batch_names),
        )
        return all_frames, video_id_meta, frame_idx_meta

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int]:
        return self.frames[idx], self._video_ids[idx], self._frame_idxs[idx]


class FrameDataManager:
    """
    Convenience wrapper: builds FrameDataset + DataLoader pairs for
    encoder-decoder pre-training.
    """

    def __init__(
        self,
        path_config: PathConfig,
        seed: int,
        batch_size: int = 64,
        num_workers: int = 4,
        data_percentage: int = 100,
        pin_memory: bool = True,
        debug: bool = False,
        datainfo_dir: str = None,
        split_prefix: str = "data_split",
    ):
        self.path_config = path_config
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_percentage = data_percentage
        self.pin_memory = pin_memory
        self.debug = debug
        self.datainfo_dir = datainfo_dir
        self.split_prefix = split_prefix

    def _make_dataset(self, flag: str) -> Dataset:
        if self.debug:
            sizes = {"train": 500, "val": 100, "test": 100}
            return DummyFrameDataset(size=sizes.get(flag, 100), seed=self.seed)
        return FrameDataset(
            path_config=self.path_config,
            flag=flag,
            seed=self.seed,
            num_workers=self.num_workers,
            data_percentage=self.data_percentage,
            datainfo_dir=self.datainfo_dir,
            split_prefix=self.split_prefix,
        )

    def _make_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def create_train_val(self):
        train_ds = self._make_dataset("train")
        val_ds = self._make_dataset("val")
        print(
            f"FrameDataset — train: {len(train_ds)} frames, val: {len(val_ds)} frames",
            flush=True,
        )
        return (
            self._make_loader(train_ds, shuffle=True),
            self._make_loader(val_ds, shuffle=False),
        )


if __name__ == "__main__":
    from config import PathConfig, TrainingConfig
    path_config = PathConfig(K=15, N=24, stride=10)
    training_config = TrainingConfig(
        data_percentage=10,
        context_length=15,
        rollout_length=24,
        stride=10,
    )
    seed, debug = 1, False

    data_manager = DataManager(
        path_config=path_config,
        training_config=training_config,
        seed=seed,
        debug=debug,
    )
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss / (1024 ** 3)  # GB

    time_start = time.time()
    train_dataset, val_dataset = data_manager.create_train_val_datasets()
    time_end = time.time()

    memory_after = process.memory_info().rss / (1024 ** 3)  # GB
    memory_used = memory_after - memory_before

    print(f"Time taken to create train and validation datasets: {time_end - time_start:.2f} seconds")
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")
    print(f"Memory before loading datasets: {memory_before:.2f} GB", flush=True)
    print(f"Memory after loading datasets: {memory_after:.2f} GB", flush=True)
    print(f"Memory used by datasets: {memory_used:.2f} GB", flush=True)

    # Visualize a sample sequence (DataManager returns 4-tuple: ctx, tgt, batch_name, start)
    os.makedirs("test_dataset", exist_ok=True)
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        context_frames, target_frames, batch_name, start_frame = sample

        # Gather the frames: all context (input) + first 10 target frames
        input_viz = context_frames  # (K, 3, 128, 128)
        target_viz = target_frames[:10]  # (<=10, 3, 128, 128)
        frames_to_show = torch.cat([input_viz, target_viz], dim=0)  # (K+10, 3, 128, 128)
        L = frames_to_show.shape[0]

        frames_np = frames_to_show.detach().cpu().numpy()
        frames_np = np.clip(frames_np, 0.0, 1.0)

        fig, axes = plt.subplots(1, L, figsize=(2 * L, 2.5))
        if L == 1:
            axes = [axes]

        K_ctx = context_frames.shape[0]
        for i in range(L):
            ax = axes[i]
            frame_img = frames_np[i].transpose(1, 2, 0)
            ax.imshow(frame_img)
            frame_idx = start_frame + i
            if i < K_ctx:
                ax.set_title(f"Input t={frame_idx}", fontsize=10, color="blue")
            else:
                ax.set_title(f"Target t={frame_idx}", fontsize=10)
            ax.axis("off")

        fig.suptitle(f"Batch: {batch_name}", fontsize=12, fontweight="bold")
        plt.tight_layout()

        out_path = os.path.join("test_dataset", f"sequence_{batch_name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor='white')
        plt.close()

        print(f"Saved visualization to {out_path}", flush=True)