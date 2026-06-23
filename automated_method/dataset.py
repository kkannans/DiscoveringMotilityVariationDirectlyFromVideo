"""
organoid_dataset.py — Organoid video dataset compatible with OpenSTL.

Wraps video data loading into the format OpenSTL expects.

Usage:
    from organoid_dataset import create_organoid_dataloaders
    train_loader, val_loader, test_loader = create_organoid_dataloaders(
        seed=1, K=10, N=2, stride=2, batch_size=16,
    )
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import Dataset, DataLoader

from config import PathConfig
from video_dataset import (
    VideoSequenceDataset,
    _get_batch_names_for_split,
)


def _get_limited_batch_names(path_config, seed, split, data_percentage=100,
                              max_videos=None, exclude_videos=None,
                              datainfo_dir=None, split_prefix="data_split_covering"):
    """Get batch names with optional hard cap on number of videos.

    data_percentage is applied first (random sample), then max_videos caps.
    exclude_videos removes specific batch names (for LOOCV).
    This avoids loading dozens of videos during debug runs.
    """
    import random
    batch_names = _get_batch_names_for_split(path_config, seed, split,
                                              datainfo_dir=datainfo_dir,
                                              split_prefix=split_prefix)
    if exclude_videos:
        batch_names = [b for b in batch_names if b not in exclude_videos]
    if data_percentage < 100:
        n_keep = max(1, int(len(batch_names) * data_percentage / 100))
        rng = random.Random(seed)
        batch_names = rng.sample(batch_names, min(n_keep, len(batch_names)))
    if max_videos is not None and len(batch_names) > max_videos:
        batch_names = batch_names[:max_videos]
    return batch_names


def get_all_batch_names(seed, K=6, N=6, stride=2):
    """Return sorted list of all unique batch names across all splits."""
    path_config = PathConfig(K=K, N=N, stride=stride)
    all_batches = set()
    for split in ["train", "val", "test"]:
        names = _get_batch_names_for_split(path_config, seed, split)
        all_batches.update(names)
    return sorted(all_batches)


class OrganoidOpenSTLDataset(Dataset):
    """
    Wraps VideoSequenceDataset into OpenSTL format.

    OpenSTL expects:
        __getitem__ returns (input_tensor, target_tensor)
        input_tensor:  (T_in, C, H, W)  — context frames
        target_tensor: (T_out, C, H, W) — target frames
    """

    def __init__(self, split, seed, K, N, stride,
                 data_percentage=100,
                 max_videos=None, exclude_videos=None,
                 datainfo_dir=None, split_prefix="data_split_covering"):
        """
        Args:
            exclude_videos: list of batch_name strings to exclude (for LOOCV).
            datainfo_dir: Override directory for split files.
            split_prefix: Filename prefix for split files (default "data_split").
        """
        path_config = PathConfig(K=K, N=N, stride=stride)

        # Pre-filter batch names to avoid loading unnecessary videos
        batch_names = _get_limited_batch_names(
            path_config, seed, split, data_percentage, max_videos,
            exclude_videos=exclude_videos,
            datainfo_dir=datainfo_dir, split_prefix=split_prefix)

        self.inner = VideoSequenceDataset(
            path_config=path_config,
            flag=split,
            seed=seed,
            context_length=K,
            rollout_length=N,
            stride=stride,
            data_percentage=100,  # already filtered above
            batch_names=batch_names,
            return_nnfm_format=False,
        )

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        ctx, tgt, batch_name, start = self.inner[idx]
        # ctx: (K, 3, 128, 128), tgt: (N, 3, 128, 128)
        return ctx, tgt


def create_organoid_dataloaders(seed=1, K=10, N=2, stride=2,
                                 batch_size=16, num_workers=4,
                                 data_percentage=100,
                                 max_videos=None,
                                 splits=None,
                                 exclude_videos=None,
                                 datainfo_dir=None,
                                 split_prefix="data_split_covering"):
    """Create train/val/test dataloaders for OpenSTL.

    Args:
        max_videos: Hard cap on number of videos per split (for fast debug).
                    None means no cap.
        splits: List of splits to load, e.g. ["test"]. Default loads all three.
        exclude_videos: list of batch_name strings to exclude (for LOOCV).
        datainfo_dir: Override directory for split files.
        split_prefix: Filename prefix for split files (default "data_split").
    """
    if splits is None:
        splits = ["train", "val", "test"]
    loaders = {}
    for split in splits:
        ds = OrganoidOpenSTLDataset(
            split=split, seed=seed, K=K, N=N, stride=stride,
            data_percentage=data_percentage,
            max_videos=max_videos,
            exclude_videos=exclude_videos,
            datainfo_dir=datainfo_dir,
            split_prefix=split_prefix,
        )
        shuffle = (split == "train")
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=(split == "train"),
        )
        print(f"  {split}: {len(ds)} sequences, {len(loaders[split])} batches")
    if splits == ["train", "val", "test"]:
        return loaders["train"], loaders["val"], loaders["test"]
    return loaders
