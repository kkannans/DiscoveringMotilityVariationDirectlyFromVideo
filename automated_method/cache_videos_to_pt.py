"""
Video frame loader.

`load_video_frames` reads an MP4 into a (T, C, H, W) tensor normalized to [0, 1]. Used by
`video_dataset.py` and `extract_prediction_surprise.py`. The module has no command-line use.
"""
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


def load_video_frames(video_path: Path, start_frame: int = None, num_frames: int = None) -> torch.Tensor:
    """Load video frames as (T, C, H, W) tensor normalized to [0, 1]."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError(f"Could not read frames from {video_path}")

    H, W, C = first_frame.shape

    if start_frame is None:
        start_frame = 0
    elif start_frame < 0:
        start_frame = max(0, total_frames + start_frame)

    if num_frames is None:
        num_frames = total_frames - start_frame
    else:
        num_frames = min(num_frames, total_frames - start_frame)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames_array = np.zeros((num_frames, H, W, C), dtype=np.float32)

    loaded = 0
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames_array[i] = frame.astype(np.float32) / 255.0
        loaded += 1

    cap.release()
    return torch.from_numpy(frames_array[:loaded]).permute(0, 3, 1, 2)


if __name__ == "__main__":
    sys.exit(
        "This module provides load_video_frames and has no command-line entry point.\n"
        "Training reads dataset/batch-XXXXXX.mp4 directly via video_dataset.py."
    )
