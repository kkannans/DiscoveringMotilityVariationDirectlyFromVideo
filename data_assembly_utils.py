"""
Utilities for assembling video frames from folder/images for training.
"""
import os
import numpy as np
import torch
import glob
from PIL import Image
from datetime import datetime
from pprint import pprint
from typing import Dict, List
import cv2
from tqdm import tqdm

def get_image_file_paths(image_folder_path:str):
    """Get image file paths from folder."""
    image_file_paths = []
    for file in os.listdir(os.path.expanduser(image_folder_path)):
        if file.endswith(".jpg"):
            image_file_paths.append(os.path.join(os.path.expanduser(image_folder_path), file))
    return image_file_paths

def get_command_file_path(commands_folder_path:str, command_name:str):
    """Get command file path from folder."""
    command_file_paths = []
    command_file_paths = glob.glob(os.path.join(os.path.expanduser(commands_folder_path), f"*{command_name}*.json*"))
    # sort by creation time
    command_file_paths.sort(key=lambda x: os.path.getctime(x))
    return command_file_paths

def get_timestamp_from_file_path(file_path: str):
    """Get timestamp from file path.
    
    Args:
        file_path: Path like 'df7c66d1-13b7-4f8e-a3f5-3f5e0edfab4a-2025-03-04T17.58.20.593717.jpg'
    
    Returns:
        str: Timestamp in format '2025-03-04T17.58.20.593717'
    """
    basename = os.path.basename(file_path)
    # Split on the UUID part to get the timestamp
    parts = basename.split('-')
    timestamp = '-'.join(parts[-3:])[:-4]  # Join the date parts back and remove .jpg
    # convert to ISO format - handle both with and without microseconds
    try:
        # Try with microseconds first
        timestamp = datetime.strptime(timestamp, '%Y-%m-%dT%H.%M.%S.%f').isoformat()
    except ValueError:
        # Fall back to format without microseconds
        timestamp = datetime.strptime(timestamp, '%Y-%m-%dT%H.%M.%S').isoformat()
    return timestamp

def process_frame(frame: np.ndarray):
    """Process frame to be used for training
    
    Args:
        frame: np.ndarray of shape (H, W, 3)
    
    Returns:
        np.ndarray of shape (128, 128, 3)
    """
    # resize to 128x128
    frame = cv2.resize(frame, (128, 128))
    return frame

def get_temp_frequencies(image_file_paths: List[str]):
    """Get temp frequencies from image file paths."""
    time_stamps = [get_timestamp_from_file_path(image_file_path) for image_file_path in image_file_paths]
    temp_frequencies = []
    for time_stamp_i, time_stamp_j in zip(time_stamps[:-1], time_stamps[1:]):
        # convert to datetime (timestamps are already in ISO format)
        time_stamp_i = datetime.fromisoformat(time_stamp_i)
        time_stamp_j = datetime.fromisoformat(time_stamp_j)
        # calculate time difference
        time_diff = time_stamp_j - time_stamp_i
        # calculate frequency
        frequency = 1 / time_diff.total_seconds()
        temp_frequencies.append(frequency)
    return temp_frequencies


def divide_into_size_limited_clusters(clusters: List[Dict], min_size: int = 24, max_size: int = 60):
    """Divide image file paths into clusters of size less than or equal to max_size."""
    filtered_clusters = []
    
    for cluster in clusters:
        if min_size <= len(cluster["images"]) <= max_size:
            filtered_clusters.append(cluster)
        else: 
            # split into max_size clusters
            for i in range(0, len(cluster["images"]), max_size):
                end_idx = min(i + max_size, len(cluster["images"]))
                chunk_images = cluster["images"][i:end_idx]
                
                # Skip chunks that are smaller than min_size
                if len(chunk_images) < min_size:
                    continue
                    
                start_frame_path = chunk_images[0]
                end_frame_path = chunk_images[-1]
                start_frame_timestamp = get_timestamp_from_file_path(start_frame_path)
                end_frame_timestamp = get_timestamp_from_file_path(end_frame_path)
                # add to filtered_clusters
                filtered_clusters.append({
                    "images": chunk_images,
                    "start_time": start_frame_timestamp,
                    "end_time": end_frame_timestamp
                })
    
    return filtered_clusters

def cluster_image_file_paths_by_timestamp(
    image_file_paths_by_timestamp: List[str],
    time_stamps: List[str],
    batch_id: str,
) -> List[Dict[str, List]]:
    """Cluster image file paths by timestamp
    
    Args:
        image_file_paths_by_timestamp: List of image file paths sorted by timestamp
        time_stamps: List of ISO format timestamps
        batch_id: String identifier for the batch of images
    
    Returns:
        List of dictionaries containing 'images', 'start_time', and 'end_time' keys
    """
    
    if not image_file_paths_by_timestamp or not time_stamps:
        raise ValueError("Input lists cannot be empty")
    if len(image_file_paths_by_timestamp) != len(time_stamps):
        raise ValueError("Image paths and timestamps lists must have the same length")
    
    expected_diff = 2
    TIME_GAP_THRESHOLD = expected_diff * 5
    MIN_CLUSTER_SIZE = 4
    
    times = [datetime.fromisoformat(ts) for ts in time_stamps]
    clusters = []
    current_cluster = [image_file_paths_by_timestamp[0]]

    for i in range(len(times) - 1):
        time_diff = (times[i + 1] - times[i]).total_seconds()
        if time_diff > TIME_GAP_THRESHOLD:
            if len(current_cluster) >= MIN_CLUSTER_SIZE:
                cluster_start = times[image_file_paths_by_timestamp.index(current_cluster[0])]
                cluster_end = times[image_file_paths_by_timestamp.index(current_cluster[-1])]
                clusters.append({
                    "images": current_cluster,
                    "start_time": cluster_start,
                    "end_time": cluster_end
                })
            current_cluster = [image_file_paths_by_timestamp[i + 1]]
        else:   
            current_cluster.append(image_file_paths_by_timestamp[i + 1])

    if current_cluster and len(current_cluster) >= MIN_CLUSTER_SIZE:
        cluster_start = times[image_file_paths_by_timestamp.index(current_cluster[0])]
        cluster_end = times[image_file_paths_by_timestamp.index(current_cluster[-1])]
        clusters.append({
            "images": current_cluster,
            "start_time": cluster_start,
            "end_time": cluster_end
        })

    print(f"\nClustering information for batch {batch_id}:")
    print(f"Expected time difference: {expected_diff:.2f} seconds")
    print(f"Gap threshold: {TIME_GAP_THRESHOLD:.2f} seconds")
    print(f"Minimum cluster size: {MIN_CLUSTER_SIZE} frames")
    print(f"Number of clusters by observation gaps: {len(clusters)}")

    return clusters
