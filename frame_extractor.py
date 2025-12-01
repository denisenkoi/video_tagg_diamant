import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from config_manager import get_video_path, get_processing_params


def extract_frames(video_path: str, fps: int) -> List[np.ndarray]:
    """Extract frames from video at specified FPS rate"""
    # Validate input parameters
    assert fps > 0, 'FPS must be positive'

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f'Cannot open video: {video_path}'

    # Get original video FPS
    original_fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate frame interval for sampling
    frame_interval = int(original_fps / fps)

    frames = []
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Extract frame at specified interval
        if frame_count % frame_interval == 0:
            frames.append(frame)

        frame_count += 1

    cap.release()

    # Validate extraction results
    assert len(frames) > 0, 'No frames extracted from video'

    return frames


def extract_frames_with_timestamps(video_path: str, fps: int) -> Tuple[List[np.ndarray], List[str], float]:
    """Extract frames with timestamps and return original FPS"""
    # Validate input parameters
    assert fps > 0, 'FPS must be positive'

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f'Cannot open video: {video_path}'

    # Get original video FPS
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video original FPS: {original_fps}, extracting at {fps} FPS")

    # Calculate frame interval for sampling
    frame_interval = int(original_fps / fps)
    print(f"Frame interval: {frame_interval} (taking every {frame_interval}th frame)")

    frames = []
    timestamps = []
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Extract frame at specified interval
        if frame_count % frame_interval == 0:
            frames.append(frame)

            # Calculate timestamp based on actual frame position in video
            timestamp_seconds = frame_count / original_fps
            timestamp_formatted = seconds_to_timestamp(timestamp_seconds)
            timestamps.append(timestamp_formatted)

        frame_count += 1

    cap.release()

    # Validate extraction results
    assert len(frames) > 0, 'No frames extracted from video'
    assert len(frames) == len(timestamps), 'Frames and timestamps count mismatch'

    print(f"Extracted {len(frames)} frames from {frame_count} total frames")
    print(f"First timestamp: {timestamps[0]}, Last timestamp: {timestamps[-1]}")

    return frames, timestamps, original_fps


def get_video_info(video_path: str) -> Dict[str, Any]:
    """Get video metadata information"""
    # Open video capture for metadata extraction
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f'Cannot open video: {video_path}'

    # Extract video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    # Return metadata dictionary
    video_info = {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': duration,
        'resolution': f'{width}x{height}'
    }

    return video_info


def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS timestamp format with proper rounding"""
    # Round seconds instead of truncating
    total_seconds = round(seconds)

    # Calculate hours, minutes, seconds
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = int(total_seconds % 60)

    # Format timestamp string
    timestamp = f'{hours:02d}:{minutes:02d}:{secs:02d}'
    return timestamp


def frames_to_timestamps(frame_indices: List[int], original_fps: float) -> List[str]:
    """Convert frame indices to HH:MM:SS timestamp format"""
    # Validate input parameters
    assert original_fps > 0, 'Original FPS must be positive'

    timestamps = []

    for frame_index in frame_indices:
        # Calculate total seconds from frame index
        total_seconds = frame_index / original_fps
        timestamp = seconds_to_timestamp(total_seconds)
        timestamps.append(timestamp)

    return timestamps


def process_video_frames():
    """Main processing function with hardcoded configuration"""
    # Hardcoded variables from config
    fps = 10  # hardcoded from config, not parameter

    # Get video path from config manager
    video_path = get_video_path()

    # Extract frames at specified FPS
    extracted_frames = extract_frames(video_path, fps)

    # Get video information
    video_info = get_video_info(video_path)

    # Generate frame indices for timestamp conversion
    frame_indices = list(range(0, len(extracted_frames)))

    # Convert frame indices to timestamps
    timestamps = frames_to_timestamps(frame_indices, video_info['fps'])

    return extracted_frames, video_info, timestamps