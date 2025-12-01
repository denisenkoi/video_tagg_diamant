import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from config_manager import get_output_dir


def export_scenes_csv(scene_events: List[Dict[str, Any]], output_path: str) -> None:
    """Export scene analysis results to CSV file with UTF-8 encoding."""
    # Hardcoded variables
    encoding = 'utf-8'  # hardcoded, not parameter
    index = False  # hardcoded, not parameter

    # Validation
    assert len(scene_events) > 0, 'No scene events to export'
    assert Path(output_path).suffix == '.csv', 'Output file must be CSV'

    # Create DataFrame and export to CSV
    df = pd.DataFrame(scene_events)
    df.to_csv(output_path, index=index, encoding=encoding, columns=df.columns[1:3])


def export_objects_csv(object_events: List[Dict[str, Any]], output_path: str) -> None:
    """Export object detection results to CSV file with UTF-8 encoding."""
    # Hardcoded variables
    encoding = 'utf-8'  # hardcoded, not parameter
    index = False  # hardcoded, not parameter

    # Validation
    assert len(object_events) > 0, 'No object events to export'
    assert Path(output_path).suffix == '.csv', 'Output file must be CSV'

    # Create DataFrame and export to CSV
    df = pd.DataFrame(object_events)
    df.to_csv(output_path, index=index, encoding=encoding)


def export_faces_csv(face_events: List[Dict[str, Any]], output_path: str) -> None:
    """Export face detection results to CSV file with UTF-8 encoding."""
    # Hardcoded variables
    encoding = 'utf-8'  # hardcoded, not parameter
    index = False  # hardcoded, not parameter

    # Validation
    assert len(face_events) > 0, 'No face events to export'
    assert Path(output_path).suffix == '.csv', 'Output file must be CSV'

    # Create DataFrame and export to CSV
    df = pd.DataFrame(face_events)
    df.to_csv(output_path, index=index, encoding=encoding)


def export_face_timeline_csv(timeline_events: List[Dict[str, Any]], output_path: str) -> None:
    """Export face timeline events to CSV file with UTF-8 encoding."""
    # Hardcoded variables
    encoding = 'utf-8'  # hardcoded, not parameter
    index = False  # hardcoded, not parameter

    # Validation
    assert len(timeline_events) > 0, 'No timeline events to export'
    assert Path(output_path).suffix == '.csv', 'Output file must be CSV'

    # Create DataFrame and export to CSV
    df = pd.DataFrame(timeline_events)
    df.to_csv(output_path, index=index, encoding=encoding)


def export_transcription_csv(transcription_data: List[Dict[str, Any]], output_path: str) -> None:
    """Export transcription results to CSV file with UTF-8 encoding."""
    # Hardcoded variables
    encoding = 'utf-8'  # hardcoded, not parameter
    index = False  # hardcoded, not parameter
    ensure_ascii = False  # hardcoded, not parameter

    # Validation
    assert len(transcription_data) > 0, 'No transcription data to export'
    assert Path(output_path).suffix == '.csv', 'Output file must be CSV'

    # Create DataFrame and export to CSV with proper Unicode support
    df = pd.DataFrame(transcription_data)
    df.to_csv(output_path, index=index, encoding=encoding)


def create_output_filename(video_path: str, suffix: str) -> str:
    """Create output filename from video name and suffix."""
    # Extract video name without extension
    video_name = Path(video_path).stem

    # Create filename with suffix
    filename = f'{video_name}_{suffix}.csv'

    # Get full output path
    output_path = Path(get_output_dir()) / filename

    return str(output_path)