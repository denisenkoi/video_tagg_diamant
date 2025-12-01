"""
Audio Extractor - Вспомогательный модуль для извлечения аудио сегментов
"""

import subprocess
import os
from pathlib import Path
from typing import Optional


def extract_audio_segment(video_path: str, start_time: float, 
                         duration: float, output_path: str) -> bool:
    """
    Extract audio segment from video using ffmpeg
    
    Args:
        video_path: Path to input video
        start_time: Start time in seconds
        duration: Duration in seconds  
        output_path: Path for output audio file
        
    Returns:
        True if successful, False otherwise
    """
    
    try:
        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-i', video_path,           # Input video
            '-ss', str(start_time),     # Start time
            '-t', str(duration),        # Duration
            '-vn',                      # No video
            '-acodec', 'pcm_s16le',     # Audio codec
            '-ar', '16000',             # Sample rate (Whisper prefers 16kHz)
            '-ac', '1',                 # Mono channel
            '-y',                       # Overwrite output file
            output_path
        ]
        
        # Run ffmpeg
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            timeout=60  # 60 second timeout
        )
        
        if result.returncode == 0:
            # Check if output file was created and has size > 0
            if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                return True
            else:
                print(f"  ffmpeg output file is empty or missing: {output_path}")
                return False
        else:
            print(f"  ffmpeg error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ffmpeg timeout for segment {start_time}s-{start_time+duration}s")
        return False
    except Exception as e:
        print(f"  ffmpeg exception: {e}")
        return False


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available in system"""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'], 
            capture_output=True, 
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except:
        return False


def cleanup_temp_files(pattern: str = "temp_segment_*.wav") -> None:
    """Clean up temporary audio files"""
    import glob
    for file_path in glob.glob(pattern):
        try:
            os.remove(file_path)
        except:
            pass