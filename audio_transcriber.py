# transcribe audio track from video using whisper

import json
import whisper
import subprocess
from typing import List, Dict, Any
from pathlib import Path
from config_manager import get_processing_params, get_video_path, get_audio_dir


def initialize_whisper_model() -> Any:
    """Initialize Whisper large-v3 model for audio transcription"""
    import torch

    model_size = 'large-v3'  # hardcoded, not parameter

    # Check CUDA availability
    if torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA available! Using GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        device = "cpu"
        print("CUDA not available, using CPU (will be slower)")

    # Load model with proper device and settings
    model = whisper.load_model(model_size, device=device)

    print(f"Whisper model '{model_size}' loaded successfully on {device}")
    return model


def extract_audio_from_video(video_path: str, audio_dir: str) -> str:
    """Extract audio from video file using ffmpeg"""
    # Create audio directory if it doesn't exist
    Path(audio_dir).mkdir(exist_ok=True)

    # Generate audio file path
    video_filename = Path(video_path).stem
    audio_path = Path(audio_dir) / f"{video_filename}.mp3"

    # Build ffmpeg command for MP3 extraction with 192kbps quality
    ffmpeg_cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vn',  # disable video
        '-acodec', 'mp3',
        '-ab', '192k',  # 192kbps bitrate
        '-ar', '44100',  # 44.1kHz sample rate
        '-y',  # overwrite output file
        str(audio_path)
    ]

    # Execute ffmpeg command
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"FFmpeg failed: {result.stderr}"

    # Verify audio file was created
    assert audio_path.exists(), f"Audio file not created: {audio_path}"

    return str(audio_path)


def transcribe_audio(audio_path: str, model: Any, language: str) -> List[Dict[str, Any]]:
    """Transcribe audio file with word timestamps"""
    # Validate input parameters
    assert Path(audio_path).exists(), f'Audio file not found: {audio_path}'
    assert language in ['auto', 'kk', 'ru', 'en'], f'Unsupported language: {language}'

    # Perform transcription with word timestamps
    result = model.transcribe(audio_path, language=language, word_timestamps=True)
    segments = result['segments']

    return segments


def process_transcription_segments(segments: List[Dict], word_timestamps: bool) -> List[Dict[str, Any]]:
    """Process transcription segments into structured format"""
    processed_segments = []

    for segment in segments:
        processed_segment = {
            'start': segment['start'],
            'end': segment['end'],
            'text': segment['text'].strip(),
            'start_formatted': format_timestamp(segment['start']),
            'end_formatted': format_timestamp(segment['end'])
        }

        # Add word-level timestamps if requested and available
        if word_timestamps and 'words' in segment:
            processed_segment['words'] = []
            for word in segment['words']:
                word_data = {
                    'word': word['word'].strip(),
                    'start': word['start'],
                    'end': word['end'],
                    'start_formatted': format_timestamp(word['start']),
                    'end_formatted': format_timestamp(word['end'])
                }
                processed_segment['words'].append(word_data)

        processed_segments.append(processed_segment)

    return processed_segments


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    timestamp = f'{hours:02d}:{minutes:02d}:{seconds:02d}'
    return timestamp


def transcribe_video_file() -> List[Dict[str, Any]]:
    """Main function to transcribe video file with hardcoded configuration"""
    # Get configuration
    params = get_processing_params()
    language = params['whisper_language']  # 'kk' from config
    word_timestamps = params['whisper_word_timestamps']  # True from config

    # Get paths
    video_path = get_video_path()
    audio_dir = get_audio_dir()

    # Extract audio from video
    audio_path = extract_audio_from_video(video_path, audio_dir)

    # Initialize Whisper model
    model = initialize_whisper_model()

    # Transcribe audio
    segments = transcribe_audio(audio_path, model, language)

    # Process segments with word timestamps
    processed_segments = process_transcription_segments(segments, word_timestamps)

    return processed_segments