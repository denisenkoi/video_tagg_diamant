from dotenv import load_dotenv
import os
from pathlib import Path
from typing import Dict, Any

# Global variable for current video path
_current_video_path = None


def set_video_path(video_path: str) -> None:
    """Set current video path for processing"""
    global _current_video_path

    # Validate video path
    assert Path(video_path).exists(), f'Video file not found: {video_path}'
    assert video_path.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')), 'Unsupported video format'

    _current_video_path = video_path
    print(f"Video path set to: {video_path}")


def load_config() -> Dict[str, Any]:
    """Load configuration from .env file with validation"""
    load_dotenv()

    # Load and validate video path (default from .env)
    video_path = os.environ['VIDEO_PATH']
    assert Path(video_path).exists(), f'Video file not found: {video_path}'
    assert video_path.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')), 'Unsupported video format'

    # Load and validate output directory
    output_dir = os.environ['OUTPUT_DIR']

    # Load and validate processing parameters
    scene_similarity_threshold = float(os.environ['SCENE_SIMILARITY_THRESHOLD'])
    assert 0.0 <= scene_similarity_threshold < 1.0, 'Scene similarity threshold must be between 0 and 1'

    frames_per_second = int(os.environ['FRAMES_PER_SECOND'])
    assert frames_per_second > 0, 'Invalid FPS'

    # Load and validate face similarity threshold
    face_similarity_threshold = float(os.environ['FACE_SIMILARITY_THRESHOLD'])
    assert 0.0 < face_similarity_threshold < 1.0, 'Face similarity threshold must be between 0 and 1'

    # Load object detection confidence threshold
    object_confidence_threshold = float(os.environ['OBJECT_CONFIDENCE_THRESHOLD'])
    assert 0.0 < object_confidence_threshold < 1.0, 'Object confidence threshold must be between 0 and 1'

    # Load face detection minimum size
    face_min_size = int(os.environ['FACE_MIN_SIZE'])
    assert face_min_size > 0, 'Face minimum size must be positive'

    # Translation settings (optional)
    enable_translation = os.environ.get('ENABLE_TRANSLATION', 'false').lower() == 'true'

    # Return complete configuration
    config = {
        'video_path': video_path,
        'output_dir': output_dir,
        'scene_similarity_threshold': scene_similarity_threshold,
        'frames_per_second': frames_per_second,
        'face_similarity_threshold': face_similarity_threshold,
        'object_confidence_threshold': object_confidence_threshold,
        'face_min_size': face_min_size,
        'enable_translation': enable_translation
    }

    return config


def get_video_path() -> str:
    """Returns current video file path (from set_video_path or .env default)"""
    global _current_video_path

    if _current_video_path:
        # Use currently set video path
        video_path = _current_video_path
    else:
        # Use default from .env
        load_dotenv()
        video_path = os.environ['VIDEO_PATH']

    assert Path(video_path).exists(), f'Video file not found: {video_path}'
    assert video_path.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')), 'Unsupported video format'
    return video_path


def get_video_info() -> Dict[str, Any]:
    """Get video metadata information using frame_extractor"""
    from frame_extractor import get_video_info as extract_video_info
    video_path = get_video_path()
    return extract_video_info(video_path)


def get_audio_dir() -> str:
    """Returns audio directory path"""
    load_dotenv()
    audio_dir = os.environ['AUDIO_DIR']
    return audio_dir


def get_output_dir() -> str:
    """Returns output directory path"""
    load_dotenv()
    output_dir = os.environ['OUTPUT_DIR']
    return output_dir


def get_processing_params() -> Dict[str, Any]:
    """Returns all processing parameters including face clustering threshold"""
    load_dotenv()

    # Load and validate scene similarity threshold
    scene_similarity_threshold = float(os.environ['SCENE_SIMILARITY_THRESHOLD'])
    assert 0.0 <= scene_similarity_threshold < 1.0, 'Scene similarity threshold must be between 0 and 1'

    # Load and validate frames per second
    frames_per_second = int(os.environ['FRAMES_PER_SECOND'])
    assert frames_per_second > 0, 'Invalid FPS'

    # Load and validate face similarity threshold
    face_similarity_threshold = float(os.environ['FACE_SIMILARITY_THRESHOLD'])
    assert 0.0 < face_similarity_threshold < 1.0, 'Face similarity threshold must be between 0 and 1'

    # Load object detection confidence threshold
    object_confidence_threshold = float(os.environ['OBJECT_CONFIDENCE_THRESHOLD'])
    assert 0.0 < object_confidence_threshold < 1.0, 'Object confidence threshold must be between 0 and 1'

    # Load face detection minimum size
    face_min_size = int(os.environ['FACE_MIN_SIZE'])
    assert face_min_size > 0, 'Face minimum size must be positive'

    # Load batch processing size
    batch_size = int(os.environ.get('BATCH_SIZE', 16))
    assert batch_size > 0, 'Batch size must be positive'

    whisper_model = os.environ['WHISPER_MODEL']
    whisper_language = os.environ['WHISPER_LANGUAGE']
    whisper_word_timestamps = os.environ['WHISPER_WORD_TIMESTAMPS'].lower() == 'true'

    # Translation settings (optional)
    enable_translation = os.environ.get('ENABLE_TRANSLATION', 'false').lower() == 'true'
    translation_api_base = os.environ.get('TRANSLATION_API_BASE', 'http://127.0.0.1:5000')
    translation_model = os.environ.get('TRANSLATION_MODEL', 'qwen2.5-coder-14b-instruct')

    processing_params = {
        'scene_similarity_threshold': scene_similarity_threshold,
        'frames_per_second': frames_per_second,
        'face_similarity_threshold': face_similarity_threshold,
        'object_confidence_threshold': object_confidence_threshold,
        'face_min_size': face_min_size,
        'batch_size': batch_size,
        'whisper_model': whisper_model,
        'whisper_language': whisper_language,
        'whisper_word_timestamps': whisper_word_timestamps,
        'enable_translation': enable_translation,
        'translation_api_base': translation_api_base,
        'translation_model': translation_model
    }

    return processing_params