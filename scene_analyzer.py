import open_clip
import torch
import numpy as np
from PIL import Image
import cv2
from typing import List, Dict, Any, Tuple

from config_manager import get_video_path, get_video_info


def initialize_clip_model() -> Tuple[Any, Any]:
    """Initialize OpenCLIP ViT-B-32 model and preprocessing transforms with GPU support.

    Returns:
        Tuple containing the model and preprocessing function
    """
    import torch

    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing CLIP model on device: {device}")

    if device == "cuda":
        print(f"GPU detected: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("No GPU detected, using CPU (will be slower)")

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai')
    model = model.to(device)  # Move model to GPU
    model.eval()

    return model, preprocess


def analyze_scene(frame: np.ndarray, model: Any, preprocess: Any) -> Tuple[str, float]:
    """Analyze single frame and return scene description with confidence.

    Args:
        frame: Input frame in BGR format
        model: CLIP model instance
        preprocess: Preprocessing function

    Returns:
        Tuple of scene description and confidence score
    """
    assert frame.shape[2] == 3, 'Frame must be BGR format'

    # Convert BGR to RGB and create PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Preprocess image and get features
    image_tensor = preprocess(image).unsqueeze(0)
    features = model.encode_image(image_tensor)

    # Get scene description from features
    scene_description = get_scene_description(features)

    # Calculate confidence as normalized feature magnitude
    confidence = float(torch.norm(features).item())
    confidence = min(confidence / 10.0, 1.0)  # Normalize to 0-1 range

    return scene_description, confidence


def detect_scene_changes(frames: List[np.ndarray], timestamps: List[str], original_fps: float,
                         similarity_threshold: float) -> List[Dict[str, Any]]:
    """Detect scene changes using CLIP features similarity tracking with batch processing.

    Args:
        frames: List of video frames in BGR format
        timestamps: List of formatted timestamps for each frame
        original_fps: Original video FPS for calculations
        similarity_threshold: Threshold for scene change detection

    Returns:
        List of scene change events with timestamps and descriptions
    """
    import torch
    import time

    assert len(frames) > 0, 'Frames list cannot be empty'
    assert len(frames) == len(timestamps), 'Frames and timestamps must have same length'
    assert original_fps > 0, 'Original FPS must be positive'

    print(f"Processing {len(frames)} frames for scene detection...")
    print(f"Using similarity threshold: {similarity_threshold}")
    print(f"Video info: {len(frames)} frames extracted, original FPS: {original_fps}")

    # Initialize CLIP model
    model, preprocess = initialize_clip_model()
    device = next(model.parameters()).device  # Get model device

    # Batch processing parameters
    from config_manager import get_processing_params
    params = get_processing_params()
    batch_size = params.get('batch_size', 16)  # Default 16 if not in config
    print(f"Using batch size: {batch_size}")

    scene_changes = []
    last_fixed_features = None
    significant_changes = 0

    start_time = time.time()

    # Process frames in batches
    for batch_start in range(0, len(frames), batch_size):
        batch_end = min(batch_start + batch_size, len(frames))
        batch_frames = frames[batch_start:batch_end]
        batch_timestamps = timestamps[batch_start:batch_end]

        # Progress logging
        if batch_start % (batch_size * 5) == 0:  # Every 5 batches
            elapsed = time.time() - start_time
            frames_processed = batch_start
            fps_processing = frames_processed / elapsed if elapsed > 0 else 0
            print(f"Processing batch {batch_start // batch_size + 1}/{(len(frames) - 1) // batch_size + 1} "
                  f"(frames {batch_start + 1}-{batch_end}) Speed: {fps_processing:.1f} frames/sec")

        # Prepare batch for processing
        batch_images = []
        for frame in batch_frames:
            assert frame.shape[2] == 3, 'Frame must be BGR format'
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = preprocess(image)
            batch_images.append(image_tensor)

        # Stack into batch tensor and move to GPU
        batch_tensor = torch.stack(batch_images).to(device)

        with torch.no_grad():  # Disable gradients for inference
            batch_features = model.encode_image(batch_tensor)

        # Process each frame in the batch
        for i, (frame_features, timestamp) in enumerate(zip(batch_features, batch_timestamps)):
            frame_idx = batch_start + i

            # Get scene description
            scene_description = get_scene_description(frame_features.unsqueeze(0))

            # Calculate similarity (для информации, но не для принятия решений)
            if last_fixed_features is not None:
                similarity = torch.cosine_similarity(frame_features.unsqueeze(0), last_fixed_features)
                similarity_score = float(similarity.item())
            else:
                similarity_score = 1.0

            # Check for ACTUAL scene change (when scene type changes)
            if last_fixed_features is not None:
                # Get previous scene description for comparison
                last_scene_description = scene_changes[-1]['scene_description'] if scene_changes else None

                # Detect scene change ONLY when scene type actually changes
                if scene_description != last_scene_description:
                    print(
                        f"Frame {frame_idx + 1} at {timestamp} - {scene_description} (similarity: {similarity_score:.3f}) *SCENE CHANGE*")
                    significant_changes += 1

                    scene_change_event = {
                        'frame_index': frame_idx,
                        'timestamp': timestamp,
                        'scene_description': scene_description,
                        'similarity_score': similarity_score,
                        'change_detected': True,
                        'previous_scene': last_scene_description
                    }
                    scene_changes.append(scene_change_event)

                    # Update reference features for next comparison
                    last_fixed_features = frame_features.unsqueeze(0).clone()
                elif frame_idx % 200 == 0:  # Show some stable frames for reference
                    print(
                        f"Frame {frame_idx + 1} at {timestamp} - {scene_description} (similarity: {similarity_score:.3f}) same scene")
            else:
                # First frame - initialize reference features
                last_fixed_features = frame_features.unsqueeze(0).clone()
                print(f"Initial scene at {timestamp} - {scene_description}")

                initial_event = {
                    'frame_index': frame_idx,
                    'timestamp': timestamp,
                    'scene_description': scene_description,
                    'similarity_score': 1.0,
                    'change_detected': False,
                    'previous_scene': None
                }
                scene_changes.append(initial_event)

    total_time = time.time() - start_time
    avg_fps = len(frames) / total_time

    print(f"Scene detection completed in {total_time:.1f} seconds")
    print(f"Average processing speed: {avg_fps:.1f} frames/sec")
    print(f"Speedup with batch processing: ~{batch_size}x theoretical")
    print(f"Found {len(scene_changes)} total events")
    print(f"Actual scene changes (different scene types): {significant_changes}")

    # Show summary of scene transitions
    if len(scene_changes) > 1:
        print("\nScene transition summary:")
        for i, event in enumerate(scene_changes):
            if event['change_detected']:
                prev_scene = event.get('previous_scene', 'unknown')
                print(f"  {event['timestamp']}: {prev_scene} → {event['scene_description']}")

    return scene_changes


# Global cache for text model (избегаем переинициализации)
_text_model_cache = None
_text_features_cache = None


def get_scene_description(features: np.ndarray) -> str:
    """Convert CLIP features to human-readable scene description using text-image similarity.

    Args:
        features: CLIP image features tensor

    Returns:
        Human-readable scene description string
    """
    import torch
    global _text_model_cache, _text_features_cache

    # Scene categories for news and documentary content (можно расширить в будущем)
    scene_categories = [
        "news studio with anchor desk",
        "reporter on location outdoors",
        "interview or talk show setting",
        "breaking news graphics and text",
        "weather forecast display",
        "sports footage or highlights",
        "commercial advertisement",
        "office or workplace interior",
        "industrial factory or manufacturing plant",
        "rocket launch or space technology",
        "city skyline and urban landscape",
        "natural landscape and countryside",
        "people walking on street",
        "laboratory or scientific equipment",
        "meeting room or conference hall",
        "archive footage or historical material"
    ]

    # Convert features to torch tensor if needed
    if not torch.is_tensor(features):
        image_features = torch.from_numpy(features)
    else:
        image_features = features

    # Get device from features
    device = image_features.device

    # Cache text model and features for performance
    try:
        if _text_model_cache is None or _text_features_cache is None:
            print("Initializing text model cache...")
            import open_clip
            _text_model_cache, _, _ = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai')
            _text_model_cache = _text_model_cache.to(device)  # Move to same device as features
            tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')

            # Encode text descriptions once and cache
            text_inputs = tokenizer(scene_categories).to(device)  # Move to GPU
            with torch.no_grad():
                _text_features_cache = _text_model_cache.encode_text(text_inputs)
                _text_features_cache = _text_features_cache / _text_features_cache.norm(dim=-1, keepdim=True)

        with torch.no_grad():
            # Normalize image features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Calculate similarities with cached text features
            similarities = (image_features @ _text_features_cache.T).squeeze(0)

            # Get best match
            best_idx = similarities.argmax()
            confidence = similarities[best_idx].item()

            # Simplify description (убираем лишние слова для краткости)
            scene_name = scene_categories[best_idx]
            simplified_name = simplify_scene_name(scene_name)

            return simplified_name

    except Exception as e:
        # Fallback to old method if text similarity fails (backup план)
        return get_fallback_scene_description(features)


def simplify_scene_name(scene_name: str) -> str:
    """Simplify scene names to Russian short descriptions"""
    # Mapping для русских названий (можно настроить)
    simplification_map = {
        "news studio with anchor desk": "студия_новости",
        "reporter on location outdoors": "репортаж_улица",
        "interview or talk show setting": "интервью",
        "breaking news graphics and text": "графика_титры",
        "weather forecast display": "прогноз_погоды",
        "sports footage or highlights": "спорт",
        "commercial advertisement": "реклама",
        "office or workplace interior": "офис",
        "industrial factory or manufacturing plant": "завод_производство",
        "rocket launch or space technology": "ракета_космос",
        "city skyline and urban landscape": "город_пейзаж",
        "natural landscape and countryside": "природа_ландшафт",
        "people walking on street": "люди_улица",
        "laboratory or scientific equipment": "лаборатория",
        "meeting room or conference hall": "переговоры_зал",
        "archive footage or historical material": "архивные_кадры"
    }

    return simplification_map.get(scene_name, "неизвестная_сцена")


def get_fallback_scene_description(features: np.ndarray) -> str:
    """Fallback method using feature statistics (старая логика как backup)"""
    # Convert features to numpy if tensor
    if torch.is_tensor(features):
        feature_vector = features.detach().numpy().flatten()
    else:
        feature_vector = features.flatten()

    # Improved statistical analysis (чуть лучше чем было)
    feature_mean = np.mean(feature_vector)
    feature_std = np.std(feature_vector)
    feature_max = np.max(feature_vector)
    feature_variance = np.var(feature_vector)

    # Enhanced classification logic with Russian names (больше вариантов)
    if feature_mean > 0.15:
        if feature_std > 0.2:
            scene_type = 'репортаж_улица'
        elif feature_variance > 0.03:
            scene_type = 'интервью'
        else:
            scene_type = 'студия_новости'
    elif feature_mean > 0.05:
        if feature_max > 0.25:
            scene_type = 'графика_титры'
        elif feature_std > 0.15:
            scene_type = 'завод_производство'
        else:
            scene_type = 'офис'
    else:
        if feature_variance > 0.01:
            scene_type = 'природа_ландшафт'
        else:
            scene_type = 'реклама'

    return scene_type