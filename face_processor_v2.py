"""
Face detection and clustering module using InsightFace for high performance.
Provides face detection, embedding extraction, attribute analysis, and similarity-based clustering.

Based on InsightFace (SCRFD detector + ArcFace embeddings) + FER for emotion detection.
No TensorFlow dependency for main processing.
"""

import numpy as np
import cv2
import time
from typing import List, Dict, Any, Tuple, Optional
from config_manager import get_processing_params

# Global model cache
_face_analyzer = None
_emotion_detector = None
_current_det_size = None


def get_optimal_det_size(frame_width: int, frame_height: int, max_size: int = 1920) -> Tuple[int, int]:
    """
    Calculate optimal detection size based on frame resolution.

    Uses original resolution up to max_size, then scales down proportionally.
    Rounds to multiple of 32 (InsightFace requirement).

    Args:
        frame_width: Original frame width
        frame_height: Original frame height
        max_size: Maximum detection size (default 1920 for Full HD)

    Returns:
        Tuple of (det_width, det_height) rounded to multiple of 32
    """
    # Use original if within limits
    if frame_width <= max_size and frame_height <= max_size:
        det_w = frame_width
        det_h = frame_height
    else:
        # Scale down proportionally
        scale = max_size / max(frame_width, frame_height)
        det_w = int(frame_width * scale)
        det_h = int(frame_height * scale)

    # Round to multiple of 32 (InsightFace requirement)
    det_w = (det_w // 32) * 32
    det_h = (det_h // 32) * 32

    # Minimum size 128
    det_w = max(128, det_w)
    det_h = max(128, det_h)

    return (det_w, det_h)


def initialize_face_models(det_size: Optional[Tuple[int, int]] = None):
    """
    Initialize InsightFace and FER models with GPU support.

    Args:
        det_size: Optional detection size tuple (width, height).
                  If None, uses default (640, 640).
                  Use get_optimal_det_size() to calculate from frame size.
    """
    global _face_analyzer, _emotion_detector, _current_det_size

    if det_size is None:
        det_size = (640, 640)

    # Skip re-initialization if same det_size
    if _face_analyzer is not None and _current_det_size == det_size:
        return

    print(f"Initializing InsightFace models with det_size={det_size}...")

    from insightface.app import FaceAnalysis

    _face_analyzer = FaceAnalysis(
        name='buffalo_l',
        root='~/.insightface',
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )

    # ctx_id: 0 = GPU, -1 = CPU
    _face_analyzer.prepare(ctx_id=0, det_size=det_size)
    _current_det_size = det_size

    print(f"InsightFace initialized with {len(_face_analyzer.models)} models:")
    for name in _face_analyzer.models:
        print(f"  - {name}")

    # FER emotion detector
    print("Initializing FER emotion detector...")
    from fer.fer import FER
    _emotion_detector = FER(mtcnn=False)  # Don't use MTCNN, we have InsightFace for detection
    print("FER initialized")

    print("All face models ready!")


def detect_faces_single_frame(frame: np.ndarray, min_face_size: int) -> List[Dict[str, Any]]:
    """
    Detect faces in single frame using InsightFace SCRFD detector.

    Args:
        frame: Input BGR image frame
        min_face_size: Minimum face size for filtering

    Returns:
        List of face data dictionaries with coordinates, embeddings, and attributes
    """
    global _face_analyzer

    assert frame.shape[2] == 3, 'Frame must be BGR format'
    assert min_face_size > 0, 'Minimum face size must be positive'

    if _face_analyzer is None:
        initialize_face_models()

    # InsightFace expects BGR (OpenCV format) - OK
    faces = _face_analyzer.get(frame)

    detected_faces = []
    for i, face in enumerate(faces):
        # face.bbox = [x1, y1, x2, y2]
        x1, y1, x2, y2 = face.bbox.astype(int)
        w = x2 - x1
        h = y2 - y1

        # Size filter
        if w >= min_face_size and h >= min_face_size:
            # Clamp coordinates to frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            face_data = {
                'coordinates': (x1, y1, w, h),
                'region': frame[y1:y2, x1:x2].copy(),
                'size': (w, h),
                'face_id': f'face_{i:03d}',
                'center_x': x1 + w // 2,
                'center_y': y1 + h // 2,
                # InsightFace provides all attributes!
                'embedding': face.normed_embedding.tolist(),  # 512D normalized
                'age': int(face.age),
                'gender': 'Man' if face.gender == 1 else 'Woman',
                'det_score': float(face.det_score),
            }
            detected_faces.append(face_data)

    return detected_faces


def add_emotion_to_faces(faces_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add emotion detection using FER library"""
    global _emotion_detector

    if _emotion_detector is None:
        from fer.fer import FER
        _emotion_detector = FER(mtcnn=False)

    for face in faces_data:
        emotion = 'neutral'
        if 'region' in face and face['region'] is not None and face['region'].size > 0:
            # FER expects RGB
            region_rgb = cv2.cvtColor(face['region'], cv2.COLOR_BGR2RGB)
            result = _emotion_detector.detect_emotions(region_rgb)
            if result and len(result) > 0:
                emotions = result[0].get('emotions', {})
                if emotions:
                    emotion = max(emotions, key=emotions.get)
        face['emotion'] = emotion

    return faces_data


def cluster_faces_to_persons(
    faces: List[Dict[str, Any]],
    known_persons: List[Dict[str, Any]],
    similarity_threshold: float
) -> List[Dict[str, Any]]:
    """
    Cluster faces to existing persons or create new persons.

    Args:
        faces: List of face data dictionaries with embeddings
        known_persons: Global list of known persons (modified in place)
        similarity_threshold: Threshold for person identification (0-1)

    Returns:
        List of face data with assigned person_ids
    """
    assert 0.0 < similarity_threshold < 1.0, 'Similarity threshold must be between 0 and 1'

    clustered_faces = []

    for face in faces:
        face_embedding = np.array(face['embedding'])

        # Find matching person
        matched_person = None
        best_similarity = 0.0

        for person in known_persons:
            similarity = cosine_similarity(face_embedding, np.array(person['representative_embedding']))
            if similarity >= similarity_threshold and similarity > best_similarity:
                matched_person = person
                best_similarity = similarity

        if matched_person:
            matched_person['face_count'] += 1
            person_id = matched_person['person_id']
        else:
            # Create new person
            person_id = f'person_{len(known_persons) + 1:02d}'
            new_person = {
                'person_id': person_id,
                'representative_embedding': face['embedding'],
                'face_count': 1,
                'first_seen': face.get('timestamp', 0.0)
            }
            known_persons.append(new_person)

        # Create clustered face data
        clustered_face = {
            'person_id': person_id,
            'coordinates': face['coordinates'],
            'center_x': face['center_x'],
            'center_y': face['center_y'],
            'size': face['size'],
            'age': face['age'],
            'gender': face['gender'],
            'emotion': face.get('emotion', 'neutral'),
            'det_score': face['det_score'],
            'embedding': face['embedding']
        }
        clustered_faces.append(clustered_face)

    return clustered_faces


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings"""
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def detect_faces_batch(
    frames: List[np.ndarray],
    timestamps: List[float]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Detect and cluster faces across all frames.

    Args:
        frames: List of input BGR image frames
        timestamps: List of timestamps (seconds) for each frame

    Returns:
        Tuple of (known_persons, all_detections)
    """
    assert len(frames) > 0, 'Frames list cannot be empty'
    assert len(frames) == len(timestamps), 'Frames and timestamps must have same length'

    print(f"Processing {len(frames)} frames for face detection...")

    # Get configuration
    params = get_processing_params()
    min_face_size = params.get('face_min_size', 50)
    similarity_threshold = params.get('face_similarity_threshold', 0.7)

    print(f"Using min face size: {min_face_size}, similarity threshold: {similarity_threshold}")

    # Initialize models
    initialize_face_models()

    # Storage
    known_persons = []
    all_detections = []
    start_time = time.time()

    for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
        if i % 100 == 0:
            elapsed = time.time() - start_time
            fps = i / elapsed if elapsed > 0 else 0
            print(f"Processing frame {i+1}/{len(frames)} ({fps:.1f} fps)")

        # Detect faces
        frame_faces = detect_faces_single_frame(frame, min_face_size)

        if frame_faces:
            # Add emotion detection
            frame_faces = add_emotion_to_faces(frame_faces)

            # Add timestamp
            for face in frame_faces:
                face['timestamp'] = timestamp
                face['frame_idx'] = i

            # Cluster to persons
            clustered_faces = cluster_faces_to_persons(frame_faces, known_persons, similarity_threshold)

            # Store detections
            all_detections.extend(clustered_faces)

    total_time = time.time() - start_time
    avg_fps = len(frames) / total_time if total_time > 0 else 0

    print(f"Face detection completed in {total_time:.1f}s ({avg_fps:.1f} fps)")
    print(f"Total faces detected: {len(all_detections)}")
    print(f"Unique persons identified: {len(known_persons)}")

    return known_persons, all_detections


if __name__ == "__main__":
    # Quick test
    print("=== Face Processor V2 Test ===")

    # Initialize models
    initialize_face_models()

    # Create test image with face-like region
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_frame, (200, 100), (400, 350), (128, 128, 128), -1)

    print("\nTesting face detection on synthetic image...")
    faces = detect_faces_single_frame(test_frame, min_face_size=50)
    print(f"Detected {len(faces)} faces (expected 0 on synthetic image)")

    print("\nFace Processor V2 is working!")
