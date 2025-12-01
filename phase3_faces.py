"""
Phase 3: Face Detection with On-the-fly Clustering

Detects faces, clusters them in real-time, and saves to database with:
- face_thumbnail (128px tight crop for embedding)
- face_crop_context (200px +25% padding for UI display)

Usage:
    python phase3_faces.py              # Process all pending videos
    python phase3_faces.py --video X    # Process single video
    python phase3_faces.py --all        # Force reprocess all videos
"""

import cv2
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

from face_processor_v2 import (
    initialize_face_models,
    detect_faces_single_frame,
    get_optimal_det_size,
    cosine_similarity
)
from db_manager_postgres import PostgresDBManager


def encode_image(image: np.ndarray, max_size: int, quality: int = 85) -> Optional[bytes]:
    """Encode image as JPEG with size limit"""
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer.tobytes()


def extract_face_crops(frame: np.ndarray, bbox: tuple, padding: float = 0.25) -> tuple:
    """
    Extract tight crop and context crop from frame.

    Args:
        frame: Full video frame
        bbox: (x, y, w, h) face bounding box
        padding: Padding percentage for context crop (0.25 = 25%)

    Returns:
        (tight_crop, context_crop) - both as numpy arrays
    """
    frame_h, frame_w = frame.shape[:2]
    x, y, w, h = bbox

    # Tight crop (original bbox)
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(frame_w, x + w)
    y2 = min(frame_h, y + h)
    tight_crop = frame[y1:y2, x1:x2].copy()

    # Context crop (+25% padding)
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    cx1 = max(0, x - pad_w)
    cy1 = max(0, y - pad_h)
    cx2 = min(frame_w, x + w + pad_w)
    cy2 = min(frame_h, y + h + pad_h)
    context_crop = frame[cy1:cy2, cx1:cx2].copy()

    return tight_crop, context_crop


def cluster_face_to_person(
    face_embedding: List[float],
    known_persons: List[Dict[str, Any]],
    similarity_threshold: float,
    timestamp: float
) -> str:
    """
    Assign face to existing person or create new one.

    Args:
        face_embedding: 512D face embedding
        known_persons: List of known persons (modified in place)
        similarity_threshold: Cosine similarity threshold
        timestamp: Current timestamp for first_seen

    Returns:
        person_id string
    """
    embedding = np.array(face_embedding)

    # Find best matching person
    best_person = None
    best_similarity = 0.0

    for person in known_persons:
        similarity = cosine_similarity(embedding, np.array(person['representative_embedding']))
        if similarity >= similarity_threshold and similarity > best_similarity:
            best_person = person
            best_similarity = similarity

    if best_person:
        best_person['face_count'] += 1
        return best_person['person_id']
    else:
        # Create new person
        person_id = f'person_{len(known_persons) + 1:03d}'
        new_person = {
            'person_id': person_id,
            'representative_embedding': face_embedding,
            'face_count': 1,
            'first_seen': timestamp
        }
        known_persons.append(new_person)
        return person_id


def process_video(
    video_path: str,
    db: PostgresDBManager,
    min_face_size: int = 50,
    similarity_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Process video for face detection with on-the-fly clustering.

    Args:
        video_path: Path to video file
        db: Database manager
        min_face_size: Minimum face size in pixels
        similarity_threshold: Threshold for person matching

    Returns:
        Processing statistics
    """
    video_name = Path(video_path).stem
    print(f"\n{'='*60}")
    print(f"Phase 3: Face Detection - {video_name}")
    print(f"{'='*60}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return {'error': 'Cannot open video'}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    # Frame skip = fps (1 frame per second)
    frame_skip = max(1, int(fps))

    print(f"Video: {frame_width}x{frame_height}, {fps:.1f}fps, {duration:.1f}s")
    print(f"Processing: 1 frame/second (skip={frame_skip})")
    print(f"Min face size: {min_face_size}px, similarity threshold: {similarity_threshold}")

    # Initialize InsightFace
    det_size = get_optimal_det_size(frame_width, frame_height, max_size=1920)
    print(f"Detection size: {det_size}")
    initialize_face_models(det_size=det_size)

    # Update status
    db.update_face_status(video_name, face_status='processing', face_start_time=datetime.now())

    # Processing storage
    known_persons = []
    all_detections = []
    thumbnails = []
    context_crops = []

    frame_idx = 0
    faces_found = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        timestamp = frame_idx / fps

        # Progress logging
        if frame_idx % (frame_skip * 60) == 0:  # Every 60 seconds of video
            elapsed = time.time() - start_time
            progress = frame_idx / total_frames * 100
            print(f"  {progress:.1f}% | {timestamp:.0f}s/{duration:.0f}s | {faces_found} faces | {len(known_persons)} persons | {elapsed:.1f}s")

        # Detect faces
        faces = detect_faces_single_frame(frame, min_face_size)

        for face in faces:
            bbox = face['coordinates']

            # Cluster to person
            person_id = cluster_face_to_person(
                face['embedding'],
                known_persons,
                similarity_threshold,
                timestamp
            )

            # Extract crops
            tight_crop, context_crop = extract_face_crops(frame, bbox, padding=0.25)

            # Prepare detection record
            detection = {
                'video_name': video_name,
                'person_id': person_id,
                'timestamp': timestamp,
                'bbox': bbox,
                'age': face['age'],
                'gender': face['gender'],
                'emotion': None,
                'det_score': face['det_score'],
                'embedding': face['embedding']
            }
            all_detections.append(detection)

            # Encode images
            thumb_bytes = encode_image(tight_crop, max_size=128)
            context_bytes = encode_image(context_crop, max_size=200)
            thumbnails.append(thumb_bytes)
            context_crops.append(context_bytes)

            faces_found += 1

        frame_idx += 1

    cap.release()

    total_time = time.time() - start_time
    print(f"\nDetection complete: {faces_found} faces, {len(known_persons)} persons in {total_time:.1f}s")

    # Save to database
    if all_detections:
        print(f"Saving to database...")

        # First save persons (FK constraint requires persons exist before detections)
        for person in known_persons:
            db.create_person(
                person_id=person['person_id'],
                video_name=video_name,
                first_seen_time=person['first_seen'],
                representative_embedding=person['representative_embedding']
            )
        print(f"  Saved {len(known_persons)} persons")

        # Batch insert detections
        db._ensure_connection()
        with db.conn.cursor() as cur:
            from psycopg2.extras import execute_values

            values = []
            for d in all_detections:
                bbox = d['bbox']
                values.append((
                    d['video_name'],
                    d['person_id'],
                    float(d['timestamp']),
                    int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                    int(d['age']) if d.get('age') is not None else None,
                    d.get('gender'),
                    d.get('emotion'),
                    float(d['det_score']) if d.get('det_score') is not None else None,
                    d.get('embedding')
                ))

            # Insert and get IDs back
            result = execute_values(cur, """
                INSERT INTO face_detections
                (video_name, person_id, timestamp, bbox_x, bbox_y, bbox_w, bbox_h,
                 age, gender, emotion, det_score, embedding)
                VALUES %s
                RETURNING id
            """, values, fetch=True)

            detection_ids = [r[0] for r in result]

        db.conn.commit()
        print(f"  Inserted {len(detection_ids)} detections")

        # Save thumbnails
        thumb_pairs = [(det_id, thumb) for det_id, thumb in zip(detection_ids, thumbnails) if thumb]
        if thumb_pairs:
            db.save_face_thumbnails_batch(thumb_pairs)
            print(f"  Saved {len(thumb_pairs)} thumbnails")

        # Save context crops
        context_pairs = [(det_id, ctx) for det_id, ctx in zip(detection_ids, context_crops) if ctx]
        if context_pairs:
            save_context_crops_batch(db, context_pairs)
            print(f"  Saved {len(context_pairs)} context crops")

    # Update status
    db.update_face_status(
        video_name,
        face_status='completed',
        face_persons_count=len(known_persons),
        face_detections_count=faces_found
    )

    print(f"Done: {video_name}")

    return {
        'video_name': video_name,
        'faces_detected': faces_found,
        'persons_identified': len(known_persons),
        'processing_time': total_time
    }


def save_context_crops_batch(db: PostgresDBManager, crops: List[tuple]) -> int:
    """Save context crops to database"""
    if not crops:
        return 0

    db._ensure_connection()
    with db.conn.cursor() as cur:
        from psycopg2.extras import execute_batch
        execute_batch(cur, """
            UPDATE face_detections
            SET face_crop_context = %s
            WHERE id = %s
        """, [(c[1], c[0]) for c in crops])

    db.conn.commit()
    return len(crops)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Phase 3: Face Detection')
    parser.add_argument('--video', type=str, help='Process single video by name')
    parser.add_argument('--all', action='store_true', help='Force reprocess all videos')
    args = parser.parse_args()

    print("="*60)
    print("Phase 3: Face Detection with On-the-fly Clustering")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*60)

    video_dir = Path("video")
    if not video_dir.exists():
        print(f"ERROR: Video directory not found: {video_dir}")
        sys.exit(1)

    db = PostgresDBManager()

    # Clear existing data for fresh run
    print("\nClearing existing face data...")
    db._ensure_connection()
    with db.conn.cursor() as cur:
        cur.execute("DELETE FROM face_detections")
        cur.execute("DELETE FROM persons")
    db.conn.commit()
    print("Cleared.")

    # Get videos to process
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.webm'}

    if args.video:
        # Single video
        video_path = video_dir / f"{args.video}.mp4"
        if not video_path.exists():
            # Try other extensions
            for ext in video_extensions:
                video_path = video_dir / f"{args.video}{ext}"
                if video_path.exists():
                    break

        if not video_path.exists():
            print(f"ERROR: Video not found: {args.video}")
            sys.exit(1)

        videos = [video_path]
    else:
        videos = [f for f in video_dir.iterdir() if f.suffix.lower() in video_extensions]

    if not videos:
        print("No videos found")
        sys.exit(1)

    print(f"Found {len(videos)} videos to process")

    # Process videos
    total_faces = 0
    total_persons = 0
    total_time = 0

    for video_path in sorted(videos):
        result = process_video(str(video_path), db)
        if 'error' not in result:
            total_faces += result['faces_detected']
            total_persons += result['persons_identified']
            total_time += result['processing_time']

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Videos processed: {len(videos)}")
    print(f"Total faces: {total_faces}")
    print(f"Total persons: {total_persons}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Completed: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
