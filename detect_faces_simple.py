"""
Simple face detection script - detects all faces and saves to DB with thumbnails.
No clustering - just raw detection for visual inspection.
"""

import cv2
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from face_processor_v2 import initialize_face_models, detect_faces_single_frame, get_optimal_det_size
from db_manager_postgres import PostgresDBManager


def encode_thumbnail(face_region, max_size=128):
    """Encode face region as JPEG thumbnail"""
    h, w = face_region.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        face_region = cv2.resize(face_region, (int(w * scale), int(h * scale)))

    _, buffer = cv2.imencode('.jpg', face_region, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buffer.tobytes()


def process_video(video_path: str, db: PostgresDBManager, frame_skip: int = 60, min_face_size: int = 50):
    """
    Process video for face detection only (no clustering).

    Args:
        video_path: Path to video file
        db: Database manager
        frame_skip: Process every N-th frame (60 = ~1fps for 60fps video)
        min_face_size: Minimum face size in pixels
    """
    video_name = Path(video_path).stem
    print(f"\n{'='*60}")
    print(f"Processing: {video_name}")
    print(f"{'='*60}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video: {frame_width}x{frame_height}, {fps:.1f}fps, {duration:.1f}s, {total_frames} frames")
    print(f"Processing every {frame_skip} frames (~{fps/frame_skip:.2f} fps effective)")

    # Initialize InsightFace with optimal det_size
    det_size = get_optimal_det_size(frame_width, frame_height, max_size=1920)
    print(f"Detection size: {det_size}")
    initialize_face_models(det_size=det_size)

    # Update status
    db.update_face_status(video_name, face_status='processing', face_start_time=datetime.now())

    # Process
    all_detections = []
    thumbnails = []
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

        # Progress
        if frame_idx % (frame_skip * 50) == 0:
            elapsed = time.time() - start_time
            progress = frame_idx / total_frames * 100
            print(f"  {progress:.1f}% | frame {frame_idx}/{total_frames} | {faces_found} faces | {elapsed:.1f}s")

        # Detect faces
        faces = detect_faces_single_frame(frame, min_face_size)

        for face in faces:
            # Prepare detection record (no person_id - raw detection)
            detection = {
                'video_name': video_name,
                'person_id': None,  # No clustering yet
                'timestamp': timestamp,
                'bbox': face['coordinates'],
                'age': face['age'],
                'gender': face['gender'],
                'emotion': None,  # Skip emotion for speed
                'det_score': face['det_score'],
                'embedding': face['embedding']
            }
            all_detections.append(detection)

            # Prepare thumbnail
            if 'region' in face and face['region'] is not None and face['region'].size > 0:
                thumb_bytes = encode_thumbnail(face['region'])
                thumbnails.append(thumb_bytes)
            else:
                thumbnails.append(None)

            faces_found += 1

        frame_idx += 1

    cap.release()

    total_time = time.time() - start_time
    print(f"\nDetection complete: {faces_found} faces in {total_time:.1f}s")

    # Save to DB
    if all_detections:
        print(f"Saving {len(all_detections)} detections to database...")

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

    # Update status
    db.update_face_status(
        video_name,
        face_status='completed',
        face_persons_count=0,  # No clustering
        face_detections_count=faces_found
    )

    print(f"Done: {video_name}")
    return faces_found


def main():
    print("="*60)
    print("Simple Face Detection (no clustering)")
    print(f"Started: {datetime.now().isoformat()}")
    print("="*60)

    # Config
    video_dir = Path("video")
    frame_skip = 60  # Every 60th frame (~1fps for 60fps video, ~2fps for 30fps)
    min_face_size = 50

    if not video_dir.exists():
        print(f"ERROR: Video directory not found: {video_dir}")
        sys.exit(1)

    # Get videos
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.webm'}
    videos = [f for f in video_dir.iterdir() if f.suffix.lower() in video_extensions]

    if not videos:
        print("No videos found")
        sys.exit(1)

    print(f"Found {len(videos)} videos")
    print(f"Frame skip: {frame_skip}")
    print(f"Min face size: {min_face_size}")

    # Connect to DB
    db = PostgresDBManager()

    # Clear existing detections (fresh start)
    print("\nClearing existing face_detections...")
    db._ensure_connection()
    with db.conn.cursor() as cur:
        cur.execute("DELETE FROM face_detections")
        cur.execute("DELETE FROM persons")
    db.conn.commit()
    print("Cleared.")

    # Process each video
    total_faces = 0
    for video_path in sorted(videos):
        faces = process_video(str(video_path), db, frame_skip, min_face_size)
        total_faces += faces if faces else 0

    print("\n" + "="*60)
    print(f"TOTAL: {total_faces} faces detected")
    print(f"Completed: {datetime.now().isoformat()}")
    print("="*60)


if __name__ == "__main__":
    main()
