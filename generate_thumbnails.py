"""
Generate face thumbnails from video files.

Reads face detections from database, extracts face regions from video,
and saves thumbnails to face_thumbnail column.

Usage:
    python generate_thumbnails.py [video_name]
    python generate_thumbnails.py --all
"""

import sys
import cv2
import io
from typing import Dict, List, Tuple
from db_manager_postgres import PostgresDBManager


THUMBNAIL_SIZE = 100  # 100x100 pixels
JPEG_QUALITY = 85


def extract_thumbnail(
    frame: 'np.ndarray',
    bbox: Tuple[int, int, int, int],
    padding: float = 0.2
) -> bytes:
    """
    Extract face thumbnail from frame.

    Args:
        frame: Video frame (BGR)
        bbox: (x, y, w, h) bounding box
        padding: Extra padding around face (0.2 = 20%)

    Returns:
        JPEG bytes
    """
    x, y, w, h = bbox
    frame_h, frame_w = frame.shape[:2]

    # Add padding
    pad_w = int(w * padding)
    pad_h = int(h * padding)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(frame_w, x + w + pad_w)
    y2 = min(frame_h, y + h + pad_h)

    # Crop face region
    face_crop = frame[y1:y2, x1:x2]

    # Resize to square
    face_resized = cv2.resize(face_crop, (THUMBNAIL_SIZE, THUMBNAIL_SIZE))

    # Encode to JPEG
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    _, buffer = cv2.imencode('.jpg', face_resized, encode_params)

    return buffer.tobytes()


def generate_thumbnails_for_video(
    db: PostgresDBManager,
    video_name: str,
    video_path: str
) -> Dict[str, int]:
    """
    Generate thumbnails for all detections in video.

    Args:
        db: Database manager
        video_name: Video name
        video_path: Path to video file

    Returns:
        Stats dict
    """
    print(f"\n=== Generating thumbnails for {video_name} ===")

    # Get detections without thumbnails
    detections = db.get_detections_without_thumbnail(video_name)
    if not detections:
        print("All detections already have thumbnails")
        return {'generated': 0, 'total': 0}

    print(f"Found {len(detections)} detections without thumbnails")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return {'generated': 0, 'total': len(detections), 'error': 'video_not_found'}

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")

    # Group detections by timestamp to minimize seeks
    by_timestamp: Dict[float, List[Dict]] = {}
    for det in detections:
        ts = det['timestamp']
        if ts not in by_timestamp:
            by_timestamp[ts] = []
        by_timestamp[ts].append(det)

    # Process each unique timestamp
    thumbnails = []
    processed = 0
    total_timestamps = len(by_timestamp)

    for i, (timestamp, dets) in enumerate(sorted(by_timestamp.items())):
        # Seek to frame
        frame_num = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            print(f"Warning: Cannot read frame at {timestamp}s")
            continue

        # Extract thumbnails for all faces at this timestamp
        for det in dets:
            bbox = (det['bbox_x'], det['bbox_y'], det['bbox_w'], det['bbox_h'])
            thumb_bytes = extract_thumbnail(frame, bbox)
            thumbnails.append((det['id'], thumb_bytes))
            processed += 1

        # Progress
        if (i + 1) % 20 == 0 or i == total_timestamps - 1:
            print(f"Progress: {i + 1}/{total_timestamps} timestamps, {processed} thumbnails")

    cap.release()

    # Save to database
    if thumbnails:
        print(f"Saving {len(thumbnails)} thumbnails to database...")
        db.save_face_thumbnails_batch(thumbnails)

    print(f"Generated {len(thumbnails)} thumbnails")
    return {'generated': len(thumbnails), 'total': len(detections)}


def main():
    """Main entry point"""
    db = PostgresDBManager()

    if len(sys.argv) < 2:
        print("Usage: python generate_thumbnails.py <video_name>")
        print("       python generate_thumbnails.py --all")
        sys.exit(1)

    if sys.argv[1] == '--all':
        # Process all videos
        videos = db.list_videos()
        total_generated = 0

        for video in videos:
            video_name = video['video_name']
            video_path = video.get('video_path', f'video/{video_name}.mp4')
            stats = generate_thumbnails_for_video(db, video_name, video_path)
            total_generated += stats.get('generated', 0)

        print(f"\n=== Summary ===")
        print(f"Videos processed: {len(videos)}")
        print(f"Total thumbnails generated: {total_generated}")
    else:
        # Process single video
        video_name = sys.argv[1]
        video_info = db.get_video_status(video_name)

        if video_info:
            video_path = video_info.get('video_path', f'video/{video_name}.mp4')
        else:
            video_path = f'video/{video_name}.mp4'

        generate_thumbnails_for_video(db, video_name, video_path)


if __name__ == "__main__":
    main()
