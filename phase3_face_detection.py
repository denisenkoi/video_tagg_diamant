"""
Phase 3: Face Detection and Storage

Processes video files to detect faces, cluster them by person,
and store results in PostgreSQL database.

Uses InsightFace for detection/recognition and FER for emotion detection.
"""

import cv2
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from face_processor_v2 import (
    initialize_face_models,
    detect_faces_single_frame,
    add_emotion_to_faces,
    cluster_faces_to_persons,
    cosine_similarity,
    get_optimal_det_size
)
from db_manager_postgres import PostgresDBManager
from config_manager import get_processing_params
import numpy as np


class Phase3FaceDetector:
    """Face detection and storage for video files"""

    def __init__(self, db_manager: Optional[PostgresDBManager] = None):
        """
        Initialize Phase 3 processor.

        Args:
            db_manager: Database manager instance (creates new if None)
        """
        self.db = db_manager or PostgresDBManager()
        self.params = get_processing_params()

        # Configuration
        self.min_face_size = self.params.get('face_min_size', 50)
        self.similarity_threshold = self.params.get('face_similarity_threshold', 0.7)
        self.frame_skip = self.params.get('face_frame_skip', 5)  # Process every Nth frame
        self.max_det_size = self.params.get('max_det_size', 1920)  # Max detection resolution

        print(f"Phase 3 initialized:")
        print(f"  - Min face size: {self.min_face_size}")
        print(f"  - Similarity threshold: {self.similarity_threshold}")
        print(f"  - Frame skip: {self.frame_skip}")
        print(f"  - Max det size: {self.max_det_size}")

    def process_video(self, video_path: str) -> Dict[str, Any]:
        """
        Process single video for face detection.

        Args:
            video_path: Path to video file

        Returns:
            Processing statistics
        """
        video_name = Path(video_path).stem
        print(f"\n=== Phase 3: Face Detection for {video_name} ===")

        # Update status
        self.db.update_face_status(
            video_name,
            face_status='processing',
            face_start_time=datetime.now()
        )

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        print(f"Video: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s")
        print(f"Resolution: {frame_width}x{frame_height}")

        # Calculate optimal detection size based on video resolution
        det_size = get_optimal_det_size(frame_width, frame_height, self.max_det_size)
        print(f"Using detection size: {det_size[0]}x{det_size[1]}")

        # Initialize face models with adaptive det_size
        initialize_face_models(det_size=det_size)

        # Storage
        known_persons = []
        all_detections = []
        frame_idx = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames for performance
            if frame_idx % self.frame_skip != 0:
                frame_idx += 1
                continue

            timestamp = frame_idx / fps

            # Progress logging
            if frame_idx % (self.frame_skip * 100) == 0:
                elapsed = time.time() - start_time
                progress = frame_idx / total_frames * 100
                print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames} frames, {elapsed:.1f}s)")

            # Detect faces
            faces = detect_faces_single_frame(frame, self.min_face_size)

            if faces:
                # Add emotion
                faces = add_emotion_to_faces(faces)

                # Add timestamp
                for face in faces:
                    face['timestamp'] = timestamp
                    face['frame_idx'] = frame_idx

                # Cluster
                clustered = cluster_faces_to_persons(faces, known_persons, self.similarity_threshold)

                # Store detections
                for face in clustered:
                    face['video_name'] = video_name
                    all_detections.append(face)

            frame_idx += 1

        cap.release()

        total_time = time.time() - start_time
        processed_frames = frame_idx // self.frame_skip

        print(f"\nDetection completed in {total_time:.1f}s")
        print(f"Processed frames: {processed_frames}")
        print(f"Faces detected: {len(all_detections)}")
        print(f"Unique persons: {len(known_persons)}")

        # Store to database
        print("\nStoring to database...")
        self._store_results(video_name, known_persons, all_detections)

        # Update status
        self.db.update_face_status(
            video_name,
            face_status='completed',
            face_persons_count=len(known_persons),
            face_detections_count=len(all_detections)
        )

        return {
            'video_name': video_name,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'faces_detected': len(all_detections),
            'persons_identified': len(known_persons),
            'processing_time': total_time
        }

    def _store_results(
        self,
        video_name: str,
        known_persons: List[Dict],
        all_detections: List[Dict]
    ) -> None:
        """Store detection results to database"""

        # Store persons
        for person in known_persons:
            self.db.create_person(
                person_id=person['person_id'],
                video_name=video_name,
                first_seen_time=person.get('first_seen', 0.0),
                representative_embedding=person['representative_embedding']
            )

        print(f"  Stored {len(known_persons)} persons")

        # Store detections in batch
        detections_for_db = []
        for det in all_detections:
            detections_for_db.append({
                'video_name': det['video_name'],
                'person_id': det['person_id'],
                'timestamp': det.get('timestamp', 0.0),
                'bbox': det['coordinates'],
                'age': det['age'],
                'gender': det['gender'],
                'emotion': det.get('emotion', 'neutral'),
                'det_score': det['det_score'],
                'embedding': det['embedding']
            })

        count = self.db.add_face_detections_batch(detections_for_db)
        print(f"  Stored {count} face detections")

    def process_all_videos(self, video_dir: str = "video") -> List[Dict]:
        """
        Process all videos in directory.

        Args:
            video_dir: Directory containing video files

        Returns:
            List of processing results
        """
        video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.webm'}
        video_files = []

        for f in Path(video_dir).iterdir():
            if f.suffix.lower() in video_extensions:
                video_files.append(str(f))

        print(f"Found {len(video_files)} video files")

        results = []
        for video_path in sorted(video_files):
            result = self.process_video(video_path)
            results.append(result)

        return results


def main():
    """Main entry point"""
    print("=== Phase 3: Face Detection ===")
    print(f"Started at: {datetime.now().isoformat()}")

    # Check video directory
    video_dir = "video"
    if not os.path.exists(video_dir):
        print(f"Error: Video directory '{video_dir}' not found")
        sys.exit(1)

    # Initialize processor
    processor = Phase3FaceDetector()

    # Get videos to process
    videos = processor.db.list_videos({'face_status': ['pending', 'processing']})

    if not videos:
        print("No videos pending face detection")
        # Try processing all videos anyway
        print("Processing all videos in directory...")
        results = processor.process_all_videos(video_dir)
    else:
        print(f"Found {len(videos)} videos to process")
        results = []
        for video in videos:
            video_path = video.get('video_path')
            if video_path and os.path.exists(video_path):
                result = processor.process_video(video_path)
                results.append(result)

    # Summary
    print("\n=== Summary ===")
    total_faces = sum(r['faces_detected'] for r in results)
    total_persons = sum(r['persons_identified'] for r in results)
    total_time = sum(r['processing_time'] for r in results)

    print(f"Videos processed: {len(results)}")
    print(f"Total faces detected: {total_faces}")
    print(f"Total persons identified: {total_persons}")
    print(f"Total processing time: {total_time:.1f}s")

    print(f"\nCompleted at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
