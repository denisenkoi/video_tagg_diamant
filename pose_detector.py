"""
Human pose detection and gesture analysis module using MediaPipe Pose.
Detects 33 body keypoints and analyzes gestures/activities for video analysis.
Optimized for political/news content analysis (speakers, gestures, body language).
"""

import cv2
import numpy as np
import mediapipe as mp
import time
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import math
from config_manager import get_processing_params, get_video_path, get_output_dir, set_video_path
from frame_extractor import extract_frames_with_timestamps
from result_exporter import export_objects_csv, create_output_filename


def initialize_pose_model():
    """Initialize MediaPipe Pose model for human pose detection"""
    print("Initializing MediaPipe Pose detection model...")
    
    # Initialize MediaPipe pose model
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # Configure pose model with optimized settings
    pose_model = mp_pose.Pose(
        static_image_mode=False,          # Video mode
        model_complexity=1,               # 0=lite, 1=full, 2=heavy
        smooth_landmarks=True,            # Smooth landmarks over time
        enable_segmentation=False,        # Don't need segmentation for speed
        smooth_segmentation=False,
        min_detection_confidence=0.5,    # Confidence threshold for detection
        min_tracking_confidence=0.5      # Confidence threshold for tracking
    )
    
    print("MediaPipe Pose model initialized successfully")
    print("Model configuration:")
    print(f"  - Model complexity: 1 (full)")
    print(f"  - Detection confidence: 0.5") 
    print(f"  - Tracking confidence: 0.5")
    print(f"  - Keypoints detected: 33 body landmarks")
    
    return pose_model, mp_pose, mp_drawing


def detect_poses_batch(frames: List[np.ndarray], timestamps: List[str], 
                      pose_model, mp_pose, confidence_threshold: float) -> List[Dict[str, Any]]:
    """
    Detect human poses in frames using MediaPipe with gesture analysis.
    
    Args:
        frames: List of video frames in BGR format
        timestamps: List of formatted timestamps for each frame
        pose_model: MediaPipe Pose model instance
        mp_pose: MediaPipe pose module
        confidence_threshold: Minimum confidence for pose detection
        
    Returns:
        List of grouped pose detection events by timestamp
    """
    assert len(frames) > 0, 'Frames list cannot be empty'
    assert len(frames) == len(timestamps), 'Frames and timestamps must have same length'
    assert 0.0 < confidence_threshold < 1.0, 'Confidence threshold must be between 0 and 1'
    
    print(f"Processing {len(frames)} frames for pose detection...")
    print(f"Using confidence threshold: {confidence_threshold}")
    
    # Get batch size from config
    params = get_processing_params()
    batch_size = params.get('batch_size', 32)  # Higher batch size for MediaPipe
    print(f"Using batch size: {batch_size}")
    
    # Group detections by timestamp
    detections_by_timestamp = {}
    start_time = time.time()
    
    # Process frames in batches
    for batch_start in range(0, len(frames), batch_size):
        batch_end = min(batch_start + batch_size, len(frames))
        batch_frames = frames[batch_start:batch_end]
        batch_timestamps = timestamps[batch_start:batch_end]
        
        # Progress logging
        if batch_start % (batch_size * 3) == 0:  # Every 3 batches
            elapsed = time.time() - start_time
            frames_processed = batch_start
            fps_processing = frames_processed / elapsed if elapsed > 0 else 0
            print(f"Processing batch {batch_start // batch_size + 1}/{(len(frames) - 1) // batch_size + 1} "
                  f"(frames {batch_start + 1}-{batch_end}) Speed: {fps_processing:.1f} frames/sec")
        
        # Process each frame in batch
        for i, (frame, timestamp) in enumerate(zip(batch_frames, batch_timestamps)):
            frame_idx = batch_start + i
            frame_detections = process_single_frame_pose(frame, pose_model, mp_pose, confidence_threshold)
            
            # Group by timestamp
            if len(frame_detections) > 0:
                if timestamp not in detections_by_timestamp:
                    detections_by_timestamp[timestamp] = []
                detections_by_timestamp[timestamp].extend(frame_detections)
                
                # Log detections for this frame
                poses_found = [det['gesture'] for det in frame_detections]
                unique_gestures = list(set(poses_found))
                print(f"Frame {frame_idx + 1} at {timestamp} - Found {len(frame_detections)} people: {unique_gestures}")
    
    # Convert grouped detections to final format
    grouped_detections = []
    for timestamp, detections in detections_by_timestamp.items():
        grouped_detection = group_poses_by_timestamp(timestamp, detections)
        grouped_detections.append(grouped_detection)
    
    total_time = time.time() - start_time
    avg_fps = len(frames) / total_time
    total_poses = sum(len(dets) for dets in detections_by_timestamp.values())
    
    print(f"Pose detection completed in {total_time:.1f} seconds")
    print(f"Average processing speed: {avg_fps:.1f} frames/sec")
    print(f"Total poses detected: {total_poses}")
    print(f"Timestamps with poses: {len(grouped_detections)}")
    
    # Show summary of detected gestures
    if grouped_detections:
        all_gestures = []
        for detection in grouped_detections:
            if detection['gestures']:
                all_gestures.extend(detection['gestures'].split(','))
        
        unique_gestures = list(set(all_gestures))
        print(f"Gestures detected: {unique_gestures}")
        
        # Count gestures by type
        gesture_counts = {gesture: all_gestures.count(gesture) for gesture in unique_gestures}
        print("Gesture counts:")
        for gesture, count in sorted(gesture_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {gesture}: {count}")
    
    return grouped_detections


def process_single_frame_pose(frame: np.ndarray, pose_model, mp_pose, 
                             confidence_threshold: float) -> List[Dict[str, Any]]:
    """Process single frame for pose detection and gesture analysis"""
    assert frame.shape[2] == 3, 'Frame must be BGR format'
    
    detections = []
    
    try:
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = pose_model.process(rgb_frame)
        
        if results.pose_landmarks:
            # Extract pose landmarks
            landmarks = results.pose_landmarks.landmark
            
            # Calculate pose confidence (average of visible landmarks)
            visible_landmarks = [lm for lm in landmarks if lm.visibility > 0.5]
            if len(visible_landmarks) == 0:
                return detections
            
            pose_confidence = sum(lm.visibility for lm in visible_landmarks) / len(visible_landmarks)
            
            if pose_confidence >= confidence_threshold:
                # Analyze gesture/activity
                gesture, activity_confidence = analyze_gesture(landmarks, mp_pose)
                
                # Calculate bounding box from visible keypoints
                bbox, center_x, center_y = calculate_pose_bbox(landmarks, frame.shape)
                
                # Extract key pose metrics
                pose_metrics = extract_pose_metrics(landmarks, mp_pose)
                
                detection = {
                    'person_id': f'person_{len(detections) + 1}',
                    'gesture': gesture,
                    'pose_confidence': pose_confidence,
                    'activity_confidence': activity_confidence,
                    'center_x': center_x,
                    'center_y': center_y,
                    'bbox': bbox,
                    'head_pose': pose_metrics['head_pose'],
                    'arm_position': pose_metrics['arm_position'],
                    'body_orientation': pose_metrics['body_orientation'],
                    'keypoints_count': len(visible_landmarks)
                }
                
                detections.append(detection)
    
    except Exception as e:
        print(f"Pose detection error: {e}")
        return []
    
    return detections


def analyze_gesture(landmarks, mp_pose) -> Tuple[str, float]:
    """Analyze gesture/activity from pose landmarks"""
    try:
        # Get key landmarks
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        
        confidence = 0.8  # Base confidence
        
        # Check for hands raised (speaking/presenting)
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        
        # Hands above shoulders (presenting/gesturing)
        if (left_wrist.y < shoulder_y - 0.1 and left_wrist.visibility > 0.5) or \
           (right_wrist.y < shoulder_y - 0.1 and right_wrist.visibility > 0.5):
            return "presenting", confidence
        
        # Hands near face/head (speaking)
        head_y = nose.y
        if (left_wrist.y < head_y + 0.1 and left_wrist.visibility > 0.5) or \
           (right_wrist.y < head_y + 0.1 and right_wrist.visibility > 0.5):
            return "speaking", confidence
        
        # One hand raised (pointing/gesturing)
        if (left_wrist.y < left_shoulder.y and left_wrist.visibility > 0.5) or \
           (right_wrist.y < right_shoulder.y and right_wrist.visibility > 0.5):
            return "gesturing", confidence
        
        # Arms crossed or at sides
        if left_wrist.y > left_shoulder.y + 0.2 and right_wrist.y > right_shoulder.y + 0.2:
            return "standing", confidence * 0.9
        
        # Default neutral pose
        return "neutral", confidence * 0.7
        
    except Exception as e:
        return "unknown", 0.5


def extract_pose_metrics(landmarks, mp_pose) -> Dict[str, str]:
    """Extract detailed pose metrics for analysis"""
    try:
        # Head pose estimation
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
        right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
        
        # Head orientation
        if left_ear.visibility > 0.5 and right_ear.visibility > 0.5:
            ear_diff = left_ear.x - right_ear.x
            if abs(ear_diff) < 0.05:
                head_pose = "front"
            elif ear_diff > 0:
                head_pose = "right"
            else:
                head_pose = "left"
        else:
            head_pose = "unknown"
        
        # Arm position analysis
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        left_raised = left_wrist.y < left_shoulder.y if left_wrist.visibility > 0.5 else False
        right_raised = right_wrist.y < right_shoulder.y if right_wrist.visibility > 0.5 else False
        
        if left_raised and right_raised:
            arm_position = "both_raised"
        elif left_raised:
            arm_position = "left_raised"
        elif right_raised:
            arm_position = "right_raised"
        else:
            arm_position = "down"
        
        # Body orientation
        left_shoulder_vis = left_shoulder.visibility > 0.5
        right_shoulder_vis = right_shoulder.visibility > 0.5
        
        if left_shoulder_vis and right_shoulder_vis:
            shoulder_diff = abs(left_shoulder.x - right_shoulder.x)
            if shoulder_diff > 0.15:
                body_orientation = "angled"
            else:
                body_orientation = "front"
        else:
            body_orientation = "side"
        
        return {
            'head_pose': head_pose,
            'arm_position': arm_position,
            'body_orientation': body_orientation
        }
        
    except Exception as e:
        return {
            'head_pose': 'unknown',
            'arm_position': 'unknown', 
            'body_orientation': 'unknown'
        }


def calculate_pose_bbox(landmarks, frame_shape) -> Tuple[List[int], int, int]:
    """Calculate bounding box from pose landmarks"""
    height, width = frame_shape[:2]
    
    # Get visible landmarks
    visible_landmarks = [(lm.x * width, lm.y * height) for lm in landmarks if lm.visibility > 0.5]
    
    if not visible_landmarks:
        return [0, 0, 0, 0], 0, 0
    
    # Calculate bounding box
    x_coords = [pt[0] for pt in visible_landmarks]
    y_coords = [pt[1] for pt in visible_landmarks]
    
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    
    # Add padding
    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(width, x_max + padding)
    y_max = min(height, y_max + padding)
    
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    
    return [x_min, y_min, x_max, y_max], center_x, center_y


def group_poses_by_timestamp(timestamp: str, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Group multiple pose detections for the same timestamp into single record"""
    if not detections:
        return {
            'timestamp': timestamp,
            'person_count': 0,
            'gestures': '',
            'confidences': '',
            'coordinates': '',
            'head_poses': '',
            'arm_positions': '',
            'body_orientations': ''
        }
    
    # Extract data from detections
    gestures = [det['gesture'] for det in detections]
    confidences = [f"{det['pose_confidence']:.2f}" for det in detections]
    coordinates = [f"({det['center_x']},{det['center_y']})" for det in detections]
    head_poses = [det['head_pose'] for det in detections]
    arm_positions = [det['arm_position'] for det in detections]
    body_orientations = [det['body_orientation'] for det in detections]
    
    # Create grouped record
    grouped = {
        'timestamp': timestamp,
        'person_count': len(detections),
        'gestures': ','.join(gestures),
        'confidences': ','.join(confidences),
        'coordinates': ','.join(coordinates),
        'head_poses': ','.join(head_poses),
        'arm_positions': ','.join(arm_positions),
        'body_orientations': ','.join(body_orientations)
    }
    
    return grouped


def export_pose_results(poses: List[Dict[str, Any]], video_path: str) -> None:
    """Export pose detection results to CSV"""
    if not poses:
        print("No poses detected to export")
        return
    
    poses_path = create_output_filename(video_path, 'poses')
    
    print(f"Exporting pose detection results to: {poses_path}")
    
    # Export poses to CSV using existing exporter
    export_objects_csv(poses, poses_path)
    
    print(f"Pose detection results exported successfully")
    print(f"Total timestamps with poses: {len(poses)}")
    
    # Calculate total people detected
    total_people = sum(pose['person_count'] for pose in poses if pose['person_count'] > 0)
    print(f"Total people detected: {total_people}")


def process_single_video_poses(video_path: str, models: Tuple) -> None:
    """Process single video file for pose detection"""
    print(f"\n=== Pose Detection: {Path(video_path).name} ===")
    
    pose_model, mp_pose, mp_drawing = models
    
    set_video_path(video_path)
    
    # Validate video file exists
    if not Path(video_path).exists():
        print(f"ERROR: Video file not found: {video_path}")
        return
    
    print(f"Video file confirmed: {video_path}")
    
    # Extract frames with timestamps
    print("Extracting video frames...")
    config = get_processing_params()
    fps = config['frames_per_second']
    frames, timestamps, original_fps = extract_frames_with_timestamps(video_path, fps)
    print(f"Extracted {len(frames)} frames at {fps} FPS (original: {original_fps} FPS)")
    
    # Detect poses
    print("Processing pose detection...")
    confidence_threshold = config.get('object_confidence_threshold', 0.5)
    poses = detect_poses_batch(frames, timestamps, pose_model, mp_pose, confidence_threshold)
    print(f"Pose detection completed. Found poses in {len(poses)} timestamps")
    
    # Export results
    export_pose_results(poses, video_path)
    
    print(f"=== Pose Detection Completed: {Path(video_path).name} ===\n")


def main() -> None:
    """Main function for pose detection on multiple videos"""
    print("=== Pose Detection Pipeline Started ===")
    
    # List of video files to process
    video_files = [
        "video/AlmaAta.mp4",
        "video/news.mp4",
        "video/SovKz.mp4"
    ]
    
    # Initialize pose detection models once
    print("Initializing pose detection models...")
    models = initialize_pose_model()
    print("Pose detection models loaded successfully")
    
    # Process each video file
    total_videos = len(video_files)
    successful_videos = 0
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n*** Processing video {i}/{total_videos} ***")
        
        try:
            process_single_video_poses(video_path, models)
            successful_videos += 1
            print(f"✅ Successfully processed: {Path(video_path).name}")
            
        except Exception as e:
            print(f"❌ ERROR processing {video_path}: {str(e)}")
            print("Continuing with next video...")
            continue
    
    print(f"\n=== Pose Detection Pipeline Completed ===")
    print(f"Successfully processed: {successful_videos}/{total_videos} videos")
    print(f"Failed: {total_videos - successful_videos} videos")


if __name__ == "__main__":
    main()