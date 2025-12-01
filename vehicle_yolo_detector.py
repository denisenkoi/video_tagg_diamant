"""
YOLO-based vehicle detection module for video processing pipeline.
Detects vehicles by type: car, truck, bus, motorcycle, bicycle, van.
Uses pre-trained YOLOv11 weights for high-accuracy vehicle detection.
"""

from ultralytics import YOLO
import numpy as np
import torch
import time
import cv2
from typing import List, Dict, Any, Tuple
from pathlib import Path
from config_manager import get_processing_params, get_video_path, get_output_dir, set_video_path
from frame_extractor import extract_frames_with_timestamps
from result_exporter import export_objects_csv, create_output_filename


def initialize_yolo_vehicle_model() -> Any:
    """Initialize YOLOv11 model specifically configured for vehicle detection"""
    print("Initializing YOLO vehicle detection model...")

    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"YOLO vehicle detector using device: {device}")

    # Initialize YOLO model with vehicle-optimized settings
    model = YOLO('yolo11n.pt')  # Using nano for speed, can upgrade to yolo11s.pt or yolo11m.pt

    # Move model to GPU if available
    if device == "cuda":
        model.to(device)
        print(f"YOLO vehicle model moved to GPU: {torch.cuda.get_device_name()}")

    print("YOLO vehicle detection model loaded successfully")
    return model


def detect_vehicles_batch(frames: List[np.ndarray], timestamps: List[str], 
                         model: Any, confidence_threshold: float) -> List[Dict[str, Any]]:
    """
    Detect vehicles in frames using YOLO with vehicle-specific filtering.
    
    Args:
        frames: List of video frames in BGR format
        timestamps: List of formatted timestamps for each frame
        model: YOLO model instance
        confidence_threshold: Minimum confidence for vehicle detection
        
    Returns:
        List of grouped vehicle detection events by timestamp
    """
    assert len(frames) > 0, 'Frames list cannot be empty'
    assert len(frames) == len(timestamps), 'Frames and timestamps must have same length'
    assert 0.0 < confidence_threshold < 1.0, 'Confidence threshold must be between 0 and 1'

    print(f"Processing {len(frames)} frames for vehicle detection...")
    print(f"Using confidence threshold: {confidence_threshold}")

    # Vehicle classes from COCO dataset (YOLO uses COCO class IDs)
    vehicle_classes = {
        2: 'car',          # COCO class 2
        3: 'motorcycle',   # COCO class 3
        5: 'bus',          # COCO class 5
        7: 'truck',        # COCO class 7
        1: 'bicycle'       # COCO class 1 (bicycle)
    }

    print(f"Detecting vehicle classes: {list(vehicle_classes.values())}")

    # Get batch size from config
    params = get_processing_params()
    batch_size = params.get('batch_size', 16)
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
        if batch_start % (batch_size * 5) == 0:  # Every 5 batches
            elapsed = time.time() - start_time
            frames_processed = batch_start
            fps_processing = frames_processed / elapsed if elapsed > 0 else 0
            print(f"Processing batch {batch_start // batch_size + 1}/{(len(frames) - 1) // batch_size + 1} "
                  f"(frames {batch_start + 1}-{batch_end}) Speed: {fps_processing:.1f} frames/sec")

        # Validate frames
        for frame in batch_frames:
            assert frame.shape[2] == 3, 'Frame must be BGR format'

        # Run YOLO detection on batch
        results = model(batch_frames, conf=confidence_threshold, verbose=False)

        # Process results for each frame in batch
        for i, (result, timestamp) in enumerate(zip(results, batch_timestamps)):
            frame_idx = batch_start + i
            frame_detections = process_yolo_vehicle_results(result, vehicle_classes)

            # Group by timestamp
            if len(frame_detections) > 0:
                if timestamp not in detections_by_timestamp:
                    detections_by_timestamp[timestamp] = []
                detections_by_timestamp[timestamp].extend(frame_detections)

                # Log detections for this frame
                vehicles_found = [det['vehicle_type'] for det in frame_detections]
                vehicle_counts = {}
                for vehicle in vehicles_found:
                    vehicle_counts[vehicle] = vehicle_counts.get(vehicle, 0) + 1
                
                count_str = ', '.join([f"{v}:{c}" for v, c in vehicle_counts.items()])
                print(f"Frame {frame_idx + 1} at {timestamp} - Found {len(frame_detections)} vehicles: {count_str}")

    # Convert grouped detections to final format
    grouped_detections = []
    for timestamp, detections in detections_by_timestamp.items():
        grouped_detection = group_vehicles_by_timestamp(timestamp, detections)
        grouped_detections.append(grouped_detection)

    total_time = time.time() - start_time
    avg_fps = len(frames) / total_time
    total_vehicles = sum(len(dets) for dets in detections_by_timestamp.values())

    print(f"Vehicle detection completed in {total_time:.1f} seconds")
    print(f"Average processing speed: {avg_fps:.1f} frames/sec")
    print(f"Total vehicles detected: {total_vehicles}")
    print(f"Timestamps with vehicles: {len(grouped_detections)}")

    # Show summary of detected vehicle types
    if grouped_detections:
        all_vehicles = []
        for detection in grouped_detections:
            if detection['vehicle_types']:
                all_vehicles.extend(detection['vehicle_types'].split(','))

        unique_vehicles = list(set(all_vehicles))
        print(f"Vehicle types found: {unique_vehicles}")

        # Count vehicles by type
        vehicle_type_counts = {vtype: all_vehicles.count(vtype) for vtype in unique_vehicles}
        print("Vehicle type counts:")
        for vtype, count in sorted(vehicle_type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {vtype}: {count}")

    return grouped_detections


def process_yolo_vehicle_results(result: Any, vehicle_classes: Dict[int, str]) -> List[Dict[str, Any]]:
    """
    Process YOLO results for single frame, filtering only vehicle classes.
    
    Args:
        result: YOLO detection result for single frame
        vehicle_classes: Dictionary mapping COCO class IDs to vehicle type names
        
    Returns:
        List of vehicle detection dictionaries for this frame
    """
    detections = []

    # Extract boxes from result
    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:
        for i in range(len(boxes)):
            # Get class ID and check if it's a vehicle
            cls_id = int(boxes.cls[i].cpu().numpy())
            
            if cls_id in vehicle_classes:
                # Get bounding box coordinates
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                
                # Get confidence
                conf = float(boxes.conf[i].cpu().numpy())
                
                # Get vehicle type name
                vehicle_type = vehicle_classes[cls_id]
                
                # Calculate center coordinates and dimensions
                center_x, center_y = get_vehicle_center(bbox)
                width = int(bbox[2] - bbox[0])
                height = int(bbox[3] - bbox[1])
                
                # Create detection dictionary
                detection = {
                    'vehicle_type': vehicle_type,
                    'confidence': conf,
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': width,
                    'height': height,
                    'bbox': bbox
                }
                
                detections.append(detection)

    return detections


def group_vehicles_by_timestamp(timestamp: str, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Group multiple vehicle detections for the same timestamp into single record.
    
    Args:
        timestamp: Formatted timestamp string
        detections: List of individual vehicle detections for this timestamp
        
    Returns:
        Single grouped vehicle detection record
    """
    if not detections:
        return {
            'timestamp': timestamp,
            'vehicle_types': '',
            'confidences': '',
            'coordinates': '',
            'count': 0,
            'vehicle_summary': ''
        }

    # Extract data from detections
    vehicle_types = [det['vehicle_type'] for det in detections]
    confidences = [f"{det['confidence']:.2f}" for det in detections]
    coordinates = [f"({det['center_x']},{det['center_y']})" for det in detections]
    
    # Create vehicle summary (counts by type)
    vehicle_counts = {}
    for vtype in vehicle_types:
        vehicle_counts[vtype] = vehicle_counts.get(vtype, 0) + 1
    
    vehicle_summary = ', '.join([f"{vtype}:{count}" for vtype, count in vehicle_counts.items()])

    # Create grouped record
    grouped = {
        'timestamp': timestamp,
        'vehicle_types': ','.join(vehicle_types),
        'confidences': ','.join(confidences),
        'coordinates': ','.join(coordinates),
        'count': len(detections),
        'vehicle_summary': vehicle_summary
    }

    return grouped


def get_vehicle_center(bbox: List[float]) -> Tuple[int, int]:
    """
    Calculate center coordinates from bounding box.
    
    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        
    Returns:
        Tuple of center coordinates (center_x, center_y)
    """
    center_x = int((bbox[0] + bbox[2]) / 2)
    center_y = int((bbox[1] + bbox[3]) / 2)
    return center_x, center_y


def export_vehicle_results(vehicles: List[Dict[str, Any]], video_path: str) -> None:
    """Export vehicle detection results to CSV"""
    if not vehicles:
        print("No vehicles detected to export")
        return
        
    vehicles_path = create_output_filename(video_path, 'vehicles_yolo')
    
    print(f"Exporting YOLO vehicle detections to: {vehicles_path}")
    
    # Export vehicles to CSV using existing exporter
    export_objects_csv(vehicles, vehicles_path)
    
    print(f"YOLO vehicle detection results exported successfully")
    print(f"Total timestamps with vehicles: {len(vehicles)}")
    
    # Calculate total individual vehicles
    total_vehicles = sum(vehicle['count'] for vehicle in vehicles if vehicle['count'] > 0)
    print(f"Total individual vehicles detected: {total_vehicles}")


def process_single_video_yolo_vehicles(video_path: str, model: Any) -> None:
    """Process single video file for YOLO vehicle detection"""
    print(f"\n=== YOLO Vehicle Detection: {Path(video_path).name} ===")
    
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
    
    # Detect vehicles
    print("Processing vehicle detection...")
    confidence_threshold = config.get('object_confidence_threshold', 0.5)
    vehicles = detect_vehicles_batch(frames, timestamps, model, confidence_threshold)
    print(f"Vehicle detection completed. Found vehicles in {len(vehicles)} timestamps")
    
    # Export results
    export_vehicle_results(vehicles, video_path)
    
    print(f"=== YOLO Vehicle Detection Completed: {Path(video_path).name} ===\n")


def main() -> None:
    """Main function for YOLO vehicle detection on multiple videos"""
    print("=== YOLO Vehicle Detection Pipeline Started ===")
    
    # List of video files to process
    video_files = [
        "video/AlmaAta.mp4",
        "video/news.mp4", 
        "video/SovKz.mp4"
    ]
    
    # Initialize YOLO vehicle model once
    print("Initializing YOLO vehicle detection model...")
    model = initialize_yolo_vehicle_model()
    print("YOLO vehicle model loaded successfully")
    
    # Process each video file
    total_videos = len(video_files)
    successful_videos = 0
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n*** Processing video {i}/{total_videos} ***")
        
        try:
            process_single_video_yolo_vehicles(video_path, model)
            successful_videos += 1
            print(f"✅ Successfully processed: {Path(video_path).name}")
            
        except Exception as e:
            print(f"❌ ERROR processing {video_path}: {str(e)}")
            print("Continuing with next video...")
            continue
    
    print(f"\n=== YOLO Vehicle Detection Pipeline Completed ===")
    print(f"Successfully processed: {successful_videos}/{total_videos} videos")
    print(f"Failed: {total_videos - successful_videos} videos")


if __name__ == "__main__":
    main()