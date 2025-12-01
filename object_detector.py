from ultralytics import YOLO
import numpy as np
import torch
import time
from typing import List, Dict, Any, Tuple
from config_manager import get_processing_params


def initialize_yolo_model() -> Any:
    """Initialize YOLOv11n model for object detection with GPU support"""
    print("Initializing YOLO model...")

    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"YOLO using device: {device}")

    # Initialize YOLO model
    model = YOLO('yolo11n.pt')

    # Move model to GPU if available
    if device == "cuda":
        model.to(device)
        print(f"YOLO model moved to GPU: {torch.cuda.get_device_name()}")

    print("YOLO model loaded successfully")
    return model


def detect_objects_batch(frames: List[np.ndarray], timestamps: List[str], model: Any, confidence_threshold: float) -> \
List[Dict[str, Any]]:
    """Detect objects in frames using batch processing

    Args:
        frames: List of video frames in BGR format
        timestamps: List of formatted timestamps for each frame
        model: YOLO model instance
        confidence_threshold: Minimum confidence for object detection

    Returns:
        List of grouped object detection events by timestamp
    """
    assert len(frames) > 0, 'Frames list cannot be empty'
    assert len(frames) == len(timestamps), 'Frames and timestamps must have same length'
    assert 0.0 < confidence_threshold < 1.0, 'Confidence threshold must be between 0 and 1'

    print(f"Processing {len(frames)} frames for object detection...")
    print(f"Using confidence threshold: {confidence_threshold}")

    # Get batch size from config
    from config_manager import get_processing_params
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
            frame_detections = process_yolo_results_single(result)

            # Group by timestamp
            if len(frame_detections) > 0:
                if timestamp not in detections_by_timestamp:
                    detections_by_timestamp[timestamp] = []
                detections_by_timestamp[timestamp].extend(frame_detections)

                # Log detections for this frame
                objects_found = [det['class'] for det in frame_detections]
                unique_objects = list(set(objects_found))
                print(f"Frame {frame_idx + 1} at {timestamp} - Found {len(frame_detections)} objects: {unique_objects}")

    # Convert grouped detections to final format
    grouped_detections = []
    for timestamp, detections in detections_by_timestamp.items():
        grouped_detection = group_objects_by_timestamp(timestamp, detections)
        grouped_detections.append(grouped_detection)

    total_time = time.time() - start_time
    avg_fps = len(frames) / total_time
    total_objects = sum(len(dets) for dets in detections_by_timestamp.values())

    print(f"Object detection completed in {total_time:.1f} seconds")
    print(f"Average processing speed: {avg_fps:.1f} frames/sec")
    print(f"Total objects detected: {total_objects}")
    print(f"Timestamps with objects: {len(grouped_detections)}")

    # Show summary of detected object types
    if grouped_detections:
        all_classes = []
        for detection in grouped_detections:
            if detection['objects']:
                all_classes.extend(detection['objects'].split(','))

        unique_classes = list(set(all_classes))
        print(f"Object types found: {unique_classes}")

        # Count objects by type
        class_counts = {cls: all_classes.count(cls) for cls in unique_classes}
        print("Object counts:")
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls}: {count}")

    return grouped_detections


def process_yolo_results_single(result: Any) -> List[Dict[str, Any]]:
    """Process YOLO results for single frame into list of individual detections

    Args:
        result: YOLO detection result for single frame

    Returns:
        List of individual detection dictionaries for this frame
    """
    detections = []

    # Extract boxes from result
    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:
        for i in range(len(boxes)):
            # Get bounding box coordinates
            bbox = boxes.xyxy[i].cpu().numpy().tolist()

            # Get class and confidence
            cls = int(boxes.cls[i].cpu().numpy())
            conf = float(boxes.conf[i].cpu().numpy())

            # Get class name
            class_name = result.names[cls]

            # Calculate center coordinates and dimensions
            center_x, center_y = get_object_center(bbox)
            width = int(bbox[2] - bbox[0])
            height = int(bbox[3] - bbox[1])

            # Create detection dictionary
            detection = {
                'class': class_name,
                'confidence': conf,
                'center_x': center_x,
                'center_y': center_y,
                'width': width,
                'height': height
            }

            detections.append(detection)

    return detections


def group_objects_by_timestamp(timestamp: str, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Group multiple object detections for the same timestamp into single record

    Args:
        timestamp: Formatted timestamp string
        detections: List of individual detections for this timestamp

    Returns:
        Single grouped detection record
    """
    if not detections:
        return {
            'timestamp': timestamp,
            'objects': '',
            'confidences': '',
            'coordinates': '',
            'count': 0
        }

    # Extract data from detections
    objects = [det['class'] for det in detections]
    confidences = [f"{det['confidence']:.2f}" for det in detections]
    coordinates = [f"({det['center_x']},{det['center_y']})" for det in detections]

    # Create grouped record
    grouped = {
        'timestamp': timestamp,
        'objects': ','.join(objects),
        'confidences': ','.join(confidences),
        'coordinates': ','.join(coordinates),
        'count': len(detections)
    }

    return grouped


def get_object_center(bbox: List[float]) -> Tuple[int, int]:
    """Calculate center coordinates from bounding box

    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2]

    Returns:
        Tuple of center coordinates (center_x, center_y)
    """
    center_x = int((bbox[0] + bbox[2]) / 2)
    center_y = int((bbox[1] + bbox[3]) / 2)
    return center_x, center_y


def detect_objects(frame: np.ndarray, model: Any, confidence_threshold: float) -> List[Dict[str, Any]]:
    """Detect objects in single frame (legacy function for compatibility)

    Args:
        frame: Input BGR image frame
        model: YOLO model instance
        confidence_threshold: Minimum confidence for detection

    Returns:
        List of detection dictionaries
    """
    # Validate input frame format
    assert frame.shape[2] == 3, 'Frame must be BGR format'
    assert 0.0 < confidence_threshold < 1.0, 'Confidence threshold must be between 0 and 1'

    # Run YOLO detection
    results = model(frame, conf=confidence_threshold, verbose=False)

    # Process results into structured format using the new function
    detections = process_yolo_results_single(results[0])

    # Add timestamp for compatibility (legacy format)
    for detection in detections:
        detection['timestamp'] = "00:00:00"

    return detections