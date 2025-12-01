"""
ANPR (Automatic Number Plate Recognition) module for license plate detection and recognition.
Two-stage approach: YOLO detection + PaddleOCR recognition with pattern validation.
Supports Kazakh, Russian, Soviet, and European license plate formats.
"""

import cv2
import numpy as np
import time
import re
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from ultralytics import YOLO
from paddleocr import PaddleOCR
from config_manager import get_processing_params, get_video_path, get_output_dir, set_video_path
from frame_extractor import extract_frames_with_timestamps
from result_exporter import export_objects_csv, create_output_filename


def initialize_anpr_models():
    """Initialize YOLO detector and PaddleOCR for license plate recognition"""
    print("Initializing ANPR models...")
    
    # Initialize YOLO for license plate detection
    print("Loading YOLO model for plate detection...")
    yolo_model = YOLO('yolo11n.pt')  # Will detect vehicles first
    
    # Initialize PaddleOCR for text recognition
    print("Loading PaddleOCR for plate text recognition...")
    try:
        ocr_model = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            use_gpu=True,
            show_log=False,
            drop_score=0.1,  # Lower threshold for license plates
            det_algorithm='DB',
            rec_algorithm='CRNN'
        )
        print("✓ ANPR models initialized with GPU support")
    except Exception as e:
        print(f"GPU initialization failed: {e}")
        ocr_model = PaddleOCR(
            use_angle_cls=True,
            lang='en', 
            use_gpu=False,
            show_log=False,
            drop_score=0.1
        )
        print("✓ ANPR models initialized with CPU")
    
    # Load license plate patterns
    plate_patterns = get_license_plate_patterns()
    
    print("ANPR configuration:")
    print("  - Detection: YOLO vehicle detection + region extraction")
    print("  - Recognition: PaddleOCR with low threshold (0.1)")
    print(f"  - Supported formats: {len(plate_patterns)} different patterns")
    print("  - Countries: Kazakhstan, Russia, Soviet, Germany, France")
    
    return yolo_model, ocr_model, plate_patterns


def get_license_plate_patterns() -> Dict[str, Dict]:
    """Define license plate patterns for different countries/regions"""
    patterns = {
        # Kazakhstan formats
        'kz_new': {
            'pattern': r'^(\d{3})([A-Z]{3})(\d{2})$',
            'format': 'XXX ABC XX',
            'country': 'Kazakhstan',
            'type': 'standard',
            'regions': get_kazakhstan_regions()
        },
        'kz_old': {
            'pattern': r'^([A-Z])(\d{3})([A-Z]{2})(\d{2})$', 
            'format': 'A XXX BC XX',
            'country': 'Kazakhstan',
            'type': 'old_format',
            'regions': get_kazakhstan_regions()
        },
        
        # Russia formats
        'ru_standard': {
            'pattern': r'^([А-Я])(\d{3})([А-Я]{2})(\d{2,3})$',
            'format': 'А XXX БВ ГГ',
            'country': 'Russia',
            'type': 'standard',
            'regions': get_russia_regions()
        },
        'ru_taxi': {
            'pattern': r'^([А-Я]{2})(\d{3})(\d{2,3})$',
            'format': 'АА XXX ГГ',
            'country': 'Russia', 
            'type': 'taxi',
            'regions': get_russia_regions()
        },
        'ru_transit': {
            'pattern': r'^(\d{4})([ТР]{2})(\d{2,3})$',
            'format': 'XXXX ТР ГГ',
            'country': 'Russia',
            'type': 'transit',
            'regions': get_russia_regions()
        },
        
        # Soviet/legacy formats
        'soviet': {
            'pattern': r'^(\d{2})-(\d{2})([А-Я]{3})$',
            'format': 'XX-XX МСК',
            'country': 'Soviet',
            'type': 'historical',
            'regions': {}
        },
        
        # European formats
        'germany': {
            'pattern': r'^([A-Z]{1,3})-([A-Z]{1,2})\s?(\d{1,4})$',
            'format': 'B-MW XXXX',
            'country': 'Germany',
            'type': 'standard',
            'regions': {}
        },
        'france': {
            'pattern': r'^(\d{3})-([A-Z]{2})-(\d{2})$',
            'format': 'XXX-AB-XX',
            'country': 'France',
            'type': 'standard', 
            'regions': {}
        }
    }
    
    return patterns


def get_kazakhstan_regions() -> Dict[str, str]:
    """Kazakhstan region codes"""
    return {
        '01': 'Almaty',
        '02': 'Almaty Region',
        '03': 'Akmola Region',
        '04': 'Aktobe Region', 
        '05': 'Atyrau Region',
        '06': 'West Kazakhstan Region',
        '07': 'Zhambyl Region',
        '08': 'Karaganda Region',
        '09': 'Kostanay Region',
        '10': 'Kyzylorda Region',
        '11': 'Mangystau Region',
        '12': 'Pavlodar Region',
        '13': 'North Kazakhstan Region',
        '14': 'South Kazakhstan Region',
        '15': 'East Kazakhstan Region',
        '16': 'Nur-Sultan',
        '17': 'Shymkent'
    }


def get_russia_regions() -> Dict[str, str]:
    """Russia region codes (sample)"""
    return {
        '77': 'Moscow',
        '78': 'Saint Petersburg',
        '50': 'Moscow Region',
        '178': 'Saint Petersburg',
        '199': 'Moscow',
        '777': 'Moscow'
    }


def detect_license_plates_batch(frames: List[np.ndarray], timestamps: List[str],
                               yolo_model, ocr_model, plate_patterns: Dict,
                               confidence_threshold: float) -> List[Dict[str, Any]]:
    """
    Detect and recognize license plates using two-stage approach.
    
    Args:
        frames: List of video frames in BGR format
        timestamps: List of formatted timestamps for each frame
        yolo_model: YOLO model for vehicle detection
        ocr_model: PaddleOCR model for text recognition
        plate_patterns: Dictionary of license plate patterns
        confidence_threshold: Minimum confidence for detection
        
    Returns:
        List of grouped license plate detection events by timestamp
    """
    assert len(frames) > 0, 'Frames list cannot be empty'
    assert len(frames) == len(timestamps), 'Frames and timestamps must have same length'
    
    print(f"Processing {len(frames)} frames for license plate detection...")
    print(f"Using confidence threshold: {confidence_threshold}")
    
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
        if batch_start % (batch_size * 3) == 0:  # Every 3 batches
            elapsed = time.time() - start_time
            frames_processed = batch_start
            fps_processing = frames_processed / elapsed if elapsed > 0 else 0
            print(f"Processing batch {batch_start // batch_size + 1}/{(len(frames) - 1) // batch_size + 1} "
                  f"(frames {batch_start + 1}-{batch_end}) Speed: {fps_processing:.1f} frames/sec")
        
        # Stage 1: YOLO vehicle detection
        yolo_results = yolo_model(batch_frames, conf=confidence_threshold, verbose=False)
        
        # Stage 2: Process each frame
        for i, (frame, timestamp, yolo_result) in enumerate(zip(batch_frames, batch_timestamps, yolo_results)):
            frame_idx = batch_start + i
            frame_detections = process_frame_anpr(
                frame, yolo_result, ocr_model, plate_patterns
            )
            
            # Group by timestamp
            if len(frame_detections) > 0:
                if timestamp not in detections_by_timestamp:
                    detections_by_timestamp[timestamp] = []
                detections_by_timestamp[timestamp].extend(frame_detections)
                
                # Log detections for this frame
                plates_found = [f"{det['plate_number']} ({det['country']})" for det in frame_detections]
                print(f"Frame {frame_idx + 1} at {timestamp} - Found plates: {', '.join(plates_found)}")
    
    # Convert grouped detections to final format
    grouped_detections = []
    for timestamp, detections in detections_by_timestamp.items():
        grouped_detection = group_plates_by_timestamp(timestamp, detections)
        grouped_detections.append(grouped_detection)
    
    total_time = time.time() - start_time
    avg_fps = len(frames) / total_time
    total_plates = sum(len(dets) for dets in detections_by_timestamp.values())
    
    print(f"License plate detection completed in {total_time:.1f} seconds")
    print(f"Average processing speed: {avg_fps:.1f} frames/sec")
    print(f"Total license plates detected: {total_plates}")
    print(f"Timestamps with plates: {len(grouped_detections)}")
    
    # Show summary of detected plate countries and types
    if grouped_detections:
        all_countries = []
        all_types = []
        for detection in grouped_detections:
            if detection['countries']:
                all_countries.extend(detection['countries'].split(','))
            if detection['plate_types']:
                all_types.extend(detection['plate_types'].split(','))
        
        unique_countries = list(set(all_countries))
        unique_types = list(set(all_types))
        
        print(f"Plate countries detected: {unique_countries}")
        print(f"Plate types detected: {unique_types}")
        
        # Count by country
        country_counts = {country: all_countries.count(country) for country in unique_countries}
        print("Country counts:")
        for country, count in sorted(country_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {country}: {count}")
    
    return grouped_detections


def process_frame_anpr(frame: np.ndarray, yolo_result: Any, ocr_model, 
                      plate_patterns: Dict) -> List[Dict[str, Any]]:
    """Process single frame: detect vehicles and extract license plate regions"""
    detections = []
    
    # Extract vehicle boxes from YOLO results
    boxes = yolo_result.boxes
    
    if boxes is not None and len(boxes) > 0:
        # Vehicle classes from COCO dataset
        vehicle_class_ids = {2, 3, 5, 7}  # car, motorcycle, bus, truck
        
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].cpu().numpy())
            
            if cls_id in vehicle_class_ids:
                # Get vehicle bounding box
                bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())
                
                x1, y1, x2, y2 = bbox
                vehicle_region = frame[y1:y2, x1:x2]
                
                if vehicle_region.size > 0:
                    # Extract potential license plate regions from vehicle
                    plate_regions = extract_plate_regions(vehicle_region)
                    
                    for plate_region, relative_coords in plate_regions:
                        # Recognize text in plate region
                        plate_text = recognize_plate_text(plate_region, ocr_model)
                        
                        if plate_text:
                            # Validate and classify license plate
                            plate_info = validate_license_plate(plate_text, plate_patterns)
                            
                            if plate_info:
                                # Calculate absolute coordinates
                                abs_x = x1 + relative_coords[0]
                                abs_y = y1 + relative_coords[1]
                                abs_w = relative_coords[2]
                                abs_h = relative_coords[3]
                                
                                detection = {
                                    'plate_number': plate_info['plate_number'],
                                    'country': plate_info['country'],
                                    'plate_type': plate_info['type'],
                                    'region': plate_info.get('region', 'Unknown'),
                                    'confidence': conf,
                                    'ocr_confidence': plate_info['confidence'],
                                    'center_x': abs_x + abs_w // 2,
                                    'center_y': abs_y + abs_h // 2,
                                    'width': abs_w,
                                    'height': abs_h,
                                    'vehicle_bbox': bbox.tolist()
                                }
                                
                                detections.append(detection)
    
    return detections


def extract_plate_regions(vehicle_region: np.ndarray) -> List[Tuple[np.ndarray, List[int]]]:
    """Extract potential license plate regions from vehicle image"""
    plate_regions = []
    
    try:
        height, width = vehicle_region.shape[:2]
        
        # License plates are typically in the lower portion of vehicles
        # and have specific aspect ratios
        
        # Convert to grayscale for better edge detection
        gray = cv2.cvtColor(vehicle_region, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size and aspect ratio (typical license plate proportions)
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            
            # License plate criteria
            if (2.0 < aspect_ratio < 6.0 and  # Typical plate aspect ratio
                area > (width * height * 0.01) and  # Minimum size
                area < (width * height * 0.3) and   # Maximum size
                y > height * 0.4):  # Lower portion of vehicle
                
                # Extract plate region with padding
                padding = 5
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(width, x + w + padding)
                y2 = min(height, y + h + padding)
                
                plate_region = vehicle_region[y1:y2, x1:x2]
                
                if plate_region.size > 0:
                    plate_regions.append((plate_region, [x1, y1, x2 - x1, y2 - y1]))
        
        # If no good regions found, try bottom portion of vehicle
        if not plate_regions and height > 50:
            bottom_region = vehicle_region[int(height * 0.7):, :]
            if bottom_region.size > 0:
                plate_regions.append((bottom_region, [0, int(height * 0.7), width, int(height * 0.3)]))
    
    except Exception as e:
        print(f"Plate region extraction error: {e}")
    
    return plate_regions


def recognize_plate_text(plate_region: np.ndarray, ocr_model) -> Optional[str]:
    """Recognize text in license plate region using OCR"""
    try:
        # Enhance plate region for better OCR
        enhanced_region = enhance_plate_image(plate_region)
        
        # Run OCR
        ocr_results = ocr_model.ocr(enhanced_region, cls=True)
        
        if ocr_results and ocr_results[0]:
            # Combine all detected text
            text_parts = []
            for line in ocr_results[0]:
                if line:
                    _, (text, confidence) = line
                    if confidence > 0.3:  # Lower threshold for plates
                        text_parts.append(text.strip())
            
            if text_parts:
                # Combine and clean text
                combined_text = ''.join(text_parts).replace(' ', '').upper()
                return combined_text
    
    except Exception as e:
        print(f"OCR recognition error: {e}")
    
    return None


def enhance_plate_image(plate_region: np.ndarray) -> np.ndarray:
    """Enhance license plate image for better OCR recognition"""
    try:
        # Convert to grayscale
        if len(plate_region.shape) == 3:
            gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_region
        
        # Resize for better OCR (if too small)
        height, width = gray.shape
        if height < 30 or width < 100:
            scale_factor = max(30 / height, 100 / width, 2.0)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Enhance contrast
        gray = cv2.equalizeHist(gray)
        
        # Convert back to BGR for PaddleOCR
        enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        return enhanced
    
    except Exception as e:
        print(f"Image enhancement error: {e}")
        return plate_region


def validate_license_plate(text: str, plate_patterns: Dict) -> Optional[Dict[str, Any]]:
    """Validate and classify license plate text against known patterns"""
    if not text or len(text) < 4:
        return None
    
    # Clean text
    cleaned_text = re.sub(r'[^A-ZА-Я0-9]', '', text.upper())
    
    # Try to match against each pattern
    for pattern_name, pattern_info in plate_patterns.items():
        pattern = pattern_info['pattern']
        
        # Convert cyrillic to latin for some patterns
        test_text = cleaned_text
        if 'kz_' in pattern_name or 'germany' in pattern_name or 'france' in pattern_name:
            test_text = convert_cyrillic_to_latin(cleaned_text)
        
        match = re.match(pattern, test_text)
        if match:
            # Extract region information if available
            region = 'Unknown'
            if pattern_info['regions'] and len(match.groups()) > 0:
                # Try to find region code in the last group (usually region code)
                region_code = match.groups()[-1]
                region = pattern_info['regions'].get(region_code, f'Region {region_code}')
            
            return {
                'plate_number': cleaned_text,
                'country': pattern_info['country'],
                'type': pattern_info['type'],
                'region': region,
                'confidence': 0.8,  # Base confidence for pattern match
                'pattern_matched': pattern_name
            }
    
    return None


def convert_cyrillic_to_latin(text: str) -> str:
    """Convert cyrillic characters to latin equivalents for pattern matching"""
    cyrillic_to_latin = {
        'А': 'A', 'В': 'B', 'Е': 'E', 'К': 'K', 'М': 'M', 'Н': 'H',
        'О': 'O', 'Р': 'P', 'С': 'C', 'Т': 'T', 'У': 'Y', 'Х': 'X',
        # Дополнительные символы для полной совместимости
        'Я': 'R'  # В некоторых случаях Я может заменяться на R
    }
    
    result = ''
    for char in text:
        result += cyrillic_to_latin.get(char, char)
    
    return result


def group_plates_by_timestamp(timestamp: str, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Group multiple license plate detections for the same timestamp into single record"""
    if not detections:
        return {
            'timestamp': timestamp,
            'plate_numbers': '',
            'countries': '',
            'plate_types': '',
            'regions': '',
            'confidences': '',
            'coordinates': '',
            'count': 0
        }
    
    # Extract data from detections
    plate_numbers = [det['plate_number'] for det in detections]
    countries = [det['country'] for det in detections]
    plate_types = [det['plate_type'] for det in detections]
    regions = [det['region'] for det in detections]
    confidences = [f"{det['confidence']:.2f}" for det in detections]
    coordinates = [f"({det['center_x']},{det['center_y']})" for det in detections]
    
    # Create grouped record
    grouped = {
        'timestamp': timestamp,
        'plate_numbers': ','.join(plate_numbers),
        'countries': ','.join(countries),
        'plate_types': ','.join(plate_types),
        'regions': ','.join(regions),
        'confidences': ','.join(confidences),
        'coordinates': ','.join(coordinates),
        'count': len(detections)
    }
    
    return grouped


def export_anpr_results(plates: List[Dict[str, Any]], video_path: str) -> None:
    """Export ANPR license plate detection results to CSV"""
    if not plates:
        print("No license plates detected to export")
        return
    
    plates_path = create_output_filename(video_path, 'plates_anpr')
    
    print(f"Exporting ANPR license plate results to: {plates_path}")
    
    # Export plates to CSV using existing exporter
    export_objects_csv(plates, plates_path)
    
    print(f"ANPR license plate results exported successfully")
    print(f"Total timestamps with plates: {len(plates)}")
    
    # Calculate total plates detected
    total_plates = sum(plate['count'] for plate in plates if plate['count'] > 0)
    print(f"Total individual license plates detected: {total_plates}")


def process_single_video_anpr(video_path: str, models: Tuple) -> None:
    """Process single video file for ANPR license plate detection"""
    print(f"\n=== ANPR License Plate Detection: {Path(video_path).name} ===")
    
    yolo_model, ocr_model, plate_patterns = models
    
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
    
    # Detect license plates
    print("Processing license plate detection...")
    confidence_threshold = config.get('object_confidence_threshold', 0.5)
    plates = detect_license_plates_batch(
        frames, timestamps, yolo_model, ocr_model, plate_patterns, confidence_threshold
    )
    print(f"License plate detection completed. Found plates in {len(plates)} timestamps")
    
    # Export results
    export_anpr_results(plates, video_path)
    
    print(f"=== ANPR License Plate Detection Completed: {Path(video_path).name} ===\n")


def main() -> None:
    """Main function for ANPR license plate detection on multiple videos"""
    print("=== ANPR License Plate Detection Pipeline Started ===")
    
    # List of video files to process
    video_files = [
        "video/AlmaAta.mp4",
        "video/news.mp4",
        "video/SovKz.mp4"
    ]
    
    # Initialize ANPR models once
    print("Initializing ANPR license plate detection models...")
    models = initialize_anpr_models()
    print("ANPR models loaded successfully")
    
    # Process each video file
    total_videos = len(video_files)
    successful_videos = 0
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n*** Processing video {i}/{total_videos} ***")
        
        try:
            process_single_video_anpr(video_path, models)
            successful_videos += 1
            print(f"✅ Successfully processed: {Path(video_path).name}")
            
        except Exception as e:
            print(f"❌ ERROR processing {video_path}: {str(e)}")
            print("Continuing with next video...")
            continue
    
    print(f"\n=== ANPR License Plate Detection Pipeline Completed ===")
    print(f"Successfully processed: {successful_videos}/{total_videos} videos")
    print(f"Failed: {total_videos - successful_videos} videos")


if __name__ == "__main__":
    main()