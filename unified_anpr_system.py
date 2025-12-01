#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified ANPR System - Main orchestrator module
Combines detection and recognition for complete traffic analysis
"""

import cv2
import numpy as np
import re
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Import our modules
from plate_detector import PlateDetector
from plate_recognizer import PlateRecognizer, PlateStructure


@dataclass
class RadarData:
    """Radar information from frame"""
    speed: Optional[int] = None
    speed_unit: str = 'km/h'
    direction: Optional[str] = None  # 'approaching', 'receding'
    timestamp: Optional[str] = None
    raw_text: List[str] = None
    confidence: float = 0.0


@dataclass
class Vehicle:
    """Vehicle detection result with plate"""
    vehicle_type: str
    bbox: List[int]
    confidence: float
    is_primary: bool = False
    plate: Optional[PlateStructure] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for compatibility"""
        result = {
            'vehicle_type': self.vehicle_type,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'is_primary': self.is_primary
        }
        if self.plate:
            result['plate'] = {
                'text': self.plate.full_text,
                'country': self.plate.country_code or 'Unknown',
                'confidence': self.plate.confidence
            }
        return result


class UnifiedANPRSystem:
    """
    Main ANPR system orchestrator
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize ANPR system
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        self._setup_logging()
        
        # Debug flags
        self.debug_mode = self.config.get('debug_mode', False)
        self.license_plate_debug = self.config.get('license_plate_debug', False)
        
        # Initialize components
        self.logger.info("Initializing ANPR system components...")
        
        # Detector config
        detector_config = {
            'yolo_vehicle_model': self.config.get('yolo_model', 'yolov8n.pt'),
            'yolo_plate_model': self.config.get('yolo_plate_model', 'license_plate_detector.pt'),
            'min_vehicle_confidence': self.config.get('min_vehicle_confidence', 0.5),
            'min_plate_confidence': self.config.get('min_plate_confidence', 0.5),
            'plate_expansion_pixels': 40
        }
        self.detector = PlateDetector(detector_config)
        
        # Recognizer
        self.recognizer = PlateRecognizer()
        
        self.logger.info("ANPR system initialized successfully")
    
    def _get_default_config(self) -> Dict:
        """Default configuration"""
        return {
            'yolo_model': 'yolov8n.pt',
            'min_plate_confidence': 0.5,
            'min_vehicle_confidence': 0.5,
            'enable_radar_extraction': True,
            'enable_triple_pass_ocr': True,
            'radar_zone_height': 0.05,  # Top 5% of frame
            'plate_zone_bottom': 0.4,  # Bottom 40% of vehicle
            'debug_mode': False,
            'license_plate_debug': True
        }
    
    def _setup_logging(self):
        """Setup dual logging (console + file)"""
        import os
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
        
        # Log filename with timestamp
        log_filename = f'logs/anpr_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler with UTF-8 encoding
        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)
        
        self.logger.info(f"Logging initialized: {log_filename}")

        # Prevent log duplication
        self.logger.propagate = False
    
    def _extract_radar_data(self, frame: np.ndarray) -> RadarData:
        """Extract radar information from top of frame"""
        height, width = frame.shape[:2]
        radar_zone = frame[0:int(height * self.config['radar_zone_height']), 0:min(1000, width)]
        
        # DEBUG: Show radar zone
        if self.debug_mode:
            cv2.imshow('DEBUG: Radar Zone CUT', radar_zone)
            self.logger.debug(f"Radar zone shape: {radar_zone.shape}")
            cv2.waitKey(1)
        
        radar_data = RadarData(raw_text=[])
        
        # Run OCR on radar zone
        recognition_result = self.recognizer.recognize(radar_zone)
        ocr_results = recognition_result['raw_ocr']
        
        for text, confidence, _, _, angle in ocr_results:
            radar_data.raw_text.append(text)
            self.logger.debug(f"Radar OCR: '{text}' (conf: {confidence:.2f}, angle: {angle}°)")
            
            # Extract speed
            speed_match = re.search(r'(\d+)\s*(km/h|mph)', text, re.IGNORECASE)
            if speed_match:
                radar_data.speed = int(speed_match.group(1))
                radar_data.speed_unit = speed_match.group(2).lower()
                radar_data.confidence = max(radar_data.confidence, confidence)
                self.logger.debug(f"Speed detected: {radar_data.speed} {radar_data.speed_unit}")
            
            # Extract direction
            if 'head' in text.lower():
                radar_data.direction = 'approaching'
                self.logger.debug("Direction: approaching")
            elif 'tail' in text.lower():
                radar_data.direction = 'receding'
                self.logger.debug("Direction: receding")
            
            # Extract timestamp
            time_match = re.search(r'\d{2}:\d{2}:\d{2}', text)
            if time_match:
                radar_data.timestamp = time_match.group()
                self.logger.debug(f"Timestamp: {radar_data.timestamp}")
        
        if self.debug_mode:
            self.logger.debug(f"Radar data summary - Speed: {radar_data.speed}, Direction: {radar_data.direction}")
            self.logger.debug(f"All radar texts: {radar_data.raw_text}")
        
        return radar_data
    
    def _process_yolo_plates(self, frame: np.ndarray, yolo_detections: List[Dict]) -> List[Dict]:
        """
        Process YOLO plate detections with OCR
        
        Args:
            frame: Input frame
            yolo_detections: List of YOLO plate detections
            
        Returns:
            List of plates with OCR results and full geometry
        """
        processed_plates = []
        
        for detection in yolo_detections:
            # Use expanded bbox if available
            if detection.get('expanded_bbox'):
                x, y, w, h = detection['expanded_bbox']
            else:
                x, y, w, h = detection['bbox']
            
            # Extract plate region
            plate_region = frame[y:y+h, x:x+w]
            
            if plate_region.size == 0:
                continue
            
            # Enhance and recognize
            enhanced = self._enhance_plate_image(plate_region)
            enhanced = plate_region
            recognition_result = self.recognizer.recognize(enhanced)
            plate_structure = recognition_result['plate_structure']
            raw_ocr = recognition_result['raw_ocr']
            
            # Build plate info with FULL geometry
            plate_info = {
                'bbox': detection['bbox'],
                'expanded_bbox': detection.get('expanded_bbox'),
                'yolo_confidence': detection['confidence'],
                'plate_structure': plate_structure,
                'extraction_success': plate_structure.extraction_success,
                'recognized_text': plate_structure.full_text if plate_structure.extraction_success else None,
                'layout_type': plate_structure.layout_type,
                'region_code': plate_structure.region_code,
                'country_code': plate_structure.country_code,
                'plate_number': plate_structure.plate_number,
                'ocr_confidence': plate_structure.confidence,
                'combined_confidence': (detection['confidence'] + plate_structure.confidence) / 2.0,
                # ADD FULL GEOMETRY INFO
                'geometry_analysis': {
                    'elements_count': len(raw_ocr),
                    'elements_by_position': []
                }
            }
            
            # Add detailed geometry for each OCR element
            for idx, (text, conf, bbox, poly, angle) in enumerate(raw_ocr):
                elem_info = {
                    'index': idx,
                    'text': text,
                    'confidence': conf,
                    'angle': angle,
                    'bbox': bbox,
                    'polygon': poly
                }
                
                if poly and len(poly) >= 4:
                    elem_info['bounds'] = {
                        'top': min(p[1] for p in poly),
                        'bottom': max(p[1] for p in poly),
                        'left': min(p[0] for p in poly),
                        'right': max(p[0] for p in poly),
                        'width': max(p[0] for p in poly) - min(p[0] for p in poly),
                        'height': max(p[1] for p in poly) - min(p[1] for p in poly)
                    }
                
                plate_info['geometry_analysis']['elements_by_position'].append(elem_info)
            
            processed_plates.append(plate_info)
        
        return processed_plates
    
    def _ocr_pass_fullscreen(self, frame: np.ndarray) -> List[PlateStructure]:
        """OCR pass on full screen"""
        plates = []
        
        recognition_result = self.recognizer.recognize(frame)
        plate_structure = recognition_result['plate_structure']
        
        if plate_structure.extraction_success:
            plates.append(plate_structure)
        
        return plates
    
    def _ocr_pass_vehicle_zones(self, frame: np.ndarray, vehicles: List[Dict]) -> List[PlateStructure]:
        """OCR pass on vehicle zones"""
        plates = []
        
        for vehicle in vehicles:
            x, y, w, h = vehicle['bbox']
            vehicle_region = frame[y:y+h, x:x+w]
            
            if vehicle_region.size == 0:
                continue
            
            recognition_result = self.recognizer.recognize(vehicle_region)
            plate_structure = recognition_result['plate_structure']
            
            if plate_structure.extraction_success:
                plates.append(plate_structure)
        
        return plates
    
    def _ocr_pass_plate_zones(self, frame: np.ndarray, vehicles: List[Dict]) -> List[PlateStructure]:
        """OCR pass on extracted plate zones from vehicles"""
        plates = []
        
        for vehicle in vehicles:
            x, y, w, h = vehicle['bbox']
            vehicle_region = frame[y:y+h, x:x+w]
            
            if vehicle_region.size == 0:
                continue
            
            # Extract potential plate zones
            plate_zones = self.detector.extract_plate_zones_from_vehicle(
                vehicle_region, 
                self.config['plate_zone_bottom']
            )
            
            for zone in plate_zones:
                if zone.size == 0:
                    continue
                
                enhanced = self._enhance_plate_image(zone)
                recognition_result = self.recognizer.recognize(enhanced)
                plate_structure = recognition_result['plate_structure']
                
                if plate_structure.extraction_success:
                    plates.append(plate_structure)
        
        return plates
    
    def _enhance_plate_image(self, plate_region: np.ndarray) -> np.ndarray:
        """Enhance plate image for better OCR"""
        # Convert to grayscale
        if len(plate_region.shape) == 3:
            gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_region
        
        # Resize if too small
        height, width = gray.shape
        if height < 50:
            scale = 50 / height
            new_width = int(width * scale)
            gray = cv2.resize(gray, (new_width, 50), interpolation=cv2.INTER_CUBIC)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        # Convert back to BGR for OCR
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return enhanced
    
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Main analysis method - processes single frame
        
        Args:
            frame: Input image in BGR format
            
        Returns:
            Complete analysis results
        """
        start_time = time.time()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'processing_time': 0,
            'radar_data': None,
            'vehicles': [],
            'license_plates': [],
            'yolo_plate_detections': [],
            'ocr_passes': {
                'fullscreen': [],
                'vehicle_zones': [],
                'plate_zones': []
            },
            'statistics': {}
        }
        
        # Debug visualization
        if self.debug_mode or self.license_plate_debug:
            debug_frame = frame.copy()
            vehicle_counter = 0
        
        # Step 1: Extract radar data
        if self.config['enable_radar_extraction']:
            results['radar_data'] = self._extract_radar_data(frame)
        
        # Step 2: Detect vehicles
        vehicles = self.detector.detect_vehicles(frame)
        
        # Step 3: Select primary vehicle
        frame_height, frame_width = frame.shape[:2]
        primary_idx = self.detector.select_primary_vehicle(vehicles, frame_width)
        
        # Mark primary vehicle
        for idx, vehicle in enumerate(vehicles):
            vehicle['is_primary'] = (idx == primary_idx)
        
        # Step 4: Detect license plates with YOLO
        yolo_plates = self.detector.detect_plates(frame)
        
        # Step 5: Process YOLO plates with OCR
        processed_plates = self._process_yolo_plates(frame, yolo_plates)
        results['yolo_plate_detections'] = processed_plates

        # DEBUG: Show YOLO plate detections with FULL geometry
        if self.license_plate_debug:
            self.logger.info("\n=== YOLO11 LICENSE PLATE DETECTION WITH GEOMETRIC OCR ===")
            self.logger.info(f"Found {len(yolo_plates)} license plate zones:")

            for i, plate in enumerate(processed_plates):
                # Original bbox
                x, y, w, h = plate['bbox']

                # Expanded bbox (if available)
                exp_x, exp_y, exp_w, exp_h = plate.get('expanded_bbox', [x, y, w, h])

                # Draw bounding boxes on debug frame
                if self.license_plate_debug:
                    if plate['extraction_success']:
                        color = (0, 255, 0)  # Green for successful
                        struct = plate['plate_structure']
                        label_parts = []
                        if struct.region_code:
                            label_parts.append(f"R:{struct.region_code}")
                        if struct.country_code:
                            label_parts.append(f"C:{struct.country_code}")
                        if struct.plate_number:
                            label_parts.append(f"N:{struct.plate_number}")
                        label = f"YOLO {i + 1}: {' | '.join(label_parts)} [{struct.layout_type}]"
                    else:
                        color = (0, 255, 255)  # Yellow for failed
                        label = f"YOLO {i + 1}: Detection only ({plate['yolo_confidence']:.2f})"

                    # Draw EXPANDED bbox (thick line)
                    cv2.rectangle(debug_frame, (exp_x, exp_y), (exp_x + exp_w, exp_y + exp_h), (0, 255, 255), 3)

                    # Draw ORIGINAL bbox inside expanded (thin line)
                    cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (255, 0, 255), 1)  # Magenta thin line

                    cv2.putText(debug_frame, label, (exp_x, exp_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # Add expansion info text
                    expansion_text = f"Exp: +{exp_w - w}x{exp_h - h}px"
                    cv2.putText(debug_frame, expansion_text, (exp_x, exp_y + exp_h + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

                # Log detailed info
                self.logger.info(f"  Plate {i + 1}:")
                self.logger.info(f"    Original bbox: [{x}, {y}, {w}, {h}]")
                self.logger.info(f"    Expanded bbox: [{exp_x}, {exp_y}, {exp_w}, {exp_h}]")
                self.logger.info(f"    Expansion: +{exp_w - w}px width, +{exp_h - h}px height")
                self.logger.info(f"    YOLO confidence: {plate['yolo_confidence']:.3f}")
                self.logger.info(f"    Extraction success: {plate['extraction_success']}")
                self.logger.info(f"    Layout type: {plate['layout_type']}")

                if plate['extraction_success']:
                    self.logger.info(f"    Структура: регион='{plate['region_code']}', "
                                     f"страна='{plate['country_code']}', номер='{plate['plate_number']}'")

                # Show both zones in separate windows
                # Original zone
                original_zone = frame[y:y + h, x:x + w]
                if original_zone.size > 0:
                    cv2.imshow(f'Original Zone {i + 1}', cv2.resize(original_zone, (200, 100)))

                # Expanded zone (the one actually used for OCR)
                expanded_zone = frame[exp_y:exp_y + exp_h, exp_x:exp_x + exp_w]
                if expanded_zone.size > 0:
                    # Draw original bbox outline on expanded zone for reference
                    expanded_with_outline = expanded_zone.copy()
                    orig_x_rel = x - exp_x
                    orig_y_rel = y - exp_y
                    cv2.rectangle(expanded_with_outline,
                                  (orig_x_rel, orig_y_rel),
                                  (orig_x_rel + w, orig_y_rel + h),
                                  (255, 0, 255), 2)  # Magenta outline

                    cv2.imshow(f'Expanded Zone {i + 1} (OCR)', cv2.resize(expanded_with_outline, (250, 125)))

                # Log geometry details...
                geom_info = plate['geometry_analysis']
                self.logger.info(f"    Геометрия: элементов={geom_info['elements_count']}")
                
                if geom_info['elements_by_position']:
                    self.logger.info(f"    Детальная геометрия элементов:")
                    for elem in geom_info['elements_by_position']:
                        self.logger.info(f"      [{elem['index']}] '{elem['text']}' "
                                        f"(conf={elem['confidence']:.2f}, angle={elem.get('angle', 0):.1f}°)")
                        if elem.get('bbox'):
                            self.logger.info(f"        BBox: {elem['bbox']}")
                        if elem.get('polygon'):
                            poly_str = ', '.join([f"({p[0]:.0f},{p[1]:.0f})" for p in elem['polygon']])
                            self.logger.info(f"        Polygon: [{poly_str}]")
                        if elem.get('bounds'):
                            b = elem['bounds']
                            self.logger.info(f"        Bounds: top={b['top']:.0f}, bottom={b['bottom']:.0f}, "
                                           f"left={b['left']:.0f}, right={b['right']:.0f}, "
                                           f"size={b['width']:.0f}x{b['height']:.0f}")
                
                # Show individual plate zone
                plate_zone = frame[y:y+h, x:x+w]
                if plate_zone.size > 0:
                    cv2.imshow(f'YOLO Plate Zone {i+1}', cv2.resize(plate_zone, (200, 100)))
        
        # Step 6: Triple-pass OCR if enabled
        if self.config['enable_triple_pass_ocr']:
            # Pass 1: Full screen
            fullscreen_plates = self._ocr_pass_fullscreen(frame)
            results['ocr_passes']['fullscreen'] = fullscreen_plates
            
            # Pass 2: Vehicle zones  
            vehicle_plates = self._ocr_pass_vehicle_zones(frame, vehicles)
            results['ocr_passes']['vehicle_zones'] = vehicle_plates
            
            # Pass 3: Plate zones from vehicles
            plate_zone_plates = self._ocr_pass_plate_zones(frame, vehicles)
            results['ocr_passes']['plate_zones'] = plate_zone_plates

            # Заполнить license_plates для совместимости с walker
            for plate in fullscreen_plates + vehicle_plates + plate_zone_plates:
                if plate.extraction_success:
                    results['license_plates'].append({
                        'text': plate.full_text,
                        'country': plate.country_code or 'Unknown',
                        'pattern_type': 'structured',
                        'confidence': plate.confidence,
                        'ocr_method': 'triple_pass',
                        'bbox': [0, 0, 0, 0]
                    })

            # Assign best plate to vehicles and show debug info
            for idx, vehicle in enumerate(vehicles):
                # DEBUG: Show vehicle zone
                if self.debug_mode:
                    vehicle_counter += 1
                    x, y, w, h = vehicle['bbox']
                    color = (0, 255, 255) if vehicle['is_primary'] else (255, 0, 0)
                    cv2.rectangle(debug_frame, (x, y), (x+w, y+h), color, 2)
                    label = f"Vehicle {vehicle_counter}: {vehicle['vehicle_type']}"
                    if vehicle['is_primary']:
                        label += " (PRIMARY)"
                    cv2.putText(debug_frame, label, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Show vehicle window
                    vehicle_region = frame[y:y+h, x:x+w]
                    cv2.imshow(f'DEBUG: Vehicle {vehicle_counter}', vehicle_region)
                    self.logger.debug(f"Vehicle {vehicle_counter} - Type: {vehicle['vehicle_type']}, "
                                    f"Primary: {vehicle['is_primary']}")
                
                # Collect all plates from this vehicle's region
                all_plates = vehicle_plates + plate_zone_plates
                
                if all_plates:
                    # Select plate with highest confidence
                    best_plate = max(all_plates, key=lambda p: p.confidence)
                    vehicle_obj = Vehicle(
                        vehicle_type=vehicle['vehicle_type'],
                        bbox=vehicle['bbox'],
                        confidence=vehicle['confidence'],
                        is_primary=vehicle['is_primary'],
                        plate=best_plate
                    )
                    
                    # DEBUG: Show detected plate
                    if self.debug_mode:
                        plate_text = f"Plate: {best_plate.full_text}"
                        cv2.putText(debug_frame, plate_text, (x, y+h+20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        self.logger.debug(f"Plate detected for vehicle {vehicle_counter}: "
                                        f"{best_plate.full_text} (conf: {best_plate.confidence:.2f})")
                else:
                    vehicle_obj = Vehicle(
                        vehicle_type=vehicle['vehicle_type'],
                        bbox=vehicle['bbox'],
                        confidence=vehicle['confidence'],
                        is_primary=vehicle['is_primary'],
                        plate=None
                    )
                
                # Convert to dict for compatibility with walker
                results['vehicles'].append(vehicle_obj.to_dict())
        else:
            # No triple-pass, just use vehicles without plates
            for vehicle in vehicles:
                vehicle_obj = Vehicle(
                    vehicle_type=vehicle['vehicle_type'],
                    bbox=vehicle['bbox'],
                    confidence=vehicle['confidence'],
                    is_primary=vehicle['is_primary'],
                    plate=None
                )
                results['vehicles'].append(vehicle_obj.to_dict())
        
        # Calculate statistics
        successful_yolo_plates = [p for p in processed_plates if p['extraction_success']]
        
        results['statistics'] = {
            'vehicles_detected': len(vehicles),
            'yolo_plates_detected': len(yolo_plates),
            'yolo_plates_recognized': len(successful_yolo_plates),
            'has_primary_vehicle': primary_idx is not None,
            'radar_speed': results['radar_data'].speed if results['radar_data'] else None
        }
        
        results['processing_time'] = time.time() - start_time
        
        # Final debug visualization
        if self.debug_mode or self.license_plate_debug:
            # Add radar info text
            if results['radar_data'] and results['radar_data'].speed:
                speed_text = f"Speed: {results['radar_data'].speed} km/h"
                cv2.putText(debug_frame, speed_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add YOLO plate detection info
            if yolo_plates:
                plates_text = f"YOLO Plates: {len(yolo_plates)}"
                cv2.putText(debug_frame, plates_text, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imshow('DEBUG: Final Analysis', cv2.resize(debug_frame, (1280, 720)))
            
            self.logger.info("\n=== ANALYSIS SUMMARY ===")
            self.logger.info(f"YOLO detected {len(yolo_plates)} license plate zones")
            self.logger.info(f"YOLO recognized texts in {len(successful_yolo_plates)} zones")
            self.logger.info(f"Total vehicles detected: {len(vehicles)}")
            if results['radar_data'] and results['radar_data'].speed:
                self.logger.info(f"Radar speed: {results['radar_data'].speed} km/h")
            
            self.logger.info("Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print()
        
        return results
    
    def process_video(self, video_path: str, output_path: str = None) -> Dict[str, Any]:
        """
        Process entire video file
        
        Args:
            video_path: Path to video file
            output_path: Optional output path for results
            
        Returns:
            Video processing results
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        results = {
            'video_path': video_path,
            'fps': fps,
            'total_frames': total_frames,
            'frames_analyzed': [],
            'summary': {}
        }
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame
            if frame_idx % 30 == 0:  # Process 1 frame per second for 30fps video
                frame_results = self.analyze_frame(frame)
                frame_results['frame_idx'] = frame_idx
                frame_results['timestamp_sec'] = frame_idx / fps
                results['frames_analyzed'].append(frame_results)
                
                # Log progress
                if frame_idx % 300 == 0:
                    self.logger.info(f"Processed frame {frame_idx}/{total_frames}")
            
            frame_idx += 1
        
        cap.release()
        
        # Generate summary
        results['summary'] = self._generate_summary(results['frames_analyzed'])
        
        # Save results
        if output_path:
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        return results
    
    def _generate_summary(self, frames_analyzed: List[Dict]) -> Dict:
        """Generate summary statistics"""
        if not frames_analyzed:
            return {}
        
        all_speeds = []
        vehicle_counts = []
        successful_plates = 0
        
        for frame in frames_analyzed:
            if frame.get('radar_data') and frame['radar_data'].speed:
                all_speeds.append(frame['radar_data'].speed)
            vehicle_counts.append(len(frame.get('vehicles', [])))
            successful_plates += len([p for p in frame.get('yolo_plate_detections', []) 
                                     if p.get('extraction_success')])
        
        return {
            'total_frames_analyzed': len(frames_analyzed),
            'total_successful_plate_recognitions': successful_plates,
            'avg_speed': np.mean(all_speeds) if all_speeds else None,
            'max_speed': max(all_speeds) if all_speeds else None,
            'min_speed': min(all_speeds) if all_speeds else None,
            'avg_vehicles_per_frame': np.mean(vehicle_counts) if vehicle_counts else 0,
            'max_vehicles_in_frame': max(vehicle_counts) if vehicle_counts else 0
        }


def main():
    """Example usage"""
    # Initialize system
    anpr = UnifiedANPRSystem()
    
    # Process single image
    image_path = "test_image.jpg"
    if Path(image_path).exists():
        image = cv2.imread(image_path)
        results = anpr.analyze_frame(image)
        
        print(f"Radar Speed: {results['radar_data'].speed if results['radar_data'] else 'N/A'}")
        print(f"Vehicles: {len(results['vehicles'])}")
        print(f"Successful plates: {len([p for p in results['yolo_plate_detections'] if p['extraction_success']])}")
    
    # Process video
    video_path = "test_video.mp4"
    if Path(video_path).exists():
        results = anpr.process_video(video_path, "results.json")
        print(f"Video processed: {results['summary']}")


if __name__ == "__main__":
    main()
