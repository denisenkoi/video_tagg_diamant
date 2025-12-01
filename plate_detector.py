#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plate Detector Module - YOLO-based detection for vehicles and license plates
Handles object detection using YOLO models
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO


class PlateDetector:
    """
    YOLO-based detector for vehicles and license plates
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize detector with YOLO models
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self._init_models()
        
        # Vehicle class mappings (COCO dataset)
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck',
            1: 'bicycle'
        }
    
    def _get_default_config(self) -> Dict:
        """Default detector configuration"""
        return {
            'yolo_vehicle_model': 'yolov8n.pt',
            'yolo_plate_model': 'license_plate_detector.pt',
            'min_vehicle_confidence': 0.5,
            'min_plate_confidence': 0.5,
            'plate_expansion_pixels': 10  # Expand plate bbox for better OCR
        }
    
    def _init_models(self):
        """Initialize YOLO models"""
        # Vehicle detection model
        self.logger.info(f"Loading YOLO vehicle model: {self.config['yolo_vehicle_model']}")
        self.yolo_vehicle = YOLO(self.config['yolo_vehicle_model'])
        
        # Check for enhanced vehicle model
        enhanced_path = Path('yolov8-vehicle-enhanced.pt')
        if enhanced_path.exists():
            self.logger.info("Loading enhanced vehicle model")
            self.yolo_enhanced = YOLO(str(enhanced_path))
            self.has_enhanced_vehicle = True
        else:
            self.yolo_enhanced = None
            self.has_enhanced_vehicle = False
        
        # License plate detection model
        plate_model_path = Path(self.config['yolo_plate_model'])
        if plate_model_path.exists():
            self.logger.info(f"Loading YOLO plate model: {plate_model_path}")
            self.yolo_plate = YOLO(str(plate_model_path))
            self.has_plate_detector = True
        else:
            self.logger.warning(f"Plate detection model not found: {plate_model_path}")
            self.yolo_plate = None
            self.has_plate_detector = False
    
    def detect_vehicles(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect vehicles in frame
        
        Args:
            frame: Input image as numpy array
            
        Returns:
            List of detected vehicles with bbox and confidence
        """
        vehicles = []
        
        # Use enhanced model if available
        model = self.yolo_enhanced if self.has_enhanced_vehicle else self.yolo_vehicle
        results = model(frame, conf=self.config['min_vehicle_confidence'], verbose=False)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    
                    # Check if it's a vehicle class
                    if cls_id in self.vehicle_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        vehicle = {
                            'vehicle_type': self.vehicle_classes[cls_id],
                            'bbox': [x1, y1, x2 - x1, y2 - y1],  # [x, y, width, height]
                            'confidence': float(box.conf[0]),
                            'class_id': cls_id
                        }
                        
                        vehicles.append(vehicle)
        
        self.logger.debug(f"Detected {len(vehicles)} vehicles")
        return vehicles

    def detect_plates(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect license plates in frame using YOLO

        Args:
            frame: Input image as numpy array

        Returns:
            List of detected plates with bbox
        """
        plates = []

        if not self.has_plate_detector:
            self.logger.debug("Plate detector not available")
            return plates

        # Run YOLO plate detection
        results = self.yolo_plate(frame, conf=self.config['min_plate_confidence'], verbose=False)

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    plate = {
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'confidence': float(box.conf[0]),
                        'expanded_bbox': None  # Will be filled if expansion is needed
                    }

                    # Calculate expanded bbox for OCR
                    if self.config.get('plate_expansion_pixels', 0) > 0:
                        expansion = self.config['plate_expansion_pixels']
                        vertical_expansion = expansion // 2  # Half expansion for vertical
                        frame_h, frame_w = frame.shape[:2]

                        exp_x1 = max(0, x1 - expansion)
                        exp_y1 = max(0, y1 - vertical_expansion)
                        exp_x2 = min(frame_w, x2 + expansion)
                        exp_y2 = min(frame_h, y2 + vertical_expansion)

                        plate['expanded_bbox'] = [exp_x1, exp_y1, exp_x2 - exp_x1, exp_y2 - exp_y1]

                    plates.append(plate)

        self.logger.debug(f"Detected {len(plates)} license plates")
        return plates
    
    def extract_region(self, frame: np.ndarray, bbox: List[int], use_expanded: bool = False) -> np.ndarray:
        """
        Extract region from frame using bbox
        
        Args:
            frame: Input image
            bbox: Bounding box [x, y, width, height]
            use_expanded: Use expanded bbox if available
            
        Returns:
            Extracted region as numpy array
        """
        x, y, w, h = bbox
        region = frame[y:y+h, x:x+w]
        return region
    
    def select_primary_vehicle(self, vehicles: List[Dict], frame_width: int = 1280) -> Optional[int]:
        """
        Select primary vehicle based on size and position
        
        Args:
            vehicles: List of detected vehicles
            frame_width: Width of frame for center calculation
            
        Returns:
            Index of primary vehicle or None
        """
        if not vehicles:
            return None
        
        best_score = 0
        best_idx = None
        
        for idx, vehicle in enumerate(vehicles):
            x, y, w, h = vehicle['bbox']
            area = w * h
            
            # Prefer larger vehicles (closer to camera)
            # Prefer centered vehicles
            center_x = x + w / 2
            center_offset = abs(center_x - frame_width/2) / (frame_width/2)
            
            # Score calculation: larger area and more centered = higher score
            score = area * (1 - center_offset * 0.3)
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        if best_idx is not None:
            self.logger.debug(f"Selected vehicle {best_idx} as primary")
        
        return best_idx
    
    def extract_plate_zones_from_vehicle(self, vehicle_region: np.ndarray, plate_zone_bottom: float = 0.4) -> List[np.ndarray]:
        """
        Extract potential license plate zones from vehicle region
        
        Args:
            vehicle_region: Cropped vehicle image
            plate_zone_bottom: Percentage of bottom area to search (0.4 = bottom 40%)
            
        Returns:
            List of potential plate zones
        """
        import cv2
        zones = []
        height, width = vehicle_region.shape[:2]
        
        # Method 1: Bottom portion of vehicle
        bottom_start = int(height * (1 - plate_zone_bottom))
        bottom_zone = vehicle_region[bottom_start:, :]
        zones.append(bottom_zone)
        
        # Method 2: Edge detection for rectangular regions
        gray = cv2.cvtColor(vehicle_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            
            # License plate characteristics
            if (2.0 < aspect_ratio < 6.0 and 
                area > (width * height * 0.01) and 
                area < (width * height * 0.3)):
                
                zone = vehicle_region[y:y+h, x:x+w]
                if zone.size > 0:
                    zones.append(zone)
        
        # Limit to 3 zones maximum
        return zones[:3]
