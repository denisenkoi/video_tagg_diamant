#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANPR Walker - Recursive image processor with unified ANPR system
Walks through directories, processes images, saves metadata
FIXED: Complete field preservation from unified_anpr_system
"""

import os
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import cv2
import numpy as np
from dotenv import load_dotenv

# Import the unified ANPR system
from unified_anpr_system import UnifiedANPRSystem
from pdf_generator import ViolationReportGenerator


class ANPRWalker:
    """
    Walker for recursive image processing with ANPR
    """

    def __init__(self, env_file: str = '.env'):
        """
        Initialize walker with configuration

        Args:
            env_file: Path to .env configuration file
        """
        print("DEBUG: ANPRWalker.__init__ START")

        # Load environment variables
        load_dotenv(env_file)

        # Essential configuration only
        self.images_directory = os.getenv('IMAGES_DIRECTORY', 'images/')
        self.min_confidence = float(os.getenv('MIN_CONFIDENCE', '0.5'))
        self.skip_processed = os.getenv('SKIP_PROCESSED', 'false').lower() == 'true'
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')

        print(f"DEBUG: Config loaded - images_directory={self.images_directory}")

        # Metadata directory name (relative to each image)
        self.metadata_dir_name = 'violations_metadata'

        # Supported image formats
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        # Setup logging FIRST - before ANPR system to avoid conflicts
        self._setup_logging()

        # Log startup immediately after logger setup
        self.logger.info("ANPR Walker initializing...")
        print("DEBUG: Logger initialized")

        # Statistics
        self.stats = {
            'start_time': None,
            'end_time': None,
            'total_files': 0,
            'processed': 0,
            'skipped': 0,
            'corrupted': 0,
            'errors': 0,
            'detections': {
                'plates': 0,
                'vehicles': 0,
                'speeds': []
            }
        }

        # Initialize ANPR system
        self.logger.info("Initializing ANPR system...")
        print("DEBUG: About to initialize ANPR system")

        anpr_config = {
            'yolo_model': 'yolov8n.pt',
            'yolo_plate_model': 'license_plate_detector.pt',
            'min_plate_confidence': self.min_confidence,
            'min_vehicle_confidence': self.min_confidence,
            'enable_radar_extraction': True,
            'enable_triple_pass_ocr': True,
            'radar_zone_height': 0.05,
            'plate_zone_bottom': 0.4,
            'debug_mode': False,
            'license_plate_debug': False,
            'plate_expansion_pixels': 40  # Expansion for better OCR
        }
        self.anpr = UnifiedANPRSystem(config=anpr_config)
        print("DEBUG: ANPR system initialized")
        self.logger.info("ANPR system initialized successfully")

        # Initialize PDF generator
        print("DEBUG: About to initialize PDF generator")
        self.pdf_generator = ViolationReportGenerator()
        print("DEBUG: PDF generator initialized")

        print("DEBUG: ANPRWalker.__init__ END")

    def _setup_logging(self):
        """Configure unified logging for all components"""
        log_levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR
        }

        # Configure root logger to handle all modules
        root_logger = logging.getLogger()

        # Clear any existing handlers to avoid duplicates
        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        # Set up console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_levels.get(self.log_level, logging.INFO))
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)

        # Add handler to root logger
        root_logger.addHandler(console_handler)
        root_logger.setLevel(log_levels.get(self.log_level, logging.INFO))

        # Prevent propagation conflicts
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = True

    def _parse_timestamp_from_filename(self, image_path: Path) -> Optional[datetime]:
        """
        Parse timestamp from filename like 20250922144438199-1.jpg

        Args:
            image_path: Path to image file

        Returns:
            Parsed datetime or None if parsing failed
        """
        print(f"DEBUG: Parsing timestamp from {image_path.absolute()}")

        filename = image_path.stem
        if '-' not in filename:
            print(f"DEBUG: No '-' in filename {filename} from {image_path.absolute()}")
            return None

        timestamp_str = filename.split('-')[0]
        if len(timestamp_str) < 14:
            print(f"DEBUG: Timestamp string too short: {timestamp_str} from {image_path.absolute()}")
            return None

        # Parse: 20250922144438199 -> 2025-09-22 14:44:38.199
        year = int(timestamp_str[:4])
        month = int(timestamp_str[4:6])
        day = int(timestamp_str[6:8])
        hour = int(timestamp_str[8:10])
        minute = int(timestamp_str[10:12])
        second = int(timestamp_str[12:14])
        ms = int(timestamp_str[14:17]) if len(timestamp_str) >= 17 else 0

        parsed_time = datetime(year, month, day, hour, minute, second, ms * 1000)
        print(f"DEBUG: Parsed timestamp: {parsed_time} from {image_path.absolute()}")
        return parsed_time

    def _group_violation_sequences(self, image_paths: List[Path]) -> Dict[str, List[Path]]:
        """
        Group images by violation timestamp with FIXED logic for frame numbers

        Args:
            image_paths: List of image file paths

        Returns:
            Dictionary mapping group keys to lists of image paths
        """
        print(f"DEBUG: GROUPER FIXED - Grouping {len(image_paths)} images into violation sequences")
        groups = {}

        # Sort paths by filename for consistent processing
        sorted_paths = sorted(image_paths, key=lambda p: p.name)

        for image_path in sorted_paths:
            print(f"DEBUG: GROUPER FIXED - Processing {image_path.name} for grouping")

            # Parse filename: 20250922144438199-1.jpg
            filename = image_path.stem
            if '-' not in filename:
                print(f"DEBUG: GROUPER FIXED - No '-' in filename, creating single group")
                groups[str(image_path)] = [image_path]
                continue

            parts = filename.split('-')
            if len(parts) != 2:
                print(f"DEBUG: GROUPER FIXED - Invalid filename format, creating single group")
                groups[str(image_path)] = [image_path]
                continue

            timestamp_str = parts[0]
            frame_number_str = parts[1]

            # Parse frame number
            frame_number = int(frame_number_str)

            # Parse timestamp
            timestamp = self._parse_timestamp_from_filename(image_path)
            if not timestamp:
                print(f"DEBUG: GROUPER FIXED - Failed to parse timestamp, creating single group")
                groups[str(image_path)] = [image_path]
                continue

            print(f"DEBUG: GROUPER FIXED - Parsed: timestamp={timestamp}, frame={frame_number}")

            # CRITICAL FIX: If frame number is 1, ALWAYS start new series and EXIT
            if frame_number == 1:
                group_key = f"violation_{timestamp.isoformat()}_frame_{frame_number}"
                print(f"DEBUG: GROUPER FIXED - Frame 1 detected - CREATING NEW series: {group_key}")
                groups[group_key] = [image_path]
                print(
                    f"DEBUG: GROUPER FIXED - Created group {group_key} with files: {[f.name for f in groups[group_key]]}")
                continue  # EXIT HERE - do not search for existing groups!

            # Only reach here if frame_number > 1
            print(f"DEBUG: GROUPER FIXED - Frame {frame_number} > 1, searching for existing series to continue")

            # For frame numbers > 1, try to find existing series to continue
            found_group = False

            for group_key, group_files in groups.items():
                if not group_files:
                    continue

                # Check if this could be a continuation of existing series
                last_file = group_files[-1]  # Get last file in series
                last_filename = last_file.stem

                if '-' not in last_filename:
                    continue

                last_parts = last_filename.split('-')
                if len(last_parts) != 2:
                    continue

                last_frame_number = int(last_parts[1])

                last_timestamp = self._parse_timestamp_from_filename(last_file)
                if not last_timestamp:
                    continue

                # Check conditions for series continuation:
                # 1. Frame number is sequential (last_frame + 1)
                # 2. Time difference <= 0.5 seconds
                time_diff = abs((timestamp - last_timestamp).total_seconds())
                is_sequential_frame = (frame_number == last_frame_number + 1)
                is_within_time_window = (time_diff <= 0.5)

                print(f"DEBUG: GROUPER FIXED - Checking against group {group_key}:")
                print(
                    f"DEBUG: GROUPER FIXED -   Last frame: {last_frame_number}, current: {frame_number}, sequential: {is_sequential_frame}")
                print(f"DEBUG: GROUPER FIXED -   Time diff: {time_diff:.3f}s, within window: {is_within_time_window}")

                if is_sequential_frame and is_within_time_window:
                    print(f"DEBUG: GROUPER FIXED - Adding to existing series {group_key}")
                    group_files.append(image_path)
                    print(
                        f"DEBUG: GROUPER FIXED - Updated group {group_key} with files: {[f.name for f in group_files]}")
                    found_group = True
                    break

            if not found_group:
                # Create new series (shouldn't happen often for frame > 1, but possible if sequence broken)
                group_key = f"violation_{timestamp.isoformat()}_frame_{frame_number}"
                print(f"DEBUG: GROUPER FIXED - Creating new series for orphaned frame: {group_key}")
                groups[group_key] = [image_path]
                print(
                    f"DEBUG: GROUPER FIXED - Created orphaned group {group_key} with files: {[f.name for f in groups[group_key]]}")

        print(f"DEBUG: GROUPER FIXED - Final result: {len(groups)} violation groups")
        for group_key, group_files in groups.items():
            print(f"DEBUG: GROUPER FIXED - Group {group_key}: {len(group_files)} files")
            for gf in group_files:
                print(f"DEBUG: GROUPER FIXED -   {gf.name}")

        return groups

    def _select_best_frame_from_group(self, violation_group: List[Path]) -> Path:
        """
        Select best frame from violation group based on processing success and OCR confidence

        Args:
            violation_group: List of image paths in the same violation

        Returns:
            Path to the best frame
        """
        print(f"DEBUG: Selecting best frame from {len(violation_group)} files")

        if len(violation_group) == 1:
            print(f"DEBUG: Single file, returning {violation_group[0].name}")
            return violation_group[0]

        best_frame = None
        best_score = -1

        for image_path in violation_group:
            print(f"DEBUG: Evaluating {image_path.name}")
            metadata_path = self.get_metadata_path(image_path)
            if not metadata_path.exists():
                print(f"DEBUG: No metadata for {image_path.name}")
                continue

            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            score = 0

            # Priority 1: Successful processing
            if metadata.get('status') == 'success':
                score += 100
                print(f"DEBUG: {image_path.name} has success status (+100)")

            # Priority 2: Best OCR confidence from YOLO plates
            yolo_plates = metadata.get('yolo_plate_detections', [])
            if yolo_plates:
                # Look for plates with recognized text
                max_confidence = 0
                for p in yolo_plates:
                    # Check both possible fields
                    if p.get('recognized_text') or p.get('best_recognized_text'):
                        conf = p.get('ocr_confidence', p.get('best_ocr_confidence', 0))
                        max_confidence = max(max_confidence, conf)

                score += max_confidence * 10
                print(f"DEBUG: {image_path.name} OCR confidence: {max_confidence} (+{max_confidence * 10})")

            print(f"DEBUG: {image_path.name} total score: {score}")

            if score > best_score:
                best_score = score
                best_frame = image_path
                print(f"DEBUG: New best frame: {image_path.name}")

        result = best_frame or violation_group[0]
        print(f"DEBUG: Final best frame: {result.name}")
        return result

    def _safe_imread(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Safe image reading that bypasses ultralytics patches
        Validates image header first, then uses PIL for loading

        Args:
            image_path: Path to image file

        Returns:
            Image as numpy array or None if failed
        """
        # Check if file exists and not empty
        if not image_path.exists():
            self.logger.error(f"File does not exist: {image_path}")
            return None

        file_size = image_path.stat().st_size
        if file_size == 0:
            self.logger.error(f"File is empty: {image_path}")
            return None

        # Check file extension
        if image_path.suffix.lower() not in self.image_extensions:
            self.logger.error(f"Unsupported file extension: {image_path}")
            return None

        # Read first few bytes to validate image header
        with open(image_path, 'rb') as f:
            header = f.read(12)

        if len(header) < 4:
            self.logger.error(f"File too small to have valid header: {image_path}")
            return None

        # Check for valid image headers
        is_valid_image = (
                header.startswith(b'\xff\xd8\xff') or  # JPEG
                header.startswith(b'\x89PNG') or  # PNG
                header.startswith(b'BM') or  # BMP
                header.startswith(b'RIFF') or  # WebP/AVI
                header.startswith(b'GIF87a') or  # GIF87a
                header.startswith(b'GIF89a') or  # GIF89a
                header[6:].startswith(b'JFIF') or  # JPEG with JFIF
                header[6:].startswith(b'Exif')  # JPEG with EXIF
        )

        if not is_valid_image:
            self.logger.error(f"File does not have valid image header: {image_path}")
            return None

        # Use PIL to load the image (bypasses ultralytics completely)
        from PIL import Image as PILImage

        with PILImage.open(image_path) as pil_image:
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')

            # Convert PIL to OpenCV format (RGB to BGR)
            image_array = np.array(pil_image)
            image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

            # Validate image dimensions
            if len(image.shape) != 3 or image.shape[2] != 3:
                self.logger.error(f"Invalid image format: {image_path}, shape: {image.shape}")
                return None

            self.logger.debug(f"Successfully loaded image: {image_path.name}, shape: {image.shape}")
            return image

    def find_images(self, directory: str = None) -> Dict[Path, List[Path]]:
        """
        Find all images organized by directory

        Args:
            directory: Directory to search (uses configured if None)

        Returns:
            Dictionary mapping directory paths to lists of image files in that directory
        """
        search_dir = Path(directory or self.images_directory)
        print(f"DEBUG: Finding images in {search_dir.absolute()}")

        if not search_dir.exists():
            self.logger.error(f"Directory not found: {search_dir}")
            print(f"DEBUG: Directory {search_dir.absolute()} does not exist")
            return {}

        self.logger.info(f"Searching for images in: {search_dir}")

        # Group images by their parent directory
        images_by_dir = {}

        # Find all image files recursively
        all_images = []
        for ext in self.image_extensions:
            all_images.extend(search_dir.rglob(f'*{ext}'))
            all_images.extend(search_dir.rglob(f'*{ext.upper()}'))

        # Remove duplicates
        all_images = sorted(list(set(all_images)))

        # Group by parent directory
        for img_path in all_images:
            parent_dir = img_path.parent
            if parent_dir not in images_by_dir:
                images_by_dir[parent_dir] = []
            images_by_dir[parent_dir].append(img_path)

        # Sort images within each directory
        for dir_path in images_by_dir:
            images_by_dir[dir_path] = sorted(images_by_dir[dir_path])

        total_images = sum(len(imgs) for imgs in images_by_dir.values())
        print(f"DEBUG: Found {total_images} images in {len(images_by_dir)} directories")

        for dir_path, images in images_by_dir.items():
            print(f"DEBUG: Directory {dir_path.absolute()}: {len(images)} images")
            for img in images[:3]:  # Show first 3 per directory
                print(f"DEBUG:   Image: {img.absolute()}")
            if len(images) > 3:
                print(f"DEBUG:   ... and {len(images) - 3} more")

        self.logger.info(f"Found {total_images} images in {len(images_by_dir)} directories")
        return images_by_dir

    def get_metadata_path(self, image_path: Path) -> Path:
        """
        Get metadata JSON path for an image

        Args:
            image_path: Path to image file

        Returns:
            Path to metadata JSON file
        """
        # Create metadata directory next to image
        metadata_dir = image_path.parent / self.metadata_dir_name
        metadata_dir.mkdir(exist_ok=True)

        # JSON file with same name as image
        json_name = image_path.stem + '.json'
        return metadata_dir / json_name

    def should_process(self, image_path: Path) -> bool:
        """
        Check if image should be processed

        Args:
            image_path: Path to image file

        Returns:
            True if should process, False if skip
        """
        if not self.skip_processed:
            return True

        metadata_path = self.get_metadata_path(image_path)

        if not metadata_path.exists():
            return True

        # Check if previous processing was successful
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Re-process if previous attempt failed
        if metadata.get('status') == 'error':
            self.logger.debug(f"Re-processing failed image: {image_path.name}")
            return True

        # Skip if successfully processed
        if metadata.get('status') == 'success':
            return False

        return True

    def process_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Process single image with ANPR

        Args:
            image_path: Path to image file

        Returns:
            Processing results
        """
        print(f"DEBUG: Processing image {image_path.name}")
        start_time = time.time()

        # Safe image loading that bypasses ultralytics patches
        image = self._safe_imread(image_path)
        if image is None:
            print(f"DEBUG: Failed to load image {image_path.name}")
            self.logger.warning(f"Skipping corrupted/invalid image: {image_path}")
            # Return basic metadata for skipped file
            return {
                'status': 'skipped',
                'error': 'Invalid or corrupted image file',
                'file_info': {
                    'path': str(image_path),
                    'name': image_path.name,
                    'size': image_path.stat().st_size if image_path.exists() else 0
                },
                'processing_info': {
                    'timestamp': datetime.now().isoformat(),
                    'processing_time': time.time() - start_time,
                    'anpr_version': '2.0'
                }
            }

        # Run ANPR analysis
        print(f"DEBUG: Running ANPR analysis on {image_path.name}")
        self.logger.debug(f"Analyzing: {image_path.name}")
        anpr_results = self.anpr.analyze_frame(image)
        print(f"DEBUG: ANPR analysis complete for {image_path.name}")

        # Prepare metadata
        metadata = {
            'status': 'success',
            'file_info': {
                'path': str(image_path),
                'name': image_path.name,
                'size': image_path.stat().st_size,
                'modified': datetime.fromtimestamp(image_path.stat().st_mtime).isoformat(),
                'dimensions': {
                    'width': image.shape[1],
                    'height': image.shape[0]
                }
            },
            'processing_info': {
                'timestamp': datetime.now().isoformat(),
                'processing_time': time.time() - start_time,
                'anpr_version': '2.0',
                'min_confidence': self.min_confidence
            },
            'radar_data': None,
            'vehicles': [],
            'license_plates': [],
            'yolo_plate_detections': [],
            'statistics': {}
        }

        # Extract radar data
        if anpr_results.get('radar_data'):
            radar = anpr_results['radar_data']
            metadata['radar_data'] = {
                'speed': radar.speed,
                'speed_unit': radar.speed_unit,
                'direction': radar.direction,
                'timestamp': radar.timestamp,
                'confidence': radar.confidence
            }
            if radar.speed:
                self.stats['detections']['speeds'].append(radar.speed)
                print(f"DEBUG: Found radar speed: {radar.speed} km/h")

        # FIXED: Extract YOLO plate detections with ALL fields
        yolo_plates = anpr_results.get('yolo_plate_detections', [])
        print(f"DEBUG: Found {len(yolo_plates)} YOLO plate detections")

        for i, plate in enumerate(yolo_plates):
            print(f"DEBUG: Processing YOLO plate {i + 1}, keys: {list(plate.keys())}")

            # Copy ALL fields from unified_anpr_system
            plate_data = {
                # Core detection fields
                'bbox': plate.get('bbox'),
                'expanded_bbox': plate.get('expanded_bbox'),  # CRITICAL for PDF extraction!
                'yolo_confidence': plate.get('yolo_confidence'),

                # Recognition status
                'extraction_success': plate.get('extraction_success'),

                # Recognized text components
                'recognized_text': plate.get('recognized_text'),  # Full text like "01 459 ADG"
                'region_code': plate.get('region_code'),  # "01"
                'country_code': plate.get('country_code'),  # "KG?"
                'plate_number': plate.get('plate_number'),  # "459 ADG"

                # Layout and confidence
                'layout_type': plate.get('layout_type'),
                'ocr_confidence': plate.get('ocr_confidence'),
                'combined_confidence': plate.get('combined_confidence'),

                # Geometry analysis (if needed)
                'geometry_analysis': plate.get('geometry_analysis'),

                # Aliases for backward compatibility with old code
                'best_recognized_text': plate.get('recognized_text'),  # Alias
                'best_ocr_confidence': plate.get('ocr_confidence'),  # Alias

                # Source identifier
                'source': 'yolo11_detection_with_ocr'
            }

            print(f"DEBUG: YOLO plate {i + 1} extracted data:")
            print(f"  - bbox: {plate_data['bbox']}")
            print(f"  - expanded_bbox: {plate_data['expanded_bbox']}")
            print(f"  - extraction_success: {plate_data['extraction_success']}")
            print(f"  - recognized_text: {plate_data['recognized_text']}")
            print(f"  - region_code: {plate_data['region_code']}")
            print(f"  - country_code: {plate_data['country_code']}")
            print(f"  - plate_number: {plate_data['plate_number']}")

            metadata['yolo_plate_detections'].append(plate_data)

        # Extract vehicles
        vehicles_data = anpr_results.get('vehicles', [])
        print(f"DEBUG: Found {len(vehicles_data)} vehicles")
        for vehicle in vehicles_data:
            vehicle_data = {
                'type': vehicle.get('vehicle_type'),
                'bbox': vehicle.get('bbox'),
                'confidence': vehicle.get('confidence'),
                'is_primary': vehicle.get('is_primary', False)
            }

            # Add plate if detected
            if vehicle.get('plate'):
                plate = vehicle['plate']
                vehicle_data['plate'] = {
                    'text': plate['text'],
                    'country': plate['country'],
                    'confidence': plate['confidence']
                }

            metadata['vehicles'].append(vehicle_data)

        # Extract all plates (including those not assigned to vehicles)
        license_plates_data = anpr_results.get('license_plates', [])
        print(f"DEBUG: Found {len(license_plates_data)} license plates")
        for plate in license_plates_data:
            metadata['license_plates'].append({
                'text': plate['text'],
                'country': plate['country'],
                'pattern_type': plate['pattern_type'],
                'confidence': plate['confidence'],
                'ocr_method': plate['ocr_method'],
                'bbox': plate['bbox']
            })

        # Statistics
        metadata['statistics'] = {
            'vehicles_detected': len(metadata['vehicles']),
            'plates_detected': len(metadata['license_plates']),
            'yolo_plates_detected': len(metadata['yolo_plate_detections']),
            'yolo_plates_with_ocr': len([p for p in metadata['yolo_plate_detections'] if p.get('recognized_text')]),
            'unique_plates': len(set(p['text'] for p in metadata['license_plates'])),
            'has_speed_data': metadata['radar_data'] is not None and metadata['radar_data']['speed'] is not None,
            'primary_vehicle_detected': any(v['is_primary'] for v in metadata['vehicles'])
        }

        # Update global stats
        self.stats['detections']['vehicles'] += metadata['statistics']['vehicles_detected']
        self.stats['detections']['plates'] += metadata['statistics']['plates_detected']

        # Add processing time
        metadata['processing_info']['processing_time'] = time.time() - start_time

        print(f"DEBUG: Processed {image_path.name} successfully")
        print(f"DEBUG: Detections: {metadata['statistics']['vehicles_detected']} vehicles, "
              f"{metadata['statistics']['plates_detected']} plates, "
              f"{metadata['statistics']['yolo_plates_detected']} YOLO plates, "
              f"speed: {metadata['radar_data']['speed'] if metadata['radar_data'] else 'N/A'} km/h")

        return metadata

    def save_metadata(self, image_path: Path, metadata: Dict[str, Any]) -> bool:
        """
        Save metadata to JSON file

        Args:
            image_path: Path to image file
            metadata: Metadata to save

        Returns:
            True if saved successfully
        """
        metadata_path = self.get_metadata_path(image_path)

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

        self.logger.debug(f"Saved metadata: {metadata_path}")
        print(f"DEBUG: Saved metadata for {image_path.name}")
        return True

    def run(self, directory: str = None) -> Dict[str, Any]:
        """
        Run walker on directory

        Args:
            directory: Directory to process (uses configured if None)

        Returns:
            Processing statistics
        """
        print("DEBUG: ANPRWalker.run() START")
        self.stats['start_time'] = datetime.now()

        self.logger.info("=" * 60)
        self.logger.info("ANPR Walker Started")
        print("DEBUG: ANPR Walker Started")

        self.logger.info(f"Configuration:")
        self.logger.info(f"  Images directory: {directory or self.images_directory}")
        self.logger.info(f"  Min confidence: {self.min_confidence}")
        self.logger.info(f"  Skip processed: {self.skip_processed}")
        self.logger.info("=" * 60)

        # Find all images organized by directory
        print("DEBUG: About to find images by directory")
        images_by_dir = self.find_images(directory)

        total_files = sum(len(imgs) for imgs in images_by_dir.values())
        self.stats['total_files'] = total_files

        if not images_by_dir:
            print("DEBUG: No images found in any directory")
            self.logger.warning("No images found to process")
            return self._finalize_stats()

        print(f"DEBUG: Found images in {len(images_by_dir)} directories")

        # Process each directory separately
        violation_idx = 0
        for dir_path, dir_images in images_by_dir.items():
            print(f"DEBUG: Processing directory: {dir_path.absolute()}")
            print(f"DEBUG: Directory has {len(dir_images)} images")

            # Group images by violation sequences WITHIN THIS DIRECTORY
            print(f"DEBUG: About to group violation sequences for directory {dir_path.absolute()}")
            violation_groups = self._group_violation_sequences(dir_images)
            self.logger.info(
                f"Directory {dir_path.name}: Grouped {len(dir_images)} images into {len(violation_groups)} violations")
            print(
                f"DEBUG: Directory {dir_path.absolute()}: Grouped {len(dir_images)} images into {len(violation_groups)} violations")

            # Process each violation group in this directory
            print(f"DEBUG: Starting to process {len(violation_groups)} violation groups in {dir_path.absolute()}")

            for group_key, group_files in violation_groups.items():
                violation_idx += 1
                print(f"DEBUG: Processing violation {violation_idx} (global): {group_key} in {dir_path.absolute()}")

                if len(group_files) > 1:
                    self.logger.info(
                        f"[Violation {violation_idx}] Processing violation group in {dir_path.name}: {len(group_files)} frames")
                    print(f"DEBUG: Multi-frame group with {len(group_files)} frames in {dir_path.absolute()}")
                    for frame_path in group_files:
                        self.logger.info(f"  Frame: {frame_path.name}")
                        print(f"DEBUG:   Frame: {frame_path.absolute()}")
                else:
                    self.logger.info(
                        f"[Violation {violation_idx}] Processing single frame in {dir_path.name}: {group_files[0].name}")
                    print(f"DEBUG: Single frame: {group_files[0].absolute()}")

                # Process all frames in group
                processed_frames = []
                print(f"DEBUG: Processing {len(group_files)} frames in this group")

                for frame_idx, image_path in enumerate(group_files):
                    print(f"DEBUG: Processing frame {frame_idx + 1}/{len(group_files)}: {image_path.absolute()}")

                    if not self.should_process(image_path):
                        self.logger.info(f"    Skipping (already processed): {image_path.name}")
                        print(f"DEBUG: Skipping {image_path.absolute()} (already processed)")
                        self.stats['skipped'] += 1

                        # Load existing metadata for PDF generation
                        metadata_path = self.get_metadata_path(image_path)
                        if metadata_path.exists():
                            with open(metadata_path, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                            if metadata.get('status') == 'success':
                                processed_frames.append((image_path, metadata))
                        continue

                    # Process image
                    print(f"DEBUG: About to call process_image for {image_path.absolute()}")
                    metadata = self.process_image(image_path)
                    print(f"DEBUG: process_image returned for {image_path.absolute()}")

                    # Save metadata
                    if self.save_metadata(image_path, metadata):
                        if metadata['status'] == 'success':
                            self.stats['processed'] += 1
                            processed_frames.append((image_path, metadata))
                            print(f"DEBUG: Added {image_path.absolute()} to processed_frames")
                        elif metadata['status'] == 'skipped':
                            self.stats['corrupted'] += 1
                            self.logger.warning(f"  Corrupted file: {metadata.get('error', 'Unknown error')}")

                    # Log detection summary for successful processing
                    if metadata['status'] == 'success':
                        stats = metadata['statistics']
                        detections = []

                        if stats['vehicles_detected'] > 0:
                            detections.append(f"{stats['vehicles_detected']} vehicles")
                        if stats['plates_detected'] > 0:
                            detections.append(f"{stats['plates_detected']} plates")
                        if stats['yolo_plates_detected'] > 0:
                            detections.append(f"{stats['yolo_plates_detected']} YOLO plates")
                        if stats['yolo_plates_with_ocr'] > 0:
                            detections.append(f"{stats['yolo_plates_with_ocr']} with OCR")
                        if metadata['radar_data'] and metadata['radar_data']['speed']:
                            detections.append(f"speed: {metadata['radar_data']['speed']} km/h")

                        if detections:
                            self.logger.info(f"    Found: {', '.join(detections)}")
                            print(f"DEBUG: Detections: {', '.join(detections)}")
                        else:
                            self.logger.info(f"    No detections")
                            print(f"DEBUG: No detections for {image_path.absolute()}")

                print(f"DEBUG: Finished processing {len(group_files)} frames in group {group_key}")
                print(f"DEBUG: processed_frames count for this group: {len(processed_frames)}")

                # Select best frame and generate PDF
                if processed_frames:
                    print(f"DEBUG: GROUP HAS PROCESSED FRAMES - GENERATING PDF")
                    print(f"DEBUG: Selecting best frame from {len(processed_frames)} processed frames")
                    best_frame_path = self._select_best_frame_from_group([p[0] for p in processed_frames])
                    best_metadata = next(m[1] for m in processed_frames if m[0] == best_frame_path)

                    self.logger.info(f"  Selected best frame: {best_frame_path.name}")
                    print(f"DEBUG: Selected best frame: {best_frame_path.absolute()}")

                    # Generate PDF for best frame
                    print(f"DEBUG: About to load image for PDF generation: {best_frame_path.absolute()}")
                    best_image = self._safe_imread(best_frame_path)
                    if best_image is not None:
                        print(f"DEBUG: Image loaded successfully, calling PDF generator")
                        print(f"DEBUG: PDF generator object: {self.pdf_generator}")
                        print(f"DEBUG: About to call create_violation_report")

                        pdf_path = self.pdf_generator.create_violation_report(
                            best_frame_path, best_image, best_metadata
                        )
                        print(f"DEBUG: PDF generator returned: {pdf_path}")
                        self.logger.info(f"  Generated PDF: {pdf_path.name}")
                        print(f"DEBUG: Generated PDF: {pdf_path.absolute()}")
                    else:
                        print(f"DEBUG: Failed to load image for PDF generation: {best_frame_path.absolute()}")
                else:
                    print(f"DEBUG: NO PROCESSED FRAMES FOR THIS GROUP - SKIPPING PDF")

                print(f"DEBUG: FINISHED PROCESSING GROUP {group_key}")

                # Progress update every 10 violations
                if violation_idx % 10 == 0:
                    elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
                    rate = violation_idx / elapsed if elapsed > 0 else 0
                    self.logger.info(f"Progress: {violation_idx} violations processed, {rate:.1f} violations/sec")

            print(f"DEBUG: FINISHED PROCESSING ALL GROUPS IN DIRECTORY {dir_path.absolute()}")

        print("DEBUG: Finished processing all directories and violation groups")
        return self._finalize_stats()

    def _finalize_stats(self) -> Dict[str, Any]:
        """Finalize and return statistics"""
        print("DEBUG: Finalizing stats")
        self.stats['end_time'] = datetime.now()

        if self.stats['start_time']:
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            self.stats['duration_seconds'] = duration

        # Calculate speed statistics
        if self.stats['detections']['speeds']:
            speeds = self.stats['detections']['speeds']
            self.stats['detections']['speed_stats'] = {
                'min': min(speeds),
                'max': max(speeds),
                'avg': sum(speeds) / len(speeds),
                'count': len(speeds)
            }

        # Print summary
        self.logger.info("=" * 60)
        self.logger.info("ANPR Walker Completed")
        self.logger.info(f"Total files: {self.stats['total_files']}")
        self.logger.info(f"Processed: {self.stats['processed']}")
        self.logger.info(f"Skipped: {self.stats['skipped']}")
        self.logger.info(f"Corrupted: {self.stats['corrupted']}")
        self.logger.info(f"Errors: {self.stats['errors']}")
        self.logger.info(f"Total vehicles detected: {self.stats['detections']['vehicles']}")
        self.logger.info(f"Total plates detected: {self.stats['detections']['plates']}")

        if self.stats['detections'].get('speed_stats'):
            speed_stats = self.stats['detections']['speed_stats']
            self.logger.info(f"Speed data: {speed_stats['count']} readings")
            self.logger.info(f"  Min: {speed_stats['min']} km/h")
            self.logger.info(f"  Max: {speed_stats['max']} km/h")
            self.logger.info(f"  Avg: {speed_stats['avg']:.1f} km/h")

        if 'duration_seconds' in self.stats:
            self.logger.info(f"Duration: {self.stats['duration_seconds']:.1f} seconds")
            if self.stats['processed'] > 0:
                rate = self.stats['processed'] / self.stats['duration_seconds']
                self.logger.info(f"Processing rate: {rate:.2f} images/second")

        self.logger.info("=" * 60)
        print("DEBUG: Stats finalized")

        return self.stats


def main():
    """Main entry point"""
    print("DEBUG: main() function START")
    import argparse

    parser = argparse.ArgumentParser(description='ANPR Walker - Process images with ANPR')
    parser.add_argument('--directory', type=str, help='Directory to process (overrides .env)')
    parser.add_argument('--env', type=str, default='.env', help='Path to .env file')

    args = parser.parse_args()
    print(f"DEBUG: Arguments parsed - directory={args.directory}, env={args.env}")

    # Create and run walker
    print("DEBUG: About to create ANPRWalker")
    walker = ANPRWalker(env_file=args.env)
    print("DEBUG: ANPRWalker created, about to call run()")

    stats = walker.run(directory=args.directory)
    print("DEBUG: walker.run() completed")

    # Save final statistics
    stats_file = Path('anpr_walker_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\nStatistics saved to: {stats_file}")
    print("DEBUG: main() function END")

    # Exit code based on errors
    exit(0 if stats['errors'] == 0 else 1)


if __name__ == "__main__":
    print("DEBUG: Script started")
    main()
    print("DEBUG: Script ended")