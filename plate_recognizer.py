#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plate Recognizer Module - OCR and plate structure detection
Handles text recognition, geometric analysis, and plate parsing
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from paddleocr import PaddleOCR


@dataclass
class PlateStructure:
    """Structured license plate data"""
    region_code: Optional[str] = None
    country_code: Optional[str] = None
    plate_number: Optional[str] = None
    full_text: str = ''
    layout_type: str = 'unknown'  # 'stacked', 'linear', 'single', 'linear_3_elements', 'linear_4_elements'
    extraction_success: bool = False  # indicates if extraction was successful
    confidence: float = 0.0


class PlateRecognizer:
    """
    OCR and plate structure recognition
    """

    def __init__(self):
        """Initialize recognizer"""
        self.logger = logging.getLogger(__name__)

        # Initialize PaddleOCR with English for better plate recognition
        self.logger.info("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(lang='en', use_angle_cls=True)

        # Load plate patterns
        self.plate_patterns = self._load_plate_patterns()

    def _load_plate_patterns(self) -> Dict[str, Dict]:
        """Load license plate patterns for Central Asian countries"""
        return {
            # Kazakhstan
            'kz_new': {
                'pattern': r'^(\d{3})([A-Z]{3})(\d{2})$',
                'format': 'XXXAAA##',
                'country': 'Kazakhstan',
                'type': 'new_format'
            },
            'kz_old': {
                'pattern': r'^([A-Z])(\d{3})([A-Z]{2})(\d{2})$',
                'format': 'AXXXBB##',
                'country': 'Kazakhstan',
                'type': 'old_format'
            },

            # Kyrgyzstan
            'kg_new': {
                'pattern': r'^(\d{2})(KG)(\d{3})([A-Z]{3})$',
                'format': '##KG###AAA',
                'country': 'Kyrgyzstan',
                'type': 'new_format'
            },
            'kg_old': {
                'pattern': r'^([A-Z])\s*(\d{4})\s*([A-Z]{2})$',
                'format': 'A #### BB',
                'country': 'Kyrgyzstan',
                'type': 'old_format'
            },

            # Uzbekistan
            'uz_new': {
                'pattern': r'^(\d{2})([A-Z])(\d{3})([A-Z]{2})$',
                'format': '##A###BB',
                'country': 'Uzbekistan',
                'type': 'standard'
            },

            # Tajikistan
            'tj_standard': {
                'pattern': r'^(\d{4})([A-Z]{2})(\d{2})$',
                'format': '####AA##',
                'country': 'Tajikistan',
                'type': 'standard'
            },

            # Russia (common in region)
            'ru_standard': {
                'pattern': r'^([A-Z])(\d{3})([A-Z]{2})(\d{2,3})$',
                'format': 'A###BB##(#)',
                'country': 'Russia',
                'type': 'standard'
            }
        }

    def recognize(self, image) -> Dict[str, Any]:
        """
        Main recognition method

        Args:
            image: Image array (numpy array)

        Returns:
            Dictionary with OCR results and structured plate data
        """
        ocr_results = self.ocr.ocr(image)
        ocr_extracted = self._safe_extract_ocr_results(ocr_results)

        # Detect plate structure
        plate_structure = self._detect_plate_structure(ocr_extracted)

        return {
            'raw_ocr': ocr_extracted,
            'plate_structure': plate_structure
        }

    def _safe_extract_ocr_results(self, ocr_results) -> List[
        Tuple[str, float, Optional[List[int]], Optional[List[List[int]]], Optional[float]]]:
        """
        Safely extract text, confidence, bbox, polygon, and angle from PaddleOCR results

        Returns:
            List of (text, confidence, bbox, polygon, angle) tuples
        """
        extracted_results = []

        if not ocr_results or not ocr_results[0]:
            return extracted_results

        # Handle different OCR result formats
        if isinstance(ocr_results[0], dict):
            # New format with dict keys
            texts = ocr_results[0].get('rec_texts', [])
            scores = ocr_results[0].get('rec_scores', [])
            boxes = ocr_results[0].get('rec_boxes', [])
            polys = ocr_results[0].get('rec_polys', [])

            # Extract angles
            angles = ocr_results[0].get('textline_orientation_angles', [])
            if not angles:
                angles = ocr_results[0].get('rec_angles', [])

            for i, (text, confidence) in enumerate(zip(texts, scores)):
                if text and len(text.strip()) > 0:
                    # Extract bbox
                    bbox = None
                    if i < len(boxes) and boxes[i] is not None:
                        bbox = boxes[i].tolist() if hasattr(boxes[i], 'tolist') else list(boxes[i])

                    # Extract polygon
                    polygon = None
                    if i < len(polys) and polys[i] is not None:
                        polygon = polys[i].tolist() if hasattr(polys[i], 'tolist') else list(polys[i])

                    # Extract angle
                    angle = None
                    if i < len(angles) and angles[i] is not None:
                        angle = float(angles[i])

                    extracted_results.append((text.strip(), float(confidence), bbox, polygon, angle))

                    # LOG angle for analysis
                    if angle is not None:
                        self.logger.debug(f"OCR text: '{text.strip()}', angle: {angle:.1f}°")

        elif isinstance(ocr_results[0], list):
            # Old format
            for item in ocr_results[0]:
                if len(item) >= 2:
                    bbox = None
                    polygon = None
                    angle = None  # No angle info in old format

                    if len(item) > 0 and item[0] is not None:
                        if hasattr(item[0], '__iter__') and len(item[0]) >= 4:
                            poly_points = item[0]
                            if all(len(p) >= 2 for p in poly_points):
                                polygon = [[int(p[0]), int(p[1])] for p in poly_points]
                                x_coords = [p[0] for p in polygon]
                                y_coords = [p[1] for p in polygon]
                                x, y = min(x_coords), min(y_coords)
                                w, h = max(x_coords) - x, max(y_coords) - y
                                bbox = [int(x), int(y), int(w), int(h)]

                    if isinstance(item[1], tuple) and len(item[1]) >= 2:
                        text, confidence = item[1]
                        if text and len(text.strip()) > 0:
                            extracted_results.append((text.strip(), float(confidence), bbox, polygon, angle))
                    elif isinstance(item[1], str):
                        text = item[1]
                        confidence = item[2] if len(item) > 2 else 0.5
                        if text and len(text.strip()) > 0:
                            extracted_results.append((text.strip(), float(confidence), bbox, polygon, angle))

        return extracted_results

    def _detect_plate_structure(self, ocr_results: List[
        Tuple[str, float, Optional[List[int]], Optional[List[List[int]]], Optional[float]]]) -> PlateStructure:
        """
        Analyze geometric structure and text to detect plate components

        Args:
            ocr_results: List of (text, confidence, bbox, polygon, angle) tuples

        Returns:
            PlateStructure with parsed components
        """
        plate = PlateStructure()

        if not ocr_results:
            return plate

        num_elements = len(ocr_results)

        # BLOCK 1: Find main plate number (4+ chars with letters AND digits)
        main_plate_idx = None
        main_plate_text = None

        for idx, (text, conf, _, _, angle) in enumerate(ocr_results):
            # Check if element is 4+ chars with both letters and digits
            if len(text) >= 4:
                has_letters = bool(re.search(r'[A-Z]', text.upper()))
                has_digits = bool(re.search(r'\d', text))

                if has_letters and has_digits:
                    if main_plate_idx is None:
                        # Found first valid plate number
                        main_plate_idx = idx
                        main_plate_text = text
                    else:
                        # Found second 4+ element - FAIL case
                        self.logger.debug(
                            f"Multiple 4+ char elements found: '{main_plate_text}' and '{text}' - marking as failed")
                        plate.extraction_success = False
                        plate.layout_type = 'ambiguous_multiple_plates'
                        return plate

        # BLOCK 2: If main plate found - analyze remaining elements
        if main_plate_idx is not None:
            plate.plate_number = main_plate_text
            plate.extraction_success = True

            # Get remaining elements
            remaining = []
            for idx, elem in enumerate(ocr_results):
                if idx != main_plate_idx:
                    remaining.append(elem)

            # Analyze remaining elements for region/country
            if len(remaining) == 1:
                text, conf, bbox, poly, angle = remaining[0]
                fixed_text = fix_kg_text(text, is_region=True)

                # Check if it's a 2-digit region code
                if re.match(r'^\d{2}$', fixed_text):
                    plate.region_code = fixed_text
                    plate.country_code = "KG?"
                    plate.layout_type = 'linear'
                elif text.upper() in ['KG', 'KS', 'KZ']:
                    plate.country_code = 'KG' if text.upper() == 'KS' else text.upper()
                    plate.layout_type = 'linear'

            elif len(remaining) == 2:
                # Check if they are stacked (region above country)
                elem1, elem2 = remaining[0], remaining[1]
                poly1, poly2 = elem1[3], elem2[3]

                if is_above(poly1, poly2):
                    # First above second
                    region_text = fix_kg_text(elem1[0], is_region=True)
                    country_text = elem2[0].upper()

                    if re.match(r'^\d{2}$', region_text):
                        plate.region_code = region_text
                        if country_text in ['KG', 'KS']:
                            plate.country_code = 'KG'
                        plate.layout_type = 'stacked'

                elif is_above(poly2, poly1):
                    # Second above first
                    region_text = fix_kg_text(elem2[0], is_region=True)
                    country_text = elem1[0].upper()

                    if re.match(r'^\d{2}$', region_text):
                        plate.region_code = region_text
                        if country_text in ['KG', 'KS']:
                            plate.country_code = 'KG'
                        plate.layout_type = 'stacked'

            # Build full text
            parts = []
            if plate.region_code:
                parts.append(plate.region_code)
            if plate.country_code and plate.country_code != "KG?":
                parts.append(plate.country_code)
            if plate.plate_number:
                parts.append(plate.plate_number)
            plate.full_text = ' '.join(parts)

        # BLOCK 3: If no main plate found - check for stacked/linear layout (3-4 elements)
        else:
            if num_elements == 3:
                found_plate = False

                # First try to find linear sequence (3 digits + 2-3 letters)
                for i in range(num_elements - 1):
                    for j in range(i + 1, num_elements):
                        elem_left = ocr_results[i]
                        elem_right = ocr_results[j]

                        # Get boundaries
                        left_poly = elem_left[3]
                        right_poly = elem_right[3]

                        left_x_coords = [p[0] for p in left_poly]
                        right_x_coords = [p[0] for p in right_poly]
                        left_y_coords = [p[1] for p in left_poly]
                        right_y_coords = [p[1] for p in right_poly]

                        left_left = min(left_x_coords)
                        left_right = max(left_x_coords)
                        right_left = min(right_x_coords)
                        right_right = max(right_x_coords)

                        left_top = min(left_y_coords)
                        left_bottom = max(left_y_coords)
                        right_top = min(right_y_coords)
                        right_bottom = max(right_y_coords)

                        left_width = left_right - left_left
                        right_width = right_right - right_left
                        left_height = left_bottom - left_top
                        right_height = right_bottom - right_top

                        # Check if elements are horizontally sequential
                        min_height = min(left_height, right_height)
                        top_diff = abs(left_top - right_top)
                        bottom_diff = abs(left_bottom - right_bottom)
                        is_vertically_aligned = (top_diff < min_height * 0.25) and (bottom_diff < min_height * 0.25)

                        is_sequential = False
                        if left_left < right_left:  # Ensure left is actually on the left
                            overlap = min(left_right, right_right) - max(left_left, right_left)
                            if overlap > 0:
                                min_width = min(left_width, right_width)
                                overlap_percent = overlap / min_width
                                is_sequential = overlap_percent < 0.3
                            else:
                                is_sequential = True  # No overlap at all

                        # Check if it forms a valid plate number (3 digits + 2-3 letters)
                        if is_vertically_aligned and is_sequential:
                            left_text = elem_left[0]
                            right_text = elem_right[0]

                            # Check patterns: ###AAA or AAA###
                            if re.match(r'^\d{3}$', left_text) and re.match(r'^[A-Z]{2,3}$', right_text.upper()):
                                plate.plate_number = left_text + right_text.upper()
                                plate.extraction_success = True
                                plate.layout_type = 'linear_3_elements'
                                found_plate = True

                                # Remaining element should be region
                                for k in range(num_elements):
                                    if k != i and k != j:
                                        region_text = fix_kg_text(ocr_results[k][0], is_region=True)
                                        if re.match(r'^\d{2}$', region_text):
                                            plate.region_code = region_text
                                            plate.country_code = "KG?"
                                break
                            elif re.match(r'^[A-Z]{3}$', left_text.upper()) and re.match(r'^\d{3}$', right_text):
                                plate.plate_number = left_text.upper() + right_text
                                plate.extraction_success = True
                                plate.layout_type = 'linear_3_elements'
                                found_plate = True

                                # Remaining element should be region
                                for k in range(num_elements):
                                    if k != i and k != j:
                                        region_text = fix_kg_text(ocr_results[k][0], is_region=True)
                                        if re.match(r'^\d{2}$', region_text):
                                            plate.region_code = region_text
                                            plate.country_code = "KG?"
                                break

                    if found_plate:
                        break

                # If no linear sequence found, look for stacked pair (3 digits above 2-3 letters)
                if not found_plate:
                    for i in range(num_elements):
                        for j in range(num_elements):
                            if i != j and is_above(ocr_results[i][3], ocr_results[j][3]):
                                upper_text = ocr_results[i][0]
                                lower_text = ocr_results[j][0]

                                # Check if upper is 3 digits and lower is 2-3 letters
                                if re.match(r'^\d{3}$', upper_text) and re.match(r'^[A-Z]{2,3}$', lower_text.upper()):
                                    plate.plate_number = upper_text + lower_text.upper()
                                    plate.extraction_success = True
                                    found_plate = True

                                    # Remaining element should be region
                                    for k in range(num_elements):
                                        if k != i and k != j:
                                            region_text = fix_kg_text(ocr_results[k][0], is_region=True)
                                            if re.match(r'^\d{2}$', region_text):
                                                plate.region_code = region_text
                                                plate.country_code = "KG?"

                                    plate.layout_type = 'stacked_with_region'
                                    break
                        if found_plate:
                            break

                if not found_plate:
                    plate.extraction_success = False
                    plate.layout_type = 'unrecognized_3_elements'

            elif num_elements == 4:
                # First check if plate number is split into 2 sequential elements
                elements_by_x = sorted(range(num_elements), key=lambda i: min([p[0] for p in ocr_results[i][3]]))

                # Check all pairs for horizontal sequence
                merged_plate_found = False
                for i in range(num_elements - 1):
                    for j in range(i + 1, num_elements):
                        elem_left = ocr_results[i]
                        elem_right = ocr_results[j]

                        # Get boundaries
                        left_poly = elem_left[3]
                        right_poly = elem_right[3]

                        left_x_coords = [p[0] for p in left_poly]
                        right_x_coords = [p[0] for p in right_poly]
                        left_y_coords = [p[1] for p in left_poly]
                        right_y_coords = [p[1] for p in right_poly]

                        left_left = min(left_x_coords)
                        left_right = max(left_x_coords)
                        right_left = min(right_x_coords)
                        right_right = max(right_x_coords)

                        left_top = min(left_y_coords)
                        left_bottom = max(left_y_coords)
                        right_top = min(right_y_coords)
                        right_bottom = max(right_y_coords)

                        left_width = left_right - left_left
                        right_width = right_right - right_left
                        left_height = left_bottom - left_top
                        right_height = right_bottom - right_top

                        # Check if elements are horizontally sequential
                        # Condition 1: vertical alignment
                        min_height = min(left_height, right_height)
                        top_diff = abs(left_top - right_top)
                        bottom_diff = abs(left_bottom - right_bottom)
                        is_vertically_aligned = (top_diff < min_height * 0.25) and (bottom_diff < min_height * 0.25)

                        # Condition 2: horizontal sequence (left element is to the left of right element)
                        is_sequential = False
                        if left_left < right_left:  # Ensure left is actually on the left
                            overlap = min(left_right, right_right) - max(left_left, right_left)
                            if overlap > 0:
                                min_width = min(left_width, right_width)
                                overlap_percent = overlap / min_width
                                is_sequential = overlap_percent < 0.3
                            else:
                                is_sequential = True  # No overlap at all

                        # If conditions met, try to merge
                        if is_vertically_aligned and is_sequential:
                            merged_text = elem_left[0] + elem_right[0]

                            # Check if merged text is a valid plate number
                            if len(merged_text) >= 4:
                                has_letters = bool(re.search(r'[A-Z]', merged_text.upper()))
                                has_digits = bool(re.search(r'\d', merged_text))

                                if has_letters and has_digits:
                                    # Found valid merged plate number
                                    plate.plate_number = merged_text.upper()
                                    plate.extraction_success = True
                                    plate.layout_type = 'linear_4_elements'
                                    merged_plate_found = True

                                    # Analyze remaining elements for region/country
                                    remaining_indices = [k for k in range(num_elements) if k != i and k != j]

                                    if len(remaining_indices) == 2:
                                        elem1 = ocr_results[remaining_indices[0]]
                                        elem2 = ocr_results[remaining_indices[1]]
                                        poly1, poly2 = elem1[3], elem2[3]

                                        # Check if elements are stacked
                                        if is_above(poly1, poly2):
                                            # elem1 is above elem2
                                            upper_elem = elem1
                                            lower_elem = elem2
                                            are_stacked = True
                                        elif is_above(poly2, poly1):
                                            # elem2 is above elem1
                                            upper_elem = elem2
                                            lower_elem = elem1
                                            are_stacked = True
                                        else:
                                            are_stacked = False

                                        if are_stacked:
                                            # Try both interpretation variants
                                            # Standard variant: upper=region, lower=country (typical KG layout)
                                            variant_a_country = lower_elem[0].upper()
                                            # Check if KG or KS is contained in the text (not exact match)
                                            if 'KG' in variant_a_country or 'KS' in variant_a_country:
                                                # Only apply fix_kg_text if we found valid country
                                                variant_a_region = fix_kg_text(upper_elem[0], is_region=True)
                                                variant_a_valid = re.match(r'^\d{2}$', variant_a_region) is not None
                                            else:
                                                variant_a_valid = False

                                            # Inverted variant: upper=country, lower=region (OCR might have mixed up boxes)
                                            variant_b_country = upper_elem[0].upper()
                                            # Check if KG or KS is contained in the text (not exact match)
                                            if 'KG' in variant_b_country or 'KS' in variant_b_country:
                                                # Only apply fix_kg_text if we found valid country
                                                variant_b_region = fix_kg_text(lower_elem[0], is_region=True)
                                                variant_b_valid = re.match(r'^\d{2}$', variant_b_region) is not None
                                            else:
                                                variant_b_valid = False

                                            # Accept the valid variant (prefer standard if both valid)
                                            if variant_a_valid:
                                                plate.region_code = variant_a_region
                                                plate.country_code = 'KG'
                                            elif variant_b_valid:
                                                plate.region_code = variant_b_region
                                                plate.country_code = 'KG'
                                            # If neither valid, leave as None

                                    break

                    if merged_plate_found:
                        break

                # If no merged plate found, check for double stacked layout
                if not merged_plate_found:
                    # Look for 2 stacked pairs, right pair should be 3 digits above 2-3 letters
                    pairs = []

                    # Find all stacked pairs
                    for i in range(num_elements):
                        for j in range(num_elements):
                            if i < j and is_above(ocr_results[i][3], ocr_results[j][3]):
                                pairs.append((i, j))

                    if len(pairs) == 2:
                        # Check which pair is on the right (larger X coordinate)
                        pair1_x = min(
                            [p[0] for p in ocr_results[pairs[0][0]][3]] + [p[0] for p in ocr_results[pairs[0][1]][3]])
                        pair2_x = min(
                            [p[0] for p in ocr_results[pairs[1][0]][3]] + [p[0] for p in ocr_results[pairs[1][1]][3]])

                        right_pair = pairs[1] if pair2_x > pair1_x else pairs[0]
                        left_pair = pairs[0] if right_pair == pairs[1] else pairs[1]

                        # Check right pair for 3 digits above 2-3 letters
                        upper_text = ocr_results[right_pair[0]][0]
                        lower_text = ocr_results[right_pair[1]][0]

                        if re.match(r'^\d{3}$', upper_text) and re.match(r'^[A-Z]{2,3}$', lower_text.upper()):
                            plate.plate_number = upper_text + lower_text.upper()
                            plate.extraction_success = True

                            # Left pair should be region above country
                            region_text = fix_kg_text(ocr_results[left_pair[0]][0], is_region=True)
                            country_text = ocr_results[left_pair[1]][0].upper()

                            if re.match(r'^\d{2}$', region_text):
                                plate.region_code = region_text
                            if country_text in ['KG', 'KS']:
                                plate.country_code = 'KG'

                            plate.layout_type = 'double_stacked'
                        else:
                            plate.extraction_success = False
                            plate.layout_type = 'invalid_stacked_pattern'
                    else:
                        plate.extraction_success = False
                        plate.layout_type = 'unrecognized_4_elements'

            else:
                # 1-2 elements or 5+ elements
                if num_elements == 1:
                    # Single element might be the whole plate
                    text = ocr_results[0][0]
                    if len(text) >= 4:
                        plate.plate_number = text
                        plate.full_text = text
                        plate.layout_type = 'single'
                        plate.extraction_success = True
                else:
                    plate.extraction_success = False
                    plate.layout_type = f'unhandled_{num_elements}_elements'

            # Build full text for all layouts
            if plate.extraction_success:
                parts = []
                if plate.region_code:
                    parts.append(plate.region_code)
                if plate.country_code and plate.country_code != "KG?":
                    parts.append(plate.country_code)
                if plate.plate_number:
                    parts.append(plate.plate_number)
                plate.full_text = ' '.join(parts)

        # Final check: mark as failed only if critical data is missing
        if plate.extraction_success:
            # Must have at least a plate number to be considered successful
            if not plate.plate_number:
                plate.extraction_success = False
            # If we have absolutely no location info (neither region nor country), mark as failed
            elif not plate.region_code and not plate.country_code:
                plate.extraction_success = False
            # "KG?" is a valid value - it means "probably Kyrgyzstan"

        # Calculate confidence
        if ocr_results:
            total_conf = sum(elem[1] for elem in ocr_results)
            plate.confidence = total_conf / num_elements

        self.logger.debug(f"Plate structure detected: success={plate.extraction_success}, "
                          f"layout={plate.layout_type}, full_text='{plate.full_text}'")

        return plate


# Helper functions (outside class)
def fix_kg_text(text: str, is_region: bool = False) -> str:
    """Fix common OCR errors in Kyrgyz plates"""
    if is_region:
        # In region code G/D/T → 0/0/1
        text = text.replace('G', '0').replace('D', '0').replace('TO', '01').replace('O', '0').replace('61', '01')
        text = text.replace('To', '01').replace('T0', '01').replace('10', '01').replace('1O', '01').replace('1o', '01')
    else:
        # In plate number first 3 symbols might have G instead of 0
        if len(text) >= 3:
            first_three = text[:3]
            first_three = first_three.replace('G', '0').replace('O', '0')
            text = first_three + text[3:]
    return text


def is_above(upper_poly, lower_poly) -> bool:
    """Check if upper element is above lower element"""
    if not upper_poly or not lower_poly:
        return False

    # Get Y coordinates
    upper_y_bottom = max(p[1] for p in upper_poly)
    upper_y_top = min(p[1] for p in upper_poly)
    lower_y_top = min(p[1] for p in lower_poly)
    lower_y_bottom = max(p[1] for p in lower_poly)

    # Calculate heights
    upper_height = upper_y_bottom - upper_y_top
    lower_height = lower_y_bottom - lower_y_top
    min_height = min(upper_height, lower_height)

    # IRON RULE: no overlap at all
    if upper_y_bottom < lower_y_top:
        return True

    # SOFT RULES for overlapping elements
    if upper_y_top >= lower_y_top:
        return False

    vertical_overlap = upper_y_bottom - lower_y_top
    if vertical_overlap > min_height * 0.4:
        return False

    # Check X overlap (50% of smaller width)
    upper_x_coords = [p[0] for p in upper_poly]
    lower_x_coords = [p[0] for p in lower_poly]

    upper_x_min, upper_x_max = min(upper_x_coords), max(upper_x_coords)
    lower_x_min, lower_x_max = min(lower_x_coords), max(lower_x_coords)

    upper_width = upper_x_max - upper_x_min
    lower_width = lower_x_max - lower_x_min
    min_width = min(upper_width, lower_width)

    overlap_start = max(upper_x_min, lower_x_min)
    overlap_end = min(upper_x_max, lower_x_max)
    overlap_width = max(0, overlap_end - overlap_start)

    if overlap_width < min_width * 0.5:
        return False

    return True