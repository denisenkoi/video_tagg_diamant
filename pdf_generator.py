#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF Generator for ANPR Violation Reports
Creates PDF reports in TargetEYE style with Qubit branding
FIXED: Using correct field names from yolo_plate_detections
"""

import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import black, red, blue, gray
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io
import logging


class ViolationReportGenerator:
    """
    Generates PDF reports for traffic violations
    """

    def __init__(self, logo_path: str = 'qubit_logo.png',
                 font_path: str = 'djv_fonts/dejavu-lgc-fonts-ttf-2.37/ttf/DejaVuLGCSans.ttf'):
        """
        Initialize PDF generator

        Args:
            logo_path: Path to Qubit logo file
            font_path: Path to DejaVu LGC Sans TTF font
        """
        self.logger = logging.getLogger(__name__)
        self.logo_path = Path(logo_path)
        self.font_path = Path(font_path)

        # Company branding
        self.company_name = "Qubit"
        self.system_name = "ANPR System"
        self.domain = "qubit.kg"

        # Register Russian font FIRST
        self._register_fonts()

        # Load logo if exists
        if self.logo_path.exists():
            self.logo_data = self._load_logo_as_base64()
        else:
            self.logger.warning(f"Logo not found at {logo_path}")
            self.logo_data = None

    def _register_fonts(self):
        """Register DejaVu LGC fonts for Russian text support"""
        # Check if font file exists
        if not self.font_path.exists():
            raise FileNotFoundError(f"DejaVu font not found at {self.font_path}. Please ensure font file is available.")

        # Register main font
        pdfmetrics.registerFont(TTFont('DejaVuLGC', str(self.font_path)))

        # Try to register Bold variant if available
        font_dir = self.font_path.parent
        bold_font_path = font_dir / 'DejaVuLGCSans-Bold.ttf'
        if bold_font_path.exists():
            pdfmetrics.registerFont(TTFont('DejaVuLGC-Bold', str(bold_font_path)))

        self.logger.info(f"Registered DejaVu LGC font from {self.font_path}")

    def _load_logo_as_base64(self) -> str:
        """Load logo and convert to base64 string"""
        with open(self.logo_path, 'rb') as f:
            logo_bytes = f.read()
        return base64.b64encode(logo_bytes).decode('utf-8')

    def _create_header(self, canvas, doc):
        """Create PDF header with logo and branding"""
        # Save canvas state
        canvas.saveState()

        # Header height
        header_height = 1.5 * cm
        y_start = A4[1] - header_height

        # Logo on left
        if self.logo_data:
            logo_bytes = base64.b64decode(self.logo_data)
            logo_io = io.BytesIO(logo_bytes)
            logo_reader = ImageReader(logo_io)
            canvas.drawImage(logo_reader, 2 * cm, y_start + 0.2 * cm, width=2 * cm, height=1 * cm)

        # Domain on right
        canvas.setFont("DejaVuLGC", 12)
        canvas.drawRightString(A4[0] - 2 * cm, y_start + 0.6 * cm, self.domain)

        # Horizontal line
        canvas.setStrokeColor(black)
        canvas.setLineWidth(1)
        canvas.line(2 * cm, y_start, A4[0] - 2 * cm, y_start)

        canvas.restoreState()

    def _create_footer(self, canvas, doc):
        """Create PDF footer"""
        canvas.saveState()
        canvas.setFont("DejaVuLGC", 9)
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
        canvas.drawString(2 * cm, 1.5 * cm, f"Документ создан {timestamp}")
        canvas.restoreState()

    def _extract_plate_image(self, full_image: np.ndarray, plate_data: Dict) -> Optional[np.ndarray]:
        """
        Extract plate region from full image using expanded_bbox if available

        Args:
            full_image: Original image
            plate_data: Plate detection data with bbox and expanded_bbox

        Returns:
            Cropped plate image or None if failed
        """
        print(f"DEBUG: _extract_plate_image called with plate_data keys: {list(plate_data.keys())}")

        # Use expanded_bbox if available for better OCR region
        if plate_data.get('expanded_bbox'):
            x, y, w, h = plate_data['expanded_bbox']
            print(f"DEBUG: Using expanded bbox: [{x}, {y}, {w}, {h}]")
        else:
            # Fallback to regular bbox
            bbox = plate_data.get('bbox')
            if not bbox or len(bbox) != 4:
                print(f"DEBUG: No valid bbox found, bbox: {bbox}")
                return None
            x, y, w, h = bbox
            print(f"DEBUG: Using regular bbox: [{x}, {y}, {w}, {h}]")

        height, width = full_image.shape[:2]
        print(f"DEBUG: Full image dimensions: {width}x{height}")

        # Validate coordinates
        if x < 0 or y < 0 or x + w > width or y + h > height:
            print(f"DEBUG: Plate coordinates out of bounds: [{x}, {y}, {w}, {h}], image: {width}x{height}")
            return None

        # Extract region
        plate_region = full_image[y:y + h, x:x + w]
        print(f"DEBUG: Extracted plate region shape: {plate_region.shape}")

        if plate_region.size == 0:
            print("DEBUG: Extracted plate region is empty")
            return None

        # Resize if too small for PDF display
        if plate_region.shape[0] < 50:
            scale = 50 / plate_region.shape[0]
            new_width = int(plate_region.shape[1] * scale)
            plate_region = cv2.resize(plate_region, (new_width, 50))
            print(f"DEBUG: Resized plate region to: {new_width}x50")

        print(f"DEBUG: Final plate region shape: {plate_region.shape}")
        return plate_region

    def _cv2_to_reportlab_image(self, cv2_image: np.ndarray) -> Image:
        """Convert OpenCV image to ReportLab Image"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

        # Create PIL Image
        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(rgb_image)

        # Convert to bytes
        img_io = io.BytesIO()
        pil_image.save(img_io, format='JPEG', quality=85)
        img_io.seek(0)

        # platypus.Image accepts BytesIO directly
        return Image(img_io)

    def create_violation_report(self, image_path: Path, full_image: np.ndarray,
                                metadata: Dict[str, Any]) -> Path:
        """
        Create PDF violation report

        Args:
            image_path: Path to original image
            full_image: Loaded image data
            metadata: ANPR processing results

        Returns:
            Path to generated PDF
        """
        print("DEBUG: create_violation_report() STARTED")
        print(f"DEBUG: metadata keys: {list(metadata.keys())}")
        if 'yolo_plate_detections' in metadata:
            print(f"DEBUG: yolo_plate_detections count: {len(metadata['yolo_plate_detections'])}")

        # Create PDF path in same metadata directory
        metadata_dir = image_path.parent / 'violations_metadata'
        pdf_path = metadata_dir / f"{image_path.stem}_report.pdf"
        print(f"DEBUG: PDF path: {pdf_path}")

        # Create PDF document
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=A4,
            rightMargin=2 * cm, leftMargin=2 * cm,
            topMargin=2.5 * cm, bottomMargin=2 * cm
        )

        # Build story (content)
        story = []
        styles = getSampleStyleSheet()

        # Apply DejaVu LGC font to all styles for Russian text support
        styles['Normal'].fontName = 'DejaVuLGC'
        styles['Heading1'].fontName = 'DejaVuLGC-Bold' if 'DejaVuLGC-Bold' in pdfmetrics._fonts else 'DejaVuLGC'
        styles['Heading2'].fontName = 'DejaVuLGC-Bold' if 'DejaVuLGC-Bold' in pdfmetrics._fonts else 'DejaVuLGC'

        # Custom styles with DejaVu font
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='DejaVuLGC-Bold' if 'DejaVuLGC-Bold' in pdfmetrics._fonts else 'DejaVuLGC'
        )

        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading2'],
            fontSize=12,
            spaceAfter=6,
            textColor=blue,
            fontName='DejaVuLGC-Bold' if 'DejaVuLGC-Bold' in pdfmetrics._fonts else 'DejaVuLGC'
        )

        # Determine plate number for title
        plate_number = "НЕИЗВЕСТЕН"
        recognition_successful = False

        print("DEBUG: Checking for plate recognition...")

        # Check YOLO plate detections - FIXED: use correct field names
        if metadata.get('yolo_plate_detections'):
            print(f"DEBUG: Found {len(metadata['yolo_plate_detections'])} yolo_plate_detections")
            for i, plate in enumerate(metadata['yolo_plate_detections']):
                print(f"DEBUG: Plate {i + 1} structure: {list(plate.keys())}")
                # FIXED: Check for recognized_text instead of best_recognized_text
                if plate.get('recognized_text'):
                    plate_number = plate['recognized_text']
                    recognition_successful = True
                    print(f"DEBUG: Found recognized_text: {plate_number}")
                    break

        # Fallback to license_plates if still not found
        if not recognition_successful and metadata.get('license_plates'):
            print(f"DEBUG: Fallback to license_plates, found {len(metadata['license_plates'])}")
            for plate in metadata['license_plates']:
                if plate.get('text'):
                    plate_number = plate['text']
                    recognition_successful = True
                    print(f"DEBUG: Found license plate text: {plate_number}")
                    break

        print(f"DEBUG: Final recognition_successful = {recognition_successful}")
        print(f"DEBUG: Final plate_number = {plate_number}")

        # Title - plate number or failure message
        if not recognition_successful:
            failure_style = ParagraphStyle(
                'FailureTitle',
                parent=title_style,
                textColor=red,
                fontSize=20,
                fontName='DejaVuLGC-Bold' if 'DejaVuLGC-Bold' in pdfmetrics._fonts else 'DejaVuLGC'
            )
            story.append(Paragraph("НЕ УДАЛОСЬ РАСПОЗНАТЬ НОМЕР", failure_style))

        story.append(Spacer(1, 10))

        # Generate unique ID
        violation_id = f"qbt-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hash(str(image_path)) % 10000:04d}"

        # Metadata table
        file_info = metadata.get('file_info', {})
        processing_info = metadata.get('processing_info', {})
        radar_data = metadata.get('radar_data', {})

        # Parse timestamp from filename or use file timestamp
        filename = image_path.stem
        if '-' in filename:
            timestamp_str = filename.split('-')[0]
            if len(timestamp_str) >= 14:
                year = timestamp_str[:4]
                month = timestamp_str[4:6]
                day = timestamp_str[6:8]
                hour = timestamp_str[8:10]
                minute = timestamp_str[10:12]
                second = timestamp_str[12:14]
                event_time = f"{day}/{month}/{year} {hour}:{minute}:{second}"
            else:
                event_time = processing_info.get('timestamp', 'Unknown')[:19]
        else:
            event_time = processing_info.get('timestamp', 'Unknown')[:19]

        # Calculate speed violation dynamically with fines
        allowed_speed_kmh = 10  # Default allowed speed
        current_speed_kmh = None
        violation_description = 'Нарушение не определено'
        violation_article = 'Статья не определена'
        fine_amount = 0

        if radar_data and radar_data.get('speed'):
            current_speed_kmh = radar_data['speed']
            speed_diff = current_speed_kmh - allowed_speed_kmh

            # Calculate violation article, description and fine based on speed difference
            if speed_diff > 60:
                violation_article = 'ст. 187 ч.4 КоАП КР'
                violation_description = f'Превышение скорости на {speed_diff} км/ч'
                fine_amount = 7500
            elif speed_diff > 40:
                violation_article = 'ст. 187 ч.3 КоАП КР'
                violation_description = f'Превышение скорости на {speed_diff} км/ч'
                fine_amount = 5500
            elif speed_diff > 20:
                violation_article = 'ст. 187 ч.2 КоАП КР'
                violation_description = f'Превышение скорости на {speed_diff} км/ч'
                fine_amount = 3000
            elif speed_diff > 10:
                violation_article = 'ст. 187 ч.1 КоАП КР'
                violation_description = f'Превышение скорости на {speed_diff} км/ч'
                fine_amount = 1000
            else:
                violation_article = 'Нарушения нет'
                violation_description = 'Скорость в пределах нормы'
                fine_amount = 0

        info_data = [
            ['id:', violation_id],
            ['Дата:', event_time],
            ['Номер:', plate_number],  # Added plate number to table
            ['Место:', 'ул. Кийизбаевой, 54, Бишкек'],
            ['Статья:', violation_article],
            ['Нарушение:', violation_description],
            ['Штраф:', f'{fine_amount} сом' if fine_amount > 0 else 'Без штрафа'],
            ['Камера:', 'CAM-BSK-001']
        ]

        info_table = Table(info_data, colWidths=[3 * cm, 8 * cm])
        info_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'DejaVuLGC-Bold' if 'DejaVuLGC-Bold' in pdfmetrics._fonts else 'DejaVuLGC'),
            ('FONTNAME', (1, 0), (1, -1), 'DejaVuLGC'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))

        story.append(info_table)
        story.append(Spacer(1, 8))

        # Show extracted PLATE IMAGE above main image if recognized
        if recognition_successful and metadata.get('yolo_plate_detections'):
            print(f"DEBUG: Processing {len(metadata['yolo_plate_detections'])} YOLO plate detections for PDF")

            # Find best plate with expanded bbox for extraction - FIXED: use correct field names
            best_plate_data = None
            for i, plate_data in enumerate(metadata['yolo_plate_detections']):
                print(f"DEBUG: Plate {i + 1} - has recognized_text: {plate_data.get('recognized_text') is not None}")
                print(f"DEBUG: Plate {i + 1} - extraction_success: {plate_data.get('extraction_success')}")
                print(f"DEBUG: Plate {i + 1} - bbox: {plate_data.get('bbox')}")
                print(f"DEBUG: Plate {i + 1} - expanded_bbox: {plate_data.get('expanded_bbox')}")

                # FIXED: Check extraction_success and recognized_text
                if plate_data.get('extraction_success') and plate_data.get('recognized_text'):
                    best_plate_data = plate_data
                    print(f"DEBUG: Selected plate {i + 1} as best plate")
                    break

            if best_plate_data:
                # Extract ACTUAL PLATE IMAGE from coordinates
                print("DEBUG: Attempting to extract plate image")
                plate_img = self._extract_plate_image(full_image, best_plate_data)

                if plate_img is not None:
                    print(f"DEBUG: Successfully extracted plate image with shape: {plate_img.shape}")

                    # Convert extracted plate image to ReportLab format
                    plate_reportlab_img = self._cv2_to_reportlab_image(plate_img)
                    # Make plate image larger and more visible
                    plate_reportlab_img.drawHeight = 3 * cm
                    plate_reportlab_img.drawWidth = 9 * cm

                    # Add text info about the plate with all components
                    confidence = best_plate_data.get('combined_confidence', 0) * 100
                    plate_text_style = ParagraphStyle(
                        'PlateText',
                        fontSize=14,
                        alignment=TA_CENTER,
                        fontName='DejaVuLGC-Bold' if 'DejaVuLGC-Bold' in pdfmetrics._fonts else 'DejaVuLGC',
                        spaceAfter=5
                    )

                    # Build detailed plate info
                    plate_parts = []
                    if best_plate_data.get('region_code'):
                        plate_parts.append(f"Регион: {best_plate_data['region_code']}")
                    if best_plate_data.get('country_code'):
                        plate_parts.append(f"Страна: {best_plate_data['country_code']}")
                    if best_plate_data.get('plate_number'):
                        plate_parts.append(f"Номер: {best_plate_data['plate_number']}")

                    # FIXED: Use recognized_text instead of best_recognized_text
                    full_plate_text = best_plate_data['recognized_text']
                    plate_info_text = f"Номерной знак: {full_plate_text}"
                    if plate_parts:
                        plate_info_text += f" ({', '.join(plate_parts)})"
                    plate_info_text += f" [Уверенность: {confidence:.1f}%]"

                    story.append(Paragraph(plate_info_text, plate_text_style))

                    # Show the ACTUAL EXTRACTED PLATE IMAGE
                    story.append(plate_reportlab_img)
                    story.append(Spacer(1, 10))
                    print("DEBUG: Added plate image to PDF story")
                else:
                    print("DEBUG: Failed to extract plate image - plate_img is None")
            else:
                print("DEBUG: No best plate data found for extraction")

        # Main vehicle image with proper aspect ratio
        main_image = self._cv2_to_reportlab_image(full_image)

        # Calculate proper aspect ratio for main image
        img_height, img_width = full_image.shape[:2]
        aspect_ratio = img_width / img_height

        # Set desired height and calculate width to preserve aspect ratio
        desired_height = 8 * cm
        calculated_width = desired_height * aspect_ratio

        main_image.drawHeight = desired_height
        main_image.drawWidth = calculated_width

        story.append(main_image)
        story.append(Spacer(1, 8))

        # Additional info header
        story.append(Paragraph("Дополнительная информация:", header_style))

        # Vehicle and detection info
        vehicles = metadata.get('vehicles', [])
        vehicle_type = 'неизвестен'
        if vehicles:
            vehicle_type = vehicles[0].get('type', 'неизвестен')

        additional_data = [
            ['Тип автотранспорта:', vehicle_type],
            ['Марка:', 'не определена'],
            ['Модель:', 'не определена'],
            ['Цвет:', 'не определен']
        ]

        # Add speed data
        speed_text = 'не определена'
        allowed_speed = f'{allowed_speed_kmh} км/ч'
        if radar_data and radar_data.get('speed'):
            speed_text = f"{radar_data['speed']} км/ч"

        additional_data.append(['Скорость:', speed_text])
        additional_data.append(['Разрешенная скорость:', allowed_speed])

        additional_table = Table(additional_data, colWidths=[4 * cm, 7 * cm])
        additional_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'DejaVuLGC-Bold' if 'DejaVuLGC-Bold' in pdfmetrics._fonts else 'DejaVuLGC'),
            ('FONTNAME', (1, 0), (1, -1), 'DejaVuLGC'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))

        story.append(additional_table)
        story.append(Spacer(1, 10))

        # Build PDF with custom header/footer
        doc.build(story, onFirstPage=self._create_header, onLaterPages=self._create_header)

        self.logger.info(f"Generated PDF report: {pdf_path}")
        return pdf_path

    def should_generate_pdf(self, metadata: Dict[str, Any]) -> bool:
        """
        Check if PDF should be generated for this metadata

        Args:
            metadata: Processing results

        Returns:
            True if PDF should be generated
        """
        # Generate PDF for all processed images (successful or not)
        return metadata.get('status') == 'success'