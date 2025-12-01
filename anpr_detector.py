#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANPR Детектор - РЕАЛЬНАЯ боевая версия для распознавания номерных знаков
Использует реальные ML модели: PaddleOCR, YOLO, OpenCV
"""

import logging
import time
import re
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image
import paddleocr
from ultralytics import YOLO

logger = logging.getLogger(__name__)

class ANPRDetector:
    """
    РЕАЛЬНЫЙ класс для детекции номерных знаков и анализа дорожных изображений
    Использует PaddleOCR для OCR и YOLO для детекции объектов
    """
    
    def __init__(self, config):
        """
        Инициализация детектора
        
        Args:
            config: Объект конфигурации
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info("🚀 Инициализация РЕАЛЬНОГО ANPR детектора...")
        
        # Паттерны для киргизских номеров
        self.kg_patterns = [
            # Новый формат: 01KG001AAA
            r'^(\d{2})(KG)(\d{3})([A-Z]{3})$',
            # Старый формат с кириллицей: В 5431 АА
            r'^([АВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЫЭЮЯ])\s*(\d{4})\s*([АВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЫЭЮЯ]{2})$',
            # Латинский формат: B 9999 CD
            r'^([A-Z])\s*(\d{4})\s*([A-Z]{2})$',
            # Современный формат: T 1212 A
            r'^([A-Z])\s*(\d{4})\s*([A-Z])$'
        ]
        
        # Загрузка моделей
        self._load_models()
        
    def _load_models(self):
        """
        Загрузка реальных моделей машинного обучения
        """
        self.logger.info("📥 Загрузка моделей...")
        
        try:
            # Загружаем PaddleOCR для распознавания текста
            self.logger.info("  🔤 Загрузка PaddleOCR...")
            self.ocr = paddleocr.PaddleOCR(
                use_angle_cls=True, 
                lang='en',  # Основной язык
                use_gpu=True if self._check_gpu_available() else False,
                show_log=False,
                det_algorithm='DB',
                rec_algorithm='SVTR_LCNet'
            )
            
            # Загружаем YOLO для детекции объектов
            self.logger.info("  🎯 Загрузка YOLO модели...")
            yolo_model_path = self.config.yolo_model_path if hasattr(self.config, 'yolo_model_path') else 'yolov8n.pt'
            self.yolo_model = YOLO(yolo_model_path)
            
            # Дополнительная YOLO для номерных знаков (если есть специализированная)
            license_plate_model_path = getattr(self.config, 'license_plate_model_path', None)
            if license_plate_model_path and os.path.exists(license_plate_model_path):
                self.logger.info("  🎯 Загрузка специализированной модели для номеров...")
                self.license_plate_model = YOLO(license_plate_model_path)
            else:
                self.license_plate_model = self.yolo_model
            
            self.models_loaded = True
            self.logger.info("✅ Все модели успешно загружены!")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка загрузки моделей: {e}")
            self.models_loaded = False
            raise
            
    def _check_gpu_available(self) -> bool:
        """Проверяет доступность GPU"""
        try:
            import paddle
            return paddle.device.cuda.device_count() > 0
        except:
            return False
    
    def _preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Предобработка изображения для лучшего распознавания
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            np.ndarray: Обработанное изображение
        """
        # Загружаем изображение
        img = cv2.imread(str(image_path))
        
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        # Увеличиваем контрастность
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        img = cv2.merge([l, a, b])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        
        # Убираем шум
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        
        return img
    
    def _extract_text_regions(self, image: np.ndarray) -> List[Dict]:
        """
        Извлекает текстовые регионы с помощью OCR
        
        Args:
            image: Изображение для анализа
            
        Returns:
            List[Dict]: Список найденных текстовых регионов
        """
        try:
            # Запускаем OCR
            results = self.ocr.ocr(image, cls=True)
            
            text_regions = []
            if results and results[0]:
                for line in results[0]:
                    if line:
                        bbox, (text, confidence) = line
                        
                        # Фильтруем по уверенности
                        if confidence >= self.config.min_confidence:
                            text_regions.append({
                                'text': text,
                                'confidence': confidence,
                                'bbox': bbox
                            })
            
            return text_regions
            
        except Exception as e:
            self.logger.error(f"Ошибка OCR: {e}")
            return []
    
    def _detect_vehicles(self, image: np.ndarray) -> List[Dict]:
        """
        Детекция транспортных средств с помощью YOLO
        
        Args:
            image: Изображение для анализа
            
        Returns:
            List[Dict]: Список найденных транспортных средств
        """
        try:
            results = self.yolo_model(image, verbose=False)
            
            vehicles = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Классы транспорта в COCO: car (2), truck (7), bus (5), motorcycle (3)
                        if class_id in [2, 3, 5, 7] and confidence >= self.config.min_confidence:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            vehicles.append({
                                'class_id': class_id,
                                'class_name': self._get_vehicle_class_name(class_id),
                                'confidence': confidence,
                                'bbox': [x1, y1, x2-x1, y2-y1]  # x, y, width, height
                            })
            
            return vehicles
            
        except Exception as e:
            self.logger.error(f"Ошибка детекции транспорта: {e}")
            return []
    
    def _get_vehicle_class_name(self, class_id: int) -> str:
        """Возвращает название класса транспорта"""
        vehicle_classes = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        return vehicle_classes.get(class_id, 'unknown')
    
    def _validate_license_plate(self, text: str) -> Optional[Dict]:
        """
        Проверяет, является ли текст киргизским номерным знаком
        
        Args:
            text: Текст для проверки
            
        Returns:
            Optional[Dict]: Информация о номере, если валиден
        """
        # Очистка текста
        cleaned_text = text.strip().upper()
        
        for pattern in self.kg_patterns:
            match = re.match(pattern, cleaned_text)
            if match:
                return {
                    'original_text': text,
                    'cleaned_text': cleaned_text,
                    'pattern': pattern,
                    'country': 'Kyrgyzstan',
                    'region': self._get_region_from_plate(cleaned_text),
                    'type': self._get_plate_type(cleaned_text),
                    'format': self._get_plate_format(cleaned_text),
                    'is_valid': True
                }
        
        return None
    
    def detect_license_plates(self, image_path: str) -> List[Dict]:
        """
        РЕАЛЬНАЯ детекция номерных знаков на изображении
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            List[Dict]: Список найденных номерных знаков
        """
        if not self.config.enable_plate_detection:
            return []
            
        self.logger.debug(f"🔍 Детекция номерных знаков: {image_path}")
        
        try:
            # Предобработка изображения
            image = self._preprocess_image(image_path)
            
            # Детекция транспортных средств
            vehicles = self._detect_vehicles(image)
            
            # Извлечение текстовых регионов
            text_regions = self._extract_text_regions(image)
            
            license_plates = []
            
            # Анализируем каждый текстовый регион
            for text_region in text_regions:
                plate_info = self._validate_license_plate(text_region['text'])
                
                if plate_info:
                    # Конвертируем bbox в нужный формат
                    bbox_points = text_region['bbox']
                    x_coords = [point[0] for point in bbox_points]
                    y_coords = [point[1] for point in bbox_points]
                    x, y = min(x_coords), min(y_coords)
                    w, h = max(x_coords) - x, max(y_coords) - y
                    
                    license_plates.append({
                        'text': plate_info['cleaned_text'],
                        'original_text': plate_info['original_text'],
                        'confidence': text_region['confidence'],
                        'bbox': [x, y, w, h],
                        'country': plate_info['country'],
                        'region': plate_info['region'],
                        'type': plate_info['type'],
                        'format': plate_info['format'],
                        'vehicle_nearby': len(vehicles) > 0
                    })
            
            self.logger.info(f"✅ Найдено номерных знаков: {len(license_plates)}")
            return license_plates
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка детекции номеров: {e}")
            return []
    
    def detect_violations(self, image_path: str) -> List[Dict]:
        """
        РЕАЛЬНАЯ детекция нарушений ПДД
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            List[Dict]: Список найденных нарушений
        """
        if not self.config.enable_violation_detection:
            return []
            
        self.logger.debug(f"🚔 Детекция нарушений: {image_path}")
        
        try:
            image = cv2.imread(str(image_path))
            violations = []
            
            # Анализ с помощью YOLO
            results = self.yolo_model(image, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Анализируем потенциальные нарушения
                        violation = self._analyze_for_violations(class_id, confidence, box.xyxy[0])
                        if violation and confidence >= self.config.min_confidence:
                            violations.append(violation)
            
            # Дополнительный анализ по цветам светофора
            traffic_light_violations = self._detect_traffic_light_violations(image)
            violations.extend(traffic_light_violations)
            
            self.logger.info(f"🚨 Найдено нарушений: {len(violations)}")
            return violations
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка детекции нарушений: {e}")
            return []
    
    def _analyze_for_violations(self, class_id: int, confidence: float, bbox) -> Optional[Dict]:
        """Анализ объекта на предмет нарушений"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Примеры анализа нарушений (можно расширить)
        if class_id == 3:  # motorcycle
            # Проверка наличия шлема (требует дополнительной модели)
            return {
                'type': 'potential_no_helmet',
                'confidence': confidence * 0.7,  # Снижаем уверенность для потенциального нарушения
                'bbox': [x1, y1, x2-x1, y2-y1],
                'description': 'Потенциальное нарушение: мотоциклист без шлема'
            }
        
        return None
    
    def _detect_traffic_light_violations(self, image: np.ndarray) -> List[Dict]:
        """Детекция нарушений сигналов светофора"""
        violations = []
        
        # Простая детекция красного цвета (базовый алгоритм)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Диапазон для красного цвета
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        # Поиск контуров красных объектов
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Минимальная площадь для светофора
                x, y, w, h = cv2.boundingRect(contour)
                
                # Простая эвристика: если красный объект в верхней части изображения
                if y < image.shape[0] * 0.3 and w > 10 and h > 10:
                    violations.append({
                        'type': 'potential_red_light',
                        'confidence': 0.6,
                        'bbox': [x, y, w, h],
                        'description': 'Потенциальное нарушение: проезд на красный сигнал'
                    })
        
        return violations
    
    def detect_streets(self, image_path: str) -> List[Dict]:
        """
        РЕАЛЬНАЯ детекция и распознавание названий улиц
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            List[Dict]: Список найденных улиц
        """
        if not self.config.enable_street_detection:
            return []
            
        self.logger.debug(f"🛣️ Детекция улиц: {image_path}")
        
        try:
            image = self._preprocess_image(image_path)
            
            # Ищем текстовые регионы
            text_regions = self._extract_text_regions(image)
            
            streets = []
            street_keywords = [
                'улица', 'проспект', 'переулок', 'бульвар', 'шоссе',
                'ул.', 'пр.', 'пер.', 'бул.', 'ш.',
                'street', 'avenue', 'boulevard', 'highway'
            ]
            
            for text_region in text_regions:
                text = text_region['text'].lower()
                
                # Проверяем наличие ключевых слов улиц
                for keyword in street_keywords:
                    if keyword in text:
                        bbox_points = text_region['bbox']
                        x_coords = [point[0] for point in bbox_points]
                        y_coords = [point[1] for point in bbox_points]
                        x, y = min(x_coords), min(y_coords)
                        w, h = max(x_coords) - x, max(y_coords) - y
                        
                        streets.append({
                            'name': text_region['text'],
                            'confidence': text_region['confidence'],
                            'bbox': [x, y, w, h],
                            'type': 'street_sign',
                            'keyword_found': keyword
                        })
                        break
            
            self.logger.info(f"🗺️ Найдено улиц: {len(streets)}")
            return streets
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка детекции улиц: {e}")
            return []
    
    def detect_car_brands(self, image_path: str) -> List[Dict]:
        """
        РЕАЛЬНАЯ детекция марок автомобилей
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            List[Dict]: Список найденных марок автомобилей
        """
        if not self.config.enable_car_brand_detection:
            return []
            
        self.logger.debug(f"🚗 Детекция марок авто: {image_path}")
        
        try:
            image = cv2.imread(str(image_path))
            
            # Детекция транспортных средств
            vehicles = self._detect_vehicles(image)
            
            cars = []
            
            for vehicle in vehicles:
                if vehicle['class_name'] == 'car':
                    # Извлекаем область с автомобилем
                    x, y, w, h = vehicle['bbox']
                    car_region = image[y:y+h, x:x+w]
                    
                    # Анализ логотипов и эмблем (базовая реализация)
                    brand_info = self._analyze_car_brand(car_region)
                    
                    if brand_info:
                        cars.append({
                            'brand': brand_info['brand'],
                            'confidence': vehicle['confidence'] * brand_info['confidence'],
                            'bbox': vehicle['bbox'],
                            'color': self._detect_car_color(car_region),
                            'type': self._classify_car_type(car_region),
                            'method': brand_info['method']
                        })
                    else:
                        # Если не удалось определить марку, добавляем базовую информацию
                        cars.append({
                            'brand': 'unknown',
                            'confidence': vehicle['confidence'],
                            'bbox': vehicle['bbox'],
                            'color': self._detect_car_color(car_region),
                            'type': self._classify_car_type(car_region),
                            'method': 'vehicle_detection_only'
                        })
            
            self.logger.info(f"🏷️ Найдено автомобилей: {len(cars)}")
            return cars
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка детекции марок: {e}")
            return []
    
    def _analyze_car_brand(self, car_region: np.ndarray) -> Optional[Dict]:
        """
        Анализ марки автомобиля по изображению
        (Базовая реализация, можно улучшить специализированной моделью)
        """
        try:
            # Поиск текста на автомобиле (номера, надписи)
            text_regions = self._extract_text_regions(car_region)
            
            # Известные марки автомобилей
            known_brands = [
                'TOYOTA', 'HONDA', 'HYUNDAI', 'KIA', 'NISSAN',
                'MAZDA', 'VOLKSWAGEN', 'BMW', 'Mercedes', 'AUDI',
                'LADA', 'CHEVROLET', 'FORD', 'OPEL', 'RENAULT'
            ]
            
            for text_region in text_regions:
                text = text_region['text'].upper()
                for brand in known_brands:
                    if brand in text:
                        return {
                            'brand': brand,
                            'confidence': text_region['confidence'],
                            'method': 'text_recognition'
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа марки: {e}")
            return None
    
    def _detect_car_color(self, car_region: np.ndarray) -> str:
        """Определение цвета автомобиля"""
        try:
            # Конвертируем в HSV для лучшего анализа цвета
            hsv = cv2.cvtColor(car_region, cv2.COLOR_BGR2HSV)
            
            # Анализируем доминирующий цвет
            h, w, _ = hsv.shape
            center_region = hsv[h//4:3*h//4, w//4:3*w//4]  # Центральная область
            
            # Средние значения HSV
            mean_h = np.mean(center_region[:, :, 0])
            mean_s = np.mean(center_region[:, :, 1])
            mean_v = np.mean(center_region[:, :, 2])
            
            # Простая классификация цветов
            if mean_v < 50:
                return 'черный'
            elif mean_v > 200 and mean_s < 50:
                return 'белый'
            elif mean_s < 50:
                return 'серый'
            elif 0 <= mean_h <= 10 or 170 <= mean_h <= 180:
                return 'красный'
            elif 35 <= mean_h <= 85:
                return 'зеленый'
            elif 100 <= mean_h <= 130:
                return 'синий'
            elif 15 <= mean_h <= 35:
                return 'желтый'
            else:
                return 'другой'
                
        except Exception:
            return 'неопределен'
    
    def _classify_car_type(self, car_region: np.ndarray) -> str:
        """Классификация типа автомобиля по размерам"""
        try:
            h, w, _ = car_region.shape
            aspect_ratio = w / h
            
            # Простая классификация по соотношению сторон
            if aspect_ratio > 2.5:
                return 'лимузин'
            elif aspect_ratio > 2.0:
                return 'седан'
            elif aspect_ratio > 1.8:
                return 'хетчбек'
            elif aspect_ratio > 1.5:
                return 'внедорожник'
            else:
                return 'компактный'
                
        except Exception:
            return 'неопределен'
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        РЕАЛЬНЫЙ полный анализ изображения со всеми типами детекции
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Dict[str, Any]: Результаты всех видов анализа
        """
        start_time = time.time()
        self.logger.info(f"🔍 Начинаем РЕАЛЬНЫЙ анализ изображения: {image_path}")
        
        # Проверяем существование файла
        if not Path(image_path).exists():
            error_msg = f"Файл не найден: {image_path}"
            self.logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            # Выполняем все виды детекции
            results = {
                'status': 'success',
                'image_path': str(image_path),
                'timestamp': datetime.now().isoformat(),
                'processing_time': 0,  # Будет заполнено в конце
                'model_info': {
                    'ocr_enabled': hasattr(self, 'ocr'),
                    'yolo_enabled': hasattr(self, 'yolo_model'),
                    'gpu_used': self._check_gpu_available()
                },
                'results': {
                    'license_plates': self.detect_license_plates(image_path),
                    'violations': self.detect_violations(image_path),
                    'streets': self.detect_streets(image_path),
                    'car_brands': self.detect_car_brands(image_path)
                },
                'summary': {}
            }
            
            # Добавляем сводку
            results['summary'] = {
                'plates_found': len(results['results']['license_plates']),
                'violations_found': len(results['results']['violations']),
                'streets_found': len(results['results']['streets']),
                'cars_found': len(results['results']['car_brands']),
                'has_detections': any([
                    results['results']['license_plates'],
                    results['results']['violations'],
                    results['results']['streets'],
                    results['results']['car_brands']
                ])
            }
            
            # Записываем время обработки
            processing_time = time.time() - start_time
            results['processing_time'] = round(processing_time, 3)
            
            self.logger.info(f"✅ РЕАЛЬНЫЙ анализ завершен за {processing_time:.3f}с. "
                           f"Найдено: {results['summary']['plates_found']} номеров, "
                           f"{results['summary']['violations_found']} нарушений, "
                           f"{results['summary']['streets_found']} улиц, "
                           f"{results['summary']['cars_found']} авто")
            
            return results
            
        except Exception as e:
            error_msg = f"Ошибка при РЕАЛЬНОМ анализе {image_path}: {str(e)}"
            self.logger.error(error_msg)
            return {
                'status': 'error',
                'error': error_msg,
                'image_path': str(image_path),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_region_from_plate(self, plate_text: str) -> str:
        """Определяет регион по номерному знаку"""
        if plate_text.startswith('В') or plate_text.startswith('B'):
            return 'Bishkek'
        elif plate_text.startswith('З') or plate_text.startswith('Z'):
            return 'Osh'
        elif plate_text.startswith('01'):
            return 'Bishkek'
        elif plate_text.startswith('02'):
            return 'Osh'
        elif plate_text.startswith('03'):
            return 'Jalal-Abad'
        elif plate_text.startswith('04'):
            return 'Issyk-Kul'
        elif plate_text.startswith('05'):
            return 'Naryn'
        elif plate_text.startswith('06'):
            return 'Batken'
        elif plate_text.startswith('07'):
            return 'Talas'
        elif plate_text.startswith('08'):
            return 'Chui'
        else:
            return 'Unknown'
    
    def _get_plate_type(self, plate_text: str) -> str:
        """Определяет тип номерного знака"""
        if len(plate_text) >= 8 and plate_text[2:4] == 'KG':
            return 'kg_new_format'
        elif any(char in 'АВГДЕЖЗИКЛМНОПРСТУФХЦЧШЩЫЭЮЯ' for char in plate_text):
            return 'kg_old_individual_cyrillic'
        else:
            return 'kg_individual_latin'
    
    def _get_plate_format(self, plate_text: str) -> str:
        """Определяет формат номерного знака"""
        if len(plate_text) >= 8 and 'KG' in plate_text:
            return 'NNKGNNNAAA'
        elif len(plate_text.replace(' ', '')) == 7:
            return 'LNNNNLL'
        elif len(plate_text.replace(' ', '')) == 6:
            return 'LNNNL'
        else:
            return 'unknown'