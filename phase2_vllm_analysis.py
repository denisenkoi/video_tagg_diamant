"""
Phase 2: VLLM Analysis
Независимый модуль для анализа подготовленных данных через VLLM
Минимальные зависимости: requests, json, pickle, tqdm
"""

import json
import pickle
import requests
import base64
import time
import os
import platform
from pathlib import Path
from typing import List, Dict, Any, Optional
import shutil
from datetime import datetime
from db_manager import ChromaDBManager

# КОНФИГУРАЦИЯ
FULL_REPROCESS = False  # True = полная перезапись, False = только недостающие

# Try to import tqdm, fallback if not available
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", leave=True):
        print(f"{desc}...")
        return iterable

# Try to import cv2 for image encoding
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not available - will use pre-encoded frames")


class Phase2VLLMAnalyzer:
    def __init__(self, config_path: str = "vllm_config.json", db_manager: ChromaDBManager = None):
        """Initialize Phase 2 VLLM analyzer"""
        self.config = self.load_config(config_path)
        self.session = requests.Session()
        self.db_manager = db_manager if db_manager else ChromaDBManager()

        # VLLM settings
        self.api_base = self.config["vllm_settings"]["api_base"]
        self.model_name = self.config["vllm_settings"]["model_name"]
        self.temperature = self.config["vllm_settings"]["temperature"]
        self.max_tokens = self.config["vllm_settings"]["max_tokens"]
        self.timeout = self.config["vllm_settings"]["timeout"]

        # Кэш результатов текущей сессии для поиска контекста
        self.current_session_results = {}

        # Рабочая директория для кроссплатформенности
        self.working_dir = Path.cwd()  # Текущая рабочая директория

        # Performance optimization settings
        self.preventive_restart_interval = 40  # Перезапуск каждые N сегментов
        self.processed_segments_count = 0
        self._last_description = None  # Для детекции повторяющихся ответов
        self.quality_degradation_count = 0

        print(f"✓ Phase2VLLMAnalyzer initialized")
        print(f"  API: {self.api_base}")
        print(f"  Model: {self.model_name}")
        print(f"  Timeout: {self.timeout}")
        print(f"  Platform: {platform.system()}")
        print(f"  Working dir: {self.working_dir}")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def test_vllm_connection(self) -> bool:
        """Test connection to VLLM API"""
        try:
            print("Testing VLLM connection...")
            # Test with models endpoint
            response = self.session.get(f"{self.api_base}/v1/models", timeout=5)
            if response.status_code == 200:
                print("✓ VLLM API connection successful")
                return True
            else:
                print(f"✗ VLLM API returned {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ VLLM API connection failed: {type(e).__name__}")
            return False
    
    def load_phase1_data(self, video_name: str) -> List[Dict]:
        """Load Phase 1 results"""
        pickle_path = Path('output') / f'{video_name}_phase1_data.pkl'
        
        if not pickle_path.exists():
            raise FileNotFoundError(f"Phase 1 data not found: {pickle_path}")
        
        print(f"Loading Phase 1 data from: {pickle_path}")
        with open(pickle_path, 'rb') as f:
            segment_data = pickle.load(f)
        
        print(f"✓ Loaded {len(segment_data)} segments from Phase 1")
        return segment_data
    
    def create_vllm_prompt(self, previous_context: str, audio_transcript: str) -> str:
        """Create structured prompt for VLLM"""
        
        base_prompt = """Analyze this video segment and provide a detailed description in Russian language ONLY.

PREVIOUS SEGMENT CONTEXT:
{previous_context}

AUDIO TRANSCRIPT:
{audio_transcript}

IMPORTANT: Answer ONLY in Russian language. Do not use any other languages.

TASKS:
1. Describe what's happening in the video: people, actions, environment, objects
2. Translate any speech to Russian if needed
3. Extract keywords (exact terms from the video)
4. Generate search_terms: keywords + synonyms, slang, related concepts for better search
5. Determine content type and atmosphere
6. Note if the topic or character changed compared to the previous segment

Respond with ONLY the JSON object below - no explanations, no additional text, no descriptions.

CRITICAL FORMATTING RULES:
- Return ONLY the JSON structure, nothing else
- Do NOT use line breaks (\\n) or tabs (\\t) inside string values  
- Write all text in single lines without line breaks
- Do NOT add any explanatory text before or after JSON
- Ensure strict JSON syntax

Required JSON format:
{{
  "description": "подробное описание происходящего на русском языке",
  "dialogue_translation": "перевод речи на РУССКИЙ язык",
  "keywords": ["точные", "ключевые", "слова", "из", "видео"],
  "search_terms": "ключевые слова плюс синонимы жаргон разговорные варианты и связанные понятия через пробел для поиска",
  "content_type": "тип контента на русском",
  "mood_atmosphere": "настроение и атмосфера на русском",
  "scene_change": true/false,
  "new_information": "новая информация на русском языке",
  "confidence": "высокая/средняя/низкая"
}}

CRITICAL: 
- Use ONLY Russian language in all fields. No ANY another languages.
- Return ONLY the JSON object above, nothing more.
- Do NOT explain what you're doing.
- Do NOT add any text before or after the JSON."""
        
        return base_prompt.format(
            previous_context=previous_context or "Нет (это первый сегмент)",
            audio_transcript=audio_transcript or "Нет транскрипции"
        )
    
    def detect_quality_degradation(self, response: str, analysis_json: dict) -> tuple:
        """Детекция признаков деградации качества ответа VLLM

        Returns:
            tuple: (is_degraded: bool, degradation_flags: list, is_critical_loop: bool)
        """

        degradation_flags = []
        critical_loop_flags = ["keyword_loop", "keyword_flood", "search_terms_loop"]

        # 1. Проверка на китайские символы в русскоязычных полях
        chinese_chars = any('\u4e00' <= char <= '\u9fff' for char in response)
        if chinese_chars:
            degradation_flags.append("chinese_characters")
            print("  ⚠️ Обнаружены китайские символы - деградация качества")

        # 2. Проверка на слишком короткие описания
        description = analysis_json.get('description', '')
        if len(description) < 20:
            degradation_flags.append("short_description")
            print(f"  ⚠️ Слишком короткое описание ({len(description)} символов) - возможная деградация")

        # 3. Проверка на повторяющиеся ответы
        if self._last_description and self._last_description == description:
            degradation_flags.append("duplicate_description")
            print("  ⚠️ Повторяющееся описание - деградация качества")

        # 4. Проверка на пустые обязательные поля
        required_fields = ['description', 'confidence']
        empty_fields = [f for f in required_fields if not analysis_json.get(f, '').strip()]
        if empty_fields:
            degradation_flags.append("empty_required_fields")
            print(f"  ⚠️ Пустые обязательные поля: {empty_fields}")

        # 5. Проверка на некорректные значения confidence
        confidence = analysis_json.get('confidence', '').lower()
        valid_confidence = ['высокая', 'средняя', 'низкая', 'high', 'medium', 'low']
        if confidence and confidence not in valid_confidence:
            degradation_flags.append("invalid_confidence")
            print(f"  ⚠️ Некорректное значение confidence: {confidence}")

        # 6. Детекция повторяющихся слов в keywords (LLM loop) - КРИТИЧЕСКАЯ
        keywords = analysis_json.get('keywords', [])
        if isinstance(keywords, list) and len(keywords) > 5:
            # Проверяем есть ли слово которое повторяется >3 раз подряд
            for i in range(len(keywords) - 3):
                if keywords[i] == keywords[i+1] == keywords[i+2] == keywords[i+3]:
                    degradation_flags.append("keyword_loop")
                    print(f"  🔴 CRITICAL: LLM loop detected: '{keywords[i]}' повторяется 4+ раз подряд")
                    break
            # Также проверяем общее количество уникальных vs всего
            if len(keywords) > 10:
                unique_ratio = len(set(keywords)) / len(keywords)
                if unique_ratio < 0.3:  # Менее 30% уникальных
                    degradation_flags.append("keyword_flood")
                    print(f"  🔴 CRITICAL: Keyword flood: {len(set(keywords))}/{len(keywords)} уникальных ({unique_ratio:.1%})")

        # 7. Детекция повторов в search_terms - КРИТИЧЕСКАЯ
        search_terms = analysis_json.get('search_terms', '')
        if isinstance(search_terms, str) and len(search_terms) > 100:
            words = search_terms.split()
            if len(words) > 10:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.3:
                    degradation_flags.append("search_terms_loop")
                    print(f"  🔴 CRITICAL: Search terms loop: {len(set(words))}/{len(words)} уникальных ({unique_ratio:.1%})")

        # Обновляем последнее описание
        self._last_description = description

        # Проверяем критические loop флаги
        is_critical_loop = any(flag in critical_loop_flags for flag in degradation_flags)

        # Если найдены признаки деградации
        if degradation_flags:
            self.quality_degradation_count += 1
            print(f"  📊 Обнаружена деградация качества: {', '.join(degradation_flags)}")
            print(f"  📊 Общее количество деградаций: {self.quality_degradation_count}")
            if is_critical_loop:
                print(f"  🔴 ТРЕБУЕТСЯ НЕМЕДЛЕННЫЙ ПЕРЕЗАПУСК LLM!")
            return (True, degradation_flags, is_critical_loop)

        return (False, [], False)
    
    def check_preventive_restart(self, segment_index: int) -> bool:
        """Проверка необходимости профилактического перезапуска"""
        
        # Профилактический перезапуск каждые N сегментов
        if (segment_index + 1) % self.preventive_restart_interval == 0:
            print(f"  🔄 Время для профилактического перезапуска после {segment_index + 1} сегментов")
            return True
        
        # Перезапуск при накоплении деградаций качества
        if self.quality_degradation_count >= 3:
            print(f"  🔄 Перезапуск из-за накопления деградаций ({self.quality_degradation_count})")
            self.quality_degradation_count = 0  # Сбрасываем счетчик
            return True
        
        return False
    
    def _extract_from_markdown_blocks(self, response: str) -> str:
        """Extract JSON from markdown code blocks"""
        # Более надежная очистка markdown оберток
        if '```json' in response:
            # Найти начало и конец JSON блока
            start_marker = '```json'
            end_marker = '```'
            
            start_idx = response.find(start_marker)
            if start_idx != -1:
                # Начинаем поиск после ```json
                json_start = start_idx + len(start_marker)
                
                # Ищем закрывающий ```
                end_idx = response.find(end_marker, json_start)
                if end_idx != -1:
                    return response[json_start:end_idx].strip()
                else:
                    # Если нет закрывающего - берем все после ```json
                    return response[json_start:].strip()
        
        # Если просто начинается с ```
        elif response.startswith('```') and response.endswith('```'):
            lines = response.split('\n')
            if len(lines) > 2:
                return '\n'.join(lines[1:-1])
        
        # Return as-is if no markdown wrapping
        return response
    
    def _is_structured_markdown(self, text: str) -> bool:
        """Check if text looks like structured markdown with our expected fields"""
        # Поиск признаков структурированного markdown ответа
        markdown_indicators = [
            '**Описание:**', '**Description:**',
            '**Диалог:**', '**Dialogue:**', 
            '**Ключевые слова:**', '**Keywords:**',
            '**Тип контента:**', '**Content type:**',
            '**Атмосfera:**', '**Mood:**',
            '- Описание:', '- Description:',
            '- Диалог:', '- Dialogue:',
            '- Ключевые слова:', '- Keywords:'
        ]
        
        # Если есть хотя бы 2 индикатора, считаем structured
        found_indicators = sum(1 for indicator in markdown_indicators if indicator.lower() in text.lower())
        return found_indicators >= 2
    
    def _convert_markdown_to_json(self, markdown: str) -> str:
        """Convert structured markdown to JSON"""
        import re
        
        try:
            # Результирующий JSON объект
            result = {}
            
            # Шаблоны для поиска полей
            patterns = {
                'description': [
                    r'\*\*Описание:\*\*\s*(.+?)(?=\*\*|\n\n|$)',
                    r'\*\*Description:\*\*\s*(.+?)(?=\*\*|\n\n|$)',
                    r'- Описание:\s*(.+?)(?=\n-|\n\n|$)',
                    r'- Description:\s*(.+?)(?=\n-|\n\n|$)'
                ],
                'dialogue_translation': [
                    r'\*\*Диалог:\*\*\s*(.+?)(?=\*\*|\n\n|$)',
                    r'\*\*Dialogue:\*\*\s*(.+?)(?=\*\*|\n\n|$)',
                    r'- Диалог:\s*(.+?)(?=\n-|\n\n|$)',
                    r'- Dialogue:\s*(.+?)(?=\n-|\n\n|$)'
                ],
                'keywords': [
                    r'\*\*Ключевые слова:\*\*\s*(.+?)(?=\*\*|\n\n|$)',
                    r'\*\*Keywords:\*\*\s*(.+?)(?=\*\*|\n\n|$)',
                    r'- Ключевые слова:\s*(.+?)(?=\n-|\n\n|$)',
                    r'- Keywords:\s*(.+?)(?=\n-|\n\n|$)'
                ],
                'content_type': [
                    r'\*\*Тип контента:\*\*\s*(.+?)(?=\*\*|\n\n|$)',
                    r'\*\*Content type:\*\*\s*(.+?)(?=\*\*|\n\n|$)',
                    r'- Тип контента:\s*(.+?)(?=\n-|\n\n|$)',
                    r'- Content type:\s*(.+?)(?=\n-|\n\n|$)'
                ],
                'mood_atmosphere': [
                    r'\*\*Атмосфера:\*\*\s*(.+?)(?=\*\*|\n\n|$)',
                    r'\*\*Mood:\*\*\s*(.+?)(?=\*\*|\n\n|$)',
                    r'- Атмосфера:\s*(.+?)(?=\n-|\n\n|$)',
                    r'- Mood:\s*(.+?)(?=\n-|\n\n|$)'
                ]
            }
            
            # Извлекаем каждое поле
            for field, field_patterns in patterns.items():
                for pattern in field_patterns:
                    match = re.search(pattern, markdown, re.DOTALL | re.IGNORECASE)
                    if match:
                        value = match.group(1).strip()
                        
                        # Для ключевых слов пытаемся распарсить как список
                        if field == 'keywords' and value:
                            # Удаляем markdown списки и разделяем
                            value = re.sub(r'^[-*]\s*', '', value, flags=re.MULTILINE)
                            keywords = [kw.strip() for kw in re.split(r'[,;]|\n', value) if kw.strip()]
                            if keywords:
                                result[field] = keywords
                            else:
                                result[field] = [value]
                        else:
                            result[field] = value
                        break
            
            # Добавляем дефолтные значения если поля отсутствуют
            if 'description' not in result:
                result['description'] = 'Описание недоступно'
            if 'confidence' not in result:
                result['confidence'] = 'средняя'
            if 'scene_change' not in result:
                result['scene_change'] = False
            if 'new_information' not in result:
                result['new_information'] = 'Информация извлечена из markdown'
                
            # Конвертируем в JSON
            import json
            json_str = json.dumps(result, ensure_ascii=False, indent=2)
            
            # Проверяем что получился валидный JSON
            json.loads(json_str)
            return json_str
            
        except Exception as e:
            print(f"    ❌ Error converting markdown to JSON: {e}")
            return None
    
    def extract_json_from_response(self, response: str) -> str:
        """Extract JSON from markdown code block or convert markdown structure to JSON"""
        import re
        import json
        
        response = response.strip()
        
        # 1. Сначала попробуем найти JSON в markdown блоках
        extracted = self._extract_from_markdown_blocks(response)
        
        # 2. Если нашли JSON в блоках, используем его
        if extracted != response:
            print(f"    📝 Extracted from markdown code block")
        
        # 3. Если ответ не JSON, но структурированный markdown - пытаемся конвертировать
        elif self._is_structured_markdown(response):
            print(f"    🔄 Detected structured markdown, attempting conversion...")
            converted = self._convert_markdown_to_json(response)
            if converted:
                print(f"    ✅ Successfully converted markdown to JSON")
                extracted = converted
            else:
                print(f"    ❌ Failed to convert markdown to JSON")
                extracted = response
        else:
            # Return as-is if no special handling needed
            extracted = response
        
        # АККУРАТНАЯ ОЧИСТКА JSON
        
        # 1. Удалить только явно проблемные контрольные символы
        # Но сохранить переносы строк и табы для структуры JSON
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', extracted)
        
        # 2. Убрать лишние пробелы только в конце строк, но сохранить структуру
        lines = cleaned.split('\n')
        cleaned_lines = [line.rstrip() for line in lines]
        cleaned = '\n'.join(cleaned_lines)
        
        # 3. Удалить trailing commas если есть
        cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)
        
        # 4. Проверить что это валидный JSON
        cleaned = cleaned.strip()
        
        # Если пустой после очистки
        if not cleaned:
            print(f"    ⚠️ JSON empty after basic cleaning")
            return '{}'
        
        # Попробовать распарсить
        try:
            json.loads(cleaned)
            print(f"    ✓ JSON is valid after cleaning")
            return cleaned
        except json.JSONDecodeError as e:
            print(f"    JSON still invalid after cleaning: {e}")
            print(f"    Cleaned length: {len(cleaned)}")
            print(f"    First 300 chars: {repr(cleaned[:300])}")
            
            # Попытка починить отсутствующую закрывающую скобку
            if cleaned.startswith('{') and not cleaned.rstrip().endswith('}'):
                fixed = cleaned.rstrip() + '}'
                try:
                    json.loads(fixed)
                    print(f"    ✓ Fixed by adding closing brace")
                    return fixed
                except:
                    print(f"    ❌ Still broken after adding closing brace")
            
            # Если всё сломано, возвращаем оригинальный ответ как есть
            # НЕ подменяем на заглушку - сохраняем что вернул LLM
            print(f"    → Returning original LLM response as fallback")
            return cleaned
    
    def get_base64_frames(self, data: Dict) -> List[str]:
        """Get base64 frames from segment data"""
        # Try to use pre-encoded base64 frames first
        if 'frames_base64' in data and data['frames_base64']:
            return data['frames_base64']
        
        # Fallback to encoding frames if cv2 is available
        if HAS_CV2 and 'frames' in data:
            frames = data['frames']
            base64_images = []
            for frame in tqdm(frames, desc="    Converting frames to base64", leave=False):
                success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if success:
                    base64_string = base64.b64encode(buffer).decode('utf-8')
                    base64_images.append(base64_string)
            return base64_images
        
        # No frames available
        return []
    
    def analyze_segment_vllm(self, base64_frames: List[str], prompt: str) -> Optional[str]:
        """Analyze segment frames with VLLM"""
        
        if not base64_frames:
            return None
        
        # Prepare VLLM request
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt}
                ] + [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img}"}
                    } for img in base64_frames
                ]
            }
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        # Make request with connection error handling
        try:
            response = self.session.post(
                f"{self.api_base}/v1/chat/completions",
                json=payload,
                timeout=self.timeout
            )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            print(f"    ❌ VLLM connection error: {type(e).__name__}")
            return None
        except Exception as e:
            print(f"    ❌ VLLM request error: {type(e).__name__}: {e}")
            return None
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                if content:
                    # Clean up response: trim whitespace and newlines
                    cleaned_content = content.strip()
                    if cleaned_content:
                        return cleaned_content
                    else:
                        print(f"    ⚠️ VLLM returned content but it's empty after cleaning")
                else:
                    print(f"    ⚠️ VLLM returned null/empty content")
            else:
                print(f"    ⚠️ VLLM response missing choices: {result}")
        else:
            print(f"    ❌ VLLM HTTP error {response.status_code}: {response.text}")
        
        return None
    
    def get_flag_path(self, filename: str) -> Path:
        """Получить кроссплатформенный путь к файлу флага"""
        # Всегда используем текущую рабочую директорию
        return self.working_dir / filename
    
    def write_flag_file(self, filename: str, data: Dict[str, Any]) -> bool:
        """Кроссплатформенная запись файла флага"""
        flag_path = self.get_flag_path(filename)
        
        try:
            # Используем безопасную кодировку для Windows и Linux
            with open(flag_path, 'w', encoding='utf-8', newline='\n') as f:
                json.dump(data, f, ensure_ascii=True, indent=2)
            
            print(f"✓ Flag file created: {flag_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating flag file {flag_path}: {e}")
            return False
    
    def read_status_file(self, filename: str) -> Optional[Dict[str, Any]]:
        """Кроссплатформенное чтение файла статуса"""
        status_path = self.get_flag_path(filename)
        
        if not status_path.exists():
            return None
        
        try:
            with open(status_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error reading status file {status_path}: {e}")
            return None
    
    def stop_model(self, reason: str = "manual") -> None:
        """Остановить модель через флаг оркестратора"""
        print(f"🛑 Запрашиваем остановку модели у оркестратора (причина: {reason})")
        
        flag_data = {
            "action": "stop",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "reason": reason,
            "platform": platform.system()
        }
        
        if self.write_flag_file('restart_vllm.flag', flag_data):
            print(f"✅ Флаг остановки создан, ожидаем обработки оркестратором")
        else:
            print(f"❌ Ошибка создания флага остановки")
    
    def start_model(self, model_script: str = "qwen2_5_vl_32b.sh", reason: str = "manual") -> None:
        """Запустить модель через флаг"""
        flag_data = {
            "action": "restart",
            "model_script": model_script,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "reason": reason,
            "platform": platform.system()
        }
        
        if self.write_flag_file('restart_vllm.flag', flag_data):
            print(f"🚀 Запрошен запуск модели: {model_script} (причина: {reason})")
        else:
            print(f"❌ Ошибка создания флага запуска")
    
    def restart_model(self, model_script: str = "qwen3_vl_32b.sh", reason: str = "error") -> None:
        """Перезапустить модель через флаг оркестратора"""
        print(f"🔄 Запрашиваем перезапуск модели у оркестратора: {model_script} (причина: {reason})")
        
        flag_data = {
            "action": "restart",
            "model_script": model_script,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "reason": reason,
            "platform": platform.system()
        }
        
        if self.write_flag_file('restart_vllm.flag', flag_data):
            print(f"✅ Флаг перезапуска создан, ожидаем обработки оркестратором")
        else:
            print(f"❌ Ошибка создания флага перезапуска")
    
    def get_model_status(self) -> Optional[Dict[str, Any]]:
        """Получить статус модели от оркестратора"""
        return self.read_status_file('model_status.json')
    
    def check_model_health(self) -> bool:
        """Проверить здоровье модели (доступность API)"""
        try:
            return self.test_vllm_connection()
        except Exception as e:
            print(f"❌ Ошибка проверки здоровья модели: {e}")
            return False
    
    def auto_start_model_if_needed(self, max_wait_time: int = 180) -> bool:
        """Автозапуск модели если она недоступна"""
        print("🚀 Автозапуск модели...")
        
        # 1. Сначала пробуем через оркестратор (только на Linux/WSL)
        if platform.system() == "Linux":
            print("   Запрашиваем запуск через оркестратор...")
            try:
                import subprocess
                result = subprocess.run(['./model_orchestrator_v2.sh', 'cron'], 
                                      capture_output=True, text=True, timeout=60,
                                      cwd=str(self.working_dir))
                if result.returncode == 0:
                    print("   ✅ Оркестратор запущен")
                else:
                    print(f"   ⚠️ Оркестратор завершился с кодом {result.returncode}")
                    print(f"   stderr: {result.stderr[:200] if result.stderr else 'нет ошибок'}")
            except Exception as e:
                print(f"   ❌ Ошибка запуска оркестратора: {e}")
        else:
            print("   Windows detected - используем только флаговый запуск...")
        
        # 2. В любом случае создаем флаг запуска (универсально)
        print("   Создаем флаг запуска модели...")
        self.start_model("qwen2_5_vl_32b.sh", "auto_start_python")
        
        # 3. Ждем запуска модели
        print(f"   ⏳ Ожидание запуска модели (максимум {max_wait_time} секунд)...")
        
        for attempt in range(max_wait_time // 10):
            time.sleep(10)
            if self.test_vllm_connection():
                print(f"   ✅ Модель запустилась после {(attempt + 1) * 10} секунд")
                return True
            else:
                print(f"   ⏳ Попытка {attempt + 1}/{max_wait_time // 10}: модель еще не готова...")
        
        print(f"   ❌ Модель не запустилась за {max_wait_time} секунд")
        return False
    
    def wait_for_model_stop(self, timeout: int = 60) -> bool:
        """Ждать остановки модели"""
        print(f"⏳ Ожидание остановки модели (максимум {timeout} секунд)...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = self.get_model_status()
            if not status or status.get('status') == 'stopped':
                print("✅ Модель остановлена")
                return True
            time.sleep(2)
        
        print("❌ Модель не остановилась в отведенное время")
        return False
    
    def request_model_restart(self, model_script: str = "qwen2_5_vl_32b.sh", reason: str = "error") -> None:
        """Запрос перезапуска модели через флаг (для совместимости)"""
        self.restart_model(model_script, reason)
    
    def wait_for_model_restart(self, timeout: int = 180) -> bool:
        """Ждать перезапуска модели и проверить готовность"""
        print(f"⏳ Ожидание перезапуска модели (максимум {timeout} секунд)...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Проверить что флаг исчез (оркестратор его обработал)
            flag_file = self.get_flag_path('restart_vllm.flag')
            if not flag_file.exists():
                print("✓ Флаг перезапуска обработан оркестратором")
                break
            time.sleep(5)
        
        # Дать время на перезапуск
        print("⏳ Даем время на полный перезапуск...")
        time.sleep(45)  # Увеличиваем время ожидания
        
        # Проверить готовность API
        max_checks = 18  # 90 секунд
        for i in range(max_checks):
            if self.test_vllm_connection():
                print(f"✅ Модель готова к работе после перезапуска!")
                return True
            else:
                print(f"   Проверка {i+1}/{max_checks}: модель еще не готова...")
                time.sleep(5)
        
        print("❌ Модель не готова после перезапуска")
        return False

    def get_previous_segment_context(self, video_name: str, current_index: int) -> str:
        """Получить описание предыдущего сегмента для контекста"""
        if current_index == 0:
            return ""  # Первый сегмент - нет контекста
        
        previous_index = current_index - 1
        
        # Сначала проверяем кэш текущей сессии
        if previous_index in self.current_session_results:
            cached_result = self.current_session_results[previous_index]
            if cached_result.get('success', False):
                try:
                    analysis = json.loads(cached_result.get('analysis', '{}'))
                    description = analysis.get('description', '')
                    if description:
                        print(f"  ✓ Найден контекст от сегмента {previous_index + 1} (текущая сессия): {description[:100]}...")
                        return description
                except:
                    pass
        
        # Если не найден в кэше, ищем в сохраненных результатах
        existing = self.load_existing_results(video_name)
        if existing:
            for segment in existing.get('segments', []):
                if segment.get('segment_index') == previous_index:
                    # Если есть успешно обработанный предыдущий сегмент
                    if segment.get('success', False) or ('parse_error' not in segment and 'error' not in segment):
                        try:
                            analysis = json.loads(segment.get('analysis', '{}'))
                            description = analysis.get('description', '')
                            if description:
                                print(f"  ✓ Найден контекст от сегмента {previous_index + 1} (файл): {description[:100]}...")
                                return description
                        except:
                            pass
        
        print(f"  ⚠️ Контекст предыдущего сегмента {previous_index + 1} не найден")
        return ""

    def process_phase2_analysis(self, video_name: str) -> None:
        """Process Phase 2: VLLM Analysis"""
        print(f"\n=== PHASE 2: VLLM Analysis for {video_name} ===")

        # === Check ChromaDB status ===
        video_status = self.db_manager.get_video_status(video_name)

        if not video_status:
            print(f"❌ Video {video_name} not found in ChromaDB - run Phase 1 first!")
            return

        if video_status.get("phase1_status") != "completed":
            print(f"⚠️ Phase 1 not completed for {video_name}, skipping...")
            return

        if video_status.get("phase2_status") == "completed":
            print(f"✓ Phase 2 already completed for {video_name}, skipping...")
            return

        if video_status.get("phase2_status") == "processing":
            # Check timeout
            start_time = datetime.fromisoformat(video_status["phase2_start_time"])
            video_duration = video_status["video_duration"]
            timeout = video_duration * 2

            if (datetime.now() - start_time).total_seconds() < timeout:
                print(f"⚠️ Phase 2 still processing for {video_name}, skipping...")
                return
            else:
                print(f"🔄 Phase 2 timeout detected for {video_name}, reprocessing...")

        # Set status to "processing"
        self.db_manager.update_video_status(
            video_name=video_name,
            phase2_status="processing",
            phase2_start_time=datetime.now().isoformat()
        )

        # Test VLLM connection and auto-start if needed
        if not self.test_vllm_connection():
            print("⚠️ VLLM API недоступен, запускаем модель автоматически...")
            self.auto_start_model_if_needed()

            # Проверяем еще раз после запуска
            if not self.test_vllm_connection():
                raise ConnectionError("VLLM API недоступен даже после автозапуска")
        
        # Load Phase 1 data
        segment_data = self.load_phase1_data(video_name)
        
        # Определить какие сегменты нужно обработать
        segments_to_process = self.get_segments_to_process(video_name, len(segment_data))
        
        if not segments_to_process:
            print("✅ Все сегменты уже успешно обработаны!")
            return
        
        print(f"\n🚀 Начинаем обработку {len(segments_to_process)} сегментов")
        
        # Обрабатываем только нужные сегменты
        results = []
        
        # Фильтруем segment_data по индексам которые нужно обработать
        segments_to_analyze = [segment_data[i] for i in segments_to_process]
        
        for data in tqdm(segments_to_analyze, desc="Phase 2: VLLM Analysis"):
            i = data['index']
            segment = data['segment']
            transcript = data['transcript']
            
            print(f"\nAnalyzing segment {i+1}/60 with VLLM")
            print(f"  Time: {segment['start']:.1f}s - {segment['end']:.1f}s")
            
            # Проверка профилактического перезапуска
            if self.check_preventive_restart(i):
                print(f"  🔄 Выполняется профилактический перезапуск VLLM...")
                self.restart_model(reason=f"preventive_restart_segment_{i+1}")
                
                # Ожидание стабилизации после перезапуска
                print(f"  ⏳ Ожидание стабилизации VLLM (180 секунд)...")
                time.sleep(180)
                
                # Проверка доступности после перезапуска
                if not self.check_model_health():
                    print(f"  ❌ VLLM недоступен после перезапуска")
                    print(f"  🛑 ОСТАНОВКА: невозможно продолжить без VLLM")
                    break
                
                print(f"  ✅ VLLM готов к работе после перезапуска")
            
            # Get base64 frames
            base64_frames = self.get_base64_frames(data)
            if not base64_frames:
                print(f"  ❌ No frames available for segment {i+1}")
                print(f"  🛑 ОСТАНОВКА: не удается получить кадры")
                break
            
            print(f"  ✓ Got {len(base64_frames)} base64 frames")
            
            # Получаем контекст предыдущего сегмента
            previous_description = self.get_previous_segment_context(video_name, i)
            
            # Create prompt
            prompt = self.create_vllm_prompt(previous_description, transcript)
            
            # Analyze with VLLM (with retry logic)
            analysis = None
            max_retries = 5
            
            for attempt in range(max_retries):
                print(f"  VLLM attempt {attempt+1}/{max_retries}...")
                analysis = self.analyze_segment_vllm(base64_frames, prompt)
                
                if analysis and analysis.strip():
                    print(f"  ✓ VLLM analysis completed")
                    # Debug: show what VLLM returned
                    print(f"  Raw VLLM response: {analysis}")
                    break
                else:
                    print(f"  ⚠️ Empty/invalid VLLM response on attempt {attempt+1}")
                    if attempt < max_retries - 1:
                        print(f"    Retrying...")
                        time.sleep(1.0)  # Brief pause before retry
                    else:
                        print(f"  🛑 ОСТАНОВКА: VLLM не отвечает после {max_retries} попыток")
                        print(f"     Возможно VLLM сервер нужно перезапустить")
                        break
            
            if not analysis or not analysis.strip():
                print(f"  ⚠️ Пустой ответ от VLLM для сегмента {i+1}")
                print(f"     Запрашиваем перезапуск модели...")
                
                # Запросить перезапуск модели
                self.restart_model(reason=f"empty_response_segment_{i+1}")
                
                # Дождаться перезапуска
                if self.wait_for_model_restart():
                    print(f"  🔄 Модель перезапущена, повторяем сегмент {i+1}")
                    # Повторить анализ этого же сегмента после перезапуска
                    analysis = self.analyze_segment_vllm(base64_frames, prompt)
                    if analysis and analysis.strip():
                        print(f"  ✅ После перезапуска получен ответ для сегмента {i+1}")
                    else:
                        print(f"  🛑 После перезапуска все еще пустой ответ - останавливаем")
                        break
                else:
                    print(f"  🛑 Перезапуск не удался - останавливаем обработку")
                    break
                
            if analysis and analysis.strip():
                try:
                    # Extract and validate JSON - метод уже включает валидацию
                    clean_json = self.extract_json_from_response(analysis)
                    print(f"  ✓ JSON cleaned and validated")
                    
                    # Парсим уже очищенный и валидированный JSON
                    analysis_json = json.loads(clean_json)
                    
                    # Исправляем распространённые ошибки в именах полей
                    if 'mchange' in analysis_json and 'scene_change' not in analysis_json:
                        analysis_json['scene_change'] = analysis_json['mchange']
                        del analysis_json['mchange']
                    
                    if 'information' in analysis_json and 'new_information' not in analysis_json:
                        analysis_json['new_information'] = analysis_json['information'] 
                        del analysis_json['information']
                    
                    # Проверяем обязательные поля (мягкая проверка)
                    required_fields = ['description', 'confidence']
                    missing_fields = [f for f in required_fields if f not in analysis_json]
                    if missing_fields:
                        print(f"  ⚠️ Missing fields: {missing_fields} - using defaults")
                        if 'description' not in analysis_json:
                            analysis_json['description'] = "Описание недоступно"
                        if 'confidence' not in analysis_json:
                            analysis_json['confidence'] = "средняя"
                    
                    # Детекция деградации качества VLLM
                    quality_degraded, degradation_flags, is_critical_loop = self.detect_quality_degradation(analysis, analysis_json)
                    if quality_degraded:
                        print(f"  ⚠️ Обнаружена деградация качества для сегмента {i+1}")

                        # При критическом loop - немедленный перезапуск и retry
                        if is_critical_loop:
                            print(f"  🔴 CRITICAL LOOP DETECTED! Перезапуск LLM и повтор сегмента...")
                            self.restart_model(reason=f"critical_loop_segment_{i+1}")
                            if self.wait_for_model_restart():
                                print(f"  🔄 LLM перезапущена, повторяем сегмент {i+1}")
                                # Повторить этот же сегмент
                                retry_analysis = self.analyze_segment_vllm(base64_frames, prompt)
                                if retry_analysis and retry_analysis.strip():
                                    retry_clean_json = self.extract_json_from_response(retry_analysis)
                                    try:
                                        retry_analysis_json = json.loads(retry_clean_json)
                                        # Используем retry результат вместо оригинального
                                        analysis_json = retry_analysis_json
                                        clean_json = retry_clean_json
                                        print(f"  ✅ Сегмент {i+1} успешно переобработан после перезапуска LLM")
                                    except json.JSONDecodeError:
                                        print(f"  ⚠️ JSON ошибка после retry, сохраняем оригинал")
                            else:
                                print(f"  ⚠️ Перезапуск не удался, сохраняем данные как есть")
                    
                    # Проверяем что keywords это список
                    if 'keywords' in analysis_json and not isinstance(analysis_json['keywords'], list):
                        if isinstance(analysis_json['keywords'], str):
                            analysis_json['keywords'] = [analysis_json['keywords']]
                        else:
                            analysis_json['keywords'] = []
                    
                    previous_description = analysis_json.get('description', '')
                    
                    result = {
                        'segment_index': i,
                        'start_time': segment['start'],
                        'end_time': segment['end'],
                        'duration': segment['duration'],
                        'transcript': transcript,
                        'analysis': clean_json,  # Используем очищенный JSON вместо оригинального
                        'frame_count': len(base64_frames),
                        'success': True
                    }
                    results.append(result)
                    # Сохраняем в кэш текущей сессии для поиска контекста
                    self.current_session_results[i] = result
                    
                    # Обновляем счетчик для оптимизации производительности
                    self.processed_segments_count += 1
                    
                    print(f"  ✅ Segment {i+1} processed successfully")
                    print(f"  📊 Всего обработано сегментов: {self.processed_segments_count}")
                    
                except json.JSONDecodeError as e:
                    print(f"  ❌ JSON parsing error for segment {i+1}:")
                    print(f"      Error: {e}")
                    print(f"      Raw VLLM response length: {len(analysis)} chars")
                    print(f"      First 200 chars of raw response: {repr(analysis[:200])}")
                    print(f"      Cleaned JSON length: {len(clean_json)} chars")  
                    print(f"      First 200 chars of cleaned JSON: {repr(clean_json[:200])}")
                    
                    # Несколько попыток перезапуска модели и повтора
                    max_restart_attempts = 3
                    restart_successful = False
                    
                    for restart_attempt in range(max_restart_attempts):
                        print(f"  🔄 Попытка перезапуска {restart_attempt+1}/{max_restart_attempts} из-за JSON ошибки...")
                        self.restart_model(reason=f"json_parse_error_segment_{i+1}_attempt_{restart_attempt+1}")
                        
                        # Дождаться перезапуска
                        if self.wait_for_model_restart():
                            print(f"  🔄 Модель перезапущена, повторяем сегмент {i+1}")
                            # Повторить анализ этого же сегмента после перезапуска
                            retry_analysis = self.analyze_segment_vllm(base64_frames, prompt)
                            if retry_analysis and retry_analysis.strip():
                                try:
                                    # Попробовать распарсить после перезапуска
                                    retry_clean_json = self.extract_json_from_response(retry_analysis)
                                    retry_analysis_json = json.loads(retry_clean_json)
                                    
                                    # Проверки и исправления как выше
                                    if 'mchange' in retry_analysis_json and 'scene_change' not in retry_analysis_json:
                                        retry_analysis_json['scene_change'] = retry_analysis_json['mchange']
                                        del retry_analysis_json['mchange']
                                    
                                    if 'information' in retry_analysis_json and 'new_information' not in retry_analysis_json:
                                        retry_analysis_json['new_information'] = retry_analysis_json['information'] 
                                        del retry_analysis_json['information']
                                    
                                    required_fields = ['description', 'confidence']
                                    missing_fields = [f for f in required_fields if f not in retry_analysis_json]
                                    if missing_fields:
                                        if 'description' not in retry_analysis_json:
                                            retry_analysis_json['description'] = "Описание недоступно"
                                        if 'confidence' not in retry_analysis_json:
                                            retry_analysis_json['confidence'] = "средняя"
                                    
                                    if 'keywords' in retry_analysis_json and not isinstance(retry_analysis_json['keywords'], list):
                                        if isinstance(retry_analysis_json['keywords'], str):
                                            retry_analysis_json['keywords'] = [retry_analysis_json['keywords']]
                                        else:
                                            retry_analysis_json['keywords'] = []
                                    
                                    result = {
                                        'segment_index': i,
                                        'start_time': segment['start'],
                                        'end_time': segment['end'],
                                        'duration': segment['duration'],
                                        'transcript': transcript,
                                        'analysis': retry_clean_json,
                                        'frame_count': len(base64_frames),
                                        'success': True,
                                        'retry_after_restart': True,
                                        'restart_attempts': restart_attempt + 1
                                    }
                                    results.append(result)
                                    self.current_session_results[i] = result
                                    print(f"  ✅ Segment {i+1} processed successfully after restart (attempt {restart_attempt+1})")
                                    restart_successful = True
                                    break  # Успешно - выходим из цикла попыток
                                    
                                except json.JSONDecodeError as retry_e:
                                    print(f"  ❌ JSON ошибка на попытке {restart_attempt+1}: {retry_e}")
                                    if restart_attempt < max_restart_attempts - 1:
                                        print(f"    Пробуем еще раз...")
                                        continue
                                except Exception as retry_e:
                                    print(f"  ❌ Неожиданная ошибка на попытке {restart_attempt+1}: {retry_e}")
                                    if restart_attempt < max_restart_attempts - 1:
                                        print(f"    Пробуем еще раз...")
                                        continue
                            else:
                                print(f"  ❌ Пустой ответ от VLLM на попытке {restart_attempt+1}")
                                if restart_attempt < max_restart_attempts - 1:
                                    print(f"    Пробуем еще раз...")
                                    continue
                        else:
                            print(f"  ❌ Перезапуск не удался на попытке {restart_attempt+1}")
                            if restart_attempt < max_restart_attempts - 1:
                                print(f"    Пробуем еще раз...")
                                continue
                    
                    # Если все попытки провалились
                    if restart_successful:
                        continue  # Переходим к следующему сегменту
                    else:
                        print(f"  ⚠️ Все {max_restart_attempts} попыток перезапуска провалились")
                    
                    # Если все попытки провалились, сохраняем как fallback
                    result = {
                        'segment_index': i,
                        'start_time': segment['start'],
                        'end_time': segment['end'],
                        'duration': segment['duration'],
                        'transcript': transcript,
                        'analysis': clean_json,  # Сохраняем оригинальный ответ LLM как есть
                        'frame_count': len(base64_frames),
                        'parse_error': str(e),
                        'success': False
                    }
                    results.append(result)
                    print(f"  ⚠️ Segment {i+1} saved as fallback with original LLM response")
                    
                except Exception as e:
                    print(f"  ❌ Unexpected error processing segment {i+1}:")
                    print(f"      Error: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"      Raw VLLM response: {repr(analysis)}")
                    
                    # Сохраняем как fallback даже при неожиданных ошибках
                    result = {
                        'segment_index': i,
                        'start_time': segment['start'],
                        'end_time': segment['end'],
                        'duration': segment['duration'],
                        'transcript': transcript,
                        'analysis': analysis,  # Сохраняем сырой ответ
                        'frame_count': len(base64_frames),
                        'error': str(e)
                    }
                    results.append(result)
                    print(f"  ⚠️ Segment {i+1} saved as fallback with raw response")
            else:
                print(f"  ❌ VLLM analysis failed for segment {i+1} after {max_retries} attempts")
        
        print(f"\n✓ Phase 2 batch complete: {len(results)} новых сегментов обработано")

        # Save results
        self.save_phase2_results(video_name, results)

        # Update ChromaDB status to "completed"
        self.db_manager.update_video_status(
            video_name=video_name,
            phase2_status="completed",
            phase2_segments_analyzed=len(results)
        )

        print(f"✓ Phase 2 analysis complete for {video_name}. ChromaDB status updated.")
    
    def load_existing_results(self, video_name: str) -> Optional[Dict[str, Any]]:
        """Загрузить существующие результаты Phase 2"""
        output_path = Path('output') / f'{video_name}_phase2_vllm_analysis.json'
        
        if not output_path.exists():
            print(f"No existing results found: {output_path}")
            return None
            
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"✓ Loaded existing results: {len(data.get('segments', []))} segments")
                return data
        except Exception as e:
            print(f"❌ Error loading existing results: {e}")
            return None
    
    def get_segments_to_process(self, video_name: str, total_segments: int) -> List[int]:
        """Определить какие сегменты нужно обработать"""
        if FULL_REPROCESS:
            print(f"🔄 FULL_REPROCESS=True - обрабатываем все {total_segments} сегментов заново")
            return list(range(total_segments))
        
        existing = self.load_existing_results(video_name)
        if not existing:
            print(f"📝 Нет существующих результатов - обрабатываем все {total_segments} сегментов")
            return list(range(total_segments))
        
        # Находим успешно обработанные сегменты
        processed_indices = set()
        for segment in existing.get('segments', []):
            # Для обратной совместимости: если нет поля success, но есть analysis без ошибок
            is_success = segment.get('success', False) or (
                'analysis' in segment and 
                'parse_error' not in segment and 
                'error' not in segment
            )
            if is_success:
                processed_indices.add(segment.get('segment_index'))
        
        # Определяем недостающие
        missing = [i for i in range(total_segments) if i not in processed_indices]
        
        print(f"📊 Анализ существующих результатов:")
        print(f"   Всего сегментов: {total_segments}")
        print(f"   Уже обработано успешно: {len(processed_indices)}")
        print(f"   Нужно обработать: {len(missing)}")
        if missing:
            print(f"   Пропущенные индексы: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        
        return missing
    
    def merge_results(self, video_name: str, new_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Объединить новые результаты с существующими"""
        existing = self.load_existing_results(video_name)
        
        if not existing or FULL_REPROCESS:
            # Создаем новую структуру
            all_segments = new_results
        else:
            # Объединяем результаты
            existing_segments = {seg['segment_index']: seg for seg in existing.get('segments', [])}
            
            # Добавляем новые результаты
            for new_seg in new_results:
                existing_segments[new_seg['segment_index']] = new_seg
            
            # Создаем отсортированный список
            all_segments = [existing_segments[i] for i in sorted(existing_segments.keys())]
        
        # Подсчитываем статистику
        success_count = sum(1 for seg in all_segments if seg.get('success', False))
        total_count = len(all_segments)
        expected_total = self.load_phase1_data(video_name)
        expected_count = len(expected_total) if expected_total else 60
        
        analysis_complete = (total_count == expected_count and success_count == expected_count)
        
        return {
            'video_info': {
                'video_name': video_name,
                'total_segments': expected_count,
                'processed_segments': total_count,
                'success_segments': success_count,
                'analysis_complete': analysis_complete,
                'analysis_method': 'phase2_vllm_chunked_50s_overlap_15s',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'segments': all_segments
        }

    def save_phase2_results(self, video_name: str, results: List[Dict[str, Any]]) -> None:
        """Save Phase 2 VLLM analysis results with merge and backup"""
        if not results:
            print("No results to save")
            return
        
        output_path = Path('output') / f'{video_name}_phase2_vllm_analysis.json'
        backup_path = Path('output') / f'{video_name}_phase2_vllm_analysis.json.bak'
        
        # Создать backup существующего файла
        if output_path.exists():
            try:
                shutil.copy2(output_path, backup_path)
                print(f"✓ Backup created: {backup_path}")
            except Exception as e:
                print(f"⚠️ Failed to create backup: {e}")
        
        # Добавляем пометки успешности к новым результатам
        for result in results:
            # Успешность определяется отсутствием ошибок
            result['success'] = not ('parse_error' in result or 'error' in result)
        
        # Объединяем результаты
        merged_data = self.merge_results(video_name, results)
        
        # Сохраняем
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n✓ Phase 2 results saved to: {output_path}")
            
            # Показываем статистику
            video_info = merged_data['video_info']
            print(f"📊 Статистика:")
            print(f"   Всего сегментов: {video_info['total_segments']}")
            print(f"   Обработано: {video_info['processed_segments']}")
            print(f"   Успешно: {video_info['success_segments']}")
            print(f"   Анализ завершен: {'✅ ДА' if video_info['analysis_complete'] else '❌ НЕТ'}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            # Восстанавливаем из backup если есть
            if backup_path.exists():
                try:
                    shutil.copy2(backup_path, output_path)
                    print(f"✓ Restored from backup")
                except:
                    pass
        
        # Show sample results 
        print("\nSample VLLM analysis results:")
        for i, result in enumerate(results[:3]):
            print(f"\n  Segment {i+1} ({result['start_time']:.1f}s - {result['end_time']:.1f}s):")
            print(f"    Transcript: {result['transcript']}")
            print(f"    Analysis: {result['analysis']}")
        
        if len(results) > 3:
            print(f"\n  ... and {len(results) - 3} more segments")


def test_model_control():
    """Тестирование прямого управления моделью"""
    print("\n=== Тестирование управления моделью ===")
    
    analyzer = Phase2VLLMAnalyzer()
    
    # Получить текущий статус
    print("\n1. Получение статуса модели:")
    status = analyzer.get_model_status()
    if status:
        print(f"   Статус: {status.get('status')}")
        print(f"   Модель: {status.get('model')}")
        print(f"   PID: {status.get('pid')}")
        print(f"   Время работы: {status.get('uptime_seconds')} сек")
    else:
        print("   Статус недоступен")
    
    # Тест остановки модели
    print("\n2. Тест остановки модели:")
    analyzer.stop_model(reason="test_stop")
    
    # Ждем остановки
    if analyzer.wait_for_model_stop():
        print("   ✅ Модель успешно остановлена")
    else:
        print("   ❌ Модель не остановилась")
    
    # Ждем немного
    time.sleep(5)
    
    # Тест запуска модели
    print("\n3. Тест запуска модели:")
    analyzer.start_model(reason="test_start")
    
    # Ждем готовности
    if analyzer.wait_for_model_restart():
        print("   ✅ Модель успешно запущена")
    else:
        print("   ❌ Модель не запустилась")
    
    # Финальный статус
    print("\n4. Финальный статус:")
    status = analyzer.get_model_status()
    if status:
        print(f"   Статус: {status.get('status')}")
        print(f"   Модель: {status.get('model')}")
        print(f"   PID: {status.get('pid')}")
    else:
        print("   Статус недоступен")
    
    # Тест подключения
    print("\n5. Тест подключения к API:")
    if analyzer.test_vllm_connection():
        print("   ✅ API доступен")
    else:
        print("   ❌ API недоступен")
    
    print("\n=== Тестирование завершено ===")


def main():
    """Main function for Phase 2"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test-control":
        test_model_control()
        return

    print("=== Phase 2: VLLM Analysis ===")

    analyzer = Phase2VLLMAnalyzer()

    # Get videos from ChromaDB where phase1 completed and phase2 pending/processing
    videos_to_process = analyzer.db_manager.list_videos(
        status_filter={
            "phase1_status": "completed",
            "phase2_status": ["pending", "processing"]
        }
    )

    if not videos_to_process:
        print("✅ No videos to process - all Phase 2 complete or Phase 1 not ready!")
        return

    print(f"\n📋 Found {len(videos_to_process)} videos to process:")
    for video_info in videos_to_process:
        print(f"  - {video_info['video_name']}: phase2_status={video_info['phase2_status']}")

    # Process videos
    for video_info in videos_to_process:
        video_name = video_info["video_name"]

        print(f"\n*** Processing video: {video_name} ***")

        # Check if Phase 1 data exists
        phase1_file = Path('output') / f'{video_name}_phase1_data.pkl'
        if not phase1_file.exists():
            print(f"❌ Phase 1 data not found: {phase1_file}")
            continue

        analyzer.process_phase2_analysis(video_name)
        print(f"✅ Phase 2 completed for: {video_name}")

    print("\n=== Phase 2: All videos analyzed ===")


if __name__ == "__main__":
    main()