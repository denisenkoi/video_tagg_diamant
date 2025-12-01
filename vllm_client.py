"""
VLLM Client for video frame analysis using Qwen 2.5 VL model
Простой клиент для анализа кадров через OpenAI-совместимый API
"""

import requests
import base64
import cv2
import numpy as np
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path


class VLLMClient:
    def __init__(self, config_path: str = "vllm_config.json"):
        """Initialize VLLM client with configuration"""
        self.config = self.load_config(config_path)
        self.session = requests.Session()
        
        # Основные настройки
        self.api_base = self.config["vllm_settings"]["api_base"]
        self.model_name = self.config["vllm_settings"]["model_name"]
        self.temperature = self.config["vllm_settings"]["temperature"]
        self.max_tokens = self.config["vllm_settings"]["max_tokens"]
        self.timeout = self.config["vllm_settings"]["timeout"]
        
        # Настройки ретраев
        self.max_retries = self.config["retry_settings"]["max_retries"]
        self.retry_delay = self.config["retry_settings"]["retry_delay"]
        
        print(f"✓ VLLM Client initialized")
        print(f"  API: {self.api_base}")
        print(f"  Model: {self.model_name}")
        print(f"  Max retries: {self.max_retries}")

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def resize_frame_if_needed(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame if larger than max_resolution"""
        max_res = self.config["frame_processing"]["max_resolution"]
        height, width = frame.shape[:2]
        
        if height > max_res or width > max_res:
            # Сохраняем пропорции
            if height > width:
                new_height = max_res
                new_width = int(width * (max_res / height))
            else:
                new_width = max_res
                new_height = int(height * (max_res / width))
            
            resized = cv2.resize(frame, (new_width, new_height))
            print(f"  Resized frame: {width}x{height} -> {new_width}x{new_height}")
            return resized
        
        return frame

    def frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert frame to base64 string"""
        # Resize if needed
        frame = self.resize_frame_if_needed(frame)
        
        # Encode to JPEG
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise ValueError("Failed to encode frame to JPEG")
        
        # Convert to base64
        base64_string = base64.b64encode(buffer).decode('utf-8')
        return base64_string

    def analyze_frame(self, frame: np.ndarray, context: Optional[str] = None) -> Optional[str]:
        """Analyze single frame using VLLM"""
        
        try:
            # Convert frame to base64
            base64_image = self.frame_to_base64(frame)
            
            # Prepare prompt
            prompt = self.config["prompts"]["simple_analysis"]
            if context:
                prompt = f"{prompt}\n\nДополнительный контекст: {context}"
            
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            # Make request with retries
            for attempt in range(self.max_retries):
                try:
                    print(f"  VLLM request attempt {attempt + 1}/{self.max_retries}")
                    
                    response = self.session.post(
                        f"{self.api_base}/v1/chat/completions",
                        json=payload,
                        timeout=self.timeout,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if "choices" in result and len(result["choices"]) > 0:
                            content = result["choices"][0]["message"]["content"]
                            if content and content.strip():
                                return content.strip()
                            else:
                                print(f"  Empty response on attempt {attempt + 1}")
                        else:
                            print(f"  No choices in response on attempt {attempt + 1}")
                    else:
                        print(f"  HTTP {response.status_code} on attempt {attempt + 1}: {response.text}")
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        
                except requests.exceptions.RequestException as e:
                    print(f"  Network error on attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
            
            print("  All VLLM attempts failed")
            return None
            
        except Exception as e:
            print(f"  VLLM analysis error: {e}")
            return None

    def test_connection(self) -> bool:
        """Test connection to VLLM API"""
        try:
            response = self.session.get(f"{self.api_base}/v1/models", timeout=5)
            if response.status_code == 200:
                print("✓ VLLM API connection successful")
                return True
            else:
                print(f"✗ VLLM API returned {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ VLLM API connection failed: {e}")
            return False


def analyze_video_frames_vllm(frames: List[np.ndarray], timestamps: List[str], 
                             all_analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Analyze video frames using VLLM
    
    Args:
        frames: List of video frames
        timestamps: Corresponding timestamps
        all_analysis_results: Results from other analyzers for context
        
    Returns:
        List of VLLM analysis results
    """
    
    # Load frame processing config
    with open("vllm_config.json", 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    max_minutes = config["frame_processing"]["max_video_minutes"]
    
    print(f"VLLM Analysis: max_duration={max_minutes}min")
    
    # Initialize client
    client = VLLMClient()
    
    # Test connection
    if not client.test_connection():
        print("VLLM API not available - skipping analysis")
        return []
    
    # Просто анализируем все извлеченные кадры (они уже каждые N секунд)
    total_frames = len(frames)
    
    # Limit to max_minutes (frames уже извлечены с нужным интервалом)
    interval_seconds = config["frame_processing"]["interval_seconds"] 
    max_frames = max_minutes * 60 / interval_seconds  # кадров за max_minutes
    
    if total_frames > max_frames:
        frames = frames[:int(max_frames)]
        timestamps = timestamps[:int(max_frames)]
        print(f"Limited to first {max_minutes} minutes: {len(frames)} frames")
    
    print(f"Analyzing all {len(frames)} extracted frames")
    
    results = []
    
    # Анализируем все кадры по порядку
    for i, (frame, timestamp) in enumerate(zip(frames, timestamps)):
        print(f"Analyzing frame {i+1}/{len(frames)} at {timestamp}")
        
        # Prepare context from other analysis results
        context = prepare_context_for_frame(timestamp, all_analysis_results)
        
        # Analyze frame
        description = client.analyze_frame(frame, context)
        
        if description:
            result = {
                'timestamp': timestamp,
                'frame_index': i,
                'description': description,
                'context_provided': bool(context)
            }
            results.append(result)
            print(f"  ✓ Got description: {description}")
        else:
            print(f"  ✗ No description received")
    
    print(f"VLLM analysis completed: {len(results)} descriptions generated")
    return results


def prepare_context_for_frame(timestamp: str, all_analysis_results: Dict[str, Any]) -> Optional[str]:
    """Prepare context information for VLLM analysis"""
    
    context_parts = []
    
    # Add detected objects if available
    if 'objects' in all_analysis_results:
        for obj_data in all_analysis_results['objects']:
            if obj_data['timestamp'] == timestamp and obj_data.get('objects'):
                objects = obj_data['objects'].split(',')[:5]  # First 5 objects
                context_parts.append(f"Объекты: {', '.join(objects)}")
    
    # Add detected faces if available  
    if 'faces' in all_analysis_results:
        for face_data in all_analysis_results['faces']:
            if face_data['timestamp'] == timestamp and face_data.get('persons'):
                persons = face_data['persons'].split(',')[:3]  # First 3 persons
                context_parts.append(f"Люди: {', '.join(persons)}")
    
    # Add detected text if available
    if 'texts_ocr' in all_analysis_results:
        for text_data in all_analysis_results['texts_ocr']:
            if text_data['timestamp'] == timestamp and text_data.get('texts'):
                texts = text_data['texts'].split('|')[:3]  # First 3 texts
                context_parts.append(f"Текст: {', '.join(texts)}")
    
    return "; ".join(context_parts) if context_parts else None


if __name__ == "__main__":
    # Test the client
    client = VLLMClient()
    client.test_connection()