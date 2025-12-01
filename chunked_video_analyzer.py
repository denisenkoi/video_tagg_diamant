"""
Chunked Video Analyzer - Прототип v2
Синхронизированный анализ 50-сек блоков с контекстом предыдущего сегмента
"""

import cv2
import whisper
import torch
import numpy as np
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from vllm_client import VLLMClient
from config_manager import set_video_path
from result_exporter import create_output_filename
from audio_extractor import extract_audio_segment, check_ffmpeg_available, cleanup_temp_files


class ChunkedVideoAnalyzer:
    def __init__(self, config_path: str = "vllm_config.json"):
        """Initialize the chunked analyzer"""
        self.config = self.load_config(config_path)
        self.whisper_model = None
        self.vllm_client = None
        
        # Параметры сегментации 
        self.segment_duration = 50.0  # 50 секунд
        self.overlap_duration = 15.0  # 15 секунд перекрытие
        self.frames_per_segment = 6   # 6 кадров (каждые ~10 сек)
        
        print(f"✓ ChunkedVideoAnalyzer initialized")
        print(f"  Segment duration: {self.segment_duration}s")
        print(f"  Overlap: {self.overlap_duration}s") 
        print(f"  Frames per segment: {self.frames_per_segment}")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def initialize_models(self):
        """Initialize Whisper and VLLM models"""
        print("Initializing models...")
        
        # Check ffmpeg availability
        if not check_ffmpeg_available():
            raise RuntimeError("ffmpeg not found - required for audio extraction")
        print("✓ ffmpeg available")
        
        # Initialize Whisper
        print("Loading Whisper large-v3...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = whisper.load_model("large-v3", device=device)
        print(f"✓ Whisper loaded on {device}")
        
        # Initialize VLLM client
        print("Connecting to VLLM...")
        self.vllm_client = VLLMClient()
        if not self.vllm_client.test_connection():
            raise ConnectionError("VLLM API not available")
        print("✓ VLLM connected")
    
    def get_video_info(self, video_path: str) -> Tuple[float, int, int, float]:
        """Get video information"""
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        
        cap.release()
        
        print(f"Video info: {width}x{height}, {fps} FPS, {duration:.1f}s")
        return duration, width, height, fps
    
    def calculate_frame_resolution(self, video_width: int, video_height: int) -> Tuple[int, int]:
        """Calculate proportional frame resolution with 320 as base"""
        if video_width > video_height:  # landscape
            frame_width = 320
            frame_height = int(320 * video_height / video_width)
        else:  # portrait
            frame_height = 320
            frame_width = int(320 * video_width / video_height)
        
        # Ensure even dimensions for video encoding
        frame_width = frame_width + (frame_width % 2)
        frame_height = frame_height + (frame_height % 2)
        
        print(f"Frame resolution: {frame_width}x{frame_height}")
        return frame_width, frame_height
    
    def extract_segment_frames(self, video_path: str, start_time: float, 
                              duration: float, target_width: int, target_height: int) -> List[np.ndarray]:
        """Extract frames from video segment"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames = []
        frame_interval = duration / self.frames_per_segment  # интервал между кадрами
        
        for i in tqdm(range(self.frames_per_segment), desc="    Extracting frames", leave=False):
            timestamp = start_time + i * frame_interval
            frame_number = int(timestamp * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                # Resize frame proportionally 
                resized_frame = cv2.resize(frame, (target_width, target_height))
                frames.append(resized_frame)
        
        cap.release()
        return frames
    
    def transcribe_segment(self, video_path: str, start_time: float, duration: float) -> str:
        """Transcribe audio segment using Whisper"""
        temp_audio = f"temp_segment_{start_time:.1f}_{duration:.1f}.wav"
        
        # Extract audio segment
        if not extract_audio_segment(video_path, start_time, duration, temp_audio):
            return ""
        
        # Transcribe with Whisper
        result = self.whisper_model.transcribe(temp_audio)
        
        # Extract text from segments
        text_parts = []
        for segment in result.get('segments', []):
            text = segment.get('text', '').strip()
            if text:
                text_parts.append(text)
        
        # Clean up temp file
        Path(temp_audio).unlink(missing_ok=True)
        
        return ' '.join(text_parts)
    
    def create_vllm_prompt(self, previous_context: str, audio_transcript: str) -> str:
        """Create structured prompt for VLLM"""
        
        base_prompt = """Анализируй 6 кадров из 50-секундного видео сегмента.

ВАЖНО: Данный сегмент имеет 15-секундное перекрытие с предыдущим сегментом.

КОНТЕКСТ ПРЕДЫДУЩЕГО СЕГМЕНТА:
{previous_context}

АУДИО ТРАНСКРИПТ ТЕКУЩЕГО СЕГМЕНТА:
{audio_transcript}

ЗАДАЧИ:
1. Опиши что происходит в кадрах (учитывай перекрытие с предыдущим сегментом)
2. Переведи все диалоги на русский язык 
3. Выдели 5-10 ключевых слов для поиска
4. Отметь смену сцен, если есть
5. Определи новую информацию (исключи дубли из 15-сек перекрытия)

Отвечай в формате JSON:
{{
  "description": "подробное описание происходящего",
  "dialogue_translation": "перевод диалогов на русский",
  "keywords": ["ключевое", "слово", "для", "поиска"],
  "scene_change": true/false,
  "new_information": "что нового по сравнению с предыдущим сегментом",
  "confidence": "высокая/средняя/низкая"
}}"""
        
        return base_prompt.format(
            previous_context=previous_context or "Нет (это первый сегмент)",
            audio_transcript=audio_transcript or "Нет транскрипции"
        )
    
    def analyze_segment(self, frames: List[np.ndarray], prompt: str) -> Optional[str]:
        """Analyze segment frames with VLLM"""
        # Convert all frames to base64
        base64_images = []
        for frame in tqdm(frames, desc="    Converting frames to base64", leave=False):
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                import base64
                base64_string = base64.b64encode(buffer).decode('utf-8')
                base64_images.append(base64_string)
        
        if not base64_images:
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
                    } for img in base64_images
                ]
            }
        ]
        
        payload = {
            "model": self.vllm_client.model_name,
            "messages": messages,
            "temperature": self.vllm_client.temperature,
            "max_tokens": self.vllm_client.max_tokens
        }
        
        # Make request
        response = self.vllm_client.session.post(
            f"{self.vllm_client.api_base}/v1/chat/completions",
            json=payload,
            timeout=self.vllm_client.timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                if content:
                    # Clean up response: trim whitespace and newlines
                    cleaned_content = content.strip()
                    if cleaned_content:
                        return cleaned_content
        
        return None
    
    def process_video(self, video_path: str) -> List[Dict[str, Any]]:
        """Process entire video in chunks"""
        print(f"\n=== Processing video: {Path(video_path).name} ===")
        
        # Set video path
        set_video_path(video_path)
        
        # Get video info
        video_duration, video_width, video_height, fps = self.get_video_info(video_path)
        frame_width, frame_height = self.calculate_frame_resolution(video_width, video_height)
        
        # Calculate segments
        segments = []
        current_start = 0.0
        segment_index = 0
        
        while current_start < video_duration:
            segment_end = min(current_start + self.segment_duration, video_duration)
            actual_duration = segment_end - current_start
            
            segments.append({
                'index': segment_index,
                'start': current_start,
                'end': segment_end, 
                'duration': actual_duration
            })
            
            # Next segment starts with overlap
            current_start += (self.segment_duration - self.overlap_duration)
            segment_index += 1
        
        print(f"Created {len(segments)} segments")
        
        # PHASE 1: Extract all frames and transcribe all audio first
        print("\n=== PHASE 1: Extracting frames and transcribing audio ===")
        segment_data = []
        
        for i, segment in enumerate(tqdm(segments, desc="Phase 1: Frames + Audio")):
            print(f"\nPreparing segment {i+1}/{len(segments)}")
            print(f"  Time: {segment['start']:.1f}s - {segment['end']:.1f}s ({segment['duration']:.1f}s)")
            
            # Extract frames
            print(f"  Extracting {self.frames_per_segment} frames...")
            frames = self.extract_segment_frames(
                video_path, segment['start'], segment['duration'],
                frame_width, frame_height
            )
            
            if not frames:
                print(f"  ❌ No frames extracted for segment {i+1}")
                continue
            
            print(f"  ✓ Extracted {len(frames)} frames")
            
            # Transcribe audio
            print(f"  Transcribing audio...")
            transcript = self.transcribe_segment(video_path, segment['start'], segment['duration'])
            if transcript:
                print(f"  ✓ Transcript: {transcript}")
            else:
                print(f"  ⚠️ Empty transcript for segment {i+1}")
            
            segment_data.append({
                'segment': segment,
                'index': i,
                'frames': frames,
                'transcript': transcript
            })
        
        print(f"\n✓ Phase 1 complete: {len(segment_data)} segments prepared")
        
        # PHASE 2: VLLM analysis
        print("\n=== PHASE 2: VLLM Analysis ===")
        results = []
        previous_description = ""
        
        for data in tqdm(segment_data, desc="Phase 2: VLLM Analysis"):
            i = data['index']
            segment = data['segment']
            frames = data['frames']
            transcript = data['transcript']
            
            print(f"\nAnalyzing segment {i+1}/{len(segments)} with VLLM")
            print(f"  Time: {segment['start']:.1f}s - {segment['end']:.1f}s")
            
            # Create prompt
            prompt = self.create_vllm_prompt(previous_description, transcript)
            
            # Analyze with VLLM (with retry logic)
            analysis = None
            max_retries = 3
            
            for attempt in range(max_retries):
                print(f"  VLLM attempt {attempt+1}/{max_retries}...")
                analysis = self.analyze_segment(frames, prompt)
                
                if analysis and analysis.strip():
                    print(f"  ✓ VLLM analysis completed")
                    # Debug: show what VLLM returned
                    print(f"  Raw VLLM response: {analysis}")
                    break
                else:
                    print(f"  ⚠️ Empty/invalid VLLM response on attempt {attempt+1}")
                    if attempt < max_retries - 1:
                        print(f"    Retrying...")
            
            if analysis and analysis.strip():
                # Parse JSON
                analysis_json = json.loads(analysis)
                previous_description = analysis_json.get('description', '')
                
                result = {
                    'segment_index': i,
                    'start_time': segment['start'],
                    'end_time': segment['end'],
                    'duration': segment['duration'],
                    'transcript': transcript,
                    'analysis': analysis,
                    'frame_count': len(frames)
                }
                results.append(result)
            else:
                print(f"  ❌ VLLM analysis failed for segment {i+1} after {max_retries} attempts")
        
        print(f"\n✓ Phase 2 complete: {len(results)}/{len(segments)} segments analyzed successfully")
        return results
    
    def export_results(self, results: List[Dict[str, Any]], video_path: str):
        """Export results to JSON file"""
        if not results:
            print("No results to export")
            return
        
        # Create JSON output path manually since create_output_filename only does CSV
        video_name = Path(video_path).stem
        output_path = Path('output') / f'{video_name}_chunked_analysis.json'
        
        export_data = {
            'video_info': {
                'path': str(Path(video_path).name),
                'total_segments': len(results),
                'analysis_method': 'chunked_50s_overlap_15s'
            },
            'segments': results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults exported to: {output_path}")
        
        # Show sample results 
        print("\nSample analysis results:")
        for i, result in enumerate(results[:3]):
            print(f"\n  Segment {i+1} ({result['start_time']:.1f}s - {result['end_time']:.1f}s):")
            print(f"    Transcript: {result['transcript']}")
            print(f"    Analysis: {result['analysis']}")
        
        if len(results) > 3:
            print(f"\n  ... and {len(results) - 3} more segments")


def main():
    """Main function"""
    print("=== Chunked Video Analyzer v2 ===")
    
    # Video to analyze
    video_files = [
        "video/news.mp4"
    ]
    
    analyzer = ChunkedVideoAnalyzer()
    
    # Initialize models
    try:
        analyzer.initialize_models()
    except Exception as e:
        print(f"Failed to initialize models: {e}")
        return
    
    # Process videos
    for video_path in tqdm(video_files, desc="Processing videos"):
        if not Path(video_path).exists():
            print(f"Video not found: {video_path}")
            continue
        
        try:
            results = analyzer.process_video(video_path)
            analyzer.export_results(results, video_path)
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Clean up any remaining temp files
    cleanup_temp_files()
    print("\n=== Chunked Video Analysis Complete ===")


if __name__ == "__main__":
    main()