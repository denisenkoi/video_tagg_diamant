"""
VLLM Scene Director - "Режиссер" для умной нарезки видео на сцены
Использует мини-кадры (320x200) для анализа всего видео и определения логических границ сцен
"""

import cv2
import numpy as np
import json
import time
import base64
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from vllm_client import VLLMClient
from frame_extractor import extract_frames_with_timestamps
from config_manager import set_video_path, load_config


class VLLMSceneDirector:
    def __init__(self):
        """Initialize VLLM Scene Director"""
        self.client = VLLMClient()
        self.mini_resolution = (320, 200)  # Мини-разрешение для предпросмотра
        
        print("✓ VLLM Scene Director initialized")
        print(f"  Mini-frame resolution: {self.mini_resolution}")

    def load_transcription_segments(self, video_path: str) -> List[Dict[str, Any]]:
        """Load transcription segments from CSV"""
        video_name = Path(video_path).stem
        transcription_file = f"output/{video_name}_transcription.csv"
        
        if not Path(transcription_file).exists():
            print(f"WARNING: No transcription file found: {transcription_file}")
            return []
        
        try:
            df = pd.read_csv(transcription_file)
            segments = []
            
            for _, row in df.iterrows():
                segment = {
                    'start': float(row['start']),
                    'end': float(row['end']),
                    'text': str(row['text']),
                    'start_formatted': str(row['start_formatted']),
                    'end_formatted': str(row['end_formatted'])
                }
                segments.append(segment)
            
            print(f"✓ Loaded {len(segments)} transcription segments")
            return segments
            
        except Exception as e:
            print(f"ERROR loading transcription: {e}")
            return []

    def create_video_segments(self, transcription_segments: List[Dict], video_duration: float) -> List[Dict[str, Any]]:
        """
        Create video segments based on Whisper boundaries with 70-second limit
        Implements logic from scene_segmentation_logic.md
        """
        print("Creating video segments based on transcription boundaries...")
        
        segments = []
        current_start = 0.0
        max_segment_duration = 40.0
        min_silence_gap = 3.0
        
        if not transcription_segments:
            # No transcription - create fixed-time segments
            print("No transcription available, creating fixed-time segments")
            current_time = 0.0
            while current_time < video_duration:
                segment_end = min(current_time + max_segment_duration, video_duration)
                
                segment = {
                    'start': current_time,
                    'end': segment_end,
                    'type': 'silence',
                    'audio_text': None,
                    'duration': segment_end - current_time,
                    'has_speech': False
                }
                segments.append(segment)
                current_time = segment_end
            
            return segments
        
        # Process transcription segments
        i = 0
        while i < len(transcription_segments):
            segment_start = current_start
            segment_text_parts = []
            segment_end = current_start
            
            # Handle silence before first speech
            if i == 0 and transcription_segments[i]['start'] > 0:
                silence_duration = transcription_segments[i]['start']
                if silence_duration > max_segment_duration:
                    # Split long silence into chunks
                    silence_start = 0.0
                    while silence_start < silence_duration:
                        chunk_end = min(silence_start + max_segment_duration, silence_duration)
                        segment = {
                            'start': silence_start,
                            'end': chunk_end,
                            'type': 'silence',
                            'audio_text': None,
                            'duration': chunk_end - silence_start,
                            'has_speech': False
                        }
                        segments.append(segment)
                        silence_start = chunk_end - 5.0  # 5-second overlap
                    current_start = transcription_segments[i]['start']
                    continue
            
            # Collect speech segments that fit in max_segment_duration
            while i < len(transcription_segments):
                speech_segment = transcription_segments[i]
                potential_end = speech_segment['end']
                
                # Check if adding this segment would exceed limit
                if potential_end - segment_start > max_segment_duration:
                    # If this is the first segment and it's too long by itself
                    if len(segment_text_parts) == 0:
                        print(f"WARNING: Single speech segment too long ({potential_end - segment_start:.1f}s): {speech_segment['text'][:50]}...")
                        # Take it anyway - individual speech segments shouldn't be split
                        segment_text_parts.append(speech_segment['text'])
                        segment_end = potential_end
                        i += 1
                    break
                
                # Add segment
                segment_text_parts.append(speech_segment['text'])
                segment_end = potential_end
                i += 1
                
                # Check for silence gap to next segment
                if i < len(transcription_segments):
                    next_start = transcription_segments[i]['start']
                    silence_gap = next_start - segment_end
                    
                    # If big silence gap, handle it
                    if silence_gap > min_silence_gap:
                        # Check if we can fit the silence in current segment
                        if next_start - segment_start <= max_segment_duration:
                            # Include silence in current segment
                            segment_end = next_start
                        else:
                            # Close current segment, handle silence separately
                            break
            
            # Create speech segment
            if segment_text_parts:
                segment = {
                    'start': segment_start,
                    'end': segment_end,
                    'type': 'speech',
                    'audio_text': ' '.join(segment_text_parts),
                    'duration': segment_end - segment_start,
                    'has_speech': True
                }
                segments.append(segment)
                current_start = segment_end
            
            # Handle silence after speech (if exists)
            if i < len(transcription_segments):
                next_speech_start = transcription_segments[i]['start']
                if next_speech_start > current_start:
                    silence_duration = next_speech_start - current_start
                    
                    if silence_duration > max_segment_duration:
                        # Split long silence
                        silence_pos = current_start
                        while silence_pos < next_speech_start:
                            chunk_end = min(silence_pos + max_segment_duration, next_speech_start)
                            segment = {
                                'start': silence_pos,
                                'end': chunk_end,
                                'type': 'silence',
                                'audio_text': None,
                                'duration': chunk_end - silence_pos,
                                'has_speech': False
                            }
                            segments.append(segment)
                            silence_pos = chunk_end - 10.0  # overlap
                    else:
                        # Short silence - add as segment
                        segment = {
                            'start': current_start,
                            'end': next_speech_start,
                            'type': 'silence',
                            'audio_text': None,
                            'duration': silence_duration,
                            'has_speech': False
                        }
                        segments.append(segment)
                    
                    current_start = next_speech_start
        
        # Handle remaining video after last speech
        if current_start < video_duration:
            remaining_duration = video_duration - current_start
            segment = {
                'start': current_start,
                'end': video_duration,
                'type': 'silence',
                'audio_text': None,
                'duration': remaining_duration,
                'has_speech': False
            }
            segments.append(segment)
        
        print(f"✓ Created {len(segments)} video segments")
        
        # Show segment statistics
        durations = [seg['duration'] for seg in segments]
        speech_segments = [seg for seg in segments if seg['has_speech']]
        
        print(f"  Total segments: {len(segments)}")
        print(f"  Speech segments: {len(speech_segments)}")
        print(f"  Average duration: {np.mean(durations):.1f}s")
        print(f"  Max duration: {np.max(durations):.1f}s")
        print(f"  Min duration: {np.min(durations):.1f}s")
        
        # Check for segments over limit
        over_limit = [seg for seg in segments if seg['duration'] > max_segment_duration + 1]
        if over_limit:
            print(f"⚠️ {len(over_limit)} segments exceed {max_segment_duration}s limit:")
            for seg in over_limit[:3]:
                text_preview = seg['audio_text'][:50] + "..." if seg['audio_text'] else "silence"
                print(f"    {seg['start']:.1f}-{seg['end']:.1f} ({seg['duration']:.1f}s): {text_preview}")
        
        return segments

    def extract_mini_frames(self, video_path: str, segment: Dict) -> Tuple[List[np.ndarray], List[str]]:
        """Extract mini-resolution frames for a video segment"""
        start_time = segment['start']
        end_time = segment['end']
        duration = segment['duration']
        
        # Extract 1 frame per second for the segment
        fps = 1.0  # 1 frame per second
        
        print(f"  Extracting mini-frames: {start_time:.1f}s-{end_time:.1f}s ({fps} FPS)")
        
        # Extract frames using existing function
        frames, timestamps, original_fps = extract_frames_with_timestamps(video_path, fps)
        
        # Filter frames for this segment
        segment_frames = []
        segment_timestamps = []
        
        for frame, timestamp in zip(frames, timestamps):
            # Parse timestamp to seconds
            time_parts = timestamp.split(':')
            timestamp_seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
            
            if start_time <= timestamp_seconds <= end_time:
                # Resize to mini resolution
                mini_frame = cv2.resize(frame, self.mini_resolution)
                segment_frames.append(mini_frame)
                segment_timestamps.append(timestamp)
        
        print(f"    Got {len(segment_frames)} mini-frames for segment")
        return segment_frames, segment_timestamps

    def analyze_segment_with_vllm(self, mini_frames: List[np.ndarray], segment: Dict) -> str:
        """Analyze video segment using VLLM with ALL mini-frames"""
        if not mini_frames:
            return "Пустой сегмент без кадров"
        
        print(f"  Sending {len(mini_frames)} mini-frames to VLLM...")
        
        # Prepare context prompt
        audio_context = f"Аудио: \"{segment['audio_text']}\"" if segment['audio_text'] else "Аудио: тишина"
        
        prompt = f"""Проанализируй видео сегмент длительностью {segment['duration']:.1f} секунд.

{audio_context}

Тип сегмента: {segment['type']}

Вот {len(mini_frames)} кадров этого сегмента (каждую секунду). Опиши что происходит от начала к концу(description), как развивается действие(actions), какая общая атмосфера и контекст сцены(context).  На каких по счету кадрах происходят смены сцены, если происходят(scene_change)?

Переведи на русский разговор. Отметь, есть ли странности со словами? Требуется ли заново транскрибировать этот фрагмент видео?(re_transcribe_needs) И есть ли кадры, требующие более скурпулезного анализа в высоком разрешении. на которых могут быть важные детали(hi_res_nums)? 

Сделай ответ в формате json

"""
        
        # Send ALL mini-frames to VLLM
        description = self.analyze_multiple_frames_vllm(mini_frames, prompt)
        
        if description:
            print(f"  ✓ Got VLLM analysis: {description}")
            return description
        else:
            return "Не удалось получить анализ от VLLM"

    def analyze_multiple_frames_vllm(self, frames: List[np.ndarray], prompt: str) -> str:
        """Send multiple frames to VLLM in single request"""
        try:
            # Convert all frames to base64
            base64_images = []
            for frame in frames:
                # Resize frame if needed
                frame = self.client.resize_frame_if_needed(frame)
                
                # Encode to base64
                success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if success:
                    base64_string = base64.b64encode(buffer).decode('utf-8')
                    base64_images.append(base64_string)
            
            if not base64_images:
                return None
            
            # Prepare multi-image request payload
            content = [{"type": "text", "text": prompt}]
            
            # Add all images
            for base64_img in base64_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_img}"
                    }
                })
            
            payload = {
                "model": self.client.model_name,
                "messages": [{"role": "user", "content": content}],
                "temperature": self.client.temperature,
                "max_tokens": self.client.max_tokens
            }
            
            # Make request with retries
            for attempt in range(self.client.max_retries):
                try:
                    print(f"    VLLM multi-frame request attempt {attempt + 1}/{self.client.max_retries}")
                    
                    response = self.client.session.post(
                        f"{self.client.api_base}/v1/chat/completions",
                        json=payload,
                        timeout=self.client.timeout,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if "choices" in result and len(result["choices"]) > 0:
                            content = result["choices"][0]["message"]["content"]
                            if content and content.strip():
                                return content.strip()
                            else:
                                print(f"    Empty response on attempt {attempt + 1}")
                        else:
                            print(f"    No choices in response on attempt {attempt + 1}")
                    else:
                        print(f"    HTTP {response.status_code} on attempt {attempt + 1}: {response.text}")
                    
                    if attempt < self.client.max_retries - 1:
                        time.sleep(self.client.retry_delay)
                        
                except Exception as e:
                    print(f"    Network error on attempt {attempt + 1}: {e}")
                    if attempt < self.client.max_retries - 1:
                        time.sleep(self.client.retry_delay)
            
            print("    All VLLM multi-frame attempts failed")
            return None
            
        except Exception as e:
            print(f"    VLLM multi-frame analysis error: {e}")
            return None

    def process_video_segmentation(self, video_path: str) -> List[Dict[str, Any]]:
        """Main processing function"""
        print(f"\n=== VLLM Scene Director: {Path(video_path).name} ===")
        
        # Validate video file
        if not Path(video_path).exists():
            print(f"ERROR: Video file not found: {video_path}")
            return []
        
        set_video_path(video_path)
        
        # Get video duration (rough estimate)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 3600  # fallback
        cap.release()
        
        print(f"Video duration: {duration:.1f} seconds")
        
        # Test VLLM connection
        if not self.client.test_connection():
            print("ERROR: VLLM API not available")
            return []
        
        # Load transcription segments
        transcription_segments = self.load_transcription_segments(video_path)
        
        # Create video segments based on speech boundaries
        video_segments = self.create_video_segments(transcription_segments, duration)
        
        if not video_segments:
            print("ERROR: No video segments created")
            return []
        
        # Analyze ALL segments
        results = []
        
        for i, segment in enumerate(video_segments):
            print(f"\nAnalyzing segment {i+1}/{len(video_segments)}")
            print(f"  Time: {segment['start']:.1f}s - {segment['end']:.1f}s ({segment['duration']:.1f}s)")
            print(f"  Type: {segment['type']}")
            if segment['audio_text']:
                print(f"  Speech: {segment['audio_text']}")
            
            try:
                # Extract mini frames for this segment
                mini_frames, mini_timestamps = self.extract_mini_frames(video_path, segment)
                
                if mini_frames:
                    # Analyze with VLLM
                    analysis = self.analyze_segment_with_vllm(mini_frames, segment)
                    
                    # Store result
                    result = {
                        **segment,
                        'vllm_analysis': analysis,
                        'mini_frame_count': len(mini_frames)
                    }
                    results.append(result)
                else:
                    print("  ⚠️ No mini-frames extracted for segment")
                    
            except Exception as e:
                print(f"  ❌ Error analyzing segment: {e}")
                continue
        
        print(f"\n✓ Scene director analysis completed: {len(results)} segments analyzed")
        return results


def export_segmentation_results(results: List[Dict[str, Any]], video_path: str) -> None:
    """Export segmentation results to CSV"""
    if not results:
        print("No segmentation results to export")
        return
    
    video_name = Path(video_path).stem
    output_path = f"output/{video_name}_scene_segmentation.csv"
    
    print(f"Exporting scene segmentation to: {output_path}")
    
    # Prepare data for CSV
    csv_data = []
    for result in results:
        csv_row = {
            'start': result['start'],
            'end': result['end'],
            'duration': result['duration'],
            'type': result['type'],
            'has_speech': result['has_speech'],
            'audio_text': result['audio_text'] or '',
            'vllm_analysis': result['vllm_analysis'],
            'mini_frame_count': result['mini_frame_count']
        }
        csv_data.append(csv_row)
    
    # Save to CSV
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"✓ Scene segmentation results exported")
    print(f"Total segments: {len(results)}")


def main() -> None:
    """Main function for scene director module"""
    print("=== VLLM Scene Director Module Started ===")
    
    # Video to process
    video_files = ["video/news.mp4"]
    
    # Initialize director
    director = VLLMSceneDirector()
    
    for video_path in video_files:
        try:
            # Process video
            results = director.process_video_segmentation(video_path)
            
            if results:
                # Export results
                export_segmentation_results(results, video_path)
                print(f"✅ Successfully processed: {Path(video_path).name}")
            else:
                print(f"❌ No results for: {Path(video_path).name}")
                
        except Exception as e:
            print(f"❌ ERROR processing {video_path}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n=== VLLM Scene Director Module Completed ===")


if __name__ == "__main__":
    main()