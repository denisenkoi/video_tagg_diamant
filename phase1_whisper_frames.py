"""
Phase 1: Whisper Transcription + Frame Extraction
Независимый модуль для подготовки данных
"""

import cv2
import whisper
import torch
import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from datetime import datetime
from config_manager import set_video_path
from audio_extractor import extract_audio_segment, check_ffmpeg_available, cleanup_temp_files
from db_manager import ChromaDBManager
from db_manager_postgres import PostgresDBManager


class Phase1WhisperFrames:
    def __init__(self, db_manager: ChromaDBManager = None):
        """Initialize Phase 1 processor"""
        self.whisper_model = None
        self.db_manager = db_manager if db_manager else ChromaDBManager()

        # Параметры сегментации (такие же как в основном модуле)
        self.segment_duration = 50.0  # 50 секунд
        self.overlap_duration = 15.0  # 15 секунд перекрытие
        self.frames_per_segment = 6   # 6 кадров (каждые ~10 сек)

        print(f"✓ Phase1WhisperFrames initialized")
        print(f"  Segment duration: {self.segment_duration}s")
        print(f"  Overlap: {self.overlap_duration}s")
        print(f"  Frames per segment: {self.frames_per_segment}")
    
    def initialize_whisper(self):
        """Initialize Whisper model"""
        print("Initializing Whisper...")
        
        # Check ffmpeg availability
        if not check_ffmpeg_available():
            raise RuntimeError("ffmpeg not found - required for audio extraction")
        print("✓ ffmpeg available")
        
        # Initialize Whisper
        print("Loading Whisper large-v3...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = whisper.load_model("large-v3", device=device)
        print(f"✓ Whisper loaded on {device}")
    
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
    
    def create_segments(self, video_duration: float) -> List[Dict[str, Any]]:
        """Create segment definitions"""
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
        
        return segments
    
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
    
    def transcribe_segment(self, video_path: str, start_time: float, duration: float, language: str = None) -> str:
        """Transcribe audio segment using Whisper"""
        temp_audio = f"temp_segment_{start_time:.1f}_{duration:.1f}.wav"

        # Extract audio segment
        if not extract_audio_segment(video_path, start_time, duration, temp_audio):
            return ""

        # Transcribe with Whisper
        if language:
            result = self.whisper_model.transcribe(temp_audio, language=language)
        else:
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
    
    def process_video_phase1(self, video_path: str, language: str = None) -> None:
        """Process video - Phase 1: Whisper + Frames"""
        video_name = Path(video_path).stem

        print(f"\n=== PHASE 1: Processing {video_name} ===")
        if language:
            print(f"  Language: {language}")

        # Get video info first
        video_duration, video_width, video_height, fps = self.get_video_info(video_path)
        frame_width, frame_height = self.calculate_frame_resolution(video_width, video_height)

        # Calculate segments
        segments = self.create_segments(video_duration)

        # === Check ChromaDB status ===
        video_status = self.db_manager.get_video_status(video_name)

        if video_status:
            # Record exists - check status
            if video_status.get("phase1_status") == "completed":
                print(f"✓ Phase 1 already completed for {video_name}, skipping...")
                return

            if video_status.get("phase1_status") == "processing":
                # Check timeout
                start_time = datetime.fromisoformat(video_status["phase1_start_time"])
                timeout = video_duration * 2

                if (datetime.now() - start_time).total_seconds() < timeout:
                    print(f"⚠️ Phase 1 still processing for {video_name}, skipping...")
                    return
                else:
                    print(f"🔄 Phase 1 timeout detected for {video_name}, reprocessing...")
        else:
            # First run - create record
            print(f"➕ Creating new video_files record for {video_name}...")

        # Set status to "processing"
        self.db_manager.create_or_update_video_file(
            video_name=video_name,
            video_path=video_path,
            language=language or "unknown",
            video_duration=video_duration,
            total_segments=len(segments),
            width=video_width,
            height=video_height,
            fps=fps,
            phase1_status="processing",
            phase1_start_time=datetime.now().isoformat()
        )

        # Set video path
        set_video_path(video_path)

        print(f"Processing {len(segments)} segments...")
        
        # Process all segments
        segment_data = []
        
        for i, segment in enumerate(tqdm(segments, desc="Phase 1: Frames + Audio")):
            print(f"\nProcessing segment {i+1}/{len(segments)}")
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
            transcript = self.transcribe_segment(video_path, segment['start'], segment['duration'], language=language)
            if transcript:
                print(f"  ✓ Transcript: {transcript}")
            else:
                print(f"  ⚠️ Empty transcript for segment {i+1}")
            
            # Convert frames to base64 for Phase 2 compatibility
            base64_frames = []
            for frame in frames:
                success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if success:
                    import base64
                    base64_string = base64.b64encode(buffer).decode('utf-8')
                    base64_frames.append(base64_string)
            
            segment_data.append({
                'segment': segment,
                'index': i,
                'frames': frames,  # Keep original frames
                'frames_base64': base64_frames,  # Add base64 for Phase 2
                'transcript': transcript
            })
        
        print(f"\n✓ Phase 1 complete: {len(segment_data)} segments prepared")

        # Save results
        self.save_phase1_results(video_path, segment_data)

        # Update ChromaDB status to "completed"
        self.db_manager.update_video_status(
            video_name=video_name,
            phase1_status="completed",
            phase1_segments_created=len(segment_data)
        )

        # Clean up
        cleanup_temp_files()

        print(f"✓ Phase 1 data saved. ChromaDB status updated. Ready for Phase 2 VLLM analysis.")
    
    def save_phase1_results(self, video_path: str, segment_data: List[Dict]) -> None:
        """Save Phase 1 results for Phase 2"""
        video_name = Path(video_path).stem

        # Create output directory if it doesn't exist
        output_dir = Path('output')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as pickle for Python objects (frames are numpy arrays)
        pickle_path = output_dir / f'{video_name}_phase1_data.pkl'

        print(f"\nSaving Phase 1 results...")
        print(f"  Pickle file: {pickle_path}")

        with open(pickle_path, 'wb') as f:
            pickle.dump(segment_data, f)

        # Also save summary as JSON (without frames data)
        json_path = output_dir / f'{video_name}_phase1_summary.json'
        
        summary_data = {
            'video_path': str(video_path),
            'video_name': video_name,
            'total_segments': len(segment_data),
            'segments_info': []
        }
        
        for data in segment_data:
            summary_data['segments_info'].append({
                'index': data['index'],
                'start': data['segment']['start'],
                'end': data['segment']['end'],
                'duration': data['segment']['duration'],
                'frame_count': len(data['frames']),
                'has_transcript': bool(data['transcript']),
                'transcript_preview': data['transcript'][:100] if data['transcript'] else None
            })
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        print(f"  Summary JSON: {json_path}")
        print(f"  Total size: {len(segment_data)} segments saved")


def load_videos_config(config_path: str = "videos_config.json") -> list:
    """Load videos configuration from JSON file"""
    config_file = Path(config_path)

    if not config_file.exists():
        print(f"⚠️ Config file not found: {config_path}")
        print("Using default configuration...")
        return [{"path": "video/news.mp4", "language": None}]

    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    return config.get('videos', [])


def delete_output_files(video_name: str, output_dir: str = "output"):
    """Delete output JSON files for a video"""
    output_path = Path(output_dir)
    patterns = [
        f"{video_name}_phase1_summary.json",
        f"{video_name}_phase2_vllm_analysis.json",
        f"{video_name}_phase3_faces.json"
    ]
    deleted = []
    for pattern in patterns:
        file_path = output_path / pattern
        if file_path.exists():
            file_path.unlink()
            deleted.append(pattern)
    if deleted:
        print(f"🗑️ Deleted output files: {', '.join(deleted)}")


def main():
    """Main function for Phase 1"""
    print("=== Phase 1: Whisper Transcription + Frame Extraction ===")

    # Load videos configuration
    videos_config = load_videos_config("videos_config.json")
    print(f"\n📋 Loaded {len(videos_config)} videos from config")

    # Initialize PostgresDBManager for force operations
    pg_db = None

    processor = Phase1WhisperFrames()

    # Initialize Whisper
    processor.initialize_whisper()

    # Process videos
    for video_config in tqdm(videos_config, desc="Processing videos"):
        video_path = video_config.get('path')
        language = video_config.get('language')
        force = video_config.get('force', False)
        video_name = video_config.get('name') or Path(video_path).stem

        if not video_path:
            print(f"⚠️ Skipping video config without path: {video_config}")
            continue

        if not Path(video_path).exists():
            print(f"❌ Video not found: {video_path}")
            continue

        # Handle force reindexing
        if force:
            print(f"\n🔄 FORCE reindex requested for {video_name}")
            if pg_db is None:
                pg_db = PostgresDBManager()
            # Delete all data from DB
            pg_db.delete_video_data(video_name)
            # Delete output JSON files
            delete_output_files(video_name)

        processor.process_video_phase1(video_path, language=language)
        print(f"✅ Phase 1 completed for: {Path(video_path).name}")

    print("\n=== Phase 1: All videos processed ===")
    print("Ready to run Phase 2 VLLM analysis!")


if __name__ == "__main__":
    main()