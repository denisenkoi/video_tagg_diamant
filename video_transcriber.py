"""
Video Transcription Module - Отдельный модуль для транскрипции видео через Whisper
Запускается независимо, читает видео, транскрибирует аудио, сохраняет результаты
"""

import whisper
import torch
import time
from pathlib import Path
from typing import List, Dict, Any
from config_manager import get_video_path, set_video_path, load_config
from result_exporter import create_output_filename, export_transcription_csv


def initialize_whisper_model():
    """Initialize Whisper model for audio transcription"""
    print("Initializing Whisper model...")
    
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Whisper using device: {device}")
    
    if device == "cuda":
        print(f"GPU detected: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    
    # Load Whisper model (can be: tiny, base, small, medium, large-v3)
    model_size = "large-v3"  # Best accuracy
    model = whisper.load_model(model_size, device=device)
    
    print(f"✓ Whisper model '{model_size}' loaded successfully")
    print(f"Model capabilities:")
    print(f"  - Languages: 99 languages supported")
    print(f"  - Word-level timestamps: Yes")
    print(f"  - Voice activity detection: Yes")
    
    return model


def transcribe_video_file(video_path: str, model) -> List[Dict[str, Any]]:
    """
    Transcribe video file using Whisper with detailed timestamps
    
    Args:
        video_path: Path to video file
        model: Whisper model instance
        
    Returns:
        List of transcription segments with timestamps
    """
    print(f"Starting transcription of: {Path(video_path).name}")
    
    # Validate video file
    if not Path(video_path).exists():
        print(f"ERROR: Video file not found: {video_path}")
        return []
    
    start_time = time.time()
    
    # Transcribe with default settings - large-v3 model
    result = model.transcribe(video_path)
    
    transcription_time = time.time() - start_time
    
    print(f"Transcription completed in {transcription_time:.1f} seconds")
    print(f"Detected language: {result.get('language', 'unknown')}")
    print(f"Total segments: {len(result['segments'])}")
    
    # Process segments
    processed_segments = process_transcription_segments(result['segments'])
    
    print(f"Processed {len(processed_segments)} segments")
    
    # Calculate total duration
    if processed_segments:
        total_duration = processed_segments[-1]['end'] - processed_segments[0]['start']
        print(f"Total transcribed duration: {total_duration:.1f} seconds")
        
        # Show segment statistics
        segment_lengths = [seg['end'] - seg['start'] for seg in processed_segments]
        avg_length = sum(segment_lengths) / len(segment_lengths)
        max_length = max(segment_lengths)
        
        print(f"Average segment length: {avg_length:.1f} seconds")
        print(f"Longest segment: {max_length:.1f} seconds")
        
        # Check for long segments (>60 seconds)
        long_segments = [seg for seg in processed_segments if seg['end'] - seg['start'] > 60]
        if long_segments:
            print(f"⚠️ Found {len(long_segments)} segments longer than 60 seconds:")
            for seg in long_segments[:3]:  # Show first 3
                duration = seg['end'] - seg['start']
                text_preview = seg['text'][:50] + "..." if len(seg['text']) > 50 else seg['text']
                print(f"  {seg['start_formatted']}-{seg['end_formatted']} ({duration:.1f}s): {text_preview}")
    
    return processed_segments


def process_transcription_segments(segments: List[Dict], word_timestamps: bool = True) -> List[Dict[str, Any]]:
    """
    Process Whisper transcription segments into standardized format
    
    Args:
        segments: Raw segments from Whisper
        word_timestamps: Include word-level timestamps
        
    Returns:
        Processed segments with consistent formatting
    """
    processed = []
    
    for i, segment in enumerate(segments):
        # Extract basic info
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text'].strip()
        
        if not text:  # Skip empty segments
            continue
        
        # Format timestamps
        start_formatted = format_timestamp(start_time)
        end_formatted = format_timestamp(end_time)
        
        # Process words if available
        words_data = []
        if word_timestamps and 'words' in segment:
            for word in segment['words']:
                word_info = {
                    'word': word['word'],
                    'start': word['start'],
                    'end': word['end'],
                    'probability': word.get('probability', 0.0)
                }
                words_data.append(word_info)
        
        # Create processed segment
        processed_segment = {
            'id': i,
            'start': start_time,
            'end': end_time,
            'start_formatted': start_formatted,
            'end_formatted': end_formatted,
            'text': text,
            'duration': end_time - start_time,
            'word_count': len(text.split()),
            'words': words_data if word_timestamps else []
        }
        
        processed.append(processed_segment)
    
    return processed


def clean_repetitive_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean segments with repetitive text patterns"""
    cleaned = []
    
    for segment in segments:
        text = segment['text'].strip()
        
        # Skip empty segments
        if not text:
            continue
            
        # Check for repetitive patterns
        words = text.split()
        if len(words) > 5:
            # Check if more than 70% of words are the same
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Find most common word
            max_count = max(word_counts.values()) if word_counts else 0
            repetition_ratio = max_count / len(words) if words else 0
            
            if repetition_ratio > 0.7:
                print(f"  ⚠️ Skipping repetitive segment: '{text[:50]}...'")
                continue
        
        # Check for exact duplicates with previous segment
        if cleaned and cleaned[-1]['text'] == text:
            print(f"  ⚠️ Skipping duplicate segment: '{text[:50]}...'")
            continue
            
        cleaned.append(segment)
    
    return cleaned


def format_timestamp(seconds: float) -> str:
    """Format seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def export_transcription_results(segments: List[Dict[str, Any]], video_path: str) -> None:
    """Export transcription results to CSV"""
    if not segments:
        print("No transcription segments to export")
        return
    
    # Create output filename
    output_path = create_output_filename(video_path, 'transcription')
    
    print(f"Exporting transcription to: {output_path}")
    
    # Export using existing CSV exporter
    export_transcription_csv(segments, output_path)
    
    print(f"Transcription results exported successfully")
    print(f"Total segments: {len(segments)}")
    
    # Show sample results
    print("\nSample transcription segments:")
    for i, segment in enumerate(segments[:3]):
        text_preview = segment['text'][:80] + "..." if len(segment['text']) > 80 else segment['text']
        print(f"  {i+1}. {segment['start_formatted']}-{segment['end_formatted']}: {text_preview}")
    
    if len(segments) > 3:
        print(f"  ... and {len(segments) - 3} more segments")


def process_single_video_transcription(video_path: str) -> None:
    """Process single video file for transcription"""
    print(f"\n=== Video Transcription: {Path(video_path).name} ===")
    
    # Set video path
    set_video_path(video_path)
    
    # Validate video file
    if not Path(video_path).exists():
        print(f"ERROR: Video file not found: {video_path}")
        return
    
    print(f"Video file confirmed: {video_path}")
    
    # Initialize Whisper model
    model = initialize_whisper_model()
    
    # Transcribe video
    print("Starting video transcription...")
    segments = transcribe_video_file(video_path, model)
    
    if not segments:
        print("No transcription results generated")
        return
    
    # Export results
    export_transcription_results(segments, video_path)
    
    print(f"=== Video Transcription Completed: {Path(video_path).name} ===\n")


def main() -> None:
    """Main function for video transcription module"""
    print("=== Video Transcription Module Started ===")
    
    # List of videos to process
    video_files = [
        "video/news.mp4",
        # "video/AlmaAta.mp4",
        # "video/SovKz.mp4"
    ]
    
    total_videos = len(video_files)
    successful_videos = 0
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n*** Processing video {i}/{total_videos} ***")
        
        try:
            process_single_video_transcription(video_path)
            successful_videos += 1
            print(f"✅ Successfully processed: {Path(video_path).name}")
            
        except Exception as e:
            print(f"❌ ERROR processing {video_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n=== Video Transcription Module Completed ===")
    print(f"Successfully processed: {successful_videos}/{total_videos} videos")
    print(f"Failed: {total_videos - successful_videos} videos")


if __name__ == "__main__":
    main()