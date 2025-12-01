"""
VLLM Video Analysis Module - Отдельный модуль для анализа видео через Vision-Language модель
Запускается независимо, читает видео, анализирует кадры, сохраняет результаты
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
from vllm_client import VLLMClient, analyze_video_frames_vllm
from frame_extractor import extract_frames_with_timestamps
from config_manager import get_video_path, set_video_path, load_config
from result_exporter import create_output_filename, export_objects_csv


def load_context_from_other_modules(video_name: str) -> Dict[str, Any]:
    """
    Загружает результаты других модулей как контекст для VLLM
    """
    context_data = {}
    
    try:
        # Загружаем результаты объектов
        objects_file = f"output/{video_name}_objects.csv"
        if Path(objects_file).exists():
            print(f"Loading objects context from {objects_file}")
            # Простая загрузка - можно улучшить
            context_data['objects'] = []
    
        # Загружаем результаты лиц
        faces_file = f"output/{video_name}_faces.csv" 
        if Path(faces_file).exists():
            print(f"Loading faces context from {faces_file}")
            context_data['faces'] = []
            
        # Загружаем результаты OCR
        ocr_file = f"output/{video_name}_texts_ocr.csv"
        if Path(ocr_file).exists():
            print(f"Loading OCR context from {ocr_file}")
            context_data['texts_ocr'] = []
            
        print(f"Loaded context from {len(context_data)} modules")
        
    except Exception as e:
        print(f"Warning: Could not load context data: {e}")
        
    return context_data


def process_single_video_vllm(video_path: str) -> None:
    """Process single video file for VLLM analysis"""
    print(f"\n=== VLLM Analysis: {Path(video_path).name} ===")
    
    # Set video path
    set_video_path(video_path)
    
    # Validate video file
    if not Path(video_path).exists():
        print(f"ERROR: Video file not found: {video_path}")
        return
    
    print(f"Video file confirmed: {video_path}")
    
    # Load VLLM-specific configuration
    with open("vllm_config.json", 'r', encoding='utf-8') as f:
        vllm_config = json.load(f)
    
    interval_seconds = vllm_config["frame_processing"]["interval_seconds"]
    
    # Extract frames at interval_seconds rate (каждые N секунд = 1/N FPS)
    fps = 1.0 / interval_seconds  # каждые 2 сек = 0.5 FPS
    
    # Extract frames
    print(f"Extracting frames every {interval_seconds} seconds ({fps} FPS)...")
    frames, timestamps, original_fps = extract_frames_with_timestamps(video_path, fps)
    print(f"Extracted {len(frames)} frames (every {interval_seconds}s from original {original_fps} FPS)")
    
    max_minutes = vllm_config["frame_processing"]["max_video_minutes"]
    
    # Limit frames to max_minutes
    max_frames = int(max_minutes * 60 * fps)
    if len(frames) > max_frames:
        frames = frames[:max_frames]
        timestamps = timestamps[:max_frames]
        print(f"Limited to first {max_minutes} minutes: {len(frames)} frames")
    
    # Load context from other modules
    video_name = Path(video_path).stem
    context_data = load_context_from_other_modules(video_name)
    
    # Initialize VLLM client
    print("Initializing VLLM client...")
    client = VLLMClient()
    
    # Test connection
    if not client.test_connection():
        print("ERROR: VLLM API not available")
        return
    
    # Process frames with VLLM
    print("Starting VLLM video analysis...")
    results = analyze_video_frames_vllm(frames, timestamps, context_data)
    
    if not results:
        print("No VLLM analysis results generated")
        return
    
    # Export results
    export_vllm_results(results, video_path)
    
    print(f"=== VLLM Analysis Completed: {Path(video_path).name} ===\n")


def export_vllm_results(results: List[Dict[str, Any]], video_path: str) -> None:
    """Export VLLM analysis results to CSV"""
    
    if not results:
        print("No VLLM results to export")
        return
    
    # Create output filename
    output_path = create_output_filename(video_path, 'vllm_analysis')
    
    print(f"Exporting VLLM results to: {output_path}")
    
    # Convert results to CSV-friendly format
    csv_data = []
    for result in results:
        csv_row = {
            'timestamp': result['timestamp'],
            'frame_index': result['frame_index'], 
            'description': result['description'],
            'context_provided': result['context_provided'],
            'description_length': len(result['description'])
        }
        csv_data.append(csv_row)
    
    # Export using existing CSV exporter
    export_objects_csv(csv_data, output_path)
    
    print(f"VLLM analysis results exported successfully")
    print(f"Total analyzed segments: {len(results)}")
    
    # Show sample results - БЕЗ ОГРАНИЧЕНИЙ
    print("\nSample VLLM descriptions:")
    for i, result in enumerate(results[:3]):
        desc = result['description']  # Убрали ограничение [:80]
        print(f"  {i+1}. {result['timestamp']}: {desc}")
    
    if len(results) > 3:
        print(f"  ... and {len(results) - 3} more descriptions")


def main() -> None:
    """Main function for VLLM analysis module"""
    print("=== VLLM Video Analysis Module Started ===")
    
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
            process_single_video_vllm(video_path)
            successful_videos += 1
            print(f"✅ Successfully processed: {Path(video_path).name}")
            
        except Exception as e:
            print(f"❌ ERROR processing {video_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n=== VLLM Analysis Module Completed ===")
    print(f"Successfully processed: {successful_videos}/{total_videos} videos")
    print(f"Failed: {total_videos - successful_videos} videos")


if __name__ == "__main__":
    main()