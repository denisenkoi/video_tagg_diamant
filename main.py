from typing import List, Dict, Any, Tuple
import numpy as np
import os
import shutil
from pathlib import Path
from config_manager import load_config, get_video_path, get_output_dir, set_video_path
from frame_extractor import extract_frames_with_timestamps, get_video_info
from scene_analyzer import initialize_clip_model, detect_scene_changes
from object_detector import initialize_yolo_model, detect_objects_batch
from face_processor import detect_faces_batch
from audio_transcriber import transcribe_video_file
#from translator import translate_transcription_segments
from result_exporter import (export_scenes_csv, export_objects_csv, export_faces_csv,
                             export_face_timeline_csv, export_transcription_csv, create_output_filename)


def setup_output_directories() -> None:
    """Setup and clean output directories"""
    output_dir = get_output_dir()
    output_path = Path(output_dir)

    # Clean previous results if directory exists
    if output_path.exists():
        print(f"Cleaning previous results from {output_dir}")
        shutil.rmtree(output_path)

    # Create fresh output directory
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {output_dir}")


def extract_video_frames() -> Tuple[List[np.ndarray], List[str], float]:
    """Extract video frames with timestamps (shared for all modules)"""
    # Load configuration parameters
    config = load_config()
    fps = config['frames_per_second']

    print(f"Extracting frames at {fps} FPS...")

    # Get video path and extract frames with timestamps
    video_path = get_video_path()
    frames, timestamps, original_fps = extract_frames_with_timestamps(video_path, fps)

    print(f"Extracted {len(frames)} frames from video (original FPS: {original_fps})")

    # Validate frame extraction
    assert len(frames) > 0, 'No frames extracted from video'

    return frames, timestamps, original_fps


def process_video_scenes(frames: List[np.ndarray], timestamps: List[str], original_fps: float) -> List[Dict]:
    """Processes video frames for scene detection"""
    # Load configuration parameters
    config = load_config()
    similarity_threshold = config['scene_similarity_threshold']

    print(f"Detecting scene changes with threshold {similarity_threshold}...")

    # Detect scene changes using real parameters from config
    scene_events = detect_scene_changes(frames, timestamps, original_fps, similarity_threshold)

    return scene_events


def process_video_objects(frames: List[np.ndarray], timestamps: List[str], yolo_model: Any) -> List[Dict]:
    """Processes video frames for object detection"""
    # Load configuration parameters
    config = load_config()
    confidence_threshold = config['object_confidence_threshold']

    print(f"Detecting objects with confidence threshold {confidence_threshold}...")

    # Detect objects using batch processing
    object_detections = detect_objects_batch(frames, timestamps, yolo_model, confidence_threshold)

    return object_detections


def process_video_faces(frames: List[np.ndarray], timestamps: List[str]) -> Tuple[List[Dict], List[Dict]]:
    """Processes video frames for face detection and clustering with attributes"""
    print("Starting face detection and person identification with attribute analysis...")

    # Detect and cluster faces using batch processing with attributes
    face_detections, timeline_events = detect_faces_batch(frames, timestamps)

    return face_detections, timeline_events


def process_audio_transcription() -> List[Dict]:
    """Process audio transcription from video file"""
    print("Starting audio transcription...")

    # Transcribe video file using existing module
    transcription_segments = transcribe_video_file()

    print(f"Audio transcription completed. Found {len(transcription_segments)} segments")

    return transcription_segments


def process_translation(transcription_segments: List[Dict]) -> List[Dict]:
    """Process translation of transcription from Kazakh to Russian"""
    print("Starting translation of transcription from Kazakh to Russian...")

    # Translate transcription segments using local LLM
    translated_segments = translate_transcription_segments(transcription_segments)

    print(f"Translation completed. Processed {len(translated_segments)} segments")

    return translated_segments


def export_scene_results(scenes: List[Dict]) -> None:
    """Export scene detection results"""
    # Get video path for filename generation
    video_path = get_video_path()

    # Create output file path
    scenes_path = create_output_filename(video_path, 'scenes')

    print(f"Exporting scenes to: {scenes_path}")

    # Export scenes to CSV
    if scenes:
        export_scenes_csv(scenes, scenes_path)
        print(f"Scene detection results exported successfully")
        print(f"Total scenes detected: {len(scenes)}")

        # Show sample results
        scene_changes = [s for s in scenes if s.get('change_detected', False)]
        print(f"Scene changes detected: {len(scene_changes)}")

        if scene_changes:
            print("First few scene changes:")
            for i, scene in enumerate(scene_changes[:3]):
                print(
                    f"  {i + 1}. {scene['timestamp']} - {scene['scene_description']} (similarity: {scene['similarity_score']:.3f})")
    else:
        print("No scene changes detected")


def export_object_results(objects: List[Dict]) -> None:
    """Export object detection results"""
    # Get video path for filename generation
    video_path = get_video_path()

    # Create output file path
    objects_path = create_output_filename(video_path, 'objects')

    print(f"Exporting objects to: {objects_path}")

    # Export objects to CSV
    if objects:
        export_objects_csv(objects, objects_path)
        print(f"Object detection results exported successfully")
        print(f"Total grouped timestamps: {len(objects)}")

        # Calculate total individual objects
        total_objects = sum(obj['count'] for obj in objects if obj['count'] > 0)
        print(f"Total individual objects: {total_objects}")

        # Show sample results
        all_objects = []
        for obj in objects:
            if obj['objects']:  # Skip empty object lists
                all_objects.extend(obj['objects'].split(','))

        unique_classes = list(set(all_objects))
        print(f"Object types found: {unique_classes}")

        # Count objects by type
        if all_objects:
            class_counts = {cls: all_objects.count(cls) for cls in unique_classes}
            print("Object counts:")
            for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {cls}: {count}")

        if len(objects) > 0:
            print("First few grouped detections:")
            for i, obj in enumerate(objects[:5]):
                if obj['count'] > 0:
                    print(f"  {i + 1}. {obj['timestamp']} - {obj['objects']} (count: {obj['count']})")
    else:
        print("No objects detected")


def export_face_results(faces: List[Dict], timeline_events: List[Dict]) -> None:
    """Export face detection and clustering results with timeline"""
    # Get video path for filename generation
    video_path = get_video_path()

    # Create output file paths
    faces_path = create_output_filename(video_path, 'faces')
    timeline_path = create_output_filename(video_path, 'face_timeline')

    print(f"Exporting faces to: {faces_path}")
    print(f"Exporting face timeline to: {timeline_path}")

    # Export main faces CSV
    if faces:
        export_faces_csv(faces, faces_path)
        print(f"Face detection results exported successfully")
        print(f"Total grouped timestamps: {len(faces)}")

        # Calculate total individual faces
        total_faces = sum(face['count'] for face in faces if face['count'] > 0)
        print(f"Total individual faces: {total_faces}")

        # Show sample results
        all_persons = []
        for face in faces:
            if face['persons']:  # Skip empty person lists
                all_persons.extend(face['persons'].split(','))

        unique_persons = list(set(all_persons))
        print(f"Unique persons identified: {unique_persons}")

        # Count appearances by person
        if all_persons:
            person_counts = {person: all_persons.count(person) for person in unique_persons}
            print("Person appearance counts:")
            for person, count in sorted(person_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {person}: {count} appearances")

        if len(faces) > 0:
            print("First few grouped face detections:")
            for i, face in enumerate(faces[:5]):
                if face['count'] > 0:
                    print(f"  {i + 1}. {face['timestamp']} - {face['persons']} (count: {face['count']})")
    else:
        print("No faces detected")

    # Export timeline CSV
    if timeline_events:
        export_face_timeline_csv(timeline_events, timeline_path)
        print(f"Face timeline exported successfully")
        print(f"Total timeline events: {len(timeline_events)}")

        # Show sample timeline events
        print("First few timeline events:")
        for i, event in enumerate(timeline_events[:5]):
            print(f"  {i + 1}. {event['timestamp']} - {event['person_id']}: "
                  f"age {event['age']}, gender {event['gender']}, "
                  f"emotion {event['emotion']}, race {event['race']}")
    else:
        print("No timeline events to export")


def export_translated_transcription_results(translated_transcription: List[Dict]) -> None:
    """Export translated transcription results"""
    # Get video path for filename generation
    video_path = get_video_path()

    # Create output file path
    translated_path = create_output_filename(video_path, 'transcription_translated')

    print(f"Exporting translated transcription to: {translated_path}")

    # Export translated transcription to CSV
    if translated_transcription:
        export_transcription_csv(translated_transcription, translated_path)
        print(f"Translated transcription results exported successfully")
        print(f"Total translated segments: {len(translated_transcription)}")

        # Show sample results
        print("First few translated segments:")
        for i, segment in enumerate(translated_transcription[:3]):
            original_text = segment.get('original_text', '')[:80]
            translated_text = segment.get('translated_text', segment.get('text', ''))[:80]

            print(f"  {i + 1}. {segment['start_formatted']}-{segment['end_formatted']}:")
            print(f"     Kazakh: {original_text}...")
            print(f"     Russian: {translated_text}...")
    else:
        print("No translated transcription data to export")


def export_transcription_results(transcription: List[Dict]) -> None:
    """Export audio transcription results"""
    # Get video path for filename generation
    video_path = get_video_path()

    # Create output file path
    transcription_path = create_output_filename(video_path, 'transcription')

    print(f"Exporting transcription to: {transcription_path}")

    # Export transcription to CSV
    if transcription:
        export_transcription_csv(transcription, transcription_path)
        print(f"Audio transcription results exported successfully")
        print(f"Total transcription segments: {len(transcription)}")

        # Calculate total duration
        if transcription:
            total_duration = transcription[-1]['end'] - transcription[0]['start']
            print(f"Total transcribed duration: {total_duration:.1f} seconds")

        # Show sample results - check for both original and translated text
        print("First few transcription segments:")
        for i, segment in enumerate(transcription[:3]):
            original_text = segment.get('original_text', segment.get('text', ''))[:100]
            translated_text = segment.get('translated_text', '')[:100] if segment.get('translated_text') else 'N/A'

            print(f"  {i + 1}. {segment['start_formatted']}-{segment['end_formatted']}:")
            print(f"     Original: {original_text}...")
            if translated_text != 'N/A':
                print(f"     Translation: {translated_text}...")
    else:
        print("No transcription data to export")


def process_single_video(video_path: str, clip_model: Any, clip_preprocess: Any, yolo_model: Any) -> None:
    """Process single video file through the entire pipeline"""
    print(f"\n=== Processing Video: {video_path} ===")

    # Set current video path for all modules
    set_video_path(video_path)

    # Validate video file exists
    if not Path(video_path).exists():
        print(f"ERROR: Video file not found: {video_path}")
        return

    print(f"Video file confirmed: {video_path}")

    # Process video frames (shared for all modules)
    print("Extracting video frames...")
    frames, timestamps, original_fps = extract_video_frames()

    # Process video scenes
    print("Processing video scenes...")
    scenes = process_video_scenes(frames, timestamps, original_fps)
    print(f"Scene detection completed. Found {len(scenes)} scene events")

    # Export scene results immediately
    export_scene_results(scenes)

    # Process object detection
    print("Processing object detection...")
    objects = process_video_objects(frames, timestamps, yolo_model)
    print(f"Object detection completed. Found {len(objects)} object detections")

    # Export object results immediately
    export_object_results(objects)

    # Process face detection with attributes
    print("Processing face detection and clustering...")
    faces, timeline_events = process_video_faces(frames, timestamps)
    print(f"Face detection completed. Found {len(faces)} face detections")

    # Export face results immediately
    export_face_results(faces, timeline_events)

    # Process audio transcription
    print("Processing audio transcription...")
    transcription = process_audio_transcription()
    print(f"Audio transcription completed. Found {len(transcription)} segments")

    # Export original transcription results immediately
    export_transcription_results(transcription)

    # Process translation (optional based on config)
    config = load_config()
    enable_translation = config.get('enable_translation', False)

    if enable_translation and transcription:
        print("Processing translation...")
        translated_transcription = process_translation(transcription)
        print(f"Translation completed. Processed {len(translated_transcription)} segments")

        # Export translated transcription results
        export_translated_transcription_results(translated_transcription)
    elif not enable_translation:
        print("Translation disabled in config - skipping")
    else:
        print("No transcription to translate")

    print(f"=== Video Processing Completed: {Path(video_path).name} ===\n")


def main() -> None:
    """Main function that orchestrates the video processing pipeline for multiple videos"""
    print("=== Multi-Video Processing Pipeline Started ===")

    # List of video files to process
    video_files = [
        #"video/AlmaAta.mp4",
        "video/news.mp4",
        #"video/SovKz.mp4"
    ]

    # Load initial configuration
    print("Loading configuration...")
    config = load_config()

    # Setup output directories
    setup_output_directories()

    # Initialize models once for all videos
    print("Initializing models...")
    clip_model, clip_preprocess = initialize_clip_model()
    print("CLIP model loaded successfully")

    yolo_model = initialize_yolo_model()
    print("YOLO model loaded successfully")

    print("DeepFace will be initialized on first face detection")

    # Process each video file
    total_videos = len(video_files)
    successful_videos = 0

    for i, video_path in enumerate(video_files, 1):
        print(f"\n*** Processing video {i}/{total_videos} ***")

        try:
            process_single_video(video_path, clip_model, clip_preprocess, yolo_model)
            successful_videos += 1
            print(f"✅ Successfully processed: {Path(video_path).name}")

        except Exception as e:
            print(f"❌ ERROR processing {video_path}: {str(e)}")
            print("Continuing with next video...")
            continue

    print(f"\n=== Pipeline Completed ===")
    print(f"Successfully processed: {successful_videos}/{total_videos} videos")
    print(f"Failed: {total_videos - successful_videos} videos")


if __name__ == "__main__":
    main()