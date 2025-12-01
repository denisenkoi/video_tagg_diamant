"""
Load Phase 2 VLLM analysis data into PostgreSQL
Reads JSON files from output/ and stores in video_segments table with embeddings
"""
import json
import sys
from pathlib import Path
from typing import Optional
import logging
from datetime import datetime

from db_manager_postgres import PostgresDBManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_phase2_json(video_name: str, output_dir: str = "output") -> Optional[dict]:
    """
    Load Phase 2 VLLM analysis JSON file

    Args:
        video_name: Video name (without extension)
        output_dir: Directory containing Phase 2 JSON files

    Returns:
        Parsed JSON data or None if file not found
    """
    json_path = Path(output_dir) / f"{video_name}_phase2_vllm_analysis.json"

    if not json_path.exists():
        logger.error(f"Phase 2 JSON not found: {json_path}")
        return None

    logger.info(f"Loading Phase 2 data from: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    logger.info(f"✓ Loaded JSON with {len(data.get('segments', []))} segments")
    return data


def load_video_to_db(video_name: str, video_path: str,
                     db_manager: PostgresDBManager,
                     output_dir: str = "output") -> bool:
    """
    Load a single video's Phase 2 data into PostgreSQL (ON CONFLICT = update)

    Args:
        video_name: Video name (without extension)
        video_path: Path to the actual video file
        db_manager: DBManager instance
        output_dir: Directory containing Phase 2 JSON files

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"\n=== Processing {video_name} ===")

    # === Check DB status ===
    video_status = db_manager.get_video_status(video_name)

    if not video_status:
        logger.error(f"Video {video_name} not found in DB - run Phase 1 first!")
        return False

    if video_status.get("phase2_status") != "completed":
        logger.warning(f"Phase 2 not completed for {video_name}, skipping...")
        return False

    if video_status.get("db_load_status") == "completed":
        logger.info(f"✓ DB load already completed for {video_name}, skipping...")
        return True

    if video_status.get("db_load_status") == "processing":
        # Check timeout
        start_time = datetime.fromisoformat(video_status["db_load_start_time"])
        video_duration = video_status["video_duration"]
        timeout = video_duration * 2

        if (datetime.now() - start_time).total_seconds() < timeout:
            logger.warning(f"⚠️ DB load still processing for {video_name}, skipping...")
            return False
        else:
            logger.info(f"🔄 DB load timeout detected for {video_name}, reprocessing...")

    # Set status to "processing"
    db_manager.update_video_status(
        video_name=video_name,
        db_load_status="processing",
        db_load_start_time=datetime.now().isoformat()
    )

    # Load Phase 2 JSON
    phase2_data = load_phase2_json(video_name, output_dir)
    if not phase2_data:
        return False

    # Extract segments
    segments = phase2_data.get('segments', [])
    if not segments:
        logger.warning(f"No segments found in {video_name}")
        return False

    # Get video info
    video_info = phase2_data.get('video_info', {})
    total_segments = video_info.get('total_segments', len(segments))
    success_segments = video_info.get('success_segments', 0)
    analysis_complete = video_info.get('analysis_complete', False)

    logger.info(f"Video info:")
    logger.info(f"  Total segments: {total_segments}")
    logger.info(f"  Success segments: {success_segments}")
    logger.info(f"  Analysis complete: {analysis_complete}")

    # Add segments to DB
    added_count = db_manager.add_segments(video_path, segments)

    if added_count > 0:
        # Update DB status to "completed"
        db_manager.update_video_status(
            video_name=video_name,
            db_load_status="completed",
            db_segments_loaded=added_count
        )

        logger.info(f"✅ Successfully loaded {added_count} segments from {video_name}. DB status updated.")
        return True
    else:
        logger.error(f"❌ Failed to load segments from {video_name}")
        return False


def main():
    """Main function to load Phase 2 data into DB"""
    print("=== Load Phase 2 Data to DB ===\n")

    # Initialize PostgreSQL manager
    logger.info("Initializing PostgreSQL manager...")
    db_manager = PostgresDBManager()

    # Show current stats
    stats = db_manager.get_collection_stats()
    logger.info(f"Current collection stats:")
    logger.info(f"  Video files: {stats['video_files_count']}")
    logger.info(f"  Video segments: {stats['video_segments_count']}")

    # Get videos from DB where phase2 completed and db_load pending/processing
    videos_to_load = db_manager.list_videos(
        status_filter={
            "phase2_status": "completed",
            "db_load_status": ["pending", "processing"]
        }
    )

    if not videos_to_load:
        print("✅ No videos to load - all DB loads complete or Phase 2 not ready!")
        return 0

    logger.info(f"\n📋 Found {len(videos_to_load)} videos to load:")
    for video_info in videos_to_load:
        logger.info(f"  - {video_info['video_name']}: db_load_status={video_info['db_load_status']}")

    # Process each video
    success_count = 0
    for video_info in videos_to_load:
        video_name = video_info["video_name"]
        video_path = video_info["video_path"]

        success = load_video_to_db(
            video_name=video_name,
            video_path=video_path,
            db_manager=db_manager,
            output_dir="output"
        )

        if success:
            success_count += 1

    # Final stats
    print("\n=== Loading Complete ===")
    final_stats = db_manager.get_collection_stats()
    logger.info(f"Final collection stats:")
    logger.info(f"  Total segments in DB: {final_stats['video_segments_count']}")
    logger.info(f"  Videos processed: {success_count}/{len(videos_to_load)}")

    if success_count == len(videos_to_load):
        print("\n✅ All videos loaded successfully!")
        return 0
    else:
        print(f"\n⚠️ {len(videos_to_load) - success_count} video(s) failed to load")
        return 1


if __name__ == "__main__":
    sys.exit(main())
