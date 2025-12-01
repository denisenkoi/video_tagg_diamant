"""
ChromaDB Manager for Video Tagging
Manages vector database storage and search for video segment descriptions
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChromaDBManager:
    """Manager for ChromaDB vector database operations"""

    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB manager with two collections

        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        self.persist_directory = Path(persist_directory)

        # Create persist directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistence
        logger.info(f"Initializing ChromaDB at {self.persist_directory}")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(allow_reset=True)
        )

        # Get or create video_files collection (metadata + statuses)
        logger.info("Getting or creating collection: video_files")
        self.video_files_collection = self.client.get_or_create_collection(
            name="video_files",
            metadata={"description": "Video files metadata and processing status"}
        )

        # Get or create video_segments collection (segment descriptions)
        logger.info("Getting or creating collection: video_segments")
        self.video_segments_collection = self.client.get_or_create_collection(
            name="video_segments",
            metadata={"description": "Video segment descriptions from VLLM analysis"}
        )

        logger.info(f"✓ ChromaDB initialized:")
        logger.info(f"  - video_files: {self.video_files_collection.count()} records")
        logger.info(f"  - video_segments: {self.video_segments_collection.count()} segments")

    def add_segments(self, video_path: str, segments: List[Dict[str, Any]]) -> int:
        """
        Add video segments to ChromaDB

        Args:
            video_path: Path to the video file
            segments: List of segment dictionaries from phase2_vllm_analysis.json

        Returns:
            Number of segments added

        Example segment structure:
        {
            "segment_index": 0,
            "start_time": 0.0,
            "end_time": 50.0,
            "duration": 50.0,
            "transcript": "audio text",
            "analysis": {
                "description": "что происходит",
                "dialogue_translation": "перевод речи",
                "keywords": ["ключ1", "ключ2"],
                "confidence": "высокая"
            }
        }
        """
        if not segments:
            logger.warning("No segments to add")
            return 0

        video_name = Path(video_path).stem

        documents = []
        metadatas = []
        ids = []

        for segment in segments:
            segment_index = segment.get('segment_index', 0)
            analysis = segment.get('analysis', {})

            # Parse analysis if it's a JSON string
            if isinstance(analysis, str):
                import json
                try:
                    analysis = json.loads(analysis)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse analysis for segment {segment_index}")
                    analysis = {}

            # Extract fields
            description = analysis.get('description', '')
            dialogue = analysis.get('dialogue_translation', '')
            keywords = analysis.get('keywords', [])
            keywords_text = ' '.join(keywords) if isinstance(keywords, list) else str(keywords)

            # Combine text for embedding (ChromaDB will auto-embed this)
            document_text = f"{description} {dialogue} {keywords_text}".strip()

            if not document_text:
                logger.warning(f"Empty document for segment {segment_index}, skipping")
                continue

            # Create unique ID
            doc_id = f"{video_name}_seg_{segment_index}"

            # Create metadata
            metadata = {
                "video_path": video_path,
                "video_name": video_name,
                "segment_index": segment_index,
                "start_time": segment.get('start_time', 0.0),
                "end_time": segment.get('end_time', 0.0),
                "duration": segment.get('duration', 0.0),
                "description": description[:500],  # Limit length for metadata
                "keywords": keywords_text[:200],
                "confidence": analysis.get('confidence', 'средняя'),
                "content_type": analysis.get('content_type', '')
            }

            documents.append(document_text)
            metadatas.append(metadata)
            ids.append(doc_id)

        if not documents:
            logger.warning("No valid documents to add")
            return 0

        # Add to ChromaDB video_segments collection
        logger.info(f"Adding {len(documents)} segments to ChromaDB...")
        self.video_segments_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        logger.info(f"✓ Added {len(documents)} segments from {video_name}")
        return len(documents)

    def search(self, query: str, limit: int = 10,
               video_filter: Optional[str] = None,
               confidence_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for video segments by text query

        Args:
            query: Text query (русский or English)
            limit: Maximum number of results
            video_filter: Filter by video name
            confidence_filter: Filter by confidence level

        Returns:
            List of matching segments with metadata
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []

        # Build metadata filter
        where_filter = {}
        if video_filter:
            where_filter["video_name"] = video_filter
        if confidence_filter:
            where_filter["confidence"] = confidence_filter

        # Query ChromaDB video_segments collection
        logger.info(f"Searching for: '{query}' (limit={limit})")

        try:
            results = self.video_segments_collection.query(
                query_texts=[query],
                n_results=limit,
                where=where_filter if where_filter else None
            )
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

        # Format results
        formatted_results = []

        if results and results['documents'] and len(results['documents']) > 0:
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            ids = results['ids'][0]

            for i in range(len(documents)):
                result = {
                    'id': ids[i],
                    'video_path': metadatas[i].get('video_path'),
                    'video_name': metadatas[i].get('video_name'),
                    'segment_index': metadatas[i].get('segment_index'),
                    'start_time': metadatas[i].get('start_time'),
                    'end_time': metadatas[i].get('end_time'),
                    'duration': metadatas[i].get('duration'),
                    'description': metadatas[i].get('description'),
                    'keywords': metadatas[i].get('keywords'),
                    'confidence': metadatas[i].get('confidence'),
                    'relevance_score': 1.0 - distances[i],  # Convert distance to similarity
                    'distance': distances[i]
                }
                formatted_results.append(result)

        logger.info(f"✓ Found {len(formatted_results)} results")
        return formatted_results

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        return {
            "video_files_count": self.video_files_collection.count(),
            "video_segments_count": self.video_segments_collection.count(),
            "persist_directory": str(self.persist_directory)
        }

    def delete_video_segments(self, video_name: str) -> int:
        """
        Delete all segments for a specific video

        Args:
            video_name: Name of the video (without extension)

        Returns:
            Number of segments deleted
        """
        # Get all IDs for this video
        results = self.video_segments_collection.get(
            where={"video_name": video_name}
        )

        if results and results['ids']:
            ids_to_delete = results['ids']
            self.video_segments_collection.delete(ids=ids_to_delete)
            logger.info(f"✓ Deleted {len(ids_to_delete)} segments from {video_name}")
            return len(ids_to_delete)
        else:
            logger.info(f"No segments found for {video_name}")
            return 0

    def reset_collections(self):
        """Reset (clear) both collections"""
        logger.warning("Resetting all collections")

        # Reset video_files
        self.client.delete_collection(name="video_files")
        self.video_files_collection = self.client.create_collection(
            name="video_files",
            metadata={"description": "Video files metadata and processing status"}
        )

        # Reset video_segments
        self.client.delete_collection(name="video_segments")
        self.video_segments_collection = self.client.create_collection(
            name="video_segments",
            metadata={"description": "Video segment descriptions from VLLM analysis"}
        )

        logger.info("✓ Collections reset complete")

    # === Methods for video_files collection ===

    def create_or_update_video_file(
        self,
        video_name: str,
        video_path: str,
        language: str,
        video_duration: float,
        total_segments: int,
        width: int,
        height: int,
        fps: float,
        **status_fields
    ) -> None:
        """
        Create or update video_files record

        Args:
            video_name: Video name (without extension)
            video_path: Full path to video file
            language: Language code (ru, kk, etc.)
            video_duration: Video duration in seconds
            total_segments: Total number of segments
            width: Video width in pixels
            height: Video height in pixels
            fps: Frames per second
            **status_fields: Status fields (phase1_status, phase1_start_time, etc.)
        """
        from datetime import datetime

        # Get existing record if it exists
        try:
            existing = self.video_files_collection.get(ids=[video_name])
            is_update = existing and existing['ids']
        except:
            is_update = False

        # Build metadata
        metadata = {
            "video_name": video_name,
            "video_path": video_path,
            "language": language,
            "video_duration": video_duration,
            "total_segments": total_segments,
            "width": width,
            "height": height,
            "fps": fps,
            "updated_at": datetime.now().isoformat()
        }

        # Add status fields
        if not is_update:
            metadata["created_at"] = datetime.now().isoformat()
            # Initialize default statuses
            metadata["phase1_status"] = "pending"
            metadata["phase2_status"] = "pending"
            metadata["db_load_status"] = "pending"

        # Update with provided status fields
        metadata.update(status_fields)

        # Document for embeddings (будущая суммаризация)
        document = f"{video_name} {language}"

        if is_update:
            # Update existing record
            self.video_files_collection.update(
                ids=[video_name],
                documents=[document],
                metadatas=[metadata]
            )
            logger.info(f"✓ Updated video_files record for {video_name}")
        else:
            # Create new record
            self.video_files_collection.add(
                ids=[video_name],
                documents=[document],
                metadatas=[metadata]
            )
            logger.info(f"✓ Created video_files record for {video_name}")

    def get_video_status(self, video_name: str) -> Optional[Dict[str, Any]]:
        """
        Get video processing status

        Args:
            video_name: Video name (without extension)

        Returns:
            Video metadata dict or None if not found
        """
        try:
            result = self.video_files_collection.get(ids=[video_name])

            if result and result['ids']:
                metadata = result['metadatas'][0]
                return metadata
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting video status: {e}")
            return None

    def update_video_status(self, video_name: str, **status_fields) -> None:
        """
        Update video processing statuses

        Args:
            video_name: Video name (without extension)
            **status_fields: Status fields to update
        """
        from datetime import datetime

        # Get existing record
        existing_status = self.get_video_status(video_name)

        if not existing_status:
            logger.error(f"Video {video_name} not found in video_files collection")
            return

        # Update metadata
        existing_status.update(status_fields)
        existing_status["updated_at"] = datetime.now().isoformat()

        # Update ChromaDB
        document = f"{video_name} {existing_status.get('language', '')}"

        self.video_files_collection.update(
            ids=[video_name],
            metadatas=[existing_status],
            documents=[document]
        )

        logger.info(f"✓ Updated status for {video_name}: {status_fields}")

    def list_videos(self, status_filter: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Get list of videos with optional filtering

        Args:
            status_filter: Dictionary with status filters
                Example: {"phase1_status": "completed"}
                Example: {"phase2_status": ["pending", "processing"]}

        Returns:
            List of video metadata dictionaries
        """
        try:
            # Build where clause
            where_clause = {}

            if status_filter:
                for key, value in status_filter.items():
                    if isinstance(value, list):
                        # ChromaDB doesn't support IN operator, need to query separately
                        # For now, we'll get all and filter in Python
                        pass
                    else:
                        where_clause[key] = value

            # Get all records
            if where_clause:
                results = self.video_files_collection.get(where=where_clause)
            else:
                results = self.video_files_collection.get()

            if not results or not results['ids']:
                return []

            videos = []
            for i, video_id in enumerate(results['ids']):
                metadata = results['metadatas'][i]

                # Apply list filters in Python
                if status_filter:
                    matches = True
                    for key, value in status_filter.items():
                        if isinstance(value, list):
                            if metadata.get(key) not in value:
                                matches = False
                                break
                    if not matches:
                        continue

                videos.append(metadata)

            return videos

        except Exception as e:
            logger.error(f"Error listing videos: {e}")
            return []


if __name__ == "__main__":
    # Quick test
    print("=== ChromaDB Manager Test ===")

    manager = ChromaDBManager()
    stats = manager.get_collection_stats()

    print(f"\nStats:")
    print(f"  Video files: {stats['video_files_count']}")
    print(f"  Video segments: {stats['video_segments_count']}")
    print(f"  Persist directory: {stats['persist_directory']}")

    print("\n✓ ChromaDB Manager is working!")
