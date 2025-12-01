"""
PostgreSQL Database Manager for Video Tagging
Manages database storage and search for video segment descriptions with pgvector
"""
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from sentence_transformers import SentenceTransformer
from pgvector.psycopg2 import register_vector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Singleton for sentence transformer model"""
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.info("Loading sentence-transformers model...")
            cls._model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("✓ Model loaded")
        return cls._instance

    def encode(self, text: str) -> List[float]:
        """Encode text to embedding vector"""
        if not text.strip():
            # Return zero vector for empty text
            return [0.0] * 384
        return self._model.encode(text).tolist()


class PostgresDBManager:
    """Manager for PostgreSQL database operations with pgvector"""

    def __init__(self,
                 host: str = "localhost",
                 port: int = 5432,
                 database: str = "video_tagging_db",
                 user: str = "video_user",
                 password: str = "video_pass_2024"):
        """
        Initialize PostgreSQL connection

        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
        """
        self.connection_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }

        # Test connection
        logger.info(f"Connecting to PostgreSQL at {host}:{port}/{database}")
        self.conn = self._get_connection()
        register_vector(self.conn)

        # Initialize embedding model
        self.embedding_model = EmbeddingModel()

        # Get stats
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM video_files")
            video_files_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM video_segments")
            video_segments_count = cur.fetchone()[0]

        logger.info(f"✓ PostgreSQL connected:")
        logger.info(f"  - video_files: {video_files_count} records")
        logger.info(f"  - video_segments: {video_segments_count} segments")

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.connection_params)

    def _ensure_connection(self):
        """Ensure connection is alive"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            logger.info("Reconnecting to database...")
            self.conn = self._get_connection()
            register_vector(self.conn)

    # === Methods for video_files ===

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
        self._ensure_connection()

        with self.conn.cursor() as cur:
            # Check if exists
            cur.execute("SELECT video_name FROM video_files WHERE video_name = %s", (video_name,))
            exists = cur.fetchone() is not None

            if exists:
                # Update existing
                set_parts = [
                    "video_path = %s",
                    "language = %s",
                    "video_duration = %s",
                    "total_segments = %s",
                    "width = %s",
                    "height = %s",
                    "fps = %s",
                    "updated_at = NOW()"
                ]
                values = [video_path, language, video_duration, total_segments, width, height, fps]

                # Add status fields
                for key, value in status_fields.items():
                    set_parts.append(f"{key} = %s")
                    values.append(value)

                values.append(video_name)  # For WHERE clause

                sql = f"UPDATE video_files SET {', '.join(set_parts)} WHERE video_name = %s"
                cur.execute(sql, values)
                logger.info(f"✓ Updated video_files record for {video_name}")
            else:
                # Insert new
                columns = [
                    'video_name', 'video_path', 'language', 'video_duration',
                    'total_segments', 'width', 'height', 'fps'
                ]
                values = [video_name, video_path, language, video_duration, total_segments, width, height, fps]

                # Add status fields
                for key, value in status_fields.items():
                    columns.append(key)
                    values.append(value)

                placeholders = ', '.join(['%s'] * len(values))
                sql = f"INSERT INTO video_files ({', '.join(columns)}) VALUES ({placeholders})"
                cur.execute(sql, values)
                logger.info(f"✓ Created video_files record for {video_name}")

        self.conn.commit()

    def get_video_status(self, video_name: str) -> Optional[Dict[str, Any]]:
        """
        Get video processing status

        Args:
            video_name: Video name (without extension)

        Returns:
            Video metadata dict or None if not found
        """
        self._ensure_connection()

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM video_files WHERE video_name = %s", (video_name,))
            result = cur.fetchone()

            if result:
                # Convert to dict and handle datetime serialization
                result = dict(result)
                for key, value in result.items():
                    if isinstance(value, datetime):
                        result[key] = value.isoformat()
                return result
            return None

    def update_video_status(self, video_name: str, **status_fields) -> None:
        """
        Update video processing statuses

        Args:
            video_name: Video name (without extension)
            **status_fields: Status fields to update
        """
        self._ensure_connection()

        if not status_fields:
            return

        set_parts = ["updated_at = NOW()"]
        values = []

        for key, value in status_fields.items():
            set_parts.append(f"{key} = %s")
            values.append(value)

        values.append(video_name)

        sql = f"UPDATE video_files SET {', '.join(set_parts)} WHERE video_name = %s"

        with self.conn.cursor() as cur:
            cur.execute(sql, values)

        self.conn.commit()
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
        self._ensure_connection()

        sql = "SELECT * FROM video_files"
        where_parts = []
        values = []

        if status_filter:
            for key, value in status_filter.items():
                if isinstance(value, list):
                    placeholders = ', '.join(['%s'] * len(value))
                    where_parts.append(f"{key} IN ({placeholders})")
                    values.extend(value)
                else:
                    where_parts.append(f"{key} = %s")
                    values.append(value)

        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, values)
            results = cur.fetchall()

            # Convert datetime to isoformat
            videos = []
            for result in results:
                result = dict(result)
                for key, value in result.items():
                    if isinstance(value, datetime):
                        result[key] = value.isoformat()
                videos.append(result)

            return videos

    # === Methods for video_segments ===

    def add_segments(self, video_path: str, segments: List[Dict[str, Any]]) -> int:
        """
        Add video segments to database

        Args:
            video_path: Path to the video file
            segments: List of segment dictionaries from phase2_vllm_analysis.json

        Returns:
            Number of segments added
        """
        if not segments:
            logger.warning("No segments to add")
            return 0

        self._ensure_connection()

        from pathlib import Path
        video_name = Path(video_path).stem

        added_count = 0

        with self.conn.cursor() as cur:
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

                # Combine text for embedding
                document_text = f"{description} {dialogue} {keywords_text}".strip()

                if not document_text:
                    logger.warning(f"Empty document for segment {segment_index}, skipping")
                    continue

                # Generate embedding
                embedding = self.embedding_model.encode(document_text)

                # Create unique ID
                doc_id = f"{video_name}_seg_{segment_index}"

                # Insert segment
                sql = """
                    INSERT INTO video_segments (
                        id, video_name, video_path, segment_index,
                        start_time, end_time, duration,
                        description, keywords, confidence, content_type,
                        embedding
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        description = EXCLUDED.description,
                        keywords = EXCLUDED.keywords,
                        confidence = EXCLUDED.confidence,
                        content_type = EXCLUDED.content_type,
                        embedding = EXCLUDED.embedding
                """

                cur.execute(sql, (
                    doc_id,
                    video_name,
                    video_path,
                    segment_index,
                    segment.get('start_time', 0.0),
                    segment.get('end_time', 0.0),
                    segment.get('duration', 0.0),
                    description[:500] if description else None,
                    keywords_text[:200] if keywords_text else None,
                    analysis.get('confidence', 'средняя'),
                    analysis.get('content_type', ''),
                    embedding
                ))

                added_count += 1

        self.conn.commit()
        logger.info(f"✓ Added {added_count} segments from {video_name}")
        return added_count

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

        self._ensure_connection()

        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)

        # Build SQL query
        sql = """
            SELECT
                id,
                video_name,
                video_path,
                segment_index,
                start_time,
                end_time,
                duration,
                description,
                keywords,
                confidence,
                1 - (embedding <=> %s::vector) as relevance_score,
                embedding <=> %s::vector as distance
            FROM video_segments
        """

        where_parts = []
        values = [query_embedding, query_embedding]

        if video_filter:
            where_parts.append("video_name = %s")
            values.append(video_filter)

        if confidence_filter:
            where_parts.append("confidence = %s")
            values.append(confidence_filter)

        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)

        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        values.extend([query_embedding, limit])

        logger.info(f"Searching for: '{query}' (limit={limit})")

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, values)
            results = cur.fetchall()

        formatted_results = [dict(row) for row in results]
        logger.info(f"✓ Found {len(formatted_results)} results")
        return formatted_results

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        self._ensure_connection()

        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM video_files")
            video_files_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM video_segments")
            video_segments_count = cur.fetchone()[0]

        return {
            "video_files_count": video_files_count,
            "video_segments_count": video_segments_count,
            "database": self.connection_params['database']
        }

    def delete_video_segments(self, video_name: str) -> int:
        """
        Delete all segments for a specific video

        Args:
            video_name: Name of the video (without extension)

        Returns:
            Number of segments deleted
        """
        self._ensure_connection()

        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM video_segments WHERE video_name = %s", (video_name,))
            deleted_count = cur.rowcount

        self.conn.commit()

        if deleted_count > 0:
            logger.info(f"✓ Deleted {deleted_count} segments from {video_name}")
        else:
            logger.info(f"No segments found for {video_name}")

        return deleted_count

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            if logger:
                logger.info("Database connection closed")

    def __del__(self):
        """Cleanup on deletion"""
        self.close()


# Alias for backwards compatibility
ChromaDBManager = PostgresDBManager


if __name__ == "__main__":
    # Quick test
    print("=== PostgreSQL DB Manager Test ===")

    manager = PostgresDBManager()
    stats = manager.get_collection_stats()

    print(f"\nStats:")
    print(f"  Video files: {stats['video_files_count']}")
    print(f"  Video segments: {stats['video_segments_count']}")
    print(f"  Database: {stats['database']}")

    print("\n✓ PostgreSQL DB Manager is working!")
