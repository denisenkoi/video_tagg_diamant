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

    def delete_video_data(self, video_name: str) -> Dict[str, int]:
        """
        Delete all data for a video (for force reindexing)

        Args:
            video_name: Video name (without extension)

        Returns:
            Dict with counts of deleted records
        """
        self._ensure_connection()

        deleted = {}

        with self.conn.cursor() as cur:
            # Delete video_segments
            cur.execute("DELETE FROM video_segments WHERE video_name = %s", (video_name,))
            deleted['video_segments'] = cur.rowcount

            # Delete face_detections
            cur.execute("DELETE FROM face_detections WHERE video_name = %s", (video_name,))
            deleted['face_detections'] = cur.rowcount

            # Delete persons
            cur.execute("DELETE FROM persons WHERE video_name = %s", (video_name,))
            deleted['persons'] = cur.rowcount

            # Reset statuses in video_files (keep the record)
            cur.execute("""
                UPDATE video_files SET
                    phase1_status = 'pending',
                    phase1_start_time = NULL,
                    phase1_segments_created = NULL,
                    phase2_status = 'pending',
                    phase2_start_time = NULL,
                    phase2_segments_analyzed = NULL,
                    face_status = 'pending',
                    face_start_time = NULL,
                    face_persons_count = NULL,
                    face_detections_count = NULL,
                    db_load_status = 'pending',
                    db_load_start_time = NULL,
                    db_segments_loaded = 0,
                    updated_at = NOW()
                WHERE video_name = %s
            """, (video_name,))

        self.conn.commit()

        logger.info(f"🗑️ Deleted data for {video_name}:")
        logger.info(f"  - video_segments: {deleted['video_segments']}")
        logger.info(f"  - face_detections: {deleted['face_detections']}")
        logger.info(f"  - persons: {deleted['persons']}")
        logger.info(f"  - Statuses reset to 'pending'")

        return deleted

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
                search_terms = analysis.get('search_terms', '')

                # Combine text for embedding
                document_text = f"{description} {dialogue} {keywords_text}".strip()

                if not document_text:
                    logger.warning(f"Empty document for segment {segment_index}, skipping")
                    continue

                # Generate embeddings
                embedding = self.embedding_model.encode(document_text)
                search_terms_embedding = self.embedding_model.encode(search_terms) if search_terms else None

                # Create unique ID
                doc_id = f"{video_name}_seg_{segment_index}"

                # Insert segment
                sql = """
                    INSERT INTO video_segments (
                        id, video_name, video_path, segment_index,
                        start_time, end_time, duration,
                        description, keywords, confidence, content_type,
                        embedding, search_terms, search_terms_embedding
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        description = EXCLUDED.description,
                        keywords = EXCLUDED.keywords,
                        confidence = EXCLUDED.confidence,
                        content_type = EXCLUDED.content_type,
                        embedding = EXCLUDED.embedding,
                        search_terms = EXCLUDED.search_terms,
                        search_terms_embedding = EXCLUDED.search_terms_embedding
                """

                content_type = analysis.get('content_type', '')[:50] if analysis.get('content_type') else ''
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
                    analysis.get('confidence', 'средняя')[:20] if analysis.get('confidence') else 'средняя',
                    content_type,
                    embedding,
                    search_terms[:500] if search_terms else None,
                    search_terms_embedding
                ))

                added_count += 1

        self.conn.commit()
        logger.info(f"✓ Added {added_count} segments from {video_name}")
        return added_count

    def search(self, query: str, limit: int = 10,
               video_filter: Optional[str] = None,
               confidence_filter: Optional[str] = None,
               hybrid: bool = True) -> List[Dict[str, Any]]:
        """
        Search for video segments by text query (hybrid: description + search_terms)

        Args:
            query: Text query (русский or English)
            limit: Maximum number of results
            video_filter: Filter by video name
            confidence_filter: Filter by confidence level
            hybrid: Use hybrid search (description + search_terms embeddings)

        Returns:
            List of matching segments with metadata
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []

        self._ensure_connection()

        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)

        # Build SQL query with hybrid scoring
        # Hybrid: 0.5 * description_similarity + 0.5 * search_terms_similarity (if exists)
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
                search_terms,
                confidence,
                1 - (embedding <=> %s::vector) as desc_score,
                CASE
                    WHEN search_terms_embedding IS NOT NULL
                    THEN 1 - (search_terms_embedding <=> %s::vector)
                    ELSE 0
                END as search_score,
                CASE
                    WHEN search_terms_embedding IS NOT NULL
                    THEN 0.5 * (1 - (embedding <=> %s::vector)) + 0.5 * (1 - (search_terms_embedding <=> %s::vector))
                    ELSE 1 - (embedding <=> %s::vector)
                END as relevance_score
            FROM video_segments
        """

        where_parts = []
        values = [query_embedding, query_embedding, query_embedding, query_embedding, query_embedding]

        if video_filter:
            where_parts.append("video_name = %s")
            values.append(video_filter)

        if confidence_filter:
            where_parts.append("confidence = %s")
            values.append(confidence_filter)

        if where_parts:
            sql += " WHERE " + " AND ".join(where_parts)

        sql += " ORDER BY relevance_score DESC LIMIT %s"
        values.append(limit)

        logger.info(f"Searching for: '{query}' (limit={limit}, hybrid={hybrid})")

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

    # === Methods for persons (face clustering) ===

    def create_person(
        self,
        person_id: str,
        video_name: str,
        first_seen_time: float,
        representative_embedding: List[float],
        name: Optional[str] = None
    ) -> None:
        """Create new person record"""
        self._ensure_connection()

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO persons (person_id, video_name, first_seen_time, representative_embedding, name)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (person_id) DO UPDATE SET
                    representative_embedding = EXCLUDED.representative_embedding,
                    appearances_count = persons.appearances_count + 1
            """, (person_id, video_name, first_seen_time, representative_embedding, name))

        self.conn.commit()
        logger.info(f"✓ Created/updated person {person_id}")

    def get_person(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Get person by ID"""
        self._ensure_connection()

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM persons WHERE person_id = %s", (person_id,))
            result = cur.fetchone()
            return dict(result) if result else None

    def search_similar_persons(
        self,
        embedding: List[float],
        threshold: float = 0.7,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar persons by embedding (cosine similarity)"""
        self._ensure_connection()

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT person_id, video_name, name, appearances_count,
                       1 - (representative_embedding <=> %s::vector) as similarity
                FROM persons
                WHERE 1 - (representative_embedding <=> %s::vector) >= %s
                ORDER BY similarity DESC
                LIMIT %s
            """, (embedding, embedding, threshold, limit))
            results = cur.fetchall()

        return [dict(row) for row in results]

    def increment_person_appearances(self, person_id: str) -> None:
        """Increment appearances count for person"""
        self._ensure_connection()

        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE persons SET appearances_count = appearances_count + 1
                WHERE person_id = %s
            """, (person_id,))

        self.conn.commit()

    # === Methods for face_detections ===

    def add_face_detection(
        self,
        video_name: str,
        person_id: Optional[str],
        timestamp: float,
        bbox: tuple,  # (x, y, w, h)
        age: int,
        gender: str,
        emotion: str,
        det_score: float,
        embedding: List[float]
    ) -> int:
        """Add face detection, returns detection ID"""
        self._ensure_connection()

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO face_detections
                (video_name, person_id, timestamp, bbox_x, bbox_y, bbox_w, bbox_h,
                 age, gender, emotion, det_score, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (video_name, person_id, timestamp, bbox[0], bbox[1], bbox[2], bbox[3],
                  age, gender, emotion, det_score, embedding))
            detection_id = cur.fetchone()[0]

        self.conn.commit()
        return detection_id

    def add_face_detections_batch(self, detections: List[Dict[str, Any]]) -> int:
        """Bulk insert face detections, returns count"""
        if not detections:
            return 0

        self._ensure_connection()

        with self.conn.cursor() as cur:
            from psycopg2.extras import execute_values

            values = []
            for d in detections:
                bbox = d.get('bbox', (0, 0, 0, 0))
                # Convert numpy types to native Python types for psycopg2
                values.append((
                    d['video_name'],
                    d.get('person_id'),
                    float(d['timestamp']),
                    int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
                    int(d['age']) if d.get('age') is not None else None,
                    d.get('gender'),
                    d.get('emotion'),
                    float(d['det_score']) if d.get('det_score') is not None else None,
                    d.get('embedding')
                ))

            execute_values(cur, """
                INSERT INTO face_detections
                (video_name, person_id, timestamp, bbox_x, bbox_y, bbox_w, bbox_h,
                 age, gender, emotion, det_score, embedding)
                VALUES %s
            """, values)

            count = cur.rowcount

        self.conn.commit()
        logger.info(f"✓ Added {count} face detections")
        return count

    def get_face_detections_by_video(
        self,
        video_name: str,
        person_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all face detections for video"""
        self._ensure_connection()

        sql = "SELECT * FROM face_detections WHERE video_name = %s"
        values = [video_name]

        if person_id:
            sql += " AND person_id = %s"
            values.append(person_id)

        sql += " ORDER BY timestamp"

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, values)
            results = cur.fetchall()

        return [dict(row) for row in results]

    def get_face_detections_by_person(self, person_id: str) -> List[Dict[str, Any]]:
        """Get all appearances of a person"""
        self._ensure_connection()

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM face_detections
                WHERE person_id = %s
                ORDER BY video_name, timestamp
            """, (person_id,))
            results = cur.fetchall()

        return [dict(row) for row in results]

    def search_similar_faces(
        self,
        embedding: List[float],
        threshold: float = 0.7,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find similar faces by embedding"""
        self._ensure_connection()

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, video_name, person_id, timestamp, age, gender, emotion,
                       1 - (embedding <=> %s::vector) as similarity
                FROM face_detections
                WHERE 1 - (embedding <=> %s::vector) >= %s
                ORDER BY similarity DESC
                LIMIT %s
            """, (embedding, embedding, threshold, limit))
            results = cur.fetchall()

        return [dict(row) for row in results]

    def get_persons_by_video(self, video_name: str) -> List[Dict[str, Any]]:
        """Get all unique persons in video"""
        self._ensure_connection()

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT p.*, COUNT(fd.id) as detections_in_video
                FROM persons p
                JOIN face_detections fd ON p.person_id = fd.person_id
                WHERE fd.video_name = %s
                GROUP BY p.person_id
                ORDER BY detections_in_video DESC
            """, (video_name,))
            results = cur.fetchall()

        return [dict(row) for row in results]

    def update_face_status(self, video_name: str, **status_fields) -> None:
        """Update face processing status for video"""
        self.update_video_status(video_name, **status_fields)

    # === Methods for post-clustering ===

    def get_embeddings_for_clustering(self, video_name: str) -> List[Dict[str, Any]]:
        """
        Get all face embeddings for clustering.

        Args:
            video_name: Video name to get embeddings from

        Returns:
            List of dicts with id and embedding
        """
        self._ensure_connection()

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, embedding
                FROM face_detections
                WHERE video_name = %s AND embedding IS NOT NULL
                ORDER BY id
            """, (video_name,))
            results = cur.fetchall()

        return [dict(row) for row in results]

    def update_postclustering_labels(
        self,
        video_name: str,
        labels_dict: Dict[int, str]
    ) -> int:
        """
        Update person_id_postclustering for multiple detections.

        Args:
            video_name: Video name
            labels_dict: Dict mapping detection_id -> cluster_label

        Returns:
            Number of rows updated
        """
        if not labels_dict:
            return 0

        self._ensure_connection()

        with self.conn.cursor() as cur:
            for detection_id, label in labels_dict.items():
                cur.execute("""
                    UPDATE face_detections
                    SET person_id_postclustering = %s
                    WHERE id = %s AND video_name = %s
                """, (label, detection_id, video_name))

        self.conn.commit()
        logger.info(f"Updated {len(labels_dict)} postclustering labels for {video_name}")
        return len(labels_dict)

    # === Methods for thumbnails ===

    def save_face_thumbnail(self, detection_id: int, thumbnail_bytes: bytes) -> None:
        """Save face thumbnail image"""
        self._ensure_connection()

        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE face_detections
                SET face_thumbnail = %s
                WHERE id = %s
            """, (thumbnail_bytes, detection_id))

        self.conn.commit()

    def save_face_thumbnails_batch(self, thumbnails: List[tuple]) -> int:
        """
        Batch save face thumbnails.

        Args:
            thumbnails: List of (detection_id, thumbnail_bytes) tuples

        Returns:
            Number of thumbnails saved
        """
        if not thumbnails:
            return 0

        self._ensure_connection()

        with self.conn.cursor() as cur:
            from psycopg2.extras import execute_batch
            execute_batch(cur, """
                UPDATE face_detections
                SET face_thumbnail = %s
                WHERE id = %s
            """, [(t[1], t[0]) for t in thumbnails])

        self.conn.commit()
        logger.info(f"Saved {len(thumbnails)} face thumbnails")
        return len(thumbnails)

    def get_face_thumbnail(self, detection_id: int) -> Optional[bytes]:
        """Get face thumbnail image"""
        self._ensure_connection()

        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT face_thumbnail FROM face_detections WHERE id = %s
            """, (detection_id,))
            result = cur.fetchone()

        return result[0] if result else None

    def get_detections_without_thumbnail(self, video_name: str) -> List[Dict[str, Any]]:
        """Get detections that need thumbnail generation"""
        self._ensure_connection()

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, timestamp, bbox_x, bbox_y, bbox_w, bbox_h
                FROM face_detections
                WHERE video_name = %s AND face_thumbnail IS NULL
                ORDER BY timestamp
            """, (video_name,))
            results = cur.fetchall()

        return [dict(row) for row in results]

    # === Methods for clustered persons ===

    def get_persons_by_video_clustered(
        self,
        video_name: str,
        clustering_type: str = 'onthefly'
    ) -> List[Dict[str, Any]]:
        """
        Get unique persons grouped by clustering type.

        Args:
            video_name: Video name
            clustering_type: 'onthefly', 'postclustering', or 'alt'

        Returns:
            List of person groups with detection counts
        """
        self._ensure_connection()

        # Map clustering type to column
        column_map = {
            'onthefly': 'person_id',
            'postclustering': 'person_id_postclustering',
            'alt': 'person_id_alt'
        }
        column = column_map.get(clustering_type, 'person_id')

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"""
                SELECT
                    {column} as cluster_id,
                    COUNT(*) as detection_count,
                    MIN(timestamp) as first_seen,
                    MAX(timestamp) as last_seen,
                    AVG(age) as avg_age,
                    MODE() WITHIN GROUP (ORDER BY gender) as common_gender,
                    ARRAY_AGG(DISTINCT id ORDER BY id) as detection_ids
                FROM face_detections
                WHERE video_name = %s AND {column} IS NOT NULL
                GROUP BY {column}
                ORDER BY detection_count DESC
            """, (video_name,))
            results = cur.fetchall()

        return [dict(row) for row in results]

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
