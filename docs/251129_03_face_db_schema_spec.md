# Specification: Face Data PostgreSQL Schema

**Date:** 2025-11-29 03:00
**Task:** VIDEO-18 (TBD)
**Status:** Draft

---

## Goal

Add face detection data storage to PostgreSQL database with:
- Person clustering and identification
- Full embedding storage for re-clustering
- Vector search capability (pgvector)
- Timeline tracking

---

## Current Database Structure

### Existing Tables
- `video_files` - video metadata + processing status
- `video_segments` - segments with text embeddings (384D)

### Connection
```
Host: localhost
Port: 5432
Database: video_tagging_db
User: video_user
Password: video_pass_2024
```

---

## New Tables

### 1. persons - Unique Persons (Clusters)

```sql
CREATE TABLE persons (
    person_id VARCHAR(50) PRIMARY KEY,      -- 'person_01', 'person_02'...
    video_name VARCHAR(255) NOT NULL,       -- first appearance video
    first_seen_time FLOAT NOT NULL,         -- timestamp of first appearance
    representative_embedding vector(512),   -- ArcFace reference embedding
    name VARCHAR(255),                       -- manual annotation (optional)
    appearances_count INTEGER DEFAULT 1,    -- total detections count
    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT fk_persons_video
        FOREIGN KEY (video_name)
        REFERENCES video_files(video_name)
        ON DELETE CASCADE
);

-- Index for vector search on persons
CREATE INDEX persons_embedding_idx ON persons
    USING ivfflat (representative_embedding vector_cosine_ops)
    WITH (lists = 100);

-- Index for video filtering
CREATE INDEX persons_video_name_idx ON persons(video_name);
```

### 2. face_detections - All Face Detections

```sql
CREATE TABLE face_detections (
    id SERIAL PRIMARY KEY,
    video_name VARCHAR(255) NOT NULL,       -- source video
    person_id VARCHAR(50),                  -- linked person (nullable for unclustered)
    timestamp FLOAT NOT NULL,               -- time in video (seconds)

    -- Bounding box
    bbox_x INTEGER NOT NULL,
    bbox_y INTEGER NOT NULL,
    bbox_w INTEGER NOT NULL,
    bbox_h INTEGER NOT NULL,

    -- Attributes
    age INTEGER,
    gender VARCHAR(10),                     -- 'Man' / 'Woman'
    emotion VARCHAR(20),                    -- 'happy', 'neutral', 'angry'...
    det_score FLOAT,                        -- detection confidence

    -- Embedding for re-clustering
    embedding vector(512),                  -- ArcFace embedding (512D)

    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT fk_detections_video
        FOREIGN KEY (video_name)
        REFERENCES video_files(video_name)
        ON DELETE CASCADE,

    CONSTRAINT fk_detections_person
        FOREIGN KEY (person_id)
        REFERENCES persons(person_id)
        ON DELETE SET NULL
);

-- Index for vector search on detections
CREATE INDEX face_detections_embedding_idx ON face_detections
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Index for video filtering
CREATE INDEX face_detections_video_idx ON face_detections(video_name);

-- Index for person filtering
CREATE INDEX face_detections_person_idx ON face_detections(person_id);

-- Index for timeline queries
CREATE INDEX face_detections_timestamp_idx ON face_detections(video_name, timestamp);

-- Index for attribute filtering
CREATE INDEX face_detections_gender_idx ON face_detections(gender);
CREATE INDEX face_detections_age_idx ON face_detections(age);
```

---

## Data Size Estimation

### Embedding Storage
- 512 floats × 4 bytes = 2KB per detection
- 10,000 detections = 20MB
- 100,000 detections = 200MB

### Acceptable for:
- Exhibition demos
- Small-medium video collections
- Re-clustering scenarios

---

## db_manager_postgres.py Methods

### New Methods to Add

```python
# === Methods for persons ===

def create_person(
    self,
    person_id: str,
    video_name: str,
    first_seen_time: float,
    representative_embedding: List[float],
    name: Optional[str] = None
) -> None:
    """Create new person record"""
    pass

def get_person(self, person_id: str) -> Optional[Dict]:
    """Get person by ID"""
    pass

def update_person_embedding(
    self,
    person_id: str,
    embedding: List[float]
) -> None:
    """Update representative embedding"""
    pass

def increment_person_appearances(self, person_id: str) -> None:
    """Increment appearances count"""
    pass

def search_similar_persons(
    self,
    embedding: List[float],
    threshold: float = 0.7,
    limit: int = 5
) -> List[Dict]:
    """Find similar persons by embedding"""
    pass

def set_person_name(self, person_id: str, name: str) -> None:
    """Set manual name annotation"""
    pass


# === Methods for face_detections ===

def add_face_detection(
    self,
    video_name: str,
    person_id: Optional[str],
    timestamp: float,
    bbox: Tuple[int, int, int, int],  # x, y, w, h
    age: int,
    gender: str,
    emotion: str,
    det_score: float,
    embedding: List[float]
) -> int:
    """Add face detection, returns detection ID"""
    pass

def get_face_detections_by_video(
    self,
    video_name: str,
    person_id: Optional[str] = None
) -> List[Dict]:
    """Get all face detections for video"""
    pass

def get_face_detections_by_person(
    self,
    person_id: str
) -> List[Dict]:
    """Get all appearances of a person"""
    pass

def search_similar_faces(
    self,
    embedding: List[float],
    threshold: float = 0.7,
    limit: int = 10
) -> List[Dict]:
    """Find similar faces by embedding"""
    pass

def get_face_timeline(
    self,
    video_name: str
) -> List[Dict]:
    """Get timeline of face appearances"""
    pass


# === Bulk operations ===

def add_face_detections_batch(
    self,
    detections: List[Dict]
) -> int:
    """Bulk insert face detections, returns count"""
    pass

def recluster_faces(
    self,
    video_name: Optional[str] = None,
    threshold: float = 0.7
) -> Dict[str, int]:
    """Re-cluster faces, returns stats"""
    pass
```

---

## SQL Queries

### Find Similar Persons
```sql
SELECT person_id, name,
       1 - (representative_embedding <=> $1::vector) as similarity
FROM persons
WHERE 1 - (representative_embedding <=> $1::vector) >= $2
ORDER BY similarity DESC
LIMIT $3;
```

### Get Person Timeline
```sql
SELECT fd.timestamp, fd.age, fd.gender, fd.emotion, fd.det_score,
       fd.bbox_x, fd.bbox_y, fd.bbox_w, fd.bbox_h
FROM face_detections fd
WHERE fd.video_name = $1 AND fd.person_id = $2
ORDER BY fd.timestamp;
```

### Face Detection Statistics
```sql
SELECT
    p.person_id,
    p.name,
    COUNT(fd.id) as appearances,
    AVG(fd.age) as avg_age,
    MODE() WITHIN GROUP (ORDER BY fd.gender) as common_gender,
    MODE() WITHIN GROUP (ORDER BY fd.emotion) as common_emotion
FROM persons p
LEFT JOIN face_detections fd ON p.person_id = fd.person_id
WHERE p.video_name = $1
GROUP BY p.person_id, p.name;
```

### All Faces in Time Range
```sql
SELECT fd.*, p.name as person_name
FROM face_detections fd
LEFT JOIN persons p ON fd.person_id = p.person_id
WHERE fd.video_name = $1
  AND fd.timestamp BETWEEN $2 AND $3
ORDER BY fd.timestamp;
```

---

## Migration Script

```sql
-- Run as superuser or video_user with CREATE permissions

-- Enable pgvector if not already
CREATE EXTENSION IF NOT EXISTS vector;

-- Create persons table
CREATE TABLE IF NOT EXISTS persons (
    person_id VARCHAR(50) PRIMARY KEY,
    video_name VARCHAR(255) NOT NULL REFERENCES video_files(video_name) ON DELETE CASCADE,
    first_seen_time FLOAT NOT NULL,
    representative_embedding vector(512),
    name VARCHAR(255),
    appearances_count INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create face_detections table
CREATE TABLE IF NOT EXISTS face_detections (
    id SERIAL PRIMARY KEY,
    video_name VARCHAR(255) NOT NULL REFERENCES video_files(video_name) ON DELETE CASCADE,
    person_id VARCHAR(50) REFERENCES persons(person_id) ON DELETE SET NULL,
    timestamp FLOAT NOT NULL,
    bbox_x INTEGER NOT NULL,
    bbox_y INTEGER NOT NULL,
    bbox_w INTEGER NOT NULL,
    bbox_h INTEGER NOT NULL,
    age INTEGER,
    gender VARCHAR(10),
    emotion VARCHAR(20),
    det_score FLOAT,
    embedding vector(512),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS persons_embedding_idx ON persons
    USING ivfflat (representative_embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS persons_video_name_idx ON persons(video_name);

CREATE INDEX IF NOT EXISTS face_detections_embedding_idx ON face_detections
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS face_detections_video_idx ON face_detections(video_name);
CREATE INDEX IF NOT EXISTS face_detections_person_idx ON face_detections(person_id);
CREATE INDEX IF NOT EXISTS face_detections_timestamp_idx ON face_detections(video_name, timestamp);

-- Add face processing status to video_files
ALTER TABLE video_files
    ADD COLUMN IF NOT EXISTS face_status VARCHAR(20) DEFAULT 'pending',
    ADD COLUMN IF NOT EXISTS face_start_time TIMESTAMP,
    ADD COLUMN IF NOT EXISTS face_persons_count INTEGER,
    ADD COLUMN IF NOT EXISTS face_detections_count INTEGER;

-- Verify
SELECT 'persons' as table_name, COUNT(*) as rows FROM persons
UNION ALL
SELECT 'face_detections', COUNT(*) FROM face_detections;
```

---

## Testing Checklist

1. [ ] Tables created successfully
2. [ ] Indexes created (check with `\di`)
3. [ ] Foreign keys work (cascade delete)
4. [ ] Vector search works on persons
5. [ ] Vector search works on face_detections
6. [ ] Bulk insert performance acceptable
7. [ ] video_files has face_status column

---

## Use Cases

### 1. Exhibition Demo - "Find this person"
```python
# User uploads photo, extract embedding
embedding = extract_face_embedding(photo)

# Search in database
similar = db.search_similar_persons(embedding, threshold=0.6)

# Show videos with this person
for person in similar:
    videos = db.get_face_detections_by_person(person['person_id'])
```

### 2. Video Analysis - "Who appears in this video?"
```python
# Get all unique persons in video
persons = db.get_persons_by_video(video_name)

# Show timeline
for person in persons:
    timeline = db.get_face_timeline(video_name, person['person_id'])
```

### 3. Re-clustering after collecting more data
```python
# Re-cluster with stricter threshold
stats = db.recluster_faces(threshold=0.75)
print(f"Merged {stats['merged']} duplicates, created {stats['new']} new persons")
```

---

**Document Path (Windows):** E:\Projects\Quantum\Video_tagging_db\docs\251129_03_face_db_schema_spec.md
