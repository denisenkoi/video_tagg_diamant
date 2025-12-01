"""
Persons/Faces API endpoints
"""
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import Response
from typing import List, Optional
from ..database import get_db

router = APIRouter(prefix="/api", tags=["persons"])


@router.get("/persons")
async def get_persons(
    video: str = Query(..., description="Video name"),
    clustering: str = Query("onthefly", description="Clustering type: onthefly, postclustering, alt")
):
    """
    Get all persons/clusters for a video
    """
    db = get_db()

    if clustering not in ("onthefly", "postclustering", "alt"):
        raise HTTPException(400, f"Invalid clustering type: {clustering}")

    persons = db.get_persons_by_video_clustered(video, clustering)

    return {
        "video": video,
        "clustering": clustering,
        "count": len(persons),
        "persons": [
            {
                "cluster_id": p["cluster_id"],
                "detection_count": p["detection_count"],
                "first_seen": round(p["first_seen"], 2) if p["first_seen"] else None,
                "last_seen": round(p["last_seen"], 2) if p["last_seen"] else None,
                "avg_age": round(p["avg_age"]) if p["avg_age"] else None,
                "gender": p["common_gender"],
                "detection_ids": p["detection_ids"][:5]  # First 5 for preview
            }
            for p in persons
        ]
    }


@router.get("/persons/{cluster_id}/detections")
async def get_person_detections(
    cluster_id: str,
    video: str = Query(..., description="Video name"),
    clustering: str = Query("onthefly", description="Clustering type")
):
    """
    Get all detections for a person/cluster
    """
    db = get_db()

    # Map clustering type to column
    column_map = {
        'onthefly': 'person_id',
        'postclustering': 'person_id_postclustering',
        'alt': 'person_id_alt'
    }
    column = column_map.get(clustering, 'person_id')

    # Get detections
    db._ensure_connection()
    with db.conn.cursor() as cur:
        cur.execute(f"""
            SELECT id, timestamp, bbox_x, bbox_y, bbox_w, bbox_h,
                   age, gender, emotion, det_score
            FROM face_detections
            WHERE video_name = %s AND {column} = %s
            ORDER BY timestamp
        """, (video, cluster_id))

        from psycopg2.extras import RealDictCursor
        cur2 = db.conn.cursor(cursor_factory=RealDictCursor)
        cur2.execute(f"""
            SELECT id, timestamp, bbox_x, bbox_y, bbox_w, bbox_h,
                   age, gender, emotion, det_score
            FROM face_detections
            WHERE video_name = %s AND {column} = %s
            ORDER BY timestamp
        """, (video, cluster_id))
        results = cur2.fetchall()

    return {
        "cluster_id": cluster_id,
        "video": video,
        "clustering": clustering,
        "count": len(results),
        "detections": [
            {
                "id": r["id"],
                "timestamp": round(r["timestamp"], 2),
                "bbox": [r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"]],
                "age": r["age"],
                "gender": r["gender"],
                "emotion": r["emotion"],
                "det_score": round(r["det_score"], 3) if r["det_score"] else None
            }
            for r in results
        ]
    }


@router.get("/thumbnail/{detection_id}")
async def get_thumbnail(detection_id: int):
    """
    Get face thumbnail image
    """
    db = get_db()
    thumbnail = db.get_face_thumbnail(detection_id)

    if not thumbnail:
        raise HTTPException(404, "Thumbnail not found")

    return Response(
        content=thumbnail,
        media_type="image/jpeg",
        headers={"Cache-Control": "max-age=86400"}  # Cache 24h
    )


@router.get("/faces")
async def get_all_faces(
    video: str = Query(..., description="Video name"),
    limit: int = Query(500, description="Max faces to return"),
    offset: int = Query(0, description="Offset for pagination")
):
    """
    Get all face detections for a video (no clustering, raw detections)
    """
    db = get_db()
    db._ensure_connection()

    from psycopg2.extras import RealDictCursor

    with db.conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Get total count
        cur.execute(
            "SELECT COUNT(*) as cnt FROM face_detections WHERE video_name = %s",
            (video,)
        )
        total = cur.fetchone()["cnt"]

        # Get faces
        cur.execute("""
            SELECT id, timestamp, bbox_x, bbox_y, bbox_w, bbox_h,
                   age, gender, emotion, det_score, person_id
            FROM face_detections
            WHERE video_name = %s
            ORDER BY timestamp, id
            LIMIT %s OFFSET %s
        """, (video, limit, offset))
        results = cur.fetchall()

    return {
        "video": video,
        "total": total,
        "limit": limit,
        "offset": offset,
        "count": len(results),
        "faces": [
            {
                "id": r["id"],
                "timestamp": round(r["timestamp"], 2),
                "bbox": [r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"]],
                "age": r["age"],
                "gender": r["gender"],
                "emotion": r["emotion"],
                "det_score": round(r["det_score"], 3) if r["det_score"] else None,
                "person_id": r["person_id"]
            }
            for r in results
        ]
    }
