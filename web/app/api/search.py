"""
Search API endpoints
"""
from fastapi import APIRouter, Query
from typing import List, Optional
from ..database import get_db

router = APIRouter(prefix="/api", tags=["search"])


@router.get("/search")
async def search_videos(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(10, ge=1, le=100),
    video: Optional[str] = Query(None, description="Filter by video name")
):
    """
    Search video segments by text query (semantic search)
    """
    db = get_db()
    results = db.search(query=q, limit=limit, video_filter=video)

    return {
        "query": q,
        "count": len(results),
        "results": [
            {
                "id": r["id"],
                "video_name": r["video_name"],
                "segment_index": r["segment_index"],
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "description": r["description"],
                "keywords": r["keywords"],
                "search_terms": r.get("search_terms"),
                "relevance": round(r["relevance_score"], 3) if r.get("relevance_score") else None,
                "desc_score": round(r["desc_score"], 3) if r.get("desc_score") else None,
                "search_score": round(r["search_score"], 3) if r.get("search_score") else None
            }
            for r in results
        ]
    }
