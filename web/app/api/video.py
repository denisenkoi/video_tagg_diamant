"""
Video API endpoints
"""
import os
import cv2
from fastapi import APIRouter, Query, HTTPException, Request
from fastapi.responses import Response, StreamingResponse, FileResponse
from typing import Optional
from pathlib import Path
from ..database import get_db
from .. import config

router = APIRouter(prefix="/api", tags=["video"])


@router.get("/videos")
async def list_videos():
    """
    List all available videos
    """
    db = get_db()
    videos = db.list_videos()

    return {
        "count": len(videos),
        "videos": [
            {
                "name": v["video_name"],
                "path": v.get("video_path"),
                "duration": v.get("video_duration"),
                "width": v.get("width"),
                "height": v.get("height"),
                "fps": v.get("fps")
            }
            for v in videos
        ]
    }


@router.get("/video/{video_name}/stream")
async def stream_video(video_name: str, request: Request):
    """
    Stream video file with Range support for seeking
    """
    db = get_db()
    video_info = db.get_video_status(video_name)

    if not video_info:
        raise HTTPException(404, f"Video not found: {video_name}")

    video_path = video_info.get("video_path", f"video/{video_name}.mp4")

    # Make path absolute
    if not os.path.isabs(video_path):
        video_path = config.BASE_DIR / video_path

    if not os.path.exists(video_path):
        raise HTTPException(404, f"Video file not found: {video_path}")

    file_size = os.path.getsize(video_path)

    # Handle Range header for seeking
    range_header = request.headers.get("range")

    if range_header:
        # Parse range
        range_str = range_header.replace("bytes=", "")
        start_str, end_str = range_str.split("-")
        start = int(start_str)
        end = int(end_str) if end_str else file_size - 1

        # Limit chunk size
        chunk_size = min(end - start + 1, 1024 * 1024)  # 1MB max
        end = start + chunk_size - 1

        def iter_file():
            with open(video_path, "rb") as f:
                f.seek(start)
                yield f.read(chunk_size)

        return StreamingResponse(
            iter_file(),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk_size)
            }
        )

    # Full file response
    return FileResponse(
        video_path,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"}
    )


@router.get("/video/{video_name}/frame")
async def get_frame(
    video_name: str,
    t: float = Query(..., description="Timestamp in seconds")
):
    """
    Get a single frame from video at timestamp
    """
    db = get_db()
    video_info = db.get_video_status(video_name)

    if not video_info:
        raise HTTPException(404, f"Video not found: {video_name}")

    video_path = video_info.get("video_path", f"video/{video_name}.mp4")

    # Make path absolute
    if not os.path.isabs(video_path):
        video_path = config.BASE_DIR / video_path

    if not os.path.exists(video_path):
        raise HTTPException(404, f"Video file not found: {video_path}")

    # Extract frame
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = int(t * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(400, f"Cannot read frame at {t}s")

    # Resize for web
    max_width = 800
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))

    # Encode to JPEG
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "max-age=3600"}
    )
