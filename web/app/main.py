"""
Video Tagging Web Application
FastAPI application for video search and face gallery
"""
import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from . import config
from .database import get_db
from .api import search, persons, video

# Initialize FastAPI
app = FastAPI(
    title="Video Tagging",
    description="Video search and face gallery",
    version="1.0.0"
)

# Templates
BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Static files
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include API routers
app.include_router(search.router)
app.include_router(persons.router)
app.include_router(video.router)


# === HTML Pages ===

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page - search"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request):
    """Search page"""
    return templates.TemplateResponse("search.html", {"request": request})


@app.get("/persons", response_class=HTMLResponse)
async def persons_page(request: Request, video: str = None):
    """Persons gallery page"""
    db = get_db()
    videos = db.list_videos()

    return templates.TemplateResponse("persons.html", {
        "request": request,
        "videos": [v["video_name"] for v in videos],
        "selected_video": video or (videos[0]["video_name"] if videos else None)
    })


@app.get("/person/{cluster_id}", response_class=HTMLResponse)
async def person_detail_page(
    request: Request,
    cluster_id: str,
    video: str = None,
    clustering: str = "onthefly"
):
    """Person detail page"""
    return templates.TemplateResponse("person_detail.html", {
        "request": request,
        "cluster_id": cluster_id,
        "video": video,
        "clustering": clustering
    })


@app.get("/player/{video_name}", response_class=HTMLResponse)
async def player_page(request: Request, video_name: str, t: float = 0):
    """Video player page"""
    db = get_db()
    video_info = db.get_video_status(video_name)

    return templates.TemplateResponse("player.html", {
        "request": request,
        "video_name": video_name,
        "video_info": video_info,
        "start_time": t
    })


@app.get("/face-search", response_class=HTMLResponse)
async def face_search_page(request: Request):
    """Face search page (placeholder)"""
    return templates.TemplateResponse("face_search.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    db = get_db()
    stats = db.get_collection_stats()

    return {
        "status": "ok",
        "database": stats["database"],
        "video_files": stats["video_files_count"],
        "video_segments": stats["video_segments_count"]
    }


@app.on_event("startup")
async def startup_event():
    """Application startup"""
    print("Starting Video Tagging Web Application...")
    print(f"Video directory: {config.VIDEO_DIR}")

    # Test database connection
    db = get_db()
    stats = db.get_collection_stats()
    print(f"Database connected: {stats['video_files_count']} videos, {stats['video_segments_count']} segments")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=config.APP_HOST,
        port=config.APP_PORT,
        reload=config.DEBUG
    )
