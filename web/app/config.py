"""
Application configuration
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # video_tagging_db/
VIDEO_DIR = BASE_DIR / "video"

# Database
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "video_tagging_db")
DB_USER = os.getenv("DB_USER", "video_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "video_pass_2024")

# App settings
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", 8000))
