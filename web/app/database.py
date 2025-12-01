"""
Database connection for web application
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from db_manager_postgres import PostgresDBManager
from . import config

# Global database instance
_db: PostgresDBManager = None


def get_db() -> PostgresDBManager:
    """Get database connection (singleton)"""
    global _db
    if _db is None:
        _db = PostgresDBManager(
            host=config.DB_HOST,
            port=config.DB_PORT,
            database=config.DB_NAME,
            user=config.DB_USER,
            password=config.DB_PASSWORD
        )
    return _db
