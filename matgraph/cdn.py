"""
Zero-config caching layer for MatGraph.

Uses Python's built-in sqlite3 for a fast, persistent, local cache.
No external services, no credentials, no cost.
Repeated queries return instantly without hitting the Materials Project API.

Cache location: ~/.matgraph_cache/cache.db
Default TTL: 1 hour (3600 seconds)
"""

import os
import json
import hashlib
import time
import sqlite3
from pathlib import Path
from typing import Optional, Any

def _cache_dir() -> Path:
    from matgraph.settings import settings, cache_db_path
    return settings.cache_dir

def _cache_db() -> Path:
    from matgraph.settings import cache_db_path
    return cache_db_path()

CACHE_DIR = _cache_dir()
CACHE_DB = _cache_db()


def _init_db() -> sqlite3.Connection:
    """Initialize the SQLite cache database."""
    from matgraph.settings import settings
    db = _cache_db()
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    # WAL for concurrency, no hardcode
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        pass
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def _get_cache_key(prefix: str, **kwargs) -> str:
    """Generate a deterministic cache key from query parameters."""
    raw = json.dumps(kwargs, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{prefix}:{digest}"


def cache_get(prefix: str, ttl: Optional[int] = None, **kwargs) -> Optional[Any]:
    """
    Retrieve cached result if it exists and is within TTL.
    TTL comes from settings.get_ttl(prefix) if not passed.
    """
    from matgraph.settings import get_ttl
    if ttl is None:
        ttl = get_ttl(prefix)
    key = _get_cache_key(prefix, **kwargs)
    try:
        conn = _init_db()
        row = conn.execute(
            "SELECT value, created_at FROM cache WHERE key = ?", (key,)
        ).fetchone()
        conn.close()

        if row is None:
            return None

        value, created_at = row
        if (time.time() - created_at) > ttl:
            return None  # expired

        return json.loads(value)
    except Exception:
        return None


def cache_put(prefix: str, data: Any, **kwargs):
    """Write result to cache."""
    key = _get_cache_key(prefix, **kwargs)
    try:
        conn = _init_db()
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(data, default=str), time.time()),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # caching is best-effort, never break the pipeline


def cache_clear():
    """Clear the entire cache."""
    try:
        conn = _init_db()
        conn.execute("DELETE FROM cache")
        conn.commit()
        conn.close()
    except Exception:
        pass


def cache_stats() -> dict:
    """Return cache statistics."""
    try:
        from matgraph.settings import cache_db_path
        db = cache_db_path()
        conn = _init_db()
        total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        size_bytes = db.stat().st_size if db.exists() else 0
        conn.close()
        return {
            "entries": total,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "location": str(db),
        }
    except Exception:
        from matgraph.settings import cache_db_path
        return {"entries": 0, "size_mb": 0, "location": str(cache_db_path())}
