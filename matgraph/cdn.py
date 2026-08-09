"""
AWS CloudFront CDN integration for MatGraph.

Caches Materials Project API responses and model predictions on S3,
served globally via CloudFront edge locations. This reduces API calls,
speeds up repeated queries, and lowers latency for researchers worldwide.

Usage:
    # Environment variables required:
    #   AWS_ACCESS_KEY_ID
    #   AWS_SECRET_ACCESS_KEY
    #   AWS_REGION (default: us-east-1)
    #   MATGRAPH_S3_BUCKET (default: matgraph-cdn-cache)
    #   MATGRAPH_CDN_URL (your CloudFront distribution URL)
"""

import os
import json
import hashlib
import time
from pathlib import Path
from typing import Optional, Any

LOCAL_CACHE_DIR = Path.home() / ".matgraph_cache"


def _get_cache_key(prefix: str, **kwargs) -> str:
    """Generate a deterministic cache key from query parameters."""
    raw = json.dumps(kwargs, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"{prefix}/{digest}.json"


def _local_cache_path(cache_key: str) -> Path:
    path = LOCAL_CACHE_DIR / cache_key
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _is_local_fresh(path: Path, ttl_seconds: int = 3600) -> bool:
    """Check if local cache file exists and is within TTL."""
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < ttl_seconds


# ---------------------------------------------------------------------------
# Local-only cache (works without AWS credentials)
# ---------------------------------------------------------------------------

def local_get(cache_key: str, ttl: int = 3600) -> Optional[Any]:
    """Retrieve from local disk cache."""
    path = _local_cache_path(cache_key)
    if _is_local_fresh(path, ttl):
        with open(path, "r") as f:
            return json.load(f)
    return None


def local_put(cache_key: str, data: Any):
    """Write to local disk cache."""
    path = _local_cache_path(cache_key)
    with open(path, "w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# S3 + CloudFront layer (optional, requires boto3 + AWS creds)
# ---------------------------------------------------------------------------

def _get_s3_client():
    try:
        import boto3
    except ImportError:
        return None

    region = os.environ.get("AWS_REGION", "us-east-1")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    if not access_key or not secret_key:
        return None

    return boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def _bucket() -> str:
    return os.environ.get("MATGRAPH_S3_BUCKET", "matgraph-cdn-cache")


def cdn_url() -> Optional[str]:
    return os.environ.get("MATGRAPH_CDN_URL")


def s3_get(cache_key: str) -> Optional[Any]:
    """Retrieve a cached object from S3."""
    s3 = _get_s3_client()
    if s3 is None:
        return None
    try:
        obj = s3.get_object(Bucket=_bucket(), Key=cache_key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        return None


def s3_put(cache_key: str, data: Any):
    """Upload a cached object to S3 (auto-served via CloudFront)."""
    s3 = _get_s3_client()
    if s3 is None:
        return
    try:
        s3.put_object(
            Bucket=_bucket(),
            Key=cache_key,
            Body=json.dumps(data).encode("utf-8"),
            ContentType="application/json",
        )
    except Exception:
        pass  # Fail silently; caching is best-effort


# ---------------------------------------------------------------------------
# Unified cache interface (local -> S3/CDN -> origin)
# ---------------------------------------------------------------------------

def cache_get(prefix: str, ttl: int = 3600, **kwargs) -> Optional[Any]:
    """
    Try local cache first, then S3/CDN.
    Returns None on miss so caller can fetch from origin.
    """
    key = _get_cache_key(prefix, **kwargs)

    # 1. Local disk
    hit = local_get(key, ttl)
    if hit is not None:
        return hit

    # 2. S3 / CloudFront
    hit = s3_get(key)
    if hit is not None:
        local_put(key, hit)  # backfill local
        return hit

    return None


def cache_put(prefix: str, data: Any, **kwargs):
    """Write to both local and S3 caches."""
    key = _get_cache_key(prefix, **kwargs)
    local_put(key, data)
    s3_put(key, data)
