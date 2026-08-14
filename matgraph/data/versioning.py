"""DVC-like dataset versioning — local, no server."""
from __future__ import annotations
import hashlib, json, time, sqlite3
from pathlib import Path
from typing import List, Dict

def _db() -> Path:
    from matgraph.tracking.store import _dir
    return _dir() / "datasets.db"

def _init():
    p=_db()
    p.parent.mkdir(parents=True, exist_ok=True)
    import sqlite3
    conn=sqlite3.connect(str(p))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, path TEXT, hash TEXT, version TEXT, created_at REAL, meta TEXT)""")
    conn.commit()
    conn.close()

def version_dataset(path: str, meta: dict | None=None) -> str:
    _init()
    p=Path(path)
    h=hashlib.sha256()
    if p.is_file():
        with open(p,"rb") as f:
            for c in iter(lambda: f.read(8192), b""):
                h.update(c)
    elif p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.is_file():
                h.update(f.name.encode())
                with open(f,"rb") as fh:
                    h.update(fh.read(8192))
    digest=h.hexdigest()[:12]
    vid=f"v{int(time.time())%100000}-{digest}"
    import sqlite3, json
    conn=sqlite3.connect(str(_db()))
    conn.execute("INSERT INTO datasets (id, path, hash, version, created_at, meta) VALUES (?,?,?,?,?,?)",
                 (vid, str(p), digest, vid, time.time(), json.dumps(meta or {})))
    conn.commit()
    conn.close()
    return vid

def list_datasets():
    _init()
    import sqlite3, json
    conn=sqlite3.connect(str(_db()))
    rows=conn.execute("SELECT id, path, hash, version, created_at, meta FROM datasets ORDER BY created_at DESC").fetchall()
    conn.close()
    return [{"id":r[0],"path":r[1],"hash":r[2],"version":r[3],"created_at":r[4],"meta":json.loads(r[5])} for r in rows]
