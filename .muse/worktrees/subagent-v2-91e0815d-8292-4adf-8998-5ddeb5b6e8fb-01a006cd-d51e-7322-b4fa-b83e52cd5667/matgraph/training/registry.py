"""Model registry — like wandb Artifacts + DVC, local SQLite."""
from __future__ import annotations
import json, time, hashlib, sqlite3
from pathlib import Path
from typing import List, Dict, Any

def _db() -> Path:
    from matgraph.tracking.store import _dir
    return _dir() / "registry.db"

def _init():
    p = _db()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS models (
        id TEXT PRIMARY KEY, name TEXT, version TEXT, base TEXT, dataset TEXT, metrics TEXT, artifact_path TEXT, created_at REAL
    )""")
    conn.commit()
    conn.close()

def register_model(name: str, base: str, dataset: str, metrics: Dict[str,Any], artifact_path: str) -> str:
    _init()
    import secrets
    vid = f"v{int(time.time()) % 100000}"
    mid = f"{name}:{vid}"
    conn = sqlite3.connect(str(_db()))
    conn.execute("INSERT INTO models (id, name, version, base, dataset, metrics, artifact_path, created_at) VALUES (?,?,?,?,?,?,?,?)",
                 (mid, name, vid, base, dataset, json.dumps(metrics), artifact_path, time.time()))
    conn.commit()
    conn.close()
    return mid

def list_models() -> List[Dict[str,Any]]:
    _init()
    conn = sqlite3.connect(str(_db()))
    rows = conn.execute("SELECT id, name, version, base, dataset, metrics, artifact_path, created_at FROM models ORDER BY created_at DESC").fetchall()
    conn.close()
    return [{"id":r[0],"name":r[1],"version":r[2],"base":r[3],"dataset":r[4],"metrics":json.loads(r[5]),"artifact":r[6],"created_at":r[7]} for r in rows]

def get_model_artifact(model_id: str) -> Dict[str,Any] | None:
    _init()
    conn = sqlite3.connect(str(_db()))
    row = conn.execute("SELECT id, name, version, base, dataset, metrics, artifact_path FROM models WHERE id=?", (model_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return {"id":row[0],"name":row[1],"version":row[2],"base":row[3],"dataset":row[4],"metrics":json.loads(row[5]),"artifact":row[6]}
