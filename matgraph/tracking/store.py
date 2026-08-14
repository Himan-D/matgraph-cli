"""Local store — SQLite + files, like wandb local."""
from __future__ import annotations
import json, time, hashlib, sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any

def _dir() -> Path:
    from matgraph.settings import settings
    import os, tempfile
    p = os.getenv("MATGRAPH_TRACKING_DIR")
    if p:
        return Path(p).expanduser()
    base = settings.cache_dir / "tracking"
    try:
        base.mkdir(parents=True, exist_ok=True)
        return base
    except PermissionError:
        # sandbox fallback
        fb = Path(tempfile.gettempdir()) / "matgraph_tracking"
        fb.mkdir(parents=True, exist_ok=True)
        return fb

def _db() -> Path:
    return _dir() / "runs.db"

def _init():
    d = _dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        import tempfile
        from pathlib import Path
        d = Path(tempfile.gettempdir()) / "matgraph_tracking"
        d.mkdir(parents=True, exist_ok=True)
    db = _db()
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        project TEXT, name TEXT, config TEXT, created_at REAL, updated_at REAL, status TEXT, summary TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS metrics (
        run_id TEXT, step INTEGER, ts REAL, data TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS artifacts (
        run_id TEXT, name TEXT, type TEXT, path TEXT, digest TEXT, created_at REAL
    )""")
    conn.commit()
    conn.close()

def new_id() -> str:
    import secrets
    return secrets.token_hex(4)

def create_run(project: str, name: Optional[str], config: Dict[str,Any]) -> str:
    _init()
    rid = new_id()
    now = time.time()
    conn = sqlite3.connect(str(_db()))
    conn.execute("INSERT INTO runs (id, project, name, config, created_at, updated_at, status, summary) VALUES (?,?,?,?,?,?,?,?)",
                 (rid, project, name or rid, json.dumps(config), now, now, "running", "{}"))
    conn.commit()
    conn.close()
    # create run dir
    (_dir() / rid).mkdir(exist_ok=True)
    return rid

def log_metrics(run_id: str, metrics: Dict[str,Any], step: Optional[int]=None):
    _init()
    conn = sqlite3.connect(str(_db()))
    # auto step = max+1
    if step is None:
        cur = conn.execute("SELECT MAX(step) FROM metrics WHERE run_id=?", (run_id,)).fetchone()[0]
        step = (cur or 0) + 1
    conn.execute("INSERT INTO metrics (run_id, step, ts, data) VALUES (?,?,?,?)",
                 (run_id, step, time.time(), json.dumps(metrics, default=str)))
    # update summary
    row = conn.execute("SELECT summary FROM runs WHERE id=?", (run_id,)).fetchone()
    summary = json.loads(row[0]) if row and row[0] else {}
    summary.update(metrics)
    conn.execute("UPDATE runs SET summary=?, updated_at=? WHERE id=?", (json.dumps(summary), time.time(), run_id))
    conn.commit()
    conn.close()

def log_artifact(run_id: str, path: str, typ: str="dataset"):
    _init()
    p = Path(path)
    digest = ""
    if p.exists():
        h = hashlib.sha256()
        with open(p,"rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        digest = h.hexdigest()[:12]
    conn = sqlite3.connect(str(_db()))
    conn.execute("INSERT INTO artifacts (run_id, name, type, path, digest, created_at) VALUES (?,?,?,?,?,?)",
                 (run_id, p.name, typ, str(p), digest, time.time()))
    conn.commit()
    conn.close()

def finish_run(run_id: str):
    _init()
    conn = sqlite3.connect(str(_db()))
    conn.execute("UPDATE runs SET status=?, updated_at=? WHERE id=?", ("finished", time.time(), run_id))
    conn.commit()
    conn.close()

def list_runs(project: Optional[str]=None) -> List[Dict[str,Any]]:
    _init()
    conn = sqlite3.connect(str(_db()))
    if project:
        rows = conn.execute("SELECT id, project, name, config, created_at, status, summary FROM runs WHERE project=? ORDER BY created_at DESC", (project,)).fetchall()
    else:
        rows = conn.execute("SELECT id, project, name, config, created_at, status, summary FROM runs ORDER BY created_at DESC").fetchall()
    conn.close()
    out=[]
    for r in rows:
        out.append({"id":r[0],"project":r[1],"name":r[2],"config":json.loads(r[3]),"created_at":r[4],"status":r[5],"summary":json.loads(r[6])})
    return out

def get_run(run_id: str) -> Optional[Dict[str,Any]]:
    _init()
    conn = sqlite3.connect(str(_db()))
    row = conn.execute("SELECT id, project, name, config, created_at, status, summary FROM runs WHERE id=?", (run_id,)).fetchone()
    if not row:
        return None
    metrics = conn.execute("SELECT step, ts, data FROM metrics WHERE run_id=? ORDER BY step", (run_id,)).fetchall()
    arts = conn.execute("SELECT name, type, path, digest, created_at FROM artifacts WHERE run_id=?", (run_id,)).fetchall()
    conn.close()
    return {
        "id":row[0],"project":row[1],"name":row[2],"config":json.loads(row[3]),"created_at":row[4],"status":row[5],"summary":json.loads(row[6]),
        "metrics":[{"step":m[0],"ts":m[1],"data":json.loads(m[2])} for m in metrics],
        "artifacts":[{"name":a[0],"type":a[1],"path":a[2],"digest":a[3],"created_at":a[4]} for a in arts]
    }
