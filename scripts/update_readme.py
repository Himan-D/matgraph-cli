#!/usr/bin/env python3
"""Dynamic README updater — run on release, keeps version + verticals in sync."""
import re
from pathlib import Path
try:
    import tomllib
    ver = tomllib.loads(Path("pyproject.toml").read_bytes().decode())["project"]["version"]
except Exception:
    m = re.search(r'version\s*=\s*"([^"]+)"', Path("pyproject.toml").read_text())
    ver = m.group(1) if m else "unknown"
txt = Path("README.md").read_text()
txt = re.sub(r"> \*\*Current: v[^*]+\*\*", f"> **Current: v{ver}**", txt)
txt = re.sub(r"### Deep Learning Models \([^)]+\)", f"### Deep Learning Models ({ver} — six FMMs inc. OMat24)", txt)
Path("README.md").write_text(txt)
print(f"README updated to v{ver}")
