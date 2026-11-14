#!/usr/bin/env python3
"""CI guard: prove the base install has no heavy dependencies.

Fails if any heavy-stack module is importable (do NOT import matgraph here —
its meta-path finder would mask the check by design).
"""

HEAVY = ["torch", "pymatgen", "mp_api", "matgl", "ase", "gradio", "wandb", "fastapi", "strawberry"]

failed = []
for name in HEAVY:
    try:
        mod = __import__(name)
    except ImportError:
        continue
    if getattr(mod, "__file__", None) is None:
        # Namespace shadow (e.g. ./wandb logs dir on sys.path from an
        # editable install), not a real installed package.
        continue
    failed.append(name)

if failed:
    raise SystemExit(f"slim env polluted — these must stay extras: {sorted(failed)}")
print(f"slim OK: none of {len(HEAVY)} heavy modules importable")
