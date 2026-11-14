"""Helpful errors for optional dependency groups.

The base ``pip install matgraph-cli`` is intentionally slim (CLI + cache +
demo mode). Heavy stacks live in extras::

    pip install "matgraph-cli[mp]"   # Materials Project data (pymatgen, mp-api)
    pip install "matgraph-cli[ml]"   # ML potentials (torch, matgl, ase, ...)
    pip install "matgraph-cli[api]"  # GraphQL/REST server
    pip install "matgraph-cli[ui]"   # Gradio dashboard

This module installs a meta-path finder (last in ``sys.meta_path``, so it only
fires when normal imports fail) that turns ``ModuleNotFoundError: No module
named 'torch'`` into an actionable message telling the user which extra to
install. The raised error subclasses ``ImportError``, so existing
``except ImportError`` / ``except Exception`` handlers keep working.
"""

from __future__ import annotations

import importlib.abc
import sys

# top-level module -> extra name (without the "matgraph-cli[...]" wrapper)
EXTRA_FOR_MODULE = {
    # data layer
    "mp_api": "mp",
    "pymatgen": "mp",
    "emmet": "mp",
    # ML layer
    "torch": "ml",
    "matgl": "ml",
    "ase": "ml",
    "sklearn": "ml",
    "huggingface_hub": "ml",
    "dgl": "ml",
    # serving / UI / tracking
    "fastapi": "api",
    "strawberry": "api",
    "uvicorn": "api",
    "gradio": "ui",
    "wandb": "tracking",
    # data formats
    "pandas": "parquet",
    "pyarrow": "parquet",
    "polars": "polars",
    # research extras
    "pennylane": "quantum",
    "fairchem": "omat24",
    "BoltzTraP2": "thermo",
}


class MissingExtraError(ImportError):
    """ImportError with the exact `pip install` command to fix it."""


def hint_for(module: str) -> str | None:
    extra = EXTRA_FOR_MODULE.get(module.split(".")[0])
    if extra is None:
        return None
    return f'MatGraph needs the "{extra}" extra for this (missing module: {module}). Install it with: pip install "matgraph-cli[{extra}]"'


def require(module: str, extra: str | None = None):
    """Import ``module`` or raise MissingExtraError with an install hint."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:
        msg = hint_for(module)
        if msg is None:
            if extra is None:
                raise
            msg = f'MatGraph needs the "{extra}" extra for this. Install it with: pip install "matgraph-cli[{extra}]"'
        raise MissingExtraError(msg) from e


class _ExtraHintLoader(importlib.abc.Loader):
    def __init__(self, fullname: str):
        self.fullname = fullname

    def create_module(self, spec):
        raise MissingExtraError(hint_for(self.fullname))

    def exec_module(self, module):
        raise MissingExtraError(hint_for(self.fullname))  # pragma: no cover


class _ExtraHintFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder, appended LAST: only reached when the module is truly missing."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in EXTRA_FOR_MODULE:
            import importlib.machinery

            return importlib.machinery.ModuleSpec(fullname, _ExtraHintLoader(fullname))
        return None


def install() -> None:
    """Install the hint finder once (idempotent). Called on `import matgraph`."""
    if not any(isinstance(f, _ExtraHintFinder) for f in sys.meta_path):
        sys.meta_path.append(_ExtraHintFinder())
