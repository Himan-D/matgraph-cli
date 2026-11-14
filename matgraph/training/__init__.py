"""Training — finetune (needs [ml]) + registry (stdlib-only, works on slim).

Imports are lazy (PEP 562) so `from matgraph.training.registry import ...`
works on the slim base install without pulling in torch via finetune.
"""

__all__ = ["finetune", "register_model", "list_models", "get_model_artifact"]


def __getattr__(name: str):
    if name == "finetune":
        from .finetune import finetune

        return finetune
    if name in ("register_model", "list_models", "get_model_artifact"):
        from . import registry as _registry

        return getattr(_registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
