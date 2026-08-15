from matgraph.core import substitute_material
from matgraph.ga import CrystalGA

def diffusion_generate(chemistry: str, count: int) -> list[str]:
    """Real diffusion only — no silent heuristic fallback. Requires mattergen/diffcsp/cdvae checkpoint."""
    import os
    from matgraph.settings import settings
    from matgraph.exceptions import ModelInferenceError
    model = os.getenv("MATGRAPH_DIFFUSION_MODEL", getattr(settings, "diffusion_model", "auto")).lower()
    # Explicit model request must exist
    if model in ("mattergen","matter-gen"):
        try:
            import mattergen  # type: ignore
            gen = getattr(mattergen, "generate", None) or getattr(mattergen, "MatterGen", None)
            if gen is None:
                raise ImportError("mattergen.generate not found")
            raise ModelInferenceError("MatterGen checkpoint/runtime not configured — set mattergen checkpoint and MATGRAPH_DIFFUSION_MODEL=mattergen; see docs/models.md")
        except ModelInferenceError:
            raise
        except Exception as e:
            raise ModelInferenceError(f"MatterGen not installed: {e} — pip install matgraph-cli[diffusion]") from e
    if model in ("diffcsp","diff-csp"):
        try:
            import diffcsp  # type: ignore
            if not hasattr(diffcsp, "DiffCSP"):
                raise ImportError("diffcsp.DiffCSP not found")
            raise ModelInferenceError("DiffCSP checkpoint not configured — MATGRAPH_DIFFUSION_MODEL=diffcsp requires checkpoint; see docs/models.md")
        except ModelInferenceError:
            raise
        except Exception as e:
            raise ModelInferenceError(f"DiffCSP not installed: {e} — pip install matgraph-cli[diffusion]") from e
    if model == "cdvae":
        try:
            import cdvae  # type: ignore
            if not hasattr(cdvae, "CDVAE"):
                raise ImportError("cdvae.CDVAE not found")
            raise ModelInferenceError("CDVAE checkpoint not configured — MATGRAPH_DIFFUSION_MODEL=cdvae requires checkpoint")
        except ModelInferenceError:
            raise
        except Exception as e:
            raise ModelInferenceError(f"CDVAE not installed: {e} — pip install matgraph-cli[diffusion]") from e
    # auto: fail loudly — never return random chemistry
    raise ModelInferenceError(
        f"Real diffusion unavailable (MATGRAPH_DIFFUSION_MODEL={model}). Install one of: pip install matgraph-cli[diffusion] + mattergen/diffcsp/cdvae checkpoint. "
        "Refusing to return heuristic random formulas — see audit fix."
    )

__all__ = ["substitute_material","CrystalGA","diffusion_generate"]
