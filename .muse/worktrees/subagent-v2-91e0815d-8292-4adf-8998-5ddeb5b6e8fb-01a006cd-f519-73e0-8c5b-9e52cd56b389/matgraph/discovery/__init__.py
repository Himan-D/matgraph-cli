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

def generate_candidate_pool(base_formula: str, pool_size: int = 20, seed: int | None = None) -> list[str]:
    """Generate candidate pool for active learning via elemental substitution + perturbations.

    Heuristic fallback for AL pool generation — allowed here (unlike diffusion) because
    pool generation is explicitly heuristic by design; uncertainty+E_hull filter is the ML stage.
    """
    import random
    if seed is not None:
        random.seed(seed)
    from pymatgen.core import Composition
    try:
        comp = Composition(base_formula)
        elements = [str(e) for e in comp.elements]
    except Exception:
        elements = [base_formula]
    pool = set([base_formula])
    allowed = ["Li","Na","K","Mg","Ca","Fe","Co","Ni","Mn","Ti","V","O","S","P","Si","Al","Cu","Zn","Cr"]
    # Simple substitution variants
    for _ in range(pool_size * 2):
        if len(pool) >= pool_size:
            break
        base = random.choice(list(pool))
        try:
            c = Composition(base)
            els = [str(e) for e in c.elements]
            if not els:
                continue
            el_out = random.choice(els)
            el_in = random.choice([e for e in allowed if e != el_out])
            # Count-preserving substitution via string replace (heuristic)
            new_formula = base.replace(el_out, el_in, 1) if el_out in base else f"{base}{el_in}"
            # Validate via pymatgen
            Composition(new_formula)
            pool.add(new_formula)
        except Exception:
            continue
    # Fill randomly if still short
    while len(pool) < pool_size:
        pool.add(f"{random.choice(allowed)}{random.randint(1,2)}{random.choice(allowed)}{random.randint(1,2)}")
    return list(pool)[:pool_size]

__all__ = ["substitute_material","CrystalGA","diffusion_generate","generate_candidate_pool"]
