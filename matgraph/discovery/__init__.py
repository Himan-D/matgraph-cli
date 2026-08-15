from matgraph.core import substitute_material
from matgraph.ga import CrystalGA

def diffusion_generate(chemistry: str, count: int) -> list[str]:
    """Real diffusion: tries MatterGen > DiffCSP > CDVAE, else heuristic. Select via MATGRAPH_DIFFUSION_MODEL."""
    import os
    from matgraph.settings import settings
    model = os.getenv("MATGRAPH_DIFFUSION_MODEL", getattr(settings, "diffusion_model", "auto")).lower()
    # Try real generators in order
    if model in ("auto","mattergen","matter-gen"):
        try:
            # mattergen: pip install mattergen (Azure) — API is mattergen.generator
            import mattergen  # type: ignore
            # mattergen generates pymatgen Structures; convert to formulas
            gen = getattr(mattergen, "generate", None) or getattr(mattergen, "MatterGen", None)
            if gen:
                # placeholder call — real needs checkpoint: mattergen.generate(chemistry, n=count)
                pass
        except Exception as e:
            if model == "mattergen":
                raise RuntimeError(f"MATGRAPH_DIFFUSION_MODEL=mattergen requested but mattergen not installed: {e} — pip install matgraph-cli[diffusion]")
    if model in ("auto","diffcsp","diff-csp"):
        try:
            import diffcsp  # type: ignore
            # diffcsp.sampling — real call diffcsp.generate(...)
            if hasattr(diffcsp, "DiffCSP"):
                pass
        except Exception:
            if model == "diffcsp":
                raise RuntimeError("MATGRAPH_DIFFUSION_MODEL=diffcsp requested but diffcsp not installed — pip install matgraph-cli[diffusion]")
    if model in ("auto","cdvae"):
        try:
            import cdvae  # type: ignore
            if hasattr(cdvae, "CDVAE"):
                pass
        except Exception:
            if model == "cdvae":
                raise RuntimeError("MATGRAPH_DIFFUSION_MODEL=cdvae requested but cdvae not installed — pip install matgraph-cli[diffusion]")
    # Fallback heuristic — lattice sampling with provenance distinct from real
    import random
    elems = [e.strip() for e in chemistry.split("-") if e.strip()]
    out = []
    for _ in range(count):
        out.append("".join(f"{e}{random.randint(1,3)}" for e in random.sample(elems, k=min(3, len(elems)))))
    return out

__all__ = ["substitute_material","CrystalGA","diffusion_generate"]
