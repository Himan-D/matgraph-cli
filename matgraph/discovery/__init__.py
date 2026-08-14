from matgraph.core import substitute_material
from matgraph.ga import CrystalGA

def diffusion_generate(chemistry: str, count: int) -> list[str]:
    """CDVAE/diffusion stub: heuristic + lattice sampling; replace with real cdvae when installed."""
    import random
    elems = [e.strip() for e in chemistry.split("-") if e.strip()]
    out = []
    for _ in range(count):
        out.append("".join(f"{e}{random.randint(1,3)}" for e in random.sample(elems, k=min(3, len(elems)))))
    return out

__all__ = ["substitute_material","CrystalGA","diffusion_generate"]
