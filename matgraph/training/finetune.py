"""Finetune — real training when ml deps present, else honest error (no fake metrics)."""
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path

def finetune(data_path: str, base: str = "chgnet", epochs: int = 5, project: str = "finetune") -> Dict[str,Any]:
    """Fine-tune FMM on user's DFT data. Requires torch + data; never invents loss/mae."""
    from matgraph.exceptions import ModelInferenceError
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"No data at {data_path}: need CSV or CIF dir")
    # Count data honestly
    n = 0
    if p.is_file() and p.suffix==".csv":
        try:
            import pandas as pd
            n = len(pd.read_csv(p))
        except Exception as e:
            raise ModelInferenceError(f"Cannot read CSV {data_path}: {e}") from e
    elif p.is_dir():
        n = len(list(p.glob("*.cif")) + list(p.glob("*.vasp")) + list(p.glob("*.json")))
    if n==0:
        raise ValueError(f"No training samples at {data_path}")
    # Real training requires torch + matgl; if missing, fail loudly — do not simulate
    try:
        import torch  # noqa
        import matgl  # noqa
    except Exception as e:
        raise ModelInferenceError(f"Real fine-tuning requires torch+matgl — pip install matgraph-cli[ml]: {e}") from e
    # Placeholder for real training loop — must be implemented with actual model, not random
    raise ModelInferenceError(
        "Real fine-tuning not yet wired for this base — training loop must use actual CHGNet/M3GNet/MEGNet weights, not simulated loss. "
        "This stub now fails honestly instead of writing fake JSON .pt. Implement via matgl + pytorch-lightning and re-enable."
    )

def simulate_finetune(data_path: str, base: str = "chgnet", epochs: int = 5, project: str = "finetune") -> Dict[str,Any]:
    """Demo helper that simulates finetuning metrics — explicitly NOT real training."""
    import json, random, time
    from matgraph.tracking import init
    from .registry import register_model
    p = Path(data_path); n = 2
    if p.is_file():
        try:
            import pandas as pd; n = len(pd.read_csv(p))
        except Exception:
            n = 2
    run = init(project=project, name=f"{base}-simulate", config={"base":base,"epochs":epochs,"data":str(p),"n":n})
    metrics = {}
    for ep in range(1, epochs+1):
        loss = 0.5 * (0.9 ** ep) + random.uniform(-0.02,0.02)
        mae = 0.08 * (0.95 ** ep) + random.uniform(-0.005,0.005)
        run.log({"epoch":ep, "loss":loss, "mae":mae, "n":n})
        metrics = {"loss":loss, "mae":mae, "r2": 0.85 + 0.1*(1 - 0.9**ep)}
    artifact_path = str(Path(f"/tmp/matgraph_simulate_{base}_{int(time.time())}.json"))
    Path(artifact_path).write_text(json.dumps({"base":base,"epochs":epochs,"n":n, **metrics, "simulated": True}))
    run.log_artifact(artifact_path, type="model")
    mid = register_model(name=f"{base}-simulated", base=base, dataset=str(p), metrics=metrics, artifact_path=artifact_path)
    run.finish()
    return {"model_id": mid, "artifact": artifact_path, "metrics": metrics, "n": n, "simulated": True}
