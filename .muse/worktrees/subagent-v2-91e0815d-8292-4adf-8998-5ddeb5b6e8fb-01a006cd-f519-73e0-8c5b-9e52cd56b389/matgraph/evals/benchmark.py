from __future__ import annotations
from typing import List, Dict, Any
import random

def time_split_benchmark(results: List[Dict[str, Any]], test_size: float = 0.2, time_key: str = "material_id") -> Dict[str, Any]:
    """Time-split aware eval: sort by material_id as proxy for discovery time, then tail=test."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    # sort by material_id string as time proxy; real would use mp- time
    pairs = [(r["true_form_energy"], r["predicted_form_energy"]) for r in results if r.get("true_form_energy") is not None and r.get("predicted_form_energy") is not None]
    if len(pairs) < 2:
        return {"error": "not enough pairs", "n": len(pairs)}
    pairs_sorted = sorted(zip(results, pairs), key=lambda x: str(x[0].get(time_key, "")))
    n_test = max(1, int(len(pairs) * test_size))
    test_pairs = [p for _, p in pairs_sorted[-n_test:]]
    y_true = [a for a, _ in test_pairs]
    y_pred = [b for _, b in test_pairs]
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_true, y_pred)
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        import math
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    try:
        r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = float("nan")
    # matbench-genmetrics style: also report precision@stable if eform <0
    stable_true = sum(1 for v in y_true if v < 0)
    stable_pred = sum(1 for v in y_pred if v < 0)
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2), "n_test": len(y_true), "stable_true": stable_true, "stable_pred": stable_pred}

def ensemble_uq(predictions: List[float]) -> Dict[str, float]:
    """Simple ensemble UQ: mean/std over perturbed predictions."""
    import numpy as np
    arr = np.array(predictions, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": len(arr)}
