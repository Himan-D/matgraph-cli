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
    """Ensemble UQ: mean, variance, std, 95% CI (±1.96σ). Supports 4 perturbed or 5-model ensembles."""
    import numpy as np
    arr = np.asarray(predictions, dtype=float).ravel()
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "variance": float("nan"), "n": 0, "ci_95": [float("nan"), float("nan")], "ci_95_lower": float("nan"), "ci_95_upper": float("nan")}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=0))
    var = float(np.var(arr, ddof=0))
    ci_low = float(mean - 1.96 * std)
    ci_high = float(mean + 1.96 * std)
    return {"mean": mean, "std": std, "variance": var, "n": int(arr.size), "ci_95": [ci_low, ci_high], "ci_95_lower": ci_low, "ci_95_upper": ci_high}


def calibration_curve(y_true, y_pred, y_std, n_bins: int = 10) -> Dict[str, Any]:
    """Calibration curve: binned predicted std vs observed |error|."""
    import numpy as np
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_std = np.asarray(y_std, dtype=float).ravel()
    if y_true.size == 0 or y_true.size != y_pred.size or y_true.size != y_std.size:
        return {"bins": [], "n": 0, "error": "mismatched or empty inputs"}
    abs_err = np.abs(y_true - y_pred)
    order = np.argsort(y_std)
    y_std_s, abs_err_s = y_std[order], abs_err[order]
    bins = []
    for i in range(n_bins):
        lo = int(i * len(y_std_s) / n_bins)
        hi = int((i + 1) * len(y_std_s) / n_bins)
        if hi <= lo:
            continue
        bins.append({"bin": i, "pred_std_mean": float(np.mean(y_std_s[lo:hi])), "observed_error_mean": float(np.mean(abs_err_s[lo:hi])), "observed_rmse": float(np.sqrt(np.mean(abs_err_s[lo:hi] ** 2))), "count": int(hi - lo)})
    return {"bins": bins, "n": int(y_true.size), "n_bins": n_bins}


def coverage(y_true, y_pred, y_std, ci: float = 1.96) -> Dict[str, Any]:
    """Coverage: fraction within ±ci·σ (0.95 expected for ci=1.96)."""
    import numpy as np
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_std = np.asarray(y_std, dtype=float).ravel()
    if y_true.size == 0:
        return {"coverage": float("nan"), "n": 0, "ci": ci}
    within = np.abs(y_true - y_pred) <= ci * y_std
    return {"coverage": float(np.mean(within)), "n": int(y_true.size), "ci": ci, "expected": 0.95 if abs(ci - 1.96) < 1e-6 else None}


def calibrate_uncertainty(y_true, y_pred, y_std) -> Dict[str, Any]:
    """Find scaling factor s for σ so coverage ≈ 0.95."""
    import numpy as np
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_std = np.asarray(y_std, dtype=float).ravel()
    if y_true.size == 0:
        return {"scale": 1.0, "coverage_before": float("nan"), "coverage_after": float("nan")}
    before = coverage(y_true, y_pred, y_std, ci=1.96)
    best_s, best_cov, best_diff = 1.0, before["coverage"], abs(before["coverage"] - 0.95)
    for s in [i / 20 for i in range(2, 101)]:
        cov = float(np.mean(np.abs(y_true - y_pred) <= 1.96 * y_std * s))
        diff = abs(cov - 0.95)
        if diff < best_diff:
            best_diff, best_s, best_cov = diff, float(s), cov
    return {"scale": best_s, "coverage_before": float(before["coverage"]), "coverage_after": float(best_cov), "n": int(y_true.size)}
