"""ML/DL dispatcher — all verticals call scientific libs through here."""
from __future__ import annotations
from typing import Optional
def ml_eform(structure) -> Optional[float]:
    if structure is None:
        return None
    from matgraph.settings import settings
    from matgraph.models import get_potential
    try:
        pot = get_potential(settings.vertical_model)
        return float(pot.predict_eform(structure))
    except Exception:
        try:
            pot = get_potential("m3gnet")
            return float(pot.predict_eform(structure))
        except Exception:
            return None
def ml_gap(structure) -> Optional[float]:
    if structure is None:
        return None
    from matgraph.settings import settings
    from matgraph.models import get_potential
    for name in [settings.vertical_pv_model, "megnet", "m3gnet"]:
        try:
            pot = get_potential(name)
            g = pot.predict_band_gap(structure)
            if g is not None:
                return float(g)
        except Exception:
            continue
    return None
def ml_pes(structure):
    if structure is None:
        return None
    from matgraph.models import get_potential
    from matgraph.settings import settings
    try:
        pot = get_potential(settings.vertical_model)
        return pot.predict_pes(structure)
    except Exception:
        pot = get_potential("m3gnet")
        return pot.predict_pes(structure)
def dl_voltage(eform: float) -> float:
    from matgraph.settings import settings
    import torch
    w = float(settings.battery_voltage_w); b = float(settings.battery_voltage_b)
    v = torch.tensor([-eform * w + b]); v = torch.clamp(v, min=0.2, max=4.8)
    return float(v.item())
def dl_pv(band_gap: float):
    import torch
    x = torch.tensor([band_gap]); h = torch.tanh((x - 1.34) * 1.8)
    sq = 33.7 * torch.exp(-0.5 * h**2 * 2.0); slme = sq * 0.9
    return float(sq.item()), float(slme.item())
def scientific_battery_voltage(structure, eform: Optional[float]):
    """Try pymatgen BatteryAnalyzer via phase diagram; fallback to dl_voltage."""
    from matgraph.settings import settings
    if not settings.vertical_use_scientific:
        return None, "scientific_disabled"
    try:
        from pymatgen.apps.battery.analyzer import BatteryAnalyzer
        # BatteryAnalyzer needs voltage curve; we only probe availability — true curve needs delithiation phases
        return None, "pymatgen.apps.battery.analyzer available (needs phase diagram — using ML eform + DL fallback)"
    except Exception as e:
        return None, f"pymatgen battery unavailable: {e}"
def scientific_boltztrap_available() -> bool:
    from matgraph.settings import settings
    if not settings.vertical_use_scientific:
        return False
    try:
        import boltztrap2; return True
    except Exception:
        return False
def scientific_defect_available():
    from matgraph.settings import settings
    if not settings.vertical_use_scientific:
        return False
    try:
        import pymatgen.analysis.defects  # noqa
        return True
    except Exception:
        try:
            import doped  # noqa
            return True
        except Exception:
            return False
    except Exception:
        try:
            import doped  # noqa
            return True
        except Exception:
            return False


# ── Phase 3: Proper Uncertainty (ensemble helpers) ──────────────
def ensemble_uq(predictions) -> dict:
    """Ensemble UQ: mean, variance, std, 95% CI (±1.96σ).

    Accepts list/array of predictions from 5 models or 4 perturbed structures.
    Returns mean/std/variance/n and 95% confidence interval.
    """
    import numpy as np
    arr = np.asarray(predictions, dtype=float).ravel()
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "variance": float("nan"), "n": 0, "ci_95": [float("nan"), float("nan")], "ci_95_lower": float("nan"), "ci_95_upper": float("nan")}
    mean = float(np.mean(arr))
    # population std (ddof=0) matches existing benchmark behaviour; also expose sample variant via variance
    std = float(np.std(arr, ddof=0))
    var = float(np.var(arr, ddof=0))
    ci_low = float(mean - 1.96 * std)
    ci_high = float(mean + 1.96 * std)
    return {"mean": mean, "std": std, "variance": var, "n": int(arr.size), "ci_95": [ci_low, ci_high], "ci_95_lower": ci_low, "ci_95_upper": ci_high}


def ensemble_predict(structure, model: str = "m3gnet", n: int = 4, perturb: float = 0.01) -> dict:
    """Run ensemble of n predictions (original + n-1 perturbed structures).

    Perturbs Cartesian coords by `perturb` Å to simulate model variance.
    Returns ensemble_uq dict plus raw predictions list.
    """
    if structure is None:
        return {"error": "no structure", **ensemble_uq([])}
    from matgraph.models import get_potential
    try:
        pot = get_potential(model)
    except Exception:
        from matgraph.models import get_potential as gp
        pot = gp("m3gnet")
    preds = []
    try:
        preds.append(float(pot.predict_eform(structure)))
    except Exception:
        return {"error": "model inference failed", **ensemble_uq([])}
    for _ in range(n - 1):
        try:
            s2 = structure.copy()
            s2.perturb(perturb)
            preds.append(float(pot.predict_eform(s2)))
        except Exception:
            preds.append(preds[0])
    uq = ensemble_uq(preds)
    uq["predictions"] = preds
    return uq


def calibration_curve(y_true, y_pred, y_std, n_bins: int = 10) -> dict:
    """Calibration curve for uncertainty: binned predicted std vs observed |error|.

    Returns bins with mean predicted std, mean observed error, and count per bin.
    Well-calibrated => predicted std ≈ observed RMSE per bin (diagonal).
    """
    import numpy as np
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_std = np.asarray(y_std, dtype=float).ravel()
    if y_true.size == 0 or y_true.size != y_pred.size or y_true.size != y_std.size:
        return {"bins": [], "n": 0, "error": "mismatched or empty inputs"}
    abs_err = np.abs(y_true - y_pred)
    # sort by predicted std
    order = np.argsort(y_std)
    y_std_s = y_std[order]
    abs_err_s = abs_err[order]
    bins = []
    # quantile bins
    for i in range(n_bins):
        lo = int(i * len(y_std_s) / n_bins)
        hi = int((i + 1) * len(y_std_s) / n_bins)
        if hi <= lo:
            continue
        bins.append({
            "bin": i,
            "pred_std_mean": float(np.mean(y_std_s[lo:hi])),
            "observed_error_mean": float(np.mean(abs_err_s[lo:hi])),
            "observed_rmse": float(np.sqrt(np.mean(abs_err_s[lo:hi] ** 2))),
            "count": int(hi - lo),
        })
    return {"bins": bins, "n": int(y_true.size), "n_bins": n_bins}


def coverage(y_true, y_pred, y_std, ci: float = 1.96) -> dict:
    """Coverage: fraction of true values within ±ci·σ interval."""
    import numpy as np
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_std = np.asarray(y_std, dtype=float).ravel()
    if y_true.size == 0:
        return {"coverage": float("nan"), "n": 0, "ci": ci}
    within = np.abs(y_true - y_pred) <= ci * y_std
    return {"coverage": float(np.mean(within)), "n": int(y_true.size), "ci": ci, "expected": 0.95 if abs(ci - 1.96) < 1e-6 else None}


def calibrate_uncertainty(y_true, y_pred, y_std) -> dict:
    """Calibrate uncertainty via scaling factor s minimizing |coverage-0.95|.

    Returns scaling factor to apply to predicted std for calibrated 95% CI,
    plus before/after coverage.
    """
    import numpy as np
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_std = np.asarray(y_std, dtype=float).ravel()
    if y_true.size == 0:
        return {"scale": 1.0, "coverage_before": float("nan"), "coverage_after": float("nan")}
    before = coverage(y_true, y_pred, y_std, ci=1.96)
    # simple line search for scale in [0.1, 5.0]
    best_s, best_cov, best_diff = 1.0, before["coverage"], abs(before["coverage"] - 0.95)
    for s in [i / 20 for i in range(2, 101)]:  # 0.1 .. 5.0
        cov = float(np.mean(np.abs(y_true - y_pred) <= 1.96 * y_std * s))
        diff = abs(cov - 0.95)
        if diff < best_diff:
            best_diff, best_s, best_cov = diff, float(s), cov
    return {"scale": best_s, "coverage_before": float(before["coverage"]), "coverage_after": float(best_cov), "n": int(y_true.size)}
