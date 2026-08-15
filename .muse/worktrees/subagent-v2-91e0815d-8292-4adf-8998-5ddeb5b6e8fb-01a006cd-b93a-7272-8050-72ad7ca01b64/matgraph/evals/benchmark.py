"""Benchmark engine Phase 2 — splits + metrics.

Splits:
 - random: random shuffle split
 - element: hold out one or more elements (OOD)
 - chemsys / chemical-system-disjoint / system_ood: disjoint chemical systems
 - prototype: split by prototype / crystal system proxy
 - temporal / time_split: sort by material_id as discovery-time proxy
 - ood: union of element + chemsys OOD

Metrics:
 - MAE, RMSE, R2
 - Spearman rho (rank correlation)
 - ROC-AUC (stability classification: e_form < 0)
 - calibration: ECE for stability classification
 - OOD: delta MAE (OOD - ID) and OOD ROC-AUC
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import random
import math


def _get_pairs(results: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    return [(r["true_form_energy"], r["predicted_form_energy"]) for r in results if r.get("true_form_energy") is not None and r.get("predicted_form_energy") is not None]


def _chemical_system(formula: str) -> str:
    """Return canonical chemical system key like 'Fe-Li-O' sorted."""
    # lightweight: parse formula by extracting element symbols via regex
    import re
    # match element symbols: capital + optional lower
    elems = re.findall(r"[A-Z][a-z]?", formula or "")
    # unique sorted
    uniq = sorted(set(elems))
    return "-".join(uniq) if uniq else ""


def _chemsys_of_result(r: Dict[str, Any]) -> str:
    # prefer formula field, fallback to material_id
    formula = r.get("formula") or r.get("formula_pretty") or ""
    if formula:
        return _chemical_system(formula)
    # fallback: naive from material_id
    return str(r.get("material_id", ""))


def _compute_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, Any]:
    """Compute MAE/RMSE/R2/Spearman/ROC-AUC/calibration."""
    n = len(y_true)
    if n == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "spearman": float("nan"), "roc_auc": float("nan"), "ece": float("nan"), "n": 0}
    # try sklearn, else manual
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
        mae = float(mean_absolute_error(y_true, y_pred))
        try:
            rmse = float(mean_squared_error(y_true, y_pred, squared=False))
        except TypeError:
            rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
        try:
            r2 = float(r2_score(y_true, y_pred))
        except Exception:
            r2 = float("nan")
    except Exception:
        # manual fallback
        mae = sum(abs(a - b) for a, b in zip(y_true, y_pred)) / n
        mse = sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n
        rmse = math.sqrt(mse)
        # r2 manual
        mean_true = sum(y_true) / n
        ss_tot = sum((a - mean_true) ** 2 for a in y_true)
        ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    # Spearman
    try:
        from scipy.stats import spearmanr
        spear = float(spearmanr(y_true, y_pred).correlation)
    except Exception:
        try:
            raise ImportError("force manual fallback")
        except Exception:
            # manual rank correlation via sorted ranks
            try:
                # simple spearman via rank differences
                def rankdata(a):
                    sorted_a = sorted((v, i) for i, v in enumerate(a))
                    ranks = [0]*len(a)
                    for rank, (_, idx) in enumerate(sorted_a):
                        ranks[idx] = rank
                    return ranks
                rx = rankdata(y_true)
                ry = rankdata(y_pred)
                # pearson on ranks
                mx = sum(rx)/n
                my = sum(ry)/n
                num = sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
                den = math.sqrt(sum((rx[i]-mx)**2 for i in range(n)) * sum((ry[i]-my)**2 for i in range(n)))
                spear = float(num/den) if den != 0 else float("nan")
            except Exception:
                spear = float("nan")

    # ROC-AUC for stability classification (true label = e_form < 0)
    y_true_stable = [1 if v < 0 else 0 for v in y_true]
    # use predicted stability probability proxy: sigmoid(-y_pred) or simple threshold
    # We'll map predicted e_form to probability of stable via logistic
    def _sigmoid(x):
        try:
            return 1/(1+math.exp(x))
        except OverflowError:
            return 0.0 if x > 0 else 1.0
    y_score = [_sigmoid(v*2) for v in y_pred]  # lower predicted eform -> higher score
    roc_auc = float("nan")
    try:
        if len(set(y_true_stable)) == 2:
            from sklearn.metrics import roc_auc_score
            roc_auc = float(roc_auc_score(y_true_stable, y_score))
        else:
            roc_auc = float("nan")
    except Exception:
        roc_auc = float("nan")

    # Calibration: ECE for stability classification (10 bins)
    ece = float("nan")
    try:
        n_bins = 10
        bins = [[] for _ in range(n_bins)]
        for score, label in zip(y_score, y_true_stable):
            idx = min(int(score * n_bins), n_bins - 1)
            bins[idx].append((score, label))
        ece_val = 0.0
        for b in bins:
            if not b:
                continue
            acc = sum(lbl for _, lbl in b)/len(b)
            conf = sum(s for s, _ in b)/len(b)
            ece_val += abs(acc - conf) * len(b)/n
        ece = float(ece_val)
    except Exception:
        ece = float("nan")

    # OOD not computed here; caller does delta
    stable_true = sum(1 for v in y_true if v < 0)
    stable_pred = sum(1 for v in y_pred if v < 0)
    return {"mae": mae, "rmse": rmse, "r2": r2, "spearman": spear, "roc_auc": roc_auc, "ece": ece, "n": n, "stable_true": stable_true, "stable_pred": stable_pred}


def time_split_benchmark(results: List[Dict[str, Any]], test_size: float = 0.2, time_key: str = "material_id") -> Dict[str, Any]:
    """Time-split aware eval: sort by material_id as proxy for discovery time, then tail=test."""
    pairs = _get_pairs(results)
    if len(pairs) < 2:
        return {"error": "not enough pairs", "n": len(pairs)}
    # sort by time_key proxy
    pairs_sorted = sorted(zip(results, pairs), key=lambda x: str(x[0].get(time_key, "")))
    n_test = max(1, int(len(pairs) * test_size))
    test_pairs = [p for _, p in pairs_sorted[-n_test:]]
    y_true = [a for a, _ in test_pairs]
    y_pred = [b for _, b in test_pairs]
    m = _compute_metrics(y_true, y_pred)
    return {"mae": float(m["mae"]), "rmse": float(m["rmse"]), "r2": float(m["r2"]), "spearman": float(m["spearman"]), "roc_auc": float(m["roc_auc"]), "ece": float(m["ece"]), "n_test": int(m["n"]), "stable_true": int(m["stable_true"]), "stable_pred": int(m["stable_pred"])}


def random_split_benchmark(results: List[Dict[str, Any]], test_size: float = 0.2, seed: int = 42) -> Dict[str, Any]:
    """Random split benchmark."""
    pairs = _get_pairs(results)
    if len(pairs) < 2:
        return {"error": "not enough pairs", "n": len(pairs)}
    rnd = random.Random(seed)
    idx = list(range(len(pairs)))
    rnd.shuffle(idx)
    n_test = max(1, int(len(pairs) * test_size))
    test_idx = set(idx[-n_test:])
    y_true = [pairs[i][0] for i in test_idx]
    y_pred = [pairs[i][1] for i in test_idx]
    m = _compute_metrics(y_true, y_pred)
    return {"mae": float(m["mae"]), "rmse": float(m["rmse"]), "r2": float(m["r2"]), "spearman": float(m["spearman"]), "roc_auc": float(m["roc_auc"]), "ece": float(m["ece"]), "n_test": int(m["n"]), "stable_true": int(m["stable_true"]), "stable_pred": int(m["stable_pred"])}


def chemical_system_split(results: List[Dict[str, Any]], test_size: float = 0.2, seed: int = 42) -> Dict[str, Any]:
    """Chemical-system-disjoint split: no chemsys overlap between train/test."""
    # filter to valid pairs with chemsys
    valid = [(r, r["true_form_energy"], r["predicted_form_energy"]) for r in results if r.get("true_form_energy") is not None and r.get("predicted_form_energy") is not None]
    if len(valid) < 2:
        return {"error": "not enough pairs", "n": len(valid)}
    # group by chemsys
    from collections import defaultdict
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, (r, _, _) in enumerate(valid):
        cs = _chemsys_of_result(r)
        groups[cs].append(i)
    chemsys_keys = list(groups.keys())
    rnd = random.Random(seed)
    rnd.shuffle(chemsys_keys)
    n_test_groups = max(1, int(len(chemsys_keys) * test_size))
    # ensure at least test_size fraction of samples, but disjoint groups
    test_keys = set(chemsys_keys[-n_test_groups:])
    # if too few samples, expand until test_size fraction reached (still disjoint)
    total = len(valid)
    test_indices = [idx for k in test_keys for idx in groups[k]]
    # expand if needed
    p = 0
    while len(test_indices) < int(total * test_size) and p < len(chemsys_keys):
        k = chemsys_keys[-(n_test_groups + p + 1) % len(chemsys_keys)] if len(chemsys_keys) > n_test_groups + p else None
        if k and k not in test_keys:
            test_keys.add(k)
            test_indices.extend(groups[k])
        p += 1
        if len(test_keys) >= len(chemsys_keys) - 1:
            break
    y_true = [valid[i][1] for i in test_indices]
    y_pred = [valid[i][2] for i in test_indices]
    m = _compute_metrics(y_true, y_pred)
    return {"mae": float(m["mae"]), "rmse": float(m["rmse"]), "r2": float(m["r2"]), "spearman": float(m["spearman"]), "roc_auc": float(m["roc_auc"]), "ece": float(m["ece"]), "n_test": int(m["n"]), "stable_true": int(m["stable_true"]), "stable_pred": int(m["stable_pred"]), "n_chemsys_test": len(test_keys), "n_chemsys_total": len(chemsys_keys)}


# alias for spec wording
chemsys_split = chemical_system_split
system_ood_split = chemical_system_split


def element_ood_split(results: List[Dict[str, Any]], held_out_elements: Optional[List[str]] = None, test_size: float = 0.2, seed: int = 42) -> Dict[str, Any]:
    """Element-OOD split: hold out materials containing held_out_elements (or random elements)."""
    valid = [(r, r["true_form_energy"], r["predicted_form_energy"]) for r in results if r.get("true_form_energy") is not None and r.get("predicted_form_energy") is not None]
    if len(valid) < 2:
        return {"error": "not enough pairs", "n": len(valid)}
    import re
    def elems_of(r):
        formula = r.get("formula") or r.get("formula_pretty") or ""
        return set(re.findall(r"[A-Z][a-z]?", formula))
    # collect all elements
    all_elems = set()
    for r, _, _ in valid:
        all_elems.update(elems_of(r))
    if not all_elems:
        return random_split_benchmark(results, test_size=test_size, seed=seed)
    rnd = random.Random(seed)
    if held_out_elements is None:
        # pick 20% of distinct elements as held-out, at least 1
        n_hold = max(1, int(len(all_elems) * 0.2))
        held_out_elements = rnd.sample(sorted(all_elems), n_hold)
    held = set(held_out_elements)
    # test = any material containing held element
    test_indices = [i for i, (r, _, _) in enumerate(valid) if elems_of(r) & held]
    # if too small or too large, adjust via random fallback
    if len(test_indices) == 0 or len(test_indices) >= len(valid):
        # fallback: treat held_out as chemsys heuristic: pick random subset containing rare element
        return chemical_system_split(results, test_size=test_size, seed=seed)
    # if test fraction too far from desired, we still report but note
    y_true = [valid[i][1] for i in test_indices]
    y_pred = [valid[i][2] for i in test_indices]
    m = _compute_metrics(y_true, y_pred)
    return {"mae": float(m["mae"]), "rmse": float(m["rmse"]), "r2": float(m["r2"]), "spearman": float(m["spearman"]), "roc_auc": float(m["roc_auc"]), "ece": float(m["ece"]), "n_test": int(m["n"]), "stable_true": int(m["stable_true"]), "stable_pred": int(m["stable_pred"]), "held_out_elements": sorted(held)}


def prototype_split(results: List[Dict[str, Any]], test_size: float = 0.2, seed: int = 42) -> Dict[str, Any]:
    """Prototype split: disjoint crystal_system / prototype proxy."""
    valid = [(r, r["true_form_energy"], r["predicted_form_energy"]) for r in results if r.get("true_form_energy") is not None and r.get("predicted_form_energy") is not None]
    if len(valid) < 2:
        return {"error": "not enough pairs", "n": len(valid)}
    from collections import defaultdict
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, (r, _, _) in enumerate(valid):
        key = str(r.get("crystal_system") or r.get("prototype") or "Unknown")
        groups[key].append(i)
    keys = list(groups.keys())
    rnd = random.Random(seed)
    rnd.shuffle(keys)
    n_test_keys = max(1, int(len(keys) * test_size))
    test_keys = set(keys[-n_test_keys:])
    test_indices = [idx for k in test_keys for idx in groups[k]]
    if not test_indices:
        return random_split_benchmark(results, test_size=test_size, seed=seed)
    y_true = [valid[i][1] for i in test_indices]
    y_pred = [valid[i][2] for i in test_indices]
    m = _compute_metrics(y_true, y_pred)
    return {"mae": float(m["mae"]), "rmse": float(m["rmse"]), "r2": float(m["r2"]), "spearman": float(m["spearman"]), "roc_auc": float(m["roc_auc"]), "ece": float(m["ece"]), "n_test": int(m["n"]), "stable_true": int(m["stable_true"]), "stable_pred": int(m["stable_pred"]), "n_prototype_test": len(test_keys)}


def ood_split(results: List[Dict[str, Any]], test_size: float = 0.2, seed: int = 42) -> Dict[str, Any]:
    """OOD split: union of element-OOD and system-OOD for stricter generalization."""
    # For report, OOD is max of element and system OOD MAE
    e = element_ood_split(results, test_size=test_size, seed=seed)
    s = chemical_system_split(results, test_size=test_size, seed=seed + 1)
    if "error" in e or "error" in s:
        return s if "error" not in s else e
    # combine: take worse MAE as OOD, merge metrics
    mae_ood = max(e["mae"], s["mae"])
    return {"mae": mae_ood, "rmse": max(e["rmse"], s["rmse"]), "r2": min(e["r2"], s["r2"]), "spearman": min(e["spearman"], s["spearman"]), "roc_auc": min(e["roc_auc"], s["roc_auc"]), "ece": max(e["ece"], s["ece"]), "n_test": max(e["n_test"], s["n_test"]), "element": e, "system": s}


def benchmark_report(results: List[Dict[str, Any]], test_size: float = 0.2, seed: int = 42) -> Dict[str, Any]:
    """Produce report table Random / Element-OOD / System-OOD with all metrics."""
    rand = random_split_benchmark(results, test_size=test_size, seed=seed)
    elem = element_ood_split(results, test_size=test_size, seed=seed)
    sys = chemical_system_split(results, test_size=test_size, seed=seed)
    # temporal and prototype also included if requested
    temp = time_split_benchmark(results, test_size=test_size)
    proto = prototype_split(results, test_size=test_size, seed=seed)
    ood = ood_split(results, test_size=test_size, seed=seed)
    # OOD gap: MAE_ood - MAE_random
    def gap(ood_m, id_m):
        try:
            return float(ood_m["mae"] - id_m["mae"])
        except Exception:
            return float("nan")
    table = {
        "Random": rand,
        "Element-OOD": elem,
        "System-OOD": sys,
        "Prototype-OOD": proto,
        "Temporal": temp,
        "OOD": ood,
        "OOD_gap_Element_vs_Random": gap(elem, rand),
        "OOD_gap_System_vs_Random": gap(sys, rand),
    }
    return table


def format_report_table(report: Dict[str, Any]) -> str:
    """Return a markdown-style table string for display."""
    headers = ["Split", "MAE", "RMSE", "R2", "Spearman", "ROC-AUC", "ECE", "n"]
    rows = []
    for name in ["Random", "Element-OOD", "System-OOD"]:
        m = report.get(name, {})
        if "error" in m:
            rows.append([name, f"error:{m.get('error')}", "-", "-", "-", "-", "-", str(m.get("n", 0))])
        else:
            rows.append([
                name,
                f"{m.get('mae', float('nan')):.4f}",
                f"{m.get('rmse', float('nan')):.4f}",
                f"{m.get('r2', float('nan')):.3f}",
                f"{m.get('spearman', float('nan')):.3f}",
                f"{m.get('roc_auc', float('nan')):.3f}" if not math.isnan(m.get('roc_auc', float('nan'))) else "nan",
                f"{m.get('ece', float('nan')):.3f}" if not math.isnan(m.get('ece', float('nan'))) else "nan",
                str(m.get("n_test", m.get("n", 0))),
            ])
    # build markdown table
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def ensemble_uq(predictions: List[float]) -> Dict[str, float]:
    """Simple ensemble UQ: mean/std over perturbed predictions."""
    try:
        import numpy as np
        arr = np.array(predictions, dtype=float)
        return {"mean": float(arr.mean()), "std": float(arr.std()), "n": len(arr)}
    except Exception:
        n = len(predictions)
        if n == 0:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        mean = sum(predictions)/n
        var = sum((x-mean)**2 for x in predictions)/n
        return {"mean": float(mean), "std": float(math.sqrt(var)), "n": n}
