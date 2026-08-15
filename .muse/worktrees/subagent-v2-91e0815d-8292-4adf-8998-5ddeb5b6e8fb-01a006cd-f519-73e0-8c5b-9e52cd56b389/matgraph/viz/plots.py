from __future__ import annotations
from typing import List, Dict
def parity_data(results: List[Dict], y_key_true: str = "true_form_energy", y_key_pred: str = "predicted_form_energy"):
    pts = [(r[y_key_true], r[y_key_pred]) for r in results if r.get(y_key_true) is not None and r.get(y_key_pred) is not None]
    return pts
