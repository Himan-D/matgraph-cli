import os
import asyncio
from typing import Optional, List, Dict, Any, Union
from matgraph.core import (
    run_pipeline, substitute_material, simulate_xrd, fetch_materials_data,
    fetch_phonon_dos, inverse_design, relax_structure, export_dft,
    stability_hull, fetch_band_structure, fetch_elastic, fetch_dielectric, fetch_magnetic
)
from matgraph.config import get_api_key
from matgraph.exceptions import ValidationError

class MatGraphSDK:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or get_api_key()
        if not self.api_key:
            raise ValueError("Materials Project API key is required. Run 'matgraph setup <KEY>' or set MP_API_KEY.")

    def predict(self, formula: str, model: str = "m3gnet", min_gap: Optional[float] = None, max_gap: Optional[float] = None, crystal_system: Optional[str] = None, seed: Optional[int] = None, as_frame: Optional[str] = None, track: bool = False, project: str = "matgraph") -> Union[List[Dict[str, Any]], Any]:
        """Predict with optional DataFrame return: as_frame='pandas'|'polars'. If track=True, logs wandb-like run."""
        results = run_pipeline(formula=formula, api_key=self.api_key, min_gap=min_gap, max_gap=max_gap, crystal_system=crystal_system, model=model, seed=seed)
        if track:
            try:
                from matgraph.tracking import init
                run = init(project=project, name=f"predict-{formula}", config={"formula":formula,"model":model,"seed":seed})
                for r in results:
                    run.log({"formula":r.get("formula"),"eform":r.get("predicted_form_energy"),"band_gap":r.get("true_band_gap")})
                # log table artifact
                try:
                    run.log_table("predictions", ["formula","eform","band_gap"], [[r.get("formula"),r.get("predicted_form_energy"),r.get("true_band_gap")] for r in results])
                except Exception:
                    pass
                run.finish()
            except Exception:
                pass
        if as_frame:
            return self._to_frame(results, as_frame)
        return results

    def predict_many(self, formulas: List[str], model: str = "m3gnet", crystal_system: Optional[str] = None, seed: Optional[int] = None, as_frame: Optional[str] = None, max_workers: int = 4) -> Union[List[Dict[str, Any]], Any]:
        """Batch predict — not hardcoded concurrency."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        out = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(run_pipeline, f, self.api_key, None, None, crystal_system, model, seed): f for f in formulas}
            for fut in as_completed(futs):
                try:
                    out.extend(fut.result())
                except Exception as e:
                    out.append({"formula": futs[fut], "error": str(e)})
        if as_frame:
            return self._to_frame([r for r in out if "material_id" in r], as_frame)
        return out

    async def predict_async(self, formula: str, **kw) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.predict, formula, **kw)

    def from_structures(self, structures: List[Any], model: str = "m3gnet", seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """Predict directly from pymatgen Structures — no MP fetch."""
        from matgraph.core import extract_features, _provenance, m3gnet_predict_pes, get_matgl_eform_model
        prov = _provenance(seed=seed)
        out = []
        for s in structures:
            feats = extract_features(s)
            energy, forces, stresses = m3gnet_predict_pes(s)
            eform = float(get_matgl_eform_model().predict_structure(s).detach().item())
            out.append({"formula": s.composition.reduced_formula, "features": feats, "m3gnet_energy": float(energy), "m3gnet_forces": forces.tolist(), "m3gnet_stresses": stresses.tolist(), "predicted_form_energy": eform, "predicted_band_gap": None, "band_gap_source": "mp_experimental", "provenance": prov, "model_used": model.upper()})
        return out

    def _to_frame(self, results: List[dict], kind: str):
        if kind == "pandas":
            import pandas as pd
            flat = []
            for r in results:
                flat.append({**{k: v for k, v in r.items() if k not in ("features","provenance","structure","m3gnet_forces","m3gnet_stresses")}, **{"density": r.get("features",{}).get("density"), "volume": r.get("features",{}).get("volume")}})
            return pd.DataFrame(flat)
        if kind == "polars":
            import polars as pl
            flat = []
            for r in results:
                flat.append({**{k: v for k, v in r.items() if k not in ("features","provenance","structure","m3gnet_forces","m3gnet_stresses")}, **{"density": r.get("features",{}).get("density"), "volume": r.get("features",{}).get("volume")}})
            return pl.DataFrame(flat)
        raise ValidationError("as_frame must be 'pandas' or 'polars'")

    def evaluate(self, formula: str, model: str = "m3gnet", seed: Optional[int] = None) -> Dict[str, float]:
        # honest: only formation energy MAE, band gap MAE excluded unless model predicts it
        results = self.predict(formula, model=model, seed=seed)
        form_errors = []
        gap_errors = []
        for r in results:
            if r.get("true_band_gap") is not None and r.get("predicted_band_gap") is not None:
                gap_errors.append(abs(r["true_band_gap"] - r["predicted_band_gap"]))
            if r.get("true_form_energy") is not None and r.get("predicted_form_energy") is not None:
                form_errors.append(abs(r["true_form_energy"] - r["predicted_form_energy"]))
        return {
            "band_gap_mae": sum(gap_errors)/len(gap_errors) if gap_errors else None,
            "band_gap_mae_note": None if gap_errors else "No predicted_band_gap available — model does not predict band gap",
            "formation_energy_mae": sum(form_errors)/len(form_errors) if form_errors else 0.0,
            "samples_evaluated": len(results)
        }

    def substitute(self, formula: str, element_out: str, element_in: str, seed: Optional[int] = None) -> Dict[str, Any]:
        return substitute_material(formula, element_out, element_in, self.api_key, seed=seed)

    def xrd(self, formula: str) -> Dict[str, Any]:
        docs = fetch_materials_data(formula, self.api_key)
        if not docs or not docs[0].structure:
            raise ValueError(f"No crystal structure found for {formula}")
        return simulate_xrd(docs[0].structure)

    def phonon_dos(self, formula: str, method: str = "dfpt") -> Dict[str, Any]:
        return fetch_phonon_dos(formula, self.api_key, phonon_method=method)

    def design(self, min_gap: float = None, max_gap: float = None, crystal_system: str = None, exclude_elements: List[str] = None, include_elements: List[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        return inverse_design(api_key=self.api_key, min_gap=min_gap, max_gap=max_gap, crystal_system=crystal_system, exclude_elements=exclude_elements, include_elements=include_elements, limit=limit)

    def relax(self, formula: str, steps: int = 10, seed: Optional[int] = None) -> Dict[str, Any]:
        return relax_structure(formula, self.api_key, steps=steps, seed=seed)

    def evolve(self, formula: str, population_size: int = 10, generations: int = 5, allowed_elements: Optional[List[str]] = None, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        from matgraph.ga import CrystalGA
        ga = CrystalGA(base_formula=formula, api_key=self.api_key, population_size=population_size, allowed_elements=allowed_elements, seed=seed)
        return ga.run(generations=generations)

    def export_dft(self, formula: str, code: str = "vasp", output_dir: str = "dft_inputs", seed: Optional[int] = None) -> Dict[str, Any]:
        return export_dft(formula, self.api_key, code=code, output_dir=output_dir, seed=seed)

    def stability(self, formula: str) -> List[Dict[str, Any]]:
        return stability_hull(formula, self.api_key)

    def band_structure(self, formula: str) -> Dict[str, Any]:
        return fetch_band_structure(formula, self.api_key)

    def elastic(self, formula: str) -> List[Dict[str, Any]]:
        return fetch_elastic(formula, self.api_key)

    def dielectric(self, formula: str) -> List[Dict[str, Any]]:
        return fetch_dielectric(formula, self.api_key)

    def magnetic(self, formula: str) -> List[Dict[str, Any]]:
        return fetch_magnetic(formula, self.api_key)
