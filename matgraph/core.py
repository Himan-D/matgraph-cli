import os
import json
import csv
from typing import Optional, Tuple, List
from mp_api.client import MPRester

def fetch_materials_data(
    formula: str, 
    api_key: str,
    band_gap_range: Optional[Tuple[float, float]] = None,
    crystal_system: Optional[str] = None
):
    """Fetch material data from Materials Project with advanced filters."""
    search_kwargs = {
        "formula": formula,
        "fields": ["material_id", "formula_pretty", "structure", "band_gap", "formation_energy_per_atom", "density", "symmetry"]
    }
    
    if band_gap_range:
        search_kwargs["band_gap"] = band_gap_range
    if crystal_system:
        search_kwargs["crystal_system"] = crystal_system

    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(**search_kwargs)
    return docs

def extract_features(structure):
    """Extract features from pymatgen structure."""
    comp = structure.composition
    return {
        "num_elements": len(comp.elements),
        "mean_atomic_mass": comp.weight / comp.num_atoms,
        "volume": structure.volume,
        "density": structure.density
    }

from matgraph.cgcnn import cgcnn_predict
from matgraph.megnet import megnet_predict

def m3gnet_predict(features):
    from matgraph.m3gnet import M3GNet
    model = M3GNet()
    model.eval()
    import torch
    with torch.no_grad():
        return model(features)

def simulate_xrd(structure):
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    xrd_calc = XRDCalculator(wavelength="CuKa")
    pattern = xrd_calc.get_pattern(structure)
    return {
        "two_theta": pattern.x.tolist(),
        "intensity": pattern.y.tolist(),
        "hkls": [[hkl[0] for hkl in hkls] for hkls in pattern.hkls]
    }

def run_pipeline(formula: str, api_key: str, min_gap: Optional[float] = None, max_gap: Optional[float] = None, crystal_system: Optional[str] = None, model: str = "cgcnn"):
    docs = fetch_materials_data(formula, api_key)
    
    if min_gap is not None or max_gap is not None:
        docs = [d for d in docs if d.band_gap is not None]
        if min_gap is not None:
            docs = [d for d in docs if d.band_gap >= min_gap]
        if max_gap is not None:
            docs = [d for d in docs if d.band_gap <= max_gap]
            
    if crystal_system is not None:
        docs = [d for d in docs if d.symmetry and d.symmetry.crystal_system.name.lower() == crystal_system.lower()]
        
    results = []
    for doc in docs:
        if not doc.structure:
            continue
            
        c_sys = doc.symmetry.crystal_system.name if doc.symmetry else "Unknown"
        features = extract_features(doc.structure)
        
        pred_gap, pred_form_energy = None, None
        energy, forces, stresses = None, None, None
        
        if model.lower() == "megnet":
            pred_gap, pred_form_energy = megnet_predict(features)
        elif model.lower() == "m3gnet":
            energy, forces, stresses = m3gnet_predict(features)
        else:
            pred_gap, pred_form_energy = cgcnn_predict(features)
            
        results.append({
            "material_id": str(doc.material_id),
            "formula": doc.formula_pretty,
            "true_band_gap": doc.band_gap,
            "predicted_band_gap": pred_gap,
            "true_form_energy": doc.formation_energy_per_atom,
            "predicted_form_energy": pred_form_energy,
            "m3gnet_energy": energy,
            "m3gnet_forces": forces,
            "m3gnet_stresses": stresses,
            "crystal_system": c_sys,
            "features": features,
            "model_used": model.upper(),
            "structure": doc.structure
        })
        
    return results

def save_results(results: List[dict], output_file: str, file_format: str):
    """Save the pipeline results to CSV or JSON formats."""
    if file_format == "json":
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
    elif file_format == "csv":
        if not results:
            return
        keys = ["material_id", "formula", "crystal_system", "true_band_gap", "predicted_band_gap", "true_form_energy", "predicted_form_energy", "density", "volume"]
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in results:
                row = {
                    "material_id": r["material_id"],
                    "formula": r["formula"],
                    "crystal_system": r["crystal_system"],
                    "true_band_gap": r["true_band_gap"],
                    "predicted_band_gap": r["predicted_band_gap"],
                    "true_form_energy": r["true_form_energy"],
                    "predicted_form_energy": r["predicted_form_energy"],
                    "density": r["features"]["density"],
                    "volume": r["features"]["volume"],
                }
                writer.writerow(row)
