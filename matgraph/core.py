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
        # MP API expects CrystalSystem enum, but string usually works if properly formatted.
        # Often mapped as kwargs for symmetry parameters. We pass crystal_system strictly.
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

def run_pipeline(
    formula: str, 
    api_key: str,
    min_gap: Optional[float] = None,
    max_gap: Optional[float] = None,
    crystal_system: Optional[str] = None
):
    """Orchestrates the entire Fetch, Filter, Featurize, and Predict pipeline."""
    band_gap_range = None
    if min_gap is not None or max_gap is not None:
        band_gap_range = (min_gap or 0.0, max_gap or 100.0)

    docs = fetch_materials_data(
        formula, api_key, 
        band_gap_range=band_gap_range, 
        crystal_system=crystal_system
    )
    
    results = []
    for doc in docs:
        if not doc.structure:
            continue
            
        features = extract_features(doc.structure)
        prediction = cgcnn_predict(features)
        
        # Ensure robust retrieval of crystal system
        c_sys = "Unknown"
        if doc.symmetry and hasattr(doc.symmetry, "crystal_system"):
            c_sys = doc.symmetry.crystal_system.name if hasattr(doc.symmetry.crystal_system, 'name') else str(doc.symmetry.crystal_system)
        
        results.append({
            "material_id": str(doc.material_id),
            "formula": doc.formula_pretty,
            "true_band_gap": doc.band_gap,
            "predicted_band_gap": prediction,
            "crystal_system": c_sys,
            "features": features
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
        keys = ["material_id", "formula", "crystal_system", "true_band_gap", "predicted_band_gap", "density", "volume"]
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
                    "density": r["features"]["density"],
                    "volume": r["features"]["volume"],
                }
                writer.writerow(row)
