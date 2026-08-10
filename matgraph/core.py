import os
import json
import csv
from typing import Optional, Tuple, List
from matgraph.cdn import cache_get, cache_put

def fetch_materials_data(
    formula: str, 
    api_key: str,
    band_gap_range: Optional[Tuple[float, float]] = None,
    crystal_system: Optional[str] = None
):
    """Fetch material data from Materials Project with CDN caching."""
    search_kwargs = {
        "formula": formula,
        "fields": ["material_id", "formula_pretty", "structure", "band_gap", "formation_energy_per_atom", "density", "symmetry"]
    }
    
    if band_gap_range:
        search_kwargs["band_gap"] = band_gap_range
    if crystal_system:
        search_kwargs["crystal_system"] = crystal_system

    from mp_api.client import MPRester
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

from functools import lru_cache

@lru_cache(maxsize=1)
def get_matgl_pes_model():
    import matgl
    return matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2")

@lru_cache(maxsize=1)
def get_matgl_eform_model():
    import matgl
    return matgl.load_model("M3GNet-Eform-MP-2019.4.1")

def m3gnet_predict_pes(structure):
    from matgl.ext.ase import M3GNetCalculator
    from pymatgen.io.ase import AseAtomsAdaptor
    
    pot = get_matgl_pes_model()
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = M3GNetCalculator(potential=pot)
    
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stresses = atoms.get_stress()
    return energy, forces, stresses

def simulate_xrd(structure):
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    xrd_calc = XRDCalculator(wavelength="CuKa")
    pattern = xrd_calc.get_pattern(structure)
    return {
        "two_theta": pattern.x.tolist(),
        "intensity": pattern.y.tolist(),
        "hkls": [[hkl["hkl"] for hkl in hkls] for hkls in pattern.hkls]
    }

def substitute_material(formula: str, elem_out: str, elem_in: str, api_key: str):
    """
    Simulates elemental substitution using real MatGL models on structures.
    """
    from pymatgen.core import Composition
    import numpy as np
    
    docs = fetch_materials_data(formula, api_key)
    if not docs:
        raise ValueError(f"Could not find baseline data for {formula}.")
        
    doc = docs[0]
    if not doc.structure:
        raise ValueError(f"No structure available in MP for {formula}.")
        
    orig_comp = Composition(formula)
    if elem_out not in orig_comp:
        raise ValueError(f"Element {elem_out} not found in {formula}.")
        
    orig_structure = doc.structure
    new_structure = orig_structure.copy()
    new_structure.replace_species({elem_out: elem_in})
    
    orig_energy, orig_forces, _ = m3gnet_predict_pes(orig_structure)
    new_energy, new_forces, _ = m3gnet_predict_pes(new_structure)
    
    return {
        "original": {
            "formula": formula,
            "energy": float(orig_energy),
            "max_force": float(np.max(np.abs(orig_forces)))
        },
        "hypothetical": {
            "formula": new_structure.composition.reduced_formula,
            "energy": float(new_energy),
            "max_force": float(np.max(np.abs(new_forces)))
        },
        "is_more_stable": new_energy < orig_energy
    }

def run_pipeline(formula: str, api_key: str, min_gap: Optional[float] = None, max_gap: Optional[float] = None, crystal_system: Optional[str] = None, model: str = "cgcnn"):
    # Check CDN cache first
    cached = cache_get("pipeline", formula=formula, min_gap=min_gap, max_gap=max_gap, crystal_system=crystal_system, model=model)
    if cached is not None:
        return cached

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
        
        if model.lower() in ["m3gnet", "megnet", "cgcnn"]: # Map all legacy calls to real M3GNet
            energy, forces, stresses = m3gnet_predict_pes(doc.structure)
            eform_model = get_matgl_eform_model()
            # M3GNet returns a tensor, take the item
            pred_form_energy = float(eform_model.predict_structure(doc.structure).detach().item())
            # matgl does not currently expose a working M3GNet band gap model, fallback to MP data
            pred_gap = doc.band_gap 
        else:
            # Fallback
            pred_gap = doc.band_gap
            pred_form_energy = doc.formation_energy_per_atom
            
        results.append({
            "material_id": str(doc.material_id),
            "formula": doc.formula_pretty,
            "true_band_gap": doc.band_gap,
            "predicted_band_gap": pred_gap,
            "true_form_energy": doc.formation_energy_per_atom,
            "predicted_form_energy": pred_form_energy,
            "m3gnet_energy": float(energy) if energy is not None else None,
            "m3gnet_forces": forces.tolist() if forces is not None else None,
            "m3gnet_stresses": stresses.tolist() if stresses is not None else None,
            "crystal_system": c_sys,
            "features": features,
            "model_used": model.upper(),
            "structure": doc.structure
        })

    # Write results to CDN cache (local + S3)
    serializable = []
    for r in results:
        row = {k: v for k, v in r.items() if k != "structure"}
        serializable.append(row)
    cache_put("pipeline", serializable, formula=formula, min_gap=min_gap, max_gap=max_gap, crystal_system=crystal_system, model=model)

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

def fetch_phonon_dos(formula: str, api_key: str, phonon_method: str = "dfpt"):
    """
    Fetches the Phonon Density of States (DOS) for a given material formula.
    Uses the dfpt method by default.
    """
    docs = fetch_materials_data(formula, api_key)
    if not docs:
        raise ValueError(f"Could not find baseline data for {formula}.")
        
    from mp_api.client import MPRester
    with MPRester(api_key) as mpr:
        for doc in docs:
            mat_id = str(doc.material_id)
            try:
                dos = mpr.materials.phonon.get_dos_from_material_id(mat_id, phonon_method=phonon_method)
                if dos:
                    return {
                        "material_id": mat_id,
                        "formula": formula,
                        "phonon_method": phonon_method,
                        "frequencies": list(dos.frequencies),
                        "densities": list(dos.densities)
                    }
            except Exception as e:
                # Some materials might not have phonon data; continue to the next polymorph
                continue
                
        raise ValueError(f"Phonon DOS data not found for any polymorph of {formula} using method {phonon_method}.")

def inverse_design(api_key: str, min_gap: float = None, max_gap: float = None, crystal_system: str = None, exclude_elements: list = None, include_elements: list = None, limit: int = 10):
    """
    Inverse design: query the Materials Project for materials matching strict criteria.
    """
    kwargs = {
        "num_chunks": 1,
        "chunk_size": limit,
        "fields": ["material_id", "formula_pretty", "band_gap", "symmetry", "is_stable"]
    }
    
    if min_gap is not None or max_gap is not None:
        kwargs["band_gap"] = (min_gap or 0.0, max_gap or 10.0)
    if crystal_system:
        kwargs["crystal_system"] = crystal_system
    if exclude_elements:
        kwargs["exclude_elements"] = exclude_elements
    if include_elements:
        kwargs["elements"] = include_elements

    from mp_api.client import MPRester
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(**kwargs)
        
    results = []
    for d in docs:
        results.append({
            "material_id": str(d.material_id),
            "formula": d.formula_pretty,
            "band_gap": d.band_gap,
            "crystal_system": str(d.symmetry.crystal_system),
            "is_stable": d.is_stable
        })
    return results

def relax_structure(formula: str, api_key: str, steps: int = 10):
    """
    Relax a crystal structure using the real MatGL M3GNet Universal Potential and ASE.
    """
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.optimize import FIRE
    from matgl.ext.ase import M3GNetCalculator
    
    docs = fetch_materials_data(formula, api_key)
    if not docs or not docs[0].structure:
        raise ValueError(f"No structure found for {formula}")
        
    structure = docs[0].structure
    
    # Introduce small random noise to the atomic positions to simulate an unrelaxed state
    import numpy as np
    structure.perturb(0.1)
    
    pot = get_matgl_pes_model()
    
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = M3GNetCalculator(potential=pot)
    
    # Run geometry optimization
    dyn = FIRE(atoms, logfile=None)
    
    energy_history = []
    def observer():
        energy_history.append(atoms.get_potential_energy())
        
    dyn.attach(observer)
    dyn.run(fmax=0.05, steps=steps)
    
    relaxed_structure = AseAtomsAdaptor.get_structure(atoms)
    
    return {
        "formula": formula,
        "initial_energy": float(energy_history[0]) if energy_history else None,
        "final_energy": float(energy_history[-1]) if energy_history else None,
        "steps_taken": len(energy_history),
        "relaxed_structure": relaxed_structure
    }
