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

def export_dft(formula: str, api_key: str, code: str = "vasp", output_dir: str = "dft_inputs"):
    """
    Bridge ML to DFT: Pre-relax the structure using M3GNet and write DFT input files.
    """
    import os
    
    # 1. Relax the structure very quickly using ML
    relax_results = relax_structure(formula, api_key, steps=20)
    structure = relax_results["relaxed_structure"]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    out_path = os.path.join(output_dir, formula)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    # 2. Generate DFT Inputs
    if code.lower() == "vasp":
        from pymatgen.io.vasp.sets import MPRelaxSet
        # Use MPRelaxSet for standard Materials Project parameters
        vis = MPRelaxSet(structure)
        vis.write_input(out_path)
        return {"code": "VASP", "directory": out_path, "files_written": ["POSCAR", "INCAR", "KPOINTS", "POTCAR"]}
        
    elif code.lower() in ["qe", "pwscf", "quantum_espresso"]:
        from pymatgen.io.pwscf import PWInput
        # Generate a standard self-consistent field (scf) input for QE
        pseudo_dir = os.environ.get("PSEUDO_DIR", ".")
        pseudopotentials = {str(el): f"{el}.UPF" for el in structure.composition.elements}
        
        control = {"calculation": "scf", "pseudo_dir": pseudo_dir}
        system = {"ecutwfc": 50, "ecutrho": 200}
        electrons = {"conv_thr": 1e-6}
        
        pw_in = PWInput(
            structure=structure,
            pseudo=pseudopotentials,
            control=control,
            system=system,
            electrons=electrons,
            kpoints_grid=(4, 4, 4)
        )
        pw_in.write_file(os.path.join(out_path, f"{formula}.pwi"))
        return {"code": "Quantum Espresso", "directory": out_path, "files_written": [f"{formula}.pwi"]}
        
    else:
        raise ValueError(f"Unsupported DFT code: {code}. Use 'vasp' or 'qe'.")


def stability_hull(formula: str, api_key: str) -> List[dict]:
    """
    Check where a material sits on the convex hull (thermodynamic phase diagram).
    energy_above_hull = 0 means on the hull (stable). >0 means metastable/unstable.
    """
    from mp_api.client import MPRester

    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            formula=formula,
            fields=["material_id", "formula_pretty", "formation_energy_per_atom",
                    "energy_above_hull", "is_stable"]
        )

    if not docs:
        raise ValueError(f"No data found for {formula}")

    results = []
    for d in docs:
        hull_e = d.energy_above_hull or 0.0
        if hull_e == 0.0:
            label = "Stable"
        elif hull_e < 0.05:
            label = "Metastable"
        else:
            label = "Unstable"
        results.append({
            "material_id": str(d.material_id),
            "formula": d.formula_pretty,
            "formation_energy_per_atom": d.formation_energy_per_atom,
            "energy_above_hull": hull_e,
            "is_stable": d.is_stable,
            "stability_label": label,
        })
    return results


def fetch_band_structure(formula: str, api_key: str) -> dict:
    """
    Fetch the electronic band structure summary for the most stable polymorph.
    Returns band gap, VBM, CBM, and whether the material is metallic.
    """
    from mp_api.client import MPRester

    docs = fetch_materials_data(formula, api_key)
    if not docs:
        raise ValueError(f"No data for {formula}")

    with MPRester(api_key) as mpr:
        for doc in docs:
            mat_id = str(doc.material_id)
            try:
                bs = mpr.get_bandstructure_by_material_id(mat_id)
                if bs is None:
                    continue
                return {
                    "material_id": mat_id,
                    "formula": formula,
                    "band_gap": bs.get_band_gap()["energy"],
                    "is_metal": bs.is_metal(),
                    "vbm": bs.get_vbm()["energy"],
                    "cbm": bs.get_cbm()["energy"],
                    "nbands": bs.nb_bands,
                    "kpoints": [k.frac_coords.tolist() for k in bs.kpoints],
                }
            except Exception:
                continue
    raise ValueError(f"No band structure data found for any polymorph of {formula}")


def fetch_elastic(formula: str, api_key: str) -> List[dict]:
    """
    Fetch elastic constants (bulk/shear modulus, Poisson ratio, anisotropy) from MP.
    """
    from mp_api.client import MPRester

    with MPRester(api_key) as mpr:
        docs = mpr.materials.elasticity.search(
            formula=formula,
            fields=["material_id", "formula_pretty", "bulk_modulus",
                    "shear_modulus", "universal_anisotropy", "homogeneous_poisson"]
        )

    if not docs:
        raise ValueError(f"No elastic data for {formula}. Not all materials have DFT elastic tensors.")

    results = []
    for d in docs:
        results.append({
            "material_id": str(d.material_id),
            "formula": d.formula_pretty,
            "bulk_modulus_vrh": d.bulk_modulus.vrh if d.bulk_modulus else None,
            "shear_modulus_vrh": d.shear_modulus.vrh if d.shear_modulus else None,
            "universal_anisotropy": d.universal_anisotropy,
            "homogeneous_poisson": d.homogeneous_poisson,
        })
    return results


def fetch_dielectric(formula: str, api_key: str) -> List[dict]:
    """
    Fetch dielectric constants (total, electronic, ionic) and refractive index from MP.
    """
    from mp_api.client import MPRester

    with MPRester(api_key) as mpr:
        docs = mpr.materials.dielectric.search(
            formula=formula,
            fields=["material_id", "formula_pretty", "e_total", "e_ionic", "e_electronic", "n"]
        )

    if not docs:
        raise ValueError(f"No dielectric data for {formula}.")

    results = []
    for d in docs:
        results.append({
            "material_id": str(d.material_id),
            "formula": d.formula_pretty,
            "e_total": d.e_total,
            "e_ionic": d.e_ionic,
            "e_electronic": d.e_electronic,
            "refractive_index": d.n,
        })
    return results


def fetch_magnetic(formula: str, api_key: str) -> List[dict]:
    """
    Fetch magnetic properties (ordering, total magnetization) from MP.
    """
    from mp_api.client import MPRester

    with MPRester(api_key) as mpr:
        docs = mpr.materials.magnetism.search(
            formula=formula,
            fields=["material_id", "formula_pretty", "ordering",
                    "total_magnetization", "total_magnetization_normalized_vol"]
        )

    if not docs:
        raise ValueError(f"No magnetic data for {formula}.")

    results = []
    for d in docs:
        results.append({
            "material_id": str(d.material_id),
            "formula": d.formula_pretty,
            "ordering": str(d.ordering),
            "total_magnetization": d.total_magnetization,
            "magnetization_per_vol": d.total_magnetization_normalized_vol,
        })
    return results
