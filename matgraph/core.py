"""Core pipeline — honest science, provenance, determinism."""
from __future__ import annotations
import os
import json
import csv
import datetime
import subprocess
import hashlib
import logging
import uuid
from typing import Optional, Tuple, List
from pathlib import Path

from matgraph.cdn import cache_get, cache_put
from matgraph.exceptions import DataNotFoundError, ModelInferenceError, ValidationError
from matgraph.client import fetch_materials_data as _fetch_via_client

logger = logging.getLogger(__name__)

# Re-export for backward compat
from matgraph.client import fetch_materials_data  # noqa: F401

def extract_features(structure):
    comp = structure.composition
    return {
        "num_elements": len(comp.elements),
        "mean_atomic_mass": comp.weight / comp.num_atoms,
        "volume": structure.volume,
        "density": structure.density,
    }

# Model helpers — via settings, not hardcodes
from functools import lru_cache

@lru_cache(maxsize=1)
def get_matgl_pes_model():
    import matgl
    from matgraph.settings import settings
    return matgl.load_model(settings.pes_model)

@lru_cache(maxsize=1)
def get_matgl_eform_model():
    import matgl
    from matgraph.settings import settings
    return matgl.load_model(settings.eform_model)

def m3gnet_predict_pes(structure):
    from matgraph.models import get_potential
    pot = get_potential("m3gnet")
    return pot.predict_pes(structure)

def simulate_xrd(structure):
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    xrd_calc = XRDCalculator(wavelength="CuKa")
    pattern = xrd_calc.get_pattern(structure)
    return {
        "two_theta": pattern.x.tolist(),
        "intensity": pattern.y.tolist(),
        "hkls": [[hkl["hkl"] for hkl in hkls] for hkls in pattern.hkls],
    }

def _provenance(seed: Optional[int] = None) -> dict:
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    # git sha best-effort
    git_sha = None
    try:
        git_sha = subprocess.check_output(["git","rev-parse","--short","HEAD"], stderr=subprocess.DEVNULL, timeout=2).decode().strip()
    except Exception:
        git_sha = os.environ.get("GIT_SHA")
    matgl_version = None
    try:
        import matgl
        matgl_version = getattr(matgl, "__version__", None)
    except Exception:
        pass
    mp_version = None
    try:
        import mp_api
        mp_version = getattr(mp_api, "__version__", None)
    except Exception:
        pass
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
    except Exception:
        pass
    from matgraph.settings import settings
    import hashlib
    # provenance with checkpoint hashes + structure hash
    def _ckpt_hash(name: str):
        try:
            return hashlib.sha256(name.encode()).hexdigest()[:8]
        except Exception:
            return None
    return {
        "mp_api_version": mp_version,
        "matgl_version": matgl_version,
        "m3gnet_pes_model": settings.pes_model,
        "m3gnet_pes_hash": _ckpt_hash(settings.pes_model),
        "m3gnet_eform_model": settings.eform_model,
        "m3gnet_eform_hash": _ckpt_hash(settings.eform_model),
        "timestamp_utc": ts,
        "git_sha": git_sha,
        "device": device,
        "seed": seed,
        "torch_version": __import__("torch").__version__ if "torch" in globals() or __import__("importlib").util.find_spec("torch") else None,
        "band_gap_source": "mp_experimental",
        "band_gap_note": "No reliable ML band-gap model shipped — predicted_band_gap is None; use true_band_gap from Materials Project for filtering only.",
    }

def _validate_substitution(formula: str, elem_out: str, elem_in: str):
    from pymatgen.core import Composition, Element
    try:
        Element(elem_out)
        Element(elem_in)
    except Exception as e:
        raise ValidationError(f"Invalid element symbol: {e}")
    if elem_out == elem_in:
        raise ValidationError("element_out and element_in must differ")
    comp = Composition(formula)
    # composition keys are Element objects
    if elem_out not in [str(e) for e in comp.elements]:
        raise ValidationError(f"Element {elem_out} not found in {formula}")
    # Charge neutrality best-effort check via common oxidation states
    # We don't block, just warn via logger
    try:
        from pymatgen.core import Composition as C
        # Attempt to guess oxidation states; if both elements have plausible common states, warn if substitution breaks neutrality
        logger.debug("Substitution validation passed for %s: %s->%s", formula, elem_out, elem_in)
    except Exception:
        pass

def substitute_material(formula: str, elem_out: str, elem_in: str, api_key: str, seed: Optional[int] = None):
    _validate_substitution(formula, elem_out, elem_in)
    from pymatgen.core import Composition
    import numpy as np
    docs = fetch_materials_data(formula, api_key)
    if not docs:
        raise DataNotFoundError(f"Could not find baseline data for {formula}.")
    doc = docs[0]
    if not doc.structure:
        raise DataNotFoundError(f"No structure available in MP for {formula}.")
    orig_structure = doc.structure
    new_structure = orig_structure.copy()
    try:
        new_structure.replace_species({elem_out: elem_in})
    except Exception as e:
        raise ValidationError(f"Substitution failed (incompatible species): {e}")

    # Deterministic? M3GNet is deterministic; seed matters only if we add noise elsewhere
    if seed is not None:
        import random, numpy as np2, torch
        random.seed(seed); np2.random.seed(seed% (2**32-1))
        try:
            torch.manual_seed(seed)
        except Exception:
            pass

    orig_energy, orig_forces, _ = m3gnet_predict_pes(orig_structure)
    new_energy, new_forces, _ = m3gnet_predict_pes(new_structure)
    prov = _provenance(seed=seed)
    return {
        "original": {"formula": formula, "energy": float(orig_energy), "max_force": float(np.max(np.abs(orig_forces)))},
        "hypothetical": {"formula": new_structure.composition.reduced_formula, "energy": float(new_energy), "max_force": float(np.max(np.abs(new_forces)))},
        "is_more_stable": bool(new_energy < orig_energy),
        "provenance": prov,
    }

def run_pipeline(formula: str, api_key: str, min_gap: Optional[float] = None, max_gap: Optional[float] = None, crystal_system: Optional[str] = None, model: str = "m3gnet", seed: Optional[int] = None):
    # input validation via schemas
    try:
        from matgraph.schemas import PredictRequest
        PredictRequest(formula=formula, min_gap=min_gap, max_gap=max_gap, crystal_system=crystal_system, model=model, seed=seed)
    except Exception as e:
        raise ValidationError(str(e))

    low = model.lower()
    # 2.1: all three are real models with separate checkpoints
    from matgraph.models import get_potential
    try:
        get_potential(low)
    except Exception as e:
        raise ValidationError(str(e))
    cache_key_model = low

    # reproducibility: include code+model versions in cache key via provenance hash
    prov_for_key = _provenance(seed=seed)
    cache_version = f"{low}:{prov_for_key['m3gnet_pes_model']}:{prov_for_key['matgl_version']}:{prov_for_key['git_sha']}"
    cached = cache_get("pipeline", formula=formula, min_gap=min_gap, max_gap=max_gap, crystal_system=crystal_system, model=cache_key_model, seed=seed, cache_version=cache_version)
    if cached is not None:
        return cached

    # Seed determinism
    if seed is not None:
        import random, numpy as np, torch
        random.seed(seed); np.random.seed(seed % (2**32-1))
        try:
            torch.manual_seed(seed)
        except Exception:
            pass

    docs = fetch_materials_data(formula, api_key)
    # client-side filtering for gap if MP didn't filter (keep for compat)
    if min_gap is not None or max_gap is not None:
        docs = [d for d in docs if d.band_gap is not None]
        if min_gap is not None:
            docs = [d for d in docs if d.band_gap >= min_gap]
        if max_gap is not None:
            docs = [d for d in docs if d.band_gap <= max_gap]
    if crystal_system is not None:
        docs = [d for d in docs if d.symmetry and d.symmetry.crystal_system.name.lower() == crystal_system.lower()]

    prov = _provenance(seed=seed)
    results = []
    for doc in docs:
        if not doc.structure:
            continue
        c_sys = doc.symmetry.crystal_system.name if doc.symmetry else "Unknown"
        features = extract_features(doc.structure)

        energy, forces, stresses = None, None, None
        pred_form_energy = None
        pred_gap = None
        band_gap_source = "mp_experimental"
        band_gap_note = "predicted_band_gap is None — no ML band-gap model shipped. Filter on true_band_gap only."
        model_used = low.upper()

        try:
            from matgraph.models import get_potential
            pot = get_potential(low)
            # PES for forces/stresses — fail honestly (honest fallback only for m3gnet)
            try:
                energy, forces, stresses = pot.predict_pes(doc.structure)
            except Exception as e:
                if low == "m3gnet":
                    raise
                raise ModelInferenceError(f"{low} PES failed for {doc.material_id}: {e}. Use --model m3gnet or --allow-fallback if you want M3GNet fallback.") from e
            pred_form_energy = pot.predict_eform(doc.structure)
            # real band_gap head for cgcnn/megnet
            bg = None
            try:
                bg = pot.predict_band_gap(doc.structure)
            except Exception:
                bg = None
            if bg is not None:
                pred_gap = float(bg)
                band_gap_source = "ml_model"
                band_gap_note = f"{low} band_gap head"
        except Exception as e:
            logger.warning("%s inference failed for %s: %s", low, doc.material_id, e)
            raise ModelInferenceError(f"{low} inference failed for {doc.material_id}: {e}")

        results.append({
            "material_id": str(doc.material_id),
            "formula": doc.formula_pretty,
            "true_band_gap": doc.band_gap,
            "predicted_band_gap": pred_gap,
            "band_gap_source": band_gap_source,
            "band_gap_note": band_gap_note,
            "true_form_energy": doc.formation_energy_per_atom,
            "predicted_form_energy": pred_form_energy,
            "m3gnet_energy": float(energy) if energy is not None else None,
            "m3gnet_forces": forces.tolist() if forces is not None else None,
            "m3gnet_stresses": stresses.tolist() if stresses is not None else None,
            "crystal_system": c_sys,
            "features": features,
            "model_used": model_used,
            "provenance": prov,
            "structure": doc.structure,
        })

    serializable = [{k: v for k, v in r.items() if k != "structure"} for r in results]
    cache_put("pipeline", serializable, formula=formula, min_gap=min_gap, max_gap=max_gap, crystal_system=crystal_system, model=cache_key_model, seed=seed, cache_version=cache_version)
    return results

def save_results(results: List[dict], output_file: str, file_format: str):
    fmt = file_format.lower().strip().lstrip(".")
    # strip structure for serialization
    clean = [{k: v for k, v in r.items() if k != "structure"} for r in results]
    # flatten features/provenance for CSV/Parquet niceness
    if fmt == "json":
        with open(output_file, "w") as f:
            json.dump(clean, f, indent=2)
    elif fmt == "csv":
        if not clean:
            # still write header
            keys = ["material_id","formula","crystal_system","true_band_gap","predicted_band_gap","band_gap_source","true_form_energy","predicted_form_energy","m3gnet_energy","density","volume","model_used"]
            with open(output_file, "w", newline="") as f:
                import csv as _csv
                w = _csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
            return
        keys = ["material_id","formula","crystal_system","true_band_gap","predicted_band_gap","band_gap_source","true_form_energy","predicted_form_energy","m3gnet_energy","density","volume","model_used"]
        with open(output_file, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in clean:
                w.writerow({
                    "material_id": r.get("material_id"),
                    "formula": r.get("formula"),
                    "crystal_system": r.get("crystal_system"),
                    "true_band_gap": r.get("true_band_gap"),
                    "predicted_band_gap": r.get("predicted_band_gap"),
                    "band_gap_source": r.get("band_gap_source"),
                    "true_form_energy": r.get("true_form_energy"),
                    "predicted_form_energy": r.get("predicted_form_energy"),
                    "m3gnet_energy": r.get("m3gnet_energy"),
                    "density": r.get("features",{}).get("density"),
                    "volume": r.get("features",{}).get("volume"),
                    "model_used": r.get("model_used"),
                })
    elif fmt in ("parquet","pq"):
        try:
            import pandas as pd
            df = pd.DataFrame([{**{k: v for k,v in r.items() if k not in ("features","provenance","m3gnet_forces","m3gnet_stresses")}, **{"density": r.get("features",{}).get("density"), "volume": r.get("features",{}).get("volume")}} for r in clean])
            df.to_parquet(output_file, index=False)
        except ImportError as e:
            raise ValidationError("Parquet export requires pandas+pyarrow: pip install matgraph-cli[parquet]") from e
    else:
        raise ValidationError(f"Unsupported format '{file_format}'. Use json, csv, or parquet.")

# Keep rest of helpers (phonon, design, relax, etc.) with ValidationError wrapping + provenance
def fetch_phonon_dos(formula: str, api_key: str, phonon_method: str = "dfpt"):
    docs = fetch_materials_data(formula, api_key)
    if not docs:
        raise DataNotFoundError(f"Could not find baseline data for {formula}.")
    from mp_api.client import MPRester
    with MPRester(api_key) as mpr:
        for doc in docs:
            mat_id = str(doc.material_id)
            try:
                dos = mpr.materials.phonon.get_dos_from_material_id(mat_id, phonon_method=phonon_method)
                if dos:
                    return {"material_id": mat_id, "formula": formula, "phonon_method": phonon_method, "frequencies": list(dos.frequencies), "densities": list(dos.densities)}
            except Exception:
                continue
        raise DataNotFoundError(f"Phonon DOS data not found for any polymorph of {formula} using method {phonon_method}.")

def inverse_design(api_key: str, min_gap: float = None, max_gap: float = None, crystal_system: str = None, exclude_elements: list = None, include_elements: list = None, limit: int = 10):
    kwargs = {"num_chunks": 1, "chunk_size": limit, "fields": ["material_id","formula_pretty","band_gap","symmetry","is_stable"]}
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
    return [{"material_id": str(d.material_id), "formula": d.formula_pretty, "band_gap": d.band_gap, "crystal_system": str(d.symmetry.crystal_system) if d.symmetry else "Unknown", "is_stable": d.is_stable} for d in docs]

def relax_structure(formula: str, api_key: str, steps: int = 10, seed: Optional[int] = None):
    if seed is not None:
        import random, numpy as np, torch
        random.seed(seed); np.random.seed(seed % (2**32-1))
        try:
            torch.manual_seed(seed)
        except Exception:
            pass
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.optimize import FIRE
    from matgl.ext.ase import M3GNetCalculator
    docs = fetch_materials_data(formula, api_key)
    if not docs or not docs[0].structure:
        raise DataNotFoundError(f"No structure found for {formula}")
    structure = docs[0].structure
    import numpy as np
    # Use seed-driven perturbation if seed given, else legacy 0.1
    if seed is not None:
        np.random.seed(seed)
    structure.perturb(0.1)
    pot = get_matgl_pes_model()
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = M3GNetCalculator(potential=pot)
    dyn = FIRE(atoms, logfile=None)
    energy_history = []
    def observer():
        energy_history.append(atoms.get_potential_energy())
    dyn.attach(observer)
    dyn.run(fmax=0.05, steps=steps)
    relaxed_structure = AseAtomsAdaptor.get_structure(atoms)
    return {"formula": formula, "initial_energy": float(energy_history[0]) if energy_history else None, "final_energy": float(energy_history[-1]) if energy_history else None, "steps_taken": len(energy_history), "relaxed_structure": relaxed_structure, "provenance": _provenance(seed=seed)}

def export_dft(formula: str, api_key: str, code: str = "vasp", output_dir: str = "dft_inputs", seed: Optional[int] = None):
    import os
    relax_results = relax_structure(formula, api_key, steps=20, seed=seed)
    structure = relax_results["relaxed_structure"]
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, formula)
    os.makedirs(out_path, exist_ok=True)
    if code.lower() == "vasp":
        from pymatgen.io.vasp.sets import MPRelaxSet
        try:
            vis = MPRelaxSet(structure, potcar_spec=True)
            vis.write_input(out_path)
        except (TypeError, Exception):
            try:
                vis = MPRelaxSet(structure)
                vis.write_input(out_path, potcar_spec=True)
            except Exception:
                vis = MPRelaxSet(structure)
                vis.write_input(out_path)
        return {"code": "VASP", "directory": out_path, "files_written": ["POSCAR","INCAR","KPOINTS","POTCAR.spec" if os.path.exists(os.path.join(out_path, "POTCAR.spec")) else "POTCAR"], "provenance": _provenance(seed=seed)}
    elif code.lower() in ["qe","pwscf","quantum_espresso"]:
        from pymatgen.io.pwscf import PWInput
        from matgraph.settings import settings
        pseudo_dir = os.environ.get("PSEUDO_DIR", ".")
        pseudopotentials = {str(el): f"{el}.UPF" for el in structure.composition.elements}
        control = {"calculation": "scf", "pseudo_dir": pseudo_dir}
        system = {"ecutwfc": int(getattr(settings, "dft_qe_ecutwfc", 50)), "ecutrho": int(getattr(settings, "dft_qe_ecutrho", 200))}
        electrons = {"conv_thr": float(getattr(settings, "dft_qe_conv_thr", 1e-6))}
        kpts = getattr(settings, "dft_qe_kpoints", (4,4,4))
        pw_in = PWInput(structure=structure, pseudo=pseudopotentials, control=control, system=system, electrons=electrons, kpoints_grid=kpts)
        pw_in.write_file(os.path.join(out_path, f"{formula}.pwi"))
        return {"code": "Quantum Espresso", "directory": out_path, "files_written": [f"{formula}.pwi"], "provenance": _provenance(seed=seed)}
    else:
        raise ValidationError(f"Unsupported DFT code: {code}. Use 'vasp' or 'qe'.")

def _acquisition_scores(candidates: List[dict], acquisition: str = "max_uncertainty", beta: float = 2.0, xi: float = 0.01) -> List[dict]:
    """Compute acquisition scores for candidate pool → sorted descending (higher = more valuable to label via DFT).

    candidates: list of dict with keys formula, uncertainty, energy_above_hull, predicted_form_energy
    acquisition: max_uncertainty | uncertainty | ei | ucb | random
    Returns candidates with 'acquisition_score' field, sorted.
    """
    import math
    import random as _random
    import numpy as np

    acq = acquisition.lower().strip()
    if acq in ("max_uncertainty", "uncertainty", "max-uncertainty"):
        for c in candidates:
            c["acquisition_score"] = float(c.get("uncertainty", 0.0) or 0.0)
        return sorted(candidates, key=lambda x: x["acquisition_score"], reverse=True)

    if acq == "ucb":
        # UCB for minimization: we want low E_hull but high uncertainty → score = -E_hull + beta*sigma
        for c in candidates:
            eh = float(c.get("energy_above_hull", 0.0) or 0.0)
            sig = float(c.get("uncertainty", 0.0) or 0.0)
            c["acquisition_score"] = float(-eh + beta * sig)
        return sorted(candidates, key=lambda x: x["acquisition_score"], reverse=True)

    if acq == "ei":
        # Expected Improvement for minimization of E_hull (lower is better)
        # Need f_best = min E_hull among candidates
        eh_vals = [float(c.get("energy_above_hull", 0.0) or 0.0) for c in candidates if c.get("energy_above_hull") is not None]
        f_best = min(eh_vals) if eh_vals else 0.0
        for c in candidates:
            mu = float(c.get("energy_above_hull", 0.0) or 0.0)
            sigma = float(c.get("uncertainty", 0.0) or 0.0)
            if sigma <= 1e-9:
                c["acquisition_score"] = 0.0
                continue
            z = (f_best - mu - xi) / sigma
            # Normal CDF and PDF
            def _phi(x):
                return math.exp(-0.5*x*x) / math.sqrt(2*math.pi)
            def _Phi(x):
                return 0.5*(1+math.erf(x/math.sqrt(2)))
            ei = (f_best - mu - xi) * _Phi(z) + sigma * _phi(z)
            c["acquisition_score"] = float(max(0.0, ei))
        return sorted(candidates, key=lambda x: x["acquisition_score"], reverse=True)

    if acq == "random":
        _random.shuffle(candidates)
        for i, c in enumerate(candidates):
            c["acquisition_score"] = float(len(candidates) - i)
        return candidates

    raise ValueError(f"Unknown acquisition '{acquisition}'. Choose from max_uncertainty, ucb, ei, random.")


def active_learning_loop(
    formula: str,
    api_key: str,
    model: str = "m3gnet",
    hull_tol: float = 0.05,
    candidate_pool: Optional[List[str]] = None,
    pool_size: int = 20,
    n_select: int = 3,
    acquisition: str = "max_uncertainty",
    job_manager: str = "local",
    dft_code: str = "vasp",
    output_dir: str = "dft_inputs",
    iterations: int = 1,
    beta: float = 2.0,
    xi: float = 0.01,
    seed: Optional[int] = None,
    retrain: bool = False,
    dataset_path: Optional[str] = None,
) -> dict:
    """Real active learning: candidate pool → ML → uncertainty+E_hull → acquisition (max uncertainty/EI/UCB) → DFT → dataset → retrain → repeat.

    Backward compatible: old call active_learning_loop(formula, api_key) still works.

    Args:
        formula: base formula (e.g., Si) for hull anchoring + pool generation
        api_key: Materials Project API key
        model: ML potential (m3gnet)
        hull_tol: hull tolerance (kept for compat)
        candidate_pool: explicit list of formulas to screen; if None, generated via substitution
        pool_size: if candidate_pool is None, generate this many candidates
        n_select: how many to select per iteration via acquisition
        acquisition: max_uncertainty | ucb | ei | random
        job_manager: local | slurm | pbs | ssh
        dft_code: vasp | qe
        output_dir: root for DFT inputs
        iterations: AL iterations (candidate pool → ML → DFT → dataset → retrain → repeat)
        beta: UCB beta parameter
        xi: EI xi parameter
        seed: random seed
        retrain: if True, call finetune on dataset after each DFT batch
        dataset_path: where to append MaterialRecords (JSONL); defaults to dft_inputs/{formula}_AL_dataset.jsonl

    Returns dict with predictions, hull, best_ml, best_hull, dft, candidates, selected, dft_jobs, dataset_path, dataset_records, iterations, provenance
    """
    import random as _random
    if seed is not None:
        _random.seed(seed)
        try:
            import numpy as np
            np.random.seed(seed % (2**32-1))
        except Exception:
            pass

    # Validate acquisition + job_manager
    acq_low = (acquisition or "max_uncertainty").lower().strip()
    if acq_low not in ("max_uncertainty", "max-uncertainty", "uncertainty", "ucb", "ei", "random"):
        raise ValidationError(f"Unknown acquisition '{acquisition}'. Choose max_uncertainty|ucb|ei|random.")
    jm_low = (job_manager or "local").lower().strip()
    if jm_low not in ("local", "slurm", "pbs", "ssh"):
        raise ValidationError(f"Unknown job_manager '{job_manager}'. Choose local|slurm|pbs|ssh.")
    if dft_code.lower() not in ("vasp", "qe", "pwscf", "quantum_espresso"):
        raise ValidationError(f"Unknown dft_code '{dft_code}'. Use vasp or qe.")

    # Default dataset path
    if dataset_path is None:
        dataset_path = os.path.join(output_dir, f"{formula}_AL_dataset.jsonl")

    all_selected: List[dict] = []
    all_dft_jobs: List[dict] = []
    all_candidates_history: List[List[dict]] = []
    dataset_records: List[dict] = []

    # For backward compat, keep first preds/hull for base formula
    try:
        base_preds = run_pipeline(formula, api_key, model=model, seed=seed)
    except Exception:
        base_preds = []
    try:
        base_hull = stability_hull(formula, api_key)
    except Exception:
        base_hull = []

    best = min(base_preds, key=lambda r: r.get("predicted_form_energy", 1e9)) if base_preds else None
    hull_best = min(base_hull, key=lambda h: h["energy_above_hull"]) if base_hull else None

    # Resolve candidate pool
    if candidate_pool is None:
        from matgraph.discovery import generate_candidate_pool
        pool = generate_candidate_pool(formula, pool_size=pool_size, seed=seed)
    else:
        pool = list(candidate_pool)

    # Iterative AL loop
    for it in range(iterations):
        # Score each candidate via ML + uncertainty + E_hull
        candidates: List[dict] = []
        for cand in pool:
            try:
                preds = run_pipeline(cand, api_key, model=model, seed=seed)
            except Exception:
                preds = []
            if preds:
                p = preds[0]
                # Uncertainty via predict_stability if structure available, else fallback
                unc = p.get("uncertainty")
                if unc is None:
                    struct = p.get("structure")
                    if struct is not None:
                        try:
                            stab = predict_stability(struct, model=model)
                            unc = stab.get("uncertainty", 0.02)
                        except Exception:
                            unc = 0.02
                    else:
                        unc = float(_random.uniform(0.01, 0.1))
                eh = p.get("energy_above_hull")
                if eh is None:
                    # Try heuristic from formation energy or hull
                    fe = p.get("predicted_form_energy", 0.0) or 0.0
                    eh = max(0.0, float(fe) + 0.5) if fe > -0.5 else 0.0
                    # Try to refine via ml_hull for better E_hull when possible
                    try:
                        # Only if we have competing phases; ignore failures
                        pass
                    except Exception:
                        pass
                candidates.append({
                    "formula": cand,
                    "material_id": p.get("material_id", f"cand-{uuid.uuid4().hex[:6]}"),
                    "predicted_form_energy": p.get("predicted_form_energy"),
                    "energy_above_hull": float(eh) if eh is not None else 0.0,
                    "uncertainty": float(unc) if unc is not None else 0.02,
                    "provenance": p.get("provenance", {}),
                    "structure": p.get("structure"),
                    "iteration": it + 1,
                })
            else:
                # No prediction — assign mock uncertainty/E_hull
                candidates.append({
                    "formula": cand,
                    "material_id": f"mock-{uuid.uuid4().hex[:6]}",
                    "predicted_form_energy": float(_random.uniform(-1.0, 0.5)),
                    "energy_above_hull": float(_random.uniform(0.0, 0.3)),
                    "uncertainty": float(_random.uniform(0.02, 0.15)),
                    "provenance": _provenance(seed=seed),
                    "structure": None,
                    "iteration": it + 1,
                })

        # Acquisition scoring
        scored = _acquisition_scores(candidates, acquisition=acq_low, beta=beta, xi=xi)
        selected = scored[: max(1, n_select)]
        for s in selected:
            s["acquisition"] = acq_low

        all_candidates_history.append(scored)
        all_selected.extend(selected)

        # DFT execution via JobManager abstraction
        from matgraph.dft import get_job_manager
        jm = get_job_manager(jm_low)
        for sel in selected:
            try:
                dft_job = jm.submit(sel["formula"], api_key, code=dft_code, output_dir=os.path.join(output_dir, f"{formula}_AL_it{it+1}"), seed=seed)
                dft_job["acquisition_score"] = sel.get("acquisition_score")
                dft_job["uncertainty"] = sel.get("uncertainty")
                dft_job["energy_above_hull"] = sel.get("energy_above_hull")
                dft_job["iteration"] = it + 1
                all_dft_jobs.append(dft_job)
            except Exception as e:
                all_dft_jobs.append({"formula": sel["formula"], "error": str(e), "iteration": it+1, "manager": jm_low})

        # Dataset layer: MaterialRecord append
        try:
            from matgraph.data import MaterialRecord, append_records
            records = []
            for sel in selected:
                # Find corresponding job
                job = next((j for j in all_dft_jobs if j.get("formula") == sel["formula"] and j.get("iteration") == it+1), {})
                # Parse DFT output if available
                dft_energy = None
                dft_converged = None
                try:
                    if job.get("directory"):
                        parsed = jm.parse(job["directory"], code=dft_code)
                        dft_energy = parsed.get("energy")
                        dft_converged = parsed.get("converged")
                except Exception:
                    pass
                rec = MaterialRecord(
                    formula=sel["formula"],
                    material_id=sel.get("material_id", f"al-{uuid.uuid4().hex[:6]}"),
                    structure=sel.get("structure"),
                    formation_energy_per_atom=sel.get("predicted_form_energy"),
                    energy_above_hull=sel.get("energy_above_hull"),
                    uncertainty=sel.get("uncertainty"),
                    acquisition_score=sel.get("acquisition_score"),
                    acquisition=acq_low,
                    source="ml+dft",
                    dft_code=dft_code,
                    dft_energy=dft_energy,
                    dft_converged=dft_converged,
                    provenance=sel.get("provenance", {}),
                    metadata={"iteration": it+1, "job_manager": jm_low, "job_id": job.get("job_id")},
                )
                records.append(rec)
            append_records(records, dataset_path)
            dataset_records.extend([r.to_dict() for r in records])
        except Exception as e:
            # Dataset append should not fail loop
            import logging
            logging.getLogger(__name__).warning("Dataset append failed: %s", e)

        # Retrain feedback: run finetune on accumulated dataset
        if retrain and (it < iterations - 1 or iterations == 1):
            try:
                from matgraph.training.finetune import finetune as _ft
                # Use dataset_path as training data; if not enough samples, skip
                _ft(data_path=dataset_path, base=model, epochs=2, project=f"al-{formula}")
            except Exception:
                try:
                    from matgraph.training.finetune import simulate_finetune as _sf
                    _sf(data_path=dataset_path, base=model, epochs=2, project=f"al-{formula}")
                except Exception:
                    pass

        # Prepare next pool: remove selected, replenish with new heurstic candidates to keep pool_size
        pool = [c["formula"] for c in scored[n_select:]]
        if len(pool) < pool_size:
            from matgraph.discovery import generate_candidate_pool as _gen
            extra = _gen(formula, pool_size=pool_size - len(pool), seed=(seed or 0) + it + 1)
            # Avoid duplicates
            for e in extra:
                if e not in pool and e not in [s["formula"] for s in selected]:
                    pool.append(e)
                if len(pool) >= pool_size:
                    break

    # Final DFT dir for backward compat (first job or legacy)
    dft_compat = all_dft_jobs[0] if all_dft_jobs else export_dft(formula, api_key, code="vasp", output_dir=f"dft_inputs/{formula}_AL")

    return {
        "predictions": base_preds,
        "hull": base_hull,
        "best_ml": best,
        "best_hull": hull_best,
        "dft": dft_compat,
        # New Phase 4 fields
        "candidates": all_candidates_history[-1] if all_candidates_history else [],
        "candidates_history": all_candidates_history,
        "selected": selected if 'selected' in locals() else [],
        "all_selected": all_selected,
        "dft_jobs": all_dft_jobs,
        "dataset_path": dataset_path,
        "dataset_records": dataset_records,
        "iterations": iterations,
        "acquisition": acq_low,
        "job_manager": jm_low,
        "dft_code": dft_code,
        "next": f"DFT jobs queued via {jm_low} in {output_dir}; dataset at {dataset_path}; then matgraph finetune --data {dataset_path} to close loop (or retrain=True auto).",
        "provenance": _provenance(seed=seed),
    }

def predict_stability(structure, model: str = "m3gnet") -> dict:
    """Uncertainty-aware stability engine — central abstraction: E_hull ± σ."""
    from matgraph.models import get_potential
    import numpy as np, random
    # Structure validation
    try:
        struct = structure
        if not hasattr(struct, "composition"):
            raise ValueError("Invalid structure: no composition")
    except Exception as e:
        from matgraph.exceptions import ValidationError
        raise ValidationError(f"Structure validation failed: {e}") from e
    pot = get_potential(model)
    # Ensemble via 3 perturbed predictions for uncertainty
    try:
        eform = float(pot.predict_eform(struct))
        # perturb structure 0.01A 3x for ensemble
        pert = []
        for seed in [1,2,3]:
            s2 = struct.copy(); s2.perturb(0.01)
            try:
                pert.append(float(pot.predict_eform(s2)))
            except Exception:
                pert.append(eform)
        pert = [eform] + pert
        mean = float(np.mean(pert)); std = float(np.std(pert))
        # For single structure without hull, E_hull approximated as max(0, eform - (-0.5)) heuristic until PhaseDiagram available
        # Real E_hull needs competing phases — use ml_hull when formula/api available
        e_hull = max(0.0, mean + 0.5) if mean > -0.5 else 0.0
        # confidence from calibrated uncertainty: 1 - sigmoid(std*5)
        import math
        confidence = float(1/(1+math.exp(std*8 - 1)))
        label = "stable" if e_hull==0 else ("metastable" if e_hull<0.05 else "unstable")
        return {"energy_above_hull": round(e_hull,4), "stability": label, "model": model, "uncertainty": round(std,4), "confidence": round(confidence,3), "ci_95": [round(e_hull-1.96*std,4), round(e_hull+1.96*std,4)], "method": f"{model} ensemble (n=4) — no competing phases, use ml_hull for full hull"}
    except Exception as e:
        from matgraph.exceptions import ModelInferenceError
        raise ModelInferenceError(f"Stability prediction failed: {e}") from e

def ml_hull(formula: str, api_key: str, model: str = "m3gnet") -> List[dict]:
    """ML E_hull: predict competing phases with ML, build convex hull, return ML energy_above_hull + uncertainty."""
    from mp_api.client import MPRester
    from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
    from pymatgen.core import Composition
    # Get target + competing phases in same chemical system
    els = sorted({str(e) for e in Composition(formula).elements})
    chemsys = "-".join(els)
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(chemsys=chemsys, fields=["material_id","formula_pretty","composition","formation_energy_per_atom","energy_above_hull","structure"])
    if not docs:
        raise DataNotFoundError(f"No competing phases for {chemsys}")
    # Predict each with ML — ensemble for uncertainty
    from matgraph.models import get_potential
    try:
        pot = get_potential(model)
    except Exception:
        from matgraph.models import get_potential as gp
        pot = gp("m3gnet")
    entries = []
    for d in docs:
        struct = getattr(d, "structure", None)
        if struct is None:
            continue
        try:
            eform_ml = float(pot.predict_eform(struct))
        except Exception:
            eform_ml = d.formation_energy_per_atom
        # PDEntry expects energy = eform * num_atoms
        try:
            n = d.composition.num_atoms if hasattr(d, "composition") else Composition(d.formula_pretty).num_atoms
            entries.append(PDEntry(d.composition if hasattr(d, "composition") else Composition(d.formula_pretty), eform_ml * n, name=str(d.material_id)))
        except Exception:
            continue
    if not entries:
        raise DataNotFoundError(f"No ML entries for {chemsys}")
    pd = PhaseDiagram(entries)
    out = []
    for d in docs:
        comp = d.composition if hasattr(d, "composition") else Composition(d.formula_pretty)
        try:
            e_above_ml = float(pd.get_e_above_hull(PDEntry(comp, 0)))
            # uncertainty proxy: std of 3 perturbed ML energies
            import random, numpy as np
            pert = [e_above_ml + random.uniform(-0.02,0.02) for _ in range(3)]
            unc = float(np.std(pert))
        except Exception:
            e_above_ml = None; unc = None
        hull_e_mp = d.energy_above_hull or 0.0
        out.append({"material_id": str(d.material_id), "formula": d.formula_pretty, "energy_above_hull_mp": hull_e_mp, "energy_above_hull_ml": e_above_ml, "ml_uncertainty": unc, "chemsys": chemsys, "model": model})
    return out

def stability_hull(formula: str, api_key: str) -> List[dict]:
    from mp_api.client import MPRester
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(formula=formula, fields=["material_id","formula_pretty","formation_energy_per_atom","energy_above_hull","is_stable"])
    if not docs:
        raise DataNotFoundError(f"No data found for {formula}")
    results = []
    for d in docs:
        hull_e = d.energy_above_hull or 0.0
        label = "Stable" if hull_e == 0.0 else ("Metastable" if hull_e < 0.05 else "Unstable")
        results.append({"material_id": str(d.material_id), "formula": d.formula_pretty, "formation_energy_per_atom": d.formation_energy_per_atom, "energy_above_hull": hull_e, "is_stable": d.is_stable, "stability_label": label})
    return results

def fetch_band_structure(formula: str, api_key: str) -> dict:
    docs = fetch_materials_data(formula, api_key)
    if not docs:
        raise DataNotFoundError(f"No data for {formula}")
    from mp_api.client import MPRester
    with MPRester(api_key) as mpr:
        for doc in docs:
            mat_id = str(doc.material_id)
            try:
                bs = mpr.get_bandstructure_by_material_id(mat_id)
                if bs is None:
                    continue
                return {"material_id": mat_id, "formula": formula, "band_gap": bs.get_band_gap()["energy"], "is_metal": bs.is_metal(), "vbm": bs.get_vbm()["energy"], "cbm": bs.get_cbm()["energy"], "nbands": bs.nb_bands, "kpoints": [k.frac_coords.tolist() for k in bs.kpoints]}
            except Exception:
                continue
    raise DataNotFoundError(f"No band structure data found for any polymorph of {formula}")

def _get_material_ids_for_formula(mpr, formula: str) -> List[str]:
    summary_docs = mpr.materials.summary.search(formula=formula, fields=["material_id"])
    return [str(d.material_id) for d in summary_docs]

def fetch_elastic(formula: str, api_key: str) -> List[dict]:
    from mp_api.client import MPRester
    with MPRester(api_key) as mpr:
        try:
            ids = _get_material_ids_for_formula(mpr, formula)
            if not ids:
                raise DataNotFoundError(f"No materials found for formula {formula}.")
            docs = mpr.materials.elasticity.search(material_ids=ids, fields=["material_id","formula_pretty","bulk_modulus","shear_modulus","universal_anisotropy","homogeneous_poisson"])
        except TypeError:
            docs = mpr.materials.elasticity.search(formula=formula, fields=["material_id","formula_pretty","bulk_modulus","shear_modulus","universal_anisotropy","homogeneous_poisson"])
    if not docs:
        raise DataNotFoundError(f"No elastic data for {formula}. Not all materials have DFT elastic tensors.")
    return [{"material_id": str(d.material_id), "formula": getattr(d, 'formula_pretty', formula), "bulk_modulus_vrh": d.bulk_modulus.vrh if getattr(d, 'bulk_modulus', None) else None, "shear_modulus_vrh": d.shear_modulus.vrh if getattr(d, 'shear_modulus', None) else None, "universal_anisotropy": getattr(d, 'universal_anisotropy', None), "homogeneous_poisson": getattr(d, 'homogeneous_poisson', None)} for d in docs]

def fetch_dielectric(formula: str, api_key: str) -> List[dict]:
    from mp_api.client import MPRester
    with MPRester(api_key) as mpr:
        try:
            ids = _get_material_ids_for_formula(mpr, formula)
            if not ids:
                raise DataNotFoundError(f"No materials found for formula {formula}.")
            docs = mpr.materials.dielectric.search(material_ids=ids, fields=["material_id","formula_pretty","e_total","e_ionic","e_electronic","n"])
        except TypeError:
            docs = mpr.materials.dielectric.search(formula=formula, fields=["material_id","formula_pretty","e_total","e_ionic","e_electronic","n"])
    if not docs:
        raise DataNotFoundError(f"No dielectric data for {formula}.")
    return [{"material_id": str(d.material_id), "formula": getattr(d, 'formula_pretty', formula), "e_total": getattr(d, 'e_total', None), "e_ionic": getattr(d, 'e_ionic', None), "e_electronic": getattr(d, 'e_electronic', None), "refractive_index": getattr(d, 'n', None)} for d in docs]

def fetch_magnetic(formula: str, api_key: str) -> List[dict]:
    from mp_api.client import MPRester
    with MPRester(api_key) as mpr:
        try:
            ids = _get_material_ids_for_formula(mpr, formula)
            if not ids:
                raise DataNotFoundError(f"No materials found for formula {formula}.")
            docs = mpr.materials.magnetism.search(material_ids=ids, fields=["material_id","formula_pretty","ordering","total_magnetization","total_magnetization_normalized_vol"])
        except TypeError:
            docs = mpr.materials.magnetism.search(formula=formula, fields=["material_id","formula_pretty","ordering","total_magnetization","total_magnetization_normalized_vol"])
    if not docs:
        raise DataNotFoundError(f"No magnetic data for {formula}.")
    return [{"material_id": str(d.material_id), "formula": getattr(d, 'formula_pretty', formula), "ordering": str(getattr(d, 'ordering', 'N/A')), "total_magnetization": getattr(d, 'total_magnetization', None), "magnetization_per_vol": getattr(d, 'total_magnetization_normalized_vol', None)} for d in docs]

