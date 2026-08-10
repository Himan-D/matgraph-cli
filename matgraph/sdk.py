import os
from typing import Optional, List, Dict, Any
from matgraph.core import run_pipeline, substitute_material, simulate_xrd, fetch_materials_data, fetch_phonon_dos, inverse_design, relax_structure

class MatGraphSDK:
    """
    Python SDK for MatGraph.
    Perfect for Jupyter Notebooks, ML pipelines, and custom Python scripts.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("MP_API_KEY")
        if not self.api_key:
            raise ValueError("Materials Project API key is required. Pass it or set MP_API_KEY.")

    def predict(self, formula: str, model: str = "m3gnet", min_gap: Optional[float] = None, max_gap: Optional[float] = None, crystal_system: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Runs the full ML prediction pipeline on a material.
        
        Args:
            formula: Chemical formula (e.g., 'LiFePO4')
            model: 'm3gnet' (legacy 'cgcnn', 'megnet' also route to 'm3gnet' now)
            min_gap: Minimum true band gap
            max_gap: Maximum true band gap
            crystal_system: e.g., 'Cubic'
            
        Returns:
            List of dictionaries containing predictions, true values, and structural data.
        """
        return run_pipeline(
            formula=formula,
            api_key=self.api_key,
            min_gap=min_gap,
            max_gap=max_gap,
            crystal_system=crystal_system,
            model=model
        )

    def evaluate(self, formula: str, model: str = "m3gnet") -> Dict[str, float]:
        """
        Evaluates the Mean Absolute Error (MAE) for a given formula across available polymorphs.
        
        Returns:
            Dictionary with 'band_gap_mae' and 'formation_energy_mae'.
        """
        results = self.predict(formula, model=model)
        gap_errors, form_errors = [], []
        
        for r in results:
            if r.get("true_band_gap") is not None and r.get("predicted_band_gap") is not None:
                gap_errors.append(abs(r["true_band_gap"] - r["predicted_band_gap"]))
            if r.get("true_form_energy") is not None and r.get("predicted_form_energy") is not None:
                form_errors.append(abs(r["true_form_energy"] - r["predicted_form_energy"]))
                
        return {
            "band_gap_mae": sum(gap_errors) / len(gap_errors) if gap_errors else 0.0,
            "formation_energy_mae": sum(form_errors) / len(form_errors) if form_errors else 0.0,
            "samples_evaluated": len(results)
        }

    def substitute(self, formula: str, element_out: str, element_in: str) -> Dict[str, Any]:
        """
        Simulate generative discovery by substituting elements and predicting thermodynamic stability.
        
        Args:
            formula: Base material formula (e.g., 'LiFePO4')
            element_out: Element to remove (e.g., 'Li')
            element_in: Element to insert (e.g., 'Na')
            
        Returns:
            Dictionary comparing the original and hypothetical structures' stability.
        """
        return substitute_material(formula, element_out, element_in, self.api_key)

    def xrd(self, formula: str) -> Dict[str, Any]:
        """
        Simulates the X-Ray Diffraction (XRD) pattern for the most stable polymorph of a formula.
        
        Returns:
            Dictionary with 'two_theta', 'intensity', and 'hkls' arrays.
        """
        docs = fetch_materials_data(formula, self.api_key)
        if not docs or not docs[0].structure:
            raise ValueError(f"No crystal structure found for {formula}")
            
        return simulate_xrd(docs[0].structure)
        
    def phonon_dos(self, formula: str, method: str = "dfpt") -> Dict[str, Any]:
        """
        Fetches the Phonon Density of States (DOS) for the most stable polymorph of a formula.
        
        Args:
            formula: Chemical formula (e.g., 'Si', 'NaCl')
            method: 'dfpt', 'finite_difference', or 'line_mode'
            
        Returns:
            Dictionary containing frequencies and densities arrays.
        """
        return fetch_phonon_dos(formula, self.api_key, phonon_method=method)

    def design(self, min_gap: float = None, max_gap: float = None, crystal_system: str = None, exclude_elements: List[str] = None, include_elements: List[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Inverse design: Generates or searches for materials that match specific properties.
        """
        return inverse_design(
            api_key=self.api_key,
            min_gap=min_gap,
            max_gap=max_gap,
            crystal_system=crystal_system,
            exclude_elements=exclude_elements,
            include_elements=include_elements,
            limit=limit
        )

    def relax(self, formula: str, steps: int = 10) -> Dict[str, Any]:
        """
        Relax a crystal structure using the MatGraph Universal Potential (M3GNet) and ASE.
        """
        return relax_structure(formula, self.api_key, steps=steps)
