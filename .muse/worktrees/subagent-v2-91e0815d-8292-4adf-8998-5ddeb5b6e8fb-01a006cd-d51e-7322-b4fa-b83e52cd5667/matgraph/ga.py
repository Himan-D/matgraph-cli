import random
import logging
from typing import List, Dict, Any, Optional
from pymatgen.core import Structure
from pymatgen.transformations.standard_transformations import SubstitutionTransformation, PerturbStructureTransformation
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import FIRE
import numpy as np

from matgraph.core import get_matgl_pes_model, get_matgl_eform_model, fetch_materials_data
from matgraph.settings import settings

logger = logging.getLogger(__name__)

class CrystalGA:
    def __init__(self, base_formula: str, api_key: str, population_size: int = 10, target_property: str = "formation_energy",
                 allowed_elements: Optional[List[str]] = None, seed: Optional[int] = None,
                 mutate_intensity: Optional[float] = None, init_mutate_intensity: Optional[float] = None,
                 scale_jitter: Optional[float] = None, relax_fmax: Optional[float] = None, relax_steps: Optional[int] = None,
                 elite_frac: Optional[float] = None):
        self.base_formula = base_formula
        self.api_key = api_key
        self.population_size = population_size
        self.target_property = target_property
        self.population: List[Structure] = []
        self.seed = seed
        if seed is not None:
            random.seed(seed); np.random.seed(seed % (2**32-1))
        # all tunables from settings, not hardcode
        self.allowed_elements = allowed_elements if allowed_elements is not None else list(settings.ga_allowed_elements)
        self.mutate_intensity = mutate_intensity if mutate_intensity is not None else settings.ga_mutate_intensity
        self.init_mutate_intensity = init_mutate_intensity if init_mutate_intensity is not None else settings.ga_init_mutate_intensity
        self.scale_jitter = scale_jitter if scale_jitter is not None else settings.ga_scale_jitter
        self.relax_fmax = relax_fmax if relax_fmax is not None else settings.ga_relax_fmax
        self.relax_steps = relax_steps if relax_steps is not None else settings.ga_relax_steps
        self.elite_frac = elite_frac if elite_frac is not None else settings.ga_elite_frac

    def _initialize_population(self):
        docs = fetch_materials_data(self.base_formula, self.api_key)
        if not docs or not docs[0].structure:
            raise ValueError(f"Could not fetch baseline structure for {self.base_formula}")
        base_structure = docs[0].structure
        self.population.append(base_structure)
        for _ in range(self.population_size - 1):
            mutated = self._mutate(base_structure, intensity=self.init_mutate_intensity)
            self.population.append(mutated)

    def _mutate(self, structure: Structure, intensity: Optional[float] = None) -> Structure:
        if intensity is None:
            intensity = self.mutate_intensity
        new_struct = structure.copy()
        mutation_type = random.choice(["perturb", "substitute", "scale"])
        try:
            if mutation_type == "perturb":
                trans = PerturbStructureTransformation(distance=float(intensity) * 2.0)
                new_struct = trans.apply_transformation(new_struct)
            elif mutation_type == "substitute":
                elements_in_struct = [str(el) for el in new_struct.composition.elements]
                el_to_replace = random.choice(elements_in_struct)
                candidates = [e for e in self.allowed_elements if e != el_to_replace]
                if not candidates:
                    return new_struct
                new_el = random.choice(candidates)
                trans = SubstitutionTransformation({el_to_replace: new_el})
                new_struct = trans.apply_transformation(new_struct)
            elif mutation_type == "scale":
                scale_factor = 1.0 + random.uniform(-self.scale_jitter, self.scale_jitter) * float(intensity)
                # scale_lattice expects new volume
                new_struct.scale_lattice(new_struct.volume * scale_factor)
        except Exception as e:
            logger.debug("mutate failed %s->%s: %s", mutation_type, e, exc_info=True)
        return new_struct

    def _crossover(self, parent1: Structure, parent2: Structure) -> Structure:
        child = parent1.copy()
        p2_elements = [str(el) for el in parent2.composition.elements]
        p1_elements = [str(el) for el in child.composition.elements]
        if set(p1_elements) != set(p2_elements):
            try:
                el_to_add = random.choice(list(set(p2_elements) - set(p1_elements)))
                el_to_remove = random.choice(p1_elements)
                trans = SubstitutionTransformation({el_to_remove: el_to_add})
                child = trans.apply_transformation(child)
            except Exception as e:
                logger.debug("crossover failed: %s", e)
        return child

    def _evaluate(self, structure: Structure) -> float:
        from matgl.ext.ase import M3GNetCalculator
        pot = get_matgl_pes_model()
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = M3GNetCalculator(potential=pot)
        dyn = FIRE(atoms, logfile=None)
        dyn.run(fmax=self.relax_fmax, steps=self.relax_steps)
        relaxed_struct = AseAtomsAdaptor.get_structure(atoms)
        eform_model = get_matgl_eform_model()
        formation_energy = float(eform_model.predict_structure(relaxed_struct).detach().item())
        for i, site in enumerate(relaxed_struct):
            structure[i].frac_coords = site.frac_coords
        structure.lattice = relaxed_struct.lattice
        return formation_energy

    def run(self, generations: int = 5) -> List[Dict[str, Any]]:
        self._initialize_population()
        history = []
        for gen in range(generations):
            scored_population = []
            for struct in self.population:
                try:
                    fitness = self._evaluate(struct)
                    scored_population.append((fitness, struct))
                except Exception as e:
                    logger.warning("GA evaluate failed gen %d: %s", gen+1, e)
                    scored_population.append((float("inf"), struct))
            scored_population.sort(key=lambda x: x[0])
            best_fitness, best_struct = scored_population[0]
            history.append({"generation": gen + 1, "best_formula": best_struct.composition.reduced_formula, "best_fitness": best_fitness, "structure": best_struct})
            elite_count = max(1, int(self.population_size * self.elite_frac))
            next_generation = [s for f, s in scored_population[:elite_count]]
            while len(next_generation) < self.population_size:
                if random.random() < 0.3 and len(scored_population) > 1:
                    p1 = random.choice(scored_population[:elite_count])[1]
                    p2 = random.choice(scored_population[:elite_count])[1]
                    child = self._crossover(p1, p2)
                    next_generation.append(child)
                else:
                    parent = random.choice(scored_population[:elite_count])[1]
                    child = self._mutate(parent)
                    next_generation.append(child)
            self.population = next_generation
        return history
