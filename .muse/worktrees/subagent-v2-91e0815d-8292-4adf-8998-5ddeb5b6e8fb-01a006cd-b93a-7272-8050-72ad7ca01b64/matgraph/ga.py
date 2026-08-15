import random
import copy
from typing import List, Dict, Any
from pymatgen.core import Structure
from pymatgen.transformations.standard_transformations import SubstitutionTransformation, PerturbStructureTransformation
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import FIRE
import numpy as np

# We import the cached models from core
from matgraph.core import get_matgl_pes_model, get_matgl_eform_model, fetch_materials_data

class CrystalGA:
    def __init__(self, base_formula: str, api_key: str, population_size: int = 10, target_property: str = "formation_energy"):
        self.base_formula = base_formula
        self.api_key = api_key
        self.population_size = population_size
        self.target_property = target_property
        self.population: List[Structure] = []
        
        # We need elements to mutate into. Let's use a list of common solid-state elements
        self.allowed_elements = ["Li", "Na", "K", "Mg", "Ca", "Fe", "Co", "Ni", "Mn", "Ti", "V", "O", "S", "P", "Si"]

    def _initialize_population(self):
        """Fetch the base structure and generate initial mutants."""
        docs = fetch_materials_data(self.base_formula, self.api_key)
        if not docs or not docs[0].structure:
            raise ValueError(f"Could not fetch baseline structure for {self.base_formula}")
            
        base_structure = docs[0].structure
        self.population.append(base_structure)
        
        # Generate initial population via random permutations
        for _ in range(self.population_size - 1):
            mutated = self._mutate(base_structure, intensity=0.2)
            self.population.append(mutated)

    def _mutate(self, structure: Structure, intensity: float = 0.1) -> Structure:
        """Apply random mutations: coordinate perturbation, lattice scaling, or elemental substitution."""
        new_struct = structure.copy()
        mutation_type = random.choice(["perturb", "substitute", "scale"])
        
        try:
            if mutation_type == "perturb":
                # Randomly perturb atomic coordinates
                trans = PerturbStructureTransformation(distance=intensity * 2.0)
                new_struct = trans.apply_transformation(new_struct)
                
            elif mutation_type == "substitute":
                # Pick a random site and substitute its species
                elements_in_struct = [str(el) for el in new_struct.composition.elements]
                el_to_replace = random.choice(elements_in_struct)
                new_el = random.choice([e for e in self.allowed_elements if e != el_to_replace])
                trans = SubstitutionTransformation({el_to_replace: new_el})
                new_struct = trans.apply_transformation(new_struct)
                
            elif mutation_type == "scale":
                # Scale the lattice volume by +/- 5%
                scale_factor = 1.0 + random.uniform(-0.05, 0.05) * intensity
                new_struct.scale_lattice(new_struct.volume * scale_factor)
        except Exception:
            pass # Fallback to original if transformation fails (e.g., incompatible substitution)
            
        return new_struct

    def _crossover(self, parent1: Structure, parent2: Structure) -> Structure:
        """Very simple crossover: take lattice from parent1, but try to mix species."""
        child = parent1.copy()
        # In a real GA, you'd slice the fractional coordinates. 
        # Here we just introduce a random species from parent2 into parent1.
        p2_elements = [str(el) for el in parent2.composition.elements]
        p1_elements = [str(el) for el in child.composition.elements]
        
        if set(p1_elements) != set(p2_elements):
            try:
                el_to_add = random.choice(list(set(p2_elements) - set(p1_elements)))
                el_to_remove = random.choice(p1_elements)
                trans = SubstitutionTransformation({el_to_remove: el_to_add})
                child = trans.apply_transformation(child)
            except Exception:
                pass
        return child

    def _evaluate(self, structure: Structure) -> float:
        """Evaluate fitness using M3GNet. Lower is better (we are minimizing formation energy)."""
        # Relax the structure first so we don't evaluate unphysical states
        from matgl.ext.ase import M3GNetCalculator
        pot = get_matgl_pes_model()
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = M3GNetCalculator(potential=pot)
        
        dyn = FIRE(atoms, logfile=None)
        dyn.run(fmax=0.1, steps=20) # Fast relaxation
        
        relaxed_struct = AseAtomsAdaptor.get_structure(atoms)
        
        eform_model = get_matgl_eform_model()
        formation_energy = float(eform_model.predict_structure(relaxed_struct).detach().item())
        
        # Update the structure with its relaxed coordinates
        for i, site in enumerate(relaxed_struct):
            structure[i].frac_coords = site.frac_coords
        structure.lattice = relaxed_struct.lattice
            
        return formation_energy

    def run(self, generations: int = 5) -> List[Dict[str, Any]]:
        """Run the genetic algorithm evolution."""
        self._initialize_population()
        
        history = []
        
        for gen in range(generations):
            # Evaluate all
            scored_population = []
            for struct in self.population:
                try:
                    fitness = self._evaluate(struct)
                    scored_population.append((fitness, struct))
                except Exception as e:
                    # Penalize failed evaluations
                    scored_population.append((999.0, struct))
            
            # Sort by fitness (minimize formation energy)
            scored_population.sort(key=lambda x: x[0])
            
            best_fitness, best_struct = scored_population[0]
            history.append({
                "generation": gen + 1,
                "best_formula": best_struct.composition.reduced_formula,
                "best_fitness": best_fitness,
                "structure": best_struct
            })
            
            # Elitism: keep top 20%
            elite_count = max(1, int(self.population_size * 0.2))
            next_generation = [s for f, s in scored_population[:elite_count]]
            
            # Crossover and Mutation to fill the rest
            while len(next_generation) < self.population_size:
                if random.random() < 0.3 and len(scored_population) > 1:
                    # Crossover
                    p1 = random.choice(scored_population[:elite_count])[1]
                    p2 = random.choice(scored_population[:elite_count])[1]
                    child = self._crossover(p1, p2)
                    next_generation.append(child)
                else:
                    # Mutate
                    parent = random.choice(scored_population[:elite_count])[1]
                    child = self._mutate(parent)
                    next_generation.append(child)
                    
            self.population = next_generation
            
        return history
