import typer
from rich.console import Console
from rich.table import Table
import os
from typing import Optional
from matgraph.core import run_pipeline, save_results, substitute_material, simulate_xrd
from matgraph.sdk import MatGraphSDK
from matgraph.auth import generate_api_key
from matgraph.config import get_api_key, save_api_key

import sys
import importlib.metadata

app = typer.Typer(help="MatGraph CLI: Deep Learning for Material Science", no_args_is_help=True)
api_app = typer.Typer(help="Manage API keys for the GraphQL Server")
cache_app = typer.Typer(help="Manage the local query cache")
app.add_typer(api_app, name="auth")
app.add_typer(cache_app, name="cache")
console = Console()

@cache_app.command("stats")
def cache_stats_cmd():
    """Show cache size and entry count."""
    from matgraph.cdn import cache_stats
    stats = cache_stats()
    console.print(f"Entries: [bold cyan]{stats['entries']}[/bold cyan]")
    console.print(f"Size: [bold cyan]{stats['size_mb']} MB[/bold cyan]")
    console.print(f"Location: [bold]{stats['location']}[/bold]")

@cache_app.command("clear")
def cache_clear_cmd():
    """Clear all cached query results."""
    from matgraph.cdn import cache_clear
    cache_clear()
    console.print("[bold green]Cache cleared.[/bold green]")

@api_app.command("generate")
def generate_key(user: str = typer.Option(..., "--user", "-u", help="Username or identifier for this API key")):
    """Generates a secure API key for authenticating with the MatGraph GraphQL Server."""
    console.print(f"[bold cyan]Generating API Key for {user}...[/bold cyan]")
    key = generate_api_key(user)
    console.print(f"[bold green]Success![/bold green] Your API Key is: [bold yellow]{key}[/bold yellow]")
    console.print("[bold red]Please save this key securely! It grants access to the GraphQL API.[/bold red]")

def version_callback(value: bool):
    if value:
        try:
            version = importlib.metadata.version("matgraph-cli")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        console.print(f"MatGraph CLI Version: [bold green]{version}[/bold green]")
        console.print(f"Python Version: [bold cyan]{sys.version.split()[0]}[/bold cyan]")
        raise typer.Exit()

@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True, help="Show the application and Python version."
    )
):
    pass

@app.command()
def setup(api_key: str):
    """Set up your Materials Project API Key."""
    save_api_key(api_key)
    console.print(f"[bold green]Success![/bold green] Materials Project API key saved securely.")
    console.print("[dim]It is stored in ~/.matgraph/config.json[/dim]")

@app.command()
def predict(
    formula: str, 
    min_gap: Optional[float] = typer.Option(None, "--min-gap", help="Minimum true band gap to filter (eV)"),
    max_gap: Optional[float] = typer.Option(None, "--max-gap", help="Maximum true band gap to filter (eV)"),
    crystal_system: Optional[str] = typer.Option(None, "--crystal-system", help="Filter by crystal system (e.g., Cubic, Hexagonal)"),
    save: Optional[str] = typer.Option(None, "--save", help="File path to save results (e.g., results.csv)"),
    format: str = typer.Option("csv", "--format", help="Save format: 'csv' or 'json'"),
    model: str = typer.Option("m3gnet", "--model", help="Model to use for prediction: 'm3gnet'"),
    cif: bool = typer.Option(False, "--cif", help="Export the raw crystal structure of the results to .cif files")
):
    """Run the complete ML pipeline with advanced search filters and data saving."""
    api_key = get_api_key()
    if not api_key:
        console.print("[red]Error: MP_API_KEY is not set.[/red]")
        raise typer.Exit(code=1)
    
    console.print(f"[cyan]Searching and running ML pipeline for {formula} using {model.upper()}...[/cyan]")
    if min_gap or max_gap or crystal_system:
        console.print(f"[dim]Filters applied - Min Gap: {min_gap}, Max Gap: {max_gap}, System: {crystal_system}[/dim]")
    
    try:
        results = run_pipeline(
            formula=formula, 
            api_key=api_key, 
            min_gap=min_gap, 
            max_gap=max_gap, 
            crystal_system=crystal_system,
            model=model
        )
        
        if not results:
            console.print(f"[yellow]No materials found matching the criteria.[/yellow]")
            return
            
        table = Table(title=f"Prediction Results for {formula} ({model.upper()})")
        table.add_column("ID", style="cyan")
        table.add_column("Formula", style="magenta")
        
        if model.lower() == "m3gnet":
            table.add_column("Energy (eV)", style="bold blue")
            table.add_column("Max Force", style="green")
            table.add_column("Crystal Sys", style="yellow")
        else:
            table.add_column("True Gap (eV)", style="green")
            table.add_column("Pred Gap", style="bold blue")
            table.add_column("True Form (eV)", style="green")
            table.add_column("Pred Form", style="bold blue")
        
        for r in results:
            if model.lower() == "m3gnet":
                import numpy as np
                energy = f"{r['m3gnet_energy']:.3f}" if r['m3gnet_energy'] is not None else "N/A"
                max_force = f"{np.max(np.abs(r['m3gnet_forces'])):.3f}" if r.get('m3gnet_forces') else "N/A"
                table.add_row(str(r["material_id"]).replace("mp-", ""), r["formula"], energy, max_force, r["crystal_system"])
            else:
                true_form = str(round(r["true_form_energy"], 3)) if r["true_form_energy"] is not None else "N/A"
                true_gap = str(round(r["true_band_gap"], 3)) if r["true_band_gap"] is not None else "N/A"
                pred_gap = str(round(r["predicted_band_gap"], 3)) if r["predicted_band_gap"] is not None else "N/A"
                pred_form = str(round(r["predicted_form_energy"], 3)) if r["predicted_form_energy"] is not None else "N/A"
                
                table.add_row(
                    str(r["material_id"]).replace("mp-", ""),
                    r["formula"],
                    true_gap,
                    pred_gap,
                    true_form,
                    pred_form
                )
            
            if cif and r.get("structure"):
                cif_filename = f"{r['material_id']}_{r['formula']}.cif"
                r["structure"].to(filename=cif_filename)
                console.print(f"[dim]Exported structure to {cif_filename}[/dim]")
            
        console.print(table)
        console.print(f"[green]Pipeline completed successfully! Found {len(results)} items.[/green]")
        
        if save:
            save_results(results, save, format.lower())
            console.print(f"[bold green]Saved {len(results)} results to {save} (Format: {format.upper()})[/bold green]")

    except Exception as e:
        console.print(f"[red]Pipeline Error: {e}[/red]")

@app.command()
def xrd(formula: str):
    """Simulate X-Ray Diffraction (XRD) patterns for materials."""
    from matgraph.core import fetch_materials_data, simulate_xrd
    api_key = get_api_key()
    if not api_key:
        console.print("[red]Error: MP_API_KEY is not set.[/red]")
        raise typer.Exit(code=1)
        
    console.print(f"[cyan]Fetching structures and simulating XRD for {formula}...[/cyan]")
    try:
        docs = fetch_materials_data(formula, api_key)
        if not docs:
            console.print("[yellow]No materials found.[/yellow]")
            return
            
        # Simulate XRD for the best match
        doc = docs[0]
        if not doc.structure:
            console.print("[red]No structure available to simulate XRD.[/red]")
            return
            
        xrd_data = simulate_xrd(doc.structure)
        
        table = Table(title=f"XRD Peaks (Cu-Ka) for {doc.formula_pretty} ({doc.material_id})")
        table.add_column("2θ (degrees)", justify="right", style="cyan")
        table.add_column("Intensity (%)", justify="right", style="green")
        table.add_column("hkl", style="magenta")
        
        # Display top 10 peaks
        peaks = list(zip(xrd_data["two_theta"], xrd_data["intensity"], xrd_data["hkls"]))
        peaks.sort(key=lambda x: x[1], reverse=True)
        
        for theta, intensity, hkl in peaks[:10]:
            hkl_str = str(hkl[0]) # take the primary hkl family
            table.add_row(f"{theta:.2f}", f"{intensity:.1f}", hkl_str)
            
        console.print(table)
        console.print(f"[dim]Showing top 10 peaks. Use python API for full pattern export.[/dim]")
    except Exception as e:
        console.print(f"[red]XRD Error: {e}[/red]")

@app.command()
def evaluate(formula: str, model: str = typer.Option("m3gnet", "--model", help="Model to evaluate")):
    """Evaluate model accuracy (MAE) against true Materials Project data."""
    api_key = get_api_key()
    if not api_key:
        console.print("[red]Error: MP_API_KEY is not set.[/red]")
        raise typer.Exit(code=1)
        
    console.print(f"[cyan]Evaluating {model.upper()} accuracy for {formula} variants...[/cyan]")
    try:
        results = run_pipeline(formula=formula, api_key=api_key, model=model)
        
        if not results:
            console.print("[yellow]No data found for evaluation.[/yellow]")
            return
            
        gap_errors = []
        form_errors = []
        
        for r in results:
            if r.get("true_band_gap") is not None and r.get("predicted_band_gap") is not None:
                gap_errors.append(abs(r["true_band_gap"] - r["predicted_band_gap"]))
            if r.get("true_form_energy") is not None and r.get("predicted_form_energy") is not None:
                form_errors.append(abs(r["true_form_energy"] - r["predicted_form_energy"]))
                
        gap_mae = sum(gap_errors) / len(gap_errors) if gap_errors else 0.0
        form_mae = sum(form_errors) / len(form_errors) if form_errors else 0.0
        
        console.print(f"\n[bold magenta]Model Analytics Report ({model.upper()})[/bold magenta]")
        console.print(f"Total Samples Evaluated: {len(results)}")
        if gap_errors:
            console.print(f"Band Gap MAE: [bold yellow]{gap_mae:.3f} eV[/bold yellow]")
        if form_errors:
            console.print(f"Formation Energy MAE: [bold yellow]{form_mae:.3f} eV/atom[/bold yellow]\n")
        
    except Exception as e:
        console.print(f"[red]Evaluation Error: {e}[/red]")

@app.command()
def substitute(formula: str, elem_out: str, elem_in: str):
    """
    (GNoME-inspired) Perform hypothetical elemental substitution to predict stability of a new material.
    e.g., matgraph substitute LiFePO4 Li Na
    """
    from matgraph.core import substitute_material
    api_key = get_api_key()
    if not api_key:
        console.print("[red]Error: MP_API_KEY is not set.[/red]")
        raise typer.Exit(code=1)
        
    console.print(f"[cyan]Generative Discovery: Substituting {elem_out} with {elem_in} in {formula}...[/cyan]")
    try:
        res = substitute_material(formula, elem_out, elem_in, api_key)
        
        table = Table(title=f"Thermodynamic Stability Analysis")
        table.add_column("Material", style="magenta")
        table.add_column("Predicted Energy (M3GNet)", style="blue")
        table.add_column("Status", style="bold")
        
        o_energy = res['original']['energy']
        h_energy = res['hypothetical']['energy']
        
        table.add_row(res['original']['formula'] + " (Baseline)", f"{o_energy:.3f} eV", "[green]Known[/green]")
        
        status_color = "[bold green]Stable Candidate[/bold green]" if res['is_more_stable'] else "[bold red]Likely Unstable[/bold red]"
        table.add_row(res['hypothetical']['formula'] + " (Generated)", f"{h_energy:.3f} eV", status_color)
        
        console.print(table)
        
        if res['is_more_stable']:
            console.print(f"\n🎉 [bold green]Discovery![/bold green] The hypothetical material {res['hypothetical']['formula']} is predicted to be thermodynamically MORE stable than the baseline!")
        else:
            console.print(f"\n[yellow]Analysis:[/yellow] The substitution increases internal energy. {res['hypothetical']['formula']} might decompose.")
            
    except Exception as e:
        console.print(f"[red]Substitution Error: {e}[/red]")

@app.command()
def phonon(formula: str, method: str = typer.Option("dfpt", "--method", help="Phonon calculation method ('dfpt', 'finite_difference', 'line_mode')")):
    """
    Fetch and display the Phonon Density of States (DOS).
    """
    from matgraph.core import fetch_phonon_dos
    api_key = get_api_key()
    if not api_key:
        console.print("[red]Error: MP_API_KEY is not set.[/red]")
        raise typer.Exit(code=1)
        
    console.print(f"[cyan]Fetching Phonon DOS for {formula} using method '{method}'...[/cyan]")
    try:
        dos_data = fetch_phonon_dos(formula, api_key, phonon_method=method)
        freqs = dos_data["frequencies"]
        if method == "dfpt" and dos_data.get("frequencies") is not None:
            console.print(f"Phonon DOS fetched successfully via {method} for {formula}.")
            console.print(f"Number of frequency points: {len(dos_data['frequencies'])}")
            
            # Simple text-based plot simulation
            console.print("[bold cyan]Frequency vs DOS (Preview):[/bold cyan]")
            step = max(1, len(dos_data['frequencies']) // 10)
            for i in range(0, len(dos_data['frequencies']), step):
                freq = dos_data['frequencies'][i]
                dens = dos_data['densities'][i]
                bar = "#" * int(dens * 10)
                console.print(f"{freq:8.2f} THz | {bar}")
        else:
            console.print(f"Fetched Phonon data for {formula} via {method}.")
        
    except Exception as e:
        console.print(f"[red]Phonon Error: {e}[/red]")

@app.command()
def design(
    min_gap: float = typer.Option(0.0, "--min-gap", help="Minimum band gap (eV)"),
    max_gap: float = typer.Option(10.0, "--max-gap", help="Maximum band gap (eV)"),
    crystal_system: str = typer.Option(None, "--crystal-system", help="Target crystal system (e.g. Cubic, Hexagonal)"),
    exclude: str = typer.Option(None, "--exclude", help="Comma-separated elements to exclude"),
    include: str = typer.Option(None, "--include", help="Comma-separated elements to include"),
    limit: int = typer.Option(10, "--limit", help="Max results to fetch"),
    api_key: str = typer.Option(None, envvar="MP_API_KEY", help="Materials Project API Key")
):
    """
    Inverse design: Search for materials by target properties.
    """
    sdk = MatGraphSDK(api_key=api_key)
    exclude_elements = exclude.split(",") if exclude else None
    include_elements = include.split(",") if include else None
    
    with console.status(f"[bold green]Searching for materials with gap {min_gap}-{max_gap} eV..."):
        results = sdk.design(
            min_gap=min_gap, max_gap=max_gap, 
            crystal_system=crystal_system, 
            exclude_elements=exclude_elements,
            include_elements=include_elements,
            limit=limit
        )
        
    if not results:
        console.print("[bold red]No materials found matching criteria.[/bold red]")
        return
        
    table = Table(title=f"Inverse Design Results ({len(results)} found)")
    table.add_column("Formula", style="cyan")
    table.add_column("Band Gap (eV)", style="magenta")
    table.add_column("Crystal System", style="blue")
    table.add_column("Stable?", style="green")
    
    for r in results:
        stable_str = "Yes" if r["is_stable"] else "No"
        table.add_row(r["formula"], f"{r['band_gap']:.3f}", r["crystal_system"], stable_str)
        
    console.print(table)
    
@app.command()
def relax(
    formula: str,
    steps: int = typer.Option(10, "--steps", help="Number of relaxation steps"),
    api_key: str = typer.Option(None, envvar="MP_API_KEY", help="Materials Project API Key")
):
    """
    Relax a crystal structure using the MatGraph Universal Potential (M3GNet) and ASE.
    """
    sdk = MatGraphSDK(api_key=api_key)
    with console.status(f"[bold green]Relaxing {formula} structure with M3GNet + ASE for {steps} steps..."):
        try:
            result = sdk.relax(formula, steps=steps)
            
            console.print(f"[bold cyan]Relaxation Complete for {formula}![/bold cyan]")
            console.print(f"Steps taken: {result['steps_taken']}")
            console.print(f"Initial Energy: {result['initial_energy']:.4f} eV")
            console.print(f"Final Energy: {result['final_energy']:.4f} eV")
            delta_e = result['final_energy'] - result['initial_energy']
            console.print(f"Energy Change: {delta_e:.4f} eV")
            
        except Exception as e:
            console.print(f"[bold red]Error during relaxation: {e}[/bold red]")

@app.command()
def evolve(
    formula: str,
    population: int = typer.Option(10, "--population", help="Number of structures in each generation"),
    generations: int = typer.Option(5, "--generations", help="Number of generations to evolve"),
    api_key: str = typer.Option(None, envvar="MP_API_KEY", help="Materials Project API Key")
):
    """
    Genetic Algorithm Discovery: Evolve crystal structures to find more stable alternatives.
    """
    from matgraph.sdk import MatGraphSDK
    sdk = MatGraphSDK(api_key=api_key)
    
    console.print(f"[bold cyan]🧬 Starting Genetic Algorithm Evolution for {formula}[/bold cyan]")
    console.print(f"Population Size: {population} | Generations: {generations}")
    
    with console.status(f"[bold green]Running Generation Evolution (Evaluating M3GNet Energies)..."):
        try:
            history = sdk.evolve(formula, population_size=population, generations=generations)
            
            console.print("\n[bold green]Evolution Complete![/bold green]")
            table = Table(title="Evolution History (Best per Generation)")
            table.add_column("Gen", justify="right", style="cyan")
            table.add_column("Best Formula", style="magenta")
            table.add_column("Formation Energy (eV/atom)", justify="right", style="blue")
            
            for entry in history:
                table.add_row(
                    str(entry["generation"]),
                    entry["best_formula"],
                    f"{entry['best_fitness']:.4f}"
                )
                
            console.print(table)
            
            best_all_time = history[-1]
            console.print(f"\n[bold yellow]🏆 Top Discovered Material:[/bold yellow] [bold magenta]{best_all_time['best_formula']}[/bold magenta]")
            console.print(f"Predicted Formation Energy: {best_all_time['best_fitness']:.4f} eV/atom")
            
        except Exception as e:
            console.print(f"[bold red]Error during evolution: {e}[/bold red]")

@app.command()
def dft(
    formula: str,
    code: str = typer.Option("vasp", "--code", help="DFT code: 'vasp' or 'qe'"),
    output_dir: str = typer.Option("dft_inputs", "--output-dir", help="Output directory for DFT files"),
    api_key: str = typer.Option(None, envvar="MP_API_KEY"),
):
    """
    ML-to-DFT bridge: pre-relax with M3GNet, then write VASP or Quantum Espresso input files.
    """
    sdk = MatGraphSDK(api_key=api_key)
    console.print(f"[cyan]Pre-relaxing {formula} with M3GNet and generating {code.upper()} inputs...[/cyan]")
    try:
        result = sdk.export_dft(formula, code=code, output_dir=output_dir)
        console.print(f"[bold green]DFT inputs written to: {result['directory']}[/bold green]")
        for f in result["files_written"]:
            console.print(f"  - {f}")
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")


@app.command()
def stability(
    formula: str,
    api_key: str = typer.Option(None, envvar="MP_API_KEY"),
):
    """
    Check thermodynamic stability (convex hull distance) for all polymorphs of a material.
    """
    sdk = MatGraphSDK(api_key=api_key)
    with console.status(f"[bold green]Checking convex hull stability for {formula}..."):
        try:
            results = sdk.stability(formula)
            table = Table(title=f"Convex Hull Stability: {formula}")
            table.add_column("Material ID", style="cyan")
            table.add_column("Formula", style="magenta")
            table.add_column("E_above_hull (eV/atom)", justify="right", style="blue")
            table.add_column("Status", justify="center")
            for r in results:
                color = "green" if r["stability_label"] == "Stable" else (
                    "yellow" if r["stability_label"] == "Metastable" else "red"
                )
                table.add_row(
                    r["material_id"], r["formula"],
                    f"{r['energy_above_hull']:.4f}",
                    f"[{color}]{r['stability_label']}[/{color}]"
                )
            console.print(table)
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")


@app.command()
def bands(
    formula: str,
    api_key: str = typer.Option(None, envvar="MP_API_KEY"),
):
    """
    Fetch and display the electronic band structure summary (VBM, CBM, gap, metal/insulator).
    """
    sdk = MatGraphSDK(api_key=api_key)
    with console.status(f"[bold green]Fetching band structure for {formula}..."):
        try:
            result = sdk.band_structure(formula)
            console.print(f"[bold cyan]Band Structure: {formula} ({result['material_id']})[/bold cyan]")
            console.print(f"  Metal:           {result['is_metal']}")
            console.print(f"  Band Gap:        {result['band_gap']:.3f} eV")
            console.print(f"  VBM:             {result['vbm']:.3f} eV")
            console.print(f"  CBM:             {result['cbm']:.3f} eV")
            console.print(f"  Number of Bands: {result['nbands']}")
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")


@app.command()
def elastic(
    formula: str,
    api_key: str = typer.Option(None, envvar="MP_API_KEY"),
):
    """
    Fetch and display elastic constants (Bulk/Shear modulus, Poisson ratio, anisotropy).
    """
    sdk = MatGraphSDK(api_key=api_key)
    with console.status(f"[bold green]Fetching elastic properties for {formula}..."):
        try:
            results = sdk.elastic(formula)
            table = Table(title=f"Elastic Constants: {formula}")
            table.add_column("Material ID", style="cyan")
            table.add_column("K_vrh (GPa)", justify="right", style="blue")
            table.add_column("G_vrh (GPa)", justify="right", style="magenta")
            table.add_column("Poisson Ratio", justify="right", style="yellow")
            table.add_column("Anisotropy", justify="right")
            for r in results:
                table.add_row(
                    r["material_id"],
                    f"{r['bulk_modulus_vrh']:.1f}" if r["bulk_modulus_vrh"] else "N/A",
                    f"{r['shear_modulus_vrh']:.1f}" if r["shear_modulus_vrh"] else "N/A",
                    f"{r['homogeneous_poisson']:.3f}" if r["homogeneous_poisson"] else "N/A",
                    f"{r['universal_anisotropy']:.3f}" if r["universal_anisotropy"] else "N/A",
                )
            console.print(table)
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")


@app.command()
def dielectric(
    formula: str,
    api_key: str = typer.Option(None, envvar="MP_API_KEY"),
):
    """
    Fetch dielectric constants (total, electronic, ionic) and refractive index.
    """
    sdk = MatGraphSDK(api_key=api_key)
    with console.status(f"[bold green]Fetching dielectric properties for {formula}..."):
        try:
            results = sdk.dielectric(formula)
            table = Table(title=f"Dielectric Constants: {formula}")
            table.add_column("Material ID", style="cyan")
            table.add_column("e_total", justify="right")
            table.add_column("e_ionic", justify="right")
            table.add_column("e_electronic", justify="right")
            table.add_column("Refractive Index (n)", justify="right", style="magenta")
            for r in results:
                table.add_row(
                    r["material_id"],
                    f"{r['e_total']:.3f}" if r["e_total"] else "N/A",
                    f"{r['e_ionic']:.3f}" if r["e_ionic"] else "N/A",
                    f"{r['e_electronic']:.3f}" if r["e_electronic"] else "N/A",
                    f"{r['refractive_index']:.3f}" if r["refractive_index"] else "N/A",
                )
            console.print(table)
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")


@app.command()
def magnetic(
    formula: str,
    api_key: str = typer.Option(None, envvar="MP_API_KEY"),
):
    """
    Fetch magnetic properties (ordering, total magnetization) for a material.
    """
    sdk = MatGraphSDK(api_key=api_key)
    with console.status(f"[bold green]Fetching magnetic properties for {formula}..."):
        try:
            results = sdk.magnetic(formula)
            table = Table(title=f"Magnetic Properties: {formula}")
            table.add_column("Material ID", style="cyan")
            table.add_column("Formula", style="magenta")
            table.add_column("Ordering", style="blue")
            table.add_column("Total Magnetization (uB)", justify="right")
            for r in results:
                table.add_row(
                    r["material_id"],
                    r["formula"],
                    r["ordering"],
                    f"{r['total_magnetization']:.3f}" if r["total_magnetization"] is not None else "N/A",
                )
            console.print(table)
        except Exception as e:
            console.print(f"[bold red]Error: {e}[/bold red]")


@app.command()
def serve(port: int = 8000):
    """Start the robust GraphQL API server."""
    api_key = get_api_key()
    if not api_key:
        console.print("[yellow]Warning: MP_API_KEY is not set. GraphQL queries will fail.[/yellow]")

    console.print(f"[green]Starting modern GraphQL server on port {port}...[/green]")
    console.print(f"[cyan]Explore the API at http://localhost:{port}/graphql[/cyan]")
    import uvicorn
    uvicorn.run("matgraph.graphql_app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    app()
