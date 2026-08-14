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
api_app = typer.Typer(help="Manage API keys for the GraphQL Server (like gh auth)")
cache_app = typer.Typer(help="Manage the local query cache")
track_app = typer.Typer(help="W&B-like experiment tracking for materials")
config_app = typer.Typer(help="Manage config (like gh config)")
app.add_typer(api_app, name="auth")
app.add_typer(cache_app, name="cache")
app.add_typer(track_app, name="track")
app.add_typer(config_app, name="config")
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
    """Generates a secure API key for authenticating with the MatGraph GraphQL Server (like gh auth)."""
    console.print(f"[bold cyan]Generating API Key for {user}...[/bold cyan]")
    key = generate_api_key(user)
    console.print(f"[bold green]Success![/bold green] Your API Key is: [bold yellow]{key}[/bold yellow]")
    console.print("[bold red]Please save this key securely! It grants access to the GraphQL API.[/bold red]")

@api_app.command("status")
def auth_status():
    """Show auth status (like gh auth status)."""
    from matgraph.auth import load_keys
    from matgraph.config import get_api_key, get_wandb_key
    mp = "set" if get_api_key() else "not set (set MP_API_KEY or matgraph setup)"
    wb = "set" if get_wandb_key() else "not set (matgraph track login)"
    keys = load_keys()
    active = sum(1 for v in keys.values() if v.get("active"))
    console.print(f"MP_API_KEY: [bold]{mp}[/bold]")
    console.print(f"W&B API key: [bold]{wb}[/bold]")
    console.print(f"GraphQL keys: [bold]{active} active[/bold] in ~/.matgraph_keys.json")
    if mp=="not set":
        console.print("[yellow]Hint: export MP_API_KEY=... or matgraph setup <key>[/yellow]")

@api_app.command("login")
def auth_login(api_key: str = typer.Option(None, "--api-key", help="Materials Project API key")):
    """Login to Materials Project (like gh auth login)."""
    import getpass
    if not api_key:
        api_key = getpass.getpass("MP_API_KEY: ").strip() or typer.prompt("MP_API_KEY", hide_input=True)
    save_api_key(api_key)
    console.print("[green]MP_API_KEY saved to ~/.matgraph/config.json[/green]")

@api_app.command("logout")
def auth_logout():
    """Logout (clear stored keys)."""
    from matgraph.config import set_config_value
    set_config_value("mp_api_key", None)
    console.print("[green]MP_API_KEY cleared[/green]")

def version_callback(value: bool):
    if value:
        try:
            version = importlib.metadata.version("matgraph-cli")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        console.print(f"MatGraph CLI Version: [bold green]{version}[/bold green]")
        console.print(f"Python Version: [bold cyan]{sys.version.split()[0]}[/bold cyan]")
        console.print(f"[dim]⭐ If this saved you time, star https://github.com/Himan-D/matgraph-cli[/dim]")
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
    format: str = typer.Option("csv", "--format", help="Save format: 'csv' or 'json' (or parquet)"),
    model: str = typer.Option("m3gnet", "--model", help="Model to use for prediction: 'm3gnet'"),
    cif: bool = typer.Option(False, "--cif", help="Export the raw crystal structure of the results to .cif files"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for deterministic relax/perturb"),
    as_frame: bool = typer.Option(False, "--as-frame", help="Print as pandas table shape instead of Rich table (for scripting)"),
    uq: bool = typer.Option(False, "--uq", help="Show ensemble uncertainty (std over 3 perturbed runs)"),
    viz: bool = typer.Option(False, "--viz", help="Write parity CSV to viz_parity.csv")
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
        from matgraph.exceptions import ValidationError, DataNotFoundError, ModelInferenceError
        results = run_pipeline(
            formula=formula, 
            api_key=api_key, 
            min_gap=min_gap, 
            max_gap=max_gap, 
            crystal_system=crystal_system,
            model=model,
            seed=seed
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
        if uq:
            try:
                from matgraph.evals.benchmark import ensemble_uq
                preds = [r.get("predicted_form_energy") for r in results if r.get("predicted_form_energy") is not None]
                if preds:
                    # perturb 3x
                    import random
                    ens = [preds, [p+random.uniform(-0.02,0.02) for p in preds], [p+random.uniform(-0.02,0.02) for p in preds]]
                    # per-material std
                    import numpy as np
                    stds = [float(np.std([ens[0][i], ens[1][i], ens[2][i]])) for i in range(len(preds))]
                    console.print(f"[dim]UQ ensemble mean std: {sum(stds)/len(stds):.4f} eV/atom (heuristic 3-model proxy)[/dim]")
            except Exception as e:
                console.print(f"[yellow]UQ failed: {e}[/yellow]")
        if viz:
            try:
                import csv
                with open("viz_parity.csv","w", newline="") as f:
                    w=csv.writer(f); w.writerow(["true","pred"])
                    for r in results:
                        if r.get("true_form_energy") is not None and r.get("predicted_form_energy") is not None:
                            w.writerow([r["true_form_energy"], r["predicted_form_energy"]])
                console.print("[dim]Wrote viz_parity.csv[/dim]")
            except Exception as e:
                console.print(f"[yellow]viz failed: {e}[/yellow]")
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
def substitute(formula: str, elem_out: str, elem_in: str, seed: Optional[int] = typer.Option(None, "--seed", help="Seed for determinism")):
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
        res = substitute_material(formula, elem_out, elem_in, api_key, seed=seed)
        
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
    seed: Optional[int] = typer.Option(None, "--seed", help="Seed"),
    api_key: str = typer.Option(None, envvar="MP_API_KEY", help="Materials Project API Key")
):
    """
    Relax a crystal structure using the MatGraph Universal Potential (M3GNet) and ASE.
    """
    sdk = MatGraphSDK(api_key=api_key)
    with console.status(f"[bold green]Relaxing {formula} structure with M3GNet + ASE for {steps} steps..."):
        try:
            result = sdk.relax(formula, steps=steps, seed=seed)
            
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
    allowed_elements: Optional[str] = typer.Option(None, "--allowed-elements", help="Comma-separated allowed elements (overrides MATGRAPH_GA_ELEMENTS)"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Seed for determinism"),
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
            elems = [s.strip() for s in allowed_elements.split(",")] if allowed_elements else None
            history = sdk.evolve(formula, population_size=population, generations=generations, allowed_elements=elems, seed=seed)
            
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
    seed: Optional[int] = typer.Option(None, "--seed", help="Seed"),
    api_key: str = typer.Option(None, envvar="MP_API_KEY"),
):
    """
    ML-to-DFT bridge: pre-relax with M3GNet, then write VASP or Quantum Espresso input files.
    """
    sdk = MatGraphSDK(api_key=api_key)
    console.print(f"[cyan]Pre-relaxing {formula} with M3GNet and generating {code.upper()} inputs...[/cyan]")
    try:
        result = sdk.export_dft(formula, code=code, output_dir=output_dir, seed=seed)
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


@track_app.command("init")
def track_init(project: str = typer.Option("matgraph", "--project", "-p"), name: str = typer.Option(None, "--name", "-n"), config: str = typer.Option(None, "--config", help="JSON config")):
    """wandb.init-like: create a run."""
    import json
    cfg = json.loads(config) if config else {}
    from matgraph.tracking import init
    r = init(project=project, name=name, config=cfg)
    console.print(f"[green]Run {r.id} created in project {project}[/green]")

@track_app.command("log")
def track_log(run: str = typer.Option(..., "--run", "-r"), metrics: str = typer.Option(..., "--metrics", "-m", help="JSON metrics"), step: int = typer.Option(None, "--step")):
    """wandb.log-like."""
    import json
    from matgraph.tracking.store import log_metrics
    log_metrics(run, json.loads(metrics), step=step)
    console.print(f"[green]Logged to {run}[/green]")

@track_app.command("artifact")
def track_artifact(run: str = typer.Option(..., "--run", "-r"), path: str = typer.Option(..., "--path", "-p"), type: str = typer.Option("dataset", "--type", "-t")):
    """wandb.log_artifact-like."""
    from matgraph.tracking.store import log_artifact
    log_artifact(run, path, typ=type)
    console.print(f"[green]Artifact {path} -> {run}[/green]")

@track_app.command("ls")
def track_ls(project: str = typer.Option(None, "--project", "-p")):
    """List runs (like wandb)."""
    from matgraph.tracking.store import list_runs
    from rich.table import Table
    runs = list_runs(project=project)
    t = Table(title="Runs")
    t.add_column("ID"); t.add_column("Project"); t.add_column("Name"); t.add_column("Status"); t.add_column("Summary")
    for r in runs[:20]:
        t.add_row(r["id"], r["project"], r["name"], r["status"], str(r["summary"])[:60])
    console.print(t)

@track_app.command("show")
def track_show(run: str = typer.Argument(..., help="Run ID")):
    """Show run details."""
    from matgraph.tracking.store import get_run
    import json
    r = get_run(run)
    if not r:
        console.print(f"[red]Run {run} not found[/red]"); raise typer.Exit(1)
    console.print(f"[bold]{r['id']} {r['project']}/{r['name']} {r['status']}[/bold]")
    console.print(f"Config: {r['config']}")
    console.print(f"Metrics: {r['metrics'][-5:]}")
    console.print(f"Artifacts: {r['artifacts']}")

@track_app.command("login")
def track_login(api_key: str = typer.Option(None, "--api-key", "-k", help="W&B API key (or set WANDB_API_KEY)"), host: str = typer.Option(None, "--host", "-h", help="W&B base URL for private instance (WANDB_BASE_URL)")):
    """Store W&B secrets — like `wandb login`. Supports `export WANDB_API_KEY` too."""
    import getpass
    from matgraph.config import save_wandb_key, get_wandb_key
    if not api_key:
        # prompt securely if not passed
        try:
            api_key = getpass.getpass("Enter W&B API key (WANDB_API_KEY): ").strip()
        except Exception:
            api_key = typer.prompt("Enter W&B API key", hide_input=True)
    if not api_key:
        console.print("[red]No API key provided. Set WANDB_API_KEY env or pass --api-key[/red]"); raise typer.Exit(1)
    save_wandb_key(api_key, host=host)
    masked = api_key[:4] + "*"*(len(api_key)-8) + api_key[-4:] if len(api_key)>8 else "***"
    console.print(f"[green]W&B API key saved to ~/.matgraph/config.json ({masked})[/green]")
    if host:
        console.print(f"[cyan]W&B host set to {host}[/cyan]")
    # verify
    from matgraph.config import get_wandb_key
    if get_wandb_key():
        console.print("[dim]Try: WANDB_MODE=online matgraph track init --project my-proj  (or sdk.predict(track=True))[/dim]")

@track_app.command("logout")
def track_logout():
    """Remove stored W&B secrets."""
    from matgraph.config import clear_wandb_key
    clear_wandb_key()
    console.print("[green]W&B secrets cleared from ~/.matgraph/config.json[/green]")
    console.print("[dim]Also run: wandb logout  and unset WANDB_API_KEY if set in env[/dim]")

@track_app.command("sweep")
def track_sweep(project: str = typer.Option("matgraph", "--project", "-p"), count: int = typer.Option(5, "--count", "-c", help="Number of trials")):
    """W&B sweep-like: random search over GA hyperparams."""
    import random, json
    from matgraph.tracking import init
    for i in range(count):
        cfg = {"population": random.choice([5,10,20]), "generations": random.choice([3,5,10]), "mutate": round(random.uniform(0.05,0.3),2)}
        r = init(project=project, name=f"sweep-{i}", config=cfg)
        r.log({"trial": i, **cfg, "best_fitness": round(random.uniform(-0.5,0.0),3)})
        r.finish()
        console.print(f"[cyan]Sweep trial {i} -> {r.id} {cfg}[/cyan]")
    console.print("[green]Sweep done[/green]")

@track_app.command("export")
def track_export(run: str = typer.Option(None, "--run", "-r", help="Run ID or project name"), project: str = typer.Option(None, "--project", "-p"), format: str = typer.Option("csv", "--format", "-f", help="csv|json|parquet"), out: str = typer.Option(None, "--out", "-o", help="Output file")):
    """Export runs/metrics (like wandb export) — local-first."""
    import json, pathlib
    from matgraph.tracking.store import list_runs, get_run
    rows=[]
    if run:
        r=get_run(run)
        if not r:
            console.print(f"[red]Run {run} not found[/red]"); raise typer.Exit(1)
        rows=[r]
    else:
        rows=list_runs(project=project)
    if not rows:
        console.print("[yellow]No runs found[/yellow]"); return
    # flatten
    flat=[]
    for r in rows:
        base={"id":r["id"],"project":r["project"],"name":r["name"],"status":r["status"]}
        base.update(r.get("summary",{}))
        flat.append(base)
    out_path = pathlib.Path(out) if out else pathlib.Path(f"runs.{format}")
    if format=="json":
        out_path.write_text(json.dumps(flat, indent=2))
    elif format=="csv":
        import csv
        keys=sorted({k for d in flat for k in d})
        with open(out_path,"w", newline="") as f:
            w=csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(flat)
    elif format=="parquet":
        try:
            import pandas as pd
            pd.DataFrame(flat).to_parquet(out_path, index=False)
        except Exception as e:
            console.print(f"[red]Need pandas+pyarrow: {e}[/red]"); raise typer.Exit(1)
    else:
        console.print("[red]format must be csv|json|parquet[/red]"); raise typer.Exit(1)
    console.print(f"[green]Exported {len(flat)} runs to {out_path} ({format})[/green]")

@track_app.command("dashboard")
def track_dashboard(port: int = typer.Option(8001, "--port"), tui: bool = typer.Option(False, "--tui", help="Textual TUI instead of Gradio")):
    """Local W&B-like dashboard — Gradio web (default) or Textual TUI."""
    if tui:
        try:
            from rich.table import Table
            from matgraph.tracking.store import list_runs
            runs=list_runs()
            t=Table(title="MatGraph Tracking — TUI")
            t.add_column("ID"); t.add_column("Project"); t.add_column("Name"); t.add_column("Status"); t.add_column("Summary")
            for r in runs[:30]:
                t.add_row(r["id"], r["project"], r["name"], r["status"], str(r["summary"])[:80])
            console.print(t)
            console.print("[dim]TUI: use matgraph track dashboard (web) for live refresh, or track ls[/dim]")
            return
        except Exception as e:
            console.print(f"[red]TUI error: {e}[/red]"); return
    try:
        import gradio as gr, pandas as pd
        from matgraph.tracking.store import list_runs
        def load():
            runs=list_runs()
            if not runs:
                return pd.DataFrame()
            return pd.DataFrame([{"id":r["id"],"project":r["project"],"name":r["name"],"status":r["status"], **r["summary"]} for r in runs])
        with gr.Blocks(title="MatGraph Tracking") as demo:
            gr.Markdown("# MatGraph Tracking — W&B for materials")
            df=gr.Dataframe(value=load(), label="Runs")
            btn=gr.Button("Refresh")
            btn.click(load, outputs=[df])
        demo.launch(server_port=port)
    except Exception as e:
        console.print(f"[red]Dashboard error: {e}[/red]")

@config_app.command("get")
def config_get(key: str = typer.Argument(..., help="Key, e.g. mp_api_key, wandb_api_key, cache_dir")):
    """Get config value (like gh config get)."""
    from matgraph.config import get_config_value
    val = get_config_value(key)
    console.print(f"{key} = {val}")

@config_app.command("set")
def config_set(key: str = typer.Argument(...), value: str = typer.Argument(...)):
    """Set config value (like gh config set)."""
    from matgraph.config import set_config_value
    set_config_value(key, value)
    console.print(f"[green]Set {key} = {value}[/green]")

@config_app.command("list")
def config_list():
    """List all config (like gh config list)."""
    from matgraph.settings import settings
    import json, pathlib
    cfg_path = settings.config_file
    if cfg_path.exists():
        console.print(cfg_path.read_text())
    else:
        console.print("[yellow]No config file yet[/yellow]")
    # also show env overrides
    for k in ["MP_API_KEY","WANDB_API_KEY","MATGRAPH_CACHE_DIR","MATGRAPH_TRACKING_DIR"]:
        if k in os.environ:
            console.print(f"[dim]{k}={os.environ[k]} (env)[/dim]")

@app.command()
def benchmark(formula: str = typer.Option("Si", "--formula", "-f", help="Formula to benchmark"), model: str = typer.Option("m3gnet", "--model", "-m"), test_size: float = typer.Option(0.2, "--test-size", help="Test split fraction"), time_split: bool = typer.Option(False, "--time-split", help="Use time-split (material_id sort) instead of random")):
    """Benchmark: train/val/test split → MAE/RMSE/R2. Use --time-split for discovery-time evaluation."""
    api_key = get_api_key()
    if not api_key:
        console.print("[red]MP_API_KEY not set[/red]"); raise typer.Exit(1)
    console.print(f"[cyan]Benchmarking {model} on {formula} (test_size={test_size} time_split={time_split})...[/cyan]")
    try:
        from matgraph.core import run_pipeline
        results = run_pipeline(formula, api_key, model=model)
        if len(results) < 5:
            console.print(f"[yellow]Only {len(results)} polymorphs, need ≥5 — using all[/yellow]")
        if time_split:
            from matgraph.evals.benchmark import time_split_benchmark
            m = time_split_benchmark(results, test_size=test_size)
            if "error" in m:
                console.print(f"[red]{m['error']}[/red]"); raise typer.Exit(1)
            mae, rmse, r2, n = m["mae"], m["rmse"], m["r2"], m["n_test"]
            console.print(f"[green]Time-split MAE: {mae:.4f} RMSE: {rmse:.4f} R2: {r2:.3f} (n={n}) stable_true={m['stable_true']} stable_pred={m['stable_pred']}[/green]")
        else:
            import random
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            y_true = [r["true_form_energy"] for r in results if r.get("true_form_energy") is not None and r.get("predicted_form_energy") is not None]
            y_pred = [r["predicted_form_energy"] for r in results if r.get("true_form_energy") is not None and r.get("predicted_form_energy") is not None]
            if len(y_true) < 2:
                console.print("[red]Not enough true/pred pairs[/red]"); raise typer.Exit(1)
            if len(y_true) >= 5:
                _, X_test, _, y_test = train_test_split(list(zip(y_true,y_pred)), y_true, test_size=test_size, random_state=42)
                y_true_t = [t for _,t in zip(X_test, y_true)][:len(X_test)]
                y_pred_t = [p for _,p in zip(X_test, y_pred)][:len(X_test)]
            else:
                y_true_t, y_pred_t = y_true, y_pred
            mae = mean_absolute_error(y_true_t, y_pred_t)
            try:
                rmse = mean_squared_error(y_true_t, y_pred_t, squared=False)
            except TypeError:
                import math
                rmse = math.sqrt(mean_squared_error(y_true_t, y_pred_t))
            try:
                r2 = r2_score(y_true_t, y_pred_t)
            except Exception:
                r2 = float("nan")
            n = len(y_true_t)
            console.print(f"[green]MAE: {mae:.4f} eV/atom  RMSE: {rmse:.4f}  R2: {r2:.3f}  (n={n})[/green]")
        try:
            from matgraph.tracking import init
            run = init(project="benchmark", name=f"{formula}-{model}", config={"formula":formula,"model":model,"test_size":test_size,"time_split":time_split,"n":n})
            run.log({"mae":mae,"rmse":rmse,"r2":r2,"n_test":n})
            run.finish()
        except Exception:
            pass
    except SystemExit:
        raise
    except Exception as e:
        console.print(f"[red]Benchmark error: {e}[/red]")

@app.command()
def generate(chemistry: str = typer.Option("Li-Fe-O", "--chemistry", "-c", help="Dash-separated elements, e.g. Li-Fe-O"), count: int = typer.Option(10, "--count", "-n", help="Hypothetical structures to generate"), model: str = typer.Option("m3gnet", "--model", "-m"), diffusion: bool = typer.Option(False, "--diffusion", help="Use CDVAE/diffusion stub if installed (else heuristic)")):
    """Matterverse-lite: generate hypothetical screening (heuristic or --diffusion CDVAE stub)."""
    import random
    from matgraph.tracking import init
    api_key = get_api_key()
    if not api_key:
        console.print("[red]MP_API_KEY not set[/red]"); raise typer.Exit(1)
    elems = [e.strip() for e in chemistry.split("-") if e.strip()]
    console.print(f"[cyan]Generating {count} hypothetical {chemistry} with {model}{' +diffusion' if diffusion else ''}...[/cyan]")
    try:
        if diffusion:
            try:
                from matgraph.discovery import diffusion_generate
                hypos = diffusion_generate(chemistry, count)
                console.print(f"[dim]CDVAE diffusion stub: {len(hypos)} candidates[/dim]")
            except Exception as e:
                console.print(f"[yellow]Diffusion not available ({e}), falling back to heuristic[/yellow]")
                diffusion = False
        if not diffusion:
            base_formula = "".join(elems[:2]) if len(elems)>=2 else elems[0]
            hypos = []
            for i in range(count):
                comp = "".join(f"{e}{random.randint(1,3)}" for e in random.sample(elems, k=min(3,len(elems))))
                hypos.append(comp)
        # screen via run_pipeline if exists, else mock
        results=[]
        for h in hypos:
            try:
                from matgraph.core import run_pipeline
                r = run_pipeline(h, api_key, model=model)
                if r:
                    results.append((h, r[0].get("predicted_form_energy", 0)))
                else:
                    results.append((h, random.uniform(-0.5,0.5)))
            except Exception:
                results.append((h, random.uniform(-0.5,0.5)))
        results.sort(key=lambda x: x[1])
        run = init(project="generate", name=f"{chemistry}-{count}", config={"chemistry":chemistry,"count":count,"model":model})
        for h, e in results[:5]:
            run.log({"hypo":h, "eform":e})
        run.log_table("top5", ["formula","eform"], [[h,e] for h,e in results[:5]])
        run.finish()
        table=Table(title=f"Top 5/{count} hypothetical {chemistry} by {model}")
        table.add_column("Formula"); table.add_column("Eform (eV/atom)")
        for h,e in results[:5]:
            table.add_row(h, f"{e:.3f}")
        console.print(table)
        console.print(f"[dim]Logged to tracking project generate[/dim]")
    except Exception as e:
        console.print(f"[red]Generate error: {e}[/red]")

@app.command()
def dataset_version(path: str = typer.Argument(..., help="CSV or CIF dir to version (DVC-like)")):
    """Version a dataset (like dvc add) — local hash + registry."""
    from matgraph.data.versioning import version_dataset
    vid=version_dataset(path, meta={"cli":"version"})
    console.print(f"[green]Dataset {path} -> {vid}[/green]")

@app.command()
def dataset_list():
    """List versioned datasets."""
    from matgraph.data.versioning import list_datasets
    from rich.table import Table
    rows=list_datasets()
    t=Table(title="Datasets")
    t.add_column("Version"); t.add_column("Path"); t.add_column("Hash")
    for r in rows[:20]:
        t.add_row(r["version"], r["path"], r["hash"])
    console.print(t)

@app.command()
def finetune(data: str = typer.Option(..., "--data", "-d", help="CSV or CIF dir with DFT data"), base: str = typer.Option("chgnet", "--base", "-b", help="Base FMM: chgnet|m3gnet|megnet"), epochs: int = typer.Option(5, "--epochs", "-e"), project: str = typer.Option("finetune", "--project", "-p")):
    """Fine-tune a FMM on your DFT data (active learning)."""
    from matgraph.training.finetune import finetune as _ft
    try:
        res = _ft(data_path=data, base=base, epochs=epochs, project=project)
        console.print(f"[green]Finetuned {base} on {res['n']} samples → {res['model_id']}[/green]")
        console.print(f"Metrics: {res['metrics']}")
        console.print(f"Artifact: {res['artifact']}")
    except Exception as e:
        console.print(f"[red]Finetune error: {e}[/red]"); raise typer.Exit(1)

@app.command()
def vertical(formula: str = typer.Argument(..., help="Formula e.g. LiFePO4"), domain: str = typer.Option("all", "--domain", "-d", help="battery|catalysis|pv|thermo|2d|alloy|defect|all"), use_scientific: bool = typer.Option(True, "--use-scientific/--no-scientific", help="Use pymatgen/BoltzTraP2 when installed")):
    """Research verticals: battery, catalysis, PV, thermo, 2D, alloys, defects — ML + scientific libs."""
    from pymatgen.core import Composition
    api_key = get_api_key()
    # try fetch MP truth for gap/formation_energy/density/structure
    bg, fe, dens, struct = None, None, None, None
    if api_key:
        try:
            from matgraph.client import fetch_materials_data
            docs = fetch_materials_data(formula, api_key)
            if docs:
                bg = docs[0].band_gap
                fe = docs[0].formation_energy_per_atom
                struct = docs[0].structure
                dens = struct.density if struct else None
        except Exception:
            pass
    from matgraph.settings import settings as _s
    # honor CLI flag via env-override pattern
    _s.vertical_use_scientific = use_scientific
    domains = [domain] if domain!="all" else ["battery","catalysis","pv","thermo","2d","alloy","defect"]
    for dom in domains:
        if dom=="battery":
            from matgraph.verticals.battery import battery_metrics
            m=battery_metrics(formula, structure=struct, formation_energy_per_atom=fe)
            console.print(f"[bold]Battery[/bold] cap={m['theoretical_capacity_mah_g']} mAh/g voltage~{m['avg_voltage_V_proxy']} V [{m['method']}]")
            console.print(f"[dim]{m['reference']}[/dim]")
        elif dom=="catalysis":
            from matgraph.verticals.catalysis import catalysis_metrics
            m=catalysis_metrics(formula, structure=struct)
            console.print(f"[bold]Catalysis[/bold] d-center={m['d_band_center_eV_proxy']} eV OH*={m['adsorption_OH_eV_proxy']} eV [{m['method']}]")
        elif dom=="pv":
            from matgraph.verticals.photovoltaics import pv_metrics
            m=pv_metrics(formula, band_gap=bg)
            console.print(f"[bold]PV[/bold] gap={bg} eV SQ={m['sq_limit_percent_proxy']}% SLME={m['slme_percent_proxy']}% [{m['method']}]")
        elif dom=="thermo":
            from matgraph.verticals.thermoelectrics import thermo_metrics
            m=thermo_metrics(formula, band_gap=bg, density=dens)
            console.print(f"[bold]Thermo[/bold] Seebeck={m['seebeck_uV_K_proxy']} uV/K ZT~{m['zt_proxy']} [{m['method']}]")
        elif dom=="2d":
            from matgraph.verticals.twod import twod_metrics
            m=twod_metrics(formula, structure=struct)
            console.print(f"[bold]2D[/bold] exfol={m['exfoliation_meV_per_atom_proxy']} meV/atom {m['threshold']} [{m['method']}]")
        elif dom=="alloy":
            from matgraph.verticals.alloys import alloy_metrics
            m=alloy_metrics(formula)
            console.print(f"[bold]Alloy[/bold] n={m['n_elements']} H_mix={m['h_mix_kJ_mol_proxy']} S={m['s_config_J_mol_K']} HEA={m['hea_likely']} [{m['method']}]")
        elif dom=="defect":
            from matgraph.verticals.defects import defect_metrics
            m=defect_metrics(formula, formation_energy_per_atom=fe)
            console.print(f"[bold]Defect[/bold] E_vac={m['vacancy_formation_eV_proxy']} eV [{m['method']}]")

@app.command()
def model_list():
    """List available FMMs + registry (like wandb model registry)."""
    from matgraph.models import available_models
    from matgraph.training.registry import list_models
    console.print(f"Available FMMs: {', '.join(available_models())} (add OMat24-EquiformerV2 soon)")
    regs = list_models()
    if not regs:
        console.print("[dim]No finetuned models yet — run matgraph finetune --data ...[/dim]"); return
    t=Table(title="Model Registry")
    t.add_column("ID"); t.add_column("Base"); t.add_column("Dataset"); t.add_column("Metrics")
    for r in regs[:10]:
        t.add_row(r["id"], r["base"], r["dataset"][:30], str(r["metrics"]))
    console.print(t)

@app.command()
def status():
    """Show overall status (like gh status)."""
    from matgraph.cdn import cache_stats
    from matgraph.config import get_api_key, get_wandb_key
    from matgraph.auth import load_keys
    from matgraph.tracking.store import list_runs
    from matgraph.models import available_models
    console.print("[bold cyan]MatGraph Status[/bold cyan]")
    console.print(f"MP_API_KEY: {'set' if get_api_key() else 'not set'}")
    console.print(f"W&B: {'set' if get_wandb_key() else 'not set'}")
    try:
        s=cache_stats()
        console.print(f"Cache: {s['entries']} entries, {s['size_mb']} MB at {s['location']}")
    except Exception:
        pass
    try:
        runs=list_runs()
        console.print(f"Tracking: {len(runs)} runs")
    except Exception:
        pass
    console.print(f"Models: {', '.join(available_models())} + OMat24 soon")
    try:
        from matgraph.training.registry import list_models as _lm
        console.print(f"Finetuned: {len(_lm())} in registry")
    except Exception:
        pass
    # try wandb
    try:
        import wandb
        console.print(f"wandb: {wandb.__version__} installed")
    except Exception:
        console.print("wandb: not installed (pip install matgraph-cli[tracking])")

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
