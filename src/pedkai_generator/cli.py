"""
CLI entrypoint for the Pedkai Synthetic Data Generator.

Orchestrates all pipeline steps with progress display and configurable options.
Run individual steps or the full pipeline end-to-end.

Usage:
    pedkai-generate run --all                   # Full pipeline
    pedkai-generate run --step 1                # Single step
    pedkai-generate run --step 1 --step 2       # Multiple steps
    pedkai-generate config --show               # Show current config
    pedkai-generate config --save config.yaml   # Save config to file
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from pedkai_generator.config.settings import GeneratorConfig

console = Console()

# ---------------------------------------------------------------------------
# Step registry — maps step number to (module_path, function_name, description)
# ---------------------------------------------------------------------------

STEP_REGISTRY: dict[int, tuple[str, str, str]] = {
    0: (
        "pedkai_generator.step_00_schema.contracts",
        "generate_schema_contracts",
        "Schema Contracts — define Parquet column contracts for all output files",
    ),
    1: (
        "pedkai_generator.step_01_sites.generate",
        "generate_sites_and_cells",
        "Site & Cell Schema — 21,100 sites, 64,700 logical cell-layers",
    ),
    2: (
        "pedkai_generator.step_02_topology.generate",
        "generate_topology",
        "Full Topology — ~1.49M entities, ~2.21M relationships across 6 domains",
    ),
    3: (
        "pedkai_generator.step_03_radio_kpis.generate",
        "generate_radio_kpis",
        "Radio Physics + Cell KPIs — SINR→CQI→MCS→throughput for 64,700 cells × 720h",
    ),
    4: (
        "pedkai_generator.step_04_domain_kpis.generate",
        "generate_domain_kpis",
        "Multi-Domain KPIs — transport, fixed broadband, enterprise, core, power",
    ),
    5: (
        "pedkai_generator.step_05_scenarios.generate",
        "inject_scenarios",
        "Scenario Injection — sleeping cell, congestion, fibre cut, power failure, etc.",
    ),
    6: (
        "pedkai_generator.step_06_events.generate",
        "generate_events",
        "Events & Alarms — multi-domain alarms aligned with scenario injections",
    ),
    7: (
        "pedkai_generator.step_07_customers.generate",
        "generate_customers",
        "Customer & BSS — 1M subscribers with billing, service plans, site associations",
    ),
    8: (
        "pedkai_generator.step_08_cmdb_degradation.generate",
        "degrade_cmdb",
        "CMDB Degradation — 6 Dark Graph divergence types applied to ground truth",
    ),
    9: (
        "pedkai_generator.step_09_vendor_naming.generate",
        "apply_vendor_naming",
        "Vendor Naming — map internal KPIs to Ericsson/Nokia PM counter names",
    ),
    10: (
        "pedkai_generator.step_10_validation.validate",
        "validate_all",
        "Validation — schema compliance, FK integrity, range checks, cross-domain consistency",
    ),
    11: (
        "pedkai_generator.step_11_loader.loader",
        "load_into_pedkai",
        "Pedkai Loader — ingest all Parquet files into Pedkai's database",
    ),
}

# Steps that can run in parallel (no dependency between them)
PARALLEL_GROUPS: list[list[int]] = [
    [0],  # Phase 0: scaffolding, must be first
    [1],  # Phase 1: sites + cells, foundation for everything
    [2],  # Phase 2: topology, needs sites
    [3, 7],  # Phase 3 (radio KPIs) and Phase 7 (customers) can run in parallel — both need topology
    [4],  # Phase 4: domain KPIs, needs topology
    [5],  # Phase 5: scenario injection, needs all KPIs
    [6],  # Phase 6: events, needs scenarios
    [8],  # Phase 8: CMDB degradation, needs ground truth topology
    [9],  # Phase 9: vendor naming, needs KPIs
    [10],  # Phase 10: validation, needs everything
    [11],  # Phase 11: loader, needs everything
]

# Dependency map: step -> list of steps it depends on
STEP_DEPENDENCIES: dict[int, list[int]] = {
    0: [],
    1: [0],
    2: [1],
    3: [1],
    4: [2],
    5: [2, 3, 4],
    6: [5],
    7: [2],
    8: [2],
    9: [3, 4],
    10: [1, 2, 3, 4, 5, 6, 7, 8, 9],
    11: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}


def _import_step_function(step_num: int):
    """Dynamically import and return the step's generator function."""
    module_path, func_name, _ = STEP_REGISTRY[step_num]
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def _resolve_execution_order(requested_steps: list[int]) -> list[int]:
    """
    Given a set of requested steps, resolve the full execution order
    including all dependencies, in topological order.
    """
    needed: set[int] = set()

    def _add_with_deps(step: int) -> None:
        if step in needed:
            return
        for dep in STEP_DEPENDENCIES.get(step, []):
            _add_with_deps(dep)
        needed.add(step)

    for s in requested_steps:
        _add_with_deps(s)

    # Return in numeric order (which is topological by design)
    return sorted(needed)


def _display_plan(steps: list[int]) -> None:
    """Display the execution plan as a rich table."""
    table = Table(title="Execution Plan", show_header=True, header_style="bold cyan")
    table.add_column("Step", style="bold", width=6, justify="center")
    table.add_column("Phase", width=55)
    table.add_column("Dependencies", width=20)

    for step_num in steps:
        _, _, desc = STEP_REGISTRY[step_num]
        deps = STEP_DEPENDENCIES.get(step_num, [])
        deps_str = ", ".join(str(d) for d in deps) if deps else "—"
        table.add_row(str(step_num), desc, deps_str)

    console.print()
    console.print(table)
    console.print()


def _run_step(step_num: int, config: GeneratorConfig) -> float:
    """
    Run a single pipeline step. Returns elapsed time in seconds.
    """
    _, _, desc = STEP_REGISTRY[step_num]
    console.print(f"\n[bold blue]━━━ Step {step_num}: {desc} ━━━[/bold blue]")

    start = time.time()
    try:
        func = _import_step_function(step_num)
        func(config)
        elapsed = time.time() - start
        console.print(f"[bold green]✓ Step {step_num} completed[/bold green] in {elapsed:.1f}s")
        return elapsed
    except Exception as e:
        elapsed = time.time() - start
        console.print(f"[bold red]✗ Step {step_num} FAILED[/bold red] after {elapsed:.1f}s: {e}")
        raise


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version="0.1.0", prog_name="pedkai-generate")
def main():
    """
    Pedkai Synthetic Data Generator.

    Generates a UK-scale converged operator dataset with radio-layer physics,
    multi-domain KPIs, scenario injection, and CMDB degradation for Dark Graph training.

    Output: ~6.6 GB across 14 Parquet files → /Volumes/Projects/Pedkai Data Store/
    """
    pass


@main.command()
@click.option("--all", "run_all", is_flag=True, help="Run the full pipeline (all 12 steps).")
@click.option(
    "--step",
    "steps",
    type=int,
    multiple=True,
    help="Run specific step(s) by number (0-11). Can be specified multiple times.",
)
@click.option(
    "--from-step",
    "from_step",
    type=int,
    default=None,
    help="Run from this step onwards (inclusive), including all dependencies.",
)
@click.option(
    "--config-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to YAML config file. Overrides defaults.",
)
@click.option("--seed", type=int, default=None, help="Override global seed.")
@click.option("--tenant-id", type=str, default=None, help="Override tenant ID.")
@click.option(
    "--data-store",
    type=click.Path(path_type=Path),
    default=None,
    help="Override data store root path.",
)
@click.option("--dry-run", is_flag=True, help="Show execution plan without running anything.")
@click.option(
    "--days",
    type=int,
    default=None,
    help="Override simulation days (default: 30). Use a smaller value for testing.",
)
def run(
    run_all: bool,
    steps: tuple[int, ...],
    from_step: int | None,
    config_file: Path | None,
    seed: int | None,
    tenant_id: str | None,
    data_store: Path | None,
    dry_run: bool,
    days: int | None,
):
    """Run the synthetic data generation pipeline."""

    # ── Build configuration ──────────────────────────────────
    if config_file:
        config = GeneratorConfig.from_yaml(config_file)
        console.print(f"[dim]Loaded config from {config_file}[/dim]")
    else:
        config = GeneratorConfig()

    # Apply CLI overrides
    if seed is not None:
        config.global_seed = seed
    if tenant_id is not None:
        config.tenant_id = tenant_id
    if data_store is not None:
        config.paths.data_store_root = data_store
    if days is not None:
        config.simulation.simulation_days = days

    # ── Determine which steps to run ─────────────────────────
    if run_all:
        requested = list(range(12))
    elif from_step is not None:
        requested = list(range(from_step, 12))
    elif steps:
        requested = list(steps)
    else:
        console.print("[bold red]Error:[/bold red] Specify --all, --step N, or --from-step N.")
        raise SystemExit(1)

    # Validate step numbers
    for s in requested:
        if s not in STEP_REGISTRY:
            console.print(f"[bold red]Error:[/bold red] Unknown step {s}. Valid steps: 0-11.")
            raise SystemExit(1)

    execution_order = _resolve_execution_order(requested)

    # ── Display header ───────────────────────────────────────
    header_text = (
        f"[bold]Global Seed:[/bold] {config.global_seed}\n"
        f"[bold]Tenant ID:[/bold] {config.tenant_id}\n"
        f"[bold]Sites:[/bold] {config.sites.total:,}\n"
        f"[bold]Logical Cell-Layers:[/bold] {config.rat_split.total_logical_cell_layers:,}\n"
        f"[bold]Simulation:[/bold] {config.simulation.simulation_days}d × {config.simulation.reporting_interval_hours}h intervals = {config.simulation.total_intervals:,} steps\n"
        f"[bold]Target Entities:[/bold] ~{config.entities.total_entities:,}\n"
        f"[bold]Target Relationships:[/bold] ~{config.entities.total_relationships:,}\n"
        f"[bold]Data Store:[/bold] {config.paths.data_store_root}\n"
        f"[bold]Steps:[/bold] {len(execution_order)} ({', '.join(str(s) for s in execution_order)})"
    )
    console.print(Panel(header_text, title="Pedkai Synthetic Data Generator", border_style="cyan"))

    _display_plan(execution_order)

    if dry_run:
        console.print("[yellow]Dry run — no steps executed.[/yellow]")
        return

    # ── Ensure output directories exist ──────────────────────
    config.ensure_output_dirs()

    # ── Save config snapshot for reproducibility ─────────────
    config_snapshot_path = config.paths.output_dir / "generator_config.yaml"
    config.save_yaml(config_snapshot_path)
    console.print(f"[dim]Config snapshot saved to {config_snapshot_path}[/dim]")

    # ── Execute steps ────────────────────────────────────────
    timings: dict[int, float] = {}
    total_start = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        overall_task = progress.add_task("Overall pipeline", total=len(execution_order))

        for step_num in execution_order:
            _, _, desc = STEP_REGISTRY[step_num]
            progress.update(overall_task, description=f"Step {step_num}: {desc[:50]}...")

            try:
                elapsed = _run_step(step_num, config)
                timings[step_num] = elapsed
            except Exception:
                console.print(
                    f"\n[bold red]Pipeline halted at step {step_num}.[/bold red] "
                    f"Fix the error and re-run with --from-step {step_num}"
                )
                raise SystemExit(1)

            progress.advance(overall_task)

    total_elapsed = time.time() - total_start

    # ── Summary ──────────────────────────────────────────────
    summary_table = Table(title="Execution Summary", show_header=True, header_style="bold green")
    summary_table.add_column("Step", width=6, justify="center")
    summary_table.add_column("Phase", width=55)
    summary_table.add_column("Time", width=12, justify="right")

    for step_num in execution_order:
        _, _, desc = STEP_REGISTRY[step_num]
        t = timings.get(step_num, 0.0)
        if t < 60:
            time_str = f"{t:.1f}s"
        elif t < 3600:
            time_str = f"{t / 60:.1f}m"
        else:
            time_str = f"{t / 3600:.1f}h"
        summary_table.add_row(str(step_num), desc, time_str)

    summary_table.add_section()
    if total_elapsed < 60:
        total_str = f"{total_elapsed:.1f}s"
    elif total_elapsed < 3600:
        total_str = f"{total_elapsed / 60:.1f}m"
    else:
        total_str = f"{total_elapsed / 3600:.1f}h"
    summary_table.add_row("", "[bold]Total[/bold]", f"[bold]{total_str}[/bold]")

    console.print()
    console.print(summary_table)
    console.print()
    console.print(f"[bold green]✓ Pipeline complete.[/bold green] Output at: {config.paths.data_store_root}")


@main.command()
@click.option("--show", is_flag=True, help="Display current configuration.")
@click.option(
    "--save",
    "save_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Save default configuration to a YAML file.",
)
@click.option(
    "--load",
    "load_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Load and display a configuration file.",
)
def config(show: bool, save_path: Path | None, load_path: Path | None):
    """View, save, or load generator configuration."""

    if load_path:
        cfg = GeneratorConfig.from_yaml(load_path)
        console.print(f"[dim]Loaded from {load_path}[/dim]")
    else:
        cfg = GeneratorConfig()

    if show or load_path:
        console.print()
        console.print_json(data=cfg.to_dict())
        console.print()

    if save_path:
        cfg.save_yaml(save_path)
        console.print(f"[green]Configuration saved to {save_path}[/green]")

    if not show and not save_path and not load_path:
        console.print("[yellow]Specify --show, --save, or --load.[/yellow]")


@main.command(name="steps")
def list_steps():
    """List all pipeline steps and their dependencies."""
    table = Table(title="Pipeline Steps", show_header=True, header_style="bold cyan")
    table.add_column("Step", style="bold", width=6, justify="center")
    table.add_column("Description", width=65)
    table.add_column("Depends On", width=20)

    for step_num in sorted(STEP_REGISTRY.keys()):
        _, _, desc = STEP_REGISTRY[step_num]
        deps = STEP_DEPENDENCIES.get(step_num, [])
        deps_str = ", ".join(str(d) for d in deps) if deps else "—"
        table.add_row(str(step_num), desc, deps_str)

    console.print()
    console.print(table)
    console.print()
    console.print(f"[dim]Total: {len(STEP_REGISTRY)} steps[/dim]")


if __name__ == "__main__":
    main()
