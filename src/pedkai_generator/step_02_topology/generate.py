"""
Step 02: Full Network Topology Generation — Main Orchestrator.

Coordinates all domain-specific topology builders and writes the final
output Parquet files:

  - output/ground_truth_entities.parquet     (~1.49M entities)
  - output/ground_truth_relationships.parquet (~2.21M relationships)
  - output/neighbour_relations.parquet       (~200K pairs)

Build order:
  1. Mobile RAN (reads sites/cells from Step 01 intermediate files)
  2. Transport (shared backbone — convergence point)
  3. Fixed Broadband Access (FTTP/FTTC/enterprise)
  4. Core Network (EPC/5GC/IMS)
  5. Logical/Service (slices, tracking areas, QoS profiles)
  6. Power/Environment (supplement for exchanges + generators)
  7. Cell Neighbour Relations (spatial index + handover metrics)

The shared TopologyAccumulator collects entities and relationships from
all domain builders, then serialises them to Parquet at the end.

Design references:
  - THREAD_SUMMARY Section 5: Complete Telecom Topology Model
  - THREAD_SUMMARY Section 7: Final Output Specification (files 1, 2, 14)
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table

from pedkai_generator.config.settings import GeneratorConfig
from pedkai_generator.step_02_topology.builders import TopologyAccumulator
from pedkai_generator.step_02_topology.mobile_ran import build_mobile_ran_topology
from pedkai_generator.step_02_topology.neighbours import build_neighbour_relations
from pedkai_generator.step_02_topology.other_domains import (
    build_core_network_topology,
    build_fixed_broadband_topology,
    build_logical_service_topology,
    build_power_environment_supplement,
    build_transport_topology,
)

console = Console()


# ---------------------------------------------------------------------------
# DataFrame conversion helpers
# ---------------------------------------------------------------------------

# Columns expected in ground_truth_entities.parquet
# (must match the contract in step_00_schema/contracts.py)
ENTITY_COLUMNS = [
    "entity_id",
    "tenant_id",
    "entity_type",
    "name",
    "external_id",
    "domain",
    "geo_lat",
    "geo_lon",
    "site_id",
    "site_type",
    "deployment_profile",
    "province",
    "timezone",
    "vendor",
    "rat_type",
    "band",
    "bandwidth_mhz",
    "max_tx_power_dbm",
    "max_prbs",
    "frequency_mhz",
    "sector_id",
    "azimuth_deg",
    "electrical_tilt_deg",
    "antenna_height_m",
    "inter_site_distance_m",
    "revenue_weight",
    "sla_tier",
    "is_nsa_anchor",
    "nsa_anchor_cell_id",
    "parent_entity_id",
    "properties_json",
]

RELATIONSHIP_COLUMNS = [
    "relationship_id",
    "tenant_id",
    "from_entity_id",
    "from_entity_type",
    "relationship_type",
    "to_entity_id",
    "to_entity_type",
    "domain",
    "properties_json",
]

NEIGHBOUR_COLUMNS = [
    "relation_id",
    "tenant_id",
    "from_cell_id",
    "from_cell_rat",
    "from_cell_band",
    "to_cell_id",
    "to_cell_rat",
    "to_cell_band",
    "neighbour_type",
    "is_intra_site",
    "distance_m",
    "handover_attempts",
    "handover_success_rate",
    "cio_offset_db",
    "no_remove_flag",
]


def _normalise_entity_row(row: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure an entity row has all expected columns, filling missing ones
    with None.  Also coerces types where necessary.
    """
    normalised: dict[str, Any] = {}
    for col in ENTITY_COLUMNS:
        val = row.get(col)
        # Type coercions
        if col == "max_prbs" and val is not None:
            val = int(val) if val is not None else None
        if col == "sector_id" and val is not None:
            val = int(val) if val is not None else None
        normalised[col] = val
    return normalised


def _normalise_relationship_row(row: dict[str, Any]) -> dict[str, Any]:
    """Ensure a relationship row has all expected columns."""
    return {col: row.get(col) for col in RELATIONSHIP_COLUMNS}


def _normalise_neighbour_row(row: dict[str, Any]) -> dict[str, Any]:
    """Ensure a neighbour relation row has all expected columns."""
    return {col: row.get(col) for col in NEIGHBOUR_COLUMNS}


def _rows_to_entities_df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """
    Convert a list of entity row dicts to a Polars DataFrame with correct
    column types matching the ground_truth_entities contract.
    """
    normalised = [_normalise_entity_row(r) for r in rows]

    # Build column-oriented data
    data: dict[str, list] = {col: [] for col in ENTITY_COLUMNS}
    for row in normalised:
        for col in ENTITY_COLUMNS:
            data[col].append(row.get(col))

    # Create DataFrame with explicit types to avoid inference issues
    df = pl.DataFrame(data)

    # Cast columns to correct types
    cast_exprs = []

    # String columns (most columns)
    string_cols = [
        "entity_id",
        "tenant_id",
        "entity_type",
        "name",
        "external_id",
        "domain",
        "site_id",
        "site_type",
        "deployment_profile",
        "province",
        "timezone",
        "vendor",
        "rat_type",
        "band",
        "sla_tier",
        "nsa_anchor_cell_id",
        "parent_entity_id",
        "properties_json",
    ]
    for col in string_cols:
        if col in df.columns:
            cast_exprs.append(pl.col(col).cast(pl.Utf8))

    # Float columns
    float_cols = [
        "geo_lat",
        "geo_lon",
        "bandwidth_mhz",
        "max_tx_power_dbm",
        "frequency_mhz",
        "azimuth_deg",
        "electrical_tilt_deg",
        "antenna_height_m",
        "inter_site_distance_m",
        "revenue_weight",
    ]
    for col in float_cols:
        if col in df.columns:
            cast_exprs.append(pl.col(col).cast(pl.Float64))

    # Int columns
    if "max_prbs" in df.columns:
        cast_exprs.append(pl.col("max_prbs").cast(pl.Int32))
    if "sector_id" in df.columns:
        cast_exprs.append(pl.col("sector_id").cast(pl.Int32))

    # Boolean columns
    if "is_nsa_anchor" in df.columns:
        cast_exprs.append(pl.col("is_nsa_anchor").cast(pl.Boolean))

    if cast_exprs:
        df = df.with_columns(cast_exprs)

    return df


def _rows_to_relationships_df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Convert relationship row dicts to a Polars DataFrame."""
    normalised = [_normalise_relationship_row(r) for r in rows]

    data: dict[str, list] = {col: [] for col in RELATIONSHIP_COLUMNS}
    for row in normalised:
        for col in RELATIONSHIP_COLUMNS:
            data[col].append(row.get(col))

    df = pl.DataFrame(data)

    # All columns are strings
    cast_exprs = [pl.col(col).cast(pl.Utf8) for col in RELATIONSHIP_COLUMNS]
    df = df.with_columns(cast_exprs)

    return df


def _rows_to_neighbours_df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Convert neighbour relation row dicts to a Polars DataFrame."""
    normalised = [_normalise_neighbour_row(r) for r in rows]

    data: dict[str, list] = {col: [] for col in NEIGHBOUR_COLUMNS}
    for row in normalised:
        for col in NEIGHBOUR_COLUMNS:
            data[col].append(row.get(col))

    df = pl.DataFrame(data)

    # Type casts
    string_cols = [
        "relation_id",
        "tenant_id",
        "from_cell_id",
        "from_cell_rat",
        "from_cell_band",
        "to_cell_id",
        "to_cell_rat",
        "to_cell_band",
        "neighbour_type",
    ]
    float_cols = [
        "distance_m",
        "handover_attempts",
        "handover_success_rate",
        "cio_offset_db",
    ]
    bool_cols = ["is_intra_site", "no_remove_flag"]

    cast_exprs = []
    for col in string_cols:
        if col in df.columns:
            cast_exprs.append(pl.col(col).cast(pl.Utf8))
    for col in float_cols:
        if col in df.columns:
            cast_exprs.append(pl.col(col).cast(pl.Float64))
    for col in bool_cols:
        if col in df.columns:
            cast_exprs.append(pl.col(col).cast(pl.Boolean))

    if cast_exprs:
        df = df.with_columns(cast_exprs)

    return df


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------


def _print_topology_summary(acc: TopologyAccumulator) -> None:
    """Print a detailed summary of the generated topology."""

    # ── Entity counts by domain ──────────────────────────────
    console.print("\n  [bold]Entity Counts by Domain:[/bold]")
    domain_table = Table(show_header=True, header_style="bold cyan", padding=(0, 2))
    domain_table.add_column("Domain", width=25)
    domain_table.add_column("Entities", justify="right", width=12)
    domain_table.add_column("% of Total", justify="right", width=10)

    domain_counts = acc.entity_count_by_domain()
    total_entities = acc.entity_count
    for domain in sorted(domain_counts.keys()):
        count = domain_counts[domain]
        pct = count / total_entities * 100 if total_entities > 0 else 0
        domain_table.add_row(domain, f"{count:,}", f"{pct:.1f}%")
    domain_table.add_section()
    domain_table.add_row("[bold]Total[/bold]", f"[bold]{total_entities:,}[/bold]", "[bold]100.0%[/bold]")
    console.print(domain_table)

    # ── Relationship counts by domain ────────────────────────
    console.print("\n  [bold]Relationship Counts by Domain:[/bold]")
    rel_table = Table(show_header=True, header_style="bold cyan", padding=(0, 2))
    rel_table.add_column("Domain", width=25)
    rel_table.add_column("Relationships", justify="right", width=12)
    rel_table.add_column("% of Total", justify="right", width=10)

    rel_domain_counts = acc.relationship_count_by_domain()
    total_rels = acc.relationship_count
    for domain in sorted(rel_domain_counts.keys()):
        count = rel_domain_counts[domain]
        pct = count / total_rels * 100 if total_rels > 0 else 0
        rel_table.add_row(domain, f"{count:,}", f"{pct:.1f}%")
    rel_table.add_section()
    rel_table.add_row("[bold]Total[/bold]", f"[bold]{total_rels:,}[/bold]", "[bold]100.0%[/bold]")
    console.print(rel_table)

    # ── Entity counts by type (top 30) ───────────────────────
    console.print("\n  [bold]Entity Counts by Type (top 30):[/bold]")
    type_table = Table(show_header=True, header_style="bold cyan", padding=(0, 2))
    type_table.add_column("Entity Type", width=25)
    type_table.add_column("Count", justify="right", width=12)

    type_counts = acc.entity_count_by_type()
    sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
    for etype, count in sorted_types[:30]:
        type_table.add_row(etype, f"{count:,}")
    if len(sorted_types) > 30:
        remaining = sum(c for _, c in sorted_types[30:])
        type_table.add_section()
        type_table.add_row(f"[dim]({len(sorted_types) - 30} more types)[/dim]", f"[dim]{remaining:,}[/dim]")
    console.print(type_table)

    # ── Neighbour relation summary ───────────────────────────
    if acc.neighbour_relation_count > 0:
        console.print(f"\n  [bold]Neighbour Relations:[/bold] {acc.neighbour_relation_count:,} pairs")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_topology(config: GeneratorConfig) -> None:
    """
    Step 02 entry point: Generate the full converged-operator topology.

    Orchestrates all domain builders in the correct order, then writes
    the three output Parquet files. Deterministic given config.global_seed.

    Output files:
        - output/ground_truth_entities.parquet
        - output/ground_truth_relationships.parquet
        - output/neighbour_relations.parquet
    """
    seed = config.seed_for("step_02_topology")
    rng = np.random.default_rng(seed)
    console.print(f"  [dim]Seed for step_02: {seed}[/dim]")

    # Shared accumulator for all domain builders
    acc = TopologyAccumulator()

    timings: dict[str, float] = {}

    # ── 1. Mobile RAN ────────────────────────────────────────
    console.print("\n  [bold]Building Mobile RAN topology...[/bold]")
    t0 = time.time()
    build_mobile_ran_topology(config, acc, rng)
    timings["Mobile RAN"] = time.time() - t0
    console.print(f"    [dim]{timings['Mobile RAN']:.1f}s[/dim]")

    # ── 2. Transport ─────────────────────────────────────────
    console.print("\n  [bold]Building Transport topology...[/bold]")
    t0 = time.time()
    build_transport_topology(config, acc, rng)
    timings["Transport"] = time.time() - t0
    console.print(f"    [dim]{timings['Transport']:.1f}s[/dim]")

    # ── 3. Fixed Broadband Access ────────────────────────────
    console.print("\n  [bold]Building Fixed Broadband Access topology...[/bold]")
    t0 = time.time()
    build_fixed_broadband_topology(config, acc, rng)
    timings["Fixed Broadband"] = time.time() - t0
    console.print(f"    [dim]{timings['Fixed Broadband']:.1f}s[/dim]")

    # ── 4. Core Network ──────────────────────────────────────
    console.print("\n  [bold]Building Core Network topology...[/bold]")
    t0 = time.time()
    build_core_network_topology(config, acc, rng)
    timings["Core Network"] = time.time() - t0
    console.print(f"    [dim]{timings['Core Network']:.1f}s[/dim]")

    # ── 5. Logical / Service ─────────────────────────────────
    console.print("\n  [bold]Building Logical/Service topology...[/bold]")
    t0 = time.time()
    build_logical_service_topology(config, acc, rng)
    timings["Logical/Service"] = time.time() - t0
    console.print(f"    [dim]{timings['Logical/Service']:.1f}s[/dim]")

    # ── 6. Power / Environment (supplement) ──────────────────
    console.print("\n  [bold]Building Power/Environment supplement...[/bold]")
    t0 = time.time()
    build_power_environment_supplement(config, acc, rng)
    timings["Power/Environment"] = time.time() - t0
    console.print(f"    [dim]{timings['Power/Environment']:.1f}s[/dim]")

    # ── 7. Cell Neighbour Relations ──────────────────────────
    console.print("\n  [bold]Building Cell Neighbour Relations...[/bold]")
    t0 = time.time()
    build_neighbour_relations(config, acc, rng)
    timings["Neighbours"] = time.time() - t0
    console.print(f"    [dim]{timings['Neighbours']:.1f}s[/dim]")

    # ── Print summary ────────────────────────────────────────
    _print_topology_summary(acc)

    # Print build timings
    console.print("\n  [bold]Build Timings:[/bold]")
    timing_table = Table(show_header=True, header_style="bold cyan", padding=(0, 2))
    timing_table.add_column("Domain", width=25)
    timing_table.add_column("Time", justify="right", width=10)
    total_time = 0.0
    for domain, elapsed in timings.items():
        total_time += elapsed
        if elapsed < 60:
            time_str = f"{elapsed:.1f}s"
        else:
            time_str = f"{elapsed / 60:.1f}m"
        timing_table.add_row(domain, time_str)
    timing_table.add_section()
    if total_time < 60:
        total_str = f"{total_time:.1f}s"
    else:
        total_str = f"{total_time / 60:.1f}m"
    timing_table.add_row("[bold]Total[/bold]", f"[bold]{total_str}[/bold]")
    console.print(timing_table)

    # ── Write Parquet files ──────────────────────────────────
    output_dir = config.paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ground truth entities
    console.print("\n  [bold]Writing ground_truth_entities.parquet...[/bold]")
    t0 = time.time()
    entities_df = _rows_to_entities_df(acc.entities)
    entities_path = output_dir / "ground_truth_entities.parquet"
    entities_df.write_parquet(entities_path, compression="zstd", compression_level=3)
    entities_size_mb = entities_path.stat().st_size / (1024 * 1024)
    console.print(
        f"    [green]✓[/green] {entities_path.name}: "
        f"{entities_df.height:,} rows, {entities_df.width} columns, "
        f"{entities_size_mb:.1f} MB ({time.time() - t0:.1f}s)"
    )

    # 2. Ground truth relationships
    console.print("  [bold]Writing ground_truth_relationships.parquet...[/bold]")
    t0 = time.time()
    rels_df = _rows_to_relationships_df(acc.relationships)
    rels_path = output_dir / "ground_truth_relationships.parquet"
    rels_df.write_parquet(rels_path, compression="zstd", compression_level=3)
    rels_size_mb = rels_path.stat().st_size / (1024 * 1024)
    console.print(
        f"    [green]✓[/green] {rels_path.name}: "
        f"{rels_df.height:,} rows, {rels_df.width} columns, "
        f"{rels_size_mb:.1f} MB ({time.time() - t0:.1f}s)"
    )

    # 3. Neighbour relations
    console.print("  [bold]Writing neighbour_relations.parquet...[/bold]")
    t0 = time.time()
    if acc.neighbour_relations:
        neighbours_df = _rows_to_neighbours_df(acc.neighbour_relations)
    else:
        # Empty DataFrame with correct schema
        neighbours_df = pl.DataFrame({col: [] for col in NEIGHBOUR_COLUMNS})
    neighbours_path = output_dir / "neighbour_relations.parquet"
    neighbours_df.write_parquet(neighbours_path, compression="zstd", compression_level=3)
    neighbours_size_mb = neighbours_path.stat().st_size / (1024 * 1024)
    console.print(
        f"    [green]✓[/green] {neighbours_path.name}: "
        f"{neighbours_df.height:,} rows, {neighbours_df.width} columns, "
        f"{neighbours_size_mb:.1f} MB ({time.time() - t0:.1f}s)"
    )

    # ── Final totals ─────────────────────────────────────────
    total_size_mb = entities_size_mb + rels_size_mb + neighbours_size_mb
    console.print(
        f"\n  [bold green]✓ Topology generation complete:[/bold green] "
        f"{acc.entity_count:,} entities, "
        f"{acc.relationship_count:,} relationships, "
        f"{acc.neighbour_relation_count:,} neighbour pairs"
    )
    console.print(f"    Total Parquet size: {total_size_mb:.1f} MB (3 files in {output_dir})")

    # ── Also save topology metadata for downstream steps ─────
    # Save a small JSON with counts and cross-references that downstream
    # steps may need without re-reading the full Parquet files.
    import json

    metadata = {
        "entity_count": acc.entity_count,
        "relationship_count": acc.relationship_count,
        "neighbour_relation_count": acc.neighbour_relation_count,
        "entity_count_by_domain": acc.entity_count_by_domain(),
        "relationship_count_by_domain": acc.relationship_count_by_domain(),
        "entity_count_by_type": acc.entity_count_by_type(),
        "build_timings": timings,
    }

    # Store cross-domain reference IDs for use by Steps 03-08
    transport_refs = getattr(acc, "_transport_refs", {})
    fixed_refs = getattr(acc, "_fixed_refs", {})
    core_refs = getattr(acc, "_core_refs", {})

    # Save just the counts (not the full ID lists which can be huge)
    metadata["cross_domain_ref_counts"] = {
        "pe_routers": len(transport_refs.get("pe_router_ids", [])),
        "p_routers": len(transport_refs.get("p_router_ids", [])),
        "agg_switches": len(transport_refs.get("agg_switch_ids", [])),
        "bngs": len(transport_refs.get("bng_ids", [])),
        "l3vpns": len(transport_refs.get("l3vpn_ids", [])),
        "exchanges": len(fixed_refs.get("exchange_ids", [])),
        "olts": len(fixed_refs.get("olt_ids", [])),
        "onts": len(fixed_refs.get("ont_ids", [])),
        "ntes": len(fixed_refs.get("nte_ids", [])),
    }

    metadata_path = config.paths.intermediate_dir / "topology_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    console.print(f"    [dim]Metadata saved to {metadata_path}[/dim]")
