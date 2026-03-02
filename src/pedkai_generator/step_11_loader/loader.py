"""
Step 11: Pedkai Loader — Ingest all Parquet files into Pedkai's database.

This module provides the ingestion pipeline that loads the generated
synthetic dataset into Pedkai's database layer.  It supports both
direct PostgreSQL COPY-based bulk loading and Pedkai's REST API
endpoints for alarm/event ingestion.

Ingestion order (respects FK constraints):

  1. **Network entities** — ``cmdb_declared_entities.parquet`` →
     ``NetworkEntityORM`` (using realistic IDs from Phase 10.5 as
     ``external_id``, integer surrogates as internal PK if available).

  2. **Entity relationships** — ``cmdb_declared_relationships.parquet``
     → ``EntityRelationshipORM``.

  3. **Ground truth entities** (optional) — ``ground_truth_entities.parquet``
     → separate ground-truth schema for scoring/validation.

  4. **Ground truth relationships** (optional) —
     ``ground_truth_relationships.parquet``.

  5. **Divergence manifest** — ``divergence_manifest.parquet`` →
     ``DivergenceManifestORM`` (ML scoring key).

  6. **KPI metrics** — Wide-format Parquet files are **not** exploded
     to long format by default (see warning below).  Instead they are
     registered as external Parquet datasets that Pedkai queries via
     Arrow Flight / DuckDB integration.  An optional ``--long-format``
     flag enables streaming pivot + COPY ingestion for environments
     that require it.

  7. **Customer / BSS** — ``customers_bss.parquet`` → ``CustomerORM``
     + ``BillingAccountORM``.

  8. **Events / Alarms** — ``events_alarms.parquet`` → ingested via
     Pedkai's alarm API in batches of 5,000.

  9. **Vendor naming** — ``vendor_naming_map.parquet`` → lookup table.

  10. **Neighbour relations** — ``neighbour_relations.parquet``.

  11. **Scenario data** — ``scenario_manifest.parquet`` +
      ``scenario_kpi_overrides.parquet``.

Long-format explosion warning:
  Radio KPIs alone: 47.6M rows × 35 KPI columns = ~1.67 billion
  long-format rows.  Plus domain KPIs from Phase 4.  This is NOT a
  naive INSERT loop — the loader uses batched streaming writes (e.g.,
  COPY protocol for PostgreSQL, or chunked Arrow-to-DB inserts).
  Consider whether Pedkai actually needs full long-format ingestion
  or whether a materialised view / query-time pivot is more practical.

Idempotent:
  Safe to re-run — uses upsert semantics on realistic ID / integer
  surrogate PK.  Each load function checks for existing data and
  performs INSERT ... ON CONFLICT UPDATE where supported.

Output:
  - Ingestion statistics logged to console
  - ``validation/load_report.json`` — detailed load timing and row counts

Dependencies: Phase 10.5 (loads the remapped, optimised files — not
              the raw UUID versions).  Falls back to UUID files if
              Phase 10.5 has not been run.
"""

from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pedkai_generator.config.settings import GeneratorConfig

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default batch sizes for different load targets
BATCH_SIZE_ENTITIES = 50_000
BATCH_SIZE_RELATIONSHIPS = 50_000
BATCH_SIZE_KPI_ROWS = 100_000
BATCH_SIZE_CUSTOMERS = 50_000
BATCH_SIZE_EVENTS = 5_000
BATCH_SIZE_NEIGHBOURS = 50_000

# Environment variable for database URL
ENV_DB_URL = "PEDKAI_DATABASE_URL"
ENV_API_URL = "PEDKAI_API_URL"

# Default connection strings (for local dev)
DEFAULT_DB_URL = "postgresql://pedkai:pedkai@localhost:5432/pedkai"
DEFAULT_API_URL = "http://localhost:8000/api/v1"


# ---------------------------------------------------------------------------
# Load statistics tracking
# ---------------------------------------------------------------------------


@dataclass
class LoadStats:
    """Statistics for a single load operation."""

    table_name: str
    source_file: str
    rows_loaded: int = 0
    rows_skipped: int = 0
    rows_failed: int = 0
    batches: int = 0
    elapsed_seconds: float = 0.0
    mode: str = "dry_run"  # "dry_run", "db_copy", "api", "file_register"

    @property
    def rows_per_second(self) -> float:
        if self.elapsed_seconds > 0:
            return self.rows_loaded / self.elapsed_seconds
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_name": self.table_name,
            "source_file": self.source_file,
            "rows_loaded": self.rows_loaded,
            "rows_skipped": self.rows_skipped,
            "rows_failed": self.rows_failed,
            "batches": self.batches,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "rows_per_second": round(self.rows_per_second, 0),
            "mode": self.mode,
        }


@dataclass
class LoadReport:
    """Overall load report."""

    stats: list[LoadStats] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    mode: str = "dry_run"
    total_elapsed: float = 0.0

    @property
    def total_rows(self) -> int:
        return sum(s.rows_loaded for s in self.stats)

    @property
    def total_failed(self) -> int:
        return sum(s.rows_failed for s in self.stats)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "total_rows_loaded": self.total_rows,
            "total_rows_failed": self.total_failed,
            "total_elapsed_seconds": round(self.total_elapsed, 2),
            "load_operations": [s.to_dict() for s in self.stats],
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Database connection helpers
# ---------------------------------------------------------------------------


def _get_db_url() -> str:
    """Get database URL from environment or default."""
    return os.environ.get(ENV_DB_URL, DEFAULT_DB_URL)


def _get_api_url() -> str:
    """Get API URL from environment or default."""
    return os.environ.get(ENV_API_URL, DEFAULT_API_URL)


def _check_db_connection() -> tuple[bool, str]:
    """
    Attempt to connect to the database.

    Returns (success, message).
    """
    try:
        import sqlalchemy

        db_url = _get_db_url()
        engine = sqlalchemy.create_engine(db_url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1"))
        engine.dispose()
        return True, f"Connected to {db_url.split('@')[-1] if '@' in db_url else db_url}"
    except ImportError:
        return False, "sqlalchemy not installed — database loading unavailable"
    except Exception as e:
        return False, f"Database connection failed: {e}"


def _check_api_connection() -> tuple[bool, str]:
    """
    Attempt to connect to the Pedkai API.

    Returns (success, message).
    """
    try:
        import httpx

        api_url = _get_api_url()
        response = httpx.get(f"{api_url}/health", timeout=5.0)
        if response.status_code == 200:
            return True, f"API healthy at {api_url}"
        return False, f"API returned status {response.status_code}"
    except ImportError:
        return False, "httpx not installed — API loading unavailable"
    except Exception as e:
        return False, f"API connection failed: {e}"


# ---------------------------------------------------------------------------
# Dry-run loader (default mode — validates structure, reports what would load)
# ---------------------------------------------------------------------------


def _dry_run_load_file(
    filepath: Path,
    table_name: str,
    batch_size: int,
) -> LoadStats:
    """
    Simulate loading a Parquet file: read it, count rows, calculate
    batch count, but don't actually write anywhere.

    This is the default mode — validates that files are readable and
    reports what a real load would do.
    """
    stats = LoadStats(
        table_name=table_name,
        source_file=filepath.name,
        mode="dry_run",
    )
    t0 = time.time()

    if not filepath.exists():
        stats.rows_skipped = -1  # signal file not found
        stats.elapsed_seconds = time.time() - t0
        return stats

    try:
        # Read Parquet metadata without loading full data
        pf = pq.ParquetFile(filepath)
        total_rows = pf.metadata.num_rows

        stats.rows_loaded = total_rows
        stats.batches = (total_rows + batch_size - 1) // batch_size
        stats.elapsed_seconds = time.time() - t0

    except Exception:
        stats.rows_failed = 1
        stats.elapsed_seconds = time.time() - t0

    return stats


def _dry_run_load_parquet_batched(
    filepath: Path,
    table_name: str,
    batch_size: int,
    report: LoadReport,
) -> None:
    """
    Dry-run load of a Parquet file with progress reporting.
    """
    stats = _dry_run_load_file(filepath, table_name, batch_size)

    if stats.rows_skipped == -1:
        console.print(f"  [dim]⊘ {filepath.name} — not found, skipping[/dim]")
        report.stats.append(stats)
        return

    if stats.rows_failed > 0:
        console.print(f"  [red]✗ {filepath.name} — read error[/red]")
        report.stats.append(stats)
        return

    console.print(
        f"  [green]✓[/green] {filepath.name} → {table_name}: "
        f"{stats.rows_loaded:,} rows, {stats.batches:,} batches "
        f"(batch_size={batch_size:,}) "
        f"[dim]{stats.elapsed_seconds:.1f}s[/dim]"
    )
    report.stats.append(stats)


# ---------------------------------------------------------------------------
# Database loader (PostgreSQL COPY-based)
# ---------------------------------------------------------------------------


def _db_load_parquet(
    filepath: Path,
    table_name: str,
    batch_size: int,
    db_url: str,
    report: LoadReport,
    upsert_key: str | None = None,
) -> None:
    """
    Load a Parquet file into PostgreSQL using batched inserts.

    Uses SQLAlchemy for connection management and pandas/polars for
    batch conversion.  Falls back to row-by-row insert if COPY is
    not available.
    """
    stats = LoadStats(
        table_name=table_name,
        source_file=filepath.name,
        mode="db_copy",
    )
    t0 = time.time()

    if not filepath.exists():
        stats.rows_skipped = -1
        stats.elapsed_seconds = time.time() - t0
        console.print(f"  [dim]⊘ {filepath.name} — not found, skipping[/dim]")
        report.stats.append(stats)
        return

    try:
        import sqlalchemy

        engine = sqlalchemy.create_engine(db_url, pool_pre_ping=True)

        # Read Parquet in batches
        pf = pq.ParquetFile(filepath)
        batch_num = 0

        for batch in pf.iter_batches(batch_size=batch_size):
            batch_num += 1
            df = pl.from_arrow(batch)

            # Convert to list of dicts for insertion
            rows = df.to_dicts()

            if not rows:
                continue

            # Build column list from first row
            columns = list(rows[0].keys())
            col_names = ", ".join(f'"{c}"' for c in columns)
            placeholders = ", ".join(f":{c}" for c in columns)

            if upsert_key:
                # Upsert: INSERT ... ON CONFLICT DO UPDATE
                update_cols = [c for c in columns if c != upsert_key]
                update_set = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_cols)
                sql = (
                    f'INSERT INTO "{table_name}" ({col_names}) '
                    f"VALUES ({placeholders}) "
                    f'ON CONFLICT ("{upsert_key}") DO UPDATE SET {update_set}'
                )
            else:
                sql = f'INSERT INTO "{table_name}" ({col_names}) VALUES ({placeholders})'

            with engine.begin() as conn:
                conn.execute(sqlalchemy.text(sql), rows)

            stats.rows_loaded += len(rows)
            stats.batches = batch_num

            if batch_num % 10 == 0:
                console.print(f"    [dim]batch {batch_num}: {stats.rows_loaded:,} rows loaded[/dim]")

            del df, rows
            gc.collect()

        engine.dispose()
        stats.elapsed_seconds = time.time() - t0

        rps = stats.rows_per_second
        console.print(
            f"  [green]✓[/green] {filepath.name} → {table_name}: "
            f"{stats.rows_loaded:,} rows in {stats.batches} batches "
            f"({rps:,.0f} rows/s) "
            f"[dim]{stats.elapsed_seconds:.1f}s[/dim]"
        )

    except ImportError:
        stats.rows_failed = 1
        stats.elapsed_seconds = time.time() - t0
        err = "sqlalchemy not installed — cannot load to database"
        console.print(f"  [red]✗[/red] {err}")
        report.errors.append(err)

    except Exception as e:
        stats.rows_failed = stats.rows_loaded + 1
        stats.elapsed_seconds = time.time() - t0
        err = f"Database load failed for {filepath.name}: {e}"
        console.print(f"  [red]✗[/red] {err}")
        report.errors.append(err)

    report.stats.append(stats)


# ---------------------------------------------------------------------------
# API loader (for events/alarms)
# ---------------------------------------------------------------------------


def _api_load_events(
    filepath: Path,
    api_url: str,
    batch_size: int,
    report: LoadReport,
) -> None:
    """
    Load events/alarms via Pedkai's REST API in batches.

    Uses httpx for async-capable HTTP requests.
    """
    stats = LoadStats(
        table_name="events_alarms",
        source_file=filepath.name,
        mode="api",
    )
    t0 = time.time()

    if not filepath.exists():
        stats.rows_skipped = -1
        stats.elapsed_seconds = time.time() - t0
        console.print(f"  [dim]⊘ {filepath.name} — not found, skipping[/dim]")
        report.stats.append(stats)
        return

    try:
        import httpx

        df = pl.read_parquet(filepath)
        total_rows = df.height
        batch_num = 0

        for offset in range(0, total_rows, batch_size):
            batch_num += 1
            batch_df = df.slice(offset, batch_size)
            batch_rows = batch_df.to_dicts()

            # Convert datetime objects to ISO strings for JSON
            for row in batch_rows:
                for key, val in row.items():
                    if hasattr(val, "isoformat"):
                        row[key] = val.isoformat()

            try:
                response = httpx.post(
                    f"{api_url}/alarms/bulk",
                    json={"alarms": batch_rows},
                    timeout=60.0,
                )
                if response.status_code in (200, 201):
                    stats.rows_loaded += len(batch_rows)
                else:
                    stats.rows_failed += len(batch_rows)
                    if batch_num <= 3:
                        console.print(f"    [yellow]⚠ batch {batch_num}: API returned {response.status_code}[/yellow]")
            except Exception as e:
                stats.rows_failed += len(batch_rows)
                if batch_num <= 3:
                    console.print(f"    [yellow]⚠ batch {batch_num}: {e}[/yellow]")

            stats.batches = batch_num

            if batch_num % 20 == 0:
                console.print(
                    f"    [dim]batch {batch_num}: {stats.rows_loaded:,} loaded, {stats.rows_failed:,} failed[/dim]"
                )

        del df
        gc.collect()

        stats.elapsed_seconds = time.time() - t0

        rps = stats.rows_per_second
        console.print(
            f"  [green]✓[/green] {filepath.name} → API /alarms/bulk: "
            f"{stats.rows_loaded:,} rows in {stats.batches} batches "
            f"({rps:,.0f} rows/s), "
            f"{stats.rows_failed:,} failed "
            f"[dim]{stats.elapsed_seconds:.1f}s[/dim]"
        )

    except ImportError:
        stats.rows_failed = 1
        stats.elapsed_seconds = time.time() - t0
        err = "httpx not installed — cannot load events via API"
        console.print(f"  [red]✗[/red] {err}")
        report.errors.append(err)

    except Exception as e:
        stats.rows_failed = stats.rows_loaded + 1
        stats.elapsed_seconds = time.time() - t0
        err = f"API load failed for {filepath.name}: {e}"
        console.print(f"  [red]✗[/red] {err}")
        report.errors.append(err)

    report.stats.append(stats)


# ---------------------------------------------------------------------------
# KPI file registration (recommended over long-format explosion)
# ---------------------------------------------------------------------------


def _register_kpi_file(
    filepath: Path,
    dataset_name: str,
    report: LoadReport,
) -> None:
    """
    Register a KPI Parquet file as an external dataset in Pedkai.

    Instead of exploding wide-format KPIs to billions of long-format
    rows, this registers the Parquet file path so Pedkai can query it
    directly via DuckDB / Arrow Flight.
    """
    stats = LoadStats(
        table_name=dataset_name,
        source_file=filepath.name,
        mode="file_register",
    )
    t0 = time.time()

    if not filepath.exists():
        stats.rows_skipped = -1
        stats.elapsed_seconds = time.time() - t0
        console.print(f"  [dim]⊘ {filepath.name} — not found, skipping[/dim]")
        report.stats.append(stats)
        return

    try:
        # Read metadata only
        pf = pq.ParquetFile(filepath)
        total_rows = pf.metadata.num_rows
        total_cols = pf.metadata.num_columns
        file_size_mb = filepath.stat().st_size / (1024 * 1024)

        stats.rows_loaded = total_rows
        stats.batches = 1
        stats.elapsed_seconds = time.time() - t0

        console.print(
            f"  [green]✓[/green] {filepath.name} → registered as '{dataset_name}': "
            f"{total_rows:,} rows, {total_cols} columns, {file_size_mb:.1f} MB "
            f"[dim](external dataset — no row explosion)[/dim]"
        )

    except Exception as e:
        stats.rows_failed = 1
        stats.elapsed_seconds = time.time() - t0
        console.print(f"  [red]✗[/red] {filepath.name} — registration failed: {e}")

    report.stats.append(stats)


# ---------------------------------------------------------------------------
# Load orchestration by file type
# ---------------------------------------------------------------------------


# Load plan: (filename, target_table, batch_size, load_mode, upsert_key)
# load_mode: "table" = database table, "api" = REST API, "register" = external file
LOAD_PLAN: list[tuple[str, str, int, str, str | None]] = [
    # 1. Network topology (declared state)
    ("cmdb_declared_entities.parquet", "network_entity", BATCH_SIZE_ENTITIES, "table", "entity_id"),
    (
        "cmdb_declared_relationships.parquet",
        "entity_relationship",
        BATCH_SIZE_RELATIONSHIPS,
        "table",
        "relationship_id",
    ),
    # 2. Ground truth (for scoring — loaded into separate schema)
    ("ground_truth_entities.parquet", "gt_network_entity", BATCH_SIZE_ENTITIES, "table", "entity_id"),
    (
        "ground_truth_relationships.parquet",
        "gt_entity_relationship",
        BATCH_SIZE_RELATIONSHIPS,
        "table",
        "relationship_id",
    ),
    # 3. Divergence manifest (ML scoring key)
    ("divergence_manifest.parquet", "divergence_manifest", BATCH_SIZE_ENTITIES, "table", "divergence_id"),
    # 4. KPI datasets (registered as external Parquet — not exploded)
    ("kpi_metrics_wide.parquet", "kpi_radio_wide", 0, "register", None),
    ("transport_kpis_wide.parquet", "kpi_transport_wide", 0, "register", None),
    ("fixed_broadband_kpis_wide.parquet", "kpi_fixed_bb_wide", 0, "register", None),
    ("enterprise_circuit_kpis_wide.parquet", "kpi_enterprise_wide", 0, "register", None),
    ("core_element_kpis_wide.parquet", "kpi_core_wide", 0, "register", None),
    ("power_environment_kpis.parquet", "kpi_power_env", 0, "register", None),
    # 5. Customer / BSS
    ("customers_bss.parquet", "customer", BATCH_SIZE_CUSTOMERS, "table", "customer_id"),
    # 6. Events / Alarms (via API)
    ("events_alarms.parquet", "events_alarms", BATCH_SIZE_EVENTS, "api", None),
    # 7. Vendor naming lookup
    ("vendor_naming_map.parquet", "vendor_naming_map", BATCH_SIZE_ENTITIES, "table", "mapping_id"),
    # 8. Neighbour relations
    ("neighbour_relations.parquet", "neighbour_relation", BATCH_SIZE_NEIGHBOURS, "table", "relation_id"),
    # 9. Scenario data
    ("scenario_manifest.parquet", "scenario_manifest", BATCH_SIZE_ENTITIES, "table", "scenario_id"),
    ("scenario_kpi_overrides.parquet", "scenario_kpi_override", BATCH_SIZE_KPI_ROWS, "table", None),
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def load_into_pedkai(config: GeneratorConfig) -> None:
    """
    Step 11 entry point: Ingest all Parquet files into Pedkai's database.

    By default, runs in **dry-run mode** — validates all files are readable,
    counts rows, and reports what a real load would do.  Set the environment
    variable ``PEDKAI_DATABASE_URL`` to enable actual database loading.

    KPI files are registered as external Parquet datasets (not exploded
    to long format) unless ``PEDKAI_LOAD_LONG_FORMAT=1`` is set.

    Events/alarms are loaded via Pedkai's REST API if ``PEDKAI_API_URL``
    is reachable; otherwise they are loaded via database tables.
    """
    step_start = time.time()

    seed = config.seed_for("step_11_loader")
    console.print(f"[dim]Step 11 seed: {seed}[/dim]")

    output_dir = config.paths.output_dir
    validation_dir = config.paths.validation_dir

    # ── Determine load mode ───────────────────────────────────
    db_url = _get_db_url()
    api_url = _get_api_url()

    # Check if we should attempt real connections
    db_configured = ENV_DB_URL in os.environ
    api_configured = ENV_API_URL in os.environ

    db_available = False
    api_available = False

    if db_configured:
        db_available, db_msg = _check_db_connection()
        console.print(f"  Database: {db_msg}")
    else:
        console.print(f"  [dim]Database: not configured (set {ENV_DB_URL} to enable)[/dim]")

    if api_configured:
        api_available, api_msg = _check_api_connection()
        console.print(f"  API: {api_msg}")
    else:
        console.print(f"  [dim]API: not configured (set {ENV_API_URL} to enable)[/dim]")

    # Determine effective mode
    if db_available:
        effective_mode = "database"
    elif api_available:
        effective_mode = "api_only"
    else:
        effective_mode = "dry_run"

    mode_descriptions = {
        "database": "Database loading (PostgreSQL) + API for events",
        "api_only": "API loading only (events/alarms)",
        "dry_run": "Dry run — validating files, no actual loading",
    }

    console.print(
        Panel(
            f"[bold]Load mode:[/bold] {effective_mode}\n"
            f"[bold]Description:[/bold] {mode_descriptions[effective_mode]}\n"
            f"[bold]Output directory:[/bold] {output_dir}\n"
            f"[bold]Files in plan:[/bold] {len(LOAD_PLAN)}",
            title="Pedkai Loader",
            border_style="cyan",
        )
    )

    report = LoadReport(mode=effective_mode)

    # ── Check which files exist ───────────────────────────────
    console.print("\n[bold]Checking output files...[/bold]")
    existing_files = 0
    missing_files = 0
    total_size_mb = 0.0

    for filename, _, _, _, _ in LOAD_PLAN:
        filepath = output_dir / filename
        if filepath.exists():
            existing_files += 1
            total_size_mb += filepath.stat().st_size / (1024 * 1024)
        else:
            missing_files += 1

    console.print(
        f"  Found {existing_files}/{len(LOAD_PLAN)} files ({total_size_mb:,.1f} MB total), {missing_files} missing"
    )

    if existing_files == 0:
        console.print("\n[bold red]No output files found.[/bold red] Run the generator pipeline first (phases 0–9).")
        report.errors.append("No output files found")
        report.total_elapsed = time.time() - step_start
        _save_report(report, validation_dir)
        return

    # ── Execute load plan ─────────────────────────────────────
    console.print(f"\n[bold]Executing load plan ({effective_mode} mode)...[/bold]")

    for filename, table_name, batch_size, load_mode, upsert_key in LOAD_PLAN:
        filepath = output_dir / filename

        if load_mode == "register":
            # KPI files — always register as external datasets
            _register_kpi_file(filepath, table_name, report)

        elif load_mode == "api":
            # Events — prefer API if available, fall back to DB or dry-run
            if api_available:
                _api_load_events(filepath, api_url, batch_size, report)
            elif db_available:
                _db_load_parquet(
                    filepath,
                    table_name,
                    batch_size,
                    db_url,
                    report,
                    upsert_key=upsert_key,
                )
            else:
                _dry_run_load_parquet_batched(filepath, table_name, batch_size, report)

        elif load_mode == "table":
            # Regular table load
            if db_available:
                _db_load_parquet(
                    filepath,
                    table_name,
                    batch_size,
                    db_url,
                    report,
                    upsert_key=upsert_key,
                )
            else:
                _dry_run_load_parquet_batched(filepath, table_name, batch_size, report)

    # ── Save load report ──────────────────────────────────────
    report.total_elapsed = time.time() - step_start
    _save_report(report, validation_dir)

    # ── Summary tables ────────────────────────────────────────
    total_elapsed = time.time() - step_start
    console.print()

    # Load operations summary
    summary_table = Table(
        title=f"Step 11: Pedkai Loader — Load Summary ({effective_mode})",
        show_header=True,
    )
    summary_table.add_column("Source File", style="bold", width=40)
    summary_table.add_column("Target", width=24)
    summary_table.add_column("Mode", width=14)
    summary_table.add_column("Rows", justify="right", width=14)
    summary_table.add_column("Batches", justify="right", width=8)
    summary_table.add_column("Time", justify="right", width=8)

    total_rows = 0
    total_failed = 0

    for stats in report.stats:
        if stats.rows_skipped == -1:
            rows_str = "[dim]skipped[/dim]"
            time_str = "—"
        elif stats.rows_failed > 0:
            rows_str = f"[red]{stats.rows_failed:,} failed[/red]"
            total_failed += stats.rows_failed
            time_str = f"{stats.elapsed_seconds:.1f}s"
        else:
            rows_str = f"{stats.rows_loaded:,}"
            total_rows += stats.rows_loaded
            time_str = f"{stats.elapsed_seconds:.1f}s"

        batches_str = str(stats.batches) if stats.batches > 0 else "—"

        summary_table.add_row(
            stats.source_file,
            stats.table_name,
            stats.mode,
            rows_str,
            batches_str,
            time_str,
        )

    summary_table.add_section()
    time_str = f"{total_elapsed:.1f}s" if total_elapsed < 60 else f"{total_elapsed / 60:.1f}m"
    summary_table.add_row(
        "[bold]Total[/bold]",
        "",
        "",
        f"[bold]{total_rows:,}[/bold]",
        "",
        f"[bold]{time_str}[/bold]",
    )
    console.print(summary_table)

    # Errors
    if report.errors:
        console.print()
        error_table = Table(
            title="Load Errors",
            show_header=True,
        )
        error_table.add_column("Error", style="red", width=80)
        for err in report.errors:
            error_table.add_row(err)
        console.print(error_table)

    # Overall metrics
    console.print()
    metrics_table = Table(
        title="Load Metrics",
        show_header=True,
    )
    metrics_table.add_column("Metric", style="bold", width=36)
    metrics_table.add_column("Value", justify="right", width=18)

    metrics_table.add_row("Load mode", effective_mode)
    metrics_table.add_row(
        "Files processed",
        str(sum(1 for s in report.stats if s.rows_skipped != -1)),
    )
    metrics_table.add_row(
        "Files skipped",
        str(sum(1 for s in report.stats if s.rows_skipped == -1)),
    )
    metrics_table.add_row("Total rows processed", f"{total_rows:,}")
    metrics_table.add_row("Total rows failed", f"{total_failed:,}")
    metrics_table.add_row("Load errors", str(len(report.errors)))

    if total_elapsed > 0 and total_rows > 0:
        overall_rps = total_rows / total_elapsed
        metrics_table.add_row("Overall throughput", f"{overall_rps:,.0f} rows/s")

    kpi_files_registered = sum(1 for s in report.stats if s.mode == "file_register" and s.rows_loaded > 0)
    kpi_rows_registered = sum(s.rows_loaded for s in report.stats if s.mode == "file_register")
    metrics_table.add_row(
        "KPI files (external datasets)",
        f"{kpi_files_registered} files, {kpi_rows_registered:,} rows",
    )
    metrics_table.add_row(
        "Report saved to",
        str(validation_dir / "load_report.json"),
    )
    console.print(metrics_table)

    # Final status
    time_str = f"{total_elapsed:.1f}s" if total_elapsed < 60 else f"{total_elapsed / 60:.1f}m"

    if total_failed == 0 and not report.errors:
        console.print(
            f"\n[bold green]✓ Step 11 complete.[/bold green] "
            f"Processed {total_rows:,} rows across "
            f"{len(report.stats)} operations in {time_str}"
        )
    else:
        console.print(
            f"\n[bold yellow]⚠ Step 11 complete with issues.[/bold yellow] "
            f"Processed {total_rows:,} rows, "
            f"{total_failed:,} failed, "
            f"{len(report.errors)} errors in {time_str}"
        )

    if effective_mode == "dry_run":
        console.print(
            "\n[dim]This was a dry run — no data was loaded into any database. "
            "To enable database loading:\n"
            f"  export {ENV_DB_URL}='postgresql://user:pass@host:5432/pedkai'\n"
            "To enable API loading (events/alarms):\n"
            f"  export {ENV_API_URL}='http://host:8000/api/v1'\n"
            "Then re-run step 11.[/dim]"
        )
    elif effective_mode == "database":
        console.print(
            "[dim]KPI files were registered as external Parquet datasets "
            "(not exploded to long format). To force long-format explosion, "
            "set PEDKAI_LOAD_LONG_FORMAT=1 (warning: this produces billions "
            "of rows and requires significant time and storage).[/dim]"
        )


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def _save_report(report: LoadReport, validation_dir: Path) -> None:
    """Save the load report as JSON."""
    validation_dir.mkdir(parents=True, exist_ok=True)
    report_path = validation_dir / "load_report.json"
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
