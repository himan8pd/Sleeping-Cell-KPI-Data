"""
Step 03: Radio-Layer Physics Model & Cell KPI Generation — Orchestrator.

Reads the cell inventory from Step 01, generates environmental conditions
using a **streaming AR(1) generator** (one hour at a time), runs the
vectorised physics chain (SINR → CQI → MCS → throughput) for every hourly
interval, and writes the final `kpi_metrics_wide.parquet`.

MEMORY-SAFE ARCHITECTURE:
  The previous implementation pre-allocated four (720, 66131) float64 matrices
  totalling ~1.45 GB, plus temporaries during generation (~4 GB peak).
  On a 16 GB machine with heavy OS usage this caused OOM / kernel panic.

  This rewrite uses a StreamingEnvironmentGenerator that maintains only
  (n_cells,) AR(1) state vectors (~500 KB each).  Each hour produces ~2 MB
  of environmental conditions, the physics engine produces ~4 MB of KPIs,
  and a single RecordBatch (~30 MB) is immediately flushed as a Parquet
  row group.  Peak working memory ≈ 150-200 MB regardless of simulation
  length.

  Additional size savings:
  - Removed 8 alias columns that were byte-for-byte duplicates (throughput_mbps,
    prb_utilization, traffic_volume, etc.).  These can be trivially computed on
    read by downstream consumers.
  - Use zstd level 9 for better compression of float64 KPIs.
  - Row-group-per-hour gives Parquet better dictionary encoding on the repeated
    string columns (cell_id, site_id, etc.).

Realism enhancements (RED_FLAG_REPORT remediation):
  - RF-04: Stochastic null injection (~0.15% per KPI column) with correlated
    patterns simulating partial ROP failures, PM file corruption, and
    counter schema changes per 3GPP TS 32.401/32.432.
  - RF-05: PM collection gap simulation — ~0.08% of cell-hours are dropped
    entirely (row absent) in correlated per-site bursts, simulating planned
    maintenance, NMS failover, and SFTP collection timeouts.
  - RF-15: Rare rural traffic spike outliers — 0.1% of rural/deep_rural
    cells receive temporary traffic spikes exceeding urban P75, simulating
    festivals, transit cells, resort areas.

Output:
  - output/kpi_metrics_wide.parquet
    ~66,131 cells × 720 hours ≈ ~47.6M rows × 44 KPI columns
    (slightly fewer due to RF-05 collection gaps)
    Target size: ~3 GB (zstd-9, no alias bloat)

Design references:
  - THREAD_SUMMARY Section 3: Simulation Parameters
  - THREAD_SUMMARY Section 5: KPIs by Domain (Cell-Level Radio KPIs)
  - THREAD_SUMMARY Section 8: Size Estimates (~3 GB)
"""

from __future__ import annotations

import gc
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from pedkai_generator.config.settings import GeneratorConfig
from pedkai_generator.step_03_radio_kpis.physics import (
    CellPhysicsInput,
    HourlyConditions,
    compute_cell_kpis_vectorised,
)
from pedkai_generator.step_03_radio_kpis.profiles import (
    StreamingEnvironmentGenerator,
    TrafficProfileConfig,
)

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Simulation epoch: 2024-01-01 00:00:00 UTC (a Monday)
SIMULATION_EPOCH = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# Flush every hour — keeps PyArrow buffer at ~30 MB.
# With 720 hours this gives 720 row groups which is fine for Parquet
# (Parquet footer is tiny, and readers can efficiently skip row groups).
HOURS_PER_ROW_GROUP = 1

# Maximum cells per vectorised physics call.  66k fits easily in ~50 MB.
MAX_CELLS_PER_BATCH = 80_000

# How often to run gc.collect() (every N row-group flushes)
GC_EVERY_N_FLUSHES = 24

# ---------------------------------------------------------------------------
# RF-04: Null injection parameters
# ---------------------------------------------------------------------------

# Base probability that any individual KPI value is nullified in a given row.
# Real PM systems lose ~0.1-0.5% of values due to node restarts, SNMP
# timeouts, PM file parsing errors, and counter schema changes.
NULL_INJECTION_RATE = 0.0015  # 0.15% per KPI cell

# When one KPI in a row is nullified, the probability that 1-3 additional
# KPIs in the *same row* are also nullified (simulating a partial ROP
# failure that corrupts multiple counters in the same PM file).
NULL_CORRELATION_PROB = 0.40

# Maximum number of additional correlated nulls per row (when triggered).
NULL_CORRELATION_MAX_EXTRA = 3

# ---------------------------------------------------------------------------
# RF-05: Collection gap parameters
# ---------------------------------------------------------------------------

# Probability that a given cell-hour is entirely missing from the output
# (row dropped), simulating PM collection failures, planned maintenance,
# eNB/gNB software restarts, and NMS database failovers.
COLLECTION_GAP_RATE = 0.0008  # 0.08% of cell-hours

# Probability that a gap is "site-wide" — all cells on the same site are
# missing for that hour (e.g. site power outage, site maintenance window).
SITE_WIDE_GAP_PROB = 0.30

# When a site-wide gap occurs, the duration in consecutive hours (burst).
SITE_GAP_BURST_HOURS_MIN = 1
SITE_GAP_BURST_HOURS_MAX = 4

# ---------------------------------------------------------------------------
# RF-15: Rural outlier spike parameters
# ---------------------------------------------------------------------------

# Fraction of rural/deep_rural cells that receive a temporary traffic spike
# in any given hour (simulating festivals, transit corridors, resorts).
RURAL_SPIKE_RATE = 0.001  # 0.1% of rural cells per hour

# Spike multiplier range — pushes traffic above urban P75.
RURAL_SPIKE_MULTIPLIER_MIN = 2.5
RURAL_SPIKE_MULTIPLIER_MAX = 5.0


# ---------------------------------------------------------------------------
# PyArrow schema for kpi_metrics_wide.parquet
#
# 9 metadata columns + 35 KPI columns = 44 total.
# The 8 alias columns from the previous version are REMOVED to save ~30%
# file size.  Downstream code can compute them trivially:
#   throughput_mbps       = dl_throughput_mbps
#   prb_utilization       = (prb_utilization_dl + prb_utilization_ul) / 2
#   traffic_volume        = traffic_volume_gb
#   active_users_count    = active_ue_avg
#   prb_utilization_pct   = prb_utilization  (same as above)
#   latency_ms_alias      = latency_ms
#   data_throughput_gbps  = dl_throughput_mbps / 1000
#   packet_loss_pct_alias = packet_loss_pct
# ---------------------------------------------------------------------------


def _build_output_schema() -> pa.Schema:
    """
    Build the PyArrow schema for the output Parquet file.

    44 columns: 9 metadata + 35 KPI values.
    """
    # Use float32 for all KPI columns: 32-bit precision is more than sufficient
    # for radio KPIs (RSRP to 0.01 dB, throughput to 0.01 Mbps, etc.) and
    # halves the data footprint of the dominant float columns (~90% of file).
    # float32 has ~7 significant decimal digits — plenty for PM counters.
    f = pa.float32()
    return pa.schema(
        [
            # --- Metadata / dimension columns ---
            pa.field("cell_id", pa.string(), nullable=False),
            pa.field("tenant_id", pa.string(), nullable=False),
            pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("rat_type", pa.string(), nullable=False),
            pa.field("band", pa.string()),
            pa.field("site_id", pa.string(), nullable=False),
            pa.field("vendor", pa.string(), nullable=False),
            pa.field("deployment_profile", pa.string(), nullable=False),
            pa.field("is_nsa_scg_leg", pa.bool_(), nullable=False),
            # --- Radio KPIs (35 columns, float32) ---
            pa.field("rsrp_dbm", f),
            pa.field("rsrq_db", f),
            pa.field("sinr_db", f),
            pa.field("cqi_mean", f),
            pa.field("mcs_dl", f),
            pa.field("mcs_ul", f),
            pa.field("dl_bler_pct", f),
            pa.field("ul_bler_pct", f),
            pa.field("prb_utilization_dl", f),
            pa.field("prb_utilization_ul", f),
            pa.field("rach_attempts", f),
            pa.field("rach_success_rate", f),
            pa.field("rrc_setup_attempts", f),
            pa.field("rrc_setup_success_rate", f),
            pa.field("dl_throughput_mbps", f),
            pa.field("ul_throughput_mbps", f),
            pa.field("latency_ms", f),
            pa.field("jitter_ms", f),
            pa.field("packet_loss_pct", f),
            pa.field("active_ue_avg", f),
            pa.field("active_ue_max", f),
            pa.field("traffic_volume_gb", f),
            pa.field("dl_rlc_retransmission_pct", f),
            pa.field("ul_rlc_retransmission_pct", f),
            pa.field("ho_attempt", f),
            pa.field("ho_success_rate", f),
            pa.field("cell_availability_pct", f),
            pa.field("interference_iot_db", f),
            pa.field("paging_discard_rate", f),
            pa.field("cce_utilization_pct", f),
            pa.field("volte_erlangs", f),
            pa.field("csfb_attempts", f),
            pa.field("csfb_success_rate", f),
            pa.field("pdcp_dl_volume_mb", f),
            pa.field("pdcp_ul_volume_mb", f),
        ]
    )


# Ordered list of the 35 KPI column names (must match schema order after the
# 9 metadata columns).
_KPI_COLUMNS: list[str] = [
    "rsrp_dbm",
    "rsrq_db",
    "sinr_db",
    "cqi_mean",
    "mcs_dl",
    "mcs_ul",
    "dl_bler_pct",
    "ul_bler_pct",
    "prb_utilization_dl",
    "prb_utilization_ul",
    "rach_attempts",
    "rach_success_rate",
    "rrc_setup_attempts",
    "rrc_setup_success_rate",
    "dl_throughput_mbps",
    "ul_throughput_mbps",
    "latency_ms",
    "jitter_ms",
    "packet_loss_pct",
    "active_ue_avg",
    "active_ue_max",
    "traffic_volume_gb",
    "dl_rlc_retransmission_pct",
    "ul_rlc_retransmission_pct",
    "ho_attempt",
    "ho_success_rate",
    "cell_availability_pct",
    "interference_iot_db",
    "paging_discard_rate",
    "cce_utilization_pct",
    "volte_erlangs",
    "csfb_attempts",
    "csfb_success_rate",
    "pdcp_dl_volume_mb",
    "pdcp_ul_volume_mb",
]


# ---------------------------------------------------------------------------
# Cell data loader
# ---------------------------------------------------------------------------


def _load_cell_data(config: GeneratorConfig):
    """
    Load cells.parquet from the intermediate directory.

    Uses polars for fast columnar reads, but we immediately extract numpy
    arrays and release the DataFrame to keep memory tight.
    """
    import polars as pl

    cells_path = config.paths.intermediate_dir / "cells.parquet"
    if not cells_path.exists():
        raise FileNotFoundError(f"Cell inventory not found at {cells_path}. Run Step 01 (sites & cells) first.")
    cells = pl.read_parquet(str(cells_path))
    size_mb = cells_path.stat().st_size / 1024 / 1024
    console.print(f"[dim]Loaded {cells.shape[0]:,} cells from {cells_path} ({size_mb:.1f} MB)[/dim]")
    return cells


def _extract_arrays(cells) -> dict[str, np.ndarray]:
    """
    Extract all needed columns as numpy arrays from a Polars DataFrame,
    then allow the DataFrame to be garbage-collected.

    Returns a dict of column_name → numpy array.
    """
    cols = {}
    cols["cell_id"] = cells["cell_id"].to_numpy().astype(str)
    cols["site_id"] = cells["site_id"].to_numpy().astype(str)
    cols["rat_type"] = cells["rat_type"].to_numpy().astype(str)
    cols["band"] = cells["band"].to_numpy().astype(str)
    cols["vendor"] = cells["vendor"].to_numpy().astype(str)
    cols["deployment_profile"] = cells["deployment_profile"].to_numpy().astype(str)
    cols["timezone"] = cells["timezone"].to_numpy().astype(str)
    cols["is_nsa_scg_leg"] = cells["is_nsa_scg_leg"].to_numpy().astype(bool)
    cols["frequency_mhz"] = cells["frequency_mhz"].to_numpy().astype(np.float64)
    cols["bandwidth_mhz"] = cells["bandwidth_mhz"].to_numpy().astype(np.float64)
    cols["max_tx_power_dbm"] = cells["max_tx_power_dbm"].to_numpy().astype(np.float64)
    cols["max_prbs"] = cells["max_prbs"].to_numpy().astype(np.int64)
    cols["antenna_height_m"] = cells["antenna_height_m"].to_numpy().astype(np.float64)
    cols["inter_site_distance_m"] = cells["inter_site_distance_m"].to_numpy().astype(np.float64)
    return cols


def _build_physics_input(cols: dict[str, np.ndarray]) -> CellPhysicsInput:
    """Build the CellPhysicsInput struct from extracted column arrays."""
    return CellPhysicsInput(
        cell_id=cols["cell_id"],
        rat_type=cols["rat_type"],
        band=cols["band"],
        deployment_profile=cols["deployment_profile"],
        freq_mhz=cols["frequency_mhz"],
        bandwidth_mhz=cols["bandwidth_mhz"],
        max_tx_power_dbm=cols["max_tx_power_dbm"],
        max_prbs=cols["max_prbs"],
        antenna_height_m=cols["antenna_height_m"],
        isd_m=cols["inter_site_distance_m"],
        is_nsa_scg_leg=cols["is_nsa_scg_leg"],
    )


# ---------------------------------------------------------------------------
# RecordBatch builder — converts physics output to a PyArrow batch
# ---------------------------------------------------------------------------


def _inject_nulls(
    kpi_dict: dict[str, np.ndarray],
    rng: np.random.Generator,
    n: int,
) -> dict[str, np.ndarray]:
    """
    RF-04: Inject stochastic null (NaN) values into KPI arrays.

    Simulates real-world PM collection imperfections:
      - Node restarts: counter MIBs reset during software upgrades
      - SNMP/SFTP collection timeouts
      - PM file parsing errors (malformed XML, corrupt gzip)
      - Counter schema changes after software upgrades
      - Measurement configuration changes

    Null patterns are *correlated*: when one KPI in a row is nullified,
    there's a 40% chance that 1-3 additional KPIs in the same row are
    also nullified (simulating a partial ROP failure).

    Parameters
    ----------
    kpi_dict : dict of KPI name → numpy array (modified in-place)
    rng : random generator
    n : number of cells (rows)

    Returns
    -------
    kpi_dict with some values replaced by NaN.
    """
    kpi_names = [k for k in _KPI_COLUMNS if k in kpi_dict]
    n_kpis = len(kpi_names)
    if n_kpis == 0:
        return kpi_dict

    # Step 1: Generate base null mask — independent per (row, column)
    null_mask = rng.random(size=(n, n_kpis)) < NULL_INJECTION_RATE

    # Step 2: Correlated nulls — for rows with at least one null, sometimes
    # null out additional columns in the same row (partial ROP failure).
    rows_with_nulls = np.where(null_mask.any(axis=1))[0]
    if len(rows_with_nulls) > 0:
        correlated_trigger = rng.random(size=len(rows_with_nulls)) < NULL_CORRELATION_PROB
        for idx in rows_with_nulls[correlated_trigger]:
            n_extra = rng.integers(1, NULL_CORRELATION_MAX_EXTRA + 1)
            extra_cols = rng.choice(n_kpis, size=min(n_extra, n_kpis), replace=False)
            null_mask[idx, extra_cols] = True

    # Step 3: Apply nulls to each KPI column
    for col_idx, col_name in enumerate(kpi_names):
        col_nulls = null_mask[:, col_idx]
        if col_nulls.any():
            arr = kpi_dict[col_name].copy()
            arr[col_nulls] = np.nan
            kpi_dict[col_name] = arr

    return kpi_dict


def _compute_collection_gap_mask(
    n: int,
    cols: dict[str, np.ndarray],
    hour_idx: int,
    rng: np.random.Generator,
    site_gap_schedule: dict[str, set[int]],
) -> np.ndarray:
    """
    RF-05: Compute a boolean mask of rows to *keep* (True = keep, False = drop).

    Simulates PM collection gaps:
      - Random individual cell-hour drops (SFTP timeout, PM file corruption)
      - Site-wide correlated drops (maintenance window, power outage, NMS
        failover) — all cells on the same site missing for 1-4 consecutive
        hours.

    Parameters
    ----------
    n : number of cells
    cols : cell metadata arrays (need site_id)
    hour_idx : current simulation hour (0-based)
    rng : random generator
    site_gap_schedule : mutable dict mapping site_id → set of hour indices
        where all cells on that site should be dropped.  Pre-populated with
        burst schedules; checked and extended as new gaps are triggered.

    Returns
    -------
    Boolean mask of shape (n,), True = keep row, False = drop row.
    """
    keep_mask = np.ones(n, dtype=bool)
    site_ids = cols["site_id"]

    # 1. Check pre-scheduled site-wide gaps (from previous burst triggers)
    for i in range(n):
        sid = site_ids[i]
        if sid in site_gap_schedule and hour_idx in site_gap_schedule[sid]:
            keep_mask[i] = False

    # 2. Random individual cell-hour drops
    random_drops = rng.random(size=n) < COLLECTION_GAP_RATE
    keep_mask[random_drops] = False

    # 3. Trigger new site-wide burst gaps
    # Pick a small number of sites to start a new maintenance burst this hour
    unique_sites = np.unique(site_ids)
    n_sites = len(unique_sites)
    # Expected: ~0.03% of sites start a burst per hour
    site_burst_prob = COLLECTION_GAP_RATE * SITE_WIDE_GAP_PROB * 5.0
    site_burst_trigger = rng.random(size=n_sites) < site_burst_prob
    triggered_sites = unique_sites[site_burst_trigger]

    for sid in triggered_sites:
        if sid in site_gap_schedule:
            continue  # Already has a scheduled gap, don't double-trigger
        burst_len = rng.integers(SITE_GAP_BURST_HOURS_MIN, SITE_GAP_BURST_HOURS_MAX + 1)
        gap_hours = set(range(hour_idx, hour_idx + burst_len))
        site_gap_schedule[sid] = gap_hours
        # Apply to current hour
        site_mask = site_ids == sid
        keep_mask[site_mask] = False

    return keep_mask


def _apply_rural_spikes(
    kpi_dict: dict[str, np.ndarray],
    cols: dict[str, np.ndarray],
    rng: np.random.Generator,
    n: int,
) -> dict[str, np.ndarray]:
    """
    RF-15 + NC-02: Inject rare traffic spikes into rural/deep_rural cells.

    Simulates festivals, harvest markets, transit corridors (highway cells
    classified as rural), resort areas, and cell-breathing when a neighbour
    fails.  Approximately 0.1% of rural cells per hour receive a spike that
    pushes their traffic above urban P75.

    NC-02 remediation: The original implementation only boosted UE count,
    traffic volume, PRB, RACH, and RRC — but left throughput, latency,
    handover, CCE, and VoLTE unchanged.  In a real festival spike:
      - Aggregate DL throughput increases (more total data) but per-user
        throughput *decreases* (congestion).
      - Latency increases via the hockey-stick effect at high PRB.
      - Handover attempts increase (more mobile UEs).
      - CCE utilisation spikes (more scheduling grants).
      - VoLTE erlangs increase (more voice calls).

    Affected KPIs: active_ue_avg, active_ue_max, traffic_volume_gb,
    prb_utilization_dl, prb_utilization_ul, rach_attempts, rrc_setup_attempts,
    dl_throughput_mbps, ul_throughput_mbps, latency_ms, ho_attempt,
    cce_utilization_pct, volte_erlangs.
    """
    profiles = cols["deployment_profile"]
    is_rural = (profiles == "rural") | (profiles == "deep_rural")
    rural_indices = np.where(is_rural)[0]

    if len(rural_indices) == 0:
        return kpi_dict

    # Select which rural cells get a spike this hour
    spike_trigger = rng.random(size=len(rural_indices)) < RURAL_SPIKE_RATE
    spike_indices = rural_indices[spike_trigger]

    if len(spike_indices) == 0:
        return kpi_dict

    # Generate per-cell spike multiplier
    n_spikes = len(spike_indices)
    multiplier = rng.uniform(RURAL_SPIKE_MULTIPLIER_MIN, RURAL_SPIKE_MULTIPLIER_MAX, size=n_spikes)

    # Apply spike to load-dependent KPIs
    for kpi_name in ("active_ue_avg", "active_ue_max", "traffic_volume_gb", "rach_attempts", "rrc_setup_attempts"):
        if kpi_name in kpi_dict:
            arr = kpi_dict[kpi_name].copy()
            arr[spike_indices] *= multiplier
            kpi_dict[kpi_name] = arr

    # PRB utilisation: boost but cap at 99%
    for kpi_name in ("prb_utilization_dl", "prb_utilization_ul"):
        if kpi_name in kpi_dict:
            arr = kpi_dict[kpi_name].copy()
            arr[spike_indices] = np.minimum(arr[spike_indices] * multiplier * 0.6, 99.0)
            kpi_dict[kpi_name] = arr

    # NC-02: Aggregate throughput increases (more total data served), but
    # the increase is sub-linear because congestion limits per-user rates.
    # A 4× UE spike might only yield 1.5–2× aggregate throughput because
    # the scheduler is resource-constrained.  sqrt(multiplier) gives ~1.6–2.2×.
    if "dl_throughput_mbps" in kpi_dict:
        arr = kpi_dict["dl_throughput_mbps"].copy()
        arr[spike_indices] *= np.sqrt(multiplier)
        kpi_dict["dl_throughput_mbps"] = arr
    if "ul_throughput_mbps" in kpi_dict:
        arr = kpi_dict["ul_throughput_mbps"].copy()
        arr[spike_indices] *= np.sqrt(multiplier) * 0.8  # UL benefits less
        kpi_dict["ul_throughput_mbps"] = arr

    # NC-02: Latency increases — at high PRB the hockey-stick kicks in.
    # Model a proportional latency penalty: more UEs = longer queues.
    # Scale by multiplier^0.7 (sub-linear, but significant).
    if "latency_ms" in kpi_dict:
        arr = kpi_dict["latency_ms"].copy()
        arr[spike_indices] *= np.power(multiplier, 0.7)
        kpi_dict["latency_ms"] = arr

    # NC-02: Handover attempts increase — more mobile UEs at a festival.
    if "ho_attempt" in kpi_dict:
        arr = kpi_dict["ho_attempt"].copy()
        arr[spike_indices] *= multiplier * 0.8  # HO scales ~linearly with UEs
        kpi_dict["ho_attempt"] = arr

    # NC-02: CCE utilisation spikes — more scheduling grants needed.
    if "cce_utilization_pct" in kpi_dict:
        arr = kpi_dict["cce_utilization_pct"].copy()
        arr[spike_indices] = np.minimum(arr[spike_indices] * multiplier * 0.5, 100.0)
        kpi_dict["cce_utilization_pct"] = arr

    # NC-02: VoLTE erlangs increase — more voice calls at a festival.
    if "volte_erlangs" in kpi_dict:
        arr = kpi_dict["volte_erlangs"].copy()
        arr[spike_indices] *= multiplier * 0.6  # Voice scales sub-linearly
        kpi_dict["volte_erlangs"] = arr

    return kpi_dict


def _build_record_batch(
    kpi_dict: dict[str, np.ndarray],
    cols: dict[str, np.ndarray],
    tenant_id: str,
    timestamp: datetime,
    schema: pa.Schema,
    n: int,
    keep_mask: np.ndarray | None = None,
) -> pa.RecordBatch:
    """
    Combine cell metadata + physics KPIs into a PyArrow RecordBatch.

    Called once per hourly interval; produces n rows (one per cell),
    optionally filtered by ``keep_mask`` (RF-05 collection gaps).

    Parameters
    ----------
    keep_mask : optional boolean array — if provided, only rows where
        keep_mask[i] is True are included in the output batch.
    """
    # Timestamp array (same value repeated n times)
    ts_arr = pa.array([timestamp] * n, type=pa.timestamp("us", tz="UTC"))

    # 9 metadata arrays
    arrays: list[pa.Array] = [
        pa.array(cols["cell_id"], type=pa.string()),
        pa.array([tenant_id] * n, type=pa.string()),
        ts_arr,
        pa.array(cols["rat_type"], type=pa.string()),
        pa.array(cols["band"], type=pa.string()),
        pa.array(cols["site_id"], type=pa.string()),
        pa.array(cols["vendor"], type=pa.string()),
        pa.array(cols["deployment_profile"], type=pa.string()),
        pa.array(cols["is_nsa_scg_leg"], type=pa.bool_()),
    ]

    # 35 KPI arrays (in schema order, cast to float32 for compactness)
    nan_fallback = np.full(n, np.nan, dtype=np.float32)
    f32 = pa.float32()
    for col_name in _KPI_COLUMNS:
        arr = kpi_dict.get(col_name)
        if arr is not None:
            arrays.append(pa.array(arr.astype(np.float32), type=f32))
        else:
            arrays.append(pa.array(nan_fallback, type=f32))

    batch = pa.RecordBatch.from_arrays(arrays, schema=schema)

    # RF-05: Drop rows for simulated PM collection gaps
    if keep_mask is not None and not keep_mask.all():
        keep_indices = np.where(keep_mask)[0]
        if len(keep_indices) == 0:
            # Entire hour dropped (extremely rare) — return empty batch
            return batch.slice(0, 0)
        batch = batch.take(pa.array(keep_indices, type=pa.int32()))

    return batch


# ---------------------------------------------------------------------------
# Batched physics runner (for n_cells > MAX_CELLS_PER_BATCH)
# ---------------------------------------------------------------------------


def _run_batched_physics(
    physics_input: CellPhysicsInput,
    conditions: HourlyConditions,
    rng: np.random.Generator,
    n_cells: int,
    app_mix_state: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Run physics in sub-batches when n_cells exceeds MAX_CELLS_PER_BATCH.
    Splits input arrays, runs physics on each chunk, concatenates results.

    NC-03: Threads per-batch slices of ``app_mix_state`` through so each
    sub-batch gets its own AR(1) state segment and the reassembled output
    contains the full ``_app_mix_state`` vector for the next hour.
    """
    results: dict[str, list[np.ndarray]] = {}
    batch_size = MAX_CELLS_PER_BATCH

    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        s = slice(start, end)

        sub_input = CellPhysicsInput(
            cell_id=physics_input.cell_id[s],
            rat_type=physics_input.rat_type[s],
            band=physics_input.band[s],
            deployment_profile=physics_input.deployment_profile[s],
            freq_mhz=physics_input.freq_mhz[s],
            bandwidth_mhz=physics_input.bandwidth_mhz[s],
            max_tx_power_dbm=physics_input.max_tx_power_dbm[s],
            max_prbs=physics_input.max_prbs[s],
            antenna_height_m=physics_input.antenna_height_m[s],
            isd_m=physics_input.isd_m[s],
            is_nsa_scg_leg=physics_input.is_nsa_scg_leg[s],
        )
        sub_conditions = HourlyConditions(
            load_factor=conditions.load_factor[s],
            shadow_fading_db=conditions.shadow_fading_db[s],
            interference_delta_db=conditions.interference_delta_db[s],
            active_ue_multiplier=conditions.active_ue_multiplier[s],
        )
        sub_app_mix = app_mix_state[s] if app_mix_state is not None else None
        sub_kpis = compute_cell_kpis_vectorised(sub_input, sub_conditions, rng, app_mix_state=sub_app_mix)
        for key, arr in sub_kpis.items():
            results.setdefault(key, []).append(arr)

    return {key: np.concatenate(arrs) for key, arrs in results.items()}


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------


def _sanity_check(output_path: Path, expected_rows: int) -> None:
    """Quick validation of the written Parquet file."""
    meta = pq.read_metadata(str(output_path))

    # RF-05: With collection gaps, actual rows will be slightly fewer than
    # the theoretical maximum (n_cells × n_hours).  Allow up to 1% fewer.
    row_diff = expected_rows - meta.num_rows
    gap_pct = row_diff / expected_rows * 100.0 if expected_rows > 0 else 0.0
    if row_diff < 0 or gap_pct > 1.0:
        console.print(
            f"[bold yellow]⚠ Row count unexpected:[/bold yellow] "
            f"expected ~{expected_rows:,} (minus gaps), got {meta.num_rows:,} "
            f"(diff: {row_diff:,}, {gap_pct:.3f}%)"
        )
    else:
        console.print(
            f"[dim]Sanity check passed: {meta.num_rows:,} rows "
            f"({row_diff:,} dropped as collection gaps, {gap_pct:.3f}%), "
            f"{meta.num_columns} columns, "
            f"{meta.num_row_groups} row groups[/dim]"
        )

    # Spot-check value ranges on a small sample
    try:
        sample_cols = [
            "rsrp_dbm",
            "sinr_db",
            "cqi_mean",
            "dl_throughput_mbps",
            "prb_utilization_dl",
            "cell_availability_pct",
            "active_ue_avg",
            "latency_ms",
        ]
        sample = pq.read_table(
            str(output_path),
            columns=sample_cols,
        ).slice(0, min(1000, expected_rows))

        # RF-08: Widened SINR range to [-20, 50], RSRP ceiling to -30 dBm
        checks = {
            "rsrp_dbm": (-140.0, -30.0),
            "sinr_db": (-20.0, 50.0),
            "cqi_mean": (0.0, 15.0),
            "cell_availability_pct": (0.0, 100.0),
        }
        issues = []
        for col_name, (lo, hi) in checks.items():
            vals = sample.column(col_name).to_pylist()
            # RF-04: Skip None/NaN values (injected nulls are expected)
            if any(v is not None and not (v != v) and (v < lo or v > hi) for v in vals):
                issues.append(f"{col_name} out of [{lo}, {hi}]")

        if issues:
            for issue in issues:
                console.print(f"[bold yellow]⚠ Range check: {issue}[/bold yellow]")
        else:
            console.print("[dim]Range spot-check passed (1000-row sample, nulls expected)[/dim]")
    except Exception as e:
        console.print(f"[yellow]Spot-check skipped: {e}[/yellow]")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_radio_kpis(config: GeneratorConfig) -> None:
    """
    Generate radio-layer KPIs for all cells across the full simulation period.

    This is the Step 03 entry point called by the CLI pipeline.

    Memory budget:
      - Cell metadata arrays (string + numeric): ~50 MB for 66k cells
      - CellPhysicsInput (numeric views): ~10 MB
      - StreamingEnvironmentGenerator state: ~5 MB
      - Per-hour: HourlyEnvironment ~2 MB + physics temps ~20 MB + RecordBatch ~30 MB
      - PyArrow writer buffer: ~30 MB (flushed every hour)
      - Peak total: ~150-200 MB

    Steps:
    1. Load cell inventory; extract numpy arrays; release DataFrame.
    2. Create StreamingEnvironmentGenerator (tiny state vectors).
    3. Create Parquet writer with zstd-9 compression.
    4. For each of 720 hours:
       a. Stream one hour of environmental conditions (~2 MB).
       b. Run vectorised physics chain (~20 MB temporary).
       b2. RF-15: Inject rare rural traffic spikes.
       b3. RF-04: Inject stochastic null values (~0.15% per KPI).
       c. RF-05: Compute collection gap mask (drop ~0.08% of rows).
       d. Build RecordBatch (with gap filtering) (~30 MB).
       e. Flush to Parquet as one row group; release batch.
       f. Periodic gc.collect().
    5. Close writer; print summary; sanity check.
    """
    step_start = time.time()

    # ── 0. Derive deterministic seed ─────────────────────────
    seed = config.seed_for("step_03_radio_kpis")
    rng = np.random.default_rng(seed)
    console.print(f"[dim]Step 03 seed: {seed}[/dim]")

    # ── 1. Load cell inventory ───────────────────────────────
    cells_df = _load_cell_data(config)
    n_cells = cells_df.shape[0]
    total_hours = config.simulation.total_intervals
    total_rows = n_cells * total_hours

    console.print(
        f"[bold]Generating radio KPIs:[/bold] {n_cells:,} cells × {total_hours:,} hours = {total_rows:,} rows"
    )

    # Extract numpy arrays and release the Polars DataFrame
    cols = _extract_arrays(cells_df)
    del cells_df
    gc.collect()

    physics_input = _build_physics_input(cols)

    console.print(
        f"[dim]Cell arrays extracted. Metadata: ~{sum(a.nbytes for a in cols.values()) / 1024 / 1024:.0f} MB[/dim]"
    )

    # ── 2. Create streaming environment generator ────────────
    console.print("[dim]Initialising streaming environment generator (AR(1) state vectors)...[/dim]")
    env_init_start = time.time()

    profile_config = TrafficProfileConfig(
        simulation_days=config.simulation.simulation_days,
        start_day_of_week=0,  # Simulation starts on Monday
    )

    env_gen = StreamingEnvironmentGenerator(
        n_cells=n_cells,
        deployment_profiles=cols["deployment_profile"],
        timezones=cols["timezone"],
        simulation_days=config.simulation.simulation_days,
        rng=rng,
        profile_config=profile_config,
    )
    env_init_elapsed = time.time() - env_init_start
    console.print(
        f"[dim]Environment generator ready in {env_init_elapsed:.2f}s "
        f"(state: ~{n_cells * 8 * 4 / 1024 / 1024:.1f} MB)[/dim]"
    )

    # ── 3. Set up Parquet writer ─────────────────────────────
    output_path = config.paths.output_dir / "kpi_metrics_wide.parquet"
    schema = _build_output_schema()

    writer = pq.ParquetWriter(
        str(output_path),
        schema=schema,
        compression="zstd",
        compression_level=9,  # Higher compression than before (was 3)
        use_dictionary=True,  # Dictionary-encode repeated strings
        write_statistics=True,
        version="2.6",
    )

    # ── 4. Main generation loop (streaming, 1 hour at a time) ─
    console.print(
        f"[bold cyan]Starting KPI generation: {total_hours} hourly intervals, flush-per-hour, zstd-9...[/bold cyan]"
    )

    rows_written = 0
    rows_dropped = 0
    row_groups_written = 0
    gen_start = time.time()

    # RF-05: Site-wide gap schedule — tracks which sites have active
    # maintenance bursts and for which hour indices rows should be dropped.
    site_gap_schedule: dict[str, set[int]] = {}

    # NC-03: Application-mix AR(1) state — threaded across hours so that
    # each cell's traffic-volume/throughput ratio is temporally persistent
    # (ρ ≈ 0.70), matching the observed autocorrelation in real networks.
    app_mix_state: np.ndarray | None = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("({task.fields[rows_so_far]} rows)"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=1,  # Low refresh rate to reduce overhead
    ) as progress:
        task = progress.add_task(
            "Generating KPIs",
            total=total_hours,
            rows_so_far="0",
        )

        for hour_idx in range(total_hours):
            # Timestamp for this interval
            timestamp = SIMULATION_EPOCH + timedelta(hours=hour_idx)

            # ── 4a. Stream one hour of environmental conditions ───
            env_hour = env_gen.next_hour()

            conditions = HourlyConditions(
                load_factor=env_hour.load_factor,
                shadow_fading_db=env_hour.shadow_fading_db,
                interference_delta_db=env_hour.interference_delta_db,
                active_ue_multiplier=env_hour.active_ue_multiplier,
            )

            # ── 4b. Per-hour sub-seed for within-hour reproducibility
            hour_rng = np.random.default_rng(rng.integers(0, 2**31) + hour_idx)

            # ── 4c. Run physics chain (vectorised across all cells) ─
            if n_cells <= MAX_CELLS_PER_BATCH:
                kpi_dict = compute_cell_kpis_vectorised(
                    physics_input, conditions, hour_rng, app_mix_state=app_mix_state
                )
            else:
                kpi_dict = _run_batched_physics(
                    physics_input, conditions, hour_rng, n_cells, app_mix_state=app_mix_state
                )

            # NC-03: Extract and carry forward the app-mix AR(1) state
            app_mix_state = kpi_dict.pop("_app_mix_state", None)

            # ── 4c2. RF-15: Inject rare rural traffic spikes ─────
            kpi_dict = _apply_rural_spikes(kpi_dict, cols, hour_rng, n_cells)

            # ── 4c3. RF-04: Inject stochastic null values ────────
            kpi_dict = _inject_nulls(kpi_dict, hour_rng, n_cells)

            # ── 4c4. RF-05: Compute PM collection gap mask ───────
            keep_mask = _compute_collection_gap_mask(
                n_cells,
                cols,
                hour_idx,
                hour_rng,
                site_gap_schedule,
            )

            # ── 4d. Build RecordBatch and flush immediately ──────
            batch = _build_record_batch(
                kpi_dict=kpi_dict,
                cols=cols,
                tenant_id=config.tenant_id,
                timestamp=timestamp,
                schema=schema,
                n=n_cells,
                keep_mask=keep_mask,
            )

            # Write as a single row group and release
            batch_rows = batch.num_rows
            table = pa.Table.from_batches([batch])
            writer.write_table(table)
            row_groups_written += 1
            rows_written += batch_rows
            rows_dropped += n_cells - batch_rows

            # Explicitly release references to per-hour data
            del env_hour, conditions, kpi_dict, batch, table, keep_mask

            # ── 4e. Periodic GC ──────────────────────────────────
            if row_groups_written % GC_EVERY_N_FLUSHES == 0:
                gc.collect()

            progress.update(
                task,
                advance=1,
                rows_so_far=f"{rows_written:,}",
            )

    # ── 5. Close writer ──────────────────────────────────────
    writer.close()
    gen_elapsed = time.time() - gen_start

    # Release remaining large objects
    del physics_input, cols, env_gen
    gc.collect()

    # ── 6. Report results ────────────────────────────────────
    output_size_mb = output_path.stat().st_size / 1024 / 1024
    total_elapsed = time.time() - step_start

    console.print()

    summary = Table(
        title="Step 03: Radio KPI Generation — Summary",
        show_header=True,
    )
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")
    summary.add_row("Cells", f"{n_cells:,}")
    summary.add_row("Hours", f"{total_hours:,}")
    summary.add_row("Total rows written", f"{rows_written:,}")
    summary.add_row("Rows dropped (RF-05 gaps)", f"{rows_dropped:,}")
    summary.add_row(
        "Collection gap rate",
        f"{rows_dropped / (rows_written + rows_dropped) * 100:.3f}%" if (rows_written + rows_dropped) > 0 else "N/A",
    )
    summary.add_row("Row groups", f"{row_groups_written:,}")
    summary.add_row("Columns", f"{len(schema)}")
    summary.add_row("Output file", str(output_path))
    summary.add_row("Output size", f"{output_size_mb:.1f} MB")
    summary.add_row("Env init time", f"{env_init_elapsed:.2f}s")
    summary.add_row("KPI generation time", f"{gen_elapsed:.1f}s")
    summary.add_row("Total step time", f"{total_elapsed:.1f}s")
    summary.add_row(
        "Throughput",
        f"{rows_written / gen_elapsed:,.0f} rows/s" if gen_elapsed > 0 else "N/A",
    )
    summary.add_row(
        "Bytes/row",
        f"{output_path.stat().st_size / rows_written:.1f}" if rows_written > 0 else "N/A",
    )
    console.print(summary)

    # ── 7. Quick sanity check ────────────────────────────────
    _sanity_check(output_path, rows_written + rows_dropped)

    console.print(
        f"\n[bold green]✓ Step 03 complete.[/bold green] "
        f"Wrote {rows_written:,} rows ({output_size_mb:.1f} MB) "
        f"to {output_path}"
    )
