"""
Step 04: Multi-Domain KPI Generation.

Generates hourly KPIs for all non-radio domains:
  1. Transport     — PE_ROUTER, AGG_SWITCH, MICROWAVE_LINK, FIBRE_CABLE, DWDM, LSP, L3VPN
  2. Fixed BB      — OLT, PON_PORT
  3. Enterprise    — ETHERNET_CIRCUIT, L3VPN (enterprise instances)
  4. Core Network  — MME, SGW, PGW, AMF, SMF, UPF, HSS, UDM, NSSF, P-CSCF, S-CSCF, BNG, etc.
  5. Power/Env     — per-site power, battery, temperature, humidity, cooling

Architecture:
  Mirrors Phase 3's streaming approach — one hour at a time, flush each
  row-group immediately, explicit GC, peak memory < 200 MB per domain.

  Each domain uses a lightweight AR(1) state machine for temporal
  correlation (utilisation, latency, temperature, etc.) and diurnal
  profiles where appropriate.

  KPI columns use float32 to halve storage cost (matching Phase 3).

Entity discovery:
  Reads ``ground_truth_entities.parquet`` from Phase 2 output and filters
  by entity_type to determine which entities get KPI streams.  This ensures
  row counts are driven by *actual* topology, not config estimates.

Output files:
  - output/transport_kpis_wide.parquet
  - output/fixed_broadband_kpis_wide.parquet
  - output/enterprise_circuit_kpis_wide.parquet
  - output/core_element_kpis_wide.parquet
  - output/power_environment_kpis.parquet
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
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

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIMULATION_EPOCH = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
GC_EVERY_N_FLUSHES = 24
F32 = pa.float32()

# Timezone UTC offsets for Indonesian timezones
_TZ_OFFSET = {"WIB": 7, "WITA": 8, "WIT": 9}


# ---------------------------------------------------------------------------
# AR(1) state helper (identical pattern to Phase 3)
# ---------------------------------------------------------------------------


@dataclass
class _AR1:
    """Mutable AR(1) state: x[t] = rho * x[t-1] + N(0, innov_std)."""

    value: np.ndarray  # (n,)
    rho: float
    innov_std: float | np.ndarray

    def advance(self, rng: np.random.Generator) -> np.ndarray:
        self.value = self.rho * self.value + rng.normal(0.0, self.innov_std, size=self.value.shape)
        return self.value


# ---------------------------------------------------------------------------
# Diurnal profile helpers
# ---------------------------------------------------------------------------

# Simple 24-hour traffic profile (weekday).  Values are fraction of peak.
_DIURNAL_WEEKDAY = np.array(
    [
        0.15,
        0.10,
        0.08,
        0.07,
        0.07,
        0.10,
        0.20,
        0.45,  # 00-07
        0.70,
        0.85,
        0.90,
        0.92,
        0.88,
        0.90,
        0.93,
        0.95,  # 08-15
        0.90,
        0.85,
        0.80,
        0.75,
        0.65,
        0.50,
        0.35,
        0.22,  # 16-23
    ],
    dtype=np.float64,
)

_DIURNAL_WEEKEND = np.array(
    [
        0.18,
        0.12,
        0.09,
        0.08,
        0.08,
        0.09,
        0.15,
        0.30,  # 00-07
        0.50,
        0.60,
        0.65,
        0.68,
        0.70,
        0.68,
        0.65,
        0.60,  # 08-15
        0.55,
        0.52,
        0.50,
        0.48,
        0.42,
        0.35,
        0.28,
        0.20,  # 16-23
    ],
    dtype=np.float64,
)

# Power-specific: temperature follows a different diurnal curve (warmer midday)
_TEMP_DIURNAL = np.array(
    [
        0.40,
        0.35,
        0.32,
        0.30,
        0.30,
        0.32,
        0.38,
        0.50,  # 00-07
        0.65,
        0.78,
        0.88,
        0.95,
        1.00,
        0.98,
        0.95,
        0.90,  # 08-15
        0.82,
        0.75,
        0.68,
        0.60,
        0.55,
        0.50,
        0.46,
        0.42,  # 16-23
    ],
    dtype=np.float64,
)


def _diurnal_factor(hour_idx: int, start_dow: int = 0) -> float:
    """Return a scalar diurnal factor for the given hour index."""
    day = hour_idx // 24
    hour = hour_idx % 24
    dow = (start_dow + day) % 7
    profile = _DIURNAL_WEEKEND if dow >= 5 else _DIURNAL_WEEKDAY
    return float(profile[hour])


def _temp_diurnal_factor(hour_idx: int) -> float:
    """Return a scalar temperature diurnal factor."""
    hour = hour_idx % 24
    return float(_TEMP_DIURNAL[hour])


def _diurnal_factors_vec(
    hour_idx: int,
    utc_offsets: np.ndarray,
    start_dow: int = 0,
) -> np.ndarray:
    """Return per-entity diurnal factors accounting for timezone offsets."""
    n = len(utc_offsets)
    result = np.empty(n, dtype=np.float64)
    day = hour_idx // 24
    base_hour = hour_idx % 24
    for tz_off in np.unique(utc_offsets):
        mask = utc_offsets == tz_off
        local_hour = (base_hour + int(tz_off)) % 24
        dow = (start_dow + day) % 7
        profile = _DIURNAL_WEEKEND if dow >= 5 else _DIURNAL_WEEKDAY
        result[mask] = profile[local_hour]
    return result


# ---------------------------------------------------------------------------
# Entity loader — reads Phase 2 ground_truth_entities.parquet
# ---------------------------------------------------------------------------


def _load_entities(
    config: GeneratorConfig,
    entity_types: list[str],
    columns: list[str] | None = None,
) -> pl.DataFrame:
    """Load entities of specified types from Phase 2 output."""
    path = config.paths.output_dir / "ground_truth_entities.parquet"
    if not path.exists():
        raise FileNotFoundError(f"ground_truth_entities.parquet not found at {path}. Run Phase 2 first.")
    # Read only needed columns for memory efficiency
    read_cols = ["entity_id", "entity_type", "tenant_id"]
    if columns:
        read_cols = list(set(read_cols + columns))

    df = pl.read_parquet(str(path), columns=read_cols)
    df = df.filter(pl.col("entity_type").is_in(entity_types))
    return df


def _load_sites(config: GeneratorConfig) -> pl.DataFrame:
    """Load site inventory from Phase 1 intermediate."""
    path = config.paths.intermediate_dir / "sites.parquet"
    if not path.exists():
        raise FileNotFoundError(f"sites.parquet not found at {path}. Run Phase 1 first.")
    return pl.read_parquet(str(path))


# ---------------------------------------------------------------------------
# Parquet writer helper
# ---------------------------------------------------------------------------


def _make_writer(path: Path, schema: pa.Schema) -> pq.ParquetWriter:
    return pq.ParquetWriter(
        str(path),
        schema=schema,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )


def _write_hour(writer: pq.ParquetWriter, batch: pa.RecordBatch) -> None:
    table = pa.Table.from_batches([batch])
    writer.write_table(table)
    del table


# ---------------------------------------------------------------------------
# Shared KPI computation helpers
# ---------------------------------------------------------------------------


def _clip(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(arr, lo, hi)


def _pct(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 100.0)


def _pos(arr: np.ndarray) -> np.ndarray:
    return np.maximum(arr, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
# 1. TRANSPORT DOMAIN KPIs
# ═══════════════════════════════════════════════════════════════════════════

# Which entity types get transport KPI rows
TRANSPORT_KPI_ENTITY_TYPES = [
    "PE_ROUTER",
    "AGGREGATION_SWITCH",
    "MICROWAVE_LINK",
    "FIBRE_CABLE",
    "DWDM_SYSTEM",
    "LSP",
    "L3VPN",
    "BNG",
    "ACCESS_SWITCH",
]

TRANSPORT_SCHEMA = pa.schema(
    [
        pa.field("entity_id", pa.string(), nullable=False),
        pa.field("tenant_id", pa.string(), nullable=False),
        pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("entity_type", pa.string(), nullable=False),
        pa.field("site_id", pa.string()),
        # 15 KPI columns
        pa.field("interface_utilization_in_pct", F32),
        pa.field("interface_utilization_out_pct", F32),
        pa.field("interface_errors_in", F32),
        pa.field("interface_errors_out", F32),
        pa.field("interface_discards_in", F32),
        pa.field("interface_discards_out", F32),
        pa.field("optical_rx_power_dbm", F32),
        pa.field("optical_snr_db", F32),
        pa.field("lsp_utilization_pct", F32),
        pa.field("lsp_latency_ms", F32),
        pa.field("bgp_prefixes_received", F32),
        pa.field("bgp_session_flaps", F32),
        pa.field("microwave_modulation", pa.string()),
        pa.field("microwave_capacity_mbps", F32),
        pa.field("microwave_availability_pct", F32),
    ]
)

_MW_MODULATIONS = ["QPSK", "16QAM", "32QAM", "64QAM", "128QAM", "256QAM", "1024QAM", "2048QAM", "4096QAM"]

# Baseline parameter ranges per entity type
_TRANSPORT_BASELINES: dict[str, dict[str, tuple[float, float]]] = {
    "PE_ROUTER": {
        "util_base": (30.0, 70.0),
        "optical_rx": (-12.0, -3.0),
        "optical_snr": (18.0, 32.0),
        "bgp_prefixes": (5000.0, 50000.0),
    },
    "AGGREGATION_SWITCH": {
        "util_base": (20.0, 60.0),
        "optical_rx": (-15.0, -5.0),
        "optical_snr": (15.0, 28.0),
        "bgp_prefixes": (100.0, 2000.0),
    },
    "MICROWAVE_LINK": {
        "util_base": (25.0, 75.0),
        "mw_capacity": (500.0, 4000.0),
    },
    "ACCESS_SWITCH": {
        "util_base": (15.0, 50.0),
        "optical_rx": (-18.0, -8.0),
        "optical_snr": (12.0, 25.0),
        "bgp_prefixes": (10.0, 100.0),
    },
    "FIBRE_CABLE": {
        "util_base": (10.0, 40.0),
        "optical_rx": (-8.0, -1.0),
        "optical_snr": (22.0, 38.0),
    },
    "DWDM_SYSTEM": {
        "util_base": (20.0, 60.0),
        "optical_rx": (-6.0, -1.0),
        "optical_snr": (25.0, 38.0),
    },
    "LSP": {
        "util_base": (15.0, 55.0),
        "lsp_latency_base": (2.0, 15.0),
    },
    "L3VPN": {
        "util_base": (10.0, 50.0),
        "bgp_prefixes": (50.0, 5000.0),
    },
    "BNG": {
        "util_base": (25.0, 65.0),
        "optical_rx": (-10.0, -2.0),
        "optical_snr": (20.0, 35.0),
        "bgp_prefixes": (1000.0, 20000.0),
    },
}


@dataclass
class _TransportState:
    """Mutable state for transport domain streaming."""

    n: int
    entity_ids: np.ndarray
    entity_types: np.ndarray
    site_ids: np.ndarray
    tenant_id: str
    # Per-entity timezone offset (hours from UTC) — RF-11 remediation:
    # transport entities inherit the timezone of their associated site so
    # that per-region diurnal patterns are applied instead of a single
    # national curve.
    utc_offsets: np.ndarray
    # Per-entity baselines (computed once)
    util_base_in: np.ndarray
    util_base_out: np.ndarray
    optical_rx_base: np.ndarray
    optical_snr_base: np.ndarray
    bgp_prefix_base: np.ndarray
    lsp_latency_base: np.ndarray
    mw_capacity_base: np.ndarray
    is_mw: np.ndarray
    is_lsp: np.ndarray
    is_optical: np.ndarray
    # AR(1) states
    util_ar: _AR1
    latency_ar: _AR1


def _init_transport(config: GeneratorConfig, rng: np.random.Generator) -> _TransportState | None:
    """Load transport entities and initialise state."""
    df = _load_entities(
        config,
        TRANSPORT_KPI_ENTITY_TYPES,
        columns=["site_id", "properties_json"],
    )
    if df.height == 0:
        console.print("    [yellow]No transport KPI entities found — skipping.[/yellow]")
        return None

    n = df.height
    entity_ids = df["entity_id"].to_numpy().astype(str)
    entity_types = df["entity_type"].to_numpy().astype(str)
    site_ids = df["site_id"].fill_null("").to_numpy().astype(str)

    # RF-11: Resolve per-entity UTC offset by joining site_id → site timezone.
    # Transport links serve specific geographic regions, so their diurnal
    # utilisation pattern must follow the local timezone of the site they
    # are attached to (e.g. a PE_ROUTER in Papua/WIT peaks ~2h before one
    # in Sumatra/WIB).
    try:
        sites_df = _load_sites(config)
        site_tz_map: dict[str, int] = {}
        for row in sites_df.select(["site_id", "timezone"]).iter_rows():
            site_tz_map[str(row[0])] = _TZ_OFFSET.get(str(row[1]), 7)
        del sites_df
        gc.collect()
    except FileNotFoundError:
        site_tz_map = {}

    utc_offsets = np.array(
        [site_tz_map.get(sid, 7) for sid in site_ids],  # default WIB (UTC+7)
        dtype=np.int32,
    )

    # Compute per-entity baselines
    util_base_in = np.empty(n, dtype=np.float64)
    util_base_out = np.empty(n, dtype=np.float64)
    optical_rx_base = np.full(n, -10.0, dtype=np.float64)
    optical_snr_base = np.full(n, 25.0, dtype=np.float64)
    bgp_prefix_base = np.full(n, 500.0, dtype=np.float64)
    lsp_latency_base = np.full(n, 5.0, dtype=np.float64)
    mw_capacity_base = np.full(n, 1000.0, dtype=np.float64)

    for i in range(n):
        etype = entity_types[i]
        bl = _TRANSPORT_BASELINES.get(etype, _TRANSPORT_BASELINES["AGGREGATION_SWITCH"])
        lo, hi = bl.get("util_base", (15.0, 50.0))
        util_base_in[i] = rng.uniform(lo, hi)
        util_base_out[i] = rng.uniform(lo, hi)
        if "optical_rx" in bl:
            lo_o, hi_o = bl["optical_rx"]
            optical_rx_base[i] = rng.uniform(lo_o, hi_o)
        if "optical_snr" in bl:
            lo_s, hi_s = bl["optical_snr"]
            optical_snr_base[i] = rng.uniform(lo_s, hi_s)
        if "bgp_prefixes" in bl:
            lo_b, hi_b = bl["bgp_prefixes"]
            bgp_prefix_base[i] = rng.uniform(lo_b, hi_b)
        if "lsp_latency_base" in bl:
            lo_l, hi_l = bl["lsp_latency_base"]
            lsp_latency_base[i] = rng.uniform(lo_l, hi_l)
        if "mw_capacity" in bl:
            lo_m, hi_m = bl["mw_capacity"]
            mw_capacity_base[i] = rng.uniform(lo_m, hi_m)

    is_mw = entity_types == "MICROWAVE_LINK"
    is_lsp = (entity_types == "LSP") | (entity_types == "L3VPN")
    is_optical = ~is_mw & ~(entity_types == "L3VPN")

    del df
    gc.collect()

    return _TransportState(
        n=n,
        entity_ids=entity_ids,
        entity_types=entity_types,
        site_ids=site_ids,
        tenant_id=config.tenant_id,
        utc_offsets=utc_offsets,
        util_base_in=util_base_in,
        util_base_out=util_base_out,
        optical_rx_base=optical_rx_base,
        optical_snr_base=optical_snr_base,
        bgp_prefix_base=bgp_prefix_base,
        lsp_latency_base=lsp_latency_base,
        mw_capacity_base=mw_capacity_base,
        is_mw=is_mw,
        is_lsp=is_lsp,
        is_optical=is_optical,
        util_ar=_AR1(
            value=rng.normal(0.0, 3.0, size=n),
            rho=0.80,
            innov_std=3.0 * np.sqrt(1 - 0.80**2),
        ),
        latency_ar=_AR1(
            value=rng.normal(0.0, 0.5, size=n),
            rho=0.70,
            innov_std=0.5 * np.sqrt(1 - 0.70**2),
        ),
    )


def _transport_hour(
    state: _TransportState,
    hour_idx: int,
    rng: np.random.Generator,
    timestamp: datetime,
) -> pa.RecordBatch:
    """Generate one hour of transport KPIs."""
    n = state.n

    # RF-11: Per-entity diurnal factor using site timezone instead of a
    # single national curve.  A PE_ROUTER in Papua (WIT, UTC+9) peaks
    # ~2 hours before one in Sumatra (WIB, UTC+7).
    diurnal_vec = _diurnal_factors_vec(hour_idx, state.utc_offsets)

    # Advance AR(1) states
    util_jitter = state.util_ar.advance(rng)
    lat_jitter = state.latency_ar.advance(rng)

    # Utilisation
    util_in = _pct(state.util_base_in * diurnal_vec + util_jitter + rng.normal(0, 1.5, n))
    util_out = _pct(state.util_base_out * diurnal_vec + util_jitter * 0.8 + rng.normal(0, 1.5, n))

    # Interface errors/discards — correlated with utilisation
    err_rate = np.where(util_in > 80, 0.005, 0.0005)
    errors_in = _pos(rng.poisson(err_rate * 100, n).astype(np.float32))
    errors_out = _pos(rng.poisson(err_rate * 80, n).astype(np.float32))
    disc_rate = np.where(util_in > 85, 0.01, 0.001)
    discards_in = _pos(rng.poisson(disc_rate * 50, n).astype(np.float32))
    discards_out = _pos(rng.poisson(disc_rate * 40, n).astype(np.float32))

    # Optical metrics (NaN for non-optical entities)
    optical_rx = np.where(
        state.is_optical,
        state.optical_rx_base + rng.normal(0, 0.3, n),
        np.nan,
    )
    optical_snr = np.where(
        state.is_optical,
        state.optical_snr_base + rng.normal(0, 0.5, n),
        np.nan,
    )

    # LSP metrics
    lsp_util = np.where(
        state.is_lsp,
        _pct(state.util_base_in * diurnal_vec * 0.8 + rng.normal(0, 3, n)),
        np.nan,
    )
    lsp_latency = np.where(
        state.is_lsp,
        _pos(state.lsp_latency_base + lat_jitter + diurnal_vec * 2.0 + rng.normal(0, 0.3, n)),
        np.nan,
    )

    # BGP metrics
    has_bgp = np.isin(state.entity_types, ["PE_ROUTER", "AGGREGATION_SWITCH", "BNG", "L3VPN", "ACCESS_SWITCH"])
    bgp_prefixes = np.where(
        has_bgp,
        _pos(state.bgp_prefix_base + rng.normal(0, state.bgp_prefix_base * 0.02, n)),
        np.nan,
    )
    # Flaps are rare events — Poisson with very low rate, higher under stress
    flap_rate = np.where((has_bgp) & (util_in > 85), 0.3, 0.02)
    bgp_flaps = np.where(
        has_bgp,
        rng.poisson(flap_rate, n).astype(np.float64),
        np.nan,
    )

    # Microwave-specific
    mw_avail = np.where(
        state.is_mw,
        _pct(np.full(n, 99.5) + rng.normal(0, 0.3, n)),
        np.nan,
    )
    mw_capacity = np.where(
        state.is_mw,
        _pos(state.mw_capacity_base * (0.9 + 0.1 * diurnal_vec) + rng.normal(0, 50, n)),
        np.nan,
    )
    # Modulation — pick based on capacity/quality
    mw_mod_arr = np.full(n, "", dtype=object)
    mw_indices = np.where(state.is_mw)[0]
    if len(mw_indices) > 0:
        mod_idx = np.clip(
            (mw_capacity[mw_indices] / state.mw_capacity_base[mw_indices] * 5).astype(int),
            0,
            len(_MW_MODULATIONS) - 1,
        )
        for i, mi in zip(mw_indices, mod_idx):
            mw_mod_arr[i] = _MW_MODULATIONS[mi]
    # Replace empty string with None for non-MW
    mw_mod_list = [s if s else None for s in mw_mod_arr]

    ts_arr = pa.array([timestamp] * n, type=pa.timestamp("us", tz="UTC"))

    arrays = [
        pa.array(state.entity_ids, type=pa.string()),
        pa.array([state.tenant_id] * n, type=pa.string()),
        ts_arr,
        pa.array(state.entity_types, type=pa.string()),
        pa.array(state.site_ids, type=pa.string()),
        pa.array(util_in.astype(np.float32), type=F32),
        pa.array(util_out.astype(np.float32), type=F32),
        pa.array(errors_in.astype(np.float32), type=F32),
        pa.array(errors_out.astype(np.float32), type=F32),
        pa.array(discards_in.astype(np.float32), type=F32),
        pa.array(discards_out.astype(np.float32), type=F32),
        pa.array(optical_rx.astype(np.float32), type=F32),
        pa.array(optical_snr.astype(np.float32), type=F32),
        pa.array(lsp_util.astype(np.float32), type=F32),
        pa.array(lsp_latency.astype(np.float32), type=F32),
        pa.array(bgp_prefixes.astype(np.float32), type=F32),
        pa.array(bgp_flaps.astype(np.float32), type=F32),
        pa.array(mw_mod_list, type=pa.string()),
        pa.array(mw_capacity.astype(np.float32), type=F32),
        pa.array(mw_avail.astype(np.float32), type=F32),
    ]

    return pa.RecordBatch.from_arrays(arrays, schema=TRANSPORT_SCHEMA)


# ═══════════════════════════════════════════════════════════════════════════
# 2. FIXED BROADBAND DOMAIN KPIs
# ═══════════════════════════════════════════════════════════════════════════

FIXED_BB_KPI_ENTITY_TYPES = ["OLT", "PON_PORT"]

FIXED_BB_SCHEMA = pa.schema(
    [
        pa.field("entity_id", pa.string(), nullable=False),
        pa.field("tenant_id", pa.string(), nullable=False),
        pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("entity_type", pa.string(), nullable=False),
        pa.field("exchange_id", pa.string()),
        # 14 KPI columns
        pa.field("pon_rx_power_dbm", F32),
        pa.field("pon_tx_power_dbm", F32),
        pa.field("pon_ber", F32),
        pa.field("olt_port_utilization_pct", F32),
        pa.field("ont_uptime_pct", F32),
        pa.field("broadband_sync_rate_down_mbps", F32),
        pa.field("broadband_sync_rate_up_mbps", F32),
        pa.field("broadband_throughput_down_mbps", F32),
        pa.field("broadband_throughput_up_mbps", F32),
        pa.field("broadband_latency_ms", F32),
        pa.field("broadband_packet_loss_pct", F32),
        pa.field("pppoe_session_count", F32),
        pa.field("dhcp_lease_failures", F32),
        pa.field("dns_query_latency_ms", F32),
    ]
)


@dataclass
class _FixedBBState:
    n: int
    entity_ids: np.ndarray
    entity_types: np.ndarray
    exchange_ids: np.ndarray
    tenant_id: str
    # Baselines
    sync_down_base: np.ndarray  # per-entity downstream sync rate
    sync_up_base: np.ndarray
    pon_rx_base: np.ndarray
    pon_tx_base: np.ndarray
    session_base: np.ndarray  # PPPoE session baseline
    is_olt: np.ndarray
    # AR(1)
    util_ar: _AR1
    latency_ar: _AR1


def _init_fixed_bb(config: GeneratorConfig, rng: np.random.Generator) -> _FixedBBState | None:
    df = _load_entities(config, FIXED_BB_KPI_ENTITY_TYPES, columns=["site_id", "parent_entity_id"])
    if df.height == 0:
        console.print("    [yellow]No fixed broadband KPI entities found — skipping.[/yellow]")
        return None

    n = df.height
    entity_ids = df["entity_id"].to_numpy().astype(str)
    entity_types = df["entity_type"].to_numpy().astype(str)
    # For OLTs, site_id is the exchange; for PON_PORTs, parent_entity_id is the OLT
    # Use site_id as exchange_id proxy
    exchange_ids = df["site_id"].fill_null("").to_numpy().astype(str)

    is_olt = entity_types == "OLT"

    # Sync rate baselines — OLTs aggregate many subscribers
    sync_down_base = np.where(
        is_olt,
        rng.uniform(500.0, 2000.0, n),  # OLT aggregate downstream
        rng.uniform(100.0, 1000.0, n),  # PON port
    )
    sync_up_base = sync_down_base * rng.uniform(0.05, 0.15, n)

    # Optical
    pon_rx_base = rng.uniform(-22.0, -8.0, n)
    pon_tx_base = rng.uniform(2.0, 7.0, n)

    # PPPoE sessions
    session_base = np.where(
        is_olt,
        rng.uniform(2000.0, 15000.0, n),
        rng.uniform(5.0, 32.0, n),  # PON port serves ~16 ONTs
    )

    del df
    gc.collect()

    return _FixedBBState(
        n=n,
        entity_ids=entity_ids,
        entity_types=entity_types,
        exchange_ids=exchange_ids,
        tenant_id=config.tenant_id,
        sync_down_base=sync_down_base,
        sync_up_base=sync_up_base,
        pon_rx_base=pon_rx_base,
        pon_tx_base=pon_tx_base,
        session_base=session_base,
        is_olt=is_olt,
        util_ar=_AR1(
            value=rng.normal(0.0, 2.0, size=n),
            rho=0.75,
            innov_std=2.0 * np.sqrt(1 - 0.75**2),
        ),
        latency_ar=_AR1(
            value=rng.normal(0.0, 1.0, size=n),
            rho=0.70,
            innov_std=1.0 * np.sqrt(1 - 0.70**2),
        ),
    )


def _fixed_bb_hour(
    state: _FixedBBState,
    hour_idx: int,
    rng: np.random.Generator,
    timestamp: datetime,
) -> pa.RecordBatch:
    n = state.n
    diurnal = _diurnal_factor(hour_idx)

    util_jitter = state.util_ar.advance(rng)
    lat_jitter = state.latency_ar.advance(rng)

    # PON optical
    pon_rx = state.pon_rx_base + rng.normal(0, 0.2, n)
    pon_tx = state.pon_tx_base + rng.normal(0, 0.15, n)

    # BER — very low normally, correlated with rx power degradation
    ber_base = 1e-10
    rx_penalty = np.where(pon_rx < -25.0, 1e-6, np.where(pon_rx < -20.0, 1e-8, ber_base))
    pon_ber = _pos(rx_penalty + rng.exponential(rx_penalty * 0.1, n))

    # Port utilisation
    olt_util = _pct(
        np.where(
            state.is_olt,
            40.0 * diurnal + util_jitter + rng.normal(0, 2, n),
            30.0 * diurnal + util_jitter * 0.8 + rng.normal(0, 2, n),
        )
    )

    # ONT uptime — normally high, slight random outages
    ont_uptime = _pct(np.full(n, 99.8) + rng.normal(0, 0.15, n))

    # Throughput tracks diurnal load
    tp_down = _pos(state.sync_down_base * diurnal * 0.7 + rng.normal(0, state.sync_down_base * 0.03, n))
    tp_up = _pos(state.sync_up_base * diurnal * 0.6 + rng.normal(0, state.sync_up_base * 0.05, n))

    # Sync rates are more stable (physical layer)
    sync_down = _pos(state.sync_down_base + rng.normal(0, state.sync_down_base * 0.005, n))
    sync_up = _pos(state.sync_up_base + rng.normal(0, state.sync_up_base * 0.01, n))

    # Latency
    latency = _pos(8.0 + diurnal * 5.0 + lat_jitter + rng.normal(0, 0.5, n))

    # Packet loss
    pkt_loss = _pct(np.where(olt_util > 80, 0.5, 0.02) + rng.exponential(0.01, n))

    # PPPoE sessions — diurnal
    sessions = _pos(state.session_base * diurnal + rng.normal(0, state.session_base * 0.03, n))

    # DHCP failures — rare
    dhcp_fail = _pos(rng.poisson(np.where(sessions > state.session_base * 0.9, 0.5, 0.05), n).astype(np.float64))

    # DNS latency
    dns_lat = _pos(2.0 + diurnal * 3.0 + rng.normal(0, 0.3, n))

    ts_arr = pa.array([timestamp] * n, type=pa.timestamp("us", tz="UTC"))

    arrays = [
        pa.array(state.entity_ids, type=pa.string()),
        pa.array([state.tenant_id] * n, type=pa.string()),
        ts_arr,
        pa.array(state.entity_types, type=pa.string()),
        pa.array(state.exchange_ids, type=pa.string()),
        pa.array(pon_rx.astype(np.float32), type=F32),
        pa.array(pon_tx.astype(np.float32), type=F32),
        pa.array(pon_ber.astype(np.float32), type=F32),
        pa.array(olt_util.astype(np.float32), type=F32),
        pa.array(ont_uptime.astype(np.float32), type=F32),
        pa.array(sync_down.astype(np.float32), type=F32),
        pa.array(sync_up.astype(np.float32), type=F32),
        pa.array(tp_down.astype(np.float32), type=F32),
        pa.array(tp_up.astype(np.float32), type=F32),
        pa.array(latency.astype(np.float32), type=F32),
        pa.array(pkt_loss.astype(np.float32), type=F32),
        pa.array(sessions.astype(np.float32), type=F32),
        pa.array(dhcp_fail.astype(np.float32), type=F32),
        pa.array(dns_lat.astype(np.float32), type=F32),
    ]

    return pa.RecordBatch.from_arrays(arrays, schema=FIXED_BB_SCHEMA)


# ═══════════════════════════════════════════════════════════════════════════
# 3. ENTERPRISE CIRCUIT KPIs
# ═══════════════════════════════════════════════════════════════════════════

ENTERPRISE_KPI_ENTITY_TYPES = ["ETHERNET_CIRCUIT"]

ENTERPRISE_SCHEMA = pa.schema(
    [
        pa.field("entity_id", pa.string(), nullable=False),
        pa.field("tenant_id", pa.string(), nullable=False),
        pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("entity_type", pa.string(), nullable=False),
        pa.field("sla_tier", pa.string()),
        # 10 KPI columns
        pa.field("circuit_availability_pct", F32),
        pa.field("circuit_throughput_in_mbps", F32),
        pa.field("circuit_throughput_out_mbps", F32),
        pa.field("circuit_latency_ms", F32),
        pa.field("circuit_jitter_ms", F32),
        pa.field("circuit_packet_loss_pct", F32),
        pa.field("vpn_prefix_count", F32),
        pa.field("sla_breach_count", F32),
        pa.field("cos_queue_drops", F32),
        pa.field("circuit_uptime_seconds", F32),
    ]
)

# SLA tier → latency budget & loss budget
_SLA_PARAMS: dict[str, dict[str, float]] = {
    "GOLD": {"latency_base": 3.0, "loss_budget": 0.01, "avail_target": 99.99},
    "SILVER": {"latency_base": 8.0, "loss_budget": 0.05, "avail_target": 99.95},
    "BRONZE": {"latency_base": 15.0, "loss_budget": 0.10, "avail_target": 99.90},
}


@dataclass
class _EnterpriseState:
    n: int
    entity_ids: np.ndarray
    entity_types: np.ndarray
    sla_tiers: np.ndarray
    tenant_id: str
    # Baselines
    speed_mbps: np.ndarray  # circuit committed speed
    latency_base: np.ndarray
    loss_budget: np.ndarray
    avail_target: np.ndarray
    prefix_base: np.ndarray
    # AR(1)
    util_ar: _AR1
    latency_ar: _AR1


def _init_enterprise(config: GeneratorConfig, rng: np.random.Generator) -> _EnterpriseState | None:
    df = _load_entities(config, ENTERPRISE_KPI_ENTITY_TYPES, columns=["sla_tier", "properties_json"])
    if df.height == 0:
        console.print("    [yellow]No enterprise circuit entities found — skipping.[/yellow]")
        return None

    n = df.height
    entity_ids = df["entity_id"].to_numpy().astype(str)
    entity_types = df["entity_type"].to_numpy().astype(str)
    sla_raw = df["sla_tier"].fill_null("SILVER").to_numpy().astype(str)

    # Parse speed from properties_json
    props_col = df["properties_json"].fill_null("{}").to_list()
    speed_mbps = np.empty(n, dtype=np.float64)
    for i, pj in enumerate(props_col):
        try:
            d = json.loads(pj) if isinstance(pj, str) else (pj or {})
            speed_mbps[i] = float(d.get("speed_mbps", rng.choice([100, 1000, 10000])))
        except (json.JSONDecodeError, TypeError):
            speed_mbps[i] = float(rng.choice([100, 1000, 10000]))

    latency_base = np.array([_SLA_PARAMS.get(s, _SLA_PARAMS["SILVER"])["latency_base"] for s in sla_raw])
    loss_budget = np.array([_SLA_PARAMS.get(s, _SLA_PARAMS["SILVER"])["loss_budget"] for s in sla_raw])
    avail_target = np.array([_SLA_PARAMS.get(s, _SLA_PARAMS["SILVER"])["avail_target"] for s in sla_raw])

    prefix_base = rng.uniform(10.0, 500.0, n)

    del df
    gc.collect()

    return _EnterpriseState(
        n=n,
        entity_ids=entity_ids,
        entity_types=entity_types,
        sla_tiers=sla_raw,
        tenant_id=config.tenant_id,
        speed_mbps=speed_mbps,
        latency_base=latency_base,
        loss_budget=loss_budget,
        avail_target=avail_target,
        prefix_base=prefix_base,
        util_ar=_AR1(
            value=rng.normal(0.0, 2.0, size=n),
            rho=0.80,
            innov_std=2.0 * np.sqrt(1 - 0.80**2),
        ),
        latency_ar=_AR1(
            value=rng.normal(0.0, 0.5, size=n),
            rho=0.75,
            innov_std=0.5 * np.sqrt(1 - 0.75**2),
        ),
    )


def _enterprise_hour(
    state: _EnterpriseState,
    hour_idx: int,
    rng: np.random.Generator,
    timestamp: datetime,
) -> pa.RecordBatch:
    n = state.n
    diurnal = _diurnal_factor(hour_idx)

    util_jitter = state.util_ar.advance(rng)
    lat_jitter = state.latency_ar.advance(rng)

    # Throughput
    tp_factor = diurnal * 0.6 + 0.2  # enterprise has more constant baseline
    tp_in = _pos(
        state.speed_mbps * tp_factor + util_jitter * state.speed_mbps * 0.01 + rng.normal(0, state.speed_mbps * 0.02, n)
    )
    tp_out = _pos(
        state.speed_mbps * tp_factor * 0.7
        + util_jitter * state.speed_mbps * 0.008
        + rng.normal(0, state.speed_mbps * 0.015, n)
    )
    # Cap at committed speed
    tp_in = np.minimum(tp_in, state.speed_mbps)
    tp_out = np.minimum(tp_out, state.speed_mbps)

    # Latency
    latency = _pos(state.latency_base + lat_jitter + diurnal * 1.5 + rng.normal(0, 0.3, n))

    # Jitter — typically ~10% of latency
    jitter = _pos(latency * 0.1 + rng.normal(0, 0.1, n))

    # Packet loss — normally within SLA budget
    pkt_loss = _pct(_pos(state.loss_budget * 0.3 + rng.exponential(state.loss_budget * 0.1, n)))

    # Availability — mostly perfect, occasional micro-outages
    # Each hour is 3600 seconds; availability = uptime_seconds / 3600
    outage_seconds = rng.exponential(3600 * (100 - state.avail_target) / 100.0, n)
    outage_seconds = np.clip(outage_seconds, 0, 3600)
    uptime_seconds = np.clip(3600.0 - outage_seconds, 0, 3600)
    avail_pct = _pct(uptime_seconds / 3600.0 * 100.0)

    # VPN prefix count — slowly varying
    prefixes = _pos(state.prefix_base + rng.normal(0, state.prefix_base * 0.01, n))

    # SLA breaches — when latency or loss exceed budget
    breach_latency = latency > (state.latency_base * 2.0)
    breach_loss = pkt_loss > (state.loss_budget * 3.0)
    sla_breaches = breach_latency.astype(np.float64) + breach_loss.astype(np.float64)

    # CoS queue drops — rare, correlated with high utilisation
    util_ratio = tp_in / np.maximum(state.speed_mbps, 1.0)
    queue_drop_rate = np.where(util_ratio > 0.85, 5.0, 0.1)
    cos_drops = _pos(rng.poisson(queue_drop_rate, n).astype(np.float64))

    ts_arr = pa.array([timestamp] * n, type=pa.timestamp("us", tz="UTC"))

    arrays = [
        pa.array(state.entity_ids, type=pa.string()),
        pa.array([state.tenant_id] * n, type=pa.string()),
        ts_arr,
        pa.array(state.entity_types, type=pa.string()),
        pa.array(state.sla_tiers, type=pa.string()),
        pa.array(avail_pct.astype(np.float32), type=F32),
        pa.array(tp_in.astype(np.float32), type=F32),
        pa.array(tp_out.astype(np.float32), type=F32),
        pa.array(latency.astype(np.float32), type=F32),
        pa.array(jitter.astype(np.float32), type=F32),
        pa.array(pkt_loss.astype(np.float32), type=F32),
        pa.array(prefixes.astype(np.float32), type=F32),
        pa.array(sla_breaches.astype(np.float32), type=F32),
        pa.array(cos_drops.astype(np.float32), type=F32),
        pa.array(uptime_seconds.astype(np.float32), type=F32),
    ]

    return pa.RecordBatch.from_arrays(arrays, schema=ENTERPRISE_SCHEMA)


# ═══════════════════════════════════════════════════════════════════════════
# 4. CORE NETWORK ELEMENT KPIs
# ═══════════════════════════════════════════════════════════════════════════

CORE_KPI_ENTITY_TYPES = [
    "MME",
    "SGW",
    "PGW",
    "HSS",
    "AMF",
    "SMF",
    "UPF",
    "NSSF",
    "PCF",
    "UDM",
    "NWDAF",
    "P_CSCF",
    "S_CSCF",
    "TAS",
    "MGCF",
    "RADIUS_SERVER",
    "DHCP_SERVER",
    "DNS_RESOLVER",
    "POLICY_SERVER",
    "SOFTSWITCH",
    "SBC",
    "MEDIA_GATEWAY",
    "SIP_TRUNK",
    "SD_WAN_CONTROLLER",
    "FIREWALL_SERVICE",
    "CE_ROUTER",
    "BNG",
]

CORE_SCHEMA = pa.schema(
    [
        pa.field("entity_id", pa.string(), nullable=False),
        pa.field("tenant_id", pa.string(), nullable=False),
        pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("entity_type", pa.string(), nullable=False),
        pa.field("core_domain", pa.string()),
        # 20 KPI columns
        pa.field("cpu_utilization_pct", F32),
        pa.field("memory_utilization_pct", F32),
        pa.field("active_sessions", F32),
        pa.field("session_setup_rate", F32),
        pa.field("session_setup_success_pct", F32),
        pa.field("signalling_load_pct", F32),
        pa.field("throughput_gbps", F32),
        pa.field("control_plane_latency_ms", F32),
        pa.field("user_plane_latency_ms", F32),
        pa.field("error_rate_pct", F32),
        pa.field("paging_attempts", F32),
        pa.field("paging_success_pct", F32),
        pa.field("handover_in_count", F32),
        pa.field("handover_out_count", F32),
        pa.field("bearer_activation_success_pct", F32),
        pa.field("dns_query_count", F32),
        pa.field("radius_auth_latency_ms", F32),
        pa.field("sip_registrations", F32),
        pa.field("volte_call_setup_success_pct", F32),
        pa.field("element_availability_pct", F32),
    ]
)

# Map entity type → core sub-domain
_CORE_DOMAIN_MAP: dict[str, str] = {
    "MME": "epc",
    "SGW": "epc",
    "PGW": "epc",
    "HSS": "epc",
    "AMF": "5gc",
    "SMF": "5gc",
    "UPF": "5gc",
    "NSSF": "5gc",
    "PCF": "5gc",
    "UDM": "5gc",
    "NWDAF": "5gc",
    "P_CSCF": "ims",
    "S_CSCF": "ims",
    "TAS": "ims",
    "MGCF": "ims",
    "RADIUS_SERVER": "broadband_control",
    "DHCP_SERVER": "broadband_control",
    "DNS_RESOLVER": "broadband_control",
    "POLICY_SERVER": "broadband_control",
    "SOFTSWITCH": "voice",
    "SBC": "voice",
    "MEDIA_GATEWAY": "voice",
    "SIP_TRUNK": "voice",
    "SD_WAN_CONTROLLER": "enterprise_control",
    "FIREWALL_SERVICE": "enterprise_control",
    "CE_ROUTER": "enterprise_control",
    "BNG": "broadband_control",
}

# Baseline CPU/memory per entity type
_CORE_BASELINES: dict[str, dict[str, float]] = {
    "MME": {"cpu": 40, "mem": 55, "sessions": 500000, "throughput": 0.0},
    "SGW": {"cpu": 35, "mem": 50, "sessions": 400000, "throughput": 50.0},
    "PGW": {"cpu": 45, "mem": 60, "sessions": 300000, "throughput": 80.0},
    "HSS": {"cpu": 25, "mem": 40, "sessions": 0, "throughput": 0.0},
    "AMF": {"cpu": 35, "mem": 50, "sessions": 600000, "throughput": 0.0},
    "SMF": {"cpu": 30, "mem": 45, "sessions": 500000, "throughput": 0.0},
    "UPF": {"cpu": 50, "mem": 65, "sessions": 400000, "throughput": 150.0},
    "NSSF": {"cpu": 15, "mem": 25, "sessions": 0, "throughput": 0.0},
    "PCF": {"cpu": 20, "mem": 30, "sessions": 0, "throughput": 0.0},
    "UDM": {"cpu": 20, "mem": 35, "sessions": 0, "throughput": 0.0},
    "NWDAF": {"cpu": 55, "mem": 70, "sessions": 0, "throughput": 0.0},
    "P_CSCF": {"cpu": 30, "mem": 40, "sessions": 100000, "throughput": 0.0},
    "S_CSCF": {"cpu": 35, "mem": 45, "sessions": 100000, "throughput": 0.0},
    "TAS": {"cpu": 40, "mem": 50, "sessions": 50000, "throughput": 0.0},
    "MGCF": {"cpu": 25, "mem": 35, "sessions": 20000, "throughput": 0.0},
    "RADIUS_SERVER": {"cpu": 30, "mem": 40, "sessions": 0, "throughput": 0.0},
    "DHCP_SERVER": {"cpu": 20, "mem": 30, "sessions": 0, "throughput": 0.0},
    "DNS_RESOLVER": {"cpu": 35, "mem": 45, "sessions": 0, "throughput": 0.0},
    "POLICY_SERVER": {"cpu": 25, "mem": 35, "sessions": 0, "throughput": 0.0},
    "SOFTSWITCH": {"cpu": 30, "mem": 40, "sessions": 50000, "throughput": 0.0},
    "SBC": {"cpu": 35, "mem": 45, "sessions": 80000, "throughput": 5.0},
    "MEDIA_GATEWAY": {"cpu": 40, "mem": 50, "sessions": 30000, "throughput": 2.0},
    "SIP_TRUNK": {"cpu": 20, "mem": 30, "sessions": 10000, "throughput": 1.0},
    "SD_WAN_CONTROLLER": {"cpu": 45, "mem": 55, "sessions": 5000, "throughput": 0.0},
    "FIREWALL_SERVICE": {"cpu": 50, "mem": 60, "sessions": 200000, "throughput": 20.0},
    "CE_ROUTER": {"cpu": 20, "mem": 30, "sessions": 0, "throughput": 1.0},
    "BNG": {"cpu": 45, "mem": 60, "sessions": 150000, "throughput": 100.0},
}


@dataclass
class _CoreState:
    n: int
    entity_ids: np.ndarray
    entity_types: np.ndarray
    core_domains: np.ndarray
    tenant_id: str
    # Baselines
    cpu_base: np.ndarray
    mem_base: np.ndarray
    session_base: np.ndarray
    throughput_base: np.ndarray
    is_signalling: np.ndarray  # entities that handle signalling (MME, AMF, etc.)
    is_user_plane: np.ndarray  # entities that handle user plane (SGW, PGW, UPF, etc.)
    is_dns: np.ndarray
    is_radius: np.ndarray
    is_ims: np.ndarray
    # AR(1)
    cpu_ar: _AR1
    session_ar: _AR1


def _init_core(config: GeneratorConfig, rng: np.random.Generator) -> _CoreState | None:
    df = _load_entities(config, CORE_KPI_ENTITY_TYPES, columns=["properties_json"])
    if df.height == 0:
        console.print("    [yellow]No core element entities found — skipping.[/yellow]")
        return None

    n = df.height
    entity_ids = df["entity_id"].to_numpy().astype(str)
    entity_types = df["entity_type"].to_numpy().astype(str)
    core_domains = np.array([_CORE_DOMAIN_MAP.get(et, "unknown") for et in entity_types])

    cpu_base = np.empty(n, dtype=np.float64)
    mem_base = np.empty(n, dtype=np.float64)
    session_base = np.empty(n, dtype=np.float64)
    throughput_base = np.empty(n, dtype=np.float64)

    for i, et in enumerate(entity_types):
        bl = _CORE_BASELINES.get(et, {"cpu": 30, "mem": 40, "sessions": 0, "throughput": 0})
        cpu_base[i] = bl["cpu"] + rng.normal(0, 3)
        mem_base[i] = bl["mem"] + rng.normal(0, 3)
        session_base[i] = bl["sessions"] * (1 + rng.normal(0, 0.1))
        throughput_base[i] = bl["throughput"] * (1 + rng.normal(0, 0.1))

    is_signalling = np.isin(entity_types, ["MME", "AMF", "SMF", "P_CSCF", "S_CSCF", "NSSF", "PCF"])
    is_user_plane = np.isin(entity_types, ["SGW", "PGW", "UPF", "BNG", "MEDIA_GATEWAY", "SBC"])
    is_dns = entity_types == "DNS_RESOLVER"
    is_radius = entity_types == "RADIUS_SERVER"
    is_ims = np.isin(entity_types, ["P_CSCF", "S_CSCF", "TAS", "MGCF", "SBC", "MEDIA_GATEWAY", "SIP_TRUNK"])

    del df
    gc.collect()

    return _CoreState(
        n=n,
        entity_ids=entity_ids,
        entity_types=entity_types,
        core_domains=core_domains,
        tenant_id=config.tenant_id,
        cpu_base=cpu_base,
        mem_base=mem_base,
        session_base=session_base,
        throughput_base=throughput_base,
        is_signalling=is_signalling,
        is_user_plane=is_user_plane,
        is_dns=is_dns,
        is_radius=is_radius,
        is_ims=is_ims,
        cpu_ar=_AR1(
            value=rng.normal(0.0, 2.0, size=n),
            rho=0.85,
            innov_std=2.0 * np.sqrt(1 - 0.85**2),
        ),
        session_ar=_AR1(
            value=rng.normal(0.0, 0.02, size=n),
            rho=0.80,
            innov_std=0.02 * np.sqrt(1 - 0.80**2),
        ),
    )


def _core_hour(
    state: _CoreState,
    hour_idx: int,
    rng: np.random.Generator,
    timestamp: datetime,
) -> pa.RecordBatch:
    n = state.n
    diurnal = _diurnal_factor(hour_idx)

    cpu_jitter = state.cpu_ar.advance(rng)
    session_jitter = state.session_ar.advance(rng)

    # CPU & memory — track diurnal load
    cpu = _pct(state.cpu_base * (0.5 + 0.5 * diurnal) + cpu_jitter + rng.normal(0, 1.5, n))
    mem = _pct(state.mem_base * (0.7 + 0.3 * diurnal) + rng.normal(0, 1.0, n))

    # Active sessions
    sessions = _pos(state.session_base * diurnal * (1.0 + session_jitter) + rng.normal(0, state.session_base * 0.01, n))

    # Session setup rate (per second) — proportional to session churn
    setup_rate = _pos(sessions * 0.001 * diurnal + rng.normal(0, 0.5, n))

    # Session setup success — normally very high
    setup_success = _pct(np.full(n, 99.5) + rng.normal(0, 0.3, n) - np.where(cpu > 80, 2.0, 0.0))

    # Signalling load (for signalling elements)
    sig_load = np.where(
        state.is_signalling,
        _pct(cpu * 0.8 + diurnal * 10.0 + rng.normal(0, 2.0, n)),
        np.nan,
    )

    # Throughput (for user-plane elements)
    tp = np.where(
        state.is_user_plane,
        _pos(state.throughput_base * diurnal * (1 + session_jitter) + rng.normal(0, state.throughput_base * 0.02, n)),
        np.nan,
    )

    # Control plane latency
    cp_latency = _pos(2.0 + diurnal * 3.0 + rng.normal(0, 0.3, n) + np.where(cpu > 80, 5.0, 0.0))

    # User plane latency
    up_latency = np.where(
        state.is_user_plane,
        _pos(1.0 + diurnal * 1.5 + rng.normal(0, 0.2, n) + np.where(cpu > 85, 3.0, 0.0)),
        np.nan,
    )

    # Error rate
    error_rate = _pct(_pos(0.01 + rng.exponential(0.005, n) + np.where(cpu > 90, 0.5, 0.0)))

    # Paging (for MME/AMF)
    has_paging = np.isin(state.entity_types, ["MME", "AMF"])
    paging_att = np.where(has_paging, _pos(sessions * 0.1 * diurnal + rng.normal(0, 100, n)), np.nan)
    paging_succ = np.where(has_paging, _pct(np.full(n, 98.5) + rng.normal(0, 0.5, n)), np.nan)

    # Handovers (for MME/AMF)
    ho_in = np.where(has_paging, _pos(sessions * 0.05 * diurnal + rng.normal(0, 50, n)), np.nan)
    ho_out = np.where(has_paging, _pos(sessions * 0.05 * diurnal + rng.normal(0, 50, n)), np.nan)

    # Bearer activation (for SGW/PGW/SMF/UPF)
    has_bearer = np.isin(state.entity_types, ["SGW", "PGW", "SMF", "UPF"])
    bearer_succ = np.where(has_bearer, _pct(np.full(n, 99.2) + rng.normal(0, 0.3, n)), np.nan)

    # DNS queries (DNS resolvers only)
    dns_queries = np.where(
        state.is_dns,
        _pos(50000 * diurnal + rng.normal(0, 2000, n)),
        np.nan,
    )

    # RADIUS auth latency
    radius_lat = np.where(
        state.is_radius,
        _pos(5.0 + diurnal * 3.0 + rng.normal(0, 0.5, n)),
        np.nan,
    )

    # IMS / VoLTE metrics
    sip_reg = np.where(
        state.is_ims,
        _pos(sessions * 0.2 * diurnal + rng.normal(0, 100, n)),
        np.nan,
    )
    volte_succ = np.where(
        state.is_ims,
        _pct(np.full(n, 98.5) + rng.normal(0, 0.5, n)),
        np.nan,
    )

    # Element availability
    avail = _pct(np.full(n, 99.99) + rng.normal(0, 0.005, n))

    ts_arr = pa.array([timestamp] * n, type=pa.timestamp("us", tz="UTC"))

    arrays = [
        pa.array(state.entity_ids, type=pa.string()),
        pa.array([state.tenant_id] * n, type=pa.string()),
        ts_arr,
        pa.array(state.entity_types, type=pa.string()),
        pa.array(state.core_domains, type=pa.string()),
        pa.array(cpu.astype(np.float32), type=F32),
        pa.array(mem.astype(np.float32), type=F32),
        pa.array(sessions.astype(np.float32), type=F32),
        pa.array(setup_rate.astype(np.float32), type=F32),
        pa.array(setup_success.astype(np.float32), type=F32),
        pa.array(sig_load.astype(np.float32), type=F32),
        pa.array(tp.astype(np.float32), type=F32),
        pa.array(cp_latency.astype(np.float32), type=F32),
        pa.array(up_latency.astype(np.float32), type=F32),
        pa.array(error_rate.astype(np.float32), type=F32),
        pa.array(paging_att.astype(np.float32), type=F32),
        pa.array(paging_succ.astype(np.float32), type=F32),
        pa.array(ho_in.astype(np.float32), type=F32),
        pa.array(ho_out.astype(np.float32), type=F32),
        pa.array(bearer_succ.astype(np.float32), type=F32),
        pa.array(dns_queries.astype(np.float32), type=F32),
        pa.array(radius_lat.astype(np.float32), type=F32),
        pa.array(sip_reg.astype(np.float32), type=F32),
        pa.array(volte_succ.astype(np.float32), type=F32),
        pa.array(avail.astype(np.float32), type=F32),
    ]

    return pa.RecordBatch.from_arrays(arrays, schema=CORE_SCHEMA)


# ═══════════════════════════════════════════════════════════════════════════
# 5. POWER / ENVIRONMENT KPIs
# ═══════════════════════════════════════════════════════════════════════════

POWER_SCHEMA = pa.schema(
    [
        pa.field("site_id", pa.string(), nullable=False),
        pa.field("tenant_id", pa.string(), nullable=False),
        pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("site_type", pa.string(), nullable=False),
        pa.field("deployment_profile", pa.string(), nullable=False),
        # 7 KPI columns
        pa.field("mains_power_status", F32),
        pa.field("battery_voltage_v", F32),
        pa.field("battery_charge_pct", F32),
        pa.field("cabinet_temperature_c", F32),
        pa.field("cabinet_humidity_pct", F32),
        pa.field("generator_runtime_hours", F32),
        pa.field("cooling_status", F32),
    ]
)


@dataclass
class _PowerState:
    n: int
    site_ids: np.ndarray
    site_types: np.ndarray
    profiles: np.ndarray
    tenant_id: str
    utc_offsets: np.ndarray
    # Baselines
    ambient_temp_base: np.ndarray  # per-site ambient temperature baseline
    humidity_base: np.ndarray
    has_generator: np.ndarray
    gen_runtime_cumulative: np.ndarray  # cumulative generator runtime (hours)
    battery_capacity_v: np.ndarray
    # AR(1)
    temp_ar: _AR1
    humidity_ar: _AR1


def _init_power(config: GeneratorConfig, rng: np.random.Generator) -> _PowerState | None:
    sites_df = _load_sites(config)
    if sites_df.height == 0:
        console.print("    [yellow]No sites found — skipping power KPIs.[/yellow]")
        return None

    n = sites_df.height
    site_ids = sites_df["site_id"].to_numpy().astype(str)
    site_types = sites_df["site_type"].to_numpy().astype(str)
    profiles = sites_df["deployment_profile"].to_numpy().astype(str)
    timezones = sites_df["timezone"].to_numpy().astype(str)
    utc_offsets = np.array([_TZ_OFFSET.get(tz, 7) for tz in timezones], dtype=np.int32)

    # Ambient temperature varies by deployment profile
    temp_map = {
        "dense_urban": 32.0,
        "urban": 30.0,
        "suburban": 28.0,
        "rural": 27.0,
        "deep_rural": 26.0,
        "indoor": 24.0,
    }
    ambient_temp_base = np.array([temp_map.get(p, 28.0) for p in profiles]) + rng.normal(0, 2, n)

    humidity_base = rng.uniform(55.0, 85.0, n)  # Indonesia is tropical — high humidity

    # Generator presence (approximate — rural/deep_rural have them more often)
    has_generator = np.zeros(n, dtype=bool)
    for i, (st, dp) in enumerate(zip(site_types, profiles)):
        if st == "greenfield" and dp in ("rural", "deep_rural"):
            has_generator[i] = rng.random() < 0.6
        elif st == "in_building":
            has_generator[i] = rng.random() < 0.3
        elif dp == "dense_urban":
            has_generator[i] = rng.random() < 0.15

    gen_runtime = np.where(has_generator, rng.uniform(0, 5000, n), 0.0)
    battery_v = rng.uniform(47.0, 54.0, n)  # 48V DC system, nominal 48V

    del sites_df
    gc.collect()

    return _PowerState(
        n=n,
        site_ids=site_ids,
        site_types=site_types,
        profiles=profiles,
        tenant_id=config.tenant_id,
        utc_offsets=utc_offsets,
        ambient_temp_base=ambient_temp_base,
        humidity_base=humidity_base,
        has_generator=has_generator,
        gen_runtime_cumulative=gen_runtime,
        battery_capacity_v=battery_v,
        temp_ar=_AR1(
            value=rng.normal(0.0, 0.5, size=n),
            rho=0.92,
            innov_std=0.5 * np.sqrt(1 - 0.92**2),
        ),
        humidity_ar=_AR1(
            value=rng.normal(0.0, 1.0, size=n),
            rho=0.88,
            innov_std=1.0 * np.sqrt(1 - 0.88**2),
        ),
    )


def _power_hour(
    state: _PowerState,
    hour_idx: int,
    rng: np.random.Generator,
    timestamp: datetime,
) -> pa.RecordBatch:
    n = state.n

    temp_jitter = state.temp_ar.advance(rng)
    hum_jitter = state.humidity_ar.advance(rng)

    # Temperature follows its own diurnal + ambient + AR(1)
    # Use per-site local hour for temperature diurnal (peak at local noon)
    base_hour = hour_idx % 24
    local_hours = (base_hour + state.utc_offsets) % 24
    temp_diurnal = _TEMP_DIURNAL[local_hours]
    temp_range = 8.0  # degrees swing from min to max over day
    cabinet_temp = np.clip(
        state.ambient_temp_base + temp_range * (temp_diurnal - 0.5) + temp_jitter + rng.normal(0, 0.3, n),
        -5.0,
        55.0,
    )

    # Humidity — inversely correlated with temperature (hotter → slightly drier indoors with cooling)
    cabinet_humidity = np.clip(
        state.humidity_base - (cabinet_temp - state.ambient_temp_base) * 0.5 + hum_jitter + rng.normal(0, 0.5, n),
        10.0,
        98.0,
    )

    # Mains power — normally 1.0 (OK).  Very rare random outages (~0.05% per hour)
    mains_fail_prob = 0.0005
    mains_status = np.where(rng.random(n) < mains_fail_prob, 0.0, 1.0)

    # RF-13: Battery voltage uses -48V DC convention (ITU-T L.1200).
    # All telecom sites globally use negative polarity.  The magnitude is
    # stored as a negative value so domain experts recognise the convention.
    # Float charge ~-54V, nominal ~-48V, discharge cutoff ~-43V.
    battery_v = -state.battery_capacity_v.copy()  # Flip to negative polarity
    battery_v += rng.normal(0, 0.1, n)  # Small jitter around baseline
    # Mains failure → battery starts discharging (magnitude decreases towards 0,
    # i.e. voltage moves from -48V towards -43V, so we ADD +0.5V)
    battery_v = np.where(mains_status < 0.5, battery_v + 0.5, battery_v)
    battery_v = np.clip(battery_v, -56.0, -40.0)

    # Battery charge percentage (48V system: 100% at ~-54V, 0% at ~-42V)
    # Use absolute voltage magnitude for charge calculation
    battery_v_abs = np.abs(battery_v)
    battery_charge = np.clip((battery_v_abs - 42.0) / (54.0 - 42.0) * 100.0, 0.0, 100.0)

    # Generator runtime — only accumulates when mains is down AND has_generator
    gen_running = (mains_status < 0.5) & state.has_generator
    state.gen_runtime_cumulative += gen_running.astype(np.float64)
    gen_runtime = np.where(state.has_generator, state.gen_runtime_cumulative, np.nan)

    # Cooling status — normally 1.0.  Rare failure (~0.01% per hour), slightly
    # more likely when cabinet temp is very high
    cool_fail_prob = np.where(cabinet_temp > 45, 0.005, 0.0001)
    cooling_status = np.where(rng.random(n) < cool_fail_prob, 0.0, 1.0)

    ts_arr = pa.array([timestamp] * n, type=pa.timestamp("us", tz="UTC"))

    arrays = [
        pa.array(state.site_ids, type=pa.string()),
        pa.array([state.tenant_id] * n, type=pa.string()),
        ts_arr,
        pa.array(state.site_types, type=pa.string()),
        pa.array(state.profiles, type=pa.string()),
        pa.array(mains_status.astype(np.float32), type=F32),
        pa.array(battery_v.astype(np.float32), type=F32),
        pa.array(battery_charge.astype(np.float32), type=F32),
        pa.array(cabinet_temp.astype(np.float32), type=F32),
        pa.array(cabinet_humidity.astype(np.float32), type=F32),
        pa.array(gen_runtime.astype(np.float32), type=F32),
        pa.array(cooling_status.astype(np.float32), type=F32),
    ]

    return pa.RecordBatch.from_arrays(arrays, schema=POWER_SCHEMA)


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — runs all 5 domain KPI generators
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class _DomainSpec:
    """Specification for one domain's KPI generation."""

    name: str
    filename: str
    schema: pa.Schema
    state: Any  # domain-specific state dataclass
    hour_fn: Any  # callable(state, hour_idx, rng, timestamp) -> RecordBatch


def _generate_single_domain(
    spec: _DomainSpec,
    config: GeneratorConfig,
    rng: np.random.Generator,
    total_hours: int,
) -> dict[str, Any]:
    """Generate all hours for a single domain. Returns summary dict."""
    output_path = config.paths.output_dir / spec.filename
    writer = _make_writer(output_path, spec.schema)

    rows_written = 0
    rg_written = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("({task.fields[rows]} rows)"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=1,
    ) as progress:
        task = progress.add_task(
            f"  {spec.name}",
            total=total_hours,
            rows="0",
        )

        for hour_idx in range(total_hours):
            timestamp = SIMULATION_EPOCH + timedelta(hours=hour_idx)

            # Per-hour sub-seed for reproducibility
            hour_rng = np.random.default_rng(rng.integers(0, 2**31) + hour_idx)

            batch = spec.hour_fn(spec.state, hour_idx, hour_rng, timestamp)
            _write_hour(writer, batch)

            rows_written += batch.num_rows
            rg_written += 1

            del batch
            if rg_written % GC_EVERY_N_FLUSHES == 0:
                gc.collect()

            progress.update(task, advance=1, rows=f"{rows_written:,}")

    writer.close()

    size_mb = output_path.stat().st_size / 1024 / 1024
    return {
        "name": spec.name,
        "filename": spec.filename,
        "entities": spec.state.n,
        "rows": rows_written,
        "row_groups": rg_written,
        "columns": len(spec.schema),
        "size_mb": size_mb,
        "path": str(output_path),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_domain_kpis(config: GeneratorConfig) -> None:
    """
    Step 04 entry point: Generate multi-domain KPIs.

    Produces 5 Parquet files — one per domain — using the streaming
    per-hour flush architecture from Phase 3.

    Memory budget per domain:
      - Entity metadata arrays: <10 MB (largest is transport ~30K entities)
      - AR(1) state vectors: <1 MB per domain
      - Per-hour: KPI temporaries ~5 MB + RecordBatch ~10 MB
      - Peak: ~50 MB per domain (domains are generated sequentially)
    """
    step_start = time.time()

    seed = config.seed_for("step_04_domain_kpis")
    rng = np.random.default_rng(seed)
    console.print(f"[dim]Step 04 seed: {seed}[/dim]")

    total_hours = config.simulation.total_intervals
    console.print(f"[bold]Generating multi-domain KPIs:[/bold] {total_hours:,} hourly intervals across 5 domains")

    config.ensure_output_dirs()

    # ── Initialise all domain states ──────────────────────────
    console.print("\n[bold]Initialising domain states...[/bold]")

    specs: list[_DomainSpec] = []

    # 1. Transport
    console.print("  [dim]Loading transport entities...[/dim]")
    t_state = _init_transport(config, rng)
    if t_state is not None:
        console.print(f"    [green]✓[/green] {t_state.n:,} transport entities")
        specs.append(
            _DomainSpec(
                name="Transport",
                filename="transport_kpis_wide.parquet",
                schema=TRANSPORT_SCHEMA,
                state=t_state,
                hour_fn=_transport_hour,
            )
        )

    # 2. Fixed Broadband
    console.print("  [dim]Loading fixed broadband entities...[/dim]")
    fb_state = _init_fixed_bb(config, rng)
    if fb_state is not None:
        console.print(f"    [green]✓[/green] {fb_state.n:,} fixed broadband entities")
        specs.append(
            _DomainSpec(
                name="Fixed Broadband",
                filename="fixed_broadband_kpis_wide.parquet",
                schema=FIXED_BB_SCHEMA,
                state=fb_state,
                hour_fn=_fixed_bb_hour,
            )
        )

    # 3. Enterprise
    console.print("  [dim]Loading enterprise circuit entities...[/dim]")
    ent_state = _init_enterprise(config, rng)
    if ent_state is not None:
        console.print(f"    [green]✓[/green] {ent_state.n:,} enterprise circuit entities")
        specs.append(
            _DomainSpec(
                name="Enterprise Circuits",
                filename="enterprise_circuit_kpis_wide.parquet",
                schema=ENTERPRISE_SCHEMA,
                state=ent_state,
                hour_fn=_enterprise_hour,
            )
        )

    # 4. Core Network
    console.print("  [dim]Loading core network entities...[/dim]")
    core_state = _init_core(config, rng)
    if core_state is not None:
        console.print(f"    [green]✓[/green] {core_state.n:,} core network entities")
        specs.append(
            _DomainSpec(
                name="Core Network",
                filename="core_element_kpis_wide.parquet",
                schema=CORE_SCHEMA,
                state=core_state,
                hour_fn=_core_hour,
            )
        )

    # 5. Power / Environment
    console.print("  [dim]Loading site inventory for power KPIs...[/dim]")
    pwr_state = _init_power(config, rng)
    if pwr_state is not None:
        console.print(f"    [green]✓[/green] {pwr_state.n:,} sites")
        specs.append(
            _DomainSpec(
                name="Power/Environment",
                filename="power_environment_kpis.parquet",
                schema=POWER_SCHEMA,
                state=pwr_state,
                hour_fn=_power_hour,
            )
        )

    if not specs:
        console.print("[bold red]No domain entities found — nothing to generate. Ensure Phase 2 has run.[/bold red]")
        return

    console.print(f"\n[bold cyan]Generating KPIs for {len(specs)} domains × {total_hours:,} hours...[/bold cyan]\n")

    # ── Generate each domain sequentially ─────────────────────
    summaries: list[dict[str, Any]] = []
    domain_timings: dict[str, float] = {}

    for spec in specs:
        console.print(f"\n[bold blue]━━━ {spec.name} ({spec.state.n:,} entities × {total_hours:,}h) ━━━[/bold blue]")
        t0 = time.time()

        # Each domain gets its own sub-RNG for independence
        domain_rng = np.random.default_rng(rng.integers(0, 2**31))
        summary = _generate_single_domain(spec, config, domain_rng, total_hours)
        elapsed = time.time() - t0
        domain_timings[spec.name] = elapsed
        summaries.append(summary)

        console.print(
            f"  [green]✓[/green] {summary['rows']:,} rows, "
            f"{summary['size_mb']:.1f} MB, {elapsed:.1f}s "
            f"({summary['rows'] / elapsed:,.0f} rows/s)"
        )

        # Release domain state to free memory before next domain
        spec.state = None
        gc.collect()

    # ── Summary table ─────────────────────────────────────────
    total_elapsed = time.time() - step_start
    total_rows = sum(s["rows"] for s in summaries)
    total_size = sum(s["size_mb"] for s in summaries)

    console.print()
    summary_table = Table(
        title="Step 04: Multi-Domain KPI Generation — Summary",
        show_header=True,
    )
    summary_table.add_column("Domain", style="bold", width=22)
    summary_table.add_column("Entities", justify="right", width=10)
    summary_table.add_column("Rows", justify="right", width=14)
    summary_table.add_column("Columns", justify="right", width=8)
    summary_table.add_column("Size", justify="right", width=10)
    summary_table.add_column("Time", justify="right", width=10)
    summary_table.add_column("Rate", justify="right", width=14)

    for s in summaries:
        t = domain_timings.get(s["name"], 0)
        rate = f"{s['rows'] / t:,.0f} r/s" if t > 0 else "N/A"
        time_str = f"{t:.1f}s" if t < 60 else f"{t / 60:.1f}m"
        summary_table.add_row(
            s["name"],
            f"{s['entities']:,}",
            f"{s['rows']:,}",
            str(s["columns"]),
            f"{s['size_mb']:.1f} MB",
            time_str,
            rate,
        )

    summary_table.add_section()
    total_time_str = f"{total_elapsed:.1f}s" if total_elapsed < 60 else f"{total_elapsed / 60:.1f}m"
    summary_table.add_row(
        "[bold]Total[/bold]",
        "",
        f"[bold]{total_rows:,}[/bold]",
        "",
        f"[bold]{total_size:.1f} MB[/bold]",
        f"[bold]{total_time_str}[/bold]",
        f"{total_rows / total_elapsed:,.0f} r/s" if total_elapsed > 0 else "N/A",
    )
    console.print(summary_table)

    console.print(
        f"\n[bold green]✓ Step 04 complete.[/bold green] "
        f"Generated {total_rows:,} rows ({total_size:.1f} MB) "
        f"across {len(summaries)} domain files in {total_time_str}"
    )
