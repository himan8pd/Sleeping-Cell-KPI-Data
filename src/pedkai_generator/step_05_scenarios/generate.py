"""
Step 05: Scenario Injection System.

Injects realistic failure/degradation scenarios into the KPI time series
produced by Phases 3 and 4, using an **overlay strategy** — baseline KPI
files are never mutated.  Instead, two output files are produced:

  - output/scenario_manifest.parquet   — master schedule of all injected scenarios
  - output/scenario_kpi_overrides.parquet — sparse override table

Consumers apply overrides on read:
    effective_value = COALESCE(override, baseline)

Scenario types (8):
  1. Sleeping cell   — subtle traffic/throughput degradation, NO alarm (the whole point)
  2. Congestion      — PRB >85%, throughput collapse, latency spike
  3. Coverage hole   — RSRP/RSRQ degradation in spatial cluster
  4. Hardware fault  — cell availability drop, BLER spike, abrupt onset
  5. Interference    — IoT elevation, CQI/MCS degradation, SINR drop
  6. Transport failure — backhaul link down, cascade to served cells
  7. Power failure   — site-level, all co-located equipment affected
  8. Fibre cut       — cross-domain cascade: transport + cells + fixed BB

Cross-domain cascades walk the ``ground_truth_relationships`` graph to
identify downstream entities affected by infrastructure failures.

Architecture:
  - Deterministic seeding from config.global_seed
  - Memory-efficient: scenarios are generated as small DataFrames, never
    loading the full 8.5 GB+ KPI files
  - Idempotent: re-running Phase 5 regenerates only overlay files

Dependencies: Phase 2 (topology graph), Phase 3 (radio KPI schema),
             Phase 4 (domain KPI schemas)
"""

from __future__ import annotations

import gc
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Union

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
F32 = pa.float32()

# Scenario type identifiers — used in manifest and overrides
SCENARIO_SLEEPING_CELL = "sleeping_cell"
SCENARIO_CONGESTION = "congestion"
SCENARIO_COVERAGE_HOLE = "coverage_hole"
SCENARIO_HARDWARE_FAULT = "hardware_fault"
SCENARIO_INTERFERENCE = "interference"
SCENARIO_TRANSPORT_FAILURE = "transport_failure"
SCENARIO_POWER_FAILURE = "power_failure"
SCENARIO_FIBRE_CUT = "fibre_cut"

ALL_SCENARIO_TYPES = [
    SCENARIO_SLEEPING_CELL,
    SCENARIO_CONGESTION,
    SCENARIO_COVERAGE_HOLE,
    SCENARIO_HARDWARE_FAULT,
    SCENARIO_INTERFERENCE,
    SCENARIO_TRANSPORT_FAILURE,
    SCENARIO_POWER_FAILURE,
    SCENARIO_FIBRE_CUT,
]

# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("scenario_id", pa.string(), nullable=False),
        pa.field("tenant_id", pa.string(), nullable=False),
        pa.field("scenario_type", pa.string(), nullable=False),
        pa.field("severity", pa.string(), nullable=False),  # low / medium / high / critical
        pa.field("primary_entity_id", pa.string(), nullable=False),
        pa.field("primary_entity_type", pa.string(), nullable=False),
        pa.field("primary_domain", pa.string(), nullable=False),
        pa.field("affected_entity_ids", pa.string(), nullable=False),  # JSON array
        pa.field("affected_entity_count", pa.int32(), nullable=False),
        pa.field("start_hour", pa.int32(), nullable=False),
        pa.field("end_hour", pa.int32(), nullable=False),
        pa.field("duration_hours", pa.int32(), nullable=False),
        pa.field("cascade_chain", pa.string()),  # JSON array of cascade steps, null if none
        pa.field("ramp_up_hours", pa.int32(), nullable=False),
        pa.field("ramp_down_hours", pa.int32(), nullable=False),
        pa.field("parameters_json", pa.string()),  # JSON dict of scenario-specific params
    ]
)

OVERRIDES_SCHEMA = pa.schema(
    [
        pa.field("entity_id", pa.string(), nullable=False),
        pa.field("tenant_id", pa.string(), nullable=False),
        pa.field("timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("kpi_column", pa.string(), nullable=False),
        pa.field("override_value", F32, nullable=False),
        pa.field("scenario_id", pa.string(), nullable=False),
        pa.field("scenario_type", pa.string(), nullable=False),
        pa.field("source_file", pa.string(), nullable=False),  # which baseline file this overrides
    ]
)


# ---------------------------------------------------------------------------
# Helper dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ScenarioInstance:
    """A single scenario injection instance."""

    scenario_id: str
    scenario_type: str
    severity: str
    primary_entity_id: str
    primary_entity_type: str
    primary_domain: str
    affected_entity_ids: list[str]
    start_hour: int
    end_hour: int
    ramp_up_hours: int = 1
    ramp_down_hours: int = 1
    cascade_chain: list[dict[str, str]] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class KPIOverride:
    """A single KPI override row."""

    entity_id: str
    timestamp: datetime
    kpi_column: str
    override_value: Union[float, "np.floating[Any]"]
    scenario_id: str
    scenario_type: str
    source_file: str


# ---------------------------------------------------------------------------
# Topology graph loader and cascade walker
# ---------------------------------------------------------------------------


def _load_entities(config: GeneratorConfig) -> pl.DataFrame:
    """Load ground_truth_entities.parquet."""
    path = config.paths.output_dir / "ground_truth_entities.parquet"
    if not path.exists():
        raise FileNotFoundError(f"ground_truth_entities.parquet not found at {path}. Run Phase 2 first.")
    return pl.read_parquet(str(path))


def _load_relationships(config: GeneratorConfig) -> pl.DataFrame:
    """Load ground_truth_relationships.parquet."""
    path = config.paths.output_dir / "ground_truth_relationships.parquet"
    if not path.exists():
        raise FileNotFoundError(f"ground_truth_relationships.parquet not found at {path}. Run Phase 2 first.")
    return pl.read_parquet(str(path))


def _load_sites(config: GeneratorConfig) -> pl.DataFrame:
    """Load sites from intermediate."""
    path = config.paths.intermediate_dir / "sites.parquet"
    if not path.exists():
        raise FileNotFoundError(f"sites.parquet not found at {path}. Run Phase 1 first.")
    return pl.read_parquet(str(path))


def _load_cells(config: GeneratorConfig) -> pl.DataFrame:
    """Load cells from intermediate."""
    path = config.paths.intermediate_dir / "cells.parquet"
    if not path.exists():
        raise FileNotFoundError(f"cells.parquet not found at {path}. Run Phase 1 first.")
    return pl.read_parquet(str(path))


class TopologyGraph:
    """
    Lightweight in-memory topology graph for cascade walking.

    Stores adjacency lists built from ground_truth_relationships.
    Supports finding all downstream entities from a given entity
    (BFS/DFS traversal following relationship direction).
    """

    def __init__(self, rels_df: pl.DataFrame, entities_df: pl.DataFrame):
        # Build forward adjacency: from_entity_id -> list of (to_entity_id, to_entity_type, rel_type)
        self._forward: dict[str, list[tuple[str, str, str]]] = {}
        # Build reverse adjacency: to_entity_id -> list of (from_entity_id, from_entity_type, rel_type)
        self._reverse: dict[str, list[tuple[str, str, str]]] = {}
        # Entity type lookup
        self._entity_type: dict[str, str] = {}
        # Entity domain lookup
        self._entity_domain: dict[str, str] = {}
        # Site ID lookup — entity_id → site_id (via relationships or direct)
        self._entity_site: dict[str, str] = {}

        # Build entity lookups
        for row in entities_df.iter_rows(named=True):
            eid = row["entity_id"]
            self._entity_type[eid] = row["entity_type"]
            if "domain" in row:
                self._entity_domain[eid] = row.get("domain", "unknown")

        # Build adjacency
        from_ids = rels_df["from_entity_id"].to_list()
        from_types = rels_df["from_entity_type"].to_list()
        to_ids = rels_df["to_entity_id"].to_list()
        to_types = rels_df["to_entity_type"].to_list()
        rel_types = rels_df["relationship_type"].to_list()

        for i in range(len(from_ids)):
            fid = from_ids[i]
            tid = to_ids[i]
            ft = from_types[i]
            tt = to_types[i]
            rt = rel_types[i]

            if fid not in self._forward:
                self._forward[fid] = []
            self._forward[fid].append((tid, tt, rt))

            if tid not in self._reverse:
                self._reverse[tid] = []
            self._reverse[tid].append((fid, ft, rt))

        # Build site mapping from HOSTS / LOCATED_AT relationships
        site_rel_types = {"HOSTS", "LOCATED_AT", "INSTALLED_AT", "MOUNTED_ON", "SITE_HAS_CABINET"}
        for i in range(len(from_ids)):
            ftype = from_types[i]
            rt = rel_types[i]
            if ftype == "SITE" or rt in site_rel_types:
                # SITE -> child entity: map child to site
                if from_types[i] == "SITE":
                    self._entity_site[to_ids[i]] = from_ids[i]

    def get_type(self, entity_id: str) -> str:
        return self._entity_type.get(entity_id, "UNKNOWN")

    def get_domain(self, entity_id: str) -> str:
        return self._entity_domain.get(entity_id, "unknown")

    def get_site(self, entity_id: str) -> str | None:
        return self._entity_site.get(entity_id)

    def downstream(
        self,
        entity_id: str,
        max_depth: int = 5,
        type_filter: set[str] | None = None,
    ) -> list[tuple[str, str, list[dict[str, str]]]]:
        """
        BFS to find all downstream entities reachable from entity_id.

        Returns list of (entity_id, entity_type, cascade_path) tuples.
        cascade_path is a list of dicts recording each hop.
        """
        visited: set[str] = {entity_id}
        results: list[tuple[str, str, list[dict[str, str]]]] = []
        # Queue: (entity_id, depth, path_so_far)
        queue: list[tuple[str, int, list[dict[str, str]]]] = [(entity_id, 0, [])]

        while queue:
            current, depth, path = queue.pop(0)
            if depth >= max_depth:
                continue
            for child_id, child_type, rel_type in self._forward.get(current, []):
                if child_id in visited:
                    continue
                if type_filter is not None and child_type not in type_filter:
                    continue
                visited.add(child_id)
                new_path = path + [
                    {
                        "from": current,
                        "to": child_id,
                        "rel": rel_type,
                        "to_type": child_type,
                    }
                ]
                results.append((child_id, child_type, new_path))
                queue.append((child_id, depth + 1, new_path))

        return results

    def entities_at_site(self, site_id: str) -> list[tuple[str, str]]:
        """Return all (entity_id, entity_type) pairs located at a given site."""
        results: list[tuple[str, str]] = []
        for eid, sid in self._entity_site.items():
            if sid == site_id:
                etype = self.get_type(eid)
                results.append((eid, etype))
        return results

    def find_cells_at_site(self, site_id: str) -> list[str]:
        """Return cell entity IDs at a site (LTE_CELL, NR_CELL)."""
        cell_types = {"LTE_CELL", "NR_CELL", "NR_NSA_CELL"}
        results = []
        for child_id, child_type, _ in self._forward.get(site_id, []):
            if child_type in cell_types:
                results.append(child_id)
        # Also check deeper: site -> cabinet -> bbu -> cell, site -> enodeb -> cell
        for child_id, child_type, _ in self._forward.get(site_id, []):
            for grandchild_id, gc_type, _ in self._forward.get(child_id, []):
                if gc_type in cell_types:
                    if grandchild_id not in results:
                        results.append(grandchild_id)
                # Go one more level: site -> cabinet -> bbu -> rru -> cell
                for ggc_id, ggc_type, _ in self._forward.get(grandchild_id, []):
                    if ggc_type in cell_types:
                        if ggc_id not in results:
                            results.append(ggc_id)
                    for gggc_id, gggc_type, _ in self._forward.get(ggc_id, []):
                        if gggc_type in cell_types:
                            if gggc_id not in results:
                                results.append(gggc_id)
        return results

    def find_entities_served_by(self, entity_id: str, target_types: set[str], max_depth: int = 5) -> list[str]:
        """Find all entities of target_types reachable downstream from entity_id."""
        results = []
        for eid, etype, _ in self.downstream(entity_id, max_depth=max_depth):
            if etype in target_types:
                results.append(eid)
        return results


# ---------------------------------------------------------------------------
# Severity assignment
# ---------------------------------------------------------------------------

_SEVERITY_MAP = {
    SCENARIO_SLEEPING_CELL: ["low", "medium"],
    SCENARIO_CONGESTION: ["medium", "high"],
    SCENARIO_COVERAGE_HOLE: ["low", "medium"],
    SCENARIO_HARDWARE_FAULT: ["high", "critical"],
    SCENARIO_INTERFERENCE: ["low", "medium", "high"],
    SCENARIO_TRANSPORT_FAILURE: ["high", "critical"],
    SCENARIO_POWER_FAILURE: ["critical"],
    SCENARIO_FIBRE_CUT: ["critical"],
}


def _pick_severity(scenario_type: str, rng: np.random.Generator) -> str:
    options = _SEVERITY_MAP.get(scenario_type, ["medium"])
    return str(rng.choice(options))


# ---------------------------------------------------------------------------
# Duration helpers
# ---------------------------------------------------------------------------


def _pick_duration(
    scenario_type: str,
    rng: np.random.Generator,
    total_hours: int,
) -> tuple[int, int, int, int]:
    """
    Pick (start_hour, end_hour, ramp_up_hours, ramp_down_hours) for a scenario.

    Different scenario types have characteristic duration distributions.
    """
    if scenario_type == SCENARIO_SLEEPING_CELL:
        # Sleeping cells are long-duration — days to weeks (the whole point is
        # they persist undetected). 72h to 504h (3-21 days).
        duration = int(rng.integers(72, min(504, total_hours - 24)))
        ramp_up = int(rng.integers(6, 24))  # slow onset
        ramp_down = 0  # sleeping cells don't self-resolve
    elif scenario_type == SCENARIO_CONGESTION:
        # Congestion episodes: 2h to 48h, often during busy hours
        duration = int(rng.integers(2, min(48, total_hours - 12)))
        ramp_up = int(rng.integers(1, 4))
        ramp_down = int(rng.integers(1, 4))
    elif scenario_type == SCENARIO_COVERAGE_HOLE:
        # Coverage holes are persistent — environment change or antenna issue
        duration = int(rng.integers(48, min(360, total_hours - 24)))
        ramp_up = int(rng.integers(1, 12))
        ramp_down = int(rng.integers(1, 12))
    elif scenario_type == SCENARIO_HARDWARE_FAULT:
        # Hardware faults: abrupt onset, variable duration (4h to 168h / 1 week)
        duration = int(rng.integers(4, min(168, total_hours - 12)))
        ramp_up = 0  # abrupt
        ramp_down = 0  # abrupt recovery (replacement/restart)
    elif scenario_type == SCENARIO_INTERFERENCE:
        # Interference: hours to days
        duration = int(rng.integers(6, min(120, total_hours - 12)))
        ramp_up = int(rng.integers(1, 6))
        ramp_down = int(rng.integers(1, 6))
    elif scenario_type == SCENARIO_TRANSPORT_FAILURE:
        # Transport failures: 1h to 24h (MTTR for transport)
        duration = int(rng.integers(1, min(24, total_hours - 6)))
        ramp_up = 0  # abrupt
        ramp_down = 0  # abrupt
    elif scenario_type == SCENARIO_POWER_FAILURE:
        # Power failures: 1h to 12h
        duration = int(rng.integers(1, min(12, total_hours - 4)))
        ramp_up = 0  # abrupt
        ramp_down = 0  # abrupt (power restored)
    elif scenario_type == SCENARIO_FIBRE_CUT:
        # Fibre cuts: 2h to 48h (repair crew dispatch)
        duration = int(rng.integers(2, min(48, total_hours - 12)))
        ramp_up = 0  # abrupt
        ramp_down = 0  # abrupt
    else:
        duration = int(rng.integers(4, min(48, total_hours - 12)))
        ramp_up = 1
        ramp_down = 1

    # Pick a start hour — leave room for full duration + ramp_down
    latest_start = total_hours - duration - ramp_down - 1
    if latest_start < ramp_up:
        latest_start = ramp_up
    start_hour = int(rng.integers(ramp_up, max(ramp_up + 1, latest_start)))
    end_hour = min(start_hour + duration, total_hours - 1)

    return start_hour, end_hour, ramp_up, ramp_down


def _ramp_factor(
    hour_idx: int,
    start_hour: int,
    end_hour: int,
    ramp_up_hours: int,
    ramp_down_hours: int,
) -> float:
    """
    Compute a 0.0→1.0 ramp factor for a given hour within a scenario window.

    Returns 0.0 outside the scenario window.
    Returns a smooth ramp from 0→1 during ramp-up, 1.0 at full severity,
    and 1→0 during ramp-down.
    """
    if hour_idx < start_hour or hour_idx > end_hour:
        return 0.0

    # Ramp-up phase
    if ramp_up_hours > 0 and hour_idx < start_hour + ramp_up_hours:
        t = (hour_idx - start_hour) / ramp_up_hours
        # Smooth (cosine) ramp
        return 0.5 * (1.0 - np.cos(np.pi * t))

    # Ramp-down phase
    if ramp_down_hours > 0 and hour_idx > end_hour - ramp_down_hours:
        t = (end_hour - hour_idx) / ramp_down_hours
        return 0.5 * (1.0 - np.cos(np.pi * t))

    return 1.0


# ---------------------------------------------------------------------------
# Scenario-specific KPI override generators
# ---------------------------------------------------------------------------

# Map scenario type → which baseline parquet file(s) it overrides
_SOURCE_FILE_MAP = {
    "radio": "kpi_metrics_wide.parquet",
    "transport": "transport_kpis_wide.parquet",
    "fixed_bb": "fixed_broadband_kpis_wide.parquet",
    "enterprise": "enterprise_circuit_kpis_wide.parquet",
    "core": "core_element_kpis_wide.parquet",
    "power": "power_environment_kpis.parquet",
}


def _generate_sleeping_cell_overrides(
    scenario: ScenarioInstance,
    rng: np.random.Generator,
    total_hours: int,
    tenant_id: str,
) -> list[KPIOverride]:
    """
    Sleeping cell: subtle degradation that doesn't trigger standard alarms.

    Key signature:
    - Traffic volume drops 30-70% (users camp elsewhere)
    - Throughput (DL/UL) drops 20-50%
    - Active UEs drop 30-60%
    - PRB utilisation drops (less traffic, but NOT from congestion)
    - Cell availability remains ~100% (it's "up" but underperforming)
    - RSRP/RSRQ/SINR may have slight degradation or remain normal
    - NO alarm is generated (that's the whole problem)

    The subtlety is critical — this must NOT look like an obvious outage.
    """
    overrides: list[KPIOverride] = []
    severity_factor = {"low": 0.5, "medium": 0.75, "high": 1.0}.get(scenario.severity, 0.6)

    # Per-cell randomised degradation factors
    n_cells = len(scenario.affected_entity_ids)
    traffic_drop = rng.uniform(0.30, 0.70, n_cells) * severity_factor
    throughput_drop_dl = rng.uniform(0.20, 0.50, n_cells) * severity_factor
    throughput_drop_ul = rng.uniform(0.15, 0.40, n_cells) * severity_factor
    ue_drop = rng.uniform(0.30, 0.60, n_cells) * severity_factor
    prb_drop = rng.uniform(0.10, 0.35, n_cells) * severity_factor
    # Subtle SINR degradation (0-3 dB) — not always present
    sinr_degrade = rng.uniform(0.0, 3.0, n_cells) * severity_factor * rng.choice([0, 1], n_cells, p=[0.4, 0.6])

    for hour_idx in range(scenario.start_hour, scenario.end_hour + 1):
        ramp = _ramp_factor(
            hour_idx,
            scenario.start_hour,
            scenario.end_hour,
            scenario.ramp_up_hours,
            scenario.ramp_down_hours,
        )
        if ramp < 0.01:
            continue

        ts = SIMULATION_EPOCH + timedelta(hours=hour_idx)

        # Add some hourly jitter so it doesn't look perfectly uniform
        hour_jitter = rng.normal(0, 0.05, n_cells)

        for i, eid in enumerate(scenario.affected_entity_ids):
            jitter = max(0.0, min(1.0, ramp + hour_jitter[i]))

            # Traffic volume: multiply baseline by (1 - drop * ramp)
            tv_factor = max(0.05, 1.0 - traffic_drop[i] * jitter)
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "traffic_volume_gb",
                    np.float32(tv_factor),  # This is a multiplier; we store it as-is
                    scenario.scenario_id,
                    SCENARIO_SLEEPING_CELL,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

            # DL throughput
            dl_factor = max(0.1, 1.0 - throughput_drop_dl[i] * jitter)
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "dl_throughput_mbps",
                    np.float32(dl_factor),
                    scenario.scenario_id,
                    SCENARIO_SLEEPING_CELL,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

            # UL throughput
            ul_factor = max(0.1, 1.0 - throughput_drop_ul[i] * jitter)
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "ul_throughput_mbps",
                    np.float32(ul_factor),
                    scenario.scenario_id,
                    SCENARIO_SLEEPING_CELL,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

            # Active UEs
            ue_factor = max(0.05, 1.0 - ue_drop[i] * jitter)
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "active_ue_avg",
                    np.float32(ue_factor),
                    scenario.scenario_id,
                    SCENARIO_SLEEPING_CELL,
                    _SOURCE_FILE_MAP["radio"],
                )
            )
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "active_ue_max",
                    np.float32(ue_factor),
                    scenario.scenario_id,
                    SCENARIO_SLEEPING_CELL,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

            # PRB utilisation drops (less load)
            prb_factor = max(0.05, 1.0 - prb_drop[i] * jitter)
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "prb_utilization_dl",
                    np.float32(prb_factor),
                    scenario.scenario_id,
                    SCENARIO_SLEEPING_CELL,
                    _SOURCE_FILE_MAP["radio"],
                )
            )
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "prb_utilization_ul",
                    np.float32(prb_factor),
                    scenario.scenario_id,
                    SCENARIO_SLEEPING_CELL,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

            # Subtle SINR degradation (additive, negative)
            if sinr_degrade[i] > 0.1:
                overrides.append(
                    KPIOverride(
                        eid,
                        ts,
                        "sinr_db",
                        np.float32(-sinr_degrade[i] * jitter),
                        scenario.scenario_id,
                        SCENARIO_SLEEPING_CELL,
                        _SOURCE_FILE_MAP["radio"],
                    )
                )

            # PDCP volumes drop (correlated with traffic)
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "pdcp_dl_volume_mb",
                    np.float32(tv_factor),
                    scenario.scenario_id,
                    SCENARIO_SLEEPING_CELL,
                    _SOURCE_FILE_MAP["radio"],
                )
            )
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "pdcp_ul_volume_mb",
                    np.float32(tv_factor * 0.3),  # UL typically lower
                    scenario.scenario_id,
                    SCENARIO_SLEEPING_CELL,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

    return overrides


def _generate_congestion_overrides(
    scenario: ScenarioInstance,
    rng: np.random.Generator,
    total_hours: int,
    tenant_id: str,
) -> list[KPIOverride]:
    """
    Congestion: PRB utilisation spikes >85%, throughput collapses,
    latency spikes, packet loss increases.
    """
    overrides: list[KPIOverride] = []
    severity_factor = {"medium": 0.7, "high": 1.0}.get(scenario.severity, 0.85)

    n_cells = len(scenario.affected_entity_ids)
    prb_spike = rng.uniform(85.0, 99.0, n_cells) * severity_factor
    throughput_crush = rng.uniform(0.50, 0.80, n_cells) * severity_factor
    latency_spike = rng.uniform(30.0, 200.0, n_cells) * severity_factor
    pkt_loss_spike = rng.uniform(1.0, 8.0, n_cells) * severity_factor
    jitter_spike = rng.uniform(5.0, 30.0, n_cells) * severity_factor

    for hour_idx in range(scenario.start_hour, scenario.end_hour + 1):
        ramp = _ramp_factor(
            hour_idx,
            scenario.start_hour,
            scenario.end_hour,
            scenario.ramp_up_hours,
            scenario.ramp_down_hours,
        )
        if ramp < 0.01:
            continue

        ts = SIMULATION_EPOCH + timedelta(hours=hour_idx)
        h_jitter = rng.normal(0, 0.03, n_cells)

        for i, eid in enumerate(scenario.affected_entity_ids):
            j = max(0.0, min(1.0, ramp + h_jitter[i]))

            # PRB utilisation overridden to high absolute value
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "prb_utilization_dl",
                    np.float32(prb_spike[i] * j + (1.0 - j) * 50.0),
                    scenario.scenario_id,
                    SCENARIO_CONGESTION,
                    _SOURCE_FILE_MAP["radio"],
                )
            )
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "prb_utilization_ul",
                    np.float32((prb_spike[i] - 10.0) * j + (1.0 - j) * 40.0),
                    scenario.scenario_id,
                    SCENARIO_CONGESTION,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

            # Throughput collapse (multiplicative factor)
            dl_factor = max(0.1, 1.0 - throughput_crush[i] * j)
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "dl_throughput_mbps",
                    np.float32(dl_factor),
                    scenario.scenario_id,
                    SCENARIO_CONGESTION,
                    _SOURCE_FILE_MAP["radio"],
                )
            )
            ul_factor = max(0.1, 1.0 - throughput_crush[i] * 0.8 * j)
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "ul_throughput_mbps",
                    np.float32(ul_factor),
                    scenario.scenario_id,
                    SCENARIO_CONGESTION,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

            # Latency spike (absolute override)
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "latency_ms",
                    np.float32(latency_spike[i] * j + 15.0),
                    scenario.scenario_id,
                    SCENARIO_CONGESTION,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

            # Jitter
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "jitter_ms",
                    np.float32(jitter_spike[i] * j + 3.0),
                    scenario.scenario_id,
                    SCENARIO_CONGESTION,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

            # Packet loss
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "packet_loss_pct",
                    np.float32(pkt_loss_spike[i] * j),
                    scenario.scenario_id,
                    SCENARIO_CONGESTION,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

            # CCE utilisation spikes
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "cce_utilization_pct",
                    np.float32(min(95.0, 70.0 + 25.0 * j)),
                    scenario.scenario_id,
                    SCENARIO_CONGESTION,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

            # Active UE spike
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "active_ue_avg",
                    np.float32(1.0 + 0.8 * severity_factor * j),  # multiplier
                    scenario.scenario_id,
                    SCENARIO_CONGESTION,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

    return overrides


def _generate_coverage_hole_overrides(
    scenario: ScenarioInstance,
    rng: np.random.Generator,
    total_hours: int,
    tenant_id: str,
) -> list[KPIOverride]:
    """
    Coverage hole: RSRP/RSRQ degradation, higher BLER, reduced throughput.
    Cells in a spatial cluster are affected (e.g. antenna tilt change, new building).
    """
    overrides: list[KPIOverride] = []
    severity_factor = {"low": 0.5, "medium": 0.75}.get(scenario.severity, 0.6)

    n_cells = len(scenario.affected_entity_ids)
    rsrp_drop = rng.uniform(8.0, 20.0, n_cells) * severity_factor  # dB drop
    rsrq_drop = rng.uniform(3.0, 10.0, n_cells) * severity_factor
    sinr_drop = rng.uniform(4.0, 12.0, n_cells) * severity_factor
    bler_increase = rng.uniform(2.0, 15.0, n_cells) * severity_factor

    for hour_idx in range(scenario.start_hour, scenario.end_hour + 1):
        ramp = _ramp_factor(
            hour_idx,
            scenario.start_hour,
            scenario.end_hour,
            scenario.ramp_up_hours,
            scenario.ramp_down_hours,
        )
        if ramp < 0.01:
            continue

        ts = SIMULATION_EPOCH + timedelta(hours=hour_idx)

        for i, eid in enumerate(scenario.affected_entity_ids):
            # RSRP drop (additive, negative)
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "rsrp_dbm",
                    np.float32(-rsrp_drop[i] * ramp),
                    scenario.scenario_id,
                    SCENARIO_COVERAGE_HOLE,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

            # RSRQ drop
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "rsrq_db",
                    np.float32(-rsrq_drop[i] * ramp),
                    scenario.scenario_id,
                    SCENARIO_COVERAGE_HOLE,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

            # SINR drop
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "sinr_db",
                    np.float32(-sinr_drop[i] * ramp),
                    scenario.scenario_id,
                    SCENARIO_COVERAGE_HOLE,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

            # CQI degrades (absolute, pulled down)
            cqi_degraded = max(1.0, 8.0 - 5.0 * severity_factor * ramp)
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "cqi_mean",
                    np.float32(cqi_degraded),
                    scenario.scenario_id,
                    SCENARIO_COVERAGE_HOLE,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

            # BLER increases
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "dl_bler_pct",
                    np.float32(bler_increase[i] * ramp),
                    scenario.scenario_id,
                    SCENARIO_COVERAGE_HOLE,
                    _SOURCE_FILE_MAP["radio"],
                )
            )
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "ul_bler_pct",
                    np.float32(bler_increase[i] * 0.7 * ramp),
                    scenario.scenario_id,
                    SCENARIO_COVERAGE_HOLE,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

            # Throughput reduction
            tp_factor = max(0.2, 1.0 - 0.4 * severity_factor * ramp)
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "dl_throughput_mbps",
                    np.float32(tp_factor),
                    scenario.scenario_id,
                    SCENARIO_COVERAGE_HOLE,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

            # Handover attempts increase (UEs trying to move to better cells)
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "ho_attempt",
                    np.float32(1.0 + 1.5 * severity_factor * ramp),  # multiplier
                    scenario.scenario_id,
                    SCENARIO_COVERAGE_HOLE,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

            # Handover success rate drops
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "ho_success_rate",
                    np.float32(max(70.0, 95.0 - 15.0 * severity_factor * ramp)),
                    scenario.scenario_id,
                    SCENARIO_COVERAGE_HOLE,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

    return overrides


def _generate_hardware_fault_overrides(
    scenario: ScenarioInstance,
    rng: np.random.Generator,
    total_hours: int,
    tenant_id: str,
) -> list[KPIOverride]:
    """
    Hardware fault: abrupt cell availability drop, high BLER, possible
    complete outage. Distinct from sleeping cell by being sudden and obvious.
    """
    overrides: list[KPIOverride] = []
    severity_factor = {"high": 0.8, "critical": 1.0}.get(scenario.severity, 0.9)

    n_cells = len(scenario.affected_entity_ids)
    # Some cells go to complete outage, others partial
    is_full_outage = rng.random(n_cells) < (0.3 * severity_factor)
    avail_degraded = np.where(is_full_outage, 0.0, rng.uniform(20.0, 70.0, n_cells))

    for hour_idx in range(scenario.start_hour, scenario.end_hour + 1):
        ramp = _ramp_factor(
            hour_idx,
            scenario.start_hour,
            scenario.end_hour,
            scenario.ramp_up_hours,
            scenario.ramp_down_hours,
        )
        if ramp < 0.01:
            continue

        ts = SIMULATION_EPOCH + timedelta(hours=hour_idx)

        for i, eid in enumerate(scenario.affected_entity_ids):
            if is_full_outage[i] and ramp > 0.5:
                # Complete cell outage
                overrides.append(
                    KPIOverride(
                        eid,
                        ts,
                        "cell_availability_pct",
                        np.float32(0.0),
                        scenario.scenario_id,
                        SCENARIO_HARDWARE_FAULT,
                        _SOURCE_FILE_MAP["radio"],
                    )
                )
                overrides.append(
                    KPIOverride(
                        eid,
                        ts,
                        "dl_throughput_mbps",
                        np.float32(0.0),
                        scenario.scenario_id,
                        SCENARIO_HARDWARE_FAULT,
                        _SOURCE_FILE_MAP["radio"],
                    )
                )
                overrides.append(
                    KPIOverride(
                        eid,
                        ts,
                        "ul_throughput_mbps",
                        np.float32(0.0),
                        scenario.scenario_id,
                        SCENARIO_HARDWARE_FAULT,
                        _SOURCE_FILE_MAP["radio"],
                    )
                )
                overrides.append(
                    KPIOverride(
                        eid,
                        ts,
                        "active_ue_avg",
                        np.float32(0.0),
                        scenario.scenario_id,
                        SCENARIO_HARDWARE_FAULT,
                        _SOURCE_FILE_MAP["radio"],
                    )
                )
                overrides.append(
                    KPIOverride(
                        eid,
                        ts,
                        "traffic_volume_gb",
                        np.float32(0.0),
                        scenario.scenario_id,
                        SCENARIO_HARDWARE_FAULT,
                        _SOURCE_FILE_MAP["radio"],
                    )
                )
            else:
                # Partial degradation
                avail = avail_degraded[i] * ramp + 100.0 * (1.0 - ramp)
                overrides.append(
                    KPIOverride(
                        eid,
                        ts,
                        "cell_availability_pct",
                        np.float32(avail),
                        scenario.scenario_id,
                        SCENARIO_HARDWARE_FAULT,
                        _SOURCE_FILE_MAP["radio"],
                    )
                )

                # BLER spike
                overrides.append(
                    KPIOverride(
                        eid,
                        ts,
                        "dl_bler_pct",
                        np.float32(min(30.0, 15.0 * severity_factor * ramp)),
                        scenario.scenario_id,
                        SCENARIO_HARDWARE_FAULT,
                        _SOURCE_FILE_MAP["radio"],
                    )
                )
                overrides.append(
                    KPIOverride(
                        eid,
                        ts,
                        "ul_bler_pct",
                        np.float32(min(25.0, 12.0 * severity_factor * ramp)),
                        scenario.scenario_id,
                        SCENARIO_HARDWARE_FAULT,
                        _SOURCE_FILE_MAP["radio"],
                    )
                )

                # Throughput reduction
                tp_factor = max(0.1, 1.0 - 0.6 * severity_factor * ramp)
                overrides.append(
                    KPIOverride(
                        eid,
                        ts,
                        "dl_throughput_mbps",
                        np.float32(tp_factor),
                        scenario.scenario_id,
                        SCENARIO_HARDWARE_FAULT,
                        _SOURCE_FILE_MAP["radio"],
                    )
                )

                # RLC retransmission spikes
                overrides.append(
                    KPIOverride(
                        eid,
                        ts,
                        "dl_rlc_retransmission_pct",
                        np.float32(min(20.0, 10.0 * severity_factor * ramp)),
                        scenario.scenario_id,
                        SCENARIO_HARDWARE_FAULT,
                        _SOURCE_FILE_MAP["radio"],
                    )
                )

                # RACH success drops
                overrides.append(
                    KPIOverride(
                        eid,
                        ts,
                        "rach_success_rate",
                        np.float32(max(60.0, 95.0 - 25.0 * severity_factor * ramp)),
                        scenario.scenario_id,
                        SCENARIO_HARDWARE_FAULT,
                        _SOURCE_FILE_MAP["radio"],
                    )
                )

    return overrides


def _generate_interference_overrides(
    scenario: ScenarioInstance,
    rng: np.random.Generator,
    total_hours: int,
    tenant_id: str,
) -> list[KPIOverride]:
    """
    Interference: IoT elevation, SINR drop, CQI/MCS degradation.
    Can be intermittent (external interference source may be periodic).
    """
    overrides: list[KPIOverride] = []
    severity_factor = {"low": 0.5, "medium": 0.75, "high": 1.0}.get(scenario.severity, 0.7)

    n_cells = len(scenario.affected_entity_ids)
    iot_elevation = rng.uniform(3.0, 12.0, n_cells) * severity_factor  # dB
    sinr_impact = rng.uniform(3.0, 10.0, n_cells) * severity_factor

    # Some interference sources are periodic (e.g. industrial equipment)
    is_periodic = rng.random(n_cells) < 0.3
    periodic_period_h = rng.choice([4, 6, 8, 12, 24], n_cells)
    periodic_duty_cycle = rng.uniform(0.3, 0.7, n_cells)

    for hour_idx in range(scenario.start_hour, scenario.end_hour + 1):
        ramp = _ramp_factor(
            hour_idx,
            scenario.start_hour,
            scenario.end_hour,
            scenario.ramp_up_hours,
            scenario.ramp_down_hours,
        )
        if ramp < 0.01:
            continue

        ts = SIMULATION_EPOCH + timedelta(hours=hour_idx)

        for i, eid in enumerate(scenario.affected_entity_ids):
            # Periodic interference: only active part of the time
            if is_periodic[i]:
                phase = (hour_idx % periodic_period_h[i]) / periodic_period_h[i]
                if phase > periodic_duty_cycle[i]:
                    continue  # interference source is OFF this hour

            j = ramp

            # IoT elevation
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "interference_iot_db",
                    np.float32(iot_elevation[i] * j),  # additive
                    scenario.scenario_id,
                    SCENARIO_INTERFERENCE,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

            # SINR degradation
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "sinr_db",
                    np.float32(-sinr_impact[i] * j),  # additive, negative
                    scenario.scenario_id,
                    SCENARIO_INTERFERENCE,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

            # CQI degrades
            cqi_drop = min(8.0, 4.0 * severity_factor * j)
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "cqi_mean",
                    np.float32(-cqi_drop),  # additive, negative
                    scenario.scenario_id,
                    SCENARIO_INTERFERENCE,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

            # MCS degrades
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "mcs_dl",
                    np.float32(-min(10.0, 6.0 * severity_factor * j)),
                    scenario.scenario_id,
                    SCENARIO_INTERFERENCE,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

            # Throughput degrades
            tp_factor = max(0.2, 1.0 - 0.4 * severity_factor * j)
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "dl_throughput_mbps",
                    np.float32(tp_factor),
                    scenario.scenario_id,
                    SCENARIO_INTERFERENCE,
                    _SOURCE_FILE_MAP["radio"],
                )
            )
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "ul_throughput_mbps",
                    np.float32(tp_factor * 1.1),  # UL slightly less affected
                    scenario.scenario_id,
                    SCENARIO_INTERFERENCE,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

            # BLER increases
            overrides.append(
                KPIOverride(
                    eid,
                    ts,
                    "dl_bler_pct",
                    np.float32(min(10.0, 5.0 * severity_factor * j)),
                    scenario.scenario_id,
                    SCENARIO_INTERFERENCE,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

    return overrides


def _generate_transport_failure_overrides(
    scenario: ScenarioInstance,
    rng: np.random.Generator,
    total_hours: int,
    tenant_id: str,
    graph: TopologyGraph,
) -> list[KPIOverride]:
    """
    Transport failure: backhaul link goes down.
    Cascade: transport entity → all cells served by that link.

    Transport entity itself: utilisation → 0%, errors → spike.
    Downstream cells: throughput collapse, high packet loss.
    """
    overrides: list[KPIOverride] = []
    primary_eid = scenario.primary_entity_id

    for hour_idx in range(scenario.start_hour, scenario.end_hour + 1):
        ramp = _ramp_factor(
            hour_idx,
            scenario.start_hour,
            scenario.end_hour,
            scenario.ramp_up_hours,
            scenario.ramp_down_hours,
        )
        if ramp < 0.01:
            continue

        ts = SIMULATION_EPOCH + timedelta(hours=hour_idx)

        # Transport entity itself
        overrides.append(
            KPIOverride(
                primary_eid,
                ts,
                "interface_utilization_in_pct",
                np.float32(0.0),
                scenario.scenario_id,
                SCENARIO_TRANSPORT_FAILURE,
                _SOURCE_FILE_MAP["transport"],
            )
        )
        overrides.append(
            KPIOverride(
                primary_eid,
                ts,
                "interface_utilization_out_pct",
                np.float32(0.0),
                scenario.scenario_id,
                SCENARIO_TRANSPORT_FAILURE,
                _SOURCE_FILE_MAP["transport"],
            )
        )
        overrides.append(
            KPIOverride(
                primary_eid,
                ts,
                "interface_errors_in",
                np.float32(1000.0 * ramp),
                scenario.scenario_id,
                SCENARIO_TRANSPORT_FAILURE,
                _SOURCE_FILE_MAP["transport"],
            )
        )
        overrides.append(
            KPIOverride(
                primary_eid,
                ts,
                "interface_errors_out",
                np.float32(1000.0 * ramp),
                scenario.scenario_id,
                SCENARIO_TRANSPORT_FAILURE,
                _SOURCE_FILE_MAP["transport"],
            )
        )

        # LSP/L3VPN may have utilisation → 0
        transport_type = graph.get_type(primary_eid)
        if transport_type in ("MICROWAVE_LINK",):
            overrides.append(
                KPIOverride(
                    primary_eid,
                    ts,
                    "microwave_availability_pct",
                    np.float32(0.0),
                    scenario.scenario_id,
                    SCENARIO_TRANSPORT_FAILURE,
                    _SOURCE_FILE_MAP["transport"],
                )
            )

        # Downstream cells: throughput collapse, high packet loss
        cell_types = {"LTE_CELL", "NR_CELL", "NR_NSA_CELL"}
        downstream_cells = [
            eid for eid in scenario.affected_entity_ids if eid != primary_eid and graph.get_type(eid) in cell_types
        ]
        for cell_id in downstream_cells:
            # Cells behind a failed backhaul: severe throughput drop
            overrides.append(
                KPIOverride(
                    cell_id,
                    ts,
                    "dl_throughput_mbps",
                    np.float32(max(0.0, 0.05 * (1.0 - ramp))),  # near zero
                    scenario.scenario_id,
                    SCENARIO_TRANSPORT_FAILURE,
                    _SOURCE_FILE_MAP["radio"],
                )
            )
            overrides.append(
                KPIOverride(
                    cell_id,
                    ts,
                    "ul_throughput_mbps",
                    np.float32(max(0.0, 0.05 * (1.0 - ramp))),
                    scenario.scenario_id,
                    SCENARIO_TRANSPORT_FAILURE,
                    _SOURCE_FILE_MAP["radio"],
                )
            )
            overrides.append(
                KPIOverride(
                    cell_id,
                    ts,
                    "packet_loss_pct",
                    np.float32(min(80.0, 50.0 * ramp)),
                    scenario.scenario_id,
                    SCENARIO_TRANSPORT_FAILURE,
                    _SOURCE_FILE_MAP["radio"],
                )
            )
            overrides.append(
                KPIOverride(
                    cell_id,
                    ts,
                    "latency_ms",
                    np.float32(500.0 * ramp + 20.0),
                    scenario.scenario_id,
                    SCENARIO_TRANSPORT_FAILURE,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

        # Downstream transport entities (e.g. AGG_SWITCH behind a PE_ROUTER)
        downstream_transport = [
            eid for eid in scenario.affected_entity_ids if eid != primary_eid and graph.get_type(eid) not in cell_types
        ]
        for t_eid in downstream_transport:
            overrides.append(
                KPIOverride(
                    t_eid,
                    ts,
                    "interface_utilization_in_pct",
                    np.float32(0.0),
                    scenario.scenario_id,
                    SCENARIO_TRANSPORT_FAILURE,
                    _SOURCE_FILE_MAP["transport"],
                )
            )
            overrides.append(
                KPIOverride(
                    t_eid,
                    ts,
                    "interface_utilization_out_pct",
                    np.float32(0.0),
                    scenario.scenario_id,
                    SCENARIO_TRANSPORT_FAILURE,
                    _SOURCE_FILE_MAP["transport"],
                )
            )

    return overrides


def _generate_power_failure_overrides(
    scenario: ScenarioInstance,
    rng: np.random.Generator,
    total_hours: int,
    tenant_id: str,
    graph: TopologyGraph,
) -> list[KPIOverride]:
    """
    Power failure: site-level event.
    All co-located equipment affected (cells go down, transport degrades).

    Power KPIs: mains_power_status → 0, battery drains, cabinet temp rises.
    Radio KPIs: cell_availability → 0 after battery depletes.
    """
    overrides: list[KPIOverride] = []
    primary_site = scenario.primary_entity_id
    battery_life_hours = int(rng.integers(2, 6))  # battery backup duration

    cell_types = {"LTE_CELL", "NR_CELL", "NR_NSA_CELL"}
    affected_cells = [eid for eid in scenario.affected_entity_ids if graph.get_type(eid) in cell_types]
    affected_transport = [
        eid for eid in scenario.affected_entity_ids if graph.get_type(eid) not in cell_types and eid != primary_site
    ]

    for hour_idx in range(scenario.start_hour, scenario.end_hour + 1):
        ramp = _ramp_factor(
            hour_idx,
            scenario.start_hour,
            scenario.end_hour,
            scenario.ramp_up_hours,
            scenario.ramp_down_hours,
        )
        if ramp < 0.01:
            continue

        ts = SIMULATION_EPOCH + timedelta(hours=hour_idx)
        hours_into_outage = hour_idx - scenario.start_hour

        # ── Power KPIs (site-level) ──
        overrides.append(
            KPIOverride(
                primary_site,
                ts,
                "mains_power_status",
                np.float32(0.0),
                scenario.scenario_id,
                SCENARIO_POWER_FAILURE,
                _SOURCE_FILE_MAP["power"],
            )
        )

        # Battery drains over time
        battery_pct = max(0.0, 100.0 * (1.0 - hours_into_outage / battery_life_hours))
        battery_v = max(42.0, 54.0 - (12.0 * hours_into_outage / battery_life_hours))
        overrides.append(
            KPIOverride(
                primary_site,
                ts,
                "battery_charge_pct",
                np.float32(battery_pct),
                scenario.scenario_id,
                SCENARIO_POWER_FAILURE,
                _SOURCE_FILE_MAP["power"],
            )
        )
        overrides.append(
            KPIOverride(
                primary_site,
                ts,
                "battery_voltage_v",
                np.float32(battery_v),
                scenario.scenario_id,
                SCENARIO_POWER_FAILURE,
                _SOURCE_FILE_MAP["power"],
            )
        )

        # Cooling fails → temp rises
        if hours_into_outage > 1:
            overrides.append(
                KPIOverride(
                    primary_site,
                    ts,
                    "cooling_status",
                    np.float32(0.0),
                    scenario.scenario_id,
                    SCENARIO_POWER_FAILURE,
                    _SOURCE_FILE_MAP["power"],
                )
            )
            temp_rise = min(15.0, 2.0 * hours_into_outage)
            overrides.append(
                KPIOverride(
                    primary_site,
                    ts,
                    "cabinet_temperature_c",
                    np.float32(35.0 + temp_rise),
                    scenario.scenario_id,
                    SCENARIO_POWER_FAILURE,
                    _SOURCE_FILE_MAP["power"],
                )
            )

        # ── Cell KPIs — battery holds for battery_life_hours, then cells go down ──
        battery_depleted = hours_into_outage >= battery_life_hours

        for cell_id in affected_cells:
            if battery_depleted:
                # Complete outage
                overrides.append(
                    KPIOverride(
                        cell_id,
                        ts,
                        "cell_availability_pct",
                        np.float32(0.0),
                        scenario.scenario_id,
                        SCENARIO_POWER_FAILURE,
                        _SOURCE_FILE_MAP["radio"],
                    )
                )
                overrides.append(
                    KPIOverride(
                        cell_id,
                        ts,
                        "dl_throughput_mbps",
                        np.float32(0.0),
                        scenario.scenario_id,
                        SCENARIO_POWER_FAILURE,
                        _SOURCE_FILE_MAP["radio"],
                    )
                )
                overrides.append(
                    KPIOverride(
                        cell_id,
                        ts,
                        "ul_throughput_mbps",
                        np.float32(0.0),
                        scenario.scenario_id,
                        SCENARIO_POWER_FAILURE,
                        _SOURCE_FILE_MAP["radio"],
                    )
                )
                overrides.append(
                    KPIOverride(
                        cell_id,
                        ts,
                        "active_ue_avg",
                        np.float32(0.0),
                        scenario.scenario_id,
                        SCENARIO_POWER_FAILURE,
                        _SOURCE_FILE_MAP["radio"],
                    )
                )
                overrides.append(
                    KPIOverride(
                        cell_id,
                        ts,
                        "traffic_volume_gb",
                        np.float32(0.0),
                        scenario.scenario_id,
                        SCENARIO_POWER_FAILURE,
                        _SOURCE_FILE_MAP["radio"],
                    )
                )
            else:
                # Battery backup — degraded but functional
                degrade = hours_into_outage / battery_life_hours
                overrides.append(
                    KPIOverride(
                        cell_id,
                        ts,
                        "dl_throughput_mbps",
                        np.float32(max(0.3, 1.0 - 0.5 * degrade)),
                        scenario.scenario_id,
                        SCENARIO_POWER_FAILURE,
                        _SOURCE_FILE_MAP["radio"],
                    )
                )

        # Transport entities at site also affected
        for t_eid in affected_transport:
            if battery_depleted:
                overrides.append(
                    KPIOverride(
                        t_eid,
                        ts,
                        "interface_utilization_in_pct",
                        np.float32(0.0),
                        scenario.scenario_id,
                        SCENARIO_POWER_FAILURE,
                        _SOURCE_FILE_MAP["transport"],
                    )
                )
                overrides.append(
                    KPIOverride(
                        t_eid,
                        ts,
                        "interface_utilization_out_pct",
                        np.float32(0.0),
                        scenario.scenario_id,
                        SCENARIO_POWER_FAILURE,
                        _SOURCE_FILE_MAP["transport"],
                    )
                )

    return overrides


def _generate_fibre_cut_overrides(
    scenario: ScenarioInstance,
    rng: np.random.Generator,
    total_hours: int,
    tenant_id: str,
    graph: TopologyGraph,
) -> list[KPIOverride]:
    """
    Fibre cut: cross-domain cascade.
    Fibre cable → connected transport → downstream cells.
    Also affects fixed broadband (OLT/PON_PORT) if served by the fibre.
    """
    overrides: list[KPIOverride] = []
    primary_eid = scenario.primary_entity_id

    cell_types = {"LTE_CELL", "NR_CELL", "NR_NSA_CELL"}
    transport_types = {
        "PE_ROUTER",
        "AGGREGATION_SWITCH",
        "MICROWAVE_LINK",
        "DWDM_SYSTEM",
        "LSP",
        "L3VPN",
        "BNG",
        "ACCESS_SWITCH",
    }
    fixed_bb_types = {"OLT", "PON_PORT"}

    affected_cells = [eid for eid in scenario.affected_entity_ids if graph.get_type(eid) in cell_types]
    affected_transport = [eid for eid in scenario.affected_entity_ids if graph.get_type(eid) in transport_types]
    affected_fixed_bb = [eid for eid in scenario.affected_entity_ids if graph.get_type(eid) in fixed_bb_types]

    for hour_idx in range(scenario.start_hour, scenario.end_hour + 1):
        ramp = _ramp_factor(
            hour_idx,
            scenario.start_hour,
            scenario.end_hour,
            scenario.ramp_up_hours,
            scenario.ramp_down_hours,
        )
        if ramp < 0.01:
            continue

        ts = SIMULATION_EPOCH + timedelta(hours=hour_idx)

        # Fibre cable itself: optical signal lost
        overrides.append(
            KPIOverride(
                primary_eid,
                ts,
                "interface_utilization_in_pct",
                np.float32(0.0),
                scenario.scenario_id,
                SCENARIO_FIBRE_CUT,
                _SOURCE_FILE_MAP["transport"],
            )
        )
        overrides.append(
            KPIOverride(
                primary_eid,
                ts,
                "interface_utilization_out_pct",
                np.float32(0.0),
                scenario.scenario_id,
                SCENARIO_FIBRE_CUT,
                _SOURCE_FILE_MAP["transport"],
            )
        )
        overrides.append(
            KPIOverride(
                primary_eid,
                ts,
                "optical_rx_power_dbm",
                np.float32(-40.0),  # signal lost
                scenario.scenario_id,
                SCENARIO_FIBRE_CUT,
                _SOURCE_FILE_MAP["transport"],
            )
        )
        overrides.append(
            KPIOverride(
                primary_eid,
                ts,
                "optical_snr_db",
                np.float32(0.0),  # no signal
                scenario.scenario_id,
                SCENARIO_FIBRE_CUT,
                _SOURCE_FILE_MAP["transport"],
            )
        )

        # Downstream transport entities
        for t_eid in affected_transport:
            if t_eid == primary_eid:
                continue
            overrides.append(
                KPIOverride(
                    t_eid,
                    ts,
                    "interface_utilization_in_pct",
                    np.float32(0.0),
                    scenario.scenario_id,
                    SCENARIO_FIBRE_CUT,
                    _SOURCE_FILE_MAP["transport"],
                )
            )
            overrides.append(
                KPIOverride(
                    t_eid,
                    ts,
                    "interface_errors_in",
                    np.float32(500.0 * ramp),
                    scenario.scenario_id,
                    SCENARIO_FIBRE_CUT,
                    _SOURCE_FILE_MAP["transport"],
                )
            )

        # Downstream cells
        for cell_id in affected_cells:
            overrides.append(
                KPIOverride(
                    cell_id,
                    ts,
                    "dl_throughput_mbps",
                    np.float32(0.0),
                    scenario.scenario_id,
                    SCENARIO_FIBRE_CUT,
                    _SOURCE_FILE_MAP["radio"],
                )
            )
            overrides.append(
                KPIOverride(
                    cell_id,
                    ts,
                    "ul_throughput_mbps",
                    np.float32(0.0),
                    scenario.scenario_id,
                    SCENARIO_FIBRE_CUT,
                    _SOURCE_FILE_MAP["radio"],
                )
            )
            overrides.append(
                KPIOverride(
                    cell_id,
                    ts,
                    "packet_loss_pct",
                    np.float32(100.0 * ramp),
                    scenario.scenario_id,
                    SCENARIO_FIBRE_CUT,
                    _SOURCE_FILE_MAP["radio"],
                )
            )

        # Downstream fixed broadband
        for fb_eid in affected_fixed_bb:
            overrides.append(
                KPIOverride(
                    fb_eid,
                    ts,
                    "pon_rx_power_dbm",
                    np.float32(-40.0),
                    scenario.scenario_id,
                    SCENARIO_FIBRE_CUT,
                    _SOURCE_FILE_MAP["fixed_bb"],
                )
            )
            overrides.append(
                KPIOverride(
                    fb_eid,
                    ts,
                    "broadband_throughput_down_mbps",
                    np.float32(0.0),
                    scenario.scenario_id,
                    SCENARIO_FIBRE_CUT,
                    _SOURCE_FILE_MAP["fixed_bb"],
                )
            )
            overrides.append(
                KPIOverride(
                    fb_eid,
                    ts,
                    "broadband_throughput_up_mbps",
                    np.float32(0.0),
                    scenario.scenario_id,
                    SCENARIO_FIBRE_CUT,
                    _SOURCE_FILE_MAP["fixed_bb"],
                )
            )
            overrides.append(
                KPIOverride(
                    fb_eid,
                    ts,
                    "broadband_packet_loss_pct",
                    np.float32(100.0 * ramp),
                    scenario.scenario_id,
                    SCENARIO_FIBRE_CUT,
                    _SOURCE_FILE_MAP["fixed_bb"],
                )
            )

    return overrides


# ---------------------------------------------------------------------------
# Entity selection helpers
# ---------------------------------------------------------------------------


def _select_random_cells(
    entities_df: pl.DataFrame,
    rate: float,
    rng: np.random.Generator,
    cell_types: list[str] | None = None,
) -> list[str]:
    """Select a random fraction of cells."""
    if cell_types is None:
        cell_types = ["LTE_CELL", "NR_CELL"]
    cells = entities_df.filter(pl.col("entity_type").is_in(cell_types))
    n = cells.height
    count = max(1, int(n * rate))
    indices = rng.choice(n, size=min(count, n), replace=False)
    return cells["entity_id"].gather(indices.tolist()).to_list()


def _select_spatial_clusters(
    cells_df: pl.DataFrame,
    rate: float,
    rng: np.random.Generator,
    cluster_size: int = 5,
) -> list[list[str]]:
    """
    Select spatial clusters of cells for coverage hole scenarios.

    Groups cells by site_id, then picks clusters of co-sited or nearby cells.
    """
    # Group by site
    if "site_id" not in cells_df.columns:
        # Fall back to random selection
        n = cells_df.height
        count = max(1, int(n * rate))
        n_clusters = max(1, count // cluster_size)
        clusters = []
        all_ids = cells_df["cell_id"].to_list() if "cell_id" in cells_df.columns else cells_df["entity_id"].to_list()
        for _ in range(n_clusters):
            size = min(cluster_size, len(all_ids))
            idx = rng.choice(len(all_ids), size=size, replace=False)
            clusters.append([all_ids[i] for i in idx])
        return clusters

    sites = cells_df["site_id"].unique().to_list()
    n_clusters = max(1, int(len(sites) * rate))
    selected_sites = rng.choice(sites, size=min(n_clusters, len(sites)), replace=False)

    clusters = []
    id_col = "cell_id" if "cell_id" in cells_df.columns else "entity_id"
    for site_id in selected_sites:
        site_cells = cells_df.filter(pl.col("site_id") == site_id)[id_col].to_list()
        if site_cells:
            clusters.append(site_cells[:cluster_size])

    return clusters


def _select_transport_links(
    entities_df: pl.DataFrame,
    rate: float,
    rng: np.random.Generator,
    link_types: list[str] | None = None,
) -> list[str]:
    """Select random transport link entities."""
    if link_types is None:
        link_types = ["MICROWAVE_LINK", "FIBRE_CABLE", "PE_ROUTER", "AGGREGATION_SWITCH"]
    links = entities_df.filter(pl.col("entity_type").is_in(link_types))
    n = links.height
    if n == 0:
        return []
    count = max(1, int(n * rate))
    indices = rng.choice(n, size=min(count, n), replace=False)
    return links["entity_id"].gather(indices.tolist()).to_list()


def _select_fibre_links(
    entities_df: pl.DataFrame,
    rate: float,
    rng: np.random.Generator,
) -> list[str]:
    """Select random FIBRE_CABLE entities."""
    fibres = entities_df.filter(pl.col("entity_type") == "FIBRE_CABLE")
    n = fibres.height
    if n == 0:
        return []
    count = max(1, int(n * rate))
    indices = rng.choice(n, size=min(count, n), replace=False)
    return fibres["entity_id"].gather(indices.tolist()).to_list()


def _select_sites(
    sites_df: pl.DataFrame,
    rate: float,
    rng: np.random.Generator,
) -> list[str]:
    """Select random sites for power failure scenarios."""
    n = sites_df.height
    count = max(1, int(n * rate))
    indices = rng.choice(n, size=min(count, n), replace=False)
    return sites_df["site_id"].gather(indices.tolist()).to_list()


# ---------------------------------------------------------------------------
# Main scenario generation orchestrator
# ---------------------------------------------------------------------------


def _build_manifest_rows(scenarios: list[ScenarioInstance], tenant_id: str) -> list[dict[str, Any]]:
    """Convert ScenarioInstance list to manifest table rows."""
    rows = []
    for s in scenarios:
        rows.append(
            {
                "scenario_id": s.scenario_id,
                "tenant_id": tenant_id,
                "scenario_type": s.scenario_type,
                "severity": s.severity,
                "primary_entity_id": s.primary_entity_id,
                "primary_entity_type": s.primary_entity_type,
                "primary_domain": s.primary_domain,
                "affected_entity_ids": json.dumps(s.affected_entity_ids),
                "affected_entity_count": len(s.affected_entity_ids),
                "start_hour": s.start_hour,
                "end_hour": s.end_hour,
                "duration_hours": s.end_hour - s.start_hour,
                "cascade_chain": json.dumps(s.cascade_chain) if s.cascade_chain else None,
                "ramp_up_hours": s.ramp_up_hours,
                "ramp_down_hours": s.ramp_down_hours,
                "parameters_json": json.dumps(s.parameters) if s.parameters else None,
            }
        )
    return rows


def _overrides_to_rows(
    overrides: list[KPIOverride],
    tenant_id: str,
) -> list[dict[str, Any]]:
    """Convert KPIOverride list to override table rows."""
    return [
        {
            "entity_id": o.entity_id,
            "tenant_id": tenant_id,
            "timestamp": o.timestamp,
            "kpi_column": o.kpi_column,
            "override_value": float(o.override_value),
            "scenario_id": o.scenario_id,
            "scenario_type": o.scenario_type,
            "source_file": o.source_file,
        }
        for o in overrides
    ]


def _write_parquet(rows: list[dict[str, Any]], schema: pa.Schema, path: Path) -> int:
    """Write a list of dicts as a Parquet file. Returns row count."""
    if not rows:
        # Write empty file with correct schema
        table = pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)
        pq.write_table(
            table,
            str(path),
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
            write_statistics=True,
        )
        return 0

    # Build columnar arrays from row dicts
    arrays = {}
    for f in schema:
        col_values = [row.get(f.name) for row in rows]
        arrays[f.name] = pa.array(col_values, type=f.type)

    table = pa.table(arrays, schema=schema)
    pq.write_table(
        table,
        str(path),
        compression="zstd",
        compression_level=3,
        use_dictionary=True,
        write_statistics=True,
    )
    return table.num_rows


# ---------------------------------------------------------------------------
# Chunked override writer for memory efficiency
# ---------------------------------------------------------------------------

_OVERRIDE_CHUNK_SIZE = 500_000  # Flush to disk every 500K override rows


class _OverrideWriter:
    """
    Streaming writer for scenario_kpi_overrides.parquet.

    Accumulates override rows in memory up to _OVERRIDE_CHUNK_SIZE, then
    appends them as a new row group. This keeps peak memory bounded even
    when millions of overrides are generated.
    """

    _path: Path
    _tenant_id: str
    _writer: pq.ParquetWriter | None
    _buffer: list[KPIOverride]
    _total_rows: int

    def __init__(self, path: Path, tenant_id: str):
        self._path = path
        self._tenant_id = tenant_id
        self._writer = None
        self._buffer = []
        self._total_rows = 0

    def add(self, overrides: list[KPIOverride]) -> None:
        """Add overrides to the buffer, flushing if needed."""
        self._buffer.extend(overrides)
        while len(self._buffer) >= _OVERRIDE_CHUNK_SIZE:
            self._flush(_OVERRIDE_CHUNK_SIZE)

    def _flush(self, count: int | None = None) -> None:
        """Flush up to `count` rows from the buffer (or all if None)."""
        if not self._buffer:
            return

        if count is None:
            to_write = self._buffer
            self._buffer = []
        else:
            to_write = self._buffer[:count]
            self._buffer = self._buffer[count:]

        rows = _overrides_to_rows(to_write, self._tenant_id)

        arrays = {}
        for f in OVERRIDES_SCHEMA:
            col_values = [row.get(f.name) for row in rows]
            arrays[f.name] = pa.array(col_values, type=f.type)

        table = pa.table(arrays, schema=OVERRIDES_SCHEMA)

        if self._writer is None:
            self._writer = pq.ParquetWriter(
                str(self._path),
                schema=OVERRIDES_SCHEMA,
                compression="zstd",
                compression_level=3,
                use_dictionary=True,
                write_statistics=True,
                version="2.6",
            )

        self._writer.write_table(table)
        self._total_rows += table.num_rows
        del table, rows, to_write
        gc.collect()

    def close(self) -> int:
        """Flush remaining buffer and close the writer. Returns total rows."""
        self._flush()
        if self._writer is not None:
            self._writer.close()
        elif self._total_rows == 0:
            # No overrides at all — write empty file
            _write_parquet([], OVERRIDES_SCHEMA, self._path)
        return self._total_rows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def inject_scenarios(config: GeneratorConfig) -> None:
    """
    Step 05 entry point: Inject degradation/failure scenarios into KPI baselines.

    Produces:
      - output/scenario_manifest.parquet
      - output/scenario_kpi_overrides.parquet

    These are overlay files — baseline KPI Parquet files from Phases 3–4 are
    never modified.
    """
    step_start = time.time()

    seed = config.seed_for("step_05_scenarios")
    rng = np.random.default_rng(seed)
    console.print(f"[dim]Step 05 seed: {seed}[/dim]")

    total_hours = config.simulation.total_intervals
    tenant_id = config.tenant_id
    sc = config.scenario_injection

    console.print(f"[bold]Scenario Injection:[/bold] {total_hours:,} hours, 8 scenario types")

    config.ensure_output_dirs()

    # ── Load topology ─────────────────────────────────────────
    console.print("\n[bold]Loading topology graph...[/bold]")
    t0 = time.time()
    entities_df = _load_entities(config)
    rels_df = _load_relationships(config)
    sites_df = _load_sites(config)
    cells_df = _load_cells(config)
    graph = TopologyGraph(rels_df, entities_df)
    console.print(
        f"  [green]✓[/green] Loaded {entities_df.height:,} entities, "
        f"{rels_df.height:,} relationships, "
        f"{sites_df.height:,} sites, "
        f"{cells_df.height:,} cells in {time.time() - t0:.1f}s"
    )

    # Free the raw relationship DataFrame — we only need the graph
    del rels_df
    gc.collect()

    # ── Generate scenario instances ───────────────────────────
    console.print("\n[bold]Generating scenario instances...[/bold]")
    all_scenarios: list[ScenarioInstance] = []
    scenario_counts: dict[str, int] = {}

    # ── 1. Sleeping Cell ──────────────────────────────────────
    console.print("  [dim]1. Sleeping cell scenarios...[/dim]")
    sleeping_cells = _select_random_cells(entities_df, sc.sleeping_cell_rate, rng)
    for cell_id in sleeping_cells:
        start_h, end_h, ramp_up, ramp_down = _pick_duration(SCENARIO_SLEEPING_CELL, rng, total_hours)
        sev = _pick_severity(SCENARIO_SLEEPING_CELL, rng)
        scenario = ScenarioInstance(
            scenario_id=str(uuid.uuid4()),
            scenario_type=SCENARIO_SLEEPING_CELL,
            severity=sev,
            primary_entity_id=cell_id,
            primary_entity_type=graph.get_type(cell_id),
            primary_domain="mobile_ran",
            affected_entity_ids=[cell_id],
            start_hour=start_h,
            end_hour=end_h,
            ramp_up_hours=ramp_up,
            ramp_down_hours=ramp_down,
            parameters={"traffic_drop_mode": "gradual", "alarm_suppressed": True},
        )
        all_scenarios.append(scenario)
    scenario_counts[SCENARIO_SLEEPING_CELL] = len(sleeping_cells)
    console.print(f"    [green]✓[/green] {len(sleeping_cells):,} sleeping cell scenarios")

    # ── 2. Congestion ─────────────────────────────────────────
    console.print("  [dim]2. Congestion scenarios...[/dim]")
    congested_cells = _select_random_cells(entities_df, sc.congestion_rate, rng)
    for cell_id in congested_cells:
        start_h, end_h, ramp_up, ramp_down = _pick_duration(SCENARIO_CONGESTION, rng, total_hours)
        sev = _pick_severity(SCENARIO_CONGESTION, rng)
        scenario = ScenarioInstance(
            scenario_id=str(uuid.uuid4()),
            scenario_type=SCENARIO_CONGESTION,
            severity=sev,
            primary_entity_id=cell_id,
            primary_entity_type=graph.get_type(cell_id),
            primary_domain="mobile_ran",
            affected_entity_ids=[cell_id],
            start_hour=start_h,
            end_hour=end_h,
            ramp_up_hours=ramp_up,
            ramp_down_hours=ramp_down,
        )
        all_scenarios.append(scenario)
    scenario_counts[SCENARIO_CONGESTION] = len(congested_cells)
    console.print(f"    [green]✓[/green] {len(congested_cells):,} congestion scenarios")

    # ── 3. Coverage Hole ──────────────────────────────────────
    console.print("  [dim]3. Coverage hole scenarios...[/dim]")
    coverage_clusters = _select_spatial_clusters(cells_df, sc.coverage_hole_rate, rng, cluster_size=5)
    for cluster in coverage_clusters:
        start_h, end_h, ramp_up, ramp_down = _pick_duration(SCENARIO_COVERAGE_HOLE, rng, total_hours)
        sev = _pick_severity(SCENARIO_COVERAGE_HOLE, rng)
        primary = cluster[0]
        scenario = ScenarioInstance(
            scenario_id=str(uuid.uuid4()),
            scenario_type=SCENARIO_COVERAGE_HOLE,
            severity=sev,
            primary_entity_id=primary,
            primary_entity_type=graph.get_type(primary) if graph.get_type(primary) != "UNKNOWN" else "LTE_CELL",
            primary_domain="mobile_ran",
            affected_entity_ids=cluster,
            start_hour=start_h,
            end_hour=end_h,
            ramp_up_hours=ramp_up,
            ramp_down_hours=ramp_down,
            parameters={"cluster_size": len(cluster)},
        )
        all_scenarios.append(scenario)
    scenario_counts[SCENARIO_COVERAGE_HOLE] = len(coverage_clusters)
    console.print(
        f"    [green]✓[/green] {len(coverage_clusters):,} coverage hole clusters ({sum(len(c) for c in coverage_clusters):,} cells)"
    )

    # ── 4. Hardware Fault ─────────────────────────────────────
    console.print("  [dim]4. Hardware fault scenarios...[/dim]")
    hw_fault_cells = _select_random_cells(entities_df, sc.hardware_fault_rate, rng)
    for cell_id in hw_fault_cells:
        start_h, end_h, ramp_up, ramp_down = _pick_duration(SCENARIO_HARDWARE_FAULT, rng, total_hours)
        sev = _pick_severity(SCENARIO_HARDWARE_FAULT, rng)
        scenario = ScenarioInstance(
            scenario_id=str(uuid.uuid4()),
            scenario_type=SCENARIO_HARDWARE_FAULT,
            severity=sev,
            primary_entity_id=cell_id,
            primary_entity_type=graph.get_type(cell_id),
            primary_domain="mobile_ran",
            affected_entity_ids=[cell_id],
            start_hour=start_h,
            end_hour=end_h,
            ramp_up_hours=ramp_up,
            ramp_down_hours=ramp_down,
        )
        all_scenarios.append(scenario)
    scenario_counts[SCENARIO_HARDWARE_FAULT] = len(hw_fault_cells)
    console.print(f"    [green]✓[/green] {len(hw_fault_cells):,} hardware fault scenarios")

    # ── 5. Interference ───────────────────────────────────────
    console.print("  [dim]5. Interference scenarios...[/dim]")
    interference_cells = _select_random_cells(entities_df, sc.interference_rate, rng)
    for cell_id in interference_cells:
        start_h, end_h, ramp_up, ramp_down = _pick_duration(SCENARIO_INTERFERENCE, rng, total_hours)
        sev = _pick_severity(SCENARIO_INTERFERENCE, rng)
        scenario = ScenarioInstance(
            scenario_id=str(uuid.uuid4()),
            scenario_type=SCENARIO_INTERFERENCE,
            severity=sev,
            primary_entity_id=cell_id,
            primary_entity_type=graph.get_type(cell_id),
            primary_domain="mobile_ran",
            affected_entity_ids=[cell_id],
            start_hour=start_h,
            end_hour=end_h,
            ramp_up_hours=ramp_up,
            ramp_down_hours=ramp_down,
            parameters={"periodic": bool(rng.random() < 0.3)},
        )
        all_scenarios.append(scenario)
    scenario_counts[SCENARIO_INTERFERENCE] = len(interference_cells)
    console.print(f"    [green]✓[/green] {len(interference_cells):,} interference scenarios")

    # ── 6. Transport Failure (with cascade) ───────────────────
    console.print("  [dim]6. Transport failure scenarios...[/dim]")
    transport_links = _select_transport_links(
        entities_df,
        sc.transport_failure_rate,
        rng,
        link_types=["MICROWAVE_LINK", "AGGREGATION_SWITCH"],
    )
    for link_id in transport_links:
        start_h, end_h, ramp_up, ramp_down = _pick_duration(SCENARIO_TRANSPORT_FAILURE, rng, total_hours)
        sev = _pick_severity(SCENARIO_TRANSPORT_FAILURE, rng)

        # Walk the topology graph to find downstream affected entities
        cell_types_set = {"LTE_CELL", "NR_CELL", "NR_NSA_CELL"}
        transport_types_set = {
            "PE_ROUTER",
            "AGGREGATION_SWITCH",
            "MICROWAVE_LINK",
            "DWDM_SYSTEM",
            "LSP",
            "L3VPN",
            "BNG",
            "ACCESS_SWITCH",
        }
        downstream = graph.downstream(
            link_id,
            max_depth=4,
            type_filter=cell_types_set | transport_types_set,
        )
        affected_ids = [link_id] + [eid for eid, _, _ in downstream]
        cascade_chain = [
            {"from": link_id, "to": eid, "to_type": etype, "path": json.dumps(path)} for eid, etype, path in downstream
        ][:20]  # Cap cascade chain detail to keep manifest manageable

        scenario = ScenarioInstance(
            scenario_id=str(uuid.uuid4()),
            scenario_type=SCENARIO_TRANSPORT_FAILURE,
            severity=sev,
            primary_entity_id=link_id,
            primary_entity_type=graph.get_type(link_id),
            primary_domain="transport",
            affected_entity_ids=affected_ids,
            start_hour=start_h,
            end_hour=end_h,
            ramp_up_hours=ramp_up,
            ramp_down_hours=ramp_down,
            cascade_chain=cascade_chain,
            parameters={"cascade_depth": len(downstream)},
        )
        all_scenarios.append(scenario)
    scenario_counts[SCENARIO_TRANSPORT_FAILURE] = len(transport_links)
    total_cascade_transport = sum(
        len(s.affected_entity_ids) - 1 for s in all_scenarios if s.scenario_type == SCENARIO_TRANSPORT_FAILURE
    )
    console.print(
        f"    [green]✓[/green] {len(transport_links):,} transport failures "
        f"(cascading to {total_cascade_transport:,} downstream entities)"
    )

    # ── 7. Power Failure (with cascade) ───────────────────────
    console.print("  [dim]7. Power failure scenarios...[/dim]")
    power_sites = _select_sites(sites_df, sc.power_failure_rate, rng)
    for site_id in power_sites:
        start_h, end_h, ramp_up, ramp_down = _pick_duration(SCENARIO_POWER_FAILURE, rng, total_hours)
        sev = _pick_severity(SCENARIO_POWER_FAILURE, rng)

        # Find all entities at this site
        site_entities = graph.entities_at_site(site_id)
        cells_at_site = graph.find_cells_at_site(site_id)
        affected_ids = [site_id] + [eid for eid, _ in site_entities] + cells_at_site
        # Deduplicate
        affected_ids = list(dict.fromkeys(affected_ids))

        cascade_chain = [
            {"from": site_id, "to": eid, "to_type": etype, "mechanism": "power_loss"} for eid, etype in site_entities
        ][:30]

        scenario = ScenarioInstance(
            scenario_id=str(uuid.uuid4()),
            scenario_type=SCENARIO_POWER_FAILURE,
            severity=sev,
            primary_entity_id=site_id,
            primary_entity_type="SITE",
            primary_domain="power",
            affected_entity_ids=affected_ids,
            start_hour=start_h,
            end_hour=end_h,
            ramp_up_hours=ramp_up,
            ramp_down_hours=ramp_down,
            cascade_chain=cascade_chain,
            parameters={"battery_backup_hours": int(rng.integers(2, 6))},
        )
        all_scenarios.append(scenario)
    scenario_counts[SCENARIO_POWER_FAILURE] = len(power_sites)
    total_cascade_power = sum(
        len(s.affected_entity_ids) - 1 for s in all_scenarios if s.scenario_type == SCENARIO_POWER_FAILURE
    )
    console.print(
        f"    [green]✓[/green] {len(power_sites):,} power failures (cascading to {total_cascade_power:,} entities)"
    )

    # ── 8. Fibre Cut (with cross-domain cascade) ─────────────
    console.print("  [dim]8. Fibre cut scenarios...[/dim]")
    fibre_links = _select_fibre_links(entities_df, sc.fibre_cut_rate, rng)
    for fibre_id in fibre_links:
        start_h, end_h, ramp_up, ramp_down = _pick_duration(SCENARIO_FIBRE_CUT, rng, total_hours)
        sev = _pick_severity(SCENARIO_FIBRE_CUT, rng)

        # Walk topology for cross-domain cascade
        all_target_types = {
            "PE_ROUTER",
            "AGGREGATION_SWITCH",
            "DWDM_SYSTEM",
            "LSP",
            "L3VPN",
            "BNG",
            "ACCESS_SWITCH",
            "MICROWAVE_LINK",
            "LTE_CELL",
            "NR_CELL",
            "NR_NSA_CELL",
            "OLT",
            "PON_PORT",
        }
        downstream = graph.downstream(fibre_id, max_depth=5, type_filter=all_target_types)
        affected_ids = [fibre_id] + [eid for eid, _, _ in downstream]
        affected_ids = list(dict.fromkeys(affected_ids))

        cascade_chain = [
            {"from": fibre_id, "to": eid, "to_type": etype, "path": json.dumps(path)} for eid, etype, path in downstream
        ][:30]

        scenario = ScenarioInstance(
            scenario_id=str(uuid.uuid4()),
            scenario_type=SCENARIO_FIBRE_CUT,
            severity=sev,
            primary_entity_id=fibre_id,
            primary_entity_type="FIBRE_CABLE",
            primary_domain="transport",
            affected_entity_ids=affected_ids,
            start_hour=start_h,
            end_hour=end_h,
            ramp_up_hours=ramp_up,
            ramp_down_hours=ramp_down,
            cascade_chain=cascade_chain,
            parameters={"cascade_depth": len(downstream), "cross_domain": True},
        )
        all_scenarios.append(scenario)
    scenario_counts[SCENARIO_FIBRE_CUT] = len(fibre_links)
    total_cascade_fibre = sum(
        len(s.affected_entity_ids) - 1 for s in all_scenarios if s.scenario_type == SCENARIO_FIBRE_CUT
    )
    console.print(
        f"    [green]✓[/green] {len(fibre_links):,} fibre cuts "
        f"(cascading to {total_cascade_fibre:,} entities across domains)"
    )

    console.print(f"\n[bold cyan]Total: {len(all_scenarios):,} scenario instances across 8 types[/bold cyan]")

    # ── Free entity/site DataFrames ───────────────────────────
    del entities_df, sites_df, cells_df
    gc.collect()

    # ── Generate KPI overrides ────────────────────────────────
    console.print("\n[bold]Generating KPI overrides...[/bold]")

    override_path = config.paths.output_dir / "scenario_kpi_overrides.parquet"
    writer = _OverrideWriter(override_path, tenant_id)

    override_counts: dict[str, int] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("({task.fields[overrides]} overrides)"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=2,
    ) as progress:
        task = progress.add_task("  Generating overrides", total=len(all_scenarios), overrides="0")

        for idx, scenario in enumerate(all_scenarios):
            # Dispatch to the appropriate override generator
            if scenario.scenario_type == SCENARIO_SLEEPING_CELL:
                overrides = _generate_sleeping_cell_overrides(scenario, rng, total_hours, tenant_id)
            elif scenario.scenario_type == SCENARIO_CONGESTION:
                overrides = _generate_congestion_overrides(scenario, rng, total_hours, tenant_id)
            elif scenario.scenario_type == SCENARIO_COVERAGE_HOLE:
                overrides = _generate_coverage_hole_overrides(scenario, rng, total_hours, tenant_id)
            elif scenario.scenario_type == SCENARIO_HARDWARE_FAULT:
                overrides = _generate_hardware_fault_overrides(scenario, rng, total_hours, tenant_id)
            elif scenario.scenario_type == SCENARIO_INTERFERENCE:
                overrides = _generate_interference_overrides(scenario, rng, total_hours, tenant_id)
            elif scenario.scenario_type == SCENARIO_TRANSPORT_FAILURE:
                overrides = _generate_transport_failure_overrides(
                    scenario,
                    rng,
                    total_hours,
                    tenant_id,
                    graph,
                )
            elif scenario.scenario_type == SCENARIO_POWER_FAILURE:
                overrides = _generate_power_failure_overrides(
                    scenario,
                    rng,
                    total_hours,
                    tenant_id,
                    graph,
                )
            elif scenario.scenario_type == SCENARIO_FIBRE_CUT:
                overrides = _generate_fibre_cut_overrides(
                    scenario,
                    rng,
                    total_hours,
                    tenant_id,
                    graph,
                )
            else:
                overrides = []

            writer.add(overrides)
            override_counts[scenario.scenario_type] = override_counts.get(scenario.scenario_type, 0) + len(overrides)
            del overrides

            if idx % 100 == 0:
                gc.collect()

            progress.update(
                task,
                advance=1,
                overrides=f"{sum(override_counts.values()):,}",
            )

    total_override_rows = writer.close()
    del writer
    gc.collect()

    override_size_mb = override_path.stat().st_size / (1024 * 1024)
    console.print(f"  [green]✓[/green] {total_override_rows:,} override rows written ({override_size_mb:.1f} MB)")

    # ── Write scenario manifest ───────────────────────────────
    console.print("\n[bold]Writing scenario manifest...[/bold]")
    manifest_path = config.paths.output_dir / "scenario_manifest.parquet"
    manifest_rows = _build_manifest_rows(all_scenarios, tenant_id)
    manifest_count = _write_parquet(manifest_rows, MANIFEST_SCHEMA, manifest_path)
    manifest_size_mb = manifest_path.stat().st_size / (1024 * 1024)
    console.print(f"  [green]✓[/green] {manifest_count:,} manifest rows written ({manifest_size_mb:.1f} MB)")

    # ── Summary table ─────────────────────────────────────────
    total_elapsed = time.time() - step_start

    console.print()
    summary_table = Table(
        title="Step 05: Scenario Injection — Summary",
        show_header=True,
    )
    summary_table.add_column("Scenario Type", style="bold", width=22)
    summary_table.add_column("Instances", justify="right", width=10)
    summary_table.add_column("Override Rows", justify="right", width=14)
    summary_table.add_column("Avg Overrides/Instance", justify="right", width=22)

    for st in ALL_SCENARIO_TYPES:
        count = scenario_counts.get(st, 0)
        ovr = override_counts.get(st, 0)
        avg = f"{ovr / count:,.0f}" if count > 0 else "N/A"
        summary_table.add_row(st, f"{count:,}", f"{ovr:,}", avg)

    summary_table.add_section()
    summary_table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{len(all_scenarios):,}[/bold]",
        f"[bold]{total_override_rows:,}[/bold]",
        f"{total_override_rows / max(1, len(all_scenarios)):,.0f}",
    )
    console.print(summary_table)

    # Output file summary
    console.print()
    file_table = Table(title="Output Files", show_header=True)
    file_table.add_column("File", style="bold", width=40)
    file_table.add_column("Rows", justify="right", width=14)
    file_table.add_column("Size", justify="right", width=10)

    file_table.add_row(
        "scenario_manifest.parquet",
        f"{manifest_count:,}",
        f"{manifest_size_mb:.1f} MB",
    )
    file_table.add_row(
        "scenario_kpi_overrides.parquet",
        f"{total_override_rows:,}",
        f"{override_size_mb:.1f} MB",
    )
    file_table.add_section()
    total_size = manifest_size_mb + override_size_mb
    file_table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{manifest_count + total_override_rows:,}[/bold]",
        f"[bold]{total_size:.1f} MB[/bold]",
    )
    console.print(file_table)

    time_str = f"{total_elapsed:.1f}s" if total_elapsed < 60 else f"{total_elapsed / 60:.1f}m"
    console.print(
        f"\n[bold green]✓ Step 05 complete.[/bold green] "
        f"Injected {len(all_scenarios):,} scenarios → "
        f"{total_override_rows:,} KPI overrides ({total_size:.1f} MB) "
        f"in {time_str}"
    )
    console.print(
        "[dim]Override semantics: consumers apply "
        "COALESCE(override_value, baseline) per (entity_id, timestamp, kpi_column). "
        "Multiplicative overrides (0.0-1.0 range) are stored as factors; "
        "additive overrides (negative dB values, absolute counts) are stored as deltas. "
        "Check scenario_type + kpi_column to determine interpretation.[/dim]"
    )
