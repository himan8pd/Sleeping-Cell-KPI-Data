"""
Step 06: Events & Alarms Generation.

Generates multi-domain alarm/event data aligned with scenario injections
from Phase 5.  Each non-sleeping-cell scenario produces a temporally
aligned chain of alarms; cross-domain cascades (e.g. fibre cut →
transport link-down → radio cell throughput alarm) produce correlated
alarm chains sharing a ``correlation_group_id``.

**Critical design rule:**
  Sleeping-cell scenarios produce **NO** alarms — that is the entire
  point of the sleeping-cell problem.  The alarm gap is what makes them
  hard to detect.

Alarm lifecycle:
  Each alarm has a ``raised_at`` timestamp (scenario start + optional
  detection delay) and a ``cleared_at`` timestamp (scenario end + optional
  recovery delay, or null if still active at simulation end).

Organic (background) alarms:
  In addition to scenario-driven alarms, a realistic background of
  organic/transient alarms is generated — brief interface flaps, minor
  temperature warnings, transient RACH spikes — to provide noise that
  any real alarm correlation system must filter through.

TMF642 compliance:
  Alarm structure is compatible with Pedkai's alarm ingestion API
  (alarm_id, entity_id, alarm_type, severity, raised_at, cleared_at,
  source_system, probable_cause, domain).

Output:
  - output/events_alarms.parquet

Dependencies: Phase 5 (reads scenario_manifest.parquet),
             Phase 2 (reads ground_truth_entities.parquet for entity metadata)
"""

from __future__ import annotations

import gc
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.table import Table

from pedkai_generator.config.settings import GeneratorConfig

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIMULATION_EPOCH = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# Scenario types (must match Phase 5 constants)
SCENARIO_SLEEPING_CELL = "sleeping_cell"
SCENARIO_CONGESTION = "congestion"
SCENARIO_COVERAGE_HOLE = "coverage_hole"
SCENARIO_HARDWARE_FAULT = "hardware_fault"
SCENARIO_INTERFERENCE = "interference"
SCENARIO_TRANSPORT_FAILURE = "transport_failure"
SCENARIO_POWER_FAILURE = "power_failure"
SCENARIO_FIBRE_CUT = "fibre_cut"

# Alarm severities (from Phase 0 contract)
SEV_MINOR = "minor"
SEV_MAJOR = "major"
SEV_CRITICAL = "critical"

# Alarm source systems (from Phase 0 contract)
SOURCE_ERICSSON_ENM = "ericsson_enm"
SOURCE_NOKIA_NETACT = "nokia_netact"
SOURCE_SNMP = "snmp"
SOURCE_OSS_VENDOR = "oss_vendor"
SOURCE_SYSLOG = "syslog"
SOURCE_OSS_SYNTHETIC = "oss_synthetic"

# Domain identifiers
DOMAIN_RADIO = "radio"
DOMAIN_TRANSPORT = "transport"
DOMAIN_POWER = "power"
DOMAIN_CORE = "core"
DOMAIN_FIXED_BB = "fixed_broadband"

# ---------------------------------------------------------------------------
# Output schema — 15 columns (14 from Phase 0 contract + correlation_group_id)
#
# The Phase 0 contract defines the minimum set; we add correlation_group_id
# as specified in the README for cross-domain alarm chain correlation.
# ---------------------------------------------------------------------------

EVENTS_SCHEMA = pa.schema(
    [
        pa.field("alarm_id", pa.string(), nullable=False),
        pa.field("tenant_id", pa.string(), nullable=False),
        pa.field("entity_id", pa.string(), nullable=False),
        pa.field("entity_type", pa.string(), nullable=False),
        pa.field("alarm_type", pa.string(), nullable=False),
        pa.field("severity", pa.string(), nullable=False),
        pa.field("raised_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("cleared_at", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("source_system", pa.string(), nullable=False),
        pa.field("probable_cause", pa.string(), nullable=True),
        pa.field("domain", pa.string(), nullable=False),
        pa.field("scenario_id", pa.string(), nullable=True),
        pa.field("is_synthetic_scenario", pa.bool_(), nullable=False),
        pa.field("additional_text", pa.string(), nullable=True),
        pa.field("correlation_group_id", pa.string(), nullable=True),
    ]
)


# ---------------------------------------------------------------------------
# Helper dataclass for alarm rows
# ---------------------------------------------------------------------------


@dataclass
class AlarmRow:
    """A single alarm event row."""

    alarm_id: str
    tenant_id: str
    entity_id: str
    entity_type: str
    alarm_type: str
    severity: str
    raised_at: datetime
    cleared_at: datetime | None
    source_system: str
    probable_cause: str | None
    domain: str
    scenario_id: str | None
    is_synthetic_scenario: bool
    additional_text: str | None
    correlation_group_id: str | None


# ---------------------------------------------------------------------------
# Entity type → domain mapping
# ---------------------------------------------------------------------------

_ENTITY_DOMAIN: dict[str, str] = {
    # Radio
    "LTE_CELL": DOMAIN_RADIO,
    "NR_CELL": DOMAIN_RADIO,
    "NR_NSA_CELL": DOMAIN_RADIO,
    "ENODEB": DOMAIN_RADIO,
    "GNODEB": DOMAIN_RADIO,
    "GNODEB_DU": DOMAIN_RADIO,
    "GNODEB_CU_CP": DOMAIN_RADIO,
    "GNODEB_CU_UP": DOMAIN_RADIO,
    "BBU": DOMAIN_RADIO,
    "RRU": DOMAIN_RADIO,
    "ANTENNA": DOMAIN_RADIO,
    "ANTENNA_SYSTEM": DOMAIN_RADIO,
    # Transport
    "PE_ROUTER": DOMAIN_TRANSPORT,
    "P_ROUTER": DOMAIN_TRANSPORT,
    "AGGREGATION_SWITCH": DOMAIN_TRANSPORT,
    "ACCESS_SWITCH": DOMAIN_TRANSPORT,
    "MICROWAVE_LINK": DOMAIN_TRANSPORT,
    "FIBRE_CABLE": DOMAIN_TRANSPORT,
    "DWDM_SYSTEM": DOMAIN_TRANSPORT,
    "LSP": DOMAIN_TRANSPORT,
    "L3VPN": DOMAIN_TRANSPORT,
    "BNG": DOMAIN_TRANSPORT,
    "FIREWALL": DOMAIN_TRANSPORT,
    # Power
    "POWER_SUPPLY": DOMAIN_POWER,
    "BATTERY_BANK": DOMAIN_POWER,
    "BATTERY": DOMAIN_POWER,
    "GENERATOR": DOMAIN_POWER,
    "CLIMATE_CONTROL": DOMAIN_POWER,
    "MAINS_CONNECTION": DOMAIN_POWER,
    "SITE": DOMAIN_POWER,
    # Core
    "MME": DOMAIN_CORE,
    "SGW": DOMAIN_CORE,
    "PGW": DOMAIN_CORE,
    "HSS": DOMAIN_CORE,
    "AMF": DOMAIN_CORE,
    "SMF": DOMAIN_CORE,
    "UPF": DOMAIN_CORE,
    "NSSF": DOMAIN_CORE,
    "PCF": DOMAIN_CORE,
    "UDM": DOMAIN_CORE,
    "NWDAF": DOMAIN_CORE,
    "P_CSCF": DOMAIN_CORE,
    "S_CSCF": DOMAIN_CORE,
    "TAS": DOMAIN_CORE,
    "MGCF": DOMAIN_CORE,
    "RADIUS_SERVER": DOMAIN_CORE,
    "DHCP_SERVER": DOMAIN_CORE,
    "DNS_RESOLVER": DOMAIN_CORE,
    "POLICY_SERVER": DOMAIN_CORE,
    "SOFTSWITCH": DOMAIN_CORE,
    "SBC": DOMAIN_CORE,
    "MEDIA_GATEWAY": DOMAIN_CORE,
    "SIP_TRUNK": DOMAIN_CORE,
    "SD_WAN_CONTROLLER": DOMAIN_CORE,
    "FIREWALL_SERVICE": DOMAIN_CORE,
    "CE_ROUTER": DOMAIN_CORE,
    # Fixed broadband
    "OLT": DOMAIN_FIXED_BB,
    "PON_PORT": DOMAIN_FIXED_BB,
    "ONT": DOMAIN_FIXED_BB,
    "SPLITTER": DOMAIN_FIXED_BB,
    "NTE": DOMAIN_FIXED_BB,
    "ETHERNET_CIRCUIT": DOMAIN_FIXED_BB,
    "EXCHANGE_BUILDING": DOMAIN_FIXED_BB,
}


def _get_domain(entity_type: str) -> str:
    """Return the domain for an entity type."""
    return _ENTITY_DOMAIN.get(entity_type, DOMAIN_RADIO)


# ---------------------------------------------------------------------------
# Scenario → alarm type mapping
#
# Each scenario type produces specific alarm types on specific entity types.
# The mapping defines: (alarm_type, severity, domain, probable_cause_template)
# ---------------------------------------------------------------------------

# Cell-level alarm types for radio scenarios
_RADIO_CELL_TYPES = {"LTE_CELL", "NR_CELL", "NR_NSA_CELL"}
_TRANSPORT_TYPES = {
    "PE_ROUTER",
    "P_ROUTER",
    "AGGREGATION_SWITCH",
    "ACCESS_SWITCH",
    "MICROWAVE_LINK",
    "FIBRE_CABLE",
    "DWDM_SYSTEM",
    "LSP",
    "L3VPN",
    "BNG",
}
_FIXED_BB_TYPES = {"OLT", "PON_PORT", "ONT"}


@dataclass
class AlarmTemplate:
    """Template for generating alarms from a scenario."""

    alarm_type: str
    severity: str
    domain: str
    probable_cause: str
    additional_text_template: str
    # Detection delay range (hours after scenario start)
    detection_delay_min_h: int = 0
    detection_delay_max_h: int = 1
    # Recovery delay range (hours after scenario end)
    recovery_delay_min_h: int = 0
    recovery_delay_max_h: int = 2
    # Entity type filter — only generate this alarm for these types
    entity_type_filter: set[str] | None = None


# ── Congestion alarms ─────────────────────────────────────────────────────

_CONGESTION_TEMPLATES = [
    AlarmTemplate(
        alarm_type="PRB_CONGESTION",
        severity=SEV_MAJOR,
        domain=DOMAIN_RADIO,
        probable_cause="PRB utilisation exceeded 85% threshold",
        additional_text_template="PRB DL utilisation sustained above congestion threshold on {entity_id}",
        detection_delay_min_h=0,
        detection_delay_max_h=2,
        recovery_delay_min_h=0,
        recovery_delay_max_h=1,
        entity_type_filter=_RADIO_CELL_TYPES,
    ),
    AlarmTemplate(
        alarm_type="HIGH_LATENCY",
        severity=SEV_MINOR,
        domain=DOMAIN_RADIO,
        probable_cause="User plane latency exceeded threshold due to congestion",
        additional_text_template="Latency spike detected on {entity_id} correlated with congestion",
        detection_delay_min_h=0,
        detection_delay_max_h=3,
        recovery_delay_min_h=0,
        recovery_delay_max_h=2,
        entity_type_filter=_RADIO_CELL_TYPES,
    ),
]

# ── Coverage hole alarms ──────────────────────────────────────────────────

_COVERAGE_HOLE_TEMPLATES = [
    AlarmTemplate(
        alarm_type="CELL_DEGRADATION",
        severity=SEV_MAJOR,
        domain=DOMAIN_RADIO,
        probable_cause="RSRP/RSRQ degradation detected below acceptable threshold",
        additional_text_template="Coverage degradation on {entity_id}: RSRP below threshold, BLER elevated",
        detection_delay_min_h=1,
        detection_delay_max_h=6,
        recovery_delay_min_h=1,
        recovery_delay_max_h=6,
        entity_type_filter=_RADIO_CELL_TYPES,
    ),
    AlarmTemplate(
        alarm_type="HANDOVER_FAILURE",
        severity=SEV_MINOR,
        domain=DOMAIN_RADIO,
        probable_cause="Handover success rate dropped below threshold in coverage hole area",
        additional_text_template="Elevated handover failures on {entity_id} indicating coverage gap",
        detection_delay_min_h=2,
        detection_delay_max_h=8,
        recovery_delay_min_h=1,
        recovery_delay_max_h=4,
        entity_type_filter=_RADIO_CELL_TYPES,
    ),
]

# ── Hardware fault alarms ─────────────────────────────────────────────────

_HARDWARE_FAULT_TEMPLATES = [
    AlarmTemplate(
        alarm_type="CELL_OUTAGE",
        severity=SEV_CRITICAL,
        domain=DOMAIN_RADIO,
        probable_cause="Cell unavailable due to hardware fault",
        additional_text_template="Cell {entity_id} outage: availability dropped to 0%, hardware fault suspected",
        detection_delay_min_h=0,
        detection_delay_max_h=0,  # Immediate detection
        recovery_delay_min_h=0,
        recovery_delay_max_h=1,
        entity_type_filter=_RADIO_CELL_TYPES,
    ),
    AlarmTemplate(
        alarm_type="HIGH_BLER",
        severity=SEV_MAJOR,
        domain=DOMAIN_RADIO,
        probable_cause="BLER exceeded threshold indicating hardware degradation",
        additional_text_template="DL BLER spike on {entity_id}: {severity} hardware-related block error rate",
        detection_delay_min_h=0,
        detection_delay_max_h=1,
        recovery_delay_min_h=0,
        recovery_delay_max_h=1,
        entity_type_filter=_RADIO_CELL_TYPES,
    ),
    AlarmTemplate(
        alarm_type="EQUIPMENT_FAILURE",
        severity=SEV_CRITICAL,
        domain=DOMAIN_RADIO,
        probable_cause="Radio unit hardware failure detected",
        additional_text_template="Equipment fault on {entity_id}: radio unit reporting errors",
        detection_delay_min_h=0,
        detection_delay_max_h=0,
        recovery_delay_min_h=0,
        recovery_delay_max_h=0,
        entity_type_filter=_RADIO_CELL_TYPES,
    ),
]

# ── Interference alarms ───────────────────────────────────────────────────

_INTERFERENCE_TEMPLATES = [
    AlarmTemplate(
        alarm_type="HIGH_INTERFERENCE",
        severity=SEV_MAJOR,
        domain=DOMAIN_RADIO,
        probable_cause="Interference-over-thermal (IoT) exceeded threshold",
        additional_text_template="Elevated interference on {entity_id}: IoT +{iot_db:.1f} dB above nominal",
        detection_delay_min_h=1,
        detection_delay_max_h=4,
        recovery_delay_min_h=0,
        recovery_delay_max_h=3,
        entity_type_filter=_RADIO_CELL_TYPES,
    ),
    AlarmTemplate(
        alarm_type="CELL_DEGRADATION",
        severity=SEV_MINOR,
        domain=DOMAIN_RADIO,
        probable_cause="SINR degradation due to external interference source",
        additional_text_template="SINR degradation on {entity_id} correlated with interference event",
        detection_delay_min_h=1,
        detection_delay_max_h=6,
        recovery_delay_min_h=1,
        recovery_delay_max_h=4,
        entity_type_filter=_RADIO_CELL_TYPES,
    ),
]

# ── Transport failure alarms ──────────────────────────────────────────────

_TRANSPORT_FAILURE_TEMPLATES_PRIMARY = [
    AlarmTemplate(
        alarm_type="LINK_DOWN",
        severity=SEV_CRITICAL,
        domain=DOMAIN_TRANSPORT,
        probable_cause="Transport link failure — interface down",
        additional_text_template="Link down on {entity_id}: all traffic lost on {entity_type}",
        detection_delay_min_h=0,
        detection_delay_max_h=0,
        recovery_delay_min_h=0,
        recovery_delay_max_h=0,
        entity_type_filter=_TRANSPORT_TYPES,
    ),
    AlarmTemplate(
        alarm_type="INTERFACE_DOWN",
        severity=SEV_CRITICAL,
        domain=DOMAIN_TRANSPORT,
        probable_cause="Physical interface down on transport node",
        additional_text_template="Interface down on {entity_id}: link state transitioned to DOWN",
        detection_delay_min_h=0,
        detection_delay_max_h=0,
        recovery_delay_min_h=0,
        recovery_delay_max_h=0,
        entity_type_filter=_TRANSPORT_TYPES,
    ),
]

_TRANSPORT_FAILURE_TEMPLATES_CASCADE_RADIO = [
    AlarmTemplate(
        alarm_type="HIGH_PACKET_LOSS",
        severity=SEV_MAJOR,
        domain=DOMAIN_RADIO,
        probable_cause="Packet loss elevated due to backhaul transport failure",
        additional_text_template="High packet loss on {entity_id}: backhaul degradation from transport failure",
        detection_delay_min_h=0,
        detection_delay_max_h=1,
        recovery_delay_min_h=0,
        recovery_delay_max_h=1,
        entity_type_filter=_RADIO_CELL_TYPES,
    ),
    AlarmTemplate(
        alarm_type="CELL_DEGRADATION",
        severity=SEV_MAJOR,
        domain=DOMAIN_RADIO,
        probable_cause="Cell throughput collapsed due to backhaul failure",
        additional_text_template="Throughput collapse on {entity_id}: cascade from transport failure",
        detection_delay_min_h=0,
        detection_delay_max_h=2,
        recovery_delay_min_h=0,
        recovery_delay_max_h=1,
        entity_type_filter=_RADIO_CELL_TYPES,
    ),
]

_TRANSPORT_FAILURE_TEMPLATES_CASCADE_TRANSPORT = [
    AlarmTemplate(
        alarm_type="INTERFACE_DOWN",
        severity=SEV_MAJOR,
        domain=DOMAIN_TRANSPORT,
        probable_cause="Downstream transport interface down due to upstream failure",
        additional_text_template="Interface down on {entity_id}: cascade from upstream transport failure",
        detection_delay_min_h=0,
        detection_delay_max_h=0,
        recovery_delay_min_h=0,
        recovery_delay_max_h=0,
        entity_type_filter=_TRANSPORT_TYPES,
    ),
]

# ── Power failure alarms ──────────────────────────────────────────────────

_POWER_FAILURE_TEMPLATES_PRIMARY = [
    AlarmTemplate(
        alarm_type="MAINS_FAILURE",
        severity=SEV_CRITICAL,
        domain=DOMAIN_POWER,
        probable_cause="Mains AC power supply lost",
        additional_text_template="Mains power failure at site {entity_id}: switching to battery backup",
        detection_delay_min_h=0,
        detection_delay_max_h=0,
        recovery_delay_min_h=0,
        recovery_delay_max_h=0,
        entity_type_filter={"SITE"},
    ),
    AlarmTemplate(
        alarm_type="BATTERY_LOW",
        severity=SEV_MAJOR,
        domain=DOMAIN_POWER,
        probable_cause="Battery charge below critical threshold during mains outage",
        additional_text_template="Battery low at site {entity_id}: charge below 30%, estimated {hours_remaining}h remaining",
        detection_delay_min_h=1,
        detection_delay_max_h=3,
        recovery_delay_min_h=0,
        recovery_delay_max_h=0,
        entity_type_filter={"SITE"},
    ),
    AlarmTemplate(
        alarm_type="COOLING_FAILURE",
        severity=SEV_MAJOR,
        domain=DOMAIN_POWER,
        probable_cause="Cooling system failure following mains power loss",
        additional_text_template="Cooling failure at site {entity_id}: cabinet temperature rising",
        detection_delay_min_h=1,
        detection_delay_max_h=2,
        recovery_delay_min_h=0,
        recovery_delay_max_h=1,
        entity_type_filter={"SITE"},
    ),
    AlarmTemplate(
        alarm_type="HIGH_TEMPERATURE",
        severity=SEV_MAJOR,
        domain=DOMAIN_POWER,
        probable_cause="Cabinet temperature exceeded safe operating threshold",
        additional_text_template="High temperature at site {entity_id}: cabinet temp above 45°C",
        detection_delay_min_h=2,
        detection_delay_max_h=4,
        recovery_delay_min_h=0,
        recovery_delay_max_h=2,
        entity_type_filter={"SITE"},
    ),
]

_POWER_FAILURE_TEMPLATES_CASCADE_RADIO = [
    AlarmTemplate(
        alarm_type="CELL_OUTAGE",
        severity=SEV_CRITICAL,
        domain=DOMAIN_RADIO,
        probable_cause="Cell outage due to site power failure — battery depleted",
        additional_text_template="Cell {entity_id} outage: power failure at parent site, battery exhausted",
        detection_delay_min_h=2,
        detection_delay_max_h=6,  # After battery depletes
        recovery_delay_min_h=0,
        recovery_delay_max_h=1,
        entity_type_filter=_RADIO_CELL_TYPES,
    ),
]

_POWER_FAILURE_TEMPLATES_CASCADE_TRANSPORT = [
    AlarmTemplate(
        alarm_type="INTERFACE_DOWN",
        severity=SEV_MAJOR,
        domain=DOMAIN_TRANSPORT,
        probable_cause="Transport equipment down due to site power failure",
        additional_text_template="Interface down on {entity_id}: site power failure cascade",
        detection_delay_min_h=2,
        detection_delay_max_h=6,
        recovery_delay_min_h=0,
        recovery_delay_max_h=1,
        entity_type_filter=_TRANSPORT_TYPES,
    ),
]

# ── Fibre cut alarms ──────────────────────────────────────────────────────

_FIBRE_CUT_TEMPLATES_PRIMARY = [
    AlarmTemplate(
        alarm_type="LINK_DOWN",
        severity=SEV_CRITICAL,
        domain=DOMAIN_TRANSPORT,
        probable_cause="Fibre cable cut — optical signal lost",
        additional_text_template="Fibre cut on {entity_id}: optical Rx power at -40 dBm, total signal loss",
        detection_delay_min_h=0,
        detection_delay_max_h=0,
        recovery_delay_min_h=0,
        recovery_delay_max_h=0,
        entity_type_filter={"FIBRE_CABLE"},
    ),
    AlarmTemplate(
        alarm_type="OPTICAL_POWER_LOW",
        severity=SEV_CRITICAL,
        domain=DOMAIN_TRANSPORT,
        probable_cause="Optical receive power below minimum threshold — fibre break detected",
        additional_text_template="Optical signal loss on {entity_id}: Rx power critically low",
        detection_delay_min_h=0,
        detection_delay_max_h=0,
        recovery_delay_min_h=0,
        recovery_delay_max_h=0,
        entity_type_filter={"FIBRE_CABLE", "DWDM_SYSTEM"},
    ),
]

_FIBRE_CUT_TEMPLATES_CASCADE_TRANSPORT = [
    AlarmTemplate(
        alarm_type="INTERFACE_DOWN",
        severity=SEV_CRITICAL,
        domain=DOMAIN_TRANSPORT,
        probable_cause="Downstream interface down due to upstream fibre cut",
        additional_text_template="Interface down on {entity_id}: cascade from fibre cut on upstream link",
        detection_delay_min_h=0,
        detection_delay_max_h=0,
        recovery_delay_min_h=0,
        recovery_delay_max_h=0,
        entity_type_filter=_TRANSPORT_TYPES,
    ),
]

_FIBRE_CUT_TEMPLATES_CASCADE_RADIO = [
    AlarmTemplate(
        alarm_type="CELL_DEGRADATION",
        severity=SEV_CRITICAL,
        domain=DOMAIN_RADIO,
        probable_cause="Cell service impacted by fibre cut in backhaul path",
        additional_text_template="Total throughput loss on {entity_id}: fibre cut in backhaul cascade",
        detection_delay_min_h=0,
        detection_delay_max_h=1,
        recovery_delay_min_h=0,
        recovery_delay_max_h=1,
        entity_type_filter=_RADIO_CELL_TYPES,
    ),
    AlarmTemplate(
        alarm_type="HIGH_PACKET_LOSS",
        severity=SEV_CRITICAL,
        domain=DOMAIN_RADIO,
        probable_cause="100% packet loss due to backhaul fibre cut",
        additional_text_template="Complete packet loss on {entity_id}: backhaul fibre cut cascade",
        detection_delay_min_h=0,
        detection_delay_max_h=1,
        recovery_delay_min_h=0,
        recovery_delay_max_h=1,
        entity_type_filter=_RADIO_CELL_TYPES,
    ),
]

_FIBRE_CUT_TEMPLATES_CASCADE_FIXED_BB = [
    AlarmTemplate(
        alarm_type="PON_SIGNAL_DEGRADATION",
        severity=SEV_CRITICAL,
        domain=DOMAIN_FIXED_BB,
        probable_cause="PON signal lost due to upstream fibre cut",
        additional_text_template="PON signal loss on {entity_id}: fibre cut in access network",
        detection_delay_min_h=0,
        detection_delay_max_h=0,
        recovery_delay_min_h=0,
        recovery_delay_max_h=0,
        entity_type_filter=_FIXED_BB_TYPES,
    ),
    AlarmTemplate(
        alarm_type="OLT_PORT_DOWN",
        severity=SEV_CRITICAL,
        domain=DOMAIN_FIXED_BB,
        probable_cause="OLT port down due to fibre infrastructure failure",
        additional_text_template="OLT port down on {entity_id}: upstream fibre cut detected",
        detection_delay_min_h=0,
        detection_delay_max_h=0,
        recovery_delay_min_h=0,
        recovery_delay_max_h=0,
        entity_type_filter={"OLT", "PON_PORT"},
    ),
]

# ── Organic (background) alarm templates ──────────────────────────────────

_ORGANIC_RADIO_TEMPLATES = [
    AlarmTemplate(
        alarm_type="RACH_FAILURE",
        severity=SEV_MINOR,
        domain=DOMAIN_RADIO,
        probable_cause="Transient RACH success rate dip below threshold",
        additional_text_template="Brief RACH degradation on {entity_id}: recovered automatically",
        entity_type_filter=_RADIO_CELL_TYPES,
    ),
    AlarmTemplate(
        alarm_type="HIGH_BLER",
        severity=SEV_MINOR,
        domain=DOMAIN_RADIO,
        probable_cause="Transient BLER spike (weather/propagation)",
        additional_text_template="Transient BLER elevation on {entity_id}: likely propagation event",
        entity_type_filter=_RADIO_CELL_TYPES,
    ),
    AlarmTemplate(
        alarm_type="HANDOVER_FAILURE",
        severity=SEV_MINOR,
        domain=DOMAIN_RADIO,
        probable_cause="Brief handover success rate dip",
        additional_text_template="Transient handover failures on {entity_id}",
        entity_type_filter=_RADIO_CELL_TYPES,
    ),
    AlarmTemplate(
        alarm_type="CELL_DEGRADATION",
        severity=SEV_MINOR,
        domain=DOMAIN_RADIO,
        probable_cause="Minor cell KPI degradation (transient)",
        additional_text_template="Minor degradation on {entity_id}: self-recovered",
        entity_type_filter=_RADIO_CELL_TYPES,
    ),
]

_ORGANIC_TRANSPORT_TEMPLATES = [
    AlarmTemplate(
        alarm_type="BGP_FLAP",
        severity=SEV_MINOR,
        domain=DOMAIN_TRANSPORT,
        probable_cause="BGP session flap (transient connectivity issue)",
        additional_text_template="BGP flap on {entity_id}: session recovered within seconds",
        entity_type_filter={"PE_ROUTER", "P_ROUTER", "AGGREGATION_SWITCH", "BNG"},
    ),
    AlarmTemplate(
        alarm_type="INTERFACE_DOWN",
        severity=SEV_MINOR,
        domain=DOMAIN_TRANSPORT,
        probable_cause="Brief interface flap (cable reseat, micro-outage)",
        additional_text_template="Interface flap on {entity_id}: link restored",
        entity_type_filter=_TRANSPORT_TYPES,
    ),
    AlarmTemplate(
        alarm_type="OPTICAL_POWER_LOW",
        severity=SEV_MINOR,
        domain=DOMAIN_TRANSPORT,
        probable_cause="Transient optical power warning (environmental)",
        additional_text_template="Optical power warning on {entity_id}: Rx power fluctuation",
        entity_type_filter={"FIBRE_CABLE", "DWDM_SYSTEM", "PE_ROUTER", "AGGREGATION_SWITCH"},
    ),
    AlarmTemplate(
        alarm_type="HIGH_LATENCY",
        severity=SEV_MINOR,
        domain=DOMAIN_TRANSPORT,
        probable_cause="Brief latency spike on transport path",
        additional_text_template="Transient latency spike on {entity_id}: self-recovered",
        entity_type_filter={"LSP", "L3VPN", "PE_ROUTER"},
    ),
]

_ORGANIC_POWER_TEMPLATES = [
    AlarmTemplate(
        alarm_type="HIGH_TEMPERATURE",
        severity=SEV_MINOR,
        domain=DOMAIN_POWER,
        probable_cause="Cabinet temperature briefly exceeded warning threshold",
        additional_text_template="Temperature warning at site {entity_id}: briefly above 40°C, cooling recovered",
        entity_type_filter={"SITE"},
    ),
    AlarmTemplate(
        alarm_type="POWER_SUPPLY_FAIL",
        severity=SEV_MINOR,
        domain=DOMAIN_POWER,
        probable_cause="Brief power supply voltage fluctuation",
        additional_text_template="Voltage fluctuation at site {entity_id}: rectifier auto-corrected",
        entity_type_filter={"SITE"},
    ),
]

_ORGANIC_CORE_TEMPLATES = [
    AlarmTemplate(
        alarm_type="RADIUS_TIMEOUT",
        severity=SEV_MINOR,
        domain=DOMAIN_CORE,
        probable_cause="Transient RADIUS authentication timeout",
        additional_text_template="RADIUS timeout on {entity_id}: latency spike resolved",
        entity_type_filter={"RADIUS_SERVER", "BNG"},
    ),
    AlarmTemplate(
        alarm_type="DEGRADATION",
        severity=SEV_MINOR,
        domain=DOMAIN_CORE,
        probable_cause="Brief signalling load elevation",
        additional_text_template="Transient signalling load on {entity_id}: auto-scaled",
        entity_type_filter={"MME", "AMF", "SMF", "SGW", "PGW", "UPF"},
    ),
]

_ORGANIC_FIXED_BB_TEMPLATES = [
    AlarmTemplate(
        alarm_type="PON_SIGNAL_DEGRADATION",
        severity=SEV_MINOR,
        domain=DOMAIN_FIXED_BB,
        probable_cause="Transient PON signal degradation",
        additional_text_template="Brief PON signal fluctuation on {entity_id}",
        entity_type_filter={"OLT", "PON_PORT"},
    ),
]


# ---------------------------------------------------------------------------
# Source system assignment
# ---------------------------------------------------------------------------


def _pick_source_system(
    entity_type: str,
    domain: str,
    vendor: str | None,
    rng: np.random.Generator,
) -> str:
    """
    Assign a source system based on entity type, domain, and vendor.

    Ericsson-managed entities → ericsson_enm
    Nokia-managed entities → nokia_netact
    Transport/infrastructure → snmp or oss_vendor
    Power → snmp or syslog
    """
    if domain == DOMAIN_RADIO:
        if vendor == "ericsson":
            return SOURCE_ERICSSON_ENM
        elif vendor == "nokia":
            return SOURCE_NOKIA_NETACT
        else:
            return str(rng.choice([SOURCE_ERICSSON_ENM, SOURCE_NOKIA_NETACT]))
    elif domain == DOMAIN_TRANSPORT:
        return str(rng.choice([SOURCE_SNMP, SOURCE_OSS_VENDOR]))
    elif domain == DOMAIN_POWER:
        return str(rng.choice([SOURCE_SNMP, SOURCE_SYSLOG]))
    elif domain == DOMAIN_CORE:
        return str(rng.choice([SOURCE_SNMP, SOURCE_OSS_VENDOR, SOURCE_SYSLOG]))
    elif domain == DOMAIN_FIXED_BB:
        return str(rng.choice([SOURCE_SNMP, SOURCE_OSS_VENDOR]))
    else:
        return SOURCE_OSS_SYNTHETIC


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_manifest(config: GeneratorConfig) -> pl.DataFrame:
    """Load scenario_manifest.parquet from Phase 5 output."""
    path = config.paths.output_dir / "scenario_manifest.parquet"
    if not path.exists():
        raise FileNotFoundError(f"scenario_manifest.parquet not found at {path}. Run Phase 5 first.")
    return pl.read_parquet(str(path))


def _load_entities(config: GeneratorConfig) -> pl.DataFrame:
    """Load ground_truth_entities.parquet with minimal columns."""
    path = config.paths.output_dir / "ground_truth_entities.parquet"
    if not path.exists():
        raise FileNotFoundError(f"ground_truth_entities.parquet not found at {path}. Run Phase 2 first.")
    cols = ["entity_id", "entity_type", "tenant_id", "vendor", "domain", "site_id"]
    return pl.read_parquet(str(path), columns=cols)


def _build_entity_lookup(entities_df: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """Build entity_id → {entity_type, vendor, domain, site_id} lookup."""
    lookup: dict[str, dict[str, Any]] = {}
    for row in entities_df.iter_rows(named=True):
        lookup[row["entity_id"]] = {
            "entity_type": row["entity_type"],
            "vendor": row.get("vendor"),
            "domain": row.get("domain", "unknown"),
            "site_id": row.get("site_id"),
        }
    return lookup


# ---------------------------------------------------------------------------
# Alarm generation from scenarios
# ---------------------------------------------------------------------------


def _generate_scenario_alarms(
    manifest_df: pl.DataFrame,
    entity_lookup: dict[str, dict[str, Any]],
    total_hours: int,
    tenant_id: str,
    rng: np.random.Generator,
) -> list[AlarmRow]:
    """
    Generate alarms for all non-sleeping-cell scenarios.

    For each scenario in the manifest, applies the appropriate alarm
    templates to generate alarm rows with correct timing, severity,
    and cross-domain correlation.

    Sleeping cell scenarios are **explicitly skipped** — they produce
    no alarms by design.
    """
    all_alarms: list[AlarmRow] = []
    sim_end = SIMULATION_EPOCH + timedelta(hours=total_hours)

    for row in manifest_df.iter_rows(named=True):
        scenario_type = row["scenario_type"]
        scenario_id = row["scenario_id"]
        severity = row["severity"]
        primary_entity_id = row["primary_entity_id"]
        primary_entity_type = row["primary_entity_type"]
        start_hour = row["start_hour"]
        end_hour = row["end_hour"]

        # Parse affected entity IDs from JSON array
        try:
            affected_ids = json.loads(row["affected_entity_ids"])
        except (json.JSONDecodeError, TypeError):
            affected_ids = [primary_entity_id]

        # ──────────────────────────────────────────────────────────────
        # SLEEPING CELL: NO ALARMS — this is the critical design rule
        # ──────────────────────────────────────────────────────────────
        if scenario_type == SCENARIO_SLEEPING_CELL:
            continue

        # Correlation group for cross-domain alarm chains
        correlation_group = str(uuid.uuid4())

        # Classify affected entities by type
        primary_info = entity_lookup.get(primary_entity_id, {})

        # Select templates based on scenario type
        if scenario_type == SCENARIO_CONGESTION:
            _emit_alarms_for_entities(
                alarms_out=all_alarms,
                templates=_CONGESTION_TEMPLATES,
                entity_ids=affected_ids,
                entity_lookup=entity_lookup,
                scenario_id=scenario_id,
                correlation_group=correlation_group,
                start_hour=start_hour,
                end_hour=end_hour,
                total_hours=total_hours,
                sim_end=sim_end,
                tenant_id=tenant_id,
                rng=rng,
            )

        elif scenario_type == SCENARIO_COVERAGE_HOLE:
            _emit_alarms_for_entities(
                alarms_out=all_alarms,
                templates=_COVERAGE_HOLE_TEMPLATES,
                entity_ids=affected_ids,
                entity_lookup=entity_lookup,
                scenario_id=scenario_id,
                correlation_group=correlation_group,
                start_hour=start_hour,
                end_hour=end_hour,
                total_hours=total_hours,
                sim_end=sim_end,
                tenant_id=tenant_id,
                rng=rng,
            )

        elif scenario_type == SCENARIO_HARDWARE_FAULT:
            _emit_alarms_for_entities(
                alarms_out=all_alarms,
                templates=_HARDWARE_FAULT_TEMPLATES,
                entity_ids=affected_ids,
                entity_lookup=entity_lookup,
                scenario_id=scenario_id,
                correlation_group=correlation_group,
                start_hour=start_hour,
                end_hour=end_hour,
                total_hours=total_hours,
                sim_end=sim_end,
                tenant_id=tenant_id,
                rng=rng,
            )

        elif scenario_type == SCENARIO_INTERFERENCE:
            _emit_alarms_for_entities(
                alarms_out=all_alarms,
                templates=_INTERFERENCE_TEMPLATES,
                entity_ids=affected_ids,
                entity_lookup=entity_lookup,
                scenario_id=scenario_id,
                correlation_group=correlation_group,
                start_hour=start_hour,
                end_hour=end_hour,
                total_hours=total_hours,
                sim_end=sim_end,
                tenant_id=tenant_id,
                rng=rng,
            )

        elif scenario_type == SCENARIO_TRANSPORT_FAILURE:
            # Primary transport entity — immediate alarms
            _emit_alarms_for_entities(
                alarms_out=all_alarms,
                templates=_TRANSPORT_FAILURE_TEMPLATES_PRIMARY,
                entity_ids=[primary_entity_id],
                entity_lookup=entity_lookup,
                scenario_id=scenario_id,
                correlation_group=correlation_group,
                start_hour=start_hour,
                end_hour=end_hour,
                total_hours=total_hours,
                sim_end=sim_end,
                tenant_id=tenant_id,
                rng=rng,
            )
            # Cascade to downstream transport entities
            downstream_transport = [
                eid
                for eid in affected_ids
                if eid != primary_entity_id and entity_lookup.get(eid, {}).get("entity_type", "") in _TRANSPORT_TYPES
            ]
            if downstream_transport:
                _emit_alarms_for_entities(
                    alarms_out=all_alarms,
                    templates=_TRANSPORT_FAILURE_TEMPLATES_CASCADE_TRANSPORT,
                    entity_ids=downstream_transport,
                    entity_lookup=entity_lookup,
                    scenario_id=scenario_id,
                    correlation_group=correlation_group,
                    start_hour=start_hour,
                    end_hour=end_hour,
                    total_hours=total_hours,
                    sim_end=sim_end,
                    tenant_id=tenant_id,
                    rng=rng,
                )
            # Cascade to downstream cells
            downstream_cells = [
                eid for eid in affected_ids if entity_lookup.get(eid, {}).get("entity_type", "") in _RADIO_CELL_TYPES
            ]
            if downstream_cells:
                _emit_alarms_for_entities(
                    alarms_out=all_alarms,
                    templates=_TRANSPORT_FAILURE_TEMPLATES_CASCADE_RADIO,
                    entity_ids=downstream_cells,
                    entity_lookup=entity_lookup,
                    scenario_id=scenario_id,
                    correlation_group=correlation_group,
                    start_hour=start_hour,
                    end_hour=end_hour,
                    total_hours=total_hours,
                    sim_end=sim_end,
                    tenant_id=tenant_id,
                    rng=rng,
                )

        elif scenario_type == SCENARIO_POWER_FAILURE:
            # Primary site — power alarms
            _emit_alarms_for_entities(
                alarms_out=all_alarms,
                templates=_POWER_FAILURE_TEMPLATES_PRIMARY,
                entity_ids=[primary_entity_id],
                entity_lookup=entity_lookup,
                scenario_id=scenario_id,
                correlation_group=correlation_group,
                start_hour=start_hour,
                end_hour=end_hour,
                total_hours=total_hours,
                sim_end=sim_end,
                tenant_id=tenant_id,
                rng=rng,
            )
            # Cascade: cells at site go down after battery depletes
            cascade_cells = [
                eid for eid in affected_ids if entity_lookup.get(eid, {}).get("entity_type", "") in _RADIO_CELL_TYPES
            ]
            if cascade_cells:
                _emit_alarms_for_entities(
                    alarms_out=all_alarms,
                    templates=_POWER_FAILURE_TEMPLATES_CASCADE_RADIO,
                    entity_ids=cascade_cells,
                    entity_lookup=entity_lookup,
                    scenario_id=scenario_id,
                    correlation_group=correlation_group,
                    start_hour=start_hour,
                    end_hour=end_hour,
                    total_hours=total_hours,
                    sim_end=sim_end,
                    tenant_id=tenant_id,
                    rng=rng,
                )
            # Cascade: transport at site
            cascade_transport = [
                eid
                for eid in affected_ids
                if eid != primary_entity_id and entity_lookup.get(eid, {}).get("entity_type", "") in _TRANSPORT_TYPES
            ]
            if cascade_transport:
                _emit_alarms_for_entities(
                    alarms_out=all_alarms,
                    templates=_POWER_FAILURE_TEMPLATES_CASCADE_TRANSPORT,
                    entity_ids=cascade_transport,
                    entity_lookup=entity_lookup,
                    scenario_id=scenario_id,
                    correlation_group=correlation_group,
                    start_hour=start_hour,
                    end_hour=end_hour,
                    total_hours=total_hours,
                    sim_end=sim_end,
                    tenant_id=tenant_id,
                    rng=rng,
                )

        elif scenario_type == SCENARIO_FIBRE_CUT:
            # Primary fibre entity — immediate alarms
            _emit_alarms_for_entities(
                alarms_out=all_alarms,
                templates=_FIBRE_CUT_TEMPLATES_PRIMARY,
                entity_ids=[primary_entity_id],
                entity_lookup=entity_lookup,
                scenario_id=scenario_id,
                correlation_group=correlation_group,
                start_hour=start_hour,
                end_hour=end_hour,
                total_hours=total_hours,
                sim_end=sim_end,
                tenant_id=tenant_id,
                rng=rng,
            )
            # Cascade: downstream transport
            cascade_transport = [
                eid
                for eid in affected_ids
                if eid != primary_entity_id and entity_lookup.get(eid, {}).get("entity_type", "") in _TRANSPORT_TYPES
            ]
            if cascade_transport:
                _emit_alarms_for_entities(
                    alarms_out=all_alarms,
                    templates=_FIBRE_CUT_TEMPLATES_CASCADE_TRANSPORT,
                    entity_ids=cascade_transport,
                    entity_lookup=entity_lookup,
                    scenario_id=scenario_id,
                    correlation_group=correlation_group,
                    start_hour=start_hour,
                    end_hour=end_hour,
                    total_hours=total_hours,
                    sim_end=sim_end,
                    tenant_id=tenant_id,
                    rng=rng,
                )
            # Cascade: downstream cells
            cascade_cells = [
                eid for eid in affected_ids if entity_lookup.get(eid, {}).get("entity_type", "") in _RADIO_CELL_TYPES
            ]
            if cascade_cells:
                _emit_alarms_for_entities(
                    alarms_out=all_alarms,
                    templates=_FIBRE_CUT_TEMPLATES_CASCADE_RADIO,
                    entity_ids=cascade_cells,
                    entity_lookup=entity_lookup,
                    scenario_id=scenario_id,
                    correlation_group=correlation_group,
                    start_hour=start_hour,
                    end_hour=end_hour,
                    total_hours=total_hours,
                    sim_end=sim_end,
                    tenant_id=tenant_id,
                    rng=rng,
                )
            # Cascade: downstream fixed broadband
            cascade_fbb = [
                eid for eid in affected_ids if entity_lookup.get(eid, {}).get("entity_type", "") in _FIXED_BB_TYPES
            ]
            if cascade_fbb:
                _emit_alarms_for_entities(
                    alarms_out=all_alarms,
                    templates=_FIBRE_CUT_TEMPLATES_CASCADE_FIXED_BB,
                    entity_ids=cascade_fbb,
                    entity_lookup=entity_lookup,
                    scenario_id=scenario_id,
                    correlation_group=correlation_group,
                    start_hour=start_hour,
                    end_hour=end_hour,
                    total_hours=total_hours,
                    sim_end=sim_end,
                    tenant_id=tenant_id,
                    rng=rng,
                )

    return all_alarms


def _emit_alarms_for_entities(
    *,
    alarms_out: list[AlarmRow],
    templates: list[AlarmTemplate],
    entity_ids: list[str],
    entity_lookup: dict[str, dict[str, Any]],
    scenario_id: str,
    correlation_group: str,
    start_hour: int,
    end_hour: int,
    total_hours: int,
    sim_end: datetime,
    tenant_id: str,
    rng: np.random.Generator,
) -> None:
    """
    For each entity × template combination, generate an alarm if the
    entity type matches the template's filter.
    """
    for eid in entity_ids:
        info = entity_lookup.get(eid, {})
        etype = info.get("entity_type", "UNKNOWN")
        vendor = info.get("vendor")
        domain = info.get("domain", _get_domain(etype))

        for tmpl in templates:
            # Check entity type filter
            if tmpl.entity_type_filter and etype not in tmpl.entity_type_filter:
                continue

            # Detection delay: alarm raised after scenario starts
            detect_delay = int(rng.integers(tmpl.detection_delay_min_h, tmpl.detection_delay_max_h + 1))
            raised_hour = min(start_hour + detect_delay, total_hours - 1)
            raised_at = SIMULATION_EPOCH + timedelta(hours=raised_hour)
            # Add sub-hour jitter (random minutes) for realism
            raised_at += timedelta(minutes=int(rng.integers(0, 60)))

            # Recovery delay: alarm cleared after scenario ends
            recovery_delay = int(rng.integers(tmpl.recovery_delay_min_h, tmpl.recovery_delay_max_h + 1))
            cleared_hour = end_hour + recovery_delay
            if cleared_hour >= total_hours:
                # Alarm still active at simulation end
                cleared_at = None
            else:
                cleared_at = SIMULATION_EPOCH + timedelta(hours=cleared_hour)
                cleared_at += timedelta(minutes=int(rng.integers(0, 60)))

            # Ensure cleared_at is after raised_at when both are set
            if cleared_at is not None and cleared_at <= raised_at:
                cleared_at = raised_at + timedelta(hours=1)
                if cleared_at > sim_end:
                    cleared_at = None

            # Source system based on vendor and domain
            source = _pick_source_system(etype, tmpl.domain, vendor, rng)

            # Format additional text
            additional_text = tmpl.additional_text_template.format(
                entity_id=eid,
                entity_type=etype,
                severity=tmpl.severity,
                iot_db=float(rng.uniform(3.0, 12.0)),
                hours_remaining=max(1, end_hour - raised_hour),
            )

            alarms_out.append(
                AlarmRow(
                    alarm_id=str(uuid.uuid4()),
                    tenant_id=tenant_id,
                    entity_id=eid,
                    entity_type=etype,
                    alarm_type=tmpl.alarm_type,
                    severity=tmpl.severity,
                    raised_at=raised_at,
                    cleared_at=cleared_at,
                    source_system=source,
                    probable_cause=tmpl.probable_cause,
                    domain=tmpl.domain,
                    scenario_id=scenario_id,
                    is_synthetic_scenario=True,
                    additional_text=additional_text,
                    correlation_group_id=correlation_group,
                )
            )


# ---------------------------------------------------------------------------
# Organic (background) alarm generation
# ---------------------------------------------------------------------------


def _generate_organic_alarms(
    entities_df: pl.DataFrame,
    entity_lookup: dict[str, dict[str, Any]],
    total_hours: int,
    tenant_id: str,
    rng: np.random.Generator,
    organic_rate: float = 0.0005,
) -> list[AlarmRow]:
    """
    Generate organic/background alarms that are not associated with any
    injected scenario.

    These represent normal network behaviour — brief transient issues
    that self-resolve.  They provide realistic noise for alarm
    correlation systems.

    Parameters
    ----------
    organic_rate : float
        Probability per entity per day of generating an organic alarm.
        Default 0.0005 ≈ 0.05% per entity per day.
    """
    organic_alarms: list[AlarmRow] = []
    total_days = total_hours // 24

    # Collect entity types that have organic templates
    organic_template_groups: list[tuple[list[AlarmTemplate], list[str]]] = []

    # Build entity ID lists per domain
    radio_cells = entities_df.filter(pl.col("entity_type").is_in(list(_RADIO_CELL_TYPES)))["entity_id"].to_list()

    transport_entities = entities_df.filter(pl.col("entity_type").is_in(list(_TRANSPORT_TYPES)))["entity_id"].to_list()

    # Sites for power alarms
    sites = entities_df.filter(pl.col("entity_type") == "SITE")["entity_id"].to_list()

    core_entities = entities_df.filter(
        pl.col("entity_type").is_in(
            [
                "MME",
                "AMF",
                "SMF",
                "SGW",
                "PGW",
                "UPF",
                "RADIUS_SERVER",
                "BNG",
            ]
        )
    )["entity_id"].to_list()

    fbb_entities = entities_df.filter(pl.col("entity_type").is_in(["OLT", "PON_PORT"]))["entity_id"].to_list()

    organic_template_groups = [
        (_ORGANIC_RADIO_TEMPLATES, radio_cells),
        (_ORGANIC_TRANSPORT_TEMPLATES, transport_entities),
        (_ORGANIC_POWER_TEMPLATES, sites),
        (_ORGANIC_CORE_TEMPLATES, core_entities),
        (_ORGANIC_FIXED_BB_TEMPLATES, fbb_entities),
    ]

    for templates, eid_pool in organic_template_groups:
        if not eid_pool:
            continue

        n_entities = len(eid_pool)
        # Expected organic alarms for this pool over the simulation
        expected_count = int(n_entities * organic_rate * total_days)
        if expected_count < 1:
            expected_count = max(1, int(n_entities * organic_rate))

        # Cap at a reasonable number to avoid excessive memory usage
        actual_count = min(expected_count, 50_000)

        for _ in range(actual_count):
            eid = str(rng.choice(eid_pool))
            tmpl = templates[int(rng.integers(0, len(templates)))]
            info = entity_lookup.get(eid, {})
            etype = info.get("entity_type", "UNKNOWN")
            vendor = info.get("vendor")

            # Check entity type filter for the chosen template
            if tmpl.entity_type_filter and etype not in tmpl.entity_type_filter:
                # Pick a compatible template instead
                compatible = [t for t in templates if not t.entity_type_filter or etype in t.entity_type_filter]
                if not compatible:
                    continue
                tmpl = compatible[int(rng.integers(0, len(compatible)))]

            # Pick a random hour for the alarm
            raised_hour = int(rng.integers(0, total_hours))
            raised_at = SIMULATION_EPOCH + timedelta(hours=raised_hour)
            raised_at += timedelta(minutes=int(rng.integers(0, 60)))

            # Organic alarms are transient — duration 5 min to 4 hours
            duration_min = int(rng.integers(5, 240))
            cleared_at = raised_at + timedelta(minutes=duration_min)
            sim_end_ts = SIMULATION_EPOCH + timedelta(hours=total_hours)
            if cleared_at > sim_end_ts:
                cleared_at = None

            source = _pick_source_system(etype, tmpl.domain, vendor, rng)

            additional_text = tmpl.additional_text_template.format(
                entity_id=eid,
                entity_type=etype,
                severity=tmpl.severity,
                iot_db=0.0,
                hours_remaining=0,
            )

            organic_alarms.append(
                AlarmRow(
                    alarm_id=str(uuid.uuid4()),
                    tenant_id=tenant_id,
                    entity_id=eid,
                    entity_type=etype,
                    alarm_type=tmpl.alarm_type,
                    severity=tmpl.severity,
                    raised_at=raised_at,
                    cleared_at=cleared_at,
                    source_system=source,
                    probable_cause=tmpl.probable_cause,
                    domain=tmpl.domain,
                    scenario_id=None,
                    is_synthetic_scenario=False,
                    additional_text=additional_text,
                    correlation_group_id=None,
                )
            )

    return organic_alarms


# ---------------------------------------------------------------------------
# Parquet writer
# ---------------------------------------------------------------------------

_WRITE_CHUNK_SIZE = 200_000  # rows per row group


def _write_alarms_parquet(alarms: list[AlarmRow], path: Path) -> tuple[int, float]:
    """
    Write alarm rows to a Parquet file in chunks.

    Returns (row_count, size_mb).
    """
    if not alarms:
        # Empty file with correct schema
        table = pa.table(
            {f.name: pa.array([], type=f.type) for f in EVENTS_SCHEMA},
            schema=EVENTS_SCHEMA,
        )
        pq.write_table(
            table,
            str(path),
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
            write_statistics=True,
        )
        return 0, path.stat().st_size / (1024 * 1024)

    writer = pq.ParquetWriter(
        str(path),
        schema=EVENTS_SCHEMA,
        compression="zstd",
        compression_level=3,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )

    total_rows = 0
    for chunk_start in range(0, len(alarms), _WRITE_CHUNK_SIZE):
        chunk = alarms[chunk_start : chunk_start + _WRITE_CHUNK_SIZE]

        arrays = {
            "alarm_id": pa.array([a.alarm_id for a in chunk], type=pa.string()),
            "tenant_id": pa.array([a.tenant_id for a in chunk], type=pa.string()),
            "entity_id": pa.array([a.entity_id for a in chunk], type=pa.string()),
            "entity_type": pa.array([a.entity_type for a in chunk], type=pa.string()),
            "alarm_type": pa.array([a.alarm_type for a in chunk], type=pa.string()),
            "severity": pa.array([a.severity for a in chunk], type=pa.string()),
            "raised_at": pa.array([a.raised_at for a in chunk], type=pa.timestamp("us", tz="UTC")),
            "cleared_at": pa.array([a.cleared_at for a in chunk], type=pa.timestamp("us", tz="UTC")),
            "source_system": pa.array([a.source_system for a in chunk], type=pa.string()),
            "probable_cause": pa.array([a.probable_cause for a in chunk], type=pa.string()),
            "domain": pa.array([a.domain for a in chunk], type=pa.string()),
            "scenario_id": pa.array([a.scenario_id for a in chunk], type=pa.string()),
            "is_synthetic_scenario": pa.array([a.is_synthetic_scenario for a in chunk], type=pa.bool_()),
            "additional_text": pa.array([a.additional_text for a in chunk], type=pa.string()),
            "correlation_group_id": pa.array([a.correlation_group_id for a in chunk], type=pa.string()),
        }

        table = pa.table(arrays, schema=EVENTS_SCHEMA)
        writer.write_table(table)
        total_rows += table.num_rows
        del table, arrays, chunk
        gc.collect()

    writer.close()
    size_mb = path.stat().st_size / (1024 * 1024)
    return total_rows, size_mb


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_events(config: GeneratorConfig) -> None:
    """
    Step 06 entry point: Generate events & alarms.

    Produces output/events_alarms.parquet with:
      1. Scenario-driven alarms — temporally aligned with Phase 5 scenarios
         (sleeping cell scenarios produce NO alarms)
      2. Organic/background alarms — transient noise across all domains
      3. Cross-domain correlation via correlation_group_id
    """
    step_start = time.time()

    seed = config.seed_for("step_06_events")
    rng = np.random.default_rng(seed)
    console.print(f"[dim]Step 06 seed: {seed}[/dim]")

    total_hours = config.simulation.total_intervals
    tenant_id = config.tenant_id

    console.print(f"[bold]Events & Alarms Generation:[/bold] {total_hours:,} hours simulation window")

    config.ensure_output_dirs()

    # ── Load scenario manifest ────────────────────────────────
    console.print("\n[bold]Loading scenario manifest...[/bold]")
    t0 = time.time()
    manifest_df = _load_manifest(config)
    total_scenarios = manifest_df.height
    sleeping_cell_count = manifest_df.filter(pl.col("scenario_type") == SCENARIO_SLEEPING_CELL).height
    non_sleeping = total_scenarios - sleeping_cell_count
    console.print(
        f"  [green]✓[/green] {total_scenarios:,} scenarios loaded "
        f"({sleeping_cell_count:,} sleeping cell → NO alarms, "
        f"{non_sleeping:,} will generate alarms) "
        f"in {time.time() - t0:.1f}s"
    )

    # ── Load entity metadata ──────────────────────────────────
    console.print("\n[bold]Loading entity metadata...[/bold]")
    t0 = time.time()
    entities_df = _load_entities(config)
    entity_lookup = _build_entity_lookup(entities_df)
    console.print(f"  [green]✓[/green] {entities_df.height:,} entities loaded, lookup built in {time.time() - t0:.1f}s")

    # ── Generate scenario-driven alarms ───────────────────────
    console.print("\n[bold]Generating scenario-driven alarms...[/bold]")
    t0 = time.time()
    scenario_alarms = _generate_scenario_alarms(
        manifest_df=manifest_df,
        entity_lookup=entity_lookup,
        total_hours=total_hours,
        tenant_id=tenant_id,
        rng=rng,
    )
    scenario_alarm_count = len(scenario_alarms)
    elapsed_scenario = time.time() - t0

    # Count by domain
    domain_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    sev_counts: dict[str, int] = {}
    for a in scenario_alarms:
        domain_counts[a.domain] = domain_counts.get(a.domain, 0) + 1
        type_counts[a.alarm_type] = type_counts.get(a.alarm_type, 0) + 1
        sev_counts[a.severity] = sev_counts.get(a.severity, 0) + 1

    console.print(f"  [green]✓[/green] {scenario_alarm_count:,} scenario-driven alarms in {elapsed_scenario:.1f}s")
    for dom in sorted(domain_counts.keys()):
        console.print(f"    {dom}: {domain_counts[dom]:,}")

    # ── Generate organic/background alarms ────────────────────
    console.print("\n[bold]Generating organic (background) alarms...[/bold]")
    t0 = time.time()
    organic_alarms = _generate_organic_alarms(
        entities_df=entities_df,
        entity_lookup=entity_lookup,
        total_hours=total_hours,
        tenant_id=tenant_id,
        rng=rng,
    )
    organic_count = len(organic_alarms)
    elapsed_organic = time.time() - t0
    console.print(f"  [green]✓[/green] {organic_count:,} organic alarms in {elapsed_organic:.1f}s")

    # Update domain/type/severity counts with organic
    for a in organic_alarms:
        domain_counts[a.domain] = domain_counts.get(a.domain, 0) + 1
        type_counts[a.alarm_type] = type_counts.get(a.alarm_type, 0) + 1
        sev_counts[a.severity] = sev_counts.get(a.severity, 0) + 1

    # Free entity DataFrame
    del entities_df, manifest_df
    gc.collect()

    # ── Merge and sort all alarms by raised_at ────────────────
    console.print("\n[bold]Merging and sorting alarms...[/bold]")
    all_alarms = scenario_alarms + organic_alarms
    del scenario_alarms, organic_alarms
    gc.collect()

    all_alarms.sort(key=lambda a: a.raised_at)
    total_alarm_count = len(all_alarms)
    console.print(f"  [green]✓[/green] {total_alarm_count:,} total alarms (sorted by raised_at)")

    # Count correlation groups
    corr_groups = set()
    uncorrelated_count = 0
    for a in all_alarms:
        if a.correlation_group_id:
            corr_groups.add(a.correlation_group_id)
        else:
            uncorrelated_count += 1
    console.print(f"  Correlation groups: {len(corr_groups):,} (organic/uncorrelated: {uncorrelated_count:,})")

    # ── Write Parquet ─────────────────────────────────────────
    console.print("\n[bold]Writing events_alarms.parquet...[/bold]")
    output_path = config.paths.output_dir / "events_alarms.parquet"
    t0 = time.time()
    rows_written, size_mb = _write_alarms_parquet(all_alarms, output_path)
    elapsed_write = time.time() - t0
    console.print(f"  [green]✓[/green] {rows_written:,} rows, {size_mb:.1f} MB in {elapsed_write:.1f}s")

    del all_alarms
    gc.collect()

    # ── Summary tables ────────────────────────────────────────
    total_elapsed = time.time() - step_start
    console.print()

    # Domain breakdown
    domain_table = Table(
        title="Step 06: Events & Alarms — Domain Breakdown",
        show_header=True,
    )
    domain_table.add_column("Domain", style="bold", width=20)
    domain_table.add_column("Alarms", justify="right", width=12)
    domain_table.add_column("Share", justify="right", width=10)
    for dom in sorted(domain_counts.keys()):
        count = domain_counts[dom]
        share = f"{100.0 * count / max(1, total_alarm_count):.1f}%"
        domain_table.add_row(dom, f"{count:,}", share)
    domain_table.add_section()
    domain_table.add_row("[bold]Total[/bold]", f"[bold]{total_alarm_count:,}[/bold]", "100.0%")
    console.print(domain_table)

    # Severity breakdown
    console.print()
    sev_table = Table(
        title="Severity Distribution",
        show_header=True,
    )
    sev_table.add_column("Severity", style="bold", width=12)
    sev_table.add_column("Count", justify="right", width=12)
    sev_table.add_column("Share", justify="right", width=10)
    for sev in [SEV_MINOR, SEV_MAJOR, SEV_CRITICAL]:
        count = sev_counts.get(sev, 0)
        share = f"{100.0 * count / max(1, total_alarm_count):.1f}%"
        sev_table.add_row(sev, f"{count:,}", share)
    console.print(sev_table)

    # Alarm type top-10
    console.print()
    type_table = Table(
        title="Top Alarm Types",
        show_header=True,
    )
    type_table.add_column("Alarm Type", style="bold", width=28)
    type_table.add_column("Count", justify="right", width=12)
    top_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    for atype, count in top_types:
        type_table.add_row(atype, f"{count:,}")
    console.print(type_table)

    # Source breakdown
    console.print()
    source_table = Table(
        title="Source & Category Breakdown",
        show_header=True,
    )
    source_table.add_column("Metric", style="bold", width=30)
    source_table.add_column("Value", justify="right", width=14)
    source_table.add_row("Scenario-driven alarms", f"{scenario_alarm_count:,}")
    source_table.add_row("Organic/background alarms", f"{organic_count:,}")
    source_table.add_row("Sleeping cell alarms", "[bold red]0 (by design)[/bold red]")
    source_table.add_row("Correlation groups", f"{len(corr_groups):,}")
    source_table.add_row("Columns", f"{len(EVENTS_SCHEMA)}")
    source_table.add_row("File size", f"{size_mb:.1f} MB")
    console.print(source_table)

    time_str = f"{total_elapsed:.1f}s" if total_elapsed < 60 else f"{total_elapsed / 60:.1f}m"
    console.print(
        f"\n[bold green]✓ Step 06 complete.[/bold green] "
        f"Generated {total_alarm_count:,} alarms ({size_mb:.1f} MB) "
        f"in {time_str}"
    )
    console.print(
        "[dim]Sleeping cell scenarios produce NO alarms — this is the "
        "ground-truth signal gap that Dark Graph must learn to detect. "
        "All other scenarios have correlated alarm chains via "
        "correlation_group_id.[/dim]"
    )
