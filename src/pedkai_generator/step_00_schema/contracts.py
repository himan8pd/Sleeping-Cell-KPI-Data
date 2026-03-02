"""
Step 00: Schema Contracts for all output Parquet files.

Defines the column-level contracts (name, PyArrow type, nullable, description,
valid range / allowed values) for every output file the generator produces.

These contracts serve three purposes:
  1. Documentation — single source of truth for what each file contains.
  2. Validation — Step 10 checks every Parquet file against its contract.
  3. Generation guidance — downstream steps reference these to ensure
     they produce conforming data.

The 14 output files (from THREAD_SUMMARY Section 7):
  1.  ground_truth_entities.parquet
  2.  ground_truth_relationships.parquet
  3.  cmdb_declared_entities.parquet
  4.  cmdb_declared_relationships.parquet
  5.  divergence_manifest.parquet
  6.  kpi_metrics_wide.parquet
  7.  transport_kpis_wide.parquet
  8.  fixed_broadband_kpis_wide.parquet
  9.  enterprise_circuit_kpis_wide.parquet
  10. core_element_kpis_wide.parquet
  11. power_environment_kpis.parquet
  12. events_alarms.parquet
  13. customers_bss.parquet
  14. neighbour_relations.parquet
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa

from pedkai_generator.config.settings import GeneratorConfig

# ---------------------------------------------------------------------------
# Column contract
# ---------------------------------------------------------------------------


@dataclass
class ColumnContract:
    """Contract for a single column in a Parquet file."""

    name: str
    pa_type: pa.DataType
    nullable: bool = False
    description: str = ""
    # Optional validation bounds  (inclusive)
    min_value: float | int | None = None
    max_value: float | int | None = None
    # Optional set of allowed string values
    allowed_values: list[str] | None = None


@dataclass
class FileContract:
    """Contract for a single output Parquet file."""

    filename: str
    description: str
    columns: list[ColumnContract] = field(default_factory=list)
    # Expected approximate row count (order of magnitude, for sanity checks)
    expected_row_count_approx: int | None = None

    @property
    def pa_schema(self) -> pa.Schema:
        """Build a PyArrow schema from the column contracts."""
        fields = []
        for col in self.columns:
            metadata = {}
            if col.description:
                metadata[b"description"] = col.description.encode("utf-8")
            if col.min_value is not None:
                metadata[b"min_value"] = str(col.min_value).encode("utf-8")
            if col.max_value is not None:
                metadata[b"max_value"] = str(col.max_value).encode("utf-8")
            if col.allowed_values is not None:
                metadata[b"allowed_values"] = ",".join(col.allowed_values).encode("utf-8")

            f = pa.field(col.name, col.pa_type, nullable=col.nullable, metadata=metadata)
            fields.append(f)
        return pa.schema(fields)

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def get_column(self, name: str) -> ColumnContract | None:
        for c in self.columns:
            if c.name == name:
                return c
        return None


# ============================================================================
# ENTITY TYPE / RELATIONSHIP TYPE constants
# (shared across multiple contracts)
# ============================================================================

ENTITY_TYPES = [
    # Mobile RAN
    "SITE",
    "CABINET",
    "POWER_SUPPLY",
    "BATTERY_BANK",
    "MAINS_CONNECTION",
    "CLIMATE_CONTROL",
    "TRANSMISSION_EQUIPMENT",
    "MICROWAVE_LINK",
    "FIBER_TERMINATION",
    "ANTENNA_SYSTEM",
    "ANTENNA",
    "RRU",
    "FEEDER_CABLE",
    "FIBER_JUMPER",
    "BBU",
    "ENODEB",
    "LTE_CELL",
    "GNODEB",
    "GNODEB_DU",
    "GNODEB_CU_CP",
    "GNODEB_CU_UP",
    "NR_CELL",
    "GPS_RECEIVER",
    # Fixed Broadband
    "EXCHANGE_BUILDING",
    "OLT",
    "PON_PORT",
    "SPLITTER",
    "ONT",
    "RESIDENTIAL_SERVICE",
    "PCP",
    "DP",
    "FIBRE_SPAN",
    "DSLAM",
    "STREET_CABINET",
    "COPPER_PAIR",
    "NTE",
    "ETHERNET_CIRCUIT",
    "ENTERPRISE_SERVICE",
    # Transport
    "FIBRE_CABLE",
    "FIBRE_PAIR",
    "DUCT",
    "DUCT_SECTION",
    "MANHOLE",
    "ODF",
    "DWDM_SYSTEM",
    "OPTICAL_CHANNEL",
    "ACCESS_SWITCH",
    "AGGREGATION_SWITCH",
    "PE_ROUTER",
    "P_ROUTER",
    "ROUTE_REFLECTOR",
    "L3VPN",
    "L2VPN",
    "PSEUDOWIRE",
    "LSP",
    "BNG",
    "CGNAT",
    "FIREWALL",
    "CDN_NODE",
    # Core
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
    "CE_ROUTER",
    "SD_WAN_CONTROLLER",
    "FIREWALL_SERVICE",
    "SOFTSWITCH",
    "SBC",
    "MEDIA_GATEWAY",
    "SIP_TRUNK",
    # Logical / Service
    "NETWORK_SLICE",
    "TRACKING_AREA",
    "SERVICE_AREA",
    "QOS_PROFILE",
    "EMBB_SLICE",
    "URLLC_SLICE",
    "MMTC_SLICE",
    # Customers
    "RESIDENTIAL_CUSTOMER",
    "ENTERPRISE_CUSTOMER",
    "SERVICE_SUBSCRIPTION",
    # Power / Environment (aliases used in entity table)
    "BATTERY",
    "GENERATOR",
]

RELATIONSHIP_TYPES = [
    "HOSTS",
    "POWERS",
    "COOLS",
    "CONTAINS",
    "CONNECTS_FIBRE",
    "BACKHAULS",
    "UPLINKS_TO",
    "AGGREGATES",
    "PEERS_WITH",
    "ROUTES_THROUGH",
    "MEMBER_OF_VRF",
    "TERMINATES",
    "SPLITS_TO",
    "SERVES_LINE",
    "NEIGHBOURS",
    "ANCHORS",
    "DEPENDS_ON",
    "BEARER_TO",
    "AUTHENTICATES_VIA",
    "MANAGED_BY",
    "COVERED_BY",
    "SUBSCRIBES_TO",
    "CARRIED_OVER",
    "MEMBER_OF",
    "TIMING_FROM",
    # Fixed access specific
    "CONNECTS_COPPER",
    "SERVES_BROADBAND",
]

DEPLOYMENT_PROFILES = [
    "dense_urban",
    "urban",
    "suburban",
    "rural",
    "deep_rural",
    "indoor",
]

SITE_TYPES = [
    "greenfield",
    "rooftop",
    "streetworks",
    "in_building",
    "unspecified",
]

RAT_TYPES = ["LTE", "NR_NSA", "NR_SA"]

VENDOR_NAMES = ["ericsson", "nokia"]

SLA_TIERS = ["GOLD", "SILVER", "BRONZE"]

ALARM_SEVERITIES = ["minor", "major", "critical"]

ALARM_TYPES = [
    # Radio domain
    "CELL_DEGRADATION",
    "CELL_OUTAGE",
    "HIGH_BLER",
    "HIGH_INTERFERENCE",
    "RACH_FAILURE",
    "HANDOVER_FAILURE",
    "PRB_CONGESTION",
    # Transport domain
    "LINK_DOWN",
    "INTERFACE_DOWN",
    "OPTICAL_POWER_LOW",
    "BGP_FLAP",
    "LSP_DOWN",
    "HIGH_LATENCY",
    "HIGH_PACKET_LOSS",
    # Power domain
    "POWER_SUPPLY_FAIL",
    "BATTERY_LOW",
    "MAINS_FAILURE",
    "HIGH_TEMPERATURE",
    "COOLING_FAILURE",
    # Core domain
    "MME_OVERLOAD",
    "SGW_FAILURE",
    "PGW_FAILURE",
    "AMF_OVERLOAD",
    "UPF_FAILURE",
    "SMF_FAILURE",
    "BNG_OVERLOAD",
    "RADIUS_TIMEOUT",
    # Fixed broadband
    "OLT_PORT_DOWN",
    "ONT_OFFLINE",
    "PON_SIGNAL_DEGRADATION",
    # Generic
    "DEGRADATION",
    "EQUIPMENT_FAILURE",
    "CONFIGURATION_ERROR",
]

ALARM_SOURCE_SYSTEMS = [
    "ericsson_enm",
    "nokia_netact",
    "snmp",
    "oss_vendor",
    "syslog",
    "oss_synthetic",
]

DIVERGENCE_TYPES = [
    "dark_node",
    "phantom_node",
    "dark_edge",
    "phantom_edge",
    "dark_attribute",
    "identity_mutation",
]

NEIGHBOUR_RELATION_TYPES = [
    "INTRA_FREQ_NEIGHBOUR",
    "INTER_FREQ_NEIGHBOUR",
    "INTER_RAT_NEIGHBOUR",
]


# ============================================================================
# 1. GROUND TRUTH ENTITIES
# ============================================================================


def _ground_truth_entities_contract() -> FileContract:
    return FileContract(
        filename="ground_truth_entities.parquet",
        description=(
            "Complete correct entity inventory — all ~1.49M entities across "
            "mobile RAN, fixed broadband, transport, core, logical/service, "
            "power/environment, and customer domains."
        ),
        expected_row_count_approx=1_490_000,
        columns=[
            ColumnContract("entity_id", pa.string(), description="Globally unique entity identifier (UUID v4)."),
            ColumnContract("tenant_id", pa.string(), description="Multi-tenant isolation key."),
            ColumnContract(
                "entity_type", pa.string(), description="Entity classification.", allowed_values=ENTITY_TYPES
            ),
            ColumnContract("name", pa.string(), description="Human-readable entity name."),
            ColumnContract(
                "external_id",
                pa.string(),
                nullable=True,
                description="Vendor NMS identifier (e.g., from Ericsson ENM, Nokia NetAct).",
            ),
            ColumnContract(
                "domain",
                pa.string(),
                description="Domain grouping.",
                allowed_values=[
                    "mobile_ran",
                    "fixed_access",
                    "transport",
                    "core",
                    "logical_service",
                    "power_environment",
                    "customer",
                ],
            ),
            ColumnContract(
                "geo_lat", pa.float64(), nullable=True, description="Latitude (WGS84).", min_value=-11.0, max_value=6.0
            ),
            ColumnContract(
                "geo_lon",
                pa.float64(),
                nullable=True,
                description="Longitude (WGS84).",
                min_value=94.0,
                max_value=141.0,
            ),
            ColumnContract(
                "site_id",
                pa.string(),
                nullable=True,
                description="FK to parent site entity_id (for entities associated with a physical site).",
            ),
            ColumnContract(
                "site_type", pa.string(), nullable=True, description="Physical site type.", allowed_values=SITE_TYPES
            ),
            ColumnContract(
                "deployment_profile",
                pa.string(),
                nullable=True,
                description="Deployment environment.",
                allowed_values=DEPLOYMENT_PROFILES,
            ),
            ColumnContract("province", pa.string(), nullable=True, description="Indonesian province name."),
            ColumnContract(
                "timezone",
                pa.string(),
                nullable=True,
                description="Timezone (WIB/WITA/WIT).",
                allowed_values=["WIB", "WITA", "WIT"],
            ),
            ColumnContract(
                "vendor", pa.string(), nullable=True, description="Equipment vendor.", allowed_values=VENDOR_NAMES
            ),
            ColumnContract(
                "rat_type",
                pa.string(),
                nullable=True,
                description="Radio Access Technology (cells only).",
                allowed_values=RAT_TYPES,
            ),
            ColumnContract("band", pa.string(), nullable=True, description="Frequency band name (cells only)."),
            ColumnContract(
                "bandwidth_mhz", pa.float64(), nullable=True, description="Carrier bandwidth in MHz (cells only)."
            ),
            ColumnContract(
                "max_tx_power_dbm",
                pa.float64(),
                nullable=True,
                description="Maximum transmit power in dBm (cells only).",
            ),
            ColumnContract(
                "max_prbs", pa.int32(), nullable=True, description="Maximum Physical Resource Blocks (cells only)."
            ),
            ColumnContract(
                "frequency_mhz",
                pa.float64(),
                nullable=True,
                description="Carrier centre frequency in MHz (cells only).",
            ),
            ColumnContract(
                "sector_id", pa.int32(), nullable=True, description="Sector number within site (0-indexed)."
            ),
            ColumnContract(
                "azimuth_deg",
                pa.float64(),
                nullable=True,
                description="Antenna azimuth in degrees (0-360).",
                min_value=0.0,
                max_value=360.0,
            ),
            ColumnContract(
                "electrical_tilt_deg",
                pa.float64(),
                nullable=True,
                description="Antenna electrical tilt in degrees.",
                min_value=0.0,
                max_value=15.0,
            ),
            ColumnContract(
                "antenna_height_m",
                pa.float64(),
                nullable=True,
                description="Antenna height above ground in metres.",
                min_value=1.0,
                max_value=200.0,
            ),
            ColumnContract(
                "inter_site_distance_m",
                pa.float64(),
                nullable=True,
                description="Inter-site distance in metres.",
                min_value=50.0,
                max_value=50000.0,
            ),
            ColumnContract(
                "revenue_weight",
                pa.float64(),
                nullable=True,
                description="Estimated monthly revenue flowing through this entity.",
                min_value=0.0,
            ),
            ColumnContract(
                "sla_tier", pa.string(), nullable=True, description="SLA category.", allowed_values=SLA_TIERS
            ),
            ColumnContract(
                "is_nsa_anchor", pa.bool_(), nullable=True, description="True if this LTE cell anchors an EN-DC NR SCG."
            ),
            ColumnContract(
                "nsa_anchor_cell_id",
                pa.string(),
                nullable=True,
                description="For NR SCG legs: entity_id of the LTE anchor cell.",
            ),
            ColumnContract(
                "parent_entity_id",
                pa.string(),
                nullable=True,
                description="FK to parent entity (e.g., BBU→ENODEB→CELL hierarchy).",
            ),
            ColumnContract(
                "properties_json", pa.string(), nullable=True, description="Additional properties as JSON string."
            ),
        ],
    )


# ============================================================================
# 2. GROUND TRUTH RELATIONSHIPS
# ============================================================================


def _ground_truth_relationships_contract() -> FileContract:
    return FileContract(
        filename="ground_truth_relationships.parquet",
        description=("Complete correct relationship graph — all ~2.21M relationships across all domains."),
        expected_row_count_approx=2_210_000,
        columns=[
            ColumnContract(
                "relationship_id", pa.string(), description="Globally unique relationship identifier (UUID v4)."
            ),
            ColumnContract("tenant_id", pa.string(), description="Multi-tenant isolation key."),
            ColumnContract("from_entity_id", pa.string(), description="Source entity UUID."),
            ColumnContract(
                "from_entity_type", pa.string(), description="Source entity type.", allowed_values=ENTITY_TYPES
            ),
            ColumnContract(
                "relationship_type",
                pa.string(),
                description="Relationship classification.",
                allowed_values=RELATIONSHIP_TYPES,
            ),
            ColumnContract("to_entity_id", pa.string(), description="Target entity UUID."),
            ColumnContract(
                "to_entity_type", pa.string(), description="Target entity type.", allowed_values=ENTITY_TYPES
            ),
            ColumnContract(
                "domain",
                pa.string(),
                description="Domain grouping.",
                allowed_values=[
                    "mobile_ran",
                    "fixed_access",
                    "transport",
                    "core",
                    "logical_service",
                    "power_environment",
                    "customer",
                    "cross_domain",
                ],
            ),
            ColumnContract(
                "properties_json",
                pa.string(),
                nullable=True,
                description="Additional relationship properties as JSON string (e.g., capacity, distance).",
            ),
        ],
    )


# ============================================================================
# 3. CMDB DECLARED ENTITIES (deliberately degraded)
# ============================================================================


def _cmdb_declared_entities_contract() -> FileContract:
    # Same schema as ground truth entities — but content is degraded
    base = _ground_truth_entities_contract()
    return FileContract(
        filename="cmdb_declared_entities.parquet",
        description=(
            "Deliberately degraded entity CMDB — represents what the operator "
            "*thinks* the network looks like. Missing dark nodes (~6.5%), "
            "contains phantom nodes (~3%), has mutated attributes (~15%), "
            "and identity mutations (~2%)."
        ),
        expected_row_count_approx=1_440_000,
        columns=base.columns,
    )


# ============================================================================
# 4. CMDB DECLARED RELATIONSHIPS (deliberately degraded)
# ============================================================================


def _cmdb_declared_relationships_contract() -> FileContract:
    base = _ground_truth_relationships_contract()
    return FileContract(
        filename="cmdb_declared_relationships.parquet",
        description=(
            "Deliberately degraded relationship CMDB — missing ~10% of real "
            "relationships (dark edges), contains ~5% spurious relationships "
            "(phantom edges)."
        ),
        expected_row_count_approx=2_100_000,
        columns=base.columns,
    )


# ============================================================================
# 5. DIVERGENCE MANIFEST
# ============================================================================


def _divergence_manifest_contract() -> FileContract:
    return FileContract(
        filename="divergence_manifest.parquet",
        description=("Labels for every CMDB divergence — scoring key for Dark Graph reconciliation accuracy."),
        expected_row_count_approx=300_000,
        columns=[
            ColumnContract(
                "divergence_id", pa.string(), description="Unique ID for this divergence instance (UUID v4)."
            ),
            ColumnContract("tenant_id", pa.string(), description="Multi-tenant isolation key."),
            ColumnContract(
                "divergence_type", pa.string(), description="Type of divergence.", allowed_values=DIVERGENCE_TYPES
            ),
            ColumnContract(
                "entity_or_relationship",
                pa.string(),
                description="Whether this affects an entity or relationship.",
                allowed_values=["entity", "relationship"],
            ),
            ColumnContract("target_id", pa.string(), description="The entity_id or relationship_id affected."),
            ColumnContract("target_type", pa.string(), description="Entity type or relationship type of the target."),
            ColumnContract("domain", pa.string(), description="Domain of the affected object."),
            ColumnContract("description", pa.string(), description="Human-readable description of the divergence."),
            # For dark_attribute divergences:
            ColumnContract(
                "attribute_name",
                pa.string(),
                nullable=True,
                description="Which attribute was corrupted (dark_attribute only).",
            ),
            ColumnContract(
                "ground_truth_value", pa.string(), nullable=True, description="Correct value in ground truth."
            ),
            ColumnContract(
                "cmdb_declared_value", pa.string(), nullable=True, description="Incorrect value written into CMDB."
            ),
            # For identity_mutation divergences:
            ColumnContract(
                "original_external_id", pa.string(), nullable=True, description="Original external_id in ground truth."
            ),
            ColumnContract(
                "mutated_external_id", pa.string(), nullable=True, description="Mutated external_id in CMDB."
            ),
        ],
    )


# ============================================================================
# 6. KPI METRICS WIDE (Cell-level Radio KPIs)
# ============================================================================

# The ~35 cell-level radio KPI columns
CELL_RADIO_KPI_COLUMNS: list[ColumnContract] = [
    ColumnContract(
        "rsrp_dbm",
        pa.float64(),
        nullable=True,
        description="Reference Signal Received Power.",
        min_value=-140.0,
        max_value=-40.0,
    ),
    ColumnContract(
        "rsrq_db",
        pa.float64(),
        nullable=True,
        description="Reference Signal Received Quality.",
        min_value=-20.0,
        max_value=-3.0,
    ),
    ColumnContract(
        "sinr_db",
        pa.float64(),
        nullable=True,
        description="Signal to Interference + Noise Ratio.",
        min_value=-10.0,
        max_value=40.0,
    ),
    ColumnContract(
        "cqi_mean",
        pa.float64(),
        nullable=True,
        description="Mean Channel Quality Indicator.",
        min_value=0.0,
        max_value=15.0,
    ),
    ColumnContract(
        "mcs_dl",
        pa.float64(),
        nullable=True,
        description="Downlink Modulation & Coding Scheme index.",
        min_value=0.0,
        max_value=28.0,
    ),
    ColumnContract(
        "mcs_ul",
        pa.float64(),
        nullable=True,
        description="Uplink Modulation & Coding Scheme index.",
        min_value=0.0,
        max_value=28.0,
    ),
    ColumnContract(
        "dl_bler_pct",
        pa.float64(),
        nullable=True,
        description="Downlink Block Error Rate (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "ul_bler_pct",
        pa.float64(),
        nullable=True,
        description="Uplink Block Error Rate (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "prb_utilization_dl",
        pa.float64(),
        nullable=True,
        description="Downlink PRB utilisation (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "prb_utilization_ul",
        pa.float64(),
        nullable=True,
        description="Uplink PRB utilisation (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract("rach_attempts", pa.float64(), nullable=True, description="RACH attempts count.", min_value=0.0),
    ColumnContract(
        "rach_success_rate",
        pa.float64(),
        nullable=True,
        description="RACH success rate (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "rrc_setup_attempts", pa.float64(), nullable=True, description="RRC connection setup attempts.", min_value=0.0
    ),
    ColumnContract(
        "rrc_setup_success_rate",
        pa.float64(),
        nullable=True,
        description="RRC connection setup success rate (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "dl_throughput_mbps", pa.float64(), nullable=True, description="Mean downlink throughput (Mbps).", min_value=0.0
    ),
    ColumnContract(
        "ul_throughput_mbps", pa.float64(), nullable=True, description="Mean uplink throughput (Mbps).", min_value=0.0
    ),
    ColumnContract(
        "latency_ms", pa.float64(), nullable=True, description="Mean user-plane latency (ms).", min_value=0.0
    ),
    ColumnContract("jitter_ms", pa.float64(), nullable=True, description="Mean jitter (ms).", min_value=0.0),
    ColumnContract(
        "packet_loss_pct",
        pa.float64(),
        nullable=True,
        description="Packet loss rate (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "active_ue_avg", pa.float64(), nullable=True, description="Average number of active UEs.", min_value=0.0
    ),
    ColumnContract(
        "active_ue_max",
        pa.float64(),
        nullable=True,
        description="Maximum number of active UEs in the hour.",
        min_value=0.0,
    ),
    ColumnContract(
        "traffic_volume_gb", pa.float64(), nullable=True, description="Total traffic volume in GB.", min_value=0.0
    ),
    ColumnContract(
        "dl_rlc_retransmission_pct",
        pa.float64(),
        nullable=True,
        description="Downlink RLC retransmission rate (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "ul_rlc_retransmission_pct",
        pa.float64(),
        nullable=True,
        description="Uplink RLC retransmission rate (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract("ho_attempt", pa.float64(), nullable=True, description="Handover attempt count.", min_value=0.0),
    ColumnContract(
        "ho_success_rate",
        pa.float64(),
        nullable=True,
        description="Handover success rate (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "cell_availability_pct",
        pa.float64(),
        nullable=True,
        description="Cell availability (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "interference_iot_db",
        pa.float64(),
        nullable=True,
        description="Interference over Thermal (dB).",
        min_value=-5.0,
        max_value=30.0,
    ),
    ColumnContract(
        "paging_discard_rate",
        pa.float64(),
        nullable=True,
        description="Paging discard rate (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "cce_utilization_pct",
        pa.float64(),
        nullable=True,
        description="PDCCH CCE utilisation (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "volte_erlangs", pa.float64(), nullable=True, description="VoLTE traffic in Erlangs.", min_value=0.0
    ),
    ColumnContract("csfb_attempts", pa.float64(), nullable=True, description="CS Fallback attempts.", min_value=0.0),
    ColumnContract(
        "csfb_success_rate",
        pa.float64(),
        nullable=True,
        description="CS Fallback success rate (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "pdcp_dl_volume_mb", pa.float64(), nullable=True, description="PDCP layer DL volume (MB).", min_value=0.0
    ),
    ColumnContract(
        "pdcp_ul_volume_mb", pa.float64(), nullable=True, description="PDCP layer UL volume (MB).", min_value=0.0
    ),
]

# Pedkai-compatible alias columns
PEDKAI_ALIAS_COLUMNS: list[ColumnContract] = [
    ColumnContract(
        "throughput_mbps",
        pa.float64(),
        nullable=True,
        description="Alias for dl_throughput_mbps (Pedkai compat).",
        min_value=0.0,
    ),
    ColumnContract(
        "prb_utilization",
        pa.float64(),
        nullable=True,
        description="Mean of dl+ul PRB utilisation (Pedkai compat).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "traffic_volume",
        pa.float64(),
        nullable=True,
        description="Alias for traffic_volume_gb (Pedkai compat).",
        min_value=0.0,
    ),
    ColumnContract(
        "active_users_count",
        pa.float64(),
        nullable=True,
        description="Alias for active_ue_avg (Pedkai compat).",
        min_value=0.0,
    ),
    ColumnContract(
        "prb_utilization_pct",
        pa.float64(),
        nullable=True,
        description="Alias for prb_utilization (Pedkai compat).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "latency_ms_alias", pa.float64(), nullable=True, description="Alias: same as latency_ms.", min_value=0.0
    ),
    ColumnContract(
        "data_throughput_gbps",
        pa.float64(),
        nullable=True,
        description="dl_throughput_mbps / 1000 (Pedkai compat).",
        min_value=0.0,
    ),
    ColumnContract(
        "packet_loss_pct_alias",
        pa.float64(),
        nullable=True,
        description="Alias: same as packet_loss_pct.",
        min_value=0.0,
        max_value=100.0,
    ),
]


def _kpi_metrics_wide_contract() -> FileContract:
    key_columns = [
        ColumnContract(
            "cell_id", pa.string(), description="Logical cell-layer entity_id (FK to ground_truth_entities)."
        ),
        ColumnContract("tenant_id", pa.string(), description="Multi-tenant isolation key."),
        ColumnContract(
            "timestamp", pa.timestamp("us", tz="UTC"), description="Hourly reporting interval start (ISO 8601 UTC)."
        ),
        ColumnContract("rat_type", pa.string(), description="Radio Access Technology.", allowed_values=RAT_TYPES),
        ColumnContract("band", pa.string(), nullable=True, description="Frequency band name."),
        ColumnContract("site_id", pa.string(), description="FK to parent site entity_id."),
        ColumnContract("vendor", pa.string(), description="Equipment vendor.", allowed_values=VENDOR_NAMES),
        ColumnContract(
            "deployment_profile", pa.string(), description="Deployment environment.", allowed_values=DEPLOYMENT_PROFILES
        ),
        ColumnContract(
            "is_nsa_scg_leg", pa.bool_(), description="True if this row is the NR SCG leg of an EN-DC cell."
        ),
    ]
    return FileContract(
        filename="kpi_metrics_wide.parquet",
        description=(
            "Cell-level radio KPIs in wide format. 1 row per logical cell-layer "
            "per hour. ~64,700 cell-layers × 720 hours = ~46.6M rows, ~50 columns."
        ),
        expected_row_count_approx=46_600_000,
        columns=key_columns + CELL_RADIO_KPI_COLUMNS + PEDKAI_ALIAS_COLUMNS,
    )


# ============================================================================
# 7. TRANSPORT KPIs WIDE
# ============================================================================

TRANSPORT_KPI_COLUMNS: list[ColumnContract] = [
    ColumnContract(
        "interface_utilization_in_pct",
        pa.float64(),
        nullable=True,
        description="Inbound interface utilisation (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "interface_utilization_out_pct",
        pa.float64(),
        nullable=True,
        description="Outbound interface utilisation (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "interface_errors_in", pa.float64(), nullable=True, description="Inbound interface error count.", min_value=0.0
    ),
    ColumnContract(
        "interface_errors_out",
        pa.float64(),
        nullable=True,
        description="Outbound interface error count.",
        min_value=0.0,
    ),
    ColumnContract(
        "interface_discards_in",
        pa.float64(),
        nullable=True,
        description="Inbound interface discard count.",
        min_value=0.0,
    ),
    ColumnContract(
        "interface_discards_out",
        pa.float64(),
        nullable=True,
        description="Outbound interface discard count.",
        min_value=0.0,
    ),
    ColumnContract(
        "optical_rx_power_dbm",
        pa.float64(),
        nullable=True,
        description="Optical receive power (dBm).",
        min_value=-30.0,
        max_value=5.0,
    ),
    ColumnContract(
        "optical_snr_db",
        pa.float64(),
        nullable=True,
        description="Optical signal-to-noise ratio (dB).",
        min_value=0.0,
        max_value=40.0,
    ),
    ColumnContract(
        "lsp_utilization_pct",
        pa.float64(),
        nullable=True,
        description="LSP utilisation (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract("lsp_latency_ms", pa.float64(), nullable=True, description="LSP latency (ms).", min_value=0.0),
    ColumnContract(
        "bgp_prefixes_received", pa.float64(), nullable=True, description="BGP prefixes received.", min_value=0.0
    ),
    ColumnContract(
        "bgp_session_flaps", pa.float64(), nullable=True, description="BGP session flap count.", min_value=0.0
    ),
    ColumnContract(
        "microwave_modulation",
        pa.string(),
        nullable=True,
        description="Current microwave modulation scheme.",
        allowed_values=["QPSK", "16QAM", "32QAM", "64QAM", "128QAM", "256QAM", "1024QAM", "2048QAM", "4096QAM"],
    ),
    ColumnContract(
        "microwave_capacity_mbps",
        pa.float64(),
        nullable=True,
        description="Current microwave capacity (Mbps).",
        min_value=0.0,
    ),
    ColumnContract(
        "microwave_availability_pct",
        pa.float64(),
        nullable=True,
        description="Microwave link availability (%).",
        min_value=0.0,
        max_value=100.0,
    ),
]


def _transport_kpis_wide_contract() -> FileContract:
    key_columns = [
        ColumnContract("entity_id", pa.string(), description="Transport entity_id (FK to ground_truth_entities)."),
        ColumnContract("tenant_id", pa.string(), description="Multi-tenant isolation key."),
        ColumnContract("timestamp", pa.timestamp("us", tz="UTC"), description="Hourly reporting interval start."),
        ColumnContract("entity_type", pa.string(), description="Transport entity type."),
        ColumnContract("site_id", pa.string(), nullable=True, description="Associated site (if applicable)."),
    ]
    return FileContract(
        filename="transport_kpis_wide.parquet",
        description=("Transport link/element KPIs in wide format. ~50,000 entities × 15 metrics × 720 hours."),
        expected_row_count_approx=36_000_000,
        columns=key_columns + TRANSPORT_KPI_COLUMNS,
    )


# ============================================================================
# 8. FIXED BROADBAND KPIs WIDE
# ============================================================================

FIXED_BROADBAND_KPI_COLUMNS: list[ColumnContract] = [
    ColumnContract(
        "pon_rx_power_dbm",
        pa.float64(),
        nullable=True,
        description="PON receive power (dBm).",
        min_value=-30.0,
        max_value=0.0,
    ),
    ColumnContract(
        "pon_tx_power_dbm",
        pa.float64(),
        nullable=True,
        description="PON transmit power (dBm).",
        min_value=0.0,
        max_value=10.0,
    ),
    ColumnContract(
        "pon_ber", pa.float64(), nullable=True, description="PON bit error rate.", min_value=0.0, max_value=1.0
    ),
    ColumnContract(
        "olt_port_utilization_pct",
        pa.float64(),
        nullable=True,
        description="OLT port utilisation (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "ont_uptime_pct", pa.float64(), nullable=True, description="ONT uptime (%).", min_value=0.0, max_value=100.0
    ),
    ColumnContract(
        "broadband_sync_rate_down_mbps",
        pa.float64(),
        nullable=True,
        description="Downstream sync rate (Mbps).",
        min_value=0.0,
    ),
    ColumnContract(
        "broadband_sync_rate_up_mbps",
        pa.float64(),
        nullable=True,
        description="Upstream sync rate (Mbps).",
        min_value=0.0,
    ),
    ColumnContract(
        "broadband_throughput_down_mbps",
        pa.float64(),
        nullable=True,
        description="Downstream throughput (Mbps).",
        min_value=0.0,
    ),
    ColumnContract(
        "broadband_throughput_up_mbps",
        pa.float64(),
        nullable=True,
        description="Upstream throughput (Mbps).",
        min_value=0.0,
    ),
    ColumnContract(
        "broadband_latency_ms", pa.float64(), nullable=True, description="Broadband latency (ms).", min_value=0.0
    ),
    ColumnContract(
        "broadband_packet_loss_pct",
        pa.float64(),
        nullable=True,
        description="Broadband packet loss (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "pppoe_session_count", pa.float64(), nullable=True, description="Active PPPoE session count.", min_value=0.0
    ),
    ColumnContract(
        "dhcp_lease_failures", pa.float64(), nullable=True, description="DHCP lease failure count.", min_value=0.0
    ),
    ColumnContract(
        "dns_query_latency_ms", pa.float64(), nullable=True, description="DNS query latency (ms).", min_value=0.0
    ),
]


def _fixed_broadband_kpis_wide_contract() -> FileContract:
    key_columns = [
        ColumnContract(
            "entity_id", pa.string(), description="Fixed broadband entity_id (FK to ground_truth_entities)."
        ),
        ColumnContract("tenant_id", pa.string(), description="Multi-tenant isolation key."),
        ColumnContract("timestamp", pa.timestamp("us", tz="UTC"), description="Hourly reporting interval start."),
        ColumnContract(
            "entity_type", pa.string(), description="Fixed broadband entity type (OLT, PON_PORT, ONT, etc.)."
        ),
        ColumnContract("exchange_id", pa.string(), nullable=True, description="Parent exchange entity_id."),
    ]
    return FileContract(
        filename="fixed_broadband_kpis_wide.parquet",
        description=("Fixed broadband KPIs (OLT/PON/ONT) in wide format. ~15,000 entities × 14 metrics × 720 hours."),
        expected_row_count_approx=10_800_000,
        columns=key_columns + FIXED_BROADBAND_KPI_COLUMNS,
    )


# ============================================================================
# 9. ENTERPRISE CIRCUIT KPIs WIDE
# ============================================================================

ENTERPRISE_CIRCUIT_KPI_COLUMNS: list[ColumnContract] = [
    ColumnContract(
        "circuit_availability_pct",
        pa.float64(),
        nullable=True,
        description="Circuit availability (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "circuit_throughput_in_mbps",
        pa.float64(),
        nullable=True,
        description="Inbound circuit throughput (Mbps).",
        min_value=0.0,
    ),
    ColumnContract(
        "circuit_throughput_out_mbps",
        pa.float64(),
        nullable=True,
        description="Outbound circuit throughput (Mbps).",
        min_value=0.0,
    ),
    ColumnContract(
        "circuit_latency_ms", pa.float64(), nullable=True, description="Circuit latency (ms).", min_value=0.0
    ),
    ColumnContract("circuit_jitter_ms", pa.float64(), nullable=True, description="Circuit jitter (ms).", min_value=0.0),
    ColumnContract(
        "circuit_packet_loss_pct",
        pa.float64(),
        nullable=True,
        description="Circuit packet loss (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract("vpn_prefix_count", pa.float64(), nullable=True, description="VPN prefix count.", min_value=0.0),
    ColumnContract(
        "sla_breach_count", pa.float64(), nullable=True, description="SLA breach count in this interval.", min_value=0.0
    ),
    ColumnContract("cos_queue_drops", pa.float64(), nullable=True, description="CoS queue drops.", min_value=0.0),
    ColumnContract(
        "circuit_uptime_seconds",
        pa.float64(),
        nullable=True,
        description="Seconds the circuit was up in this interval.",
        min_value=0.0,
        max_value=3600.0,
    ),
]


def _enterprise_circuit_kpis_wide_contract() -> FileContract:
    key_columns = [
        ColumnContract(
            "entity_id", pa.string(), description="Enterprise circuit/service entity_id (FK to ground_truth_entities)."
        ),
        ColumnContract("tenant_id", pa.string(), description="Multi-tenant isolation key."),
        ColumnContract("timestamp", pa.timestamp("us", tz="UTC"), description="Hourly reporting interval start."),
        ColumnContract(
            "entity_type", pa.string(), description="Enterprise entity type (ETHERNET_CIRCUIT, L3VPN, E_LINE, etc.)."
        ),
        ColumnContract("sla_tier", pa.string(), nullable=True, description="SLA tier.", allowed_values=SLA_TIERS),
    ]
    return FileContract(
        filename="enterprise_circuit_kpis_wide.parquet",
        description=("Enterprise service KPIs in wide format. ~10,000 entities × 10 metrics × 720 hours."),
        expected_row_count_approx=7_200_000,
        columns=key_columns + ENTERPRISE_CIRCUIT_KPI_COLUMNS,
    )


# ============================================================================
# 10. CORE ELEMENT KPIs WIDE
# ============================================================================

CORE_ELEMENT_KPI_COLUMNS: list[ColumnContract] = [
    ColumnContract(
        "cpu_utilization_pct",
        pa.float64(),
        nullable=True,
        description="CPU utilisation (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "memory_utilization_pct",
        pa.float64(),
        nullable=True,
        description="Memory utilisation (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract("active_sessions", pa.float64(), nullable=True, description="Active session count.", min_value=0.0),
    ColumnContract(
        "session_setup_rate", pa.float64(), nullable=True, description="Session setup rate (per second).", min_value=0.0
    ),
    ColumnContract(
        "session_setup_success_pct",
        pa.float64(),
        nullable=True,
        description="Session setup success rate (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "signalling_load_pct",
        pa.float64(),
        nullable=True,
        description="Signalling load (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "throughput_gbps", pa.float64(), nullable=True, description="User-plane throughput (Gbps).", min_value=0.0
    ),
    ColumnContract(
        "control_plane_latency_ms",
        pa.float64(),
        nullable=True,
        description="Control plane latency (ms).",
        min_value=0.0,
    ),
    ColumnContract(
        "user_plane_latency_ms", pa.float64(), nullable=True, description="User plane latency (ms).", min_value=0.0
    ),
    ColumnContract(
        "error_rate_pct", pa.float64(), nullable=True, description="Error rate (%).", min_value=0.0, max_value=100.0
    ),
    ColumnContract("paging_attempts", pa.float64(), nullable=True, description="Paging attempts.", min_value=0.0),
    ColumnContract(
        "paging_success_pct",
        pa.float64(),
        nullable=True,
        description="Paging success rate (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "handover_in_count", pa.float64(), nullable=True, description="Inbound handovers processed.", min_value=0.0
    ),
    ColumnContract(
        "handover_out_count", pa.float64(), nullable=True, description="Outbound handovers processed.", min_value=0.0
    ),
    ColumnContract(
        "bearer_activation_success_pct",
        pa.float64(),
        nullable=True,
        description="Bearer activation success rate (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "dns_query_count",
        pa.float64(),
        nullable=True,
        description="DNS query count (DNS resolvers only).",
        min_value=0.0,
    ),
    ColumnContract(
        "radius_auth_latency_ms",
        pa.float64(),
        nullable=True,
        description="RADIUS auth latency (AAA servers only).",
        min_value=0.0,
    ),
    ColumnContract(
        "sip_registrations",
        pa.float64(),
        nullable=True,
        description="SIP registrations (IMS elements only).",
        min_value=0.0,
    ),
    ColumnContract(
        "volte_call_setup_success_pct",
        pa.float64(),
        nullable=True,
        description="VoLTE call setup success (IMS TAS only) (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "element_availability_pct",
        pa.float64(),
        nullable=True,
        description="Element availability (%).",
        min_value=0.0,
        max_value=100.0,
    ),
]


def _core_element_kpis_wide_contract() -> FileContract:
    key_columns = [
        ColumnContract("entity_id", pa.string(), description="Core element entity_id (FK to ground_truth_entities)."),
        ColumnContract("tenant_id", pa.string(), description="Multi-tenant isolation key."),
        ColumnContract("timestamp", pa.timestamp("us", tz="UTC"), description="Hourly reporting interval start."),
        ColumnContract("entity_type", pa.string(), description="Core element type (MME, SGW, AMF, UPF, BNG, etc.)."),
        ColumnContract(
            "core_domain",
            pa.string(),
            nullable=True,
            description="Core sub-domain.",
            allowed_values=["epc", "5gc", "ims", "broadband_control", "enterprise_control", "voice"],
        ),
    ]
    return FileContract(
        filename="core_element_kpis_wide.parquet",
        description=("Core network element KPIs in wide format. ~500 entities × 20 metrics × 720 hours."),
        expected_row_count_approx=360_000,
        columns=key_columns + CORE_ELEMENT_KPI_COLUMNS,
    )


# ============================================================================
# 11. POWER / ENVIRONMENT KPIs
# ============================================================================

POWER_ENVIRONMENT_KPI_COLUMNS: list[ColumnContract] = [
    ColumnContract(
        "mains_power_status",
        pa.float64(),
        nullable=True,
        description="Mains power status (1.0 = OK, 0.0 = failed).",
        min_value=0.0,
        max_value=1.0,
    ),
    ColumnContract(
        "battery_voltage_v",
        pa.float64(),
        nullable=True,
        description="Battery bank voltage (V).",
        min_value=0.0,
        max_value=60.0,
    ),
    ColumnContract(
        "battery_charge_pct",
        pa.float64(),
        nullable=True,
        description="Battery charge level (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "cabinet_temperature_c",
        pa.float64(),
        nullable=True,
        description="Cabinet temperature (°C).",
        min_value=-10.0,
        max_value=60.0,
    ),
    ColumnContract(
        "cabinet_humidity_pct",
        pa.float64(),
        nullable=True,
        description="Cabinet humidity (%).",
        min_value=0.0,
        max_value=100.0,
    ),
    ColumnContract(
        "generator_runtime_hours",
        pa.float64(),
        nullable=True,
        description="Generator cumulative runtime (hours).",
        min_value=0.0,
    ),
    ColumnContract(
        "cooling_status",
        pa.float64(),
        nullable=True,
        description="Cooling system status (1.0 = OK, 0.0 = failed).",
        min_value=0.0,
        max_value=1.0,
    ),
]


def _power_environment_kpis_contract() -> FileContract:
    key_columns = [
        ColumnContract("site_id", pa.string(), description="Site entity_id (FK to ground_truth_entities)."),
        ColumnContract("tenant_id", pa.string(), description="Multi-tenant isolation key."),
        ColumnContract("timestamp", pa.timestamp("us", tz="UTC"), description="Hourly reporting interval start."),
        ColumnContract("site_type", pa.string(), description="Site type.", allowed_values=SITE_TYPES),
        ColumnContract(
            "deployment_profile", pa.string(), description="Deployment profile.", allowed_values=DEPLOYMENT_PROFILES
        ),
    ]
    return FileContract(
        filename="power_environment_kpis.parquet",
        description=("Site power/environment status. ~21,000 sites × ~7 metrics × 720 hours."),
        expected_row_count_approx=15_120_000,
        columns=key_columns + POWER_ENVIRONMENT_KPI_COLUMNS,
    )


# ============================================================================
# 12. EVENTS / ALARMS
# ============================================================================


def _events_alarms_contract() -> FileContract:
    return FileContract(
        filename="events_alarms.parquet",
        description=(
            "Multi-domain alarms. Each row is one alarm lifecycle event "
            "(raise or clear). Aligns with scenario injections; sleeping "
            "cells deliberately have NO corresponding alarms."
        ),
        expected_row_count_approx=500_000,
        columns=[
            ColumnContract("alarm_id", pa.string(), description="Unique alarm identifier (UUID v4)."),
            ColumnContract("tenant_id", pa.string(), description="Multi-tenant isolation key."),
            ColumnContract(
                "entity_id", pa.string(), description="Entity that raised the alarm (FK to ground_truth_entities)."
            ),
            ColumnContract("entity_type", pa.string(), description="Entity type.", allowed_values=ENTITY_TYPES),
            ColumnContract("alarm_type", pa.string(), description="Alarm classification.", allowed_values=ALARM_TYPES),
            ColumnContract("severity", pa.string(), description="Alarm severity.", allowed_values=ALARM_SEVERITIES),
            ColumnContract("raised_at", pa.timestamp("us", tz="UTC"), description="When the alarm was raised."),
            ColumnContract(
                "cleared_at",
                pa.timestamp("us", tz="UTC"),
                nullable=True,
                description="When the alarm was cleared (null if still active).",
            ),
            ColumnContract(
                "source_system",
                pa.string(),
                description="Source system that reported the alarm.",
                allowed_values=ALARM_SOURCE_SYSTEMS,
            ),
            ColumnContract("probable_cause", pa.string(), nullable=True, description="Probable cause description."),
            ColumnContract(
                "domain",
                pa.string(),
                description="Domain of the alarm.",
                allowed_values=["radio", "transport", "power", "core", "fixed_broadband"],
            ),
            ColumnContract(
                "scenario_id",
                pa.string(),
                nullable=True,
                description="FK to scenario injection that caused this alarm (null for organic alarms).",
            ),
            ColumnContract(
                "is_synthetic_scenario",
                pa.bool_(),
                description="True if this alarm was generated by scenario injection.",
            ),
            ColumnContract("additional_text", pa.string(), nullable=True, description="Free-text alarm detail."),
        ],
    )


# ============================================================================
# 13. CUSTOMERS / BSS
# ============================================================================


def _customers_bss_contract() -> FileContract:
    return FileContract(
        filename="customers_bss.parquet",
        description=(
            "Customer + billing + service plan data. ~1M customers with service plan associations and site bindings."
        ),
        expected_row_count_approx=1_000_000,
        columns=[
            ColumnContract("customer_id", pa.string(), description="Unique customer identifier (UUID v4)."),
            ColumnContract("tenant_id", pa.string(), description="Multi-tenant isolation key."),
            ColumnContract("external_id", pa.string(), description="Unique external ID (e.g., account number)."),
            ColumnContract(
                "customer_type", pa.string(), description="Customer type.", allowed_values=["residential", "enterprise"]
            ),
            ColumnContract("name", pa.string(), description="Customer name."),
            ColumnContract(
                "associated_site_id", pa.string(), description="FK to site entity_id that serves this customer."
            ),
            ColumnContract("province", pa.string(), description="Customer's province."),
            # Service plan
            ColumnContract("service_plan_name", pa.string(), description="Name of the subscribed service plan."),
            ColumnContract("service_plan_tier", pa.string(), description="Plan SLA tier.", allowed_values=SLA_TIERS),
            ColumnContract("monthly_fee", pa.float64(), description="Monthly subscription fee.", min_value=0.0),
            ColumnContract(
                "sla_guarantee",
                pa.string(),
                nullable=True,
                description="SLA guarantee text (e.g., '99.999% Availability').",
            ),
            # Billing account
            ColumnContract(
                "account_status",
                pa.string(),
                description="Billing account status.",
                allowed_values=["ACTIVE", "SUSPENDED", "DELINQUENT"],
            ),
            ColumnContract("avg_monthly_revenue", pa.float64(), description="Average monthly revenue.", min_value=0.0),
            ColumnContract(
                "contract_end_date", pa.timestamp("us", tz="UTC"), nullable=True, description="Contract end date."
            ),
            # CX / proactive care
            ColumnContract(
                "churn_risk_score",
                pa.float64(),
                description="Churn risk score (0.0 = low, 1.0 = high).",
                min_value=0.0,
                max_value=1.0,
            ),
            ColumnContract(
                "consent_proactive_comms", pa.bool_(), description="Customer consent for proactive communications."
            ),
            # Access technology
            ColumnContract(
                "access_type",
                pa.string(),
                description="How this customer connects.",
                allowed_values=["mobile", "fttp", "fttc", "enterprise_ethernet"],
            ),
            # For fixed broadband customers: FK to ONT/NTE
            ColumnContract(
                "access_entity_id",
                pa.string(),
                nullable=True,
                description="FK to access termination entity (ONT, NTE, etc.).",
            ),
        ],
    )


# ============================================================================
# 14. NEIGHBOUR RELATIONS
# ============================================================================


def _neighbour_relations_contract() -> FileContract:
    return FileContract(
        filename="neighbour_relations.parquet",
        description=(
            "Cell-to-cell adjacency with handover metrics. ~200K neighbour pairs "
            "with intra-freq, inter-freq, and inter-RAT relations."
        ),
        expected_row_count_approx=200_000,
        columns=[
            ColumnContract("relation_id", pa.string(), description="Unique relation identifier (UUID v4)."),
            ColumnContract("tenant_id", pa.string(), description="Multi-tenant isolation key."),
            ColumnContract("from_cell_id", pa.string(), description="Source cell entity_id."),
            ColumnContract("from_cell_rat", pa.string(), description="Source cell RAT.", allowed_values=RAT_TYPES),
            ColumnContract("from_cell_band", pa.string(), description="Source cell band."),
            ColumnContract("to_cell_id", pa.string(), description="Target cell entity_id."),
            ColumnContract("to_cell_rat", pa.string(), description="Target cell RAT.", allowed_values=RAT_TYPES),
            ColumnContract("to_cell_band", pa.string(), description="Target cell band."),
            ColumnContract(
                "neighbour_type",
                pa.string(),
                description="Neighbour relation type.",
                allowed_values=NEIGHBOUR_RELATION_TYPES,
            ),
            ColumnContract(
                "is_intra_site", pa.bool_(), description="True if both cells are on the same physical site."
            ),
            ColumnContract(
                "distance_m",
                pa.float64(),
                nullable=True,
                description="Distance between sites in metres.",
                min_value=0.0,
            ),
            # Handover metrics (aggregated over simulation period)
            ColumnContract(
                "handover_attempts",
                pa.float64(),
                description="Total handover attempts over simulation period.",
                min_value=0.0,
            ),
            ColumnContract(
                "handover_success_rate",
                pa.float64(),
                description="Handover success rate (%).",
                min_value=0.0,
                max_value=100.0,
            ),
            ColumnContract(
                "cio_offset_db",
                pa.float64(),
                description="Cell Individual Offset (dB).",
                min_value=-24.0,
                max_value=24.0,
            ),
            ColumnContract(
                "no_remove_flag", pa.bool_(), description="True if this neighbour relation is operator-locked."
            ),
        ],
    )


# ============================================================================
# CONTRACT REGISTRY
# ============================================================================


def get_all_contracts() -> dict[str, FileContract]:
    """
    Return a dict mapping filename → FileContract for all 14 output files.
    """
    contracts = [
        _ground_truth_entities_contract(),
        _ground_truth_relationships_contract(),
        _cmdb_declared_entities_contract(),
        _cmdb_declared_relationships_contract(),
        _divergence_manifest_contract(),
        _kpi_metrics_wide_contract(),
        _transport_kpis_wide_contract(),
        _fixed_broadband_kpis_wide_contract(),
        _enterprise_circuit_kpis_wide_contract(),
        _core_element_kpis_wide_contract(),
        _power_environment_kpis_contract(),
        _events_alarms_contract(),
        _customers_bss_contract(),
        _neighbour_relations_contract(),
    ]
    return {c.filename: c for c in contracts}


def get_contract(filename: str) -> FileContract:
    """Get the contract for a specific output file by filename."""
    contracts = get_all_contracts()
    if filename not in contracts:
        raise KeyError(f"No contract for '{filename}'. Available: {sorted(contracts.keys())}")
    return contracts[filename]


def get_kpi_column_names(domain: str) -> list[str]:
    """
    Get the list of KPI column names (excluding key columns) for a domain.

    Args:
        domain: One of 'radio', 'transport', 'fixed_broadband', 'enterprise',
                'core', 'power_environment'.

    Returns:
        List of KPI column names.
    """
    mapping = {
        "radio": [c.name for c in CELL_RADIO_KPI_COLUMNS],
        "radio_with_aliases": [c.name for c in CELL_RADIO_KPI_COLUMNS + PEDKAI_ALIAS_COLUMNS],
        "transport": [c.name for c in TRANSPORT_KPI_COLUMNS],
        "fixed_broadband": [c.name for c in FIXED_BROADBAND_KPI_COLUMNS],
        "enterprise": [c.name for c in ENTERPRISE_CIRCUIT_KPI_COLUMNS],
        "core": [c.name for c in CORE_ELEMENT_KPI_COLUMNS],
        "power_environment": [c.name for c in POWER_ENVIRONMENT_KPI_COLUMNS],
    }
    if domain not in mapping:
        raise KeyError(f"Unknown domain '{domain}'. Available: {sorted(mapping.keys())}")
    return mapping[domain]


def get_kpi_ranges(domain: str) -> dict[str, tuple[float | None, float | None]]:
    """
    Get min/max validation ranges for KPI columns in a given domain.

    Returns:
        Dict mapping column_name → (min_value, max_value).
        None means unbounded on that side.
    """
    col_lists = {
        "radio": CELL_RADIO_KPI_COLUMNS,
        "transport": TRANSPORT_KPI_COLUMNS,
        "fixed_broadband": FIXED_BROADBAND_KPI_COLUMNS,
        "enterprise": ENTERPRISE_CIRCUIT_KPI_COLUMNS,
        "core": CORE_ELEMENT_KPI_COLUMNS,
        "power_environment": POWER_ENVIRONMENT_KPI_COLUMNS,
    }
    if domain not in col_lists:
        raise KeyError(f"Unknown domain '{domain}'. Available: {sorted(col_lists.keys())}")

    return {c.name: (c.min_value, c.max_value) for c in col_lists[domain]}


# ============================================================================
# Step 00 entry point (called by CLI)
# ============================================================================


def generate_schema_contracts(config: GeneratorConfig) -> None:
    """
    Step 00: Generate and save schema contracts as metadata.

    This step doesn't produce data — it saves the schema contracts as
    JSON + PyArrow schema files alongside the output directory so that
    downstream steps and external tooling can reference them.
    """
    import json

    contracts = get_all_contracts()
    output_dir = config.paths.output_dir / "schemas"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for filename, contract in contracts.items():
        # Save PyArrow schema as IPC metadata
        schema_path = output_dir / f"{filename.replace('.parquet', '')}.schema.ipc"
        buf = contract.pa_schema.serialize()
        with open(schema_path, "wb") as f:
            f.write(buf.to_pybytes())

        # Build summary entry
        summary[filename] = {
            "description": contract.description,
            "expected_rows": contract.expected_row_count_approx,
            "columns": [
                {
                    "name": col.name,
                    "type": str(col.pa_type),
                    "nullable": col.nullable,
                    "description": col.description,
                    **({"min_value": col.min_value} if col.min_value is not None else {}),
                    **({"max_value": col.max_value} if col.max_value is not None else {}),
                    **({"allowed_values": col.allowed_values} if col.allowed_values is not None else {}),
                }
                for col in contract.columns
            ],
        }

    # Save summary JSON
    summary_path = output_dir / "all_contracts.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    from rich.console import Console

    console = Console()
    console.print(f"  [dim]Wrote {len(contracts)} schema contracts to {output_dir}[/dim]")
    console.print(f"  [dim]Summary: {summary_path}[/dim]")
    for filename, contract in contracts.items():
        console.print(
            f"  [dim]  {filename}: {len(contract.columns)} columns, "
            f"~{contract.expected_row_count_approx:,} rows expected[/dim]"
        )
