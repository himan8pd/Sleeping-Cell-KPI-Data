"""
Step 08: CMDB Degradation Engine.

Applies six Dark Graph divergence types to the ground-truth topology to
produce a "declared state" CMDB — the operator's imperfect view of their
own network.  The gap between ground truth (Phase 2) and declared state
(this phase) is exactly what Dark Graph must learn to reconcile.

Six divergence types (rates from ``CMDBDegradationConfig``):

  1. **Dark nodes** (6.5%): Entities exist in reality but are missing
     from the CMDB.  The entity row is simply dropped from
     ``cmdb_declared_entities``.

  2. **Phantom nodes** (3%): Entities appear in the CMDB but do not
     exist in reality.  Fabricated rows are inserted — using plausible
     but non-ground-truth attribute values.

  3. **Dark edges** (10%): Relationships exist in reality but are not
     declared in the CMDB.  The relationship row is dropped from
     ``cmdb_declared_relationships``.

  4. **Phantom edges** (5%): Relationships are declared in the CMDB but
     do not actually exist.  Fabricated rows connecting random compatible
     entities are inserted.

  5. **Dark attributes** (15%): Entity rows are present in the CMDB but
     one or more attribute values are stale / incorrect.  Typical
     mutations: wrong vendor, wrong band, stale lat/lon drift, wrong
     deployment profile, wrong SLA tier, corrupted external_id suffix.

  6. **Identity mutations** (2%): An entity's ``external_id`` has been
     corrupted — e.g. transposed characters, wrong region prefix, digit
     substitution.  The entity is still present but its identity link to
     the NMS is broken.

Output files:
  - output/cmdb_declared_entities.parquet   (same schema as ground_truth_entities)
  - output/cmdb_declared_relationships.parquet (same schema as ground_truth_relationships)
  - output/divergence_manifest.parquet      (labels for ML scoring)

Dependencies: Phase 2 (reads ground_truth_entities.parquet,
              ground_truth_relationships.parquet)
"""

from __future__ import annotations

import gc
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.table import Table

from pedkai_generator.config.settings import (
    DeploymentProfile,
    GeneratorConfig,
    SLATier,
    Vendor,
)

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Divergence type identifiers (match Phase 0 contract)
DIV_DARK_NODE = "dark_node"
DIV_PHANTOM_NODE = "phantom_node"
DIV_DARK_EDGE = "dark_edge"
DIV_PHANTOM_EDGE = "phantom_edge"
DIV_DARK_ATTRIBUTE = "dark_attribute"
DIV_IDENTITY_MUTATION = "identity_mutation"

# Entity types that should NOT be dark-noded (critical infrastructure)
# Removing a SITE would cascade-orphan too many children; we still allow
# non-SITE entities to be dark-noded.
PROTECTED_ENTITY_TYPES = {"SITE"}

# Attributes eligible for dark-attribute corruption
MUTABLE_ATTRIBUTES: list[dict[str, Any]] = [
    {
        "name": "vendor",
        "applicable_types": None,  # all entity types
        "mutate_fn": "_mutate_vendor",
    },
    {
        "name": "deployment_profile",
        "applicable_types": {"SITE", "LTE_CELL", "NR_CELL", "ENODEB", "GNODEB"},
        "mutate_fn": "_mutate_deployment_profile",
    },
    {
        "name": "sla_tier",
        "applicable_types": None,
        "mutate_fn": "_mutate_sla_tier",
    },
    {
        "name": "geo_lat",
        "applicable_types": {"SITE"},
        "mutate_fn": "_mutate_geo_lat",
    },
    {
        "name": "geo_lon",
        "applicable_types": {"SITE"},
        "mutate_fn": "_mutate_geo_lon",
    },
    {
        "name": "band",
        "applicable_types": {"LTE_CELL", "NR_CELL"},
        "mutate_fn": "_mutate_band",
    },
    {
        "name": "province",
        "applicable_types": {"SITE"},
        "mutate_fn": "_mutate_province",
    },
    {
        "name": "site_type",
        "applicable_types": {"SITE"},
        "mutate_fn": "_mutate_site_type",
    },
]

# Relationship types eligible for phantom edge injection — we pick
# compatible (from_type, rel_type, to_type) triples
PHANTOM_EDGE_TEMPLATES: list[dict[str, str]] = [
    {"from_type": "SITE", "rel_type": "HOSTS", "to_type": "CABINET"},
    {"from_type": "CABINET", "rel_type": "HOSTS", "to_type": "BBU"},
    {"from_type": "ENODEB", "rel_type": "CONTAINS", "to_type": "LTE_CELL"},
    {"from_type": "ACCESS_SWITCH", "rel_type": "UPLINKS_TO", "to_type": "AGGREGATION_SWITCH"},
    {"from_type": "AGGREGATION_SWITCH", "rel_type": "UPLINKS_TO", "to_type": "PE_ROUTER"},
    {"from_type": "OLT", "rel_type": "CONTAINS", "to_type": "PON_PORT"},
    {"from_type": "SPLITTER", "rel_type": "SPLITS_TO", "to_type": "ONT"},
    {"from_type": "P_ROUTER", "rel_type": "PEERS_WITH", "to_type": "P_ROUTER"},
    {"from_type": "PE_ROUTER", "rel_type": "PEERS_WITH", "to_type": "PE_ROUTER"},
    {"from_type": "GNODEB", "rel_type": "CONTAINS", "to_type": "NR_CELL"},
    {"from_type": "SITE", "rel_type": "HOSTS", "to_type": "POWER_SUPPLY"},
    {"from_type": "SITE", "rel_type": "HOSTS", "to_type": "ANTENNA_SYSTEM"},
]

# Province list for random corruption (subset — full list in config)
PROVINCE_NAMES = [
    "Aceh",
    "Sumatera Utara",
    "Sumatera Barat",
    "Riau",
    "DKI Jakarta",
    "Jawa Barat",
    "Jawa Tengah",
    "Jawa Timur",
    "Bali",
    "Kalimantan Timur",
    "Sulawesi Selatan",
    "Papua",
    "Banten",
    "DI Yogyakarta",
    "Lampung",
    "Kepulauan Riau",
]

BAND_NAMES_LTE = ["L900", "L1800", "L2100", "L2300"]
BAND_NAMES_NR = ["n1", "n3", "n28", "n78", "n77", "n257"]

SITE_TYPES = ["greenfield", "rooftop", "streetworks", "in_building", "unspecified"]
DEPLOYMENT_PROFILES = ["dense_urban", "urban", "suburban", "rural", "deep_rural", "indoor"]
VENDOR_NAMES = ["ericsson", "nokia"]
SLA_TIERS_LIST = ["GOLD", "SILVER", "BRONZE"]

# ---------------------------------------------------------------------------
# PyArrow schema for divergence_manifest (13 columns — Phase 0 contract)
# ---------------------------------------------------------------------------

DIVERGENCE_MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("divergence_id", pa.string(), nullable=False),
        pa.field("tenant_id", pa.string(), nullable=False),
        pa.field("divergence_type", pa.string(), nullable=False),
        pa.field("entity_or_relationship", pa.string(), nullable=False),
        pa.field("target_id", pa.string(), nullable=False),
        pa.field("target_type", pa.string(), nullable=False),
        pa.field("domain", pa.string(), nullable=False),
        pa.field("description", pa.string(), nullable=False),
        pa.field("attribute_name", pa.string(), nullable=True),
        pa.field("ground_truth_value", pa.string(), nullable=True),
        pa.field("cmdb_declared_value", pa.string(), nullable=True),
        pa.field("original_external_id", pa.string(), nullable=True),
        pa.field("mutated_external_id", pa.string(), nullable=True),
    ]
)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_entities(config: GeneratorConfig) -> pl.DataFrame:
    """Load ground_truth_entities.parquet from Phase 2 output."""
    path = config.paths.output_dir / "ground_truth_entities.parquet"
    return pl.read_parquet(path)


def _load_relationships(config: GeneratorConfig) -> pl.DataFrame:
    """Load ground_truth_relationships.parquet from Phase 2 output."""
    path = config.paths.output_dir / "ground_truth_relationships.parquet"
    return pl.read_parquet(path)


# ---------------------------------------------------------------------------
# Attribute mutation functions
# ---------------------------------------------------------------------------


def _mutate_vendor(current_value: Any, rng: np.random.Generator) -> str:
    """Flip vendor: ericsson → nokia, nokia → ericsson."""
    if current_value == "ericsson":
        return "nokia"
    return "ericsson"


def _mutate_deployment_profile(current_value: Any, rng: np.random.Generator) -> str:
    """Pick a different deployment profile."""
    options = [p for p in DEPLOYMENT_PROFILES if p != current_value]
    return str(rng.choice(options))


def _mutate_sla_tier(current_value: Any, rng: np.random.Generator) -> str:
    """Pick a different SLA tier."""
    options = [t for t in SLA_TIERS_LIST if t != current_value]
    return str(rng.choice(options))


def _mutate_geo_lat(current_value: Any, rng: np.random.Generator) -> float:
    """Add a stale drift to latitude (0.01–0.1 degrees ≈ 1–11 km)."""
    if current_value is None:
        return 0.0
    drift = float(rng.uniform(0.01, 0.1)) * rng.choice([-1, 1])
    return round(float(current_value) + drift, 6)


def _mutate_geo_lon(current_value: Any, rng: np.random.Generator) -> float:
    """Add a stale drift to longitude."""
    if current_value is None:
        return 0.0
    drift = float(rng.uniform(0.01, 0.1)) * rng.choice([-1, 1])
    return round(float(current_value) + drift, 6)


def _mutate_band(current_value: Any, rng: np.random.Generator) -> str:
    """Pick a different band."""
    if current_value is not None and str(current_value).startswith("n"):
        options = [b for b in BAND_NAMES_NR if b != current_value]
    else:
        options = [b for b in BAND_NAMES_LTE if b != current_value]
    if not options:
        return str(current_value) if current_value is not None else "L900"
    return str(rng.choice(options))


def _mutate_province(current_value: Any, rng: np.random.Generator) -> str:
    """Pick a different province."""
    options = [p for p in PROVINCE_NAMES if p != current_value]
    return str(rng.choice(options))


def _mutate_site_type(current_value: Any, rng: np.random.Generator) -> str:
    """Pick a different site type."""
    options = [s for s in SITE_TYPES if s != current_value]
    return str(rng.choice(options))


# Dispatch map
MUTATION_DISPATCH: dict[str, Any] = {
    "_mutate_vendor": _mutate_vendor,
    "_mutate_deployment_profile": _mutate_deployment_profile,
    "_mutate_sla_tier": _mutate_sla_tier,
    "_mutate_geo_lat": _mutate_geo_lat,
    "_mutate_geo_lon": _mutate_geo_lon,
    "_mutate_band": _mutate_band,
    "_mutate_province": _mutate_province,
    "_mutate_site_type": _mutate_site_type,
}


# ---------------------------------------------------------------------------
# Identity mutation helpers
# ---------------------------------------------------------------------------


def _mutate_external_id(external_id: str | None, rng: np.random.Generator) -> str:
    """
    Apply a plausible identity mutation to an external_id string.

    Mutation types (randomly selected):
      - Character transposition (swap two adjacent characters)
      - Digit substitution (replace one digit with a different digit)
      - Region prefix corruption (change the first segment)
      - Suffix truncation (drop last 2-4 characters)
      - Character duplication (repeat one character)
    """
    if external_id is None or len(external_id) < 4:
        # Generate a completely synthetic corrupted ID
        return f"CORRUPT-{uuid.uuid4().hex[:12].upper()}"

    eid = str(external_id)
    mutation_type = rng.integers(0, 5)

    if mutation_type == 0:
        # Character transposition — swap two adjacent chars
        if len(eid) < 3:
            return eid + "_X"
        pos = int(rng.integers(1, len(eid) - 1))
        chars = list(eid)
        chars[pos], chars[pos - 1] = chars[pos - 1], chars[pos]
        return "".join(chars)

    elif mutation_type == 1:
        # Digit substitution — find a digit and replace it
        digit_positions = [i for i, c in enumerate(eid) if c.isdigit()]
        if not digit_positions:
            # No digits — insert a wrong one
            return eid[:-1] + str(rng.integers(0, 10))
        pos = int(rng.choice(digit_positions))
        original_digit = int(eid[pos])
        # Pick a different digit
        new_digit = (original_digit + int(rng.integers(1, 9))) % 10
        return eid[:pos] + str(new_digit) + eid[pos + 1 :]

    elif mutation_type == 2:
        # Region prefix corruption — change first segment before separator
        separators = ["/", "-", "_", "=", ","]
        for sep in separators:
            if sep in eid:
                parts = eid.split(sep, 1)
                # Corrupt the first part
                first = parts[0]
                if len(first) > 2:
                    chars = list(first)
                    idx = int(rng.integers(0, len(chars)))
                    chars[idx] = chr(ord("A") + int(rng.integers(0, 26)))
                    return "".join(chars) + sep + parts[1]
                break
        # Fallback: just change first char
        return chr(ord("A") + int(rng.integers(0, 26))) + eid[1:]

    elif mutation_type == 3:
        # Suffix truncation — drop 2-4 chars from end
        drop = int(rng.integers(2, min(5, len(eid) - 2)))
        return eid[:-drop]

    else:
        # Character duplication — repeat one character
        pos = int(rng.integers(0, len(eid)))
        return eid[:pos] + eid[pos] + eid[pos:]


# ---------------------------------------------------------------------------
# Phantom node generation
# ---------------------------------------------------------------------------


def _generate_phantom_entity(
    template_entity: dict[str, Any],
    tenant_id: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """
    Create a phantom entity based on a template (an existing real entity).

    The phantom has:
      - A new UUID (not in ground truth)
      - The same entity_type and domain
      - A plausible but fake name
      - Slightly perturbed geo coordinates (if applicable)
      - Same site_id (it appears to be at the same site)
    """
    phantom = dict(template_entity)  # shallow copy
    phantom_id = str(uuid.uuid4())
    phantom["entity_id"] = phantom_id

    # Generate a plausible name
    etype = phantom.get("entity_type", "UNKNOWN")
    phantom["name"] = f"{etype}-PHANTOM-{phantom_id[:8].upper()}"

    # Generate a new external_id
    vendor = phantom.get("vendor", "ericsson")
    region = phantom.get("province", "")
    if vendor == "ericsson":
        phantom["external_id"] = (
            f"SubNetwork=ERBS_{str(region)[:6]},MeContext={etype[:8]}_{phantom_id[:8]},ManagedElement=1"
        )
    else:
        phantom["external_id"] = f"PLMN-PLMN/{str(region)[:4]}/{etype[:8]}-{phantom_id[:8]}"

    # Perturb geo if present
    if phantom.get("geo_lat") is not None:
        drift_lat = float(rng.uniform(-0.005, 0.005))
        drift_lon = float(rng.uniform(-0.005, 0.005))
        phantom["geo_lat"] = round(float(phantom["geo_lat"]) + drift_lat, 6)
        phantom["geo_lon"] = round(float(phantom["geo_lon"]) + drift_lon, 6)

    # Clear NSA-specific fields for non-cell types
    if etype not in ("LTE_CELL", "NR_CELL"):
        phantom["is_nsa_anchor"] = None
        phantom["nsa_anchor_cell_id"] = None

    return phantom


# ---------------------------------------------------------------------------
# Phantom edge generation
# ---------------------------------------------------------------------------


def _generate_phantom_relationship(
    entity_ids_by_type: dict[str, list[str]],
    tenant_id: str,
    existing_edge_set: set[tuple[str, str]],
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    """
    Create a phantom relationship between two compatible entities.

    Picks a random template from PHANTOM_EDGE_TEMPLATES and selects
    random entities of the required types. Returns None if unable to
    find compatible entities.
    """
    # Try a few templates before giving up
    for _ in range(10):
        template = PHANTOM_EDGE_TEMPLATES[int(rng.integers(0, len(PHANTOM_EDGE_TEMPLATES)))]
        from_type = template["from_type"]
        to_type = template["to_type"]
        rel_type = template["rel_type"]

        from_pool = entity_ids_by_type.get(from_type, [])
        to_pool = entity_ids_by_type.get(to_type, [])

        if not from_pool or not to_pool:
            continue

        from_id = from_pool[int(rng.integers(0, len(from_pool)))]
        to_id = to_pool[int(rng.integers(0, len(to_pool)))]

        # Avoid self-loops
        if from_id == to_id:
            continue

        # Avoid duplicating an existing edge
        edge_key = (from_id, to_id)
        if edge_key in existing_edge_set:
            continue

        # Determine domain
        domain = _infer_domain(from_type, to_type)

        return {
            "relationship_id": str(uuid.uuid4()),
            "tenant_id": tenant_id,
            "from_entity_id": from_id,
            "from_entity_type": from_type,
            "relationship_type": rel_type,
            "to_entity_id": to_id,
            "to_entity_type": to_type,
            "domain": domain,
            "properties_json": json.dumps({"phantom": True}),
        }

    return None


def _infer_domain(from_type: str, to_type: str) -> str:
    """Infer relationship domain from entity types."""
    ran_types = {
        "SITE",
        "CABINET",
        "POWER_SUPPLY",
        "BATTERY_BANK",
        "MAINS_CONNECTION",
        "CLIMATE_CONTROL",
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
    }
    transport_types = {
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
    }
    fixed_types = {
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
    }
    core_types = {
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
    }
    power_types = {
        "BATTERY",
        "GENERATOR",
    }

    types = {from_type, to_type}
    if types & ran_types:
        return "mobile_ran"
    if types & transport_types:
        return "transport"
    if types & fixed_types:
        return "fixed_access"
    if types & core_types:
        return "core"
    if types & power_types:
        return "power_environment"
    return "cross_domain"


# ---------------------------------------------------------------------------
# Main divergence application
# ---------------------------------------------------------------------------


def _apply_dark_nodes(
    entities_df: pl.DataFrame,
    rate: float,
    tenant_id: str,
    rng: np.random.Generator,
) -> tuple[set[str], list[dict[str, Any]]]:
    """
    Select entities to remove (dark nodes).

    Returns:
      - set of entity_ids to remove
      - list of divergence manifest rows
    """
    # Filter out protected types
    eligible_mask = ~entities_df["entity_type"].is_in(list(PROTECTED_ENTITY_TYPES))
    eligible_df = entities_df.filter(eligible_mask)
    n_eligible = eligible_df.height

    n_dark = int(n_eligible * rate)
    if n_dark == 0:
        return set(), []

    # Random sample of indices
    dark_indices = rng.choice(n_eligible, size=n_dark, replace=False)
    dark_entity_ids: set[str] = set()
    manifest_rows: list[dict[str, Any]] = []

    entity_ids = eligible_df["entity_id"].to_list()
    entity_types = eligible_df["entity_type"].to_list()
    domains = eligible_df["domain"].to_list()

    for idx in dark_indices:
        eid = str(entity_ids[idx])
        dark_entity_ids.add(eid)
        manifest_rows.append(
            {
                "divergence_id": str(uuid.uuid4()),
                "tenant_id": tenant_id,
                "divergence_type": DIV_DARK_NODE,
                "entity_or_relationship": "entity",
                "target_id": eid,
                "target_type": str(entity_types[idx]),
                "domain": str(domains[idx]),
                "description": f"Entity {eid} exists in reality but is missing from CMDB (dark node).",
                "attribute_name": None,
                "ground_truth_value": None,
                "cmdb_declared_value": None,
                "original_external_id": None,
                "mutated_external_id": None,
            }
        )

    return dark_entity_ids, manifest_rows


def _apply_phantom_nodes(
    entities_df: pl.DataFrame,
    rate: float,
    tenant_id: str,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Generate phantom entities (exist in CMDB but not in reality).

    Returns:
      - list of phantom entity row dicts
      - list of divergence manifest rows
    """
    n_entities = entities_df.height
    n_phantom = int(n_entities * rate)
    if n_phantom == 0:
        return [], []

    # Use random real entities as templates
    template_indices = rng.integers(0, n_entities, size=n_phantom)
    entity_rows = entities_df.to_dicts()

    phantom_entities: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for idx in template_indices:
        template = entity_rows[int(idx)]
        phantom = _generate_phantom_entity(template, tenant_id, rng)
        phantom_entities.append(phantom)

        manifest_rows.append(
            {
                "divergence_id": str(uuid.uuid4()),
                "tenant_id": tenant_id,
                "divergence_type": DIV_PHANTOM_NODE,
                "entity_or_relationship": "entity",
                "target_id": phantom["entity_id"],
                "target_type": phantom["entity_type"],
                "domain": phantom.get("domain", "unknown"),
                "description": (
                    f"Phantom entity {phantom['entity_id']} exists in CMDB but not in reality "
                    f"(type={phantom['entity_type']})."
                ),
                "attribute_name": None,
                "ground_truth_value": None,
                "cmdb_declared_value": None,
                "original_external_id": None,
                "mutated_external_id": None,
            }
        )

    return phantom_entities, manifest_rows


def _apply_dark_edges(
    relationships_df: pl.DataFrame,
    rate: float,
    tenant_id: str,
    rng: np.random.Generator,
) -> tuple[set[str], list[dict[str, Any]]]:
    """
    Select relationships to remove (dark edges).

    Returns:
      - set of relationship_ids to remove
      - list of divergence manifest rows
    """
    n_rels = relationships_df.height
    n_dark = int(n_rels * rate)
    if n_dark == 0:
        return set(), []

    dark_indices = rng.choice(n_rels, size=n_dark, replace=False)
    dark_rel_ids: set[str] = set()
    manifest_rows: list[dict[str, Any]] = []

    rel_ids = relationships_df["relationship_id"].to_list()
    rel_types = relationships_df["relationship_type"].to_list()
    domains = relationships_df["domain"].to_list()

    for idx in dark_indices:
        rid = str(rel_ids[idx])
        dark_rel_ids.add(rid)
        manifest_rows.append(
            {
                "divergence_id": str(uuid.uuid4()),
                "tenant_id": tenant_id,
                "divergence_type": DIV_DARK_EDGE,
                "entity_or_relationship": "relationship",
                "target_id": rid,
                "target_type": str(rel_types[idx]),
                "domain": str(domains[idx]),
                "description": (f"Relationship {rid} exists in reality but is missing from CMDB (dark edge)."),
                "attribute_name": None,
                "ground_truth_value": None,
                "cmdb_declared_value": None,
                "original_external_id": None,
                "mutated_external_id": None,
            }
        )

    return dark_rel_ids, manifest_rows


def _apply_phantom_edges(
    relationships_df: pl.DataFrame,
    entities_df: pl.DataFrame,
    rate: float,
    tenant_id: str,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Generate phantom relationships (declared but don't exist in reality).

    Returns:
      - list of phantom relationship row dicts
      - list of divergence manifest rows
    """
    n_rels = relationships_df.height
    n_phantom = int(n_rels * rate)
    if n_phantom == 0:
        return [], []

    # Build entity type → entity_id lookup for phantom edge generation
    entity_ids_by_type: dict[str, list[str]] = {}
    for row in entities_df.select("entity_id", "entity_type").iter_rows():
        eid, etype = str(row[0]), str(row[1])
        entity_ids_by_type.setdefault(etype, []).append(eid)

    # Build existing edge set to avoid duplicates
    existing_edge_set: set[tuple[str, str]] = set()
    for row in relationships_df.select("from_entity_id", "to_entity_id").iter_rows():
        existing_edge_set.add((str(row[0]), str(row[1])))

    phantom_rels: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    generated = 0
    max_attempts = n_phantom * 5  # allow some failures
    attempts = 0

    while generated < n_phantom and attempts < max_attempts:
        attempts += 1
        phantom_rel = _generate_phantom_relationship(entity_ids_by_type, tenant_id, existing_edge_set, rng)
        if phantom_rel is None:
            continue

        phantom_rels.append(phantom_rel)
        existing_edge_set.add((phantom_rel["from_entity_id"], phantom_rel["to_entity_id"]))

        manifest_rows.append(
            {
                "divergence_id": str(uuid.uuid4()),
                "tenant_id": tenant_id,
                "divergence_type": DIV_PHANTOM_EDGE,
                "entity_or_relationship": "relationship",
                "target_id": phantom_rel["relationship_id"],
                "target_type": phantom_rel["relationship_type"],
                "domain": phantom_rel.get("domain", "cross_domain"),
                "description": (
                    f"Phantom relationship {phantom_rel['relationship_id']} "
                    f"({phantom_rel['from_entity_type']} → {phantom_rel['to_entity_type']}) "
                    f"exists in CMDB but not in reality."
                ),
                "attribute_name": None,
                "ground_truth_value": None,
                "cmdb_declared_value": None,
                "original_external_id": None,
                "mutated_external_id": None,
            }
        )
        generated += 1

    return phantom_rels, manifest_rows


def _apply_dark_attributes(
    entities_df: pl.DataFrame,
    rate: float,
    tenant_id: str,
    rng: np.random.Generator,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """
    Select entities whose attributes should be corrupted in the CMDB.

    Returns:
      - dict mapping entity_id → {attr_name: new_value, ...} for overrides
      - list of divergence manifest rows
    """
    n_entities = entities_df.height
    n_corrupt = int(n_entities * rate)
    if n_corrupt == 0:
        return {}, []

    corrupt_indices = rng.choice(n_entities, size=n_corrupt, replace=False)
    entity_rows = entities_df.to_dicts()

    overrides: dict[str, dict[str, Any]] = {}
    manifest_rows: list[dict[str, Any]] = []

    for idx in corrupt_indices:
        row = entity_rows[int(idx)]
        eid = str(row["entity_id"])
        etype = str(row.get("entity_type", ""))
        domain = str(row.get("domain", "unknown"))

        # Find applicable attributes for this entity type
        applicable_attrs = []
        for attr_spec in MUTABLE_ATTRIBUTES:
            attr_name = attr_spec["name"]
            applicable_types = attr_spec["applicable_types"]

            # Skip if attribute has no value to corrupt
            if row.get(attr_name) is None:
                continue

            # Check type applicability
            if applicable_types is not None and etype not in applicable_types:
                continue

            applicable_attrs.append(attr_spec)

        if not applicable_attrs:
            continue

        # Pick 1-2 attributes to corrupt per entity
        n_attrs = min(len(applicable_attrs), int(rng.integers(1, 3)))
        chosen_indices = rng.choice(len(applicable_attrs), size=n_attrs, replace=False)

        entity_overrides: dict[str, Any] = {}

        for ai in chosen_indices:
            attr_spec = applicable_attrs[int(ai)]
            attr_name = attr_spec["name"]
            mutate_fn_name = attr_spec["mutate_fn"]
            mutate_fn = MUTATION_DISPATCH[mutate_fn_name]

            original_value = row.get(attr_name)
            new_value = mutate_fn(original_value, rng)

            entity_overrides[attr_name] = new_value

            manifest_rows.append(
                {
                    "divergence_id": str(uuid.uuid4()),
                    "tenant_id": tenant_id,
                    "divergence_type": DIV_DARK_ATTRIBUTE,
                    "entity_or_relationship": "entity",
                    "target_id": eid,
                    "target_type": etype,
                    "domain": domain,
                    "description": (
                        f"Attribute '{attr_name}' of entity {eid} is stale/wrong in CMDB: "
                        f"ground_truth='{original_value}' vs cmdb='{new_value}'."
                    ),
                    "attribute_name": attr_name,
                    "ground_truth_value": str(original_value) if original_value is not None else None,
                    "cmdb_declared_value": str(new_value) if new_value is not None else None,
                    "original_external_id": None,
                    "mutated_external_id": None,
                }
            )

        if entity_overrides:
            overrides[eid] = entity_overrides

    return overrides, manifest_rows


def _apply_identity_mutations(
    entities_df: pl.DataFrame,
    rate: float,
    tenant_id: str,
    rng: np.random.Generator,
    already_corrupted: set[str],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """
    Select entities whose external_id should be corrupted (identity mutation).

    Only targets entities NOT already affected by dark_node or dark_attribute
    to keep divergence types independent.

    Returns:
      - dict mapping entity_id → mutated_external_id
      - list of divergence manifest rows
    """
    # Filter to entities with a non-null external_id, not already corrupted
    eligible = entities_df.filter(
        pl.col("external_id").is_not_null() & ~pl.col("entity_id").is_in(list(already_corrupted))
    )
    n_eligible = eligible.height
    n_mutate = int(n_eligible * rate)
    if n_mutate == 0:
        return {}, []

    mutate_indices = rng.choice(n_eligible, size=n_mutate, replace=False)

    entity_ids = eligible["entity_id"].to_list()
    external_ids = eligible["external_id"].to_list()
    entity_types = eligible["entity_type"].to_list()
    domains = eligible["domain"].to_list()

    mutations: dict[str, str] = {}
    manifest_rows: list[dict[str, Any]] = []

    for idx in mutate_indices:
        eid = str(entity_ids[idx])
        original_ext_id = str(external_ids[idx])
        etype = str(entity_types[idx])
        domain = str(domains[idx])

        mutated_ext_id = _mutate_external_id(original_ext_id, rng)
        mutations[eid] = mutated_ext_id

        manifest_rows.append(
            {
                "divergence_id": str(uuid.uuid4()),
                "tenant_id": tenant_id,
                "divergence_type": DIV_IDENTITY_MUTATION,
                "entity_or_relationship": "entity",
                "target_id": eid,
                "target_type": etype,
                "domain": domain,
                "description": (
                    f"Identity mutation on entity {eid}: external_id corrupted from "
                    f"'{original_ext_id}' to '{mutated_ext_id}'."
                ),
                "attribute_name": "external_id",
                "ground_truth_value": None,
                "cmdb_declared_value": None,
                "original_external_id": original_ext_id,
                "mutated_external_id": mutated_ext_id,
            }
        )

    return mutations, manifest_rows


# ---------------------------------------------------------------------------
# Assembler: build CMDB outputs from ground truth + divergences
# ---------------------------------------------------------------------------


def _build_cmdb_entities(
    entities_df: pl.DataFrame,
    dark_node_ids: set[str],
    phantom_entities: list[dict[str, Any]],
    attribute_overrides: dict[str, dict[str, Any]],
    identity_mutations: dict[str, str],
) -> pl.DataFrame:
    """
    Build the degraded CMDB entity DataFrame:
      1. Remove dark nodes
      2. Apply attribute overrides
      3. Apply identity mutations
      4. Append phantom nodes
    """
    # 1. Remove dark nodes
    cmdb_df = entities_df.filter(~pl.col("entity_id").is_in(list(dark_node_ids)))

    # 2+3. Apply attribute overrides and identity mutations row-by-row
    # Convert to dicts for mutation, then back to DataFrame
    if attribute_overrides or identity_mutations:
        rows = cmdb_df.to_dicts()
        for row in rows:
            eid = str(row["entity_id"])

            # Apply attribute overrides
            if eid in attribute_overrides:
                for attr_name, new_value in attribute_overrides[eid].items():
                    row[attr_name] = new_value

            # Apply identity mutations
            if eid in identity_mutations:
                row["external_id"] = identity_mutations[eid]

        cmdb_df = pl.DataFrame(rows, schema=entities_df.schema)

    # 4. Append phantom nodes
    if phantom_entities:
        # Ensure phantom rows have all columns
        columns = entities_df.columns
        normalised_phantoms: list[dict[str, Any]] = []
        for phantom in phantom_entities:
            normalised = {col: phantom.get(col) for col in columns}
            normalised_phantoms.append(normalised)

        phantom_df = pl.DataFrame(normalised_phantoms, schema=entities_df.schema)
        cmdb_df = pl.concat([cmdb_df, phantom_df], how="vertical_relaxed")

    return cmdb_df


def _build_cmdb_relationships(
    relationships_df: pl.DataFrame,
    dark_edge_ids: set[str],
    phantom_rels: list[dict[str, Any]],
    dark_node_ids: set[str],
) -> pl.DataFrame:
    """
    Build the degraded CMDB relationship DataFrame:
      1. Remove dark edges
      2. Remove edges where either endpoint is a dark node
      3. Append phantom edges
    """
    # 1. Remove dark edges
    cmdb_df = relationships_df.filter(~pl.col("relationship_id").is_in(list(dark_edge_ids)))

    # 2. Remove edges referencing dark nodes (cascade)
    if dark_node_ids:
        dark_node_list = list(dark_node_ids)
        cmdb_df = cmdb_df.filter(
            ~pl.col("from_entity_id").is_in(dark_node_list) & ~pl.col("to_entity_id").is_in(dark_node_list)
        )

    # 3. Append phantom edges
    if phantom_rels:
        columns = relationships_df.columns
        normalised_phantoms: list[dict[str, Any]] = []
        for rel in phantom_rels:
            normalised = {col: rel.get(col) for col in columns}
            normalised_phantoms.append(normalised)

        phantom_df = pl.DataFrame(normalised_phantoms, schema=relationships_df.schema)
        cmdb_df = pl.concat([cmdb_df, phantom_df], how="vertical_relaxed")

    return cmdb_df


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------


def _write_manifest(
    manifest_rows: list[dict[str, Any]],
    output_path: Path,
) -> tuple[int, float]:
    """
    Write the divergence manifest to Parquet.

    Returns (rows_written, file_size_mb).
    """
    if not manifest_rows:
        # Write an empty file with correct schema
        empty_batch = pa.RecordBatch.from_pydict(
            {name: [] for name in DIVERGENCE_MANIFEST_SCHEMA.names},
            schema=DIVERGENCE_MANIFEST_SCHEMA,
        )
        writer = pq.ParquetWriter(
            output_path,
            schema=DIVERGENCE_MANIFEST_SCHEMA,
            compression="zstd",
            compression_level=3,
        )
        writer.write_batch(empty_batch)
        writer.close()
        return 0, 0.0

    # Build column arrays
    arrays = []
    for field in DIVERGENCE_MANIFEST_SCHEMA:
        col_data = [row.get(field.name) for row in manifest_rows]
        arrays.append(pa.array(col_data, type=field.type))

    batch = pa.RecordBatch.from_arrays(arrays, schema=DIVERGENCE_MANIFEST_SCHEMA)

    writer = pq.ParquetWriter(
        output_path,
        schema=DIVERGENCE_MANIFEST_SCHEMA,
        compression="zstd",
        compression_level=3,
    )
    writer.write_batch(batch)
    writer.close()

    size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0.0
    return len(manifest_rows), size_mb


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def degrade_cmdb(config: GeneratorConfig) -> None:
    """
    Step 08 entry point: Apply Dark Graph divergences to ground-truth topology.

    Produces:
      - output/cmdb_declared_entities.parquet
      - output/cmdb_declared_relationships.parquet
      - output/divergence_manifest.parquet
    """
    step_start = time.time()

    seed = config.seed_for("step_08_cmdb_degradation")
    rng = np.random.default_rng(seed)
    console.print(f"[dim]Step 08 seed: {seed}[/dim]")

    tenant_id = config.tenant_id
    deg = config.cmdb_degradation

    console.print(
        f"[bold]CMDB Degradation:[/bold] 6 divergence types "
        f"(dark_node={deg.dark_node_rate:.1%}, phantom_node={deg.phantom_node_rate:.1%}, "
        f"dark_edge={deg.dark_edge_rate:.1%}, phantom_edge={deg.phantom_edge_rate:.1%}, "
        f"dark_attr={deg.dark_attribute_rate:.1%}, id_mutation={deg.identity_mutation_rate:.1%})"
    )

    config.ensure_output_dirs()

    # ── Load ground truth ────────────────────────────────────
    console.print("\n[bold]Loading ground truth topology...[/bold]")
    t0 = time.time()
    entities_df = _load_entities(config)
    relationships_df = _load_relationships(config)
    console.print(
        f"  [green]✓[/green] {entities_df.height:,} entities, "
        f"{relationships_df.height:,} relationships loaded "
        f"in {time.time() - t0:.1f}s"
    )

    gt_entity_count = entities_df.height
    gt_rel_count = relationships_df.height

    all_manifest_rows: list[dict[str, Any]] = []

    # ── 1. Dark nodes ─────────────────────────────────────────
    console.print("\n[bold]1. Dark nodes[/bold] (entities missing from CMDB)...")
    t0 = time.time()
    dark_node_ids, dark_node_manifest = _apply_dark_nodes(entities_df, deg.dark_node_rate, tenant_id, rng)
    all_manifest_rows.extend(dark_node_manifest)
    console.print(
        f"  [green]✓[/green] {len(dark_node_ids):,} entities marked as dark nodes "
        f"({100.0 * len(dark_node_ids) / max(1, gt_entity_count):.1f}%) "
        f"in {time.time() - t0:.1f}s"
    )

    # ── 2. Phantom nodes ──────────────────────────────────────
    console.print("\n[bold]2. Phantom nodes[/bold] (fabricated entities in CMDB)...")
    t0 = time.time()
    phantom_entities, phantom_node_manifest = _apply_phantom_nodes(entities_df, deg.phantom_node_rate, tenant_id, rng)
    all_manifest_rows.extend(phantom_node_manifest)
    console.print(f"  [green]✓[/green] {len(phantom_entities):,} phantom entities generated in {time.time() - t0:.1f}s")

    # ── 3. Dark edges ─────────────────────────────────────────
    console.print("\n[bold]3. Dark edges[/bold] (relationships missing from CMDB)...")
    t0 = time.time()
    dark_edge_ids, dark_edge_manifest = _apply_dark_edges(relationships_df, deg.dark_edge_rate, tenant_id, rng)
    all_manifest_rows.extend(dark_edge_manifest)
    console.print(
        f"  [green]✓[/green] {len(dark_edge_ids):,} relationships marked as dark edges "
        f"({100.0 * len(dark_edge_ids) / max(1, gt_rel_count):.1f}%) "
        f"in {time.time() - t0:.1f}s"
    )

    # ── 4. Phantom edges ──────────────────────────────────────
    console.print("\n[bold]4. Phantom edges[/bold] (fabricated relationships in CMDB)...")
    t0 = time.time()
    phantom_rels, phantom_edge_manifest = _apply_phantom_edges(
        relationships_df, entities_df, deg.phantom_edge_rate, tenant_id, rng
    )
    all_manifest_rows.extend(phantom_edge_manifest)
    console.print(
        f"  [green]✓[/green] {len(phantom_rels):,} phantom relationships generated in {time.time() - t0:.1f}s"
    )

    # ── 5. Dark attributes ────────────────────────────────────
    console.print("\n[bold]5. Dark attributes[/bold] (stale/wrong attribute values in CMDB)...")
    t0 = time.time()
    attribute_overrides, dark_attr_manifest = _apply_dark_attributes(
        entities_df, deg.dark_attribute_rate, tenant_id, rng
    )
    all_manifest_rows.extend(dark_attr_manifest)
    # Count total corrupted attribute instances
    total_attr_corruptions = sum(len(v) for v in attribute_overrides.values())
    console.print(
        f"  [green]✓[/green] {len(attribute_overrides):,} entities with corrupted attributes "
        f"({total_attr_corruptions:,} total attribute corruptions) "
        f"in {time.time() - t0:.1f}s"
    )

    # ── 6. Identity mutations ─────────────────────────────────
    console.print("\n[bold]6. Identity mutations[/bold] (corrupted external_id values)...")
    t0 = time.time()
    # Already-corrupted = dark nodes + dark-attribute targets
    already_corrupted = dark_node_ids | set(attribute_overrides.keys())
    identity_mutations, id_mutation_manifest = _apply_identity_mutations(
        entities_df, deg.identity_mutation_rate, tenant_id, rng, already_corrupted
    )
    all_manifest_rows.extend(id_mutation_manifest)
    console.print(
        f"  [green]✓[/green] {len(identity_mutations):,} entities with identity mutations in {time.time() - t0:.1f}s"
    )

    # ── Build CMDB entity DataFrame ───────────────────────────
    console.print("\n[bold]Building degraded CMDB entities...[/bold]")
    t0 = time.time()
    cmdb_entities_df = _build_cmdb_entities(
        entities_df,
        dark_node_ids,
        phantom_entities,
        attribute_overrides,
        identity_mutations,
    )
    console.print(
        f"  [green]✓[/green] CMDB entities: {cmdb_entities_df.height:,} rows "
        f"(ground truth: {gt_entity_count:,}, "
        f"removed: {len(dark_node_ids):,}, "
        f"added: {len(phantom_entities):,}) "
        f"in {time.time() - t0:.1f}s"
    )

    # Free ground truth entities
    del entities_df
    gc.collect()

    # ── Build CMDB relationship DataFrame ─────────────────────
    console.print("\n[bold]Building degraded CMDB relationships...[/bold]")
    t0 = time.time()
    cmdb_rels_df = _build_cmdb_relationships(
        relationships_df,
        dark_edge_ids,
        phantom_rels,
        dark_node_ids,
    )
    # Count cascade-removed edges (edges to/from dark nodes that were not
    # explicitly marked as dark edges)
    cascade_removed = gt_rel_count - len(dark_edge_ids) - (cmdb_rels_df.height - len(phantom_rels))
    console.print(
        f"  [green]✓[/green] CMDB relationships: {cmdb_rels_df.height:,} rows "
        f"(ground truth: {gt_rel_count:,}, "
        f"dark edges: {len(dark_edge_ids):,}, "
        f"cascade-removed: {cascade_removed:,}, "
        f"phantom added: {len(phantom_rels):,}) "
        f"in {time.time() - t0:.1f}s"
    )

    # Free ground truth relationships
    del relationships_df
    gc.collect()

    # ── Write outputs ─────────────────────────────────────────

    # 1. CMDB entities
    console.print("\n[bold]Writing cmdb_declared_entities.parquet...[/bold]")
    t0 = time.time()
    entities_path = config.paths.output_dir / "cmdb_declared_entities.parquet"
    cmdb_entities_df.write_parquet(entities_path, compression="zstd", compression_level=3)
    entities_size_mb = entities_path.stat().st_size / (1024 * 1024)
    console.print(
        f"  [green]✓[/green] {cmdb_entities_df.height:,} rows, "
        f"{cmdb_entities_df.width} columns, {entities_size_mb:.1f} MB "
        f"in {time.time() - t0:.1f}s"
    )

    del cmdb_entities_df
    gc.collect()

    # 2. CMDB relationships
    console.print("\n[bold]Writing cmdb_declared_relationships.parquet...[/bold]")
    t0 = time.time()
    rels_path = config.paths.output_dir / "cmdb_declared_relationships.parquet"
    cmdb_rels_df.write_parquet(rels_path, compression="zstd", compression_level=3)
    rels_size_mb = rels_path.stat().st_size / (1024 * 1024)
    console.print(
        f"  [green]✓[/green] {cmdb_rels_df.height:,} rows, "
        f"{cmdb_rels_df.width} columns, {rels_size_mb:.1f} MB "
        f"in {time.time() - t0:.1f}s"
    )

    del cmdb_rels_df
    gc.collect()

    # 3. Divergence manifest
    console.print("\n[bold]Writing divergence_manifest.parquet...[/bold]")
    t0 = time.time()
    manifest_path = config.paths.output_dir / "divergence_manifest.parquet"
    manifest_rows_written, manifest_size_mb = _write_manifest(all_manifest_rows, manifest_path)
    console.print(
        f"  [green]✓[/green] {manifest_rows_written:,} rows, "
        f"{len(DIVERGENCE_MANIFEST_SCHEMA)} columns, {manifest_size_mb:.1f} MB "
        f"in {time.time() - t0:.1f}s"
    )

    # ── Summary tables ────────────────────────────────────────
    total_elapsed = time.time() - step_start
    console.print()

    # Divergence type breakdown
    div_table = Table(
        title="Step 08: CMDB Degradation — Divergence Summary",
        show_header=True,
    )
    div_table.add_column("Divergence Type", style="bold", width=22)
    div_table.add_column("Count", justify="right", width=12)
    div_table.add_column("Rate", justify="right", width=10)
    div_table.add_column("Scope", width=16)

    div_counts: dict[str, int] = {}
    for row in all_manifest_rows:
        dt = row["divergence_type"]
        div_counts[dt] = div_counts.get(dt, 0) + 1

    div_info = [
        (DIV_DARK_NODE, deg.dark_node_rate, "entities"),
        (DIV_PHANTOM_NODE, deg.phantom_node_rate, "entities"),
        (DIV_DARK_EDGE, deg.dark_edge_rate, "relationships"),
        (DIV_PHANTOM_EDGE, deg.phantom_edge_rate, "relationships"),
        (DIV_DARK_ATTRIBUTE, deg.dark_attribute_rate, "entities"),
        (DIV_IDENTITY_MUTATION, deg.identity_mutation_rate, "entities"),
    ]

    total_divs = 0
    for div_type, rate, scope in div_info:
        count = div_counts.get(div_type, 0)
        total_divs += count
        div_table.add_row(div_type, f"{count:,}", f"{rate:.1%}", scope)

    div_table.add_section()
    div_table.add_row("[bold]Total[/bold]", f"[bold]{total_divs:,}[/bold]", "", "")
    console.print(div_table)

    # Output file summary
    console.print()
    output_table = Table(
        title="Output Files",
        show_header=True,
    )
    output_table.add_column("File", style="bold", width=42)
    output_table.add_column("Rows", justify="right", width=14)
    output_table.add_column("Size", justify="right", width=10)

    output_table.add_row("cmdb_declared_entities.parquet", f"—", f"{entities_size_mb:.1f} MB")
    output_table.add_row("cmdb_declared_relationships.parquet", f"—", f"{rels_size_mb:.1f} MB")
    output_table.add_row("divergence_manifest.parquet", f"{manifest_rows_written:,}", f"{manifest_size_mb:.1f} MB")
    total_size = entities_size_mb + rels_size_mb + manifest_size_mb
    output_table.add_section()
    output_table.add_row("[bold]Total[/bold]", "", f"[bold]{total_size:.1f} MB[/bold]")
    console.print(output_table)

    # Cascade impact analysis
    console.print()
    impact_table = Table(
        title="Cascade Impact Analysis",
        show_header=True,
    )
    impact_table.add_column("Metric", style="bold", width=42)
    impact_table.add_column("Value", justify="right", width=14)
    impact_table.add_row("Ground truth entities", f"{gt_entity_count:,}")
    impact_table.add_row("Dark nodes removed", f"{len(dark_node_ids):,}")
    impact_table.add_row("Phantom nodes added", f"{len(phantom_entities):,}")
    impact_table.add_row("Dark edges removed", f"{len(dark_edge_ids):,}")
    impact_table.add_row("Cascade-removed edges (dark node endpoints)", f"{cascade_removed:,}")
    impact_table.add_row("Phantom edges added", f"{len(phantom_rels):,}")
    impact_table.add_row("Entities with attribute corruption", f"{len(attribute_overrides):,}")
    impact_table.add_row("Total attribute corruptions", f"{total_attr_corruptions:,}")
    impact_table.add_row("Entities with identity mutation", f"{len(identity_mutations):,}")
    impact_table.add_row("Divergence manifest entries", f"{total_divs:,}")
    console.print(impact_table)

    time_str = f"{total_elapsed:.1f}s" if total_elapsed < 60 else f"{total_elapsed / 60:.1f}m"
    console.print(
        f"\n[bold green]✓ Step 08 complete.[/bold green] "
        f"Applied {total_divs:,} divergences across 6 types, "
        f"wrote 3 files ({total_size:.1f} MB) in {time_str}"
    )
    console.print(
        "[dim]The divergence_manifest.parquet is the scoring key for "
        "Dark Graph reconciliation accuracy — every dark node, phantom node, "
        "dark edge, phantom edge, attribute corruption, and identity mutation "
        "is labelled for ML training and evaluation.[/dim]"
    )
