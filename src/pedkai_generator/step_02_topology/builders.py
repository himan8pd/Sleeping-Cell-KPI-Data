"""
Shared topology builder utilities.

Provides helper functions for creating entities and relationships with consistent
structure across all domain generators. Every entity and relationship in the
topology graph flows through these builders to ensure schema compliance with
the ground_truth_entities and ground_truth_relationships contracts.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Entity & Relationship row builders
# ---------------------------------------------------------------------------


def make_entity(
    *,
    tenant_id: str,
    entity_type: str,
    name: str,
    domain: str,
    external_id: str | None = None,
    geo_lat: float | None = None,
    geo_lon: float | None = None,
    site_id: str | None = None,
    site_type: str | None = None,
    deployment_profile: str | None = None,
    province: str | None = None,
    timezone: str | None = None,
    vendor: str | None = None,
    rat_type: str | None = None,
    band: str | None = None,
    bandwidth_mhz: float | None = None,
    max_tx_power_dbm: float | None = None,
    max_prbs: int | None = None,
    frequency_mhz: float | None = None,
    sector_id: int | None = None,
    azimuth_deg: float | None = None,
    electrical_tilt_deg: float | None = None,
    antenna_height_m: float | None = None,
    inter_site_distance_m: float | None = None,
    revenue_weight: float | None = None,
    sla_tier: str | None = None,
    is_nsa_anchor: bool | None = None,
    nsa_anchor_cell_id: str | None = None,
    parent_entity_id: str | None = None,
    properties_json: str | None = None,
    entity_id: str | None = None,
) -> dict[str, Any]:
    """
    Build a single entity row dict conforming to ground_truth_entities contract.

    If entity_id is not provided, a new UUID v4 is generated.
    """
    return {
        "entity_id": entity_id or str(uuid.uuid4()),
        "tenant_id": tenant_id,
        "entity_type": entity_type,
        "name": name,
        "external_id": external_id,
        "domain": domain,
        "geo_lat": geo_lat,
        "geo_lon": geo_lon,
        "site_id": site_id,
        "site_type": site_type,
        "deployment_profile": deployment_profile,
        "province": province,
        "timezone": timezone,
        "vendor": vendor,
        "rat_type": rat_type,
        "band": band,
        "bandwidth_mhz": bandwidth_mhz,
        "max_tx_power_dbm": max_tx_power_dbm,
        "max_prbs": max_prbs,
        "frequency_mhz": frequency_mhz,
        "sector_id": sector_id,
        "azimuth_deg": azimuth_deg,
        "electrical_tilt_deg": electrical_tilt_deg,
        "antenna_height_m": antenna_height_m,
        "inter_site_distance_m": inter_site_distance_m,
        "revenue_weight": revenue_weight,
        "sla_tier": sla_tier,
        "is_nsa_anchor": is_nsa_anchor,
        "nsa_anchor_cell_id": nsa_anchor_cell_id,
        "parent_entity_id": parent_entity_id,
        "properties_json": properties_json,
    }


def make_relationship(
    *,
    tenant_id: str,
    from_entity_id: str,
    from_entity_type: str,
    relationship_type: str,
    to_entity_id: str,
    to_entity_type: str,
    domain: str,
    properties_json: str | None = None,
    relationship_id: str | None = None,
) -> dict[str, Any]:
    """
    Build a single relationship row dict conforming to ground_truth_relationships contract.

    If relationship_id is not provided, a new UUID v4 is generated.
    """
    return {
        "relationship_id": relationship_id or str(uuid.uuid4()),
        "tenant_id": tenant_id,
        "from_entity_id": from_entity_id,
        "from_entity_type": from_entity_type,
        "relationship_type": relationship_type,
        "to_entity_id": to_entity_id,
        "to_entity_type": to_entity_type,
        "domain": domain,
        "properties_json": properties_json,
    }


def make_neighbour_relation(
    *,
    tenant_id: str,
    from_cell_id: str,
    from_cell_rat: str,
    from_cell_band: str,
    to_cell_id: str,
    to_cell_rat: str,
    to_cell_band: str,
    neighbour_type: str,
    is_intra_site: bool,
    distance_m: float | None = None,
    handover_attempts: float = 0.0,
    handover_success_rate: float = 98.0,
    cio_offset_db: float = 0.0,
    no_remove_flag: bool = False,
    relation_id: str | None = None,
) -> dict[str, Any]:
    """
    Build a single neighbour relation row dict conforming to
    neighbour_relations contract.
    """
    return {
        "relation_id": relation_id or str(uuid.uuid4()),
        "tenant_id": tenant_id,
        "from_cell_id": from_cell_id,
        "from_cell_rat": from_cell_rat,
        "from_cell_band": from_cell_band,
        "to_cell_id": to_cell_id,
        "to_cell_rat": to_cell_rat,
        "to_cell_band": to_cell_band,
        "neighbour_type": neighbour_type,
        "is_intra_site": is_intra_site,
        "distance_m": distance_m,
        "handover_attempts": handover_attempts,
        "handover_success_rate": handover_success_rate,
        "cio_offset_db": cio_offset_db,
        "no_remove_flag": no_remove_flag,
    }


# ---------------------------------------------------------------------------
# Batch ID generation
# ---------------------------------------------------------------------------


def generate_uuids(n: int) -> list[str]:
    """Generate n UUID v4 strings."""
    return [str(uuid.uuid4()) for _ in range(n)]


# ---------------------------------------------------------------------------
# Vendor external ID patterns
# ---------------------------------------------------------------------------


def ericsson_external_id(entity_type: str, region: str, entity_id: str) -> str:
    """
    Generate an Ericsson ENM-style MO path external ID.

    Pattern: SubNetwork=<region>,MeContext=<type>_<id_prefix>,ManagedElement=1,...
    """
    short_id = entity_id[:8]
    region_code = region[:6] if region else "REG"
    type_code = entity_type[:8]
    return f"SubNetwork=ERBS_{region_code},MeContext={type_code}_{short_id},ManagedElement=1"


def nokia_external_id(entity_type: str, region: str, entity_id: str) -> str:
    """
    Generate a Nokia NetAct-style external ID.

    Pattern: PLMN-PLMN/<region_code>/<type>-<id_prefix>
    """
    short_id = entity_id[:8]
    region_code = region[:4] if region else "REG"
    type_code = entity_type[:8]
    return f"PLMN-PLMN/{region_code}/{type_code}-{short_id}"


def vendor_external_id(vendor: str, entity_type: str, region: str, entity_id: str) -> str:
    """Generate vendor-appropriate external ID based on vendor assignment."""
    if vendor == "ericsson":
        return ericsson_external_id(entity_type, region, entity_id)
    else:
        return nokia_external_id(entity_type, region, entity_id)


# ---------------------------------------------------------------------------
# Name generation helpers
# ---------------------------------------------------------------------------


def entity_name(entity_type: str, index: int, region: str | None = None, suffix: str = "") -> str:
    """Generate a human-readable entity name."""
    region_part = f"-{region[:4].upper()}" if region else ""
    suffix_part = f"-{suffix}" if suffix else ""
    return f"{entity_type}{region_part}-{index:06d}{suffix_part}"


# ---------------------------------------------------------------------------
# Geographic helpers
# ---------------------------------------------------------------------------


def offset_lat_lon(
    rng: np.random.Generator,
    base_lat: float,
    base_lon: float,
    spread_km: float = 0.5,
) -> tuple[float, float]:
    """
    Generate a lat/lon offset from a base point.
    spread_km controls the standard deviation in kilometres.
    ~0.009 degrees ≈ 1 km at equator.
    """
    spread_deg = spread_km * 0.009
    lat = base_lat + float(rng.normal(0, spread_deg))
    lon = base_lon + float(rng.normal(0, spread_deg))
    # Clip to Indonesian bounds
    lat = max(-11.0, min(6.0, lat))
    lon = max(94.0, min(141.0, lon))
    return round(lat, 6), round(lon, 6)


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two points on Earth in metres.
    Uses the Haversine formula.
    """
    R = 6_371_000  # Earth radius in metres
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return float(R * c)


# ---------------------------------------------------------------------------
# JSON property helpers
# ---------------------------------------------------------------------------


def props_json(**kwargs: Any) -> str | None:
    """Convert keyword arguments to a JSON string for the properties column."""
    import json

    filtered = {k: v for k, v in kwargs.items() if v is not None}
    if not filtered:
        return None
    return json.dumps(filtered, default=str)


# ---------------------------------------------------------------------------
# Accumulator: collects entities and relationships across domain builders
# ---------------------------------------------------------------------------


@dataclass
class TopologyAccumulator:
    """
    Collects entities and relationships from all domain builders.

    Each domain builder appends to the shared lists. After all domains are
    built, the accumulator provides the complete topology for writing to
    Parquet files.

    Also maintains lookup indices for cross-domain referencing (e.g.,
    transport builders need to know which sites exist).
    """

    entities: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)
    neighbour_relations: list[dict[str, Any]] = field(default_factory=list)

    # Indices for cross-domain lookups
    _entities_by_type: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _entities_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    _entities_by_site: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def add_entity(self, entity: dict[str, Any]) -> None:
        """Add an entity and update indices."""
        self.entities.append(entity)

        etype = entity["entity_type"]
        if etype not in self._entities_by_type:
            self._entities_by_type[etype] = []
        self._entities_by_type[etype].append(entity)

        eid = entity["entity_id"]
        self._entities_by_id[eid] = entity

        site_id = entity.get("site_id")
        if site_id:
            if site_id not in self._entities_by_site:
                self._entities_by_site[site_id] = []
            self._entities_by_site[site_id].append(entity)

    def add_entities(self, entities: list[dict[str, Any]]) -> None:
        """Add multiple entities."""
        for e in entities:
            self.add_entity(e)

    def add_relationship(self, rel: dict[str, Any]) -> None:
        """Add a relationship."""
        self.relationships.append(rel)

    def add_relationships(self, rels: list[dict[str, Any]]) -> None:
        """Add multiple relationships."""
        self.relationships.extend(rels)

    def add_neighbour_relation(self, nr: dict[str, Any]) -> None:
        """Add a neighbour relation."""
        self.neighbour_relations.append(nr)

    def add_neighbour_relations(self, nrs: list[dict[str, Any]]) -> None:
        """Add multiple neighbour relations."""
        self.neighbour_relations.extend(nrs)

    def get_entities_by_type(self, entity_type: str) -> list[dict[str, Any]]:
        """Get all entities of a given type."""
        return self._entities_by_type.get(entity_type, [])

    def get_entity_by_id(self, entity_id: str) -> dict[str, Any] | None:
        """Get a single entity by its ID."""
        return self._entities_by_id.get(entity_id)

    def get_entities_at_site(self, site_id: str) -> list[dict[str, Any]]:
        """Get all entities associated with a site."""
        return self._entities_by_site.get(site_id, [])

    def get_site_ids(self) -> list[str]:
        """Get all site entity IDs."""
        return [e["entity_id"] for e in self._entities_by_type.get("SITE", [])]

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def relationship_count(self) -> int:
        return len(self.relationships)

    @property
    def neighbour_relation_count(self) -> int:
        return len(self.neighbour_relations)

    def entity_count_by_domain(self) -> dict[str, int]:
        """Count entities grouped by domain."""
        counts: dict[str, int] = {}
        for e in self.entities:
            d = e.get("domain", "unknown")
            counts[d] = counts.get(d, 0) + 1
        return counts

    def relationship_count_by_domain(self) -> dict[str, int]:
        """Count relationships grouped by domain."""
        counts: dict[str, int] = {}
        for r in self.relationships:
            d = r.get("domain", "unknown")
            counts[d] = counts.get(d, 0) + 1
        return counts

    def entity_count_by_type(self) -> dict[str, int]:
        """Count entities grouped by entity_type."""
        return {k: len(v) for k, v in self._entities_by_type.items()}
