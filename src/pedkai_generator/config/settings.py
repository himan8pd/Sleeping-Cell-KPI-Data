"""
Global configuration system for the Pedkai Synthetic Data Generator.

Implements deterministic seed management: Seed(Step_N) = F(Global_Seed, Step_ID)
All scale parameters from the thread summary are defined here as defaults.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DeploymentProfile(str, Enum):
    DENSE_URBAN = "dense_urban"
    URBAN = "urban"
    SUBURBAN = "suburban"
    RURAL = "rural"
    DEEP_RURAL = "deep_rural"
    INDOOR = "indoor"


class SiteType(str, Enum):
    GREENFIELD = "greenfield"
    ROOFTOP = "rooftop"
    STREETWORKS = "streetworks"
    IN_BUILDING = "in_building"
    UNSPECIFIED = "unspecified"


class RAT(str, Enum):
    LTE = "LTE"
    NR_NSA = "NR_NSA"  # EN-DC: LTE anchor + NR SCG
    NR_SA = "NR_SA"  # Standalone NR with 5GC


class Vendor(str, Enum):
    ERICSSON = "ericsson"
    NOKIA = "nokia"


class SLATier(str, Enum):
    GOLD = "GOLD"
    SILVER = "SILVER"
    BRONZE = "BRONZE"


# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------


def derive_seed(global_seed: int, step_id: str) -> int:
    """
    Deterministic seed derivation: Seed(Step_N) = F(Global_Seed, Step_ID).

    Uses HMAC-like hashing so that each pipeline step gets a unique but
    reproducible seed from the single global seed.
    """
    raw = hashlib.sha256(struct.pack(">Q", global_seed) + step_id.encode("utf-8")).digest()
    # Take first 8 bytes as unsigned 64-bit int, then mask to 32-bit for numpy compat
    return struct.unpack(">Q", raw[:8])[0] & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# Scale parameter dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SiteScaleConfig:
    """UK-scale operator site counts by type."""

    greenfield: int = 11_000
    rooftop: int = 4_000
    streetworks: int = 5_500
    in_building: int = 500
    unspecified: int = 100

    @property
    def total(self) -> int:
        return self.greenfield + self.rooftop + self.streetworks + self.in_building + self.unspecified

    def sectors_per_site(self, site_type: SiteType) -> int:
        mapping = {
            SiteType.GREENFIELD: 3,
            SiteType.ROOFTOP: 3,
            SiteType.STREETWORKS: 1,
            SiteType.IN_BUILDING: 2,
            SiteType.UNSPECIFIED: 2,
        }
        return mapping[site_type]


@dataclass
class RATSplitConfig:
    """Approximate cell counts per RAT."""

    lte_only: int = 32_000
    lte_plus_nsa: int = 13_000
    nr_sa: int = 6_700

    @property
    def total_physical_cells(self) -> int:
        return self.lte_only + self.lte_plus_nsa + self.nr_sa

    @property
    def nsa_nr_scg_legs(self) -> int:
        """Each NSA cell produces a second KPI stream for the NR SCG leg."""
        return self.lte_plus_nsa

    @property
    def total_logical_cell_layers(self) -> int:
        return self.total_physical_cells + self.nsa_nr_scg_legs


@dataclass
class EntityScaleConfig:
    """Entity counts per domain (UK-scale converged operator)."""

    mobile_ran_entities: int = 170_000
    mobile_ran_relationships: int = 400_000
    fixed_access_entities: int = 120_000
    fixed_access_relationships: int = 350_000
    transport_entities: int = 50_000
    transport_relationships: int = 150_000
    core_entities: int = 200
    core_relationships: int = 2_000
    logical_service_entities: int = 10_000
    logical_service_relationships: int = 50_000
    power_environment_entities: int = 40_000
    power_environment_relationships: int = 60_000
    customer_entities: int = 1_100_000
    customer_relationships: int = 1_200_000

    @property
    def total_entities(self) -> int:
        return (
            self.mobile_ran_entities
            + self.fixed_access_entities
            + self.transport_entities
            + self.core_entities
            + self.logical_service_entities
            + self.power_environment_entities
            + self.customer_entities
        )

    @property
    def total_relationships(self) -> int:
        return (
            self.mobile_ran_relationships
            + self.fixed_access_relationships
            + self.transport_relationships
            + self.core_relationships
            + self.logical_service_relationships
            + self.power_environment_relationships
            + self.customer_relationships
        )


@dataclass
class SimulationConfig:
    """Time-series simulation parameters."""

    simulation_days: int = 30
    reporting_interval_hours: int = 1

    @property
    def total_hours(self) -> int:
        return self.simulation_days * 24

    @property
    def total_intervals(self) -> int:
        return self.total_hours // self.reporting_interval_hours


@dataclass
class UserScaleConfig:
    """Subscriber / customer scale."""

    total_subscribers: int = 1_000_000
    residential_fraction: float = 0.95
    enterprise_fraction: float = 0.05

    @property
    def residential_count(self) -> int:
        return int(self.total_subscribers * self.residential_fraction)

    @property
    def enterprise_count(self) -> int:
        return self.total_subscribers - self.residential_count


@dataclass
class CMDBDegradationConfig:
    """Rates for each Dark Graph divergence type."""

    dark_node_rate: float = 0.065  # 5-8% → use 6.5% midpoint
    phantom_node_rate: float = 0.03  # 3%
    dark_edge_rate: float = 0.10  # 10%
    phantom_edge_rate: float = 0.05  # 5%
    dark_attribute_rate: float = 0.15  # 15%
    identity_mutation_rate: float = 0.02  # 2%


@dataclass
class ScenarioInjectionConfig:
    """Scenario injection rates and parameters."""

    sleeping_cell_rate: float = 0.02  # ~2% of cells → ~1,034 cases
    congestion_rate: float = 0.05  # 5% of cells experience congestion episodes
    coverage_hole_rate: float = 0.01  # 1% spatial clusters
    hardware_fault_rate: float = 0.005  # 0.5% of cells
    interference_rate: float = 0.03  # 3% of cells
    transport_failure_rate: float = 0.002  # 0.2% of transport links
    power_failure_rate: float = 0.001  # 0.1% of sites
    fibre_cut_rate: float = 0.0005  # 0.05% of fibre links


# ---------------------------------------------------------------------------
# LTE Band definitions
# ---------------------------------------------------------------------------


@dataclass
class BandConfig:
    """Radio band characteristics."""

    name: str
    frequency_mhz: float
    bandwidth_mhz: float
    max_prbs: int
    max_tx_power_dbm: float
    duplex: str  # "FDD" or "TDD"
    rat: RAT = RAT.LTE

    @property
    def subcarrier_spacing_khz(self) -> float:
        if self.rat == RAT.LTE:
            return 15.0
        # NR: depends on numerology, default to μ=1 (30 kHz) for sub-6
        if self.frequency_mhz < 6000:
            return 30.0
        return 120.0  # mmWave


# Pre-defined band library
LTE_BANDS: dict[str, BandConfig] = {
    "L900": BandConfig("L900", 900, 10, 50, 46.0, "FDD", RAT.LTE),
    "L1800": BandConfig("L1800", 1800, 20, 100, 46.0, "FDD", RAT.LTE),
    "L2100": BandConfig("L2100", 2100, 15, 75, 43.0, "FDD", RAT.LTE),
    "L2300": BandConfig("L2300", 2300, 20, 100, 38.0, "TDD", RAT.LTE),
}

NR_NSA_BANDS: dict[str, BandConfig] = {
    "n1": BandConfig("n1", 2100, 20, 106, 43.0, "FDD", RAT.NR_NSA),
    "n3": BandConfig("n3", 1800, 20, 106, 43.0, "FDD", RAT.NR_NSA),
    "n28": BandConfig("n28", 700, 10, 52, 46.0, "FDD", RAT.NR_NSA),
    "n78": BandConfig("n78", 3500, 100, 273, 38.0, "TDD", RAT.NR_NSA),
}

NR_SA_BANDS: dict[str, BandConfig] = {
    "n77": BandConfig("n77", 3700, 100, 273, 38.0, "TDD", RAT.NR_SA),
    "n78": BandConfig("n78", 3500, 100, 273, 38.0, "TDD", RAT.NR_SA),
    "n257": BandConfig("n257", 28000, 400, 264, 35.0, "TDD", RAT.NR_SA),
    "n258": BandConfig("n258", 26000, 400, 264, 35.0, "TDD", RAT.NR_SA),
}

ALL_BANDS: dict[str, BandConfig] = {**LTE_BANDS, **NR_NSA_BANDS, **NR_SA_BANDS}


# ---------------------------------------------------------------------------
# Indonesian geography
# ---------------------------------------------------------------------------


@dataclass
class ProvinceConfig:
    """Indonesian province with timezone and population density class."""

    name: str
    timezone: str  # WIB, WITA, WIT
    lat_center: float
    lon_center: float
    lat_spread: float  # approx bounding box half-height in degrees
    lon_spread: float  # approx bounding box half-width in degrees
    density_class: str  # "hyper_dense", "dense", "moderate", "sparse", "very_sparse"


# 38 Indonesian provinces
INDONESIA_PROVINCES: list[ProvinceConfig] = [
    # -- WIB (Western Indonesia Time, UTC+7) --
    ProvinceConfig("Aceh", "WIB", 4.70, 96.75, 1.5, 1.5, "sparse"),
    ProvinceConfig("Sumatera Utara", "WIB", 2.50, 99.00, 2.0, 1.5, "moderate"),
    ProvinceConfig("Sumatera Barat", "WIB", -0.95, 100.40, 1.0, 0.8, "moderate"),
    ProvinceConfig("Riau", "WIB", 0.50, 102.15, 1.5, 1.5, "moderate"),
    ProvinceConfig("Kepulauan Riau", "WIB", 1.00, 104.50, 1.0, 1.5, "sparse"),
    ProvinceConfig("Jambi", "WIB", -1.60, 103.60, 1.0, 1.0, "sparse"),
    ProvinceConfig("Sumatera Selatan", "WIB", -3.30, 104.75, 1.5, 1.5, "moderate"),
    ProvinceConfig("Bengkulu", "WIB", -3.80, 102.30, 1.0, 0.5, "sparse"),
    ProvinceConfig("Lampung", "WIB", -5.00, 105.00, 1.0, 1.0, "moderate"),
    ProvinceConfig("Bangka Belitung", "WIB", -2.70, 106.60, 0.5, 1.0, "sparse"),
    ProvinceConfig("DKI Jakarta", "WIB", -6.20, 106.85, 0.15, 0.15, "hyper_dense"),
    ProvinceConfig("Jawa Barat", "WIB", -6.90, 107.60, 0.8, 1.2, "dense"),
    ProvinceConfig("Banten", "WIB", -6.40, 106.15, 0.5, 0.5, "dense"),
    ProvinceConfig("Jawa Tengah", "WIB", -7.15, 110.00, 0.8, 1.5, "dense"),
    ProvinceConfig("DI Yogyakarta", "WIB", -7.80, 110.35, 0.2, 0.3, "dense"),
    ProvinceConfig("Jawa Timur", "WIB", -7.55, 112.75, 1.0, 1.5, "dense"),
    # -- WITA (Central Indonesia Time, UTC+8) --
    ProvinceConfig("Bali", "WITA", -8.40, 115.20, 0.3, 0.5, "dense"),
    ProvinceConfig("Nusa Tenggara Barat", "WITA", -8.65, 117.20, 0.5, 1.0, "moderate"),
    ProvinceConfig("Nusa Tenggara Timur", "WITA", -9.50, 121.00, 1.0, 2.0, "sparse"),
    ProvinceConfig("Kalimantan Barat", "WITA", 0.00, 109.50, 2.0, 1.5, "sparse"),
    ProvinceConfig("Kalimantan Tengah", "WITA", -1.50, 114.00, 2.0, 2.0, "very_sparse"),
    ProvinceConfig("Kalimantan Selatan", "WITA", -3.30, 115.50, 1.0, 1.0, "moderate"),
    ProvinceConfig("Kalimantan Timur", "WITA", 0.50, 116.50, 3.0, 1.5, "sparse"),
    ProvinceConfig("Kalimantan Utara", "WITA", 3.00, 116.50, 1.5, 1.0, "very_sparse"),
    ProvinceConfig("Sulawesi Utara", "WITA", 1.20, 124.80, 0.5, 0.8, "moderate"),
    ProvinceConfig("Gorontalo", "WITA", 0.55, 122.45, 0.3, 0.5, "sparse"),
    ProvinceConfig("Sulawesi Tengah", "WITA", -1.40, 121.50, 2.0, 2.0, "sparse"),
    ProvinceConfig("Sulawesi Selatan", "WITA", -3.70, 120.00, 1.5, 1.5, "moderate"),
    ProvinceConfig("Sulawesi Barat", "WITA", -2.80, 119.00, 0.5, 0.5, "sparse"),
    ProvinceConfig("Sulawesi Tenggara", "WITA", -4.00, 122.50, 1.5, 1.5, "sparse"),
    # -- WIT (Eastern Indonesia Time, UTC+9) --
    ProvinceConfig("Maluku", "WIT", -3.50, 128.50, 2.0, 2.0, "very_sparse"),
    ProvinceConfig("Maluku Utara", "WIT", 1.50, 128.00, 1.5, 1.5, "very_sparse"),
    ProvinceConfig("Papua", "WIT", -4.00, 138.50, 3.0, 3.0, "very_sparse"),
    ProvinceConfig("Papua Barat", "WIT", -1.50, 133.50, 1.5, 2.0, "very_sparse"),
    ProvinceConfig("Papua Selatan", "WIT", -6.50, 139.00, 1.5, 1.5, "very_sparse"),
    ProvinceConfig("Papua Tengah", "WIT", -3.80, 137.00, 1.5, 1.5, "very_sparse"),
    ProvinceConfig("Papua Pegunungan", "WIT", -4.20, 139.50, 1.5, 1.5, "very_sparse"),
    ProvinceConfig("Papua Barat Daya", "WIT", -2.50, 131.50, 1.0, 1.5, "very_sparse"),
]


# ---------------------------------------------------------------------------
# Deployment profile → site type mapping
# ---------------------------------------------------------------------------

SITE_TYPE_DEPLOYMENT_PROFILES: dict[SiteType, list[DeploymentProfile]] = {
    SiteType.GREENFIELD: [
        DeploymentProfile.SUBURBAN,
        DeploymentProfile.RURAL,
        DeploymentProfile.DEEP_RURAL,
    ],
    SiteType.ROOFTOP: [
        DeploymentProfile.URBAN,
        DeploymentProfile.DENSE_URBAN,
    ],
    SiteType.STREETWORKS: [
        DeploymentProfile.DENSE_URBAN,
        DeploymentProfile.URBAN,
    ],
    SiteType.IN_BUILDING: [
        DeploymentProfile.INDOOR,
    ],
    SiteType.UNSPECIFIED: [
        DeploymentProfile.SUBURBAN,
        DeploymentProfile.URBAN,
    ],
}


# Province density → deployment profile weighting
# Used to distribute sites realistically across provinces
DENSITY_DEPLOYMENT_WEIGHTS: dict[str, dict[DeploymentProfile, float]] = {
    "hyper_dense": {
        DeploymentProfile.DENSE_URBAN: 0.55,
        DeploymentProfile.URBAN: 0.30,
        DeploymentProfile.SUBURBAN: 0.10,
        DeploymentProfile.INDOOR: 0.05,
    },
    "dense": {
        DeploymentProfile.DENSE_URBAN: 0.15,
        DeploymentProfile.URBAN: 0.40,
        DeploymentProfile.SUBURBAN: 0.35,
        DeploymentProfile.INDOOR: 0.10,
    },
    "moderate": {
        DeploymentProfile.URBAN: 0.20,
        DeploymentProfile.SUBURBAN: 0.50,
        DeploymentProfile.RURAL: 0.25,
        DeploymentProfile.INDOOR: 0.05,
    },
    "sparse": {
        DeploymentProfile.SUBURBAN: 0.20,
        DeploymentProfile.RURAL: 0.60,
        DeploymentProfile.DEEP_RURAL: 0.15,
        DeploymentProfile.INDOOR: 0.05,
    },
    "very_sparse": {
        DeploymentProfile.RURAL: 0.40,
        DeploymentProfile.DEEP_RURAL: 0.55,
        DeploymentProfile.SUBURBAN: 0.05,
    },
}


# ---------------------------------------------------------------------------
# Output path configuration
# ---------------------------------------------------------------------------


@dataclass
class OutputPathConfig:
    """Where generated data lands."""

    data_store_root: Path = field(default_factory=lambda: Path("/Volumes/Projects/Pedkai Data Store"))

    @property
    def output_dir(self) -> Path:
        return self.data_store_root / "output"

    @property
    def intermediate_dir(self) -> Path:
        return self.data_store_root / "intermediate"

    @property
    def validation_dir(self) -> Path:
        return self.data_store_root / "validation"

    def ensure_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)
        self.validation_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Master configuration
# ---------------------------------------------------------------------------


@dataclass
class GeneratorConfig:
    """
    Master configuration for the entire synthetic data generator.

    All parameters from the thread summary are captured here as defaults.
    Can be overridden via YAML config file or CLI arguments.
    """

    # Global seed — single source of determinism for the entire pipeline
    global_seed: int = 42_000_001

    # Tenant ID — every row in every table must carry this
    tenant_id: str = "pedkai_synthetic_01"

    # Scale parameters
    sites: SiteScaleConfig = field(default_factory=SiteScaleConfig)
    rat_split: RATSplitConfig = field(default_factory=RATSplitConfig)
    entities: EntityScaleConfig = field(default_factory=EntityScaleConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    users: UserScaleConfig = field(default_factory=UserScaleConfig)
    cmdb_degradation: CMDBDegradationConfig = field(default_factory=CMDBDegradationConfig)
    scenario_injection: ScenarioInjectionConfig = field(default_factory=ScenarioInjectionConfig)

    # Output paths
    paths: OutputPathConfig = field(default_factory=OutputPathConfig)

    # Vendor split (fraction of sites assigned to Ericsson vs Nokia)
    ericsson_fraction: float = 0.55
    nokia_fraction: float = 0.45

    def seed_for(self, step_id: str) -> int:
        """Derive a deterministic seed for a specific pipeline step."""
        return derive_seed(self.global_seed, step_id)

    def ensure_output_dirs(self) -> None:
        """Create all output directories."""
        self.paths.ensure_dirs()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "GeneratorConfig":
        """Load configuration from a YAML file, merging with defaults."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, d: dict[str, Any]) -> "GeneratorConfig":
        """Construct config from a flat/nested dict, using defaults for missing keys."""
        config = cls()

        if "global_seed" in d:
            config.global_seed = int(d["global_seed"])
        if "tenant_id" in d:
            config.tenant_id = str(d["tenant_id"])
        if "ericsson_fraction" in d:
            config.ericsson_fraction = float(d["ericsson_fraction"])
            config.nokia_fraction = 1.0 - config.ericsson_fraction

        # Sites
        if "sites" in d:
            s = d["sites"]
            config.sites = SiteScaleConfig(
                greenfield=s.get("greenfield", config.sites.greenfield),
                rooftop=s.get("rooftop", config.sites.rooftop),
                streetworks=s.get("streetworks", config.sites.streetworks),
                in_building=s.get("in_building", config.sites.in_building),
                unspecified=s.get("unspecified", config.sites.unspecified),
            )

        # RAT split
        if "rat_split" in d:
            r = d["rat_split"]
            config.rat_split = RATSplitConfig(
                lte_only=r.get("lte_only", config.rat_split.lte_only),
                lte_plus_nsa=r.get("lte_plus_nsa", config.rat_split.lte_plus_nsa),
                nr_sa=r.get("nr_sa", config.rat_split.nr_sa),
            )

        # Simulation
        if "simulation" in d:
            sim = d["simulation"]
            config.simulation = SimulationConfig(
                simulation_days=sim.get("simulation_days", config.simulation.simulation_days),
                reporting_interval_hours=sim.get(
                    "reporting_interval_hours",
                    config.simulation.reporting_interval_hours,
                ),
            )

        # Users
        if "users" in d:
            u = d["users"]
            config.users = UserScaleConfig(
                total_subscribers=u.get("total_subscribers", config.users.total_subscribers),
                residential_fraction=u.get("residential_fraction", config.users.residential_fraction),
                enterprise_fraction=u.get("enterprise_fraction", config.users.enterprise_fraction),
            )

        # CMDB degradation
        if "cmdb_degradation" in d:
            cd = d["cmdb_degradation"]
            config.cmdb_degradation = CMDBDegradationConfig(
                dark_node_rate=cd.get("dark_node_rate", config.cmdb_degradation.dark_node_rate),
                phantom_node_rate=cd.get("phantom_node_rate", config.cmdb_degradation.phantom_node_rate),
                dark_edge_rate=cd.get("dark_edge_rate", config.cmdb_degradation.dark_edge_rate),
                phantom_edge_rate=cd.get("phantom_edge_rate", config.cmdb_degradation.phantom_edge_rate),
                dark_attribute_rate=cd.get("dark_attribute_rate", config.cmdb_degradation.dark_attribute_rate),
                identity_mutation_rate=cd.get(
                    "identity_mutation_rate",
                    config.cmdb_degradation.identity_mutation_rate,
                ),
            )

        # Scenario injection
        if "scenario_injection" in d:
            si = d["scenario_injection"]
            config.scenario_injection = ScenarioInjectionConfig(
                sleeping_cell_rate=si.get("sleeping_cell_rate", config.scenario_injection.sleeping_cell_rate),
                congestion_rate=si.get("congestion_rate", config.scenario_injection.congestion_rate),
                coverage_hole_rate=si.get("coverage_hole_rate", config.scenario_injection.coverage_hole_rate),
                hardware_fault_rate=si.get("hardware_fault_rate", config.scenario_injection.hardware_fault_rate),
                interference_rate=si.get("interference_rate", config.scenario_injection.interference_rate),
                transport_failure_rate=si.get(
                    "transport_failure_rate",
                    config.scenario_injection.transport_failure_rate,
                ),
                power_failure_rate=si.get("power_failure_rate", config.scenario_injection.power_failure_rate),
                fibre_cut_rate=si.get("fibre_cut_rate", config.scenario_injection.fibre_cut_rate),
            )

        # Paths
        if "data_store_root" in d:
            config.paths = OutputPathConfig(data_store_root=Path(d["data_store_root"]))

        return config

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a dict (for logging / saving alongside output)."""
        return {
            "global_seed": self.global_seed,
            "tenant_id": self.tenant_id,
            "ericsson_fraction": self.ericsson_fraction,
            "nokia_fraction": self.nokia_fraction,
            "sites": {
                "greenfield": self.sites.greenfield,
                "rooftop": self.sites.rooftop,
                "streetworks": self.sites.streetworks,
                "in_building": self.sites.in_building,
                "unspecified": self.sites.unspecified,
                "total": self.sites.total,
            },
            "rat_split": {
                "lte_only": self.rat_split.lte_only,
                "lte_plus_nsa": self.rat_split.lte_plus_nsa,
                "nr_sa": self.rat_split.nr_sa,
                "total_physical_cells": self.rat_split.total_physical_cells,
                "total_logical_cell_layers": self.rat_split.total_logical_cell_layers,
            },
            "simulation": {
                "simulation_days": self.simulation.simulation_days,
                "reporting_interval_hours": self.simulation.reporting_interval_hours,
                "total_hours": self.simulation.total_hours,
                "total_intervals": self.simulation.total_intervals,
            },
            "users": {
                "total_subscribers": self.users.total_subscribers,
                "residential_count": self.users.residential_count,
                "enterprise_count": self.users.enterprise_count,
            },
            "entities": {
                "total_entities": self.entities.total_entities,
                "total_relationships": self.entities.total_relationships,
            },
            "cmdb_degradation": {
                "dark_node_rate": self.cmdb_degradation.dark_node_rate,
                "phantom_node_rate": self.cmdb_degradation.phantom_node_rate,
                "dark_edge_rate": self.cmdb_degradation.dark_edge_rate,
                "phantom_edge_rate": self.cmdb_degradation.phantom_edge_rate,
                "dark_attribute_rate": self.cmdb_degradation.dark_attribute_rate,
                "identity_mutation_rate": self.cmdb_degradation.identity_mutation_rate,
            },
            "scenario_injection": {
                "sleeping_cell_rate": self.scenario_injection.sleeping_cell_rate,
                "congestion_rate": self.scenario_injection.congestion_rate,
                "coverage_hole_rate": self.scenario_injection.coverage_hole_rate,
                "hardware_fault_rate": self.scenario_injection.hardware_fault_rate,
                "interference_rate": self.scenario_injection.interference_rate,
                "transport_failure_rate": self.scenario_injection.transport_failure_rate,
                "power_failure_rate": self.scenario_injection.power_failure_rate,
                "fibre_cut_rate": self.scenario_injection.fibre_cut_rate,
            },
            "data_store_root": str(self.paths.data_store_root),
        }

    def save_yaml(self, path: str | Path) -> None:
        """Save current config as YAML for reproducibility."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Convenience: default config singleton
# ---------------------------------------------------------------------------


def get_default_config() -> GeneratorConfig:
    """Return a GeneratorConfig with all thread-summary defaults."""
    return GeneratorConfig()
