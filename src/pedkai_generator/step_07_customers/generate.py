"""
Step 07: Customer & BSS Data Generation.

Generates ~1M synthetic subscribers with billing accounts, service plans,
SLA tiers, and site/cell associations for a UK-scale converged operator.

Customer profiles include:
  - Residential (~95%) and Enterprise (~5%) customers
  - Service plans: prepaid/postpaid for residential; managed connectivity
    for enterprise
  - SLA tiers: Gold / Silver / Bronze
  - Site-cell associations via FK to ground_truth_entities
  - Billing accounts with status, revenue, contract dates
  - Churn risk scores and proactive comms consent
  - Access type: mobile, fttp, fttc, enterprise_ethernet

Design notes:
  - Customer-to-cell/site mapping uses topology data from Phase 2.
  - Downstream Pedkai analytics (CX Impact Analysis) joins customers
    against scenario-affected cells at query time, so the FK from
    ``associated_site_id`` → ``ground_truth_entities.entity_id`` must
    be valid.
  - Phase 5 scenario injection does NOT need customer data — scenarios
    target network entities.
  - Fixed broadband customers get an ``access_entity_id`` FK pointing
    to their ONT or NTE entity from the topology.

Output:
  - output/customers_bss.parquet (~200 MB, ~1M rows, 18 columns)

Dependencies: Phase 2 (reads ground_truth_entities.parquet for site and
              access entity metadata)
"""

from __future__ import annotations

import gc
import time
import uuid
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

# Customer types
CUSTOMER_TYPE_RESIDENTIAL = "residential"
CUSTOMER_TYPE_ENTERPRISE = "enterprise"

# Access types
ACCESS_MOBILE = "mobile"
ACCESS_FTTP = "fttp"
ACCESS_FTTC = "fttc"
ACCESS_ENTERPRISE_ETHERNET = "enterprise_ethernet"

# Account statuses
STATUS_ACTIVE = "ACTIVE"
STATUS_SUSPENDED = "SUSPENDED"
STATUS_DELINQUENT = "DELINQUENT"

# SLA tiers
SLA_GOLD = "GOLD"
SLA_SILVER = "SILVER"
SLA_BRONZE = "BRONZE"

# ---------------------------------------------------------------------------
# PyArrow output schema — 18 columns (matches Phase 0 contract)
# ---------------------------------------------------------------------------

CUSTOMERS_SCHEMA = pa.schema(
    [
        pa.field("customer_id", pa.string(), nullable=False),
        pa.field("tenant_id", pa.string(), nullable=False),
        pa.field("external_id", pa.string(), nullable=False),
        pa.field("customer_type", pa.string(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("associated_site_id", pa.string(), nullable=False),
        pa.field("province", pa.string(), nullable=False),
        pa.field("service_plan_name", pa.string(), nullable=False),
        pa.field("service_plan_tier", pa.string(), nullable=False),
        pa.field("monthly_fee", pa.float64(), nullable=False),
        pa.field("sla_guarantee", pa.string(), nullable=True),
        pa.field("account_status", pa.string(), nullable=False),
        pa.field("avg_monthly_revenue", pa.float64(), nullable=False),
        pa.field("contract_end_date", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("churn_risk_score", pa.float64(), nullable=False),
        pa.field("consent_proactive_comms", pa.bool_(), nullable=False),
        pa.field("access_type", pa.string(), nullable=False),
        pa.field("access_entity_id", pa.string(), nullable=True),
    ]
)

# ---------------------------------------------------------------------------
# Service plan definitions
# ---------------------------------------------------------------------------

# Residential service plans
RESIDENTIAL_PLANS_MOBILE: list[dict[str, Any]] = [
    {
        "name": "SIM Only 5GB",
        "tier": SLA_BRONZE,
        "monthly_fee": 8.0,
        "sla": None,
        "access_type": ACCESS_MOBILE,
    },
    {
        "name": "SIM Only 20GB",
        "tier": SLA_BRONZE,
        "monthly_fee": 15.0,
        "sla": None,
        "access_type": ACCESS_MOBILE,
    },
    {
        "name": "SIM Only Unlimited",
        "tier": SLA_SILVER,
        "monthly_fee": 25.0,
        "sla": None,
        "access_type": ACCESS_MOBILE,
    },
    {
        "name": "Handset Plan Standard",
        "tier": SLA_BRONZE,
        "monthly_fee": 35.0,
        "sla": None,
        "access_type": ACCESS_MOBILE,
    },
    {
        "name": "Handset Plan Premium",
        "tier": SLA_SILVER,
        "monthly_fee": 55.0,
        "sla": None,
        "access_type": ACCESS_MOBILE,
    },
    {
        "name": "Handset Plan Max",
        "tier": SLA_GOLD,
        "monthly_fee": 75.0,
        "sla": "99.9% Network Availability",
        "access_type": ACCESS_MOBILE,
    },
]

RESIDENTIAL_PLANS_FTTP: list[dict[str, Any]] = [
    {
        "name": "Fibre 36",
        "tier": SLA_BRONZE,
        "monthly_fee": 24.99,
        "sla": None,
        "access_type": ACCESS_FTTP,
    },
    {
        "name": "Fibre 80",
        "tier": SLA_BRONZE,
        "monthly_fee": 29.99,
        "sla": None,
        "access_type": ACCESS_FTTP,
    },
    {
        "name": "Fibre 150",
        "tier": SLA_SILVER,
        "monthly_fee": 34.99,
        "sla": None,
        "access_type": ACCESS_FTTP,
    },
    {
        "name": "Full Fibre 300",
        "tier": SLA_SILVER,
        "monthly_fee": 39.99,
        "sla": "99.9% Availability",
        "access_type": ACCESS_FTTP,
    },
    {
        "name": "Full Fibre 900",
        "tier": SLA_GOLD,
        "monthly_fee": 54.99,
        "sla": "99.95% Availability",
        "access_type": ACCESS_FTTP,
    },
]

RESIDENTIAL_PLANS_FTTC: list[dict[str, Any]] = [
    {
        "name": "Broadband Essential",
        "tier": SLA_BRONZE,
        "monthly_fee": 19.99,
        "sla": None,
        "access_type": ACCESS_FTTC,
    },
    {
        "name": "Superfast 55",
        "tier": SLA_BRONZE,
        "monthly_fee": 24.99,
        "sla": None,
        "access_type": ACCESS_FTTC,
    },
    {
        "name": "Superfast 80",
        "tier": SLA_SILVER,
        "monthly_fee": 29.99,
        "sla": None,
        "access_type": ACCESS_FTTC,
    },
]

ENTERPRISE_PLANS: list[dict[str, Any]] = [
    {
        "name": "Business Mobile Standard",
        "tier": SLA_SILVER,
        "monthly_fee": 22.0,
        "sla": "99.9% Network Availability",
        "access_type": ACCESS_MOBILE,
    },
    {
        "name": "Business Mobile Premium",
        "tier": SLA_GOLD,
        "monthly_fee": 45.0,
        "sla": "99.95% Network Availability",
        "access_type": ACCESS_MOBILE,
    },
    {
        "name": "Business Fibre 100",
        "tier": SLA_SILVER,
        "monthly_fee": 49.99,
        "sla": "99.9% Availability",
        "access_type": ACCESS_FTTP,
    },
    {
        "name": "Business Fibre 500",
        "tier": SLA_GOLD,
        "monthly_fee": 89.99,
        "sla": "99.95% Availability",
        "access_type": ACCESS_FTTP,
    },
    {
        "name": "Ethernet 10 Mbps",
        "tier": SLA_SILVER,
        "monthly_fee": 199.0,
        "sla": "99.95% Availability, 5h MTTR",
        "access_type": ACCESS_ENTERPRISE_ETHERNET,
    },
    {
        "name": "Ethernet 100 Mbps",
        "tier": SLA_GOLD,
        "monthly_fee": 399.0,
        "sla": "99.99% Availability, 4h MTTR",
        "access_type": ACCESS_ENTERPRISE_ETHERNET,
    },
    {
        "name": "Ethernet 1 Gbps",
        "tier": SLA_GOLD,
        "monthly_fee": 799.0,
        "sla": "99.999% Availability, 4h MTTR",
        "access_type": ACCESS_ENTERPRISE_ETHERNET,
    },
    {
        "name": "SD-WAN Managed",
        "tier": SLA_GOLD,
        "monthly_fee": 599.0,
        "sla": "99.99% Availability, Managed SLA",
        "access_type": ACCESS_ENTERPRISE_ETHERNET,
    },
]

# Residential plan probability weights (mobile-dominant market)
RESIDENTIAL_PLAN_WEIGHTS_MOBILE = [0.10, 0.25, 0.25, 0.20, 0.12, 0.08]
RESIDENTIAL_PLAN_WEIGHTS_FTTP = [0.15, 0.30, 0.25, 0.20, 0.10]
RESIDENTIAL_PLAN_WEIGHTS_FTTC = [0.30, 0.40, 0.30]
ENTERPRISE_PLAN_WEIGHTS = [0.20, 0.10, 0.15, 0.10, 0.20, 0.12, 0.05, 0.08]

# Residential access type distribution
# Mobile-dominant: 60% mobile, 25% FTTP, 15% FTTC
RESIDENTIAL_ACCESS_TYPE_WEIGHTS = {
    ACCESS_MOBILE: 0.60,
    ACCESS_FTTP: 0.25,
    ACCESS_FTTC: 0.15,
}

# Enterprise access type distribution
# More diverse: mobile 25%, FTTP 20%, ethernet 55%
# (mapped through plan selection rather than separate weights)

# Account status weights
STATUS_WEIGHTS_RESIDENTIAL = [0.92, 0.04, 0.04]  # ACTIVE, SUSPENDED, DELINQUENT
STATUS_WEIGHTS_ENTERPRISE = [0.96, 0.02, 0.02]

# ---------------------------------------------------------------------------
# Name generators (UK-style)
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Oliver",
    "George",
    "Harry",
    "Jack",
    "Jacob",
    "Noah",
    "Charlie",
    "Muhammad",
    "Thomas",
    "Oscar",
    "William",
    "James",
    "Leo",
    "Alfie",
    "Henry",
    "Archie",
    "Ethan",
    "Joseph",
    "Freddie",
    "Samuel",
    "Olivia",
    "Amelia",
    "Isla",
    "Ava",
    "Emily",
    "Jessica",
    "Ella",
    "Mia",
    "Sophie",
    "Grace",
    "Lily",
    "Chloe",
    "Sophia",
    "Isabella",
    "Freya",
    "Charlotte",
    "Daisy",
    "Poppy",
    "Ruby",
    "Alice",
    "David",
    "Daniel",
    "Alexander",
    "Matthew",
    "Ryan",
    "Luke",
    "Benjamin",
    "Nathan",
    "Andrew",
    "Robert",
    "Christopher",
    "Michael",
    "Sarah",
    "Emma",
    "Hannah",
    "Laura",
    "Kate",
    "Rachel",
    "Louise",
    "Rebecca",
    "Victoria",
    "Jennifer",
    "Helen",
    "Stephanie",
    "Ellie",
    "Phoebe",
    "Evie",
    "Scarlett",
    "Sienna",
    "Layla",
    "Rosie",
    "Zara",
    "Aiden",
    "Liam",
    "Mason",
    "Logan",
    "Jayden",
    "Lucas",
    "Caleb",
    "Theo",
    "Max",
    "Finn",
    "Arthur",
    "Edward",
    "Isaac",
    "Jasper",
    "Dylan",
    "Adam",
    "Sebastian",
    "Toby",
    "Zachary",
    "Kai",
    "Harper",
    "Willow",
    "Ivy",
    "Aurora",
    "Penelope",
    "Luna",
    "Nora",
    "Aria",
    "Eleanor",
    "Violet",
]

LAST_NAMES = [
    "Smith",
    "Jones",
    "Williams",
    "Taylor",
    "Brown",
    "Davies",
    "Evans",
    "Wilson",
    "Thomas",
    "Roberts",
    "Johnson",
    "Lewis",
    "Walker",
    "Robinson",
    "Wood",
    "Thompson",
    "White",
    "Watson",
    "Jackson",
    "Wright",
    "Green",
    "Harris",
    "Cooper",
    "King",
    "Lee",
    "Martin",
    "Clarke",
    "James",
    "Morgan",
    "Hughes",
    "Edwards",
    "Hill",
    "Moore",
    "Clark",
    "Harrison",
    "Scott",
    "Young",
    "Morris",
    "Hall",
    "Ward",
    "Turner",
    "Carter",
    "Phillips",
    "Mitchell",
    "Patel",
    "Adams",
    "Campbell",
    "Anderson",
    "Allen",
    "Cook",
    "Bailey",
    "Palmer",
    "Stevens",
    "Bell",
    "Richardson",
    "Fox",
    "Gray",
    "Rose",
    "Chapman",
    "Hunt",
    "Robertson",
    "Shaw",
    "Simpson",
    "Ellis",
    "Bennett",
    "Murray",
    "Kelly",
    "Graham",
    "Stewart",
    "Stone",
    "Knight",
    "Webb",
    "Spencer",
    "Watts",
    "Butler",
    "Price",
    "Fisher",
    "Holmes",
    "Mills",
    "Ross",
    "Saunders",
    "Barker",
    "Powell",
    "Sullivan",
    "Russell",
    "Dixon",
    "Hamilton",
    "Gibson",
    "Grant",
    "Reynolds",
    "Marshall",
    "Griffiths",
    "Armstrong",
    "Henderson",
    "Perry",
    "Payne",
    "Burton",
    "Gordon",
    "Collins",
    "Murray",
]

COMPANY_PREFIXES = [
    "Apex",
    "Blue",
    "Central",
    "Delta",
    "Eagle",
    "Falcon",
    "Global",
    "Harbour",
    "Icon",
    "Jade",
    "Keystone",
    "Link",
    "Metro",
    "Nova",
    "Omega",
    "Peak",
    "Quantum",
    "Ridge",
    "Summit",
    "Titan",
    "Ultra",
    "Vertex",
    "Western",
    "Xenon",
    "York",
    "Zenith",
    "Albion",
    "Beacon",
    "Crown",
    "Drake",
    "Emerald",
    "Forge",
    "Granite",
    "Highland",
    "Imperial",
    "Jupiter",
    "Kestrel",
    "Linden",
    "Meridian",
    "Neptune",
    "Obsidian",
    "Pinnacle",
    "Regency",
    "Sterling",
    "Tempest",
    "Unity",
    "Vanguard",
    "Windsor",
]

COMPANY_SUFFIXES = [
    "Solutions",
    "Systems",
    "Group",
    "Holdings",
    "Partners",
    "Consulting",
    "Industries",
    "Services",
    "Technologies",
    "Enterprises",
    "Corp",
    "Associates",
    "Capital",
    "Ventures",
    "Networks",
    "Digital",
    "Engineering",
    "Logistics",
    "Trading",
    "Management",
]

# ---------------------------------------------------------------------------
# Helper: load entities from Phase 2
# ---------------------------------------------------------------------------


def _load_entities(config: GeneratorConfig) -> pl.DataFrame:
    """Load ground_truth_entities.parquet from Phase 2 output."""
    path = config.paths.output_dir / "ground_truth_entities.parquet"
    return pl.read_parquet(path)


def _extract_sites(entities_df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract SITE entities with their geo/province information.

    Returns a DataFrame with columns:
        entity_id, province, geo_lat, geo_lon, site_type, deployment_profile
    """
    return entities_df.filter(pl.col("entity_type") == "SITE").select(
        "entity_id",
        "province",
        "geo_lat",
        "geo_lon",
        "site_type",
        "deployment_profile",
    )


def _extract_ont_entities(entities_df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract ONT entities (fixed broadband FTTP termination points).

    Returns DataFrame with columns: entity_id, site_id
    """
    return entities_df.filter(pl.col("entity_type") == "ONT").select("entity_id", "site_id")


def _extract_nte_entities(entities_df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract NTE entities (enterprise Ethernet termination points).

    Returns DataFrame with columns: entity_id, site_id
    """
    return entities_df.filter(pl.col("entity_type") == "NTE").select("entity_id", "site_id")


def _extract_dslam_entities(entities_df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract DSLAM entities (FTTC copper-based access).

    Returns DataFrame with columns: entity_id, site_id
    We use the street cabinet / DSLAM as the access entity for FTTC.
    """
    return entities_df.filter(pl.col("entity_type").is_in(["DSLAM", "STREET_CABINET"])).select("entity_id", "site_id")


# ---------------------------------------------------------------------------
# Name generation
# ---------------------------------------------------------------------------


def _generate_residential_name(rng: np.random.Generator) -> str:
    """Generate a realistic UK-style residential customer name."""
    first = FIRST_NAMES[rng.integers(0, len(FIRST_NAMES))]
    last = LAST_NAMES[rng.integers(0, len(LAST_NAMES))]
    return f"{first} {last}"


def _generate_enterprise_name(rng: np.random.Generator) -> str:
    """Generate a realistic UK-style enterprise name."""
    prefix = COMPANY_PREFIXES[rng.integers(0, len(COMPANY_PREFIXES))]
    suffix = COMPANY_SUFFIXES[rng.integers(0, len(COMPANY_SUFFIXES))]
    return f"{prefix} {suffix} Ltd"


# ---------------------------------------------------------------------------
# External ID generation
# ---------------------------------------------------------------------------


def _generate_external_id(
    customer_type: str,
    index: int,
    rng: np.random.Generator,
) -> str:
    """
    Generate a realistic external account number.

    Residential: ACC-RES-XXXXXXXX  (8-digit zero-padded)
    Enterprise:  ACC-ENT-XXXXXXXX  (8-digit zero-padded)
    """
    if customer_type == CUSTOMER_TYPE_RESIDENTIAL:
        prefix = "ACC-RES"
    else:
        prefix = "ACC-ENT"
    # Use index + random offset to create semi-sequential but non-trivial IDs
    num = 10_000_000 + index + rng.integers(0, 100)
    return f"{prefix}-{num:08d}"


# ---------------------------------------------------------------------------
# Churn risk score generation
# ---------------------------------------------------------------------------


def _generate_churn_risk_scores(
    n: int,
    customer_type: str,
    account_statuses: np.ndarray,
    monthly_fees: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate churn risk scores in [0.0, 1.0].

    Higher risk for:
      - Delinquent/suspended accounts
      - Lower-value plans (less sticky)
      - Residential customers (more churn than enterprise)
    """
    # Base: Beta distribution skewed low (most customers are low-risk)
    if customer_type == CUSTOMER_TYPE_RESIDENTIAL:
        scores = rng.beta(2.0, 5.0, size=n)  # skewed towards low risk
    else:
        scores = rng.beta(1.5, 8.0, size=n)  # enterprise: even lower base risk

    # Boost for suspended/delinquent accounts
    for i in range(n):
        if account_statuses[i] == STATUS_SUSPENDED:
            scores[i] = min(1.0, scores[i] + rng.uniform(0.2, 0.4))
        elif account_statuses[i] == STATUS_DELINQUENT:
            scores[i] = min(1.0, scores[i] + rng.uniform(0.3, 0.5))

    # Slight boost for very cheap plans (less stickiness)
    low_value_mask = monthly_fees < 15.0
    scores[low_value_mask] = np.minimum(
        1.0, scores[low_value_mask] + rng.uniform(0.0, 0.15, size=int(low_value_mask.sum()))
    )

    return np.clip(scores, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Contract end date generation
# ---------------------------------------------------------------------------


def _generate_contract_end_dates(
    n: int,
    customer_type: str,
    rng: np.random.Generator,
) -> list[datetime | None]:
    """
    Generate contract end dates.

    - ~30% of residential have no contract (rolling monthly) → None
    - ~60% have a 12-month contract, ~10% have a 24-month contract
    - Enterprise: ~5% no contract, ~40% 12m, ~35% 24m, ~20% 36m
    - Dates spread across the next 0-24 months from simulation epoch
    """
    dates: list[datetime | None] = []

    if customer_type == CUSTOMER_TYPE_RESIDENTIAL:
        contract_type_probs = [0.30, 0.45, 0.25]  # none, 12m, 24m
        contract_months = [0, 12, 24]
    else:
        contract_type_probs = [0.05, 0.35, 0.35, 0.25]  # none, 12m, 24m, 36m
        contract_months = [0, 12, 24, 36]

    choices = rng.choice(len(contract_type_probs), size=n, p=contract_type_probs)

    for i in range(n):
        ct = choices[i]
        if contract_months[ct] == 0:
            dates.append(None)
        else:
            # Contract started at some point in the past 0-contract_months months,
            # so end date is 0 to contract_months months in the future
            months_remaining = rng.integers(1, contract_months[ct] + 1)
            end_date = SIMULATION_EPOCH + timedelta(days=int(months_remaining * 30.44))
            dates.append(end_date)

    return dates


# ---------------------------------------------------------------------------
# Revenue generation
# ---------------------------------------------------------------------------


def _generate_avg_revenue(
    monthly_fees: np.ndarray,
    customer_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate average monthly revenue.

    Revenue ≈ monthly_fee * usage_multiplier + overage_charges.
    Enterprise customers tend to have more consistent revenue.
    """
    n = len(monthly_fees)
    if customer_type == CUSTOMER_TYPE_RESIDENTIAL:
        # Revenue is fee + some variance (overages, add-ons, etc.)
        multiplier = rng.normal(1.05, 0.15, size=n)
        overage = rng.exponential(2.0, size=n)  # occasional overages
    else:
        # Enterprise: more predictable, higher base
        multiplier = rng.normal(1.02, 0.05, size=n)
        overage = rng.exponential(10.0, size=n)

    revenue = monthly_fees * np.maximum(0.8, multiplier) + overage
    return np.maximum(0.0, revenue).round(2)


# ---------------------------------------------------------------------------
# Batch writer for memory safety
# ---------------------------------------------------------------------------

BATCH_SIZE = 100_000  # Write 100k customers at a time


class _CustomerWriter:
    """
    Streaming Parquet writer that flushes in batches to keep memory bounded.
    """

    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path
        self._writer: pq.ParquetWriter | None = None
        self._rows_written = 0

    def write_batch(self, batch: pa.RecordBatch) -> None:
        """Write a single RecordBatch to the Parquet file."""
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self._output_path,
                schema=CUSTOMERS_SCHEMA,
                compression="zstd",
                compression_level=3,
            )
        self._writer.write_batch(batch)
        self._rows_written += batch.num_rows

    def close(self) -> tuple[int, float]:
        """Close the writer and return (rows_written, file_size_mb)."""
        if self._writer is not None:
            self._writer.close()
        size_mb = 0.0
        if self._output_path.exists():
            size_mb = self._output_path.stat().st_size / (1024 * 1024)
        return self._rows_written, size_mb


# ---------------------------------------------------------------------------
# Core batch generation
# ---------------------------------------------------------------------------


def _generate_customer_batch(
    batch_start: int,
    batch_size: int,
    customer_type: str,
    site_ids: np.ndarray,
    site_provinces: np.ndarray,
    ont_entity_ids: np.ndarray | None,
    ont_site_ids: np.ndarray | None,
    nte_entity_ids: np.ndarray | None,
    nte_site_ids: np.ndarray | None,
    dslam_entity_ids: np.ndarray | None,
    dslam_site_ids: np.ndarray | None,
    tenant_id: str,
    rng: np.random.Generator,
) -> pa.RecordBatch:
    """
    Generate a batch of customer records as a PyArrow RecordBatch.

    This function handles:
      - Customer ID generation (UUID v4)
      - Name generation
      - Site association (random site selection, province lookup)
      - Service plan selection (with access-type routing)
      - Billing account generation
      - Access entity FK assignment for fixed customers
    """
    # Pre-allocate arrays
    customer_ids: list[str] = []
    tenant_ids: list[str] = []
    external_ids: list[str] = []
    customer_types: list[str] = []
    names: list[str] = []
    associated_site_ids: list[str] = []
    provinces: list[str] = []
    service_plan_names: list[str] = []
    service_plan_tiers: list[str] = []
    monthly_fees_list: list[float] = []
    sla_guarantees: list[str | None] = []
    account_statuses_list: list[str] = []
    access_types: list[str] = []
    access_entity_ids: list[str | None] = []

    # Build ONT lookup by site_id for FTTP access entity assignment
    ont_by_site: dict[str, list[str]] = {}
    if ont_entity_ids is not None and ont_site_ids is not None:
        for eid, sid in zip(ont_entity_ids, ont_site_ids):
            if sid is not None and eid is not None:
                ont_by_site.setdefault(str(sid), []).append(str(eid))

    # Build NTE lookup by site_id for enterprise ethernet
    nte_by_site: dict[str, list[str]] = {}
    if nte_entity_ids is not None and nte_site_ids is not None:
        for eid, sid in zip(nte_entity_ids, nte_site_ids):
            if sid is not None and eid is not None:
                nte_by_site.setdefault(str(sid), []).append(str(eid))

    # Build DSLAM/street cabinet lookup for FTTC
    dslam_by_site: dict[str, list[str]] = {}
    if dslam_entity_ids is not None and dslam_site_ids is not None:
        for eid, sid in zip(dslam_entity_ids, dslam_site_ids):
            if sid is not None and eid is not None:
                dslam_by_site.setdefault(str(sid), []).append(str(eid))

    # Collect site IDs that have each access type available
    ont_site_id_set = set(ont_by_site.keys())
    dslam_site_id_set = set(dslam_by_site.keys())

    # Status weights
    if customer_type == CUSTOMER_TYPE_RESIDENTIAL:
        status_choices = rng.choice(
            [STATUS_ACTIVE, STATUS_SUSPENDED, STATUS_DELINQUENT],
            size=batch_size,
            p=STATUS_WEIGHTS_RESIDENTIAL,
        )
    else:
        status_choices = rng.choice(
            [STATUS_ACTIVE, STATUS_SUSPENDED, STATUS_DELINQUENT],
            size=batch_size,
            p=STATUS_WEIGHTS_ENTERPRISE,
        )

    # Pre-select site indices
    site_indices = rng.integers(0, len(site_ids), size=batch_size)

    for i in range(batch_size):
        # Customer ID
        cid = str(uuid.UUID(int=rng.integers(0, 2**128), version=4))
        customer_ids.append(cid)
        tenant_ids.append(tenant_id)

        # External ID
        ext_id = _generate_external_id(customer_type, batch_start + i, rng)
        external_ids.append(ext_id)

        # Type
        customer_types.append(customer_type)

        # Name
        if customer_type == CUSTOMER_TYPE_RESIDENTIAL:
            names.append(_generate_residential_name(rng))
        else:
            names.append(_generate_enterprise_name(rng))

        # Site association
        site_idx = site_indices[i]
        assoc_site_id = str(site_ids[site_idx])
        associated_site_ids.append(assoc_site_id)
        provinces.append(str(site_provinces[site_idx]))

        # Account status
        account_statuses_list.append(str(status_choices[i]))

        # Plan selection depends on customer type and available access
        if customer_type == CUSTOMER_TYPE_RESIDENTIAL:
            # Determine access type based on what's available at this site
            # and weighted probabilities
            available_access = [ACCESS_MOBILE]  # mobile always available
            available_weights = [RESIDENTIAL_ACCESS_TYPE_WEIGHTS[ACCESS_MOBILE]]

            if assoc_site_id in ont_site_id_set:
                available_access.append(ACCESS_FTTP)
                available_weights.append(RESIDENTIAL_ACCESS_TYPE_WEIGHTS[ACCESS_FTTP])

            if assoc_site_id in dslam_site_id_set:
                available_access.append(ACCESS_FTTC)
                available_weights.append(RESIDENTIAL_ACCESS_TYPE_WEIGHTS[ACCESS_FTTC])

            # Normalise weights
            total_w = sum(available_weights)
            norm_weights = [w / total_w for w in available_weights]

            access_type = rng.choice(available_access, p=norm_weights)

            if access_type == ACCESS_MOBILE:
                plan_idx = rng.choice(
                    len(RESIDENTIAL_PLANS_MOBILE),
                    p=RESIDENTIAL_PLAN_WEIGHTS_MOBILE,
                )
                plan = RESIDENTIAL_PLANS_MOBILE[plan_idx]
            elif access_type == ACCESS_FTTP:
                plan_idx = rng.choice(
                    len(RESIDENTIAL_PLANS_FTTP),
                    p=RESIDENTIAL_PLAN_WEIGHTS_FTTP,
                )
                plan = RESIDENTIAL_PLANS_FTTP[plan_idx]
            else:  # FTTC
                plan_idx = rng.choice(
                    len(RESIDENTIAL_PLANS_FTTC),
                    p=RESIDENTIAL_PLAN_WEIGHTS_FTTC,
                )
                plan = RESIDENTIAL_PLANS_FTTC[plan_idx]

        else:
            # Enterprise
            plan_idx = rng.choice(
                len(ENTERPRISE_PLANS),
                p=ENTERPRISE_PLAN_WEIGHTS,
            )
            plan = ENTERPRISE_PLANS[plan_idx]
            access_type = plan["access_type"]

        access_types.append(access_type)
        service_plan_names.append(plan["name"])
        service_plan_tiers.append(plan["tier"])
        monthly_fees_list.append(float(plan["monthly_fee"]))
        sla_guarantees.append(plan["sla"])

        # Access entity ID — FK to ONT, NTE, or DSLAM
        if access_type == ACCESS_FTTP:
            onts = ont_by_site.get(assoc_site_id, [])
            if onts:
                access_entity_ids.append(onts[rng.integers(0, len(onts))])
            else:
                access_entity_ids.append(None)
        elif access_type == ACCESS_ENTERPRISE_ETHERNET:
            ntes = nte_by_site.get(assoc_site_id, [])
            if ntes:
                access_entity_ids.append(ntes[rng.integers(0, len(ntes))])
            else:
                access_entity_ids.append(None)
        elif access_type == ACCESS_FTTC:
            dslams = dslam_by_site.get(assoc_site_id, [])
            if dslams:
                access_entity_ids.append(dslams[rng.integers(0, len(dslams))])
            else:
                access_entity_ids.append(None)
        else:
            # Mobile — no access entity
            access_entity_ids.append(None)

    # Vectorised post-processing
    monthly_fees_arr = np.array(monthly_fees_list, dtype=np.float64)
    account_statuses_arr = np.array(account_statuses_list)

    # Churn risk scores (vectorised)
    churn_scores = _generate_churn_risk_scores(batch_size, customer_type, account_statuses_arr, monthly_fees_arr, rng)

    # Average monthly revenue (vectorised)
    avg_revenue = _generate_avg_revenue(monthly_fees_arr, customer_type, rng)

    # Contract end dates
    contract_dates = _generate_contract_end_dates(batch_size, customer_type, rng)

    # Consent for proactive comms (~70% residential, ~85% enterprise)
    if customer_type == CUSTOMER_TYPE_RESIDENTIAL:
        consent = rng.random(batch_size) < 0.70
    else:
        consent = rng.random(batch_size) < 0.85

    # Convert contract dates to microsecond timestamps for PyArrow
    contract_date_us: list[int | None] = []
    for dt in contract_dates:
        if dt is None:
            contract_date_us.append(None)
        else:
            contract_date_us.append(int(dt.timestamp() * 1_000_000))

    # Build PyArrow arrays
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array(customer_ids, type=pa.string()),
            pa.array(tenant_ids, type=pa.string()),
            pa.array(external_ids, type=pa.string()),
            pa.array(customer_types, type=pa.string()),
            pa.array(names, type=pa.string()),
            pa.array(associated_site_ids, type=pa.string()),
            pa.array(provinces, type=pa.string()),
            pa.array(service_plan_names, type=pa.string()),
            pa.array(service_plan_tiers, type=pa.string()),
            pa.array(monthly_fees_arr.tolist(), type=pa.float64()),
            pa.array(sla_guarantees, type=pa.string()),
            pa.array(account_statuses_list, type=pa.string()),
            pa.array(avg_revenue.tolist(), type=pa.float64()),
            pa.array(contract_date_us, type=pa.timestamp("us", tz="UTC")),
            pa.array(churn_scores.tolist(), type=pa.float64()),
            pa.array(consent.tolist(), type=pa.bool_()),
            pa.array(access_types, type=pa.string()),
            pa.array(access_entity_ids, type=pa.string()),
        ],
        schema=CUSTOMERS_SCHEMA,
    )

    return batch


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_customers(config: GeneratorConfig) -> None:
    """
    Step 07 entry point: Generate customer & BSS data.

    Produces output/customers_bss.parquet with ~1M customers containing:
      - Customer profiles (residential 95% / enterprise 5%)
      - Service plans with SLA tiers (Gold/Silver/Bronze)
      - Billing accounts with status and revenue
      - Site associations (FK to ground_truth_entities)
      - Access entity FKs for fixed broadband customers (ONT, NTE, DSLAM)
      - Churn risk scores and proactive comms consent
    """
    step_start = time.time()

    seed = config.seed_for("step_07_customers")
    rng = np.random.default_rng(seed)
    console.print(f"[dim]Step 07 seed: {seed}[/dim]")

    total_subscribers = config.users.total_subscribers
    residential_count = config.users.residential_count
    enterprise_count = config.users.enterprise_count
    tenant_id = config.tenant_id

    console.print(
        f"[bold]Customer & BSS Generation:[/bold] "
        f"{total_subscribers:,} total subscribers "
        f"({residential_count:,} residential, {enterprise_count:,} enterprise)"
    )

    config.ensure_output_dirs()

    # ── Load topology data ────────────────────────────────────
    console.print("\n[bold]Loading topology data...[/bold]")
    t0 = time.time()
    entities_df = _load_entities(config)
    console.print(f"  [green]✓[/green] {entities_df.height:,} entities loaded in {time.time() - t0:.1f}s")

    # Extract sites
    t0 = time.time()
    sites_df = _extract_sites(entities_df)
    site_count = sites_df.height
    console.print(f"  [green]✓[/green] {site_count:,} sites extracted")

    if site_count == 0:
        console.print(
            "[bold red]Error:[/bold red] No SITE entities found in ground_truth_entities.parquet. Run Phase 2 first."
        )
        raise RuntimeError("No sites found — Phase 2 must be run first.")

    # Extract access entities for FK assignment
    ont_df = _extract_ont_entities(entities_df)
    nte_df = _extract_nte_entities(entities_df)
    dslam_df = _extract_dslam_entities(entities_df)
    console.print(
        f"  [green]✓[/green] Access entities: "
        f"{ont_df.height:,} ONTs, {nte_df.height:,} NTEs, "
        f"{dslam_df.height:,} DSLAMs/Street Cabinets "
        f"in {time.time() - t0:.1f}s"
    )

    # Convert to numpy arrays for fast access
    site_ids = sites_df["entity_id"].to_numpy()
    site_provinces = sites_df["province"].to_numpy()
    site_id_set = set(str(s) for s in site_ids)

    ont_entity_ids = ont_df["entity_id"].to_numpy() if ont_df.height > 0 else None
    ont_site_ids = ont_df["site_id"].to_numpy() if ont_df.height > 0 else None

    nte_entity_ids = nte_df["entity_id"].to_numpy() if nte_df.height > 0 else None
    nte_site_ids = nte_df["site_id"].to_numpy() if nte_df.height > 0 else None

    dslam_entity_ids = dslam_df["entity_id"].to_numpy() if dslam_df.height > 0 else None
    dslam_site_ids = dslam_df["site_id"].to_numpy() if dslam_df.height > 0 else None

    # Free entity DataFrame (keep only what we need)
    del entities_df, sites_df, ont_df, nte_df, dslam_df
    gc.collect()

    # ── Generate residential customers ────────────────────────
    console.print(f"\n[bold]Generating {residential_count:,} residential customers...[/bold]")
    output_path = config.paths.output_dir / "customers_bss.parquet"
    writer = _CustomerWriter(output_path)

    t0 = time.time()
    res_generated = 0
    res_batch_num = 0

    while res_generated < residential_count:
        batch_sz = min(BATCH_SIZE, residential_count - res_generated)
        batch = _generate_customer_batch(
            batch_start=res_generated,
            batch_size=batch_sz,
            customer_type=CUSTOMER_TYPE_RESIDENTIAL,
            site_ids=site_ids,
            site_provinces=site_provinces,
            ont_entity_ids=ont_entity_ids,
            ont_site_ids=ont_site_ids,
            nte_entity_ids=nte_entity_ids,
            nte_site_ids=nte_site_ids,
            dslam_entity_ids=dslam_entity_ids,
            dslam_site_ids=dslam_site_ids,
            tenant_id=tenant_id,
            rng=rng,
        )
        writer.write_batch(batch)
        res_generated += batch_sz
        res_batch_num += 1

        if res_batch_num % 5 == 0 or res_generated >= residential_count:
            console.print(
                f"  [dim]Residential: {res_generated:,}/{residential_count:,} "
                f"({100.0 * res_generated / residential_count:.0f}%)[/dim]"
            )

        del batch
        gc.collect()

    elapsed_res = time.time() - t0
    console.print(f"  [green]✓[/green] {res_generated:,} residential customers in {elapsed_res:.1f}s")

    # ── Generate enterprise customers ─────────────────────────
    console.print(f"\n[bold]Generating {enterprise_count:,} enterprise customers...[/bold]")
    t0 = time.time()
    ent_generated = 0
    ent_batch_num = 0

    while ent_generated < enterprise_count:
        batch_sz = min(BATCH_SIZE, enterprise_count - ent_generated)
        batch = _generate_customer_batch(
            batch_start=residential_count + ent_generated,
            batch_size=batch_sz,
            customer_type=CUSTOMER_TYPE_ENTERPRISE,
            site_ids=site_ids,
            site_provinces=site_provinces,
            ont_entity_ids=ont_entity_ids,
            ont_site_ids=ont_site_ids,
            nte_entity_ids=nte_entity_ids,
            nte_site_ids=nte_site_ids,
            dslam_entity_ids=dslam_entity_ids,
            dslam_site_ids=dslam_site_ids,
            tenant_id=tenant_id,
            rng=rng,
        )
        writer.write_batch(batch)
        ent_generated += batch_sz
        ent_batch_num += 1

        if ent_batch_num % 5 == 0 or ent_generated >= enterprise_count:
            console.print(
                f"  [dim]Enterprise: {ent_generated:,}/{enterprise_count:,} "
                f"({100.0 * ent_generated / enterprise_count:.0f}%)[/dim]"
            )

        del batch
        gc.collect()

    elapsed_ent = time.time() - t0
    console.print(f"  [green]✓[/green] {ent_generated:,} enterprise customers in {elapsed_ent:.1f}s")

    # ── Finalise output ───────────────────────────────────────
    console.print("\n[bold]Finalising customers_bss.parquet...[/bold]")
    rows_written, size_mb = writer.close()
    console.print(f"  [green]✓[/green] {rows_written:,} rows, {size_mb:.1f} MB")

    total_generated = res_generated + ent_generated

    # ── Quick validation read-back ────────────────────────────
    console.print("\n[bold]Validating output...[/bold]")
    t0 = time.time()
    validation_df = pl.read_parquet(output_path)
    val_rows = validation_df.height
    val_cols = validation_df.width

    # Check column count
    expected_cols = len(CUSTOMERS_SCHEMA)
    if val_cols != expected_cols:
        console.print(f"  [bold red]⚠ Column count mismatch:[/bold red] expected {expected_cols}, got {val_cols}")
    else:
        console.print(f"  [green]✓[/green] Column count: {val_cols} (matches contract)")

    # Check row count
    if val_rows != total_generated:
        console.print(f"  [bold red]⚠ Row count mismatch:[/bold red] expected {total_generated:,}, got {val_rows:,}")
    else:
        console.print(f"  [green]✓[/green] Row count: {val_rows:,}")

    # Check null counts on non-nullable columns
    non_nullable_cols = [
        "customer_id",
        "tenant_id",
        "external_id",
        "customer_type",
        "name",
        "associated_site_id",
        "province",
        "service_plan_name",
        "service_plan_tier",
        "monthly_fee",
        "account_status",
        "avg_monthly_revenue",
        "churn_risk_score",
        "consent_proactive_comms",
        "access_type",
    ]
    null_issues = []
    for col_name in non_nullable_cols:
        if col_name in validation_df.columns:
            null_count = validation_df[col_name].null_count()
            if null_count > 0:
                null_issues.append(f"{col_name}: {null_count:,} nulls")

    if null_issues:
        console.print(f"  [bold yellow]⚠ Unexpected nulls:[/bold yellow] " + "; ".join(null_issues))
    else:
        console.print(f"  [green]✓[/green] No unexpected nulls in non-nullable columns")

    # Check FK validity: all associated_site_ids should be in site_id_set
    assoc_sites = set(validation_df["associated_site_id"].unique().to_list())
    invalid_sites = assoc_sites - site_id_set
    if invalid_sites:
        console.print(
            f"  [bold red]⚠ Invalid site FKs:[/bold red] "
            f"{len(invalid_sites):,} associated_site_ids not in ground_truth_entities"
        )
    else:
        console.print(
            f"  [green]✓[/green] All {len(assoc_sites):,} associated_site_ids are valid FKs to ground_truth_entities"
        )

    # Compute distributions for summary
    type_dist = validation_df.group_by("customer_type").len().sort("customer_type")
    access_dist = validation_df.group_by("access_type").len().sort("access_type")
    tier_dist = validation_df.group_by("service_plan_tier").len().sort("service_plan_tier")
    status_dist = validation_df.group_by("account_status").len().sort("account_status")

    # Revenue statistics
    total_monthly_revenue = validation_df["avg_monthly_revenue"].sum()
    avg_rev = validation_df["avg_monthly_revenue"].mean()
    median_rev = validation_df["avg_monthly_revenue"].median()

    # Access entity coverage
    access_entity_count = validation_df.filter(pl.col("access_entity_id").is_not_null()).height
    fixed_customer_count = validation_df.filter(
        pl.col("access_type").is_in([ACCESS_FTTP, ACCESS_FTTC, ACCESS_ENTERPRISE_ETHERNET])
    ).height

    console.print(f"  Validation completed in {time.time() - t0:.1f}s")

    del validation_df
    gc.collect()

    # ── Summary tables ────────────────────────────────────────
    total_elapsed = time.time() - step_start
    console.print()

    # Customer type breakdown
    type_table = Table(
        title="Step 07: Customers & BSS — Customer Type Distribution",
        show_header=True,
    )
    type_table.add_column("Type", style="bold", width=16)
    type_table.add_column("Count", justify="right", width=14)
    type_table.add_column("Share", justify="right", width=10)
    for row in type_dist.iter_rows():
        ctype, count = row[0], row[1]
        share = f"{100.0 * count / max(1, total_generated):.1f}%"
        type_table.add_row(str(ctype), f"{count:,}", share)
    type_table.add_section()
    type_table.add_row("[bold]Total[/bold]", f"[bold]{total_generated:,}[/bold]", "100.0%")
    console.print(type_table)

    # Access type breakdown
    console.print()
    access_table = Table(
        title="Access Type Distribution",
        show_header=True,
    )
    access_table.add_column("Access Type", style="bold", width=24)
    access_table.add_column("Count", justify="right", width=14)
    access_table.add_column("Share", justify="right", width=10)
    for row in access_dist.iter_rows():
        atype, count = row[0], row[1]
        share = f"{100.0 * count / max(1, total_generated):.1f}%"
        access_table.add_row(str(atype), f"{count:,}", share)
    console.print(access_table)

    # SLA tier breakdown
    console.print()
    tier_table = Table(
        title="SLA Tier Distribution",
        show_header=True,
    )
    tier_table.add_column("Tier", style="bold", width=12)
    tier_table.add_column("Count", justify="right", width=14)
    tier_table.add_column("Share", justify="right", width=10)
    for row in tier_dist.iter_rows():
        tier, count = row[0], row[1]
        share = f"{100.0 * count / max(1, total_generated):.1f}%"
        tier_table.add_row(str(tier), f"{count:,}", share)
    console.print(tier_table)

    # Account status breakdown
    console.print()
    status_table = Table(
        title="Account Status Distribution",
        show_header=True,
    )
    status_table.add_column("Status", style="bold", width=16)
    status_table.add_column("Count", justify="right", width=14)
    status_table.add_column("Share", justify="right", width=10)
    for row in status_dist.iter_rows():
        status, count = row[0], row[1]
        share = f"{100.0 * count / max(1, total_generated):.1f}%"
        status_table.add_row(str(status), f"{count:,}", share)
    console.print(status_table)

    # Revenue & metrics summary
    console.print()
    metrics_table = Table(
        title="Revenue & Metrics Summary",
        show_header=True,
    )
    metrics_table.add_column("Metric", style="bold", width=36)
    metrics_table.add_column("Value", justify="right", width=18)
    metrics_table.add_row("Total monthly revenue", f"£{total_monthly_revenue:,.0f}")
    metrics_table.add_row("Average monthly revenue", f"£{avg_rev:,.2f}")
    metrics_table.add_row("Median monthly revenue", f"£{median_rev:,.2f}")
    metrics_table.add_row("Fixed customers with access_entity_id", f"{access_entity_count:,}")
    metrics_table.add_row("Total fixed customers", f"{fixed_customer_count:,}")
    if fixed_customer_count > 0:
        coverage = 100.0 * access_entity_count / fixed_customer_count
        metrics_table.add_row("Access entity FK coverage", f"{coverage:.1f}%")
    metrics_table.add_row("Columns", f"{expected_cols}")
    metrics_table.add_row("File size", f"{size_mb:.1f} MB")
    console.print(metrics_table)

    time_str = f"{total_elapsed:.1f}s" if total_elapsed < 60 else f"{total_elapsed / 60:.1f}m"
    console.print(
        f"\n[bold green]✓ Step 07 complete.[/bold green] "
        f"Generated {total_generated:,} customers ({size_mb:.1f} MB) "
        f"in {time_str}"
    )
    console.print(
        f"[dim]Customer-to-site FKs are valid for join with "
        f"scenario-affected cells in CX Impact Analysis. "
        f"Fixed broadband customers carry access_entity_id FKs "
        f"to ONT/NTE/DSLAM entities from Phase 2.[/dim]"
    )
