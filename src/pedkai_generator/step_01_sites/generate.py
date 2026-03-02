"""
Step 01: Site & Cell Schema Generation.

Generates the physical site inventory and cell-layer schema for a UK-scale
converged operator set in Indonesian geography.

This is the foundation for the entire pipeline — every downstream step
references sites and cells generated here.

Output:
    - intermediate/sites.parquet   (~21,100 sites)
    - intermediate/cells.parquet   (~64,700 logical cell-layers)

Design references:
    - THREAD_SUMMARY Section 3: Scale Parameters
    - THREAD_SUMMARY Section 5: Complete Telecom Topology Model (Physical Layer)
    - THREAD_SUMMARY Section 2: Design Decisions (Geography, RAT Support)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from rich.console import Console
from rich.progress import track

from pedkai_generator.config.settings import (
    ALL_BANDS,
    DENSITY_DEPLOYMENT_WEIGHTS,
    INDONESIA_PROVINCES,
    LTE_BANDS,
    NR_NSA_BANDS,
    NR_SA_BANDS,
    RAT,
    BandConfig,
    DeploymentProfile,
    GeneratorConfig,
    ProvinceConfig,
    SiteType,
    Vendor,
)

console = Console()

# ---------------------------------------------------------------------------
# Constants: deployment-profile-dependent radio parameters
# ---------------------------------------------------------------------------

# Inter-site distance (metres) by deployment profile
ISD_BY_PROFILE: dict[DeploymentProfile, tuple[float, float]] = {
    DeploymentProfile.DENSE_URBAN: (100.0, 400.0),
    DeploymentProfile.URBAN: (300.0, 800.0),
    DeploymentProfile.SUBURBAN: (800.0, 2500.0),
    DeploymentProfile.RURAL: (2000.0, 8000.0),
    DeploymentProfile.DEEP_RURAL: (5000.0, 30000.0),
    DeploymentProfile.INDOOR: (50.0, 200.0),
}

# Antenna height (metres) by deployment profile
ANTENNA_HEIGHT_BY_PROFILE: dict[DeploymentProfile, tuple[float, float]] = {
    DeploymentProfile.DENSE_URBAN: (15.0, 35.0),
    DeploymentProfile.URBAN: (20.0, 40.0),
    DeploymentProfile.SUBURBAN: (25.0, 50.0),
    DeploymentProfile.RURAL: (30.0, 60.0),
    DeploymentProfile.DEEP_RURAL: (30.0, 80.0),
    DeploymentProfile.INDOOR: (3.0, 10.0),
}

# Electrical tilt (degrees) by deployment profile
TILT_BY_PROFILE: dict[DeploymentProfile, tuple[float, float]] = {
    DeploymentProfile.DENSE_URBAN: (4.0, 12.0),
    DeploymentProfile.URBAN: (3.0, 10.0),
    DeploymentProfile.SUBURBAN: (2.0, 8.0),
    DeploymentProfile.RURAL: (1.0, 5.0),
    DeploymentProfile.DEEP_RURAL: (0.5, 3.0),
    DeploymentProfile.INDOOR: (0.0, 2.0),
}

# Which LTE bands are deployed at each profile (probability weights)
LTE_BAND_WEIGHTS_BY_PROFILE: dict[DeploymentProfile, dict[str, float]] = {
    DeploymentProfile.DENSE_URBAN: {"L1800": 0.35, "L2100": 0.30, "L2300": 0.25, "L900": 0.10},
    DeploymentProfile.URBAN: {"L1800": 0.35, "L2100": 0.25, "L900": 0.20, "L2300": 0.20},
    DeploymentProfile.SUBURBAN: {"L900": 0.35, "L1800": 0.35, "L2100": 0.20, "L2300": 0.10},
    DeploymentProfile.RURAL: {"L900": 0.55, "L1800": 0.30, "L2100": 0.10, "L2300": 0.05},
    DeploymentProfile.DEEP_RURAL: {"L900": 0.70, "L1800": 0.25, "L2100": 0.05},
    DeploymentProfile.INDOOR: {"L1800": 0.40, "L2300": 0.35, "L2100": 0.25},
}

# NR NSA band weights (NR SCG leg band)
NR_NSA_BAND_WEIGHTS_BY_PROFILE: dict[DeploymentProfile, dict[str, float]] = {
    DeploymentProfile.DENSE_URBAN: {"n78": 0.60, "n1": 0.15, "n3": 0.15, "n28": 0.10},
    DeploymentProfile.URBAN: {"n78": 0.45, "n1": 0.20, "n3": 0.20, "n28": 0.15},
    DeploymentProfile.SUBURBAN: {"n78": 0.25, "n1": 0.25, "n3": 0.25, "n28": 0.25},
    DeploymentProfile.RURAL: {"n28": 0.50, "n1": 0.25, "n3": 0.15, "n78": 0.10},
    DeploymentProfile.DEEP_RURAL: {"n28": 0.60, "n1": 0.25, "n3": 0.15},
    DeploymentProfile.INDOOR: {"n78": 0.50, "n3": 0.25, "n1": 0.25},
}

# NR SA band weights
NR_SA_BAND_WEIGHTS_BY_PROFILE: dict[DeploymentProfile, dict[str, float]] = {
    DeploymentProfile.DENSE_URBAN: {"n78": 0.45, "n77": 0.25, "n257": 0.15, "n258": 0.15},
    DeploymentProfile.URBAN: {"n78": 0.50, "n77": 0.30, "n257": 0.10, "n258": 0.10},
    DeploymentProfile.SUBURBAN: {"n78": 0.55, "n77": 0.40, "n257": 0.05},
    DeploymentProfile.RURAL: {"n78": 0.60, "n77": 0.40},
    DeploymentProfile.DEEP_RURAL: {"n78": 0.60, "n77": 0.40},
    DeploymentProfile.INDOOR: {"n78": 0.40, "n258": 0.30, "n257": 0.30},
}

# RAT assignment probability by site type + deployment profile
# The thread summary defines the RAT split as:
#   LTE-only:        ~32,000   Most greenfield (rural/deep rural), some suburban, unspecified
#   LTE + 5G NSA:    ~13,000   Suburban greenfield, most rooftop, some streetworks
#   5G SA:           ~6,700    Dense urban streetworks, some rooftop, enterprise in-building
# We translate this into per-cell RAT probability based on site type and profile.
# Target overall split: ~62% LTE (32K), ~25% NSA (13K), ~13% SA (6.7K)
# Greenfield = 11,000 sites × 3 sectors = 33,000 cells (dominant contributor)
# Rooftop = 4,000 × 3 = 12,000 cells
# Streetworks = 5,500 × 1 = 5,500 cells
# In-Building = 500 × 2 = 1,000 cells
# Unspecified = 100 × 2 = 200 cells
# Weights are tuned empirically against these volumes to hit the targets.
RAT_WEIGHTS_BY_SITE_PROFILE: dict[tuple[SiteType, DeploymentProfile], dict[RAT, float]] = {
    # Greenfield (33K cells — must be heavily LTE to pull overall average up)
    (SiteType.GREENFIELD, DeploymentProfile.SUBURBAN): {RAT.LTE: 0.68, RAT.NR_NSA: 0.25, RAT.NR_SA: 0.07},
    (SiteType.GREENFIELD, DeploymentProfile.RURAL): {RAT.LTE: 0.92, RAT.NR_NSA: 0.06, RAT.NR_SA: 0.02},
    (SiteType.GREENFIELD, DeploymentProfile.DEEP_RURAL): {RAT.LTE: 0.97, RAT.NR_NSA: 0.025, RAT.NR_SA: 0.005},
    # Rooftop (12K cells — NSA-heavy: "most rooftop" per thread summary)
    (SiteType.ROOFTOP, DeploymentProfile.URBAN): {RAT.LTE: 0.35, RAT.NR_NSA: 0.45, RAT.NR_SA: 0.20},
    (SiteType.ROOFTOP, DeploymentProfile.DENSE_URBAN): {RAT.LTE: 0.25, RAT.NR_NSA: 0.42, RAT.NR_SA: 0.33},
    # Streetworks (5.5K cells — SA-heavy for dense urban per thread summary)
    (SiteType.STREETWORKS, DeploymentProfile.DENSE_URBAN): {RAT.LTE: 0.12, RAT.NR_NSA: 0.23, RAT.NR_SA: 0.65},
    (SiteType.STREETWORKS, DeploymentProfile.URBAN): {RAT.LTE: 0.40, RAT.NR_NSA: 0.35, RAT.NR_SA: 0.25},
    # In-Building (1K cells — enterprise = SA-forward per thread summary)
    (SiteType.IN_BUILDING, DeploymentProfile.INDOOR): {RAT.LTE: 0.30, RAT.NR_NSA: 0.28, RAT.NR_SA: 0.42},
    # Unspecified (200 cells — minor impact)
    (SiteType.UNSPECIFIED, DeploymentProfile.SUBURBAN): {RAT.LTE: 0.65, RAT.NR_NSA: 0.25, RAT.NR_SA: 0.10},
    (SiteType.UNSPECIFIED, DeploymentProfile.URBAN): {RAT.LTE: 0.45, RAT.NR_NSA: 0.35, RAT.NR_SA: 0.20},
}


# ---------------------------------------------------------------------------
# Province allocation: distribute sites across provinces by density
# ---------------------------------------------------------------------------


def _compute_province_weights(provinces: list[ProvinceConfig]) -> np.ndarray:
    """
    Compute relative site-allocation weights for each province based on
    population density class.  Denser provinces get more sites.
    """
    density_weight_map = {
        "hyper_dense": 25.0,
        "dense": 8.0,
        "moderate": 3.0,
        "sparse": 1.0,
        "very_sparse": 0.3,
    }
    weights = np.array([density_weight_map.get(p.density_class, 1.0) for p in provinces])
    return weights / weights.sum()


def _choose_weighted(rng: np.random.Generator, items: list[str], weights: dict[str, float], size: int) -> list[str]:
    """Choose `size` items from a weighted dict."""
    keys = list(weights.keys())
    probs = np.array([weights[k] for k in keys])
    probs = probs / probs.sum()
    indices = rng.choice(len(keys), size=size, p=probs)
    return [keys[i] for i in indices]


def _generate_lat_lon(
    rng: np.random.Generator,
    province: ProvinceConfig,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate lat/lon coordinates within a province's bounding box with Gaussian clustering."""
    lats = rng.normal(province.lat_center, province.lat_spread * 0.35, size=count)
    lons = rng.normal(province.lon_center, province.lon_spread * 0.35, size=count)
    # Clip to plausible Indonesian bounds
    lats = np.clip(lats, -11.0, 6.0)
    lons = np.clip(lons, 94.0, 141.0)
    return lats, lons


# ---------------------------------------------------------------------------
# Site generation
# ---------------------------------------------------------------------------


def _generate_sites(config: GeneratorConfig, rng: np.random.Generator) -> pl.DataFrame:
    """
    Generate ~21,100 sites distributed across 38 Indonesian provinces.

    Each site gets:
      - Unique site_id (UUID v4)
      - Site type (greenfield, rooftop, streetworks, in_building, unspecified)
      - Deployment profile (dense_urban, urban, suburban, rural, deep_rural, indoor)
      - Province, timezone, lat/lon
      - Vendor assignment (ericsson / nokia)
      - SLA tier

    Returns:
        Polars DataFrame with site records.
    """
    provinces = INDONESIA_PROVINCES
    province_weights = _compute_province_weights(provinces)

    # Build the full site list by type
    site_type_counts: list[tuple[SiteType, int]] = [
        (SiteType.GREENFIELD, config.sites.greenfield),
        (SiteType.ROOFTOP, config.sites.rooftop),
        (SiteType.STREETWORKS, config.sites.streetworks),
        (SiteType.IN_BUILDING, config.sites.in_building),
        (SiteType.UNSPECIFIED, config.sites.unspecified),
    ]

    all_rows: list[dict[str, Any]] = []

    for site_type, count in site_type_counts:
        # Allocate this site type across provinces
        province_indices = rng.choice(len(provinces), size=count, p=province_weights)

        # Decide deployment profile for each site based on its province density
        for i in range(count):
            prov = provinces[province_indices[i]]

            # Get deployment profile weights for this province's density
            profile_weights = DENSITY_DEPLOYMENT_WEIGHTS.get(prov.density_class, {})

            # Filter to profiles valid for this site type
            from pedkai_generator.config.settings import SITE_TYPE_DEPLOYMENT_PROFILES

            valid_profiles = SITE_TYPE_DEPLOYMENT_PROFILES[site_type]

            # Intersect and re-normalise
            filtered = {p: profile_weights.get(p, 0.0) for p in valid_profiles}
            total_w = sum(filtered.values())
            if total_w == 0:
                # Fallback: uniform over valid profiles
                filtered = {p: 1.0 / len(valid_profiles) for p in valid_profiles}
                total_w = 1.0

            profile_keys = list(filtered.keys())
            profile_probs = np.array([filtered[k] / total_w for k in profile_keys])
            chosen_profile = profile_keys[rng.choice(len(profile_keys), p=profile_probs)]

            # Generate location
            lat, lon = _generate_lat_lon(rng, prov, 1)

            # Vendor assignment
            vendor = Vendor.ERICSSON if rng.random() < config.ericsson_fraction else Vendor.NOKIA

            # SLA tier based on deployment profile
            sla_probs = {
                DeploymentProfile.DENSE_URBAN: [0.3, 0.5, 0.2],  # GOLD, SILVER, BRONZE
                DeploymentProfile.URBAN: [0.2, 0.5, 0.3],
                DeploymentProfile.SUBURBAN: [0.1, 0.4, 0.5],
                DeploymentProfile.RURAL: [0.05, 0.25, 0.7],
                DeploymentProfile.DEEP_RURAL: [0.02, 0.18, 0.8],
                DeploymentProfile.INDOOR: [0.4, 0.4, 0.2],
            }
            tiers = ["GOLD", "SILVER", "BRONZE"]
            probs = sla_probs.get(chosen_profile, [0.1, 0.4, 0.5])
            sla_tier = tiers[rng.choice(3, p=probs)]

            # Revenue weight: higher for denser profiles, with some variance
            revenue_base = {
                DeploymentProfile.DENSE_URBAN: 150_000.0,
                DeploymentProfile.URBAN: 80_000.0,
                DeploymentProfile.SUBURBAN: 40_000.0,
                DeploymentProfile.RURAL: 15_000.0,
                DeploymentProfile.DEEP_RURAL: 5_000.0,
                DeploymentProfile.INDOOR: 100_000.0,
            }
            base_rev = revenue_base.get(chosen_profile, 30_000.0)
            revenue = float(rng.lognormal(np.log(base_rev), 0.4))

            site_id = str(uuid.uuid4())

            all_rows.append(
                {
                    "site_id": site_id,
                    "tenant_id": config.tenant_id,
                    "entity_type": "SITE",
                    "name": f"SITE-{site_type.value[:3].upper()}-{prov.name[:3].upper()}-{i:05d}",
                    "external_id": f"NMS-{vendor.value[:3].upper()}-{site_id[:8]}",
                    "domain": "mobile_ran",
                    "site_type": site_type.value,
                    "deployment_profile": chosen_profile.value,
                    "province": prov.name,
                    "timezone": prov.timezone,
                    "geo_lat": float(lat[0]),
                    "geo_lon": float(lon[0]),
                    "vendor": vendor.value,
                    "sla_tier": sla_tier,
                    "revenue_weight": round(revenue, 2),
                    "sectors": config.sites.sectors_per_site(site_type),
                }
            )

    df = pl.DataFrame(all_rows)
    console.print(f"  [dim]Generated {len(df):,} sites across {len(provinces)} provinces[/dim]")
    return df


# ---------------------------------------------------------------------------
# Cell generation
# ---------------------------------------------------------------------------


def _assign_rat_for_cell(
    rng: np.random.Generator,
    site_type: SiteType,
    deployment_profile: DeploymentProfile,
) -> RAT:
    """
    Assign a RAT to a cell based on its site type and deployment profile.
    Falls back to LTE-biased distribution if no specific weights are defined.
    """
    key = (site_type, deployment_profile)
    weights = RAT_WEIGHTS_BY_SITE_PROFILE.get(key)
    if weights is None:
        # Fallback: heavily LTE to maintain overall ~62% target
        weights = {RAT.LTE: 0.78, RAT.NR_NSA: 0.15, RAT.NR_SA: 0.07}

    rat_keys = list(weights.keys())
    probs = np.array([weights[k] for k in rat_keys])
    probs = probs / probs.sum()
    return rat_keys[rng.choice(len(rat_keys), p=probs)]


def _pick_band(
    rng: np.random.Generator,
    rat: RAT,
    profile: DeploymentProfile,
) -> BandConfig:
    """Pick a frequency band for a cell based on its RAT and deployment profile."""
    if rat == RAT.LTE:
        weight_map = LTE_BAND_WEIGHTS_BY_PROFILE.get(profile, {"L900": 0.5, "L1800": 0.5})
        band_dict = LTE_BANDS
    elif rat == RAT.NR_NSA:
        # For NSA, this picks the NR SCG leg's band; the LTE anchor band is separate
        weight_map = NR_NSA_BAND_WEIGHTS_BY_PROFILE.get(profile, {"n78": 0.5, "n1": 0.5})
        band_dict = NR_NSA_BANDS
    elif rat == RAT.NR_SA:
        weight_map = NR_SA_BAND_WEIGHTS_BY_PROFILE.get(profile, {"n78": 0.5, "n77": 0.5})
        band_dict = NR_SA_BANDS
    else:
        raise ValueError(f"Unknown RAT: {rat}")

    band_names = list(weight_map.keys())
    probs = np.array([weight_map[k] for k in band_names])
    probs = probs / probs.sum()
    chosen = band_names[rng.choice(len(band_names), p=probs)]
    return band_dict[chosen]


def _pick_lte_anchor_band(
    rng: np.random.Generator,
    profile: DeploymentProfile,
) -> BandConfig:
    """Pick the LTE anchor band for an NSA cell."""
    # NSA anchors tend to be on lower-frequency LTE bands for coverage
    anchor_weights: dict[str, float] = {
        "L1800": 0.45,
        "L900": 0.30,
        "L2100": 0.20,
        "L2300": 0.05,
    }
    band_names = list(anchor_weights.keys())
    probs = np.array([anchor_weights[k] for k in band_names])
    probs = probs / probs.sum()
    chosen = band_names[rng.choice(len(band_names), p=probs)]
    return LTE_BANDS[chosen]


def _generate_cells(
    config: GeneratorConfig,
    sites_df: pl.DataFrame,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """
    Generate cells for all sites.

    Each site has N sectors (determined by site type). Each sector gets one
    cell with an assigned RAT.  NSA cells produce 2 logical cell-layers
    (LTE anchor + NR SCG leg), each with their own entity_id.

    Returns:
        Polars DataFrame with one row per logical cell-layer (~64,700 rows).
    """
    all_rows: list[dict[str, Any]] = []

    # Pre-convert to dicts for faster iteration
    site_records = sites_df.to_dicts()

    # Counters for RAT distribution tracking
    rat_counts = {RAT.LTE: 0, RAT.NR_NSA: 0, RAT.NR_SA: 0}

    for site in track(site_records, description="  Generating cells...", console=console):
        site_id = site["site_id"]
        site_type = SiteType(site["site_type"])
        profile = DeploymentProfile(site["deployment_profile"])
        vendor = site["vendor"]
        province = site["province"]
        tz = site["timezone"]
        geo_lat = site["geo_lat"]
        geo_lon = site["geo_lon"]
        sla_tier = site["sla_tier"]
        sectors = site["sectors"]

        # ISD for this deployment profile
        isd_range = ISD_BY_PROFILE[profile]
        isd = float(rng.uniform(isd_range[0], isd_range[1]))

        # Antenna height
        height_range = ANTENNA_HEIGHT_BY_PROFILE[profile]
        antenna_height = float(rng.uniform(height_range[0], height_range[1]))

        # Generate sector azimuths (evenly spaced with small jitter)
        if sectors == 1:
            azimuths = [float(rng.uniform(0, 360))]
        elif sectors == 2:
            base = float(rng.uniform(0, 180))
            azimuths = [base, (base + 180) % 360]
        else:
            # 3 sectors: 120° apart with ±5° jitter
            base = float(rng.uniform(0, 120))
            azimuths = [(base + i * 120 + float(rng.normal(0, 3))) % 360 for i in range(sectors)]

        for sector_idx in range(sectors):
            azimuth = azimuths[sector_idx]

            # Electrical tilt with per-cell jitter
            tilt_range = TILT_BY_PROFILE[profile]
            tilt = float(rng.uniform(tilt_range[0], tilt_range[1]))

            # Assign RAT
            rat = _assign_rat_for_cell(rng, site_type, profile)
            rat_counts[rat] += 1

            # Small per-cell lat/lon offset (cells on same site are co-located but
            # slightly offset to avoid identical coordinates in visualisations)
            cell_lat = geo_lat + float(rng.normal(0, 0.0001))
            cell_lon = geo_lon + float(rng.normal(0, 0.0001))

            if rat == RAT.LTE:
                # Single LTE cell
                band = _pick_band(rng, RAT.LTE, profile)
                cell_id = str(uuid.uuid4())
                name_prefix = "CELL-LTE"
                all_rows.append(
                    _build_cell_row(
                        cell_id=cell_id,
                        tenant_id=config.tenant_id,
                        name=f"{name_prefix}-{site_id[:8]}-S{sector_idx}-{band.name}",
                        site_id=site_id,
                        site_type=site_type.value,
                        profile=profile.value,
                        province=province,
                        timezone=tz,
                        geo_lat=cell_lat,
                        geo_lon=cell_lon,
                        vendor=vendor,
                        sla_tier=sla_tier,
                        rat=RAT.LTE,
                        band=band,
                        sector_id=sector_idx,
                        azimuth=azimuth,
                        tilt=tilt,
                        antenna_height=antenna_height,
                        isd=isd,
                        is_nsa_anchor=False,
                        is_nsa_scg_leg=False,
                        nsa_anchor_cell_id=None,
                        parent_entity_type="ENODEB",
                        cell_entity_type="LTE_CELL",
                    )
                )

            elif rat == RAT.NR_NSA:
                # EN-DC: Two logical cell-layers
                # 1) LTE anchor leg
                lte_anchor_band = _pick_lte_anchor_band(rng, profile)
                lte_anchor_id = str(uuid.uuid4())
                all_rows.append(
                    _build_cell_row(
                        cell_id=lte_anchor_id,
                        tenant_id=config.tenant_id,
                        name=f"CELL-NSA-LTE-{site_id[:8]}-S{sector_idx}-{lte_anchor_band.name}",
                        site_id=site_id,
                        site_type=site_type.value,
                        profile=profile.value,
                        province=province,
                        timezone=tz,
                        geo_lat=cell_lat,
                        geo_lon=cell_lon,
                        vendor=vendor,
                        sla_tier=sla_tier,
                        rat=RAT.NR_NSA,
                        band=lte_anchor_band,
                        sector_id=sector_idx,
                        azimuth=azimuth,
                        tilt=tilt,
                        antenna_height=antenna_height,
                        isd=isd,
                        is_nsa_anchor=True,
                        is_nsa_scg_leg=False,
                        nsa_anchor_cell_id=None,
                        parent_entity_type="ENODEB",
                        cell_entity_type="LTE_CELL",
                    )
                )

                # 2) NR SCG leg
                nr_band = _pick_band(rng, RAT.NR_NSA, profile)
                nr_scg_id = str(uuid.uuid4())
                all_rows.append(
                    _build_cell_row(
                        cell_id=nr_scg_id,
                        tenant_id=config.tenant_id,
                        name=f"CELL-NSA-NR-{site_id[:8]}-S{sector_idx}-{nr_band.name}",
                        site_id=site_id,
                        site_type=site_type.value,
                        profile=profile.value,
                        province=province,
                        timezone=tz,
                        geo_lat=cell_lat,
                        geo_lon=cell_lon,
                        vendor=vendor,
                        sla_tier=sla_tier,
                        rat=RAT.NR_NSA,
                        band=nr_band,
                        sector_id=sector_idx,
                        azimuth=azimuth,
                        tilt=tilt,
                        antenna_height=antenna_height,
                        isd=isd,
                        is_nsa_anchor=False,
                        is_nsa_scg_leg=True,
                        nsa_anchor_cell_id=lte_anchor_id,
                        parent_entity_type="GNODEB",
                        cell_entity_type="NR_CELL",
                    )
                )

            elif rat == RAT.NR_SA:
                # Single NR SA cell
                band = _pick_band(rng, RAT.NR_SA, profile)
                cell_id = str(uuid.uuid4())
                all_rows.append(
                    _build_cell_row(
                        cell_id=cell_id,
                        tenant_id=config.tenant_id,
                        name=f"CELL-SA-NR-{site_id[:8]}-S{sector_idx}-{band.name}",
                        site_id=site_id,
                        site_type=site_type.value,
                        profile=profile.value,
                        province=province,
                        timezone=tz,
                        geo_lat=cell_lat,
                        geo_lon=cell_lon,
                        vendor=vendor,
                        sla_tier=sla_tier,
                        rat=RAT.NR_SA,
                        band=band,
                        sector_id=sector_idx,
                        azimuth=azimuth,
                        tilt=tilt,
                        antenna_height=antenna_height,
                        isd=isd,
                        is_nsa_anchor=False,
                        is_nsa_scg_leg=False,
                        nsa_anchor_cell_id=None,
                        parent_entity_type="GNODEB",
                        cell_entity_type="NR_CELL",
                    )
                )

    df = pl.DataFrame(all_rows)

    # Report RAT distribution
    total_physical = sum(rat_counts.values())
    console.print(f"  [dim]Physical cells by RAT assignment:[/dim]")
    for r, c in rat_counts.items():
        console.print(f"    [dim]{r.value}: {c:,} ({c / total_physical * 100:.1f}%)[/dim]")
    console.print(f"  [dim]Logical cell-layers (incl. NSA dual-stream): {len(df):,}[/dim]")

    return df


def _build_cell_row(
    *,
    cell_id: str,
    tenant_id: str,
    name: str,
    site_id: str,
    site_type: str,
    profile: str,
    province: str,
    timezone: str,
    geo_lat: float,
    geo_lon: float,
    vendor: str,
    sla_tier: str,
    rat: RAT,
    band: BandConfig,
    sector_id: int,
    azimuth: float,
    tilt: float,
    antenna_height: float,
    isd: float,
    is_nsa_anchor: bool,
    is_nsa_scg_leg: bool,
    nsa_anchor_cell_id: str | None,
    parent_entity_type: str,
    cell_entity_type: str,
) -> dict[str, Any]:
    """Build a single cell row dict."""
    return {
        "cell_id": cell_id,
        "tenant_id": tenant_id,
        "entity_type": cell_entity_type,
        "name": name,
        "external_id": f"NMS-{vendor[:3].upper()}-{cell_id[:8]}",
        "domain": "mobile_ran",
        "site_id": site_id,
        "site_type": site_type,
        "deployment_profile": profile,
        "province": province,
        "timezone": timezone,
        "geo_lat": round(geo_lat, 6),
        "geo_lon": round(geo_lon, 6),
        "vendor": vendor,
        "sla_tier": sla_tier,
        "rat_type": rat.value,
        "band": band.name,
        "bandwidth_mhz": band.bandwidth_mhz,
        "max_tx_power_dbm": band.max_tx_power_dbm,
        "max_prbs": band.max_prbs,
        "frequency_mhz": band.frequency_mhz,
        "duplex": band.duplex,
        "sector_id": sector_id,
        "azimuth_deg": round(azimuth, 2),
        "electrical_tilt_deg": round(tilt, 2),
        "antenna_height_m": round(antenna_height, 1),
        "inter_site_distance_m": round(isd, 1),
        "is_nsa_anchor": is_nsa_anchor,
        "is_nsa_scg_leg": is_nsa_scg_leg,
        "nsa_anchor_cell_id": nsa_anchor_cell_id,
        "parent_entity_type": parent_entity_type,
    }


# ---------------------------------------------------------------------------
# RAT distribution balancing
# ---------------------------------------------------------------------------


def _check_and_report_rat_split(
    cells_df: pl.DataFrame,
    config: GeneratorConfig,
) -> None:
    """
    Report the actual RAT split vs. the target from config.
    The probabilistic assignment won't exactly hit targets, and that's fine —
    but we log the comparison.
    """
    # Physical cells = non-SCG-leg rows
    physical = cells_df.filter(pl.col("is_nsa_scg_leg") == False)  # noqa: E712

    lte_count = physical.filter(pl.col("rat_type") == "LTE").height
    nsa_count = physical.filter(pl.col("rat_type") == "NR_NSA").height
    sa_count = physical.filter(pl.col("rat_type") == "NR_SA").height
    total = lte_count + nsa_count + sa_count

    console.print(f"\n  [bold]RAT Split Comparison:[/bold]")
    console.print(f"  {'RAT':<12} {'Target':>10} {'Actual':>10} {'Delta':>10}")
    console.print(f"  {'─' * 44}")

    for label, target, actual in [
        ("LTE", config.rat_split.lte_only, lte_count),
        ("NR_NSA", config.rat_split.lte_plus_nsa, nsa_count),
        ("NR_SA", config.rat_split.nr_sa, sa_count),
    ]:
        delta = actual - target
        sign = "+" if delta >= 0 else ""
        console.print(f"  {label:<12} {target:>10,} {actual:>10,} {sign}{delta:>9,}")

    console.print(f"  {'─' * 44}")
    console.print(f"  {'Total':<12} {config.rat_split.total_physical_cells:>10,} {total:>10,}")

    # Logical cell-layers (all rows including SCG legs)
    total_logical = cells_df.height
    nsa_scg = cells_df.filter(pl.col("is_nsa_scg_leg") == True).height  # noqa: E712
    console.print(f"\n  [dim]NSA NR SCG legs: {nsa_scg:,}[/dim]")
    console.print(
        f"  [dim]Total logical cell-layers: {total_logical:,} (target: {config.rat_split.total_logical_cell_layers:,})[/dim]"
    )


# ---------------------------------------------------------------------------
# Vendor naming helper (external_id patterns)
# ---------------------------------------------------------------------------


def _build_vendor_external_ids(
    cells_df: pl.DataFrame,
    rng: np.random.Generator,
) -> pl.DataFrame:
    """
    Enrich cells with realistic vendor-specific external IDs.

    Ericsson ENM pattern: "SubNetwork=ERBS_<region>,MeContext=<enodeb>,
                           ManagedElement=1,ENodeBFunction=1,EUtranCellFDD=<cellname>"
    Nokia NetAct pattern: "PLMN-PLMN/<BSC>/<LNBTS-id>/LNCEL-<id>"

    We simplify to a recognisable pattern per vendor.
    """
    external_ids: list[str] = []

    for row in cells_df.iter_rows(named=True):
        vendor = row["vendor"]
        cell_id = row["cell_id"][:8]
        name = row["name"]
        band = row["band"]
        sector = row["sector_id"]

        if vendor == "ericsson":
            # Ericsson-style MO path
            if row["entity_type"] == "LTE_CELL":
                ext_id = f"SubNetwork=ERBS_{row['province'][:6]},MeContext=eNB_{cell_id},EUtranCellFDD={name}"
            else:
                ext_id = f"SubNetwork=NR_{row['province'][:6]},MeContext=gNB_{cell_id},NRCellDU={name}"
        else:
            # Nokia NetAct style
            if row["entity_type"] == "LTE_CELL":
                ext_id = f"PLMN-PLMN/RNC_{row['province'][:4]}/LNBTS-{cell_id}/LNCEL-{sector}-{band}"
            else:
                ext_id = f"PLMN-PLMN/NRC_{row['province'][:4]}/NRBTS-{cell_id}/NRCEL-{sector}-{band}"

        external_ids.append(ext_id)

    return cells_df.with_columns(pl.Series("external_id", external_ids))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_sites_and_cells(config: GeneratorConfig) -> None:
    """
    Step 01 entry point: Generate sites and cells, write to intermediate Parquet files.

    Called by the CLI orchestrator. Deterministic given config.global_seed.

    Outputs:
        - {intermediate_dir}/sites.parquet
        - {intermediate_dir}/cells.parquet
    """
    seed = config.seed_for("step_01_sites")
    rng = np.random.default_rng(seed)
    console.print(f"  [dim]Seed for step_01: {seed}[/dim]")

    # ── Generate sites ──────────────────────────────────────
    console.print("\n  [bold]Generating sites...[/bold]")
    sites_df = _generate_sites(config, rng)

    # Validate
    assert sites_df.height == config.sites.total, f"Expected {config.sites.total} sites, got {sites_df.height}"

    # ── Generate cells ──────────────────────────────────────
    console.print("\n  [bold]Generating cells...[/bold]")
    cells_df = _generate_cells(config, sites_df, rng)

    # Enrich with vendor-specific external IDs
    console.print("  [dim]Enriching vendor external IDs...[/dim]")
    cells_df = _build_vendor_external_ids(cells_df, rng)

    # Report RAT split
    _check_and_report_rat_split(cells_df, config)

    # ── Write intermediate Parquet files ────────────────────
    intermediate_dir = config.paths.intermediate_dir
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    sites_path = intermediate_dir / "sites.parquet"
    cells_path = intermediate_dir / "cells.parquet"

    sites_df.write_parquet(sites_path, compression="zstd", compression_level=3)
    cells_df.write_parquet(cells_path, compression="zstd", compression_level=3)

    sites_size_mb = sites_path.stat().st_size / (1024 * 1024)
    cells_size_mb = cells_path.stat().st_size / (1024 * 1024)

    console.print(
        f"\n  [bold green]✓ Sites written:[/bold green] {sites_path} ({sites_size_mb:.1f} MB, {sites_df.height:,} rows)"
    )
    console.print(
        f"  [bold green]✓ Cells written:[/bold green] {cells_path} ({cells_size_mb:.1f} MB, {cells_df.height:,} rows)"
    )

    # ── Summary statistics ──────────────────────────────────
    _print_summary(sites_df, cells_df)


def _print_summary(sites_df: pl.DataFrame, cells_df: pl.DataFrame) -> None:
    """Print summary statistics for the generated data."""
    from rich.table import Table

    console.print("\n  [bold]Site Distribution by Type:[/bold]")
    table = Table(show_header=True, header_style="bold cyan", padding=(0, 2))
    table.add_column("Site Type", width=15)
    table.add_column("Count", justify="right", width=8)
    table.add_column("% of Total", justify="right", width=10)

    total_sites = sites_df.height
    for site_type in ["greenfield", "rooftop", "streetworks", "in_building", "unspecified"]:
        count = sites_df.filter(pl.col("site_type") == site_type).height
        pct = count / total_sites * 100 if total_sites > 0 else 0
        table.add_row(site_type, f"{count:,}", f"{pct:.1f}%")
    table.add_section()
    table.add_row("[bold]Total[/bold]", f"[bold]{total_sites:,}[/bold]", "[bold]100.0%[/bold]")
    console.print(table)

    console.print("\n  [bold]Site Distribution by Deployment Profile:[/bold]")
    table2 = Table(show_header=True, header_style="bold cyan", padding=(0, 2))
    table2.add_column("Profile", width=15)
    table2.add_column("Count", justify="right", width=8)
    table2.add_column("% of Total", justify="right", width=10)

    for profile in ["dense_urban", "urban", "suburban", "rural", "deep_rural", "indoor"]:
        count = sites_df.filter(pl.col("deployment_profile") == profile).height
        pct = count / total_sites * 100 if total_sites > 0 else 0
        table2.add_row(profile, f"{count:,}", f"{pct:.1f}%")
    console.print(table2)

    console.print("\n  [bold]Site Distribution by Vendor:[/bold]")
    table3 = Table(show_header=True, header_style="bold cyan", padding=(0, 2))
    table3.add_column("Vendor", width=15)
    table3.add_column("Count", justify="right", width=8)
    table3.add_column("% of Total", justify="right", width=10)

    for vendor in ["ericsson", "nokia"]:
        count = sites_df.filter(pl.col("vendor") == vendor).height
        pct = count / total_sites * 100 if total_sites > 0 else 0
        table3.add_row(vendor, f"{count:,}", f"{pct:.1f}%")
    console.print(table3)

    console.print("\n  [bold]Top 10 Provinces by Site Count:[/bold]")
    province_counts = (
        sites_df.group_by("province").agg(pl.count().alias("count")).sort("count", descending=True).head(10)
    )
    table4 = Table(show_header=True, header_style="bold cyan", padding=(0, 2))
    table4.add_column("Province", width=25)
    table4.add_column("Sites", justify="right", width=8)
    table4.add_column("% of Total", justify="right", width=10)

    for row in province_counts.iter_rows(named=True):
        pct = row["count"] / total_sites * 100
        table4.add_row(row["province"], f"{row['count']:,}", f"{pct:.1f}%")
    console.print(table4)

    console.print("\n  [bold]Cell Distribution by Band:[/bold]")
    band_counts = cells_df.group_by("band").agg(pl.count().alias("count")).sort("count", descending=True)
    table5 = Table(show_header=True, header_style="bold cyan", padding=(0, 2))
    table5.add_column("Band", width=10)
    table5.add_column("Cells", justify="right", width=8)
    table5.add_column("% of Total", justify="right", width=10)

    total_cells = cells_df.height
    for row in band_counts.iter_rows(named=True):
        pct = row["count"] / total_cells * 100
        table5.add_row(row["band"], f"{row['count']:,}", f"{pct:.1f}%")
    console.print(table5)
