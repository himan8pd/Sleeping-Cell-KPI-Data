"""
Mobile RAN topology domain builder.

Converts the sites and cells generated in Step 01 into full topology entities
and relationships for the Mobile RAN domain:

Physical hierarchy:
  SITE → CABINET → POWER_SUPPLY / CLIMATE_CONTROL / TRANSMISSION_EQUIPMENT
  SITE → ANTENNA_SYSTEM → ANTENNA / RRU / FEEDER_CABLE
  SITE → BBU → ENODEB → LTE_CELL
  SITE → BBU → GNODEB → GNODEB_DU / GNODEB_CU_CP / GNODEB_CU_UP → NR_CELL
  SITE → GPS_RECEIVER

Relationships:
  HOSTS, POWERS, COOLS, TIMING_FROM, ANCHORS

Target: ~170,000 entities, ~400,000 relationships (mobile_ran domain)

This module reads from intermediate/sites.parquet and intermediate/cells.parquet
produced by Step 01.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import numpy as np
import polars as pl
from rich.console import Console

from pedkai_generator.config.settings import RAT, GeneratorConfig
from pedkai_generator.step_02_topology.builders import (
    TopologyAccumulator,
    make_entity,
    make_relationship,
    props_json,
    vendor_external_id,
)

console = Console()


# ---------------------------------------------------------------------------
# Helper: deterministic UUID from components (for reproducibility)
# ---------------------------------------------------------------------------


def _make_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_mobile_ran_topology(
    config: GeneratorConfig,
    acc: TopologyAccumulator,
    rng: np.random.Generator,
) -> None:
    """
    Build the entire Mobile RAN topology domain.

    Reads sites.parquet and cells.parquet from the intermediate directory,
    then creates all sub-site entities (cabinet, power, cooling, antennas,
    BBUs, eNBs, gNBs, CU/DU splits) and their relationships.

    All entities and relationships are appended to the shared TopologyAccumulator.
    """
    intermediate = config.paths.intermediate_dir

    # ── Load Step 01 outputs ──────────────────────────────────
    sites_df = pl.read_parquet(intermediate / "sites.parquet")
    cells_df = pl.read_parquet(intermediate / "cells.parquet")

    console.print(f"    [dim]Loaded {sites_df.height:,} sites, {cells_df.height:,} cell-layers[/dim]")

    # Pre-index cells by site_id for fast lookup
    cells_by_site: dict[str, list[dict[str, Any]]] = {}
    for cell in cells_df.iter_rows(named=True):
        sid = cell["site_id"]
        if sid not in cells_by_site:
            cells_by_site[sid] = []
        cells_by_site[sid].append(cell)

    tenant = config.tenant_id

    # Counters for progress reporting
    n_sites = 0
    n_cabinets = 0
    n_power = 0
    n_battery = 0
    n_mains = 0
    n_climate = 0
    n_transmission = 0
    n_antenna_sys = 0
    n_antenna = 0
    n_rru = 0
    n_feeder = 0
    n_bbu = 0
    n_enodeb = 0
    n_gnodeb = 0
    n_gnodeb_du = 0
    n_gnodeb_cu_cp = 0
    n_gnodeb_cu_up = 0
    n_gps = 0
    n_lte_cell = 0
    n_nr_cell = 0
    n_rels = 0

    # ── Process each site ─────────────────────────────────────
    for site_row in sites_df.iter_rows(named=True):
        site_id = site_row["site_id"]
        vendor = site_row["vendor"]
        province = site_row["province"]
        tz = site_row["timezone"]
        geo_lat = site_row["geo_lat"]
        geo_lon = site_row["geo_lon"]
        site_type = site_row["site_type"]
        profile = site_row["deployment_profile"]
        sla_tier = site_row["sla_tier"]
        revenue = site_row["revenue_weight"]

        site_cells = cells_by_site.get(site_id, [])
        n_sectors = site_row["sectors"]

        # --- SITE entity (already has an ID from Step 01) ---
        site_entity = make_entity(
            entity_id=site_id,
            tenant_id=tenant,
            entity_type="SITE",
            name=site_row["name"],
            external_id=site_row["external_id"],
            domain="mobile_ran",
            geo_lat=geo_lat,
            geo_lon=geo_lon,
            site_type=site_type,
            deployment_profile=profile,
            province=province,
            timezone=tz,
            vendor=vendor,
            sla_tier=sla_tier,
            revenue_weight=revenue,
        )
        acc.add_entity(site_entity)
        n_sites += 1

        # --- CABINET ---
        cabinet_id = _make_id()
        cabinet = make_entity(
            entity_id=cabinet_id,
            tenant_id=tenant,
            entity_type="CABINET",
            name=f"CAB-{site_id[:8]}",
            external_id=vendor_external_id(vendor, "CABINET", province, cabinet_id),
            domain="mobile_ran",
            geo_lat=geo_lat,
            geo_lon=geo_lon,
            site_id=site_id,
            site_type=site_type,
            deployment_profile=profile,
            province=province,
            timezone=tz,
            vendor=vendor,
        )
        acc.add_entity(cabinet)
        n_cabinets += 1

        # SITE → CABINET
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=site_id,
                from_entity_type="SITE",
                relationship_type="HOSTS",
                to_entity_id=cabinet_id,
                to_entity_type="CABINET",
                domain="mobile_ran",
            )
        )
        n_rels += 1

        # --- POWER_SUPPLY ---
        ps_id = _make_id()
        ps = make_entity(
            entity_id=ps_id,
            tenant_id=tenant,
            entity_type="POWER_SUPPLY",
            name=f"PWR-{site_id[:8]}",
            external_id=vendor_external_id(vendor, "PWR_SUPP", province, ps_id),
            domain="mobile_ran",
            site_id=site_id,
            site_type=site_type,
            deployment_profile=profile,
            province=province,
            timezone=tz,
            vendor=vendor,
            properties_json=props_json(type="rectifier", rating_kw=round(float(rng.uniform(3, 12)), 1)),
        )
        acc.add_entity(ps)
        n_power += 1

        # CABINET hosts POWER_SUPPLY
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=cabinet_id,
                from_entity_type="CABINET",
                relationship_type="HOSTS",
                to_entity_id=ps_id,
                to_entity_type="POWER_SUPPLY",
                domain="mobile_ran",
            )
        )
        n_rels += 1

        # POWER_SUPPLY → CABINET (powers)
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=ps_id,
                from_entity_type="POWER_SUPPLY",
                relationship_type="POWERS",
                to_entity_id=cabinet_id,
                to_entity_type="CABINET",
                domain="mobile_ran",
            )
        )
        n_rels += 1

        # --- BATTERY_BANK ---
        bat_id = _make_id()
        bat = make_entity(
            entity_id=bat_id,
            tenant_id=tenant,
            entity_type="BATTERY_BANK",
            name=f"BAT-{site_id[:8]}",
            external_id=vendor_external_id(vendor, "BATTERY", province, bat_id),
            domain="mobile_ran",
            site_id=site_id,
            site_type=site_type,
            province=province,
            timezone=tz,
            vendor=vendor,
            parent_entity_id=ps_id,
            properties_json=props_json(
                voltage_v=48.0,
                capacity_ah=round(float(rng.uniform(100, 400)), 0),
            ),
        )
        acc.add_entity(bat)
        n_battery += 1

        # POWER_SUPPLY hosts BATTERY_BANK
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=ps_id,
                from_entity_type="POWER_SUPPLY",
                relationship_type="HOSTS",
                to_entity_id=bat_id,
                to_entity_type="BATTERY_BANK",
                domain="mobile_ran",
            )
        )
        n_rels += 1

        # --- MAINS_CONNECTION ---
        mains_id = _make_id()
        mains = make_entity(
            entity_id=mains_id,
            tenant_id=tenant,
            entity_type="MAINS_CONNECTION",
            name=f"MAINS-{site_id[:8]}",
            domain="mobile_ran",
            site_id=site_id,
            site_type=site_type,
            province=province,
            timezone=tz,
            parent_entity_id=ps_id,
            properties_json=props_json(
                phase="3-phase" if profile in ("dense_urban", "urban") else "single-phase",
                supply_kva=round(float(rng.uniform(10, 50)), 0),
            ),
        )
        acc.add_entity(mains)
        n_mains += 1

        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=ps_id,
                from_entity_type="POWER_SUPPLY",
                relationship_type="HOSTS",
                to_entity_id=mains_id,
                to_entity_type="MAINS_CONNECTION",
                domain="mobile_ran",
            )
        )
        n_rels += 1

        # --- CLIMATE_CONTROL ---
        cc_id = _make_id()
        cc = make_entity(
            entity_id=cc_id,
            tenant_id=tenant,
            entity_type="CLIMATE_CONTROL",
            name=f"COOL-{site_id[:8]}",
            external_id=vendor_external_id(vendor, "CLIMATE", province, cc_id),
            domain="mobile_ran",
            site_id=site_id,
            site_type=site_type,
            province=province,
            timezone=tz,
            vendor=vendor,
            parent_entity_id=cabinet_id,
            properties_json=props_json(cooling_capacity_kw=round(float(rng.uniform(2, 8)), 1)),
        )
        acc.add_entity(cc)
        n_climate += 1

        # CABINET hosts CLIMATE_CONTROL
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=cabinet_id,
                from_entity_type="CABINET",
                relationship_type="HOSTS",
                to_entity_id=cc_id,
                to_entity_type="CLIMATE_CONTROL",
                domain="mobile_ran",
            )
        )
        n_rels += 1

        # CLIMATE_CONTROL cools CABINET
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=cc_id,
                from_entity_type="CLIMATE_CONTROL",
                relationship_type="COOLS",
                to_entity_id=cabinet_id,
                to_entity_type="CABINET",
                domain="mobile_ran",
            )
        )
        n_rels += 1

        # --- TRANSMISSION_EQUIPMENT ---
        tx_id = _make_id()
        # Decide backhaul type: dense/urban → fibre, rural/deep_rural → microwave, suburban → mix
        if profile in ("rural", "deep_rural"):
            bh_type = "microwave"
        elif profile in ("dense_urban", "urban", "indoor"):
            bh_type = "fibre"
        else:
            bh_type = "fibre" if rng.random() < 0.7 else "microwave"

        tx = make_entity(
            entity_id=tx_id,
            tenant_id=tenant,
            entity_type="TRANSMISSION_EQUIPMENT",
            name=f"TX-{site_id[:8]}",
            domain="mobile_ran",
            site_id=site_id,
            site_type=site_type,
            province=province,
            timezone=tz,
            vendor=vendor,
            parent_entity_id=cabinet_id,
            properties_json=props_json(backhaul_type=bh_type),
        )
        acc.add_entity(tx)
        n_transmission += 1

        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=cabinet_id,
                from_entity_type="CABINET",
                relationship_type="HOSTS",
                to_entity_id=tx_id,
                to_entity_type="TRANSMISSION_EQUIPMENT",
                domain="mobile_ran",
            )
        )
        n_rels += 1

        # --- GPS_RECEIVER ---
        gps_id = _make_id()
        gps = make_entity(
            entity_id=gps_id,
            tenant_id=tenant,
            entity_type="GPS_RECEIVER",
            name=f"GPS-{site_id[:8]}",
            domain="mobile_ran",
            site_id=site_id,
            site_type=site_type,
            province=province,
            timezone=tz,
            parent_entity_id=site_id,
        )
        acc.add_entity(gps)
        n_gps += 1

        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=site_id,
                from_entity_type="SITE",
                relationship_type="HOSTS",
                to_entity_id=gps_id,
                to_entity_type="GPS_RECEIVER",
                domain="mobile_ran",
            )
        )
        n_rels += 1

        # --- PER-SECTOR: ANTENNA_SYSTEM, ANTENNA, RRU, FEEDER, BBU ---
        # Group cells by sector to understand what bands/RATs exist per sector
        sector_cells: dict[int, list[dict[str, Any]]] = {}
        for cell in site_cells:
            s = cell["sector_id"]
            if s not in sector_cells:
                sector_cells[s] = []
            sector_cells[s].append(cell)

        # Track BBU (one per site, shared across sectors)
        bbu_id = _make_id()
        bbu = make_entity(
            entity_id=bbu_id,
            tenant_id=tenant,
            entity_type="BBU",
            name=f"BBU-{site_id[:8]}",
            external_id=vendor_external_id(vendor, "BBU", province, bbu_id),
            domain="mobile_ran",
            geo_lat=geo_lat,
            geo_lon=geo_lon,
            site_id=site_id,
            site_type=site_type,
            deployment_profile=profile,
            province=province,
            timezone=tz,
            vendor=vendor,
            parent_entity_id=site_id,
            properties_json=props_json(
                model="Baseband 6630" if vendor == "ericsson" else "AirScale ASIQ",
            ),
        )
        acc.add_entity(bbu)
        n_bbu += 1

        # SITE hosts BBU
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=site_id,
                from_entity_type="SITE",
                relationship_type="HOSTS",
                to_entity_id=bbu_id,
                to_entity_type="BBU",
                domain="mobile_ran",
            )
        )
        n_rels += 1

        # BBU → GPS (timing)
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=bbu_id,
                from_entity_type="BBU",
                relationship_type="TIMING_FROM",
                to_entity_id=gps_id,
                to_entity_type="GPS_RECEIVER",
                domain="mobile_ran",
            )
        )
        n_rels += 1

        # Track logical nodes created at this site to avoid duplicates
        # (one eNB per site regardless of how many LTE cells, same for gNB)
        enodeb_id: str | None = None
        gnodeb_id: str | None = None
        gnodeb_du_id: str | None = None
        gnodeb_cu_cp_id: str | None = None
        gnodeb_cu_up_id: str | None = None

        for sector_idx in sorted(sector_cells.keys()):
            cells_in_sector = sector_cells[sector_idx]

            # --- ANTENNA_SYSTEM (one per sector) ---
            as_id = _make_id()
            as_entity = make_entity(
                entity_id=as_id,
                tenant_id=tenant,
                entity_type="ANTENNA_SYSTEM",
                name=f"ANT-SYS-{site_id[:8]}-S{sector_idx}",
                domain="mobile_ran",
                site_id=site_id,
                site_type=site_type,
                province=province,
                timezone=tz,
                vendor=vendor,
                sector_id=sector_idx,
                parent_entity_id=site_id,
            )
            acc.add_entity(as_entity)
            n_antenna_sys += 1

            acc.add_relationship(
                make_relationship(
                    tenant_id=tenant,
                    from_entity_id=site_id,
                    from_entity_type="SITE",
                    relationship_type="HOSTS",
                    to_entity_id=as_id,
                    to_entity_type="ANTENNA_SYSTEM",
                    domain="mobile_ran",
                )
            )
            n_rels += 1

            # --- ANTENNA (physical panel — one per sector, shared across bands) ---
            ant_id = _make_id()
            # Use first cell's azimuth/tilt/height for the sector
            ref_cell = cells_in_sector[0]
            ant = make_entity(
                entity_id=ant_id,
                tenant_id=tenant,
                entity_type="ANTENNA",
                name=f"ANT-{site_id[:8]}-S{sector_idx}",
                external_id=vendor_external_id(vendor, "ANTENNA", province, ant_id),
                domain="mobile_ran",
                site_id=site_id,
                site_type=site_type,
                province=province,
                timezone=tz,
                vendor=vendor,
                sector_id=sector_idx,
                azimuth_deg=ref_cell["azimuth_deg"],
                electrical_tilt_deg=ref_cell["electrical_tilt_deg"],
                antenna_height_m=ref_cell["antenna_height_m"],
                parent_entity_id=as_id,
                properties_json=props_json(
                    antenna_type="panel" if site_type != "in_building" else "omni",
                    gain_dbi=round(float(rng.uniform(15, 21)), 1),
                ),
            )
            acc.add_entity(ant)
            n_antenna += 1

            acc.add_relationship(
                make_relationship(
                    tenant_id=tenant,
                    from_entity_id=as_id,
                    from_entity_type="ANTENNA_SYSTEM",
                    relationship_type="HOSTS",
                    to_entity_id=ant_id,
                    to_entity_type="ANTENNA",
                    domain="mobile_ran",
                )
            )
            n_rels += 1

            # --- FEEDER_CABLE (one per sector) ---
            feeder_id = _make_id()
            feeder = make_entity(
                entity_id=feeder_id,
                tenant_id=tenant,
                entity_type="FEEDER_CABLE",
                name=f"FDR-{site_id[:8]}-S{sector_idx}",
                domain="mobile_ran",
                site_id=site_id,
                province=province,
                timezone=tz,
                parent_entity_id=as_id,
                properties_json=props_json(
                    type="fibre_jumper" if profile in ("dense_urban", "indoor") else "coaxial",
                    length_m=round(float(rng.uniform(5, 80)), 1),
                ),
            )
            acc.add_entity(feeder)
            n_feeder += 1

            acc.add_relationship(
                make_relationship(
                    tenant_id=tenant,
                    from_entity_id=as_id,
                    from_entity_type="ANTENNA_SYSTEM",
                    relationship_type="HOSTS",
                    to_entity_id=feeder_id,
                    to_entity_type="FEEDER_CABLE",
                    domain="mobile_ran",
                )
            )
            n_rels += 1

            # --- RRU (one per unique band in this sector) ---
            bands_in_sector = set()
            for cell in cells_in_sector:
                bands_in_sector.add(cell["band"])

            for band_name in sorted(bands_in_sector):
                rru_id = _make_id()
                rru = make_entity(
                    entity_id=rru_id,
                    tenant_id=tenant,
                    entity_type="RRU",
                    name=f"RRU-{site_id[:8]}-S{sector_idx}-{band_name}",
                    external_id=vendor_external_id(vendor, "RRU", province, rru_id),
                    domain="mobile_ran",
                    site_id=site_id,
                    site_type=site_type,
                    province=province,
                    timezone=tz,
                    vendor=vendor,
                    band=band_name,
                    sector_id=sector_idx,
                    parent_entity_id=as_id,
                    properties_json=props_json(
                        model="Radio 4422" if vendor == "ericsson" else "AirScale ABIA",
                        band=band_name,
                    ),
                )
                acc.add_entity(rru)
                n_rru += 1

                acc.add_relationship(
                    make_relationship(
                        tenant_id=tenant,
                        from_entity_id=as_id,
                        from_entity_type="ANTENNA_SYSTEM",
                        relationship_type="HOSTS",
                        to_entity_id=rru_id,
                        to_entity_type="RRU",
                        domain="mobile_ran",
                    )
                )
                n_rels += 1

            # --- LOGICAL NODES & CELLS ---
            for cell in cells_in_sector:
                cell_id = cell["cell_id"]
                cell_rat = cell["rat_type"]
                cell_band = cell["band"]
                cell_type = cell["entity_type"]  # LTE_CELL or NR_CELL
                is_scg = cell["is_nsa_scg_leg"]
                is_anchor = cell["is_nsa_anchor"]

                if cell_type == "LTE_CELL":
                    # Create eNodeB if not yet created for this site
                    if enodeb_id is None:
                        enodeb_id = _make_id()
                        enb = make_entity(
                            entity_id=enodeb_id,
                            tenant_id=tenant,
                            entity_type="ENODEB",
                            name=f"eNB-{site_id[:8]}",
                            external_id=vendor_external_id(vendor, "ENODEB", province, enodeb_id),
                            domain="mobile_ran",
                            geo_lat=geo_lat,
                            geo_lon=geo_lon,
                            site_id=site_id,
                            site_type=site_type,
                            deployment_profile=profile,
                            province=province,
                            timezone=tz,
                            vendor=vendor,
                            sla_tier=sla_tier,
                            parent_entity_id=bbu_id,
                            properties_json=props_json(
                                enb_id=int(rng.integers(100000, 999999)),
                                plmn="510-01",
                            ),
                        )
                        acc.add_entity(enb)
                        n_enodeb += 1

                        # BBU hosts ENODEB
                        acc.add_relationship(
                            make_relationship(
                                tenant_id=tenant,
                                from_entity_id=bbu_id,
                                from_entity_type="BBU",
                                relationship_type="HOSTS",
                                to_entity_id=enodeb_id,
                                to_entity_type="ENODEB",
                                domain="mobile_ran",
                            )
                        )
                        n_rels += 1

                    # Create LTE_CELL entity
                    cell_entity = make_entity(
                        entity_id=cell_id,
                        tenant_id=tenant,
                        entity_type="LTE_CELL",
                        name=cell["name"],
                        external_id=cell["external_id"],
                        domain="mobile_ran",
                        geo_lat=cell["geo_lat"],
                        geo_lon=cell["geo_lon"],
                        site_id=site_id,
                        site_type=site_type,
                        deployment_profile=profile,
                        province=province,
                        timezone=tz,
                        vendor=vendor,
                        sla_tier=sla_tier,
                        rat_type=cell_rat,
                        band=cell_band,
                        bandwidth_mhz=cell["bandwidth_mhz"],
                        max_tx_power_dbm=cell["max_tx_power_dbm"],
                        max_prbs=cell["max_prbs"],
                        frequency_mhz=cell["frequency_mhz"],
                        sector_id=cell["sector_id"],
                        azimuth_deg=cell["azimuth_deg"],
                        electrical_tilt_deg=cell["electrical_tilt_deg"],
                        antenna_height_m=cell["antenna_height_m"],
                        inter_site_distance_m=cell["inter_site_distance_m"],
                        revenue_weight=revenue,
                        is_nsa_anchor=is_anchor,
                        nsa_anchor_cell_id=cell.get("nsa_anchor_cell_id"),
                        parent_entity_id=enodeb_id,
                    )
                    acc.add_entity(cell_entity)
                    n_lte_cell += 1

                    # ENODEB hosts LTE_CELL
                    acc.add_relationship(
                        make_relationship(
                            tenant_id=tenant,
                            from_entity_id=enodeb_id,
                            from_entity_type="ENODEB",
                            relationship_type="HOSTS",
                            to_entity_id=cell_id,
                            to_entity_type="LTE_CELL",
                            domain="mobile_ran",
                        )
                    )
                    n_rels += 1

                elif cell_type == "NR_CELL":
                    # Create gNodeB and CU/DU split if not yet created for this site
                    if gnodeb_id is None:
                        gnodeb_id = _make_id()
                        gnb = make_entity(
                            entity_id=gnodeb_id,
                            tenant_id=tenant,
                            entity_type="GNODEB",
                            name=f"gNB-{site_id[:8]}",
                            external_id=vendor_external_id(vendor, "GNODEB", province, gnodeb_id),
                            domain="mobile_ran",
                            geo_lat=geo_lat,
                            geo_lon=geo_lon,
                            site_id=site_id,
                            site_type=site_type,
                            deployment_profile=profile,
                            province=province,
                            timezone=tz,
                            vendor=vendor,
                            sla_tier=sla_tier,
                            parent_entity_id=bbu_id,
                            properties_json=props_json(
                                gnb_id=int(rng.integers(1000000, 9999999)),
                                plmn="510-01",
                                cu_du_split=(cell_rat == "NR_SA"),
                            ),
                        )
                        acc.add_entity(gnb)
                        n_gnodeb += 1

                        # BBU hosts GNODEB
                        acc.add_relationship(
                            make_relationship(
                                tenant_id=tenant,
                                from_entity_id=bbu_id,
                                from_entity_type="BBU",
                                relationship_type="HOSTS",
                                to_entity_id=gnodeb_id,
                                to_entity_type="GNODEB",
                                domain="mobile_ran",
                            )
                        )
                        n_rels += 1

                        # CU/DU split entities
                        gnodeb_du_id = _make_id()
                        du = make_entity(
                            entity_id=gnodeb_du_id,
                            tenant_id=tenant,
                            entity_type="GNODEB_DU",
                            name=f"gNB-DU-{site_id[:8]}",
                            external_id=vendor_external_id(vendor, "GNB_DU", province, gnodeb_du_id),
                            domain="mobile_ran",
                            site_id=site_id,
                            site_type=site_type,
                            province=province,
                            timezone=tz,
                            vendor=vendor,
                            parent_entity_id=gnodeb_id,
                        )
                        acc.add_entity(du)
                        n_gnodeb_du += 1

                        acc.add_relationship(
                            make_relationship(
                                tenant_id=tenant,
                                from_entity_id=gnodeb_id,
                                from_entity_type="GNODEB",
                                relationship_type="HOSTS",
                                to_entity_id=gnodeb_du_id,
                                to_entity_type="GNODEB_DU",
                                domain="mobile_ran",
                            )
                        )
                        n_rels += 1

                        gnodeb_cu_cp_id = _make_id()
                        cu_cp = make_entity(
                            entity_id=gnodeb_cu_cp_id,
                            tenant_id=tenant,
                            entity_type="GNODEB_CU_CP",
                            name=f"gNB-CU-CP-{site_id[:8]}",
                            external_id=vendor_external_id(vendor, "GNB_CUCP", province, gnodeb_cu_cp_id),
                            domain="mobile_ran",
                            site_id=site_id,
                            site_type=site_type,
                            province=province,
                            timezone=tz,
                            vendor=vendor,
                            parent_entity_id=gnodeb_id,
                        )
                        acc.add_entity(cu_cp)
                        n_gnodeb_cu_cp += 1

                        acc.add_relationship(
                            make_relationship(
                                tenant_id=tenant,
                                from_entity_id=gnodeb_id,
                                from_entity_type="GNODEB",
                                relationship_type="HOSTS",
                                to_entity_id=gnodeb_cu_cp_id,
                                to_entity_type="GNODEB_CU_CP",
                                domain="mobile_ran",
                            )
                        )
                        n_rels += 1

                        gnodeb_cu_up_id = _make_id()
                        cu_up = make_entity(
                            entity_id=gnodeb_cu_up_id,
                            tenant_id=tenant,
                            entity_type="GNODEB_CU_UP",
                            name=f"gNB-CU-UP-{site_id[:8]}",
                            external_id=vendor_external_id(vendor, "GNB_CUUP", province, gnodeb_cu_up_id),
                            domain="mobile_ran",
                            site_id=site_id,
                            site_type=site_type,
                            province=province,
                            timezone=tz,
                            vendor=vendor,
                            parent_entity_id=gnodeb_id,
                        )
                        acc.add_entity(cu_up)
                        n_gnodeb_cu_up += 1

                        acc.add_relationship(
                            make_relationship(
                                tenant_id=tenant,
                                from_entity_id=gnodeb_id,
                                from_entity_type="GNODEB",
                                relationship_type="HOSTS",
                                to_entity_id=gnodeb_cu_up_id,
                                to_entity_type="GNODEB_CU_UP",
                                domain="mobile_ran",
                            )
                        )
                        n_rels += 1

                    # Create NR_CELL entity
                    cell_entity = make_entity(
                        entity_id=cell_id,
                        tenant_id=tenant,
                        entity_type="NR_CELL",
                        name=cell["name"],
                        external_id=cell["external_id"],
                        domain="mobile_ran",
                        geo_lat=cell["geo_lat"],
                        geo_lon=cell["geo_lon"],
                        site_id=site_id,
                        site_type=site_type,
                        deployment_profile=profile,
                        province=province,
                        timezone=tz,
                        vendor=vendor,
                        sla_tier=sla_tier,
                        rat_type=cell_rat,
                        band=cell_band,
                        bandwidth_mhz=cell["bandwidth_mhz"],
                        max_tx_power_dbm=cell["max_tx_power_dbm"],
                        max_prbs=cell["max_prbs"],
                        frequency_mhz=cell["frequency_mhz"],
                        sector_id=cell["sector_id"],
                        azimuth_deg=cell["azimuth_deg"],
                        electrical_tilt_deg=cell["electrical_tilt_deg"],
                        antenna_height_m=cell["antenna_height_m"],
                        inter_site_distance_m=cell["inter_site_distance_m"],
                        revenue_weight=revenue,
                        is_nsa_anchor=False,
                        nsa_anchor_cell_id=cell.get("nsa_anchor_cell_id"),
                        parent_entity_id=gnodeb_du_id,
                    )
                    acc.add_entity(cell_entity)
                    n_nr_cell += 1

                    # GNODEB_DU hosts NR_CELL (via DU, not directly from gNB)
                    acc.add_relationship(
                        make_relationship(
                            tenant_id=tenant,
                            from_entity_id=gnodeb_du_id,
                            from_entity_type="GNODEB_DU",
                            relationship_type="HOSTS",
                            to_entity_id=cell_id,
                            to_entity_type="NR_CELL",
                            domain="mobile_ran",
                        )
                    )
                    n_rels += 1

                    # For NSA SCG legs: create ANCHORS relationship
                    if is_scg and cell.get("nsa_anchor_cell_id"):
                        acc.add_relationship(
                            make_relationship(
                                tenant_id=tenant,
                                from_entity_id=cell["nsa_anchor_cell_id"],
                                from_entity_type="LTE_CELL",
                                relationship_type="ANCHORS",
                                to_entity_id=cell_id,
                                to_entity_type="NR_CELL",
                                domain="mobile_ran",
                                properties_json=props_json(mode="EN-DC", anchor_band=cell.get("band")),
                            )
                        )
                        n_rels += 1

    # ── Summary ────────────────────────────────────────────────
    console.print(f"    [bold green]Mobile RAN topology built:[/bold green]")
    entity_counts = {
        "SITE": n_sites,
        "CABINET": n_cabinets,
        "POWER_SUPPLY": n_power,
        "BATTERY_BANK": n_battery,
        "MAINS_CONNECTION": n_mains,
        "CLIMATE_CONTROL": n_climate,
        "TRANSMISSION_EQUIPMENT": n_transmission,
        "ANTENNA_SYSTEM": n_antenna_sys,
        "ANTENNA": n_antenna,
        "RRU": n_rru,
        "FEEDER_CABLE": n_feeder,
        "BBU": n_bbu,
        "ENODEB": n_enodeb,
        "GNODEB": n_gnodeb,
        "GNODEB_DU": n_gnodeb_du,
        "GNODEB_CU_CP": n_gnodeb_cu_cp,
        "GNODEB_CU_UP": n_gnodeb_cu_up,
        "GPS_RECEIVER": n_gps,
        "LTE_CELL": n_lte_cell,
        "NR_CELL": n_nr_cell,
    }
    total_entities = sum(entity_counts.values())
    console.print(f"      Entities: {total_entities:,}")
    for etype, count in entity_counts.items():
        if count > 0:
            console.print(f"        {etype}: {count:,}")
    console.print(f"      Relationships: {n_rels:,}")
