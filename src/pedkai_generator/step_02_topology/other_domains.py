"""
Non-RAN topology domain builders.

Builds entities and relationships for all domains outside Mobile RAN:
  - Transport (IP/MPLS/fibre backbone) — ~50K entities, ~150K relationships
  - Fixed Broadband Access (FTTP/FTTC/enterprise) — ~120K entities, ~350K relationships
  - Core Network (EPC/5GC/IMS/broadband control) — ~200 entities, ~2K relationships
  - Logical/Service (slices, tracking areas, VPNs, LSPs) — ~10K entities, ~50K relationships
  - Power/Environment (already partially created in RAN; augment for exchanges) — ~40K entities, ~60K relationships

All builders append to the shared TopologyAccumulator.
"""

from __future__ import annotations

import json
import math
import uuid
from typing import Any

import numpy as np
from rich.console import Console

from pedkai_generator.config.settings import DeploymentProfile, GeneratorConfig
from pedkai_generator.step_02_topology.builders import (
    TopologyAccumulator,
    entity_name,
    make_entity,
    make_relationship,
    offset_lat_lon,
    props_json,
    vendor_external_id,
)

console = Console()


def _id() -> str:
    return str(uuid.uuid4())


# ============================================================================
# 1. TRANSPORT DOMAIN
# ============================================================================


def build_transport_topology(
    config: GeneratorConfig,
    acc: TopologyAccumulator,
    rng: np.random.Generator,
) -> None:
    """
    Build the transport/backbone topology.

    Hierarchy:
      P_ROUTER (core) ← PE_ROUTER (edge) ← AGG_SWITCH (metro) ← ACCESS_SWITCH (site)
      Overlaid with: L3VPN, L2VPN, LSP, PSEUDOWIRE
      Physical: FIBRE_CABLE, MICROWAVE_LINK (backhaul)

    Access switches are placed at mobile sites and exchange buildings.
    The transport topology provides the convergence point through which
    mobile backhaul, broadband aggregation, and enterprise circuits all flow.
    """
    tenant = config.tenant_id
    sites = acc.get_entities_by_type("SITE")
    exchanges = acc.get_entities_by_type("EXCHANGE_BUILDING")  # may be empty if called before fixed

    # --- Scale parameters ---
    # Target: ~50K entities, ~150K relationships
    n_p_routers = 30  # Core routers (national backbone)
    n_pe_routers = 250  # Provider edge routers (regional PoPs)
    n_agg_switches = 2000  # Metro aggregation switches
    n_route_reflectors = 8  # BGP route reflectors
    n_bng = 40  # Broadband Network Gateways
    n_cgnat = 20
    n_firewall = 15
    n_cdn = 25
    n_dwdm = 60
    n_l3vpn = 3000  # VPN instances
    n_l2vpn = 1500
    n_lsp = 5000  # Label Switched Paths
    n_pseudowire = 2000

    # ── P_ROUTER (core backbone) ──────────────────────────────
    p_router_ids: list[str] = []
    # Spread across major provinces
    major_provinces = [
        ("DKI Jakarta", -6.20, 106.85),
        ("Jawa Barat", -6.90, 107.60),
        ("Jawa Timur", -7.55, 112.75),
        ("Jawa Tengah", -7.15, 110.00),
        ("Bali", -8.40, 115.20),
        ("Sumatera Utara", 2.50, 99.00),
        ("Sulawesi Selatan", -3.70, 120.00),
        ("Kalimantan Timur", 0.50, 116.50),
    ]

    for i in range(n_p_routers):
        pid = _id()
        p_router_ids.append(pid)
        prov_idx = i % len(major_provinces)
        prov_name, lat, lon = major_provinces[prov_idx]
        lat_j, lon_j = offset_lat_lon(rng, lat, lon, spread_km=5.0)
        vendor = "ericsson" if rng.random() < config.ericsson_fraction else "nokia"

        acc.add_entity(
            make_entity(
                entity_id=pid,
                tenant_id=tenant,
                entity_type="P_ROUTER",
                name=entity_name("P-RTR", i, prov_name),
                external_id=vendor_external_id(vendor, "P_ROUTER", prov_name, pid),
                domain="transport",
                geo_lat=lat_j,
                geo_lon=lon_j,
                province=prov_name,
                vendor=vendor,
                properties_json=props_json(
                    role="core_backbone",
                    capacity_gbps=int(rng.choice([400, 800, 1600])),
                    model="NCS 5500" if vendor == "ericsson" else "7750 SR-14s",
                ),
            )
        )

    # P_ROUTER ↔ P_ROUTER mesh (partial mesh — each connects to 3-5 others)
    p_rels = 0
    for i, pid in enumerate(p_router_ids):
        n_peers = int(rng.integers(3, min(6, n_p_routers)))
        candidates = [j for j in range(n_p_routers) if j != i]
        peers = rng.choice(candidates, size=min(n_peers, len(candidates)), replace=False)
        for j in peers:
            acc.add_relationship(
                make_relationship(
                    tenant_id=tenant,
                    from_entity_id=pid,
                    from_entity_type="P_ROUTER",
                    relationship_type="PEERS_WITH",
                    to_entity_id=p_router_ids[j],
                    to_entity_type="P_ROUTER",
                    domain="transport",
                    properties_json=props_json(protocol="IS-IS", metric=int(rng.integers(10, 100))),
                )
            )
            p_rels += 1

    # ── ROUTE_REFLECTOR ──────────────────────────────────────
    rr_ids: list[str] = []
    for i in range(n_route_reflectors):
        rr_id = _id()
        rr_ids.append(rr_id)
        prov_idx = i % len(major_provinces)
        prov_name, lat, lon = major_provinces[prov_idx]
        lat_j, lon_j = offset_lat_lon(rng, lat, lon, spread_km=2.0)
        acc.add_entity(
            make_entity(
                entity_id=rr_id,
                tenant_id=tenant,
                entity_type="ROUTE_REFLECTOR",
                name=entity_name("RR", i, prov_name),
                domain="transport",
                geo_lat=lat_j,
                geo_lon=lon_j,
                province=prov_name,
                properties_json=props_json(role="bgp_route_reflector", cluster_id=i),
            )
        )

    # ── PE_ROUTER (provider edge) ────────────────────────────
    pe_router_ids: list[str] = []
    pe_router_provinces: list[str] = []
    for i in range(n_pe_routers):
        pe_id = _id()
        pe_router_ids.append(pe_id)
        prov_idx = i % len(major_provinces)
        prov_name, lat, lon = major_provinces[prov_idx]
        pe_router_provinces.append(prov_name)
        lat_j, lon_j = offset_lat_lon(rng, lat, lon, spread_km=15.0)
        vendor = "ericsson" if rng.random() < config.ericsson_fraction else "nokia"

        acc.add_entity(
            make_entity(
                entity_id=pe_id,
                tenant_id=tenant,
                entity_type="PE_ROUTER",
                name=entity_name("PE-RTR", i, prov_name),
                external_id=vendor_external_id(vendor, "PE_RTR", prov_name, pe_id),
                domain="transport",
                geo_lat=lat_j,
                geo_lon=lon_j,
                province=prov_name,
                vendor=vendor,
                properties_json=props_json(
                    role="provider_edge",
                    capacity_gbps=int(rng.choice([100, 200, 400])),
                ),
            )
        )

        # PE → P_ROUTER uplink (each PE connects to 1-2 P routers)
        n_up = int(rng.integers(1, 3))
        p_peers = rng.choice(len(p_router_ids), size=min(n_up, len(p_router_ids)), replace=False)
        for j in p_peers:
            acc.add_relationship(
                make_relationship(
                    tenant_id=tenant,
                    from_entity_id=pe_id,
                    from_entity_type="PE_ROUTER",
                    relationship_type="UPLINKS_TO",
                    to_entity_id=p_router_ids[j],
                    to_entity_type="P_ROUTER",
                    domain="transport",
                    properties_json=props_json(interface_speed_gbps=100),
                )
            )

        # PE → ROUTE_REFLECTOR (BGP peering)
        rr_peer = rr_ids[i % len(rr_ids)]
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=pe_id,
                from_entity_type="PE_ROUTER",
                relationship_type="PEERS_WITH",
                to_entity_id=rr_peer,
                to_entity_type="ROUTE_REFLECTOR",
                domain="transport",
                properties_json=props_json(protocol="iBGP"),
            )
        )

    # ── AGGREGATION_SWITCH ───────────────────────────────────
    agg_switch_ids: list[str] = []
    for i in range(n_agg_switches):
        agg_id = _id()
        agg_switch_ids.append(agg_id)
        prov_idx = i % len(major_provinces)
        prov_name, lat, lon = major_provinces[prov_idx]
        lat_j, lon_j = offset_lat_lon(rng, lat, lon, spread_km=20.0)
        vendor = "ericsson" if rng.random() < config.ericsson_fraction else "nokia"

        acc.add_entity(
            make_entity(
                entity_id=agg_id,
                tenant_id=tenant,
                entity_type="AGGREGATION_SWITCH",
                name=entity_name("AGG-SW", i, prov_name),
                external_id=vendor_external_id(vendor, "AGG_SW", prov_name, agg_id),
                domain="transport",
                geo_lat=lat_j,
                geo_lon=lon_j,
                province=prov_name,
                vendor=vendor,
                properties_json=props_json(
                    capacity_gbps=int(rng.choice([10, 40, 100])),
                    ports=int(rng.choice([24, 48, 96])),
                ),
            )
        )

        # AGG → PE_ROUTER uplink (each agg connects to 1-2 PEs)
        n_up = int(rng.integers(1, 3))
        pe_peers = rng.choice(len(pe_router_ids), size=min(n_up, len(pe_router_ids)), replace=False)
        for j in pe_peers:
            acc.add_relationship(
                make_relationship(
                    tenant_id=tenant,
                    from_entity_id=agg_id,
                    from_entity_type="AGGREGATION_SWITCH",
                    relationship_type="UPLINKS_TO",
                    to_entity_id=pe_router_ids[j],
                    to_entity_type="PE_ROUTER",
                    domain="transport",
                )
            )

    # ── ACCESS_SWITCH (at mobile sites — one per site) ───────
    # Only place at a subset of sites (larger sites get their own switch;
    # smaller streetworks/in-building may share via aggregation)
    access_switch_ids: list[str] = []
    access_switch_by_site: dict[str, str] = {}  # site_id → access_switch_id

    for site in sites:
        site_type = site.get("site_type", "")
        # Streetworks small cells often don't have their own access switch
        if site_type == "streetworks" and rng.random() < 0.6:
            continue
        if site_type == "in_building" and rng.random() < 0.3:
            continue

        sw_id = _id()
        access_switch_ids.append(sw_id)
        access_switch_by_site[site["entity_id"]] = sw_id
        vendor = site.get("vendor", "nokia")
        province = site.get("province", "")

        acc.add_entity(
            make_entity(
                entity_id=sw_id,
                tenant_id=tenant,
                entity_type="ACCESS_SWITCH",
                name=f"ACC-SW-{site['entity_id'][:8]}",
                external_id=vendor_external_id(vendor, "ACC_SW", province, sw_id),
                domain="transport",
                geo_lat=site.get("geo_lat"),
                geo_lon=site.get("geo_lon"),
                site_id=site["entity_id"],
                site_type=site_type,
                province=province,
                vendor=vendor,
            )
        )

        # ACCESS_SWITCH → AGGREGATION_SWITCH uplink
        agg_idx = rng.integers(0, len(agg_switch_ids))
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=sw_id,
                from_entity_type="ACCESS_SWITCH",
                relationship_type="UPLINKS_TO",
                to_entity_id=agg_switch_ids[agg_idx],
                to_entity_type="AGGREGATION_SWITCH",
                domain="transport",
            )
        )

        # BACKHAULS relationship: transport → site
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=sw_id,
                from_entity_type="ACCESS_SWITCH",
                relationship_type="BACKHAULS",
                to_entity_id=site["entity_id"],
                to_entity_type="SITE",
                domain="transport",
            )
        )

    # ── MICROWAVE_LINK (for rural/deep rural sites without fibre) ─
    mw_count = 0
    for site in sites:
        profile = site.get("deployment_profile", "")
        site_id = site["entity_id"]
        if profile not in ("rural", "deep_rural"):
            continue
        # ~80% of rural sites get microwave backhaul
        if rng.random() > 0.80:
            continue
        mw_id = _id()
        province = site.get("province", "")
        vendor = site.get("vendor", "nokia")
        lat_j, lon_j = offset_lat_lon(rng, site.get("geo_lat", 0), site.get("geo_lon", 0), spread_km=0.1)

        acc.add_entity(
            make_entity(
                entity_id=mw_id,
                tenant_id=tenant,
                entity_type="MICROWAVE_LINK",
                name=f"MW-{site_id[:8]}",
                external_id=vendor_external_id(vendor, "MW_LINK", province, mw_id),
                domain="transport",
                geo_lat=lat_j,
                geo_lon=lon_j,
                site_id=site_id,
                province=province,
                vendor=vendor,
                properties_json=props_json(
                    capacity_mbps=int(rng.choice([500, 1000, 2000, 4000])),
                    frequency_ghz=round(float(rng.choice([6, 11, 18, 23, 38, 80])), 1),
                    distance_km=round(float(rng.uniform(0.5, 15.0)), 1),
                ),
            )
        )
        mw_count += 1

        # MW_LINK backhauls SITE
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=mw_id,
                from_entity_type="MICROWAVE_LINK",
                relationship_type="BACKHAULS",
                to_entity_id=site_id,
                to_entity_type="SITE",
                domain="transport",
            )
        )

    # ── FIBRE_CABLE (backbone segments between P/PE routers) ──
    fibre_count = 0
    all_backbone = p_router_ids + pe_router_ids[:100]  # Sample
    for i in range(min(len(all_backbone) - 1, 300)):
        fc_id = _id()
        acc.add_entity(
            make_entity(
                entity_id=fc_id,
                tenant_id=tenant,
                entity_type="FIBRE_CABLE",
                name=entity_name("FIBRE", i),
                domain="transport",
                properties_json=props_json(
                    fibre_pairs=int(rng.choice([24, 48, 96, 144, 288])),
                    length_km=round(float(rng.uniform(5, 500)), 1),
                ),
            )
        )
        fibre_count += 1

        # FIBRE_CABLE connects two backbone nodes
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=all_backbone[i],
                from_entity_type="P_ROUTER" if i < len(p_router_ids) else "PE_ROUTER",
                relationship_type="CONNECTS_FIBRE",
                to_entity_id=all_backbone[i + 1],
                to_entity_type="P_ROUTER" if (i + 1) < len(p_router_ids) else "PE_ROUTER",
                domain="transport",
                properties_json=props_json(fibre_cable_id=fc_id),
            )
        )

    # ── DWDM_SYSTEM ──────────────────────────────────────────
    dwdm_ids: list[str] = []
    for i in range(n_dwdm):
        did = _id()
        dwdm_ids.append(did)
        prov_idx = i % len(major_provinces)
        prov_name = major_provinces[prov_idx][0]
        acc.add_entity(
            make_entity(
                entity_id=did,
                tenant_id=tenant,
                entity_type="DWDM_SYSTEM",
                name=entity_name("DWDM", i, prov_name),
                domain="transport",
                province=prov_name,
                properties_json=props_json(
                    wavelengths=int(rng.choice([40, 80, 96])),
                    capacity_per_lambda_gbps=int(rng.choice([100, 200, 400])),
                ),
            )
        )

    # ── OPTICAL_CHANNEL (2-4 per DWDM) ──────────────────────
    for did in dwdm_ids:
        n_ch = int(rng.integers(2, 5))
        for ch in range(n_ch):
            oc_id = _id()
            acc.add_entity(
                make_entity(
                    entity_id=oc_id,
                    tenant_id=tenant,
                    entity_type="OPTICAL_CHANNEL",
                    name=f"OCH-{did[:6]}-{ch}",
                    domain="transport",
                    parent_entity_id=did,
                    properties_json=props_json(wavelength_nm=round(1550.0 + ch * 0.8, 1)),
                )
            )
            acc.add_relationship(
                make_relationship(
                    tenant_id=tenant,
                    from_entity_id=did,
                    from_entity_type="DWDM_SYSTEM",
                    relationship_type="HOSTS",
                    to_entity_id=oc_id,
                    to_entity_type="OPTICAL_CHANNEL",
                    domain="transport",
                )
            )

    # ── BNG / CGNAT / FIREWALL / CDN ─────────────────────────
    bng_ids: list[str] = []
    for i in range(n_bng):
        bid = _id()
        bng_ids.append(bid)
        prov_idx = i % len(major_provinces)
        prov_name = major_provinces[prov_idx][0]
        acc.add_entity(
            make_entity(
                entity_id=bid,
                tenant_id=tenant,
                entity_type="BNG",
                name=entity_name("BNG", i, prov_name),
                domain="transport",
                province=prov_name,
                properties_json=props_json(
                    subscriber_capacity=int(rng.integers(50000, 200000)),
                    throughput_gbps=int(rng.choice([100, 200, 400])),
                ),
            )
        )
        # BNG → PE_ROUTER
        pe_idx = i % len(pe_router_ids)
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=bid,
                from_entity_type="BNG",
                relationship_type="UPLINKS_TO",
                to_entity_id=pe_router_ids[pe_idx],
                to_entity_type="PE_ROUTER",
                domain="transport",
            )
        )

    for i in range(n_cgnat):
        cid = _id()
        acc.add_entity(
            make_entity(
                entity_id=cid,
                tenant_id=tenant,
                entity_type="CGNAT",
                name=entity_name("CGNAT", i),
                domain="transport",
                properties_json=props_json(port_blocks=int(rng.integers(500, 2000))),
            )
        )

    for i in range(n_firewall):
        fid = _id()
        acc.add_entity(
            make_entity(
                entity_id=fid,
                tenant_id=tenant,
                entity_type="FIREWALL",
                name=entity_name("FW", i),
                domain="transport",
            )
        )

    for i in range(n_cdn):
        cid = _id()
        prov_idx = i % len(major_provinces)
        prov_name = major_provinces[prov_idx][0]
        acc.add_entity(
            make_entity(
                entity_id=cid,
                tenant_id=tenant,
                entity_type="CDN_NODE",
                name=entity_name("CDN", i, prov_name),
                domain="transport",
                province=prov_name,
                properties_json=props_json(cache_tb=int(rng.choice([50, 100, 200]))),
            )
        )

    # ── L3VPN ────────────────────────────────────────────────
    l3vpn_ids: list[str] = []
    vpn_types = ["MOBILE_BACKHAUL_VRF", "BROADBAND_VRF", "ENTERPRISE_VPN", "MANAGEMENT_VRF"]
    for i in range(n_l3vpn):
        vid = _id()
        l3vpn_ids.append(vid)
        vtype = vpn_types[i % len(vpn_types)]
        acc.add_entity(
            make_entity(
                entity_id=vid,
                tenant_id=tenant,
                entity_type="L3VPN",
                name=f"L3VPN-{vtype[:8]}-{i:05d}",
                domain="transport",
                properties_json=props_json(vpn_type=vtype, rd=f"65000:{i}"),
            )
        )
        # VPN terminates on 1-3 PE routers
        n_pe = int(rng.integers(1, 4))
        pe_sample = rng.choice(len(pe_router_ids), size=min(n_pe, len(pe_router_ids)), replace=False)
        for j in pe_sample:
            acc.add_relationship(
                make_relationship(
                    tenant_id=tenant,
                    from_entity_id=pe_router_ids[j],
                    from_entity_type="PE_ROUTER",
                    relationship_type="MEMBER_OF_VRF",
                    to_entity_id=vid,
                    to_entity_type="L3VPN",
                    domain="transport",
                )
            )

    # ── L2VPN ────────────────────────────────────────────────
    l2vpn_ids: list[str] = []
    l2_types = ["E_LINE", "E_LAN", "E_TREE"]
    for i in range(n_l2vpn):
        lid = _id()
        l2vpn_ids.append(lid)
        lt = l2_types[i % len(l2_types)]
        acc.add_entity(
            make_entity(
                entity_id=lid,
                tenant_id=tenant,
                entity_type="L2VPN",
                name=f"L2VPN-{lt}-{i:05d}",
                domain="transport",
                properties_json=props_json(service_type=lt),
            )
        )

    # ── LSP ──────────────────────────────────────────────────
    lsp_ids: list[str] = []
    for i in range(n_lsp):
        lid = _id()
        lsp_ids.append(lid)
        acc.add_entity(
            make_entity(
                entity_id=lid,
                tenant_id=tenant,
                entity_type="LSP",
                name=f"LSP-{i:06d}",
                domain="transport",
                properties_json=props_json(
                    bandwidth_mbps=int(rng.choice([1000, 5000, 10000, 40000])),
                    priority=int(rng.integers(0, 8)),
                ),
            )
        )
        # LSP routes through 2-4 P_ROUTERs
        n_hop = int(rng.integers(2, 5))
        hops = rng.choice(len(p_router_ids), size=min(n_hop, len(p_router_ids)), replace=False)
        for j in hops:
            acc.add_relationship(
                make_relationship(
                    tenant_id=tenant,
                    from_entity_id=lid,
                    from_entity_type="LSP",
                    relationship_type="ROUTES_THROUGH",
                    to_entity_id=p_router_ids[j],
                    to_entity_type="P_ROUTER",
                    domain="transport",
                )
            )

        # L3VPN → LSP (carried over)
        if i < len(l3vpn_ids):
            acc.add_relationship(
                make_relationship(
                    tenant_id=tenant,
                    from_entity_id=l3vpn_ids[i % len(l3vpn_ids)],
                    from_entity_type="L3VPN",
                    relationship_type="CARRIED_OVER",
                    to_entity_id=lid,
                    to_entity_type="LSP",
                    domain="transport",
                )
            )

    # ── PSEUDOWIRE ───────────────────────────────────────────
    for i in range(n_pseudowire):
        pw_id = _id()
        acc.add_entity(
            make_entity(
                entity_id=pw_id,
                tenant_id=tenant,
                entity_type="PSEUDOWIRE",
                name=f"PW-{i:06d}",
                domain="transport",
            )
        )
        if i < len(l2vpn_ids):
            acc.add_relationship(
                make_relationship(
                    tenant_id=tenant,
                    from_entity_id=l2vpn_ids[i % len(l2vpn_ids)],
                    from_entity_type="L2VPN",
                    relationship_type="CARRIED_OVER",
                    to_entity_id=pw_id,
                    to_entity_type="PSEUDOWIRE",
                    domain="transport",
                )
            )

    e_count = acc.entity_count_by_domain().get("transport", 0)
    r_count = acc.relationship_count_by_domain().get("transport", 0)
    console.print(
        f"    [bold green]Transport topology built:[/bold green] {e_count:,} entities, {r_count:,} relationships"
    )

    # Store cross-domain references on the accumulator for other builders
    acc._transport_refs = {
        "pe_router_ids": pe_router_ids,
        "agg_switch_ids": agg_switch_ids,
        "access_switch_by_site": access_switch_by_site,
        "bng_ids": bng_ids,
        "l3vpn_ids": l3vpn_ids,
        "l2vpn_ids": l2vpn_ids,
        "lsp_ids": lsp_ids,
        "p_router_ids": p_router_ids,
    }


# ============================================================================
# 2. FIXED BROADBAND ACCESS DOMAIN
# ============================================================================


def build_fixed_broadband_topology(
    config: GeneratorConfig,
    acc: TopologyAccumulator,
    rng: np.random.Generator,
) -> None:
    """
    Build Fixed Broadband Access topology.

    FTTP hierarchy:
      EXCHANGE_BUILDING → OLT → PON_PORT → SPLITTER → ONT → RESIDENTIAL_SERVICE

    Enterprise Ethernet:
      NTE → ETHERNET_CIRCUIT → ENTERPRISE_SERVICE

    Target: ~120K entities, ~350K relationships
    """
    tenant = config.tenant_id
    sites = acc.get_entities_by_type("SITE")

    # --- Scale ---
    n_exchanges = 500  # Exchange buildings (host OLTs)
    olts_per_exchange = 4  # Average OLTs per exchange
    pon_ports_per_olt = 16  # PON ports per OLT
    splitter_ratio = 32  # 1:32 split
    # ONTs: sampled — not all splitter ports are lit
    ont_fill_rate = 0.5  # 50% of splitter ports have active ONTs
    n_nte = 2000  # Enterprise NTEs
    n_ethernet_circuits = 2000
    n_enterprise_services = 2000

    # Use a subset of sites for exchange locations (dense/urban provinces)
    urban_sites = [
        s
        for s in sites
        if s.get("deployment_profile") in ("dense_urban", "urban") and s.get("site_type") in ("rooftop", "greenfield")
    ]
    # Deduplicate by picking one site per "area" (sample from urban sites)
    exchange_sites = []
    if len(urban_sites) >= n_exchanges:
        indices = rng.choice(len(urban_sites), size=n_exchanges, replace=False)
        exchange_sites = [urban_sites[i] for i in indices]
    else:
        exchange_sites = urban_sites[:n_exchanges]
        # Fill remainder with random sites
        remaining = n_exchanges - len(exchange_sites)
        if remaining > 0 and len(sites) > len(exchange_sites):
            extra_idx = rng.choice(len(sites), size=remaining, replace=True)
            exchange_sites.extend([sites[i] for i in extra_idx])

    exchange_ids: list[str] = []
    olt_ids: list[str] = []
    ont_ids: list[str] = []

    transport_refs = getattr(acc, "_transport_refs", {})
    agg_switch_ids = transport_refs.get("agg_switch_ids", [])
    bng_ids = transport_refs.get("bng_ids", [])

    for ex_idx, ref_site in enumerate(exchange_sites):
        # --- EXCHANGE_BUILDING ---
        ex_id = _id()
        exchange_ids.append(ex_id)
        lat = ref_site.get("geo_lat", 0) + float(rng.normal(0, 0.002))
        lon = ref_site.get("geo_lon", 0) + float(rng.normal(0, 0.002))
        province = ref_site.get("province", "")

        acc.add_entity(
            make_entity(
                entity_id=ex_id,
                tenant_id=tenant,
                entity_type="EXCHANGE_BUILDING",
                name=entity_name("EXCH", ex_idx, province),
                domain="fixed_access",
                geo_lat=round(lat, 6),
                geo_lon=round(lon, 6),
                province=province,
                timezone=ref_site.get("timezone"),
                properties_json=props_json(
                    floors=int(rng.integers(1, 5)),
                    has_backup_power=True,
                ),
            )
        )

        # Exchange → aggregation switch uplink (if transport built)
        if agg_switch_ids:
            agg_idx = ex_idx % len(agg_switch_ids)
            acc.add_relationship(
                make_relationship(
                    tenant_id=tenant,
                    from_entity_id=ex_id,
                    from_entity_type="EXCHANGE_BUILDING",
                    relationship_type="UPLINKS_TO",
                    to_entity_id=agg_switch_ids[agg_idx],
                    to_entity_type="AGGREGATION_SWITCH",
                    domain="cross_domain",
                )
            )

        # --- OLTs per exchange ---
        n_olts = max(1, int(rng.poisson(olts_per_exchange)))
        n_olts = min(n_olts, 8)  # Cap

        for olt_idx in range(n_olts):
            olt_id = _id()
            olt_ids.append(olt_id)
            vendor = "ericsson" if rng.random() < 0.3 else "nokia"  # Nokia dominates FTTP

            acc.add_entity(
                make_entity(
                    entity_id=olt_id,
                    tenant_id=tenant,
                    entity_type="OLT",
                    name=f"OLT-{ex_id[:6]}-{olt_idx}",
                    external_id=vendor_external_id(vendor, "OLT", province, olt_id),
                    domain="fixed_access",
                    geo_lat=round(lat, 6),
                    geo_lon=round(lon, 6),
                    site_id=ex_id,
                    province=province,
                    vendor=vendor,
                    properties_json=props_json(
                        model="ISAM 7360" if vendor == "nokia" else "6274",
                        pon_type=rng.choice(["GPON", "XGS-PON"]),
                        total_ports=pon_ports_per_olt,
                    ),
                )
            )

            # EXCHANGE hosts OLT
            acc.add_relationship(
                make_relationship(
                    tenant_id=tenant,
                    from_entity_id=ex_id,
                    from_entity_type="EXCHANGE_BUILDING",
                    relationship_type="HOSTS",
                    to_entity_id=olt_id,
                    to_entity_type="OLT",
                    domain="fixed_access",
                )
            )

            # OLT → BNG uplink
            if bng_ids:
                bng_idx = (ex_idx * n_olts + olt_idx) % len(bng_ids)
                acc.add_relationship(
                    make_relationship(
                        tenant_id=tenant,
                        from_entity_id=olt_id,
                        from_entity_type="OLT",
                        relationship_type="UPLINKS_TO",
                        to_entity_id=bng_ids[bng_idx],
                        to_entity_type="BNG",
                        domain="cross_domain",
                    )
                )

            # --- PON_PORT per OLT ---
            actual_pon_ports = max(4, int(rng.poisson(pon_ports_per_olt)))
            actual_pon_ports = min(actual_pon_ports, 32)

            for pp_idx in range(actual_pon_ports):
                pp_id = _id()
                acc.add_entity(
                    make_entity(
                        entity_id=pp_id,
                        tenant_id=tenant,
                        entity_type="PON_PORT",
                        name=f"PON-{olt_id[:6]}-{pp_idx}",
                        domain="fixed_access",
                        site_id=ex_id,
                        parent_entity_id=olt_id,
                    )
                )

                acc.add_relationship(
                    make_relationship(
                        tenant_id=tenant,
                        from_entity_id=olt_id,
                        from_entity_type="OLT",
                        relationship_type="HOSTS",
                        to_entity_id=pp_id,
                        to_entity_type="PON_PORT",
                        domain="fixed_access",
                    )
                )

                # --- SPLITTER per PON_PORT ---
                sp_id = _id()
                acc.add_entity(
                    make_entity(
                        entity_id=sp_id,
                        tenant_id=tenant,
                        entity_type="SPLITTER",
                        name=f"SPL-{pp_id[:6]}",
                        domain="fixed_access",
                        parent_entity_id=pp_id,
                        properties_json=props_json(ratio=f"1:{splitter_ratio}"),
                    )
                )

                acc.add_relationship(
                    make_relationship(
                        tenant_id=tenant,
                        from_entity_id=pp_id,
                        from_entity_type="PON_PORT",
                        relationship_type="SPLITS_TO",
                        to_entity_id=sp_id,
                        to_entity_type="SPLITTER",
                        domain="fixed_access",
                    )
                )

                # --- ONTs (sampled) ---
                n_ont = int(splitter_ratio * ont_fill_rate)
                # Add some variance
                n_ont = max(1, int(rng.poisson(n_ont)))
                n_ont = min(n_ont, splitter_ratio)

                for ont_idx in range(n_ont):
                    ont_id = _id()
                    ont_ids.append(ont_id)
                    ont_lat, ont_lon = offset_lat_lon(rng, lat, lon, spread_km=2.0)

                    acc.add_entity(
                        make_entity(
                            entity_id=ont_id,
                            tenant_id=tenant,
                            entity_type="ONT",
                            name=f"ONT-{sp_id[:4]}-{ont_idx}",
                            domain="fixed_access",
                            geo_lat=ont_lat,
                            geo_lon=ont_lon,
                            site_id=ex_id,
                            province=province,
                            parent_entity_id=sp_id,
                        )
                    )

                    acc.add_relationship(
                        make_relationship(
                            tenant_id=tenant,
                            from_entity_id=sp_id,
                            from_entity_type="SPLITTER",
                            relationship_type="SPLITS_TO",
                            to_entity_id=ont_id,
                            to_entity_type="ONT",
                            domain="fixed_access",
                        )
                    )

                    # ONT → RESIDENTIAL_SERVICE
                    svc_id = _id()
                    acc.add_entity(
                        make_entity(
                            entity_id=svc_id,
                            tenant_id=tenant,
                            entity_type="RESIDENTIAL_SERVICE",
                            name=f"BB-SVC-{ont_id[:6]}",
                            domain="fixed_access",
                            parent_entity_id=ont_id,
                            properties_json=props_json(
                                plan_speed_mbps=int(rng.choice([50, 100, 300, 500, 1000])),
                            ),
                        )
                    )

                    acc.add_relationship(
                        make_relationship(
                            tenant_id=tenant,
                            from_entity_id=ont_id,
                            from_entity_type="ONT",
                            relationship_type="SERVES_LINE",
                            to_entity_id=svc_id,
                            to_entity_type="RESIDENTIAL_SERVICE",
                            domain="fixed_access",
                        )
                    )

                # Cap total ONTs at ~100K to keep dataset manageable
                if len(ont_ids) >= 100_000:
                    break
            if len(ont_ids) >= 100_000:
                break
        if len(ont_ids) >= 100_000:
            break

    # ── Enterprise Ethernet Access ───────────────────────────
    pe_router_ids = transport_refs.get("pe_router_ids", [])
    nte_ids: list[str] = []

    for i in range(n_nte):
        nte_id = _id()
        nte_ids.append(nte_id)
        # Place near a random urban site
        if urban_sites:
            ref = urban_sites[i % len(urban_sites)]
            lat_n, lon_n = offset_lat_lon(rng, ref.get("geo_lat", 0), ref.get("geo_lon", 0), spread_km=3.0)
            province = ref.get("province", "")
        else:
            lat_n, lon_n = -6.2 + float(rng.normal(0, 0.5)), 106.8 + float(rng.normal(0, 0.5))
            province = "DKI Jakarta"

        acc.add_entity(
            make_entity(
                entity_id=nte_id,
                tenant_id=tenant,
                entity_type="NTE",
                name=entity_name("NTE", i, province),
                domain="fixed_access",
                geo_lat=lat_n,
                geo_lon=lon_n,
                province=province,
                sla_tier=rng.choice(["GOLD", "SILVER", "BRONZE"]),
                properties_json=props_json(interface_speed_mbps=int(rng.choice([100, 1000, 10000]))),
            )
        )

        # ETHERNET_CIRCUIT
        ec_id = _id()
        speed = int(rng.choice([100, 1000, 10000]))
        acc.add_entity(
            make_entity(
                entity_id=ec_id,
                tenant_id=tenant,
                entity_type="ETHERNET_CIRCUIT",
                name=f"ETH-CKT-{i:05d}",
                domain="fixed_access",
                parent_entity_id=nte_id,
                sla_tier=rng.choice(["GOLD", "SILVER"]),
                properties_json=props_json(speed_mbps=speed, service_type="E-Line"),
            )
        )

        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=nte_id,
                from_entity_type="NTE",
                relationship_type="HOSTS",
                to_entity_id=ec_id,
                to_entity_type="ETHERNET_CIRCUIT",
                domain="fixed_access",
            )
        )

        # ENTERPRISE_SERVICE
        es_id = _id()
        acc.add_entity(
            make_entity(
                entity_id=es_id,
                tenant_id=tenant,
                entity_type="ENTERPRISE_SERVICE",
                name=f"ENT-SVC-{i:05d}",
                domain="fixed_access",
                parent_entity_id=ec_id,
                sla_tier=rng.choice(["GOLD", "SILVER"]),
            )
        )

        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=ec_id,
                from_entity_type="ETHERNET_CIRCUIT",
                relationship_type="SERVES_LINE",
                to_entity_id=es_id,
                to_entity_type="ENTERPRISE_SERVICE",
                domain="fixed_access",
            )
        )

        # NTE → PE_ROUTER (access)
        if pe_router_ids:
            pe_idx = i % len(pe_router_ids)
            acc.add_relationship(
                make_relationship(
                    tenant_id=tenant,
                    from_entity_id=nte_id,
                    from_entity_type="NTE",
                    relationship_type="UPLINKS_TO",
                    to_entity_id=pe_router_ids[pe_idx],
                    to_entity_type="PE_ROUTER",
                    domain="cross_domain",
                )
            )

    # Store references
    acc._fixed_refs = {
        "exchange_ids": exchange_ids,
        "olt_ids": olt_ids,
        "ont_ids": ont_ids,
        "nte_ids": nte_ids,
    }

    e_count = acc.entity_count_by_domain().get("fixed_access", 0)
    r_count = acc.relationship_count_by_domain().get("fixed_access", 0)
    cd_count = acc.relationship_count_by_domain().get("cross_domain", 0)
    console.print(
        f"    [bold green]Fixed Broadband topology built:[/bold green] {e_count:,} entities, {r_count:,} relationships (+{cd_count:,} cross-domain)"
    )
    console.print(
        f"      Exchanges: {len(exchange_ids):,}, OLTs: {len(olt_ids):,}, ONTs: {len(ont_ids):,}, NTEs: {len(nte_ids):,}"
    )


# ============================================================================
# 3. CORE NETWORK DOMAIN
# ============================================================================


def build_core_network_topology(
    config: GeneratorConfig,
    acc: TopologyAccumulator,
    rng: np.random.Generator,
) -> None:
    """
    Build Core Network topology.

    EPC (4G): MME, SGW, PGW, HSS
    5GC (5G SA): AMF, SMF, UPF, NSSF, PCF, UDM, NWDAF
    IMS (Voice): P_CSCF, S_CSCF, TAS, MGCF
    Broadband Control: RADIUS, DHCP, DNS, POLICY_SERVER
    Voice: SOFTSWITCH, SBC, MEDIA_GATEWAY

    Target: ~200 entities, ~2,000 relationships
    """
    tenant = config.tenant_id
    transport_refs = getattr(acc, "_transport_refs", {})
    pe_router_ids = transport_refs.get("pe_router_ids", [])
    bng_ids = transport_refs.get("bng_ids", [])

    # Core elements are replicated across a few data centres
    dc_locations = [
        ("DKI Jakarta", -6.20, 106.85),
        ("Jawa Barat", -6.90, 107.60),
        ("Sumatera Utara", 2.50, 99.00),
        ("Jawa Timur", -7.55, 112.75),
    ]

    core_entities: dict[str, list[str]] = {}  # type -> list of IDs

    def _add_core(etype: str, count: int, sub_domain: str, **extra_props):
        ids = []
        for i in range(count):
            cid = _id()
            ids.append(cid)
            dc_idx = i % len(dc_locations)
            prov, lat, lon = dc_locations[dc_idx]
            lat_j, lon_j = offset_lat_lon(rng, lat, lon, spread_km=1.0)

            acc.add_entity(
                make_entity(
                    entity_id=cid,
                    tenant_id=tenant,
                    entity_type=etype,
                    name=entity_name(etype, i, prov),
                    domain="core",
                    geo_lat=lat_j,
                    geo_lon=lon_j,
                    province=prov,
                    properties_json=props_json(core_sub_domain=sub_domain, **extra_props),
                )
            )
        core_entities[etype] = ids
        return ids

    # ── EPC ──────────────────────────────────────────────────
    mme_ids = _add_core("MME", 12, "epc", capacity_subscribers=500000)
    sgw_ids = _add_core("SGW", 12, "epc")
    pgw_ids = _add_core("PGW", 8, "epc")
    hss_ids = _add_core("HSS", 4, "epc")

    # MME ↔ SGW, SGW → PGW, MME → HSS
    for i, mme_id in enumerate(mme_ids):
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=mme_id,
                from_entity_type="MME",
                relationship_type="BEARER_TO",
                to_entity_id=sgw_ids[i % len(sgw_ids)],
                to_entity_type="SGW",
                domain="core",
            )
        )
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=mme_id,
                from_entity_type="MME",
                relationship_type="DEPENDS_ON",
                to_entity_id=hss_ids[i % len(hss_ids)],
                to_entity_type="HSS",
                domain="core",
            )
        )
    for i, sgw_id in enumerate(sgw_ids):
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=sgw_id,
                from_entity_type="SGW",
                relationship_type="BEARER_TO",
                to_entity_id=pgw_ids[i % len(pgw_ids)],
                to_entity_type="PGW",
                domain="core",
            )
        )

    # eNBs → MME (DEPENDS_ON), eNBs → SGW (BEARER_TO)
    enodebs = acc.get_entities_by_type("ENODEB")
    for i, enb in enumerate(enodebs):
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=enb["entity_id"],
                from_entity_type="ENODEB",
                relationship_type="DEPENDS_ON",
                to_entity_id=mme_ids[i % len(mme_ids)],
                to_entity_type="MME",
                domain="core",
            )
        )
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=enb["entity_id"],
                from_entity_type="ENODEB",
                relationship_type="BEARER_TO",
                to_entity_id=sgw_ids[i % len(sgw_ids)],
                to_entity_type="SGW",
                domain="core",
            )
        )

    # ── 5GC ──────────────────────────────────────────────────
    amf_ids = _add_core("AMF", 8, "5gc")
    smf_ids = _add_core("SMF", 8, "5gc")
    upf_ids = _add_core("UPF", 16, "5gc")
    nssf_ids = _add_core("NSSF", 4, "5gc")
    pcf_ids = _add_core("PCF", 4, "5gc")
    udm_ids = _add_core("UDM", 4, "5gc")
    nwdaf_ids = _add_core("NWDAF", 4, "5gc")

    # AMF → SMF → UPF chain
    for i, amf_id in enumerate(amf_ids):
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=amf_id,
                from_entity_type="AMF",
                relationship_type="DEPENDS_ON",
                to_entity_id=smf_ids[i % len(smf_ids)],
                to_entity_type="SMF",
                domain="core",
            )
        )
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=amf_id,
                from_entity_type="AMF",
                relationship_type="DEPENDS_ON",
                to_entity_id=nssf_ids[i % len(nssf_ids)],
                to_entity_type="NSSF",
                domain="core",
            )
        )
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=amf_id,
                from_entity_type="AMF",
                relationship_type="DEPENDS_ON",
                to_entity_id=udm_ids[i % len(udm_ids)],
                to_entity_type="UDM",
                domain="core",
            )
        )
    for i, smf_id in enumerate(smf_ids):
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=smf_id,
                from_entity_type="SMF",
                relationship_type="BEARER_TO",
                to_entity_id=upf_ids[i % len(upf_ids)],
                to_entity_type="UPF",
                domain="core",
            )
        )
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=smf_id,
                from_entity_type="SMF",
                relationship_type="DEPENDS_ON",
                to_entity_id=pcf_ids[i % len(pcf_ids)],
                to_entity_type="PCF",
                domain="core",
            )
        )

    # gNBs → AMF (DEPENDS_ON), gNBs → UPF (BEARER_TO)
    gnodebs = acc.get_entities_by_type("GNODEB")
    for i, gnb in enumerate(gnodebs):
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=gnb["entity_id"],
                from_entity_type="GNODEB",
                relationship_type="DEPENDS_ON",
                to_entity_id=amf_ids[i % len(amf_ids)],
                to_entity_type="AMF",
                domain="core",
            )
        )
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=gnb["entity_id"],
                from_entity_type="GNODEB",
                relationship_type="BEARER_TO",
                to_entity_id=upf_ids[i % len(upf_ids)],
                to_entity_type="UPF",
                domain="core",
            )
        )

    # ── IMS ──────────────────────────────────────────────────
    pcscf_ids = _add_core("P_CSCF", 8, "ims")
    scscf_ids = _add_core("S_CSCF", 8, "ims")
    tas_ids = _add_core("TAS", 4, "ims")
    mgcf_ids = _add_core("MGCF", 4, "ims")

    for i, p_id in enumerate(pcscf_ids):
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=p_id,
                from_entity_type="P_CSCF",
                relationship_type="DEPENDS_ON",
                to_entity_id=scscf_ids[i % len(scscf_ids)],
                to_entity_type="S_CSCF",
                domain="core",
            )
        )
    for i, s_id in enumerate(scscf_ids):
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=s_id,
                from_entity_type="S_CSCF",
                relationship_type="DEPENDS_ON",
                to_entity_id=tas_ids[i % len(tas_ids)],
                to_entity_type="TAS",
                domain="core",
            )
        )

    # ── Broadband Service Control ─────────────────────────────
    radius_ids = _add_core("RADIUS_SERVER", 8, "broadband_control")
    dhcp_ids = _add_core("DHCP_SERVER", 8, "broadband_control")
    dns_ids = _add_core("DNS_RESOLVER", 12, "broadband_control")
    policy_ids = _add_core("POLICY_SERVER", 4, "broadband_control")

    # BNG → RADIUS, BNG → PGW
    for i, bid in enumerate(bng_ids):
        if radius_ids:
            acc.add_relationship(
                make_relationship(
                    tenant_id=tenant,
                    from_entity_id=bid,
                    from_entity_type="BNG",
                    relationship_type="AUTHENTICATES_VIA",
                    to_entity_id=radius_ids[i % len(radius_ids)],
                    to_entity_type="RADIUS_SERVER",
                    domain="core",
                )
            )
        if pgw_ids:
            acc.add_relationship(
                make_relationship(
                    tenant_id=tenant,
                    from_entity_id=bid,
                    from_entity_type="BNG",
                    relationship_type="BEARER_TO",
                    to_entity_id=pgw_ids[i % len(pgw_ids)],
                    to_entity_type="PGW",
                    domain="core",
                )
            )

    # ── Voice ────────────────────────────────────────────────
    ss_ids = _add_core("SOFTSWITCH", 4, "voice")
    sbc_ids = _add_core("SBC", 8, "voice")
    mg_ids = _add_core("MEDIA_GATEWAY", 4, "voice")
    sip_ids = _add_core("SIP_TRUNK", 8, "voice")

    for i, sbc_id in enumerate(sbc_ids):
        if pcscf_ids:
            acc.add_relationship(
                make_relationship(
                    tenant_id=tenant,
                    from_entity_id=sbc_id,
                    from_entity_type="SBC",
                    relationship_type="DEPENDS_ON",
                    to_entity_id=pcscf_ids[i % len(pcscf_ids)],
                    to_entity_type="P_CSCF",
                    domain="core",
                )
            )

    # ── Enterprise Control ───────────────────────────────────
    sdwan_ids = _add_core("SD_WAN_CONTROLLER", 4, "enterprise_control")
    fw_svc_ids = _add_core("FIREWALL_SERVICE", 8, "enterprise_control")
    ce_router_count = min(200, len(transport_refs.get("pe_router_ids", [])))
    ce_ids = _add_core("CE_ROUTER", ce_router_count, "enterprise_control")

    # CE_ROUTER → PE_ROUTER
    for i, ce_id in enumerate(ce_ids):
        if pe_router_ids:
            acc.add_relationship(
                make_relationship(
                    tenant_id=tenant,
                    from_entity_id=ce_id,
                    from_entity_type="CE_ROUTER",
                    relationship_type="UPLINKS_TO",
                    to_entity_id=pe_router_ids[i % len(pe_router_ids)],
                    to_entity_type="PE_ROUTER",
                    domain="core",
                )
            )

    acc._core_refs = {
        "mme_ids": mme_ids,
        "sgw_ids": sgw_ids,
        "pgw_ids": pgw_ids,
        "amf_ids": amf_ids,
        "smf_ids": smf_ids,
        "upf_ids": upf_ids,
        "radius_ids": radius_ids,
    }

    e_count = acc.entity_count_by_domain().get("core", 0)
    r_count = acc.relationship_count_by_domain().get("core", 0)
    console.print(
        f"    [bold green]Core Network topology built:[/bold green] {e_count:,} entities, {r_count:,} relationships"
    )


# ============================================================================
# 4. LOGICAL / SERVICE DOMAIN
# ============================================================================


def build_logical_service_topology(
    config: GeneratorConfig,
    acc: TopologyAccumulator,
    rng: np.random.Generator,
) -> None:
    """
    Build Logical/Service topology.

    - NETWORK_SLICE (eMBB, URLLC, mMTC) — 5G SA cells are members
    - TRACKING_AREA — groups of cells for paging
    - SERVICE_AREA — geographic coverage zones
    - QOS_PROFILE — QCI/5QI mappings

    Target: ~10K entities, ~50K relationships
    """
    tenant = config.tenant_id

    # ── NETWORK_SLICE ────────────────────────────────────────
    # 3 main slices + some enterprise dedicated slices
    slice_defs = [
        ("EMBB_SLICE", "eMBB-National", "enhanced Mobile Broadband"),
        ("URLLC_SLICE", "URLLC-National", "Ultra Reliable Low Latency"),
        ("MMTC_SLICE", "mMTC-National", "massive Machine Type Communications"),
    ]
    slice_ids: list[str] = []
    for stype, sname, sdesc in slice_defs:
        sid = _id()
        slice_ids.append(sid)
        acc.add_entity(
            make_entity(
                entity_id=sid,
                tenant_id=tenant,
                entity_type=stype,
                name=sname,
                domain="logical_service",
                properties_json=props_json(description=sdesc, sst=slice_defs.index((stype, sname, sdesc)) + 1),
            )
        )

    # Add enterprise-dedicated slices
    for i in range(50):
        sid = _id()
        slice_ids.append(sid)
        acc.add_entity(
            make_entity(
                entity_id=sid,
                tenant_id=tenant,
                entity_type="NETWORK_SLICE",
                name=f"ENT-SLICE-{i:04d}",
                domain="logical_service",
                sla_tier=rng.choice(["GOLD", "SILVER"]),
                properties_json=props_json(sst=1, sd=f"ENT-{i:04d}"),
            )
        )

    # NR_SA cells → NETWORK_SLICE (MEMBER_OF)
    nr_sa_cells = [e for e in acc.get_entities_by_type("NR_CELL") if e.get("rat_type") == "NR_SA"]
    for cell in nr_sa_cells:
        # Each SA cell is member of eMBB slice; some also in URLLC/mMTC
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=cell["entity_id"],
                from_entity_type="NR_CELL",
                relationship_type="MEMBER_OF",
                to_entity_id=slice_ids[0],
                to_entity_type="EMBB_SLICE",
                domain="logical_service",
            )
        )
        if rng.random() < 0.3:
            acc.add_relationship(
                make_relationship(
                    tenant_id=tenant,
                    from_entity_id=cell["entity_id"],
                    from_entity_type="NR_CELL",
                    relationship_type="MEMBER_OF",
                    to_entity_id=slice_ids[1],
                    to_entity_type="URLLC_SLICE",
                    domain="logical_service",
                )
            )

    # ── TRACKING_AREA ────────────────────────────────────────
    # Typically 50-200 cells per TA, so we need ~300-1000 TAs
    n_tracking_areas = 800
    ta_ids: list[str] = []
    for i in range(n_tracking_areas):
        ta_id = _id()
        ta_ids.append(ta_id)
        acc.add_entity(
            make_entity(
                entity_id=ta_id,
                tenant_id=tenant,
                entity_type="TRACKING_AREA",
                name=f"TA-{i:04d}",
                domain="logical_service",
                properties_json=props_json(tac=10000 + i),
            )
        )

    # Assign cells to tracking areas
    all_cells = acc.get_entities_by_type("LTE_CELL") + acc.get_entities_by_type("NR_CELL")
    for i, cell in enumerate(all_cells):
        ta_idx = i % len(ta_ids)
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=cell["entity_id"],
                from_entity_type=cell["entity_type"],
                relationship_type="MEMBER_OF",
                to_entity_id=ta_ids[ta_idx],
                to_entity_type="TRACKING_AREA",
                domain="logical_service",
            )
        )

    # ── SERVICE_AREA ─────────────────────────────────────────
    n_service_areas = 200
    sa_ids: list[str] = []
    for i in range(n_service_areas):
        sa_id = _id()
        sa_ids.append(sa_id)
        acc.add_entity(
            make_entity(
                entity_id=sa_id,
                tenant_id=tenant,
                entity_type="SERVICE_AREA",
                name=f"SA-{i:04d}",
                domain="logical_service",
                properties_json=props_json(sac=20000 + i),
            )
        )

    # Assign TAs to service areas
    for i, ta_id in enumerate(ta_ids):
        sa_idx = i % len(sa_ids)
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=ta_id,
                from_entity_type="TRACKING_AREA",
                relationship_type="MEMBER_OF",
                to_entity_id=sa_ids[sa_idx],
                to_entity_type="SERVICE_AREA",
                domain="logical_service",
            )
        )

    # ── QOS_PROFILE ──────────────────────────────────────────
    # Standard QCI values (4G) and 5QI values (5G)
    qos_profiles = [
        ("QCI-1", "Conversational Voice", 1, 100),
        ("QCI-2", "Conversational Video", 2, 150),
        ("QCI-3", "Real-time Gaming", 3, 50),
        ("QCI-5", "IMS Signalling", 5, 100),
        ("QCI-6", "Video Buffered", 6, 300),
        ("QCI-7", "Voice/Video Live Streaming", 7, 100),
        ("QCI-8", "Video TCP Premium", 8, 300),
        ("QCI-9", "Video TCP Default", 9, 300),
        ("5QI-1", "Conversational Voice 5G", 1, 100),
        ("5QI-5", "IMS Signalling 5G", 5, 100),
        ("5QI-9", "Non-GBR Default 5G", 9, 300),
        ("5QI-80", "Low-latency eMBB", 80, 10),
        ("5QI-82", "Discrete Automation", 82, 10),
        ("5QI-83", "Electricity Distribution", 83, 10),
        ("5QI-85", "V2X Messages", 85, 5),
    ]
    for name, desc, qi_val, delay in qos_profiles:
        qid = _id()
        acc.add_entity(
            make_entity(
                entity_id=qid,
                tenant_id=tenant,
                entity_type="QOS_PROFILE",
                name=name,
                domain="logical_service",
                properties_json=props_json(description=desc, qi_value=qi_val, delay_budget_ms=delay),
            )
        )

    e_count = acc.entity_count_by_domain().get("logical_service", 0)
    r_count = acc.relationship_count_by_domain().get("logical_service", 0)
    console.print(
        f"    [bold green]Logical/Service topology built:[/bold green] {e_count:,} entities, {r_count:,} relationships"
    )


# ============================================================================
# 5. POWER / ENVIRONMENT DOMAIN (supplement)
# ============================================================================


def build_power_environment_supplement(
    config: GeneratorConfig,
    acc: TopologyAccumulator,
    rng: np.random.Generator,
) -> None:
    """
    Supplement the power/environment entities already created in the
    Mobile RAN builder (POWER_SUPPLY, BATTERY_BANK, CLIMATE_CONTROL, MAINS_CONNECTION).

    This adds:
    - GENERATOR entities for sites with backup generators
    - Power entities for EXCHANGE_BUILDINGS
    - Additional BATTERY entities

    The RAN builder already created ~21K each of POWER_SUPPLY, BATTERY_BANK,
    MAINS_CONNECTION, and CLIMATE_CONTROL. We supplement to reach the ~40K
    power/environment entity target.
    """
    tenant = config.tenant_id

    # --- Generators for a subset of sites (20% of greenfield + all exchanges) ---
    sites = acc.get_entities_by_type("SITE")
    gen_count = 0

    for site in sites:
        site_id = site["entity_id"]
        site_type = site.get("site_type", "")
        profile = site.get("deployment_profile", "")

        # Generators at: greenfield rural/deep_rural (60%), all in_building (30%)
        has_generator = False
        if site_type == "greenfield" and profile in ("rural", "deep_rural"):
            has_generator = rng.random() < 0.6
        elif site_type == "in_building":
            has_generator = rng.random() < 0.3
        elif profile == "dense_urban":
            has_generator = rng.random() < 0.15

        if has_generator:
            gen_id = _id()
            acc.add_entity(
                make_entity(
                    entity_id=gen_id,
                    tenant_id=tenant,
                    entity_type="GENERATOR",
                    name=f"GEN-{site_id[:8]}",
                    domain="power_environment",
                    site_id=site_id,
                    site_type=site_type,
                    province=site.get("province"),
                    timezone=site.get("timezone"),
                    properties_json=props_json(
                        fuel="diesel",
                        capacity_kva=round(float(rng.uniform(15, 100)), 0),
                        runtime_hours=round(float(rng.uniform(0, 5000)), 0),
                    ),
                )
            )
            gen_count += 1

            # Find the power supply at this site
            site_entities = acc.get_entities_at_site(site_id)
            ps_entities = [e for e in site_entities if e["entity_type"] == "POWER_SUPPLY"]
            if ps_entities:
                acc.add_relationship(
                    make_relationship(
                        tenant_id=tenant,
                        from_entity_id=gen_id,
                        from_entity_type="GENERATOR",
                        relationship_type="POWERS",
                        to_entity_id=ps_entities[0]["entity_id"],
                        to_entity_type="POWER_SUPPLY",
                        domain="power_environment",
                    )
                )

    # --- Power + cooling for exchange buildings ---
    exchanges = acc.get_entities_by_type("EXCHANGE_BUILDING")
    for ex in exchanges:
        ex_id = ex["entity_id"]
        province = ex.get("province", "")

        # Exchange power supply
        ps_id = _id()
        acc.add_entity(
            make_entity(
                entity_id=ps_id,
                tenant_id=tenant,
                entity_type="POWER_SUPPLY",
                name=f"PWR-EXCH-{ex_id[:6]}",
                domain="power_environment",
                site_id=ex_id,
                province=province,
                properties_json=props_json(type="UPS", rating_kw=round(float(rng.uniform(20, 200)), 0)),
            )
        )
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=ex_id,
                from_entity_type="EXCHANGE_BUILDING",
                relationship_type="HOSTS",
                to_entity_id=ps_id,
                to_entity_type="POWER_SUPPLY",
                domain="power_environment",
            )
        )
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=ps_id,
                from_entity_type="POWER_SUPPLY",
                relationship_type="POWERS",
                to_entity_id=ex_id,
                to_entity_type="EXCHANGE_BUILDING",
                domain="power_environment",
            )
        )

        # Exchange battery bank
        bat_id = _id()
        acc.add_entity(
            make_entity(
                entity_id=bat_id,
                tenant_id=tenant,
                entity_type="BATTERY_BANK",
                name=f"BAT-EXCH-{ex_id[:6]}",
                domain="power_environment",
                site_id=ex_id,
                province=province,
                parent_entity_id=ps_id,
                properties_json=props_json(voltage_v=48.0, capacity_ah=round(float(rng.uniform(500, 2000)), 0)),
            )
        )
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=ps_id,
                from_entity_type="POWER_SUPPLY",
                relationship_type="HOSTS",
                to_entity_id=bat_id,
                to_entity_type="BATTERY_BANK",
                domain="power_environment",
            )
        )

        # Exchange climate control
        cc_id = _id()
        acc.add_entity(
            make_entity(
                entity_id=cc_id,
                tenant_id=tenant,
                entity_type="CLIMATE_CONTROL",
                name=f"COOL-EXCH-{ex_id[:6]}",
                domain="power_environment",
                site_id=ex_id,
                province=province,
                properties_json=props_json(cooling_capacity_kw=round(float(rng.uniform(10, 50)), 1)),
            )
        )
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=ex_id,
                from_entity_type="EXCHANGE_BUILDING",
                relationship_type="HOSTS",
                to_entity_id=cc_id,
                to_entity_type="CLIMATE_CONTROL",
                domain="power_environment",
            )
        )
        acc.add_relationship(
            make_relationship(
                tenant_id=tenant,
                from_entity_id=cc_id,
                from_entity_type="CLIMATE_CONTROL",
                relationship_type="COOLS",
                to_entity_id=ex_id,
                to_entity_type="EXCHANGE_BUILDING",
                domain="power_environment",
            )
        )

        # Exchange generator (80% of exchanges)
        if rng.random() < 0.8:
            gen_id = _id()
            acc.add_entity(
                make_entity(
                    entity_id=gen_id,
                    tenant_id=tenant,
                    entity_type="GENERATOR",
                    name=f"GEN-EXCH-{ex_id[:6]}",
                    domain="power_environment",
                    site_id=ex_id,
                    province=province,
                    properties_json=props_json(
                        fuel="diesel",
                        capacity_kva=round(float(rng.uniform(100, 500)), 0),
                    ),
                )
            )
            acc.add_relationship(
                make_relationship(
                    tenant_id=tenant,
                    from_entity_id=gen_id,
                    from_entity_type="GENERATOR",
                    relationship_type="POWERS",
                    to_entity_id=ps_id,
                    to_entity_type="POWER_SUPPLY",
                    domain="power_environment",
                )
            )
            gen_count += 1

    e_count = acc.entity_count_by_domain().get("power_environment", 0)
    r_count = acc.relationship_count_by_domain().get("power_environment", 0)
    console.print(
        f"    [bold green]Power/Environment supplement built:[/bold green] {e_count:,} entities, {r_count:,} relationships"
    )
    console.print(f"      Generators: {gen_count:,}")
