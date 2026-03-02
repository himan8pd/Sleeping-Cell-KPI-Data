# Pedkai Synthetic Data Generator

A UK-scale converged telecom operator dataset generator for [Pedkai](https://github.com/your-org/pedkai) — producing radio-layer physics, multi-domain KPIs, scenario injection, and CMDB degradation data for Dark Graph training and ML model development.

> **Origin:** This repo was originally [Sleeping-Cell-KPI-Data](https://github.com/adityonugrohoid/telecom-digital-twin) (3 CSV files of sleeping cell KPIs). It has been repurposed as the synthetic data generation pipeline for the Pedkai platform. The original CSV files are preserved in the repo root for reference.

---

## Table of Contents

- [Overview](#overview)
- [Scale Parameters](#scale-parameters)
- [Pipeline Phases (0–11)](#pipeline-phases-011)
- [Current Status](#current-status)
- [Output Files](#output-files)
- [Architecture & Memory Safety](#architecture--memory-safety)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Design References](#design-references)

---

## Overview

The generator produces **~6.6 GB across 14 Parquet files** representing a realistic converged telecom operator with:

- **21,100 sites** across 38 Indonesian provinces (3 timezones: WIB/WITA/WIT)
- **~66,100 logical cell-layers** (LTE, NR-NSA EN-DC, NR-SA) with 3GPP-aligned physics
- **~811,000 network entities** across 6 domains (Mobile RAN, Transport, Fixed Broadband, Core, Logical/Service, Power)
- **~1.97M relationships** with full cross-domain dependency chains
- **47.6M hourly KPI rows** (30 days × 720 intervals × 66k cells) with SINR→CQI→MCS→throughput physics chain
- **Multi-domain KPIs** for transport, fixed broadband, enterprise circuits, core elements, and power/environment
- **Scenario injection** — sleeping cell, congestion, fibre cut, power failure, interference, coverage holes
- **CMDB degradation** — 6 types of Dark Graph divergence (dark nodes, phantom nodes, dark edges, phantom edges, dark attributes, identity mutations)
- **1M sampled customers** with BSS/billing data
- **Ericsson & Nokia** vendor naming layers

All outputs are deterministic (seeded) and reproducible. The full design context is captured in [`THREAD_SUMMARY_SYNTHETIC_DATA_GENERATOR.md`](./THREAD_SUMMARY_SYNTHETIC_DATA_GENERATOR.md).

---

## Scale Parameters

### UK-Scale Operator Profile

| Site Type | Count | Sectors/Site | Cells | Deployment Profile |
|-----------|------:|:------------:|------:|-------------------|
| Greenfield | 11,000 | 3 | 33,000 | suburban, rural, deep_rural |
| Rooftop | 4,000 | 3 | 12,000 | urban, dense_urban |
| Streetworks | 5,500 | 1 | 5,500 | dense_urban, urban (small cells) |
| In-Building | 500 | 2 | 1,000 | indoor (DAS/small cells) |
| Unspecified | 100 | 2 | 200 | mixed |
| **Total** | **21,100** | | **51,700** | |

### RAT Split

| RAT | Physical Cells | Logical Cell-Layers | Logic |
|-----|---------------:|--------------------:|-------|
| LTE-only | ~30,000 | ~30,000 | Most greenfield (rural/deep rural), some suburban |
| LTE + 5G NSA (EN-DC) | ~14,400 | ~28,800 | Each NSA cell = LTE anchor + NR SCG leg (2 KPI streams) |
| 5G SA | ~7,300 | ~7,300 | Dense urban streetworks, some rooftop, enterprise |
| **Total** | **~51,700** | **~66,100** | |

### Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Simulation period | 30 days |
| Reporting interval | 1 hour (standard PM granularity) |
| Epoch | 2024-01-01 00:00 UTC (Monday) |
| Cell KPI rows | 66,131 × 720 = **47,614,320** |
| KPI columns | 35 radio-layer PM counters + 9 metadata |
| Global seed | 42,000,001 |
| Tenant ID | `pedkai_synthetic_01` |

---

## Pipeline Phases (0–12)

The generator is structured as a 13-step pipeline with explicit dependencies. Each step derives a deterministic seed from the global seed: `Seed(Step_N) = HMAC-SHA256(global_seed, step_id)`.

### Phase 0 — Schema Contracts ✅

> Define PyArrow column contracts for all 14 output Parquet files.

- **Module:** `step_00_schema/contracts.py`
- **Output:** `output/schemas/*.schema.ipc` + `all_contracts.json`
- **Defines:** Column names, types, allowed values, min/max ranges, expected row counts for every output file
- **Dependencies:** None

### Phase 1 — Site & Cell Generation ✅

> Generate the physical site inventory and cell-layer schema.

- **Module:** `step_01_sites/generate.py`
- **Output:** `intermediate/sites.parquet` (21,100 rows, 1.0 MB), `intermediate/cells.parquet` (66,131 rows, 4.6 MB)
- **Produces:** Site positions (lat/lon per province), deployment profiles, vendor assignment (55% Ericsson / 45% Nokia), RAT assignment with band plans (L900/L1800/L2100/L2300 for LTE; n1/n3/n28/n78 for NR-NSA; n77/n78/n257/n258 for NR-SA), sector azimuth/tilt/height, inter-site distance, NSA anchor-SCG linkage, external IDs (Ericsson/Nokia style)
- **Dependencies:** Phase 0

### Phase 2 — Full Network Topology ✅

> Build the complete multi-domain entity-relationship graph.

- **Module:** `step_02_topology/` (generate.py, mobile_ran.py, other_domains.py, builders.py, neighbours.py)
- **Output:**
  - `output/ground_truth_entities.parquet` (811,064 rows, 36 MB)
  - `output/ground_truth_relationships.parquet` (1,970,387 rows, 74 MB)
  - `output/neighbour_relations.parquet` (926,475 rows, 34 MB)
  - `intermediate/topology_metadata.json`
- **Domain breakdown:**

| Domain | Entities | Relationships | Key Entity Types |
|--------|--------:|--------------:|-----------------|
| Mobile RAN | 554,971 | 1,538,077 | SITE, CABINET, ENODEB, GNODEB, BBU, RRU, ANTENNA, LTE_CELL, NR_CELL |
| Transport | 33,531 | 66,568 | PE_ROUTER, P_ROUTER, AGG_SWITCH, MICROWAVE_LINK, FIBRE_CABLE, DWDM, LSP, L3VPN |
| Fixed Broadband | 218,992 | 216,899 | EXCHANGE, OLT, PON_PORT, SPLITTER, ONT, NTE, ETHERNET_CIRCUIT |
| Core Network | 376 | 67,580 | MME, SGW, PGW, AMF, SMF, UPF, HSS, UDM, NSSF, P-CSCF, S-CSCF |
| Logical/Service | 1,068 | 76,469 | NETWORK_SLICE, TRACKING_AREA, SERVICE_AREA, QOS_PROFILE |
| Power/Environment | 2,126 | 2,312 | POWER_SUPPLY, BATTERY_BANK, GENERATOR, CLIMATE_CONTROL |

- **Neighbour relations:** 926,475 cell-to-cell neighbour pairs (spatial grid + distance gating)
- **Dependencies:** Phase 1

### Phase 3 — Radio-Layer Physics & Cell KPIs ✅

> SINR → CQI → MCS → throughput physics chain for all cells across 30 days.

- **Module:** `step_03_radio_kpis/` (generate.py, physics.py, profiles.py)
- **Output:** `output/kpi_metrics_wide.parquet` (47,614,320 rows, 44 columns, 8.5 GB)
- **Physics chain (3GPP-aligned):**
  - Path loss: COST 231-Hata (<2 GHz), log-distance (2-6 GHz), mmWave with atmospheric absorption (>6 GHz)
  - RSRP: Tx power + antenna gain − path loss − shadow fading − indoor/environment loss
  - SINR: RSRP − noise power − interference (load-driven IoT)
  - CQI: 3GPP TS 36.213 Table 7.2.3-1 (LTE), TS 38.214 Table 5.2.2.1-3 (NR, 256QAM)
  - MCS: CQI-to-MCS mapping per RAT
  - Throughput: SE × BW × MIMO layers × PRB utilisation × (1 − overhead)
- **35 KPI columns:** RSRP, RSRQ, SINR, CQI, MCS (DL/UL), BLER (DL/UL), PRB utilisation (DL/UL), RACH (attempts/success), RRC (attempts/success), throughput (DL/UL), latency, jitter, packet loss, active UEs (avg/max), traffic volume, RLC retransmission (DL/UL), handover (attempts/success), cell availability, IoT, paging discard, CCE utilisation, VoLTE erlangs, CSFB (attempts/success), PDCP volume (DL/UL)
- **Diurnal profiles:** Per deployment profile (6 types) × weekday/weekend × 3 Indonesian timezones, with Friday evening boost and optional Ramadan overlay
- **Temporal correlation:** AR(1) shadow fading (ρ=0.95), AR(1) load jitter (ρ=0.7)
- **Memory-safe:** Streaming AR(1) generator, ~200 MB peak (see [Architecture](#architecture--memory-safety))
- **Dependencies:** Phase 1

### Phase 4 — Multi-Domain KPIs ✅

> Generate KPIs for transport, fixed broadband, enterprise circuits, core elements, and power/environment.

- **Module:** `step_04_domain_kpis/generate.py`
- **Output:** (row counts driven by **actual** Phase 2 topology output, not config estimates)
  - `output/transport_kpis_wide.parquet` — 29,736 entities (PE_ROUTER, AGG_SWITCH, MICROWAVE_LINK, FIBRE_CABLE, DWDM_SYSTEM, LSP, L3VPN, BNG, ACCESS_SWITCH) × 720h = **21,409,920 rows**, 20 columns, **1,286.9 MB**
  - `output/fixed_broadband_kpis_wide.parquet` — 6,634 entities (OLT, PON_PORT) × 720h = **4,776,480 rows**, 19 columns, **400.3 MB**. Note: the 218,992 fixed broadband entities include ~200K ONTs/SPLITTERs which report via their parent OLT/PON aggregation, not as independent PM streams.
  - `output/enterprise_circuit_kpis_wide.parquet` — 2,000 entities (ETHERNET_CIRCUIT) × 720h = **1,440,000 rows**, 15 columns, **84.3 MB**
  - `output/core_element_kpis_wide.parquet` — 416 core entities (MME, SGW, PGW, AMF, SMF, UPF, HSS, UDM, NSSF, PCF, NWDAF, P-CSCF, S-CSCF, TAS, MGCF, RADIUS_SERVER, DHCP_SERVER, DNS_RESOLVER, POLICY_SERVER, SOFTSWITCH, SBC, MEDIA_GATEWAY, SIP_TRUNK, SD_WAN_CONTROLLER, FIREWALL_SERVICE, CE_ROUTER, BNG) × 720h = **299,520 rows**, 25 columns, **20.8 MB**
  - `output/power_environment_kpis.parquet` — 21,100 sites × 720h = **15,192,000 rows**, 12 columns, **640.3 MB**
- **Total:** 43,117,920 rows across 5 files, 2,432.7 MB (2.4 GB), generated in ~1.5 minutes
- **Architecture:** Same streaming AR(1) per-hour flush as Phase 3 — peak memory ~50 MB per domain, domains generated sequentially. Each domain has its own AR(1) state vectors for temporal correlation (utilisation, latency, temperature, etc.) and diurnal profiles. All KPI columns are float32.
- **Entity discovery:** Reads `ground_truth_entities.parquet` from Phase 2 output and filters by `entity_type` to determine which entities get KPI streams — row counts are driven by actual topology, not config targets.
- **Dependencies:** Phase 2 (reads `ground_truth_entities.parquet` and `sites.parquet`)

### Phase 5 — Scenario Injection ✅

> Inject realistic failure/degradation scenarios into the KPI time series.

- **Module:** `step_05_scenarios/generate.py`
- **Scenario types (8):**
  - Sleeping cell (~2% of cells, ~1,034 cases) — subtle traffic/throughput degradation, NO alarm generated (the whole point); long duration (3–21 days), slow ramp-up
  - Congestion (5% of cells) — PRB >85%, throughput collapse, latency spike, CCE utilisation spike; 2–48h duration
  - Coverage hole (1% spatial clusters) — RSRP/RSRQ degradation in co-sited cell clusters, BLER increase, handover attempts increase; 2–15 days
  - Hardware fault (0.5% of cells) — abrupt cell availability drop, BLER spike, possible complete outage; 4h–1 week
  - Interference (3% of cells) — IoT elevation, SINR drop, CQI/MCS degradation; some periodic (industrial equipment patterns); 6h–5 days
  - Transport failure (0.2% of transport links) — backhaul link down with cascade to downstream cells and transport entities; abrupt onset/recovery; 1–24h
  - Power failure (0.1% of sites) — site-level event with battery backup modelling (2–6h); cascade: mains→battery drain→cooling failure→temperature rise→cell outage after battery depletion; 1–12h
  - Fibre cut (0.05% of fibre links) — cross-domain cascade: fibre→transport→cells+fixed broadband; optical signal loss, complete throughput collapse downstream; 2–48h
- **Cross-domain cascades:** Topology graph (BFS traversal of `ground_truth_relationships`) identifies all downstream affected entities. Fibre cut → transport KPI degradation → downstream cell throughput drop; Power failure → power KPI status flip → battery drain → cooling failure → site outage → all cells on that site go down. Cascade chains recorded in manifest for traceability.
- **Ramp-up/ramp-down curves:** Cosine-smoothed ramp profiles per scenario type — sleeping cells have slow onset (6–24h ramp), hardware faults are abrupt (0h ramp), congestion has short ramps (1–4h).
- **Override semantics:** Overrides are typed by scenario — multiplicative factors (0.0–1.0 range for throughput drops), additive deltas (negative dB for SINR/RSRP), or absolute values (0.0 for availability during outage). Consumers interpret based on `(scenario_type, kpi_column)` pair.
- **Output strategy — overlay, not in-place mutation:**
  - `output/scenario_manifest.parquet` — master schedule of all injected scenarios (scenario_id, type, severity, primary_entity_id, affected_entity_ids JSON array, start_hour, end_hour, duration_hours, cascade_chain JSON, ramp_up/down_hours, parameters_json), 16 columns
  - `output/scenario_kpi_overrides.parquet` — sparse override table (entity_id, tenant_id, timestamp, kpi_column, override_value, scenario_id, scenario_type, source_file), 8 columns. Consumers apply overrides on read: `effective_value = COALESCE(override, baseline)`.
  - This preserves the clean baseline KPI files from Phases 3–4 for A/B comparison (clean vs degraded), avoids destructive 8.5+ GB rewrites, and keeps the pipeline idempotent — re-running Phase 5 regenerates only the overlay files without touching upstream outputs.
- **Memory-safe:** Streaming `_OverrideWriter` flushes to Parquet row groups every 500K rows, keeping peak memory bounded even with millions of overrides. Scenario instances processed sequentially with periodic GC.
- **Severity model:** Each scenario type has a characteristic severity distribution (sleeping cell: low/medium; hardware fault: high/critical; power/fibre: critical).
- **Dependencies:** Phases 2 (topology graph for cascade walking), 3, 4

### Phase 6 — Events & Alarms ✅

> Generate multi-domain alarms aligned with scenario injections.

- **Module:** `step_06_events/generate.py`
- **Output:** `output/events_alarms.parquet` — 15 columns, ~500 MB
- **Two alarm categories:**
  1. **Scenario-driven alarms** — temporally aligned with Phase 5 scenarios. Each non-sleeping-cell scenario produces a chain of domain-appropriate alarms with detection delay (0–8h depending on type) and recovery delay. Cross-domain cascades (fibre cut, power failure, transport failure) produce correlated alarm chains sharing a `correlation_group_id`.
  2. **Organic/background alarms** — transient noise across all domains (brief interface flaps, minor temperature warnings, RACH spikes, BGP flaps, optical power fluctuations, RADIUS timeouts). ~0.05% per entity per day. Short duration (5 min–4h), self-clearing. Provides realistic clutter for alarm correlation systems.
- **Critical design rule:** Sleeping-cell scenarios produce **NO** alarms — that is the entire point of the sleeping-cell problem. The alarm gap is the ground-truth signal that Dark Graph must learn to detect.
- **Alarm types (from Phase 0 contract):**
  - Radio: CELL_DEGRADATION, CELL_OUTAGE, HIGH_BLER, HIGH_INTERFERENCE, RACH_FAILURE, HANDOVER_FAILURE, PRB_CONGESTION
  - Transport: LINK_DOWN, INTERFACE_DOWN, OPTICAL_POWER_LOW, BGP_FLAP, HIGH_LATENCY, HIGH_PACKET_LOSS
  - Power: MAINS_FAILURE, BATTERY_LOW, COOLING_FAILURE, HIGH_TEMPERATURE, POWER_SUPPLY_FAIL
  - Core: RADIUS_TIMEOUT, DEGRADATION
  - Fixed BB: PON_SIGNAL_DEGRADATION, OLT_PORT_DOWN
  - Generic: EQUIPMENT_FAILURE
- **Cross-domain alarm correlation:** A single scenario (e.g., fibre cut) produces a correlated alarm chain across domains — transport LINK_DOWN + OPTICAL_POWER_LOW → downstream INTERFACE_DOWN → radio CELL_DEGRADATION + HIGH_PACKET_LOSS → fixed broadband PON_SIGNAL_DEGRADATION + OLT_PORT_DOWN. Each alarm carries a `scenario_id` FK back to `scenario_manifest.parquet` for ground-truth labelling, plus a `correlation_group_id` shared across the cascade chain.
- **Source system assignment:** Ericsson-managed entities → `ericsson_enm`, Nokia → `nokia_netact`, transport → `snmp`/`oss_vendor`, power → `snmp`/`syslog`, core → mixed.
- **TMF642 compliance:** Alarm structure compatible with Pedkai's alarm ingestion API (alarm_id, entity_id, alarm_type, severity, raised_at, cleared_at, source_system, probable_cause, domain).
- **15 columns:** alarm_id, tenant_id, entity_id, entity_type, alarm_type, severity, raised_at, cleared_at, source_system, probable_cause, domain, scenario_id, is_synthetic_scenario, additional_text, correlation_group_id
- **Dependencies:** Phase 5 (reads `scenario_manifest.parquet`), Phase 2 (reads `ground_truth_entities.parquet` for entity metadata and vendor assignment)

### Phase 7 — Customers & BSS ✅

> Generate 1M sampled subscribers with billing, service plans, and site associations.

- **Module:** `step_07_customers/generate.py`
- **Output:** `output/customers_bss.parquet` (~200 MB, ~1M rows, 18 columns)
- **Customer mix:** 950,000 residential (95%) + 50,000 enterprise (5%), configurable via `UserScaleConfig`.
- **Service plans (20 distinct plans):**
  - Residential mobile: SIM Only 5GB/20GB/Unlimited, Handset Standard/Premium/Max
  - Residential FTTP: Fibre 36/80/150, Full Fibre 300/900
  - Residential FTTC: Broadband Essential, Superfast 55/80
  - Enterprise: Business Mobile Standard/Premium, Business Fibre 100/500, Ethernet 10/100/1000 Mbps, SD-WAN Managed
- **SLA tiers:** Gold / Silver / Bronze — tier assignment driven by plan selection (higher-value plans → higher tiers).
- **Access types:** mobile (60% residential default), fttp (25%), fttc (15%), enterprise_ethernet (enterprise only). Access type availability is topology-aware — FTTP only offered at sites with ONT entities, FTTC only at sites with DSLAM/Street Cabinet entities.
- **Access entity FK:** Fixed broadband customers carry `access_entity_id` FK pointing to their ONT (FTTP), NTE (enterprise ethernet), or DSLAM (FTTC) entity from Phase 2 topology.
- **Billing accounts:** Status distribution — 92% Active / 4% Suspended / 4% Delinquent (residential), 96/2/2 (enterprise). Revenue = plan fee × usage multiplier + overage charges.
- **Churn risk scores:** Beta-distributed [0, 1] with boosts for suspended/delinquent accounts and low-value plans. Enterprise customers have lower base risk.
- **Contract end dates:** ~30% residential on rolling monthly (null), remainder on 12/24-month contracts. Enterprise: ~5% no contract, rest on 12/24/36-month terms.
- **Proactive comms consent:** ~70% residential, ~85% enterprise opt-in.
- **Memory safety:** Streaming batch writer (100k customers per batch), periodic GC, PyArrow RecordBatch-based writes with zstd compression.
- **Deterministic seeding:** `config.seed_for("step_07_customers")`.
- **18 columns:** customer_id, tenant_id, external_id, customer_type, name, associated_site_id, province, service_plan_name, service_plan_tier, monthly_fee, sla_guarantee, account_status, avg_monthly_revenue, contract_end_date, churn_risk_score, consent_proactive_comms, access_type, access_entity_id
- **Built-in validation:** Post-generation read-back checks column count, row count, null constraints, and FK validity (all `associated_site_id` values verified against `ground_truth_entities`).
- **Design note:** Customer-to-cell/site mapping is generated here using topology data from Phase 2. Phase 5 scenario injection does **not** need customer data — scenarios target network entities. However, downstream Pedkai analytics (CX Impact Analysis) will join customers against scenario-affected cells at query time, so the FK from `associated_site_id` → `ground_truth_entities.entity_id` must be valid.
- **Dependencies:** Phase 2

### Phase 8 — CMDB Degradation ✅

> Apply Dark Graph divergences to the ground-truth topology to create a "declared state" CMDB.

- **Module:** `step_08_cmdb_degradation/generate.py`
- **Output:**
  - `output/cmdb_declared_entities.parquet` (degraded entity inventory — same schema as ground_truth_entities)
  - `output/cmdb_declared_relationships.parquet` (degraded relationship graph — same schema as ground_truth_relationships)
  - `output/divergence_manifest.parquet` (13-column labels for ML scoring)
- **Six divergence types (rates from `CMDBDegradationConfig`):**
  - Dark nodes (6.5%): Entities exist in reality but missing from CMDB — row dropped. SITE entities are protected from removal to avoid cascade-orphaning children.
  - Phantom nodes (3%): Fabricated entities inserted into CMDB using real entities as templates — plausible but non-ground-truth attribute values, new UUID, perturbed geo coordinates.
  - Dark edges (10%): Relationships exist but not declared — row dropped. Additionally, relationships referencing dark-noded endpoints are cascade-removed.
  - Phantom edges (5%): Fabricated relationships between compatible entities using 12 edge templates (e.g., SITE→CABINET, ENODEB→LTE_CELL, PE_ROUTER→PE_ROUTER). Avoids self-loops and duplicates of existing edges.
  - Dark attributes (15%): 1–2 attributes per entity corrupted. 8 mutable attribute types: vendor (flip), deployment_profile, sla_tier, geo_lat/lon (stale drift 1–11 km), band, province, site_type. Each corruption logged with ground_truth_value and cmdb_declared_value in the manifest.
  - Identity mutations (2%): External_id corrupted via 5 plausible mutation types — character transposition, digit substitution, region prefix corruption, suffix truncation, character duplication. Applied independently from dark_attribute targets.
- **Design note on identity mutations:** When we later remap UUIDs to realistic IDs (Phase 10.5), identity mutations must produce plausible errors for the new ID scheme — e.g., two ECGI-style cell IDs with transposed eNB digits, or a site code with a wrong region prefix — not arbitrary random swaps. The mutation logic should be ID-format-aware.
- **Divergence manifest (13 columns):** divergence_id, tenant_id, divergence_type, entity_or_relationship, target_id, target_type, domain, description, attribute_name, ground_truth_value, cmdb_declared_value, original_external_id, mutated_external_id
- **Deterministic seeding:** `config.seed_for("step_08_cmdb_degradation")`
- **Dependencies:** Phase 2

### Phase 9 — Vendor Naming ✅

> Map internal KPI names to Ericsson and Nokia PM counter naming conventions.

- **Module:** `step_09_vendor_naming/generate.py`
- **Maps:** Internal names (e.g., `dl_throughput_mbps`) → Ericsson counters (e.g., `pmRadioThpVolDl`) and Nokia counters (e.g., `VS.RAB.DLThroughput`)
- **Scope:** Applies across all 6 KPI domains — radio (Phase 3), transport, fixed broadband, enterprise circuits, core, and power/environment (all Phase 4).
- **Output:** `output/vendor_naming_map.parquet` — 11-column lookup table mapping (internal_name, domain, vendor) → vendor_counter_name. Not a rewrite of KPI files — consumers apply the mapping at read/display time.
- **Naming conventions:**
  - **Ericsson (ENM):** `pm` prefix + PascalCase metric (radio/power), `if` prefix for interface counters (transport), `cm`/`pm` for core. E.g., `pmRadioThpVolDl`, `ifHCInOctets`, `pmMmeAttachSuccRate`.
  - **Nokia (NetAct):** `VS.` dot-separated hierarchy (radio), `IF-MIB.` for interface counters (transport), `M8xxx` 3GPP-style (core), `GPON.` (fixed BB), `EQUIPMENT.` (power). E.g., `VS.RAB.DLThroughput`, `IF-MIB.ifHCInOctets`, `M8010.attachSuccRate`.
- **Coverage:** ~110 internal KPI names → ~220 vendor mappings (2 per KPI — one Ericsson, one Nokia) across 6 domains. Each mapping includes unit, description, counter_family, and 3GPP reference where applicable.
- **11 columns:** mapping_id, tenant_id, internal_name, domain, vendor, vendor_counter_name, vendor_system, unit, description, counter_family, three_gpp_ref
- **Dependencies:** Phases 3, 4

### Phase 10 — Validation ✅

> Full validation suite: schema compliance, FK integrity, range checks, cross-domain consistency.

- **Module:** `step_10_validation/validate.py`
- **Output:** `validation/` directory with per-file JSON reports + `summary_report.json`
- **Checks implemented:**
  - **Schema compliance:** column names present, no unexpected extra columns, column count matches contract
  - **Nullability:** non-nullable columns verified to have zero nulls
  - **Allowed values:** string columns with enum constraints validated against contract
  - **Range checks:** numeric columns checked against contract min/max bounds (with floating-point tolerance)
  - **Approximate row counts:** actual counts verified within ±50% of contract estimates
  - **FK integrity:** `entity_id`, `site_id`, `associated_site_id`, `from_entity_id`, `to_entity_id`, `access_entity_id` validated against `ground_truth_entities.entity_id`; `scenario_id` in events validated against `scenario_manifest.scenario_id`
  - **Neighbour relation symmetry:** spot-check that A→B implies B→A exists (sampled)
  - **CMDB divergence completeness:** dark nodes verified absent from CMDB, phantom nodes verified absent from ground truth and present in CMDB
  - **Scenario overlay integrity:** every `scenario_id` in overrides exists in manifest
  - **Events FK integrity:** entity_id and scenario_id FKs validated
- **Structured reporting:** `FileValidationReport` per file with pass/fail/warning counts; `ValidationSummary` across all files with cross-domain issues. All reports serialised as JSON.
- **Performance:** uses sampling (up to 500K–1M rows) for allowed-value and FK checks on very large files to keep validation fast.
- **Dependencies:** All previous phases (0–9)

### Phase 10.5 — ID Optimisation & Realistic Identifier Remapping ⊘ CANCELLED

> ~~Replace UUID v4 strings with realistic, deterministic, storage-efficient identifiers.~~

- **Status:** Cancelled after analysis of Pedkai's ORM schema and ingestion pathway.
- **Original plan:** Remap all UUID v4 entity/cell/site IDs to ECGI-style telecom identifiers, generate an integer surrogate key mapping table, and rewrite all 17 Parquet files in a bulk column-swap pass. Estimated 500 MB–1 GB storage savings.
- **Reason for cancellation — Pedkai's schema is UUID-native and the remapping adds unnecessary complexity:**
  1. **Pedkai's ORM models use UUID primary keys throughout.** `NetworkEntityORM.id` is `UUID(as_uuid=True)`, `KpiSampleORM.entity_id` is a UUID FK, `CustomerORM.id` is UUID, `KPIMetricORM.entity_id` is `String(255)` (accepts UUIDs directly), and `EntityRelationshipORM.from_entity_id` / `to_entity_id` are `String(255)`. The generated data already speaks Pedkai's native language. Remapping to ECGI-style strings like `510-01-A04F2-01` would force the loader to translate realistic IDs back into UUIDs for internal PKs, or populate them only in `external_id` while generating new UUIDs — adding a translation layer for no benefit.
  2. **Pedkai already has an `external_id` field for human-readable identifiers.** `NetworkEntityORM.external_id` (`String(255)`, indexed) is specifically designed for vendor-provided NMS identifiers. If ECGI-style display names are ever needed, the loader can derive them at load time and populate `external_id` — without rewriting the entire 12 GB dataset.
  3. **KPI files are registered as external Parquet datasets, not row-exploded.** Pedkai queries them via DuckDB/Arrow Flight. DuckDB handles Parquet dictionary-encoded string columns natively and efficiently. The storage savings from integer surrogates are moot for this access pattern.
  4. **Entity/relationship tables are too small for the optimisation to matter.** ~811K entity rows and ~2.1M relationship rows — the difference between 36-char UUID strings and INT32 PKs is ~50 MB at this scale. PostgreSQL btree indexes handle both efficiently.
  5. **Integer surrogates would require rewriting Pedkai's query layer.** Every graph traversal, RCA, and impact analysis query in Pedkai joins on UUID/string entity IDs. Introducing integer surrogates means either rewriting those queries or carrying both column types (defeating the purpose).
  6. **The `relation_id` / `relationship_id` cleanup is a micro-optimisation.** Replacing UUID `relation_id` with sequential INT32 on 926K rows saves ~30 MB (0.2% of total dataset). Not worth the engineering complexity.
  7. **Deterministic reproducibility is solvable without remapping.** The pipeline already uses `derive_seed()` and `config.seed_for()` throughout. If deterministic IDs are needed, `uuid.uuid5()` (name-based) with the global seed is a one-line change in the generators — far simpler than a full remapping pass.
- **If revisited:** Should Pedkai ever need realistic telecom identifiers for display or external integration, the recommended approach is to generate them at Phase 11 load time and populate the existing `NetworkEntityORM.external_id` column. No dataset rewrite required.

### Phase 11 — Pedkai Loader ✅

> Ingest all Parquet files into Pedkai's database via ORM models and API endpoints.

- **Module:** `step_11_loader/loader.py`
- **Three load modes (auto-detected):**
  1. **Database mode** (PostgreSQL) — set `PEDKAI_DATABASE_URL` env var. Uses SQLAlchemy + batched inserts with upsert semantics (`INSERT ... ON CONFLICT UPDATE`).
  2. **API mode** — set `PEDKAI_API_URL` env var. Events/alarms loaded via Pedkai's `/alarms/bulk` REST endpoint in batches of 5,000.
  3. **Dry-run mode** (default) — validates all files are readable, counts rows, calculates batch counts, reports what a real load would do. No external connections required.
- **Load plan (18 files, ordered by FK constraints):**
  1. Network topology (CMDB declared entities + relationships → database tables)
  2. Ground truth entities + relationships (optional, for scoring — separate schema)
  3. Divergence manifest (ML scoring key)
  4. KPI datasets (6 files → **registered as external Parquet datasets**, NOT exploded to long format — avoids 1.67B+ row explosion)
  5. Customer/BSS → customer table
  6. Events/alarms → API or database
  7. Vendor naming map → lookup table
  8. Neighbour relations → database
  9. Scenario manifest + overrides
- **Long-format explosion warning:** Radio KPIs alone: 47.6M rows × 35 KPI columns = **~1.67 billion** long-format rows. KPI files are registered as external Parquet datasets by default; Pedkai queries them via DuckDB/Arrow Flight. Set `PEDKAI_LOAD_LONG_FORMAT=1` to force long-format explosion (not recommended).
- **Idempotent:** Safe to re-run — upsert semantics on PK columns for all database-loaded tables.
- **Output:** `validation/load_report.json` — detailed timing, row counts, throughput per operation.
- **Dependencies:** Phase 10 (all validation must pass before loading). UUID-based files are loaded directly — Pedkai's ORM schema is UUID-native (see Phase 10.5 cancellation rationale).

---

## Current Status

| Phase | Description | Status | Output Size |
|:-----:|-------------|:------:|------------:|
| 0 | Schema Contracts | ✅ Complete | ~50 KB |
| 1 | Site & Cell Generation | ✅ Complete | 5.6 MB |
| 2 | Full Network Topology | ✅ Complete | 144 MB |
| 3 | Radio Physics & Cell KPIs | ✅ Complete | 8.5 GB |
| 4 | Multi-Domain KPIs | ✅ Complete | 2.4 GB |
| 5 | Scenario Injection | ✅ Complete | ~200 MB (overlay files) |
| 6 | Events & Alarms | ✅ Complete | ~500 MB |
| 7 | Customers & BSS | ✅ Complete | ~200 MB |
| 8 | CMDB Degradation | ✅ Complete | ~150 MB (3 files) |
| 9 | Vendor Naming | ✅ Complete | ~5 MB (lookup table) |
| 10 | Validation | ✅ Complete | (JSON reports) |
| 10.5 | ID Optimisation & Remapping | ⊘ Cancelled | — |
| 11 | Pedkai Loader | ✅ Complete | (ingestion script) |

**Total generated:** ~12.2 GB across 17 Parquet files + schemas + metadata + validation reports.

**Pipeline complete.** All implementation phases are finished. Phase 10.5 (ID Remapping) was cancelled after analysis confirmed Pedkai's ORM schema is UUID-native and the remapping would add unnecessary complexity without meaningful benefit (see Phase 10.5 section above for full rationale).

---

## Output Files

All outputs are written to `/Volumes/Projects/Pedkai Data Store/` (configurable via `--data-store`).

### Output Directory

| File | Rows | Columns | Size | Phase |
|------|-----:|--------:|-----:|:-----:|
| `schemas/*.schema.ipc` | — | — | ~50 KB | 0 |
| `generator_config.yaml` | — | — | 1 KB | — |
| `ground_truth_entities.parquet` | 811,064 | 31 | 36 MB | 2 |
| `ground_truth_relationships.parquet` | 1,970,387 | 9 | 74 MB | 2 |
| `neighbour_relations.parquet` | 926,475 | 15 | 34 MB | 2 |
| `kpi_metrics_wide.parquet` | 47,614,320 | 44 | 8.5 GB | 3 |
| `transport_kpis_wide.parquet` | 21,409,920 | 20 | 1,287 MB | 4 |
| `fixed_broadband_kpis_wide.parquet` | 4,776,480 | 19 | 400 MB | 4 |
| `enterprise_circuit_kpis_wide.parquet` | 1,440,000 | 15 | 84 MB | 4 |
| `core_element_kpis_wide.parquet` | 299,520 | 25 | 21 MB | 4 |
| `power_environment_kpis.parquet` | 15,192,000 | 12 | 640 MB | 4 |
| `scenario_manifest.parquet` | ~7,500 | 16 | ~1 MB | 5 |
| `scenario_kpi_overrides.parquet` | ~millions | 8 | ~200 MB | 5 |
| `events_alarms.parquet` | ~500,000 | 15 | ~500 MB | 6 |
| `customers_bss.parquet` | ~1,000,000 | 18 | ~200 MB | 7 |
| `cmdb_declared_entities.parquet` | ~1,440,000 | 31 | ~35 MB | 8 |
| `cmdb_declared_relationships.parquet` | ~2,100,000 | 9 | ~70 MB | 8 |
| `divergence_manifest.parquet` | ~300,000 | 13 | ~50 MB | 8 |
| `vendor_naming_map.parquet` | ~220 | 11 | ~5 KB | 9 |

### Intermediate Directory

| File | Rows | Size | Phase |
|------|-----:|-----:|:-----:|
| `sites.parquet` | 21,100 | 1.0 MB | 1 |
| `cells.parquet` | 66,131 | 4.6 MB | 1 |
| `topology_metadata.json` | — | 2.8 KB | 2 |

---

## Architecture & Memory Safety

### Streaming AR(1) Environment Generator

Phase 3 (radio KPIs) generates 47.6M rows of physics-driven KPI data. The naive approach — pre-allocating `(720 hours, 66,131 cells)` float64 matrices for load, shadow fading, interference, and UE multiplier — requires **~1.45 GB** of retained memory plus **~2.5 GB** of temporaries, totalling **~4 GB peak**. On a 16 GB machine with typical OS memory pressure, this caused a kernel panic.

The production implementation uses a **streaming AR(1) state machine**:

- **State vectors:** Four `(n_cells,)` arrays (~500 KB each) carry temporal autocorrelation forward between hours
- **Per-hour allocation:** ~2 MB for environmental conditions + ~20 MB for physics temporaries + ~30 MB for the PyArrow RecordBatch
- **Immediate flush:** Each hour's RecordBatch is written as a Parquet row group and released
- **Peak memory:** ~150-200 MB regardless of simulation length

| Metric | Bulk (crashed) | Streaming (production) |
|--------|---------------|----------------------|
| Environment memory | 1.45 GB retained | 2 MB |
| Temporaries | ~2.5 GB | ~50 MB |
| Peak total | ~4 GB | ~200 MB |
| GC strategy | Every 10 row groups | Every 24 hours + explicit `del` per hour |

### Data Type Optimization

KPI columns use **float32** instead of float64. 32-bit precision provides ~7 significant decimal digits — more than sufficient for radio PM counters (RSRP to 0.01 dB, throughput to 0.01 Mbps). This halves the dominant data cost (float columns are ~90% of the file).

### Deterministic Seeding

Every pipeline step derives its seed from the global seed via HMAC-SHA256:

```
Seed(step_id) = HMAC-SHA256(global_seed, step_id) & 0x7FFFFFFF
```

This ensures:
- Full reproducibility from a single global seed
- Step-level independence: re-running one step doesn't affect others
- Parallel execution safety: steps with no dependencies can run concurrently

---

## Project Structure

```
Sleeping-Cell-KPI-Data/
├── README.md                                    ← This file
├── THREAD_SUMMARY_SYNTHETIC_DATA_GENERATOR.md   ← Full design context (878 lines)
├── pyproject.toml                               ← Package config, dependencies, CLI entrypoint
├── src/
│   └── pedkai_generator/
│       ├── __init__.py
│       ├── cli.py                               ← CLI: pedkai-generate run/config/steps
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py                      ← All scale parameters, enums, band configs,
│       │                                           province definitions, seed management
│       ├── utils/
│       │   └── __init__.py
│       ├── step_00_schema/                      ← Phase 0: Parquet schema contracts
│       │   └── contracts.py
│       ├── step_01_sites/                       ← Phase 1: Site & cell generation
│       │   └── generate.py
│       ├── step_02_topology/                    ← Phase 2: Full network topology
│       │   ├── generate.py                        Orchestrator
│       │   ├── builders.py                        TopologyAccumulator
│       │   ├── mobile_ran.py                      Mobile RAN domain builder
│       │   ├── other_domains.py                   Transport, Fixed, Core, Logical, Power
│       │   └── neighbours.py                      Spatial neighbour relations
│       ├── step_03_radio_kpis/                  ← Phase 3: Radio physics & KPI generation
│       │   ├── __init__.py
│       │   ├── generate.py                        Orchestrator (streaming, memory-safe)
│       │   ├── physics.py                         3GPP physics engine (vectorised numpy)
│       │   └── profiles.py                        Diurnal profiles & streaming env generator
│       ├── step_04_domain_kpis/                 ← Phase 4: Multi-domain KPIs
│       │   └── generate.py                        5 domains: transport, fixed BB, enterprise, core, power
│       ├── step_05_scenarios/                   ← Phase 5: Scenario injection
│       │   └── generate.py                        8 scenario types, overlay strategy, cascade walking
│       ├── step_06_events/                      ← Phase 6: Events & alarms
│       │   └── generate.py                        Scenario-driven + organic alarms, cross-domain correlation
│       ├── step_07_customers/                   ← Phase 7: Customers & BSS (stub)
│       ├── step_08_cmdb_degradation/            ← Phase 8: CMDB degradation (stub)
│       ├── step_09_vendor_naming/               ← Phase 9: Vendor naming (stub)
│       ├── step_10_validation/                  ← Phase 10: Validation (stub)
│       ├── step_10b_id_remap/                   ← Phase 10.5: ID optimisation & remapping (stub)
│       └── step_11_loader/                      ← Phase 11: Pedkai loader (stub)
├── cell_1_KPI_Data.csv                          ← Original sleeping cell dataset (preserved)
├── cell_2_KPI_Data.csv
├── cell_3_KPI_Data.csv
└── LICENSE
```

---

## Installation

```bash
# Clone the repo
git clone https://github.com/your-org/Sleeping-Cell-KPI-Data.git
cd Sleeping-Cell-KPI-Data

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Verify
pedkai-generate --version
pedkai-generate steps
```

### Requirements

- Python ≥ 3.11
- ~16 GB RAM (streaming architecture keeps peak at ~200 MB, but downstream consumers may need more)
- ~15 GB disk space for full output
- External volume at `/Volumes/Projects/Pedkai Data Store/` (or override with `--data-store`)

---

## Usage

### CLI Commands

```bash
# Show all pipeline steps and dependencies
pedkai-generate steps

# Show current configuration
pedkai-generate config --show

# Save configuration to YAML
pedkai-generate config --save my_config.yaml

# Run the full pipeline (all 13 steps)
pedkai-generate run --all

# Run a single step
pedkai-generate run --step 3

# Run multiple specific steps (with auto-dependency resolution)
pedkai-generate run --step 3 --step 4

# Run from a specific step onwards
pedkai-generate run --from-step 3

# Dry run (show execution plan without running)
pedkai-generate run --all --dry-run

# Override parameters
pedkai-generate run --step 3 --seed 12345 --days 7 --data-store /tmp/output

# Use a custom config file
pedkai-generate run --all --config-file my_config.yaml
```

### Quick Start (Run Phases 0-3)

```bash
# Generate everything up to radio KPIs
pedkai-generate run --step 0 --step 1 --step 2 --step 3

# Or equivalently (dependency resolution pulls in 0-2 automatically)
pedkai-generate run --step 3

# For a quick test run (2 days instead of 30)
pedkai-generate run --step 3 --days 2
```

### Programmatic Usage

```python
from pedkai_generator.config.settings import GeneratorConfig
from pedkai_generator.step_03_radio_kpis.generate import generate_radio_kpis

config = GeneratorConfig()
config.simulation.simulation_days = 7  # 1-week test
config.ensure_output_dirs()

generate_radio_kpis(config)
```

---

## Phase Dependency Graph

```
Phase 0 (Schema)
  └─► Phase 1 (Sites & Cells)
        └─► Phase 2 (Topology)
              ├─► Phase 3 (Radio KPIs) ──────────────────┐
              ├─► Phase 4 (Domain KPIs) ─────────────────┤
              ├─► Phase 7 (Customers & BSS)              │
              └─► Phase 8 (CMDB Degradation)             │
                                                         ▼
                                            Phase 5 (Scenario Injection) ◄── Phase 2
                                                         │
                                                         ▼
                                            Phase 6 (Events & Alarms)
                                                         │
              Phase 3 + Phase 4 ─────────► Phase 9 (Vendor Naming)
                                                         │
                                                         ▼
              All (0–9) ───────────────► Phase 10 (Validation)
                                                         │
                                                         ▼
                                          Phase 11 (Pedkai Loader)
```

---

## Design References

The full design context — including source codebase review, Pedkai ingestion pathway analysis, complete topology model, CMDB reconciliation strategy, and ML considerations — is documented in:

- **[`THREAD_SUMMARY_SYNTHETIC_DATA_GENERATOR.md`](./THREAD_SUMMARY_SYNTHETIC_DATA_GENERATOR.md)** — 878-line design document covering all decisions

### Key 3GPP Standards Referenced

| Standard | Use |
|----------|-----|
| TS 36.213 Table 7.2.3-1 | LTE CQI → spectral efficiency mapping |
| TS 38.214 Table 5.2.2.1-3 | NR CQI → spectral efficiency (256QAM) |
| TS 36.942 | Propagation models |
| TR 38.901 | 5G channel models |
| ITU-R P.1238 | Indoor propagation |
| TMF642 | Alarm model structure |

### Pedkai Integration Points

| Pedkai Component | Fed By |
|-----------------|--------|
| `KPIMetricORM` / `KpiSampleORM` | Wide-format KPI Parquet files (pivoted to long format by loader) |
| `NetworkEntityORM` | `cmdb_declared_entities.parquet` |
| `EntityRelationshipORM` | `cmdb_declared_relationships.parquet` |
| `CustomerORM` / `BillingAccountORM` | `customers_bss.parquet` |
| Alarm ingestion API (TMF642) | `events_alarms.parquet` |
| Dark Graph scoring | `divergence_manifest.parquet` + ground truth files |
| Causal templates | Scenario injection aligns with `fiber_cut`, `prb_utilization_congestion`, `power_failure`, `backhaul_congestion` templates |

---

## Original Dataset

This repository originally contained sleeping cell KPI data from the following papers:

1. Zhao Ming et al., "Sleeping Cell Detection for Resiliency Enhancements in 5G/B5G Mobile Edge-Cloud Computing Networks", *ACM TOSN*, vol. 18, no. 3, 2022.
2. Zhao Ming et al., "Ensemble Learning Based Sleeping Cell Detection in Cloud Radio Access Networks", *IEEE ISCC*, 2020.

The original 3 CSV files (`cell_1_KPI_Data.csv`, `cell_2_KPI_Data.csv`, `cell_3_KPI_Data.csv`) containing 11 days of 36 KPIs for 3 cells are preserved in the repo root.