# Thread Summary: Pedkai Synthetic Data Generator — Full Design Context

**Date:** 2025  
**Purpose:** This document captures the complete design context for building a synthetic telecom data generator based on the `telecom-digital-twin` codebase (github.com/adityonugrohoid/telecom-digital-twin), extended to serve Pedkai's full product vision. It is intended to allow seamless continuation of work in a new thread.

---

## Table of Contents

1. [Source Codebase Review](#1-source-codebase-review)
2. [Design Decisions](#2-design-decisions)
3. [Scale Parameters](#3-scale-parameters)
4. [Pedkai Codebase Analysis](#4-pedkai-codebase-analysis)
5. [Complete Telecom Topology Model](#5-complete-telecom-topology-model)
6. [CMDB Reconciliation / Dark Graph Vision](#6-cmdb-reconciliation--dark-graph-vision)
7. [Final Output Specification](#7-final-output-specification)
8. [Size Estimates](#8-size-estimates)
9. [ML Considerations](#9-ml-considerations)
10. [Open Items / Next Steps](#10-open-items--next-steps)

---

## 1. Source Codebase Review

### telecom-digital-twin Architecture

The source codebase (github.com/adityonugrohoid/telecom-digital-twin) is a 7-step pipeline:

```
STEP 01 → Schema contracts (docs only)
STEP 02 → cells.parquet (SDV GaussianCopula synthesis)
STEP 03 → users.parquet (SDV GaussianCopula synthesis)
STEP 04 → user_behavior.parquet + network_behavior.parquet (session generation + cell hourly KPIs)
STEP 05 → sessions.parquet (KPI derivation: throughput, latency, jitter, packet loss, QoE)
STEP 06 → events.parquet (congestion, outage, handover_fail, complaint)
STEP 07 → Validation (schema, FK integrity, range checks)
```

### Critical Findings

1. **NOT vendor-specific**: Zero Huawei, Ericsson, or Nokia logic anywhere. Completely vendor-agnostic. The concern about Huawei bias is unfounded.

2. **LTE-only and superficial**: Operates at Layer 7 (application experience) not Layer 1-2 (radio). KPI model is `dl_throughput = max_dl × (1 - utilization^1.7) × lognormal_noise` — a black-box approximation that skips the entire radio chain. Missing: RACH, RRC, PRB utilisation, MCS/CQI, BLER, RLC retransmission, interference, PDCCH/CCE, handover counters, cell availability, active UEs, paging.

3. **Indonesia geography**: Hardcoded to 38 Indonesian provinces across 3 timezones (WIB/WITA/WIT). Decision: **keep it** — see below.

### What to Keep from the Source

- Pipeline architecture (7-step, causality-first)
- Deterministic seeding strategy (`Seed(Step_N) = F(Global_Seed, Step_ID)`)
- SDV GaussianCopulaSynthesizer pattern for correlated data generation
- Parquet-first storage approach
- STEP 07 validation framework (extended for new schemas)
- CLI orchestration, progress display
- Overall project structure and pyproject.toml setup

### What to Rebuild

- KPI generation engine — replace application-level model with radio-layer physics model
- Cell schema — add RAT type, deployment profile, vendor profile, NR bands
- Event system — replace hardcoded events with configurable scenario injection
- Add vendor naming/mapping layer (Ericsson/Nokia PM counter names)
- Add RAT profiles (LTE, NR-NSA, NR-SA physics differences)

---

## 2. Design Decisions

### Geography: Keep Indonesia

Indonesia is **better** than UK/Europe for synthetic data because:
- Scale and diversity: 38 provinces, 3 timezones, population density from hyper-dense Jakarta (15,000+ people/km²) to remote Papua
- Full spectrum of cell behaviours: urban micro-cells, suburban macros, rural sparse, coverage-edge
- Existing codebase already works with Indonesian geography
- Vendor-neutrality: nobody will reverse-map synthetic Indonesian data to a real operator
- Algorithm portability: sleeping cell detection, anomaly detection, etc. work on KPI patterns, not geography

**Addition**: A `deployment_profile` concept captures what geography is actually a proxy for:
- `dense_urban`: high site density, small ISD, many micro-cells, high capacity backhaul
- `urban`: medium density, mixed macro/micro
- `suburban`: macro-dominant, moderate backhaul
- `rural`: sparse sites, large ISD, limited backhaul
- `deep_rural`: very sparse, coverage-driven, satellite/microwave backhaul
- `indoor`: DAS/small-cell, venue-specific

### Not a Sleeping Cell Simulator

The goal is a **general-purpose RAN performance management dataset** — the kind of data Ericsson ENM or Nokia NetAct would export. Sleeping cell analysis is one of many downstream use cases alongside:
- Capacity planning / congestion prediction
- General anomaly detection
- QoE analytics
- RAN optimisation
- Coverage analysis
- Handover optimisation
- 5G network slicing analytics
- **CMDB reconciliation / Dark Graph inference** (Pedkai's core differentiator)

Sleeping cells are injected as one scenario profile among many (congestion, coverage holes, hardware faults, interference, etc.).

### Vendor Support

The internal physics model generates vendor-neutral KPIs. A final mapping layer renames them to:
- Ericsson PM counter names (`pmRrcConnEstabSucc`, etc.)
- Nokia counter names (`RRC_CONN_SETUP_SUCC`, etc.)

One physics model, configurable output naming.

### RAT Support

Three RAT types coexist in one dataset (reflecting real networks):
- **LTE** (EUTRA cells on L900, L1800, L2100, L2300 bands)
- **5G NSA** (EN-DC: LTE anchor + NR secondary cell group, NR bands n1, n3, n28, n78)
- **5G SA** (standalone NR with 5GC, NR bands n77, n78, n257, n258)

5G NSA cells generate **two KPI streams** (one for LTE anchor leg, one for NR SCG leg).

---

## 3. Scale Parameters

### UK-Scale Operator Profile

Based on a real UK network with ~20,000 live sites:

| Site Type | Count | Sectors/Site | Cells | Deployment Profile |
|-----------|-------|-------------|-------|--------------------|
| Greenfield | 11,000 | 3 | 33,000 | suburban, rural, deep_rural |
| Rooftop | 4,000 | 3 | 12,000 | urban, dense_urban |
| Streetworks | 5,500 | 1 | 5,500 | dense_urban, urban (small cells, single-sector) |
| In-Building | 500 | 2 | 1,000 | indoor (DAS/small cells) |
| Unspecified | 100 | 2 | 200 | mixed |
| **Total** | **21,100** | | **51,700** | |

### RAT Split

| RAT | Cells | Logic |
|-----|-------|-------|
| LTE-only | ~32,000 | Most greenfield (rural/deep rural), some suburban, unspecified |
| LTE + 5G NSA | ~13,000 | Suburban greenfield, most rooftop, some streetworks |
| 5G SA | ~6,700 | Dense urban streetworks, some rooftop, enterprise in-building |
| **Total cells** | **~51,700** | |
| **+ NSA NR SCG legs** | **+13,000** | Each NSA cell = 2 KPI streams |
| **Logical cell-layers** | **~64,700** | |

### User Scale

- Real UK operator: 20,000,000 subscribers
- **Recommended (Option B)**: 1,000,000 sampled users for session-level data
- Cell-level KPIs cover all 51,700 cells regardless
- Option B buys nothing less than Option A for any ML task (cell-level KPIs are the bottleneck, not user count)

### Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Simulation period | 30 days |
| Reporting interval | 1 hour (standard PM granularity) |
| Cell KPI rows | 64,700 cell-layers × 24h × 30d = 46,584,000 |
| KPI columns | ~50 radio-layer PM counters |

---

## 4. Pedkai Codebase Analysis

### Data Ingestion Pathways

Pedkai ingests data through 5 distinct pathways. The synthetic generator must feed all of them:

#### Pathway 1: KPI Time-Series via `KPIMetricORM` (Hot Path)

Long-format (EAV) table: one row per entity per metric per timestamp.

```python
# backend/app/models/kpi_orm.py
tenant_id = Column(String(255), primary_key=True)
entity_id = Column(String(255), primary_key=True)
timestamp = Column(DateTime(timezone=True), primary_key=True)
metric_name = Column(String(100), primary_key=True)
value = Column(Float, nullable=False)
tags = Column(JSONB, nullable=False, default=dict)
```

Used by: anomaly detector, causal analyzer, sleeping cell detector, capacity engine, drift calibration.

**Pedkai is metric-name agnostic by design** — all analysis services work on any `metric_name`. But must include existing referenced names as aliases.

Metric names currently used across Pedkai codebase:
- `throughput_mbps`, `latency_ms`, `prb_utilization` / `prb_utilization_pct`
- `traffic_volume`, `active_users_count`, `volte_cdr_pct`
- `data_throughput_gbps`, `packet_loss_pct`

LiveTestData numeric KPIs (real RAN-level):
- `RSRP`, `DL_BLER`, `DL_MCS`, `UL_BLER`, `UL_MCS`, `UL_NPRB`, `UL_SNR`
- `TX_Bytes`, `RX_Bytes`, `Estimated_UL_Buffer`
- `PRBs_DL_Current`, `PRBs_UL_Current`, `PRB_Utilization_DL`, `PRB_Utilization_UL`
- `UL_NumberOfPackets`, `DL_NumberOfPackets`

#### Pathway 2: KPI Samples via `KpiSampleORM` (Structured Path)

```python
# backend/app/models/kpi_sample_orm.py
entity_id = Column(UUID, ForeignKey("network_entities.id"))
metric_name = Column(String(100))
value = Column(Float)
timestamp = Column(DateTime(timezone=True))
source = Column(String(50))  # 'RAN_TELEMETRY', 'SYNTHETIC_TEST', etc.
```

Linked to `NetworkEntityORM` via FK. Generator should set `source = 'SYNTHETIC_DIGITAL_TWIN'`.

#### Pathway 3: Network Topology via `EntityRelationshipORM`

```python
# backend/app/models/topology_models.py
from_entity_id = Column(String(255))
from_entity_type = Column(String(50))
relationship_type = Column(String(50))
to_entity_id = Column(String(255))
to_entity_type = Column(String(50))
tenant_id = Column(String(50))
properties = Column(String)  # JSON
last_synced_at = Column(DateTime(timezone=True))
```

Network entities:
```python
# backend/app/models/network_entity_orm.py
id = Column(UUID, primary_key=True)
tenant_id = Column(String(50))
entity_type = Column(String(50))  # SITE, GNODEB, CELL, SECTOR, ROUTER, SWITCH, etc.
name = Column(String(255))
external_id = Column(String(255))  # Vendor NMS identifier
geo_lat = Column(Float)
geo_lon = Column(Float)
revenue_weight = Column(Float)
sla_tier = Column(String(50))  # GOLD, SILVER, BRONZE
```

#### Pathway 4: Alarms via TMF642 / Alarm Ingestion

```python
# backend/app/api/alarm_ingestion.py
entity_id: str
alarm_type: str       # LINK_DOWN, DEGRADATION, CELL_DEGRADATION, POWER_SUPPLY
severity: str         # minor, major, critical
raised_at: datetime
source_system: str    # oss_vendor, snmp, ericsson_enm, nokia_netact
```

Alarm normalizer already handles Ericsson XML and Nokia JSON formats (`data_fabric/alarm_normalizer.py`).

#### Pathway 5: Customer / BSS Data

```python
# backend/app/models/customer_orm.py
external_id = Column(String(100), unique=True)
name = Column(String(255))
churn_risk_score = Column(Float)  # 0.0-1.0
associated_site_id = Column(String(255))
consent_proactive_comms = Column(Boolean)

# backend/app/models/bss_orm.py
# ServicePlanORM: name, tier (GOLD/SILVER/BRONZE), monthly_fee, sla_guarantee
# BillingAccountORM: customer_id, plan_id, account_status, avg_monthly_revenue
```

### Key Pedkai Services That Consume Data

| Service | Primary Table | What It Does |
|---------|--------------|--------------|
| `SleepingCellDetector` | `KpiSampleORM` | Z-score on `traffic_volume` vs 7-day baseline |
| `AnomalyDetector` | `KPIMetricORM` | Z-score on any metric vs 24h baseline |
| `CausalAnalyzer` | `KPIMetricORM` | Granger causality tests between metric pairs |
| `CapacityEngine` | `KPIMetricORM` | PRB utilization >85% → congestion hotspots |
| `AlarmCorrelationService` | `DecisionTraceORM` | Temporal + spatial alarm clustering |
| `RootCauseAnalyzer` | `EntityRelationshipORM` | Graph traversal for upstream/downstream impact |
| `CXIntelligenceService` | `CustomerORM` + topology | Recursive graph traversal for customer impact |
| `DigitalTwinMock` | `DecisionTraceORM` | Semantic similarity for action prediction |
| `DriftCalibrationService` | `DecisionTraceORM` | False positive rate tracking |
| `CausalModelLibrary` | `causal_templates.yaml` | Expert-defined causal patterns |

### TMF API Compliance

- **TMF628 Performance Management**: `/tmf-api/performanceManagement/v4/performanceMeasurement` — reads from `KPIMetricORM`
- **TMF642 Alarm Management**: `/tmf-api/alarmManagement/v4/alarm` — reads/writes `DecisionTraceORM`, maps to TMF alarm model

### Causal Templates (must align with)

```yaml
# backend/app/data/causal_templates.yaml
causal_templates:
  - id: fiber_cut
    cause_metric: "fiber_status"
    effect_metric: "gnodeb_connectivity"
    entity_type_pair: ["FIBER", "GNODEB"]
  - id: prb_utilization_congestion
    cause_metric: "prb_utilization"
    effect_metric: "dl_throughput"
    entity_type_pair: ["CELL", "CELL"]
  - id: power_failure
    cause_metric: "mains_power"
    effect_metric: "bbu_status"
    entity_type_pair: ["SITE", "BBU"]
  - id: backhaul_congestion
    cause_metric: "transport_load"
    effect_metric: "latency"
    entity_type_pair: ["ROUTER", "GNODEB"]
```

### Compatibility Requirements

1. **Tenant isolation**: Every row must have `tenant_id`
2. **Entity ID consistency**: Topology entity IDs must match KPI entity IDs
3. **Metric names**: Must include Pedkai's existing referenced names as aliases
4. **Timestamps**: ISO 8601 with timezone (UTC)
5. **Alarm format**: Must match `AlarmIngestionRequest` schema
6. **FK integrity**: `KpiSampleORM.entity_id` → `NetworkEntityORM.id`

---

## 5. Complete Telecom Topology Model

### Design Principle

The topology must model a **converged operator** — mobile, fixed broadband, enterprise data services, and voice across a shared transport and core infrastructure. Pedkai is not limited to mobile operators; it must serve landline/broadband providers, fibre companies, and enterprise service providers. The transport layer is the convergence point.

### Physical Layer (Mobile RAN)

```
SITE (physical location — greenfield, rooftop, streetworks, in-building, temp)
├── CABINET (outdoor housing)
│   ├── POWER_SUPPLY (rectifier, battery backup, generator)
│   │   ├── BATTERY_BANK
│   │   └── MAINS_CONNECTION
│   ├── CLIMATE_CONTROL (cooling unit)
│   └── TRANSMISSION_EQUIPMENT
│       ├── MICROWAVE_LINK (point-to-point backhaul)
│       └── FIBER_TERMINATION
├── ANTENNA_SYSTEM
│   ├── ANTENNA (physical panel — shared across RATs)
│   ├── RRU / RRH (Remote Radio Unit/Head — one per band per sector)
│   └── FEEDER_CABLE / FIBER_JUMPER
├── BASEBAND_UNIT (BBU / DU — Ericsson Baseband 6630, Nokia ASIQ)
│   ├── ENODEB (LTE logical node hosted on BBU)
│   │   └── LTE_CELL (EUTRA Cell — one per sector per band)
│   └── GNODEB (NR logical node hosted on BBU or separate CU/DU)
│       ├── GNODEB_DU (Distributed Unit)
│       ├── GNODEB_CU_CP (Central Unit — Control Plane)
│       ├── GNODEB_CU_UP (Central Unit — User Plane)
│       └── NR_CELL (one per sector per NR band)
└── GPS_RECEIVER (timing source)
```

### Fixed Broadband Access

```
FTTP / FTTH:
├── EXCHANGE_BUILDING (hosts OLT equipment)
│   └── OLT (Optical Line Terminal — e.g. Nokia ISAM, Huawei MA5800)
│       └── PON_PORT (GPON/XGS-PON — 2.5G/10G downstream)
│           └── SPLITTER (1:32 or 1:64 passive optical splitter)
│               └── ONT / ONU (customer premises — one per household)
│                   └── RESIDENTIAL_SERVICE (broadband subscription)
├── PCP (Primary Cross-connect Point — street cabinet, passive)
│   └── DP (Distribution Point — final passive splice)
└── FIBRE_SPAN (physical fibre segment between nodes)

FTTC (legacy, large installed base):
├── EXCHANGE_BUILDING → DSLAM / MSAN
│   └── COPPER_LINE → STREET_CABINET (VDSL vectoring)
│       └── COPPER_PAIR → CUSTOMER_PREMISES
│           └── RESIDENTIAL_SERVICE

Enterprise Ethernet Access:
├── NTE / NTU (Network Termination Equipment at customer site)
│   └── ETHERNET_CIRCUIT (dedicated point-to-point or multipoint)
│       └── ENTERPRISE_SERVICE (e.g., 1Gbps Ethernet, 10Gbps)
├── Access can be via:
│   ├── DARK_FIBRE → directly into operator's aggregation
│   ├── ACTIVE_ETHERNET → managed L2 service
│   └── CARRIER_ETHERNET (MEF standards: E-Line, E-LAN, E-Tree)

Legacy Voice (PSTN — declining but operational):
├── EXCHANGE_BUILDING → CLASS_5_SWITCH / SOFTSWITCH
│   └── COPPER_LINE → CUSTOMER → VOICE_LINE_SERVICE
└── Being migrated to VoIP/SIP over broadband
```

### Transport Layer (Shared Backbone — Convergence Point)

Mobile backhaul, broadband aggregation, and enterprise circuits all share this:

```
PHYSICAL FIBRE LAYER:
├── FIBRE_CABLE (physical cable in duct/overhead)
│   ├── FIBRE_PAIR (individual fibre strand pair — tx/rx)
│   └── FIBRE_SPAN (segment between splice points)
├── DUCT (underground conduit carrying cables)
│   └── DUCT_SECTION (between manholes/chambers)
├── MANHOLE / CHAMBER (access point to duct network)
├── ODF (Optical Distribution Frame — fibre patching)
└── DWDM_SYSTEM (wavelength multiplexing)
    └── OPTICAL_CHANNEL (individual wavelength / lambda)

IP / ETHERNET LAYER:
├── ACCESS_SWITCH (L2 — at exchange or mobile site)
│   └── Connects: OLT uplinks, mobile site routers, enterprise NTEs
├── AGGREGATION_SWITCH (L2/L3 — metro aggregation)
│   └── Aggregates multiple access switches
├── PE_ROUTER (Provider Edge — L3 MPLS/IP boundary, VRFs live here)
├── P_ROUTER (Provider / Core — pure MPLS label switching)
└── ROUTE_REFLECTOR (BGP RR — logical)

MPLS / VPN LAYER (logical overlay):
├── L3VPN / VRF (Layer 3 VPN instance)
│   ├── MOBILE_BACKHAUL_VRF (carries S1/NG/X2/Xn for RAN)
│   ├── BROADBAND_VRF (carries residential broadband to BNG)
│   ├── ENTERPRISE_VPN (customer-dedicated L3VPN)
│   └── MANAGEMENT_VRF (OAM for all network elements)
├── L2VPN / VPLS / EVPN (Layer 2 services)
│   ├── E_LINE (point-to-point Ethernet)
│   ├── E_LAN (multipoint Ethernet)
│   └── E_TREE (rooted multipoint)
├── PSEUDOWIRE (point-to-point L2 tunnel within MPLS)
└── LSP (Label Switched Path — traffic engineered)

SERVICE EDGE:
├── BNG / BRAS (Broadband Network Gateway — terminates PPPoE/IPoE)
├── CGNAT (Carrier-Grade NAT)
├── FIREWALL / DPI
└── CDN_NODE (Content Delivery Network cache)
```

### Core Network Layer

```
EPC (4G Core):
├── MME (Mobility Management Entity)
├── SGW (Serving Gateway)
├── PGW (PDN Gateway)
└── HSS (Home Subscriber Server)

5GC (5G Core — SA only):
├── AMF (Access and Mobility Management Function)
├── SMF (Session Management Function)
├── UPF (User Plane Function)
├── NSSF (Network Slice Selection Function)
├── PCF (Policy Control Function)
├── UDM (Unified Data Management)
└── NWDAF (Network Data Analytics Function)

IMS (Voice):
├── P_CSCF (Proxy Call Session Control)
├── S_CSCF (Serving CSCF)
├── TAS (Telephony Application Server — VoLTE/VoNR)
└── MGCF (Media Gateway Control — CS fallback)

Broadband Service Control:
├── BNG / BRAS
├── RADIUS / DIAMETER_SERVER (AAA)
├── DHCP_SERVER
├── DNS_RESOLVER
└── POLICY_SERVER

Enterprise Service Control:
├── CE_ROUTER (Customer Edge — at enterprise premises)
├── PE_ROUTER (Provider Edge)
├── SD_WAN_CONTROLLER
└── FIREWALL_SERVICE

Voice (converged):
├── IMS_CORE (VoLTE, VoNR, VoWiFi)
├── SOFTSWITCH (PSTN replacement)
├── SBC (Session Border Controller)
├── MEDIA_GATEWAY (TDM-to-IP)
└── SIP_TRUNK (enterprise voice)
```

### Service Layer

```
NETWORK_SLICE (5G SA only):
├── eMBB_SLICE, URLLC_SLICE, mMTC_SLICE

SERVICE_AREA (geographic coverage zone)
├── TRACKING_AREA (groups of cells for paging)
└── ROUTING_AREA (legacy 2G/3G)

QOS_PROFILE:
├── QCI / 5QI mapping
└── ARP (Allocation and Retention Priority)
```

### Cell Adjacency / Neighbour Relationships

```
CELL ↔ CELL neighbour relations:
├── INTRA_FREQ_NEIGHBOUR (same band, same RAT)
├── INTER_FREQ_NEIGHBOUR (different band, same RAT)
├── INTER_RAT_NEIGHBOUR (LTE ↔ NR for NSA handover)
└── INTRA_SITE vs INTER_SITE

Each neighbour relation has:
├── handover_attempts (measured)
├── handover_success_rate (measured)
├── cio_offset_db (Cell Individual Offset — configurable)
└── no_remove_flag (operator-locked neighbour)
```

### Complete Relationship Types

| Relationship | From → To | Domain |
|-------------|-----------|--------|
| `HOSTS` | SITE → BBU, BBU → ENODEB, GNODEB → NR_CELL, EXCHANGE → OLT | Physical |
| `POWERS` | POWER_SUPPLY → CABINET/RACK | Physical |
| `COOLS` | CLIMATE_CONTROL → CABINET/ROOM | Physical |
| `CONTAINS` | DUCT → FIBRE_CABLE, CABINET → EQUIPMENT | Physical |
| `CONNECTS_FIBRE` | ODF → ODF (physical fibre path) | Physical |
| `BACKHAULS` | FIBRE_LINK/MW_LINK → SITE | Transport |
| `UPLINKS_TO` | ACCESS_SWITCH → AGG_SWITCH, OLT → AGG_SWITCH | Transport |
| `AGGREGATES` | AGG_SWITCH → ACCESS_SWITCH (1:N) | Transport |
| `PEERS_WITH` | PE_ROUTER ↔ PE_ROUTER (BGP/MPLS) | Transport |
| `ROUTES_THROUGH` | LSP path through P_ROUTERs | Transport |
| `MEMBER_OF_VRF` | PE_ROUTER_INTERFACE → L3VPN | Service |
| `TERMINATES` | BNG → BROADBAND_SESSION, PE → ENTERPRISE_VPN | Service |
| `SPLITS_TO` | PON_PORT → SPLITTER → ONT (FTTP tree) | Access |
| `SERVES_LINE` | ONT → RESIDENTIAL_SERVICE, NTE → ENTERPRISE_SERVICE | Access |
| `NEIGHBOURS` | CELL ↔ CELL | Radio |
| `ANCHORS` | LTE_CELL → NR_CELL (EN-DC) | Radio |
| `DEPENDS_ON` | ENODEB → MME, GNODEB → AMF | Control |
| `BEARER_TO` | ENODEB → SGW, GNODEB → UPF, BNG → PGW | User plane |
| `AUTHENTICATES_VIA` | BNG → RADIUS, ONT → RADIUS | AAA |
| `MANAGED_BY` | Any → NMS_DOMAIN | Operations |
| `COVERED_BY` | CUSTOMER → SLA | Commercial |
| `SUBSCRIBES_TO` | CUSTOMER → SERVICE | Commercial |
| `CARRIED_OVER` | L3VPN → LSP, E_LINE → PSEUDOWIRE | Logical-physical mapping |
| `MEMBER_OF` | CELL → TRACKING_AREA, CELL → NETWORK_SLICE | Logical grouping |
| `TIMING_FROM` | BBU → GPS_RECEIVER | Synchronisation |

### KPIs by Domain

#### Cell-Level Radio KPIs (~35 metrics per cell per hour)

| KPI | Unit | Purpose |
|-----|------|---------|
| `rsrp_dbm` | dBm | Reference signal received power |
| `rsrq_db` | dB | Reference signal received quality |
| `sinr_db` | dB | Signal to interference + noise ratio |
| `cqi_mean` | 0-15 | Channel Quality Indicator |
| `mcs_dl` / `mcs_ul` | 0-28 | Modulation and Coding Scheme |
| `dl_bler_pct` / `ul_bler_pct` | % | Block Error Rate |
| `prb_utilization_dl` / `prb_utilization_ul` | % | Physical Resource Block usage |
| `rach_attempts` | count | Random Access Channel attempts |
| `rach_success_rate` | % | RACH success |
| `rrc_setup_attempts` / `rrc_setup_success_rate` | count/% | Radio Resource Control |
| `dl_throughput_mbps` / `ul_throughput_mbps` | Mbps | Throughput |
| `latency_ms` / `jitter_ms` / `packet_loss_pct` | ms/ms/% | QoS |
| `active_ue_avg` / `active_ue_max` | count | Connected users |
| `traffic_volume_gb` | GB | Total traffic |
| `dl_rlc_retransmission_pct` / `ul_rlc_retransmission_pct` | % | Retransmissions |
| `ho_attempt` / `ho_success_rate` | count/% | Handover |
| `cell_availability_pct` | % | Uptime |
| `interference_iot_db` | dB | Interference over thermal |
| `paging_discard_rate` | % | Paging failures |
| `cce_utilization_pct` | % | PDCCH control channel usage |

**Pedkai-compatible aliases** (ensuring existing code paths work):
- `throughput_mbps` = `dl_throughput_mbps`
- `prb_utilization` = mean of dl + ul
- `traffic_volume` = `traffic_volume_gb`
- `active_users_count` = `active_ue_avg`
- `volte_cdr_pct` (for voice core entities)

#### Fixed Broadband KPIs (~14 metrics per OLT/ONT)

| KPI | Unit |
|-----|------|
| `pon_rx_power_dbm` / `pon_tx_power_dbm` | dBm |
| `pon_ber` | ratio |
| `olt_port_utilization_pct` | % |
| `ont_uptime_pct` | % |
| `broadband_sync_rate_down_mbps` / `broadband_sync_rate_up_mbps` | Mbps |
| `broadband_throughput_down_mbps` / `broadband_throughput_up_mbps` | Mbps |
| `broadband_latency_ms` | ms |
| `broadband_packet_loss_pct` | % |
| `pppoe_session_count` | count |
| `dhcp_lease_failures` | count |
| `dns_query_latency_ms` | ms |

#### Transport / MPLS KPIs (~15 metrics per link/interface)

| KPI | Unit |
|-----|------|
| `interface_utilization_in_pct` / `interface_utilization_out_pct` | % |
| `interface_errors_in` / `interface_errors_out` | count |
| `interface_discards_in` / `interface_discards_out` | count |
| `optical_rx_power_dbm` / `optical_snr_db` | dBm / dB |
| `lsp_utilization_pct` / `lsp_latency_ms` | % / ms |
| `bgp_prefixes_received` / `bgp_session_flaps` | count |
| `microwave_modulation` | enum |
| `microwave_capacity_mbps` / `microwave_availability_pct` | Mbps / % |

#### Enterprise Service KPIs (~10 metrics per circuit/VPN)

| KPI | Unit |
|-----|------|
| `circuit_availability_pct` | % |
| `circuit_throughput_in_mbps` / `circuit_throughput_out_mbps` | Mbps |
| `circuit_latency_ms` / `circuit_jitter_ms` / `circuit_packet_loss_pct` | ms / ms / % |
| `vpn_prefix_count` | count |
| `sla_breach_count` | count |
| `cos_queue_drops` | count |

#### Power/Environment KPIs (~5 metrics per site)

Mains power status, battery voltage, cabinet temperature, etc.

### Entity Counts (UK-Scale Converged Operator)

| Domain | Entity Types | Entity Count | Relationship Count |
|--------|-------------|-------------|-------------------|
| Mobile RAN | SITE, BBU, ENODEB, GNODEB, LTE_CELL, NR_CELL, ANTENNA, RRU | ~170,000 | ~400,000 |
| Fixed Access | EXCHANGE, OLT, PON_PORT, SPLITTER, CABINET_FTTC, ONT (sampled to 100K) | ~120,000 | ~350,000 |
| Transport | FIBRE_LINK, MW_LINK, SITE_ROUTER, ACCESS_SWITCH, AGG_SWITCH, PE_ROUTER, P_ROUTER | ~50,000 | ~150,000 |
| Core | MME, SGW, PGW, AMF, UPF, SMF, BNG, IMS entities | ~200 | ~2,000 |
| Logical/Service | L3VPN, E_LINE, LSP, TRACKING_AREA, NETWORK_SLICE, SERVICE_AREA | ~10,000 | ~50,000 |
| Power/Environment | POWER_SUPPLY, CLIMATE_CONTROL, BATTERY | ~40,000 | ~60,000 |
| Customers (sampled) | RESIDENTIAL_CUSTOMER, ENTERPRISE_CUSTOMER, SERVICE | ~1,100,000 | ~1,200,000 |
| **Total** | | **~1,490,000** | **~2,210,000** |

### Cross-Domain Cascade Scenarios

The shared transport layer enables cross-domain impact analysis:

- **Fibre cut**: Cells lose backhaul + ONTs lose broadband + enterprise circuits breach SLA — all from one physical event
- **Power failure at exchange**: OLTs go down + DSLAM dies + co-located mobile backhaul switch dies
- **Core router failure**: All VRFs affected — mobile backhaul, broadband BNG sessions, enterprise VPNs simultaneously
- **BGP session flap**: Routing instability across all services on that PE-PE path

---

## 6. CMDB Reconciliation / Dark Graph Vision

### Pedkai's Core Differentiator

Pedkai's **latent topology inference engine** (the "Dark Graph") is a primary value proposition. It reconciles observed telemetry against the declared CMDB state to discover discrepancies. This capability is:

1. **Already partially demonstrated** via the CasinoLimit simulation (cybersecurity CTF dataset — non-telecom)
2. **Designed to be domain-agnostic** — works on any infrastructure, not just telecom
3. **Sellable to ServiceNow/BMC customers** — whose CMDBs are static, manually maintained, and typically 60-70% accurate
4. **A security use case** — discovering dark nodes, lateral movement paths, undocumented SSH tunnels

### The Six Types of CMDB Divergence

From `IMPLEMENTATION_ROADMAP_V3.md`:

| Type | Description | Example |
|------|-------------|---------|
| **Dark Nodes** | Entities that exist and emit telemetry but have no CMDB record | Unplanned VNF scale-out, emergency hardware swap, shadow infrastructure |
| **Phantom Nodes** | CMDB entries that no longer exist in reality | Decommissioned cell site never cleaned from CMDB, scaled-down VNF |
| **Identity Mutations** | Same logical function, different physical entity | Hardware swap with new serial number, container migration between hosts, IP renumbering |
| **Dark Edges** | Real connections not recorded in CMDB | Undocumented backhaul reroute, temporary cross-connect, traffic flow between entities with no declared relationship |
| **Phantom Edges** | CMDB-declared connections that don't carry traffic | Decommissioned firewall rule, primary path bypassed by traffic engineering |
| **Dark Attributes** | Properties that diverge from CMDB records | Field-adjusted antenna tilt, IP renumbering, capacity upgrade not reflected |

### Three Causal Types of Dark Graph Elements

1. **Type 1: Intrusion & Anomaly Topology** — Malicious actors creating undocumented paths (cybersecurity domain)
2. **Type 2: Tribal Knowledge & Operator Behaviour** — Engineers maintaining real state in their heads, not in CMDB (most commercially valuable)
3. **Type 3: Operational Drift** — Legitimate changes that were never documented (emergency swaps, scaling events, field adjustments)

### CasinoLimit Proof Points

Already demonstrated with non-telecom data:
- 50 dark nodes discovered from telemetry with no CMDB record
- 22 phantom CIs identified (CMDB entries with zero telemetry corroboration)
- 3 dark edges from traffic flows between undeclared-as-connected entities
- 67% of hypotheses backed by ≥2 evidence sources

### Data Strategy: Two-Layer Output

The synthetic data must provide two layers:

**Layer 1: "CMDB" (Declared State)** — Deliberately degraded topology representing what the operator *thinks* the network looks like:

| Divergence | How Generated | Approximate Rate |
|-----------|---------------|-----------------|
| Dark nodes | Remove 5-8% of real entities from CMDB (keep their KPIs) | 5-8% missing |
| Phantom nodes | Add ~3% fake entities with no corresponding KPIs | 3% extra stale |
| Dark edges | Remove ~10% of relationships (KPI correlations still reveal them) | 10% missing |
| Phantom edges | Add ~5% spurious relationships with no data flow | 5% stale |
| Dark attributes | Corrupt ~15% of entity attributes (wrong band, capacity, IP) | 15% wrong |
| Identity mutations | Change external_id for ~2% of entities (CMDB vs telemetry mismatch) | 2% ID mismatches |

**Layer 2: "Telemetry" (Observed State)** — Full KPI/event data that tells the truth:
- KPIs from entities in CMDB ✅
- KPIs from dark nodes (entities NOT in CMDB) — Pedkai should discover these
- Correlated KPI patterns implying dark edges — Pedkai should infer these
- Absence of KPIs from phantom nodes — Pedkai should flag these
- Alarms referencing mutated entity IDs — Pedkai should resolve these

**Ground Truth + Scoring Key**: A `divergence_manifest.parquet` labels every dark node, phantom node, dark edge, phantom edge, dark attribute, and identity mutation — so Pedkai's reconciliation accuracy can be quantified.

### Domain-Agnostic Implication

The Dark Graph reconciliation logic doesn't care whether the entity is a cell tower, a broadband OLT, an MPLS PE router, or an enterprise server. The **pattern** is identical:
- Entity emits telemetry but isn't in CMDB → dark node
- CMDB entry has no telemetry → phantom node
- Two entities show correlated behaviour but no declared relationship → dark edge

By including mobile RAN, fixed broadband, transport, enterprise services, and core network entities **all with deliberately injected CMDB divergences**, the dataset gives Pedkai training ground for Dark Graph inference that works across any infrastructure domain.

---

## 7. Final Output Specification

### Output Files

| File | Content | Maps To (Pedkai) |
|------|---------|-----------------|
| `ground_truth_entities.parquet` | Complete correct entity inventory (all ~1.49M entities) | Reference / scoring |
| `ground_truth_relationships.parquet` | Complete correct relationship graph (~2.21M relationships) | Reference / scoring |
| `cmdb_declared_entities.parquet` | Deliberately degraded entity CMDB (missing/phantom/mutated) | Seeds `NetworkEntityORM` + DataGerry |
| `cmdb_declared_relationships.parquet` | Deliberately degraded relationship CMDB (missing/phantom) | Seeds `EntityRelationshipORM` |
| `divergence_manifest.parquet` | Labels: which entities/rels are dark/phantom/mutated | Scoring key for Dark Graph accuracy |
| `kpi_metrics_wide.parquet` | Cell-level radio KPIs (wide format: 1 row per cell per hour, ~50 cols) | Pivot → `KPIMetricORM` / `KpiSampleORM` |
| `transport_kpis_wide.parquet` | Transport link KPIs (wide format) | Pivot → `KPIMetricORM` |
| `fixed_broadband_kpis_wide.parquet` | OLT/PON/ONT KPIs (wide format) | Pivot → `KPIMetricORM` |
| `enterprise_circuit_kpis_wide.parquet` | Enterprise service KPIs (wide format) | Pivot → `KPIMetricORM` |
| `core_element_kpis_wide.parquet` | Core network element KPIs | Pivot → `KPIMetricORM` |
| `power_environment_kpis.parquet` | Site power/environment status | Pivot → `KPIMetricORM` |
| `events_alarms.parquet` | Multi-domain alarms (radio, transport, power, core) | `/api/v1/alarms/ingest` + TMF642 |
| `customers_bss.parquet` | Customer + billing + service plans | Seeds `CustomerORM` + `BillingAccountORM` |
| `neighbour_relations.parquet` | Cell-to-cell adjacency with handover metrics | Seeds topology + provides HO analysis data |

### Loader Script

A Python loader script that:
1. Seeds `NetworkEntityORM` from `cmdb_declared_entities.parquet`
2. Seeds `EntityRelationshipORM` from `cmdb_declared_relationships.parquet`
3. Pivots wide-format KPIs to long-format and bulk-inserts into `KPIMetricORM`
4. Seeds `CustomerORM` and `BillingAccountORM` from `customers_bss.parquet`
5. Ingests events via the alarm ingestion API
6. Optionally loads ground truth for validation/scoring

---

## 8. Size Estimates

### KPI Data

| Domain | Entities × Metrics × Hours | Parquet (Wide) |
|--------|---------------------------|---------------|
| Cell-level radio | 64,700 × 35 × 720 | ~3 GB |
| Transport links | 50,000 × 15 × 720 | ~1.2 GB |
| Fixed broadband (OLT/PON) | 15,000 × 12 × 720 | ~300 MB |
| Enterprise circuits | 10,000 × 10 × 720 | ~200 MB |
| Core elements | 500 × 20 × 720 | ~20 MB |
| Power/environment | 21,000 × 5 × 720 | ~200 MB |
| **Total KPI** | | **~5 GB** |

### Overall Dataset

| Component | Parquet Size |
|-----------|-------------|
| Network graph (ground truth + CMDB + divergence manifest) | ~1 GB |
| Cell-level radio KPIs | ~3 GB |
| Transport KPIs | ~1.2 GB |
| Fixed broadband KPIs | ~300 MB |
| Enterprise circuit KPIs | ~200 MB |
| Core/power/environment KPIs | ~220 MB |
| Events/alarms (all domains) | ~500 MB |
| Customers + BSS | ~200 MB |
| **Total** | **~6.6 GB** |

### Generation Time Estimate

On a decent machine (8-core, 32GB RAM):
- Option B (1M sampled users, all 51.7K cells): **2-4 hours**
- Full topology + all KPI domains: **3-5 hours**

---

## 9. ML Considerations

### What Matters for Each ML Task

| ML Task | Bottleneck Table | 51.7K Cells / 1M Users | Assessment |
|---------|-----------------|------------------------|------------|
| Sleeping cell detection | cell_hourly_kpis | ~1,034 positive cases at 2% injection | ✅ Sufficient |
| Congestion prediction | cell_hourly_kpis | 5,500 streetworks cells enable spatial cascade modelling | ✅ Sufficient |
| General anomaly detection | cell_hourly_kpis | ~46K normal + injected anomaly diversity | ✅ Strong |
| QoE prediction | sessions | 78M sessions from 1M users | ✅ Sufficient |
| Handover optimisation | events + cell KPIs | ~200K neighbour pairs | ✅ Rich |
| 5G slice analytics | sessions + cell KPIs | ~6,700 SA cells | ✅ Adequate |
| CMDB reconciliation | topology + KPIs | 1.49M entities with injected divergences | ✅ Core training ground |

### Key Insight

Going from 1M to 20M users provides zero additional ML signal. The cell-side conditions are already fully represented. Cell-level KPIs are what matter for RAN intelligence. User session data can be scaled up later if per-subscriber analytics becomes a priority.

---

## 10. Open Items / Next Steps

1. **Build the generator** — Start with extended cell schema and radio-layer physics model (SINR → CQI → MCS → throughput chain), since these are the foundation everything else depends on.

2. **Physics model for each RAT**: LTE, NR-NSA (EN-DC bearer split), NR-SA (5QI-based QoS flows). Each has distinct band plans and radio characteristics.

3. **Scenario injection system**: Configurable profiles for sleeping cell, congestion, coverage hole, hardware fault, interference, transport failure, power failure, fibre cut — each defined as a set of KPI parameter deviations applied to targeted entities for specified durations.

4. **Vendor naming layer**: Map internal KPI names to Ericsson and Nokia PM counter naming conventions.

5. **CMDB degradation engine**: Systematically inject the six types of Dark Graph divergences into the declared-state topology.

6. **Loader script**: Python utility to ingest all Parquet files into Pedkai's database tables via its existing ORM models and API endpoints.

7. **Validation framework**: Extend the telecom-digital-twin's STEP 07 validation for the new schemas, FK integrity across all tables, and range checks for all KPIs.

---

## Appendix: Key File Paths in Pedkai Codebase

```
Pedkai/
├── backend/app/
│   ├── api/
│   │   ├── tmf628.py          # TMF628 Performance Management API
│   │   ├── tmf642.py          # TMF642 Alarm Management API
│   │   ├── topology.py        # Topology graph queries + impact analysis
│   │   ├── capacity.py        # AI-driven capacity planning
│   │   ├── service_impact.py  # Alarm correlation + customer impact
│   │   ├── alarm_ingestion.py # External alarm webhook
│   │   ├── autonomous.py      # Autonomous shield + digital twin prediction
│   │   └── incidents.py       # Incident lifecycle
│   ├── models/
│   │   ├── kpi_orm.py         # KPIMetricORM (hot path time-series)
│   │   ├── kpi_sample_orm.py  # KpiSampleORM (structured, FK to entities)
│   │   ├── network_entity_orm.py # NetworkEntityORM
│   │   ├── topology_models.py # EntityRelationshipORM
│   │   ├── tmf628_models.py   # TMF628 Pydantic models
│   │   ├── tmf642_models.py   # TMF642 Pydantic models
│   │   ├── customer_orm.py    # CustomerORM + ProactiveCareORM
│   │   ├── bss_orm.py         # ServicePlanORM + BillingAccountORM
│   │   └── decision_trace_orm.py # DecisionTraceORM (alarms + decisions)
│   ├── services/
│   │   ├── sleeping_cell_detector.py  # Z-score on traffic_volume
│   │   ├── alarm_correlation.py       # Temporal + spatial clustering
│   │   ├── capacity_engine.py         # PRB >85% → hotspot identification
│   │   ├── digital_twin.py           # Semantic similarity for predictions
│   │   ├── cx_intelligence.py        # Customer impact via graph traversal
│   │   ├── causal_models.py          # Expert causal templates (YAML)
│   │   └── drift_calibration.py      # FP rate tracking + threshold adjustment
│   ├── events/
│   │   └── schemas.py         # AlarmIngestedEvent, SleepingCellDetectedEvent, etc.
│   └── data/
│       └── causal_templates.yaml  # fiber_cut, prb_congestion, power_failure, backhaul_congestion
├── anops/
│   ├── anomaly_detection.py   # Z-score anomaly detection on KPIMetricORM
│   ├── causal_analysis.py     # Granger causality tests
│   ├── root_cause_analysis.py # Graph traversal RCA
│   ├── simulate_kpis.py       # Baseline + anomaly injection demo
│   └── seed_topology.py       # Sample gNB/Cell/Customer/SLA topology
├── data_fabric/
│   ├── alarm_normalizer.py    # Ericsson XML + Nokia JSON → unified format
│   ├── dataset_loaders.py     # HuggingFace dataset adapters
│   ├── seed_database.py       # Multi-source data seeding
│   └── kaggle_loader.py       # Kaggle dataset adapter
├── LiveTestData/
│   ├── loader.py              # NUMERIC_KPI_KEYS (real RAN metrics)
│   └── adapter.py             # Row → metric event mapping
├── datagerry_cmdb/            # Open-source CMDB (DataGerry)
├── generate_cmdb.py           # CMDB population script
├── IMPLEMENTATION_ROADMAP_V3.md  # Dark Graph / latent topology inference design
├── CASINOLIMIT_REALITY_VS_VISION.md  # CMDB reconciliation proof with non-telecom data
└── Gemini_Sales_Pitch.md      # ServiceNow/BMC positioning, CMDB accuracy claims
```
