"""
Step 03: Radio-Layer Physics Model & Cell KPI Generation.

Implements the SINR → CQI → MCS → throughput physics chain and generates
30 days of hourly cell-level radio KPIs for all ~64,700 logical cell-layers.

Per-RAT physics models:
- LTE: Path loss per band, SINR, CQI (3GPP TS 36.213), MCS, throughput
- NR-NSA (EN-DC): Dual bearer split — LTE anchor + NR SCG separate streams
- NR-SA: 5QI-based QoS flows, standalone NR (3GPP TS 38.214)

Sub-modules:
- physics.py   — Core radio physics engine (path loss, RSRP, SINR, CQI,
                 MCS, throughput, BLER, latency, handover, RACH/RRC models).
                 All functions are vectorised (numpy) for performance.
- profiles.py  — Diurnal traffic profiles (24-hour load curves per deployment
                 profile and timezone), shadow fading time series, interference
                 variation, and active UE multipliers.  Includes weekday/weekend
                 differentiation, Friday evening boost, and optional Ramadan overlay.
- generate.py  — Main orchestrator: loads cell inventory from Step 01, generates
                 environmental conditions, iterates 720 hourly intervals through
                 the vectorised physics chain, and writes output incrementally
                 using PyArrow streaming row groups.

Output: kpi_metrics_wide.parquet (~3 GB, ~47.6M rows × 52 columns)

Key 3GPP references:
- TS 36.213 Table 7.2.3-1  (LTE CQI → spectral efficiency)
- TS 38.214 Table 5.2.2.1-3 (NR CQI → spectral efficiency, 256QAM)
- TS 36.942                  (propagation models)
- TR 38.901                  (5G channel models)
- ITU-R P.1238               (indoor propagation)
"""
