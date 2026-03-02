"""
Step 05: Scenario Injection System.

Configurable anomaly/fault injection engine that layers degradation scenarios
on top of baseline KPIs across all domains.

Scenario types:
  1. Sleeping cell — traffic volume drops with no alarm, subtle KPI degradation
  2. Congestion — PRB >85%, throughput collapse, latency spike
  3. Coverage hole — RSRP/RSRQ degradation in spatial cluster
  4. Hardware fault — cell availability drop, BLER spike, abrupt
  5. Interference — IoT elevation, CQI/MCS degradation, SINR drop
  6. Transport failure — backhaul link down, cascading to all served cells
  7. Power failure — site-level, all co-located equipment affected
  8. Fibre cut — cross-domain cascade: cells + ONTs + enterprise circuits

Each scenario is defined as a set of KPI parameter deviations applied to
targeted entities for specified durations, with configurable ramp-up/ramp-down
curves. Scenarios align with Pedkai's causal_templates.yaml.

Output (overlay, not in-place mutation):
  - output/scenario_manifest.parquet   — master schedule of all injected scenarios
  - output/scenario_kpi_overrides.parquet — sparse override table

Dependencies: Phase 2 (topology graph), Phase 3 (radio KPIs), Phase 4 (domain KPIs)
"""
