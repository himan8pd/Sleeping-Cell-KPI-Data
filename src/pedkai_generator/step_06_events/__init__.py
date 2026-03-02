"""
Step 06: Events & Alarms Generation.

Generates multi-domain alarm/event data aligned with scenario injections
from Phase 5.  Each non-sleeping-cell scenario produces a temporally
aligned chain of alarms; cross-domain cascades (e.g. fibre cut →
transport link-down → radio cell throughput alarm) produce correlated
alarm chains sharing a ``correlation_group_id``.

**Critical design rule:**
  Sleeping-cell scenarios produce **NO** alarms — that is the entire
  point of the sleeping-cell problem.  The alarm gap is what makes them
  hard to detect and is the ground-truth signal that Dark Graph must
  learn to identify.

Alarm categories:
  1. Scenario-driven alarms — temporally aligned with Phase 5 scenarios,
     each carrying scenario_id FK and correlation_group_id for chain tracing
  2. Organic/background alarms — transient noise (brief flaps, minor
     temperature warnings, RACH spikes) across all domains, providing
     realistic clutter for alarm correlation systems

Alarm types (from Phase 0 contract):
  Radio:      CELL_DEGRADATION, CELL_OUTAGE, HIGH_BLER, HIGH_INTERFERENCE,
              RACH_FAILURE, HANDOVER_FAILURE, PRB_CONGESTION
  Transport:  LINK_DOWN, INTERFACE_DOWN, OPTICAL_POWER_LOW, BGP_FLAP,
              LSP_DOWN, HIGH_LATENCY, HIGH_PACKET_LOSS
  Power:      POWER_SUPPLY_FAIL, BATTERY_LOW, MAINS_FAILURE,
              HIGH_TEMPERATURE, COOLING_FAILURE
  Core:       MME_OVERLOAD, SGW_FAILURE, PGW_FAILURE, AMF_OVERLOAD,
              UPF_FAILURE, SMF_FAILURE, BNG_OVERLOAD, RADIUS_TIMEOUT
  Fixed BB:   OLT_PORT_DOWN, ONT_OFFLINE, PON_SIGNAL_DEGRADATION
  Generic:    DEGRADATION, EQUIPMENT_FAILURE, CONFIGURATION_ERROR

TMF642 compliance:
  Alarm structure compatible with Pedkai's alarm ingestion API.

Output:
  - output/events_alarms.parquet  (15 columns, ~500 MB)

Dependencies: Phase 5 (reads scenario_manifest.parquet),
             Phase 2 (reads ground_truth_entities.parquet for entity metadata)
"""
