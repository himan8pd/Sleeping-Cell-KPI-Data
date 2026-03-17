"""
Step 10: Abeyance Memory Test Data Generator.

Produces deterministic, seed-reproducible datasets that exercise every
lifecycle path, state transition, and discovery-mechanism input defined
in ABEYANCE_MEMORY_LLD_V3.md.

Output files (all in output/abeyance_memory/):
  - abeyance_fragments.parquet
      All fragment lifecycle states: INGESTED, ACTIVE, NEAR_MISS, SNAPPED,
      STALE, EXPIRED, COLD. Covers all mask combinations (TTT, TTF, TFF, FFF).
  - snap_decision_records.parquet
      Per-pair scoring rows with all 5 explicit dimension scores, masks_active,
      weights_used, temporal_modifier, decision enum.
  - scenario_surprise_events.parquet
      Surprise engine inputs: snap decisions crossing the adaptive threshold
      across DISCOVERY, DRIFT_ALERT, and CALIBRATION_ALERT escalation types.
  - temporal_sequences.parquet
      entity_sequence_log rows covering stable, low-confidence, and rare
      state transitions for Expectation Violation (Mechanism #9) testing.
  - causal_pairs.parquet
      Co-occurrence event pairs for Causal Direction (Mechanism #10):
      consistent directional ordering (A→B fraction >= 0.80), borderline,
      and contradictory cases.
  - disconfirmation_events.parquet
      Operator-driven and system-driven disconfirmation batches for
      Negative Evidence (Mechanism #3), with pre/post decay scores.
  - bridge_candidates.parquet
      Accumulation-graph node records with betweenness centrality and
      domain_span annotations for Bridge Detection (Mechanism #4).

Determinism: config.seed_for("step_12_abeyance_memory") → np.random.default_rng()
All IDs use _uuid_v7(rng).

Dependencies: Phase 2 output (ground_truth_entities.parquet) for real entity IDs.
              Falls back to synthetic entity IDs if Phase 2 output not available.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.table import Table

from pedkai_generator.config.settings import GeneratorConfig

console = Console()

SIMULATION_EPOCH = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# ---------------------------------------------------------------------------
# UUIDv7 — deterministic, seeded
# ---------------------------------------------------------------------------


def _uuid_v7(rng: np.random.Generator) -> str:
    """Deterministic UUIDv7 via seeded RNG."""
    b = bytearray(rng.bytes(16))
    b[6] = (b[6] & 0x0F) | 0x70
    b[8] = (b[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(b)))


# ---------------------------------------------------------------------------
# LLD constants (must match Pedkai Abeyance Memory implementation)
# ---------------------------------------------------------------------------

SNAP_STATES = ["INGESTED", "ACTIVE", "NEAR_MISS", "SNAPPED", "STALE", "EXPIRED", "COLD"]
FAILURE_MODE_PROFILES = ["DARK_EDGE", "DARK_NODE", "IDENTITY_MUTATION", "PHANTOM_CI", "DARK_ATTRIBUTE"]
SOURCE_TYPES = ["alarm", "metric", "log", "ticket", "cmdb_delta", "trace"]
ENTITY_DOMAINS = ["RAN", "TRANSPORT", "IP", "CORE", "VNF"]
DECISION_TYPES = ["SNAP", "NEAR_MISS", "AFFINITY", "NONE"]
ESCALATION_TYPES = ["DISCOVERY", "DRIFT_ALERT", "CALIBRATION_ALERT"]

# LLD §3.4 weight profiles
WEIGHT_PROFILES: dict[str, dict[str, float]] = {
    "DARK_EDGE":         {"sem": 0.15, "topo": 0.30, "temp": 0.10, "oper": 0.15, "ent": 0.30},
    "DARK_NODE":         {"sem": 0.25, "topo": 0.10, "temp": 0.10, "oper": 0.20, "ent": 0.35},
    "IDENTITY_MUTATION": {"sem": 0.10, "topo": 0.15, "temp": 0.10, "oper": 0.20, "ent": 0.45},
    "PHANTOM_CI":        {"sem": 0.20, "topo": 0.15, "temp": 0.10, "oper": 0.25, "ent": 0.30},
    "DARK_ATTRIBUTE":    {"sem": 0.25, "topo": 0.10, "temp": 0.10, "oper": 0.25, "ent": 0.30},
}

# LLD §3.6 thresholds
BASE_SNAP_THRESHOLD = 0.75
NEAR_MISS_THRESHOLD = 0.55
AFFINITY_THRESHOLD  = 0.40

# LLD §3.3 weight redistribution
def _redistribute(profile: str, avail_dims: list[str]) -> dict[str, float]:
    base = WEIGHT_PROFILES[profile]
    total = sum(base[d] for d in avail_dims)
    return {d: base[d] / total for d in avail_dims}


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------

FRAGMENT_SCHEMA = pa.schema([
    pa.field("fragment_id",          pa.string(),  nullable=False),
    pa.field("tenant_id",            pa.string(),  nullable=False),
    pa.field("source_type",          pa.string(),  nullable=False),
    pa.field("entity_id",            pa.string(),  nullable=True),
    pa.field("entity_domain",        pa.string(),  nullable=True),
    pa.field("snap_status",          pa.string(),  nullable=False),
    pa.field("failure_mode_profile", pa.string(),  nullable=True),
    pa.field("mask_semantic",        pa.bool_(),   nullable=False),
    pa.field("mask_topological",     pa.bool_(),   nullable=False),
    pa.field("mask_operational",     pa.bool_(),   nullable=False),
    pa.field("current_decay_score",  pa.float32(), nullable=False),
    pa.field("event_timestamp",      pa.timestamp("us", tz="UTC"), nullable=False),
    pa.field("dedup_key",            pa.string(),  nullable=False),
    pa.field("entity_count",         pa.int32(),   nullable=False),
    # Snapshot of extracted entities (JSON list of identifiers)
    pa.field("extracted_entities",   pa.string(),  nullable=True),
    # Polarity for Conflict Detection (Mechanism #6)
    pa.field("polarity",             pa.string(),  nullable=True),  # UP / DOWN / NEUTRAL
    # Lifecycle metadata
    pa.field("max_lifetime_days",    pa.int32(),   nullable=False),
    pa.field("snap_partner_id",      pa.string(),  nullable=True),   # set when SNAPPED
])

SNAP_DECISION_SCHEMA = pa.schema([
    pa.field("record_id",               pa.string(),  nullable=False),
    pa.field("tenant_id",               pa.string(),  nullable=False),
    pa.field("new_fragment_id",         pa.string(),  nullable=False),
    pa.field("candidate_fragment_id",   pa.string(),  nullable=False),
    pa.field("failure_mode_profile",    pa.string(),  nullable=False),
    pa.field("score_semantic",          pa.float64(), nullable=True),
    pa.field("score_topological",       pa.float64(), nullable=True),
    pa.field("score_temporal",          pa.float64(), nullable=True),
    pa.field("score_operational",       pa.float64(), nullable=True),
    pa.field("score_entity_overlap",    pa.float64(), nullable=False),
    pa.field("masks_active",            pa.string(),  nullable=False),  # JSON
    pa.field("weights_used",            pa.string(),  nullable=False),  # JSON
    pa.field("weights_base",            pa.string(),  nullable=False),  # JSON
    pa.field("raw_composite",           pa.float64(), nullable=False),
    pa.field("temporal_modifier",       pa.float64(), nullable=False),
    pa.field("final_score",             pa.float64(), nullable=False),
    pa.field("threshold_applied",       pa.float64(), nullable=False),
    pa.field("decision",                pa.string(),  nullable=False),
    pa.field("multiple_comparisons_k",  pa.int32(),   nullable=False),
    pa.field("evaluated_at",            pa.timestamp("us", tz="UTC"), nullable=False),
])

SURPRISE_SCHEMA = pa.schema([
    pa.field("event_id",                     pa.string(),  nullable=False),
    pa.field("tenant_id",                    pa.string(),  nullable=False),
    pa.field("snap_decision_record_id",      pa.string(),  nullable=False),
    pa.field("failure_mode_profile",         pa.string(),  nullable=False),
    pa.field("surprise_value",               pa.float64(), nullable=False),
    pa.field("threshold_at_time",            pa.float64(), nullable=False),
    pa.field("escalation_type",              pa.string(),  nullable=False),
    pa.field("dimensions_contributing",      pa.string(),  nullable=False),  # JSON
    pa.field("bin_index",                    pa.int32(),   nullable=False),
    pa.field("bin_probability",              pa.float64(), nullable=False),
    pa.field("created_at",                   pa.timestamp("us", tz="UTC"), nullable=False),
])

TEMPORAL_SEQ_SCHEMA = pa.schema([
    pa.field("seq_id",          pa.string(),  nullable=False),
    pa.field("tenant_id",       pa.string(),  nullable=False),
    pa.field("entity_id",       pa.string(),  nullable=False),
    pa.field("entity_domain",   pa.string(),  nullable=False),
    pa.field("from_state",      pa.string(),  nullable=True),   # NULL on first obs
    pa.field("to_state",        pa.string(),  nullable=False),
    pa.field("fragment_id",     pa.string(),  nullable=False),
    pa.field("event_timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
    # Transition metadata for Expectation Violation scoring
    pa.field("is_rare",         pa.bool_(),   nullable=False),  # <5 historical occurrences
    pa.field("transition_count_hint", pa.int32(), nullable=False),  # synthetic prior count
])

CAUSAL_PAIR_SCHEMA = pa.schema([
    pa.field("pair_id",            pa.string(),  nullable=False),
    pa.field("tenant_id",          pa.string(),  nullable=False),
    pa.field("entity_a_id",        pa.string(),  nullable=False),
    pa.field("entity_b_id",        pa.string(),  nullable=False),
    pa.field("fragment_a_id",      pa.string(),  nullable=False),
    pa.field("fragment_b_id",      pa.string(),  nullable=False),
    pa.field("time_delta_seconds", pa.float64(), nullable=False),
    pa.field("a_precedes_b",       pa.bool_(),   nullable=False),
    pa.field("direction_category", pa.string(),  nullable=False),  # CONSISTENT / BORDERLINE / CONTRADICTORY
    pa.field("event_timestamp",    pa.timestamp("us", tz="UTC"), nullable=False),
])

DISCONFIRMATION_SCHEMA = pa.schema([
    pa.field("event_id",           pa.string(),  nullable=False),
    pa.field("tenant_id",          pa.string(),  nullable=False),
    pa.field("fragment_id",        pa.string(),  nullable=False),
    pa.field("pathway",            pa.string(),  nullable=False),  # OPERATOR / SYSTEM
    pa.field("reason",             pa.string(),  nullable=True),
    pa.field("acceleration_factor",pa.float64(), nullable=False),
    pa.field("pre_decay_score",    pa.float64(), nullable=False),
    pa.field("post_decay_score",   pa.float64(), nullable=False),
    pa.field("created_at",         pa.timestamp("us", tz="UTC"), nullable=False),
])

BRIDGE_SCHEMA = pa.schema([
    pa.field("node_id",                  pa.string(),  nullable=False),
    pa.field("tenant_id",                pa.string(),  nullable=False),
    pa.field("fragment_id",              pa.string(),  nullable=False),
    pa.field("betweenness_centrality",   pa.float64(), nullable=False),
    pa.field("domain_span",              pa.int32(),   nullable=False),
    pa.field("severity",                 pa.string(),  nullable=False),  # CRITICAL/HIGH/MEDIUM/ROUTINE
    pa.field("is_bridge_discovery",      pa.bool_(),   nullable=False),
    pa.field("entity_domains_spanned",   pa.string(),  nullable=False),  # JSON list
    pa.field("sub_component_size",       pa.int32(),   nullable=False),
    pa.field("component_fingerprint",    pa.string(),  nullable=False),  # sha256 hex
    pa.field("created_at",               pa.timestamp("us", tz="UTC"), nullable=False),
])


# ---------------------------------------------------------------------------
# Fragment lifecycle generator
# ---------------------------------------------------------------------------

@dataclass
class FragmentSpec:
    status: str
    mask: tuple[bool, bool, bool]   # semantic, topological, operational
    polarity: str
    source_type: str
    entity_domain: str
    failure_mode_profile: str
    decay_score: float
    entity_count: int


def _lifecycle_specs(
    rng: np.random.Generator,
    target_fragments: int | None = None,
    domains: list[str] | None = None,
) -> list[FragmentSpec]:
    """Build a representative set of FragmentSpecs.

    By default, covers every lifecycle state + mask combination required by
    LLD §2.5. When `target_fragments` is set, the generator will scale up
    by appending randomly sampled specs until the requested count is reached.
    """
    specs: list[FragmentSpec] = []
    states_masks = [
        # (status,    mask_sem, mask_topo, mask_oper, decay_range)
        ("INGESTED",  True,  True,  True,  (0.98, 1.00)),
        ("ACTIVE",    True,  True,  True,  (0.70, 0.97)),
        ("ACTIVE",    True,  True,  False, (0.60, 0.90)),   # T-VEC operational failure
        ("ACTIVE",    True,  False, False, (0.50, 0.80)),   # topology + operational down
        ("ACTIVE",    False, False, False, (0.40, 0.70)),   # all T-VEC down
        ("NEAR_MISS", True,  True,  True,  (0.55, 0.74)),
        ("NEAR_MISS", True,  False, True,  (0.50, 0.70)),
        ("SNAPPED",   True,  True,  True,  (0.30, 0.60)),
        ("STALE",     True,  True,  True,  (0.05, 0.29)),
        ("STALE",     False, False, False, (0.01, 0.20)),   # expired with all NULL masks
        ("EXPIRED",   True,  True,  True,  (0.00, 0.04)),
        ("COLD",      False, False, False, (0.00, 0.01)),   # tombstoned
    ]
    polarities = ["UP", "DOWN", "NEUTRAL"]

    if domains is None:
        domains = ENTITY_DOMAINS[:3]

    # Start with the canonical coverage set
    for status, ms, mt, mo, (dlo, dhi) in states_masks:
        for profile in FAILURE_MODE_PROFILES:
            for domain in domains:
                polarity = str(rng.choice(polarities))
                src = str(rng.choice(SOURCE_TYPES))
                decay = float(rng.uniform(dlo, dhi))
                n_ent = int(rng.integers(1, 8))
                specs.append(FragmentSpec(
                    status=status,
                    mask=(ms, mt, mo),
                    polarity=polarity,
                    source_type=src,
                    entity_domain=domain,
                    failure_mode_profile=profile,
                    decay_score=round(decay, 4),
                    entity_count=n_ent,
                ))

    # Scale out to match the requested number of fragments (if any)
    if target_fragments is not None and len(specs) < target_fragments:
        while len(specs) < target_fragments:
            idx = int(rng.integers(0, len(states_masks)))
            status, ms, mt, mo, (dlo, dhi) = states_masks[idx]
            profile = str(rng.choice(FAILURE_MODE_PROFILES))
            domain = str(rng.choice(domains))
            polarity = str(rng.choice(polarities))
            src = str(rng.choice(SOURCE_TYPES))
            decay = float(rng.uniform(dlo, dhi))
            n_ent = int(rng.integers(1, 8))
            specs.append(FragmentSpec(
                status=status,
                mask=(ms, mt, mo),
                polarity=polarity,
                source_type=src,
                entity_domain=domain,
                failure_mode_profile=profile,
                decay_score=round(decay, 4),
                entity_count=n_ent,
            ))

    return specs


def _build_fragments(
    specs: list[FragmentSpec],
    rng: np.random.Generator,
    tenant_id: str,
    entity_pool: list[str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """
    Materialise FragmentSpec list → fragment row dicts.
    Returns (rows, fragment_id → status) for downstream join.
    """
    rows: list[dict[str, Any]] = []
    fid_status: dict[str, str] = {}

    for i, spec in enumerate(specs):
        fid = _uuid_v7(rng)
        eid = entity_pool[i % len(entity_pool)]
        ts = SIMULATION_EPOCH + timedelta(hours=int(rng.integers(0, 720)))
        ms, mt, mo = spec.mask
        entities = [eid] + [entity_pool[(i + j) % len(entity_pool)] for j in range(1, spec.entity_count)]
        partner = None
        if spec.status == "SNAPPED":
            partner = _uuid_v7(rng)

        rows.append({
            "fragment_id":          fid,
            "tenant_id":            tenant_id,
            "source_type":          spec.source_type,
            "entity_id":            eid,
            "entity_domain":        spec.entity_domain,
            "snap_status":          spec.status,
            "failure_mode_profile": spec.failure_mode_profile,
            "mask_semantic":        ms,
            "mask_topological":     mt,
            "mask_operational":     mo,
            "current_decay_score":  np.float32(spec.decay_score),
            "event_timestamp":      ts,
            "dedup_key":            f"{spec.source_type}:{eid}:{ts.isoformat()}:{i}",
            "entity_count":         spec.entity_count,
            "extracted_entities":   json.dumps(entities),
            "polarity":             spec.polarity,
            "max_lifetime_days":    730,
            "snap_partner_id":      partner,
        })
        fid_status[fid] = spec.status

    return rows, fid_status


# ---------------------------------------------------------------------------
# Snap decision generator
# ---------------------------------------------------------------------------

def _compute_snap_decision(
    rng: np.random.Generator,
    frag_a: dict[str, Any],
    frag_b: dict[str, Any],
    profile: str,
    sidak_k: int,
    ts: datetime,
    tenant_id: str,
    target_decision: str,
) -> dict[str, Any]:
    """
    Compute a deterministic snap_decision_record for a pair (frag_a, frag_b).
    target_decision drives which score range to sample from.
    Implements LLD §3.6 composite computation.
    """
    base_weights = WEIGHT_PROFILES[profile]

    # Determine available dimensions (both fragments must have mask=TRUE)
    avail: list[str] = ["temp", "ent"]  # always available (LLD §3.1)
    if frag_a["mask_semantic"] and frag_b["mask_semantic"]:
        avail.append("sem")
    if frag_a["mask_topological"] and frag_b["mask_topological"]:
        avail.append("topo")
    if frag_a["mask_operational"] and frag_b["mask_operational"]:
        avail.append("oper")

    adj_weights = _redistribute(profile, avail)

    # Score ranges by target decision
    score_ranges = {
        "SNAP":      (0.76, 0.99),
        "NEAR_MISS": (0.56, 0.74),
        "AFFINITY":  (0.41, 0.54),
        "NONE":      (0.00, 0.39),
    }
    lo, hi = score_ranges.get(target_decision, (0.40, 0.99))

    def _dim_score(dim: str) -> float | None:
        if dim not in avail:
            return None
        # raw cosine in [0,1]
        return float(rng.uniform(max(0.0, lo - 0.05), min(1.0, hi + 0.05)))

    s_sem  = _dim_score("sem")
    s_topo = _dim_score("topo")
    s_temp = float(rng.uniform(lo, hi))  # always computed
    s_oper = _dim_score("oper")
    s_ent  = float(rng.uniform(max(0.0, lo - 0.10), min(1.0, hi + 0.10)))

    raw_composite = sum(
        adj_weights.get(d, 0.0) * v
        for d, v in [("sem", s_sem), ("topo", s_topo), ("temp", s_temp), ("oper", s_oper), ("ent", s_ent)]
        if v is not None
    )

    temp_mod = float(rng.uniform(0.50, 1.00))
    final = min(1.0, raw_composite * temp_mod)

    # Sidak correction — LLD §3.6
    threshold = 1.0 - (1.0 - BASE_SNAP_THRESHOLD) ** (1.0 / max(1, sidak_k))

    return {
        "record_id":              _uuid_v7(rng),
        "tenant_id":              tenant_id,
        "new_fragment_id":        frag_a["fragment_id"],
        "candidate_fragment_id":  frag_b["fragment_id"],
        "failure_mode_profile":   profile,
        "score_semantic":         s_sem,
        "score_topological":      s_topo,
        "score_temporal":         s_temp,
        "score_operational":      s_oper,
        "score_entity_overlap":   s_ent,
        "masks_active":           json.dumps({
            "sem": s_sem is not None,
            "topo": s_topo is not None,
            "temp": True,
            "oper": s_oper is not None,
        }),
        "weights_used":           json.dumps(adj_weights),
        "weights_base":           json.dumps(base_weights),
        "raw_composite":          round(raw_composite, 6),
        "temporal_modifier":      round(temp_mod, 6),
        "final_score":            round(final, 6),
        "threshold_applied":      round(threshold, 6),
        "decision":               target_decision,
        "multiple_comparisons_k": sidak_k,
        "evaluated_at":           ts,
    }


def _build_snap_decisions(
    fragments: list[dict[str, Any]],
    rng: np.random.Generator,
    tenant_id: str,
    snap_decisions_per_profile: int,
) -> list[dict[str, Any]]:
    """Produce snap_decision_records exercising all decision types.

    The total volume scales with the configured number of decisions per
    failure-mode profile.
    """
    rows: list[dict[str, Any]] = []

    # Group fragments by profile
    by_profile: dict[str, list[dict[str, Any]]] = {}
    for f in fragments:
        p = f.get("failure_mode_profile")
        if p:
            by_profile.setdefault(p, []).append(f)

    decisions_cycle = ["SNAP", "NEAR_MISS", "AFFINITY", "NONE"] * 4

    for profile, pool in by_profile.items():
        if len(pool) < 2:
            continue
        rng.shuffle(pool)
        for idx in range(min(snap_decisions_per_profile, len(pool) - 1)):
            fa = pool[idx]
            fb = pool[idx + 1]
            decision = decisions_cycle[idx % len(decisions_cycle)]
            ts = SIMULATION_EPOCH + timedelta(hours=int(rng.integers(0, 720)))
            sidak_k = int(rng.integers(1, 6))
            rows.append(_compute_snap_decision(
                rng, fa, fb, profile, sidak_k, ts, tenant_id, decision,
            ))

    return rows


# ---------------------------------------------------------------------------
# Surprise events generator
# ---------------------------------------------------------------------------

def _build_surprise_events(
    snap_rows: list[dict[str, Any]],
    rng: np.random.Generator,
    tenant_id: str,
    total_events: int,
) -> list[dict[str, Any]]:
    """Produce surprise_event rows covering all escalation types."""
    rows: list[dict[str, Any]] = []
    DEFAULT_THRESHOLD = 6.64  # -log2(1/100)
    CAP_BITS = 20.0

    # Base escalation sequence (original fixture pattern)
    base_seq = ["DISCOVERY"] * 6 + ["DRIFT_ALERT"] * 3 + ["CALIBRATION_ALERT"] * 2
    if total_events <= 0:
        return rows

    # Repeat/truncate to meet target size (preserves ordering)
    escalation_seq = (base_seq * ((total_events // len(base_seq)) + 1))[:total_events]

    for i, esc in enumerate(escalation_seq):
        snap = snap_rows[i % len(snap_rows)]
        # Synthesise a surprising bin (low probability → high surprise)
        bin_prob = float(rng.uniform(0.001, 0.009))
        surprise = min(CAP_BITS, -np.log2(bin_prob + 0.01))

        if esc == "DISCOVERY":
            dims = list(rng.choice(["sem", "topo", "temp", "oper"], size=int(rng.integers(1, 3)), replace=False))
        elif esc == "DRIFT_ALERT":
            dims = list(rng.choice(["sem", "topo", "temp", "oper"], size=int(rng.integers(3, 5)), replace=False))
        else:
            dims = ["sem", "topo", "temp", "oper"]
            surprise = float(rng.uniform(15.0, 20.0))

        rows.append({
            "event_id":                 _uuid_v7(rng),
            "tenant_id":                tenant_id,
            "snap_decision_record_id":  snap["record_id"],
            "failure_mode_profile":     snap["failure_mode_profile"],
            "surprise_value":           round(surprise, 6),
            "threshold_at_time":        round(DEFAULT_THRESHOLD * float(rng.uniform(0.90, 1.10)), 6),
            "escalation_type":          esc,
            "dimensions_contributing":  json.dumps(dims),
            "bin_index":                int(rng.integers(0, 50)),
            "bin_probability":          round(bin_prob, 8),
            "created_at":               snap["evaluated_at"],
        })
    return rows


# ---------------------------------------------------------------------------
# Temporal sequence generator
# ---------------------------------------------------------------------------

def _build_temporal_sequences(
    fragments: list[dict[str, Any]],
    rng: np.random.Generator,
    tenant_id: str,
    entity_count: int,
) -> list[dict[str, Any]]:
    """Generate entity_sequence_log rows for Temporal Sequence modelling.

    The number of entities and transitions scales with the fragment count.
    """
    rows: list[dict[str, Any]] = []
    all_states = ["ALARM|CRITICAL", "ALARM|MAJOR", "METRIC|WARNING", "METRIC|NORMAL",
                  "TICKET|NORMAL", "TICKET|URGENT", "LOG|ERROR", "LOG|INFO"]

    unique_ids = list({f["entity_id"] for f in fragments})
    sample_count = min(len(unique_ids), entity_count)
    entity_ids = list(rng.choice(unique_ids, size=sample_count, replace=False))

    count_profiles = [
        ("STABLE",         100, 500, False),
        ("LOW_CONFIDENCE", 5,   19,  False),
        ("RARE",           0,   4,   True),
    ]

    for eid in entity_ids:
        domain = str(rng.choice(ENTITY_DOMAINS))
        n_transitions = int(rng.integers(3, 12))
        prev_state: str | None = None

        for t in range(n_transitions):
            to_state = str(rng.choice(all_states))
            cat, cnt_lo, cnt_hi, is_rare = count_profiles[t % len(count_profiles)]
            ts = SIMULATION_EPOCH + timedelta(hours=int(rng.integers(0, 720)))
            fid = fragments[int(rng.integers(0, len(fragments)))] ["fragment_id"]

            rows.append({
                "seq_id":               _uuid_v7(rng),
                "tenant_id":            tenant_id,
                "entity_id":            eid,
                "entity_domain":        domain,
                "from_state":           prev_state,
                "to_state":             to_state,
                "fragment_id":          fid,
                "event_timestamp":      ts,
                "is_rare":              is_rare,
                "transition_count_hint": int(rng.integers(cnt_lo, max(cnt_lo + 1, cnt_hi))),
            })
            prev_state = to_state

    return rows


# ---------------------------------------------------------------------------
# Causal pair generator
# ---------------------------------------------------------------------------

def _build_causal_pairs(
    fragments: list[dict[str, Any]],
    rng: np.random.Generator,
    tenant_id: str,
    pairs_per_category: int,
) -> list[dict[str, Any]]:
    """Co-occurrence pairs for Causal Direction Testing (Mechanism #10)."""
    rows: list[dict[str, Any]] = []
    fids = [f["fragment_id"] for f in fragments]
    entities = list({f["entity_id"] for f in fragments})

    categories = [
        ("CONSISTENT",    0.82, 0.98),
        ("BORDERLINE",    0.65, 0.79),
        ("CONTRADICTORY", 0.30, 0.50),
    ]

    for direction_cat, frac_lo, frac_hi in categories:
        for _ in range(pairs_per_category):
            ea, eb = entities[int(rng.integers(0, len(entities)))], entities[int(rng.integers(0, len(entities)))]
            fa, fb = fids[int(rng.integers(0, len(fids)))], fids[int(rng.integers(0, len(fids)))]
            direction_frac = float(rng.uniform(frac_lo, frac_hi))
            a_precedes = bool(rng.random() < direction_frac)
            dt = float(rng.uniform(10.0, 3600.0))
            ts = SIMULATION_EPOCH + timedelta(hours=int(rng.integers(0, 720)))

            rows.append({
                "pair_id":            _uuid_v7(rng),
                "tenant_id":          tenant_id,
                "entity_a_id":        ea,
                "entity_b_id":        eb,
                "fragment_a_id":      fa,
                "fragment_b_id":      fb,
                "time_delta_seconds": round(dt, 2),
                "a_precedes_b":       a_precedes,
                "direction_category": direction_cat,
                "event_timestamp":    ts,
            })

    return rows


# ---------------------------------------------------------------------------
# Disconfirmation events generator
# ---------------------------------------------------------------------------

def _build_disconfirmation_events(
    fragments: list[dict[str, Any]],
    rng: np.random.Generator,
    tenant_id: str,
    max_events: int,
) -> list[dict[str, Any]]:
    """Negative Evidence batches (Mechanism #3, LLD §7.3)."""
    rows: list[dict[str, Any]] = []
    active_frags = [f for f in fragments if f["snap_status"] in ("ACTIVE", "NEAR_MISS", "SNAPPED")]

    for i, frag in enumerate(active_frags[:max_events]):
        pathway = "OPERATOR" if i % 3 != 2 else "SYSTEM"
        acc_factor = round(float(rng.uniform(2.0, 10.0)), 2)
        pre = frag["current_decay_score"]
        post = max(0.0, float(pre) * (1.0 / acc_factor))
        ts = SIMULATION_EPOCH + timedelta(hours=int(rng.integers(0, 720)))

        rows.append({
            "event_id":            _uuid_v7(rng),
            "tenant_id":           tenant_id,
            "fragment_id":         frag["fragment_id"],
            "pathway":             pathway,
            "reason":              "False positive: unrelated maintenance window" if pathway == "OPERATOR" else None,
            "acceleration_factor": acc_factor,
            "pre_decay_score":     round(float(pre), 6),
            "post_decay_score":    round(post, 6),
            "created_at":          ts,
        })

    return rows


# ---------------------------------------------------------------------------
# Bridge candidate generator
# ---------------------------------------------------------------------------

def _build_bridge_candidates(
    fragments: list[dict[str, Any]],
    rng: np.random.Generator,
    tenant_id: str,
    per_severity: int,
) -> list[dict[str, Any]]:
    """
    Accumulation graph node records for Bridge Detection (Mechanism #4).
    LLD §7.4 severity thresholds:
      CRITICAL: BC >= 0.60, sub_component >= 10
      HIGH:     BC >= 0.45 OR sub_component >= 7
      MEDIUM:   BC >= 0.30
      ROUTINE:  BC < 0.30
    Bridge discovery only when domain_span >= 2.
    """
    rows: list[dict[str, Any]] = []
    import hashlib

    bc_profiles = [
        ("CRITICAL", 0.60, 0.99, 10, 20),
        ("HIGH",     0.45, 0.59, 7,  12),
        ("MEDIUM",   0.30, 0.44, 3,  7),
        ("ROUTINE",  0.00, 0.29, 1,  4),
    ]

    frag_pool = [f for f in fragments if f["snap_status"] in ("ACTIVE", "NEAR_MISS", "SNAPPED")]

    for severity, bc_lo, bc_hi, sub_lo, sub_hi in bc_profiles:
        used_frags = frag_pool[:per_severity]
        frag_pool = frag_pool[per_severity:] or frag_pool

        for i, frag in enumerate(used_frags):
            bc = float(rng.uniform(bc_lo, bc_hi))
            sub_size = int(rng.integers(sub_lo, max(sub_lo + 1, sub_hi + 1)))
            # domain_span >= 2 required for BRIDGE_DISCOVERY classification
            n_domains = int(rng.integers(1, 6))
            domains = list(rng.choice(ENTITY_DOMAINS, size=n_domains, replace=False))
            is_discovery = len(domains) >= 2 and bc >= 0.30

            # Deterministic fingerprint from fragment_ids in component
            component_ids = sorted([frag["fragment_id"]] + [
                _uuid_v7(rng) for _ in range(sub_size - 1)
            ])
            fingerprint = hashlib.sha256("|".join(component_ids).encode()).hexdigest()

            rows.append({
                "node_id":                _uuid_v7(rng),
                "tenant_id":              tenant_id,
                "fragment_id":            frag["fragment_id"],
                "betweenness_centrality": round(bc, 6),
                "domain_span":            len(domains),
                "severity":               severity,
                "is_bridge_discovery":    is_discovery,
                "entity_domains_spanned": json.dumps(domains),
                "sub_component_size":     sub_size,
                "component_fingerprint":  fingerprint,
                "created_at":             SIMULATION_EPOCH + timedelta(hours=int(rng.integers(0, 720))),
            })

    return rows


# ---------------------------------------------------------------------------
# Writer helper
# ---------------------------------------------------------------------------

def _write_parquet(rows: list[dict[str, Any]], schema: pa.Schema, path: Path) -> int:
    if not rows:
        console.print(f"    [yellow]No rows — skipping {path.name}[/yellow]")
        return 0
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, str(path), compression="zstd", row_group_size=50_000)
    return len(rows)


# ---------------------------------------------------------------------------
# Entity pool builder
# ---------------------------------------------------------------------------

def _load_entity_pool(config: GeneratorConfig, rng: np.random.Generator) -> list[str]:
    entities_path = config.paths.output_dir / "ground_truth_entities.parquet"
    if entities_path.exists():
        try:
            import polars as pl
            df = pl.read_parquet(str(entities_path), columns=["entity_id"])
            pool = df["entity_id"].to_list()
            if pool:
                console.print(f"    [dim]Loaded {len(pool):,} real entity IDs from Phase 2[/dim]")
                return pool
        except Exception as exc:
            console.print(f"    [yellow]Could not load entities ({exc}), using synthetic IDs[/yellow]")

    # Fall back to deterministic UUIDv7 pool (seeded — fully reproducible)
    return [_uuid_v7(rng) for _ in range(500)]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_abeyance_memory_data(config: GeneratorConfig) -> None:
    """
    Generate comprehensive Abeyance Memory test datasets.

    Writes 7 Parquet files to output/abeyance_memory/.
    Fully reproducible from config.global_seed.
    """
    seed = config.seed_for("step_10_abeyance_memory")
    rng = np.random.default_rng(seed)
    tenant_id = config.tenant_id

    out_dir = config.paths.output_dir / "abeyance_memory"
    out_dir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold cyan]Phase 10 — Abeyance Memory Test Data")
    console.print(f"    seed={seed}  tenant={tenant_id}  out={out_dir}")

    # Entity pool (real IDs from Phase 2 if available)
    entity_pool = _load_entity_pool(config, rng)

    # Determine how many fragments to generate (scales with network size)
    ab_cfg = config.abeyance_memory
    entity_count = len(entity_pool)
    target_fragments = max(ab_cfg.min_fragments, int(entity_count * ab_cfg.fragment_fraction))
    if ab_cfg.max_fragments is not None:
        target_fragments = min(target_fragments, ab_cfg.max_fragments)

    # Scale factor relative to the original baseline fixture size
    base_fragments = 180
    scale = target_fragments / base_fragments

    # Derive scaled counts (0 in config means auto-scale)
    snap_per_profile = (
        ab_cfg.snap_decisions_per_profile
        or max(1, int(round(40 * scale)))
    )
    surprise_total = (
        ab_cfg.surprise_events_total
        or max(1, int(round(11 * scale)))
    )
    temporal_entities = (
        ab_cfg.temporal_entity_count
        or max(1, int(round(30 * scale)))
    )
    causal_pairs_per_cat = (
        ab_cfg.causal_pairs_per_category
        or max(1, int(round(20 * scale)))
    )
    disconfirmation_max = (
        ab_cfg.disconfirmation_max
        or max(1, int(round(60 * scale)))
    )
    bridge_candidates_per_sev = (
        ab_cfg.bridge_candidates_per_severity
        or max(1, int(round(8 * scale)))
    )

    console.print(
        f"    [dim]Generating {target_fragments:,} fragments (entity pool {entity_count:,}) "
        f"→ scale={scale:.2f} | snap={snap_per_profile} | surprise={surprise_total} | "
        f"temporal_entities={temporal_entities} | causal={causal_pairs_per_cat} | "
        f"disconf={disconfirmation_max} | bridge={bridge_candidates_per_sev}[/dim]"
    )

    # ── 1. Fragments ─────────────────────────────────────────────────────
    specs = _lifecycle_specs(rng, target_fragments=target_fragments, domains=ENTITY_DOMAINS)
    fragment_rows, fid_status = _build_fragments(specs, rng, tenant_id, entity_pool)
    n = _write_parquet(fragment_rows, FRAGMENT_SCHEMA, out_dir / "abeyance_fragments.parquet")
    console.print(f"    [green]abeyance_fragments.parquet[/green]       {n:>8,} rows")

    # ── 2. Snap decisions ────────────────────────────────────────────────
    snap_rows = _build_snap_decisions(
        fragment_rows,
        rng,
        tenant_id,
        snap_per_profile,
    )
    n = _write_parquet(snap_rows, SNAP_DECISION_SCHEMA, out_dir / "snap_decision_records.parquet")
    console.print(f"    [green]snap_decision_records.parquet[/green]    {n:>8,} rows")

    # ── 3. Surprise events ───────────────────────────────────────────────
    surprise_rows = _build_surprise_events(
        snap_rows,
        rng,
        tenant_id,
        surprise_total,
    )
    n = _write_parquet(surprise_rows, SURPRISE_SCHEMA, out_dir / "scenario_surprise_events.parquet")
    console.print(f"    [green]scenario_surprise_events.parquet[/green] {n:>8,} rows")

    # ── 4. Temporal sequences ────────────────────────────────────────────
    seq_rows = _build_temporal_sequences(
        fragment_rows,
        rng,
        tenant_id,
        temporal_entities,
    )
    n = _write_parquet(seq_rows, TEMPORAL_SEQ_SCHEMA, out_dir / "temporal_sequences.parquet")
    console.print(f"    [green]temporal_sequences.parquet[/green]       {n:>8,} rows")

    # ── 5. Causal pairs ──────────────────────────────────────────────────
    causal_rows = _build_causal_pairs(
        fragment_rows,
        rng,
        tenant_id,
        causal_pairs_per_cat,
    )
    n = _write_parquet(causal_rows, CAUSAL_PAIR_SCHEMA, out_dir / "causal_pairs.parquet")
    console.print(f"    [green]causal_pairs.parquet[/green]             {n:>8,} rows")

    # ── 6. Disconfirmation events ────────────────────────────────────────
    disconf_rows = _build_disconfirmation_events(
        fragment_rows,
        rng,
        tenant_id,
        disconfirmation_max,
    )
    n = _write_parquet(disconf_rows, DISCONFIRMATION_SCHEMA, out_dir / "disconfirmation_events.parquet")
    console.print(f"    [green]disconfirmation_events.parquet[/green]   {n:>8,} rows")

    # ── 7. Bridge candidates ─────────────────────────────────────────────
    bridge_rows = _build_bridge_candidates(
        fragment_rows,
        rng,
        tenant_id,
        bridge_candidates_per_sev,
    )
    n = _write_parquet(bridge_rows, BRIDGE_SCHEMA, out_dir / "bridge_candidates.parquet")
    console.print(f"    [green]bridge_candidates.parquet[/green]        {n:>8,} rows")

    # summary table
    t = Table(title="Phase 10 Complete", show_header=True, header_style="bold magenta")
    t.add_column("File", style="cyan")
    t.add_column("Coverage", style="white")
    t.add_row("abeyance_fragments",       "All 7 lifecycle states × 5 profiles × 3 domains × all mask combos")
    t.add_row("snap_decision_records",    "SNAP/NEAR_MISS/AFFINITY/NONE × all profiles, explicit 5-dim scores")
    t.add_row("scenario_surprise_events", "DISCOVERY / DRIFT_ALERT / CALIBRATION_ALERT escalation types")
    t.add_row("temporal_sequences",       "STABLE / LOW_CONFIDENCE / RARE transitions per entity")
    t.add_row("causal_pairs",             "CONSISTENT / BORDERLINE / CONTRADICTORY directional fractions")
    t.add_row("disconfirmation_events",   "OPERATOR + SYSTEM pathways, acc_factor in [2,10]")
    t.add_row("bridge_candidates",        "CRITICAL / HIGH / MEDIUM / ROUTINE BC, domain_span 1-5")
    console.print(t)
