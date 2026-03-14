"""
CMDB Decay Model — parametrized degradation rates for simulation.

Provides calibrated degradation rates based on published research into
CMDB accuracy decay in production telecoms environments. The module is
intentionally independent of the pipeline step (step_08_cmdb_degradation)
so it can be imported and reused without touching the existing generator.

Usage::

    from pedkai_generator.cmdb_decay_model import (
        REALISTIC_DECAY_CONFIG,
        ACCELERATED_DECAY_CONFIG,
        apply_decay,
    )

    degraded, ground_truth = apply_decay(
        cmdb_snapshot=my_df,
        config=REALISTIC_DECAY_CONFIG,
        simulation_months=12,
        seed=42,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CMDBDecayConfig:
    """Calibrated CMDB degradation rates based on published research.

    All rates are expressed as probabilities in [0, 1].  The docstring for
    each field gives the empirically observed range from which the default
    was chosen.
    """

    phantom_ci_rate_quarterly: float = 0.05
    """Probability that a CI becomes a phantom (declared but absent in reality)
    per quarter.  Observed range: 3–7 % per quarter."""

    phantom_edge_rate_annual: float = 0.11
    """Probability that a relationship row is a phantom per year.
    Observed range: 8–15 % per year."""

    identity_mutation_rate_annual: float = 0.03
    """Probability that a CI's external identifier is corrupted per year.
    Observed range: 2–4 % per year."""

    dark_node_rate_per_release: float = 0.07
    """Probability that a CI exists in reality but is absent from the CMDB
    per release cycle.  Observed range: 5–10 % per release."""

    dark_edge_rate_annual: float = 0.08
    """Probability that a relationship exists in reality but is absent from
    the CMDB per year.  Observed range: ~8 % per year."""


# ---------------------------------------------------------------------------
# Pre-built configuration instances
# ---------------------------------------------------------------------------

REALISTIC_DECAY_CONFIG: CMDBDecayConfig = CMDBDecayConfig()
"""Default / realistic decay config using mid-range published rates."""

ACCELERATED_DECAY_CONFIG: CMDBDecayConfig = CMDBDecayConfig(
    phantom_ci_rate_quarterly=0.10,
    phantom_edge_rate_annual=0.22,
    identity_mutation_rate_annual=0.06,
    dark_node_rate_per_release=0.14,
    dark_edge_rate_annual=0.16,
)
"""Accelerated decay config with all rates approximately doubled.
Useful for stress-testing detection models and shortening evaluation cycles."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compound_rate(per_period_rate: float, periods: float) -> float:
    """Return the cumulative probability of at least one event over *periods*
    independent intervals, each with probability *per_period_rate*.

    Uses the complement rule:  P = 1 - (1 - r)^n
    """
    return 1.0 - (1.0 - per_period_rate) ** periods


def _is_edge_type(entity_type: str) -> bool:
    """Return True if *entity_type* looks like a relationship / edge."""
    lower = entity_type.lower()
    return any(kw in lower for kw in ("edge", "link", "relationship", "rel", "conn"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_decay(
    cmdb_snapshot: pd.DataFrame,
    config: CMDBDecayConfig,
    simulation_months: int,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply CMDB degradation to *cmdb_snapshot* over *simulation_months*.

    The function applies five decay mechanisms in sequence.  Each mechanism
    operates independently using the same RNG state derived from *seed*, so
    the combined output is fully deterministic for a given seed.

    Parameters
    ----------
    cmdb_snapshot:
        DataFrame with at least columns ``entity_id`` (str) and
        ``entity_type`` (str).  Additional columns are preserved unchanged.
    config:
        :class:`CMDBDecayConfig` instance containing decay rates.
    simulation_months:
        Number of months to simulate.  Rates that are expressed per-quarter
        or per-year are scaled accordingly.
    seed:
        Integer random seed for reproducibility.

    Returns
    -------
    degraded_cmdb : pd.DataFrame
        Copy of *cmdb_snapshot* with:

        * phantom CI rows **removed** (they existed only in the declared
          CMDB but not in reality — so after decay the declaration is gone);
        * dark-node/dark-edge rows still present but recorded in
          *divergence_ground_truth*;
        * identity-mutation rows with ``entity_id`` suffixed by ``'-MUT'``.

    divergence_ground_truth : pd.DataFrame
        Labelled ground truth for ML evaluation with columns:

        ``entity_id`` (str), ``divergence_type`` (str),
        ``injected_at_month`` (int), ``expected_detection_score`` (float).

    Notes
    -----
    divergence_type values used:
        ``'phantom_ci'``, ``'phantom_edge'``, ``'identity_mutation'``,
        ``'dark_node'``, ``'dark_edge'``.

    expected_detection_score values:
        * ``phantom_ci`` → 0.8
        * ``identity_mutation`` → 0.7
        * ``dark_node`` / ``dark_edge`` → 0.6
    """
    if cmdb_snapshot.empty:
        empty_gt = pd.DataFrame(
            columns=["entity_id", "divergence_type", "injected_at_month", "expected_detection_score"]
        )
        return cmdb_snapshot.copy(), empty_gt

    rng = np.random.default_rng(seed)

    df = cmdb_snapshot.copy()
    ground_truth_rows: list[dict] = []

    # Derived scaled rates
    quarters = simulation_months / 3.0
    years = simulation_months / 12.0

    phantom_ci_rate = _compound_rate(config.phantom_ci_rate_quarterly, quarters)
    phantom_edge_rate = _compound_rate(config.phantom_edge_rate_annual, years)
    identity_mutation_rate = _compound_rate(config.identity_mutation_rate_annual, years)
    dark_node_rate = config.dark_node_rate_per_release  # one release cycle = simulation
    dark_edge_rate = _compound_rate(config.dark_edge_rate_annual, years)

    all_ids = df["entity_id"].values
    all_types = df["entity_type"].values

    # Partition into edge-type and node-type rows
    edge_mask = np.array([_is_edge_type(t) for t in all_types])
    node_mask = ~edge_mask

    node_ids = all_ids[node_mask]
    edge_ids = all_ids[edge_mask]

    # ------------------------------------------------------------------
    # 1. Phantom CI (nodes): remove from declared CMDB
    # ------------------------------------------------------------------
    phantom_ci_count = max(0, int(round(len(node_ids) * phantom_ci_rate)))
    if phantom_ci_count > 0:
        chosen = rng.choice(node_ids, size=phantom_ci_count, replace=False)
        inject_months = rng.integers(1, max(2, simulation_months + 1), size=phantom_ci_count)
        for eid, month in zip(chosen, inject_months):
            ground_truth_rows.append(
                {
                    "entity_id": eid,
                    "divergence_type": "phantom_ci",
                    "injected_at_month": int(month),
                    "expected_detection_score": 0.8,
                }
            )
        # Remove phantom CIs from the degraded snapshot
        df = df[~df["entity_id"].isin(set(chosen))].copy()

    # ------------------------------------------------------------------
    # 2. Phantom edge: remove from declared CMDB
    # ------------------------------------------------------------------
    phantom_edge_count = max(0, int(round(len(edge_ids) * phantom_edge_rate)))
    if phantom_edge_count > 0 and len(edge_ids) > 0:
        chosen_edges = rng.choice(edge_ids, size=min(phantom_edge_count, len(edge_ids)), replace=False)
        inject_months = rng.integers(1, max(2, simulation_months + 1), size=len(chosen_edges))
        for eid, month in zip(chosen_edges, inject_months):
            ground_truth_rows.append(
                {
                    "entity_id": eid,
                    "divergence_type": "phantom_edge",
                    "injected_at_month": int(month),
                    "expected_detection_score": 0.8,
                }
            )
        df = df[~df["entity_id"].isin(set(chosen_edges))].copy()

    # Refresh node/edge arrays after removals
    remaining_ids = df["entity_id"].values
    remaining_types = df["entity_type"].values
    remaining_edge_mask = np.array([_is_edge_type(t) for t in remaining_types])
    remaining_node_mask = ~remaining_edge_mask
    remaining_node_ids = remaining_ids[remaining_node_mask]
    remaining_edge_ids = remaining_ids[remaining_edge_mask]

    # ------------------------------------------------------------------
    # 3. Identity mutation: suffix entity_id with '-MUT'
    # ------------------------------------------------------------------
    mutation_count = max(0, int(round(len(remaining_node_ids) * identity_mutation_rate)))
    if mutation_count > 0 and len(remaining_node_ids) > 0:
        chosen = rng.choice(
            remaining_node_ids, size=min(mutation_count, len(remaining_node_ids)), replace=False
        )
        inject_months = rng.integers(1, max(2, simulation_months + 1), size=len(chosen))
        chosen_set = set(chosen)
        for eid, month in zip(chosen, inject_months):
            ground_truth_rows.append(
                {
                    "entity_id": eid,
                    "divergence_type": "identity_mutation",
                    "injected_at_month": int(month),
                    "expected_detection_score": 0.7,
                }
            )
        # Mutate entity_id in the degraded snapshot
        df["entity_id"] = df["entity_id"].apply(
            lambda eid: f"{eid}-MUT" if eid in chosen_set else eid
        )

    # ------------------------------------------------------------------
    # 4. Dark node: present in reality, absent from CMDB view
    #    (record in ground truth; row stays in df to represent reality)
    # ------------------------------------------------------------------
    dark_node_count = max(0, int(round(len(remaining_node_ids) * dark_node_rate)))
    if dark_node_count > 0 and len(remaining_node_ids) > 0:
        # Exclude already-mutated IDs for clarity
        candidates = [
            eid for eid in remaining_node_ids
            if not str(eid).endswith("-MUT")
        ]
        if candidates:
            actual_count = min(dark_node_count, len(candidates))
            chosen = rng.choice(candidates, size=actual_count, replace=False)
            inject_months = rng.integers(1, max(2, simulation_months + 1), size=len(chosen))
            for eid, month in zip(chosen, inject_months):
                ground_truth_rows.append(
                    {
                        "entity_id": eid,
                        "divergence_type": "dark_node",
                        "injected_at_month": int(month),
                        "expected_detection_score": 0.6,
                    }
                )

    # ------------------------------------------------------------------
    # 5. Dark edge: present in reality, absent from CMDB view
    # ------------------------------------------------------------------
    dark_edge_count = max(0, int(round(len(remaining_edge_ids) * dark_edge_rate)))
    if dark_edge_count > 0 and len(remaining_edge_ids) > 0:
        chosen = rng.choice(
            remaining_edge_ids, size=min(dark_edge_count, len(remaining_edge_ids)), replace=False
        )
        inject_months = rng.integers(1, max(2, simulation_months + 1), size=len(chosen))
        for eid, month in zip(chosen, inject_months):
            ground_truth_rows.append(
                {
                    "entity_id": eid,
                    "divergence_type": "dark_edge",
                    "injected_at_month": int(month),
                    "expected_detection_score": 0.6,
                }
            )

    # ------------------------------------------------------------------
    # Build divergence ground truth DataFrame
    # ------------------------------------------------------------------
    if ground_truth_rows:
        divergence_ground_truth = pd.DataFrame(ground_truth_rows).astype(
            {
                "entity_id": str,
                "divergence_type": str,
                "injected_at_month": int,
                "expected_detection_score": float,
            }
        )
    else:
        divergence_ground_truth = pd.DataFrame(
            columns=["entity_id", "divergence_type", "injected_at_month", "expected_detection_score"]
        )

    return df.reset_index(drop=True), divergence_ground_truth.reset_index(drop=True)
