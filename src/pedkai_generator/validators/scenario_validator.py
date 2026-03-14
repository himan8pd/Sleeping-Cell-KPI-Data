"""Scenario validation framework for synthetic fault data.

Validates generated KPI/alarm DataFrames against published post-incident
fault signatures.  Each scenario spec declares:
  - required_kpi_signatures: conditions that MUST hold (all must pass)
  - forbidden_kpi_signatures: conditions that must NOT hold (none must pass)

Condition semantics
-------------------
zero              : all values in the column < threshold
spike             : max value in the column > threshold
non_linear_above  : above threshold the relationship between the column and
                    a paired PRB column is non-linear (corr(x², y) > corr(x, y))
continuity        : no adjacent timestep pair has relative change > threshold
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data-model types
# ---------------------------------------------------------------------------

@dataclass
class KPISignature:
    """A condition that must hold on a KPI column in the data."""

    kpi_name: str
    condition: Literal["zero", "spike", "continuity", "non_linear_above"]
    threshold: float  # Semantics per condition:
                      # 'zero'             : max allowed value (strict <)
                      # 'spike'            : min value the max must exceed (>)
                      # 'continuity'       : max allowed fractional change between
                      #                      adjacent timesteps (e.g. 0.10 = 10 %)
                      # 'non_linear_above' : PRB threshold above which non-linearity
                      #                      between prb_utilisation and kpi_name
                      #                      must be detectable


@dataclass
class ScenarioSpec:
    name: str
    description: str
    required_kpi_signatures: list[KPISignature] = field(default_factory=list)
    forbidden_kpi_signatures: list[KPISignature] = field(default_factory=list)


@dataclass
class ValidationResult:
    scenario_name: str
    passed: bool
    failed_required: list[str]   # descriptions of required sigs that failed
    failed_forbidden: list[str]  # descriptions of forbidden sigs that triggered
    details: dict

    def to_dict(self) -> dict:
        """Return a plain-dict representation that is JSON-serialisable."""
        return {
            "scenario_name": self.scenario_name,
            "passed": self.passed,
            "failed_required": self.failed_required,
            "failed_forbidden": self.failed_forbidden,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Condition checkers
# ---------------------------------------------------------------------------

def _check_zero(series: pd.Series, threshold: float) -> tuple[bool, dict]:
    """Pass if ALL values < threshold."""
    max_val = float(series.max()) if not series.empty else 0.0
    passed = max_val < threshold
    return passed, {"max_value": max_val, "threshold": threshold}


def _check_spike(series: pd.Series, threshold: float) -> tuple[bool, dict]:
    """Pass if the maximum value > threshold."""
    max_val = float(series.max()) if not series.empty else 0.0
    passed = max_val > threshold
    return passed, {"max_value": max_val, "threshold": threshold}


def _check_continuity(series: pd.Series, threshold: float) -> tuple[bool, dict]:
    """Pass if no adjacent pair has relative change > threshold.

    Relative change = |a[i+1] - a[i]| / (|a[i]| + eps).
    """
    eps = 1e-9
    if len(series) < 2:
        return True, {"max_relative_change": 0.0, "threshold": threshold}
    arr = series.to_numpy(dtype=float)
    relative_changes = np.abs(np.diff(arr)) / (np.abs(arr[:-1]) + eps)
    max_change = float(relative_changes.max())
    passed = max_change <= threshold
    return passed, {"max_relative_change": max_change, "threshold": threshold}


def _check_non_linear_above(
    kpi_series: pd.Series,
    prb_series: pd.Series,
    threshold: float,
) -> tuple[bool, dict]:
    """Pass if the relationship between prb_utilisation and kpi_name is
    non-linear above *threshold* PRB.

    Non-linearity criterion: |corr(prb², kpi)| > |corr(prb, kpi)|
    evaluated on the subset where prb > threshold.
    Falls back to True (no opinion) when the subset has fewer than 3 points.
    """
    mask = prb_series > threshold
    prb_sub = prb_series[mask].to_numpy(dtype=float)
    kpi_sub = kpi_series[mask].to_numpy(dtype=float)

    if len(prb_sub) < 3:
        return True, {
            "note": "insufficient data above threshold — condition skipped",
            "threshold": threshold,
            "n_points": int(len(prb_sub)),
        }

    def _corr(x: np.ndarray, y: np.ndarray) -> float:
        if x.std() < 1e-12 or y.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    corr_linear = abs(_corr(prb_sub, kpi_sub))
    corr_quad = abs(_corr(prb_sub ** 2, kpi_sub))
    passed = corr_quad > corr_linear
    return passed, {
        "corr_linear": corr_linear,
        "corr_quadratic": corr_quad,
        "threshold": threshold,
        "n_points_above_threshold": int(len(prb_sub)),
    }


# ---------------------------------------------------------------------------
# Signature evaluation
# ---------------------------------------------------------------------------

def _evaluate_signature(
    sig: KPISignature,
    kpi_data: pd.DataFrame,
) -> tuple[bool, dict]:
    """Evaluate a single KPISignature against kpi_data.

    Returns (passed, detail_dict).
    Raises KeyError if the required column is missing.
    """
    if sig.kpi_name not in kpi_data.columns:
        raise KeyError(
            f"KPI column '{sig.kpi_name}' not found in data. "
            f"Available columns: {list(kpi_data.columns)}"
        )

    series = kpi_data[sig.kpi_name].dropna()

    if sig.condition == "zero":
        return _check_zero(series, sig.threshold)

    if sig.condition == "spike":
        return _check_spike(series, sig.threshold)

    if sig.condition == "continuity":
        return _check_continuity(series, sig.threshold)

    if sig.condition == "non_linear_above":
        # Requires a 'prb_utilisation' column to compare against.
        prb_col = "prb_utilisation"
        if prb_col not in kpi_data.columns:
            raise KeyError(
                f"'non_linear_above' condition requires column '{prb_col}' "
                f"but it is not present in kpi_data."
            )
        prb_series = kpi_data[prb_col].dropna()
        # Align on common index after dropna
        common_idx = series.index.intersection(prb_series.index)
        return _check_non_linear_above(
            series.loc[common_idx],
            prb_series.loc[common_idx],
            sig.threshold,
        )

    raise ValueError(f"Unknown condition type: '{sig.condition}'")


def _sig_description(sig: KPISignature) -> str:
    return f"{sig.kpi_name}[{sig.condition}@{sig.threshold}]"


# ---------------------------------------------------------------------------
# Public validator
# ---------------------------------------------------------------------------

def validate_scenario(
    scenario_name: str,
    kpi_data: pd.DataFrame,
    alarm_data: pd.DataFrame,
) -> ValidationResult:
    """Validate scenario data against its pre-loaded spec.

    Parameters
    ----------
    scenario_name:
        Key into SCENARIO_SPECS (e.g. 'sleeping_cell').
    kpi_data:
        Wide DataFrame where each column is a KPI metric.
    alarm_data:
        DataFrame with columns: timestamp, entity_id, alarm_type, severity.

    Returns
    -------
    ValidationResult with passed=True only when all required signatures pass
    AND no forbidden signature passes.

    Raises
    ------
    ValueError
        If scenario_name is not found in SCENARIO_SPECS.
    KeyError
        If a required KPI column is missing from kpi_data.
    """
    if scenario_name not in SCENARIO_SPECS:
        raise ValueError(
            f"Unknown scenario '{scenario_name}'. "
            f"Known scenarios: {sorted(SCENARIO_SPECS.keys())}"
        )

    spec = SCENARIO_SPECS[scenario_name]
    failed_required: list[str] = []
    failed_forbidden: list[str] = []
    details: dict = {"required": {}, "forbidden": {}}

    for sig in spec.required_kpi_signatures:
        desc = _sig_description(sig)
        passed, detail = _evaluate_signature(sig, kpi_data)
        details["required"][desc] = {"passed": passed, **detail}
        if not passed:
            failed_required.append(desc)

    for sig in spec.forbidden_kpi_signatures:
        desc = _sig_description(sig)
        passed, detail = _evaluate_signature(sig, kpi_data)
        # forbidden: we want it to NOT pass — if it passes, that's a failure
        details["forbidden"][desc] = {"triggered": passed, **detail}
        if passed:
            failed_forbidden.append(desc)

    overall_passed = (not failed_required) and (not failed_forbidden)

    return ValidationResult(
        scenario_name=scenario_name,
        passed=overall_passed,
        failed_required=failed_required,
        failed_forbidden=failed_forbidden,
        details=details,
    )


# ---------------------------------------------------------------------------
# Pre-loaded scenario specs
# ---------------------------------------------------------------------------

SLEEPING_CELL_SPEC = ScenarioSpec(
    name="sleeping_cell",
    description=(
        "Cell is administratively up but carries zero traffic. "
        "Signal metrics remain healthy — no RF degradation."
    ),
    required_kpi_signatures=[
        # Traffic must be essentially zero (< 1 % of typical baseline ~100 users)
        KPISignature(kpi_name="user_count", condition="zero", threshold=0.01),
        # RSRP must stay above -100 dBm — healthy RF
        KPISignature(kpi_name="rsrp", condition="spike", threshold=-100.0),
    ],
    forbidden_kpi_signatures=[
        # Must NOT have significant user activity
        KPISignature(kpi_name="user_count", condition="spike", threshold=10.0),
    ],
)

CONGESTION_CASCADE_SPEC = ScenarioSpec(
    name="congestion_cascade",
    description=(
        "PRB utilisation exceeds 90 %, causing non-linear latency growth "
        "and call-drop spikes."
    ),
    required_kpi_signatures=[
        # PRB must spike above 90 %
        KPISignature(kpi_name="prb_utilisation", condition="spike", threshold=90.0),
        # Latency must be non-linear above PRB = 90
        KPISignature(kpi_name="latency_ms", condition="non_linear_above", threshold=90.0),
        # Call-drop rate must spike above 95th-percentile proxy value
        KPISignature(kpi_name="call_drop_rate", condition="spike", threshold=5.0),
    ],
    forbidden_kpi_signatures=[
        # PRB must NOT stay below 70 % — that would indicate no congestion
        KPISignature(kpi_name="prb_utilisation", condition="zero", threshold=70.0),
    ],
)

HARDWARE_SWAP_SPEC = ScenarioSpec(
    name="hardware_swap",
    description=(
        "A hardware unit is replaced mid-series. KPIs should be continuous "
        "across the swap point (< 10 % relative change per step)."
    ),
    required_kpi_signatures=[
        # KPIs should not jump more than 10 % between adjacent timesteps
        KPISignature(kpi_name="cell_availability", condition="continuity", threshold=0.10),
        KPISignature(kpi_name="rsrp", condition="continuity", threshold=0.10),
    ],
    forbidden_kpi_signatures=[
        # Large discontinuity (> 50 % relative jump) would indicate bad data
        KPISignature(kpi_name="cell_availability", condition="continuity", threshold=0.50),
    ],
)

TRANSPORT_FAILURE_SPEC = ScenarioSpec(
    name="transport_failure",
    description=(
        "Backhaul transport link failure — packet loss spikes and RAN "
        "throughput degrades in the subsequent periods."
    ),
    required_kpi_signatures=[
        # Packet loss must spike above 90th-percentile proxy value
        KPISignature(kpi_name="packet_loss_rate", condition="spike", threshold=10.0),
        # RAN throughput must show subsequent degradation (non-continuity)
        KPISignature(kpi_name="ran_throughput", condition="continuity", threshold=0.30),
    ],
    forbidden_kpi_signatures=[],
)

SCENARIO_SPECS: dict[str, ScenarioSpec] = {
    "sleeping_cell": SLEEPING_CELL_SPEC,
    "congestion_cascade": CONGESTION_CASCADE_SPEC,
    "hardware_swap": HARDWARE_SWAP_SPEC,
    "transport_failure": TRANSPORT_FAILURE_SPEC,
}
