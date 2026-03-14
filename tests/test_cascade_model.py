"""
Tests for cascade_model.py — CascadeInjector and PropagationProfile.

Test inventory (6 required + 2 bonus):
  1. test_ran_fault_generates_core_alarm_with_correct_delay_range
  2. test_alarm_sequence_is_time_ordered
  3. test_lognormal_delays_in_range
  4. test_cascade_includes_root_fault_first
  5. test_empty_downstream_returns_only_root
  6. test_profile_selection_fallback
  7. test_exponential_delays_in_range            (bonus)
  8. test_uniform_delays_in_range                (bonus)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.pedkai_generator.cascade_model import (
    CascadeInjector,
    FaultEvent,
    PropagationProfile,
    STANDARD_PROFILES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def injector() -> CascadeInjector:
    """Return a seeded CascadeInjector for deterministic tests."""
    return CascadeInjector(seed=42)


@pytest.fixture
def ran_root_fault() -> FaultEvent:
    """A RAN fault event anchored at a known UTC timestamp."""
    return FaultEvent(
        entity_id="CELL-JK-0001-LTE-1",
        domain="RAN",
        fault_type="hardware_fault",
        timestamp=datetime(2024, 3, 15, 8, 0, 0, tzinfo=timezone.utc),
        severity="HIGH",
    )


# ---------------------------------------------------------------------------
# Test 1 — RAN root produces CORE downstream event with delay in [8, 25] min
# ---------------------------------------------------------------------------


def test_ran_fault_generates_core_alarm_with_correct_delay_range(
    injector: CascadeInjector, ran_root_fault: FaultEvent
) -> None:
    """A RAN fault with a CORE downstream entity should produce a downstream
    event whose timestamp is between root + 8 min and root + 25 min."""
    cascade = injector.inject_cascade(
        root_fault=ran_root_fault,
        downstream_entities=[("CORE-PGW-01", "CORE")],
    )

    assert len(cascade) == 2
    downstream = cascade[1]

    delay = (downstream.timestamp - ran_root_fault.timestamp).total_seconds() / 60.0

    assert delay >= 8.0, f"Delay {delay:.2f} min is below minimum (8 min)"
    assert delay <= 25.0, f"Delay {delay:.2f} min exceeds maximum (25 min)"
    assert downstream.domain == "CORE"
    assert downstream.entity_id == "CORE-PGW-01"


# ---------------------------------------------------------------------------
# Test 2 — get_alarm_sequence returns events sorted by timestamp
# ---------------------------------------------------------------------------


def test_alarm_sequence_is_time_ordered(
    injector: CascadeInjector, ran_root_fault: FaultEvent
) -> None:
    """get_alarm_sequence must return tuples in strictly ascending timestamp order."""
    cascade = injector.inject_cascade(
        root_fault=ran_root_fault,
        downstream_entities=[
            ("CORE-PGW-01", "CORE"),
            ("CORE-SGW-02", "CORE"),
            ("BSS-BILLING-01", "BSS"),
        ],
    )

    sequence = injector.get_alarm_sequence(cascade)

    assert len(sequence) == 4
    timestamps = [ts for ts, _, _ in sequence]
    assert timestamps == sorted(timestamps), (
        "Alarm sequence is not sorted by timestamp"
    )


# ---------------------------------------------------------------------------
# Test 3 — lognormal delays fall within declared range ≥ 90 % of the time
# ---------------------------------------------------------------------------


def test_lognormal_delays_in_range(injector: CascadeInjector) -> None:
    """Sample 1000 delays from the ran_to_core (lognormal) profile.
    At least 90 % must fall within [8, 25] minutes."""
    profile = STANDARD_PROFILES["ran_to_core"]
    assert profile.delay_distribution == "lognormal"

    n_samples = 1000
    in_range = 0
    for _ in range(n_samples):
        delay = injector.sample_delay(profile)
        if profile.min_delay_minutes <= delay <= profile.max_delay_minutes:
            in_range += 1

    fraction_in_range = in_range / n_samples
    assert fraction_in_range >= 0.90, (
        f"Only {fraction_in_range:.1%} of lognormal samples in [{profile.min_delay_minutes}, "
        f"{profile.max_delay_minutes}] — expected ≥ 90 %"
    )


# ---------------------------------------------------------------------------
# Test 4 — root fault is the first element of the returned list
# ---------------------------------------------------------------------------


def test_cascade_includes_root_fault_first(
    injector: CascadeInjector, ran_root_fault: FaultEvent
) -> None:
    """The first element of inject_cascade output must be the root fault object."""
    cascade = injector.inject_cascade(
        root_fault=ran_root_fault,
        downstream_entities=[("CORE-PGW-01", "CORE")],
    )

    assert cascade[0] is ran_root_fault, (
        "First element is not the root fault object"
    )


# ---------------------------------------------------------------------------
# Test 5 — empty downstream list returns only root fault
# ---------------------------------------------------------------------------


def test_empty_downstream_returns_only_root(
    injector: CascadeInjector, ran_root_fault: FaultEvent
) -> None:
    """inject_cascade with no downstream entities must return a list of length 1
    containing only the root fault."""
    cascade = injector.inject_cascade(
        root_fault=ran_root_fault,
        downstream_entities=[],
    )

    assert len(cascade) == 1
    assert cascade[0] is ran_root_fault


# ---------------------------------------------------------------------------
# Test 6 — profile selection falls back to within_domain
# ---------------------------------------------------------------------------


def test_profile_selection_fallback(injector: CascadeInjector) -> None:
    """When no specific profile exists for the source→target domain pair,
    get_profile_for_domains must return the within_domain profile."""
    # "BSS" → "RAN" has no registered specific profile
    profile = injector.get_profile_for_domains("BSS", "RAN")
    within = STANDARD_PROFILES["within_domain"]

    assert profile is within, (
        "Expected within_domain fallback profile but got a different profile"
    )


# ---------------------------------------------------------------------------
# Bonus test 7 — exponential delays are always clamped to declared range
# ---------------------------------------------------------------------------


def test_exponential_delays_in_range(injector: CascadeInjector) -> None:
    """All 500 samples from the core_to_bss (exponential) profile must be
    within [15, 45] minutes after clamping."""
    profile = STANDARD_PROFILES["core_to_bss"]
    assert profile.delay_distribution == "exponential"

    for _ in range(500):
        delay = injector.sample_delay(profile)
        assert profile.min_delay_minutes <= delay <= profile.max_delay_minutes, (
            f"Exponential delay {delay:.2f} outside [{profile.min_delay_minutes}, "
            f"{profile.max_delay_minutes}]"
        )


# ---------------------------------------------------------------------------
# Bonus test 8 — uniform delays are always within declared range
# ---------------------------------------------------------------------------


def test_uniform_delays_in_range(injector: CascadeInjector) -> None:
    """All 500 samples from the within_domain (uniform) profile must be
    within [2, 8] minutes."""
    profile = STANDARD_PROFILES["within_domain"]
    assert profile.delay_distribution == "uniform"

    for _ in range(500):
        delay = injector.sample_delay(profile)
        assert profile.min_delay_minutes <= delay <= profile.max_delay_minutes, (
            f"Uniform delay {delay:.2f} outside [{profile.min_delay_minutes}, "
            f"{profile.max_delay_minutes}]"
        )


# ---------------------------------------------------------------------------
# Bonus test 9 — sample cascade output printed for reporting
# ---------------------------------------------------------------------------


def test_sample_cascade_output(
    injector: CascadeInjector, ran_root_fault: FaultEvent, capsys
) -> None:
    """Generates a 1-root + 2-downstream cascade and prints the alarm sequence.
    Used to produce sample output in CI logs."""
    cascade = injector.inject_cascade(
        root_fault=ran_root_fault,
        downstream_entities=[
            ("CORE-PGW-01", "CORE"),
            ("BSS-BILLING-01", "BSS"),
        ],
    )

    sequence = injector.get_alarm_sequence(cascade)

    print("\n--- Sample cascade alarm sequence ---")
    for ts, eid, ftype in sequence:
        print(f"  {ts.isoformat()}  {eid:<30}  {ftype}")
    print("--- End of sample ---")

    # Basic structural assertions
    assert len(cascade) == 3
    assert sequence[0][1] == ran_root_fault.entity_id, (
        "Root entity should appear first in time-sorted sequence"
    )
    captured = capsys.readouterr()
    assert "Sample cascade alarm sequence" in captured.out
