"""
Cascade Propagation Delay Model.

Provides stochastic delay sampling and fault-event injection for cross-domain
cascade scenarios.  This is a standalone module; it does not depend on any
other pedkai_generator step and carries no I/O side-effects.

Usage example::

    from datetime import datetime, timezone
    from pedkai_generator.cascade_model import (
        CascadeInjector, FaultEvent, STANDARD_PROFILES
    )

    root = FaultEvent(
        entity_id="CELL-JK-0001-LTE-1",
        domain="RAN",
        fault_type="sleeping_cell",
        timestamp=datetime(2024, 3, 15, 8, 0, tzinfo=timezone.utc),
    )

    injector = CascadeInjector(seed=42)
    cascade = injector.inject_cascade(
        root_fault=root,
        downstream_entities=[
            ("CORE-PGW-01", "CORE"),
            ("BSS-BILLING-01", "BSS"),
        ],
    )
    for ts, eid, ftype in injector.get_alarm_sequence(cascade):
        print(ts, eid, ftype)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PropagationProfile:
    """Describes the stochastic delay distribution for a source→target domain pair."""

    source_domain: str
    target_domain: str
    min_delay_minutes: float
    max_delay_minutes: float
    delay_distribution: Literal["uniform", "lognormal", "exponential"]


@dataclass
class FaultEvent:
    """A single fault / alarm event produced by the cascade model."""

    entity_id: str
    domain: str
    fault_type: str
    timestamp: datetime
    severity: str = "HIGH"
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Standard propagation profiles
# ---------------------------------------------------------------------------

STANDARD_PROFILES: dict[str, PropagationProfile] = {
    "ran_to_core": PropagationProfile(
        source_domain="RAN",
        target_domain="CORE",
        min_delay_minutes=8.0,
        max_delay_minutes=25.0,
        delay_distribution="lognormal",
    ),
    "transport_to_ran": PropagationProfile(
        source_domain="TRANSPORT",
        target_domain="RAN",
        min_delay_minutes=3.0,
        max_delay_minutes=15.0,
        delay_distribution="lognormal",
    ),
    "core_to_bss": PropagationProfile(
        source_domain="CORE",
        target_domain="BSS",
        min_delay_minutes=15.0,
        max_delay_minutes=45.0,
        delay_distribution="exponential",
    ),
    "within_domain": PropagationProfile(
        source_domain="ANY",
        target_domain="ANY",
        min_delay_minutes=2.0,
        max_delay_minutes=8.0,
        delay_distribution="uniform",
    ),
}


# ---------------------------------------------------------------------------
# CascadeInjector
# ---------------------------------------------------------------------------


class CascadeInjector:
    """
    Generates downstream fault events with stochastic propagation delays.

    Parameters
    ----------
    seed:
        Integer seed for the internal NumPy RNG to ensure reproducibility.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_delay(self, profile: PropagationProfile) -> float:
        """Sample a propagation delay (in minutes) from the given profile.

        Distributions
        -------------
        uniform
            Uniform between ``min_delay_minutes`` and ``max_delay_minutes``.

        lognormal
            Parameterised so the mean of the underlying normal equals
            ``(min + max) / 2`` and approximately 95 % of samples fall
            within ``[min, max]``.  Concretely:

            * ``mu  = ln((min + max) / 2)``
            * ``sigma = (ln(max) - ln(min)) / 4``   (4 ≈ 2 × 1.96 on the
              log scale, placing ~95 % of the distribution inside the range)

            The sampled value is **clamped** to ``[min, max]`` so that all
            returned delays are strictly within the advertised range.

        exponential
            Mean set to ``(min + max) / 2``.  Sampled value is clamped to
            ``[min, max]``.

        Returns
        -------
        float
            Delay in minutes, guaranteed to lie in
            ``[profile.min_delay_minutes, profile.max_delay_minutes]``.
        """
        lo = profile.min_delay_minutes
        hi = profile.max_delay_minutes
        dist = profile.delay_distribution

        if dist == "uniform":
            return float(self.rng.uniform(lo, hi))

        if dist == "lognormal":
            # mu and sigma derived from desired [lo, hi] range on the log scale
            mu = np.log((lo + hi) / 2.0)
            # sigma chosen so that [mu-2sigma, mu+2sigma] on log scale maps to
            # roughly [lo, hi] — empirically ~95 % coverage.
            sigma = (np.log(hi) - np.log(lo)) / 4.0
            sample = float(self.rng.lognormal(mean=mu, sigma=sigma))
            return float(np.clip(sample, lo, hi))

        if dist == "exponential":
            mean = (lo + hi) / 2.0
            sample = float(self.rng.exponential(scale=mean))
            return float(np.clip(sample, lo, hi))

        raise ValueError(f"Unknown delay_distribution: {dist!r}")

    def inject_cascade(
        self,
        root_fault: FaultEvent,
        downstream_entities: list[tuple[str, str]],  # (entity_id, domain)
        profile: PropagationProfile | None = None,
    ) -> list[FaultEvent]:
        """Generate a list of FaultEvents representing a cascade.

        The root fault is always the first element.  For each downstream
        entity a new FaultEvent is created whose timestamp equals::

            root_fault.timestamp + timedelta(minutes=sampled_delay)

        If *profile* is ``None`` the method calls
        :meth:`get_profile_for_domains` for each downstream entity using
        its domain, falling back to the ``within_domain`` profile when no
        specific profile is registered.

        Parameters
        ----------
        root_fault:
            The originating fault event.
        downstream_entities:
            Ordered list of ``(entity_id, domain)`` tuples affected by the
            cascade.
        profile:
            Optional override profile applied to every downstream entity.
            If ``None``, per-entity profiles are selected automatically.

        Returns
        -------
        list[FaultEvent]
            ``[root_fault] + downstream_events`` where downstream events are
            **not** sorted — call :meth:`get_alarm_sequence` for a sorted view.
        """
        events: list[FaultEvent] = [root_fault]

        for entity_id, domain in downstream_entities:
            if profile is not None:
                active_profile = profile
            else:
                active_profile = self.get_profile_for_domains(root_fault.domain, domain)

            delay_minutes = self.sample_delay(active_profile)
            event_ts = root_fault.timestamp + timedelta(minutes=delay_minutes)

            downstream_event = FaultEvent(
                entity_id=entity_id,
                domain=domain,
                fault_type=root_fault.fault_type,
                timestamp=event_ts,
                severity=root_fault.severity,
                metadata={
                    "root_entity_id": root_fault.entity_id,
                    "propagation_delay_minutes": delay_minutes,
                    "profile_used": f"{active_profile.source_domain}_to_{active_profile.target_domain}",
                },
            )
            events.append(downstream_event)

        return events

    def get_alarm_sequence(
        self, cascade: list[FaultEvent]
    ) -> list[tuple[datetime, str, str]]:
        """Return the cascade as a time-ordered list of ``(timestamp, entity_id, fault_type)`` tuples.

        Parameters
        ----------
        cascade:
            List of FaultEvents as returned by :meth:`inject_cascade`.

        Returns
        -------
        list of (datetime, str, str)
            Sorted ascending by timestamp.
        """
        return sorted(
            [(e.timestamp, e.entity_id, e.fault_type) for e in cascade],
            key=lambda x: x[0],
        )

    def get_profile_for_domains(
        self, source_domain: str, target_domain: str
    ) -> PropagationProfile:
        """Select the best-matching :class:`PropagationProfile` for a source→target domain pair.

        Matching strategy (first match wins):

        1. Exact match on both ``source_domain`` and ``target_domain``.
        2. Fallback: ``within_domain`` profile.

        Parameters
        ----------
        source_domain:
            Domain of the originating fault (e.g. ``"RAN"``).
        target_domain:
            Domain of the downstream entity (e.g. ``"CORE"``).

        Returns
        -------
        PropagationProfile
        """
        for profile in STANDARD_PROFILES.values():
            if profile.source_domain == "ANY" or profile.target_domain == "ANY":
                # Skip wildcard entries during exact matching
                continue
            if profile.source_domain == source_domain and profile.target_domain == target_domain:
                return profile

        # Fallback to within_domain
        return STANDARD_PROFILES["within_domain"]
