"""
Diurnal Traffic Profiles — Streaming Hourly Load Generator.

Generates realistic 24-hour traffic profiles that vary by:
  - Deployment profile (dense_urban, urban, suburban, rural, deep_rural, indoor)
  - Day of week (weekday vs weekend)
  - Timezone (WIB/WITA/WIT — Indonesian timezones, UTC+7/+8/+9)

ARCHITECTURE (memory-safe):
  Instead of pre-allocating (720, 66k) matrices (~363 MB each × 4 = 1.45 GB),
  this module uses a **streaming AR(1) state-machine** that yields one hour's
  worth of conditions at a time.  Peak memory per environmental variable is
  O(n_cells) ≈ 66k × 8 bytes ≈ 500 KB.  Total peak ≈ ~10 MB across all four
  condition vectors, vs ~4 GB in the bulk approach.

  The AR(1) state vectors carry temporal autocorrelation forward between hours
  without storing the full time series.

The load factor is a float in [0, 1] representing the fraction of peak-hour
capacity that the cell is experiencing at each hour.  This drives:
  - PRB utilisation
  - Active UE count
  - Interference (from neighbour cell load)
  - Throughput (available resources)

The profiles are based on real-world traffic patterns observed in mobile
networks, with deployment-specific adjustments:
  - Dense urban: sharp morning commute peak, sustained midday, evening peak
  - Urban: similar but lower amplitude; office-hour bias
  - Suburban: residential pattern — morning dip, evening peak
  - Rural/deep_rural: lower overall, broader peaks, less diurnal variation
  - Indoor: office-hour dominated (enterprise DAS / in-building)

Weekend profiles shift peaks later and reduce office-hour traffic.

Each profile also includes per-cell randomisation (jitter) so cells within
the same deployment profile don't have identical load curves.

Design references:
  - THREAD_SUMMARY Section 3: Simulation Parameters (30d × 1h intervals)
  - THREAD_SUMMARY Section 5: KPIs by Domain (cell-level radio KPIs)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator

import numpy as np

# ---------------------------------------------------------------------------
# Timezone offsets (hours from UTC)
# ---------------------------------------------------------------------------

TIMEZONE_UTC_OFFSET: dict[str, int] = {
    "WIB": 7,  # Western Indonesia Time (UTC+7) — Java, Sumatra
    "WITA": 8,  # Central Indonesia Time (UTC+8) — Kalimantan, Bali, NTT/NTB, Sulawesi
    "WIT": 9,  # Eastern Indonesia Time (UTC+9) — Papua, Maluku
}


# ---------------------------------------------------------------------------
# Base 24-hour load profiles (local time, hour 0 = midnight)
#
# Values are load factors in [0, 1].  Hour indices 0..23.
# These are the "canonical" curves; per-cell jitter is added at runtime.
# ---------------------------------------------------------------------------

# Dense urban — weekday: dual-peak pattern (RF-09 remediation).
# Peak 1: morning commute 08:30-09:30, sustained office-hour plateau 10-16.
# Peak 2: evening residential streaming/social 18-20.
# This matches observed dense-urban behaviour in large metro areas where
# commuter concentration creates a distinct morning peak.
_DENSE_URBAN_WEEKDAY = np.array(
    [
        0.12,
        0.08,
        0.06,
        0.05,
        0.05,
        0.06,  # 00-05: deep night
        0.18,
        0.42,
        0.72,
        0.88,
        0.82,
        0.78,  # 06-11: morning commute peak → office plateau
        0.76,
        0.74,
        0.72,
        0.70,
        0.74,
        0.82,  # 12-17: afternoon dip + commute ramp
        0.90,
        0.96,
        1.00,
        0.90,
        0.68,
        0.38,  # 18-23: evening peak → wind-down
    ],
    dtype=np.float64,
)

# Dense urban — weekend: later wake-up, more sustained midday, slightly
# lower overall peak (less commuter concentration).
_DENSE_URBAN_WEEKEND = np.array(
    [
        0.15,
        0.10,
        0.07,
        0.06,
        0.05,
        0.05,  # 00-05
        0.08,
        0.15,
        0.30,
        0.50,
        0.62,
        0.70,  # 06-11: slower ramp
        0.75,
        0.78,
        0.76,
        0.73,
        0.75,
        0.82,  # 12-17: sustained midday
        0.88,
        0.92,
        0.95,
        0.88,
        0.68,
        0.38,  # 18-23: evening peak
    ],
    dtype=np.float64,
)

# Urban — weekday: office-hour bias with mild morning peak, moderate evening peak.
_URBAN_WEEKDAY = np.array(
    [
        0.10,
        0.07,
        0.05,
        0.04,
        0.04,
        0.06,  # 00-05
        0.15,
        0.35,
        0.62,
        0.75,
        0.72,
        0.70,  # 06-11: morning commute ramp
        0.72,
        0.70,
        0.68,
        0.66,
        0.70,
        0.78,  # 12-17: afternoon plateau + commute
        0.85,
        0.92,
        0.95,
        0.84,
        0.60,
        0.30,  # 18-23: evening peak
    ],
    dtype=np.float64,
)

_URBAN_WEEKEND = np.array(
    [
        0.12,
        0.08,
        0.06,
        0.05,
        0.04,
        0.05,  # 00-05
        0.07,
        0.12,
        0.25,
        0.42,
        0.55,
        0.63,  # 06-11
        0.68,
        0.72,
        0.70,
        0.67,
        0.70,
        0.76,  # 12-17
        0.82,
        0.88,
        0.90,
        0.82,
        0.60,
        0.32,  # 18-23
    ],
    dtype=np.float64,
)

# Suburban — weekday: residential pattern — morning dip (people leave for work),
# moderate midday (stay-at-home / WFH), strong evening peak.
_SUBURBAN_WEEKDAY = np.array(
    [
        0.08,
        0.05,
        0.04,
        0.03,
        0.03,
        0.05,  # 00-05
        0.10,
        0.22,
        0.35,
        0.42,
        0.45,
        0.48,  # 06-11
        0.50,
        0.48,
        0.46,
        0.48,
        0.55,
        0.68,  # 12-17
        0.80,
        0.90,
        0.95,
        0.85,
        0.60,
        0.28,  # 18-23: evening streaming peak
    ],
    dtype=np.float64,
)

_SUBURBAN_WEEKEND = np.array(
    [
        0.10,
        0.07,
        0.05,
        0.04,
        0.03,
        0.04,  # 00-05
        0.06,
        0.10,
        0.22,
        0.38,
        0.50,
        0.58,  # 06-11
        0.62,
        0.65,
        0.63,
        0.60,
        0.62,
        0.70,  # 12-17
        0.78,
        0.85,
        0.90,
        0.82,
        0.58,
        0.30,  # 18-23
    ],
    dtype=np.float64,
)

# Rural — weekday: broader, flatter peaks; less diurnal variation.
# Significant portion of traffic is M2M / IoT (constant background).
_RURAL_WEEKDAY = np.array(
    [
        0.08,
        0.06,
        0.05,
        0.04,
        0.04,
        0.06,  # 00-05
        0.10,
        0.18,
        0.28,
        0.35,
        0.38,
        0.40,  # 06-11
        0.42,
        0.40,
        0.38,
        0.40,
        0.45,
        0.52,  # 12-17
        0.60,
        0.68,
        0.72,
        0.62,
        0.42,
        0.20,  # 18-23
    ],
    dtype=np.float64,
)

_RURAL_WEEKEND = np.array(
    [
        0.09,
        0.07,
        0.05,
        0.04,
        0.04,
        0.05,  # 00-05
        0.07,
        0.12,
        0.22,
        0.32,
        0.38,
        0.42,  # 06-11
        0.45,
        0.46,
        0.44,
        0.42,
        0.45,
        0.52,  # 12-17
        0.58,
        0.65,
        0.70,
        0.60,
        0.40,
        0.20,  # 18-23
    ],
    dtype=np.float64,
)

# Deep rural — very flat, low overall utilisation, slight evening bump.
# High M2M/IoT fraction gives a constant baseline.
_DEEP_RURAL_WEEKDAY = np.array(
    [
        0.06,
        0.05,
        0.04,
        0.04,
        0.04,
        0.05,  # 00-05
        0.07,
        0.12,
        0.18,
        0.22,
        0.25,
        0.27,  # 06-11
        0.28,
        0.27,
        0.25,
        0.27,
        0.30,
        0.35,  # 12-17
        0.42,
        0.48,
        0.50,
        0.42,
        0.28,
        0.14,  # 18-23
    ],
    dtype=np.float64,
)

_DEEP_RURAL_WEEKEND = np.array(
    [
        0.07,
        0.05,
        0.04,
        0.04,
        0.04,
        0.04,  # 00-05
        0.06,
        0.09,
        0.15,
        0.22,
        0.27,
        0.30,  # 06-11
        0.32,
        0.33,
        0.31,
        0.30,
        0.32,
        0.37,  # 12-17
        0.42,
        0.47,
        0.50,
        0.42,
        0.28,
        0.14,  # 18-23
    ],
    dtype=np.float64,
)

# Indoor (enterprise DAS / in-building) — weekday: strong office-hour
# pattern peaking at 10:00-15:00 (RF-09 remediation).
# Enterprise DAS systems show near-zero traffic outside business hours
# because the building is empty.  The peak is firmly at midday, NOT at
# 20:00 — office workers leave by 17:00-18:00.
_INDOOR_WEEKDAY = np.array(
    [
        0.03,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,  # 00-05: building empty
        0.04,
        0.12,
        0.48,
        0.82,
        0.92,
        0.95,  # 06-11: arrival ramp → peak
        1.00,
        0.95,
        0.92,
        0.88,
        0.72,
        0.40,  # 12-17: lunch dip recovery → departure
        0.15,
        0.08,
        0.05,
        0.04,
        0.03,
        0.03,  # 18-23: building empty — near-zero
    ],
    dtype=np.float64,
)

# Indoor — weekend: much lower — only retail/leisure venues active.
# Enterprise office DAS should be near-silent on weekends; only retail
# or mixed-use buildings show moderate daytime activity.
_INDOOR_WEEKEND = np.array(
    [
        0.03,
        0.02,
        0.02,
        0.02,
        0.02,
        0.02,  # 00-05
        0.03,
        0.04,
        0.08,
        0.15,
        0.22,
        0.28,  # 06-11: retail/leisure slowly opens
        0.32,
        0.34,
        0.32,
        0.28,
        0.22,
        0.15,  # 12-17: modest daytime
        0.10,
        0.07,
        0.05,
        0.04,
        0.03,
        0.03,  # 18-23: empties out
    ],
    dtype=np.float64,
)

# ---------------------------------------------------------------------------
# Assembled profile registry
# ---------------------------------------------------------------------------

# (weekday_profile, weekend_profile) keyed by deployment_profile string.
BASE_PROFILES: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "dense_urban": (_DENSE_URBAN_WEEKDAY, _DENSE_URBAN_WEEKEND),
    "urban": (_URBAN_WEEKDAY, _URBAN_WEEKEND),
    "suburban": (_SUBURBAN_WEEKDAY, _SUBURBAN_WEEKEND),
    "rural": (_RURAL_WEEKDAY, _RURAL_WEEKEND),
    "deep_rural": (_DEEP_RURAL_WEEKDAY, _DEEP_RURAL_WEEKEND),
    "indoor": (_INDOOR_WEEKDAY, _INDOOR_WEEKEND),
}


# ---------------------------------------------------------------------------
# Special events / anomaly overlays
# ---------------------------------------------------------------------------

# Friday evening boost: social/entertainment traffic spike on Friday nights.
# Applied as a multiplier on hours 18-23 for Friday (weekday index 4).
FRIDAY_EVENING_BOOST = np.array(
    [
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,  # 00-05: no change
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,  # 06-11: no change
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.02,  # 12-17: slight pre-weekend bump
        1.05,
        1.08,
        1.10,
        1.08,
        1.05,
        1.02,  # 18-23: Friday night boost
    ],
    dtype=np.float64,
)

# Ramadan overlay for Indonesian network: late-night traffic increase
# during the holy month (pre-dawn meals, religious activities).
# This is applied optionally for a configurable date range.
RAMADAN_OVERLAY = np.array(
    [
        1.15,
        1.20,
        1.25,
        1.30,
        1.20,
        1.10,  # 00-05: suhoor / pre-dawn activity
        1.0,
        1.0,
        1.0,
        0.95,
        0.92,
        0.90,  # 06-11: slightly lower daytime (fasting)
        0.88,
        0.90,
        0.92,
        0.95,
        1.0,
        1.05,  # 12-17: approaching iftar
        1.15,
        1.18,
        1.15,
        1.10,
        1.12,
        1.15,  # 18-23: iftar + evening activity
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Profile generation dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrafficProfileConfig:
    """Configuration for traffic profile generation."""

    # Number of simulation days
    simulation_days: int = 30

    # Start day-of-week: 0 = Monday, 6 = Sunday.
    # Default: 0 (simulation starts on a Monday).
    start_day_of_week: int = 0

    # Per-cell load jitter: standard deviation of multiplicative noise
    # applied to the base profile to decorrelate cells.
    # RF-14 remediation: increased from 0.08 to 0.13 to broaden the
    # intra-day noise floor and lower the 24h FFT power concentration.
    cell_jitter_std: float = 0.13

    # Temporal autocorrelation: how much each hour's jitter is correlated
    # with the previous hour's (0 = uncorrelated, 1 = fully correlated).
    # RF-14 remediation: reduced from 0.7 to 0.55 to lower lag-24
    # autocorrelation from ~0.96 to ~0.80-0.85 (matching real networks).
    temporal_correlation: float = 0.55

    # Whether to apply Friday evening boost.
    apply_friday_boost: bool = True

    # Ramadan overlay: (start_day, end_day) as 0-indexed simulation days.
    # None means no Ramadan overlay.  Example: (0, 29) for entire simulation.
    ramadan_day_range: tuple[int, int] | None = None

    # Gradual weekly growth trend: fractional increase per week.
    # e.g., 0.02 = 2% traffic growth per week.
    weekly_growth_rate: float = 0.005

    # NSA SCG leg load correlation with LTE anchor:
    # SCG leg load tracks anchor load but with some offset.
    nsa_scg_load_fraction: float = 0.85


# ---------------------------------------------------------------------------
# Day-of-week helpers
# ---------------------------------------------------------------------------


def _get_day_of_week(sim_day: int, start_dow: int) -> int:
    """Return day-of-week (0=Mon..6=Sun) for a given simulation day."""
    return (start_dow + sim_day) % 7


def _is_weekend(dow: int) -> bool:
    """Saturday (5) or Sunday (6)."""
    return dow >= 5


def _is_friday(dow: int) -> bool:
    """Friday is day 4."""
    return dow == 4


# ---------------------------------------------------------------------------
# Streaming hourly conditions — the core memory-safe data structure
# ---------------------------------------------------------------------------


@dataclass
class HourlyEnvironment:
    """
    Environmental conditions for a single hour across all cells.

    This is the output of the streaming generator — one instance per hour,
    holding four 1-D arrays of shape (n_cells,).

    Total memory per instance ≈ 4 × n_cells × 8 bytes ≈ 2 MB for 66k cells.
    """

    load_factor: np.ndarray  # (n_cells,) — [0.01, 1.0]
    shadow_fading_db: np.ndarray  # (n_cells,) — dB
    interference_delta_db: np.ndarray  # (n_cells,) — dB
    active_ue_multiplier: np.ndarray  # (n_cells,) — [0.05, 1.2]


# ---------------------------------------------------------------------------
# AR(1) state vector — carries temporal correlation between hours
# ---------------------------------------------------------------------------


@dataclass
class _AR1State:
    """
    Mutable state for an AR(1) process: x[t] = rho * x[t-1] + innovation.

    Only stores the *current* value vector (n_cells,).
    """

    value: np.ndarray  # (n_cells,) current state
    rho: float  # autocorrelation coefficient
    innovation_std: np.ndarray  # (n_cells,) or scalar — std of the innovation term

    def advance(self, rng: np.random.Generator) -> np.ndarray:
        """Step forward one tick, return the new value (in-place update)."""
        innovation = rng.normal(0.0, self.innovation_std)
        self.value = self.rho * self.value + innovation
        return self.value


# ---------------------------------------------------------------------------
# Streaming environment generator
# ---------------------------------------------------------------------------


class StreamingEnvironmentGenerator:
    """
    Memory-safe streaming generator for hourly environmental conditions.

    Instead of pre-allocating (total_hours, n_cells) matrices, this class
    maintains compact AR(1) state vectors of shape (n_cells,) and yields
    one HourlyEnvironment per call to ``next_hour()``.

    Peak memory: ~10 MB for 66k cells (vs ~4 GB for the bulk approach).

    Usage::

        gen = StreamingEnvironmentGenerator(n_cells, deploy_profiles, timezones,
                                            simulation_days, rng)
        for hour_idx in range(total_hours):
            env = gen.next_hour()
            # env.load_factor, env.shadow_fading_db, etc. are (n_cells,) arrays
    """

    def __init__(
        self,
        n_cells: int,
        deployment_profiles: np.ndarray,
        timezones: np.ndarray,
        simulation_days: int,
        rng: np.random.Generator,
        profile_config: TrafficProfileConfig | None = None,
    ) -> None:
        self.n_cells = n_cells
        self.simulation_days = simulation_days
        self.total_hours = simulation_days * 24
        self.rng = rng

        if profile_config is None:
            self.cfg = TrafficProfileConfig(simulation_days=simulation_days)
        else:
            self.cfg = profile_config
            self.cfg.simulation_days = simulation_days

        # ── Pre-compute per-cell static properties (~500 KB total) ───
        self._utc_offsets = np.array(
            [TIMEZONE_UTC_OFFSET.get(tz, 7) for tz in timezones],
            dtype=np.int32,
        )

        # Map deployment profiles → indices for vectorised lookup
        profile_names = list(BASE_PROFILES.keys())
        profile_index_map = {name: idx for idx, name in enumerate(profile_names)}
        self._cell_profile_idx = np.array(
            [profile_index_map.get(dp, profile_index_map["suburban"]) for dp in deployment_profiles],
            dtype=np.int32,
        )

        # Stack all base profiles into (n_profiles, 2, 24) — tiny (~1 KB)
        self._all_profiles = np.zeros((len(profile_names), 2, 24), dtype=np.float64)
        for idx, name in enumerate(profile_names):
            wd, we = BASE_PROFILES[name]
            self._all_profiles[idx, 0, :] = wd
            self._all_profiles[idx, 1, :] = we

        # RF-10: Per-cell weekday multiplier — enterprise-heavy and urban
        # profiles carry significantly more traffic on weekdays than weekends
        # due to office/commuter activity.  This ensures the *national
        # aggregate* weekday traffic exceeds weekend by 15-25%.
        _WEEKDAY_BOOST_BY_PROFILE: dict[str, float] = {
            "dense_urban": 1.15,  # Strong office/commercial component
            "urban": 1.12,  # Moderate office component
            "suburban": 1.00,  # Residential — no weekday boost
            "rural": 1.00,
            "deep_rural": 1.00,
            "indoor": 1.30,  # Enterprise DAS — dramatic weekday dominance
        }
        self._weekday_multiplier = np.array(
            [_WEEKDAY_BOOST_BY_PROFILE.get(dp, 1.0) for dp in deployment_profiles],
            dtype=np.float64,
        )

        # Per-cell amplitude variation (some cells are "hotter") — (n_cells,)
        self._cell_amplitude = np.clip(1.0 + rng.normal(0.0, 0.06, size=n_cells), 0.75, 1.25)

        # Per-cell neighbour-load random offset (fixed per cell, varies across cells)
        # This represents structural differences in neighbour density.
        self._neighbour_load_offset = rng.uniform(0.0, 0.15, size=n_cells)

        # ── Initialise AR(1) state vectors ────────────────────────

        # 1. Load jitter AR(1): low autocorrelation, small amplitude
        jitter_rho = self.cfg.temporal_correlation
        jitter_innov_std = self.cfg.cell_jitter_std * np.sqrt(1.0 - jitter_rho**2) if jitter_rho < 1.0 else 0.0
        self._jitter_state = _AR1State(
            value=rng.normal(0.0, self.cfg.cell_jitter_std, size=n_cells),
            rho=jitter_rho,
            innovation_std=jitter_innov_std,
        )

        # 2. Shadow fading AR(1): high autocorrelation (slow fading)
        from pedkai_generator.step_03_radio_kpis.physics import DEPLOYMENT_PHYSICS

        sf_std = np.array(
            [
                DEPLOYMENT_PHYSICS.get(dp, DEPLOYMENT_PHYSICS["suburban"]).shadow_fading_std_db
                for dp in deployment_profiles
            ],
            dtype=np.float64,
        )
        sf_rho = 0.95
        sf_innov_std = sf_std * np.sqrt(1.0 - sf_rho**2)
        self._shadow_state = _AR1State(
            value=rng.normal(0.0, sf_std),
            rho=sf_rho,
            innovation_std=sf_innov_std,
        )

        # Current hour index (0-based from simulation start)
        self._hour_idx = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def next_hour(self) -> HourlyEnvironment:
        """
        Generate environmental conditions for the next hourly interval.

        Advances internal AR(1) states and returns a fresh HourlyEnvironment.
        Call this exactly ``total_hours`` times.

        Returns
        -------
        HourlyEnvironment with four (n_cells,) arrays.
        """
        t = self._hour_idx
        n = self.n_cells
        rng = self.rng

        # ── 1. Compute base load from diurnal profiles ────────────
        load = self._compute_load_for_hour(t)

        # ── 2. Advance shadow fading AR(1) ────────────────────────
        if t == 0:
            shadow_fading = self._shadow_state.value.copy()
        else:
            shadow_fading = self._shadow_state.advance(rng).copy()

        # ── 3. Compute interference delta (load-driven + noise) ───
        interference_delta = self._compute_interference_delta(load, rng)

        # ── 4. Compute active UE multiplier ───────────────────────
        ue_mult = self._compute_ue_multiplier(load, rng)

        self._hour_idx += 1

        return HourlyEnvironment(
            load_factor=load,
            shadow_fading_db=shadow_fading,
            interference_delta_db=interference_delta,
            active_ue_multiplier=ue_mult,
        )

    @property
    def hours_remaining(self) -> int:
        return self.total_hours - self._hour_idx

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _compute_load_for_hour(self, t: int) -> np.ndarray:
        """
        Compute per-cell load factor for hour ``t``.

        Uses the pre-computed profile lookup tables, AR(1) jitter state,
        and per-cell amplitude.  Returns a new (n_cells,) array.

        RF-10 remediation: applies a weekday multiplier for enterprise-heavy
        and urban profiles so that the national aggregate weekday traffic
        exceeds weekend by 15-25%.  In Indonesia, DKI Jakarta (21.6% of
        cells) dominates the network, and its commercial/enterprise traffic
        drops ~60% on weekends.
        """
        utc_hour_of_day = t % 24
        sim_day = t // 24

        dow = _get_day_of_week(sim_day, self.cfg.start_day_of_week)
        weekend = _is_weekend(dow)
        friday = _is_friday(dow)
        day_type_idx = 1 if weekend else 0

        # Local hour per cell (vectorised)
        local_hours = (utc_hour_of_day + self._utc_offsets) % 24

        # Look up base load: all_profiles[profile_idx, day_type, local_hour]
        base = self._all_profiles[self._cell_profile_idx, day_type_idx, local_hours]

        # RF-10: Weekday boost — enterprise/urban profiles carry more traffic
        # on weekdays (commuters, office workers).  Weekend profiles are
        # already lower in absolute terms, but this additional multiplier
        # ensures the *national aggregate* is weekday-dominant.
        if not weekend:
            weekday_boost = self._weekday_multiplier
            base = base * weekday_boost

        # Friday evening boost
        if friday and self.cfg.apply_friday_boost:
            base = base * FRIDAY_EVENING_BOOST[local_hours]

        # Ramadan overlay
        if self.cfg.ramadan_day_range is not None:
            r_start, r_end = self.cfg.ramadan_day_range
            if r_start <= sim_day <= r_end:
                base = base * RAMADAN_OVERLAY[local_hours]

        # Weekly growth trend
        week_number = sim_day / 7.0
        growth_factor = 1.0 + self.cfg.weekly_growth_rate * week_number
        base = base * growth_factor

        # Advance jitter AR(1) and apply (multiplicative)
        if t == 0:
            jitter = self._jitter_state.value
        else:
            jitter = self._jitter_state.advance(self.rng)

        load = base * self._cell_amplitude * (1.0 + jitter)

        return np.clip(load, 0.01, 1.0)

    def _compute_interference_delta(
        self,
        load: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Compute interference variation (dB delta from baseline) for one hour.

        Interference is load-dependent: when surrounding cells are heavily
        loaded, they increase inter-cell interference.

        Model:
            neighbour_load ≈ 0.6 * self_load + 0.4 * random + offset + noise
            delta_iot = 7.0 * (neighbour_load - 0.3) + noise

        Memory: allocates ~3 temporary (n_cells,) arrays ≈ ~1.5 MB total.
        """
        n = self.n_cells
        correlation = 0.6

        neighbour_load = (
            correlation * load
            + (1.0 - correlation) * rng.random(size=n)
            + self._neighbour_load_offset
            + rng.normal(0.0, 0.08, size=n)
        )
        np.clip(neighbour_load, 0.0, 1.0, out=neighbour_load)

        # Convert to dB delta: at load=0.3 → delta=0, at load=1.0 → delta≈+5 dB
        delta = 7.0 * (neighbour_load - 0.3)
        delta += rng.normal(0.0, 0.5, size=n)

        return delta

    def _compute_ue_multiplier(
        self,
        load: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Compute per-hour UE count multiplier from load factors.

        Model: ue_mult = 0.15 + 0.85 * load^0.8 + noise
        """
        ue_mult = 0.15 + 0.85 * np.power(load, 0.8)
        ue_mult += rng.normal(0.0, 0.03, size=self.n_cells)
        return np.clip(ue_mult, 0.05, 1.2)


# ---------------------------------------------------------------------------
# Convenience function for callers that want the old-style single-hour lookup
# (used by tests and simple scripts)
# ---------------------------------------------------------------------------


def get_base_load_for_hour(
    deployment_profile: str,
    local_hour: int,
    is_weekend_day: bool,
) -> float:
    """
    Look up the base load factor for a single deployment profile, hour, and
    day type (weekday/weekend).

    Returns a float in [0, 1].
    """
    profiles = BASE_PROFILES.get(deployment_profile, BASE_PROFILES["suburban"])
    profile = profiles[1] if is_weekend_day else profiles[0]
    return float(profile[local_hour % 24])
