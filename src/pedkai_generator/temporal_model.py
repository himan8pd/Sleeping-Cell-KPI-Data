"""
Temporal Model — Additive KPI Decomposition for Synthetic Data Generator.

Provides:
  - Cell-type specific diurnal profiles (residential, enterprise, transport_hub,
    stadium, mixed)
  - Day-of-week profiles
  - Seasonal calendar with Ramadan detection and Indonesian public holidays
  - Full additive model:  KPI(t) = base × diurnal(t) × dow(t) × seasonal(t) + ar_residual(t)

This module complements step_03_radio_kpis/profiles.py which handles deployment-type
profiles (dense_urban, urban, suburban, rural, indoor) and the AR(1) streaming generator.
Here we focus on cell-type semantics and seasonal effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Diurnal profiles by cell type
# ---------------------------------------------------------------------------
# Each profile is a 24-element list of hourly multipliers.
# They must sum to 24.0 for energy conservation (mean multiplier = 1.0).
#
# Design notes per cell type:
#   residential:    Low 2-6AM (~0.3), morning ramp 6-9AM, evening peak 7-10PM (~1.8)
#   enterprise:     Office-hour dominant 9AM-6PM (~1.6), low evenings (~0.5)
#   transport_hub:  Morning commute peak 7-9AM, evening 5-7PM, low nights
#   stadium:        Flat low baseline with spike during match hours (7-10PM)
#   mixed:          Average of residential + enterprise (blended land use)
# ---------------------------------------------------------------------------


@dataclass
class DiurnalProfile:
    """24-float array of hourly multipliers (must sum to 24.0 for energy conservation)."""

    name: str
    hourly_multipliers: list[float]  # length 24, sum=24.0


def _normalise_to_24(raw: list[float]) -> list[float]:
    """Scale a raw 24-element list so it sums to exactly 24.0."""
    total = sum(raw)
    scale = 24.0 / total
    return [v * scale for v in raw]


# Raw residential profile — evening-heavy, very low pre-dawn.
# Values represent *relative* traffic intensity per hour.
_RESIDENTIAL_RAW = [
    0.30,  # 00
    0.25,  # 01
    0.20,  # 02
    0.18,  # 03
    0.20,  # 04
    0.32,  # 05
    0.60,  # 06
    0.90,  # 07
    1.05,  # 08
    1.00,  # 09
    0.85,  # 10
    0.80,  # 11
    0.82,  # 12
    0.80,  # 13
    0.82,  # 14
    0.88,  # 15
    1.00,  # 16
    1.15,  # 17
    1.40,  # 18
    1.80,  # 19
    1.75,  # 20
    1.50,  # 21
    1.10,  # 22
    0.65,  # 23
]

# Enterprise profile — strong 9-17 office hours, very low outside.
_ENTERPRISE_RAW = [
    0.20,  # 00
    0.18,  # 01
    0.18,  # 02
    0.18,  # 03
    0.18,  # 04
    0.22,  # 05
    0.35,  # 06
    0.70,  # 07
    1.20,  # 08
    1.60,  # 09
    1.65,  # 10
    1.60,  # 11
    1.50,  # 12
    1.55,  # 13
    1.60,  # 14
    1.55,  # 15
    1.40,  # 16
    1.00,  # 17
    0.65,  # 18
    0.50,  # 19
    0.45,  # 20
    0.40,  # 21
    0.32,  # 22
    0.24,  # 23
]

# Transport hub — two commute peaks, minimal overnight.
_TRANSPORT_HUB_RAW = [
    0.30,  # 00
    0.25,  # 01
    0.20,  # 02
    0.18,  # 03
    0.22,  # 04
    0.50,  # 05
    0.95,  # 06
    1.60,  # 07
    1.70,  # 08
    1.30,  # 09
    1.00,  # 10
    0.90,  # 11
    0.95,  # 12
    0.92,  # 13
    0.90,  # 14
    0.98,  # 15
    1.40,  # 16
    1.65,  # 17
    1.50,  # 18
    1.20,  # 19
    0.90,  # 20
    0.70,  # 21
    0.55,  # 22
    0.40,  # 23
]

# Stadium — flat low all day, then a concentrated evening spike (7-10PM match hours).
_STADIUM_RAW = [
    0.40,  # 00
    0.35,  # 01
    0.30,  # 02
    0.28,  # 03
    0.28,  # 04
    0.30,  # 05
    0.32,  # 06
    0.35,  # 07
    0.38,  # 08
    0.40,  # 09
    0.42,  # 10
    0.45,  # 11
    0.50,  # 12
    0.55,  # 13
    0.60,  # 14
    0.70,  # 15
    0.85,  # 16
    1.00,  # 17
    1.40,  # 18
    2.20,  # 19 — match start
    2.50,  # 20 — match peak
    2.30,  # 21 — match end / post-match
    1.60,  # 22
    0.70,  # 23
]

# Mixed = average of residential + enterprise (blended land use)
_MIXED_RAW = [0.5 * (r + e) for r, e in zip(_RESIDENTIAL_RAW, _ENTERPRISE_RAW)]


CELL_TYPE_PROFILES: dict[str, DiurnalProfile] = {
    "residential": DiurnalProfile(
        name="residential",
        hourly_multipliers=_normalise_to_24(_RESIDENTIAL_RAW),
    ),
    "enterprise": DiurnalProfile(
        name="enterprise",
        hourly_multipliers=_normalise_to_24(_ENTERPRISE_RAW),
    ),
    "transport_hub": DiurnalProfile(
        name="transport_hub",
        hourly_multipliers=_normalise_to_24(_TRANSPORT_HUB_RAW),
    ),
    "stadium": DiurnalProfile(
        name="stadium",
        hourly_multipliers=_normalise_to_24(_STADIUM_RAW),
    ),
    "mixed": DiurnalProfile(
        name="mixed",
        hourly_multipliers=_normalise_to_24(_MIXED_RAW),
    ),
}


# ---------------------------------------------------------------------------
# Day-of-week profiles
# ---------------------------------------------------------------------------
# index 0 = Monday, …, 6 = Sunday.
# Multipliers are relative; 1.0 = average day.
# ---------------------------------------------------------------------------


@dataclass
class DayOfWeekProfile:
    """7-float array of daily multipliers, index 0=Monday."""

    multipliers: list[float]  # length 7


DAY_OF_WEEK_PROFILES: dict[str, DayOfWeekProfile] = {
    "enterprise": DayOfWeekProfile(
        # Strong weekday dominance; quiet weekends
        multipliers=[1.15, 1.15, 1.15, 1.15, 1.15, 0.60, 0.60]
    ),
    "residential": DayOfWeekProfile(
        # Slightly lower weekdays (people out at work), higher weekends
        multipliers=[0.90, 0.90, 0.90, 0.90, 0.95, 1.10, 1.15]
    ),
    "transport_hub": DayOfWeekProfile(
        # Heavy Mon–Fri commuter traffic, lower weekend
        multipliers=[1.10, 1.10, 1.10, 1.10, 1.10, 0.80, 0.70]
    ),
    "stadium": DayOfWeekProfile(
        # Matches mostly on weekends / Friday nights
        multipliers=[0.70, 0.70, 0.75, 0.75, 1.10, 1.30, 1.20]
    ),
    "mixed": DayOfWeekProfile(
        # Average of residential + enterprise tendency
        multipliers=[1.025, 1.025, 1.025, 1.025, 1.05, 0.85, 0.875]
    ),
}


# ---------------------------------------------------------------------------
# Seasonal calendar — Ramadan + Indonesian public holidays
# ---------------------------------------------------------------------------

# Approximate Ramadan start dates (Gregorian). Ramadan lasts ~30 days.
# These are the most commonly cited official start dates for Indonesia.
_RAMADAN_APPROX: list[date] = [
    date(2024, 3, 11),
    date(2025, 2, 28),
    date(2026, 2, 17),
]
_RAMADAN_DURATION_DAYS = 30

# Indonesian public holidays 2024–2026 (key ones with network impact).
# Key: date object.  Value: short label.
# On these days: enterprise traffic drops, residential mix rises. Net multiplier = 0.7.
_INDONESIAN_PUBLIC_HOLIDAYS: dict[date, str] = {
    # 2024
    date(2024, 1, 1): "New Year",
    date(2024, 4, 10): "Eid al-Fitr 1",
    date(2024, 4, 11): "Eid al-Fitr 2",
    date(2024, 6, 17): "Eid al-Adha",
    date(2024, 12, 25): "Christmas",
    date(2024, 12, 26): "Christmas Holiday",
    # 2025
    date(2025, 1, 1): "New Year",
    date(2025, 3, 31): "Eid al-Fitr 1",
    date(2025, 4, 1): "Eid al-Fitr 2",
    date(2025, 6, 6): "Eid al-Adha",
    date(2025, 12, 25): "Christmas",
    # 2026
    date(2026, 1, 1): "New Year",
    date(2026, 3, 20): "Eid al-Fitr 1",
    date(2026, 3, 21): "Eid al-Fitr 2",
    date(2026, 5, 27): "Eid al-Adha",
    date(2026, 12, 25): "Christmas",
}


def _is_ramadan_hijri(d: date) -> bool:
    """Attempt Ramadan detection using hijri_converter package."""
    try:
        from hijri_converter import convert  # type: ignore[import]

        hijri = convert.Gregorian(d.year, d.month, d.day).to_hijri()
        return hijri.month == 9  # Ramadan is the 9th month of the Islamic calendar
    except ImportError:
        return False


def _is_ramadan_approx(d: date) -> bool:
    """Fallback: check against hardcoded approximate Ramadan start dates."""
    for start in _RAMADAN_APPROX:
        delta = (d - start).days
        if 0 <= delta < _RAMADAN_DURATION_DAYS:
            return True
    return False


class SeasonalCalendar:
    """Returns seasonal multiplier per day, handling Ramadan and public holidays."""

    def __init__(self, use_hijri: bool | None = None) -> None:
        """
        Args:
            use_hijri: Force hijri_converter on/off.  None = auto-detect.
        """
        if use_hijri is None:
            try:
                import hijri_converter  # type: ignore[import]  # noqa: F401

                self._use_hijri = True
            except ImportError:
                self._use_hijri = False
        else:
            self._use_hijri = use_hijri

    def is_ramadan(self, d: date) -> bool:
        """Return True if the given date falls within Ramadan."""
        if self._use_hijri:
            return _is_ramadan_hijri(d)
        return _is_ramadan_approx(d)

    def is_public_holiday(self, d: date) -> bool:
        """Return True if the date is a tracked Indonesian public holiday."""
        return d in _INDONESIAN_PUBLIC_HOLIDAYS

    def get_multiplier(self, dt: datetime) -> float:
        """Return seasonal multiplier for a specific datetime.

        Rules (in priority order):
          1. Public holiday → 0.7
          2. Ramadan        → 1.15
          3. Otherwise      → 1.0
        """
        d = dt.date() if isinstance(dt, datetime) else dt
        if self.is_public_holiday(d):
            return 0.7
        if self.is_ramadan(d):
            return 1.15
        return 1.0


# ---------------------------------------------------------------------------
# Full additive KPI series generator
# ---------------------------------------------------------------------------

CellType = Literal["residential", "enterprise", "transport_hub", "stadium", "mixed"]

_DEFAULT_DOW_PROFILE_MAP: dict[str, str] = {
    "residential": "residential",
    "enterprise": "enterprise",
    "transport_hub": "transport_hub",
    "stadium": "stadium",
    "mixed": "mixed",
}


def generate_kpi_series(
    base_value: float,
    start_date: datetime,
    hours: int,
    cell_type: str,
    add_ar_residual: bool = True,
    seed: int = 42,
) -> np.ndarray:
    """Full additive model: KPI(t) = base × diurnal(t) × dow(t) × seasonal(t) + ar_residual(t).

    Args:
        base_value: Peak value (e.g., 100 active users at peak).
        start_date: datetime of first hour.
        hours: Number of hourly timesteps to generate.
        cell_type: One of residential, enterprise, transport_hub, stadium, mixed.
        add_ar_residual: If True, add AR(1) noise with phi=0.7, sigma=0.05*base_value.
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray of length `hours` with non-negative values.
    """
    if cell_type not in CELL_TYPE_PROFILES:
        raise ValueError(
            f"Unknown cell_type '{cell_type}'. "
            f"Valid options: {sorted(CELL_TYPE_PROFILES.keys())}"
        )

    diurnal_profile = CELL_TYPE_PROFILES[cell_type]
    dow_profile = DAY_OF_WEEK_PROFILES.get(
        _DEFAULT_DOW_PROFILE_MAP.get(cell_type, "mixed"),
        DAY_OF_WEEK_PROFILES["mixed"],
    )
    calendar = SeasonalCalendar()

    rng = np.random.default_rng(seed)
    phi = 0.7
    sigma = 0.05 * base_value

    # AR(1) state
    ar_state: float = 0.0

    result = np.empty(hours, dtype=np.float64)

    for i in range(hours):
        dt = start_date + timedelta(hours=i)

        hour_of_day = dt.hour  # 0..23
        dow = dt.weekday()     # 0=Monday .. 6=Sunday

        diurnal_mult = diurnal_profile.hourly_multipliers[hour_of_day]
        dow_mult = dow_profile.multipliers[dow]
        seasonal_mult = calendar.get_multiplier(dt)

        value = base_value * diurnal_mult * dow_mult * seasonal_mult

        if add_ar_residual:
            ar_state = phi * ar_state + sigma * rng.standard_normal()
            value = value + ar_state

        result[i] = value

    # Clamp to non-negative
    np.clip(result, 0.0, None, out=result)
    return result
