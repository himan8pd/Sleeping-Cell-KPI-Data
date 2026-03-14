"""Tests for pedkai_generator.temporal_model."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from pedkai_generator.temporal_model import (
    CELL_TYPE_PROFILES,
    DAY_OF_WEEK_PROFILES,
    SeasonalCalendar,
    generate_kpi_series,
)


# ---------------------------------------------------------------------------
# DiurnalProfile tests
# ---------------------------------------------------------------------------


def test_diurnal_profile_sums_to_24() -> None:
    """Each CELL_TYPE_PROFILES DiurnalProfile must sum to 24.0 (±0.01)."""
    for name, profile in CELL_TYPE_PROFILES.items():
        total = sum(profile.hourly_multipliers)
        assert abs(total - 24.0) < 0.01, (
            f"Profile '{name}' sums to {total:.4f}, expected 24.0 ± 0.01"
        )


def test_diurnal_profile_length() -> None:
    """Each profile must have exactly 24 hourly multipliers."""
    for name, profile in CELL_TYPE_PROFILES.items():
        assert len(profile.hourly_multipliers) == 24, (
            f"Profile '{name}' has {len(profile.hourly_multipliers)} entries, expected 24"
        )


def test_all_multipliers_nonnegative() -> None:
    """All hourly multipliers must be >= 0."""
    for name, profile in CELL_TYPE_PROFILES.items():
        for h, v in enumerate(profile.hourly_multipliers):
            assert v >= 0.0, f"Profile '{name}' hour {h} has negative multiplier {v}"


# ---------------------------------------------------------------------------
# Residential profile shape
# ---------------------------------------------------------------------------


def test_residential_night_lower_than_morning() -> None:
    """Traffic between 2AM-5AM < traffic 9AM-11AM for residential, by at least 40%."""
    profile = CELL_TYPE_PROFILES["residential"]
    mults = profile.hourly_multipliers

    # Night hours: 2, 3, 4, 5 (indices 2..5)
    night_avg = sum(mults[h] for h in range(2, 6)) / 4.0

    # Morning hours: 9, 10, 11 (indices 9..11)
    morning_avg = sum(mults[h] for h in range(9, 12)) / 3.0

    assert morning_avg > 0, "Morning average must be positive"
    assert night_avg < morning_avg * 0.60, (
        f"Residential night avg ({night_avg:.3f}) is not at least 40% lower "
        f"than morning avg ({morning_avg:.3f}). Ratio: {night_avg / morning_avg:.3f}"
    )


# ---------------------------------------------------------------------------
# Enterprise profile shape
# ---------------------------------------------------------------------------


def test_enterprise_office_hours_dominant() -> None:
    """Enterprise 9AM-5PM multipliers should be higher than evening (8PM) multipliers."""
    profile = CELL_TYPE_PROFILES["enterprise"]
    mults = profile.hourly_multipliers

    office_avg = sum(mults[h] for h in range(9, 17)) / 8.0  # 9AM-4PM
    evening_val = mults[20]  # 8PM

    assert office_avg > evening_val, (
        f"Enterprise office hours avg ({office_avg:.3f}) should exceed "
        f"evening 8PM ({evening_val:.3f})"
    )


# ---------------------------------------------------------------------------
# Day-of-week tests
# ---------------------------------------------------------------------------


def test_enterprise_weekday_vs_weekend() -> None:
    """Weekday enterprise traffic > weekend enterprise traffic."""
    start_weekday = datetime(2025, 3, 10, 9, 0)   # Monday 9AM
    start_weekend = datetime(2025, 3, 15, 9, 0)   # Saturday 9AM

    # 1 hour each, same time-of-day → diurnal and seasonal are equal
    # Only dow multiplier differs
    wd_series = generate_kpi_series(
        base_value=100.0,
        start_date=start_weekday,
        hours=1,
        cell_type="enterprise",
        add_ar_residual=False,
        seed=0,
    )
    we_series = generate_kpi_series(
        base_value=100.0,
        start_date=start_weekend,
        hours=1,
        cell_type="enterprise",
        add_ar_residual=False,
        seed=0,
    )

    assert wd_series[0] > we_series[0], (
        f"Enterprise weekday ({wd_series[0]:.2f}) should exceed weekend ({we_series[0]:.2f})"
    )


def test_residential_weekend_vs_weekday() -> None:
    """Residential weekend traffic > weekday traffic (people home on weekends)."""
    start_weekday = datetime(2025, 3, 10, 19, 0)   # Monday 7PM
    start_weekend = datetime(2025, 3, 15, 19, 0)   # Saturday 7PM

    wd_series = generate_kpi_series(
        base_value=100.0,
        start_date=start_weekday,
        hours=1,
        cell_type="residential",
        add_ar_residual=False,
        seed=0,
    )
    we_series = generate_kpi_series(
        base_value=100.0,
        start_date=start_weekend,
        hours=1,
        cell_type="residential",
        add_ar_residual=False,
        seed=0,
    )

    assert we_series[0] > wd_series[0], (
        f"Residential weekend ({we_series[0]:.2f}) should exceed weekday ({wd_series[0]:.2f})"
    )


# ---------------------------------------------------------------------------
# generate_kpi_series correctness
# ---------------------------------------------------------------------------


def test_generated_series_nonnegative() -> None:
    """All values in generated series >= 0."""
    for cell_type in CELL_TYPE_PROFILES:
        series = generate_kpi_series(
            base_value=50.0,
            start_date=datetime(2025, 1, 1, 0, 0),
            hours=168,  # 1 week
            cell_type=cell_type,
            add_ar_residual=True,
            seed=42,
        )
        assert np.all(series >= 0.0), (
            f"Cell type '{cell_type}' produced negative values: min={series.min():.4f}"
        )


def test_series_length() -> None:
    """generate_kpi_series returns array of correct length."""
    for hours in [1, 24, 168, 720]:
        series = generate_kpi_series(
            base_value=100.0,
            start_date=datetime(2025, 3, 1, 0, 0),
            hours=hours,
            cell_type="mixed",
            seed=0,
        )
        assert len(series) == hours, (
            f"Expected length {hours}, got {len(series)}"
        )


def test_series_dtype_float() -> None:
    """Output array should be floating point."""
    series = generate_kpi_series(
        base_value=100.0,
        start_date=datetime(2025, 3, 1, 0, 0),
        hours=24,
        cell_type="residential",
        seed=0,
    )
    assert np.issubdtype(series.dtype, np.floating), (
        f"Expected floating dtype, got {series.dtype}"
    )


def test_invalid_cell_type_raises() -> None:
    """Passing an unknown cell_type should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown cell_type"):
        generate_kpi_series(
            base_value=100.0,
            start_date=datetime(2025, 3, 1, 0, 0),
            hours=24,
            cell_type="nonexistent_type",
            seed=0,
        )


# ---------------------------------------------------------------------------
# Seasonal calendar tests
# ---------------------------------------------------------------------------


def test_ramadan_multiplier() -> None:
    """A date known to be in Ramadan 2025 (e.g., 2025-03-10) has multiplier > 1.0."""
    calendar = SeasonalCalendar(use_hijri=False)  # Use approx fallback for determinism
    # Ramadan 2025 approx starts 2025-02-28, so 2025-03-10 is day 10 of Ramadan
    dt = datetime(2025, 3, 10, 12, 0)
    mult = calendar.get_multiplier(dt)
    assert mult > 1.0, (
        f"Expected multiplier > 1.0 for Ramadan date 2025-03-10, got {mult}"
    )


def test_ramadan_approx_detection_2024() -> None:
    """2024-03-15 should be detected as Ramadan (approx)."""
    calendar = SeasonalCalendar(use_hijri=False)
    dt = datetime(2024, 3, 15, 0, 0)
    assert calendar.is_ramadan(dt.date()), "2024-03-15 should be within Ramadan 2024"


def test_ramadan_approx_detection_2026() -> None:
    """2026-02-20 should be detected as Ramadan (approx)."""
    calendar = SeasonalCalendar(use_hijri=False)
    dt = datetime(2026, 2, 20, 0, 0)
    assert calendar.is_ramadan(dt.date()), "2026-02-20 should be within Ramadan 2026"


def test_non_ramadan_multiplier() -> None:
    """A non-Ramadan, non-holiday date returns multiplier 1.0."""
    calendar = SeasonalCalendar(use_hijri=False)
    dt = datetime(2025, 7, 4, 12, 0)  # July 4 — not Ramadan, not Indonesian holiday
    mult = calendar.get_multiplier(dt)
    assert mult == 1.0, f"Expected 1.0 for ordinary date, got {mult}"


def test_public_holiday_multiplier() -> None:
    """An Indonesian public holiday returns multiplier 0.7."""
    calendar = SeasonalCalendar(use_hijri=False)
    dt = datetime(2025, 12, 25, 12, 0)  # Christmas 2025
    mult = calendar.get_multiplier(dt)
    assert mult == 0.7, f"Expected 0.7 for Christmas 2025, got {mult}"


def test_eid_al_fitr_2025_is_holiday() -> None:
    """2025-03-31 is Eid al-Fitr and should return 0.7."""
    calendar = SeasonalCalendar(use_hijri=False)
    dt = datetime(2025, 3, 31, 12, 0)
    mult = calendar.get_multiplier(dt)
    # Note: 2025-03-31 is also during Ramadan end period, but public holiday takes priority
    assert mult == 0.7, f"Expected 0.7 for Eid al-Fitr 2025-03-31, got {mult}"


def test_holiday_takes_priority_over_ramadan() -> None:
    """Public holidays should take priority over Ramadan multiplier."""
    calendar = SeasonalCalendar(use_hijri=False)
    # 2025-03-31 is both near end of Ramadan AND Eid al-Fitr (public holiday)
    dt = datetime(2025, 3, 31, 12, 0)
    mult = calendar.get_multiplier(dt)
    assert mult == 0.7, (
        f"Public holiday should override Ramadan: expected 0.7, got {mult}"
    )


# ---------------------------------------------------------------------------
# Day-of-week profile structure
# ---------------------------------------------------------------------------


def test_dow_profile_lengths() -> None:
    """All DayOfWeekProfiles must have exactly 7 multipliers."""
    for name, profile in DAY_OF_WEEK_PROFILES.items():
        assert len(profile.multipliers) == 7, (
            f"DOW profile '{name}' has {len(profile.multipliers)} entries, expected 7"
        )


def test_enterprise_dow_weekday_exceeds_weekend() -> None:
    """Enterprise DOW: average weekday (Mon-Fri) > average weekend (Sat-Sun)."""
    profile = DAY_OF_WEEK_PROFILES["enterprise"]
    weekday_avg = sum(profile.multipliers[:5]) / 5.0
    weekend_avg = sum(profile.multipliers[5:]) / 2.0
    assert weekday_avg > weekend_avg, (
        f"Enterprise weekday DOW avg ({weekday_avg:.3f}) should exceed "
        f"weekend ({weekend_avg:.3f})"
    )


def test_residential_dow_weekend_exceeds_weekday() -> None:
    """Residential DOW: average weekend > average weekday."""
    profile = DAY_OF_WEEK_PROFILES["residential"]
    weekday_avg = sum(profile.multipliers[:5]) / 5.0
    weekend_avg = sum(profile.multipliers[5:]) / 2.0
    assert weekend_avg > weekday_avg, (
        f"Residential weekend DOW avg ({weekend_avg:.3f}) should exceed "
        f"weekday ({weekday_avg:.3f})"
    )


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_seed_reproducibility() -> None:
    """Same seed must produce identical series."""
    kwargs = dict(
        base_value=100.0,
        start_date=datetime(2025, 6, 1, 0, 0),
        hours=48,
        cell_type="mixed",
        add_ar_residual=True,
        seed=99,
    )
    s1 = generate_kpi_series(**kwargs)
    s2 = generate_kpi_series(**kwargs)
    np.testing.assert_array_equal(s1, s2)
