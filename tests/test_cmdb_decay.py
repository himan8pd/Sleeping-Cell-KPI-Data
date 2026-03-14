"""Tests for pedkai_generator.cmdb_decay_model."""

from __future__ import annotations

import pandas as pd
import pytest

from pedkai_generator.cmdb_decay_model import (
    ACCELERATED_DECAY_CONFIG,
    REALISTIC_DECAY_CONFIG,
    CMDBDecayConfig,
    apply_decay,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_cmdb_df() -> pd.DataFrame:
    """Create a minimal 100-row test DataFrame with entity_id and entity_type."""
    rows = []
    for i in range(100):
        rows.append({
            "entity_id": f"ci_{i:03d}",
            "entity_type": "device" if i < 80 else "link",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


def test_realistic_config_phantom_rate(sample_cmdb_df: pd.DataFrame) -> None:
    """Apply decay with 100-row df, 12 months realistic config -> CIs removed."""
    degraded, ground_truth = apply_decay(
        cmdb_snapshot=sample_cmdb_df,
        config=REALISTIC_DECAY_CONFIG,
        simulation_months=12,
        seed=42,
    )

    # Count rows removed (phantom CIs + phantom edges)
    # For 80 nodes at 5% quarterly over 4 quarters: ~18.55% -> ~14-15 phantom CIs
    # For 20 edges at 11% annually: ~11% -> ~2 phantom edges
    # Expected total: ~16-17 rows removed
    initial_count = len(sample_cmdb_df)
    final_count = len(degraded)
    removed = initial_count - final_count

    assert removed > 0, (
        f"Expected at least 1 row removed for realistic config, got {removed}"
    )
    assert removed < initial_count, (
        f"Expected some rows to remain, got {removed} removed from {initial_count}"
    )


def test_divergence_manifest_not_empty(sample_cmdb_df: pd.DataFrame) -> None:
    """Divergence ground truth has > 0 rows."""
    degraded, divergence_gt = apply_decay(
        cmdb_snapshot=sample_cmdb_df,
        config=REALISTIC_DECAY_CONFIG,
        simulation_months=12,
        seed=42,
    )

    assert len(divergence_gt) > 0, (
        "Expected divergence ground truth to contain at least one row"
    )


def test_divergence_manifest_schema(sample_cmdb_df: pd.DataFrame) -> None:
    """Columns include entity_id, divergence_type, injected_at_month, expected_detection_score."""
    degraded, divergence_gt = apply_decay(
        cmdb_snapshot=sample_cmdb_df,
        config=REALISTIC_DECAY_CONFIG,
        simulation_months=12,
        seed=42,
    )

    expected_cols = {
        "entity_id",
        "divergence_type",
        "injected_at_month",
        "expected_detection_score",
    }
    actual_cols = set(divergence_gt.columns)

    assert expected_cols == actual_cols, (
        f"Expected columns {expected_cols}, got {actual_cols}"
    )

    # Check data types
    assert divergence_gt["entity_id"].dtype == "object" or divergence_gt["entity_id"].dtype == "string"
    assert divergence_gt["divergence_type"].dtype == "object" or divergence_gt["divergence_type"].dtype == "string"
    assert divergence_gt["injected_at_month"].dtype == "int64" or divergence_gt["injected_at_month"].dtype == "int32"
    assert divergence_gt["expected_detection_score"].dtype == "float64" or divergence_gt["expected_detection_score"].dtype == "float32"


def test_accelerated_roughly_double(sample_cmdb_df: pd.DataFrame) -> None:
    """Accelerated phantom rate > realistic phantom rate."""
    degraded_realistic, gt_realistic = apply_decay(
        cmdb_snapshot=sample_cmdb_df,
        config=REALISTIC_DECAY_CONFIG,
        simulation_months=12,
        seed=42,
    )

    degraded_accelerated, gt_accelerated = apply_decay(
        cmdb_snapshot=sample_cmdb_df,
        config=ACCELERATED_DECAY_CONFIG,
        simulation_months=12,
        seed=42,
    )

    phantom_realistic = len(
        gt_realistic[gt_realistic["divergence_type"] == "phantom_ci"]
    )
    phantom_accelerated = len(
        gt_accelerated[gt_accelerated["divergence_type"] == "phantom_ci"]
    )

    assert phantom_accelerated > phantom_realistic, (
        f"Expected accelerated ({phantom_accelerated}) > realistic ({phantom_realistic}) "
        "phantom_ci count"
    )


def test_deterministic_with_seed(sample_cmdb_df: pd.DataFrame) -> None:
    """Same seed, same output."""
    degraded_1, gt_1 = apply_decay(
        cmdb_snapshot=sample_cmdb_df,
        config=REALISTIC_DECAY_CONFIG,
        simulation_months=12,
        seed=123,
    )

    degraded_2, gt_2 = apply_decay(
        cmdb_snapshot=sample_cmdb_df,
        config=REALISTIC_DECAY_CONFIG,
        simulation_months=12,
        seed=123,
    )

    # Check that degraded DataFrames are identical
    pd.testing.assert_frame_equal(degraded_1, degraded_2)

    # Check that ground truth DataFrames are identical
    pd.testing.assert_frame_equal(gt_1, gt_2)
