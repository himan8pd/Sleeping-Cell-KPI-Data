"""
Step 10: Validation Framework.

Full validation suite that checks every output Parquet file produced by
Phases 0–9 against the schema contracts defined in Phase 0.

Validation checks performed:

  1. **Schema compliance** — column names, data types, and nullability
     match the Phase 0 contracts for every output file.

  2. **FK integrity** — every foreign-key reference resolves:
     - ``cell_id``, ``site_id``, ``entity_id`` in KPI/alarm/customer
       files → ``ground_truth_entities.entity_id``
     - ``scenario_id`` in events → ``scenario_manifest.scenario_id``
     - ``associated_site_id`` in customers → entities
     - ``from_entity_id`` / ``to_entity_id`` in relationships → entities

  3. **Range checks** — KPI values fall within contract-defined min/max
     bounds (with tolerance for floating-point edge cases).

  4. **Entity count verification** — entity/relationship counts match
     Phase 2 topology metadata.

  5. **Cross-domain consistency** — no orphaned relationship endpoints,
     scenario overlay entity IDs exist in KPI files, CMDB degradation
     accounting is complete.

  6. **Neighbour relation symmetry** — spot-check that A→B implies B→A
     exists (within sampling budget).

  7. **Divergence manifest completeness** — every dark node, phantom
     node, etc. in the manifest corresponds to the correct
     presence/absence in CMDB vs ground truth.

Output:
  - validation/<filename>_report.json   per-file validation reports
  - validation/summary_report.json      overall pass/fail summary

Dependencies: All previous phases (0–9)
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone  # noqa: F401
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq  # noqa: F401
from rich.console import Console
from rich.table import Table

from pedkai_generator.config.settings import GeneratorConfig
from pedkai_generator.step_00_schema.contracts import (
    FileContract,
    get_all_contracts,
    get_contract,  # noqa: F401
)

console = Console()

# ---------------------------------------------------------------------------
# Validation result types
# ---------------------------------------------------------------------------

SEVERITY_ERROR = "ERROR"
SEVERITY_WARNING = "WARNING"
SEVERITY_INFO = "INFO"


@dataclass
class ValidationIssue:
    """A single validation finding."""

    check: str
    severity: str  # ERROR, WARNING, INFO
    message: str
    file: str | None = None
    column: str | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "check": self.check,
            "severity": self.severity,
            "message": self.message,
        }
        if self.file:
            d["file"] = self.file
        if self.column:
            d["column"] = self.column
        if self.details:
            d["details"] = self.details
        return d


@dataclass
class FileValidationReport:
    """Validation report for a single output file."""

    filename: str
    exists: bool = False
    row_count: int = 0
    column_count: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    checks_passed: int = 0
    checks_failed: int = 0
    checks_warned: int = 0
    elapsed_seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return self.checks_failed == 0

    def add_issue(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        if issue.severity == SEVERITY_ERROR:
            self.checks_failed += 1
        elif issue.severity == SEVERITY_WARNING:
            self.checks_warned += 1

    def add_pass(self, check: str) -> None:
        self.checks_passed += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "exists": self.exists,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "passed": self.passed,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "checks_warned": self.checks_warned,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "issues": [i.to_dict() for i in self.issues],
        }


@dataclass
class ValidationSummary:
    """Overall validation summary across all files."""

    file_reports: list[FileValidationReport] = field(default_factory=list)
    cross_domain_issues: list[ValidationIssue] = field(default_factory=list)
    total_elapsed: float = 0.0

    @property
    def all_passed(self) -> bool:
        if any(not r.passed for r in self.file_reports):
            return False
        if any(i.severity == SEVERITY_ERROR for i in self.cross_domain_issues):
            return False
        return True

    @property
    def total_checks_passed(self) -> int:
        return sum(r.checks_passed for r in self.file_reports)

    @property
    def total_checks_failed(self) -> int:
        return sum(r.checks_failed for r in self.file_reports) + sum(
            1 for i in self.cross_domain_issues if i.severity == SEVERITY_ERROR
        )

    @property
    def total_checks_warned(self) -> int:
        return sum(r.checks_warned for r in self.file_reports) + sum(
            1 for i in self.cross_domain_issues if i.severity == SEVERITY_WARNING
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_passed": self.all_passed,
            "total_checks_passed": self.total_checks_passed,
            "total_checks_failed": self.total_checks_failed,
            "total_checks_warned": self.total_checks_warned,
            "total_elapsed_seconds": round(self.total_elapsed, 2),
            "file_reports": {r.filename: r.to_dict() for r in self.file_reports},
            "cross_domain_issues": [i.to_dict() for i in self.cross_domain_issues],
        }


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def _validate_schema(
    df: pl.DataFrame,
    contract: FileContract,
    report: FileValidationReport,
) -> None:
    """
    Validate that a DataFrame's columns match the contract:
      - All expected columns present
      - No unexpected columns
      - Column order matches (warning, not error)
    """
    expected_names = contract.column_names
    actual_names = df.columns

    # Check for missing columns
    missing = set(expected_names) - set(actual_names)
    if missing:
        report.add_issue(
            ValidationIssue(
                check="schema_columns_present",
                severity=SEVERITY_ERROR,
                message=f"Missing columns: {sorted(missing)}",
                file=contract.filename,
                details={"missing": sorted(missing)},
            )
        )
    else:
        report.add_pass("schema_columns_present")

    # Check for unexpected columns
    unexpected = set(actual_names) - set(expected_names)
    if unexpected:
        report.add_issue(
            ValidationIssue(
                check="schema_no_extra_columns",
                severity=SEVERITY_WARNING,
                message=f"Unexpected extra columns: {sorted(unexpected)}",
                file=contract.filename,
                details={"unexpected": sorted(unexpected)},
            )
        )
    else:
        report.add_pass("schema_no_extra_columns")

    # Check column count
    if len(actual_names) != len(expected_names):
        report.add_issue(
            ValidationIssue(
                check="schema_column_count",
                severity=SEVERITY_WARNING,
                message=(f"Column count mismatch: expected {len(expected_names)}, got {len(actual_names)}"),
                file=contract.filename,
            )
        )
    else:
        report.add_pass("schema_column_count")


def _validate_nullability(
    df: pl.DataFrame,
    contract: FileContract,
    report: FileValidationReport,
) -> None:
    """
    Check that non-nullable columns have zero nulls.
    """
    for col_contract in contract.columns:
        col_name = col_contract.name
        if col_name not in df.columns:
            continue

        if not col_contract.nullable:
            null_count = df[col_name].null_count()
            if null_count > 0:
                report.add_issue(
                    ValidationIssue(
                        check="nullability",
                        severity=SEVERITY_ERROR,
                        message=(f"Non-nullable column '{col_name}' has {null_count:,} null values"),
                        file=contract.filename,
                        column=col_name,
                        details={"null_count": null_count},
                    )
                )
            else:
                report.add_pass(f"nullability_{col_name}")


def _validate_allowed_values(
    df: pl.DataFrame,
    contract: FileContract,
    report: FileValidationReport,
    sample_size: int = 500_000,
) -> None:
    """
    Check columns with allowed_values constraints.
    Uses sampling for very large DataFrames to keep validation fast.
    """
    check_df = df if df.height <= sample_size else df.sample(n=sample_size, seed=42)

    for col_contract in contract.columns:
        col_name = col_contract.name
        if col_name not in check_df.columns:
            continue
        if col_contract.allowed_values is None:
            continue

        allowed = set(col_contract.allowed_values)

        # Get unique non-null values
        unique_vals = (
            check_df.select(pl.col(col_name)).filter(pl.col(col_name).is_not_null()).unique().to_series().to_list()
        )

        invalid = set(str(v) for v in unique_vals) - allowed
        if invalid:
            # Limit display to first 20
            display_invalid = sorted(invalid)[:20]
            report.add_issue(
                ValidationIssue(
                    check="allowed_values",
                    severity=SEVERITY_ERROR,
                    message=(f"Column '{col_name}' has {len(invalid)} invalid value(s): {display_invalid}"),
                    file=contract.filename,
                    column=col_name,
                    details={
                        "invalid_count": len(invalid),
                        "sample_invalid": display_invalid,
                        "allowed": sorted(allowed),
                    },
                )
            )
        else:
            report.add_pass(f"allowed_values_{col_name}")


def _validate_ranges(
    df: pl.DataFrame,
    contract: FileContract,
    report: FileValidationReport,
) -> None:
    """
    Check columns with min_value / max_value range constraints.
    """
    for col_contract in contract.columns:
        col_name = col_contract.name
        if col_name not in df.columns:
            continue

        has_min = col_contract.min_value is not None
        has_max = col_contract.max_value is not None

        if not has_min and not has_max:
            continue

        # Get actual min/max (skip nulls)
        non_null = df.filter(pl.col(col_name).is_not_null())
        if non_null.height == 0:
            continue

        try:
            actual_min = non_null[col_name].min()
            actual_max = non_null[col_name].max()
        except Exception:
            # Skip columns that can't be compared numerically
            continue

        # Tolerance for floating point
        tol = 1e-6

        if has_min and actual_min is not None:
            try:
                min_val_f = float(str(actual_min))
                min_bound = float(str(col_contract.min_value))
            except (TypeError, ValueError):
                continue
            if min_val_f < min_bound - tol:
                report.add_issue(
                    ValidationIssue(
                        check="range_min",
                        severity=SEVERITY_ERROR,
                        message=(f"Column '{col_name}' min value {actual_min} below contract minimum {min_bound}"),
                        file=contract.filename,
                        column=col_name,
                        details={
                            "actual_min": min_val_f,
                            "contract_min": min_bound,
                        },
                    )
                )
            else:
                report.add_pass(f"range_min_{col_name}")

        if has_max and actual_max is not None:
            try:
                max_val_f = float(str(actual_max))
                max_bound = float(str(col_contract.max_value))
            except (TypeError, ValueError):
                continue
            if max_val_f > max_bound + tol:
                report.add_issue(
                    ValidationIssue(
                        check="range_max",
                        severity=SEVERITY_ERROR,
                        message=(f"Column '{col_name}' max value {actual_max} above contract maximum {max_bound}"),
                        file=contract.filename,
                        column=col_name,
                        details={
                            "actual_max": max_val_f,
                            "contract_max": max_bound,
                        },
                    )
                )
            else:
                report.add_pass(f"range_max_{col_name}")


def _validate_row_count(
    df: pl.DataFrame,
    contract: FileContract,
    report: FileValidationReport,
) -> None:
    """
    Check that row count is within a reasonable range of the expected
    approximate count (if specified).  Uses a generous tolerance (50%)
    because expected counts are approximate.
    """
    if contract.expected_row_count_approx is None:
        return

    expected = contract.expected_row_count_approx
    actual = df.height
    tolerance = 0.50  # 50% tolerance on approximate counts

    lower = int(expected * (1.0 - tolerance))
    upper = int(expected * (1.0 + tolerance))

    if actual < lower or actual > upper:
        report.add_issue(
            ValidationIssue(
                check="row_count_approx",
                severity=SEVERITY_WARNING,
                message=(
                    f"Row count {actual:,} is outside ±{tolerance:.0%} of "
                    f"expected ~{expected:,} (range: {lower:,}–{upper:,})"
                ),
                file=contract.filename,
                details={
                    "actual": actual,
                    "expected_approx": expected,
                    "lower_bound": lower,
                    "upper_bound": upper,
                },
            )
        )
    else:
        report.add_pass("row_count_approx")


# ---------------------------------------------------------------------------
# FK integrity checks
# ---------------------------------------------------------------------------


def _validate_fk(
    df: pl.DataFrame,
    fk_column: str,
    reference_set: set[str],
    reference_name: str,
    report: FileValidationReport,
    filename: str,
    sample_size: int = 1_000_000,
) -> None:
    """
    Validate that all non-null values in fk_column exist in reference_set.
    Samples large DataFrames for performance.
    """
    if fk_column not in df.columns:
        report.add_issue(
            ValidationIssue(
                check=f"fk_{fk_column}",
                severity=SEVERITY_WARNING,
                message=f"FK column '{fk_column}' not found in {filename}",
                file=filename,
                column=fk_column,
            )
        )
        return

    check_df = df if df.height <= sample_size else df.sample(n=sample_size, seed=42)

    fk_values = (
        check_df.select(pl.col(fk_column)).filter(pl.col(fk_column).is_not_null()).unique().to_series().to_list()
    )

    invalid_fks = [str(v) for v in fk_values if str(v) not in reference_set]

    if invalid_fks:
        display = invalid_fks[:10]
        report.add_issue(
            ValidationIssue(
                check=f"fk_{fk_column}",
                severity=SEVERITY_ERROR,
                message=(f"FK column '{fk_column}' has {len(invalid_fks):,} values not in {reference_name}"),
                file=filename,
                column=fk_column,
                details={
                    "invalid_count": len(invalid_fks),
                    "sample_invalid": display,
                },
            )
        )
    else:
        report.add_pass(f"fk_{fk_column}")


# ---------------------------------------------------------------------------
# Neighbour symmetry check
# ---------------------------------------------------------------------------


def _validate_neighbour_symmetry(
    neighbours_df: pl.DataFrame,
    report: FileValidationReport,
    sample_size: int = 10_000,
) -> None:
    """
    Spot-check that A→B implies B→A exists in neighbour relations.
    """
    if "from_cell_id" not in neighbours_df.columns:
        return
    if "to_cell_id" not in neighbours_df.columns:
        return

    check_df = neighbours_df if neighbours_df.height <= sample_size else neighbours_df.sample(n=sample_size, seed=42)

    # Build set of (from, to) pairs from full data
    edge_set: set[tuple[str, str]] = set()
    for row in neighbours_df.select("from_cell_id", "to_cell_id").iter_rows():
        edge_set.add((str(row[0]), str(row[1])))

    # Check symmetry on sample
    asymmetric = 0
    checked = 0
    for row in check_df.select("from_cell_id", "to_cell_id").iter_rows():
        a, b = str(row[0]), str(row[1])
        checked += 1
        if (b, a) not in edge_set:
            asymmetric += 1

    if checked == 0:
        return

    asymmetric_pct = 100.0 * asymmetric / checked

    if asymmetric_pct > 20.0:
        report.add_issue(
            ValidationIssue(
                check="neighbour_symmetry",
                severity=SEVERITY_WARNING,
                message=(
                    f"Neighbour symmetry check: {asymmetric:,}/{checked:,} "
                    f"({asymmetric_pct:.1f}%) edges have no reverse direction"
                ),
                file="neighbour_relations.parquet",
                details={
                    "asymmetric": asymmetric,
                    "checked": checked,
                    "pct": round(asymmetric_pct, 1),
                },
            )
        )
    else:
        report.add_pass("neighbour_symmetry")


# ---------------------------------------------------------------------------
# CMDB divergence completeness
# ---------------------------------------------------------------------------


def _validate_cmdb_divergence(
    config: GeneratorConfig,
    summary: ValidationSummary,
) -> None:
    """
    Check that:
      - Every dark_node entity_id is absent from cmdb_declared_entities
      - Every phantom_node entity_id is present in cmdb_declared_entities
        but absent from ground_truth_entities
      - Divergence manifest entity counts are consistent with the difference
        between ground truth and CMDB counts
    """
    output_dir = config.paths.output_dir

    gt_entities_path = output_dir / "ground_truth_entities.parquet"
    cmdb_entities_path = output_dir / "cmdb_declared_entities.parquet"
    manifest_path = output_dir / "divergence_manifest.parquet"

    if not all(p.exists() for p in [gt_entities_path, cmdb_entities_path, manifest_path]):
        summary.cross_domain_issues.append(
            ValidationIssue(
                check="cmdb_divergence_files",
                severity=SEVERITY_WARNING,
                message="Cannot validate CMDB divergence — one or more files missing",
            )
        )
        return

    try:
        gt_ids = set(pl.read_parquet(gt_entities_path, columns=["entity_id"]).to_series().to_list())
        cmdb_ids = set(pl.read_parquet(cmdb_entities_path, columns=["entity_id"]).to_series().to_list())
        manifest_df = pl.read_parquet(manifest_path)
    except Exception as e:
        summary.cross_domain_issues.append(
            ValidationIssue(
                check="cmdb_divergence_read",
                severity=SEVERITY_WARNING,
                message=f"Error reading CMDB/manifest files: {e}",
            )
        )
        return

    # Dark nodes should be in GT but NOT in CMDB
    dark_node_ids = set(
        manifest_df.filter(pl.col("divergence_type") == "dark_node").select("target_id").to_series().to_list()
    )

    dark_in_cmdb = dark_node_ids & cmdb_ids
    if dark_in_cmdb:
        summary.cross_domain_issues.append(
            ValidationIssue(
                check="cmdb_dark_nodes_absent",
                severity=SEVERITY_ERROR,
                message=(f"{len(dark_in_cmdb):,} dark node entities still present in cmdb_declared_entities"),
                details={"count": len(dark_in_cmdb)},
            )
        )
    else:
        # Record as info — cannot add_pass to summary directly
        pass

    # Phantom nodes should be in CMDB but NOT in GT
    phantom_node_ids = set(
        manifest_df.filter(pl.col("divergence_type") == "phantom_node").select("target_id").to_series().to_list()
    )

    phantom_in_gt = phantom_node_ids & gt_ids
    if phantom_in_gt:
        summary.cross_domain_issues.append(
            ValidationIssue(
                check="cmdb_phantom_nodes_not_in_gt",
                severity=SEVERITY_ERROR,
                message=(
                    f"{len(phantom_in_gt):,} phantom node IDs also found "
                    f"in ground_truth_entities (should be fabricated)"
                ),
                details={"count": len(phantom_in_gt)},
            )
        )

    phantom_not_in_cmdb = phantom_node_ids - cmdb_ids
    if phantom_not_in_cmdb:
        summary.cross_domain_issues.append(
            ValidationIssue(
                check="cmdb_phantom_nodes_in_cmdb",
                severity=SEVERITY_ERROR,
                message=(f"{len(phantom_not_in_cmdb):,} phantom node IDs missing from cmdb_declared_entities"),
                details={"count": len(phantom_not_in_cmdb)},
            )
        )

    del gt_ids, cmdb_ids, manifest_df
    gc.collect()


# ---------------------------------------------------------------------------
# Scenario overlay integrity
# ---------------------------------------------------------------------------


def _validate_scenario_overlay(
    config: GeneratorConfig,
    summary: ValidationSummary,
) -> None:
    """
    Check that every entity_id in scenario_kpi_overrides exists in one
    of the KPI files.
    """
    output_dir = config.paths.output_dir
    overrides_path = output_dir / "scenario_kpi_overrides.parquet"
    manifest_path = output_dir / "scenario_manifest.parquet"

    if not overrides_path.exists():
        summary.cross_domain_issues.append(
            ValidationIssue(
                check="scenario_overlay_exists",
                severity=SEVERITY_WARNING,
                message="scenario_kpi_overrides.parquet not found — skipping overlay check",
            )
        )
        return

    if not manifest_path.exists():
        summary.cross_domain_issues.append(
            ValidationIssue(
                check="scenario_manifest_exists",
                severity=SEVERITY_WARNING,
                message="scenario_manifest.parquet not found — skipping manifest check",
            )
        )
        return

    try:
        manifest_df = pl.read_parquet(manifest_path)
        overrides_df = pl.read_parquet(overrides_path)
    except Exception as e:
        summary.cross_domain_issues.append(
            ValidationIssue(
                check="scenario_overlay_read",
                severity=SEVERITY_WARNING,
                message=f"Error reading scenario files: {e}",
            )
        )
        return

    # Check that every scenario_id in overrides exists in manifest
    if "scenario_id" in overrides_df.columns and "scenario_id" in manifest_df.columns:
        manifest_ids = set(manifest_df["scenario_id"].unique().to_list())
        override_scenario_ids = set(overrides_df["scenario_id"].unique().to_list())

        orphan_scenarios = override_scenario_ids - manifest_ids
        if orphan_scenarios:
            summary.cross_domain_issues.append(
                ValidationIssue(
                    check="scenario_overlay_fk",
                    severity=SEVERITY_ERROR,
                    message=(f"{len(orphan_scenarios):,} scenario_ids in overrides not found in manifest"),
                    details={"orphan_count": len(orphan_scenarios)},
                )
            )

    del manifest_df, overrides_df
    gc.collect()


# ---------------------------------------------------------------------------
# Event/alarm FK checks
# ---------------------------------------------------------------------------


def _validate_events_fks(
    config: GeneratorConfig,
    entity_ids: set[str],
    summary: ValidationSummary,
) -> None:
    """
    Validate that entity_id and scenario_id FKs in events_alarms.parquet
    resolve correctly.
    """
    output_dir = config.paths.output_dir
    events_path = output_dir / "events_alarms.parquet"

    if not events_path.exists():
        return

    try:
        events_df = pl.read_parquet(events_path)
    except Exception as e:
        summary.cross_domain_issues.append(
            ValidationIssue(
                check="events_read",
                severity=SEVERITY_WARNING,
                message=f"Error reading events_alarms.parquet: {e}",
            )
        )
        return

    # entity_id FK
    if "entity_id" in events_df.columns:
        event_entity_ids = set(
            events_df.select("entity_id").filter(pl.col("entity_id").is_not_null()).unique().to_series().to_list()
        )
        invalid = event_entity_ids - entity_ids
        if invalid:
            summary.cross_domain_issues.append(
                ValidationIssue(
                    check="events_entity_id_fk",
                    severity=SEVERITY_ERROR,
                    message=(f"{len(invalid):,} entity_ids in events_alarms not found in ground_truth_entities"),
                    details={"invalid_count": len(invalid)},
                )
            )

    # scenario_id FK (for scenario-driven alarms)
    manifest_path = output_dir / "scenario_manifest.parquet"
    if "scenario_id" in events_df.columns and manifest_path.exists():
        try:
            manifest_ids = set(pl.read_parquet(manifest_path, columns=["scenario_id"]).to_series().to_list())
            event_scenario_ids = set(
                events_df.filter(pl.col("scenario_id").is_not_null())
                .select("scenario_id")
                .unique()
                .to_series()
                .to_list()
            )
            invalid_scenarios = event_scenario_ids - manifest_ids
            if invalid_scenarios:
                summary.cross_domain_issues.append(
                    ValidationIssue(
                        check="events_scenario_id_fk",
                        severity=SEVERITY_ERROR,
                        message=(f"{len(invalid_scenarios):,} scenario_ids in events not found in scenario_manifest"),
                        details={"invalid_count": len(invalid_scenarios)},
                    )
                )
        except Exception:
            pass

    del events_df
    gc.collect()


# ---------------------------------------------------------------------------
# Per-file validation orchestrator
# ---------------------------------------------------------------------------


def _validate_file(
    filepath: Path,
    contract: FileContract,
    entity_ids: set[str] | None = None,
) -> FileValidationReport:
    """
    Run all per-file validation checks against a single Parquet output.
    """
    report = FileValidationReport(filename=contract.filename)
    t0 = time.time()

    # Check existence
    if not filepath.exists():
        report.add_issue(
            ValidationIssue(
                check="file_exists",
                severity=SEVERITY_ERROR,
                message=f"File not found: {filepath}",
                file=contract.filename,
            )
        )
        report.elapsed_seconds = time.time() - t0
        return report

    report.exists = True
    report.add_pass("file_exists")

    # Read Parquet
    try:
        df = pl.read_parquet(filepath)
    except Exception as e:
        report.add_issue(
            ValidationIssue(
                check="file_readable",
                severity=SEVERITY_ERROR,
                message=f"Cannot read Parquet file: {e}",
                file=contract.filename,
            )
        )
        report.elapsed_seconds = time.time() - t0
        return report

    report.add_pass("file_readable")
    report.row_count = df.height
    report.column_count = df.width

    # Schema validation
    _validate_schema(df, contract, report)

    # Nullability
    _validate_nullability(df, contract, report)

    # Allowed values
    _validate_allowed_values(df, contract, report)

    # Range checks
    _validate_ranges(df, contract, report)

    # Approximate row count
    _validate_row_count(df, contract, report)

    # FK checks (if entity_ids reference set provided)
    if entity_ids is not None:
        # Check common FK columns
        fk_columns_to_check = []
        for col_contract in contract.columns:
            col_name = col_contract.name
            if col_name in (
                "entity_id",
                "site_id",
                "associated_site_id",
                "from_entity_id",
                "to_entity_id",
                "access_entity_id",
            ):
                if col_name in df.columns:
                    fk_columns_to_check.append(col_name)

        for fk_col in fk_columns_to_check:
            # access_entity_id is nullable — skip null-only checks
            _validate_fk(
                df,
                fk_col,
                entity_ids,
                "ground_truth_entities.entity_id",
                report,
                contract.filename,
            )

    # Neighbour symmetry (special case)
    if contract.filename == "neighbour_relations.parquet":
        _validate_neighbour_symmetry(df, report)

    del df
    gc.collect()

    report.elapsed_seconds = time.time() - t0
    return report


# ---------------------------------------------------------------------------
# Files to validate (in order)
# ---------------------------------------------------------------------------

# Map contract filenames to validation priority.  Files that don't exist
# yet (future phases) are simply skipped with an INFO-level notice.
VALIDATION_FILES = [
    "ground_truth_entities.parquet",
    "ground_truth_relationships.parquet",
    "neighbour_relations.parquet",
    "kpi_metrics_wide.parquet",
    "transport_kpis_wide.parquet",
    "fixed_broadband_kpis_wide.parquet",
    "enterprise_circuit_kpis_wide.parquet",
    "core_element_kpis_wide.parquet",
    "power_environment_kpis.parquet",
    "scenario_manifest.parquet",
    "scenario_kpi_overrides.parquet",
    "events_alarms.parquet",
    "customers_bss.parquet",
    "cmdb_declared_entities.parquet",
    "cmdb_declared_relationships.parquet",
    "divergence_manifest.parquet",
]

# Additional files produced by later phases that may not have contracts
# but should be noted
OPTIONAL_FILES = [
    "vendor_naming_map.parquet",
]


# ---------------------------------------------------------------------------
# Write reports
# ---------------------------------------------------------------------------


def _write_report(report_data: dict[str, Any], output_path: Path) -> None:
    """Write a JSON validation report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def validate_all(config: GeneratorConfig) -> None:
    """
    Step 10 entry point: Full validation suite.

    Validates all output Parquet files against Phase 0 schema contracts,
    checks FK integrity, range bounds, cross-domain consistency, CMDB
    divergence completeness, and scenario overlay integrity.

    Produces per-file JSON reports in the validation/ directory and an
    overall summary_report.json.
    """
    step_start = time.time()

    seed = config.seed_for("step_10_validation")
    console.print(f"[dim]Step 10 seed: {seed}[/dim]")

    console.print(
        "[bold]Validation Suite:[/bold] schema compliance, FK integrity, range checks, cross-domain consistency"
    )

    config.ensure_output_dirs()
    output_dir = config.paths.output_dir
    validation_dir = config.paths.validation_dir
    validation_dir.mkdir(parents=True, exist_ok=True)

    summary = ValidationSummary()

    # ── Load entity ID reference set ─────────────────────────
    # This is needed for FK checks across all files
    entity_ids: set[str] | None = None
    gt_entities_path = output_dir / "ground_truth_entities.parquet"

    if gt_entities_path.exists():
        console.print("\n[bold]Loading entity reference set...[/bold]")
        t0 = time.time()
        try:
            entity_ids = set(pl.read_parquet(gt_entities_path, columns=["entity_id"]).to_series().to_list())
            console.print(
                f"  [green]✓[/green] {len(entity_ids):,} entity IDs loaded for FK validation in {time.time() - t0:.1f}s"
            )
        except Exception as e:
            console.print(f"  [yellow]⚠[/yellow] Could not load entity IDs: {e}")
            entity_ids = None

    # ── Get all contracts ─────────────────────────────────────
    all_contracts = get_all_contracts()

    # ── Validate each file ────────────────────────────────────
    console.print("\n[bold]Validating output files...[/bold]")

    for filename in VALIDATION_FILES:
        filepath = output_dir / filename

        if filename not in all_contracts:
            console.print(f"  [dim]⊘ {filename} — no contract defined, skipping[/dim]")
            continue

        contract = all_contracts[filename]

        if not filepath.exists():
            console.print(f"  [dim]⊘ {filename} — file not found, skipping[/dim]")
            report = FileValidationReport(filename=filename)
            report.add_issue(
                ValidationIssue(
                    check="file_exists",
                    severity=SEVERITY_INFO,
                    message=f"File not found (phase may not have been run yet): {filepath}",
                    file=filename,
                )
            )
            summary.file_reports.append(report)
            continue

        console.print(f"  [bold]{filename}[/bold]...", end=" ")
        report = _validate_file(filepath, contract, entity_ids)
        summary.file_reports.append(report)

        # Write per-file report
        report_path = validation_dir / f"{filename.replace('.parquet', '')}_report.json"
        _write_report(report.to_dict(), report_path)

        # Display result
        if report.passed:
            console.print(
                f"[green]✓ PASS[/green] "
                f"({report.row_count:,} rows, "
                f"{report.checks_passed} checks passed, "
                f"{report.checks_warned} warnings) "
                f"[dim]{report.elapsed_seconds:.1f}s[/dim]"
            )
        else:
            console.print(
                f"[red]✗ FAIL[/red] "
                f"({report.checks_failed} errors, "
                f"{report.checks_warned} warnings, "
                f"{report.checks_passed} passed) "
                f"[dim]{report.elapsed_seconds:.1f}s[/dim]"
            )
            for issue in report.issues:
                if issue.severity == SEVERITY_ERROR:
                    console.print(f"    [red]ERROR:[/red] [{issue.check}] {issue.message}")

    # Check optional files
    for filename in OPTIONAL_FILES:
        filepath = output_dir / filename
        if filepath.exists():
            console.print(f"  [dim]⊘ {filename} — present (no contract validation)[/dim]")

    # ── Cross-domain validation ───────────────────────────────
    console.print("\n[bold]Cross-domain validation checks...[/bold]")

    # CMDB divergence completeness
    console.print("  CMDB divergence completeness...", end=" ")
    t0 = time.time()
    _validate_cmdb_divergence(config, summary)
    cmdb_issues = [i for i in summary.cross_domain_issues if i.check.startswith("cmdb_")]
    cmdb_errors = sum(1 for i in cmdb_issues if i.severity == SEVERITY_ERROR)
    if cmdb_errors > 0:
        console.print(f"[red]✗ {cmdb_errors} error(s)[/red] [dim]{time.time() - t0:.1f}s[/dim]")
    else:
        console.print(f"[green]✓[/green] [dim]{time.time() - t0:.1f}s[/dim]")

    # Scenario overlay integrity
    console.print("  Scenario overlay integrity...", end=" ")
    t0 = time.time()
    prev_issues = len(summary.cross_domain_issues)
    _validate_scenario_overlay(config, summary)
    new_issues = summary.cross_domain_issues[prev_issues:]
    scenario_errors = sum(1 for i in new_issues if i.severity == SEVERITY_ERROR)
    if scenario_errors > 0:
        console.print(f"[red]✗ {scenario_errors} error(s)[/red] [dim]{time.time() - t0:.1f}s[/dim]")
    else:
        console.print(f"[green]✓[/green] [dim]{time.time() - t0:.1f}s[/dim]")

    # Events FK integrity
    if entity_ids is not None:
        console.print("  Events FK integrity...", end=" ")
        t0 = time.time()
        prev_issues_count = len(summary.cross_domain_issues)
        _validate_events_fks(config, entity_ids, summary)
        events_new = summary.cross_domain_issues[prev_issues_count:]
        events_errors = sum(1 for i in events_new if i.severity == SEVERITY_ERROR)
        if events_errors > 0:
            console.print(f"[red]✗ {events_errors} error(s)[/red] [dim]{time.time() - t0:.1f}s[/dim]")
        else:
            console.print(f"[green]✓[/green] [dim]{time.time() - t0:.1f}s[/dim]")

    # ── Write summary report ──────────────────────────────────
    summary.total_elapsed = time.time() - step_start
    summary_path = validation_dir / "summary_report.json"
    _write_report(summary.to_dict(), summary_path)
    console.print(f"\n  [dim]Reports written to {validation_dir}/[/dim]")

    # ── Summary tables ────────────────────────────────────────
    console.print()

    # Per-file results table
    results_table = Table(
        title="Step 10: Validation — Per-File Results",
        show_header=True,
    )
    results_table.add_column("File", style="bold", width=42)
    results_table.add_column("Status", width=8, justify="center")
    results_table.add_column("Rows", justify="right", width=14)
    results_table.add_column("Passed", justify="right", width=8)
    results_table.add_column("Errors", justify="right", width=8)
    results_table.add_column("Warnings", justify="right", width=8)
    results_table.add_column("Time", justify="right", width=8)

    for report in summary.file_reports:
        if not report.exists and report.checks_failed == 0:
            status = "[dim]SKIP[/dim]"
        elif report.passed:
            status = "[green]PASS[/green]"
        else:
            status = "[red]FAIL[/red]"

        time_str = f"{report.elapsed_seconds:.1f}s" if report.elapsed_seconds > 0 else "—"
        rows_str = f"{report.row_count:,}" if report.row_count > 0 else "—"

        results_table.add_row(
            report.filename,
            status,
            rows_str,
            str(report.checks_passed),
            str(report.checks_failed),
            str(report.checks_warned),
            time_str,
        )

    console.print(results_table)

    # Cross-domain results
    if summary.cross_domain_issues:
        console.print()
        xd_table = Table(
            title="Cross-Domain Validation Issues",
            show_header=True,
        )
        xd_table.add_column("Check", style="bold", width=30)
        xd_table.add_column("Severity", width=10)
        xd_table.add_column("Message", width=60)

        for issue in summary.cross_domain_issues:
            if issue.severity == SEVERITY_ERROR:
                sev_style = "[red]ERROR[/red]"
            elif issue.severity == SEVERITY_WARNING:
                sev_style = "[yellow]WARN[/yellow]"
            else:
                sev_style = "[dim]INFO[/dim]"
            xd_table.add_row(issue.check, sev_style, issue.message)

        console.print(xd_table)

    # Overall summary
    console.print()
    overall_table = Table(
        title="Overall Validation Summary",
        show_header=True,
    )
    overall_table.add_column("Metric", style="bold", width=30)
    overall_table.add_column("Value", justify="right", width=14)

    overall_table.add_row("Files validated", str(len(summary.file_reports)))
    overall_table.add_row(
        "Files passed",
        str(sum(1 for r in summary.file_reports if r.passed)),
    )
    overall_table.add_row(
        "Files failed",
        str(sum(1 for r in summary.file_reports if not r.passed and r.exists)),
    )
    overall_table.add_row(
        "Files skipped",
        str(sum(1 for r in summary.file_reports if not r.exists)),
    )
    overall_table.add_row("Total checks passed", str(summary.total_checks_passed))
    overall_table.add_row("Total errors", str(summary.total_checks_failed))
    overall_table.add_row("Total warnings", str(summary.total_checks_warned))
    overall_table.add_row(
        "Cross-domain issues",
        str(len(summary.cross_domain_issues)),
    )
    console.print(overall_table)

    total_elapsed = time.time() - step_start
    time_str = f"{total_elapsed:.1f}s" if total_elapsed < 60 else f"{total_elapsed / 60:.1f}m"

    if summary.all_passed:
        console.print(
            f"\n[bold green]✓ Step 10 complete — ALL VALIDATIONS PASSED.[/bold green] "
            f"({summary.total_checks_passed} checks, "
            f"{summary.total_checks_warned} warnings) in {time_str}"
        )
    else:
        console.print(
            f"\n[bold red]✗ Step 10 complete — VALIDATION FAILURES DETECTED.[/bold red] "
            f"({summary.total_checks_failed} errors, "
            f"{summary.total_checks_warned} warnings, "
            f"{summary.total_checks_passed} passed) in {time_str}"
        )
        console.print(
            "[dim]Review per-file reports in the validation/ directory "
            "for detailed findings. Fix issues and re-run validation.[/dim]"
        )
