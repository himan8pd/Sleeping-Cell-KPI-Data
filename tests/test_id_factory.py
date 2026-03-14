"""
Tests for the deterministic ID factory.

Validates:
- Uniqueness of generated IDs (no UUID collisions in 10K batch)
- ID format compliance (no UUID v4 format)
- Province code dictionary completeness (38 provinces)
- Determinism (same inputs produce same outputs)
"""

import pytest
from src.pedkai_generator.id_factory import (
    generate_cell_id,
    generate_site_id,
    generate_work_order_id,
    generate_alarm_id,
    generate_cmdb_ci_id,
    PROVINCE_CODES,
)


class TestProvinceCodeDictionary:
    """Validate province code dictionary."""

    def test_all_38_provinces_present(self):
        """Province codes dict should contain all 38 Indonesian provinces + special regions."""
        assert len(PROVINCE_CODES) == 38, (
            f"Expected 38 provinces, got {len(PROVINCE_CODES)}"
        )

    def test_all_codes_are_two_letters(self):
        """All province codes should be exactly 2-3 characters (letters and optional digits)."""
        for province, code in PROVINCE_CODES.items():
            assert 2 <= len(code) <= 3, (
                f"Province {province} has invalid code length: {code}"
            )
            assert code.replace('0', '').replace('1', '').replace('2', '').isalpha(), (
                f"Province {province} code contains invalid chars: {code}"
            )

    def test_no_duplicate_codes(self):
        """Province codes should be unique (no duplicates)."""
        codes = list(PROVINCE_CODES.values())
        assert len(codes) == len(set(codes)), (
            f"Duplicate province codes detected"
        )


class TestCellIdGeneration:
    """Test deterministic cell ID generation."""

    def test_cell_id_format(self):
        """Cell ID should match pattern: VENDOR-SITE_ID-CELL_TYPE-NUM."""
        cell_id = generate_cell_id("ericsson", "SITE-JK-0001", "LTE", 1)
        assert cell_id == "ERI-SITE-JK-0001-LTE-1"

    def test_cell_id_case_insensitivity_vendor(self):
        """Vendor name should be uppercase regardless of input case."""
        id1 = generate_cell_id("ericsson", "site-123", "lte", 1)
        id2 = generate_cell_id("ERICSSON", "site-123", "LTE", 1)
        assert id1 == id2

    def test_cell_id_determinism(self):
        """Same inputs should always produce same cell ID."""
        id1 = generate_cell_id("nokia", "SITE-BA-2847", "NR_SA", 5)
        id2 = generate_cell_id("nokia", "SITE-BA-2847", "NR_SA", 5)
        assert id1 == id2

    def test_10k_cell_ids_unique(self):
        """10,000 cell IDs with different sequences should all be unique."""
        cell_ids = set()
        for i in range(10000):
            cell_id = generate_cell_id("ericsson", "SITE-JK-0001", "LTE", i)
            assert cell_id not in cell_ids, f"Duplicate cell ID at sequence {i}: {cell_id}"
            cell_ids.add(cell_id)

        assert len(cell_ids) == 10000


class TestSiteIdGeneration:
    """Test deterministic site ID generation."""

    def test_site_id_format(self):
        """Site ID should match pattern: SITE-PROVINCE_CODE-SEQUENCE."""
        site_id = generate_site_id("JK", 1847)
        assert site_id == "SITE-JK-1847"

    def test_site_id_sequence_padding(self):
        """Site ID sequence should be zero-padded to 4 digits."""
        site_id = generate_site_id("BA", 5)
        assert site_id == "SITE-BA-0005"

    def test_site_id_determinism(self):
        """Same inputs should always produce same site ID."""
        id1 = generate_site_id("JT", 2500)
        id2 = generate_site_id("JT", 2500)
        assert id1 == id2

    def test_10k_site_ids_unique(self):
        """10,000 site IDs with different sequences should all be unique."""
        site_ids = set()
        for i in range(10000):
            site_id = generate_site_id("JK", i)
            assert site_id not in site_ids, f"Duplicate site ID at sequence {i}: {site_id}"
            site_ids.add(site_id)

        assert len(site_ids) == 10000


class TestWorkOrderIdGeneration:
    """Test deterministic work order ID generation."""

    def test_work_order_id_format(self):
        """Work order ID should match pattern: WO-YEAR-PROVINCE_CODE-SEQUENCE."""
        wo_id = generate_work_order_id(2024, "JK", 18472)
        assert wo_id == "WO-2024-JK-18472"

    def test_work_order_id_sequence_padding(self):
        """Work order sequence should be zero-padded to 5 digits."""
        wo_id = generate_work_order_id(2025, "BA", 42)
        assert wo_id == "WO-2025-BA-00042"

    def test_work_order_id_determinism(self):
        """Same inputs should always produce same work order ID."""
        id1 = generate_work_order_id(2024, "JB", 5000)
        id2 = generate_work_order_id(2024, "JB", 5000)
        assert id1 == id2

    def test_10k_work_order_ids_unique(self):
        """10,000 work order IDs with different sequences should all be unique."""
        wo_ids = set()
        for i in range(10000):
            wo_id = generate_work_order_id(2024, "JK", i)
            assert wo_id not in wo_ids, f"Duplicate work order ID at sequence {i}: {wo_id}"
            wo_ids.add(wo_id)

        assert len(wo_ids) == 10000


class TestAlarmIdGeneration:
    """Test deterministic alarm ID generation."""

    def test_alarm_id_format(self):
        """Alarm ID should match pattern: ALM-TIMESTAMP-VENDOR-SEQUENCE."""
        alarm_id = generate_alarm_id(1703123456, "ericsson", 1)
        assert alarm_id == "ALM-1703123456-ERI-0001"

    def test_alarm_id_sequence_padding(self):
        """Alarm sequence should be zero-padded to 4 digits."""
        alarm_id = generate_alarm_id(1703123456, "nokia", 42)
        assert alarm_id == "ALM-1703123456-NOK-0042"

    def test_alarm_id_determinism(self):
        """Same inputs should always produce same alarm ID."""
        id1 = generate_alarm_id(1704500000, "ericsson", 500)
        id2 = generate_alarm_id(1704500000, "ericsson", 500)
        assert id1 == id2

    def test_10k_alarm_ids_unique(self):
        """10,000 alarm IDs with different sequences should all be unique."""
        alarm_ids = set()
        for i in range(10000):
            alarm_id = generate_alarm_id(1703123456, "ericsson", i)
            assert alarm_id not in alarm_ids, f"Duplicate alarm ID at sequence {i}: {alarm_id}"
            alarm_ids.add(alarm_id)

        assert len(alarm_ids) == 10000


class TestCmdbCiIdGeneration:
    """Test deterministic CMDB CI ID generation."""

    def test_cmdb_ci_id_format(self):
        """CMDB CI ID should match pattern: CI-TYPE_CODE-SITE_ID-SEQUENCE."""
        ci_id = generate_cmdb_ci_id("RRU", "SITE-JK-1847", 1)
        assert ci_id == "CI-RRU-SITE-JK-1847-001"

    def test_cmdb_ci_id_sequence_padding(self):
        """CMDB CI sequence should be zero-padded to 3 digits."""
        ci_id = generate_cmdb_ci_id("PSU", "SITE-BA-0042", 5)
        assert ci_id == "CI-PSU-SITE-BA-0042-005"

    def test_cmdb_ci_id_determinism(self):
        """Same inputs should always produce same CMDB CI ID."""
        id1 = generate_cmdb_ci_id("BBU", "SITE-JT-2500", 100)
        id2 = generate_cmdb_ci_id("BBU", "SITE-JT-2500", 100)
        assert id1 == id2

    def test_10k_cmdb_ci_ids_unique(self):
        """10,000 CMDB CI IDs with different sequences should all be unique."""
        ci_ids = set()
        for i in range(10000):
            ci_id = generate_cmdb_ci_id("RRU", "SITE-JK-0001", i)
            assert ci_id not in ci_ids, f"Duplicate CMDB CI ID at sequence {i}: {ci_id}"
            ci_ids.add(ci_id)

        assert len(ci_ids) == 10000


class TestNoUuidFormat:
    """Verify that generated IDs do NOT use UUID v4 format."""

    def test_no_uuid_format_in_cell_ids(self):
        """Cell IDs should not contain UUID format (8-4-4-4-12 hex)."""
        cell_id = generate_cell_id("ericsson", "SITE-JK-0001", "LTE", 1)
        # UUID v4 pattern: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
        assert not _looks_like_uuid(cell_id), (
            f"Cell ID looks like UUID: {cell_id}"
        )

    def test_no_uuid_format_in_site_ids(self):
        """Site IDs should not contain UUID format."""
        site_id = generate_site_id("JK", 1847)
        assert not _looks_like_uuid(site_id), (
            f"Site ID looks like UUID: {site_id}"
        )

    def test_no_uuid_format_in_work_order_ids(self):
        """Work order IDs should not contain UUID format."""
        wo_id = generate_work_order_id(2024, "JK", 18472)
        assert not _looks_like_uuid(wo_id), (
            f"Work order ID looks like UUID: {wo_id}"
        )

    def test_no_uuid_format_in_alarm_ids(self):
        """Alarm IDs should not contain UUID format."""
        alarm_id = generate_alarm_id(1703123456, "ericsson", 1)
        assert not _looks_like_uuid(alarm_id), (
            f"Alarm ID looks like UUID: {alarm_id}"
        )

    def test_no_uuid_format_in_cmdb_ci_ids(self):
        """CMDB CI IDs should not contain UUID format."""
        ci_id = generate_cmdb_ci_id("RRU", "SITE-JK-1847", 1)
        assert not _looks_like_uuid(ci_id), (
            f"CMDB CI ID looks like UUID: {ci_id}"
        )


def _looks_like_uuid(s: str) -> bool:
    """
    Check if a string resembles UUID v4 format.

    UUID v4 format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
    Where x is hex and y is 8, 9, a, or b.
    """
    import re

    uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
    return bool(re.match(uuid_pattern, s, re.IGNORECASE))
