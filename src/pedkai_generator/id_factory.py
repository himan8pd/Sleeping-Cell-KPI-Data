"""
Deterministic ID factory for synthetic data generation.

Replaces UUID V4 identifiers with human-readable, collision-safe IDs based on
entity type, location (province), and sequence numbers.

Province codes follow Indonesian administrative boundaries (38 provinces + special regions).
"""

PROVINCE_CODES = {
    "Jakarta": "JK",
    "Jawa Barat": "JB",
    "Jawa Tengah": "JT",
    "Jawa Timur": "JI",
    "Banten": "BT",
    "Yogyakarta": "YO",
    "Bali": "BA",
    "Sumatera Utara": "SU",
    "Sumatera Barat": "SB",
    "Sumatera Selatan": "SS",
    "Riau": "RI",
    "Kepulauan Riau": "KR",
    "Jambi": "JA",
    "Bengkulu": "BE",
    "Lampung": "LA",
    "Bangka Belitung": "BB",
    "Aceh": "AC",
    "Kalimantan Barat": "KB",
    "Kalimantan Tengah": "KT",
    "Kalimantan Selatan": "KS",
    "Kalimantan Timur": "KU",
    "Kalimantan Utara": "KU2",
    "Sulawesi Utara": "SA",
    "Sulawesi Tengah": "ST",
    "Sulawesi Selatan": "SN",
    "Sulawesi Tenggara": "SG",
    "Gorontalo": "GO",
    "Sulawesi Barat": "SR",
    "Maluku": "MA",
    "Maluku Utara": "MU",
    "Papua": "PA",
    "Papua Barat": "PB",
    "Papua Selatan": "PS",
    "Papua Tengah": "PG",
    "Papua Pegunungan": "PP",
    "Nusa Tenggara Barat": "NB",
    "Nusa Tenggara Timur": "NT",
    "Kepulauan Seribu": "KP",
}

# Collision-safe sequence counters (per ID type)
_counters = {}


def generate_cell_id(vendor: str, site_id: str, cell_type: str, cell_num: int) -> str:
    """
    Generate a deterministic cell identifier.

    Format: {VENDOR}-{SITE_ID}-{CELL_TYPE}-{CELL_NUM}
    Example: ERB-SITE-JKT-1847-LTE-1

    Args:
        vendor: Equipment vendor (e.g., "ericsson", "nokia")
        site_id: Site identifier
        cell_type: Cell type (e.g., "LTE", "NR_SA", "NR_NSA")
        cell_num: Cell number/sequence within sector

    Returns:
        Formatted cell ID string
    """
    return f"{vendor[:3].upper()}-{site_id}-{cell_type.upper()}-{cell_num}"


def generate_site_id(province_code: str, sequence: int) -> str:
    """
    Generate a deterministic site identifier.

    Format: SITE-{PROVINCE_CODE}-{SEQUENCE}
    Example: SITE-JKT-1847

    Args:
        province_code: Two-letter province code
        sequence: Sequential number within province

    Returns:
        Formatted site ID string
    """
    return f"SITE-{province_code.upper()}-{sequence:04d}"


def generate_work_order_id(year: int, province_code: str, sequence: int) -> str:
    """
    Generate a deterministic work order identifier.

    Format: WO-{YEAR}-{PROVINCE_CODE}-{SEQUENCE}
    Example: WO-2024-JKT-18472

    Args:
        year: Year of work order issuance
        province_code: Two-letter province code
        sequence: Sequential number within province for that year

    Returns:
        Formatted work order ID string
    """
    return f"WO-{year}-{province_code.upper()}-{sequence:05d}"


def generate_alarm_id(timestamp: int, vendor: str, sequence: int) -> str:
    """
    Generate a deterministic alarm identifier.

    Format: ALM-{TIMESTAMP}-{VENDOR}-{SEQUENCE}
    Example: ALM-1703123456-ERB-0001

    Args:
        timestamp: Unix timestamp when alarm was raised
        vendor: Equipment vendor (e.g., "ericsson", "nokia")
        sequence: Sequence number for alarm within timestamp

    Returns:
        Formatted alarm ID string
    """
    return f"ALM-{timestamp}-{vendor[:3].upper()}-{sequence:04d}"


def generate_cmdb_ci_id(type_code: str, site_id: str, sequence: int) -> str:
    """
    Generate a deterministic CMDB Configuration Item (CI) identifier.

    Format: CI-{TYPE_CODE}-{SITE_ID}-{SEQUENCE}
    Example: CI-RRU-SITE-JKT-1847-001

    Args:
        type_code: Equipment type code (e.g., "RRU", "BBU", "PSU")
        site_id: Site identifier
        sequence: Sequential number within site for this equipment type

    Returns:
        Formatted CMDB CI ID string
    """
    return f"CI-{type_code.upper()}-{site_id}-{sequence:03d}"
