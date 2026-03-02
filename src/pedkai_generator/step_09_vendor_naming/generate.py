"""
Step 09: Vendor Naming Layer.

Maps internal (vendor-neutral) KPI column names from Phases 3–4 to their
Ericsson and Nokia PM counter equivalents.  Real operators never see names
like ``dl_throughput_mbps`` — they see ``pmRadioThpVolDl`` (Ericsson ENM)
or ``VS.RAB.DLThroughput`` (Nokia NetAct).

This step produces a **lookup table** — it does NOT rewrite the KPI
Parquet files.  Consumers apply the mapping at read/display time by
joining ``(internal_name, domain, vendor)`` → ``vendor_counter_name``.

Naming conventions implemented:

  **Ericsson (ENM / OSS-RC)**
    - Radio: ``pm`` prefix + PascalCase metric, e.g. ``pmRadioThpVolDl``
    - Transport: ``if`` prefix for interface counters, ``pm`` for system,
      e.g. ``ifHCInOctets``, ``pmOpticalRxPower``
    - Core: ``cm`` / ``pm`` prefix, e.g. ``pmMmeAttachSucc``
    - Power: ``pm`` prefix, e.g. ``pmBatteryVoltage``
    - Fixed BB: ``pm`` prefix, e.g. ``pmPonRxPower``

  **Nokia (NetAct / NSP)**
    - Radio: ``VS.`` dot-separated hierarchy, e.g. ``VS.RAB.DLThroughput``
    - Transport: ``IF-MIB`` style or ``EQUIPMENT`` counters,
      e.g. ``IF-MIB.ifHCInOctets``, ``EQUIPMENT.optRxPower``
    - Core: ``M8xxx`` 3GPP-style counters, e.g. ``M8010.attachSucc``
    - Power: ``EQUIPMENT.`` prefix, e.g. ``EQUIPMENT.batteryVoltage``
    - Fixed BB: ``GPON.`` prefix, e.g. ``GPON.rxOpticalPower``

Output:
  - output/vendor_naming_map.parquet  (~5 MB, lookup table)

Dependencies: Phases 3, 4 (reads column schemas — no actual KPI data needed)
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from rich.console import Console
from rich.table import Table

from pedkai_generator.config.settings import GeneratorConfig

console = Console()

# ---------------------------------------------------------------------------
# PyArrow output schema — vendor naming lookup table
# ---------------------------------------------------------------------------

VENDOR_NAMING_SCHEMA = pa.schema(
    [
        pa.field("mapping_id", pa.string(), nullable=False),
        pa.field("tenant_id", pa.string(), nullable=False),
        pa.field("internal_name", pa.string(), nullable=False),
        pa.field("domain", pa.string(), nullable=False),
        pa.field("vendor", pa.string(), nullable=False),
        pa.field("vendor_counter_name", pa.string(), nullable=False),
        pa.field("vendor_system", pa.string(), nullable=False),
        pa.field("unit", pa.string(), nullable=True),
        pa.field("description", pa.string(), nullable=True),
        pa.field("counter_family", pa.string(), nullable=True),
        pa.field("three_gpp_ref", pa.string(), nullable=True),
    ]
)

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------

DOMAIN_RADIO = "radio"
DOMAIN_TRANSPORT = "transport"
DOMAIN_FIXED_BB = "fixed_broadband"
DOMAIN_ENTERPRISE = "enterprise"
DOMAIN_CORE = "core"
DOMAIN_POWER = "power_environment"

VENDOR_ERICSSON = "ericsson"
VENDOR_NOKIA = "nokia"

SYSTEM_ERICSSON_ENM = "ericsson_enm"
SYSTEM_NOKIA_NETACT = "nokia_netact"

# ---------------------------------------------------------------------------
# Radio KPI mappings (Phase 3 — kpi_metrics_wide.parquet)
#
# Internal column names from step_03_radio_kpis
# ---------------------------------------------------------------------------

RADIO_KPI_MAPPINGS: list[dict[str, Any]] = [
    # --- SINR / Signal Quality ---
    {
        "internal": "sinr_mean_db",
        "ericsson": "pmRadioSinrMean",
        "nokia": "VS.SINR.Mean",
        "unit": "dB",
        "description": "Mean SINR across all UEs",
        "family": "RadioQuality",
        "ref": "TS 36.214 / TS 38.215",
    },
    {
        "internal": "sinr_p10_db",
        "ericsson": "pmRadioSinrPercentile10",
        "nokia": "VS.SINR.Pct10",
        "unit": "dB",
        "description": "10th percentile SINR",
        "family": "RadioQuality",
        "ref": None,
    },
    {
        "internal": "sinr_p50_db",
        "ericsson": "pmRadioSinrPercentile50",
        "nokia": "VS.SINR.Pct50",
        "unit": "dB",
        "description": "50th percentile SINR (median)",
        "family": "RadioQuality",
        "ref": None,
    },
    {
        "internal": "sinr_p90_db",
        "ericsson": "pmRadioSinrPercentile90",
        "nokia": "VS.SINR.Pct90",
        "unit": "dB",
        "description": "90th percentile SINR",
        "family": "RadioQuality",
        "ref": None,
    },
    # --- CQI ---
    {
        "internal": "cqi_mode",
        "ericsson": "pmRadioCqiDistrMode",
        "nokia": "VS.CQI.Mode",
        "unit": "index",
        "description": "Mode CQI reported by UEs (0-15)",
        "family": "RadioQuality",
        "ref": "TS 36.213 Table 7.2.3-1",
    },
    {
        "internal": "cqi_mean",
        "ericsson": "pmRadioCqiMean",
        "nokia": "VS.CQI.Mean",
        "unit": "index",
        "description": "Mean CQI across all UE reports",
        "family": "RadioQuality",
        "ref": "TS 36.213",
    },
    # --- Throughput ---
    {
        "internal": "dl_throughput_mbps",
        "ericsson": "pmRadioThpVolDl",
        "nokia": "VS.RAB.DLThroughput",
        "unit": "Mbps",
        "description": "Average downlink cell throughput",
        "family": "Throughput",
        "ref": "TS 32.425",
    },
    {
        "internal": "ul_throughput_mbps",
        "ericsson": "pmRadioThpVolUl",
        "nokia": "VS.RAB.ULThroughput",
        "unit": "Mbps",
        "description": "Average uplink cell throughput",
        "family": "Throughput",
        "ref": "TS 32.425",
    },
    {
        "internal": "dl_spectral_eff",
        "ericsson": "pmRadioSpectEffDl",
        "nokia": "VS.SpectralEff.DL",
        "unit": "bps/Hz",
        "description": "Downlink spectral efficiency",
        "family": "Throughput",
        "ref": None,
    },
    {
        "internal": "ul_spectral_eff",
        "ericsson": "pmRadioSpectEffUl",
        "nokia": "VS.SpectralEff.UL",
        "unit": "bps/Hz",
        "description": "Uplink spectral efficiency",
        "family": "Throughput",
        "ref": None,
    },
    # --- PRB Utilisation ---
    {
        "internal": "prb_util_dl_pct",
        "ericsson": "pmRadioPrbUsedDlDistr",
        "nokia": "VS.PRB.Util.DL",
        "unit": "%",
        "description": "Downlink PRB utilisation percentage",
        "family": "Resource",
        "ref": "TS 32.425 Clause 4.1.6",
    },
    {
        "internal": "prb_util_ul_pct",
        "ericsson": "pmRadioPrbUsedUlDistr",
        "nokia": "VS.PRB.Util.UL",
        "unit": "%",
        "description": "Uplink PRB utilisation percentage",
        "family": "Resource",
        "ref": "TS 32.425",
    },
    # --- Traffic Volume ---
    {
        "internal": "dl_data_volume_gb",
        "ericsson": "pmPdcpVolDlDrb",
        "nokia": "VS.PDCP.DL.DataVol",
        "unit": "GB",
        "description": "Downlink PDCP data volume",
        "family": "Traffic",
        "ref": "TS 32.425 Clause 4.1.1",
    },
    {
        "internal": "ul_data_volume_gb",
        "ericsson": "pmPdcpVolUlDrb",
        "nokia": "VS.PDCP.UL.DataVol",
        "unit": "GB",
        "description": "Uplink PDCP data volume",
        "family": "Traffic",
        "ref": "TS 32.425",
    },
    # --- Connected Users ---
    {
        "internal": "connected_users_mean",
        "ericsson": "pmRrcConnUsersMean",
        "nokia": "VS.RRC.ConnMean",
        "unit": "count",
        "description": "Mean number of RRC-connected users",
        "family": "Capacity",
        "ref": "TS 32.425 Clause 4.6",
    },
    {
        "internal": "connected_users_max",
        "ericsson": "pmRrcConnUsersMax",
        "nokia": "VS.RRC.ConnMax",
        "unit": "count",
        "description": "Maximum number of RRC-connected users",
        "family": "Capacity",
        "ref": "TS 32.425",
    },
    # --- RACH ---
    {
        "internal": "rach_attempts",
        "ericsson": "pmRachPreambleAtt",
        "nokia": "VS.RACH.PreambleAtt",
        "unit": "count",
        "description": "Total RACH preamble attempts",
        "family": "Access",
        "ref": "TS 32.425 Clause 4.3",
    },
    {
        "internal": "rach_success_rate",
        "ericsson": "pmRachPreambleSuccRate",
        "nokia": "VS.RACH.SuccRate",
        "unit": "%",
        "description": "RACH preamble success rate",
        "family": "Access",
        "ref": "TS 32.425",
    },
    # --- Handover ---
    {
        "internal": "ho_success_rate",
        "ericsson": "pmHoExeSuccRate",
        "nokia": "VS.HO.SuccRate",
        "unit": "%",
        "description": "Handover execution success rate",
        "family": "Mobility",
        "ref": "TS 32.425 Clause 4.4",
    },
    {
        "internal": "ho_attempts",
        "ericsson": "pmHoExeAtt",
        "nokia": "VS.HO.ExeAtt",
        "unit": "count",
        "description": "Total handover execution attempts",
        "family": "Mobility",
        "ref": "TS 32.425",
    },
    # --- BLER ---
    {
        "internal": "dl_bler_pct",
        "ericsson": "pmRadioDlBlerDistr",
        "nokia": "VS.DL.BLER",
        "unit": "%",
        "description": "Downlink block error rate",
        "family": "RadioQuality",
        "ref": "TS 36.213",
    },
    {
        "internal": "ul_bler_pct",
        "ericsson": "pmRadioUlBlerDistr",
        "nokia": "VS.UL.BLER",
        "unit": "%",
        "description": "Uplink block error rate",
        "family": "RadioQuality",
        "ref": None,
    },
    # --- Latency ---
    {
        "internal": "latency_mean_ms",
        "ericsson": "pmRadioLatencyMean",
        "nokia": "VS.Latency.Mean",
        "unit": "ms",
        "description": "Mean user-plane latency (one-way)",
        "family": "Latency",
        "ref": None,
    },
    # --- Transmit Power ---
    {
        "internal": "tx_power_mean_dbm",
        "ericsson": "pmRadioTxPowerMean",
        "nokia": "VS.TxPower.Mean",
        "unit": "dBm",
        "description": "Mean cell transmit power",
        "family": "RadioQuality",
        "ref": None,
    },
    # --- Timing Advance ---
    {
        "internal": "ta_mean_us",
        "ericsson": "pmRadioTaMean",
        "nokia": "VS.TA.Mean",
        "unit": "μs",
        "description": "Mean timing advance (cell range proxy)",
        "family": "Coverage",
        "ref": None,
    },
    # --- RRC Setup ---
    {
        "internal": "rrc_setup_success_rate",
        "ericsson": "pmRrcConnEstabSuccRate",
        "nokia": "VS.RRC.SetupSuccRate",
        "unit": "%",
        "description": "RRC connection setup success rate",
        "family": "Access",
        "ref": "TS 32.425 Clause 4.6.1",
    },
    # --- E-RAB / DRB Setup ---
    {
        "internal": "erab_setup_success_rate",
        "ericsson": "pmErabEstabSuccRate",
        "nokia": "VS.ERAB.SetupSuccRate",
        "unit": "%",
        "description": "E-RAB / DRB setup success rate",
        "family": "Access",
        "ref": "TS 32.425 Clause 4.2",
    },
    # --- Interference ---
    {
        "internal": "interference_mean_dbm",
        "ericsson": "pmRadioUlInterferenceMean",
        "nokia": "VS.UL.Interference.Mean",
        "unit": "dBm",
        "description": "Mean uplink interference power",
        "family": "RadioQuality",
        "ref": None,
    },
    # --- Availability ---
    {
        "internal": "cell_avail_pct",
        "ericsson": "pmCellAvailability",
        "nokia": "VS.Cell.Avail",
        "unit": "%",
        "description": "Cell availability percentage",
        "family": "Availability",
        "ref": "TS 32.425 Clause 4.7",
    },
    # --- RSRP (for reference signal quality) ---
    {
        "internal": "rsrp_mean_dbm",
        "ericsson": "pmRadioRsrpMean",
        "nokia": "VS.RSRP.Mean",
        "unit": "dBm",
        "description": "Mean Reference Signal Received Power",
        "family": "RadioQuality",
        "ref": "TS 36.214",
    },
    # --- PDCP Retransmission ---
    {
        "internal": "pdcp_retx_dl_pct",
        "ericsson": "pmPdcpPduRetxDl",
        "nokia": "VS.PDCP.Retx.DL",
        "unit": "%",
        "description": "Downlink PDCP PDU retransmission rate",
        "family": "RadioQuality",
        "ref": None,
    },
    # --- Paging ---
    {
        "internal": "paging_attempts",
        "ericsson": "pmPagDiscardAtt",
        "nokia": "VS.Paging.Att",
        "unit": "count",
        "description": "Paging message attempts",
        "family": "Access",
        "ref": None,
    },
    # --- MCS ---
    {
        "internal": "mcs_mean",
        "ericsson": "pmRadioMcsMean",
        "nokia": "VS.MCS.Mean",
        "unit": "index",
        "description": "Mean Modulation and Coding Scheme index",
        "family": "RadioQuality",
        "ref": "TS 36.213",
    },
    # --- Rank Indicator ---
    {
        "internal": "rank_mean",
        "ericsson": "pmRadioRiMean",
        "nokia": "VS.RI.Mean",
        "unit": "index",
        "description": "Mean Rank Indicator (MIMO layers)",
        "family": "RadioQuality",
        "ref": "TS 36.213",
    },
]

# ---------------------------------------------------------------------------
# Transport KPI mappings (Phase 4 — transport_kpis_wide.parquet)
# ---------------------------------------------------------------------------

TRANSPORT_KPI_MAPPINGS: list[dict[str, Any]] = [
    {
        "internal": "rx_bytes",
        "ericsson": "ifHCInOctets",
        "nokia": "IF-MIB.ifHCInOctets",
        "unit": "bytes",
        "description": "Interface received bytes (64-bit counter)",
        "family": "Interface",
        "ref": "RFC 2863",
    },
    {
        "internal": "tx_bytes",
        "ericsson": "ifHCOutOctets",
        "nokia": "IF-MIB.ifHCOutOctets",
        "unit": "bytes",
        "description": "Interface transmitted bytes (64-bit counter)",
        "family": "Interface",
        "ref": "RFC 2863",
    },
    {
        "internal": "rx_packets",
        "ericsson": "ifHCInUcastPkts",
        "nokia": "IF-MIB.ifHCInUcastPkts",
        "unit": "count",
        "description": "Interface received unicast packets",
        "family": "Interface",
        "ref": "RFC 2863",
    },
    {
        "internal": "tx_packets",
        "ericsson": "ifHCOutUcastPkts",
        "nokia": "IF-MIB.ifHCOutUcastPkts",
        "unit": "count",
        "description": "Interface transmitted unicast packets",
        "family": "Interface",
        "ref": "RFC 2863",
    },
    {
        "internal": "rx_errors",
        "ericsson": "ifInErrors",
        "nokia": "IF-MIB.ifInErrors",
        "unit": "count",
        "description": "Interface input error count",
        "family": "Interface",
        "ref": "RFC 2863",
    },
    {
        "internal": "tx_errors",
        "ericsson": "ifOutErrors",
        "nokia": "IF-MIB.ifOutErrors",
        "unit": "count",
        "description": "Interface output error count",
        "family": "Interface",
        "ref": "RFC 2863",
    },
    {
        "internal": "rx_drops",
        "ericsson": "ifInDiscards",
        "nokia": "IF-MIB.ifInDiscards",
        "unit": "count",
        "description": "Interface input discard count",
        "family": "Interface",
        "ref": "RFC 2863",
    },
    {
        "internal": "tx_drops",
        "ericsson": "ifOutDiscards",
        "nokia": "IF-MIB.ifOutDiscards",
        "unit": "count",
        "description": "Interface output discard count",
        "family": "Interface",
        "ref": "RFC 2863",
    },
    {
        "internal": "link_util_pct",
        "ericsson": "pmLinkUtilisation",
        "nokia": "SAP.linkUtil",
        "unit": "%",
        "description": "Link bandwidth utilisation percentage",
        "family": "Capacity",
        "ref": None,
    },
    {
        "internal": "latency_ms",
        "ericsson": "pmIpLatencyRtt",
        "nokia": "SAP.rttMs",
        "unit": "ms",
        "description": "Round-trip time latency",
        "family": "Performance",
        "ref": None,
    },
    {
        "internal": "jitter_ms",
        "ericsson": "pmIpJitter",
        "nokia": "SAP.jitterMs",
        "unit": "ms",
        "description": "IP jitter (inter-packet delay variation)",
        "family": "Performance",
        "ref": None,
    },
    {
        "internal": "packet_loss_pct",
        "ericsson": "pmIpPacketLoss",
        "nokia": "SAP.pktLossRatio",
        "unit": "%",
        "description": "Packet loss ratio",
        "family": "Performance",
        "ref": None,
    },
    {
        "internal": "optical_rx_power_dbm",
        "ericsson": "pmOpticalRxPower",
        "nokia": "EQUIPMENT.optRxPower",
        "unit": "dBm",
        "description": "Optical receiver power level",
        "family": "Optical",
        "ref": None,
    },
    {
        "internal": "optical_tx_power_dbm",
        "ericsson": "pmOpticalTxPower",
        "nokia": "EQUIPMENT.optTxPower",
        "unit": "dBm",
        "description": "Optical transmitter power level",
        "family": "Optical",
        "ref": None,
    },
    {
        "internal": "bgp_prefixes_received",
        "ericsson": "pmBgpPrefixesRcvd",
        "nokia": "BGP.prefixesReceived",
        "unit": "count",
        "description": "Number of BGP prefixes received",
        "family": "Routing",
        "ref": None,
    },
    {
        "internal": "bgp_session_uptime_pct",
        "ericsson": "pmBgpSessionUptime",
        "nokia": "BGP.sessionUptimePct",
        "unit": "%",
        "description": "BGP session uptime percentage",
        "family": "Routing",
        "ref": None,
    },
    {
        "internal": "cpu_util_pct",
        "ericsson": "pmCpuUtilisation",
        "nokia": "SYSTEM.cpuUtil",
        "unit": "%",
        "description": "Router/switch CPU utilisation",
        "family": "System",
        "ref": None,
    },
    {
        "internal": "memory_util_pct",
        "ericsson": "pmMemoryUtilisation",
        "nokia": "SYSTEM.memUtil",
        "unit": "%",
        "description": "Router/switch memory utilisation",
        "family": "System",
        "ref": None,
    },
]

# ---------------------------------------------------------------------------
# Fixed Broadband KPI mappings (Phase 4 — fixed_broadband_kpis_wide.parquet)
# ---------------------------------------------------------------------------

FIXED_BB_KPI_MAPPINGS: list[dict[str, Any]] = [
    {
        "internal": "pon_rx_power_dbm",
        "ericsson": "pmPonRxPower",
        "nokia": "GPON.rxOpticalPower",
        "unit": "dBm",
        "description": "PON receiver optical power at ONT",
        "family": "Optical",
        "ref": "ITU-T G.984",
    },
    {
        "internal": "pon_tx_power_dbm",
        "ericsson": "pmPonTxPower",
        "nokia": "GPON.txOpticalPower",
        "unit": "dBm",
        "description": "PON transmitter optical power at OLT",
        "family": "Optical",
        "ref": "ITU-T G.984",
    },
    {
        "internal": "pon_ber",
        "ericsson": "pmPonBitErrorRate",
        "nokia": "GPON.bitErrorRate",
        "unit": "ratio",
        "description": "PON bit error rate",
        "family": "Optical",
        "ref": "ITU-T G.984",
    },
    {
        "internal": "dl_throughput_mbps",
        "ericsson": "pmBbDlThroughput",
        "nokia": "GPON.dlThroughput",
        "unit": "Mbps",
        "description": "Downstream throughput (broadband)",
        "family": "Throughput",
        "ref": None,
    },
    {
        "internal": "ul_throughput_mbps",
        "ericsson": "pmBbUlThroughput",
        "nokia": "GPON.ulThroughput",
        "unit": "Mbps",
        "description": "Upstream throughput (broadband)",
        "family": "Throughput",
        "ref": None,
    },
    {
        "internal": "dl_data_volume_gb",
        "ericsson": "pmBbDlDataVol",
        "nokia": "GPON.dlDataVol",
        "unit": "GB",
        "description": "Downstream data volume",
        "family": "Traffic",
        "ref": None,
    },
    {
        "internal": "ul_data_volume_gb",
        "ericsson": "pmBbUlDataVol",
        "nokia": "GPON.ulDataVol",
        "unit": "GB",
        "description": "Upstream data volume",
        "family": "Traffic",
        "ref": None,
    },
    {
        "internal": "active_subscribers",
        "ericsson": "pmBbActiveSubscribers",
        "nokia": "GPON.activeONTs",
        "unit": "count",
        "description": "Number of active subscribers / ONTs on port",
        "family": "Capacity",
        "ref": None,
    },
    {
        "internal": "olt_cpu_util_pct",
        "ericsson": "pmOltCpuUtil",
        "nokia": "GPON.OLT.cpuUtil",
        "unit": "%",
        "description": "OLT CPU utilisation",
        "family": "System",
        "ref": None,
    },
    {
        "internal": "olt_memory_util_pct",
        "ericsson": "pmOltMemUtil",
        "nokia": "GPON.OLT.memUtil",
        "unit": "%",
        "description": "OLT memory utilisation",
        "family": "System",
        "ref": None,
    },
    {
        "internal": "pon_port_util_pct",
        "ericsson": "pmPonPortUtil",
        "nokia": "GPON.portUtil",
        "unit": "%",
        "description": "PON port bandwidth utilisation",
        "family": "Capacity",
        "ref": None,
    },
    {
        "internal": "ont_uptime_pct",
        "ericsson": "pmOntUptime",
        "nokia": "GPON.ONT.uptimePct",
        "unit": "%",
        "description": "ONT uptime percentage",
        "family": "Availability",
        "ref": None,
    },
    {
        "internal": "fec_corrections",
        "ericsson": "pmPonFecCorrections",
        "nokia": "GPON.fecCorrected",
        "unit": "count",
        "description": "FEC corrected codeword count",
        "family": "Optical",
        "ref": None,
    },
    {
        "internal": "latency_ms",
        "ericsson": "pmBbLatency",
        "nokia": "GPON.latencyMs",
        "unit": "ms",
        "description": "Broadband access latency",
        "family": "Performance",
        "ref": None,
    },
    {
        "internal": "packet_loss_pct",
        "ericsson": "pmBbPacketLoss",
        "nokia": "GPON.pktLossRatio",
        "unit": "%",
        "description": "Broadband packet loss ratio",
        "family": "Performance",
        "ref": None,
    },
    {
        "internal": "copper_snr_db",
        "ericsson": "pmDslSnrMargin",
        "nokia": "DSL.snrMarginDb",
        "unit": "dB",
        "description": "DSL SNR margin (FTTC copper leg)",
        "family": "DSL",
        "ref": "ITU-T G.997",
    },
    {
        "internal": "copper_attenuation_db",
        "ericsson": "pmDslAttenuation",
        "nokia": "DSL.attenuationDb",
        "unit": "dB",
        "description": "DSL line attenuation (FTTC copper leg)",
        "family": "DSL",
        "ref": "ITU-T G.997",
    },
]

# ---------------------------------------------------------------------------
# Enterprise circuit KPI mappings (Phase 4)
# ---------------------------------------------------------------------------

ENTERPRISE_KPI_MAPPINGS: list[dict[str, Any]] = [
    {
        "internal": "circuit_avail_pct",
        "ericsson": "pmCircuitAvailability",
        "nokia": "ETH.circuitAvail",
        "unit": "%",
        "description": "Ethernet circuit availability",
        "family": "Availability",
        "ref": None,
    },
    {
        "internal": "rx_bytes",
        "ericsson": "pmEthRxBytes",
        "nokia": "ETH.rxOctets",
        "unit": "bytes",
        "description": "Ethernet circuit received bytes",
        "family": "Interface",
        "ref": None,
    },
    {
        "internal": "tx_bytes",
        "ericsson": "pmEthTxBytes",
        "nokia": "ETH.txOctets",
        "unit": "bytes",
        "description": "Ethernet circuit transmitted bytes",
        "family": "Interface",
        "ref": None,
    },
    {
        "internal": "latency_ms",
        "ericsson": "pmEthLatency",
        "nokia": "ETH.frameDelayMs",
        "unit": "ms",
        "description": "Ethernet frame delay (one-way)",
        "family": "Performance",
        "ref": "ITU-T Y.1731",
    },
    {
        "internal": "jitter_ms",
        "ericsson": "pmEthJitter",
        "nokia": "ETH.frameDelayVar",
        "unit": "ms",
        "description": "Ethernet inter-frame delay variation",
        "family": "Performance",
        "ref": "ITU-T Y.1731",
    },
    {
        "internal": "packet_loss_pct",
        "ericsson": "pmEthFrameLoss",
        "nokia": "ETH.frameLossRatio",
        "unit": "%",
        "description": "Ethernet frame loss ratio",
        "family": "Performance",
        "ref": "ITU-T Y.1731",
    },
    {
        "internal": "link_util_pct",
        "ericsson": "pmEthLinkUtil",
        "nokia": "ETH.linkUtil",
        "unit": "%",
        "description": "Ethernet circuit link utilisation",
        "family": "Capacity",
        "ref": None,
    },
    {
        "internal": "crc_errors",
        "ericsson": "pmEthCrcErrors",
        "nokia": "ETH.crcErrors",
        "unit": "count",
        "description": "CRC error count on Ethernet circuit",
        "family": "Interface",
        "ref": None,
    },
    {
        "internal": "sla_compliance_pct",
        "ericsson": "pmEthSlaCompliance",
        "nokia": "ETH.slaCompliancePct",
        "unit": "%",
        "description": "SLA target compliance percentage",
        "family": "SLA",
        "ref": None,
    },
    {
        "internal": "mttr_hours",
        "ericsson": "pmEthMttr",
        "nokia": "ETH.mttrHours",
        "unit": "hours",
        "description": "Mean time to repair",
        "family": "SLA",
        "ref": None,
    },
    {
        "internal": "outage_minutes",
        "ericsson": "pmEthOutageMinutes",
        "nokia": "ETH.outageMinutes",
        "unit": "minutes",
        "description": "Total outage duration in reporting period",
        "family": "Availability",
        "ref": None,
    },
]

# ---------------------------------------------------------------------------
# Core element KPI mappings (Phase 4 — core_element_kpis_wide.parquet)
# ---------------------------------------------------------------------------

CORE_KPI_MAPPINGS: list[dict[str, Any]] = [
    {
        "internal": "attach_success_rate",
        "ericsson": "pmMmeAttachSuccRate",
        "nokia": "M8010.attachSuccRate",
        "unit": "%",
        "description": "MME/AMF attach procedure success rate",
        "family": "Signalling",
        "ref": "TS 32.425",
    },
    {
        "internal": "attach_attempts",
        "ericsson": "pmMmeAttachAtt",
        "nokia": "M8010.attachAtt",
        "unit": "count",
        "description": "Total attach procedure attempts",
        "family": "Signalling",
        "ref": "TS 32.425",
    },
    {
        "internal": "pdn_create_success_rate",
        "ericsson": "pmSgwPdnCreateSuccRate",
        "nokia": "M8010.pdnCreateSuccRate",
        "unit": "%",
        "description": "PDN / PDU session create success rate",
        "family": "Signalling",
        "ref": None,
    },
    {
        "internal": "service_request_success_rate",
        "ericsson": "pmMmeServReqSuccRate",
        "nokia": "M8010.serviceReqSuccRate",
        "unit": "%",
        "description": "Service request procedure success rate",
        "family": "Signalling",
        "ref": None,
    },
    {
        "internal": "tau_success_rate",
        "ericsson": "pmMmeTauSuccRate",
        "nokia": "M8010.tauSuccRate",
        "unit": "%",
        "description": "Tracking area update success rate",
        "family": "Signalling",
        "ref": None,
    },
    {
        "internal": "paging_success_rate",
        "ericsson": "pmMmePagingSuccRate",
        "nokia": "M8010.pagingSuccRate",
        "unit": "%",
        "description": "Paging procedure success rate",
        "family": "Signalling",
        "ref": None,
    },
    {
        "internal": "gtp_tunnel_success_rate",
        "ericsson": "pmSgwGtpTunnelSuccRate",
        "nokia": "M8010.gtpTunnelSuccRate",
        "unit": "%",
        "description": "GTP tunnel establishment success rate",
        "family": "Tunnelling",
        "ref": None,
    },
    {
        "internal": "active_bearers",
        "ericsson": "pmSgwActiveBearers",
        "nokia": "M8010.activeBearers",
        "unit": "count",
        "description": "Number of active EPS / 5G bearers",
        "family": "Capacity",
        "ref": None,
    },
    {
        "internal": "active_subscribers",
        "ericsson": "pmCoreActiveSubscribers",
        "nokia": "M8010.activeSubscribers",
        "unit": "count",
        "description": "Number of active attached subscribers",
        "family": "Capacity",
        "ref": None,
    },
    {
        "internal": "cpu_util_pct",
        "ericsson": "pmCoreCpuUtil",
        "nokia": "CORE-SYSTEM.cpuUtil",
        "unit": "%",
        "description": "Core element CPU utilisation",
        "family": "System",
        "ref": None,
    },
    {
        "internal": "memory_util_pct",
        "ericsson": "pmCoreMemUtil",
        "nokia": "CORE-SYSTEM.memUtil",
        "unit": "%",
        "description": "Core element memory utilisation",
        "family": "System",
        "ref": None,
    },
    {
        "internal": "throughput_gbps",
        "ericsson": "pmCoreThpVol",
        "nokia": "CORE-SYSTEM.throughputGbps",
        "unit": "Gbps",
        "description": "Core element aggregate throughput",
        "family": "Throughput",
        "ref": None,
    },
    {
        "internal": "signalling_load_pct",
        "ericsson": "pmCoreSignallingLoad",
        "nokia": "CORE-SYSTEM.signallingLoad",
        "unit": "%",
        "description": "Signalling processor load",
        "family": "Capacity",
        "ref": None,
    },
    {
        "internal": "diameter_success_rate",
        "ericsson": "pmHssDiaSuccRate",
        "nokia": "M8010.diameterSuccRate",
        "unit": "%",
        "description": "Diameter message success rate (HSS/UDM)",
        "family": "Signalling",
        "ref": None,
    },
    {
        "internal": "radius_auth_success_rate",
        "ericsson": "pmRadiusAuthSuccRate",
        "nokia": "AAA.authSuccRate",
        "unit": "%",
        "description": "RADIUS authentication success rate",
        "family": "AAA",
        "ref": None,
    },
    {
        "internal": "dns_query_success_rate",
        "ericsson": "pmDnsQuerySuccRate",
        "nokia": "DNS.querySuccRate",
        "unit": "%",
        "description": "DNS query resolution success rate",
        "family": "Services",
        "ref": None,
    },
    {
        "internal": "dhcp_lease_success_rate",
        "ericsson": "pmDhcpLeaseSuccRate",
        "nokia": "DHCP.leaseSuccRate",
        "unit": "%",
        "description": "DHCP lease allocation success rate",
        "family": "Services",
        "ref": None,
    },
    {
        "internal": "bng_sessions",
        "ericsson": "pmBngActiveSessions",
        "nokia": "BNG.activeSessions",
        "unit": "count",
        "description": "BNG active subscriber sessions",
        "family": "Capacity",
        "ref": None,
    },
    {
        "internal": "cgnat_port_util_pct",
        "ericsson": "pmCgnatPortUtil",
        "nokia": "CGNAT.portUtilPct",
        "unit": "%",
        "description": "CGNAT port block utilisation",
        "family": "Capacity",
        "ref": None,
    },
    {
        "internal": "ims_reg_success_rate",
        "ericsson": "pmImsCscfRegSuccRate",
        "nokia": "IMS.regSuccRate",
        "unit": "%",
        "description": "IMS registration success rate",
        "family": "IMS",
        "ref": None,
    },
    {
        "internal": "volte_call_setup_rate",
        "ericsson": "pmImsCallSetupSuccRate",
        "nokia": "IMS.callSetupSuccRate",
        "unit": "%",
        "description": "VoLTE call setup success rate",
        "family": "IMS",
        "ref": None,
    },
]

# ---------------------------------------------------------------------------
# Power / Environment KPI mappings (Phase 4)
# ---------------------------------------------------------------------------

POWER_KPI_MAPPINGS: list[dict[str, Any]] = [
    {
        "internal": "battery_voltage_v",
        "ericsson": "pmBatteryVoltage",
        "nokia": "EQUIPMENT.batteryVoltage",
        "unit": "V",
        "description": "Battery bank voltage",
        "family": "Power",
        "ref": None,
    },
    {
        "internal": "battery_charge_pct",
        "ericsson": "pmBatteryChargePct",
        "nokia": "EQUIPMENT.batteryCharge",
        "unit": "%",
        "description": "Battery state of charge",
        "family": "Power",
        "ref": None,
    },
    {
        "internal": "mains_voltage_v",
        "ericsson": "pmMainsVoltage",
        "nokia": "EQUIPMENT.mainsVoltage",
        "unit": "V",
        "description": "Mains input voltage",
        "family": "Power",
        "ref": None,
    },
    {
        "internal": "mains_available",
        "ericsson": "pmMainsAvail",
        "nokia": "EQUIPMENT.mainsAvail",
        "unit": "boolean",
        "description": "Mains power availability flag",
        "family": "Power",
        "ref": None,
    },
    {
        "internal": "load_current_a",
        "ericsson": "pmLoadCurrent",
        "nokia": "EQUIPMENT.loadCurrent",
        "unit": "A",
        "description": "Total DC load current",
        "family": "Power",
        "ref": None,
    },
    {
        "internal": "power_consumption_kw",
        "ericsson": "pmPowerConsumption",
        "nokia": "EQUIPMENT.powerConsumptionKw",
        "unit": "kW",
        "description": "Total site power consumption",
        "family": "Power",
        "ref": None,
    },
    {
        "internal": "indoor_temp_c",
        "ericsson": "pmTempIndoor",
        "nokia": "EQUIPMENT.indoorTemp",
        "unit": "°C",
        "description": "Indoor cabinet temperature",
        "family": "Environment",
        "ref": None,
    },
    {
        "internal": "outdoor_temp_c",
        "ericsson": "pmTempOutdoor",
        "nokia": "EQUIPMENT.outdoorTemp",
        "unit": "°C",
        "description": "Outdoor ambient temperature",
        "family": "Environment",
        "ref": None,
    },
    {
        "internal": "humidity_pct",
        "ericsson": "pmHumidity",
        "nokia": "EQUIPMENT.humidityPct",
        "unit": "%",
        "description": "Relative humidity inside shelter",
        "family": "Environment",
        "ref": None,
    },
    {
        "internal": "generator_runtime_hours",
        "ericsson": "pmGenRuntime",
        "nokia": "EQUIPMENT.genRuntimeHours",
        "unit": "hours",
        "description": "Generator cumulative runtime",
        "family": "Power",
        "ref": None,
    },
    {
        "internal": "generator_fuel_pct",
        "ericsson": "pmGenFuelLevel",
        "nokia": "EQUIPMENT.genFuelPct",
        "unit": "%",
        "description": "Generator fuel level",
        "family": "Power",
        "ref": None,
    },
    {
        "internal": "rectifier_efficiency_pct",
        "ericsson": "pmRectifierEfficiency",
        "nokia": "EQUIPMENT.rectEffPct",
        "unit": "%",
        "description": "Power rectifier conversion efficiency",
        "family": "Power",
        "ref": None,
    },
]


# ---------------------------------------------------------------------------
# Mapping builder
# ---------------------------------------------------------------------------


def _build_mapping_rows(
    mappings: list[dict[str, Any]],
    domain: str,
    tenant_id: str,
) -> list[dict[str, Any]]:
    """
    Expand a single-vendor-neutral mapping list into two rows per KPI
    (one for Ericsson, one for Nokia).
    """
    rows: list[dict[str, Any]] = []

    for m in mappings:
        internal_name = m["internal"]
        unit = m.get("unit")
        description = m.get("description")
        family = m.get("family")
        ref = m.get("ref")

        # Ericsson row
        rows.append(
            {
                "mapping_id": str(uuid.uuid4()),
                "tenant_id": tenant_id,
                "internal_name": internal_name,
                "domain": domain,
                "vendor": VENDOR_ERICSSON,
                "vendor_counter_name": m["ericsson"],
                "vendor_system": SYSTEM_ERICSSON_ENM,
                "unit": unit,
                "description": description,
                "counter_family": family,
                "three_gpp_ref": ref,
            }
        )

        # Nokia row
        rows.append(
            {
                "mapping_id": str(uuid.uuid4()),
                "tenant_id": tenant_id,
                "internal_name": internal_name,
                "domain": domain,
                "vendor": VENDOR_NOKIA,
                "vendor_counter_name": m["nokia"],
                "vendor_system": SYSTEM_NOKIA_NETACT,
                "unit": unit,
                "description": description,
                "counter_family": family,
                "three_gpp_ref": ref,
            }
        )

    return rows


def _write_mapping_parquet(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> tuple[int, float]:
    """
    Write the vendor naming mapping to Parquet.

    Returns (rows_written, file_size_mb).
    """
    arrays = []
    for field in VENDOR_NAMING_SCHEMA:
        col_data = [row.get(field.name) for row in rows]
        arrays.append(pa.array(col_data, type=field.type))

    batch = pa.RecordBatch.from_arrays(arrays, schema=VENDOR_NAMING_SCHEMA)

    writer = pq.ParquetWriter(
        output_path,
        schema=VENDOR_NAMING_SCHEMA,
        compression="zstd",
        compression_level=3,
    )
    writer.write_batch(batch)
    writer.close()

    size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0.0
    return len(rows), size_mb


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def apply_vendor_naming(config: GeneratorConfig) -> None:
    """
    Step 09 entry point: Generate vendor naming lookup table.

    Produces output/vendor_naming_map.parquet — a lookup table mapping
    (internal_name, domain, vendor) → vendor_counter_name.

    This is NOT a rewrite of KPI files — consumers apply the mapping
    at read/display time.
    """
    step_start = time.time()

    seed = config.seed_for("step_09_vendor_naming")
    console.print(f"[dim]Step 09 seed: {seed}[/dim]")

    tenant_id = config.tenant_id

    console.print("[bold]Vendor Naming Layer:[/bold] mapping internal KPI names to Ericsson/Nokia PM counters")

    config.ensure_output_dirs()

    # ── Build all mapping rows ────────────────────────────────
    all_rows: list[dict[str, Any]] = []

    # Radio
    console.print("\n[bold]Radio domain[/bold] (Phase 3 KPIs)...")
    radio_rows = _build_mapping_rows(RADIO_KPI_MAPPINGS, DOMAIN_RADIO, tenant_id)
    all_rows.extend(radio_rows)
    console.print(
        f"  [green]✓[/green] {len(RADIO_KPI_MAPPINGS)} internal names → "
        f"{len(radio_rows)} vendor mappings (Ericsson + Nokia)"
    )

    # Transport
    console.print("\n[bold]Transport domain[/bold] (Phase 4 KPIs)...")
    transport_rows = _build_mapping_rows(TRANSPORT_KPI_MAPPINGS, DOMAIN_TRANSPORT, tenant_id)
    all_rows.extend(transport_rows)
    console.print(
        f"  [green]✓[/green] {len(TRANSPORT_KPI_MAPPINGS)} internal names → {len(transport_rows)} vendor mappings"
    )

    # Fixed broadband
    console.print("\n[bold]Fixed broadband domain[/bold] (Phase 4 KPIs)...")
    fixed_bb_rows = _build_mapping_rows(FIXED_BB_KPI_MAPPINGS, DOMAIN_FIXED_BB, tenant_id)
    all_rows.extend(fixed_bb_rows)
    console.print(
        f"  [green]✓[/green] {len(FIXED_BB_KPI_MAPPINGS)} internal names → {len(fixed_bb_rows)} vendor mappings"
    )

    # Enterprise circuits
    console.print("\n[bold]Enterprise circuit domain[/bold] (Phase 4 KPIs)...")
    enterprise_rows = _build_mapping_rows(ENTERPRISE_KPI_MAPPINGS, DOMAIN_ENTERPRISE, tenant_id)
    all_rows.extend(enterprise_rows)
    console.print(
        f"  [green]✓[/green] {len(ENTERPRISE_KPI_MAPPINGS)} internal names → {len(enterprise_rows)} vendor mappings"
    )

    # Core
    console.print("\n[bold]Core network domain[/bold] (Phase 4 KPIs)...")
    core_rows = _build_mapping_rows(CORE_KPI_MAPPINGS, DOMAIN_CORE, tenant_id)
    all_rows.extend(core_rows)
    console.print(f"  [green]✓[/green] {len(CORE_KPI_MAPPINGS)} internal names → {len(core_rows)} vendor mappings")

    # Power/Environment
    console.print("\n[bold]Power/Environment domain[/bold] (Phase 4 KPIs)...")
    power_rows = _build_mapping_rows(POWER_KPI_MAPPINGS, DOMAIN_POWER, tenant_id)
    all_rows.extend(power_rows)
    console.print(f"  [green]✓[/green] {len(POWER_KPI_MAPPINGS)} internal names → {len(power_rows)} vendor mappings")

    # ── Write output ──────────────────────────────────────────
    console.print(f"\n[bold]Writing vendor_naming_map.parquet...[/bold]")
    output_path = config.paths.output_dir / "vendor_naming_map.parquet"
    rows_written, size_mb = _write_mapping_parquet(all_rows, output_path)
    console.print(f"  [green]✓[/green] {rows_written:,} rows, {len(VENDOR_NAMING_SCHEMA)} columns, {size_mb:.2f} MB")

    # ── Summary tables ────────────────────────────────────────
    total_elapsed = time.time() - step_start
    console.print()

    # Domain breakdown
    summary_table = Table(
        title="Step 09: Vendor Naming — Domain Breakdown",
        show_header=True,
    )
    summary_table.add_column("Domain", style="bold", width=22)
    summary_table.add_column("Internal KPIs", justify="right", width=16)
    summary_table.add_column("Ericsson Counters", justify="right", width=18)
    summary_table.add_column("Nokia Counters", justify="right", width=18)

    domain_data = [
        (DOMAIN_RADIO, RADIO_KPI_MAPPINGS),
        (DOMAIN_TRANSPORT, TRANSPORT_KPI_MAPPINGS),
        (DOMAIN_FIXED_BB, FIXED_BB_KPI_MAPPINGS),
        (DOMAIN_ENTERPRISE, ENTERPRISE_KPI_MAPPINGS),
        (DOMAIN_CORE, CORE_KPI_MAPPINGS),
        (DOMAIN_POWER, POWER_KPI_MAPPINGS),
    ]

    total_internal = 0
    total_vendor = 0
    for domain, mappings in domain_data:
        n = len(mappings)
        total_internal += n
        total_vendor += n * 2
        summary_table.add_row(domain, str(n), str(n), str(n))

    summary_table.add_section()
    summary_table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{total_internal}[/bold]",
        f"[bold]{total_internal}[/bold]",
        f"[bold]{total_internal}[/bold]",
    )
    console.print(summary_table)

    # Sample mappings
    console.print()
    sample_table = Table(
        title="Sample Mappings (first 10)",
        show_header=True,
    )
    sample_table.add_column("Internal Name", style="bold", width=26)
    sample_table.add_column("Domain", width=18)
    sample_table.add_column("Vendor", width=10)
    sample_table.add_column("Vendor Counter", width=32)
    sample_table.add_column("System", width=16)

    for row in all_rows[:10]:
        sample_table.add_row(
            row["internal_name"],
            row["domain"],
            row["vendor"],
            row["vendor_counter_name"],
            row["vendor_system"],
        )
    console.print(sample_table)

    # Output summary
    console.print()
    output_table = Table(
        title="Output Summary",
        show_header=True,
    )
    output_table.add_column("Metric", style="bold", width=36)
    output_table.add_column("Value", justify="right", width=18)
    output_table.add_row("Total internal KPI names", f"{total_internal}")
    output_table.add_row("Total vendor mappings", f"{total_vendor}")
    output_table.add_row("Vendors", "Ericsson, Nokia")
    output_table.add_row("Domains", f"{len(domain_data)}")
    output_table.add_row("Columns", f"{len(VENDOR_NAMING_SCHEMA)}")
    output_table.add_row("File size", f"{size_mb:.2f} MB")
    output_table.add_row(
        "Usage",
        "JOIN on (internal_name, domain, vendor)",
    )
    console.print(output_table)

    time_str = f"{total_elapsed:.1f}s" if total_elapsed < 60 else f"{total_elapsed / 60:.1f}m"
    console.print(
        f"\n[bold green]✓ Step 09 complete.[/bold green] "
        f"Generated {total_vendor} vendor naming mappings across {len(domain_data)} domains "
        f"({size_mb:.2f} MB) in {time_str}"
    )
    console.print(
        "[dim]This is a lookup table — KPI Parquet files from Phases 3–4 are "
        "NOT rewritten. Consumers apply the mapping at read/display time by "
        "joining (internal_name, domain, vendor) → vendor_counter_name. "
        "Ericsson entities use ericsson_enm counter names, Nokia entities "
        "use nokia_netact counter names.[/dim]"
    )
