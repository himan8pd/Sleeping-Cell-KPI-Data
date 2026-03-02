"""
Radio-Layer Physics Engine.

Implements the core signal-propagation and link-budget chain:

    Path Loss → RSRP → SINR → CQI → MCS → Throughput

Per-RAT models:
  - LTE:    3GPP TS 36.213 CQI table, TS 36.214 path loss, max 256QAM
  - NR-NSA: EN-DC dual bearer — LTE anchor uses LTE model, NR SCG leg
            uses NR model with wider bandwidth / higher order modulation
  - NR-SA:  3GPP TS 38.214 CQI/MCS table, standalone 5GC, 5QI-based QoS

All functions are vectorised (operate on numpy arrays) for performance.
The design is stateless: each function takes cell parameters + environmental
conditions and returns KPI arrays.  The orchestrator (generate.py) handles
the time dimension and cell iteration.

Key references:
  - 3GPP TS 36.213 Table 7.2.3-1   (LTE CQI → spectral efficiency)
  - 3GPP TS 38.214 Table 5.2.2.1-3 (NR CQI → spectral efficiency, 256QAM)
  - 3GPP TS 36.942                  (propagation models)
  - ITU-R P.1238                    (indoor propagation)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOLTZMANN_DBM_HZ = -174.0  # Thermal noise floor in dBm/Hz at 290 K
NOISE_FIGURE_UE_DB = 7.0  # Typical UE noise figure for sub-6 GHz
NOISE_FIGURE_UE_MMWAVE_DB = 10.0  # UE noise figure for mmWave
SPEED_OF_LIGHT = 3e8  # m/s


# ---------------------------------------------------------------------------
# CQI / MCS / Spectral-efficiency tables (3GPP aligned)
# ---------------------------------------------------------------------------

# LTE CQI table — 3GPP TS 36.213 Table 7.2.3-1 (4-bit CQI, 0 = out of range)
# Index: CQI 0..15
# Values: (modulation_order, code_rate_x1024, spectral_efficiency_bps_hz)
LTE_CQI_TABLE: list[tuple[str, int, float]] = [
    ("none", 0, 0.0),  # CQI 0 — out of range
    ("QPSK", 78, 0.1523),  # CQI 1
    ("QPSK", 120, 0.2344),  # CQI 2
    ("QPSK", 193, 0.3770),  # CQI 3
    ("QPSK", 308, 0.6016),  # CQI 4
    ("QPSK", 449, 0.8770),  # CQI 5
    ("QPSK", 602, 1.1758),  # CQI 6
    ("16QAM", 378, 1.4766),  # CQI 7
    ("16QAM", 490, 1.9141),  # CQI 8
    ("16QAM", 616, 2.4063),  # CQI 9
    ("64QAM", 466, 2.7305),  # CQI 10
    ("64QAM", 567, 3.3223),  # CQI 11
    ("64QAM", 666, 3.9023),  # CQI 12
    ("64QAM", 772, 4.5234),  # CQI 13
    ("64QAM", 873, 5.1152),  # CQI 14
    ("64QAM", 948, 5.5547),  # CQI 15
]

# NR CQI table — 3GPP TS 38.214 Table 5.2.2.1-3 (256QAM capable)
# Higher spectral efficiencies than LTE due to 256QAM and better LDPC coding
NR_CQI_TABLE: list[tuple[str, int, float]] = [
    ("none", 0, 0.0),  # CQI 0
    ("QPSK", 78, 0.1523),  # CQI 1
    ("QPSK", 193, 0.3770),  # CQI 2
    ("QPSK", 449, 0.8770),  # CQI 3
    ("16QAM", 378, 1.4766),  # CQI 4
    ("16QAM", 490, 1.9141),  # CQI 5
    ("16QAM", 616, 2.4063),  # CQI 6
    ("64QAM", 466, 2.7305),  # CQI 7
    ("64QAM", 567, 3.3223),  # CQI 8
    ("64QAM", 666, 3.9023),  # CQI 9
    ("64QAM", 772, 4.5234),  # CQI 10
    ("64QAM", 873, 5.1152),  # CQI 11
    ("256QAM", 711, 5.5547),  # CQI 12
    ("256QAM", 797, 6.2266),  # CQI 13
    ("256QAM", 885, 6.9141),  # CQI 14
    ("256QAM", 948, 7.4063),  # CQI 15
]

# Spectral efficiency arrays for fast vectorised lookup
LTE_SE_BY_CQI = np.array([row[2] for row in LTE_CQI_TABLE], dtype=np.float64)
NR_SE_BY_CQI = np.array([row[2] for row in NR_CQI_TABLE], dtype=np.float64)

# LTE MCS table — simplified mapping CQI → MCS index (0-28)
# MCS index approximation: floor(CQI * 28/15), capped at 28
LTE_MCS_BY_CQI = np.array(
    [0, 0, 2, 4, 6, 8, 10, 12, 14, 17, 19, 21, 23, 25, 27, 28],
    dtype=np.float64,
)

# NR MCS table (Table 5.1.3.1-2 of TS 38.214, 256QAM table)
NR_MCS_BY_CQI = np.array(
    [0, 0, 2, 5, 7, 9, 11, 14, 16, 19, 21, 23, 25, 26, 27, 28],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# SINR → CQI mapping thresholds (dB)
# ---------------------------------------------------------------------------

# SINR thresholds for LTE CQI mapping — derived from link-level simulations
# SINR < threshold[i] → CQI = i-1 (CQI 0 for SINR < -6.7)
LTE_SINR_TO_CQI_THRESHOLDS = np.array(
    [-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1, 10.3, 11.7, 14.1, 16.3, 18.7, 21.0, 22.7],
    dtype=np.float64,
)

# NR SINR → CQI thresholds (slightly different due to LDPC coding gain ~0.5-1 dB)
NR_SINR_TO_CQI_THRESHOLDS = np.array(
    [-7.2, -5.2, -2.0, 0.5, 2.5, 4.5, 6.2, 8.5, 10.5, 12.0, 14.5, 16.5, 19.0, 21.5, 23.0],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# Deployment profile physical parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeploymentPhysics:
    """Physical-layer parameters that vary by deployment environment."""

    # Path-loss model parameters
    pl_exponent: float  # Path-loss exponent (2 = free space, 3.5–4.5 typical)
    shadow_fading_std_db: float  # Log-normal shadow fading σ (dB)

    # Typical inter-site distance (meters) — used for interference estimation
    typical_isd_m: float

    # Antenna height above ground (meters) — base for height model
    typical_antenna_height_m: float

    # Indoor penetration loss (dB)
    indoor_loss_db: float

    # Clutter/environment loss (dB) — additional fixed loss for environment
    environment_loss_db: float

    # Typical number of active UEs (mean per cell in busy hour)
    typical_ue_busy_hour: float

    # Baseline interference-over-thermal (dB) in busy hour
    baseline_iot_db: float


DEPLOYMENT_PHYSICS: dict[str, DeploymentPhysics] = {
    "dense_urban": DeploymentPhysics(
        pl_exponent=3.8,
        shadow_fading_std_db=8.0,
        typical_isd_m=350,
        typical_antenna_height_m=25.0,
        indoor_loss_db=20.0,
        environment_loss_db=8.0,
        typical_ue_busy_hour=120.0,
        baseline_iot_db=8.0,
    ),
    "urban": DeploymentPhysics(
        pl_exponent=3.6,
        shadow_fading_std_db=7.0,
        typical_isd_m=500,
        typical_antenna_height_m=30.0,
        indoor_loss_db=18.0,
        environment_loss_db=5.0,
        typical_ue_busy_hour=80.0,
        baseline_iot_db=6.0,
    ),
    "suburban": DeploymentPhysics(
        pl_exponent=3.4,
        shadow_fading_std_db=6.0,
        typical_isd_m=1200,
        typical_antenna_height_m=35.0,
        indoor_loss_db=15.0,
        environment_loss_db=3.0,
        typical_ue_busy_hour=40.0,
        baseline_iot_db=4.0,
    ),
    "rural": DeploymentPhysics(
        pl_exponent=3.2,
        shadow_fading_std_db=5.0,
        typical_isd_m=3000,
        typical_antenna_height_m=40.0,
        indoor_loss_db=12.0,
        environment_loss_db=1.0,
        typical_ue_busy_hour=15.0,
        baseline_iot_db=2.0,
    ),
    "deep_rural": DeploymentPhysics(
        pl_exponent=3.0,
        shadow_fading_std_db=4.0,
        typical_isd_m=8000,
        typical_antenna_height_m=45.0,
        indoor_loss_db=10.0,
        environment_loss_db=0.0,
        typical_ue_busy_hour=5.0,
        baseline_iot_db=1.0,
    ),
    "indoor": DeploymentPhysics(
        pl_exponent=3.0,
        shadow_fading_std_db=5.0,
        typical_isd_m=50,
        typical_antenna_height_m=3.0,
        indoor_loss_db=0.0,  # Already indoor
        environment_loss_db=2.0,
        typical_ue_busy_hour=60.0,
        baseline_iot_db=5.0,
    ),
}


# ---------------------------------------------------------------------------
# Path-loss models
# ---------------------------------------------------------------------------


def cost231_hata_path_loss(
    freq_mhz: np.ndarray,
    distance_m: np.ndarray,
    bs_height_m: np.ndarray,
    ue_height_m: float = 1.5,
    environment: str = "urban",
) -> np.ndarray:
    """
    COST 231-Hata path-loss model (150 MHz – 2000 MHz).

    Returns path loss in dB.  For frequencies above 2 GHz we use an
    extended log-distance model instead (see below).

    Parameters
    ----------
    freq_mhz : array  — carrier frequency in MHz
    distance_m : array — 2D distance in metres (min 20 m)
    bs_height_m : array — base-station antenna height in metres
    ue_height_m : float — mobile antenna height (default 1.5 m)
    environment : str — "urban", "suburban", "rural"

    Returns
    -------
    path_loss_db : array
    """
    # Clamp distance to avoid log(0)
    d_km = np.maximum(distance_m, 20.0) / 1000.0
    f = np.clip(freq_mhz, 150.0, 2000.0)

    # Mobile antenna height correction factor
    a_hm = (1.1 * np.log10(f) - 0.7) * ue_height_m - (1.56 * np.log10(f) - 0.8)

    # Basic COST 231-Hata
    pl = (
        46.3
        + 33.9 * np.log10(f)
        - 13.82 * np.log10(np.maximum(bs_height_m, 1.0))
        - a_hm
        + (44.9 - 6.55 * np.log10(np.maximum(bs_height_m, 1.0))) * np.log10(d_km)
    )

    # Metropolitan correction (3 dB for dense urban)
    if environment in ("dense_urban",):
        pl += 3.0
    elif environment in ("suburban",):
        pl -= 2.0 * (np.log10(f / 28.0)) ** 2 + 5.4
    elif environment in ("rural", "deep_rural"):
        pl -= 4.78 * (np.log10(f)) ** 2 + 18.33 * np.log10(f) - 40.94

    return pl


def log_distance_path_loss(
    freq_mhz: np.ndarray,
    distance_m: np.ndarray,
    pl_exponent: float,
    reference_distance_m: float = 1.0,
) -> np.ndarray:
    """
    Generic log-distance path-loss model.

    PL(d) = PL(d0) + 10 * n * log10(d / d0)

    where PL(d0) = free-space path loss at reference distance d0.

    Suitable for frequencies above 2 GHz and for indoor environments.
    """
    d = np.maximum(distance_m, reference_distance_m)

    # Free-space path loss at reference distance
    wavelength_m = SPEED_OF_LIGHT / (freq_mhz * 1e6)
    fspl_d0 = 20.0 * np.log10(4.0 * np.pi * reference_distance_m / wavelength_m)

    pl = fspl_d0 + 10.0 * pl_exponent * np.log10(d / reference_distance_m)
    return pl


def compute_path_loss(
    freq_mhz: np.ndarray,
    distance_m: np.ndarray,
    bs_height_m: np.ndarray,
    deployment_profile: str,
) -> np.ndarray:
    """
    Compute path loss using the appropriate model for the frequency and environment.

    - Sub-2 GHz: COST 231-Hata
    - 2–6 GHz: Extended log-distance (3GPP TR 38.901-like)
    - Above 6 GHz (mmWave): Log-distance with higher exponent + atmospheric absorption
    - Indoor: Always log-distance with indoor-specific exponent
    """
    phys = DEPLOYMENT_PHYSICS.get(deployment_profile, DEPLOYMENT_PHYSICS["suburban"])

    if deployment_profile == "indoor":
        # ITU-R P.1238 indoor model
        return log_distance_path_loss(freq_mhz, distance_m, phys.pl_exponent)

    # For outdoor: use COST 231-Hata for sub-2 GHz, log-distance otherwise
    pl = np.zeros_like(freq_mhz, dtype=np.float64)

    sub2g_mask = freq_mhz <= 2000.0
    above2g_mask = (freq_mhz > 2000.0) & (freq_mhz <= 6000.0)
    mmwave_mask = freq_mhz > 6000.0

    if np.any(sub2g_mask):
        pl[sub2g_mask] = cost231_hata_path_loss(
            freq_mhz[sub2g_mask],
            distance_m[sub2g_mask] if distance_m.shape == freq_mhz.shape else distance_m[sub2g_mask],
            bs_height_m[sub2g_mask] if bs_height_m.shape == freq_mhz.shape else bs_height_m[sub2g_mask],
            environment=deployment_profile,
        )

    if np.any(above2g_mask):
        # 3GPP TR 38.901 UMa NLOS-like
        pl[above2g_mask] = log_distance_path_loss(
            freq_mhz[above2g_mask],
            distance_m[above2g_mask] if distance_m.shape == freq_mhz.shape else distance_m[above2g_mask],
            phys.pl_exponent + 0.2,  # Slightly higher for mid-band
        )

    if np.any(mmwave_mask):
        # mmWave: higher exponent + atmospheric absorption
        base_pl = log_distance_path_loss(
            freq_mhz[mmwave_mask],
            distance_m[mmwave_mask] if distance_m.shape == freq_mhz.shape else distance_m[mmwave_mask],
            phys.pl_exponent + 0.8,  # Significantly higher for mmWave
        )
        # Add atmospheric absorption: ~0.01 dB/m at 28 GHz, ~0.02 dB/m at 39 GHz
        d_mmw = distance_m[mmwave_mask] if distance_m.shape == freq_mhz.shape else distance_m[mmwave_mask]
        absorption = 0.01 * (freq_mhz[mmwave_mask] / 28000.0) * d_mmw
        pl[mmwave_mask] = base_pl + absorption

    return pl


# ---------------------------------------------------------------------------
# RSRP computation
# ---------------------------------------------------------------------------


def compute_rsrp(
    tx_power_dbm: np.ndarray,
    path_loss_db: np.ndarray,
    antenna_gain_db: np.ndarray,
    shadow_fading_db: np.ndarray,
    indoor_loss_db: np.ndarray,
    environment_loss_db: np.ndarray,
) -> np.ndarray:
    """
    Compute RSRP (Reference Signal Received Power) in dBm.

    RSRP = Tx Power + Antenna Gain - Path Loss - Shadow Fading - Indoor Loss - Env Loss

    For LTE: RSRP is measured on the reference signal, which is typically
    transmitted at full power per RE (Resource Element).  We model this as
    the total cell RS power (typically Tx power - ~3 dB for antenna port sharing).
    """
    rsrp = tx_power_dbm + antenna_gain_db - path_loss_db - shadow_fading_db - indoor_loss_db - environment_loss_db
    return np.clip(rsrp, -140.0, -30.0)


# ---------------------------------------------------------------------------
# SINR computation
# ---------------------------------------------------------------------------


def compute_noise_power_dbm(
    bandwidth_mhz: np.ndarray,
    freq_mhz: np.ndarray,
) -> np.ndarray:
    """
    Compute thermal noise power in dBm for a given bandwidth.

    N = kTB = -174 dBm/Hz + 10*log10(BW_Hz) + NF
    """
    bw_hz = bandwidth_mhz * 1e6
    nf = np.where(freq_mhz > 6000.0, NOISE_FIGURE_UE_MMWAVE_DB, NOISE_FIGURE_UE_DB)
    return BOLTZMANN_DBM_HZ + 10.0 * np.log10(bw_hz) + nf


def compute_sinr(
    rsrp_dbm: np.ndarray,
    noise_power_dbm: np.ndarray,
    interference_iot_db: np.ndarray,
) -> np.ndarray:
    """
    Compute SINR in dB.

    SINR = RSRP / (N + I)

    where I = interference modelled as IoT (Interference over Thermal) above
    the noise floor.  IoT is driven by load (PRB utilisation of neighbour cells).

    In linear domain:
        N_linear = 10^(noise_power_dbm/10)
        I_linear = N_linear * (10^(iot_db/10) - 1)
        S_linear = 10^(rsrp_dbm/10)
        SINR = S / (N + I) = S / (N * 10^(iot_db/10))
    """
    sinr_db = rsrp_dbm - noise_power_dbm - interference_iot_db
    return np.clip(sinr_db, -20.0, 50.0)


def compute_rsrq(
    rsrp_dbm: np.ndarray,
    noise_power_dbm: np.ndarray,
    interference_iot_db: np.ndarray,
    n_prb: np.ndarray,
) -> np.ndarray:
    """
    Compute RSRQ (Reference Signal Received Quality) in dB.

    RSRQ = N × RSRP / RSSI

    where RSSI ≈ N × RSRP + interference + noise (measured across N PRBs).
    In practice RSRQ ∈ [-20, -3] dB.

    Simplified model:
        RSRQ ≈ 10*log10(N) + RSRP - RSSI_per_prb
        where RSSI_per_prb encompasses signal + interference + noise
    """
    # RSSI in dBm per PRB ≈ signal + interference + noise
    # In linear: RSSI = 10^(rsrp/10) + 10^((noise + iot)/10)
    signal_lin = np.power(10.0, rsrp_dbm / 10.0)
    noise_plus_intf_lin = np.power(10.0, (noise_power_dbm + interference_iot_db) / 10.0)
    rssi_per_prb_lin = signal_lin + noise_plus_intf_lin

    # RSRQ = N * RSRP / RSSI  (linear), then to dB
    # But RSSI is already total across N PRBs, so:
    rsrq = 10.0 * np.log10(signal_lin / rssi_per_prb_lin)
    return np.clip(rsrq, -20.0, -3.0)


# ---------------------------------------------------------------------------
# CQI mapping
# ---------------------------------------------------------------------------


def sinr_to_cqi_lte(sinr_db: np.ndarray) -> np.ndarray:
    """Map SINR (dB) to LTE CQI (0-15) using 3GPP threshold table."""
    cqi = np.searchsorted(LTE_SINR_TO_CQI_THRESHOLDS, sinr_db, side="right").astype(np.float64)
    return np.clip(cqi, 0.0, 15.0)


def sinr_to_cqi_nr(sinr_db: np.ndarray) -> np.ndarray:
    """Map SINR (dB) to NR CQI (0-15) using NR threshold table."""
    cqi = np.searchsorted(NR_SINR_TO_CQI_THRESHOLDS, sinr_db, side="right").astype(np.float64)
    return np.clip(cqi, 0.0, 15.0)


# ---------------------------------------------------------------------------
# MCS mapping
# ---------------------------------------------------------------------------


def cqi_to_mcs_lte(cqi: np.ndarray) -> np.ndarray:
    """Map CQI (0-15) to LTE MCS index (0-28)."""
    idx = np.clip(np.round(cqi).astype(np.int32), 0, 15)
    return LTE_MCS_BY_CQI[idx]


def cqi_to_mcs_nr(cqi: np.ndarray) -> np.ndarray:
    """Map CQI (0-15) to NR MCS index (0-28)."""
    idx = np.clip(np.round(cqi).astype(np.int32), 0, 15)
    return NR_MCS_BY_CQI[idx]


# ---------------------------------------------------------------------------
# Throughput computation
# ---------------------------------------------------------------------------


def compute_throughput_mbps(
    spectral_efficiency: np.ndarray,
    bandwidth_mhz: np.ndarray,
    n_prb: np.ndarray,
    prb_utilization_frac: np.ndarray,
    n_mimo_layers: np.ndarray,
    overhead_fraction: float = 0.15,
    dl: bool = True,
) -> np.ndarray:
    """
    Compute achievable throughput in Mbps.

    Throughput = SE × BW_eff × MIMO_layers × PRB_util × (1 - overhead)

    where BW_eff is the effective bandwidth (accounting for guard bands).

    Parameters
    ----------
    spectral_efficiency : array — bps/Hz from CQI table
    bandwidth_mhz : array — channel bandwidth
    n_prb : array — number of PRBs (drives effective BW)
    prb_utilization_frac : array — fraction of PRBs allocated (0-1)
    n_mimo_layers : array — MIMO spatial multiplexing layers (1, 2, or 4)
    overhead_fraction : float — control channel + reference signal overhead
    dl : bool — True for downlink, False for uplink

    Returns
    -------
    throughput_mbps : array
    """
    # Effective bandwidth per PRB: 12 subcarriers × 15 kHz = 180 kHz (LTE)
    # For NR with 30 kHz SCS: 12 × 30 kHz = 360 kHz per PRB
    # We use the actual number of PRBs × 180 kHz as a simplified model
    # that works for both LTE and NR (NR has roughly double PRBs for same BW)
    bw_effective_mhz = bandwidth_mhz * (1.0 - 0.10)  # 10% guard band

    throughput = (
        spectral_efficiency * bw_effective_mhz * n_mimo_layers * prb_utilization_frac * (1.0 - overhead_fraction)
    )

    # UL is typically lower: fewer MIMO layers, lower MCS
    if not dl:
        throughput *= 0.45  # UL/DL asymmetry factor

    return np.maximum(throughput, 0.0)


def cqi_to_spectral_efficiency_lte(cqi: np.ndarray) -> np.ndarray:
    """Look up LTE spectral efficiency from CQI."""
    idx = np.clip(np.round(cqi).astype(np.int32), 0, 15)
    return LTE_SE_BY_CQI[idx]


def cqi_to_spectral_efficiency_nr(cqi: np.ndarray) -> np.ndarray:
    """Look up NR spectral efficiency from CQI."""
    idx = np.clip(np.round(cqi).astype(np.int32), 0, 15)
    return NR_SE_BY_CQI[idx]


# ---------------------------------------------------------------------------
# BLER model
# ---------------------------------------------------------------------------


def compute_bler(
    sinr_db: np.ndarray,
    target_bler: float = 0.10,
    sinr_offset_db: float = 0.0,
) -> np.ndarray:
    """
    Compute Block Error Rate using a sigmoid approximation with soft-clamped
    boundaries to avoid hard pile-ups at floor/ceiling values.

    At the CQI switching point, BLER ≈ 10%.  For higher SINR, BLER drops
    exponentially.  For lower SINR, BLER rises towards 100%.

    Model: BLER = 1 / (1 + exp(k * (SINR - SINR_target)))
    where k controls the slope and SINR_target is the operating point.

    In practice, AMC (Adaptive Modulation & Coding) keeps BLER near 10%.
    We model residual BLER after AMC: typically 0.5-5% in normal operation.

    Boundaries use exponential soft-clamp instead of hard np.clip() to
    produce smooth distribution tails (RF-03 remediation):
      - Floor: exponential decay towards 0.01% for high-SINR cells
      - Ceiling: tanh compression above 50% (radio link failure zone)
    """
    # After AMC, the residual BLER is mostly determined by channel estimation
    # errors and fading.  Higher SINR → lower residual BLER.
    # Sigmoid centered at 0 dB SINR with ~10% BLER at that point
    k = 0.3  # Slope factor
    sinr_shifted = sinr_db - sinr_offset_db

    # Residual BLER after AMC: ~1-5% for good conditions, ~10-30% for poor
    bler = 100.0 / (1.0 + np.exp(k * sinr_shifted))

    # Soft-clamp: exponential tails instead of hard clip boundaries.
    # Floor soft-clamp: values below 0.05% decay smoothly towards 0.01%
    floor_val = 0.05
    abs_floor = 0.01  # Minimum possible BLER (measurement noise floor)
    # Clamp exponent argument to avoid overflow when bler is far below floor
    floor_exp_arg = np.clip(-(floor_val - bler) / 0.03, -500.0, 0.0)
    bler = np.where(
        bler < floor_val,
        abs_floor + (floor_val - abs_floor) * np.exp(floor_exp_arg),
        bler,
    )
    # Ceiling soft-clamp: values above 40% compress via tanh towards ~55%
    # In reality, sustained BLER > ~10% triggers RLF / AMC back-off, so
    # the distribution naturally thins out above 40%.
    ceil_knee = 40.0
    ceil_max = 55.0
    bler = np.where(
        bler > ceil_knee,
        ceil_knee + (ceil_max - ceil_knee) * np.tanh((bler - ceil_knee) / 20.0),
        bler,
    )

    return bler


# ---------------------------------------------------------------------------
# Latency model
# ---------------------------------------------------------------------------


def compute_latency_ms(
    rat_type: str,
    prb_utilization_frac: np.ndarray,
    sinr_db: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Compute user-plane latency in milliseconds.

    Base latency varies by RAT:
    - LTE: ~15-20 ms (1 ms TTI + scheduling + processing)
    - NR-NSA: ~8-12 ms (LTE anchor signalling + NR data)
    - NR-SA: ~4-8 ms (shorter TTI, grant-free for URLLC)

    Latency increases with load (PRB utilisation → scheduling delays)
    and decreases with better SINR (fewer HARQ retransmissions).

    RF-07 remediation: latency now uses a hockey-stick model — near-flat
    until ~70% PRB, then exponential increase to 100+ ms at >90% PRB,
    matching real-world buffer-bloat and scheduling queue depth behaviour.
    """
    n = len(prb_utilization_frac)

    if rat_type == "LTE":
        base_ms = 18.0
    elif rat_type == "NR_NSA":
        base_ms = 10.0
    else:  # NR_SA
        base_ms = 5.0

    # Hockey-stick load penalty (RF-07): near-flat below 70%, exponential above.
    # Below 70% PRB: gentle linear increase (~5 ms over the range)
    # Above 70% PRB: exponential ramp — buffer bloat + scheduling queue depth
    linear_part = 5.0 * prb_utilization_frac
    # Exponential knee at 0.70 — grows to ~80-120 ms at 95%+ PRB
    expo_part = np.where(
        prb_utilization_frac > 0.70,
        60.0 * (np.exp(3.5 * (prb_utilization_frac - 0.70)) - 1.0),
        0.0,
    )
    load_penalty = linear_part + expo_part

    # SINR-dependent: poor SINR → more HARQ retransmissions → +latency
    # Each retransmission adds ~8 ms (LTE) or ~4 ms (NR)
    retx_penalty_per = 8.0 if rat_type == "LTE" else 4.0
    # Probability of retransmission increases for low SINR
    retx_prob = np.clip(1.0 / (1.0 + np.exp(0.3 * (sinr_db - 5.0))), 0.0, 0.5)
    retx_penalty = retx_penalty_per * retx_prob

    # Add jitter component
    jitter = rng.exponential(scale=1.5, size=n)

    latency = base_ms + load_penalty + retx_penalty + jitter
    return np.maximum(latency, 1.0)


def compute_jitter_ms(
    latency_ms: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Compute jitter (latency variation) in milliseconds.

    Jitter is typically 5-20% of the mean latency, with some randomness.
    """
    jitter_frac = 0.05 + 0.15 * rng.random(size=len(latency_ms))
    return np.maximum(latency_ms * jitter_frac, 0.1)


# ---------------------------------------------------------------------------
# Packet loss model
# ---------------------------------------------------------------------------


def compute_packet_loss_pct(
    bler_pct: np.ndarray,
    prb_utilization_frac: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Compute packet loss percentage.

    Packet loss comes from:
    1. Residual errors after HARQ (correlated with BLER)
    2. Buffer overflow at high load
    3. Random drops

    Model: PL% = BLER_residual + load_overflow + random_noise
    """
    n = len(bler_pct)

    # Residual after HARQ: ~10% of BLER makes it through to higher layers
    residual = bler_pct * 0.05

    # Buffer overflow: exponential increase above 80% utilisation
    overflow = np.where(
        prb_utilization_frac > 0.8,
        2.0 * np.exp(3.0 * (prb_utilization_frac - 0.8)) - 2.0,
        0.0,
    )

    # Random component
    noise = rng.exponential(scale=0.05, size=n)

    pkt_loss = residual + overflow + noise
    return np.clip(pkt_loss, 0.0, 25.0)


# ---------------------------------------------------------------------------
# RRC / RACH / Handover models
# ---------------------------------------------------------------------------


def compute_rach_metrics(
    active_ues: np.ndarray,
    prb_utilization_frac: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute RACH attempts and success rate.

    RACH attempts scale with active UEs (initial access + handover + re-establishment).
    Success rate degrades under high load due to preamble collisions.

    Returns: (rach_attempts, rach_success_rate_pct)
    """
    n = len(active_ues)

    # Each UE generates ~0.5-1.5 RACH attempts per hour on average
    # (including periodic TA updates, handover, etc.)
    rach_per_ue = 0.8 + 0.4 * rng.random(size=n)
    attempts = active_ues * rach_per_ue

    # Success rate: baseline 97-99%, degrades with load
    base_success = 98.5 - 3.0 * prb_utilization_frac  # drops to ~95.5% at full load
    noise = rng.normal(0, 0.3, size=n)
    success_rate = base_success + noise

    return np.maximum(attempts, 0.0), np.clip(success_rate, 85.0, 99.9)


def compute_rrc_metrics(
    active_ues: np.ndarray,
    sinr_db: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute RRC connection setup attempts and success rate.

    Returns: (rrc_attempts, rrc_success_rate_pct)
    """
    n = len(active_ues)

    # RRC setup attempts: each active UE session triggers RRC setup
    # Plus re-establishments and inter-RAT redirections
    rrc_per_ue = 1.2 + 0.6 * rng.random(size=n)
    attempts = active_ues * rrc_per_ue

    # Success rate depends on SINR (poor signal → more failures)
    sinr_factor = np.clip((sinr_db + 5.0) / 30.0, 0.0, 1.0)  # 0 at -5 dB, 1 at 25 dB
    base_success = 95.0 + 4.5 * sinr_factor
    noise = rng.normal(0, 0.2, size=n)
    success_rate = base_success + noise

    return np.maximum(attempts, 0.0), np.clip(success_rate, 80.0, 99.9)


def compute_handover_metrics(
    active_ues: np.ndarray,
    prb_utilization_frac: np.ndarray,
    deployment_profile: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute handover attempt count and success rate.

    Dense environments have more handovers due to smaller cell radius.
    Success rate depends on load (overloaded target cell → HO failure).

    Returns: (ho_attempts, ho_success_rate_pct)
    """
    n = len(active_ues)

    # Handover rate depends on cell size and mobility
    ho_rate_map = {
        "dense_urban": 0.30,  # High mobility + small cells
        "urban": 0.20,
        "suburban": 0.10,
        "rural": 0.05,
        "deep_rural": 0.02,
        "indoor": 0.05,  # Mostly stationary, some inter-cell
    }
    ho_rate = ho_rate_map.get(deployment_profile, 0.10)

    attempts = active_ues * (ho_rate + 0.05 * rng.random(size=n))

    # Success rate: baseline 96-99%, degrades with target cell load
    base_success = 98.0 - 4.0 * prb_utilization_frac
    noise = rng.normal(0, 0.3, size=n)
    success_rate = base_success + noise

    return np.maximum(attempts, 0.0), np.clip(success_rate, 85.0, 99.9)


# ---------------------------------------------------------------------------
# Paging / CCE models
# ---------------------------------------------------------------------------


def compute_paging_discard_rate(
    active_ues: np.ndarray,
    prb_utilization_frac: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Compute paging discard rate (%).

    Paging congestion occurs when too many UEs are in idle mode
    and the paging channel capacity is exceeded.  Normally < 1%.
    """
    n = len(active_ues)

    # Base discard rate near zero; increases with high UE density
    # and high resource utilisation (less capacity for paging messages)
    base = 0.1 + 0.5 * prb_utilization_frac**2
    # UE count factor: more UEs → more paging → higher discard
    ue_factor = np.clip(active_ues / 200.0, 0.0, 1.0) * 0.5
    noise = rng.exponential(scale=0.05, size=n)

    return np.clip(base + ue_factor + noise, 0.0, 15.0)


def compute_cce_utilization(
    active_ues: np.ndarray,
    prb_utilization_frac: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Compute PDCCH CCE utilisation (%).

    CCE (Control Channel Element) usage is driven by the number of
    scheduling grants (≈ active UEs × scheduling frequency).
    """
    n = len(active_ues)

    # CCE utilisation roughly proportional to # scheduling decisions
    # Each UE needs ~2-8 CCEs per TTI depending on aggregation level
    # Normalise: 100 UEs at AL=4 → ~50% CCE utilisation on 20 MHz carrier
    ue_factor = np.clip(active_ues / 150.0, 0.0, 1.0)

    # Also correlated with PRB utilisation
    combined = 0.6 * ue_factor + 0.4 * prb_utilization_frac
    noise = rng.normal(0, 2.0, size=n)

    return np.clip(combined * 100.0 + noise, 0.0, 100.0)


# ---------------------------------------------------------------------------
# VoLTE / CSFB models
# ---------------------------------------------------------------------------


def compute_volte_erlangs(
    active_ues: np.ndarray,
    rat_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Compute VoLTE traffic in Erlangs.

    VoLTE applies to LTE and NR-NSA (LTE anchor carries voice).
    NR-SA uses VoNR (modelled identically but named differently).
    Typical: ~0.02-0.04 Erlang per active UE (2-4% of time on voice).
    """
    n = len(active_ues)

    # Erlang per UE
    erlang_per_ue = 0.025 + 0.015 * rng.random(size=n)
    erlangs = active_ues * erlang_per_ue

    return np.maximum(erlangs, 0.0)


def compute_csfb_metrics(
    active_ues: np.ndarray,
    rat_type: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute CS Fallback attempts and success rate.

    CSFB is only relevant for LTE cells that still handle 2G/3G voice fallback.
    NR-NSA and NR-SA should have zero CSFB attempts; the success rate is
    returned as NaN (not 100%) because the metric is inapplicable — a rate
    with zero denominator is undefined, not "perfect" (RF-12 remediation,
    per 3GPP TS 32.401 §6.3.2 measurement semantics).

    Returns: (csfb_attempts, csfb_success_rate_pct)
    """
    n = len(active_ues)

    if rat_type in ("NR_NSA", "NR_SA"):
        # NR cells don't do CSFB — attempts=0, success_rate=NaN (undefined)
        return np.zeros(n), np.full(n, np.nan)

    # LTE: some portion of voice calls still fall back to 2G/3G
    # Declining trend: ~1-5% of voice attempts use CSFB
    csfb_rate = 0.005 + 0.005 * rng.random(size=n)  # per UE per hour
    attempts = active_ues * csfb_rate

    # Success rate: typically 95-99%
    success = 97.0 + 2.0 * rng.random(size=n)

    return np.maximum(attempts, 0.0), np.clip(success, 90.0, 100.0)


# ---------------------------------------------------------------------------
# PDCP volume model
# ---------------------------------------------------------------------------


def compute_pdcp_volume_mb(
    throughput_dl_mbps: np.ndarray,
    throughput_ul_mbps: np.ndarray,
    prb_utilization_frac: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute PDCP layer data volume in MB for the reporting interval (1 hour).

    PDCP volume ≈ throughput × utilisation × time_seconds × efficiency

    The throughput is peak achievable; actual volume depends on how much
    of the hour the cell is loaded.
    """
    # Effective throughput accounting for utilisation already embedded in
    # the throughput calc, so volume = throughput_mbps * 3600s / 8 bits_per_byte
    # But throughput is already the *mean* for the hour at given utilisation
    # Volume (MB) = throughput (Mbps) × 3600 (s) / 8 (bits/byte)
    dl_volume = throughput_dl_mbps * 3600.0 / 8.0
    ul_volume = throughput_ul_mbps * 3600.0 / 8.0

    return np.maximum(dl_volume, 0.0), np.maximum(ul_volume, 0.0)


# ---------------------------------------------------------------------------
# Aggregate traffic volume
# ---------------------------------------------------------------------------


def compute_traffic_volume_gb(
    pdcp_dl_mb: np.ndarray,
    pdcp_ul_mb: np.ndarray,
    rng: np.random.Generator,
    app_mix_state: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Total traffic volume in GB (DL + UL) with application-mix noise.

    RF-06 remediation: In a real network the ratio of traffic volume to
    throughput varies from ~0.3× to ~3.0× depending on application mix
    (video = large packets/low overhead vs VoLTE = tiny packets/high
    overhead), bursty vs steady traffic, multi-bearer aggregation, and
    scheduler efficiency.

    We model this as a per-cell log-normal scaling factor with a mean
    of 1.0 and realistic spread (CoV ≈ 25-35%), so that throughput and
    volume are correlated but not deterministically locked.

    NC-03 remediation: The application-mix factor now has AR(1) temporal
    persistence.  A cell serving a university campus has a persistently
    different volume/throughput ratio than a highway cell.  The per-cell
    log-normal factor follows an AR(1) process in log-space:

        log_factor[t] = rho * log_factor[t-1] + innovation

    with rho ≈ 0.70 (matching observed per-cell autocorrelation of
    volume/throughput ratio ρ ≈ 0.6–0.8 in real networks).  The
    stationary variance is preserved at sigma=0.29.

    Parameters
    ----------
    pdcp_dl_mb, pdcp_ul_mb : arrays of PDCP volume in MB
    rng : random generator
    app_mix_state : optional (n_cells,) array — the AR(1) state from the
        previous hour (in log-space).  Pass None for the first hour to
        initialise from the stationary distribution.

    Returns
    -------
    (traffic_volume_gb, updated_app_mix_state) — the volume array and the
    new AR(1) state to pass to the next hour's call.
    """
    base_volume = (pdcp_dl_mb + pdcp_ul_mb) / 1024.0

    n = len(pdcp_dl_mb)
    # Log-normal app-mix factor: median=1.0, spread gives CoV ≈ 30%
    # sigma=0.29 gives exp(sigma^2)-1 ≈ 0.088 → CoV ≈ sqrt(0.088) ≈ 0.30
    app_mix_sigma = 0.29
    app_mix_rho = 0.70  # AR(1) autocorrelation — temporal persistence
    app_mix_innov_std = app_mix_sigma * np.sqrt(1.0 - app_mix_rho**2)

    if app_mix_state is None:
        # First hour: draw from the stationary distribution
        log_factor = rng.normal(0.0, app_mix_sigma, size=n)
    else:
        # Subsequent hours: AR(1) step in log-space
        innovation = rng.normal(0.0, app_mix_innov_std, size=n)
        log_factor = app_mix_rho * app_mix_state + innovation

    app_mix_factor = np.exp(log_factor)
    volume = np.maximum(base_volume * app_mix_factor, 0.0)

    return volume, log_factor


# ---------------------------------------------------------------------------
# RLC retransmission model
# ---------------------------------------------------------------------------


def compute_rlc_retransmission_pct(
    bler_pct: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Compute RLC retransmission rate (%).

    RLC ARQ picks up residual errors after HARQ.  Typically much lower
    than BLER because HARQ catches most errors (~90% of BLER is resolved).
    RLC retransmission rate ≈ 5-15% of residual BLER.
    """
    n = len(bler_pct)
    rlc_factor = 0.08 + 0.06 * rng.random(size=n)
    rlc_retx = bler_pct * rlc_factor
    return np.clip(rlc_retx, 0.0, 20.0)


# ---------------------------------------------------------------------------
# Cell availability model
# ---------------------------------------------------------------------------


def compute_cell_availability(
    n_cells: int,
    rng: np.random.Generator,
    hw_fault_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute cell availability (%).

    Normal operation: 99.5-100%.
    Hardware faults reduce availability significantly.

    Parameters
    ----------
    n_cells : int
    rng : Generator
    hw_fault_mask : optional bool array — True for cells with hardware faults
    """
    # Baseline: mostly 100%, occasional minor dips
    avail = 100.0 - rng.exponential(scale=0.05, size=n_cells)

    if hw_fault_mask is not None:
        # Faulted cells: availability drops to 0-50%
        n_faulted = np.sum(hw_fault_mask)
        if n_faulted > 0:
            avail[hw_fault_mask] = rng.uniform(0.0, 50.0, size=n_faulted)

    return np.clip(avail, 0.0, 100.0)


# ---------------------------------------------------------------------------
# MIMO layer model
# ---------------------------------------------------------------------------


def get_mimo_layers(
    rat_type: str,
    band: str,
    deployment_profile: str,
) -> float:
    """
    Return the number of spatial MIMO layers for throughput calculation.

    - LTE: 2 layers (2×2 MIMO typical), 4 layers for high-band urban
    - NR-NSA: 2-4 layers
    - NR-SA: 2-4 layers (sub-6), 2 layers (mmWave with beamforming)
    """
    if rat_type == "LTE":
        # Most LTE deployments are 2×2 MIMO
        if deployment_profile in ("dense_urban", "urban") and band in ("L2100", "L2300"):
            return 4.0  # 4×4 MIMO in high-capacity urban LTE
        return 2.0

    # NR (NSA or SA)
    if band in ("n257", "n258"):
        # mmWave: beamforming rather than spatial MIMO
        return 2.0
    if deployment_profile in ("dense_urban", "urban"):
        return 4.0  # Massive MIMO for urban NR
    return 2.0


# ---------------------------------------------------------------------------
# Antenna gain model
# ---------------------------------------------------------------------------


def get_antenna_gain_db(
    deployment_profile: str,
    rat_type: str,
    band: str,
) -> float:
    """
    Return typical antenna gain in dBi.

    - Macro (greenfield/rooftop): 15-18 dBi
    - Small cell (streetworks): 5-8 dBi
    - Indoor (DAS/small cell): 2-5 dBi
    - mmWave beamforming: effective 20-25 dBi
    """
    if band in ("n257", "n258"):
        return 23.0  # mmWave beamforming array gain

    gains = {
        "dense_urban": 8.0,  # Small cells / rooftop
        "urban": 15.0,  # Macro / rooftop
        "suburban": 17.0,  # Macro tower
        "rural": 18.0,  # High-gain macro
        "deep_rural": 18.0,
        "indoor": 3.0,  # DAS / small cell
    }
    return gains.get(deployment_profile, 15.0)


# ---------------------------------------------------------------------------
# Complete per-cell physics chain
# ---------------------------------------------------------------------------


@dataclass
class CellPhysicsInput:
    """Input parameters for a batch of cells (one row per cell)."""

    cell_id: np.ndarray  # string array
    rat_type: np.ndarray  # string array: "LTE", "NR_NSA", "NR_SA"
    band: np.ndarray  # string array: "L1800", "n78", etc.
    deployment_profile: np.ndarray  # string array
    freq_mhz: np.ndarray  # float array
    bandwidth_mhz: np.ndarray  # float array
    max_tx_power_dbm: np.ndarray  # float array
    max_prbs: np.ndarray  # int array
    antenna_height_m: np.ndarray  # float array
    isd_m: np.ndarray  # float array (inter-site distance)
    is_nsa_scg_leg: np.ndarray  # bool array


@dataclass
class HourlyConditions:
    """Time-varying conditions for a single hour across all cells."""

    load_factor: np.ndarray  # float 0-1 — traffic load multiplier for this hour
    shadow_fading_db: np.ndarray  # float array — realisation of shadow fading
    interference_delta_db: np.ndarray  # float array — interference variation from baseline
    active_ue_multiplier: np.ndarray  # float array — UE count multiplier


def compute_cell_kpis_vectorised(
    cells: CellPhysicsInput,
    conditions: HourlyConditions,
    rng: np.random.Generator,
    app_mix_state: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    Run the full physics chain for one hourly interval across all cells.

    Returns a dict of KPI name → numpy array (one value per cell).
    The dict includes a special key ``_app_mix_state`` containing the
    AR(1) state vector for the application-mix factor (NC-03), which
    the caller must pass back on the next hour's call.

    This is the hot inner loop — everything is vectorised.
    """
    n = len(cells.cell_id)

    # ── 1. Resolve per-cell physical parameters ──────────────
    antenna_gain_db = np.zeros(n)
    mimo_layers = np.zeros(n)
    deployment_phys_list = []

    for i in range(n):
        dp = cells.deployment_profile[i]
        rt = cells.rat_type[i]
        bd = cells.band[i]
        antenna_gain_db[i] = get_antenna_gain_db(dp, rt, bd)
        mimo_layers[i] = get_mimo_layers(rt, bd, dp)
        deployment_phys_list.append(DEPLOYMENT_PHYSICS.get(dp, DEPLOYMENT_PHYSICS["suburban"]))

    # Extract deployment physics arrays
    env_loss = np.array([p.environment_loss_db for p in deployment_phys_list])
    indoor_loss = np.array([p.indoor_loss_db for p in deployment_phys_list])
    baseline_iot = np.array([p.baseline_iot_db for p in deployment_phys_list])
    typical_ues = np.array([p.typical_ue_busy_hour for p in deployment_phys_list])

    # For indoor cells, zero indoor loss (they're already indoor)
    is_indoor = cells.deployment_profile == "indoor"
    indoor_loss_effective = np.where(is_indoor, 0.0, indoor_loss * 0.3)
    # Only 30% of traffic is indoor for outdoor cells (averaged)

    # ── 2. Path loss (use typical distance = ISD/2 as cell-edge proxy) ──
    typical_distance = cells.isd_m * 0.35  # Average user at ~35% of cell radius

    # Use log-distance model for all (simpler and works across all freqs)
    pl_exponents = np.array([p.pl_exponent for p in deployment_phys_list])
    path_loss = log_distance_path_loss(
        cells.freq_mhz,
        typical_distance,
        pl_exponents.mean(),  # Use mean exponent for vectorised call
    )
    # Apply per-cell exponent correction
    mean_exp = pl_exponents.mean()
    exponent_correction = (pl_exponents - mean_exp) * 10.0 * np.log10(np.maximum(typical_distance, 1.0) / 1.0)
    path_loss += exponent_correction

    # ── 3. RSRP ──────────────────────────────────────────────
    rsrp = compute_rsrp(
        tx_power_dbm=cells.max_tx_power_dbm,
        path_loss_db=path_loss,
        antenna_gain_db=antenna_gain_db,
        shadow_fading_db=conditions.shadow_fading_db,
        indoor_loss_db=indoor_loss_effective,
        environment_loss_db=env_loss,
    )

    # ── 4. Noise power ───────────────────────────────────────
    noise_power = compute_noise_power_dbm(cells.bandwidth_mhz, cells.freq_mhz)

    # ── 5. Interference: baseline IoT + load-dependent + variation ─
    # Load drives neighbour-cell interference
    load = conditions.load_factor
    iot = baseline_iot * (0.3 + 0.7 * load) + conditions.interference_delta_db
    iot = np.clip(iot, 0.0, 30.0)

    # ── 6. SINR ──────────────────────────────────────────────
    sinr = compute_sinr(rsrp, noise_power, iot)

    # ── 7. RSRQ ──────────────────────────────────────────────
    rsrq = compute_rsrq(rsrp, noise_power, iot, cells.max_prbs.astype(np.float64))

    # ── 8. CQI (RAT-dependent) ───────────────────────────────
    is_lte = cells.rat_type == "LTE"
    is_nr = ~is_lte  # NR_NSA and NR_SA both use NR CQI table

    cqi = np.zeros(n)
    if np.any(is_lte):
        cqi[is_lte] = sinr_to_cqi_lte(sinr[is_lte])
    if np.any(is_nr):
        cqi[is_nr] = sinr_to_cqi_nr(sinr[is_nr])

    # Add CQI measurement noise (±0.5)
    cqi_noisy = cqi + rng.normal(0, 0.3, size=n)

    # RF-08 residual: Soft sigmoid compression at CQI boundaries to
    # eliminate the hard pile-up at CQI=0 and CQI=15.  The hard
    # np.clip(cqi_noisy, 0.0, 15.0) forced all sub-zero values to
    # exactly 0.0 and all >15 values to exactly 15.0, creating
    # artificial spikes in the distribution.  Instead we use:
    #   - Floor: sigmoid compression below 0.5 → asymptotic to 0.0
    #   - Ceiling: sigmoid compression above 14.5 → asymptotic to 15.0
    # This produces smooth distribution tails matching the BLER fix (RF-03).
    cqi_floor_knee = 0.5
    cqi_ceil_knee = 14.5
    # Floor soft-clamp: values below 0.5 compress towards 0.0 via sigmoid
    cqi_noisy = np.where(
        cqi_noisy < cqi_floor_knee,
        cqi_floor_knee * (1.0 / (1.0 + np.exp(-5.0 * (cqi_noisy - cqi_floor_knee * 0.5)))),
        cqi_noisy,
    )
    # Ceiling soft-clamp: values above 14.5 compress towards 15.0 via sigmoid
    cqi_noisy = np.where(
        cqi_noisy > cqi_ceil_knee,
        cqi_ceil_knee + (15.0 - cqi_ceil_knee) * np.tanh((cqi_noisy - cqi_ceil_knee) / 0.8),
        cqi_noisy,
    )
    # Final safety clamp (should rarely bind now)
    cqi_noisy = np.clip(cqi_noisy, 0.0, 15.0)

    # ── 9. MCS ───────────────────────────────────────────────
    mcs_dl = np.zeros(n)
    mcs_ul = np.zeros(n)
    if np.any(is_lte):
        mcs_dl[is_lte] = cqi_to_mcs_lte(cqi_noisy[is_lte])
        mcs_ul[is_lte] = np.clip(mcs_dl[is_lte] - 2.0, 0.0, 28.0)  # UL typically lower
    if np.any(is_nr):
        mcs_dl[is_nr] = cqi_to_mcs_nr(cqi_noisy[is_nr])
        mcs_ul[is_nr] = np.clip(mcs_dl[is_nr] - 1.0, 0.0, 28.0)

    # ── 10. Spectral efficiency ──────────────────────────────
    se = np.zeros(n)
    if np.any(is_lte):
        se[is_lte] = cqi_to_spectral_efficiency_lte(cqi_noisy[is_lte])
    if np.any(is_nr):
        se[is_nr] = cqi_to_spectral_efficiency_nr(cqi_noisy[is_nr])

    # ── 11. PRB utilisation ──────────────────────────────────
    # PRB utilisation is load-driven with some randomness
    prb_util_dl = load * (0.7 + 0.3 * rng.random(size=n))
    prb_util_ul = prb_util_dl * (0.4 + 0.2 * rng.random(size=n))  # UL lighter
    prb_util_dl = np.clip(prb_util_dl, 0.01, 1.0)
    prb_util_ul = np.clip(prb_util_ul, 0.01, 0.85)

    # ── 12. Throughput ───────────────────────────────────────
    dl_tp = compute_throughput_mbps(
        se,
        cells.bandwidth_mhz,
        cells.max_prbs.astype(np.float64),
        prb_util_dl,
        mimo_layers,
        overhead_fraction=0.15,
        dl=True,
    )
    ul_tp = compute_throughput_mbps(
        se,
        cells.bandwidth_mhz,
        cells.max_prbs.astype(np.float64),
        prb_util_ul,
        mimo_layers,
        overhead_fraction=0.15,
        dl=False,
    )

    # ── 12a. RF-07: Per-user throughput collapse at high PRB ─
    # In a congested cell, per-user throughput should monotonically
    # decrease because scheduler contention, PDCCH congestion, and
    # queuing delay increase.  We model this by applying a congestion
    # suppression factor to cell throughput above 80% PRB, ensuring
    # the *per-user* throughput doesn't rebound.
    # Clamp the base to non-negative before fractional exponent to avoid
    # RuntimeWarning from NaN ** 1.5 when prb_util_dl slightly exceeds 1.0
    prb_excess = np.clip((prb_util_dl - 0.80) / 0.20, 0.0, 1.0)
    congestion_factor = np.where(
        prb_util_dl > 0.80,
        1.0 - 0.35 * np.power(prb_excess, 1.5),
        1.0,
    )
    congestion_factor = np.clip(congestion_factor, 0.30, 1.0)
    dl_tp *= congestion_factor
    ul_tp *= congestion_factor

    # ── 13. BLER ─────────────────────────────────────────────
    dl_bler = compute_bler(sinr, target_bler=0.10)
    ul_bler = compute_bler(sinr, target_bler=0.10, sinr_offset_db=-3.0)  # UL slightly worse

    # ── 14. Active UEs ───────────────────────────────────────
    active_ue_avg = typical_ues * conditions.active_ue_multiplier * load
    active_ue_avg = np.maximum(active_ue_avg, 0.5)
    # Add per-cell noise
    active_ue_avg += rng.normal(0, active_ue_avg * 0.05, size=n)
    active_ue_avg = np.maximum(active_ue_avg, 0.0)

    # Max UEs: peak within the hour
    active_ue_max = active_ue_avg * (1.2 + 0.3 * rng.random(size=n))

    # NSA SCG legs: no voice/control UEs, only data UEs (~60% of LTE anchor)
    scg_mask = cells.is_nsa_scg_leg
    if np.any(scg_mask):
        active_ue_avg[scg_mask] *= 0.6
        active_ue_max[scg_mask] *= 0.6

    # ── 14a. RF-02: Ghost-load suppression ───────────────────
    # When a cell has CQI ≈ 0 / throughput ≈ 0 due to severe SINR, the
    # load-model KPIs must also be suppressed.  In a real network, UEs
    # experiencing persistently negative SINR would trigger cell
    # reselection (idle) or handover (connected) within seconds — almost
    # no UEs would remain camped.  Without this feedback, ~11.5% of cells
    # at peak hour show "58 UEs + 78% PRB + 0 throughput", which is
    # physically impossible and would mis-train anomaly detectors.
    #
    # Cell-barring factor: 0.02-0.05 for zero-throughput cells (almost all
    # UEs evacuated), scaling smoothly up to 1.0 for cells with reasonable
    # throughput.  Uses a sigmoid on DL throughput with a knee at 1 Mbps.
    ghost_suppression = np.clip(
        1.0 / (1.0 + np.exp(-2.0 * (dl_tp - 1.0))),
        0.02,
        1.0,
    )
    active_ue_avg *= ghost_suppression
    active_ue_max *= ghost_suppression
    prb_util_dl *= ghost_suppression
    prb_util_ul *= ghost_suppression

    # ── 15. Latency / jitter / packet loss ───────────────────
    # We need to handle per-RAT latency; vectorise by doing all then overwriting
    latency_arr = np.zeros(n)
    jitter_arr = np.zeros(n)
    pkt_loss_arr = np.zeros(n)

    for rt_val in ("LTE", "NR_NSA", "NR_SA"):
        mask = cells.rat_type == rt_val
        if not np.any(mask):
            continue
        lat = compute_latency_ms(rt_val, prb_util_dl[mask], sinr[mask], rng)
        latency_arr[mask] = lat
        jitter_arr[mask] = compute_jitter_ms(lat, rng)
        pkt_loss_arr[mask] = compute_packet_loss_pct(dl_bler[mask], prb_util_dl[mask], rng)

    # ── 16. RACH / RRC / Handover ────────────────────────────
    rach_attempts, rach_success = compute_rach_metrics(active_ue_avg, prb_util_dl, rng)
    rrc_attempts, rrc_success = compute_rrc_metrics(active_ue_avg, sinr, rng)

    # Handover: need deployment profile per cell
    ho_attempts = np.zeros(n)
    ho_success = np.zeros(n)
    for dp_val in np.unique(cells.deployment_profile):
        mask = cells.deployment_profile == dp_val
        if not np.any(mask):
            continue
        ha, hs = compute_handover_metrics(active_ue_avg[mask], prb_util_dl[mask], dp_val, rng)
        ho_attempts[mask] = ha
        ho_success[mask] = hs

    # ── 17. Paging / CCE ─────────────────────────────────────
    paging_discard = compute_paging_discard_rate(active_ue_avg, prb_util_dl, rng)
    cce_util = compute_cce_utilization(active_ue_avg, prb_util_dl, rng)

    # ── 18. Cell availability ────────────────────────────────
    cell_avail = compute_cell_availability(n, rng)

    # ── 19. VoLTE / CSFB ────────────────────────────────────
    volte_erlangs = np.zeros(n)
    csfb_attempts = np.zeros(n)
    csfb_success = np.full(n, np.nan)  # RF-12: default NaN (undefined for NR cells)

    for rt_val in ("LTE", "NR_NSA", "NR_SA"):
        mask = cells.rat_type == rt_val
        if not np.any(mask):
            continue
        volte_erlangs[mask] = compute_volte_erlangs(active_ue_avg[mask], rt_val, rng)
        ca, cs = compute_csfb_metrics(active_ue_avg[mask], rt_val, rng)
        csfb_attempts[mask] = ca
        csfb_success[mask] = cs

    # NSA SCG legs don't carry voice
    if np.any(scg_mask):
        volte_erlangs[scg_mask] = 0.0
        csfb_attempts[scg_mask] = 0.0

    # ── 20. PDCP volume / traffic volume ─────────────────────
    pdcp_dl, pdcp_ul = compute_pdcp_volume_mb(dl_tp, ul_tp, prb_util_dl)
    traffic_gb, new_app_mix_state = compute_traffic_volume_gb(
        pdcp_dl,
        pdcp_ul,
        rng,
        app_mix_state=app_mix_state,
    )

    # ── 21. RLC retransmission ───────────────────────────────
    dl_rlc_retx = compute_rlc_retransmission_pct(dl_bler, rng)
    ul_rlc_retx = compute_rlc_retransmission_pct(ul_bler, rng)

    # ── 22. Interference over Thermal (reported) ─────────────
    # IoT is the value used above, but add measurement noise
    iot_reported = iot + rng.normal(0, 0.3, size=n)
    iot_reported = np.clip(iot_reported, -5.0, 30.0)

    # ── Build output dict ────────────────────────────────────
    # Include the NC-03 app-mix AR(1) state so the caller can thread
    # it through to the next hour for temporal persistence.
    result = {
        "rsrp_dbm": rsrp,
        "rsrq_db": rsrq,
        "sinr_db": sinr,
        "cqi_mean": cqi_noisy,
        "mcs_dl": mcs_dl,
        "mcs_ul": mcs_ul,
        "dl_bler_pct": dl_bler,
        "ul_bler_pct": ul_bler,
        "prb_utilization_dl": prb_util_dl * 100.0,
        "prb_utilization_ul": prb_util_ul * 100.0,
        "rach_attempts": rach_attempts,
        "rach_success_rate": rach_success,
        "rrc_setup_attempts": rrc_attempts,
        "rrc_setup_success_rate": rrc_success,
        "dl_throughput_mbps": dl_tp,
        "ul_throughput_mbps": ul_tp,
        "latency_ms": latency_arr,
        "jitter_ms": jitter_arr,
        "packet_loss_pct": pkt_loss_arr,
        "active_ue_avg": active_ue_avg,
        "active_ue_max": active_ue_max,
        "traffic_volume_gb": traffic_gb,
        "dl_rlc_retransmission_pct": dl_rlc_retx,
        "ul_rlc_retransmission_pct": ul_rlc_retx,
        "ho_attempt": ho_attempts,
        "ho_success_rate": ho_success,
        "cell_availability_pct": cell_avail,
        "interference_iot_db": iot_reported,
        "paging_discard_rate": paging_discard,
        "cce_utilization_pct": cce_util,
        "volte_erlangs": volte_erlangs,
        "csfb_attempts": csfb_attempts,
        "csfb_success_rate": csfb_success,
        "pdcp_dl_volume_mb": pdcp_dl,
        "pdcp_ul_volume_mb": pdcp_ul,
    }
    result["_app_mix_state"] = new_app_mix_state
    return result
