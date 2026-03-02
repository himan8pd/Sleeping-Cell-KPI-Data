# Telco2 Known Limitations & Telco3 Code Fixes

**Date:** 2025-07-14
**Context:** Expert review (`TELCO2_FINAL_ASSESSMENT.md`) identified 6 new data-probing findings (DF-01 through DF-06) in the regenerated Telco2 dataset. This document records their status in Telco2, explains why they cannot be patched in-place, and summarises the code fixes applied for Telco3 generation.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Why Telco2 Data Cannot Be Patched In-Place](#why-telco2-data-cannot-be-patched-in-place)
- [Telco2 Usability Despite Limitations](#telco2-usability-despite-limitations)
- [Finding-by-Finding Status](#finding-by-finding-status)
  - [DF-01 — BLER Ceiling Pile-Up at 54.9%](#df-01--bler-ceiling-pile-up-at-549)
  - [DF-02 — CQI Distribution Distorted](#df-02--cqi-distribution-distorted)
  - [DF-03 — SINR Hard-Clip Cascade (Root Cause)](#df-03--sinr-hard-clip-cascade-root-cause)
  - [DF-04 — RACH/RRC ≡ UE Coupling Too Tight](#df-04--rachrrc--ue-coupling-too-tight)
  - [DF-05 — Cell Availability Never Below 99%](#df-05--cell-availability-never-below-99)
  - [DF-06 — Mains Failures i.i.d. Per Hour](#df-06--mains-failures-iid-per-hour)
- [Code Changes Summary](#code-changes-summary)
- [Updated Validation Gates](#updated-validation-gates)
- [Telco3 Re-Validation Gate Criteria](#telco3-re-validation-gate-criteria)

---

## Executive Summary

| Finding | Severity | Telco2 Status | Patchable In-Place? | Telco3 Code Fix |
|---------|----------|---------------|---------------------|-----------------|
| DF-01 | 🟠 MAJOR | Known limitation | ❌ No | ✅ AMC-aware BLER model |
| DF-02 | 🟠 MAJOR | Known limitation | ❌ No | ✅ Via DF-03 upstream fix |
| DF-03 | 🟡 MINOR (root cause) | Known limitation | ❌ No | ✅ SINR tanh soft-compression |
| DF-04 | 🟡 MINOR | Known limitation | ❌ No | ✅ Decorrelated RACH/RRC model |
| DF-05 | 🟡 MINOR | Known limitation | ❌ No | ✅ Mixture-distribution availability |
| DF-06 | ℹ️ INFO | Known limitation | ❌ No | ✅ Persistent mains outage tracking |

**Bottom line:** All six findings are addressed in code and will take effect when Telco3 is generated. The Telco2 dataset remains usable for its primary use cases (see below). No data files were modified.

---

## Why Telco2 Data Cannot Be Patched In-Place

The expert's findings all trace back to the physics chain in the generator:

```
SINR → CQI → MCS → Throughput → PDCP → Traffic
  ↓       ↓                        ↓
 BLER   Ghost suppression     Active UEs → RACH/RRC/HO
  ↓                                ↓
Packet loss                 Paging/CCE/VoLTE/CSFB
```

Every KPI downstream of SINR is computed from it. Patching SINR values in the output Parquet files would require recalculating **every downstream KPI** for affected rows — there are 20+ dependent columns per row, and the recalculation requires the same random state, deployment profiles, and hourly conditions that were used during generation. This is functionally equivalent to regenerating the dataset.

Additionally:
- **DF-04 (RACH/RRC):** Fixing the coupling requires adding independent noise components that weren't part of the original generation. Inventing new values would break the internal consistency of the dataset.
- **DF-05 (cell availability):** A cell showing 95% availability but with throughput/UE numbers consistent with 99.9% availability would be an obvious inconsistency to any ML model.
- **DF-06 (mains persistence):** Adding multi-hour outage sequences requires state tracking across hours that doesn't exist in the generated data.

---

## Telco2 Usability Despite Limitations

Per the expert's assessment, the Telco2 dataset is **immediately usable** for:

| Use Case | Status | Notes |
|----------|--------|-------|
| Sleeping cell detection training | ✅ Ready | Scenario signatures are nuanced, ghost-load fix verified, alarm suppression correct |
| Congestion prediction | ✅ Ready | Hockey-stick latency and per-user throughput collapse are well-calibrated |
| Dark Graph CMDB reconciliation | ✅ Ready | Phantom/dark node/edge/attribute divergences correctly generated |
| Transport anomaly detection | ✅ Ready | Timezone-separated diurnal profiles with correct cascade topology |
| **Models consuming BLER as input feature** | ⚠️ Use with caution | 17.93% of BLER samples compressed into [50, 55) band (DF-01) |
| **Models consuming CQI as input feature** | ⚠️ Use with caution | 22.9% in [14,15] and 11.2% in [0,1) (DF-02) |

If your model uses BLER or CQI as primary features, consider:
1. Applying post-hoc binning (e.g., BLER > 30% → "high BLER" category) to reduce sensitivity to the boundary pile-up.
2. Excluding the boundary accumulation bands from feature normalisation statistics.
3. Waiting for Telco3 generation where the upstream fix eliminates these artifacts.

---

## Finding-by-Finding Status

### DF-01 — BLER Ceiling Pile-Up at 54.9%

**Telco2 symptom:** 17.93% of DL BLER samples fall in [50, 55), a 4.97× density spike vs the adjacent band. The tanh soft-clamp with `ceil_max = 55.0` traded the old hard pile-up at exactly 50.0% for a soft-clamp accumulation asymptote at ~54.9%.

**Root cause:** The raw sigmoid `100/(1+exp(0.3*SINR))` produces BLER > 50% for all SINR < 0 dB (15.74% of cells). The tanh ceiling compresses this mass into a narrow band rather than distributing it realistically.

**Telco3 fix applied to:** `src/pedkai_generator/step_03_radio_kpis/physics.py` — `compute_bler()`

**What changed:**
1. **AMC-aware residual BLER model** — Instead of raw sigmoid output, the model now applies an AMC back-off factor that compresses high raw BLER values towards the 10% target. Real networks use OLLA to back off MCS, keeping residual BLER at 10–15% even at negative SINR. The new model: `amc_target + 25 × tanh((raw_bler - amc_target) / 60)` for raw BLER above target.
2. **Lower, wider ceiling** — `ceil_max` reduced from 55% to 35% (the 3GPP RLF threshold), with `ceil_knee` at 20%. This spreads any remaining boundary mass over a wider, more realistic band.
3. **SINR-dependent BLER variance** — Added in `compute_cell_kpis_vectorised()` using the rng. Variance scales from σ=0.3% at high SINR to σ=3.0% at the SINR floor, eliminating the zero-variance artifact where 14k+ samples at SINR = −20 dB all mapped to identical BLER = 54.924%.

---

### DF-02 — CQI Distribution Distorted

**Telco2 symptom:** CQI [14,15] holds 22.90% of samples (4.42× the adjacent bin); CQI [0,1) holds 11.18% (3.79× the adjacent bin). The CQI soft-compression eliminated pile-ups at exactly 0.0 and 15.0, but the underlying SINR pile-ups at the hard-clip boundaries still push disproportionate mass towards the CQI extremes.

**Root cause:** Upstream SINR hard-clip at [−20, +50] dB (DF-03). 2.18% of SINR samples at exactly −20.0 dB all map to CQI ≈ 0; 0.52% at exactly +50.0 dB all map to CQI ≈ 15.

**Telco3 fix:** This is resolved by the DF-03 upstream fix (SINR soft-compression). No additional changes needed to CQI logic — the existing CQI soft-compression works correctly once the input SINR distribution is smooth.

---

### DF-03 — SINR Hard-Clip Cascade (Root Cause)

**Telco2 symptom:** 2.18% of SINR samples at exactly −20.0 dB, 0.52% at exactly +50.0 dB. These delta-function spikes cascade through the entire physics chain: SINR → CQI pile-ups (DF-02) → BLER ceiling accumulation (DF-01) → zero-variance downstream KPIs.

**Root cause:** `compute_sinr()` used `np.clip(sinr_db, -20.0, 50.0)` — a hard clip with no soft-compression, unlike the downstream BLER and CQI which already had soft-clamps.

**Telco3 fix applied to:** `src/pedkai_generator/step_03_radio_kpis/physics.py` — `compute_sinr()`

**What changed:** Replaced hard clip with tanh soft-compression at both boundaries:
- **Soft floor:** Values below −15 dB compress via `−15 − 5 × tanh((−15 − sinr) / 5)`, asymptoting to −20 dB
- **Soft ceiling:** Values above +45 dB compress via `45 + 5 × tanh((sinr − 45) / 5)`, asymptoting to +50 dB
- **Safety guard:** `np.clip(-20, 50)` retained but should rarely bind

This is the **single upstream intervention** the expert recommended. It cascades improvements through both MAJOR findings (DF-01, DF-02) and eliminates the zero-variance problem for boundary populations.

---

### DF-04 — RACH/RRC ≡ UE Coupling Too Tight

**Telco2 symptom:** RACH-UE correlation r=0.9901, RACH/UE CoV=11.5%. RRC-RACH correlation r=0.976. Both KPIs are near-deterministic functions of UE count, contributing no independent information.

**Root cause:** Original RACH model: `active_ues × (0.8 + 0.4 × random)`. Original RRC model: `active_ues × (1.2 + 0.6 × random)`. Both are linear in UE count with narrow uniform noise.

**Telco3 fix applied to:** `src/pedkai_generator/step_03_radio_kpis/physics.py` — `compute_rach_metrics()`, `compute_rrc_metrics()`, and the per-deployment-profile dispatch in `compute_cell_kpis_vectorised()`

**What changed:**

For RACH:
1. **Deployment-dependent base rate** — Dense urban: 1.2, rural: 1.4 (TA retransmissions), indoor: 0.6
2. **Independent "mobility burst" component** — Log-normal burst uncorrelated with UE count, representing handover storms, mass paging, and TA-update clustering. ~20-30% of the UE-correlated component.
3. **Wider per-cell noise** — σ increased from 0.3 to 0.8

For RRC:
1. **Service-mix dependent base rate** — Log-normal distributed (σ=0.35), varying by deployment profile
2. **Independent "re-establishment" component** — Exponential draws driven by SINR quality, not UE count
3. **Wider per-cell noise** — σ increased from 0.2 to 0.5

Both functions now accept `deployment_profile` as a keyword argument, and the orchestrator dispatches per-profile (same loop that already existed for handover metrics).

**Expected impact:** RACH-UE correlation target ~0.80-0.85 (was 0.99), RACH/UE CoV target ~25-35% (was 11.5%).

---

### DF-05 — Cell Availability Never Below 99%

**Telco2 symptom:** Minimum availability 99.076% across 1.97M samples. Zero samples below 99%. Unrealistically healthy for a 30-day simulation of 66k cells.

**Root cause:** `compute_cell_availability()` used `100.0 − exponential(scale=0.05)`, producing values in a very narrow band [~99.35, 100.0].

**Telco3 fix applied to:** `src/pedkai_generator/step_03_radio_kpis/physics.py` — `compute_cell_availability()`

**What changed:** Replaced single-tier exponential with a 4-tier mixture distribution:
1. **Tier 1 (majority):** `100.0 − exponential(scale=0.15)` — healthy baseline
2. **Tier 2 (~2% of cells):** Uniform [95.0, 99.5] — SW upgrades, watchdog restarts
3. **Tier 3 (~0.5% of cells):** Uniform [85.0, 97.0] — HW resets, transmission failovers
4. **Tier 4 (~0.1% of cells):** Uniform [50.0, 85.0] — prolonged issues

The explicit hardware fault overlay (from scenario injection) is retained unchanged.

**Expected impact:** ~2.5% of cell-hours below 99%, with a realistic long tail down to 50%.

---

### DF-06 — Mains Failures i.i.d. Per Hour

**Telco2 symptom:** Each hour's mains failure is an independent Bernoulli draw (~0.05% per site per hour). No temporal persistence — a site can lose mains at hour 5, have it restored at hour 6, and lose it again at hour 7. Battery discharge is a constant +0.5V regardless of outage duration.

**Root cause:** `_power_hour()` had no state tracking between hours for mains outages. Each hour was independent.

**Telco3 fix applied to:** `src/pedkai_generator/step_04_domain_kpis/generate.py` — `_PowerState` (new fields), `_init_power()`, `_power_hour()`

**What changed:**

1. **Persistent outage tracking** — Two new fields in `_PowerState`:
   - `mains_outage_remaining_hours: np.ndarray` (int, 0 = mains OK)
   - `cumulative_discharge_steps: np.ndarray` (int, progressive drain counter)

2. **Multi-hour outage model:**
   - New outage onset rate: unchanged at ~0.05% per site per hour
   - Outage duration: `geometric(p=0.25) + 2`, giving mean ~6 hours, range [2, 16]
   - During ongoing outage, `mains_status` stays 0.0 automatically
   - When outage resolves (remaining hours reaches 0), discharge counter resets

3. **Progressive battery discharge:**
   - Each hour of outage: −0.8V (towards −43V cutoff)
   - Generator present: discharge rate halved (−0.4V/hour) — generator keeps battery partially topped up
   - After 6+ hours without generator, battery approaches exhaustion (−43V)

**Expected impact:** >50% of mains-failure hours are now part of multi-hour outages. Battery voltage shows progressive decline proportional to outage duration.

---

## Code Changes Summary

| File | Function/Class | Finding(s) | Change Description |
|------|---------------|------------|-------------------|
| `step_03_radio_kpis/physics.py` | `compute_sinr()` | DF-03 | Tanh soft-compression at ±boundaries |
| `step_03_radio_kpis/physics.py` | `compute_bler()` | DF-01 | AMC-aware residual BLER model, lower ceiling |
| `step_03_radio_kpis/physics.py` | `compute_cell_kpis_vectorised()` | DF-01 | SINR-dependent BLER variance injection |
| `step_03_radio_kpis/physics.py` | `compute_rach_metrics()` | DF-04 | Deployment-dependent rates + mobility burst |
| `step_03_radio_kpis/physics.py` | `compute_rrc_metrics()` | DF-04 | Service-mix variability + re-establishment component |
| `step_03_radio_kpis/physics.py` | `compute_cell_kpis_vectorised()` | DF-04 | Per-profile RACH/RRC dispatch |
| `step_03_radio_kpis/physics.py` | `compute_cell_availability()` | DF-05 | 4-tier mixture distribution |
| `step_04_domain_kpis/generate.py` | `_PowerState` | DF-06 | New outage persistence fields |
| `step_04_domain_kpis/generate.py` | `_init_power()` | DF-06 | Initialise persistence arrays |
| `step_04_domain_kpis/generate.py` | `_power_hour()` | DF-06 | Multi-hour outage + progressive discharge |
| `validate_gate.py` | 7 new check functions | DF-01–06 | Validation gates for all findings |
| `validate_gate.py` | `check_rf03_bler_ceil()` | DF-01 | Updated to test both old (50%) and new (35%) ceiling |

---

## Updated Validation Gates

Seven new gate checks were added to `validate_gate.py`:

| Check ID | Function | What It Tests | Target |
|----------|----------|---------------|--------|
| DF-01 | `check_df01_bler_ceiling_pileup()` | % of BLER in [50, 55) | < 5% (was 17.93%) |
| DF-02a | `check_df02a_cqi_ceiling_ratio()` | CQI [14,15] / [13,14) ratio | < 2.0× (was 4.42×) |
| DF-02b | `check_df02b_cqi_floor_ratio()` | CQI [0,1) / [1,2) ratio | < 2.0× (was 3.79×) |
| DF-03 | `check_df03_sinr_boundary_spikes()` | % of SINR at exactly −20 or +50 | < 0.5% (was 2.70%) |
| DF-04 | `check_df04_rach_ue_coupling()` | CoV of RACH/UE ratio | > 20% (was 11.5%) |
| DF-05 | `check_df05_cell_availability_tail()` | % of availability < 99% | > 0.5% (was 0.0%) |
| DF-06 | `check_df06_mains_persistence()` | % of fail-hours in multi-hour outages | > 50% (was 0%) |

The existing `check_rf03_bler_ceil()` was also updated to test both the old ceiling at 50.0% and the new ceiling at 35.0% (from the AMC-aware model).

---

## Telco3 Re-Validation Gate Criteria

When Telco3 is generated, the following gates (from the expert's assessment) must pass:

| Check | Metric | Target | Telco2 Value |
|-------|--------|--------|-------------|
| DF-01 | % of BLER samples in [50, 55) | < 5% | 17.93% |
| DF-02a | CQI [14,15] / CQI [13,14) ratio | < 2.0× | 4.42× |
| DF-02b | CQI [0,1) / CQI [1,2) ratio | < 2.0× | 3.79× |
| DF-03 | % of SINR at exactly −20.0 or +50.0 | < 0.5% | 2.70% |
| DF-04 | CoV of RACH/UE ratio | > 20% | 11.5% |
| DF-05 | % of cell-hours with availability < 99% | > 0.5% | 0.0% |
| DF-06 | % of mains-fail hours in multi-hour outages | > 50% | 0% |

Run all gates with:

```bash
python validate_gate.py --data-store /path/to/Telco3
```

All 23 gates (16 original RF + 7 new DF) must pass for unconditional acceptance.

---

*This document was created alongside the code fixes. No Telco2 data files were modified.*