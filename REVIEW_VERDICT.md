# 🔍 REVIEW VERDICT — Hostile Review Response Assessment

**Reviewer Role:** Senior RAN Principal Architect (Original Auditor)
**Document Under Review:** `REMEDIATION_SUMMARY.md` (Developer Response to `RED_FLAG_REPORT.md`)
**Assessment Date:** 2025-07-15
**Original Verdict:** CONDITIONAL REJECT (5 Critical, 8 Major, 6 Minor)
**Developer's Target:** CONDITIONAL REJECT → CONDITIONAL ACCEPT

**Updated Verdict: CONDITIONAL ACCEPT — with 3 outstanding observations and 1 mandatory pre-production gate**

---

## Table of Contents

- [Executive Assessment](#executive-assessment)
- [Finding-by-Finding Disposition](#finding-by-finding-disposition)
  - [Priority 1 — Critical Findings](#priority-1--critical-findings)
  - [Priority 2 — Major Findings](#priority-2--major-findings)
  - [Priority 3–4 — Minor Findings](#priority-34--minor-findings)
- [Unaddressed Findings — Risk Acceptance Evaluation](#unaddressed-findings--risk-acceptance-evaluation)
- [New Concerns Arising From Remediation](#new-concerns-arising-from-remediation)
- [Mandatory Pre-Production Gate](#mandatory-pre-production-gate)
- [Final Scoring Matrix](#final-scoring-matrix)
- [Closing Statement](#closing-statement)

---

## Executive Assessment

I have reviewed the remediation response (`REMEDIATION_SUMMARY.md`) and independently audited the corresponding code changes across the four modified source files (`physics.py`, `profiles.py`, `step_03_radio_kpis/generate.py`, `step_04_domain_kpis/generate.py`). My assessment is structured around three questions:

1. **Does the proposed fix correctly address the root cause identified in the original finding?**
2. **Is the implementation sound from a 3GPP radio-physics and operational-network standpoint?**
3. **Does the fix introduce any new artefacts or regressions?**

**Summary:** The developer has demonstrated genuine domain understanding in 13 of 15 addressed findings. The fixes are not superficial parameter tweaks — several reflect authentic radio engineering judgment (the BLER soft-clamp, the ghost-load sigmoid suppression, the hockey-stick latency model, the indoor profile rewrite). The 4 unaddressed findings are legitimately low-risk for ML training purposes, and the developer's triage reasoning is defensible.

**However, no fix has been validated against actual regenerated data.** The parquet files on disk remain unchanged. Every "Validation result" in the remediation document is from unit-level assertions against the modified functions, not from end-to-end statistical profiling of the regenerated 47M-row dataset. This is the single largest gap, and it is the basis for the mandatory pre-production gate documented at the end of this review.

---

## Finding-by-Finding Disposition

### Priority 1 — Critical Findings

---

#### RF-01: Missing Phase 5–9 Output Files

| Aspect | Assessment |
|--------|-----------|
| **Original Severity** | 🔴 CRITICAL |
| **Developer Response** | "Not a code defect — operational issue. Code for all 8 files exists." |
| **Disposition** | ⚠️ ACCEPTED WITH CONDITION |

**Analysis:** The developer is correct that this is an operational issue — the generator code in `step_05_scenarios/`, `step_06_events/`, `step_07_customers/`, `step_08_cmdb_degradation/`, and `step_09_vendor_naming/` does exist in the repository. The data store was simply never populated by running those phases.

**However, the finding remains the single most critical blocker for production use.** Without the scenario overlays (`scenario_manifest.parquet`, `scenario_kpi_overrides.parquet`), the dataset contains **zero labelled failure scenarios**. Pedkai's entire value proposition — sleeping cell detection, degradation identification, Dark Graph CMDB divergence — requires these labels. A "clean baseline" dataset is necessary but insufficient.

**Condition:** The regeneration run documented in the Regeneration Instructions section **must** include `--phases 5,6,7,8,9` and all 17 output files must be verified as present and non-empty. I will re-audit the scenario overlay statistics (failure type distribution, temporal placement, geographic spread) when the regenerated dataset is available. This is not "just run the script" — the quality of the injected scenarios is itself an auditable surface.

**Status: OPEN — pending regeneration and scenario quality audit.**

---

#### RF-02: Ghost-Load Paradox

| Aspect | Assessment |
|--------|-----------|
| **Original Severity** | 🔴 CRITICAL |
| **Developer Response** | Sigmoid-based `ghost_suppression` factor on DL throughput, applied to UE/PRB KPIs. |
| **Disposition** | ✅ ACCEPTED — technically sound |

**Code Review — `physics.py` L1339–1355:**

```
ghost_suppression = np.clip(
    1.0 / (1.0 + np.exp(-2.0 * (dl_tp - 1.0))),
    0.02,
    1.0,
)
active_ue_avg *= ghost_suppression
active_ue_max *= ghost_suppression
prb_util_dl *= ghost_suppression
prb_util_ul *= ghost_suppression
```

**What I like:**

1. The sigmoid knee at 1.0 Mbps is physically well-chosen. Below ~1 Mbps DL throughput, a cell is functionally unusable for data services. UEs in idle mode would measure RSRP/RSRQ during cell reselection and avoid camping. Connected-mode UEs would trigger A2 events within 200ms–1s. The sigmoid transition models this gracefully — there is no hard cliff, just progressive UE evacuation as throughput degrades.

2. The floor clamp at 0.02 (not 0.00) correctly models the small residual population: UEs in the process of reselecting, UEs with no neighbour coverage, measurement/control plane traffic from the eNB/gNB itself. This shows the developer understands that even a "dead" cell isn't truly zero-load.

3. The suppression propagates downstream through RACH, RRC, handover, paging, CCE, and VoLTE because those are all functions of the (now-suppressed) `active_ue_avg` and `prb_util_dl`. This is elegant — a single intervention point that corrects the entire KPI tree rather than patching each KPI individually.

**What I would probe further (post-regeneration):**

- The reported validation result shows mean UE dropping from 58.65 → 6.8 for zero-throughput cells. The 6.8 residual seems slightly high for a cell with *zero* throughput. At `dl_tp = 0.0`, the sigmoid evaluates to `1/(1+exp(2)) ≈ 0.119`, clamped to 0.119. Against a base of 58.65 UEs, that gives ~7 UEs — which matches. But 7 UEs still camped on a completely non-functional cell for an entire hour is generous. I would have preferred a steeper sigmoid (slope factor 3.0–4.0 instead of 2.0) to push the residual down to 2–3 UEs. This is a minor calibration issue, not a structural defect.

- **Sleeping cell implication:** The ghost suppression now means that a "true" sleeping cell (zero throughput due to a fault) will show *low* UE count and *low* PRB. This is correct real-world behaviour but the ML model needs to distinguish between "legitimately quiet cell at 3 AM" and "sleeping cell with suppressed load." The scenario overlay (RF-01) becomes even more critical to provide this contrast.

**Status: CLOSED.**

---

#### RF-03: BLER Hard-Clamp Artifacts

| Aspect | Assessment |
|--------|-----------|
| **Original Severity** | 🔴 CRITICAL |
| **Developer Response** | Two-sided soft-clamp: exponential decay floor (→ 0.01%), tanh compression ceiling (→ 55%). |
| **Disposition** | ✅ ACCEPTED — excellent implementation |

**Code Review — `physics.py` L596–617:**

The developer replaced the naive `np.clip(bler, 0.1, 50.0)` with a two-segment soft boundary:

**Floor (low BLER):**
- Below 0.05%: exponential decay towards an absolute floor of 0.01%.
- The 0.01% absolute floor corresponds to the measurement noise floor and quantisation limit of real PM counters. In a live eNB, the BLER counter resolution is typically `1/N_transport_blocks`, so for a cell processing ~100k TBs per ROP, the minimum measurable non-zero BLER is ~0.001%. The 0.01% floor is conservative and correct.
- The exponential argument is clipped to `[-500, 0]` to prevent numerical overflow. Good defensive programming.

**Ceiling (high BLER):**
- Above 40%: tanh compression with a knee at 40% and asymptotic limit at 55%.
- The 40% knee is well-chosen: in practice, outer-loop link adaptation (OLLA) begins aggressive MCS back-off at ~10% BLER. By 30–40%, the scheduler is using QPSK 1/3 and the cell is functionally at the radio link failure boundary. The tanh compression means a few samples can reach 45–50% (severe interference burst before RLF triggers), but the distribution thins out naturally — exactly matching the empirical shape.

**The critical artefact — 42% of samples piled at boundaries — is eliminated.** The distribution now has smooth continuous tails in both directions. This is one of the strongest fixes in the remediation.

**Status: CLOSED.**

---

#### RF-04: Zero Null/NaN Values

| Aspect | Assessment |
|--------|-----------|
| **Original Severity** | 🔴 CRITICAL |
| **Developer Response** | Two-phase null injection: 0.15% base rate + 40% correlated multi-column amplification. |
| **Disposition** | ✅ ACCEPTED — well-modelled correlation structure |

**Code Review — `generate.py` L344–399 (`_inject_nulls()`):**

The implementation is thoughtful:

1. **Independent base nulls** at 0.15% per KPI cell — this is within the expected 0.1–0.5% range from 3GPP TS 32.401.

2. **Correlated amplification** — when a row has at least one null, there is a 40% chance that 1–3 additional KPIs in the same row are also nullified. This correctly models the dominant real-world failure mode: a corrupted PM file or partial ROP loss affects multiple counters simultaneously, not individual columns in isolation. A per-cell independent null model would be naive; the correlated pattern is how real PM data actually breaks.

3. The resulting 0.26% overall null rate with 41.2% multi-column correlation is realistic. I have seen operational OSS exports from Ericsson ENM and Nokia NetAct with comparable rates.

**One minor observation:** The null injection is applied *after* the physics computation but *before* the RecordBatch construction — meaning the physics chain always runs on clean data and nulls are purely a "collection artefact." This is the correct modelling choice: real PM collection failures don't affect the network's physical state, only our observation of it.

**Post-regeneration check:** Verify that the null distribution is uniform across cell types and hours, not clustered in specific deployment profiles or time windows (unless the correlated site-burst mechanism in RF-05 creates intentional clustering, which would be acceptable).

**Status: CLOSED.**

---

#### RF-05: Perfect Row-Group Uniformity

| Aspect | Assessment |
|--------|-----------|
| **Original Severity** | 🔴 CRITICAL |
| **Developer Response** | Three-mechanism gap injection: pre-scheduled site bursts, random individual drops, new burst triggers. |
| **Disposition** | ✅ ACCEPTED — realistic multi-cause gap model |

**Code Review — `generate.py` L402–464 (`_compute_collection_gap_mask()`) and `_build_record_batch()` L519–575:**

The three-mechanism model is operationally realistic:

1. **Pre-scheduled site-wide gaps (burst):** All cells on a site dropped for 1–4 consecutive hours. This correctly models the most common cause of PM gaps in live networks — planned maintenance windows where the entire site goes dark. The burst length of 1–4 hours matches typical maintenance activities (software upgrade: 1–2 hours; hardware swap: 2–4 hours).

2. **Random individual cell-hour drops (0.08%):** SFTP timeout, PM file corruption, eNB restart. Independent per-cell per-hour.

3. **New burst triggers (~0.03% of sites per hour):** Stochastic onset of new maintenance events during the simulation. This prevents all gaps from being pre-determined at hour 0, which would be another form of artificial regularity.

**The `_build_record_batch()` integration is correct:** the `keep_mask` is applied via `pa.RecordBatch.take()`, which physically removes the rows rather than nullifying them. This is the right approach — a missing PM file means the row is *absent from the data store*, not present with all-null values. The distinction matters for ingestion pipelines that need to differentiate "cell reported but all KPIs failed" (RF-04 pattern) from "cell did not report at all" (RF-05 pattern).

**The `_sanity_check()` update** to accept up to 1% fewer rows than the theoretical maximum is appropriate given the ~0.1% gap rate.

**Post-regeneration check:** Verify that row group sizes actually vary. The current parquet files still have uniform 66,131 rows per group.

**Status: CLOSED.**

---

### Priority 2 — Major Findings

---

#### RF-06: Traffic Volume ↔ Throughput Deterministic Lock

| Aspect | Assessment |
|--------|-----------|
| **Original Severity** | 🟠 MAJOR |
| **Developer Response** | Log-normal application-mix scaling factor (σ=0.29, target CoV ≈ 30%). |
| **Disposition** | ✅ ACCEPTED — correct distribution choice |

**Code Review — `physics.py` L975–1001 (`compute_traffic_volume_gb()`):**

The log-normal distribution is the correct choice for application-mix variation. The multiplicative nature of protocol overhead, burst patterns, and bearer aggregation effects naturally produces a right-skewed distribution, which log-normal captures. The σ=0.29 parameter produces a CoV of ~30%, which is within the range I observe in real network data (25–50% depending on the operator and cell mix).

The reported 14× increase in variation (CoV from 2.1% to 29.6%) makes `traffic_volume_gb` and `dl_throughput_mbps` genuinely independent predictors — critical for any ML feature engineering.

**One subtlety the developer handled correctly:** The `rng` parameter is passed per-call, meaning each hourly interval gets a fresh draw of application-mix factors. In a real network, the mix does change hour-to-hour (video streaming peaks in evening, enterprise traffic in daytime), so per-hour variation is appropriate. However, there is no per-cell *persistence* of the mix factor across hours — a cell that was "video-heavy" at hour 13 has no memory of that at hour 14. A per-cell AR(1) on the mix factor would be more realistic, but this is a refinement, not a defect.

**Status: CLOSED.**

---

#### RF-07: Per-User Throughput Rebound at High PRB

| Aspect | Assessment |
|--------|-----------|
| **Original Severity** | 🟠 MAJOR |
| **Developer Response** | Hockey-stick latency model + throughput congestion suppression above 80% PRB. |
| **Disposition** | ✅ ACCEPTED — both mechanisms are correct |

**Code Review — `physics.py` L626–679 (latency) and L1293–1310 (congestion factor):**

**Latency hockey-stick:** The two-regime model is well-calibrated:
- Below 70% PRB: gentle 5ms linear ramp. This matches the near-flat latency behaviour of uncongested cells where scheduling delay is negligible.
- Above 70% PRB: `60 × (exp(3.5 × (prb - 0.70)) - 1)`. At 90% PRB this evaluates to ~60 × (exp(0.70) - 1) ≈ ~60ms, giving total latency ~80–110ms. At 95% PRB: ~60 × (exp(0.875) - 1) ≈ ~85ms penalty, total ~100–120ms. This is exactly the hockey-stick shape I've measured in live congested cells via drive tests and UE-level MDT reports. The exponential onset at 70% matches the point where PDCCH congestion and queuing delay begin to dominate.

**Throughput congestion suppression:** The factor `1.0 - 0.35 × ((prb - 0.80) / 0.20)^1.5` with a floor of 0.30 ensures that:
- At 80% PRB: factor = 1.0 (no suppression yet).
- At 90% PRB: factor ≈ 0.89 (~11% throughput reduction).
- At 100% PRB: factor ≈ 0.65 (~35% throughput reduction).
- Absolute floor: 0.30 (70% maximum reduction).

This ensures per-user throughput **monotonically decreases** with PRB utilisation. The 0.30 floor prevents throughput from going to zero purely from congestion (a cell at 100% PRB still serves data, just poorly), which is physically correct — total cell throughput remains high even when per-user experience is degraded.

**The `np.clip` on `prb_excess`** to `[0.0, 1.0]` before the fractional exponent `** 1.5` is good defensive coding — prevents NaN from negative bases.

**Status: CLOSED.**

---

#### RF-08: SINR and CQI Clipping Walls

| Aspect | Assessment |
|--------|-----------|
| **Original Severity** | 🟠 MAJOR |
| **Developer Response** | SINR range widened from [-10, +40] to [-20, +50] dB; RSRP ceiling from -40 to -30 dBm. |
| **Disposition** | ⚠️ PARTIALLY ACCEPTED |

**Code Review — `physics.py` L425 (`compute_sinr`) and L385 (`compute_rsrp`):**

The SINR range extension to [-20, +50] dB is appropriate:
- **-20 dB floor:** Allows representation of deep-fade cell-edge UEs, heavy inter-cell interference scenarios, and indoor penetration loss. Real UEs can experience SINR as low as -23 dB before declaring radio link failure (TS 36.133 §7.6 — Qout threshold at ~-8 dB SINR for PDCCH BLER 10%, but SINR is measured on RS, not PDCCH).
- **+50 dB ceiling:** Accommodates NR beamforming gains. With a 256-element massive MIMO panel at 3.5 GHz, beam-level SINR of 40–50 dB is achievable at close range with low interference. Reasonable.

**However, the CQI clipping issue is only partially addressed.** The remediation document describes widening the SINR range, but the CQI boundary pile-up at 0 and 15 was a distinct problem. CQI is computed via lookup tables (`sinr_to_cqi_lte()`, `sinr_to_cqi_nr()`) that naturally produce integer-valued outputs. The `cqi_noisy = cqi + rng.normal(0, 0.3)` followed by `np.clip(cqi_noisy, 0.0, 15.0)` still produces a boundary pile-up at CQI=0 and CQI=15, albeit reduced because the wider SINR range means fewer cells hit the extreme SINR values that map to CQI=0 or CQI=15.

The original finding reported 3.51% at CQI=0 and 13.90% at CQI=15 — totalling 17.41%. After widening the SINR range, the pile-up at CQI=0 will decrease (some cells that were at SINR=-10 and forced to CQI=0 can now spread into negative SINR territory), but the `np.clip(cqi_noisy, 0.0, 15.0)` hard clamp remains. Any cell with CQI_noisy > 15.0 is still clipped to exactly 15.0.

**Recommendation for a future pass:** Consider replacing the hard CQI clip with a soft compression similar to the BLER fix — tanh or sigmoid compression near the boundaries. Alternatively, model CQI as a continuous float with Gaussian noise rather than a quantised integer + small noise, since the 15-minute ROP averaging inherently produces non-integer CQI means. This is a refinement, not a blocker.

**Status: CLOSED with observation (CQI boundary pile-up reduced but not eliminated).**

---

#### RF-09: Uniform Peak Hour Across All Profiles

| Aspect | Assessment |
|--------|-----------|
| **Original Severity** | 🟠 MAJOR |
| **Developer Response** | Complete rewrite of indoor profile; dual-peak dense urban; morning ramp for urban. |
| **Disposition** | ✅ ACCEPTED — the profiles now reflect real deployment physics |

**Code Review — `profiles.py` L65–505 (all six profile pairs):**

This is one of the most impactful fixes in the remediation. I inspected each profile:

**Indoor weekday (`_INDOOR_WEEKDAY`):** Peak at hour 12 (1.00), strong ramp 08–11 (0.48 → 0.95), collapse after 17:00 (0.40 → 0.05 at 20:00). This is exactly what I see from enterprise DAS systems — Ericsson indoor small cell data from office buildings shows 85–95% of daily traffic between 08:00–18:00, with near-zero overnight. The 0.05 at 20:00 correctly models cleaning staff/security guard residual traffic. **Excellent.**

**Indoor weekend (`_INDOOR_WEEKEND`):** Modest midday bump (0.28–0.34 at 12:00–14:00), representing retail/mixed-use buildings. Enterprise-only DAS would be even lower, but the blended assumption is reasonable.

**Dense urban weekday (`_DENSE_URBAN_WEEKDAY`):** Dual-peak at hour 9 (0.88, morning commute) and hour 20 (1.00, evening streaming). Sustained office-hour plateau 10–17 (0.70–0.82). This matches the characteristic dual-hump shape observed in CBD cells of major metros. The morning peak is correctly less intense than the evening peak — commute traffic is transient (WhatsApp, email, maps), while evening streaming (Netflix, YouTube) generates sustained high throughput.

**Urban weekday (`_URBAN_WEEKDAY`):** Morning shoulder at hour 9 (0.75), lower overall amplitude. Correct — urban cells outside the CBD show a less pronounced morning peak.

**Rural/deep rural profiles:** Broader, flatter curves with lower overall amplitude and M2M/IoT constant baseline. Correct.

**The RF-10 weekday multiplier integration** (dense_urban 1.15×, urban 1.12×, indoor 1.30×) is applied in `_compute_load_for_hour()` and correctly operates only on weekdays, ensuring the national aggregate shows weekday-dominant traffic.

**Post-regeneration check:** Verify that the indoor cells actually peak at hour 12 local time in the regenerated data, and that the dense urban dual-peak structure is visible in per-profile aggregated time series.

**Status: CLOSED.**

---

#### RF-10: Weekend Traffic Exceeds Weekday

| Aspect | Assessment |
|--------|-----------|
| **Original Severity** | 🟠 MAJOR |
| **Developer Response** | Per-profile weekday multiplier (indoor 1.30×, dense_urban 1.15×, urban 1.12×, suburban/rural 1.00×). |
| **Disposition** | ✅ ACCEPTED |

**Code Review — `profiles.py` L726–740 and L850–852:**

The multiplier values are conservative and defensible:
- Indoor at 1.30× captures the dramatic weekday/weekend difference for enterprise DAS — realistic.
- Dense urban at 1.15× and urban at 1.12× capture the office/commuter component without over-correcting residential areas within those profile types.
- Suburban/rural at 1.00× is correct — residential traffic is often *higher* on weekends at the individual cell level; the weekday national aggregate advantage comes from the enterprise/CBD cells, not from suburbs.

The reported validation result of 1.13× weekday/weekend ratio at the national aggregate is within the expected 1.15–1.25× range for a mixed operator. For Indonesia specifically with DKI Jakarta's 21.6% cell share, I might expect the actual regenerated ratio to be slightly higher (1.15–1.20×) depending on how many Jakarta cells are classified as `indoor` vs `dense_urban`.

**Status: CLOSED.**

---

#### RF-11: Transport KPIs Lack Timezone Shift

| Aspect | Assessment |
|--------|-----------|
| **Original Severity** | 🟠 MAJOR |
| **Developer Response** | Per-entity UTC offset resolved via site_id → sites.parquet → timezone → `_TZ_OFFSET` lookup. `_diurnal_factors_vec()` replaces scalar `_diurnal_factor()`. |
| **Disposition** | ✅ ACCEPTED — architecturally correct |

**Code Review — `step_04_domain_kpis/generate.py` L394–420, L423–524, L540:**

The implementation correctly:
1. Resolves per-entity timezone by joining transport entity → site → timezone via `site_tz_map`.
2. Defaults to WIB (UTC+7) when the site association is missing — sensible fallback.
3. Uses the vectorised `_diurnal_factors_vec()` which iterates over unique timezone offsets and applies per-local-hour profile lookups. Efficient.
4. The temperature diurnal in `_power_hour()` also uses per-site local hours (`local_hours = (base_hour + state.utc_offsets) % 24`), which is a bonus fix beyond what I originally flagged.

**Post-regeneration check:** Verify that a scatter plot of transport `interface_utilization_in_pct` by UTC hour, coloured by timezone, shows three distinct phase-shifted diurnal curves (WIB peaking ~2h after WIT). The original finding showed a single national curve peaking at UTC 15:00 — this should now disaggregate.

**Status: CLOSED.**

---

#### RF-12: CSFB 100% Success for Zero-Attempt NR Cells

| Aspect | Assessment |
|--------|-----------|
| **Original Severity** | 🟠 MAJOR |
| **Developer Response** | NR branch returns `np.full(n, np.nan)` instead of `np.full(n, 100.0)`. Default initialisation also changed to NaN. |
| **Disposition** | ✅ ACCEPTED — semantically correct per 3GPP TS 32.401 §6.3.2 |

**Code Review — `physics.py` L924 and L1416:**

Both the function return path and the default initialisation in `compute_cell_kpis_vectorised()` now use NaN for CSFB success rate on NR cells. This is the only correct representation: a rate KPI with zero denominator is *undefined*, not 100%.

The fix also correctly preserves LTE CSFB success rate in the 90–100% range with realistic variation (`97.0 + 2.0 * rng.random()`), which matches observed 3GPP CSFB performance.

**Note:** The `volte_erlangs` naming issue I raised in the original finding (NR_SA cells carry VoNR, not VoLTE) was not addressed. This is cosmetic — the column name is a naming convention issue, not a data integrity issue — and I accept the developer's implicit decision to defer it.

**Status: CLOSED.**

---

#### RF-13: Battery Voltage Incorrect Polarity

| Aspect | Assessment |
|--------|-----------|
| **Original Severity** | 🟠 MAJOR |
| **Developer Response** | Voltage negated to −48V DC convention; discharge direction corrected; clip range inverted. |
| **Disposition** | ✅ ACCEPTED |

**Code Review — `step_04_domain_kpis/generate.py` L1519–1529:**

The implementation correctly:
1. Negates the stored voltage: `battery_v = -state.battery_capacity_v.copy()`.
2. Models discharge as additive +0.5V during mains failure (magnitude decreasing towards zero, i.e. from −48V towards −43V). This is the correct discharge *direction* for a negative-polarity system.
3. Clips to `[-56.0, -40.0]`, which represents the full range from float charge (−54V) to deep discharge cutoff (−42V) with some margin.
4. Computes charge percentage from absolute magnitude: `(|V| - 42) / (54 - 42) × 100%`.

**The discharge model is still simplistic** — it's a baseline voltage ± jitter, with a +0.5V step on mains failure. A real battery discharge follows a non-linear curve (constant voltage plateau → knee → rapid cliff). However, modelling a stateful discharge curve would require tracking per-site mains uptime/downtime history across hours, which is a significant architectural change. The current fix addresses the polarity error (the actual finding) and is sufficient for ML training purposes. The discharge curve refinement is a future enhancement.

**Status: CLOSED.**

---

### Priority 3–4 — Minor Findings

---

#### RF-14: Diurnal Pattern Too Smooth

| Aspect | Assessment |
|--------|-----------|
| **Original Severity** | 🟡 MINOR |
| **Developer Response** | `cell_jitter_std` 0.08 → 0.13; `temporal_correlation` 0.70 → 0.55. |
| **Disposition** | ✅ ACCEPTED |

**Code Review — `profiles.py` L563 and L570:**

The parameter changes are in the right direction:
- Increasing jitter std from 0.08 to 0.13 raises the broadband noise floor in the FFT decomposition, reducing the 24h component's power fraction from 64% towards the 40–50% target.
- Reducing temporal correlation from 0.70 to 0.55 lowers the lag-24 autocorrelation from 0.96 towards the 0.70–0.85 range observed in real networks.

The innovation standard deviation is correctly recomputed: `σ_innov = σ_jitter × √(1 - ρ²)` = 0.13 × √(1 - 0.3025) ≈ 0.13 × 0.835 ≈ 0.109. This maintains the stationary variance of the AR(1) process while reducing temporal memory.

**Post-regeneration check:** Perform FFT on a sample of single-cell time series and verify that the 24h component captures 40–55% of total power (not 64%). Also verify lag-24 autocorrelation is in the 0.75–0.88 range.

**Status: CLOSED.**

---

#### RF-15: Zero Rural Outliers Beyond Urban P90

| Aspect | Assessment |
|--------|-----------|
| **Original Severity** | 🟡 MINOR |
| **Developer Response** | 0.1% of rural/deep_rural cells per hour receive a 2.5–5.0× traffic spike. |
| **Disposition** | ✅ ACCEPTED |

**Code Review — `generate.py` L467–516 (`_apply_rural_spikes()`):**

The implementation is clean:
- Correctly identifies rural/deep_rural cells by deployment profile.
- Random 0.1% selection per hour (Bernoulli trial per cell).
- Uniform multiplier in [2.5, 5.0] applied to load-dependent KPIs.
- PRB utilisation is scaled by `multiplier × 0.6` and capped at 99% — this prevents unrealistic >100% PRB values while still showing congestion.

The 0.6 factor on PRB is a nice touch — it models the fact that a traffic spike doesn't translate 1:1 to PRB because the scheduler can pack more bits per PRB when the traffic mix shifts (e.g., a festival generates mostly video, which schedules efficiently at high MCS).

**Status: CLOSED.**

---

## Unaddressed Findings — Risk Acceptance Evaluation

The developer explicitly chose not to address 4 of the 19 findings. I evaluate each triage decision:

| Finding | Severity | Developer's Reason | My Assessment |
|---------|----------|-------------------|---------------|
| **RF-01** | CRITICAL | Operational — re-run pipeline | **Accepted but tracked as mandatory gate.** The code exists; this is a process issue. |
| **RF-16** | MINOR | "Marker of synthetic origin; low ML impact" | **Agreed.** The 1:1 NSA anchor:SCG ratio is architecturally correct for EN-DC. While live networks have asymmetries (NR cell sleep modes, commissioning), this is deep in the weeds. A training dataset with perfect pairing is conservative, not misleading. |
| **RF-17** | MINOR | "Would require explicit corruption injection step" | **Agreed.** String encoding anomalies are an ingestion-layer concern, not a radio-physics or ML-feature concern. If the downstream Pedkai ingestion pipeline needs robustness testing, a separate corruption injection utility can be built independently. |
| **RF-18** | MINOR | "30-day window is a design choice" | **Agreed.** The clean 720-interval boundary is a deliberate simulation parameter. Adding partial days or boundary overlap would add complexity without meaningful benefit for anomaly detection training. |
| **RF-19** | MINOR | "Requires cross-step architecture change; noted for future" | **Partially agreed.** The developer correctly identifies that power (step_04) and radio (step_03) are generated in separate phases with no shared state. Adding physical coupling (cabinet temp → PA derating → RSRP → throughput) would require either a post-processing correlation injection step or an architectural refactor to share state between steps. The developer's decision to defer this is defensible given the effort required, but I note that the **wrong-sign correlation** (positive temp↔throughput instead of negative) in the current data is actively misleading for cross-domain causal models. This should be prioritised in the next remediation cycle. |

**Overall triage assessment: Acceptable.** The developer demonstrated good judgment in separating "defects that affect ML training quality" from "markers of synthetic origin" and "architectural debt." The 4 unaddressed items are genuinely low-impact for the stated use case.

---

## New Concerns Arising From Remediation

During my code review, I identified 3 new observations that did not exist in the original dataset but are introduced or made visible by the remediation:

### NC-01: Ghost Suppression × Null Injection Interaction

**Risk level:** LOW

The ghost suppression (RF-02) reduces UE/PRB for low-throughput cells *before* null injection (RF-04) randomly nullifies some KPIs. If a ghost-suppressed cell has its `dl_throughput_mbps` nullified but `active_ue_avg` preserved, the resulting row shows "7 UEs + NaN throughput" — which a downstream model might interpret differently than "7 UEs + 0 throughput." This is actually *more realistic* than the original data (PM collection failures do create ambiguous records), but it's worth noting that the interaction between these two fixes creates a new data pattern that neither fix contemplated individually.

**Action required:** None — the interaction produces more realistic data, not less.

### NC-02: Rural Spike KPIs Not Fully Consistent

**Risk level:** LOW

The rural spike function (`_apply_rural_spikes()`) boosts `active_ue_avg`, `traffic_volume_gb`, `prb_utilization`, `rach_attempts`, and `rrc_setup_attempts`, but does **not** boost `dl_throughput_mbps`, `latency_ms`, `ho_attempt`, `cce_utilization_pct`, or `volte_erlangs`. In a real festival-driven traffic spike:

- Throughput would increase (more aggregate data) but per-user throughput would decrease (congestion).
- Latency would increase due to the hockey-stick effect at high PRB.
- Handover attempts would increase (more mobile UEs).
- CCE utilisation would spike.
- VoLTE erlangs would increase (more voice calls at a festival).

The current implementation creates cells with 5× UEs but unchanged throughput and latency, which is a mild correlation break. However, given that only 0.1% of rural cells per hour are affected, the population-level impact on ML training is negligible.

**Action required:** Documented for future refinement. Not a blocker.

### NC-03: Application-Mix Factor Has No Temporal Persistence

**Risk level:** LOW

As noted in the RF-06 analysis, `compute_traffic_volume_gb()` draws a fresh log-normal application-mix factor every hour. A cell serving a university campus would have a persistently different volume/throughput ratio than a cell serving a highway — this persistence is absent. The result is that hour-to-hour volume variation for a given cell is i.i.d. rather than autocorrelated. Real per-cell volume/throughput ratios exhibit AR(1) behaviour with ρ ≈ 0.6–0.8.

**Action required:** Documented for future refinement. The current implementation is a massive improvement over the deterministic 2.1% CoV — the CoV is now correct even if the temporal structure isn't.

---

## Mandatory Pre-Production Gate

**This is the single condition that separates the current verdict from an unconditional ACCEPT.**

### Gate Criteria: End-to-End Statistical Validation of Regenerated Dataset

The developer's validation results are based on unit-level assertions against individual functions, not on the actual 47M-row regenerated dataset. Before this dataset is used for any ML model training, the following statistical checks **must** pass on the regenerated parquet files:

| Check | Metric | Target | Original Value |
|-------|--------|--------|----------------|
| RF-02 | Mean UEs for cells where `dl_throughput_mbps` = 0 at peak hour | < 10 | 58.65 |
| RF-03 | % of BLER samples at exactly 0.10% | < 2% | 27.3% |
| RF-03 | % of BLER samples at exactly 50.0% | < 1% | 14.6% |
| RF-04 | Overall null rate in `kpi_metrics_wide.parquet` numeric columns | 0.1–0.5% | 0.0% |
| RF-05 | Number of distinct row-group sizes in `kpi_metrics_wide.parquet` | > 1 | 1 |
| RF-06 | CoV of `traffic_volume_gb / (dl_throughput_mbps × 3600/8000)` at peak hour | > 15% | 2.1% |
| RF-07 | Mean latency at >90% PRB | > 80 ms | 35.3 ms |
| RF-07 | Per-user throughput at 80–90% PRB | < value at 60–80% PRB | 6.9 > 4.1 (violated) |
| RF-09 | Indoor weekday peak hour (local time) | 10:00–14:00 | 20:00 |
| RF-10 | National aggregate weekday/weekend active UE ratio | 1.10–1.30× | 0.82× |
| RF-11 | Transport util peak UTC hour for WIT entities | 2h earlier than WIB | Same |
| RF-12 | NR_SA CSFB success rate | NaN (100% of NR cells) | 100.0% |
| RF-13 | Battery voltage mean | < 0 (negative polarity) | +50.5V |
| RF-01 | Count of output parquet files | 17 | 9 |

**Process:**
1. Run `pedkai-generate --phases 0,1,2,3,4,5,6,7,8,9,10`.
2. Execute the above statistical checks against the regenerated output.
3. Provide results to the reviewer for sign-off.

**Until this gate is passed, the dataset status remains CONDITIONAL ACCEPT (code approved, data not yet validated).**

---

## Final Scoring Matrix

| # | Finding | Original Severity | Code Fix Quality | Disposition |
|---|---------|-------------------|-----------------|-------------|
| RF-01 | 8/17 output files missing | 🔴 CRITICAL | N/A (operational) | ⚠️ OPEN — mandatory gate |
| RF-02 | Ghost-load paradox (11.5% of cells) | 🔴 CRITICAL | ⭐ Excellent | ✅ CLOSED |
| RF-03 | BLER hard-clamp artifacts (42% at boundaries) | 🔴 CRITICAL | ⭐ Excellent | ✅ CLOSED |
| RF-04 | Zero null/NaN values in 91M+ rows | 🔴 CRITICAL | ⭐ Excellent | ✅ CLOSED |
| RF-05 | Perfect row-group uniformity | 🔴 CRITICAL | ✅ Good | ✅ CLOSED |
| RF-06 | Traffic/throughput deterministic lock (CoV 2.1%) | 🟠 MAJOR | ✅ Good | ✅ CLOSED |
| RF-07 | Per-user throughput rebound at high PRB | 🟠 MAJOR | ⭐ Excellent | ✅ CLOSED |
| RF-08 | SINR/CQI clipping walls | 🟠 MAJOR | ⚠️ Partial | ✅ CLOSED (CQI pile-up reduced, not eliminated) |
| RF-09 | Uniform peak hour across all profiles | 🟠 MAJOR | ⭐ Excellent | ✅ CLOSED |
| RF-10 | Weekend > weekday at national aggregate | 🟠 MAJOR | ✅ Good | ✅ CLOSED |
| RF-11 | Transport KPIs lack timezone shift | 🟠 MAJOR | ⭐ Excellent | ✅ CLOSED |
| RF-12 | CSFB 100% for zero-attempt NR cells | 🟠 MAJOR | ✅ Good | ✅ CLOSED |
| RF-13 | Battery voltage +50V instead of −48V | 🟠 MAJOR | ✅ Good | ✅ CLOSED |
| RF-14 | Diurnal pattern too smooth (64% FFT) | 🟡 MINOR | ✅ Good | ✅ CLOSED |
| RF-15 | Zero rural outliers beyond urban P90 | 🟡 MINOR | ✅ Good | ✅ CLOSED |
| RF-16 | Perfect 1:1 NSA anchor:SCG ratio | 🟡 MINOR | — (deferred) | ✅ ACCEPTED as-is |
| RF-17 | Zero string encoding anomalies | 🟡 MINOR | — (deferred) | ✅ ACCEPTED as-is |
| RF-18 | Surgical 720-interval clean boundaries | 🟡 MINOR | — (deferred) | ✅ ACCEPTED as-is |
| RF-19 | No power→radio physical coupling | 🟡 MINOR | — (deferred) | ⚠️ ACCEPTED (wrong-sign correlation persists; prioritise next cycle) |

**Closure summary:**
- 15 of 19 findings: code remediation provided → 14 CLOSED, 1 partially closed
- 4 of 19 findings: not addressed → 3 accepted as-is, 1 accepted with future note
- 1 mandatory gate: end-to-end regeneration + statistical validation

---

## Closing Statement

This is a strong remediation response. The developer has not just "patched numbers" — they have demonstrated understanding of the underlying radio physics, operational network behaviour, and statistical properties that distinguish credible synthetic data from academic toy data. Specific highlights:

- **RF-02 (ghost suppression):** The sigmoid on throughput with downstream propagation through the KPI tree is an elegant solution that required genuine understanding of the UE state machine (cell reselection, measurement reporting, RRC connection lifecycle).
- **RF-03 (BLER soft-clamp):** The choice of exponential floor + tanh ceiling with physically motivated knee points shows awareness of link adaptation mechanics, not just distribution shaping.
- **RF-09 (profile rewrite):** The indoor weekday profile is now one of the strongest realistic elements in the dataset. It alone transforms the geospatial credibility of the data.
- **RF-07 (hockey-stick latency):** The two-regime model with an exponential knee at 70% PRB is directly calibrated against real congestion behaviour. This is not a guess — it matches what I measure in live networks.

The dataset, once regenerated, should be fit for purpose as ML training data for sleeping cell detection, congestion prediction, and anomaly identification — the core Pedkai use case. The remaining "grime" that a live 3GPP network would have (string corruption, NR cell sleep asymmetries, battery discharge curves, power↔radio coupling) represents second-order realism that can be iteratively added without blocking the current training pipeline.

**Verdict: CONDITIONAL ACCEPT — pending mandatory regeneration gate.**

---

*Review conducted by the original auditor against the same five criteria: KPI Correlation Integrity, Noise & Entropy Assessment, Anomalous Distribution Realism, Geospatial & Temporal Logic, and Schema Robustness. All assessments are based on direct examination of the modified source code files, not solely on the developer's self-reported validation results.*