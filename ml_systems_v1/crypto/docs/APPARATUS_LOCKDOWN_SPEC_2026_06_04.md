# Apparatus Lock-Down Spec — Implementation Contract (2026-06-04)

> **Status:** SPEC ONLY (no code yet, per user). Consensus-backed (Wave-1 auditor + Wave-2 auditor/validator).
> **Why this is Phase 0:** every prior number was untrustworthy because the harness shipped these holes.
> Until the apparatus is fixed, NO backtest number means anything. These fixes are pure infrastructure —
> they have no capital at stake and are independent of which strategy fork (A or B) we pursue.
> **Binding finding:** the prior validation apparatus produced both false POSITIVES (the +36.8% selection
> leak; the lone PEPE survivor) AND false NEGATIVES (sub-daily nulls under the `load_panel` bug;
> taker-only framing that missed the 15m/30m maker band). Both directions are now suspect → fresh re-test.

## LD-1 — Cost + Fill model (FIX FIRST)
**Current state:** `harness.py:175,654` hardcodes `cost_rt=0.0010` (maker) with **zero p_fill modeling**;
`config/maker_cost_calibration.yaml` (empirical p_fill 0.21–0.40, adverse_selection 0.96–1.00) is **never read**.
**Decision (consensus):** maker fills are adversely-selected ~96–100% of the time → **maker execution is effectively
dead for discovery.** Use TAKER as the working baseline.
**Spec:**
- Add a `FillModel` dataclass: `mode ∈ {taker, maker_pessimistic, maker_calibrated, ideal}`.
  - `taker` (NEW DEFAULT): `cost_rt=0.0024`, `p_fill=1.0`, `adverse_selection=0.001` (deterministic, repeatable).
  - `maker_pessimistic`: `cost_rt=0.0010`, `p_fill=0.30`, `adverse_selection=0.96`.
  - `maker_calibrated`: read p_fill / adverse_selection per bucket from the yaml.
  - `ideal`: old `0.0010 / 1.0 / 0.0` — for reference comparison ONLY.
- Stochastic modes run **N≥500 Monte-Carlo passes**; report **median + p05 + p95** compound, never a single run.
- Move `cost_rt` out of `StrategySpec` into `FillModel`; update the `__contract__`.
- CDAP guard (`_invariants.yaml`): warn if the harness taker default drifts below `0.0024`.
- **Sensitivity band for every candidate:** report results at taker 0.24% AND maker p_fill∈[0.25,0.50] — a candidate that only survives at ideal fills is rejected.

## LD-2 — Look-ahead auto-detection (shift-sensitivity probe)
**Current state:** `_validate_df()` checks only that signal columns EXIST; `Q4_look_ahead_integrity` is a hardcoded
`"VERIFIED"` string. Past-only-ness + full-history-standardization are caller-trusted with no runtime check.
**Spec:** `shift_sensitivity_test(harness, shift_bars=1)` — re-run with each signal/filter column shifted ONE extra
bar into the past; compute `max_abs_delta = max_w |compound_w − compound_shifted_w|`.
- `max_abs_delta > 5pp` ⇒ `LEAK_SUSPECT`; `> 20pp` ⇒ `LEAK_HIGH_CONFIDENCE`.
- Shift `filter_col` independently (separate leak vector).
- Writes the measured value into `Q4` (replacing the hardcoded string); a SHIP-tier claim missing a passing
  `leak_probe` is WARN in CDAP.
- **🔴 CORRECTION (RWYB 2026-06-04, BTC 1d):** the fixed 5pp/20pp thresholds are CADENCE-DEPENDENT and over-trigger on coarse bars — a legit past-only WMA(10,30) on BTC 1d swung +17pp on TRAIN from a 1-bar shift and hit `max_abs_delta=33pp` (FALSE POSITIVE). The probe DOES discriminate (a 1-bar-forward-leaked control gave 85.6pp ≫ the legit 33.2pp), but the verdict must be **RELATIVE, not absolute**: compare the `+1`-shift delta against a same-cadence past-only BASELINE, or use a shift-spectrum `[+1,+2,+3,…]` and flag a **DISCONTINUITY** at the past/future boundary (past-only degrades smoothly; a leak jumps at the boundary). Implemented `shift_sensitivity_test` now tags its verdict `ADVISORY_CADENCE_SENSITIVE`; the shift-spectrum discontinuity verdict is the corrected design to build (supervised). Module: `src/wealth_bot/leak_probe.py` (additive, RWYB-verified to discriminate).
- **Known leak vectors to test explicitly:** `xd_btc_return`/`xd_btc_volatility` are SAME-BAR pass-throughs;
  daily-silver cols (`bd_`/`te_`/`hbr_`/`lob_`/`mv_`) need a **+1-day lag** in live (end-of-day availability).

## LD-3 — DSR / multiple-comparisons gate (currently a NO-OP)
**Current state:** `check_dsr_holm.py` severity is **always "warn"** → exit-2 is unreachable → the gate never
blocks a commit. And `n_trials = len(written JSONs)`, NOT the true grid size → the correction is far too weak.
**This is why the +36.8% selection leak shipped.**
**Spec:**
- `severity="critical"` (exit 2) for any **claimed-ship** candidate that fails Holm; `"warn"` for in-qualification.
- True family-N: a `_sweep_manifest.json` sidecar declaring `n_variants_tested`; `n_trials = max(written, manifest)`;
  WARN if a SHIP claim has no manifest.
- **Family-N MUST include aggregation degrees of freedom** (cells searched × book-composition choices) — the leak
  migrates from cell-selection to book-composition otherwise.

## LD-4 — The firewall: cost-matched random-ENTRY null (THE gate, replacing DSR-only)
**Consensus:** a shuffle/resample-returns null LIES; the **cost-matched random-ENTRY null** is what caught
beta-in-disguise and killed 4 fake "2nd-edge" candidates. Adopt it as the primary acceptance gate.
**Spec:** for every candidate, generate K random-entry books with the SAME #entries/asset, SAME holding policy,
SAME tiered cost (30/60/120bps by $-vol rank); a candidate must beat the random-entry null distribution **AND** be
absolute-positive on held-out (beats-null is necessary, NOT sufficient).
- **🟠 PORT FINDING (2026-06-04):** the proven archived `kill_test.py` is the reference for this PRINCIPLE but is **strategy-bespoke** (hardcoded RSI-bounce logic, pooled K-slot sim, `_btc_regime`/`_win` from archived `pooled_oversold_sweep`) — NOT a drop-in generic harness firewall. "Reuse don't rebuild" hits a wall: the **generic, harness-wrapping** random-entry-null must be built fresh (supervised, trust-critical). Generic design: given a candidate's per-window trade count + holding/exit policy, place that many random entries per window (matched count + matched holding-duration distribution), apply the same tiered cost, run K nulls, compare real compound vs null p95 per window. Keep the kill_test verdict string ("REAL ENTRY-TIMING EDGE" vs "BETA-IN-DISGUISE/DEAD") and the tiered-cost ladder.
- **✅ BUILT + VALIDATED (2026-06-04, STAGED):** `runs/staging/random_entry_null_2026_06_04.py` — generic harness-wrapping version. RWYB on BTC 1d R12 correctly returned BETA-IN-DISGUISE (real does not beat the null on any window). Remaining for supervised integration: a regime-matched-null variant (fairer for regime-gated strats), the tiered-cost ladder, and wiring it as a hard gate in the candidate pipeline.

## LD-5 — Held-out discipline (bear-inclusive)
- The 2022 mega-bear sits in TRAIN; UNSEEN (2026 H1) is an ALT bear; the 2020–21 bull is the hard test for
  trend/DD-avoidance. **Any trend/regime/timing claim MUST be evaluated across a holdout that includes a bear AND
  a bull** — a bull-only or single-month holdout cannot validate it (the breadth satellite's 1-month/15-trade
  UNSEEN is the cautionary example).
- **Pre-register the pooling/weighting rule (equal or vol-parity) BEFORE touching held-out.** Any post-held-out
  re-weighting is a fresh family that re-burns a holdout we don't have.
- Fix the `load_panel` sub-daily→daily floor bug before ANY sub-daily / multi-cadence work (use native loaders).

## Sequencing
LD-1 → LD-3 → LD-4 (cost + a working multiple-comparisons gate + the right null) are the precondition for trusting
any number. LD-2 + LD-5 ride alongside. None of this is strategy code; it is the measurement instrument. Build it
before mining EITHER fork.
