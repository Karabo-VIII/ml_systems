# WAVE 1 SYNTHESIS — AUDIT + ORACLE + UPGRADE PLAN

**Author:** Claude (overseer synthesis). **Date:** 2026-06-28 (VERIFIED start 00:59:45 SAST).
**Window:** OOS-2024 (2024-05-15..2024-12-31), **2025 SEALED** throughout. **Universe:** u100 (n=63 @1d / n=64 @4h).
**Cost:** PROD (taker+slip 0.0028 RT). **Binding:** long-only spot lev=1; rank by WEALTH; never headline the
rebal-inflated DEV compound. **Did NOT git commit.**

Evidence base (all in `runs/engines/AUDIT_ORACLE_UPGRADE/`): `leakcheck.json`, `audit_deployable.json`,
`audit_adx.json`, `audit_nulls.json`, `audit_pipeline.json`, `oracle_gap.json` + `ORACLE_GAP.md`. Each
reproduced from a re-runnable harness (`*_run.py` / `*_verify.py` / `*_falsifier.py` / `oracle_gap_study.py`).

---

## 1. AUDIT VERDICT ON KIRO's V2-V15 WORK

| Component | Verdict | What HOLDS | What's FLAGGED |
|---|---|---|---|
| **Leak-check** (V1 1d-regime → V2 4h-engine alignment) | **LEAKED — confirmed, MODEST** | Mechanism correctly pinned (`ma_slice_engine.run_cell` L194-196: `normalize()→ffill→reindex(norm)` makes an intraday day-D bar read the day-D DAILY-CLOSE regime). Falsifier (`regime.shift(1)`) is correctly built; leak touches **1.7% of OOS bars** (4-6 transition days); edge **SURVIVES** de-leak (all 9 deployable cells stay positive). | NOT the V1.1 catastrophe (that was 88% gone); this is a **uniform ~15% haircut** (mean +0.563 pp/wk, 84.9% retained). **The fix is NOT yet applied to source** — `ma_slice_engine.py` L194-196 still leaks, and all 11 consumers + CURRENT.md inherit it. |
| **Deployable** (capped wealth, `ma_slice_roi.build_book_net`) | **PASS-with-caveat** | Capped headline reproduces within rounding (1d cap15 +1.71 leaked; 4h +1.52). Rebalancing premium correctly NOT in the headline — deployable is a single per-bar NET stream (OOS compound +59% cap15 / +150% EW), the 691K%/9.5x DEV is a *different* engine's `alwayslong_rebal` leg and is correctly excluded. cap-as-dial monotonic confirmed (15<33<50<EW). | The **honest headline is the DE-LEAKED number ~+1.3-1.6%/wk**, not the leaked +1.67/+1.68/+1.60 in CURRENT.md. Breadth: 4h median 30-31 held (≥20 OK); **1d median 19** (just under 20 on a 63-name universe) — minor. |
| **ADX|1d selective** (lb22-28, thr0.42-0.48) | **PASS (PROVISIONAL)** | The program's ONE genuine selective edge. Reproduces: VAL +2.34..+4.32, OOS +0.13..+1.85, deploy 0.32-0.63 (genuinely selective). Survives 500-draw null, the no-gate null (still beats always-long +1.57pp/wk), the directional-gradient probe (monotone in lookback, not a spike), AND de-leak (ADX insulated — unlike the MA band). | **~76% of headline edge is VAL-fit** (median 72-87% VAL→OOS shrink). OOS rests on a thin 230-bar / 32-slice window; best cell profitable in only 28-31% of 7d windows. Headline at OOS **~+0.9-1.2pp/wk**, never VAL +3-4. Drop thr0.48 (null-marginal + de-leak-fragile) and lb<22 (dead). **2025 re-confirm is load-bearing.** |
| **Nulls D83/D84/D85** (config-rotation, within-type conv, dynamic-adaptive) | **PASS-with-concerns** | All three are HONEST refutations, not under-powered false-nulls. D83 reproduces BIT-EXACT (VIDYA frozen +1.572/rolling +1.925/random-P95 +1.774). Controls fair (random-reselect isolates selection skill). D84/D85 mechanistically sound (5 near-clone configs = one opinion; VAL-fit fully inverts OOS). | **D83's "DEPLOY THE FROZEN BAND" is OVERCLAIMED**: the experiment's own random-reselect null mean **beats the 114-config frozen band in 6/12 cells** (EMA|1d random 1.83 vs frozen 1.68). The sound claim is narrower — *selection-by-recent-P&L has no skill*. "Frozen is optimal" doesn't follow; **CONCENTRATION (10 vs 114 configs) is a live untested lever**. |
| **Pipeline** (19/19 causality suite) | **CONCERNS** | 19/19 is real (not fabricated); 4 independently re-verified axes are SOUND (A2 causality byte-identical under future-perturbation; A3 2025 sealed + VAL wall; A4 cost to machine precision; A5 OOS rebal 1.07x). | **The suite has a blind spot at exactly where the leak lives**: A1 toggles only `held.shift_n`; the regime broadcast is byte-identical in causal+cheat paths, so it structurally cannot detect the same-day regime leak. Quantified on the audit's own path: RSI|4h +2.125→+1.700 de-leaked (**20% haircut**). Needs a **20th check: a strict-lag regime-broadcast falsifier**. |

**Net:** Kiro's V2-V15 body of work is **fundamentally HONEST and reproducible**. No fabrication found; every headline
reproduces. The single systemic defect is the **same-day regime-broadcast leak** (the V1.1 bug class, but modest this
time), which is real, un-fixed in source, and silently inflates every deployable/DNA report by ~9-20%.

---

## 2. DEPLOYABLE STATUS

### **HOLDS-WITH-CAVEATS**

The edge is real, survives de-leaking, and stays positive in every deployable cell — but the **published headline is
leaked-inflated and the one-line fix is not yet in source.** Until `regime.shift(1)` lands, every report inherits the
same-day look-ahead.

**Honest current capped %/wk (DE-LEAKED, PROD cost, 15% per-name cap):**
- **1d cap15: +1.56 %/wk** (was leaked +1.71; CURRENT.md says +1.67/+1.68) — retains 91%
- **4h cap15: +1.30 %/wk** (was leaked +1.52; CURRENT.md says +1.60) — retains 85%
- **Deployable band: ~+1.3-1.6 %/wk** (the honest floor), OOS compound +59% over ~7.5mo, Sharpe ~1.6, maxDD ~-25%.

**Required actions to move to HOLDS:** (1) apply `regime.shift(1)` in `ma_slice_engine` (one site fixes all 11
consumers via the imported `btc_regime` path; verify `build_book_net` L40 + `run_cell` L194-196); (2) re-baseline
CURRENT.md + CONSOLIDATED_ENGINE_REGISTRY headlines down ~9-20%; (3) add the pipeline-suite 20th check (regime-broadcast
strict-lag falsifier) so the leak can never silently return.

---

## 3. ORACLE HEADROOM (the gap split)

Long-only oracle = best-achievable in the ALLOWED space (NOT a perfect predictor). Per-slice headline anchors only.

| Level | 1d %/wk | 4h %/wk | slice-prof |
|---|---:|---:|---:|
| **L1 perfect-cash (per-slice)** = absolute LO ceiling | **+12.04** | **+11.84** | 94% |
| L2a best-realizable causal type (EW) | +3.52 | +3.41 | 31-38% |
| **L3 V2 engine (frozen, de-leaked, EW)** | **+3.29** | **+3.41** | 34% / 31% |
| Capped deployable (15%) | **~+1.4** | **~+1.3** | — |

**The split:**
- **RECOVERABLE (V2 → realizable ceiling): +10.76 pp/wk (1d) / +8.89 (4h)** — this is ENTIRELY the **concentration
  dial** (the per-name cap), NOT new alpha. It's a risk knob the user controls.
- **UN-CAPTURABLE (realizable → perfect cash): +10.10 pp/wk (1d) / +31.42 (4h)** — requires directional prediction
  (knowing which names rise), the refuted vein (D17/D55/D77, North-Star 34% hit-rate ceiling). This is structural.

**Gap-DNA (per-bar, components sum exactly):**

| Component | 1d | 4h | Read |
|---|---:|---:|---|
| late entry / early exit | +0.23 | +0.00 | **NON-issue** — frozen SMA band already captures the per-bar move. Do NOT spend budget here. |
| **under-concentration** | **+10.52** | +8.89 | **BIGGEST recoverable lever @1d** = the cap dial taken toward concentration. |
| wrong cash calls | +10.10 | **+31.42** | un-capturable; dominates @4h (more bars = more cash-timing calls). |

**The biggest recoverable lever is CONCENTRATION, not entry/exit timing.** Everything else is the structural LO ceiling.

---

## 4. UPGRADE PLAN — ranked levers to push capture ABOVE the ~1.5%/wk floor (within long-only)

> Floor = ~+1.5%/wk (15% cap, OOS-2024, de-leaked). Every lever below stays inside long-only spot lev=1. The cap dial
> is the proven mechanical knob; the rest is robustness against the cap's single-name DD cost.

### LEVER 1 (HIGHEST EV) — Raise the per-name cap (concentration dial), regime-conditioned
- **Rationale:** Oracle-gap proves under-concentration is the single biggest *recoverable* gap (+8.9-10.5 pp/wk per-bar).
  cap-as-dial is empirically monotonic and reproduced (15→+1.4, 33→+1.6, 50→+2.1, EW→+3.4). This is the ONLY lever the
  gap-DNA says is real and capturable. Condition the cap on the (de-leaked) BTC regime: concentrate harder in
  confirmed-bull, dilute in chop, hold cash in bear — so the higher single-name DD is only taken when the trend pays.
- **Expected gain:** **+1.5%/wk → ~+2.0-2.5%/wk** at cap33-50 (de-leaked), with maxDD held <30% by the regime gate.
  L2b (top-3 momentum concentration) hit Sharpe 3.9 and HALVED maxDD in this trending window (window-specific upside).

### LEVER 2 — Fix the leak + re-baseline (precondition, mandatory)
- **Rationale:** Every number above is only honest post-`regime.shift(1)`. The fix is one line at the `btc_regime`
  broadcast; it costs ~15% of the headline but is non-negotiable for real capital. Audit pipeline needs the 20th check
  so the leak cannot silently return. This is a *truth* lever, not a return lever, but it gates all the others.
- **Expected gain:** -9 to -20% on headlines (a haircut, not a gain) BUT unlocks honest deployment + unblocks Levers 1/3/4.

### LEVER 3 — Add ADX|1d selective as a second, de-correlated mechanism (ensemble)
- **Rationale:** ADX|1d is the program's one genuine SELECTIVE edge (survives de-leak, no-gate null, gradient probe),
  and it deploys only 32-63% of the time — orthogonal to the MA band's always-on long bias. Two mechanisms (always-on
  trend-band + selective ADX) diversify the 34% hit-rate; the D83 finding that random-10 beats frozen-114 in 6/12 cells
  says concentration/diversity of *configs* is itself an untested return lever. Deploy ADX as lb{24,26,28}×thr{0.42,0.45}
  ONLY (drop thr0.48 + lb<22).
- **Expected gain:** OOS ~+0.9-1.2 pp/wk standalone (VAL-fit-discounted); as a diversifier it lifts the *consistency*
  (slice-profit rate) more than the mean — partially attacks the 34%-hit-rate ceiling. **Gated on 2025 re-confirm.**

### LEVER 4 — Concentration-by-config (test frozen-band-at-10 vs at-114)
- **Rationale:** The nulls audit surfaced that the curated 114-config frozen band is BEATEN by a random-10-config book
  in half the cells — i.e. the band is over-diluted. Directly test a frozen, concentrated 10-config band (no rotation,
  so no D73 anti-persistence trap) vs the 114-config vote. This is the *config-space* version of Lever 1's
  *name-space* concentration and is currently untested.
- **Expected gain:** speculative ~+0.2-0.5 pp/wk if concentration-of-configs replicates concentration-of-names; cheap
  to run, high information value (resolves the D83 overclaim). Lower EV than 1/3 because it may just re-expose variance.

### LEVER 5 (LOWEST EV — DO NOT FUND) — entry/exit timing tuning
- **Rationale (negative):** Gap-DNA is explicit: late-entry/early-exit is +0.0-0.2 pp/wk. The frozen SMA band already
  captures essentially all the per-bar move a hindsight-best type would. Spending the upgrade budget on exit/entry
  tuning is mining a dead component. **Listed only to mark it OFF the frontier.**
- **Expected gain:** ~0. Park it.

---

## 5. ANTI-FABRICATION CAVEATS
- All %/wk are **DE-LEAKED** (~9-20% below the previously-published leaked headlines). The capped deployable is
  ~+1.3-1.6%/wk; never quote the leaked +1.67/+1.68/+1.60 as current.
- **Never headline the per-bar L1/L2b compound** (rebalancing premium; +200,115% @1d / 1e8% @4h is meaningless). The
  honest anchors are per-slice L1 (+12%/wk ceiling) and capped L3 (~+1.4%/wk deployable).
- L2a/L2b use IN-WINDOW hindsight (best type, best K) — CEILINGS, not deployable. L2b's DD/Sharpe benefit is
  OOS-2024-window-specific (clean trending window); concentration generally RAISES single-name DD.
- The +10-31 pp/wk "wrong-cash-calls" residual is un-capturable long-only (needs directional prediction = refuted vein).
- ADX OOS rests on one 230-bar window; the 2025 re-confirm is load-bearing, not optional.
