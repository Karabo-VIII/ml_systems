# Phase 2 — Dynamic / Adaptive Engines: Thorough Plan

**Status: PLAN ONLY (not executed). Authored 2026-06-24.**
This is item 4 of the user's request — planned thoroughly, deliberately NOT built yet.

---

## 0. Why this phase, and what we learned from Phase 1 (static)

Phase 1 built 8 MA-type engines (validated, deployable: WMA/SMA/EMA/VIDYA/KAMA/DEMA
at 4h, +4.11%/wk OOS, 100th-pct random-null) plus 8 non-MA TI families
(RSI/MACD/Bollinger/Stochastic/ADX/Supertrend/Funding/OI).

**Hard finding from Phase 1 (see OOS_VALIDATION_RESULT.md + FORENSIC_AUDIT_RESULT.md):**
- The non-MA families' headline DEV compounds were inflated artifacts (EW-rebalancing
  premium ~9.5x + multi-year compounding + the regime gate). OOS exposed them: each
  beats always-long at only ONE scattered timeframe (noise signature), and several
  deploy 98-100% of the time (= always-long-with-tilt, not selective).
- Single-asset BTC: MACD timing LOST to buy-and-hold (+162% vs +655%).
- **Only the MA family carries a genuine, OOS-surviving, random-null-beating edge.**

**Implication for Phase 2:** the bottleneck is NOT "more indicators." It is that
STATIC parameters can't track a regime-switching market. The dynamic thesis is that
*adapting* an indicator's parameters to recent market state is a genuinely different
mechanism than adding more static indicators. This must be proven, not assumed — and
held to the SAME OOS + random-null bar the MA family cleared.

---

## 1. Core thesis & the precise mechanism

**Static engine (Phase 1):** parameters fixed for all time. `RSI(14) > 60` always.
The 7d-14d lookback only CONFIRMS a fixed signal (decoupled lookback/slice).

**Dynamic engine (Phase 2):** the lookback window COMPUTES the parameters for the
current slice from the signal's own recent behavior. Examples:
- RSI long-threshold WIDENS in high realized-vol, TIGHTENS in low-vol.
- Bollinger band-width (std multiplier) adapts to the recent vol-of-vol.
- MA period SHORTENS in trending regimes, LENGTHENS in chop (the KAMA idea,
  generalized and made explicit/auditable).
- Ensemble vote-threshold adapts: demand MORE agreement in choppy regimes.

**Critical distinction vs the dead-list D73:** D73 killed *weekly config-mapping using
EXTERNAL factors* (mapping calendar/macro state → config). Phase 2 adapts using the
**signal's OWN recent behavior** (endogenous: realized vol, autocorrelation, trend
efficiency of the price itself). This is mechanically different and UNTESTED. The plan
must cite D73 in every design doc so we don't re-walk a dead path.

---

## 2. Adaptation state variables (the "what informs the adaptation")

All computed causally from the lookback window `[t-L, t]` (L = 7-14d in bars):

| State var | Definition (causal) | Drives |
|---|---|---|
| `rv` realized vol | std of bar returns over L | threshold width, band mult, sizing |
| `vov` vol-of-vol | std of rolling-rv | regime-uncertainty → vote threshold |
| `eff` trend efficiency | \|net move\| / Σ\|bar moves\| over L | MA period (short if efficient) |
| `ac1` lag-1 autocorr | autocorr of returns over L | mean-revert vs trend mode switch |
| `dd` recent drawdown | depth of recent local drawdown | de-risk / raise entry bar |

These are deliberately ENDOGENOUS (price/vol derived), never external/macro (D73).

---

## 3. Adaptation rules — 3 concrete candidate engines to build first

### DYN-1: Vol-adaptive RSI ("RSI-VA")
- Base: RSI ensemble (reuse Phase 1 RSI configs as the static control).
- Adaptation: `long_thr = base_thr + k * z(rv)`; in high vol raise the bar (fewer,
  higher-conviction entries), in low vol lower it. `k` is the ONLY new free param.
- Static control: same engine with `k=0` (must beat its own static twin to count).

### DYN-2: Bandwidth-adaptive Bollinger ("BB-VA")
- Base: Bollinger ensemble.
- Adaptation: `std_mult = base_mult * (1 + g * z(vov))`; widen bands when vol-of-vol
  is high (avoid whipsaw), tighten when calm.
- Static control: `g=0`.

### DYN-3: Efficiency-adaptive MA period ("MA-EA")
- Base: the WINNING MA family (WMA/EMA 4h) — start from strength, not weakness.
- Adaptation: `period = base_period * (1 - m * eff)`; shorten the MA when the recent
  window is trending efficiently (catch the move faster), lengthen in chop.
- This is the highest-prior candidate: it augments the ONLY validated family.
- Static control: the frozen WMA|4h V2 engine (the current OOS champion).

Each engine adds at most ONE new free parameter (`k`/`g`/`m`) to keep overfit risk low.

---

## 4. Architecture (reuse, don't reinvent)

Reuse the validated Phase 1 infra verbatim:
- `make_hold_provider` / ensemble-vote pattern (the held-provider interface).
- V1 BTC SMA100 regime gate (Layer 1).
- Decoupled lookback/slice (7d lookback, 7d slice).
- Maker cost 0.0006 RT on turnover.
- Same `_load_asset` walled loader; `load_asset_with_features` for any chimera inputs.

NEW component: an `adapt_fn(lookback_window) -> params` injected into the provider.
The static engine is exactly `adapt_fn = const`. This makes the static-vs-dynamic
comparison a clean A/B with one knob.

---

## 5. The validation gauntlet (NON-NEGOTIABLE — same bar as MA family)

Adaptive engines are MORE prone to overfit (extra free param + adaptation can fit
noise). Therefore the bar is HIGHER, not lower:

1. **DEV build** on TRAIN+VAL (<= 2024-05-15) only. Wall enforced by loaders.
2. **Static-twin A/B:** the dynamic engine MUST beat its own `k=0`/`g=0`/`m=0` static
   twin on DEV per-slice metrics. If it can't beat its own static control, the
   adaptation adds nothing — reject.
3. **Random-null:** same gate, same deploy rate, RANDOM adaptation parameter → the
   real adaptation must beat the random-adaptation distribution (target: ≥95th pct,
   ideally 100th like the MA family).
4. **OOS validation** on 2024-05-15 → 2025-01-01 (the held-out window used in Phase 1).
   Report LINEAR/wk (honest), compound (flagged inflated), edge vs always-long, AND
   the single-asset BTC baseline (no rebalancing premium).
5. **Overfit probe:** sweep the new free param; the OOS performance surface must be
   SMOOTH (a sharp spike at one param value = overfit, reject). Plateau = robust.
6. **UNSEEN remains sealed** (>= 2025-01-01) until a deploy decision is made.

Honest-reporting rules carry over: always linear + compound, always the
single-asset baseline, never trust the EW-rebalanced compound alone.

---

## 6. Build order (V1 → V1.1 per engine, commit as you go)

1. Scaffold `adapt_fn` injection into the ensemble provider + the static-twin harness.
2. **DYN-3 (MA-EA) first** — it augments the only validated family, highest prior of success.
3. DYN-1 (RSI-VA), then DYN-2 (BB-VA).
4. Each: V1 build (DEV) → static-twin A/B → if pass, V1.1 iterate → random-null →
   OOS → overfit probe → commit. If an engine fails the static-twin A/B, STOP that
   engine and document the null (don't polish a loser).
5. Only engines clearing ALL of §5 are candidates for the paper-trader wiring.

---

## 7. Risks / kill-criteria (decide BEFORE building)

- **Overfit via the extra param:** mitigated by static-twin A/B + smooth-surface probe.
- **Adaptation lag:** params computed from a trailing window lag the regime; if DYN
  engines just chase the last regime, they'll underperform. Test with regime-transition
  slices specifically.
- **D73 recurrence:** if any "adaptation" sneaks in an external/calendar factor, it's
  D73 — reject on sight.
- **Kill criterion:** if after DYN-1/2/3 none beats its static twin OOS, conclude that
  adaptation adds no edge at this resolution (a legitimate, documentable result — same
  honesty standard as the Phase 1 non-MA null), and the MA static family stands as the
  deployable system.

---

## 8. Deliverables of Phase 2

- `crypto/src/strat/ti_engine_dynamic.py` (adapt_fn infra + 3 engines)
- `crypto/src/strat/dynamic_oos_test.py` (the §5 gauntlet)
- `crypto/runs/engines/V4_dynamic/` (results + static-twin A/B + random-null + OOS)
- A `.kiro/specs/dynamic-engines/` spec (requirements → design → tasks)
- Updates to CHARTER/CURRENT/KNOWLEDGE_MAP moving dynamic from FUTURE → ACTIVE.

**Not started. Awaiting user go-ahead to begin Phase 2.**
