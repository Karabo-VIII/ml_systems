# Deep Oracle-Config Decomposition — the pseudo-formula, across assets AND timeframes (2026-06-10)

**The depth + breadth the shallow 1D pass missed.** Quantifies *what determines the oracle MA config* per window, fits
a **pseudo-formula** `period = f(measurable factors)`, and measures it across **6 assets × 5 timeframes (30 cells)**.
All numbers VERIFIED (RWYB) from `src/strat/oracle_decompose_deep.py`; artifacts + repro in
`runs/mining/oracle_decompose_deep_*.json`. This **corrects** the earlier "config-adaptation is futile" (that was a
shallow 1D, single-var-tercile pass) — the honest result is *small-but-real, quasi-universal*.

## Method (the config is a vector of KNOBS, not "which of 8")
Per window (28 bars, rolling, every cadence): compute a **rich factor set** — efficiency-ratio (trendiness), realized
vol, vol-of-vol, lag-1 autocorr, Hurst, range, dominant-cycle, swing-asymmetry, skew, drift — and the **oracle knobs**
(the MA *period* and the *structure* that maximized capture of the 2-10% moves). Then fit a least-squares
**pseudo-formula** for `log(period)` and a **structure separation**; test **predictive adaptability** (factor_w →
period_{w+1}, held-out) with a **GBM** to size the ML headroom.

## Result 1 — the pseudo-formula EXISTS and is QUASI-UNIVERSAL (the surprise)
Across all 30 (asset, cadence) cells, `base_period` ≈ **12.3 (std 0.9)** and the standardized coefficients are
**sign-consistent**:

| factor | mean coef | sign-consistency | reading |
|---|---|---|---|
| **er** (trendiness) | **+0.27** | **100%** | trendier week → LONGER MA (ride the clean trend) |
| **autocorr** | **−0.23** | **100%** | more persistent → SHORTER MA (catch it sooner) |
| **hurst** | −0.10 | 97% | higher Hurst → shorter MA |
| **range** | −0.21 | 93% | wider range → shorter MA |
| vol | +0.46 | 87% | higher vol → longer MA (steady vol) |
| vov (vol-of-vol) | −0.31 | 83% | erratic vol → shorter MA |
| dom_cycle / skew / drift / swing_asym | ~0 | ~50-60% | **noise — not real factors** |

So the **pseudo-formula** (the "secret sauce", standardized factors):
```
log(period) ≈ log(12.3)  + 0.27·er  − 0.23·autocorr  − 0.21·range  − 0.10·hurst  + 0.46·vol  − 0.31·vov
```
Contrary to the "no universal answer across assets" prior, **the formula *structure* IS universal** across both assets
and timeframes — the same factors drive the period the same direction everywhere, and the central period is ~12.

## Result 2 — but it's WEAK, and adaptation is SMALL (the honest ceiling)
- **R² = 0.15 mean** (range 0.10-0.26). The factors explain only ~15% of the per-window best-period variance; ~85% is
  idiosyncratic noise. The relationship is real but weak — the per-window best period is intrinsically noisy.
- **Predictive adaptability** (factor_w → period_{w+1}, held-out): the edge over a FIXED period is **POSITIVE in 97%
  of cells** but **small — median +0.15pp/window** (range −1.23 to +1.12; BTC 4h is the standout at +1.5pp linear /
  +2.2pp GBM). So adapting the config *does* beat fixed, almost everywhere, but by a fraction of a percent per window.
- **ML headroom is LIMITED:** the GBM beats the linear formula by ~0 on most cells (only BTC/DOGE 4h show +0.7-1.3pp).
  The factor→config relationship is weak *and* mostly linear — ML will add little here.
- **Cadence pattern:** the oracle gap (and the adaptation edge) is largest at **1d** and shrinks finer (1d oracle ~13%
  / edge +0.38pp → 15m oracle ~1.9% / edge +0.09pp). Adaptation matters most where the per-window oracle is richest.

**Verdict on config-adaptation:** it is a **small, consistent, quasi-universal edge — not futile (my earlier call was
wrong) and not a big lever.** The formula is the right mathematical object; its weakness (R²~0.15) is the honest ceiling
of "predict the MA config from market state."

## Result 3 — PARTICIPATION (vol-rotation): looked promising on 6 assets, FAILS robustness on 10 (honest)
Portfolio **vol-rotation** (each week deploy a fixed config across the top-K highest-vol = highest-capturability
names, rolling). On a **6-asset** pool it looked OOS-robust at 4h (+10.6pp over equal-weight, 2nd-half). **But it does
NOT generalize** — on a **10-asset** pool (BTC/ETH/SOL/BNB/XRP/DOGE/ADA/AVAX/LINK/LTC) at 4h, the OOS 2nd-half is
**negative and LOSES to a random-K rotation at every topk** (topk=3: vol_rot −22.8% vs equal_wt −0.3% vs **random
+23.0%**; "vol beats random" 43-47% of weeks, < 50%). So the 6-asset "win" was **pool-specific composition, not a real
vol-timing edge.** Vol-rotation is **not a robust lever** — corrected here after the 10-asset stress test (the kind of
fragility a single pool hides).
- **1d:** also not OOS-robust (ties equal-weight; the full-sample win was high-vol = high-beta concentration).
- **1h+:** cost-dead.

## The honest synthesis (where this leaves us — fully stress-tested)
1. **Config-adaptation is real but small** — a quasi-universal *weak* formula (R²~0.15), +0.15pp/window median edge
   (97% of cells positive), ~0 ML headroom. Worth *using* as a tilt, not worth *betting the strategy on*.
2. **The structure knob is the cleanest universal signal** — autocorr (+0.39, 100% sign-consistent) and Hurst (+0.31,
   100%) separate trend-structure (price>MA) from cross-structure across all 30 cells: persistent/trending weeks want
   price>MA, choppy weeks want cross. (Descriptive; its predictive payoff is bounded by the same factor non-persistence
   as the period knob.)
3. **Participation (vol-rotation) is NOT robust** — promising on 6 assets, fails on 10 (loses to random OOS). Do not
   deploy.
4. **The unifying truth, re-confirmed at depth:** the *descriptive* decomposition is rich and quasi-universal (the
   oracle config has a real mathematical signature), but every *predictive/realizable* lever — config-adaptation,
   vol-participation — is weak or non-robust, because the config-determining state (trendiness) does not persist
   week-to-week while only magnitude (vol) does, and vol does not robustly translate to a tradeable edge.
5. **Adjacent dimensions — now TESTED, all closed the same way:**
   - **Non-MA families** (`oracle_family_compare.py`): the MA family **dominates** — median best-capture +7-11% (1d)
     vs RSI ~0 / MACD ~0-neg / **Bollinger negative** / Donchian ~0; MA wins **72-79%** of windows. Non-MA indicators
     enter late (Donchian) or mean-revert (Bollinger) → capture less. The multitude does **not** span families.
   - **EXIT knob** (hold-length, `oracle_exit_knob.py`): **even less adaptable than the entry-config** — R²(factor →
     best-hold) 0.01-0.13 (~0 at 4h); hold-adaptation **loses** (1d −1.2 to −4.2pp). Best-hold ~11-12 bars; leave fixed.
   - **MA-TYPE:** HMA/low-lag dominate within the MA family (static decompose), but factor-prediction of the type is
     the same weak story.

**FINAL CONVERGENCE.** Across every knob and dimension — entry-period, structure, exit-hold, MA-type, indicator-family,
participation/rotation, and ML(GBM) — the result is identical: the oracle config has a **rich, quasi-universal
DESCRIPTIVE signature** but **no robust PREDICTIVE/adaptive lever**. The mechanism is the same everywhere: the
move-determining state (trendiness/regime) does not persist week-to-week, the predictable state (vol) doesn't translate
robustly, and reasonable configs/exits are nearly interchangeable on the next window. **Weekly config-adaptation, at
depth, across 6 assets × 5 timeframes, cannot beat a sensible fixed config by a robust margin.**

## Result 4 — the binary REGIME-SWITCH (the cleanest adaptive formulation) also fails — and shows WHY
Predicting the exact period is noisy, so the natural fallback is a 2-state switch: use the 100%-consistent regime
signal (autocorr/hurst) to pick a **trend-config (slow MA, p21)** vs a **chop-config (fast MA, p5)** for next week.
Held-out across 6 assets × 3 cadences, it **LOSES to the best single fixed config in ~every cell** (edge −0.04 to
−1.8pp; beats best-fixed only 7-28% of weeks). Two mechanistic reasons, both quantified here:
- **The regime does not persist** — autocorr/hurst separate trend-vs-cross *within* a window (descriptive, 100%
  consistent) but do **not predict next week's** regime, so switching on this-week's signal mistimes next week.
- **The configs are nearly INTERCHANGEABLE next-week** — trend-config(p21) ≈ chop-config(p5) in realized next-week ROI
  almost everywhere (e.g. BTC 4h 0.49% vs 0.47%; BTC 1h 0.92% vs 0.93%; SOL 1h 0.36% vs 0.31%). There is barely a
  config-selection prize to win — so even *perfect* switching would gain little, and imperfect switching loses.

**This is the deepest finding of the decomposition:** the per-window oracle config has a rich, quasi-universal
*descriptive* signature, but **weekly config-adaptation cannot win** because (a) the determining state doesn't persist
and (b) reasonable configs are nearly interchangeable on the next window. The "secret-sauce formula" describes the
*past* config; it does not *predict* the future one. Config is genuinely **not** the lever — now proven at depth, with
numbers, across assets and timeframes, not as a one-line claim.

*Tools: `src/strat/oracle_decompose_deep.py` (--predictive/GBM), `src/strat/oracle_config_adapt.py` (--rotation w/
random+OOS nulls). Provenance: /orc deep-dive 2026-06-10. The pseudo-formula is the bridge to the eventual ML step —
which the headroom analysis says will be incremental, not transformational, on these factors.*
