# IRONED MA TREND SYSTEMS -- FULL-CYCLE CONFIRM (the honest next gate)

**Run:** `python -m strat.ironed_fullcycle_confirm` | **JSON:** `runs/strat/ironed_fullcycle_confirm.json`
**Window:** full 2020-01-07 .. 2026-05-28 u10 panel (ChimeraLoader, the standard strat-layer loader).
**Cost:** maker (`MAKER_RT = 0.0006`), causal / lag-1. **Grade:** `src/strat/scorecard.py` canonical splits
(SEL 2018..2025-03-15 / OOS 2025-03-15..12-31 / UNSEEN 2025-12-31..2026-06-01). **UNSEEN read once.**

**Discipline:** PRE-REGISTERED specs used VERBATIM (commit 3822c29). The ONLY things changed vs the 2020
builder are (a) the data WINDOW and (b) the regime-gate breadth-tercile FIT window, moved to a causal SEL
prefix (2020-01-07..2023-01-01) so it does NOT look ahead into OOS/UNSEEN. The family, confirm-K, gate
policy map, vol-target formula and costs are unchanged. No re-tuning on the full window.

Specs (verbatim):
- **1d = FAMILY-ONLY**: `family=True, exit_="none", conf_k=0, gate=False, voltgt=False`
- **4h = FULL-IRONED**: `family=True, exit_="none", conf_k=2, gate=True, voltgt=True`
- **COMBINED**: `{1d, 4h}` equal-weight trend CORE + funding-dispersion CARRY satellite, 70/30 capital, 3x
  satellite leverage cap. (15m fine sleeve is a separate builder -- coarse-only DEPLOY pair here.)

---

## VERDICT (two-sided, honest)

**The de-risk VALUE is REAL but it lives in the TREND-FLIP, not the regime gate; the systems GENERALIZE as
DRAWDOWN-PRESERVING beta sleeves, NOT as alpha. The 2020-bull spec did NOT earn full-cycle, but it did NOT
collapse either -- it preserves capital in the bear/UNSEEN at far lower drawdown than buy-hold. The 4h
"full-ironed" regime gate is the WEAKER of the two: it whipsaws through the 2022 bear and stays half-invested
into the UNSEEN drawdown. The 1d family-only (NO gate) is the cleaner de-risker.** [VERIFIED, this run]

This is the regrade's "regime-gated trend book: UNSEEN preserves (not earns), valid beta-as-product deploy
profile" pattern, CONFIRMED on a fresh bear-inclusive window. Name it honestly: **preservation, not alpha.**

---

## Q3a -- BEAR LEG (2022 bear, 2021-11-10 .. 2022-12-31): does maxDD come in below buy-hold? [VERIFIED]

| system | sys comp | sys maxDD | BH comp | BH maxDD | DD protection | regime-share (bull/neu/bear) | avg book exp |
|--------|---------:|----------:|--------:|---------:|--------------:|------------------------------|-------------:|
| **1d family-only** (no gate) | -40.9% | **-41.7** | -79.4% | -79.2 | **+37.5pp** | 0.087 / 0.401 / 0.512 | **0.171** |
| **4h full-ironed** (gated)   | -61.3% | **-61.1** | -79.0% | -78.9 | **+17.8pp** | 0.286 / 0.211 / 0.503 | 0.297 |

**Answer: YES, both come in below buy-hold -- materially for 1d (+37.5pp), modestly for 4h (+17.8pp).**

CRUCIAL caveat (the load-bearing finding): **for the 1d sleeve the de-risk is NOT the regime gate -- the 1d
spec has `gate=False`.** The protection is the MA family's own signal-flip going flat (fast EMA < slow EMA in
a downtrend): 1d book exposure drops from 0.81 in the 2020-21 bull to 0.17 in the 2022 bear. The "regime gate
de-risk" crease that was asserted only operates on the 4h sleeve.

**The 4h regime gate WHIPSAWS through the bear** -- it does NOT cleanly go flat. Monthly 4h book exposure
through 2022: Nov-21 0.54 -> Dec 0.10 -> Jan 0.24 -> Feb 0.45 -> Mar 0.28 -> Apr 0.41 -> May **0.00** ->
Jun 0.01 -> Jul 0.47 -> **Aug 0.69** -> Sep 0.22 -> Oct 0.16 -> Nov 0.36 -> Dec 0.31. It re-engaged to 0.69
in Aug-2022 (a bear-rally trap) right before the Sep/Nov leg down, and classified 28.6% of the bear as "bull".
A maxDD-permutation null (cost/vol-matched, timing destroyed) puts the realized 4h path's maxDD DEEPER than
96.6% of random orderings (`p_maxdd_shallower = 0.034`) -- the gate's timing actively CONCENTRATED the bear
drawdown rather than protecting it. (For 1d the same null is benign: `p_shallower = 0.712`.)

---

## Q3b -- FULL-CYCLE ROBUSTNESS: block-bootstrap p05 + beats nulls? [VERIFIED]

| stream | SEL comp | OOS comp | UNSEEN comp | full p05 | held-out p05 |
|--------|---------:|---------:|------------:|---------:|-------------:|
| 1d family-only | 3221% | -5.5% | -3.7% | **+216** | **-45.9** |
| 4h full-ironed | 1386% | +13.9% | -19.3% | **+98** | **-52.3** |
| buy-hold 1d    | 5110% | -6.3% | -27.9% | -- | -- |

- **Full-window block-bootstrap p05 > 0 for both** (+216 / +98) -- but this is dominated by the 2020-21 bull
  in the SEL window; it is NOT a clean robustness signal because the bull compounding swamps the tail.
- **Held-out (OOS+UNSEEN) block-bootstrap p05 is NEGATIVE for both** (-45.9 / -52.3). The de-risked,
  bear-inclusive held-out tail bleeds. By the scorecard's own ship gate (`heldout_p05 > 0`), **neither ships.**
- **vs random-entry null:** the compound-beat p-values are ~0.5-0.8 (permuting i.i.d. daily returns leaves
  compound ~invariant -- a weak test, reported for completeness). The DISCRIMINATING null is the maxDD-timing
  one above: 1d passes (shallower than random, p=0.71), **4h FAILS (deeper than random, p=0.034).**

---

## Q3c -- UNSEEN (test-once, read ONCE): positive, or preserve-not-earn? [VERIFIED]

| system | UNSEEN comp | UNSEEN maxDD | avg exposure | buy-hold UNSEEN | profile |
|--------|------------:|-------------:|-------------:|----------------:|---------|
| **1d family-only** | **-3.7%** | **-3.8** | 0.054 | -27.9% (DD -41.2) | **PRESERVE** (went ~flat) |
| **4h full-ironed** | **-19.3%** | -21.1 | 0.480 | -27.5% (DD -41.2) | weak preserve (stayed half-in) |

**Answer: PRESERVE-NOT-EARN, exactly the regrade's regime-gated-trend-book profile.** The 1d sleeve is the
textbook case: UNSEEN -3.7% vs market -27.9%, at -3.8 maxDD vs -41.2 -- it went almost fully flat (avg
exposure 0.054) and preserved capital. This is a VALID deploy profile per the charter (beta-as-product /
preservation), and it is named honestly: it does NOT earn on UNSEEN, it AVOIDS the drawdown.

The 4h sleeve is the weaker half: it stayed ~48% invested into the falling UNSEEN tape and lost -19.3% (only
8pp better than buy-hold, with worse DD protection than the 1d trend-flip). The 4h gate's value did NOT show
on UNSEEN. UNSEEN was read once; not re-touched.

---

## Q3d -- Does the 2020-bull-selected spec GENERALIZE, or was it bull-overfit? [VERIFIED]

**Partially. It did NOT collapse (not a pure bull artifact), but it did NOT earn full-cycle either.** On the
2020 bull the builders reported 1d +49.8% / 4h +40.8% OOS *net*. Full-cycle, the held-out (OOS+UNSEEN)
compound is roughly flat-to-negative (1d OOS -5.5% / UNSEEN -3.7%; 4h OOS +13.9% / UNSEEN -19.3%) and the
held-out p05 is negative. The spec GENERALIZES as a **drawdown-preserving trend sleeve** (the de-risk shows
on the bear leg and on UNSEEN as lower DD than buy-hold) but NOT as a positive-return generator out-of-sample
-- consistent with the standing internal-data directional ceiling. The asserted bull-OOS net does not
reproduce as net alpha full-cycle; the asserted de-risk VALUE does reproduce (1d especially).

---

## Q4 -- Does the per-TF ironed CORE add anything over daily_engine / core_satellite_book CORE? [VERIFIED]

It **CONVERGES to it, not improves on it.** Both are long-only u10 beta with a de-risk overlay:
- `daily_engine` / `core_satellite_book` CORE = vol-target buy-hold + a causal regime exposure-scalar
  (full-cycle ~+1970%, Sharpe ~1.4, maxDD ~-48%, held-out flat-to-negative -- a documented BETA engine).
- The ironed per-TF trend CORE here = MA-family trend-follow (de-risks by flattening), same family of
  outcome: positive in the bull, preserves in the bear, no held-out alpha, negative held-out p05.

The two cross-TF ironed sleeves are **0.69 correlated** (1d vs 4h) -- they are NOT diversifying; they are the
same trend factor at two clocks. The ironed CORE's only structural difference vs the daily_engine CORE is
*mechanism* (trend-flip-to-flat vs vol-target-plus-regime-scalar); on the metrics that matter
(held-out compound, held-out p05) they land in the same place. **No additive edge over the existing CORE.**

---

## COMBINED BOOK (trend CORE 70 / carry satellite 30, 3x sat cap) [VERIFIED]

| stream | SEL comp | OOS comp | UNSEEN comp | full p05 | held-out p05 | ship (scorecard) |
|--------|---------:|---------:|------------:|---------:|-------------:|:----------------:|
| trend CORE {1d,4h} | 2384% | +4.3% | -11.6% | +236 | -46.3 | NO |
| carry satellite    | 36% | +7.9% | **+10.3%** | +47 | **+11.2** | **YES** |
| **COMBINED 70/30** | 755% | +11.8% | **+0.4%** | +192 | -23.2 | NO |

- The carry satellite is **the only ship=True component** (UNSEEN +10.3%, held-out p05 +11.2, beta~0).
- core-satellite correlation ~ **0.008** (genuinely orthogonal -- confirmed full-cycle).
- The 30% carry sleeve pulls the combined UNSEEN from the core's -11.6% up to **+0.4% (break-even)** and
  lifts held-out p05 from -46 to -23 -- a real diversification benefit, but it does NOT make the combined
  book ship (the trend core's negative held-out tail still dominates the 70% capital). **The combined book's
  honest value is "preserve at ~break-even on UNSEEN at lower DD", carried by the satellite, not the trend.**

---

## BOTTOM LINE

1. **De-risk value CONFIRMED for 1d, but it is the TREND-FLIP, not a regime gate** (1d spec has no gate; it
   still cut 2022 maxDD by +37.5pp and held UNSEEN to -3.7% vs market -27.9%).
2. **The 4h regime gate is REFUTED as a clean de-risker** -- it whipsaws through the 2022 bear, fails the
   maxDD-timing null (p=0.034, deeper than 96.6% of random orderings), and stays half-invested into the
   UNSEEN drawdown.
3. **The specs GENERALIZE as preservation sleeves, not alpha** -- held-out p05 negative for both; neither
   ships on the canonical UNSEEN+full-p05 gate.
4. **The ironed CORE converges to the existing daily_engine CORE** (0.69 cross-TF corr, same held-out
   profile) -- no additive edge.
5. **The carry satellite remains the only ship-grade sleeve**; at 70/30 it lifts the combined book to UNSEEN
   break-even -- a valid "preserve-at-lower-DD, modest carry" deploy profile, named honestly.

The honest deploy candidate from this exercise is the **1d family-only trend sleeve as a drawdown-preserving
beta overlay** (NOT the 4h gate), paired with the carry satellite -- and that is functionally the
core_satellite_book the project already has. The 2020-bull "full-ironed 4h" headline was, in its de-risk
claim, partly a bull-window artifact: the gate's bear-protection did not reproduce.
