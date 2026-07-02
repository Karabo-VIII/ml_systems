# TI bots — slice-success spec + implementation (2026-07-02, TRAIN+VAL only, OOS held out)

**Goal:** audit that the TI (MA-family) bots are *profitable when deployed*; redevelop on TRAIN+VAL only; define what
success is, what to strive for, and the timeframe; implement per the canonical TI family. OOS (2024-05-15+) is **never
loaded** during this work (`WALL = VAL_END`), so it stays a genuine future test.

## What success looks like (the definition, operationalized)

A TI bot trades a fixed **7d slice** (enter, hold 7 days, exit/roll). Success on **any 7d slice** =
1. **cash-or-profitable** — the bot either sat in **cash** (flat, no loss) or **deployed and won**; equivalently, the
   *losing-deployed-week rate* → 0; **OR**
2. **delta-profitable** — the bot's average 7d slice **beats the average 7d slice of the market** (EW-u100).

Clause 2 is the achievable primary target; clause 1's "never lose a deployed week" is an ideal (noisy crypto has an
irreducible ~30-40% single-week loss rate on any long position, so a *deployed* week can't be made loss-free without
deploying ~never). **What to strive for:** delta-positive **and** as few losing deployed weeks as the gate allows,
and — the bar this arc keeps failing — the delta must be **DISTRIBUTED** (survive dropping the top slices / positive
across multiple years), not a one-regime burst.

## Timeframe: 1d (one, canonical)

The project already proved it: across 30m/1h/2h/4h/1d, **only 1d is net-positive after cost** (finer TF multiplies
turnover — cost drag 15% at 1d → 84-188% at 30m — with no added timing skill). Structure = **14d-lookback / 7d-slice**
(`held_rolling`: 14d-confirmed entry, 7d time-cap). So: **1d bars, 14d/7d, u100 cross-section, long-only spot.**

## The implementation (per TI, canonically defined)

Each MA TI (SMA/EMA/WMA/HMA/DEMA/TEMA/KAMA/VIDYA) defines the **eligible set** (names where its canonical cross is
long, `held_rolling` over the train/val band). The redeveloped bot adds the two levers the audit showed were missing:
- **WHEN (gate):** deploy only when u100 **breadth** is up (hysteresis 0.5/0.4). Cash otherwise. *(A tighter
  breadth-AND-BTC-trend-efficiency gate lowers the losing-week rate further but costs delta — use it if clause 1 is
  weighted over clause 2.)*
- **WHICH (selection):** among the TI-eligible names, hold the **momentum-14 top-5** (not the whole basket). This is
  the lever that turns delta positive — the whole basket ≈ the market (delta ≈ 0/negative); the top-5 captures the
  movers. inv-vol weighted, cap 33%.

## Results (TRAIN pre-2023-07 / VAL 2023-07→2024-05-15)

**Baseline (breadth-gated, whole basket):** delta **negative** (−0.3 to −0.65% on VAL), losing-deployed-week rate ~20%.
Fails both clauses.

**Redeveloped (breadth gate + momentum top-5):** **all 8 TIs delta-positive on BOTH TRAIN and VAL.**

| TI | full Δ | drop-top3 | **drop-top5** | **no-2021** | bootstrap P(Δ≤0) | verdict |
|---|---|---|---|---|---|---|
| HMA | +0.92% | +0.14% | **−0.20%** | **+0.14%** | 0.13 | CONCENTRATED (2021+VAL bull) |
| TEMA | +0.89% | +0.15% | **−0.16%** | **+0.14%** | 0.13 | CONCENTRATED |
| SMA | +0.56% | −0.02% | −0.30% | +0.09% | — | CONCENTRATED (dies drop-top3) |

All 8 TIs are delta-positive *as a raw backtest number* on TRAIN+VAL, and the low-lag types (HMA/TEMA) are the
least-bad — but none is distributed: **drop the top 5 of 227 slices and every one goes negative**, and ~85% of the
edge is 2021. Top-K sweet spot is 5, but that's tuning a non-significant number.

## The honest ceiling (what is NOT achieved)

- **Clause 1 (never lose a deployed week): not met** — losing-deployed-week rate is ~16-20% (best TEMA 16% VAL). The
  gate keeps ~80% of all slices cash-or-profitable; pushing losing weeks to 0 is not achievable in noisy crypto
  without deploying ~never (and would overfit).
- **The delta is modest and magnitude-driven** — the bot beats the market in only ~51% of weeks (coin-flip hit rate);
  the positive delta comes from winning bigger when it wins (top-5 momentum). Two of five years are delta-negative.
- **OOS is untouched by design** — these numbers are TRAIN(-tuned) + VAL(-validated). The genuine forward test is OOS,
  which must be run **once**, later, without further tuning.

## Clause 1 (cash-or-profitable) — achievable to ~93%, but as a capital-preserver (the two clauses are in tension)

Since clause 2 (beat the market) isn't robust, I optimized clause 1 (never lose a deployed week) via **gate tightness**
— a mechanical selectivity choice, robust by construction (`strat_ti_gate_tightness.py`, HMA + momentum top-5):

| gate | deploy% | **cash-or-profit%** (TR / VAL) | losing-wk% | delta% (TR / VAL) |
|---|---|---|---|---|
| breadth (loose) | 48-55% | 80 / 82 | 18-20% | +0.6 / +2.3 (not robust) |
| **breadth & ER>0.35** | 20-27% | **93 / 93** | 7% | **−1.4** / +0.8 |
| breadth & ER45 & BTC-up | 15-27% | 95 / 93 | 5-7% | −1.3 / −0.1 |

The tight gate makes **93% of all 7d slices cash-or-profitable on BOTH TRAIN and VAL** (~7% losing weeks) — a *robust*
result (identical across windows = mechanical, not fitted). **But its delta goes negative** (−1.4% TRAIN): deploying
only ~25% of weeks, it misses up-weeks and lands *below* the market. **The two clauses trade off:** you can have ~93%
non-losing weeks (capital-preserver, sub-market) OR a positive delta (not robust) — not both. This is the de-risked-beta
ceiling, quantified per the success criterion.

## Verdict (CORRECTED after adversarial audit — my "distributed" claim was REFUTED)

I initially called this a distributed success on the strength of drop-top3 (+0.14%). An adversarial auditor (and my
own per-year data) refuted that — my jackknife was too shallow:
- **drop-top-5 slices (2.2% of 227) → delta goes NEGATIVE** (HMA −0.20%, TEMA −0.16%). The whole +0.92% is ~5 weeks.
- **Without 2021, delta falls +0.92% → +0.14%** (2021 carries ~85%; its slice-delta +3.49% vs ~0/neg elsewhere).
- **Bootstrap P(delta≤0) = 0.13, p05 = −0.41%** — not significant at 5%, *before* correcting for the 8-type×6-config
  search that picked HMA/TEMA (no DSR/Bonferroni applied).
- **The TI is largely vacuous:** an *ungated* momentum-top-5 (no TI eligibility at all) already beats EW by +1.52% on
  VAL; the TI adds only +0.76pp; the two share 82% of picks and correlate 0.92–0.97. The median eligible set is 12 of
  49 names, so the momentum-top-5 almost always fits inside it — the MA cross rarely binds. It's ~half re-labeled momentum.

**Honest verdict:** mechanics are clean (no look-ahead, WALL enforced — a *real* backtest number), but there is **no
distributed, TI-specific, statistically-supported slice-edge.** The +0.9%/slice delta is 2021 + the VAL bull wearing a
momentum costume. What IS robust and real is **clause 1 to ~80%:** the breadth gate mechanically keeps ~80% of slices
cash-or-profitable (losing-deployed-week rate ~20%, an irreducible floor) — i.e. the TI bots are **de-risked beta**,
the same ceiling the whole arc hits, NOT a market-beating per-slice edge.

**Methodology WIN (keep this):** the TRAIN+VAL-only discipline (OOS walled out) is correct and fixes the prior
"gates tuned on OOS" contamination; the per-slice delta-vs-market framing is the right lens. **Lesson: a delta can be
positive yet concentration-fragile — always jackknife to drop-top-5/10 + drop-best-year + bootstrap + a
multiple-comparisons (DSR) gate BEFORE calling it distributed; drop-top-3 is too shallow (it fooled me twice this session).**

Artifacts: `strat_ti_slice_audit.{py,json}` (baseline), `strat_ti_redevelop.{py,json}` (the momentum-top-5 fix),
`strat_ti_slice_verify.{py,json}` (my too-shallow jackknife), auditor probes `audit_probe_tislice.py` / `audit_probe2.py`
(drop-top5/no-2021/bootstrap/vacuity — the decisive refutation).
