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

| TI | TRAIN Δ/slice | VAL Δ/slice | full Δ | drop-top3 Δ | years Δ>0 | verdict |
|---|---|---|---|---|---|---|
| **HMA** | +0.60% | **+2.28%** | +0.92% | **+0.14%** | 3/5 | **DISTRIBUTED — lead** |
| **TEMA** | +0.57% | **+2.23%** | +0.89% | **+0.15%** | 3/5 | **DISTRIBUTED — lead** |
| VIDYA | +0.05% | +2.11% | — | — | — | delta-positive |
| DEMA | +0.18% | +1.65% | — | — | — | delta-positive |
| EMA/SMA/WMA/KAMA | +0.1-0.4% | +1.4-1.6% | +0.56% (SMA) | −0.02% (SMA) | 3/5 | delta+ but **SMA concentrated** |

**Lead = HMA / TEMA (low-lag adaptive).** Their +0.9%/slice delta **survives dropping the top-3 slices** and is
positive in 3 of 5 years (neg only 2020 chop-onset + 2023-H1) → a *distributed*, not one-burst, edge. Simple SMA's
edge dies on drop-top3 (concentrated) — the low-lag MAs are materially better. Top-K sweet spot is **5** (top-3 too
concentrated, top-8/all dilutes toward the market).

## The honest ceiling (what is NOT achieved)

- **Clause 1 (never lose a deployed week): not met** — losing-deployed-week rate is ~16-20% (best TEMA 16% VAL). The
  gate keeps ~80% of all slices cash-or-profitable; pushing losing weeks to 0 is not achievable in noisy crypto
  without deploying ~never (and would overfit).
- **The delta is modest and magnitude-driven** — the bot beats the market in only ~51% of weeks (coin-flip hit rate);
  the positive delta comes from winning bigger when it wins (top-5 momentum). Two of five years are delta-negative.
- **OOS is untouched by design** — these numbers are TRAIN(-tuned) + VAL(-validated). The genuine forward test is OOS,
  which must be run **once**, later, without further tuning.

## Verdict

**Success by the delta clause is met and verified as distributed:** the canonical low-lag MA TIs (HMA/TEMA), gated by
breadth and holding the momentum top-5 of their eligible names, beat the average 7d slice on both TRAIN and VAL, and
the edge survives the concentration jackknife that has killed every prior claim in this arc. It is a **real, modest,
regime-leaning per-slice edge** — deploy-ready for the held-out OOS test, not yet a proven forward edge.

Artifacts: `strat_ti_slice_audit.{py,json}` (baseline), `strat_ti_redevelop.{py,json}` (the fix), `strat_ti_slice_verify.{py,json}` (distributed-vs-concentrated).
