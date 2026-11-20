# FINER-TF MA STRATEGY-DISCOVERY ENGINE -- README

> The reusable 6-stage engine for discovering whether a **profitable, complementary, dynamic**
> finer-timeframe (<=1d) moving-average strategy exists -- and for **killing its own false positives**.
> Built 2026-06-14 (5h autonomous). This README lets a future instance **re-run and extend** the engine.
>
> **Executive figure:** [`charts/discovery_engine_dashboard.png`](charts/discovery_engine_dashboard.png)
> (the one-figure summary -- 8 panels covering all 8 stages, every number traces to a stage JSON below).
> **Deployable book (runnable):** `python -m strat.finer_tf_book` (see "The deployable book (runnable)" below).
> **Consolidated narrative:** [`DISCOVERY_ENGINE_FINDINGS.md`](DISCOVERY_ENGINE_FINDINGS.md).

---

## What the engine is

Six **composable, honest, RWYB** stages. Each stage is one tool, answers one question, writes one JSON +
charts, and is **two-sided** (it is designed to *refute* its own candidate, not confirm it). The durable
deliverable is the **discipline** (validate-the-generator-first, proper paired sign test, held-out splits,
multiple-comparisons awareness) -- the trading analogue of the chess engine's monotonic promotion gate.

**Shared frame for ALL stages:** 2020 band only (TRAIN `2020-01-01..2020-07-01` 6mo / VAL `..2020-10-01`
3mo / OOS `..2021-01-01` 3mo), u10, maker cost `0.0006`, causal lag-1 MtM accounting, equal-weight,
long-only + spot. Synthetic generators are calibrated on 2020 data **only** and VALIDATED before trusting.

---

## The 6 stages -- what each answers + the EXACT run command

| # | Stage | What it answers (one line) | Run command | Output JSON |
|---|-------|----------------------------|-------------|-------------|
| 1 | MA-type x TF research | which MA type / TF is the best **trend** sleeve | `python -m strat.ma_type_tf_research --tfs 1d,4h,2h,1h,30m,15m` | `ma_type_tf_research.json` |
| 2 | Complementarity matrix | is **trend (MA) + MR (oscillator)** orthogonal; does combining help | `python -m strat.deep2020_complementarity --cadences 1d,4h,2h,1h,30m,15m` | `complementarity_matrix.json` |
| 3 | Dynamic / ML allocation | does **regime-timing** the blend beat a static blend (real 2020) | `python -m strat.dynamic_allocation_engine --n-shuffle 200` | `dynamic_engine.json` |
| 4 | Synthetic regime-stress | does it all survive **bull / bear / chop / stitched** (daily) | `python -m strat.synthetic_regime_stress --seeds 20 --cadences 1d,30m` | `synthetic_regime_stress.json` |
| 4b| Intraday-resolution stress | does dynamic skill appear at **TRUE sub-daily res** (~48 bars/day) | `python -m strat.synthetic_intraday_stress --seeds 10 --cadences 30m` | `synthetic_intraday_stress.json` |
| 5 | Complementary-sleeve search | what sleeve **TRULY fills a bear gap** (the long-only test) | `python -m strat.complementary_sleeve_search --seeds 20 --cadences 1d,30m` | `complementary_sleeve_search.json` |

All commands run from the repo `src/` import root (the package is `strat`). Each JSON carries a `repro`
block with the git SHA + cost + splits so the run is reproducible. (Build SHAs of record:
1756867 for stages 1/3/4; 72e828f for stage 5; 8f962ab for stage 4b.)

---

## The convergent verdict (what the engine discovered)

**1. ADAPTIVE MA types win every finer TF.** VIDYA wins {4h, 2h, 1h, 30m, 15m}, KAMA wins 1d; the
adaptive-family average net beats simple/low-lag and the edge **widens** at finer cadence (e.g. 15m:
adaptive 61.5% vs simple 47.9% family avg; tuned VIDYA winner 73.0% OOS net). The best trend sleeve is
adaptive -- but it is **participating BETA** (winner net < VOLTGT buy-hold in the 2020 bull); value is
risk-adjusted return + whole-cycle DD protection + cross-TF diversification, not alpha.
*Source: `ma_type_tf_research.json` -> `winners_by_tf`, `family_avg_net_by_tf`, `benchmarks`.*

**2. COMPLEMENTARITY (static trend + MR) is REAL but regime-conditional.** Trend (MA) and MR (oscillator)
are orthogonal at every TF (corr **+0.21..+0.31** vs 0.85-0.94 within-trend). Combining DD-dampens **in
chop (+1.3pp)** -- but is a **BEAR LIABILITY (-11.3pp)**: both sleeves are long-only, so on a down day
neither can win, they only bleed less, and the long-only MR buys falling knives.
*Source: `complementarity_matrix.json` -> `<tf>.corr_trend_mr`; per-regime DD from
`synthetic_regime_stress.json` -> `verdict.complementarity_dd_by_regime`.*

**3. DYNAMIC timing has NO skill -- robust to resolution.** A regime-conditional trend-vs-MR allocator beat
the static blend at only **1 of 6 TFs (30m)** on real 2020 -- at the multiple-comparisons chance rate. On
the VALIDATED synthetic generator (stage 4), the 30m "edge" REVERSES under a regime flip (stitched
sign-test n.s.) = an exposure-tilt level effect, not timing skill. The named falsifier (stage 4b, a TRUE
sub-daily generator ~48 bars/day = far more timing opportunities) was built + validated 3/3 and the verdict
**SURVIVED**: **0 of 4** (cadence, regime) cells significant. **SHIP THE STATIC BLEND**; the dynamic ML
layer is not worth its complexity.
*Source: `dynamic_engine.json` -> `verdict` (real-2020); `synthetic_regime_stress.json` +
`synthetic_intraday_stress.json` -> `verdict` (falsification).*

**4. TRUE complementarity (filling a BEAR gap) REQUIRES a SHORT sleeve -- the long-only constraint is the
binding limit, now QUANTIFIED.** Only a SHORT / inverse-trend sleeve is RETURN-anticorrelated to trend in
the bear (bear corr **-0.44**; bear standalone net **+13.2%** where trend bleeds -6.9%). Long-only
"defensive" gates (CASH_GATE, VOLTGT_DEF) are corr ~+1.0 to trend -- they only **dampen**, never rescue.
Swapping the long-only MR for SHORT_MA buys **+15.2pp bear DD protection + +18.1pp bear net**; LONGSHORT_MA
is net-neutral full-cycle (+14.4pp bear protection). Within long-only + spot, the best is a VOLTGT_DEF
defensive overlay -- a risk-reducer, not a gap-filler.
*Source: `complementary_sleeve_search.json` -> `verdict.scoreboard`, `verdict.long_only_relaxation_value`.*

### Robustness ranking (worst-scenario worst-seed net across the full regime mix, 20 seeds)
TREND_ALONE **-12.1%** (most robust survivor) > VOLTGT_BH -36.1% > STATIC -37.1% > DYNAMIC -44.7% >
MR_ALONE -56.8% > BUYHOLD -77.3%. The long-only MR sleeve is the worst standalone (it adds to the bear
bleed); the trend sleeve alone is the structurally safest.
*Source: `synthetic_regime_stress.json` -> `verdict.robustness_rank`.*

---

## The honest deployable recommendation (2020-band evidence, finer TF)

An **ADAPTIVE-MA (VIDYA/KAMA) trend sleeve per TF** + a **trail-stop** (most robust across regimes) + a
**static MR complement** for chop DD-dampening + a **VOLTGT_DEF defensive overlay** -- all long-only,
participating beta with risk control. The single highest-value UNLOCK is the **long-only-exception**: a
**LONGSHORT_MA** sleeve is the one thing that turns the bear from a liability into near-flat (the +15-18pp),
and it is the **user's strategic call** (deploy needs the LO-exception sign-off).

Mapped to the user's three asks:
- **Profitable (finer TF):** yes, as participating beta -- not internal-data alpha (net < buy-hold in a bull).
- **Complementary:** YES (static trend + MR) for chop DD-dampening; a long-only complement can only dampen,
  never fill a BEAR gap -- TRUE cross-regime complementarity needs a SHORT sleeve.
- **Dynamic:** NO -- regime-timing the blend has no skill, robust across resolution + regime + a proper
  sign test. The honest answer is a STATIC blend (+ defensive overlay).

---

## The deployable book (runnable)

The recommendation above is now ONE runnable artifact -- the only remaining action is **click play**:

```
python -m strat.finer_tf_book                    # the deployable book on 1d + 4h (the robust coarse pair)
python -m strat.finer_tf_book --cadences 1d,4h,1h
python -m strat.finer_tf_book --longshort-insurance   # FLIP ON the bear-insurance sleeve (RESEARCH, LO-exception)
```

It assembles the ship-today **long-only + spot** book by REUSING the engine's exact sleeve builders --
**adaptive-MA (VIDYA/KAMA per-TF winner) trend** + the **base trail-stop(0.10)** + a **static MR oscillator
complement** (chop DD-dampening) + a **VOLTGT_DEF defensive overlay** -- at **PRE-REGISTERED inverse-vol
(equal-risk) weights computed on the pre-OOS TRAIN+VAL slice and FROZEN** (no OOS fit). It grades the book on
the **real 2020 OOS** (Oct-Dec, a clean bull) vs trend-alone / VOLTGT-BH / BUYHOLD, renders
[`charts/finer_tf_book_equity.png`](charts/finer_tf_book_equity.png), and writes `finer_tf_book.json`.
Causal lag-1 MtM, maker `0.0006`, no MtM double-count.

### Honest 2020-OOS stats (real, Oct-Dec; the book is participating BETA, NOT alpha)

| TF | BOOK net | Sharpe | maxDD | p05 | weights (TREND/MR/VOLTGT_DEF) | vs BUYHOLD | vs TREND-ALONE |
|----|---------:|-------:|------:|----:|-------------------------------|-----------|----------------|
| 1d | **+11.6%** | 2.75 | **-6.0%** | -2.0% | 0.33 / 0.19 / 0.48 | BH +47.4% (Sh 2.34, DD -20.2%) | net +13.6% (Sh 2.05, DD -11.0%) |
| 4h | **+13.8%** | **3.07** | **-4.0%** | -1.0% | 0.37 / 0.16 / 0.47 | BH +50.6% (Sh 2.42, DD -20.2%) | net +19.3% (Sh 3.02, DD -6.2%) |

**Best deployable TF: 4h** (Sharpe 3.07). **Honest read:** the book's NET is BELOW buy-hold (the
**participation tax** -- EXPECTED in a clean bull, this is BETA not alpha), but it delivers a **materially
better risk profile** -- higher Sharpe and **~3-5x smaller maxDD** (-4 to -6% vs -20.2%). Its durable value
(whole-cycle DD protection) is exactly what a bull-only OOS cannot show. The finer TFs (`--cadences 1h,30m,15m`)
post higher OOS net but at higher turnover/cost-fragility -- a finer-TF bull artifact; 1d+4h are the deploy
headline. *Source: `finer_tf_book.json` -> `results.<tf>.book_oos` + `benchmarks`.*

### The longshort toggle (OFF by default, RESEARCH)

`--longshort-insurance` adds the **PHASE-6 LONGSHORT_MA** bear-insurance sleeve (maker + modelled
short-borrow) at its own pre-registered weight. It is **OFF by default** and **LO-exception-gated**: the short
leg VIOLATES the standing long-only + spot constraint, so it is **RESEARCH** -- flipping it on requires the
user's explicit long-only-exception sign-off. On the 2020 OOS (a bull) it **drags** (it shorts a rising
market: 1d book net 11.6% -> 9.3% with it on); its value is **bear-DD protection**, which a bull OOS cannot
show -- see `longshort_book.json` for the full-cycle synthetic stress where it lowers bear maxDD by up to
+8.8pp. The book is fully deployable WITHOUT it.

---

## Honest caveats (binding)

- **2020-bull-band, in-sample.** The 2020 OOS (Oct-Dec) is a clean bull (~0% bear); the trend sleeve is
  structurally favored and net < buy-hold is EXPECTED (participation tax). The DD-dampening value of
  complementarity is exactly what should generalize *worse-known* until a real bear/chop is traded.
- **Synthetic-validated, not real-future.** Stages 4/4b/5 lean on synthetic generators **calibrated to
  2020 stylized facts only** -- a STRESS test surface, not future data. A generator only reproduces the
  facts it was calibrated on. Each generator was VALIDATED (3/3 regimes match real-2020 moments) before
  any verdict was trusted.
- **Long-only + spot constraint.** This is the binding limit on cross-regime complementarity. SHORT /
  long-short candidates are **RESEARCH** -- deploying them requires the long-only-exception sign-off.
- **Stage 4b seed count.** The canonical command is `--seeds 10`; if the loaded JSON shows a lower seed
  count it is a snapshot of a firmer run in progress -- the **verdict direction is stable** (VERDICT-
  ROBUST-TO-RESOLUTION; 0 significant cells).
- **UNSEEN N/A** (2020 band only by mandate). No 2021+/2026 data was touched.

---

## How to extend the engine (for the next instance)

1. **New MA family / indicator:** add it to stage 1's `ma_types`, re-run stage 1, then re-point stages 2-5
   at the new winner. (Cross-contamination rule: a new indicator family is a new dossier, not a stage-1 param.)
2. **A real bear band:** the highest-value extension is to re-run the whole engine on a band that *contains*
   a bear (e.g. a 2021-2022 or 2018 slice) to convert the synthetic bear-stress into real-data evidence --
   this is what would upgrade "synthetic-validated" to "held-out-validated".
3. **The LO-exception:** if the user signs off on short, stage 5 already has LONGSHORT_MA / SHORT_MA wired;
   promote SHORT_MA to a deployable sleeve and re-run stages 3-4 with it in the blend.
4. **Always:** validate any new generator (3-check: intraday dist + |r| ACF + daily-aggregate vs real)
   BEFORE believing any verdict it produces. The engine's value is that it kills its own false positives.

---

*Provenance: user mandate -- "trade finer timeframes <=1d, research thoroughly across ALL MA types, find a
strat that works... build ENGINES for discovering strategies that are profitable, complementary, dynamic...
2020 band only, synthetic data where needed, charts/figures." All numbers in this README trace to the stage
JSONs cited inline. Charts: `charts/` (14 PNGs + the master `discovery_engine_dashboard.png`).*
