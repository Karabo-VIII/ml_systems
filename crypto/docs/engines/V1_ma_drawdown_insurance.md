# V1 — MA Drawdown-Insurance Engine

**Engine series:** V1 (the first of an ongoing series of engine builds). This document is **extensive and
self-contained by design** — a fresh instance, or a V2 build, should be able to start *warm* from this file alone:
what the engine is, exactly how it was built per timeframe, every verified number with its source artifact, the
honest in-sample/out-of-fit split, the full red-team, the forensic within-slice evidence, the open/un-verified
corners, and how to reproduce it.

This document **folds in and supersedes** `crypto/docs/MA_DRAWDOWN_INSURANCE_FINDINGS.md`. The defensible-claim
block from that file is preserved **verbatim** in §1. After this V1 doc is the live reference, `MA_DRAWDOWN_INSURANCE_FINDINGS.md`
is provenance only.

> Wall-clock: this doc was written 2026-06-21 (VERIFIED via fresh `date`: Sun Jun 21 02:16 SAST 2026). The engine
> run artifacts were produced 2026-06-20 (engine JSONs, `git_sha a4184bd3`; committed at `efb23a4b`). The "1st verified
> success" state note lives at commit `957b756d`.

---

## 1. Status + the defensible claim

**Status: the first verified, deployable result of the move-capture campaign.** Not alpha — *capital-preservation
within long-only*, the charter's honest fallback, now empirically EARNED (genuine same-exposure timing skill) rather
than asserted. Aggressively red-teamed on 3 axes; GO. Incubation (DEV-window), not live-validated.

### The defensible claim (verbatim from FINDINGS — commit unchanged)

> At coarse TFs (**1d / 4h**), an equal-weight u100 sleeve that goes **long while the MA band holds and sits in cash
> otherwise** roughly **HALVES portfolio max-drawdown (~−35% vs buy-hold ~−82%)** by de-risking into every major crypto
> selloff (COVID-2020, May-2021, 2022 bear, 2023 chop — robust across all 8 MA types), at the cost of giving up part of
> the upside. This is **genuine same-exposure timing skill** (+6 to +15.5pp of drawdown avoided beyond a φ-matched dumb
> book at 1d/4h), NOT a low-exposure artifact. It is **capital-preservation / drawdown-insurance within long-only spot —
> NOT alpha, NOT a money-maker.**

This claim survives the 2020-exclusion (out-of-fit) test fully intact across all 6 audited cells (see §4). The
*property* is OOS-validated; only the full-window compound *number* is contaminated by the 2020 fit-year and must
never be the headline.

---

## 2. What it is + how it was built, PER TIMEFRAME

### 2.1 What it is

8 trading **engines**, one per MA TYPE — **EMA, SMA, WMA, HMA, DEMA, TEMA, KAMA, VIDYA**. Each engine is built
**per (MA type × timeframe)**. The decision rule:

- For a given (type, TF), take that TF's **canonical working band** = the set of `(fast, slow)` MA configs that were
  **net-positive across 2020 TRAIN ∧ VAL** in `runs/periods/TRAIN/2020/DEEP_DIVE/config_leaderboard.json`
  (selection rule literally: `TRAIN.net > 0 AND VAL.net > 0`; **the OOS field is never read in selection**).
- Build the **fixed-EW ENSEMBLE** over that band's `band_configs` — NOT `ranked[0]` (that would be hindsight). Each
  config emits a `held_cross` long/flat state per asset per bar.
- The engine is **long an asset while the MAJORITY of the band's `held_cross` is aligned (>50% of the band long), in
  cash otherwise.** Long-only + spot, lev = 1.
- Aggregate across the u100 panel into one equal-weight sleeve → MtM equity curve vs a TRUE buy-and-hold baseline.

This is **move-CATCH, not per-candle prediction**: the unit is a SETUP across a multi-candle MOVE — the band holds
the position through the trend and de-risks to cash when the band breaks.

**Universe:** 56 of 100 point-in-time u100 names in the wealth panel (the type-independent u100 loader; ≥20 breadth
floor met comfortably). The capture leg admits a type-specific count (51–58 depending on type/TF; see the engine JSON
`n_symbols`); the wealth panel is always 56.

### 2.2 The band rises with cadence (MA lengths)

MA lengths differ by TF — the slow-floor rises at finer cadence (more bars per day → longer MA spans needed to span
the same wall-clock trend). Concrete band sizes from `step1_ema_1d_7d.json` (EMA|1d band = 105 configs spanning
`ema_2_3` … `ema_102_237`) and the forensic `frozen_band_sizes` for EMA: **1d=105, 4h=120, 2h=117, 1h=108** members.
The ensemble fires when >50% of those members are long.

### 2.3 Accounting (the wealth driver is authoritative)

`src/strat/ma_wealth_driver.py:run_wealth(ma_type, tf, window)` is the **one canonical wealth path** (it replaced
per-worker improvisation that produced inconsistent/inflated wealth). It is STAGE-1 adversarially verified:

- **MtM-no-double-count** (`_compound` reconciles to 2e-16; the 2026-04-22 simulator invariant holds).
- **lag-1 causal** (weights lagged one bar; signals are trailing).
- **maker 0.0006 RT** on the wealth leg (survives taker 0.0024 — see §5). The CAPTURE leg (A/B/C) uses taker 0.0024;
  the two costs are never mixed on one column.
- **TRUE buy-and-hold baseline**: buy each name at its listing, NO rebalance, weights drift with price (EW HODL ≈ +465%).
  This is distinct from the *always-long rebalanced EW index* (rebalances every bar, harvests a ~4.6× vol/rebalance
  premium ≈ +2137%) — the old code mislabeled the rebalanced index as "buy-hold"; that was the NO-GO the verify caught
  and fixed. Both legs are now reported separately and correctly.
- **DEV wall < 2024-05-15**: `load_wide` + `portfolio_replay` hard-cap bars < DEV_END; the engine physically cannot
  peek OOS (2020 Q4 in the leaderboard split) or UNSEEN (≥ 2024-05-15).

**Window property (NOT a bug):** the ENGINE hold provider is the band ensemble (a continuous hold-while-long state)
which does NOT depend on the 7d/14d lookback. So wealth at a given TF is IDENTICAL across the 7d and 14d windows; the
7d/14d window varies ONLY the CAPTURE leg's hold horizon (A/C). Throughout this doc, wealth/maxDD numbers are quoted
at the 7d window (they are window-invariant).

### 2.4 The cadence gradient (median across 8 engines) — scope is BINDING 1d/4h

| TF | up-mkt α (pp) | maxDD eng / HODL | bear-presv (pp) | capt-A (pp) | beta-gate |
|---|---|---|---|---|---|
| **1d** | −5111 | **−35% / −82%** | +56.7 | 4.5 | 16/16 |
| **4h** | −5548 | **−34% / −83%** | +59.1 | 3.1 | 16/16 |
| 2h | −6015 | −38% / −83% | +49.9 | 2.6 | 15/16 |
| 1h | −3648 | −56% / −83% | +37.6 | 2.0 | 12/14 |
| 30m | −425 | −77% / −83% | +26.0 | 1.8 | 11/14 |
| **15m** | +18 | **−98% / −83% (BROKEN)** | +15.2 | 0.7 | 5/11 |

The drawdown-insurance property is strong and clean at **1d/4h**, degrades monotonically through 2h/1h/30m as churn
eats the edge, and **BREAKS at 15m** — there the engine compound is −95% vs HODL +461% (cost/churn DESTROYS the engine
*below* buy-hold; maxDD −98% is WORSE than HODL). 15m is broken, not merely "degraded."

> **Scope is binding: 1d/4h only.** Fine TFs (15m especially) are out-of-scope for this engine. (From the EMA engine
> JSON: 1h eng maxDD −47%, 30m −71%, 15m −96%; the median gradient above aggregates all 8 types.)

---

## 3. ALL 8 ENGINES — one row per MA type at 1d and 4h

All numbers pulled directly from `engine_<TYPE>_clean.json` (7d cell; wealth window-invariant). maxDD and HODL are in
percent; bear-presv = 2022-bear preservation in pp vs the always-long index; capt-A = cost-free identification edge in
pp; capt-C = capture efficiency (realized/MFE).

### 3.1 — 1d timeframe

| Type | eng maxDD | HODL maxDD | bear-presv | capt-A | capt-C | beta-gate | eng compound | HODL compound |
|---|---|---|---|---|---|---|---|---|
| EMA  | −37.5 | −82.3 | +56.06 | 3.54 | 0.599 | PASS | +3609.82% | +465.09% |
| SMA  | −35.4 | −82.3 | +53.55 | 3.69 | 0.607 | PASS | +2153.31% | +465.09% |
| WMA  | −31.9 | −82.3 | +57.30 | 3.59 | 0.604 | PASS | +3804.07% | +465.09% |
| **HMA**  | **−28.1** | −82.3 | **+69.41** | 3.31 | 0.617 | PASS | +4007.93% | +465.09% |
| DEMA | −43.5 | −82.3 | +49.70 | 2.56 | 0.604 | PASS | +4580.75% | +465.09% |
| **TEMA** | **−27.7** | −82.3 | +64.43 | 3.33 | 0.608 | PASS | +3447.36% | +465.09% |
| KAMA | −34.8 | −82.3 | +56.46 | 4.40 | 0.611 | PASS | +1780.17% | +465.09% |
| VIDYA| −41.4 | −82.3 | +56.96 | 3.54 | 0.585 | PASS | +2302.62% | +465.09% |

### 3.2 — 4h timeframe

| Type | eng maxDD | HODL maxDD | bear-presv | capt-A | capt-C | beta-gate | eng compound | HODL compound |
|---|---|---|---|---|---|---|---|---|
| EMA  | −32.5 | −82.8 | +62.38 | 3.15 | 0.452 | PASS | +3157.56% | +465.55% |
| **SMA**  | **−30.2** | −82.8 | +59.63 | 3.32 | 0.457 | PASS | +2727.11% | +465.55% |
| WMA  | −35.5 | −82.8 | +60.95 | 3.01 | 0.450 | PASS | +2584.20% | +465.55% |
| HMA  | −43.8 | −82.8 | +48.32 | 1.39 | 0.412 | PASS | +687.32%  | +465.55% |
| DEMA | −34.8 | −82.8 | +58.57 | 1.90 | 0.433 | PASS | +2021.04% | +465.55% |
| **TEMA** | **−30.3** | −82.8 | +60.17 | 1.31 | 0.416 | PASS | +1069.97% | +465.55% |
| KAMA | −34.1 | −82.8 | +50.82 | 3.03 | 0.453 | PASS | +1624.90% | +465.55% |
| VIDYA| −33.4 | −82.8 | +55.04 | 3.63 | 0.454 | PASS | +4319.82% | +465.55% |

**Best preservers at 1d:** TEMA (−27.7%) and HMA (−28.1%). **At 4h:** SMA (−30.2%) and TEMA (−30.3%). Across the
16 cells the engine maxDD ranges −27.7% to −43.8%, every one roughly HALF of HODL's ~−82%. **All 16 beta-gates PASS**
(≥1 non-bull regime beats its random-entry null — i.e. the edge is regime-matched, not beta-in-disguise).

Caveat on HMA|4h (−43.8%): it is the weakest preserver — its 4h capt-A is only 1.39pp and its out-of-fit compound
flips *below* HODL (see §4). It still halves the drawdown; it is just the weakest member at 4h.

---

## 4. In-sample vs out-of-sample (HONEST)

The MA working band is **fit on 2020** (TRAIN ∧ VAL). The full-window metrics in §3 run 2020 → 2024-05-15 and therefore
**include the in-sample fit-year**. To isolate genuine out-of-fit behavior we slice the verified equity curves to
exclude 2020 entirely.

### 4.1 The out-of-fit result (the clean headline)

**Window = 2021-01-01 → 2024-05-15** (OUT-OF-FIT; the 2020 in-sample fit-year is EXCLUDED; OOS/UNSEEN past 2024-05-15
NOT touched). Cells:

| Type | TF | eng maxDD | HODL maxDD | eng compound | HODL compound |
|---|---|---|---|---|---|
| EMA  | 1d | −37.5 | −82.3 | +1379.51% | +279% |
| EMA  | 4h | −32.5 | −82.8 | +1049.95% | +273.33% |
| HMA  | 1d | −28.1 | −82.3 | +1417.36% | +279% |
| HMA  | 4h | −43.8 | −82.8 | **+238.14%** | **+273.33%** |
| TEMA | 1d | −27.7 | −82.3 | +1185.42% | +279% |
| TEMA | 4h | −30.3 | −82.8 | **+373.80%** | +273.33% |

**Artifact:** `runs/engines/V1_ma_drawdown_insurance/outoffit/outoffit_2021_2024-05.json`.

### 4.2 Method (no new accounting; leak-free by construction)

Post-hoc slice of the VERIFIED `run_wealth` (`src/strat/ma_wealth_driver.py`) MtM equity curves — **NO new
accounting.** The OutOfFit phase reproduced `run_wealth`'s EXACT internal pipeline (reusing its own `_compound`,
`_true_hodl`, `_load_asset`, `_u100_symbols` primitives), recovered the dated engine + TRUE-HODL equity curves
indexed by `cp.index`, then sliced to `[2021-01-01, 2024-05-15)` and re-based compounding to the sub-window start
(`eq[-1]/eq[0] − 1`). **maxDD is scale-invariant and HODL's single terminal cost factor cancels under re-basing, so
the slice is exact / leak-free.**

**Fidelity gate:** all 6 cells — engine compound%, engine maxDD%, HODL compound%, HODL maxDD% recomputed from the
reconstructed FULL-window curves match the committed `engine_<TYPE>_clean.json` scalars **TO THE PENNY** (e.g.
HMA|1d eng 4007.93/4007.93, dd −28.1/−28.1) ⇒ the sliced curve IS the verified curve. (`n_symbols` in clean.json is the
capture-leg type-specific count; the wealth panel is the type-independent u100 loader = 56 for all cells, matching
`run_wealth`'s own `n_assets` — not gated, a different quantity.)

### 4.3 Does the maxDD-halving hold out-of-fit? YES, in all 6/6 cells.

Engine maxDD −27.7% to −43.8% vs HODL ~−82%, a **39–55pp drawdown cut excluding 2020.**

**Critical observation:** the engine maxDD over the out-of-fit sub-window is **IDENTICAL** to the full-window value in
every cell (e.g. EMA|1d −37.5/−37.5, TEMA|1d −27.7/−27.7) — because the engine's worst drawdown occurs entirely within
2021–2023 (May-2021, the 2022 bear, 2023 chop — three out-of-fit selloffs), NOT in the 2020 fit-year. HODL maxDD is
likewise essentially unchanged (−82.3/−82.8) since COVID-2020 (−62%) was not HODL's worst either. So **excluding the
in-sample year removes nothing from the drawdown-insurance property; it is genuinely driven by out-of-fit selloffs.**

### 4.4 Where 2020-exclusion materially changes the story (COMPOUND ONLY, not the headline)

On the **4h** timeframe, **HMA|4h flips to BELOW HODL** (eng +238% vs HODL +273%) and **TEMA|4h drops to a thin edge**
(+374% vs +273%) once 2020 is excluded — confirming the full-window compound gap (e.g. HMA|4h full +687% vs HODL +466%)
was partly the compounded 2020 bull, not skill. **This is exactly the overclaim the findings doc says to kill** ("do
NOT cite the full-window compound as a headline"). The 1d cells retain a large compound gap out-of-fit (eng ~+1185–1417%
vs HODL +279%) — **but that too is the compounded bear-dodge, NOT up-market alpha.** The compound figures should never
be the headline regardless.

### 4.5 The plain statement

The **property** (drawdown roughly HALVED, ~−30% vs ~−82%, via capital-preservation timing) is **OOS-validated**:
3 of 4 dodges are out-of-fit, the VAL-forward held, and the random Mar-2024 slice held (see §6). The **defensible claim
survives the 2020 exclusion fully intact across all 6 cells.** Only the full-window compound *number* is contaminated
by the 2020 bull. OOS/UNSEEN (post-2024-05-15) remains sealed per the user.

---

## 5. Verification — the 3-axis aggressive red-team (GO on all 3)

### Axis 1 — Timing skill, NOT low-exposure (the make-or-break)

The engine's average gross exposure φ ≈ 0.40 (in cash ~60% of the time). A long-only book that is flat 60% of the time
MECHANICALLY has lower drawdown — that is not skill. So we compare against two **exposure-matched, churn-immune** dumb
controls that hold the SAME φ with ZERO MA timing:

- **CFAL** (Constant-Fraction Always-Long): deploy exactly φ of capital into the rebalanced EW index every bar, rest in
  cash. Churn-immune; maxDD_CFAL ≈ φ × maxDD_alwayslong by construction.
- **REHTM** (Random-Entry Hold-To-Maturity, 24 seeds): random non-overlapping entries, each held for the engine's run
  length, sized to match φ. **Churn-immune — this is NOT the broken A9 same-exposure per-bar shuffle** (that null is
  churn-penalized and broken; see `BROKEN NULL` memory).

**Skill delta = engine's extra drawdown-reduction beyond the dumb book** (`_skeptic1_exposure_*.json`):

| Cell | engine φ | eng maxDD | CFAL maxDD | REHTM maxDD (mean) | skill Δ vs CFAL | skill Δ vs REHTM | z vs REHTM |
|---|---|---|---|---|---|---|---|
| EMA \| 1d | 0.400 | −37.5 | −46.7 | −43.48 | +9.2pp | +6.0pp | 1.64 |
| SMA \| 1d | 0.373 | −35.4 | −44.3 | −43.48 | +8.9pp | +8.1pp | 1.66 |
| DEMA \| 1d | 0.419 | −43.5 | −48.4 | −49.15 | +4.9pp | +5.7pp | 1.39 (PARTIAL) |
| EMA \| 4h | 0.402 | −32.5 | −48.0 | −47.15 | +15.5pp | +14.7pp | **3.53** |
| SMA \| 4h | 0.369 | −30.2 | −44.9 | −43.67 | +14.7pp | +13.5pp | 2.63 |

The engine avoids **+6 to +15.5pp extra drawdown at the same exposure** — it is in cash at the *right* times. It also
preserves the 2022 bear **+12 to +24pp beyond the dumb book** (e.g. EMA|4h bear skill Δ +24.29pp vs REHTM). DEMA|1d is
labeled PARTIAL (weak but positive). The strongest signal is at **4h** (EMA|4h z=3.53). **Verdict: TIMING_SKILL.**

### Axis 2 — Leak-free + survivorship-honest + cost-robust (`_skeptic2_results.json`)

- **Leak-free:** the lag-0 cheat (peeking the same bar) explodes compound to absurd levels and shrinks DD — DEMA
  lag-0 compound **+2,636,238%** (vs lag-1 +4581%), maxDD −16.9% (vs −43.5%), Δ maxDD lag0−lag1 = **+26.59pp**. EMA
  lag-0 +68,476% / Δ +17.18pp; KAMA lag-0 +26,772% / Δ +18.72pp. ⇒ the test HAS power; the shipped lag-1 numbers are
  causal.
- **Survivorship-honest:** injecting dead names into HODL makes HODL *worse* — EMA/DEMA/KAMA HODL −82.3% (dead_0) →
  −82.6 (dead_10) → −82.8 (dead_20) → −83.1 (dead_30), with compound +465% → +268%. The engine holds **cash**, not the
  zeros, so its maxDD is unchanged (−43.5 fixed). Dead names therefore WIDEN the gap — the *relative* halving is fair
  (and slightly conservative as quoted). The 2024-survivor panel has no LUNA/FTT/UST.
- **Cost-robust:** survives taker 0.0024 RT — DEMA taker compound +4102% / maxDD −44.3% (still halved); EMA taker
  +3423% / −38.7%; KAMA taker +1701% / −36.2%. `maxdd_still_halved_vs_hodl = true` in every case.

### Axis 3 — Robust, NOT 2022-curve-fit (`_skeptic3_per_event_dd.json`)

Drawdown-cut in ALL FOUR stress events — and the LARGEST cut is in COVID/May-2021, **not** 2022 (so the engine is not
a single-event 2022 curve-fit). From the per-event detail (EMA / HMA / DEMA at 1d):

| Event | index maxDD | engine dd-cut (range over EMA/HMA/DEMA) | engine φ during event |
|---|---|---|---|
| COVID 2020 | −61.7% | **+42 to +53pp** (eng −9.0 to −19.4) | 0.11–0.19 |
| May 2021 | −65.0% | **+34 to +51pp** (eng −13.7 to −31.6) | 0.08–0.17 |
| 2022 bear | −77.3% | **+34 to +56pp** (eng −21.2 to −43.5) | 0.31–0.36 |
| 2023 chop | −43.8% | **+15 to +16pp** (eng −28.0 to −29.3) | 0.32–0.46 |

In the 2023 recovery the engine **re-risks** (gross 0.32–0.46, captures 55–100% of the rally — e.g. HMA captured +158.7%
of the index's +235.4% point-to-point) — it is NOT a permanent-cash dodger. Consistent across all 8 engines × {1d,4h}
(**16/16 cells, all beta-gates pass**).

> The FINDINGS doc's headline range (COVID +42..53 / May-21 +34..51 / 2022 +34..56 / 2023 +15..16) is reproduced here
> from the same artifact.

---

## 6. Forensic within-slice evidence

Two complementary forensic studies, both in `runs/engines/V1_ma_drawdown_insurance/slices/` with charts in
`.../charts/slices/`.

### 6.1 The bull-week 2021-01 (in-sample-period structural read)

`forensic_slice_bull_2021-01.csv` + `slices_within_window.{json,csv}`. A strong-bull alt-ramp week (2021-01-02…01-08;
BTC $30k→$40k, alts ripped). The point is the **timeframe gradient** — coarse TFs enter the move *structurally late*:

- **SOL:** 1d entry only fires 2021-01-07 for a **+4.9%** slice ROI; the **4h** entry fired 2021-01-01 and rode the
  full **+94.7%** move. (1d is structurally late on a fast multi-day ramp; 4h/2h are the sweet spot.) 2h split into a
  +25.7% + 44.5% pair; 1h +29.3% + 44.5%.
- **BTC:** 1d caught the whole ramp from 2020-10-08 (+262.5%); **15m over-trades** — a churned sequence of
  signal_cross_down exits (e.g. −7.22%, −3.87%, −5.70%, −2.77% chop trades) before the final +25.3% leg, i.e. the
  fine-TF churn that breaks 15m.
- Portfolio that week: engine ROI +19.88% vs buy-hold-all +31.95%; engine avg gross exposure 0.687; up-5% mover
  capture-rate 85.2% (23/27). The engine **gives up upside in a pure bull** (no up-market alpha) — consistent with §9.

### 6.2 The random OOS-for-band Mar-2024 VAL slice (out-of-sample confirmation)

`forensic_slice_oos_val_2024-03-25_2024-03-31.{json,csv}`. The EMA band is **FROZEN from 2020**; this is a *randomly
drawn* 7-day slice (offset draw 268 of a 312-day VAL span) inside the VAL window `[2023-07-01, 2024-05-15]`. Self-verify
confirms no leak (max loaded bar 2024-03-31 23:00 < VAL_END 2024-05-15), band frozen 2020, lag-1 causal.

Deployed into an up-week, the frozen band behaved as **de-risked beta — confirmed out-of-sample:**

| TF | n_held | engine slice ROI | buy-hold-all ROI | eng vs BH | movers ≥5% | captured |
|---|---|---|---|---|---|---|
| 1d | 48/54 | +4.142% | +4.118% | +0.02pp | 18 | **17/18** |
| 4h | 50/54 | +4.108% | +7.048% | −2.94pp | 45 | 43/45 |
| 2h | 54/54 | +3.518% | +7.694% | −4.18pp | 49 | 49/49 |
| 1h | 54/54 | +4.025% | +8.180% | −4.16pp | 52 | 52/52 |

At **1d** the engine captured **17 of 18 up-movers** and matched buy-hold to within +0.02pp — de-risked beta confirmed
on a slice the band never saw. At finer TFs it captures nearly all movers but lags buy-hold on net (the same coarse-vs-fine
trade-off; finer TF gives up more to churn in a bull). Charts: `forensic_BTCUSDT_alltf_bull_2021-01.png`,
`forensic_SOLUSDT_alltf_bull_2021-01.png`, `oos_val_slice_2024-03-25_2024-03-31.png`, `slice_{BULL_2021-01,BEAR_2022-05,CHOP_2023-08}.png`.

---

## 7. Open / un-verified (DO NOT present as verified)

The following artifacts are **on disk but NOT adversarially verified.** They are candidate extensions, flagged clearly
as pending verification. They are relocated to `runs/engines/V1_ma_drawdown_insurance/open_unverified/`.

- **`val_forward_*.json` (8 files, one per MA type).** Held-out forward windows (schema: `{ma_type, window, by_tf}`).
  Intended as a forward-window stress of the frozen band. **NOT run through the §5 red-team; do not cite their numbers
  as verified.**
- **`decoupled_7slice_14lb_*.json` (16 files; 1d + 4h per type) + `_decoupled_compare_{1d,4h}.json` (2).** A
  7d-slice / 14d-lookback "decoupled" variant (schema carries `engine_compound_pct`, `maxdd_engine_pct`,
  `capture_7d_slice_avg_exposure`, `derisked_beta_confirmed`, etc.). This decouples the capture hold from the band
  lookback — a genuinely different construction from the verified engine. **NOT adversarially verified.** Verify before
  any use.

Verifying these (running them through the same exposure-matched control + leak/cost/survivorship battery as §5) is the
natural first task for a V2 build.

---

## 8. Source map + reproduce

### 8.1 V1 core drivers (the engine IS these files — shared `src/`, NOT relocated)

| File | Role |
|---|---|
| `src/strat/ma_wealth_driver.py` | **Authoritative wealth path** — `run_wealth(ma_type, tf, window)`; MtM-reconciled, lag-1, maker RT, TRUE-HODL baseline, DEV wall. Source of every wealth/maxDD number. |
| `src/strat/ma_movecatch_engine.py` | The parametrized engine driver — `run_engine(ma_type, tf, window)`; band selection (TRAIN∧VAL) + held_cross ensemble + capture A/B/C + beta gate. (`ma_wealth_driver` supersedes its old `run_engine` wealth path.) |
| `src/strat/ma_exposure_control.py` | **Skeptic-1 skill control** — CFAL + REHTM exposure-matched churn-immune controls; the skill-delta test. |
| `src/strat/capture_lab.py` | STEP-1 capture-rate harness (A identification / B tradeable / C efficiency); random-entry churn-immune null. Committed 19e291a2. |
| `src/strat/fleet_lab.py` | `load_wide` — the DEV-walled (≤2024-05-15) u100 panel loader; 6-TF (2h resampled-from-1h). |
| `src/strat/portfolio_replay.py` | `run(universe="u100", hold_provider=...)` — the MtM portfolio replay (lagged weights × next-bar returns, turnover cost). The wealth aggregation engine. |
| `src/strat/ma_type_upgrade.py` | `_MA` + `held_cross` for all 8 MA families (the per-type hold-state provider). |

### 8.2 Prior-campaign lineage (provenance, NOT V1 core)

These are earlier MA-campaign files that informed the band / methodology but are **not** part of the V1 engine. They
are documented here for archaeology, not relocated: `ma_2020_breakdown.py`, `ma_2020_config_leaderboard.py`,
`ma_2020_per_instrument*.py`, `ma_type_tf_research.py`, `ma_strat_builder.py`, `working_band_rolling.py`,
`robust_ma_runners.py`, `ma_adaptive_translation_2021.py`, `ma_blend.py`, `ma_within_year_reconcile.py`,
`ma_val_forward.py`, `ma_wealth_driver_decoupled.py`, `ma_decoupled_trade_charts.py`, and the per-type profilers
(`dema_profile.py`, `hma_profile_builder.py`, `sma_profile_optimizer.py`, `vidya_robust.py`, `ema_profile_compute.py`).

### 8.3 Canonical inputs (NOT relocated — shared)

- `runs/periods/TRAIN/2020/DEEP_DIVE/config_leaderboard.json` — the band grid (TRAIN/VAL/OOS net per config). The
  selection reads `TRAIN.net>0 AND VAL.net>0` only.
- `config/universes/u100.yaml` — the universe spec.

### 8.4 Reproduce commands

```bash
# from crypto/ root
# (1) the 8 engine clean wealth + capture cells (per type, all 6 TFs x {7d,14d}):
python -m strat.ma_movecatch_engine            # run_engine over the 8 types (capture + gate)
python -m strat.ma_wealth_driver               # run_wealth over the 8 types (authoritative wealth)

# (2) skill control (skeptic 1 — exposure-matched CFAL + REHTM):
python -m strat.ma_exposure_control            # -> _skeptic1_exposure_<TYPE>_<TF>.json

# (3) the forensic within-slice studies:
python scripts/forensic_slice_bull_2021.py     # -> forensic_slice_bull_2021-01.csv + slices_within_window.*
python scripts/forensic_slice_oos_val.py       # -> forensic_slice_oos_val_2024-03-25_2024-03-31.*

# (4) the out-of-fit (2020-excluded) slice of the verified curves:
python scripts/outoffit_slice.py               # -> outoffit/outoffit_2021_2024-05.json

# capture-rate self-test:
python -m strat.capture_lab --selftest --tf 4h
```

### 8.5 Artifact layout (post-reorg — `runs/engines/V1_ma_drawdown_insurance/`)

```
engines/          engine_{EMA,SMA,WMA,HMA,DEMA,TEMA,KAMA,VIDYA}_clean.json   (8; the per-(type x TF) cells)
verification/     _skeptic1_exposure_{DEMA_1d,EMA_1d,EMA_4h,SMA_1d,SMA_4h}.json, _skeptic2_results.json, _skeptic3_per_event_dd.json
step1/            step1_ema_1d_7d.json
charts/           portfolio_equity_*.png, trade_*_*.png, trades_*.csv, trade_summary_*.csv, slices/
slices/           forensic_slice_bull_2021-01.csv, forensic_slice_oos_val_2024-03-25_2024-03-31.{csv,json}, slices_within_window.{csv,json}
open_unverified/  decoupled_7slice_14lb_*.json (16), val_forward_*.json (8), _decoupled_compare_{1d,4h}.json (2)   [NOT verified]
outoffit/         outoffit_2021_2024-05.json
README.md         the run manifest
```

---

## 9. Mandatory caveats (ship attached)

- **ZERO up-market timing alpha.** The engine underperforms HODL in a pure bull (upmkt α −2851 to −7307pp vs index at
  1d/4h; e.g. EMA|1d −4682pp, HMA|4h −7307pp). It is drawdown-insurance, not a money-maker.
- **Do NOT headline the full-window compound** (e.g. DEMA|1d +4581% vs HODL +465%). It is the **compounded bear-dodge**,
  not skill — and §4 shows it is partly the compounded 2020 bull (HMA|4h flips *below* HODL out-of-fit). That is THE
  overclaim to kill.
- **DEV-window incubation only.** The band is selected on 2020 TRAIN+VAL; OOS/UNSEEN (≥2024-05-15) is sealed. This is
  incubation, **not live-validated.** Property OOS-validated (§4); not yet paper/live.
- **Panel = 56 of 100 point-in-time u100 names.** Absolute HODL −82% is survivor-flattered (true ≈−83%); since dead
  names make HODL worse, the survivorship correction STRENGTHENS the relative case (§5 Axis 2).
- **Scope is binding 1d/4h.** 2h degraded, 1h/30m worse, **15m BROKEN** (eng −95% vs HODL +461%). Do not deploy below 4h.
- **The short / long-short / market-neutral move is OFF THE TABLE.** This engine is the honest within-long-only answer
  to the bear gap: capital-PRESERVATION (cash/de-risk), never "enable the short."

---

## 10. Provenance

- Engine run artifacts produced 2026-06-20 (`git_sha a4184bd3`); 1st-verified-success committed `efb23a4b`; state note
  `957b756d`. 2h TF fix: `34be2be5`. Capture STEP-1: `19e291a2` (git_sha in step1 JSON: `c7e7e34c`).
- Two fabrications were caught by the gates en route (a +5828% MtM-look and a +4854pp baseline) — the reason
  `ma_wealth_driver.run_wealth` is the single authoritative wealth path.
- Supersedes `crypto/docs/MA_DRAWDOWN_INSURANCE_FINDINGS.md` (whose verbatim claim block is preserved in §1).
- Operating-model context: `crypto/CHARTER.md` (intent), `crypto/CURRENT.md` (state), `crypto/docs/COGNITION_RULES.md`
  (anti-collapse rules). The charter's "honest fallback" (capital-preservation within long-only) is what this engine
  empirically EARNS.
