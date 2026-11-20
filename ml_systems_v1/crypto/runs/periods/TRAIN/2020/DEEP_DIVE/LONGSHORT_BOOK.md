# PHASE 6 -- THE LONGSHORT-MA ENGINE + THE MULTI-SLEEVE COMPLEMENTARY BOOK

**Build the ONE positive PHASE-4b finding (a SHORT sleeve is the only true bear-gap-filler) into a
deployable-candidate engine, and assemble the user's "system of strategies that covers the whole market."**

- Module: `src/strat/longshort_ma_engine.py` (selftest 4/4 PASS; RWYB `python -m strat.longshort_ma_engine --seeds 20 --cadences 1d,4h,2h,1h,30m,15m`)
- Data: `runs/periods/TRAIN/2020/DEEP_DIVE/longshort_book.json`
- Charts: `runs/periods/TRAIN/2020/DEEP_DIVE/charts/longshort_engine_by_regime.png`, `.../multisleeve_book_vs_singles.png`
- Surface: PHASE 3/4b **VALIDATED** synthetic generator (3/3 regimes match 2020 stylized facts) + the **2020 real OOS** (Oct-Dec). 20 seeds. git_sha `a183f0f`. maker cost + modelled short-borrow; lag-1 causal; no MtM double-count.
- **The LONGSHORT engine VIOLATES the standing long-only+spot constraint -> FLAGGED RESEARCH. Deploying it (or the full book that contains it) needs the user's explicit long-only-exception sign-off.** Built + validated for the learning (user expanded scope for max learnings); the long-only deployable subset is graded alongside.

---

## HEADLINE (claim-tagged, two-sided)

> **[VERIFIED-synthetic+real]** The LONGSHORT-MA engine is a **sound, net-positive, NOT-borrow-fragile**
> market-neutral sleeve that beats its cost-matched null by +4.8 to +10pp and lowers full-cycle (stitched)
> maxDD by +1.3 to +1.7pp vs the adaptive-trend book. **BUT** the naive equal-risk multi-sleeve book does
> **NOT** beat the best single sleeve (TREND-alone) on cross-regime worst-case net at **0/6** cadences --
> a static blend dilutes the regime-favored sleeve. **The actionable nugget**: adding the LONGSHORT sleeve to
> the book consistently improves the book's **worst-DD by +5 to +8.8pp** and **worst-regime net by +2 to
> +2.8pp** vs the long-only-deployable subset. So longshort is a **valuable book COMPONENT**, the user's
> "profit in every regime" goal needs **regime-routing (not a static mix)**, and the long-only subset is the
> ship-today book (without the bear rescue). **SHORT = RESEARCH.**

---

## 1. THE LONGSHORT-MA ENGINE -- construction

Symmetric long-short ADAPTIVE-MA cross, per finer TF {1d,4h,2h,1h,30m,15m}:
- **MA-type = the PHASE-1a per-TF winner** (`ma_type_tf_research` winners): KAMA (1d), VIDYA (4h/2h/1h/30m/15m), with the PHASE-1a-selected confirm-K + exit overlay. NOT a hardcoded EMA.
- **LONG on the slow-MA up-cross + SHORT on the slow-MA down-cross**, equal-weight u10, 0.5/0.5 leg blend.
- **The FIXED short trail-stop** (`complementary_sleeve_search._apply_trail_stop_short`, additive-on-the-low). The long `apply_trail_stop` has a sign bug on negative prices (`hw*(1-trail)` stops out every bar); the fixed short mirror tracks the low-water and stops on a rally `> trail` above it.
- **Short-borrow MODELLED, not free**: `(borrow_bps/1e4)/bars_per_year` charged per-bar on the short leg (borrow accrues continuously). Swept at **0/10/20/30 bps/yr** (majors range); base case **20 bps/yr**.
- **Cost-matched NULL**: a random-direction longshort at the SAME maker+borrow cost -- the edge must beat it (it is the SIGNAL, not the cost convention).

**Selftest 4/4 PASS** (planted regimes, no real calib): bear LS +13.8% vs trend -3.8% & lower DD; bull LS +1.1% (positive, long leg dominates); real engine beats the random-direction null +13.8% vs +1.95%; borrow monotonically reduces net.

---

## 2. ENGINE VALIDATION -- per TF, per regime (20 seeds, synthetic, borrow=20bps)

| TF | bull-drag (LS bull - trend bull) | bear maxDD protection vs trend | bear net | **stitched LS net** (worst) | beats null by |
|----|----:|----:|----:|----:|----:|
| 1d  | **-2.0pp** (TOLERABLE) | +0.5pp | -0.6% | **+11.8%** (-2.2%) | +10.0pp |
| 4h  | -0.9pp (TOLERABLE) | -0.0pp | +0.9% | +5.9% (-2.5%) | +4.8pp |
| 2h  | -0.8pp (TOLERABLE) | -0.0pp | +1.2% | +6.2% (-2.8%) | +5.2pp |
| 1h  | -1.2pp (TOLERABLE) | +0.3pp | +2.5% | +7.3% (-3.9%) | +6.1pp |
| 30m | -1.1pp (TOLERABLE) | +0.4pp | +2.3% | +7.1% (-4.1%) | +5.9pp |
| 15m | -1.1pp (TOLERABLE) | +0.4pp | +2.3% | +7.1% (-4.1%) | +5.9pp |

**Stitched full-cycle maxDD: LONGSHORT lowers DD vs trend at EVERY TF** (1d: -5.95% vs -7.68%; 4h: -2.81% vs -4.27%; ~+1.3-1.7pp). **Best TF for longshort (drag-penalized robustness): 30m.**

> **[VERIFIED-synthetic] Q: bear protection WITHOUT a crippling bull drag?** YES on the synthetic surface --
> bull-drag is small (-0.8 to -2.0pp) and the engine is net-positive full-cycle at every TF. The bear maxDD
> protection vs the *adaptive-trend reference* is **modest** (+0.4-0.5pp), SMALLER than PHASE 4b's +14.4pp --
> because here the trend reference is the tighter ADAPTIVE-MA book (synthetic bear maxDD only ~-5%), not the
> looser EMA trend+MR(long-only) PHASE 4b compared against (whose MR *added* to the bleed). The bear-rescue is
> real but its *marginal* size depends on how leaky the book it complements is.

> **[CAVEAT -- BINDING] The synthetic generator produces DAILY bars regardless of the cadence label.** On the
> synthetic surface 30m == 15m EXACTLY and 2h ~= 4h, because the only per-TF difference is the MA-type/overlay
> -- NOT genuine intraday resolution. The synthetic stress surface does **not** differentiate intraday TFs; the
> genuine per-TF finer-resolution behaviour is only visible on the REAL 2020 data (Section 4).

---

## 3. SHORT-BORROW SENSITIVITY -- the engine is NOT borrow-fragile

| borrow (bps/yr) | stitched net (1d) | bull | bear |
|----:|----:|----:|----:|
| 0  | +11.77% | -0.95% | -0.55% |
| 10 | +11.77% | -0.95% | -0.55% |
| 20 | +11.76% | -0.95% | -0.55% |
| 30 | +11.75% | -0.95% | -0.55% |

> **[VERIFIED]** Borrow of 0->30 bps/yr costs **~0.02pp** net on the stitched cycle -- negligible on a daily-bar
> horizon. The PHASE-4b "short advantage" was an upper bound (borrow excluded); **essentially all of it
> survives** a realistic majors borrow. (Probe at 1000/5000 bps confirms the drag IS applied and proportional;
> a real short-squeeze spike is the 30bps stress case, still tiny here.) **Q: net-positive or net-neutral after
> borrow? -> NET-POSITIVE** full-cycle at every TF, post-borrow.

---

## 4. 2020-REAL OOS ANCHOR (Oct-Dec, ~0%-bear BULL) -- the genuine per-TF picture

This is where the cadence GENUINELY varies (real 30m has 48x the bars of 1d).

| TF | LONGSHORT OOS net (Sharpe, DD, p05) | TREND OOS net (DD) | **real bull-drag** |
|----|----:|----:|----:|
| 1d  | +5.4% (1.66, -5.9, -5.0) | +13.6% (-11.0) | **-8.2pp** |
| 4h  | +4.8% (1.55, -3.9, -4.5) | +19.3% (-6.2) | -14.5pp |
| 2h  | +7.4% (1.45, -8.3, -7.2) | +34.6% (-5.6) | -27.2pp |
| 1h  | +17.4% (3.46, -4.1, +1.2) | +50.4% (-5.8) | -33.0pp |
| 30m | +12.8% (2.30, -4.6, -3.5) | +46.6% (-7.9) | -33.8pp |
| 15m | +13.8% (2.08, -6.9, -5.5) | +54.6% (-11.5) | **-40.8pp** |

> **[VERIFIED-real -- the honest correction to the synthetic]** On REAL 2020 the bull-drag is **NOT** small,
> and it **GROWS with finer TFs** (-8.2pp @1d -> -40.8pp @15m): the trend book participates far harder in a
> real bull at fine TFs, while the LONGSHORT short leg fights it. The synthetic surface (daily bars only)
> **understates the fine-TF bull-drag**. The LONGSHORT engine is *positive at every TF* (+4.8 to +17.4%) but
> always *below trend-alone in this bull* -- it is trading bull upside for bear protection a ~0%-bear window
> cannot show. **The deploy implication: a LONGSHORT engine earns its keep ONLY in a real bear/chop; in a
> sustained bull it is a drag, heaviest at fine TFs. It is a regime-conditional sleeve, not an all-weather one.**

---

## 5. THE MULTI-SLEEVE COMPLEMENTARY BOOK -- {trend + MR + longshort + voltgt_def}

PRE-REGISTERED, regime-agnostic **equal-RISK (inverse-vol)** weights, calibrated ONCE on a held-aside
synthetic-**BULL** slice (seed 0), then FROZEN -- never fit on the bear/chop/stitched stress surface. Weights
(1d): TREND 0.25 / MR 0.10 / LONGSHORT 0.38 / VOLTGT_DEF 0.27 (equal-risk under-weights high-vol MR, over-
weights low-vol market-neutral LONGSHORT).

**Cross-regime robustness (worst-regime net / full-mix-worst maxDD / worst p05), 1d, 20 seeds:**

| candidate | worst-regime net | full-mix-worst maxDD | worst p05 | mix net |
|----|----:|----:|----:|----:|
| BOOK (full, [R]) | -4.9% | -9.5% | -9.2% | +6.4% |
| LONGONLY_BOOK (deployable) | -7.5% | -16.7% | -14.1% | +8.3% |
| TREND | **-4.2%** | -11.8% | -9.9% | +11.4% |
| MR | -22.5% | **-58.4%** | -42.5% | +5.1% |
| LONGSHORT [R] | **-0.95%** | -16.5% | -6.8% | +3.4% |
| VOLTGT_DEF | -3.5% | -12.3% | -9.2% | +7.2% |

> **[VERIFIED-synthetic -- TWO-SIDED, the honest finding] Q4 (DECISIVE): does the full system beat EVERY single
> sleeve on cross-regime robustness?** **NO -- 0/6 cadences.** TREND-alone has the best worst-regime *net* at
> every TF (the synthetic regimes are trend-favorable + a static blend dilutes the regime-winner), and MR's
> fat tail (-58% bear DD) poisons the equal-risk blend even down-weighted to ~0.10. A naive static equal-risk
> mix of these four does NOT add cross-regime robustness over the best component.

> **[VERIFIED-synthetic -- the ACTIONABLE nugget] Q4b: does LONGSHORT help the BOOK?** **YES, consistently.**
> Adding LONGSHORT to the long-only-deployable subset improves the book's **worst-DD by +5.2 to +8.8pp** and
> **worst-regime net by +2.0 to +2.8pp** at every TF (1d: +2.6pp net / +7.2pp DD; 4h: +2.0/+5.2; 30m: +2.8/+8.8).
> The full book (black) is more cross-regime-robust than the deployable subset (gray) -- *because* of the
> market-neutral longshort sleeve. So longshort is a valuable book COMPONENT even though the static book is
> not the optimal construction.

**Quantified robustness gain of the full system vs the best single sleeve:** the full BOOK does NOT win
outright (TREND wins worst-net), but it dominates on DRAWDOWN containment -- the full book's worst-DD (-9.5%)
beats MR (-58%), LONGSHORT-alone (-16.5%), VOLTGT (-12.3%), and the long-only subset (-16.7%); only TREND-alone
(-11.8%) is close, and the full book still beats it. **The system's edge is risk-containment, not raw worst-net.**

---

## 6. VERDICT (two-sided, for the user's decision)

1. **The LONGSHORT-MA engine is sound and deployable-as-research.** Net-positive full-cycle, not borrow-
   fragile, beats its cost-matched null by 4.8-10pp, lowers full-cycle DD. **Best TF: 30m** (synthetic) but
   the **real-data bull-drag is heaviest at fine TFs (-41pp @15m)** -- a coarser TF (1d/4h, -8/-15pp drag) is
   the safer deploy if the LO-exception is granted, trading less bear-rescue for less bull-drag.
2. **It is a regime-conditional sleeve, NOT all-weather.** It earns its keep in a real bear/chop and is a drag
   in a sustained bull. This is the bear-INSURANCE role: pay a small bull premium for a real bear rescue.
3. **The "system that covers the whole market" is NOT a static blend.** The honest finding: equal-risk mixing
   of {trend, MR, longshort, voltgt} does not beat TREND-alone on worst-net (0/6). The next iteration is
   **regime-ROUTING** (hold trend in a detected bull, rotate to longshort/defensive in a detected bear) --
   exactly the dynamic-allocation thread, now with a genuine return-anticorrelated sleeve (longshort) to route
   TO, which the long-only book lacked. **MR should be down-weighted hard or dropped** (its -58% tail poisons
   the blend).
4. **The deployable-TODAY book is the long-only subset (trend+MR+voltgt)** -- but it is NOT bear-return-
   anticorrelated (it dampens, does not rescue): worst-DD -16.7% vs the full book's -9.5%. **The cost of the
   long-only constraint at the book level is +5 to +8.8pp of worst-case drawdown** -- that is the price the
   LO-exception sign-off would buy back.

**SIGN-OFF GATE (parked for the user):** deploying ANY short exposure (the LONGSHORT engine OR the full book
that contains it) requires the explicit long-only-exception sign-off. Until then: RESEARCH only; the long-only
subset is the ship-candidate, with the documented worst-case drawdown cost.

---

## 7. CAVEATS (binding)

1. **SYNTHETIC stress surface** from PHASE 3/4b's VALIDATED generator (2020-calibrated stylized facts ONLY,
   3/3 regimes match) + the **2020 real OOS** -- NOT real future data.
2. **The synthetic generator is DAILY-bar; it does NOT differentiate intraday TFs** (30m==15m, 2h~=4h on
   synthetic). Per-TF intraday behaviour is real ONLY on the 2020-real anchor (Section 4).
3. **The LONGSHORT engine VIOLATES long-only+spot -> RESEARCH.** Deploy needs the user's LO-exception sign-off.
4. **Short-borrow MODELLED** at 0/10/20/30 bps/yr (per-bar on the short leg); a real squeeze can spike borrow.
5. **maker cost, no MtM double-count, lag-1 causal**; the cost-matched random-direction NULL controls for the
   cost convention.
6. **>=20 seeds; distributions (mean +- spread + WORST seed)** reported; no seed cherry-picked.
7. **PRE-REGISTERED equal-risk weights** on a held-aside synthetic-BULL slice -- NOT fit on the stress surface.
8. **The synthetic bear is the 2020-COVID fast-V-crash exemplar**; a slow grind-down bear may differ.
9. **2020-bull-band + synthetic caveat**: the 2020 real OOS is a ~0%-bear bull, so the bear-rescue value is
   demonstrable only on the synthetic bear; the real-data anchor can only show the bull-drag side.
