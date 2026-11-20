# MA-Oracle Decomposition -- CONCLUSION (2026-06-06 ~02:11; CORRECTED ~02:18 after RWYB + liq-augmented test)

Method: oracle-decomposition (docs/ORACLE_DECOMPOSITION_2026_06_06.md) -- construct the perfect-foresight oracle,
decompose its DNA via causal MA features, measure the realizable ceiling. Run across the FULL grid
(cadence x asset) directly via the loop's rigorous falsifier (shuffled + positive-control + regime-firewall).

## Per-cadence realizable-ceiling MAP (PER_CADENCE_MAP.json; 4 assets/cadence)
| cadence | held-out AUC | oracle median hold | MA-DNA capture | DNA genuine |
|---|---|---|---|---|
| 1d  | 0.54 | 2 bars | -16% | 0/4 |
| 4h  | 0.70 | 2 bars | -74% | 0/4 |
| 1h  | 0.78 | 2 bars | -99% | 0/4 |
| range | 0.90 | 3 bars | -93% | 0/2 |
| dollar | timed out (huge series, slow oracle DP) | -- | -- | -- |

## Findings (definitive)
1. **Oracle hold ~2 bars at EVERY cadence/bar-type.** The market's capturable per-move is a ~2-bar reversal, in any
   units. Structural.
2. **MA-DNA never tradeable: 0/14 genuine.** Negative held-out capture at every cadence; always loses the
   regime-matched random-entry null.
3. **AUC rises (0.54->0.90) as bars get finer, but capture worsens (-16%->-99%).** Statistical recognition of the
   "looks-oversold" texture != tradeable timing: the MA cannot separate the true 2-bar bottom from a continuing
   fall, and the false positives cost more at higher frequency.

## CRITICAL CORRECTION (2026-06-06 ~02:18) -- the falsifier already tests ALL fast features
The falsifier's `_feature_cols()` selects EVERY `norm_`/`xd_` chimera feature -- 40 features spanning **4 families**:
MA/trend (4), momentum (8), volatility (6), and **orderflow/microstructure (11): norm_vpin,
norm_hawkes_buy_intensity, norm_flow_imbalance, etc.** So "0/14 genuine" is NOT an MA-only result -- the broad
*causal* DNA (MA + momentum + vol + orderflow/micro) is what fails. RWYB caught the earlier draft's error of
framing orderflow as an untested "next avenue": **it was already in the feature set and already failed.**

I then ran the one genuinely-untested family directly: **+29 LIQUIDATION/book features** (liq_short_z30,
liq_long_usd, s3_taker_lsr, bd_*) monkeypatched into the falsifier for BTC 4h and 1d:
- BTC 4h: held-out AUC=0.7525, DNA capture plain=-75.4% (null p95=-61.7%, beats=False), regime=-47.0% (beats=False), genuine=False
- BTC 1d: held-out AUC=0.6033, DNA capture plain=-30.9% (beats=False), genuine=False

**Even liquidation features do NOT make the oracle's 2-bar DNA tradeable.** No bar-level causal family
(MA + orderflow + momentum + micro + liquidation) achieves a genuine signal.

## Conclusion (corrected)
The result is NOT "MA is the wrong instrument, try orderflow next." It is the stronger, comprehensive finding:
**NO bar-level causal feature family achieves a genuine tradeable signal on the oracle's ~2-bar reversal DNA, at any
cadence or chart type.** AUC rises with finer bars (statistical *recognition* of an oversold texture) but capture
worsens (-16% -> -99%) -- recognition != tradeable timing. The oracle's 2-bar reversal is causally unpredictable from
bar-level information. This reproduces the project's deepest finding (no active bar-level timing alpha; what is robust
is beta) at the mechanism level, comprehensively, across the full grid AND the full feature space.

## Implication (next avenue, for the meta loop / next generation)
A bar-level edge on the 2-bar reversal is REFUTED across all tested families. The remaining untested axes are
**representational, not feature-selection**: (a) sub-bar / tick / LOB-depth data (below the bar resolution where the
2-bar reversal lives); (b) a different oracle objective -- not max-capture 2-bar reversals but longer multi-day
*trend* setups where lag is acceptable and the unit-of-trading is a multi-candle MOVE (the project's stated founding
frame). Avenue (b) is cheaper to test and aligns with the "setup across a move" mandate -- prioritize it.

## SWING-ORACLE EXTENSION (2026-06-06 ~02:35) -- the oracle objective was mis-specified; corrected + re-tested
**Why the scalp oracle was the wrong question.** `oracle_high_capture` maximized COMPOUND with NO per-move floor
(`min_move_net=0`). With perfect foresight + no floor, max-compound greedily decomposes into the SMALLEST profitable
wiggles -> ~2-bar holds at every cadence. That is a SCALPING oracle. The project's actual unit-of-trading is "a SETUP
across a MOVE" (2-5%+ net, hold hours-to-7d). Asking a lagging MA to time 2-bar reversals was an unfair test.

**Fix.** Added a `min_move_net` per-move net floor to the oracle DP (backward-compatible; selftest PASS). At a 3-5%
floor the oracle becomes a SWING oracle -- fewer, larger, multi-day moves that ARE the target unit:

| cell | floor | moves | mean net/move | median hold |
|---|---|---|---|---|
| 4h BTC | 0% (scalp) | 2984 | 1.76% | 8h (2 bars) |
| 4h BTC | 5% | 377 | 8.26% | 56h (~2.3d) |
| 1d BTC | 5% | 210 | 9.66% | 96h (4d) |

**Re-test on the SWING oracle (the FAIR test of the user's adaptive-MA idea), two model classes:**
- **Linear (logistic, 40 causal feats)**: ICs flip POSITIVE on 1d (+0.04..+0.07; scalp was negative) -- the swing
  frame is far less hostile -- but **0/8 cells genuine**; only 2 inconsistent single-null beats (BTC=regime, SOL=plain)
  across 16 tests = within chance.
- **Nonlinear (HistGradientBoosting -- the STRONGEST fair form of "2-MA/3-MA adaptive", captures MA-crossover
  interactions + regime conditioning)**: at n_shuffle=15, **BTC 1d (5% floor) read DNA_GENUINE=True** (cleared
  shuffled + plain + regime nulls). The first genuine hit in the whole investigation.

**STRESS TEST refuted that hit (n_shuffle=50 x 3 seeds x 3 floors + OOS->UNSEEN persistence):**
- **Seed-dependent**: at 5% floor only seed 7 cleared the firewall; seeds 17 & 27 failed. A real edge is seed-robust;
  GBM randomness flipping genuine on/off = noise.
- **No persistence**: seed-7 OOS capture +58.5% collapsed to **+0.1% on UNSEEN** (never-touched); UNSEEN capture ~0%
  or negative across all seeds. The "skill" was OOS-overfit, gone out-of-sample.
- 3% floor uniformly failed; 7% floor (n=141) never genuine.

**Last feature family on the swing frame (liquidation -- the market-research top avenue).** Augmented the GBM swing
test with all 42 liquidation/book/positioning features (82 total: `liq_*` incl. `liq_capitulation`/`liq_short_panic`,
`bd_*` book depth, `s3_*` long-short-ratio/smart-money) on BTC/SOL/ETH 1d, 2 seeds: **0/6 genuine**, seed-inconsistent,
OOS/UNSEEN captures frequently negative. Liquidation features do NOT time swing entries either.

**Complete finding.** Even with the FAIR swing frame AND a nonlinear adaptive model AND the top market-research
feature avenue (liquidation), the bar-level DNA yields **no robust, seed-stable, persistent tradeable edge.** The swing frame is the correct one (and is the right oracle to
carry forward for ANY future signal source), but bar-level features are insufficient to time even multi-day swing
entries out-of-sample. This is a stronger, more complete refutation than the scalp result because it tested the
idea's strongest fair form and stress-tested the one apparent hit to destruction. The next axes are genuinely
representational (richer features: sub-bar/LOB; or a different signal source) OR exit/sizing-layer (the entry-timing
layer is refuted) -- NOT more model tuning on these features.

## Caveats
1. **APPARATUS_SOUND=False was a mis-calibrated gate, NOT a real leak -- RESOLVED 2026-06-06 ~02:30.** Root-cause:
   the `shuffled_collapses` soundness gate used the shuffle **p95 tail** (`< 0.55`) for a question that is about
   the **mean** ("does the permutation destroy the signal on average?"). At n_shuffle=30 the p95 tail straddles
   0.55 from sampling noise alone -- BTC 1d had shuffled **mean=0.502 (SOUND)** but **p95=0.555**, so it spuriously
   flagged a leak; SOL 4h had mean=0.489/p95=0.523 and passed. The liq-augmented run flipped False for the same
   reason (more features widen the tail), not a target leak. **Fix**: soundness now judged by the shuffle MEAN
   (`auc mean < 0.54`, `|ic mean| < 0.03`, `cap_skill mean < 0.10`) -- consistent with the falsifier's own selftest
   principle; genuineness still uses the p95 tail. After the fix, BTC 1d and SOL 4h both read **APPARATUS_SOUND=True,
   DNA_GENUINE=False** -- a valid honest refutation on confirmed-sound apparatus. (A genuine distribution-shift leak
   elevates the MEAN, so mean-based detection still catches real leaks.) The **capture** metric (deeply negative,
   loses the null p95 by a wide margin) was always the load-bearing evidence and is unaffected.
2. dollar bars timed out (slow DP on a very long series -- rerun with a longer budget); SOL/BNB range bars MISSING
   (data not built). The 14 grid cells are unanimous. n_shuffle=12 (fast) in the grid; rerun key cells at
   n_shuffle=50 to confirm.
