# ML Config-Recommender — Design (2026-06-13)

**User ask (6h autonomous):** "design the ML approach that, at the end, gives feedback akin to the results we get
when we run the static rules and gives us the best config — on the whole-of-2020 TRAIN/VAL/OOS runway the other
instance already ran; I want that as an output."

**The honest framing (what the static-rule runway already taught us — so we don't re-mine noise):**
`deep2020_leaderboard.py` (the static rule) graded candidate long-only strategies per timeframe on the 2020 runway
(WIN 2020-07-01..2021-01-01, SPLIT 2020-10-01 → in-sample Jul-Sep, **OOS Oct-Dec**). Its honest verdict:
1. **Rank by NET (wealth), NOT Sharpe** — Sharpe rewards under-participation in a bull (a barely-invested MA family
   gets top Sharpe at ~18% net vs buy-hold ~50% [REPORTED, from deep2020_leaderboard VIDYA@4h]). The North Star is wealth.
2. **The real "best" is VOL-TARGETED BUY-HOLD** — highest/near-highest net at every TF, best Sharpe at fine TFs,
   lower maxDD, every-day coverage. MA families ALWAYS have lower net (they sit out the bull).
3. **MA-config ranking is mostly NOISE** (effective N ~1.2; ranks flip between Sharpe and net; VAL-best does not
   transfer → picking it is selection risk).

=> A naive "ML ranks MA configs by past performance" re-mines (3). For the ML to MATTER it must beat the real bars:
**VOLTGT_BH on NET**, the **static-rule pick**, and a **same-frequency SHUFFLE** (no-skill control). The genuine
skill lever is the **orthogonal features the static ranking ignored** (the deep-dive's CALENDAR signal + vol-regime).

## 1. Problem framing
Supervised **contextual recommendation / learning-to-rank**, NOT per-candle prediction (founding framing: a config
is a SETUP across a MOVE). Per rolling window t over the in-sample period: past-only features X(t) → label = the
candidate with the best realized NEXT-window NET (from the static eval). The model learns X(t) → recommended
candidate(s). At eval time it emits, per window, a ranked recommendation; we trade the top pick (or top-k book).

## 2. Candidate set (what the ML may recommend) — the HONEST superset
Reuse `deep2020_leaderboard` candidates: the 8 MA-type FAMILIES (EMA/SMA/WMA/HMA/DEMA/TEMA/KAMA/VIDYA slow books)
**+ BUYHOLD + VOLTGT_BH**. Including the non-MA winners is essential: the ML must be ABLE to recommend "just
vol-target buy-hold" when that's right (otherwise we've rigged it to lose to the real bar).

## 3. Features (past-only, causal — the skill lever)
- rolling **regime** (trend/chop/down) + **vol level / vol-regime** (drives VOLTGT vs MA choice)
- recent **per-candidate performance** over a lookback (the ρ≈0.7 persistence — but known weak alone)
- **breadth** (% of u10 above own MA), **whipsaw/turnover** (chop proxy)
- **CALENDAR** (weekend / US-hours tilt) — the deep-dive's ONLY orthogonal-to-beta signal; the feature the static
  leaderboard did NOT use → the most promising skill source
- **participation/coverage** state
All features bars ≤ t; warmup from full history; fit/standardize on TRAIN only.

## 4. Model
Start SIMPLE + honest (the session lesson: framing+features > model complexity). Two tiers:
- **Tier-A:** a shrinkage/regularized scorer (reuse `data_expansion.james_stein_shrink`) over the features → rank
  candidates. Overfit-resistant on the small 2020 sample.
- **Tier-B:** a gradient-boosted ranker (LightGBM/XGBoost if available; else sklearn GBM) for the richer interaction.
Pick the tier that wins on VAL; report both. NEVER tune on OOS.

## 5. Runway (mirror the static rule for apples-to-apples)
- **TRAIN** Jan-Aug 2020 (fit features/model) · **VAL** Sep 2020 (tune/select) · **OOS Oct-Dec 2020** (test — the
  SAME OOS as the static rule). Also run the tool's exact Jul-Sep→Oct-Dec for direct leaderboard comparability.
- Per timeframe sweep {1d,4h,2h,1h,30m,15m} (no silent single-cadence default).

## 6. OUTPUT (mirror `LEADERBOARD.md`) — the deliverable
Per TF, per split, a ranked table: the ML's recommended candidate(s) + realized **NET% / Sharpe / maxDD / coverage**,
side-by-side with (a) the **static-rule leaderboard winner**, (b) **VOLTGT_BH** (the honest bar), (c) **BUYHOLD**,
(d) the **ORACLE** (hindsight-best candidate per window = the ceiling). Plus a one-line "ML best config for OOS".

## 7. Honest controls (NON-NEGOTIABLE — the guard against false victory)
- **SHUFFLE control:** block-shuffle the ML's recommendation sequence (same candidate-frequency, random timing) — the
  real ML must beat its own shuffle, else the "skill" is just average exposure (the rolling-regime-book lesson).
- **vs VOLTGT_BH on NET** (the real bar) and **vs the static-rule pick** and **vs random-recommender**.
- **PBO** (config-overfit) + block-bootstrap **p05** + **two-sided selftest** (a genuine-skill synthetic case ships;
  a no-signal case shrinks to the prior / loses to vol-target).

## 8. The verdict it answers (honest either way)
Does the ML recommendation ADD SKILL over (a) the static-rule pick, (b) VOLTGT_BH on NET, (c) the shuffle — on 2020
OOS? If yes → first learned config-recommender with real skill; quantify + the robustness. If no → the deliverable
(the best-config-per-split table) is STILL produced, and we report honestly that the learned recommender does not beat
vol-target buy-hold (the internal-data ceiling, now tested with a proper ML, not asserted). The OUTPUT ships regardless.

Tools to reuse (don't reinvent): `deep2020_leaderboard.py` (candidates+grade+split), `data_expansion.py` (shrinkage),
`rolling_regime_book.py` (regime features + the shuffle-control pattern), `scorecard.py` (PBO/p05). New file:
`src/strat/ml_config_recommender.py`.
