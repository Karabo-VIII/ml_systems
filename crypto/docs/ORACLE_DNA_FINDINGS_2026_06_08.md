# Oracle-Decomposer DNA findings (2026-06-08) — the MA driver's capture is REGIME-driven, not asset-feature-driven

**Pipeline:** `src/oracle/ma_oracle_engine.py` (hindsight oracle: best MA-TI + entry-day per top-25 performer) →
`src/oracle/decomposer.py` (rolling-validity driver + 47 chimera v50/51 features + chart types) → DNA panel
(`runs/oracle/dna_panel.parquet`, 235 (asset,date) records across 16 dates 2024-02 → 2026-05, u50).
**Status:** HINDSIGHT hypothesis-generation. Overseer-RWYB (numbers re-computed independently).

## The finding (verified)
For each top-performer move we asked: **which entry-day chimera feature predicts whether the MA driver captures
WELL vs POORLY?** Answer — **essentially none at the asset level.**

- 47 features ranked by class-separation (GOOD = capture_rate tercile ≥0.286 vs POOR ≤0.0): **max |AUC−0.5| = 0.117,
  median 0.031; only 1/47 clears the ~2σ noise bar** — that is what pure noise across 47 tests produces.
- The single "winner" `xd_btc_volatility` (raw AUC 0.606) **collapses to AUC 0.533 when date-demeaned** — its power
  is a **per-DATE / cohort effect** (high-BTC-vol *days* were good-capture days for the whole top-25 cohort), NOT a
  within-cohort asset-level discriminator. Date-demeaning collapses every other candidate to ~0.5 too.

**Conclusion:** whether a given top-performer's MA crossover captures its move well is governed by the **market
regime of the day** (a cohort-wide effect) far more than by any entry-day chimera feature of that asset. The
per-asset `norm_*` micro-features (vpin, flow, funding, hawkes, efficiency, kyle-lambda, …) do **not** predict
capture quality at this resolution. The hindsight DNA — which *over*-states — already declines to support them.

## What this REDIRECTS (the value of a clean negative)
This is not "no edge" (defeatist). It is a **precise redirect** of the predictive path:
- **DO NOT** build a per-asset chimera-feature gate on the MA driver — the DNA says it won't predict capture.
- **DO** condition the MA driver on **MARKET REGIME** — a regime overlay (ride the driver in trending/high-BTC-vol
  regimes, de-weight in chop). This is exactly classic **trend-following / managed-futures**, and it coheres with
  the 2026-06-08 reframe (pure returns via an adaptive process that **beats passive holding** at lower drawdown).

## The hypothesis to VALIDATE next (held-out, not hindsight)
**H1 (regime overlay):** conditioning the adaptive-MA driver on a BTC-vol / trend regime improves *realized,
held-out* capture (or Calmar) vs an always-on driver — tested through `src/strat/candidate_gate` against the
**beta-matched passive benchmark** (`benchmark_excess`, `bear_preserved`), 10-seed, UNSEEN-once. Honest caveat:
hindsight DNA over-states; even H1's within-date discrimination is marginal (AUC ~0.53) — H1 is a *regime/timing*
switch, not a per-asset signal, and must earn its keep OOS or be reported as null.

Artifacts: `runs/oracle/dna_panel.parquet` (corpus), `runs/oracle/dna_ranked_features.csv` (47 ranked).

## H1 validation (held-out) — NULL on alpha; capital-preservation is real but confounded
Tested (`runs/staging/h1_regime_overlay_2026_06_08.py`, overseer-RWYB vs the result JSON): a long-only SMA(30/50)
driver on u10, with a BTC-regime overlay (BTC>SMA200 [±vol]), 3 arms through `candidate_gate`. Held-out (mean-asset
compound, net taker): UNSEEN A(always-on) **-31.0%**, B(overlay) **0.0%**, C(passive) **-28.5%**.

**Two honest truths (the overseer caught a definitional fork):**
1. **No ALPHA.** (All numbers in this item **VERIFIED** — overseer-RWYB vs `h1_regime_overlay_result_2026_06_08.json`.)
   The bare MA driver itself FAILS the gate (UNSEEN -31%; battery p05 -99.5; `beats_beta_held` 1/10;
   10-seed 0/10) — no standalone edge. The overlay only "helps" by going to **0% risk-on (full cash) in UNSEEN**, and
   in OOS (the one risk-on held window, 68% on) it was *marginally worse* than always-on (no asset improved). Under the
   gate's EXPOSURE-MATCHED passive (f=0 -> 0% baseline) the overlay **ties cash -> `beats_beta_held` 0/10 = NULL**.
2. **DD-control is real under the user's reframe, BUT confounded.** (Numbers **VERIFIED** — overseer-RWYB vs result JSON.)
   Under "beat BUY-AND-HOLD at lower drawdown" (the
   2026-06-08 Calmar lens), the overlay's 0% UNSEEN *beats* buy-and-hold's -28.5% with **0 DD vs -28.5% DD** — the
   trend-following capital-preservation the reframe valued. HOWEVER it is **confounded**: UNSEEN was a *pure BTC
   downtrend* (0% risk-on all window), so this "win" is trivial bear-abstention (any `BTC<SMA200 -> cash` rule does it),
   NOT timed-trading skill — and the driver under it has no edge anyway.

**Verdict:** at 1d there is **no passive-beating *adaptive-MA* process** beyond trivial regime de-risking. To honestly
test the DD-control claim the user's reframe cares about, the next run needs: (a) **Calmar/MAR-vs-buy-and-hold** scoring
(credit DD-reduction, not exposure-matched excess); (b) a **mixed-regime UNSEEN** (not a pure downtrend); (c) a
**less-binary regime** (scale exposure, not flat-cut); and (d) a driver with *some* edge (the bare crossover has none).
This is a valid foundation result, consistent with the project's prior "no active alpha at 4h/daily" + the reframe's
own honest-ceiling clause.

## Adversarial falsification (oracle_falsify + H1_falsify_trivial) — the NULL is HONEST, the "preservation" is TRIVIAL
2026-06-08, overseer-RWYB (independent recomputation, not agent-asserted). A RED-team auditor + my own re-derivation
attacked the null on six surfaces; **none turned it into a hidden signal.** Reproduce with
`python src/oracle/verify_dna_finding.py` (exit 0 = REPRODUCED + CAUSAL; closes the prior provenance gap where the
demean number lived only in frontier.json prose).
- **Causality (no look-ahead): PASS** — data-level, all 235 panel rows have `entry_date <= query_date`,
  `days_back in [0,30]`, `peak_date >= entry_date`; the causal gate is `past_closes = closes[:d_idx+1]`
  ([ma_oracle_engine.py:299](../src/oracle/ma_oracle_engine.py#L299)). Future-spike + future-poison invariants both
  bit-identical (agent-run); 6/235 entries clip at the days_back bound (does not manufacture the null).
- **DNA finding REPRODUCED:** top feature `ctx_entry__xd_btc_volatility` raw AUC **0.6064 -> 0.5329 date-demeaned**
  (= per-DATE/cohort REGIME effect, not per-asset), 47 ctx_entry features scanned, **exactly 1/47** above
  |AUC-0.5|>0.10, median 0.0213 = the noise floor. The residual within-date signal is real-but-fragile (4 balanced
  dates) and is **entry-timing-in-the-BTC-vol-cycle** = regime/timing, reinforcing "regime, not per-asset alpha."
- **Label note (H3):** capture_rate is zero-inflated (**VERIFIED**: 108/235 = 46% exactly 0.0; tercile q33 = 0.0), so GOOD-vs-POOR
  is effectively "MA entry made vs lost money." This makes a true feature signal *easier* to find, not harder — it
  cannot create a false null.
- **H1_falsify_trivial -> REFUTED (the DD-"win" is trivial):** the H1 run already contained the control. **ARM B2 =
  classic managed-futures "BTC>SMA200 else cash"** has UNSEEN risk-on fraction **0.0** -> identical 0.0% UNSEEN to the
  full overlay B. So the overlay's capital-preservation is **the trivial trend filter alone** — the adaptive-MA driver
  and the vol-conditioning add nothing in UNSEEN. And B2 is OOS **-11.8%** mean-asset: even the trivial trend rule does
  not survive OOS. The "preservation" is bear-abstention any `BTC<SMA200->cash` rule matches, NOT timed skill.

**Net:** the oracle->decomposer->DNA->H1 arc is a VERIFIED-HONEST NULL at 1d. The depth axis (per-asset chimera
features conditioning an adaptive-MA driver) is mapped and converging null; the open value is **breadth** (resolution /
chart-type / instrument) or accepting the regime-de-risk-only ceiling — a strategic fork for the user.
