# The Oracle Framework (2026-06-10) — price oracle → TI oracle → the adaptability gap

**The canonical methodology for turning the market's *realizable move-ROI* into a *rolling, adaptable* participation
decision.** Formalized at the user's direction (2026-06-10). Every number here is VERIFIED (RWYB) from committed,
deterministic tools; re-run the command in any `runs/mining/ti_oracle_*.json` / `config_adapt_*.json` repro block.

## The shape of the framework (three stages + the gap)

```
   STAGE 1                    STAGE 2 (decomposed)                  STAGE 3
   PRICE ORACLE       -->     TI ORACLE                     -->     THE ADAPTABILITY GAP
   (what's achievable)        (what a causal TI can capture)        (can we win NEXT week?)
   the 2-10% moves            structure x MA-type x exit x period   rolling state -> decision
```

### Stage 1 — the PRICE ORACLE (the ceiling)
For an (asset, cadence, window): the compound of every zigzag up-leg whose magnitude is in **[2%, 10%]** — the
realizable "moves," captured by a perfect swing-trader. It is the **upper bound** on what any long-only rule could
bank. Tool: `oracle_anchor.py`. Key fact: the price oracle **explodes at finer cadence** (same 2-week window: ETH 1d
18% → 1h 62% → 15m 75%; DOGE 1d 15% → 15m 210%) — more, smaller swings.

### Stage 2 — the TI ORACLE (decomposed into disparate dimensions)
The **best causal TI config's** capture of those moves (hindsight-selected config; the config itself is a real
past-only rule). It is decomposed into four **independent dimensions** — *the oracle is not one answer but a
multitude*:
- **Structure:** `price>MA` · `cross+flip` · `cross+mechanical-exit (time / ATR-trail / take-profit)` · `price>fast>slow` (stack).
- **MA type:** SMA · EMA · WMA · **HMA (Hull)** · DEMA · KAMA (adaptive).
- **Period:** fast (5-13) ↔ slow (20-50).
- **Exit:** signal-flip · time-stop · ATR-trail · take-profit.

Tool: `oracle_anchor.py --decompose`. Findings: **HMA / low-lag MAs dominate** (moves are fast; lag is the enemy;
EMA is the *worst* smoother); the **multitude holds** (no structure wins >46% of windows, no MA-type >51%); the best
**structure shifts with cadence** (price>MA on 1d/trending → cross+time on 4h/choppy). The **TI oracle is ~25-90% of
the price oracle per window** (it *rises* as the TI vocabulary widens: simple-SMA ~30% → full space 0.57-0.81).

### Stage 3 — THE ADAPTABILITY GAP (the whole point)
The per-window best config is **hindsight**. The question that makes it a *strategy*: at any point in time, can we set
next week's config/participation from **observable past state**, to capture next week's moves (or raise the
probability thereof)? Decompose *what determines* the oracle config → reverse-engineer a rolling rule.

Tool: `oracle_config_adapt.py`. **The decomposition's verdict:**
1. **The config is determined by TRENDINESS** (Kaufman efficiency ratio = |net move| / Σ|bar moves| = trendline-fit
   quality). Trending → price>MA / slow / loose; choppy → cross+time / mechanical / fast.
2. **Trendiness does NOT persist week-to-week** — `er_autocorr ≈ −0.02 to −0.10` (≈0). It is the same unpredictable
   thing as *direction*.
3. **So config-adaptation is futile:** no state variable lets a state-conditioned config beat a *fixed* config on
   held-out next-week capture (`beats_fixed` 0-40%, never >50%, across 6 assets × 4 state vars); all configs are
   ~interchangeable next-week (~1%/week each ≈ ~20% of the per-week oracle's 4-7%). **Config-selection is not the lever.**
4. **The REAL adaptive lever is VOL-TIMED PARTICIPATION.** The one state that *persists* is **volatility**
   (`vol_autocorr ≈ 0.30-0.52`), and it **predicts next-week CAPTURABILITY**: high-vol-state weeks → 1.5-3× the
   next-week capturable ROI (SOL 4.6% vs 1.4%; DOGE 5.1% vs 3.3%; ETH 2.1% vs 1.3%). So the secret sauce is **"size
   up / participate when vol is high (predictable), with any reasonable fixed config"** — *not* which config.

**This is the rolling-week mechanism (user-confirmed):** each week *w*, read `vol_w` (past-only) → set participation
for week *w+1* → realize → roll forward. We participate in the market **rollingly**, sized by the persistent vol signal.

## Why it lands here (the unifying math)
Direction / trendiness is unpredictable (AUC≈0.51, er_autocorr≈0 — six ways, dead-list D14/17/44/63/67). Magnitude /
vol IS predictable (vol clusters, Hurst|ret|≈0.8). The oracle framework makes this *actionable*: the TI **config**
rides direction (unpredictable → can't adapt), but **participation** rides magnitude (predictable → can adapt). The
gap between price oracle and what we realize is closed not by a smarter indicator but by *trading more when the
market is about to move more.*

## The missing gaps (open — the forward agenda)
1. **[BUILDING NOW] Vol-timed rolling participation, net-of-cost.** Realize the Stage-3 finding: deploy a fixed
   reasonable config sized by `vol_w`, rolling weekly; compare vs uniform participation and vs the regime-beta book.
   This is the cash-out of the framework. (`oracle_config_adapt.py` → participation module.)
2. **Non-MA TI oracles.** Stage 2 is MA-only. Add RSI / MACD / Bollinger / Donchian oracles so the "multitude" spans
   indicator *families*. (Does a non-MA structure capture moves the MA family can't?)
3. **Multi-cadence integration.** Stage 1 shows the price oracle explodes finer but capture collapses finer (cost-fatal
   at taker). Open: a cadence-aware participation (coarse for cost-clearing, finer only when vol justifies the cost).
4. **The cost layer.** All oracle numbers are GROSS. Stage-3 participation must be tested net of taker 0.24% (and the
   maker-fill reality) — the realizable edge is whatever survives cost.
5. **Stronger capturability predictors.** vol→capturability is weak (corr 0.02-0.21). Open: does a *multi-feature* or
   a *longer-lookback* vol-state predict capturability better? (Bounded by the magnitude-predictability ceiling.)
6. **Participation sizing law.** Binary gate vs continuous (vol-rank) vs vol-target; the right map `vol_w → weight`.
7. **The exit dimension within participation.** Once participating, the exit (Stage-2 dimension) still matters; the
   framework currently fixes a config — co-optimizing exit × participation is open.

*Tools: `src/strat/oracle_anchor.py` (Stage 1+2), `src/strat/oracle_config_adapt.py` (Stage 3). Artifacts +
repro blocks in `runs/mining/ti_oracle_*.json` and `config_adapt_*.json`. Provenance: /orc run 2026-06-09/10.*
