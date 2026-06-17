# Adaptive-MA (expert rig) — Honest Report

**Status: REFUTED.** The adaptive moving-average mechanism is fully implemented per
`docs/ADAPTIVE_MA_BRIEF_2026_06_05.md`, is verified causal (no look-ahead), and is wired to the kept
apparatus (`ChimeraLoader.load(sym,'1d')` over u100 + `src/strat/fill_model.py`, taker **0.0024**). On
held-out data it **does not beat either baseline the brief requires it to beat** — neither a fixed-config
MA nor a cost-matched random-entry null. A refutation is a valid, valuable outcome; it is logged here.

## What was built (mechanism)
`adaptive_ma.py` — all per-asset and CAUSAL (every feature `.shift(1)`; assembly verified causal):
1. **Causal rolling features:** realized vol (rolling std of log-returns, `RV_WIN=20`), trend strength
   (Kaufman Efficiency Ratio, `ER_WIN=20`, in [0,1]), cross-sectional dispersion (universe per-date
   return std, market-state feature).
2. **Causal self-normalization:** each feature → trailing rolling percentile (`PCT_WIN=252`, past-only,
   per-asset). No full-sample standardization, no TRAIN/VAL leakage into the rank.
3. **Deterministic map** `(trend_regime ∈ {chop,mod,trend}) × (vol_high ∈ {0,1}) → (fast,slow,type)`:
   trend→fast EMA (8/21), mod→SMA (10/30), chop→slow SMA (20/50); high-vol widens one notch. Constants
   fixed up-front, before seeing held-out.
4. **Adapted MA assembly** by per-bar selection from a table of pre-computed past-only MA series
   (selection index is past-only ⇒ assembled column is past-only).
5. **Entry:** LONG-ONLY adapted-fast > adapted-slow crossover, next-bar-open fill. **Exit:** ONE uniform
   policy (opposite cross). Cost = taker 0.0024 via the canonical harness, cross-checked through
   `src/strat/fill_model.py`.

## Headline numbers (all RWYB-reproduced under current code, 2026-06-05)

`run_u100.py --firewall` → `results_u100.json` (69/77 u100 assets with >400 daily bars):

| window | ADAPT mean comp | FIX mean comp | ADAPT med | FIX med | A>F assets | ADAPT exp% | FIX exp% |
|--------|----------------:|--------------:|----------:|--------:|:----------:|-----------:|---------:|
| TRAIN  | +1095.92% | +3026.60% | +56.66% | +123.78% | 16/69 | +13.79 | +24.23 |
| VAL    |   −3.78%  |   −9.16%  | −13.10% |  −18.20% | 38/69 |  +0.76 |  +0.14 |
| OOS    |   −2.92%  |   −3.64%  | −11.09% |  −19.31% | 40/69 |  +0.05 |  −0.36 |
| **UNSEEN** | **−11.27%** | **−7.95%** | −23.38% | −21.39% | **34/69** | −2.09 | −2.50 |

- **UNSEEN sign test (adaptive comp > fixed comp): 34/69 decisive, two-sided p = 1.0**, mean diff −3.32pp,
  median diff −0.02pp → adaptation is **indistinguishable from / slightly worse than** a fixed 10/30 SMA
  on the held-out segment. It does **not** earn its keep.
- **FIREWALL (cost-matched random-entry null, 200 books/asset): 0/69 assets** beat the null AND stay
  positive on held-out (OOS+UNSEEN). The adapted entry **timing adds nothing over random entry** at the
  same trade count, durations, and cost → **beta-in-disguise**, not a timing edge.

## −k falsifiers (the brief mandated these)

- **Look-ahead** — `adaptive_ma.causal_selfcheck`: re-derives the adapted columns from each truncated
  prefix `df[:t+1]` and compares to the full-series value at t. **max_abs_diff = 0.0 over 41 sampled bars
  ⇒ causal_ok=True.** No look-ahead.
- **Cost** (`fill_model_check.py`, taker via `src/strat/fill_model.py`): wiring proof
  `max|fill_model taker − spec-cost compound| ≈ 0`; adaptive held-out UNSEEN mean = **−18.93%** under the
  realistic taker fill. (OOS +62% mean is a single-asset ZEC +583% outlier — the per-asset firewall, not
  a pooled mean, is the decisive test, and that is 0/69.)
- **`falsifiers.py`** (`falsifiers.json`, 25-asset subset):
  - **F-A exit-policy sensitivity** (time-stop max_hold=20 instead of opposite-cross): **0/25** beat the
    null on held-out → the negative is not an artifact of the opposite-cross exit. Confirms the brief's
    suspicion that the exit choice does not hide a timing edge.
  - **F-C beta benchmark** (beats costless beta-matched hold on held-out): **2/25** → no broad excess.
  - **F-B adaptive vs constituent fixed configs** (pooled held-out per-trade expectancy): adaptive
    +1.96% is **worse than 5 of 6** constituent fixed configs (e.g. 15/40 SMA +4.69%, 20/50 SMA +3.26%) —
    switching among the configs **destroys** value vs just holding the better fixed one.
  - **F-D cost**: taker +13.5% vs ideal +15.65% (means, outlier-skewed); medians −17.97% / −16.90% →
    even at near-zero cost the median asset is negative; cost is not the sole killer — there is no
    underlying timing signal to rescue.

## Why it fails (mechanism diagnosis)
Adapting the MA *windows* changes *which* crossovers fire and *how many* trades you take, but a MA cross
on daily bars has **no held-out entry-timing edge to begin with** — so re-selecting among edge-less
configs by a causal regime label cannot manufacture one. Adaptive simply trades **more** (UNSEEN n=354 vs
fixed 227) at a slightly-less-negative per-trade expectancy, which compounds to a **worse** held-out
return. The firewall (0/69) is the clean proof: random entries at the same count/duration/cost are not
beaten on any held-out window for any asset.

## RWYB — exact reproduction commands
```
python src/strat/selftest_all.py                                  # apparatus sound: 4/4 PASS (gate has power)
python experiments/adaptive_ma/expert/run_u100.py --firewall      # full u100: 0/69 firewall, signtest p=1.0
python experiments/adaptive_ma/expert/run_u100.py --quick         # 20-asset fast reproduction
python experiments/adaptive_ma/expert/fill_model_check.py --n 12  # fill_model.py taker wiring proof (~0)
python experiments/adaptive_ma/expert/falsifiers.py --n 25        # F-A/B/C/D falsifiers
python -c "import sys;sys.path.insert(0,'src');sys.path.insert(0,'experiments/adaptive_ma/expert');\
import pandas as pd,adaptive_ma as A;from pipeline.chimera_loader import ChimeraLoader;\
g=ChimeraLoader().load('ETHUSDT',cadence='1d');\
df=pd.DataFrame({'date':pd.to_datetime(g['date'].to_list()),'open':g['open'].to_numpy().astype(float),\
'high':g['high'].to_numpy().astype(float),'low':g['low'].to_numpy().astype(float),'close':g['close'].to_numpy().astype(float)});\
print(A.causal_selfcheck(A.compute_features(df)))"   # causal_ok=True, max_abs_diff=0.0
```

## Verdict
The adaptive-MA mechanism is **sound as engineering** (causal, leakage-free, cost-honest, wired to the
real apparatus) but **REFUTED as alpha**: held-out, after-cost, it beats neither a fixed-MA baseline
(34/69, p=1.0) nor a cost-matched random-entry null (0/69). Consistent with the project's standing
premise that daily-bar long-only MA timing has no robust held-out edge — adapting the window does not
change that. Files: `adaptive_ma.py`, `run_u100.py`, `falsifiers.py`, `fill_model_check.py`,
`results_u100.json`, `results_quick.json`, `falsifiers.json`, `fill_model_check.json`.
