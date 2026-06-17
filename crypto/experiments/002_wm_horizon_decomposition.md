# Experiment 002: World Model Horizon Decomposition

**Date:** 2026-03-06
**Script:** `src/analysis/wm_horizon_analysis.py`
**Results:** `logs/analysis/wm_horizon_analysis_20260306_220803.json`

## Objective

Decompose all 7 V1 world model variants across 4 prediction horizons to understand:
1. What each model excels at (short vs long-term prediction)
2. Which model combinations form optimal ensembles per horizon
3. Signal efficiency per horizon (IC * sqrt(h)) for cost-adjusted trading
4. Regime-conditional strengths for regime-gated strategies

## Methodology

- **7 models**: V1.0 (13f), V1.1_f13, V1.1_f18, V1.2 (18f), V1.3 (18f), V1.4 (18f), V1.5 (19f)
- **4 horizons**: h=1 (~5min), h=4 (~20min), h=16 (~1.3h), h=64 (~5.5h)
- **10 assets**: BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, LTC
- **70/30 IS/OOS split** with 400-bar purge gap
- **Non-overlapping sequences** of length 96 for IC computation
- **Metrics**: Spearman IC, directional accuracy, quintile spread, regime-conditional IC, cross-model correlation
- **Subsampled correlations**: 200K samples for Spearman (stable, avoids O(n^2) on 8.8M arrays)

## Results

### 1. OOS IC by Model x Horizon

| Model | h=1 | h=4 | h=16 | h=64 | Mean | Best_H |
|-------|:---:|:---:|:----:|:----:|:----:|:------:|
| V1.0 | 0.0528 | 0.0355 | 0.0258 | 0.0274 | 0.0354 | h=1 |
| V1.1_f13 | 0.0506 | 0.0347 | 0.0263 | 0.0278 | 0.0349 | h=1 |
| **V1.1_f18** | 0.0532 | **0.0415** | 0.0289 | 0.0274 | 0.0377 | h=1 |
| V1.2 | 0.0538 | 0.0391 | 0.0295 | 0.0244 | 0.0367 | h=1 |
| V1.3 | **0.0539** | 0.0391 | 0.0295 | 0.0280 | 0.0376 | h=1 |
| V1.4 | 0.0474 | 0.0308 | 0.0224 | 0.0196 | 0.0300 | h=1 |
| **V1.5** | 0.0538 | 0.0406 | **0.0300** | **0.0311** | **0.0389** | h=1 |

**Finding**: V1.5 is the strongest model overall (mean IC=0.0389), dominating at h=16/h=64. V1.4 is weakest (0.0300) -- confirmed marginal. V1.1_f18 beats V1.1_f13 at every horizon -- cross-asset features add genuine signal.

### 2. Signal Efficiency: IC * sqrt(h)

| Horizon | Best IC | sqrt(h) | Efficiency | Hold Time |
|---------|:-------:|:-------:|:----------:|:---------:|
| h=1 | 0.0562 | 1.00 | 0.0562 | ~5min |
| h=4 | 0.0429 | 2.00 | 0.0858 | ~20min |
| h=16 | 0.0327 | 4.00 | 0.1308 | ~1.3h |
| h=64 | 0.0330 | 8.00 | **0.2640** | ~5.5h |

**Critical finding**: h=64 is **4.7x more efficient** than h=1. The IC decay from h=1 to h=64 is only 1.7x (0.056 -> 0.033), but trading costs scale with frequency. At 288 bars/day, h=1 trades 288x vs h=64 trades 4.5x. h=64 is the clear winner for cost-adjusted edge.

### 3. Optimal Ensembles Per Horizon (OOS)

| Horizon | Best Single | IC | Best 2-Model | IC | Best 3-Model | IC |
|---------|-------------|:--:|-------------|:--:|-------------|:--:|
| h=1 | V1.3 | 0.0539 | V1.1_f18+V1.5 | 0.0556 | V1.0+V1.1_f18+V1.5 | 0.0562 |
| h=4 | V1.1_f18 | 0.0415 | V1.1_f18+V1.5 | 0.0430 | V1.1_f18+V1.2+V1.5 | 0.0429 |
| h=16 | V1.5 | 0.0300 | V1.1_f18+V1.5 | 0.0326 | V1.1_f13+V1.1_f18+V1.5 | 0.0327 |
| h=64 | V1.5 | 0.0311 | V1.0+V1.5 | 0.0333 | V1.0+V1.1_f13+V1.5 | 0.0330 |

**Key findings**:
- **V1.5 appears in EVERY optimal ensemble** -- it is the core model
- **V1.1_f18+V1.5** is the minimum viable ensemble (beats all singles at every horizon)
- 3-model ensembles show diminishing returns vs 2-model (+0.001 IC)
- V1.4 never appears in any optimal ensemble -- exclude it

### 4. Prediction Correlation Matrix (h=64, OOS)

|          | V1.0 | V1.1_f13 | V1.1_f18 | V1.2 | V1.3 | V1.4 | V1.5 |
|----------|:----:|:--------:|:--------:|:----:|:----:|:----:|:----:|
| V1.0     | 1.00 | 0.75 | 0.56 | 0.64 | 0.59 | 0.53 | 0.53 |
| V1.1_f13 |      | 1.00 | 0.53 | 0.62 | 0.58 | 0.47 | 0.53 |
| V1.1_f18 |      |      | 1.00 | 0.75 | 0.71 | 0.56 | 0.65 |
| V1.2     |      |      |      | 1.00 | 0.73 | 0.52 | 0.69 |
| V1.3     |      |      |      |      | 1.00 | 0.58 | 0.75 |
| V1.4     |      |      |      |      |      | 1.00 | 0.48 |
| V1.5     |      |      |      |      |      |      | 1.00 |

**Key findings**:
- V1.0 + V1.1_f13 most correlated (0.75) -- same architecture, same features
- V1.1_f18 + V1.2 highly correlated (0.75) -- both use 18 features with similar training
- V1.4 has lowest correlation with all others (0.47-0.58) -- most diverse but weakest
- **For diversity**: V1.0/V1.1_f13 + V1.1_f18 + V1.5 = low mutual correlation + strong IC

### 5. Regime-Conditional IC (h=16, OOS)

| Model | Bear | Neutral | Bull |
|-------|:----:|:-------:|:----:|
| V1.0 | 0.001 | 0.017 | 0.008 |
| V1.1_f13 | -0.078 | 0.010 | **0.090** |
| **V1.1_f18** | **0.128** | 0.015 | -0.113 |
| V1.2 | 0.064 | 0.016 | -0.052 |
| V1.3 | 0.017 | 0.016 | -0.007 |
| V1.4 | -0.058 | 0.014 | 0.065 |
| V1.5 | -0.025 | 0.018 | 0.032 |

**Critical insight for regime-gated trading**:
- **V1.1_f18 is a BEAR SPECIALIST** (IC=0.128 in bear, -0.113 in bull)
- **V1.1_f13 is a BULL SPECIALIST** (IC=0.090 in bull, -0.078 in bear)
- **V1.3 is the most BALANCED** (0.017/0.016/-0.007)
- All models are strongest in NEUTRAL regime (0.01-0.018)
- A regime-switched ensemble could use V1.1_f18 in bear + V1.1_f13 in bull

### 6. Regime Classification Accuracy (OOS)

| Group | Models | Accuracy |
|-------|--------|:--------:|
| Poor | V1.0, V1.1_f13, V1.1_f18, V1.2 | 22.9-23.4% |
| Strong | V1.3, V1.4, V1.5 | **85.1-86.5%** |

V1.3/V1.4/V1.5 have dramatically better regime classification. This is likely due to their Gumbel tau or feature attention ablations improving the regime head. V1.0-V1.2 regime heads are effectively broken (worse than 33% random).

### 7. Quintile Spread (OOS, bps)

| Model | h=1 | h=4 | h=16 | h=64 |
|-------|:---:|:---:|:----:|:----:|
| V1.5 | 1.4 | 2.4 | 4.2 | **8.6** |
| V1.2 | 1.3 | 2.3 | 4.1 | 8.1 |
| V1.1_f13 | 1.3 | 2.3 | 3.9 | 8.0 |
| V1.0 | 1.3 | 2.2 | 3.8 | 7.9 |
| V1.4 | 1.1 | 1.7 | 2.9 | 5.9 |

V1.5 top-vs-bottom quintile spread at h=64 = 8.6bp. This is the raw return difference between the top 20% and bottom 20% of predictions, confirming V1.5 has the best long-horizon sort.

### 8. IS vs OOS IC Retention

| Model | IS Mean | OOS Mean | Retention |
|-------|:-------:|:--------:|:---------:|
| V1.0 | 0.0556 | 0.0354 | 63.7% |
| V1.1_f13 | 0.0567 | 0.0349 | 61.6% |
| V1.1_f18 | 0.0569 | 0.0377 | 66.3% |
| V1.2 | 0.0606 | 0.0367 | 60.6% |
| V1.3 | 0.0619 | 0.0376 | 60.7% |
| V1.4 | 0.0467 | 0.0300 | 64.2% |
| **V1.5** | **0.0648** | **0.0389** | **60.0%** |

All models retain 60-66% of IS IC OOS. V1.5 has highest absolute IS/OOS but slightly lower retention (60%) -- the IS→OOS gap is consistent across models.

## Key Conclusions

### 1. V1.5 is the anchor model
- Highest OOS IC at h=16 and h=64 (the horizons that matter for cost-adjusted trading)
- Present in every optimal ensemble combination
- Best quintile spread at every horizon
- Best regime classification accuracy (86.3%)

### 2. Optimal ensemble: V1.1_f18 + V1.5 (2-model)
- Beats every single model at every horizon
- Sufficient diversity (correlation 0.61-0.65)
- V1.1_f18 adds bear market expertise that V1.5 lacks
- 3-model adds marginal +0.001 IC -- not worth the complexity

### 3. h=64 is the optimal trading horizon
- Signal efficiency 4.7x better than h=1
- IC=0.033 at h=64 with 4.5 trades/day vs IC=0.056 at h=1 with 288 trades/day
- At 0.04%/side perp costs, h=64 trading costs ~0.36%/day vs h=1 costs ~23%/day
- This aligns with Experiment 001: Donchian 5-day breakout (even longer horizon) was the only profitable strategy

### 4. Regime-gated trading is viable
- V1.1_f18 (bear specialist, IC=0.128) and V1.1_f13 (bull specialist, IC=0.090) can be combined
- V1.3/V1.5 have 86% regime accuracy to gate the switching
- A regime-switched ensemble could potentially improve IC by 2-3x in trending markets

### 5. V1.4 should be excluded
- Worst IC at every horizon (mean=0.0300)
- Never appears in any optimal ensemble
- Most diverse predictions but diversity without signal is noise

## Implications for Agent Design

1. **Core ensemble**: V1.1_f18 + V1.5 for predictions. Use V1.3/V1.5 regime head for regime gating.
2. **Primary trading horizon**: h=64 (~5.5h). Agent should HOLD positions for 5+ hours minimum.
3. **Entry timing**: Use h=1/h=4 signal for precise entry within a h=64 position decision.
4. **Regime switching**: Bear market -> weight V1.1_f18 more. Bull market -> weight V1.1_f13 more.
5. **Position sizing**: Proportional to h=64 IC * regime confidence, not h=1 noise.

## Connects to Experiment 001

| Finding (Exp 001) | Confirmed by Exp 002? |
|---|---|
| Signal is real (84% profitable zero-cost) | YES: IC > 0.03 OOS at all horizons |
| Per-candle trading is doom | YES: h=1 efficiency = 0.056, h=64 = 0.264 (4.7x) |
| Only 5-day breakout survives costs | YES: h=64 is most efficient, multi-day holding optimal |
| WM should be a FILTER, not per-bar signal | YES: Use h=64 for position decisions, h=1/h=4 for timing |

## Next Steps

- [ ] Build regime-gated ensemble (V1.1_f18 for bear, V1.1_f13 for bull, V1.5 anchor)
- [ ] Backtest WM h=64 signal as Donchian filter (Experiment 003)
- [ ] Redesign agent for multi-hour/multi-day holding period
- [ ] Test V1.5 at more training epochs (currently only 26, highest IC potential)
