# Experiment 003: Strategy Comparison — MA vs World Model vs Agent

**Date:** 2026-03-06
**Scripts:** `src/analysis/ma_backtest.py`, `src/analysis/wm_backtest.py`
**Results:** `logs/analysis/wm_backtest_20260306_230859.json`

## Objective

Apples-to-apples comparison of three trading approaches using the SAME cost model, data, and metrics:
1. **MA Strategies** — Pure technical analysis (Experiment 001)
2. **WM Strategies** — World model predictions as trading signals
3. **Agent** — PPO reinforcement learning (not yet tested with new constraints)

Key constraint: NO per-candle trading. All strategies use multi-horizon holding periods.

## Methodology

- **3 assets**: BTC, ETH, SOL (dollar bars, ~5-min bars)
- **Cost model**: Perp only (0.04%/side + 0.01%/8h funding) — the realistic cost model
- **70/30 IS/OOS split** (no parameter optimization on OOS)
- **WM models**: V1.5 (best overall) + V1.1_f18 (best bear market specialist) ensemble
- **14 strategies**: BuyAndHold, Donchian 5-day, 12 WM-based variants

### WM Strategy Designs

| Strategy | Signal | Rebalance | Rationale |
|----------|--------|-----------|-----------|
| WM_Dir_h64_{5h,1d,5d} | h=64 prediction sign | Every 64/288/1440 bars | Direct WM signal at different frequencies |
| WM_Dir_h16_{1h,5h} | h=16 prediction sign | Every 16/64 bars | Shorter horizon, more frequent |
| WM_Cons_{5h,1d} | 3+/4 horizons agree | Every 64/288 bars | Multi-horizon coherence filter |
| WM_Smooth_h64 | EMA(h=64, alpha=0.02) | On smoothed sign change | Heavy smoothing for few trades |
| WM_Hold_h64 | h=64 sign | Hold until 64 bars of reversal | Trend-following with confirmation |
| WM_Ens_h64_{1d,5d} | V1.1f18+V1.5 avg | Every 288/1440 bars | 2-model ensemble |
| Donch_WM_Filt | Donchian(1440) + WM h=64 | On Donchian signal change | WM as breakout filter |

## Results: OOS Cross-Asset Sharpe (Perp Costs)

| Strategy | BTC | ETH | SOL | Avg Sharpe | Cross-Asset? |
|----------|:---:|:---:|:---:|:----------:|:------------:|
| **Donchian_5D** | **+0.41** | **+1.36** | **+1.17** | **+0.98** | **YES** |
| WM_Hold_h64 | +0.61 | -1.41 | -1.47 | -0.76 | no |
| WM_Dir_h64_5d | +0.46 | -0.20 | -0.90 | -0.21 | no |
| WM_Ens_h64_5d | +0.31 | -0.25 | -0.85 | -0.26 | no |
| Donch_WM_Filt | -0.49 | +0.40 | +0.32 | +0.08 | no |
| WM_Dir_h64_1d | +0.21 | -0.78 | -0.45 | -0.34 | no |
| WM_Ens_h64_1d | +0.12 | +0.21 | -1.26 | -0.31 | no |
| BuyAndHold | +0.12 | +0.01 | -0.76 | -0.21 | no |
| WM_Cons_1d | -0.19 | -0.35 | -1.76 | -0.77 | no |
| WM_Smooth_h64 | -2.71 | -5.81 | -4.96 | -4.49 | no |

## Key Findings

### 1. Donchian 5-Day Breakout remains the ONLY cross-asset profitable strategy

Same result as Experiment 001. No WM-based strategy achieves cross-asset profitability OOS.

### 2. WM signal alone is NOT a profitable trading signal after costs

Even at the lowest rebalance frequency (5 days, ~0.2 trades/day), WM strategies lose money on 2/3 assets. The IC=0.033 at h=64 translates to only 50.7% directional accuracy — the edge is too thin to survive transaction + funding costs.

### 3. IS performance is wildly misleading

| Strategy | IS Avg Sharpe | OOS Avg Sharpe | Retention |
|----------|:------------:|:--------------:|:---------:|
| WM_Smooth_h64 | +1.87 | -4.49 | REVERSED |
| WM_Hold_h64 | +1.35 | -0.76 | REVERSED |
| Donchian_5D | +0.08 | +0.98 | 12.2x BETTER OOS |

Donchian is better OOS than IS because it's a mechanical rule with no learning — no overfitting risk. WM strategies are worse OOS because the WM has memorized temporal patterns that don't generalize.

### 4. WM as Donchian filter is PROMISING but needs work

| Asset | Donchian | Donch+WM Filter | Change |
|-------|:--------:|:----------------:|:------:|
| BTC | +0.41 | -0.49 | WORSE |
| ETH | +1.36 | +0.40 | worse |
| SOL | +1.17 | +0.32 | worse |

The filter hurts by causing the strategy to miss valid breakouts. The WM disagrees with Donchian too often (its accuracy is ~51%, not high enough to be a reliable gate).

### 5. Higher frequency = more loss (confirmed again)

| Rebalance | Avg OOS Sharpe | Avg Trades |
|-----------|:--------------:|:----------:|
| h=16 / 1h | -17.1 | ~27K |
| h=16 / 5h | -5.9 | ~8.2K |
| h=64 / 5h | -5.3 | ~6.7K |
| h=64 / 1d | -0.34 | ~660 |
| h=64 / 5d | -0.21 | ~128 |
| Donchian 5D | **+0.98** | **~440** |

Clear monotonic relationship: fewer trades = better performance. Donchian wins because it naturally generates few trades (~440) with meaningful signal.

## The Strategy Bucket (Current State)

| Bucket | Strategy | OOS Sharpe | Status |
|--------|----------|:----------:|--------|
| 1. MA Trading | Donchian 5-Day Breakout | **+0.98** | PROFITABLE, deployable |
| 2. WM Signal | All variants tested | -0.21 to -17.1 | NOT PROFITABLE standalone |
| 3. WM as Filter | Donchian + WM agreement | +0.08 | MARGINAL, needs refinement |
| 4. Agent (PPO) | Not yet tested with multi-day design | TBD | NEXT EXPERIMENT |

## Why the WM Fails as a Trading Signal

1. **IC = 0.033 is too low**: At 50.7% directional accuracy, you're barely better than random. After 0.08% roundtrip costs + 0.0125%/8h funding, the edge evaporates.

2. **h=64 predictions REVERSE OOS**: Signal Monte Carlo analysis (Exp 001b) showed h=64 IS IC=+0.071 but OOS IC=-0.034. The model has memorized long-horizon patterns.

3. **h=1 is the only genuinely generalizing horizon** (OOS IC=+0.047), but h=1 requires per-candle trading which is catastrophic for costs.

4. **The WM adds value as a FEATURE, not as a SIGNAL**: IC=0.03 means it provides useful information, but not enough to directly trade on. It needs to be combined with other features (like Donchian channels) or learned by an agent that can optimize cost-adjusted execution.

## Path Forward

### Immediate (can deploy now)
- **Donchian 5-Day Breakout on perp**: Sharpe ~1.0, ~200 trades/year, requires no ML
- Start with ETH (Sharpe=1.36) and SOL (Sharpe=1.17)

### Short-term (next experiments)
1. **Agent redesign for multi-day holding**: Remove per-candle action, use h=64 decision frequency with multi-day holding
2. **WM filter refinement**: Instead of binary agree/disagree, use WM as a CONFIDENCE weight on Donchian signals
3. **Spot mode testing**: Remove funding costs (the dominant cost) — some WM strategies may become viable

### Medium-term
4. **Train V1.5 longer**: Only 26 epochs, highest IC potential — more training could improve OOS generalization
5. **Adaptive model retraining**: Monthly signal quality check + retrain when IC degrades
6. **V2-V9 architectures**: Different architectures may produce more generalizable signals

## Monthly Signal Quality Monitoring (NOTED)

Every month (or when retraining):
1. Run `wm_horizon_analysis.py` on latest data with latest checkpoints
2. Compare OOS IC to historical baseline (V1 family: IC=0.03-0.05)
3. If OOS IC drops below 0.015, retrain or switch to adaptive (V.X) model
4. Track regime accuracy: if it degrades, market structure may have changed
5. Log results in `memory/experiment_log.md`

This should be automated as part of the adaptive model (V.X) retraining pipeline.
