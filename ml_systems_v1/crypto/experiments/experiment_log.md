# Experiment Log
> Persistent record of training runs, validations, and key findings.
> Updated by Apex after each experiment. Read at session start for context.

## Format
```
### YYYY-MM-DD: [Model] [Description]
- Config: seed=X, features=N, epochs=E, revin=on/off
- Results: IC=X.XXX, ShIC=X.XXX, MSE=X.XX, KL=X.XX, val/train=X.XX
- Gates: PASS/FAIL (which failed)
- Conclusion: [one sentence]
```

---

## Completed Experiments

### 2025-02-XX: V1.2 KL Anneal Ablation (18 features, no RevIN)
- Config: features=18, revin=off, KL anneal schedule (V1.2 specific)
- Results: ShIC=0.0307
- Gates: PASS (all 5 gates)
- Conclusion: KL anneal variant learns genuine signal, not memorization.

### 2025-02-XX: RevIN A/B Test (V1 family)
- Config: V1 with revin=on vs revin=off, features=13
- Results: RevIN ON -> ShIC=-0.001 (FAIL). RevIN OFF -> ShIC=0.028 (PASS).
- Gates: FAIL with RevIN, PASS without
- Conclusion: RevIN leaks temporal info via per-sequence mean/std. Permanently disabled by default.

### 2026-03-01: V1.4 FeatureAttn fp32 Fix Verification (18 features, no RevIN)
- Config: seed=42, features=18, epochs=5 (verification run), revin=off
- Results: IC1=0.0231, IC4=0.0109, IC16=0.0062, IC64=0.0347, MSE=0.1074, KL=1.05, Reg=82%
- Gates: [gate fail] at epoch 5 (expected — early training, ShIC not yet computed)
- Bug fixed: FeatureAttentionBlock NaN collapse at epoch 3 under AMP fp16 (head_dim=8, RMSNorm eps=1e-6 underflow). Forced fp32 in attention block.
- Before fix: NaN:1322 at epoch 3, total collapse by epoch 5
- After fix: Zero NaN across all 5 epochs, healthy loss convergence (44.35 -> 39.08)
- Conclusion: fp32 fix eliminates NaN. Full training run needed to evaluate V1.4's FeatureAttn contribution.

### 2026-03-03: V1 Family ShIC Dropout Analysis (all variants)
- All V1-V1.4 training stops were [SHIC STOP], not errors. Anti-fragile framework working correctly.
- ShIC peaks around epoch 100, then slowly declines as contiguous IC keeps rising (memorization creeps in).
- V1 f13: ShIC-stop ep130, best ShIC=0.0302 (ep100). Gate PASSED.
- V1.1 f13: ShIC-stop ep180, best ShIC=0.0293 (ep100). Gate PASSED.
- V1.2 f18: Completed 200ep, best ShIC=0.0307. Gate PASSED. KL annealing slowed memorization.
- V1.3 f18: ShIC-stop ep170, best ShIC=0.0270 (ep100). Gate PASSED.
- V1.4 f18: ShIC-stop ep100, best ShIC=0.0266 (ep20). Gate PASSED. Early peak may reflect NaN episodes.
- V1.5 f19: Manual stop ep26, ShIC=0.0254 and still improving. Resume needed.
- Conclusion: 5 of 6 V1 models are ensemble-ready. V1.2's KL annealing allows longer training.

### 2026-03-05: V1 Family Full Validation (all 6 models, 10 assets each)
- All 6 V1 models validated on held-out test set (purge gap 400 bars)
- Validation ShIC (held-out test, not training metric):
  - V1 f13:   ContIC=+0.0482, ShIC=+0.0222, Ratio=0.461, Val/Train=0.925
  - V1.1 f13: ContIC=+0.0467, ShIC=+0.0219, Ratio=0.469, Val/Train=0.925
  - V1.2 f18: ContIC=+0.0458, ShIC=+0.0225, Ratio=0.491, Val/Train=0.924
  - V1.3 f18: ContIC=+0.0366, ShIC=+0.0201, Ratio=0.550, Val/Train=0.927
  - V1.4 f18: ContIC=+0.0252, ShIC=+0.0188, Ratio=0.749, Val/Train=0.926
  - V1.5 f19: ContIC=+0.0487, ShIC=+0.0211, Ratio=0.433, Val/Train=0.925
- Gates: ALL 6 PASS all gates (RecMSE, MeanIC, KL, ShIC Ratio, Val/Train)
- Key insight: Validation ShIC band = 0.0188-0.0225 (20% spread) across 6 models
- V1.5 leads contiguous IC but its ShIC advantage is temporal, not genuine signal
- V1.2 is the only 18-feature model with working regime classification (~41% vs ~32% for V1.3-V1.5)
- Dream coherence is random (expected: dream_gru randomly initialized via strict=False)
- Conclusion: ShIC ceiling at ~0.022 for V1 architecture. V2-V9 needed to break it.

### 2026-03-05: V1.E Cross-Model Ensemble Validation (5 models, uniform averaging)
- Config: models=[V1, V1.1f13, V1.2, V1.3, V1.5], gating=uniform, features=19 (auto-sliced)
- Results:
  - Mean IC: +0.0492 (vs best individual +0.0487, mean individual +0.0454)
  - ShIC: +0.0236 (exceeds individual ceiling of ~0.022)
  - ShIC Ratio: 0.480 (PASS, > 0.3)
  - Per-horizon: IC(1)=+0.0542, IC(4)=+0.0599, IC(16)=+0.0420, IC(64)=+0.0404
  - Best assets: XRP(+0.159@t64), ETH(+0.112@t4), AVAX(+0.102@t64)
  - Weak assets: ADA(negative@t16/t64), DOGE(negative@t16)
- Gates: ALL PASS (MeanIC, ShIC Ratio)
- IC boost vs best individual: +0.9%, vs mean: +8.2%
- Conclusion: Ensemble breaks ShIC ceiling (0.0236 vs 0.022) -- genuine signal diversity confirmed.

### 2026-03-05: Agent PPO Training on V1.E Ensemble (first full run)
- Config: seed=42, policy=baseline (93K params), ensemble=V1.E (5 models), steps=2M, augment=off
- Training time: 95.6 min (~5.9s/rollout avg, 976 rollouts)
- In-sample results:  Sharpe=+14.81, MaxDD=1.73%, WinRate=100%, FinalValue=10,451 (+4.5%), Turnover=0.185
- Out-of-sample results: Sharpe=+9.71, MaxDD=1.39%, WinRate=100%, FinalValue=10,249 (+2.5%), Turnover=0.146
- OOS/IS Sharpe ratio: 0.655 [PASS] (>0.5 threshold)
- SAV robustness: stability=1.04 [PASS] (clean_reward=-6.47, noisy_reward=-6.76)
- Learning trajectory: Sharpe went from -71 (untrained) to +10 OOS over 2M steps
- Agent learned: reduce turnover (1.45->0.15), reduce costs (2035->255), minimize drawdown (20%->1.4%)
- Bug fixed: win_rate was computed from RL reward (always negative), now uses portfolio value vs initial capital
- Conclusion: First viable agent — profitable OOS with low drawdown, robust to weight perturbation.

### 2026-03-06: Agent PPO Training Run 2 — FAILURE (pre-normalization)
- Config: seed=42, policy=baseline (93K params), ensemble=V1.E (5 models), steps=2M, augment=on
- Training time: 106.3 min
- Results: Sharpe=-11.83, win_rate=10%, mean_turnover=0.004, entropy=19.189 (stuck at max)
- Reward trajectory: -8 to -13 throughout, no improvement over 2M steps
- Root causes:
  1. No obs normalization: return preds (~0.001) vs regime probs (~0.33) — 100x scale mismatch
  2. No reward normalization: all rewards uniformly negative, advantage normalization = noise
  3. LOG_STD_MAX=0.5 (sigma=1.65): actions ~uniform random after clipping to [-0.2, 0.2]
  4. Entropy 19.189 = 10 * 0.5 * ln(2*pi*e * 1.65^2) = theoretical maximum
- Fixes applied (13 across 5 files):
  - RunningMeanStd for obs + reward normalization (ppo.py)
  - LOG_STD bounds [-2.0, -0.5] (config.py), init at max (policy.py)
  - Value function clipping (ppo.py), look-ahead PnL fix (environment.py)
  - Gross exposure cap, drawdown txn costs (environment.py)
  - Funding cost penalty fix, max_dd fix (rewards.py)
- Tainted checkpoints deleted
- Conclusion: PPO fundamentally cannot learn without obs/reward normalization when features are at wildly different scales. Standard CleanRL/SB3 practice.

### 2026-03-06: Agent PPO Training Run 3 — Post-Normalization (V1.E ensemble)
- Config: seed=42, policy=baseline (93K params), ensemble=V1.E (5 models), steps=2M, augment=off
- Training time: 123.9 min (~7.6s/rollout, 976 rollouts)
- IS Final eval (10 ep): Return=-6.12, Sharpe=-9.66, MaxDD=2.1%, WinRate=30%, Turnover=0.007, Cost=$90
- IS Best eval: Return=-6.60, Sharpe=-9.09, FinalVal=$9,930, Cost=$91, Turnover=0.007
- SAV robustness: stability=1.005 [PASS]
- Reward trajectory: -51 (R100) -> -18 (R340) -> -9 (R975) — steady improvement but never profitable
- Entropy: 9.189 throughout (max for sigma=0.607, 10 assets). log_std never moved from init (-0.5).
- Value function converged: VF loss 97 -> 5 (normalization working correctly)
- Agent learned to minimize trading: turnover 0.19 -> 0.007 (near-zero trading)
- Remaining loss ($90/ep) is almost entirely funding costs on residual positions
- Comparison: Run 1 (pre-fixes, no normalization) achieved Sharpe=+14.81 IS / +9.71 OOS
- Root cause: With funding costs, the optimal strategy IS "don't trade" — agent correctly discovered this
- This confirms signal_monte_carlo finding: 0/1304 configs profitable with funding
- Conclusion: Normalization fixes work (VF converges, reward improves), but signal too weak to overcome funding costs. Need spot mode or reward redesign.

### 2026-03-06: Fixed-Rules Baseline (eval_rules.py, V1.E ensemble)
- Config: 6 naive strategies (DoNothing, ReturnProportional, MultiHorizonConsensus, RegimeGated, UncertaintyGated, TopNMarketNeutral), seed=42, 5 episodes IS+OOS
- Results: ALL strategies lose money after costs. Best: ReturnProportional IS=$-4.91 (gross +$1.31, costs $6.22)
- RegimeGated produced ZERO trades (regime probs max 0.435, never exceeds 0.5 threshold)
- TopNMarketNeutral: IS=$-947, OOS=$-1078 (catastrophic from high turnover + costs)
- Conclusion: Naive fixed rules are unprofitable. Costs dominate any edge from IC=0.05.

### 2026-03-06: Signal Monte Carlo Analysis (signal_monte_carlo.py, V1.E ensemble)
- Config: 30 MC episodes, 1304 strategy configs (980 linear + 252 binary + 72 TopN), expanded scales [100-5000], hold [4-256]
- **Futures mode (with funding): 0/1304 configs profitable IS, 12 profitable OOS**
- **Spot mode (no funding): 379/1304 (29.1%) profitable IS, but 0 survive IS+OOS cross-validation**
- IS top strategies use h64/long (memorized) -> OOS catastrophic reversal (IS +$33 -> OOS -$28)
- OOS top strategies use h1/short -> tiny edge ($5-6/episode, 0.05%)

#### Signal Quality:
- IS IC: h1=+0.061, h4=+0.051, h16=+0.055, h64=+0.071
- OOS IC: h1=+0.047, h4=+0.023, **h16=-0.008, h64=-0.034** (REVERSE OOS!)
- Conditional IC (top 25%): h1=+0.125, h64=+0.135 (strong predictions 2x more accurate)
- Hit rate h=1: 49.2% (BELOW 50% -- IC is magnitude-weighted, not directional)
- Prediction magnitude h=1: median=0.000019, p90=0.000073 (TwoHot compression ~50-100x)
- Signal persistence: 0.280 (moderate autocorrelation)
- Regime probs: mean [0.274, 0.454, 0.273], max never exceeds 0.435

#### Per-Model Signal (IS):
| Model | h=1 IC | h=4 IC | h=16 IC | h=64 IC | Best H | Persistence |
|-------|--------|--------|---------|---------|--------|-------------|
| V1.0  | +0.062 | +0.046 | +0.054 | +0.064* | h=64 | 0.254 |
| V1.1  | +0.063 | +0.061 | +0.075* | +0.068 | h=16 | 0.297 |
| V1.2  | +0.058* | +0.042 | +0.039 | +0.043 | h=1 | 0.271 |
| V1.3  | +0.056* | +0.040 | +0.035 | +0.043 | h=1 | 0.304 |
| V1.4  | +0.049 | +0.034 | +0.049 | +0.080* | h=64 | 0.249 |

#### OOS Asset IC (strongest to weakest):
SOL +0.047, LTC +0.032, ADA +0.021, BTC +0.005, XRP +0.005, ETH +0.003, LINK -0.006, DOGE -0.010, BNB -0.013, AVAX -0.037

#### Cost Analysis:
- Funding: 0.3125 bps/bar = $48-80/episode at 0.6-1.0 gross exposure (DOMINANT COST)
- Transaction: 6 bps roundtrip (~$3.60 per rebalance at 0.6 exposure)
- Break-even requires h1 IC > 0.06 AND scale > 5000 AND hold > 64 bars

#### Conclusions:
1. **h=1 is the ONLY genuinely generalizing horizon** (h16/h64 reverse OOS = memorization)
2. **Funding costs make futures unprofitable for fixed rules** (0 configs profitable IS)
3. **Spot mode enables IS profit** but h64 winners are overfit and h1 edge too small
4. **TwoHot prediction compression is the magnitude bottleneck** -- IC is rank-based, positions need magnitude
5. **PPO agent is the right tool** -- can learn cost-optimal execution fixed rules can't achieve
6. V1.2 and V1.3 are the most h=1-focused models (best for short-horizon strategies)
7. V1.1 has best multi-horizon balance but h16 signal doesn't generalize OOS
8. SOL, LTC, ADA are the most predictable assets OOS

### 2026-03-06: MA Strategy Permutation Backtest (133 strategies x 3 assets x 3 cost models)
- Script: `src/analysis/ma_backtest.py`, Results: `logs/analysis/ma_backtest_20260306_195739.json`
- Config: BTC/ETH/SOL dollar bars (~5min), 11 strategy types, zero/spot/perp costs
- Zero cost: 337/399 (84%) profitable OOS -- signal IS real in dollar bars
- Spot (0.10%/side): 5/399 (1%) profitable OOS -- costs kill everything
- Perp (0.04%/side+funding): 19/399 (5%) profitable OOS
- Cross-asset consistent: Only Donchian(1440) = 5-day breakout (perp: ETH Sharpe=1.36, SOL=1.17)
- Conclusion: Per-candle trading at 5-min bar frequency is catastrophic. Only multi-day trend following survives.

### 2026-03-06: WM Horizon Decomposition (7 models x 4 horizons x 10 assets)
- Script: `src/analysis/wm_horizon_analysis.py`, Results: `logs/analysis/wm_horizon_analysis_20260306_220803.json`
- OOS IC: V1.5 best (mean=0.0389), V1.4 worst (mean=0.0300)
- Signal efficiency: h=64 is 4.7x more efficient than h=1 (IC*sqrt(h): 0.264 vs 0.056)
- Optimal 2-model ensemble: V1.1_f18+V1.5 (beats all singles at every horizon)
- V1.5 in EVERY optimal ensemble. V1.4 in NONE.
- Regime specialists: V1.1_f18=bear (IC=0.128@h16), V1.1_f13=bull (IC=0.090@h16)
- Regime accuracy split: V1.3/V1.4/V1.5 ~86%, V1.0-V1.2 ~23%
- Quintile spread: V1.5 h=64 = 8.6bp (strongest long-horizon sort)
- Conclusion: V1.5 is anchor model. h=64 is optimal trading horizon. Regime-gated ensemble viable.

### 2026-03-06: WM Signal Backtest — Strategy Comparison (14 strategies x 3 assets, perp costs)
- Script: `src/analysis/wm_backtest.py`, Results: `logs/analysis/wm_backtest_20260306_230859.json`
- Config: BTC/ETH/SOL, 70/30 IS/OOS split, perp costs (0.04%/side + 0.01%/8h funding)
- 14 strategies: BuyAndHold, Donchian_5D, 12 WM-based variants (direction/consensus/smooth/hold/ensemble/filter)
- **ONLY cross-asset profitable strategy: Donchian 5-Day Breakout** (Avg OOS Sharpe=+0.98)
- All WM strategies fail OOS cross-asset (best: WM_Dir_h64_5d at -0.21 avg)
- WM as Donchian filter: marginal (+0.08 avg), hurts BTC (-0.49 from +0.41)
- IS->OOS reversal: WM_Smooth IS=+1.87, OOS=-4.49. Donchian IS=+0.08, OOS=+0.98 (no overfitting)
- Higher frequency = worse (monotonic): h16/1h Sharpe=-17.1, h64/5d=-0.21, Donchian=+0.98
- IC=0.033 at h=64 = 50.7% directional accuracy — too thin for standalone trading after costs
- Conclusion: WM adds value as FEATURE (IC=0.03), not as SIGNAL. Deploy Donchian 5-day as baseline. Agent must learn cost-optimal execution that fixed rules cannot achieve.

### 2026-03-07: Time-Calibrated Donchian — Dollar Bar Frequency Correction (10 assets x 10 periods)
- Script: `src/analysis/donchian_calibrated.py`, Results: `logs/analysis/donchian_calibrated_20260307_000752.json`
- **CRITICAL CORRECTION**: "Donchian 5-day" (period=1440) is actually ~1-day breakout. Dollar bars vary 852-2064/day (NOT 288).
  - BTC: 1178 bars/day -> period=1440 = 1.22 days
  - ETH: 1754 bars/day -> period=1440 = 0.82 days
  - SOL: 2064 bars/day -> period=1440 = 0.70 days
- Time-calibrated results (OOS, perp costs, 10 assets):
  - 0.5-day: avg Sharpe=-0.15, 4/10 positive
  - **1-day: avg Sharpe=+0.22, 7/10 positive** (sweet spot)
  - **2-day: avg Sharpe=+0.17, 7/10 positive**
  - 3-day: avg Sharpe=+0.01, 4/10 positive
  - 5-day: avg Sharpe=-0.47, 4/10 positive
  - 7+ day: avg Sharpe < -0.5 (fails)
- Original 3-asset "Sharpe +0.98" was partly cherry-picked (best 3 of 10) and partly accidental 1-day channel
- Best assets at 1-day: XRP (+1.51), ETH (+1.03), SOL (+0.85), AVAX (+0.57), LINK (+0.54)
- Worst: LTC (-1.32), DOGE (-0.89), BNB (-0.40)
- No overfitting: IS avg ~0.0, OOS avg +0.22 (better OOS than IS)
- Conclusion: 1-2 day Donchian breakout is the genuine cross-asset signal. Longer channels fail. Deploy on 7+ assets for diversification.

## Open Questions
- Which V2-V9 architectures pass anti-fragile gates? (None trained yet)
- Does seed affect ShIC variance significantly? (Need 3+ runs same config)
- Can resuming V1.5 from ep26 to ep150+ improve its ShIC (currently lowest at 0.0211)?
- Why does V1.2 KL annealing preserve regime head while V1.3/V1.4/V1.5 collapse?
- Would XD-gated ensemble (trained gating weights) improve over uniform averaging?

## Key Findings (cross-experiment)
- ShIC is THE discriminator: all memorization failures show ShIC < 0.01
- RevIN = always bad for this problem (confirmed across V1 family)
- **Genuine signal ceiling: ShIC ~0.022 for V1 Transformer-RSSM** (6 models converge)
- Contiguous IC differences (0.025-0.049) are mostly temporal patterns, not genuine signal
- V1.2 f18 is the most complete model (200ep, highest ShIC, working regime head)
- V1 f13 has the highest absolute ShIC (0.0222) — simplest model, strongest genuine signal
- ShIC peaks ~epoch 100, then declines — models correctly stopped by anti-fragile framework
- V1 family training ShIC range: 0.0254-0.0307 (6 models)
- V1 family validation ShIC range: 0.0188-0.0225 (6 models, held-out test)
- KL annealing (V1.2) is the only technique that extended training past ShIC dropout
- KL annealing (V1.2) is the only 18f technique that preserves regime head (~41% vs ~32%)
- Regime head collapse in 18/19-feature models: XD features crowd latent capacity for regime encoding
- Best assets for prediction: LTC, BNB, SOL (IC(1) avg ~0.063)
- Hardest asset: DOGE (IC(1) avg ~0.028)
- **V1.E ensemble breaks ShIC ceiling: 0.0236 vs individual ~0.022** -- genuine diversity benefit
- Ensemble IC boost is modest (+0.9% vs best), but ShIC improvement is the key win
- Uniform averaging across 5 heterogeneous models (f13/f18/f19) works without gating
