---
name: experiment_log
description: Training run results, ShIC trajectories, autopsy findings, and open questions
type: project
---

# Experiment Log

## 2026-03-30: Ensemble Ablation (10 assets x 29 configs)

Solo, leave-one-out, feature-group, and architecture-group testing.

| Config | Avg Sharpe | 9/9 pos? | Note |
|--------|-----------|----------|------|
| solo:v1_6 | **+0.997** | Yes | Best cross-asset. Dominates 5/9 assets. |
| solo:v1_6_f25 | +0.958 | Yes | f25 slightly worse than f13 in trading |
| arch:V1.6_all | +0.939 | Yes | 3 models, good risk-adjusted |
| solo:v1_0 | +0.921 | Yes | Simplest model, 4th best |
| ALL (10 models) | +0.766 | 8/9 | **14th out of 29** -- averaging dilutes |
| f25_only | +0.737 | 8/9 | f25 models alone are mediocre |

- **Key finding**: Uniform averaging HURTS. V1.6 solo > 10-model ensemble.
- **Per-asset selection** (best config per asset): avg +1.012 (vs +0.766 for ALL)
- **Saved**: `logs/analysis/optimal_ensemble_config.json`
- **CAVEAT**: All results from SINGLE OOS window. Walk-forward NOT yet done.
- **Next**: Learned gating or GBT meta-model instead of uniform averaging.

## 2026-03-29: Feature Scaling Results (f13 vs f18 vs f25 vs f37)

V1.1, V1.4, V1.6 trained with f18, f25, f37. Key finding: **f25 is the sweet spot**.

| Model | f13 IC_h1 | f18 IC_h1 | f25 IC_h1 | f37 IC_h1 |
|-------|-----------|-----------|-----------|-----------|
| V1.1 | 0.0569 | 0.0528 (-7%) | **0.0646 (+14%)** | 0.0570 (0%) |
| V1.4 | 0.0569 | 0.0537 (-6%) | **0.0641 (+13%)** | - |
| V1.6 | 0.0527 | 0.0514 (-2%) | **0.0603 (+14%)** | - |

- f18 WORSE than f13: the 5 extended features hurt (ma_distance, whale, efficiency, ret4, ret16)
- f25 BEST: Tier1 (kurtosis, bar_duration, funding_momentum) + Hawkes (4 features) add real signal
- f37 NEUTRAL: IC-boost + cross-asset features dilute without adding signal
- Microstructure group drives 40-73% of IC across all feature counts
- Conclusion: f25 is the optimal feature count. f13 is the safe fallback.

## 2026-03-27: Price-Action Strategy Sweep (compare-risk, 10 assets)

All 20 strategies x 10 assets x 3 risk regimes (none/ATR3/SL5).

- **10/10 assets positive** with baseline (no stops), avg Sharpe **+1.032**
- Best per-asset: VPIN (BTC +1.379, ETH +0.533, AVAX +1.311), Donchian (XRP +1.351)
- ATR3 trailing stop destructive on 7/10 assets
- SL5 tolerable but always worse than baseline
- Conclusion: strategies' own exit logic beats external risk management

## 2026-03-25: WM-Ensemble Strategy Sweep (4-model V1 ensemble)

- Config: V1.0+V1.1+V1.4+V1.6 f13 ensemble, SPOT mode, long-only, 10 assets
- Source: `logs/analysis/strategy_lab_wm_20260324_223538.json`

**Results (OOS, per-asset best WM strategy):**
| Asset | Best WM Strategy | OOS Sharpe | vs Donchian | vs B&H |
|-------|-----------------|-----------|-------------|--------|
| AVAX | WM_DonchFilter | +1.233 | +0.445 | +0.430 |
| XRP | Donchian | +1.351 | +0.000 | +0.131 |
| BNB | WM_DonchFilter | +1.097 | +0.051 | +0.362 |
| DOGE | WM_Momentum | +1.029 | +0.050 | +0.167 |
| ADA | WM_Threshold | +1.022 | +0.698 | +0.936 |
| BTC | WM_Momentum | +0.927 | +0.151 | -0.010 |
| ETH | WM_Threshold | +0.790 | +0.708 | +0.814 |
| LINK | WM_Mom_h1 | +0.759 | +0.431 | +0.301 |
| LTC | WM_Threshold | +0.661 | +0.738 | +0.179 |
| SOL | BuyAndHold | -0.507 | -0.359 | +0.000 |

- **Avg selected Sharpe: +0.836** (9/10 active, 1 fallback to B&H)
- **vs Donchian avg: +0.291** (WM beats Donchian on 9/10 assets)
- **vs B&H avg: +0.331** (WM beats B&H on 8/10 assets)
- Conclusion: WM ensemble adds real edge as filter on rule-based strategies. Not standalone-tradeable but valuable as signal overlay.
- Caveat: single OOS window. Walk-forward validation needed.

## 2026-03-24: V1 Family f13 Training Complete (DRW=3.0, pairwise=0.0)

All 4 V1 variants trained with corrected settings. All PASS gates.

| Model | Epochs | Best ShIC | OOS IC1 | ShIC Trajectory | Gate |
|-------|--------|-----------|---------|-----------------|------|
| V1.0 | 115 | 0.0267 | 0.0566 | RISING (0.0248->0.0267) | PASS |
| V1.1 | 130 | 0.0270 | 0.0570 | RISING (0.0251->0.0270) | PASS |
| V1.4 | 150 | 0.0269 | 0.0588 | Peak+ShIC-stop (5 declines) | PASS |
| V1.6 | 120 | 0.0280 | 0.0535 | Peak+ShIC-stop (5 declines) | PASS |

- V1.0/V1.1: cleanest trajectories (ShIC rising at stop)
- V1.6: highest single ShIC (0.0280) but peaked ep50, declined by ep120
- V1.4: highest OOS IC1 (0.0588) but ShIC declined from peak
- All models: DRW=3.0, pairwise=0.0, bins=[-1,1], raw targets
- Ensemble ready: 4 diverse checkpoints for V1.E

## 2026-03-22: V1.0 f13 Training (DRW=1.0, pairwise=0.1 -- ShIC DECLINING)
- Config: seed=default, features=13, epochs=60 (ShIC stop), raw targets, bins=[-1,1], DRW=1.0, pairwise=0.1
- Results: IC1=0.0523, ShIC=0.0254->0.0237 (declining), val_loss=25.25, KL=1.00
- Gates: PASS (but ShIC declining from ep10)
- ShIC trajectory: 0.0254 (ep10), 0.0240, 0.0215, 0.0222, 0.0218, 0.0237 (STOP ep60)
- OOS IC1: 0.0548
- Gap widening: 0.011 (ep10) -> 0.029 (ep60) = accelerating memorization
- **Root cause**: DRW=1.0 (was 3.0 in stable Feb runs) + pairwise ranking loss (new addition)
- **Fix**: Reverted DRW to 3.0, disabled pairwise ranking (weight=0.0) in ALL versions
- Checkpoint: DELETED (trained with wrong settings)

## 2026-03-11: V1.1 f22 Training (bins [-1,1], voladj targets)
- Config: seed=default, features=22, epochs=35 (stopped by ShIC), revin=off, bins=[-1,1]
- Results: IC1=0.158, ShIC=0.096->0.079 (monotonic decline), MSE=0.066, KL=1.00
- Gates: PASS (but ShIC declining)
- ShIC trajectory: 0.096, 0.095, 0.088, 0.084, 0.082, 0.080, 0.079 (STOP)
- LR reductions: 4x (0.5, 0.25, 0.125, 0.062)
- Conclusion: Bins fix [-1,1] did NOT stop ShIC decline. Vol shortcut persists.
- Checkpoint: models/v1/v1_1/base/v1_1_f22_wm_best_ema.pt

## 2026-03-11: V1.0 f13 Training (bins [-1,1], voladj targets)
- Config: seed=default, features=13, epochs=15 (stopped by ShIC), revin=off, bins=[-1,1]
- Results: IC1=0.126, ShIC=0.100->0.068, MSE=0.062, KL=1.01
- Gates: PASS (but ShIC declining)
- ShIC trajectory: 0.100, 0.092, 0.079, 0.068 (STOP)
- LR reductions: 2x
- Conclusion: Same decline pattern as V1.1, even with only 13 features.
- Checkpoint: models/v1/v1_0/base/v1_0_f13_wm_best_ema.pt

## 2026-03-10: V1.1 f22 Autopsy (bins [-5,5], voladj targets)
- Config: features=22, epochs=20
- Autopsy findings (epoch 10 group ablation):
  - microstructure: IC1 drop = 0.042 (largest, mix of genuine + shortcut)
  - new_base (whale, efficiency): IC1 drop = 0.034
  - vol: IC1 drop = 0.010 (despite zero raw IC for norm_hl_spread!)
  - returns: IC1 drop = 0.009
  - regime: IC1 drop = 0.004
  - **xd: IC1 drop = 0.000** (DEAD WEIGHT confirmed)
- Key raw feature ICs: flow_imbalance=0.098, xd_cross_return_mean=0.034, deviation=0.021

## 2026-03-10: V1.6 f22 Training (bins [-5,5], voladj targets)
- Config: features=22, epochs=20, ATME p=0.15
- ShIC: 0.103 -> 0.097 -> 0.088 -> 0.085 (STOP)
- Conclusion: ATME slows decline slightly but doesn't prevent it.

## Pre-V51 Reference Runs (raw targets, bins [-1,1])

### V1.0 f13 (2026-02-24)
- ShIC: 0.028 -> 0.030 (stable 100+ epochs)
- Best ShIC: 0.0302

### V1.1 f18 (2026-02-27)
- ShIC: 0.028 -> 0.028 (stable 60+ epochs)
- Best ShIC: 0.0284

### V1.6 f18 (2026-03-07, bins [-0.1,0.1])
- ShIC: 0.026 -> 0.030 (RISING over 60+ epochs)
- Best ShIC: 0.0302

## 2026-03-15: V4.0 f13 Run 1 (pre-audit, stale settings)
- Config: seed=default, features=13, epochs=~20, ATME ctx_drop=0.15/seq_shuffle=0.20
- Parameters: 6,291,459 (larger architecture)
- Results: IC1=0.0342, ShIC=0.0167, ShIC/IC=0.49
- Gates: ShIC=0.0167 > 0.015 PASS (marginal)
- Conclusion: Passed ShIC threshold but ratio 0.49 indicates ~51% temporal memorization. Pre-audit (stale DIRECT_RETURN_WEIGHT=3.0, STEPS_PER_EPOCH=300).

## 2026-03-16: V4.0 f13 Run 2 (pre-audit, higher ATME)
- Config: seed=default, features=13, epochs=~25, ATME ctx_drop=0.25/seq_shuffle=0.20
- Parameters: 5,371,091 (reduced architecture)
- Results: IC1=0.0321, ShIC=0.0060, ShIC/IC=0.19
- Gates: ShIC=0.0060 < 0.015 FAIL
- Conclusion: Increased ATME didn't help. Smaller model still memorizes heavily. Pre-audit settings.

## 2026-03-17: V4.0 f13 Run 3 COMPLETE (post-audit fixes, aggressive ATME) -- 90 epochs
- Config: seed=default, features=13, epochs=90 (full run), ATME ctx_drop=0.40/seq_shuffle=0.30
- Parameters: 5,371,091
- ShIC trajectory: 0.0125 -> **0.0167** -> 0.0159 -> 0.0164 -> 0.0153 -> 0.0144 -> 0.0140 -> 0.0133 -> 0.0135
- Best ShIC: 0.0167 (epoch 20) -- peaked early, then monotonic decline for 70 epochs
- Best contiguous IC1: 0.0420 (epoch 60)
- Best val loss: 25.2965
- Gates: **NOT PASSED** (ShIC peaked at 0.0167 but declined; gate requires sustained passing)
- ShIC/IC ratio at peak: 0.0167/0.0406 = 0.41 (vs V1's ~0.70)
- Gap (contiguous - shuffled) WIDENED: 0.023 (ep10) -> 0.026 (ep80) = increasing memorization
- Note: Post-audit (DIRECT_RETURN_WEIGHT=1.0, STEPS_PER_EPOCH=2000, h_seq.detach, strict=False)
- **Conclusion: V4 Mamba has structural memorization bias. ShIC ceiling ~0.017 is 35% below V1's stable 0.025+. Architecture deprioritized.**

## 2026-03-18: V3.0 f13 COMPLETE (WaveNet-GRU, post-audit) -- 130 epochs
- Config: seed=default, features=13, epochs=130 (ShIC stop), ATME ctx_drop=0.40/seq_shuffle=0.30
- Parameters: ~5M (WaveNet-GRU + RSSM)
- ShIC trajectory: 0.0126 -> **0.0159** -> 0.0157 -> 0.0153 -> 0.0156 -> 0.0150 -> **0.0159** -> 0.0152 -> 0.0143 -> 0.0136 -> 0.0133 -> 0.0136 -> 0.0126 (STOP)
- Best ShIC: 0.0159 (epoch 70) -- peaked then declined for 60 epochs
- Best contiguous IC1: 0.0444 (epoch 130)
- Best val loss: 25.2794
- Gates: **NOT PASSED** (ShIC=0.0159 > 0.015 at peak, but declined to 0.0126; gate requires sustained)
- ShIC/IC ratio at peak: 0.0159/0.0421 = 0.38
- Gap WIDENING: 0.020 (ep10) -> 0.032 (ep130) = increasing memorization
- Note: GN:inf appearing in later epochs (gradient instability)
- **Conclusion: V3 WaveNet-GRU briefly touched ShIC gate (0.0159) but couldn't sustain. Memorization gap widened over time. Best V2-V9 result but still below V1's stable 0.025+.**

## 2026-03-21: V9.0 f13 COMPLETE (MoE, post-NaN-fix) -- 170 epochs (full run)
- Config: seed=default, features=13, epochs=170 (full), LR=1e-4, warmup=10, ATME ctx_drop=0.40/seq_shuffle=0.30
- Parameters: 9,179,159 (3-expert MoE + RSSM)
- NaN recovery: 2 recoveries (epoch 5: reinit from scratch, epoch 11: checkpoint reload). LR_multiplier=0.25.
- ShIC trajectory: -0.000 -> -0.001 -> 0.001 -> 0.004 -> 0.006 -> 0.007 -> 0.009 -> 0.011 -> 0.012 -> 0.012 -> 0.012 -> **0.0126** -> 0.012 -> 0.012 -> 0.013 -> 0.013 -> 0.012
- Best ShIC: 0.0126 (epoch 120/150) -- rose steadily, plateaued, NO decline
- Best contiguous IC1: 0.0335 (epoch 160-170)
- Best val loss: 25.2590
- Gates: **NOT PASSED** (ShIC=0.0126 < 0.015)
- ShIC/IC ratio: 0.0126/0.0335 = 0.376
- Gap STABLE: 0.020-0.021 from epoch 90-170 (no memorization acceleration)
- Router weights (ep10): B:0.35 N:0.31 Bu:0.36 (balanced)
- **Conclusion: V9 MoE is numerically stable post-fix, ShIC doesn't decline (healthiest V2-V9 trajectory), but absolute ShIC=0.013 is well below gate. 9.1M params didn't help -- excess capacity went to temporal gap not cross-sectional signal. Effective LR=2.5e-5 after NaN recoveries may be too conservative.**

## Cross-Model Comparison (All Baselines, f13, Post-Audit)

| Model | Arch | Params | Epochs | Best ShIC | Peak IC1 | ShIC/IC | Gap Trend | Gate |
|-------|------|--------|--------|-----------|----------|---------|-----------|------|
| V1.0 (pre-V51) | Transformer+RSSM | ~2M | 100+ | **0.0302** | 0.030 | ~1.0 | STABLE | **PASS** |
| V1.1 (pre-V51) | Transformer+RSSM | ~2.5M | 60+ | **0.0284** | 0.028 | ~1.0 | STABLE | **PASS** |
| V1.6 (pre-V51) | Transformer+RSSM | ~3M | 60+ | **0.0302** | 0.030 | ~1.0 | RISING | **PASS** |
| V3 | WaveNet-GRU+RSSM | ~5M | 130 | 0.0159 | 0.044 | 0.38 | WIDENING | FAIL |
| V4 | Mamba-SSM+RSSM | 5.4M | 90 | 0.0167 | 0.042 | 0.40 | DECLINING | FAIL |
| V9 | MoE(3)+RSSM | 9.1M | 170 | 0.0126 | 0.034 | 0.38 | STABLE | FAIL |

**Key findings:**
1. V1 Transformer (2-3M) achieves ShIC 0.025-0.030 with ratio ~1.0 (no memorization)
2. V3/V4/V9 (5-9M) achieve ShIC 0.013-0.017 with ratio ~0.38 (62% temporal memorization)
3. More parameters = more memorization, not more signal
4. V9 is healthiest V2-V9 (no ShIC decline) but lowest absolute ShIC

## Open Questions

1. **Does V4 memorization hurt OOS performance?** (evaluate_v4_oos.py ready to test)
2. **Should V4 anti-memorization be adapted to Mamba's nature?** (architecture-aware ATME)
3. **Does gradient surgery stop ShIC decline?** (Phase 1C - next experiment)
4. **What is OOS raw-return IC?** (Need validation with raw targets)
5. **Does removing XD features change anything?** (f17 vs f22 post-fix)
6. **Is 7-core-feature model competitive?** (f7 experiment post-fix)

---

## 2026-04-07: V1 Family f34 Tournament (COMPLETE)

f34 (34 base features, SOTA pipeline) tournament results, all 4 V1 variants:

| Model | Cont IC (best) | Shuffled IC (best) | ShIC Gap | Gate |
|-------|:--------------:|:------------------:|:--------:|:----:|
| V1.0 f34 | 0.0660 | 0.0320 | 0.0340 | PASS |
| V1.1 f34 | **0.0674** | **0.0330** | 0.0344 | PASS (record) |
| V1.4 f34 | 0.0679 | 0.0314 | 0.0365 | PASS |
| V1.6 f34 | 0.0619 | 0.0329 | 0.0290 | PASS |

**V1.1 holds the all-time ShIC record at 0.0330.**

**f34 vs f13 lift:** IC 0.066 vs 0.056 (+18%), ShIC 0.032 vs 0.027 (+18%). The 4 SOTA features (yz_vol, cs_spread, perm_entropy, kyle_lambda) added at f34 provide a clear signal lift. f34 is now the default tournament configuration.

**Config (all models):** seed=default, split=50/20/20/10, epochs=150, revin=off, target_prefix=target_return, BIN=[-1,1], NUM_BINS=255, WM_STEPS_PER_EPOCH=2000, DIRECT_RETURN_WEIGHT=3.0.

**Next:** V3-clean, V6-clean, V9-clean, V8, V11, V12, V13, V14 queued for f34 tournament. Three session fixes landed before the run (8371b2f, 2502030, 5f25f97) to unblock the clean variants — see memory/fix_logs/INDEX.md Patterns G and H.

