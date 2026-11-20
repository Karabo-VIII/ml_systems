# World Model Validation & V5+ Architectures - Executive Summary

> **V51 Update (Mar 2026)**: Pipeline now produces 24 features (18 base + 6 XD cross-asset). Models default to 22 features (17 base + 5 XD). TwoHot bins changed to [-5, 5] x 255 with voladj targets. All existing checkpoints are incompatible and must be retrained. The validation framework and V2 overfitting analysis below remain accurate. See `CLAUDE.md` for current system state.

## Critical Discovery: V2 Overfitting

### The Problem

Your V2 JEPA model showed **IC = +0.92** on standard validation (contiguous last-10% split). This appeared world-class. However, **robust validation revealed catastrophic overfitting**:

| Metric | Standard Val | Robust Val | Status |
|--------|-------------|------------|--------|
| Contiguous IC | +0.94 | +0.94 | Misleading |
| **Shuffled IC** | N/A | **~0.00** | **FAIL** |
| IC Gap | N/A | **99%** | **FAIL** |
| Hallucination Score | N/A | **0.75** | **FAIL** (threshold: 0.3) |

**Verdict:** Model memorizes temporal autocorrelation, not true predictive signal.

---

## Root Causes

1. **Contiguous validation split** - Last 10% shares temporal structure with training data
2. **BiGRU sees full context** - Bidirectional encoding encourages memorization
3. **Small dataset** (~2K sequences) - Easy to overfit
4. **No distribution shift testing** - Single split doesn't catch this

---

## Solution: Robust Validation Framework

Created comprehensive validation suite (`validation_utils.py`) that detects overfitting via:

### 1. Temporal Forward Walk
```
Train [0, 50%], Val [50%, 60%]
Train [0, 60%], Val [60%, 70%]
...
Train [0, 90%], Val [90%, 100%]
```
Tests: Does IC remain stable as validation moves forward?

### 2. Shuffled K-Fold
```
Randomly shuffle data, split into K folds
```
Tests: Does model learn features or temporal order?
**Critical**: V2's IC dropped from 0.94 → 0.00 when shuffled

### 3. Regime-Specific Holdout
```
Hold out 20% of bearish/neutral/bullish separately
```
Tests: Does model generalize across market conditions?

### 4. Hallucination Score
```
score = f(forward_degradation, shuffle_drop, regime_brittleness, instability, absurd_baseline)
```
Range: 0-1, >0.3 = suspicious

---

## Validation Usage

```bash
# Standard validation (fast, optimistic)
python backups/BKP_20260429_MODEL_HARMONIZATION/v2/v2_training/validate_world.py

# Robust validation (slow, rigorous - USE THIS)
python backups/BKP_20260429_MODEL_HARMONIZATION/v2/v2_training/validate_world.py --robust --horizon 1

# Compare checkpoints
python backups/BKP_20260429_MODEL_HARMONIZATION/v2/v2_training/validate_world.py --both --robust
```

**Next Steps:**
1. Run `--robust` on V1/V3/V4 to check for overfitting
2. Use shuffled IC as primary metric going forward
3. Never deploy a model with hallucination score > 0.3

---

## V5-V9 SOTA Architectures

Designed 5 world-class models with **explicit anti-overfitting mechanisms**:

### V5: Hybrid Mamba-Attention
- **Core**: Mamba (linear) → Downsample → Local Attention (quadratic on compressed) → Upsample
- **Innovation**: Scale separation (Mamba=local, Attention=global)
- **Anti-overfit**: Downsampling bottleneck prevents memorization
- **Expected**: Shuffled IC 0.03-0.05, IC Gap <0.10

### V6: Adversarial JEPA (Fixes V2)
- **Core**: Causal GRU + Contrastive + **Adversarial Time Discriminator**
- **Innovation**: Discriminator detects temporal structure, encoder learns to hide it
- **Anti-overfit**: Explicit penalty for temporal dependence
- **Expected**: Shuffled IC 0.04-0.07, IC Gap <0.05
- **Priority**: **CRITICAL** - Validates adversarial training approach

### V7: Vision Transformer (2D Patch-Based)
- **Core**: Treat time×features as 2D image, apply ViT
- **Innovation**: Learns joint temporal-feature correlations
- **Anti-overfit**: Patch aggregation prevents individual timestep memorization
- **Expected**: Shuffled IC 0.02-0.04, IC Gap <0.06
- **Unique**: Completely different inductive bias

### V8: Neural ODE (Continuous Dynamics)
- **Core**: Learn dh/dt = f(h,t), solve ODE for trajectory
- **Innovation**: Continuous-time dynamics, arbitrary horizon prediction
- **Anti-overfit**: Smooth dynamics prevent discrete boundary overfitting
- **Expected**: Shuffled IC 0.025-0.045, IC Gap <0.04
- **Benefit**: Can predict fractional horizons (t+2.7, t+15.3)

### V9: Mixture-of-Experts with Regime Gating
- **Core**: Router + 3 experts (Bearish=Mamba, Neutral=TCN, Bullish=Transformer)
- **Innovation**: Specialized models per regime, learned soft mixing
- **Anti-overfit**: Expert specialization limits overfitting scope
- **Expected**: Shuffled IC 0.035-0.055, IC Gap <0.08
- **Benefit**: Interpretable regime detection via router

---

## Architecture Comparison

| Model | Anti-Overfit | Shuffled IC (Est.) | IC Gap | Ensemble Value | Priority |
|-------|--------------|-------------------|--------|----------------|----------|
| V1 (Transformer) | Moderate | 0.02-0.04 | <0.08 | Baseline | - |
| V2 (JEPA) | ✗ **Broken** | **0.00** | **~0.94** | ✗ Fail | Fix |
| V3 (TCN-GRU) | Good | 0.03-0.05 | <0.07 | High | Validate |
| V4 (Mamba) | Unknown | TBD | TBD | High | Validate |
| **V5** (Mamba+Attn) | **Strong** | **0.03-0.05** | **<0.10** | Very High | **HIGH** |
| **V6** (Adv JEPA) | **Very Strong** | **0.04-0.07** | **<0.05** | Very High | **CRITICAL** |
| **V7** (ViT) | **Strong** | **0.02-0.04** | **<0.06** | Very High | Medium |
| **V8** (Neural ODE) | **Very Strong** | **0.025-0.045** | **<0.04** | Very High | Low (complex) |
| **V9** (MoE) | **Strong** | **0.035-0.055** | **<0.08** | Maximum | Medium |

---

## Implementation Roadmap

### Phase 1: Validation & Fixes (Today)
- ✅ Robust validation framework created
- ✅ V2 overfitting documented
- ⏳ **Next: Run robust validation on V1/V3/V4**
- ⏳ **Next: Implement V6 (Adversarial JEPA)**

### Phase 2: High-Priority Models (Week 1)
- V5 (Mamba+LocalAttention) - 3-4 hours
- V9 (Mixture-of-Experts) - 5 hours
- Cross-validate all models

### Phase 3: Research Models (Week 2)
- V7 (Vision Transformer) - 6 hours
- V8 (Neural ODE) - 8 hours

### Phase 4: Ensemble Deployment
- Select top 3-5 models by shuffled IC
- Implement weighted ensemble voting
- Production confidence gating

---

## New Validation Gates (For All Models)

Before agent training, **REQUIRE**:

1. ✅ Contiguous IC > 0.02
2. ✅ **Shuffled IC > 0.015** ← **NEW: CRITICAL**
3. ✅ **IC Gap < 0.10** ← **NEW: Max 10% overfitting**
4. ✅ **Hallucination Score < 0.30** ← **NEW**
5. ✅ Regime Coverage: IC > 0.01 in all 3 regimes
6. ✅ Forward Walk Stability > 0.60
7. ✅ Reconstruction MSE < 0.15

**V2 Current Status:**
- ❌ Shuffled IC: -0.001 (FAIL)
- ❌ IC Gap: 0.94 (FAIL)
- ❌ Hallucination: 0.75 (FAIL)

**BLOCKED from agent training until fixed.**

---

## Key Takeaways

### 1. Crypto Returns Have Strong Autocorrelation
- Models can achieve IC = 0.95+ by memorizing sequence order
- **Shuffled validation is MANDATORY**
- Trust shuffled IC as ground truth

### 2. Standard Validation is Dangerously Optimistic
- Contiguous last-10% split shares temporal structure with training
- Can be 10-100x overoptimistic (V2: 0.94 vs 0.00)
- Always cross-validate with robust methods

### 3. Small Datasets Amplify Overfitting
- ~2K sequences insufficient for complex models
- Either: gather more data, or use aggressive regularization
- Simpler models (V3 TCN) may outperform complex ones (V2 JEPA)

### 4. JEPA/Contrastive is Powerful But Risky
- Can learn structure efficiently (good)
- Can memorize temporal order efficiently (bad)
- Requires adversarial training to prevent overfitting

### 5. Robust Validation Caught a Critical Bug
- Without this, V2 would have been deployed with 0.00 true IC
- Robust validation framework prevents costly production mistakes
- Investment in validation infrastructure pays off

---

## Files Created

1. **`src/validation_utils.py`**
   - RobustValidator class
   - Forward walk, shuffled K-fold, regime holdout, hallucination detection
   - ~600 lines, production-ready

2. **`VALIDATION_ROBUSTNESS_FINDINGS.md`**
   - Detailed analysis of V2 overfitting
   - Root causes, fixes, recommendations
   - 15+ pages

3. **`V5_PLUS_ARCHITECTURES.md`**
   - Full specifications for V5-V9
   - Architecture diagrams, hyperparameters, expected performance
   - Anti-overfitting mechanisms for each
   - 20+ pages

4. **`backups/BKP_20260429_MODEL_HARMONIZATION/v2/v2_training/validate_world.py`** (modified)
   - Added `--robust` flag
   - Integration with RobustValidator
   - Saves robust validation results to JSON

5. **`logs/v2/robust_validation_v2_wm_best_h1_*.json`**
   - V2 robust validation results
   - Evidence of overfitting

---

## Immediate Next Steps

### For You (User):

1. **Review Findings**
   - Read `VALIDATION_ROBUSTNESS_FINDINGS.md`
   - Understand why V2 failed

2. **Run Robust Validation on V1/V3/V4**
   ```bash
   python src/wm/v1/v1_training/validate_world.py --robust --horizon 1
   python src/wm/v3/v3_training/validate_world.py --robust --horizon 1
   python src/wm/v4/v4_training/validate_world.py --robust --horizon 1
   ```
   (Need to add `--robust` flag to V1/V3/V4 validators first)

3. **Decide on V5+ Implementation Priority**
   - V6 (Adversarial JEPA) is critical for validating approach
   - V5 (Mamba+Attention) is safest production upgrade
   - V9 (MoE) provides ensemble diversity

### For Implementation:

1. **Add robust validation to V1/V3/V4 validators**
   - Copy pattern from V2
   - Run and compare

2. **Implement V6 first** (Adversarial JEPA)
   - Validates adversarial training concept
   - If successful, proves we can fix V2
   - ~4 hours

3. **Implement V5** (Mamba+LocalAttention)
   - Safest V4 upgrade
   - Proven architecture
   - ~3-4 hours

---

## Questions for Discussion

1. **V2 Recovery**: Retrain from scratch or fine-tune current checkpoint?
2. **Data Collection**: Can we gather more training data (~10K sequences target)?
3. **V5+ Priority**: Which architectures are most critical for your use case?
4. **Ensemble Strategy**: How many models should final production system use?
5. **Validation Frequency**: Should we run robust validation every N epochs during training?

---

**Status:** ✅ Robust validation operational, V2 overfitting detected, V5-V9 designs complete
**Blocker:** V2 cannot be used for agent training (hallucination score 0.75)
**Next:** Validate V1/V3/V4, implement V6+V5

**Date:** 2026-02-15
**Author:** Claude Opus 4.6
