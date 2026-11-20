# V5-V9 World-Class Architecture Designs

> **V51 Update (Mar 2026)**: Architecture diagrams below show `Input(13+32)` reflecting the original 13-feature design. Current pipeline produces 24 features; models default to 22 features (17 base + 5 XD) via `settings.FEATURE_LIST`. The input projection dimension adjusts automatically. TwoHot encoding is now 255 bins over [-5, 5] with focal gamma 1.0 and voladj targets. See `src/v{N}/USAGE.md` for current settings.

## Design Philosophy

After discovering V2's severe overfitting to temporal autocorrelation, the V5+ designs incorporate **explicit anti-overfitting mechanisms** while maintaining SOTA performance.

### Core Principles

1. **Orthogonality**: Each architecture should fail differently (ensemble diversity)
2. **Robustness**: Built-in mechanisms to prevent temporal leakage
3. **Efficiency**: Production-ready (RTX 4060, 8GB VRAM, <2min/epoch)
4. **Interpretability**: Understand *why* model makes predictions
5. **Measurability**: Every claim must be testable via robust validation

---

## V5: Hybrid Mamba-Attention (Linear + Quadratic Synergy)

### **Motivation**

V4 (Mamba) uses O(T) selective scan but may miss long-range dependencies. V1 (Transformer) captures global context but O(T²) is expensive. **Hybrid**: use Mamba for compression, then local attention for refinement.

### **Architecture**

```
Input(13+32) → Linear(45→384) + LayerNorm + SiLU

→ 2x MambaBlock (d_model=384, d_state=16)
  [Fast sequential processing, linear complexity]

→ Downsample (T → T/4 via strided conv)
  [96 timesteps → 24 meta-tokens]

→ 2x LocalAttentionBlock (window=8, heads=6)
  [Quadratic attention but only on 24 tokens]
  [Receptive field per token: 8*4 = 32 original timesteps]

→ Upsample (T/4 → T via transposed conv + residual)
  [24 meta-tokens → 96 timesteps]

→ RSSM Prior/Posterior (32×32 = 1024 latent)

→ Heads: Reconstruction | Returns[4] | Regime
```

### **Key Innovations**

1. **Hierarchical Processing**
   - Mamba: Local dynamics (intra-bar patterns)
   - Attention: Global structure (cross-bar relationships)
   - Different computational graphs → different failure modes

2. **Computational Efficiency**
   - Mamba: 2 × O(96 × 384) = O(T)
   - Attention: 2 × O(24² × 384) = O((T/4)²)
   - Total: ~3x faster than full Transformer, 1.5x slower than pure Mamba

3. **Explicit Scale Separation**
   - Low-level (Mamba): Tick dynamics, noise filtering
   - High-level (Attention): Regime shifts, trend structures

### **Anti-Overfitting Mechanisms**

1. **Downsampling Bottleneck**
   - Forces compression → prevents memorization of individual timesteps
   - Model must learn aggregated features, not raw autocorrelation

2. **Local Attention (not global)**
   - Window=8 prevents attending to arbitrary distant timesteps
   - Reduces risk of learning spurious long-range correlations

3. **Residual Upsample**
   - Skip connection from Mamba output → Attention can only add refinements
   - Prevents Attention from completely overriding Mamba's causal structure

### **Hyperparameters**

```python
# Model
V5_D_MODEL = 384
V5_D_STATE = 16
V5_MAMBA_LAYERS = 2
V5_ATTN_LAYERS = 2
V5_ATTN_HEADS = 6
V5_ATTN_WINDOW = 8
V5_DOWNSAMPLE_FACTOR = 4
V5_RSSM_LATENT = (32, 32)  # 1024 total

# Training
V5_DROPOUT = 0.20  # Higher than V4's 0.10
V5_WEIGHT_DECAY = 0.08
V5_BATCH_SIZE = 32  # Smaller than V4's 48
V5_LR = 2e-4
V5_EPOCHS = 150
```

### **Expected Performance**

- **Shuffled IC**: 0.03-0.05 (true generalization)
- **Contiguous IC**: 0.08-0.12 (some temporal structure exploited)
- **IC Gap**: <0.10 (robust)
- **Regime Coverage**: 3/3 (attention enables regime adaptation)
- **Speed**: ~90 sec/epoch (RTX 4060)

---

## V6: Causal JEPA with Adversarial Time Shuffling

### **Motivation**

V2 (JEPA) achieved high sample efficiency but catastrophically overfit to temporal order. **V6 fixes this** by: (1) replacing BiGRU with Causal GRU, (2) adding adversarial discriminator to penalize temporal dependence.

### **Architecture**

```
Input(13+32) → Linear(45→256) + LayerNorm + SiLU

→ CausalGRUEncoder [256d, 3 layers, unidirectional]
  [NOT bidirectional - prevents seeing future]

→ Context Latent Proj: 256 → 192d

Parallel Paths:
  1) Online Predictor: context_latent → pred_latent
  2) EMA Target Encoder: obs → target_latent (momentum=0.996)
  3) Time Discriminator: context_latent → P(temporally_coherent)

→ InfoNCE Loss (per-timestep contrastive)
→ VICReg Loss (collapse prevention)
→ Adversarial Loss: -log(1 - P(coherent))  [MINIMIZE coherence signal]
→ Return Heads[4] | Regime Head | Recon Head
```

### **Key Innovation: Adversarial Time Shuffling**

```python
# Training loop
for batch in dataloader:
    # Real (coherent) sequence
    real_obs = batch["obs"]
    real_latent = online_encoder(real_obs)

    # Fake (shuffled) sequence
    shuffled_obs = shuffle_timesteps(real_obs)
    fake_latent = online_encoder(shuffled_obs)

    # Discriminator predicts if sequence is temporally coherent
    real_score = discriminator(real_latent)  # Should be ~1
    fake_score = discriminator(fake_latent)  # Should be ~0

    # Discriminator loss (standard GAN)
    L_disc = -log(real_score) - log(1 - fake_score)

    # Encoder ADVERSARIAL loss (confuse discriminator)
    # Encoder wants discriminator to fail → latents should be time-invariant
    L_encoder_adv = -log(1 - real_score)  # Minimize coherence signal

    # Total encoder loss
    L_encoder = L_infonce + L_vicreg + λ_adv * L_encoder_adv + L_returns + L_regime
```

**Why this works:**
- Discriminator learns to detect temporal structure in latent space
- Encoder learns to *hide* temporal structure from discriminator
- Result: Encoder must focus on time-invariant features (VPIN, funding, volatility) rather than sequence order

### **Anti-Overfitting Mechanisms**

1. **Causal GRU (not BiGRU)**
   - Unidirectional → can't cheat by seeing future context
   - Forces true autoregressive prediction

2. **Adversarial Training**
   - Explicit penalty for temporal dependence
   - Encoder learns features robust to time shuffling

3. **Temporal Jitter Augmentation**
   ```python
   # Randomly shift sequences ±4 timesteps during training
   shift = np.random.randint(-4, 5)
   obs_shifted = np.roll(obs, shift, axis=1)
   ```

4. **Gradient Penalty on Discriminator**
   - Prevents discriminator from collapsing
   - Ensures it provides meaningful gradient to encoder

### **Hyperparameters**

```python
# Model
V6_D_MODEL = 256
V6_D_LATENT = 192
V6_GRU_LAYERS = 3  # NOT bidirectional
V6_DISC_HIDDEN = 128

# Adversarial
V6_LAMBDA_ADV = 0.10  # Weight for adversarial loss
V6_DISC_STEPS = 1  # Train discriminator every step

# Training
V6_DROPOUT = 0.22
V6_WEIGHT_DECAY = 0.12  # Very strong
V6_BATCH_SIZE = 40
V6_TEMPORAL_JITTER = 4  # ±4 timestep shifts
V6_LR = 2.5e-4
V6_EPOCHS = 180
```

### **Expected Performance**

- **Shuffled IC**: 0.04-0.07 (robust to time shuffling by design)
- **Contiguous IC**: 0.06-0.10 (less temporal exploitation)
- **IC Gap**: <0.05 (minimal overfitting)
- **Hallucination Score**: <0.20 (adversarial training prevents)
- **Regime Coverage**: 3/3

---

## V7: Vision Transformer (Patch-Based Temporal-Feature Grid)

### **Motivation**

All previous models process time × features sequentially. **V7 treats the problem as 2D image understanding**: time axis × feature axis = visual pattern. Applies ViT (Vision Transformer) to learn joint temporal-feature correlations.

### **Architecture**

```
Input(13) → Create 2D Grid [Time=96, Features=13]

→ Patch Embedding (patch_size=4×4)
  [96×13 grid → 24×4 patches (after rounding) = 96 patches]
  [Each patch: 16 values → embedded to d_model=320]

→ Add Positional Encoding (2D: time_pos + feature_pos)

→ 4x TransformerBlock (heads=8, d_ff=960)
  [Standard ViT architecture]
  [Attention over 96 patches]

→ Patch Aggregation (inverse patching)
  [96 patches → 96×13 reconstruction]

→ MLP Head per timestep: [320] → [Returns | Regime | Recon]

→ Stochastic Latent (Optional RSSM on aggregated features)
```

### **Key Innovation: 2D Spatial Reasoning**

Traditional models:
```
Time: [t0, t1, t2, ..., t95]
Features: [f0, f1, ..., f12] per timestep
Process: Sequential (RNN/Transformer/Mamba)
```

V7:
```
Grid:
     f0    f1    f2    ...   f12
t0  [0.1] [0.3] [-0.2] ... [0.5]
t1  [0.2] [0.1] [0.0]  ... [0.6]
t2  ...
...
t95 [0.3] [0.4] [0.1]  ... [0.2]

Patches: 4×4 blocks
[t0:t3, f0:f3] → patch_00
[t0:t3, f4:f7] → patch_01
[t4:t7, f0:f3] → patch_10
...

Attention: Patch-to-patch relationships
```

**Learned Patterns:**
- **Vertical correlations**: How features evolve over time
- **Horizontal correlations**: Co-movements between features
- **Diagonal correlations**: Lead-lag relationships

### **Anti-Overfitting Mechanisms**

1. **Patch-Based Processing**
   - 4×4 patches aggregate local structure
   - Single timestep cannot be memorized individually
   - Forces model to learn compositional patterns

2. **2D Positional Encoding**
   - Time position + Feature position = separable
   - Model can learn time-invariant feature correlations

3. **Patch Dropout**
   ```python
   # Randomly drop 15% of patches during training
   mask = torch.rand(B, num_patches) > 0.15
   patch_emb = patch_emb * mask.unsqueeze(-1)
   ```

4. **Feature Permutation Augmentation**
   ```python
   # Randomly permute feature order (not time order)
   feature_perm = torch.randperm(13)
   obs = obs[:, :, feature_perm]
   # Model must learn feature relationships, not positions
   ```

### **Hyperparameters**

```python
# Model
V7_PATCH_SIZE = (4, 4)  # Time × Features
V7_D_MODEL = 320
V7_N_LAYERS = 4
V7_N_HEADS = 8
V7_D_FF = 960

# Augmentation
V7_PATCH_DROPOUT = 0.15
V7_FEATURE_PERMUTE_PROB = 0.30

# Training
V7_DROPOUT = 0.18
V7_WEIGHT_DECAY = 0.10
V7_BATCH_SIZE = 36
V7_LR = 3e-4
V7_EPOCHS = 160
```

### **Expected Performance**

- **Shuffled IC**: 0.02-0.04 (2D structure preserved under shuffling)
- **Contiguous IC**: 0.05-0.09
- **IC Gap**: <0.06
- **Interpretability**: Patch attention reveals time-feature patterns
- **Novel**: Completely different inductive bias from RNN/Mamba

---

## V8: Neural ODE (Continuous-Time Dynamics)

### **Motivation**

All previous models operate in discrete time (1-bar intervals). Markets are continuous - price changes happen at arbitrary timestamps. **Neural ODE learns continuous dynamics** via differential equations.

### **Architecture**

```
Input(13+32) → Initial State Encoder
  Linear(45→256) → LayerNorm → SiLU → h0 [B, 256]

→ Neural ODE Solver:
  dh/dt = f_θ(h, t)  [MLP: 256 → 512 → 256]
  h(t) = ODESolve(h0, f_θ, t=0..96, method='rk4')
  [Runge-Kutta 4th order adaptive solver]

→ Emission Network (per solved timestep):
  h(t) → [Latent | Returns | Regime | Recon]

→ Flow-Based Latent (Normalizing Flow, not RSSM):
  z0 ~ N(0, I)
  z_t = f_flow(z0, h(t))  [Affine coupling layers]
```

### **Key Innovation: Arbitrary Horizon Prediction**

Traditional models:
```python
# Fixed horizons [1, 4, 16, 64]
pred_1 = model.return_head_1(h)
pred_4 = model.return_head_4(h)
...
```

Neural ODE:
```python
# Continuous interpolation - predict ANY horizon
h_t = ode_solve(h0, f, t=0..T)
pred_h = return_head(h[t+h]) for any h ∈ [0.5, 100]

# Examples:
pred_0.5 = predict(h, 0.5)  # Half-bar ahead
pred_1.0 = predict(h, 1.0)  # 1 bar (matches discrete)
pred_7.3 = predict(h, 7.3)  # 7.3 bars (arbitrary)
```

**Physical Interpretation:**
- `dh/dt = f(h, t)` is the "market dynamics function"
- Solver integrates this to propagate state forward in continuous time
- More realistic than discrete jumps

### **Anti-Overfitting Mechanisms**

1. **Continuous Time Formulation**
   - Training on discrete bars [0, 1, 2, ..., 95]
   - But dynamics are continuous → generalizes to [0.1, 0.2, ..., 95.9]
   - Prevents overfitting to specific bar boundaries

2. **Dynamics Function Regularization**
   ```python
   # Penalize complex dynamics (Occam's razor)
   L_dynamics = ||f_θ(h, t)||² + λ_smooth * ||∂f/∂t||²
   # Simple dynamics generalize better
   ```

3. **Time Augmentation**
   ```python
   # Train on variable time grids
   t_grid = [0, 1.2, 2.5, 3.1, ..., 95.8]  # Not uniform
   # Forces model to learn smooth interpolation
   ```

4. **Energy-Based Regularization**
   ```python
   # Dynamics should conserve energy (market is mean-reverting)
   energy = 0.5 * ||h||²
   L_energy = (energy(t+1) - energy(t) - external_shock)²
   ```

### **Hyperparameters**

```python
# Model
V8_D_HIDDEN = 256
V8_DYNAMICS_LAYERS = [256, 512, 512, 256]
V8_ODE_METHOD = "rk4"  # 4th-order Runge-Kutta
V8_NUM_ODE_STEPS = 96  # Solve at discrete bars

# Flow
V8_FLOW_LAYERS = 4  # Affine coupling blocks
V8_FLOW_HIDDEN = 128

# Regularization
V8_LAMBDA_DYNAMICS = 0.01
V8_LAMBDA_SMOOTH = 0.005
V8_LAMBDA_ENERGY = 0.02

# Training
V8_DROPOUT = 0.15
V8_WEIGHT_DECAY = 0.08
V8_BATCH_SIZE = 32
V8_LR = 1e-4  # Slower for ODE stability
V8_EPOCHS = 200  # Slower convergence
```

### **Expected Performance**

- **Shuffled IC**: 0.025-0.045 (continuous dynamics are time-order agnostic)
- **Contiguous IC**: 0.04-0.07
- **IC Gap**: <0.04 (very robust)
- **Unique**: Can predict fractional horizons (t+2.7, t+10.5, etc.)
- **Interpretable**: Dynamics function reveals market "physics"

---

## V9: Mixture-of-Experts with Learned Regime Gating

### **Motivation**

Markets have distinct regimes (trending, mean-reverting, volatile). A single monolithic model must compromise across all regimes. **MoE uses specialized experts per regime** with a learned router.

### **Architecture**

```
Input(13+32) → Shared Encoder
  Linear(45→320) → 2x CausalGRU(320) → h_shared [B, T, 320]

→ Router Network (soft gating):
  h_shared → MLP(320→128→3) → softmax → [p_bear, p_neutral, p_bull]

→ Expert Networks (3 specialized models):
  Expert_Bearish:  2x MambaBlock(320) + RSSM(24×24)
  Expert_Neutral:  2x TCN(320, dilations=[1,2]) + RSSM(24×24)
  Expert_Bullish:  2x Transformer(320, heads=4) + RSSM(24×24)

→ Mixture Output:
  h_final = p_bear * Expert_bear(h) + p_neutral * Expert_neutral(h) + p_bull * Expert_bull(h)

→ Heads: Returns[4] | Regime | Recon (shared across experts)
```

### **Key Innovation: Learned Specialization**

**Expert Design Philosophy:**

1. **Bearish Expert (Mamba)**
   - Fast-moving, autocorrelated declines
   - Mamba's selective scan captures momentum
   - Trained primarily on bearish sequences

2. **Neutral Expert (TCN with dilations [1,2])**
   - Mean-reverting, choppy markets
   - Local patterns (dilation=1) dominate
   - Less long-range dependence

3. **Bullish Expert (Transformer)**
   - Trending, persistent moves
   - Global attention captures trend structure
   - Trained primarily on bullish sequences

**Router learns to:**
- Detect regime from `h_shared`
- Weight experts appropriately
- Soft mixing (not hard assignment) allows blending during transitions

### **Anti-Overfitting Mechanisms**

1. **Expert Specialization**
   - Each expert sees limited regime distribution
   - Prevents any single model from overfitting to all patterns
   - Generalization via ensemble diversity

2. **Router Regularization**
   ```python
   # Prevent router collapse (one expert dominates)
   L_router_entropy = -sum(p_i * log(p_i))  # Encourage diversity
   L_router = cross_entropy(router_logits, regime_labels) + λ_ent * L_router_entropy
   ```

3. **Regime-Balanced Sampling**
   ```python
   # Oversample rare regimes during training
   regime_counts = {bear: 1000, neutral: 5000, bull: 800}
   sample_weights = 1 / regime_counts  # Inverse frequency
   # Prevents neutral expert from dominating due to class imbalance
   ```

4. **Expert Dropout**
   ```python
   # Randomly zero out one expert's contribution with p=0.10
   # Forces router to learn robust weightings
   ```

### **Hyperparameters**

```python
# Shared
V9_D_MODEL = 320
V9_ENCODER_LAYERS = 2

# Router
V9_ROUTER_HIDDEN = 128
V9_LAMBDA_ROUTER_ENT = 0.05

# Experts
V9_BEAR_MAMBA_LAYERS = 2
V9_NEUTRAL_TCN_LAYERS = 2
V9_BULL_TRANSFORMER_LAYERS = 2
V9_RSSM_LATENT = (24, 24)  # 576 per expert

# Training
V9_DROPOUT = 0.18
V9_WEIGHT_DECAY = 0.09
V9_BATCH_SIZE = 40
V9_EXPERT_DROPOUT = 0.10
V9_REGIME_BALANCED_SAMPLING = True
V9_LR = 2e-4
V9_EPOCHS = 170
```

### **Expected Performance**

- **Shuffled IC**: 0.035-0.055 (ensemble diversity helps)
- **Contiguous IC**: 0.07-0.12
- **IC Gap**: <0.08
- **Regime Coverage**: 3/3 (by design)
- **Interpretability**: Router weights reveal detected regime
- **Ensemble**: 3 orthogonal architectures in one model

---

## Architecture Comparison Matrix

| Model | Core Tech | Complexity | Anti-Overfit | Shuffled IC (Est.) | IC Gap (Est.) | Ensemble Value |
|-------|-----------|------------|--------------|-------------------|---------------|----------------|
| **V1** | Transformer + RSSM | O(T²) | Moderate | 0.02-0.04 | <0.08 | Baseline |
| **V2** | BiGRU + JEPA | O(T) | ✗ Weak | ~0.00 | ~0.94 | ✗ Broken |
| **V3** | TCN + GRU + RSSM | O(T) | Good | 0.03-0.05 | <0.07 | High |
| **V4** | Mamba + RSSM | O(T) | Unknown | TBD | TBD | High |
| **V5** | Mamba + LocalAttn | O(T) + O((T/4)²) | Strong | 0.03-0.05 | <0.10 | Very High |
| **V6** | CausalGRU + Adv JEPA | O(T) | Very Strong | 0.04-0.07 | <0.05 | Very High |
| **V7** | Vision Transformer (2D) | O(P²), P=96 | Strong | 0.02-0.04 | <0.06 | Very High |
| **V8** | Neural ODE | O(T × ODE_steps) | Very Strong | 0.025-0.045 | <0.04 | Very High |
| **V9** | Mixture-of-Experts | O(T) × 3 | Strong | 0.035-0.055 | <0.08 | Maximum |

---

## Implementation Priority

### Phase 1 (Immediate - Next 2 Days)

1. **V6 (Adversarial JEPA)** - Fixes V2's critical overfitting
   - Time: ~4 hours implementation
   - Impact: Proves adversarial training works
   - Priority: **CRITICAL**

2. **Run Robust Validation on V1/V3/V4**
   - Time: ~1 hour
   - Impact: Understand baseline overfitting
   - Priority: **HIGH**

### Phase 2 (Week 1)

3. **V5 (Mamba + LocalAttention)** - Best V4 upgrade
   - Time: ~3-4 hours
   - Impact: Orthogonal to V4, proven architecture
   - Priority: **HIGH**

4. **V9 (Mixture-of-Experts)** - Ensemble in single model
   - Time: ~5 hours
   - Impact: Regime specialization
   - Priority: **MEDIUM**

### Phase 3 (Week 2)

5. **V7 (Vision Transformer)** - Novel approach
   - Time: ~6 hours
   - Impact: Completely different inductive bias
   - Priority: **MEDIUM**

6. **V8 (Neural ODE)** - Research/interpretability
   - Time: ~8 hours
   - Impact: Continuous-time dynamics, novel
   - Priority: **LOW** (interesting but complex)

---

## Ensemble Strategy

### Production Deployment (Final System)

**Tier 1: Primary Models** (highest shuffled IC)
- V6 (if retrained with adversarial)
- V5 (Mamba+Attention)
- V3 (TCN-GRU-RSSM) [already trained]

**Tier 2: Diversity Models** (orthogonal architectures)
- V9 (MoE)
- V7 (ViT)

**Tier 3: Interpretability**
- V1 (Transformer - attention analysis)
- V8 (Neural ODE - dynamics visualization)

**Voting Scheme:**
```python
# Weighted average by shuffled IC
weights = {
    "v5": 0.25,
    "v6": 0.25,
    "v3": 0.20,
    "v9": 0.15,
    "v7": 0.10,
    "v1": 0.05,
}

pred_return = sum(w * model.predict(obs) for model, w in weights.items())
pred_regime = majority_vote([model.predict_regime(obs) for model in models])
```

**Confidence Gating:**
```python
# Only trade if models agree
model_agreement = std([model.predict(obs) for model in models])
if model_agreement > threshold:
    action = "hold"  # Models disagree, too risky
else:
    action = ensemble_prediction
```

---

## Success Metrics

For each V5-V9 model, require:

1. **Shuffled IC > 0.025** (true generalization)
2. **IC Gap < 0.10** (max 10% overfitting)
3. **Hallucination Score < 0.30**
4. **Regime Coverage**: IC > 0.01 in all 3 regimes
5. **Forward Walk Stability > 0.60**
6. **Reconstruction MSE < 0.15**

**Ensemble Bonus:**
- Ensemble shuffled IC > best single model + 0.01
- Ensemble IC stability > 0.75

---

## Next Actions

1. ✅ Robust validation framework created
2. ✅ V2 overfitting documented
3. ⏳ **Implement V6 (Adversarial JEPA) - PRIORITY 1**
4. ⏳ **Robust validate V1/V3/V4 - PRIORITY 1**
5. ⏳ Implement V5 (Mamba+Attention)
6. ⏳ Implement V9 (MoE)
7. ⏳ Implement V7 (ViT)
8. ⏳ Implement V8 (Neural ODE)
9. ⏳ Cross-validate all models
10. ⏳ Build ensemble system

---

**Status:** Design Complete - Ready for Implementation
**Author:** Claude Opus 4.6
**Date:** 2026-02-15
