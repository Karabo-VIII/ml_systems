# WM Cohort — World-Class Verdict + Upgrade Plan (2026-06-10)

> From an exhaustive 12-model red-team audit. "World-class for the arsenal" = a **diverse, anchored ensemble**
> feeding V10 under the post-reset SETUP/MOVE compound objective — NOT 20 identical models. The binary test:
> a real information bottleneck (RSSM-categorical OR VIB) **+ a real reconstruction decoder** (lifts ShIC off
> 0.000), on a **non-dominated** low-SNR inductive bias. Per-model evidence: the audit task `wmnrkki1c` output.

## The classification (12 models)

| Model | Bias (class) | Anchor | Recorded ShIC | Verdict |
|---|---|---|---|---|
| **V1.1** | RSSM + VSN + move/regime heads | REAL (full) | proven backbone | WORLD-CLASS (done) |
| **V12** | Cross-asset, recon anchor grafted | REAL (full) | the anchor donor | WORLD-CLASS (done) |
| **V3** | WaveNet gated-TCN + RSSM | REAL | 0.030 | **UPGRADE** (wire levers) |
| **V4** | Mamba-3 SSM + RSSM | REAL | **0.0136 (positive!)** | **UPGRADE** (wire levers) |
| **V6** | CausalGRU JEPA + VIB+recon | REAL | none on record | **UPGRADE** (wire levers + recorded run) |
| **V8** | Neural-ODE + RSSM | REAL | 0.030 (qual.) | **UPGRADE** (wire levers) |
| **V13** | TFT (VSN + GRN + attn) | HALF (VIB, no recon) | unknown (live path) | **GRAFT recon → UPGRADE** |
| **V11** | WaveNet-TCN + regime-MoE | HALF (VIB, no recon) | 0.000 | BENCHWARMER (graft; redundant w/ V3) |
| **V14** | WaveNet + DDPM diffusion | PARTIAL | 0.000 | BENCHWARMER (graft; sizing role, HIGH effort) |
| **V23** | xLSTM (sLSTM+mLSTM) | PARTIAL (recon stub) | unknown (never trained) | **GRAFT + FIX TRAINER** (highest-upside reserve) |
| **V22** | Pure iTransformer | NONE (recon=zeros) | 0.000 | **RETIRE (dominated + redundant)** |
| **V24** | TimesNet (FFT+2D-conv) | PARTIAL | unknown | **RETIRE (FFT thesis invalidated by dollar bars)** |
| **V25** | Pure iTransformer (frontier) | NONE (recon=zeros) | 0.000 (rank-collapse) | **RETIRE (rebuild, not upgrade)** |

## The world-class cohort (7) — one strong representative per non-dominated bias
**V1.1** (RSSM backbone) · **V12** (cross-asset) · **V3** (gated-conv) · **V4** (SSM/Mamba) · **V6** (JEPA self-supervised) ·
**V8** (continuous-time ODE) · **V13** (variable-selection/TFT). No two share an inductive bias → maximal ensemble decorrelation.

## The upgrade plan (code only; load-bearing first)
- **Tier A — UPGRADE (anchor already real; wire the shared move/regime + VSN levers):** V3, V4, V6, V8. Mechanical guarded-line wiring + a recorded run. ~3.5–4.5 GPU-d each.
- **Tier B — GRAFT then UPGRADE:** V13 — graft the V12-style recon decoder onto its VIB latent (the keystone), then levers. ~5 GPU-d.
- **Tier B′ — FIX TRAINER then GRAFT then UPGRADE:** V23 — fix the ~4-line un-adapted-V22 trainer (AttributeError on import), graft recon, wire levers. ~0.5 + 4 GPU-d.
- **Deprioritized:** V11 (redundant), V14 (orthogonal sizing, HIGH effort).
- Graft donor for every recon-graft: `src/wm/v12/v12_training/world_model.py:222/434/702-710`.

## Retire — and why (honest)
- **V22 / V25** — dominated pure-iTransformer; cross-feature attention disabled-by-default and sign-flips when on; no real anchor. Rescuing needs BOTH a grafted anchor AND a new bias = a *rebuild*, and V12 already delivers the anchor on a better backbone.
- **V24** — its entire thesis is clock-periodicity (8h/24h/7d), but it trains on **dollar bars** (volume-sampled) where those cycles don't exist; its inductive bias is invalidated by its own data choice.

## The binding caveat (do not skip)
Every "world-class" label here is **structural/projected until a fresh held-out run records it** under the
SETUP/MOVE **compound** objective (IC/ShIC survive ONLY as within-WM diagnostic gates: ShIC>0.015, ratio>0.3 —
NOT the objective; per-bar IC is banned as primary). The code upgrades make the cohort **world-class-capable**;
the **proof is the training run** (your compute). V4's positive-but-sub-gate ShIC (0.0136) and V6's
missing-on-record ShIC are the live reminders the anchor's payoff must be *measured*.

## Smart training order
**V1.1 A/B (`baseline` vs `vsn_fr`) is the go/no-go that de-risks the whole move/regime direction.** Run it first
(see `docs/WM_TRAINING_RUNBOOK.md`). If it wins on held-out compound → train the diverse cohort (V3/V4/V6/V8/V13).
If it doesn't → the direction is refuted and the cohort trainings are deferred, not wasted-on.
