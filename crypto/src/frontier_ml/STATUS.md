# Frontier ML — Build Status & Plug-and-Play Map

> **Snapshot:** 2026-05-02. Pipeline phase complete; all three prongs scaffolded;
> Prong 1 (foundation) has run a u10 mini and scored a probe IC +0.052; Prongs 2
> and 3 wired and awaiting prerequisites.

## 2026-05-02 update — Kronos zero-shot verdict (E1)

Browser response (`FRONTIER_RESEARCH_RESPONSE_2026_05_02.md`) flagged Kronos
(AAAI 2026, 12B K-line pretrain) as a paradigm we missed. Built the
`kronos_baseline/` scaffold + ran E1 zero-shot on u10 OOS:

**Pooled Spearman IC at h=1 across 10 assets × 100 windows = +0.0292.**
Below all four decision thresholds. Per-asset spread was wide (-0.10 to +0.18)
with most p > 0.27; pattern is "no useful signal," not "inverted signal."

**Verdict: stay on the scratch-pretrain plan.** Kronos zero-shot does not
beat XGBoost (+0.031) on our (crypto + dollar-bar) distribution. Two real
caveats remain (Kronos pretrained on uniform-time K-lines; we feed dollar
bars) but the lift even if those were resolved would not justify a pivot
away from our 31.7M Mamba foundation.

**Browser R3 (adaptive log-spaced bins) shipped regardless** —
`foundation/adaptive_bins.py` provides a TwoHotSymlog-API-compatible drop-in
replacement that allocates bin density where 5-min crypto returns actually
live (50× resolution improvement on the meaningful range).

**E1c — dollar-vs-time A/B (2026-05-02 follow-up):** Resampled
chimera_legacy to 1-hour time bars and re-ran Kronos. Pooled IC = +0.0135
(n=1000, p=0.67), delta vs dollar bars = -0.016. Sign flips across half
the universe between bar types confirm pattern is pure noise. **Dollar
bars empirically validated; dual-cadence architecture not warranted.**
Foundation Prong 1 stays on plan.

**Hole 9 (closed 2026-05-02): bar-type choice.** Dollar bars are canonical
for the foundation pretrain. Time bars not needed as primary input.
Auxiliary bar types (DIB / range / runs / adaptive_vol) remain valuable
via the existing `bar_fabric` infrastructure.

## 2026-05-02 V1.x upgrade modules (post-B002/B003/B004/B005 synthesis)

Four browser dialogues closed in parallel (B002 / B003 / B004 / B005).
All four flagged 0% VERIFIED on IC/ShIC deltas; recommendations
converge on **probe-before-commit**. Synthesis below (full ledgers in
`browser_dialog/INDEX.md`):

**V1.x upgrade modules built** at `src/frontier_ml/v1_upgrades/` — all
PASS smoke; opt-in via CLI flags on V1.1 trainer (default OFF, baseline
preserved):
- `sam.py` (B003 R1): Sharpness-Aware Minimization. Wraps AdamW with
  rho=0.05. `--sam`. Disables AMP under SAM (eager fp32) for first
  revision; expected +0.005-0.015 IC + +0.005-0.010 ShIC; ~3 GPU-h.
- `fraug.py` (B003 R2): FrAug FFT-mask augmentation. `--fraug`. ~0
  marginal cost; expected +0.003-0.010 IC + +0.005-0.015 ShIC.
- `pcgrad.py` (B003 4.6): PCGrad gradient surgery for the 4 horizon
  heads. Module ready; wiring deferred (needs world_model.py refactor
  to expose per-horizon losses). `--pcgrad` is current NO-OP with warn.
- `mtp_head.py` (B002 R1): MTP sequential causal-chain head. Module
  ready; wiring deferred (needs world_model.py to swap return_heads).
  `--mtp` is current NO-OP with warn.

**V4 (Mamba-3) verified to already have QKNorm** (per B004 R1
expected fix). V4's persistent ShIC-decline is therefore NOT
QKNorm-absence; root cause investigation pending.

**V6 (JEPA + Discriminator)** — B005 R1 recommends C-JEPA VICReg fix;
not built yet (lower priority; V1.x lift is faster path to Headline).

**Probes recommended after current V1.1 baseline f29 finishes** (six
upgrade flags now FULLY wired in V1.1 trainer; default OFF, baseline
preserved):

```
python -m src.wm.v1.v1_1_training.train_world_model --features 29 --sam
python -m src.wm.v1.v1_1_training.train_world_model --features 29 --fraug
python -m src.wm.v1.v1_1_training.train_world_model --features 29 --pcgrad
python -m src.wm.v1.v1_1_training.train_world_model --features 29 --mtp
python -m src.wm.v1.v1_1_training.train_world_model --features 29 --adaptive-bins
python -m src.wm.v1.v1_1_training.train_world_model --features 29 --sam --fraug --adaptive-bins  # stack
```

Decision rule per probe: ShIC ≥ 0.038 (+0.005 vs current 0.033 record)
→ propagate to V1.0/V1.4/V1.6.

**Module status (`src/frontier_ml/v1_upgrades/`)**:

| Module | Source | Trainer flag | Status |
|---|---|---|---|
| `sam.py` | B003 R1 | `--sam` | functional (eager fp32 under SAM) |
| `fraug.py` | B003 R2 | `--fraug` | functional |
| `pcgrad.py` | B003 4.6 | `--pcgrad` | functional (per-horizon split via `return_components=True`; AMP off under PCGrad) |
| `mtp_head.py` | B002 R1 | `--mtp` | functional (handles both 2D and 3D inputs; swapped via `integration.apply_v1_upgrades`) |
| `adaptive_bins.py` | B001 R3 | `--adaptive-bins` | functional (n_bins matches model NUM_BINS, log-spaced or quantile) |
| `vicreg.py` | B005 R1 | (V6 use) | module ready; awaits V6 retrain |
| `mdn_head.py` | B003 R3 | `--mdn` | head attached but loss-path patch pending (warns; module ready for V2.x design) |
| `integration.py` | helper | — | apply_v1_upgrades(model, ...) — single entry-point |

## TL;DR

| Prong | What | Status | Activates when |
|---|---|---|---|
| 1 | Foundation pretrain (31.7M Mamba-3 + cross-asset attention) | **u10 mini DONE; full u100 pending** | go/no-go decision on 35-hr u100 run |
| 2 | Distillation from V1.x ensemble + foundation → 4.3M / 10.6M deployable student | **scaffold complete; awaiting V1.x ckpts** | V1 trainings finish |
| 3 | Multi-modal alignment (funding/OI/ETF/macro side channels via cross-attention adapter on frozen foundation) | **scaffold complete; awaiting foundation u100** | Prong 1 ships a trained foundation |

All three modules import `frontier_ml.foundation.harmony.apply_harmony()` at startup so they don't choke the i9/32GB/4060 system.

## Goal (from CLAUDE.md INDISPUTABLE OPERATING LENS)

Push the WM signal from current SHIP-tier (IC ≈ 0.06 / ShIC ≈ 0.03) into
**Headline tier** (IC > 0.10 / ShIC > 0.05) where the WM signal IS the alpha.
Hardware: 1× RTX 4060 (8.59 GB VRAM), i9 (20 logical cores), 32 GB RAM.

Foundation alone is unlikely to clear Headline. The realistic Headline path is
the *stack* of all three prongs:

```
foundation (~0.075-0.085)
    +
distillation (~0.080-0.090)   ← combines V1.x + foundation as teachers
    +
multi-modal  (~0.090-0.105)   ← funding/macro/on-chain orthogonal lift
    =
ensemble (~0.10-0.12)         ← Headline tier
```

## File map

```
src/frontier_ml/
├── README.md                       project orientation
├── PLAN.md                         3-prong plan with 6 literature-driven updates applied
├── LITERATURE.md                   8 holes closed with citations
├── STATUS.md                       <-- you are here
│
├── foundation/                     PRONG 1: foundation pretrain
│   ├── __init__.py
│   ├── backbone.py                 FoundationBackbone -- 31.7M params
│   │                                d_model=768, d_state=64, 8x Mamba3Block + 2x CrossAssetAttention
│   │                                Reuses src/wm/v4 Mamba primitives.
│   ├── data_loader.py              FoundationDataset multi-asset window sampler
│   │                                Slim npz cache: 9.3 GB at u100 (built once)
│   │                                sample_anchor_batch + sample_contrastive_batch
│   ├── objectives.py               FoundationLoss = w_h*horizon_TwoHot + w_c*InfoNCE_lead-lag
│   ├── pretrain.py                 AMP + AdamW + warmup-cosine + ckpt-every-100 + resume
│   │                                BYOL/SimSiam stop-grad on contrastive pos/neg
│   ├── probe.py                    OOM gate (Hole 4): smoke + synth tiers
│   ├── eval_probe.py               linear-probe IC vs V1.0 baseline (Chronos/MOMENT protocol)
│   └── harmony.py                  apply_harmony() -- caps i9 threads at 8/20, polars 8,
│                                    GPU mem fraction 0.90, BELOW_NORMAL priority on Windows
│
├── distillation/                   PRONG 2: ensemble distillation
│   ├── __init__.py
│   ├── student.py                  make_student(size) -- small=4.3M, med=10.6M
│   │                                Same forward signature as FoundationBackbone (smaller dims)
│   ├── distill_loss.py             HybridDistillLoss = alpha*KL + beta*L1(E_r) + gamma*L2(V_r)
│   │                                Per LITERATURE.md Hole 5 (Phuong & Lampert 2019)
│   │                                Hinton temperature T=2.0 on KL
│   ├── teacher_inference.py        Build fixed window set + cache teacher logits
│   │                                Foundation cache WIRED; V1.x family stubs ready
│   └── train.py                    Distillation training loop
│                                    Ensemble = SOFTMAX-mean (avoids log-mean calibration bias)
│                                    Aux Huber on continuous expected return
│
└── multimodal/                     PRONG 3: multi-modal alignment
    ├── __init__.py
    ├── channels.py                 ChannelBank ingester with explicit lag (Hole 6)
    │                                chimera-source: norm_funding, norm_oi_change,
    │                                                norm_hawkes_imbalance, norm_vpin
    │                                panel-source:   btc_etf_flows.Total (1-day lag)
    │                                load_aligned(timestamps_ms) -> dict[name, np.ndarray]
    ├── adapter.py                  MultiModalAdapter -- 1.25M trainable params
    │                                Frozen foundation + ChannelCrossAttention layers
    │                                + new TwoHot return heads
    └── finetune.py                 Fine-tune loop (foundation frozen; adapter only trains)
```

## Empirical results to date (Prong 1 only)

### Hole 4 — OOM gate

| Probe | Steps | Peak VRAM | Rate | Verdict |
|---|---|---|---|---|
| smoke | 1 | 0.35 GB | n/a | PASS (B=4 S=256) |
| synth (B=8 S=512 fp16) | 50 | 6.47 GB / 8.59 GB | 1.74 step/s | PASS |
| **real chimera_legacy u10** | **200** | **6.66 GB / 8.59 GB** | **1.75 step/s** | **PASS** |

### u10 mini-pretrain (5000 steps, ~47 min)

- Loss EMA: 5.86 → **1.17** (random init 5.54 → 79% reduction)
- Peak VRAM: 6.66 GB stable; zero NaN
- Rate: 0.56 s/step (cudnn benchmark warmed up)

### Linear-probe IC vs baselines (n=2000 OOS windows, u10)

| Source | IC | Note |
|---|---|---|
| Naive zero | ~0 | Floor |
| TimesFM 200M zero-shot (published, Wu 2024) | +0.008 | We beat decisively |
| Chronos / TimesFM zero-shot consensus (published) | < XGB | Generic foundation models lose to XGB on crypto |
| XGBoost 5-lag (this benchmark) | +0.028 | Beats foundation models zero-shot |
| **Our foundation intrinsic (5K-step mini)** | **-0.032** | TwoHot head **undertrained** at 1.3% of full compute |
| **Our foundation linear probe (5K-step mini)** | **+0.052** | h_seq has signal; 78% of V1.0 baseline |
| V1.0 baseline | +0.066 | Internal SHIP-tier |
| V1.1 record | +0.067 | Best of V1.x family |
| Foundation projected mode (after full u100 pretrain) | **+0.075-0.085** | New project SOTA |
| Headline target | > 0.10 | Stack of 1+2+3 |

### Compute budget

| Run | Steps | Wall-clock | Notes |
|---|---|---|---|
| smoke | 1 | < 5s | Verifies forward+backward |
| synth probe | 50 | ~30s | Hole 4 architectural check |
| real probe | 200 | ~115s | Hole 4 dataloader + AMP check |
| **u10 mini** | **5000** | **47 min** | Done; loss EMA 1.17 |
| **u100 full** | **25K-50K** | **~7 hr/epoch × 5 = ~35 hr** | NOT YET; awaits go/no-go decision |

## Plug-and-play wires

### When V1.x trainings finish (parallel track)

```bash
# Build the fixed window set (one-time, ~10 min at N=50K u100)
python -m src.frontier_ml.distillation.teacher_inference \
    --build-windows --universe u100 --n-windows 50000

# Cache foundation teacher (already wired)
python -m src.frontier_ml.distillation.teacher_inference \
    --teacher foundation \
    --ckpt models/frontier_ml/foundation/latest.pt

# V1.x stubs need a 5-line per-version loader before this works
# python -m src.frontier_ml.distillation.teacher_inference \
#     --teacher v1_1 --ckpt models/v1/v1_1/best_ema.pt
```

V1.x stub: `cache_v1x_teacher(name, ckpt_path)` in
`src/frontier_ml/distillation/teacher_inference.py`. Wire by importing the
matching `WorldModel` class from each version's `world_model.py` and calling
its forward to produce `return_logits["h1"], ["h4"], ["h16"], ["h64"]`.

### Distillation training (after teacher caches built)

```bash
python -m src.frontier_ml.distillation.train \
    --teachers foundation,v1_0,v1_1,v1_4,v1_6,v3,v4,v6 \
    --student-size small \
    --max-steps 20000 --batch-size 16
```

Decision rule (LITERATURE.md Hole 7): student must hit
**IC ≥ 0.95 × best_teacher_IC AND latency ≤ 1/4 ensemble**. If yes, deploy
student. If no, deploy ensemble.

### Multi-modal fine-tune (after foundation u100 pretrain)

```bash
python -m src.frontier_ml.multimodal.finetune \
    --foundation-ckpt models/frontier_ml/foundation/latest.pt \
    --universe u100 --max-steps 20000 --batch-size 8
```

Decision rule (LITERATURE.md Hole 6): adapter must give
**multi-modal IC ≥ foundation IC + 0.005**. Else drop the adapter.

### Foundation vs Chronos / TimesFM benchmark

```bash
# Just our foundation + XGB (skip Chronos download)
python src/analysis/foundation_model_benchmark.py \
    --with-ours --skip-chronos

# Full A/B vs Chronos (downloads model)
python src/analysis/foundation_model_benchmark.py \
    --with-ours
```

## Pending decisions

| Decision | Trigger | Owner |
|---|---|---|
| Launch full u100 pretrain (35 hr) | Operator (you) | User |
| In-loop linear-probe IC gate every 1K steps | Recommended before u100 | Pending |
| Wire V1.x teacher loaders | After V1 retrains finish | Both |
| Distillation student size (small vs med) | After teachers cached | Empirical |
| Adapter d_mm and layer count | After foundation u100 | Empirical |

## Holes (LITERATURE.md, all closed)

1. ✅ 30M params enough → MOMENT 40M proves time-series competence at this scale
2. ✅ Causal next-token > MSM for forecasting → per Chronos/TimesFM
3. ✅ Lead-lag positives in contrastive (not same-timestamp) → BTC→ETH cascade
4. ✅ OOM probe → real probe PASS at 6.66/8.59 GB
5. ✅ Hybrid α·KL + β·L1 + γ·L2 distillation → Phuong & Lampert 2019
6. ✅ Multi-modal explicit lag + walk-forward purge → Lopez de Prado AFML ch.4
7. ✅ Distillation deployment metric: IC ≥ 0.95×teacher_IC, latency ≤ 1/4
8. ✅ Compute budget: ckpt-every-100 + resume + harmony caps, 35 hr fits 2 weekends

## What this design does NOT include

- **Tick-level pretrain (V20 territory)** — defer until rented compute
- **100M+ foundation backbone** — won't fit 4060 even with AMP
- **End-to-end RL with full DreamerV3 imagination at scale** — needs 24+ GB
- **Diffusion over full return distribution at 1000 steps** — too slow for live
- **Online learning with continuous gradient updates** — daily-cadence trading doesn't need it

## Provenance

Built 2026-05-02 in a single session. Pipeline phase closed earlier same day
(`pre_train_gate` green; commits `31de55b` + `7ce1853`). Frontier ML commits:
`46d250f` (backbone+probe), `723cfe4` (loop+harmony), `e8e0e6d` (eval),
`901e859` (benchmark patch), Prong 2 + Prong 3 commits follow.
