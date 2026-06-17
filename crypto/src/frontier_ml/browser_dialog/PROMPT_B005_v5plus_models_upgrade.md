# B005 — V5+ model upgrade review (V6 / V8 / V9 / V10 / V11 / V12 / V13 / V14 / V15-V19)

> **Status:** OPEN  •  **Sent:** 2026-05-02
> **For:** an `@browser`-routed Claude Code session with WebSearch / WebFetch.
> **From:** the model state in `src/wm/v{6,8,9,10,11,12,13,14,15,16,17,18,19}/`.
> **Tone:** /un — direct, ship-or-concede.
> **Isolation:** per `memory/feedback_search_reliability_protocol.md` §9
>  (added 2026-05-02 per user directive "Seek info in such a way as to not
>  break result quality"). This prompt is INTENTIONALLY ISOLATED from
>  B001 / B002 / B003 / B004. Do NOT cross-cite their conclusions.
>  Each prompt gets its own raw-fetch budget. Findings from sibling
>  prompts are NOT evidence in this response.

## Mission framing

The V5+ cohort is the project's **architecture-experiment + foundation-grade
+ deferred-RL** band. V1.x has the SHIP-tier track; V5+ models are intended
to either (a) replace V1.x via novel architecture, or (b) provide ensemble
diversity for V10, or (c) reach the Ambitious / Capacity tiers via
foundation-grade pretrain (V18+).

**On-disk state** (verified at prompt-write time via `ls src/wm/v*`):

| Ver | Architecture | Status | Notes |
|---|---|---|---|
| V5 | (older SSM) | ARCHIVED | in `backups/BKP_20260429_MODEL_HARMONIZATION/v5/` |
| V6 | Causal JEPA + Discriminator | ACTIVE (`src/wm/v6/`) | per WM_FINDINGS: ShIC decline 0.0236→0.0204→0.0201 (L2 training stability) |
| V7 | (archived) | ARCHIVED | in `backups/BKP_20260429_MODEL_HARMONIZATION/v7/` |
| V8 | Neural ODE (RK4) | ACTIVE (`src/wm/v8/`) | per WM_FINDINGS: no training logs; RK4 = 4× compute factor |
| V9 | GRU + 3-expert MoE | KILL-tier (`src/wm/v9/`) | per WM_FINDINGS: IC ≈ 0.007, router collapse documented |
| V10 | Meta-ensemble router | DEFER (`src/wm/v10/v10_meta/`) | gated by ≥ 2 trained inputs |
| V11 | WaveNet + MoE + Discriminator | FROZEN (`src/wm/v11/v11_training/`) | per CLAUDE.md L339 deprecated; ACTIVE_HORIZONS already restored to [1,4,16,64] per WM_FINDINGS patch |
| V12 | Cross-Asset Attention | FROZEN (`src/wm/v12/v12_training/`) | dead code in standard runner per fix log; multi_asset attention path UNREACHED |
| V13 | TFT (Variable Selection Networks) | FROZEN (`src/wm/v13/v13_training/`) | no recent training; clean ACTIVE_HORIZONS |
| V14 | Diffusion return distribution | FROZEN (`src/wm/v14/v14_training/`) | dual-path (TwoHot + diffusion); CC2 risk: meta-learner consumes scalars |
| V15 | PatchTST encoder stub | LIBRARY-ONLY (`src/wm/v15/patchtst_encoder.py`) | no trainer |
| V16 | DreamerV3 | DEFER (`src/wm/v16/v16_training/`) | smoke only; needs full v51 build |
| V17 | TD-MPC2 | DEFER (`src/wm/v17/v17_training/`) | smoke only; planning at 384× per-step |
| V18 | Chronos finetune | KILL-projected (`src/wm/v18/v18_training/`) | foundation-model paradigm conceded beyond H=1 per B001 |
| V19 | V1.x retrained on f121 | DEFER (`src/wm/v19/v19_training/`) | gated by full v51 build |

## Tasks (priority-ordered)

### Task 1 — V6 (JEPA + discriminator) ShIC-decline fix

V6 fails on the same axis V4 fails: ShIC declines mid-training. V4's failure
got QKNorm/Mamba-3 fix in a separate dialog. V6's failure mode is different:
discriminator collapse / EMA target encoder drift / VICReg/InfoNCE balance.

Search:
1. **JEPA discriminator stability fixes 2024-2026** — has subsequent JEPA
   work shipped a recipe that prevents discriminator-loss asymptote / target
   encoder collapse?
2. **EMA momentum for target encoder** — is 0.995 / 0.999 / cosine-schedule
   the 2025-2026 norm? Has anyone published an ablation?
3. **VICReg variance/invariance/covariance term balance** — what loss-weight
   schedule has been validated for time-series JEPA at 2-5M params?
4. **InfoNCE temperature for time-series** — is the project's 0.1 still
   defensible per 2025 work?

Per fix: paper, claim, applicability, expected ShIC stabilization.

### Task 2 — V8 (Neural ODE) literature freshness

V8 has zero training logs. RK4 = 4× forward passes per step. Before
investing GPU-hours, check:
1. Has 2024-2026 work shipped Neural-ODE variants for time-series that
   beat RK4 cost (e.g. fixed-step ODE, learned-step ODE, latent ODE)?
2. Specifically: are continuous-time models still competitive vs Mamba /
   Hyena for financial forecasting at <5M params, OR has the field
   abandoned NODEs for sequential applications?
3. Is V8 a defensible architectural slot, or should it be archived alongside
   V5/V7?

### Task 3 — V11 / V13 / V14 unfreeze decision

CLAUDE.md L339 marks these as deprecated/frozen but they're not archived.
WM_FINDINGS scoring put V12 at "VALIDATE-FIRST highest interest" because of
its dead-code structural bug. V11 / V13 / V14 are scored 29-36 + ?.

Investigate:
1. **V11 (WaveNet+MoE+Discriminator)**: would 2025 sparse-MoE recipes
   (256-expert / fine-grained) salvage it, OR is the combined-architecture
   pattern just inferior to single-paradigm bets?
2. **V13 (TFT)**: has TFT (Lim 2021) been superseded? VSN per-timestep
   feature selection IS unique in our cohort. Is there 2024-2026 work
   that improved on VSN?
3. **V14 (Diffusion return distribution)**: has 2025 work scaled diffusion-
   based forecasting to financial regime, AND has anyone published
   downstream-strategy IC lift specifically from quantile-vector inputs
   (vs scalar mean inputs)?

Per version: REVIVE / STAY-FROZEN / KILL with rationale.

### Task 4 — V15 / V16 / V17 (M2 cohort) wiring readiness

V15-V17 have smoke tests but no trainers. Per WM_FINDINGS they're "DEFER
gated by full v51 build" (Job 2).

Search:
1. **PatchTST 2024-2026 status** — has the 2023 baseline been overtaken
   for time-series forecasting at small scale, or is it still the
   reference channel-independent encoder?
2. **DreamerV3 trading applications** — has DreamerV3 (Hafner 2023) been
   successfully wired to financial trading anywhere? Live performance?
3. **TD-MPC2 (Hansen 2024) in non-robotics domains** — any time-series /
   trading / financial application?

Per version: WIRE-NOW / WAIT-FOR-V51 / DROP with evidence.

### Task 5 — V18 (Chronos finetune) - is the KILL still warranted?

B001 (closed 2026-05-02) tested Kronos-small zero-shot on chimera_legacy
1h bars; pooled IC = +0.0292; below all thresholds. **B005 must NOT cite
that result** per §9 isolation. Re-investigate independently:

1. Has Chronos-2 (released Oct 2025) materially changed the foundation-
   model-on-crypto verdict?
2. Have ANY foundation-model finetune cycles on crypto returns been
   reported with IC > 0.05 in 2024-2026 literature?
3. Is the V18 finetune still worth a 1-2 GPU-h cycle, or is the paradigm
   closed?

### Task 6 — V19 (V1.x retrained on f121 frontier features)

V19 is gated by the full v51 build. Independent of v51 timing:
1. Is "more features + same architecture" empirically known to lift IC at
   < 5M params on financial data? Specifically the question: at 121-feature
   input dim vs 34-feature input dim, do small models actually use the
   extra channels, or do they regularize them away?
2. Any 2024-2026 ablation on input-dimension scaling at fixed param
   budget?

## Output format

Return one document with these sections:

1. **Executive verdict** (≤ 250 words). Per-version action plan.
2. **V6 ShIC-decline fix candidates** (Task 1).
3. **V8 NODE literature freshness** (Task 2).
4. **V11 / V13 / V14 unfreeze decisions** (Task 3) — REVIVE / STAY-FROZEN / KILL.
5. **V15 / V16 / V17 wiring readiness** (Task 4).
6. **V18 KILL recheck** (Task 5).
7. **V19 input-dim scaling literature** (Task 6).
8. **Top 5 next retrains across V5+ cohort** in priority order.

## Confidence tagging (mandatory per memory/feedback_search_reliability_protocol.md §1-§8)

Every load-bearing numerical claim (IC, ShIC, Sharpe, %-improvement, dates,
param counts, money) MUST be tagged inline:

- `[VERIFIED]` — raw source fetched (arxiv abstract page, GitHub raw README,
  HuggingFace `/api/models/<id>` JSON, paper HTML/PDF body — NOT a
  summarized WebFetch response).
- `[REPORTED]` — sourced from a WebSearch snippet OR a summarized WebFetch;
  not yet re-checked against raw text.
- `[INFERRED]` — derived/computed/extrapolated; not directly stated.

End the response with a **Reliability ledger** counting VERIFIED vs
REPORTED vs INFERRED. If REPORTED > 0 on any decision-gating number,
surface those in a `## Caveats` section.

Never let a recommendation sound more confident than its verification
level.

## Result-quality isolation (mandatory per protocol §9, added 2026-05-02)

This prompt is run in isolation from B001-B004:
- DO NOT cite B001 Kronos zero-shot result, B002 frontier-lab findings,
  B003 V1.x recommendations, or B004 V3/V4 recommendations as evidence
  in this response.
- Each search-budget claim is fresh; if a paper was VERIFIED in a prior
  dialog, it stays REPORTED here unless re-fetched within this prompt's
  budget.
- Stop conditions below are PER-PROMPT.

## Operational constraints

- Hardware: 1× RTX 4060 (8.59 GB VRAM), i9 20 cores, 32 GB RAM.
- V6 retrain: ~5-7 GPU-h.
- V8 retrain: ~10-15 GPU-h (RK4 4× factor).
- V11 / V13 / V14 retrains: ~3-5 GPU-h each.
- V16 / V17 trainers don't exist; wiring effort 0.5-1 GPU-d engineering.
- V18 finetune: ~2 GPU-h.
- ShIC > IC × 0.5 must hold.
- No emojis in any output that touches Python files (Windows cp1252).

## Time / cost budget

- WebSearch calls: 8-12 (this is a 13-version cohort; budget allocated
  by NEED not by version count).
- WebFetch calls: 3-5 (prioritize V6 stability fix + V11/V13/V14 unfreeze
  evidence + V18 paradigm-status).
- Output budget: 2500-4000 words.

## Stop conditions

- If 2024-2026 literature offers no upgrade path that beats current cohort
  member by ≥ +0.01 IC at the same param budget, recommend KEEP / FREEZE /
  ARCHIVE per current WM_FINDINGS scoring.
- If one version (e.g. V6) has a clean fix path AND the others have no
  upgrades, say so without padding.
- If V18 paradigm is genuinely closed by 2026-05 evidence, recommend
  archive without revival cycle.
