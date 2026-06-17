# B004 — V2-V5 model upgrade review (active V3/V4 + archived V2/V5)

> **Status:** OPEN  •  **Sent:** 2026-05-02
> **For:** an `@browser`-routed Claude Code session with WebSearch / WebFetch.
> **From:** the trained model state in `src/wm/v{2,3,4,5}/` (V2/V5 archived per
> `backups/BKP_20260429_MODEL_HARMONIZATION/`; V3/V4 active with multiple
> sub-version variants).
> **Tone:** /un — direct, ship-or-concede.
> **Separation:** this prompt is INTENTIONALLY SEPARATE from B002 / B003 to
> avoid blurring V1.x / foundation findings with V2-V5 architecture-specific
> recommendations. Do not cross-cite B002 / B003 conclusions in this response;
> the user wants this as an independent inventory.

## Mission framing

**The V2-V5 cohort is the project's "novel architecture diversity" track**.
V1.x is the SHIP-tier reference; V2-V5 are architecture experiments meant
to either (a) replace V1.x as the new reference if they exceed it, or
(b) provide ensemble diversity for V10 meta-ensemble.

**Current state per CLAUDE.md + WM_FINDINGS_2026_04_29.md**:

| Ver | Architecture | Params | IC h=1 | ShIC h=1 | Status |
|---|---|---|---|---|---|
| V2 | (archived; was transformer hybrid) | — | — | — | ARCHIVED in `backups/BKP_20260429_MODEL_HARMONIZATION/v2/` |
| V3 | WaveNet-Direct (multi-scale dilated causal conv) | 1.9M | unmeasured at f29/f34 | unmeasured | ACTIVE; 4 sub-variants in `src/wm/v3/` (v3, v3_1, v3_2, v3_3); per WM_FINDINGS D1=? VALIDATE-tier |
| V4 | Mamba-3 + RSSM (complex-valued SSD, RoPE) | 3.5M | 0.0477 (ep30) | **0.0164 → 0.0147 (declining)** | ACTIVE; 4 sub-variants; per WM_FINDINGS ShIC declining mid-training (L2 training-stability failure) |
| V5 | (archived; was SSM variant) | — | — | — | ARCHIVED in `backups/BKP_20260429_MODEL_HARMONIZATION/v5/` |

**The user's framing**: "V2-V5 models for upgrades" — investigate whether
2024-2026 SOTA suggests upgrades to the active V3/V4 cohort AND whether
the archived V2/V5 are worth reviving with new techniques.

## Tasks (priority-ordered)

### Task 1 — V3 WaveNet upgrades (2024-2026 literature)

V3 is dilated-causal-convolution based (WaveNet, van den Oord 2016). Its
sub-variants (v3, v3_1, v3_2, v3_3) explore minor architectural shifts but
the core dilated-conv pattern is unchanged.

Search:
1. **TCN / WaveNet evolutions for time-series forecasting 2024-2026** — has
   anyone improved on the multi-scale dilated causal-conv pattern?
2. **PatchTST / WaveletNet / FEDformer hybrids** — should V3 be combined
   with patch-based or frequency-domain front-end?
3. **Causal convolution variants** — sparse causal conv, causal Mamba conv,
   gated linear unit causal conv. Any with measured IC lift on financial
   small-cap predictors?

For each: paper, claim, applicability to 1.9M-param V3, expected IC delta
if applied.

### Task 2 — V4 Mamba-3 upgrades (2024-2026 literature)

V4 is Mamba-3 (latest SSD with complex-valued state) + RSSM bottleneck.
Per WM_FINDINGS: ShIC declines mid-training (training-stability problem).

Search:
1. **Mamba-2 / Mamba-3 / Mamba-4 stability fixes** — has subsequent work
   identified or fixed the ShIC-decline pattern at mid-training?
2. **RSSM categorical bottleneck variants** — have alternatives to Dreamer's
   24×24 categorical (e.g. continuous VQ, Gumbel quantization, FSQ
   finite-scalar-quant) been shown superior?
3. **Mamba + RSSM specifically** — what 2024-2026 work combines selective
   SSM with categorical world-model latent? Has anyone reported the same
   ShIC-decline failure mode?

For each: paper, claim, fix applicability to V4, expected ShIC stabilization.

### Task 3 — Should V2 and V5 be revived?

V2 and V5 are archived. We need to decide between:
- **Stay archived**: V1.x + V3 + V4 cohort is sufficient for ensemble diversity.
- **Revive with 2025-2026 techniques**: rebuild V2 (was transformer hybrid)
  or V5 (was SSM variant) with modern training stack.

Investigate:
1. What was the most-promising 2024-2026 architecture that **isn't already
   covered** by V1.x (Transformer+RSSM), V3 (WaveNet), V4 (Mamba), V8
   (Neural ODE), V9 (GRU+MoE)?
2. Specifically: **xLSTM, RetNet, RWKV-7, S4D updates, Liquid neural
   networks, Hyena**. Which of these has a 2025-2026 measured benefit on
   financial / time-series data?
3. Is V5 (SSM-class) made redundant by V4 (Mamba-3) or distinct enough to
   re-enable?

### Task 4 — Cross-version invariant compliance check (V3/V4)

CLAUDE.md prescribes a "Cross-Version Training Invariants" table (e.g.
DIRECT_RETURN_WEIGHT=3.0, BIN_MIN/MAX=-1/+1, NUM_BINS=255, ACTIVE_HORIZONS,
TWOHOT_FOCAL_GAMMA=0.0, WM_BATCH_SIZE=32, WM_STEPS_PER_EPOCH=2000). Per
WM_FINDINGS D7 = 5 for V3 and V4 (post-2026-03-17 audit clean).

Has 2024-2026 work surfaced any NEW invariants we should be enforcing on
V3/V4 specifically? E.g. learning-rate schedules, warmup patterns,
gradient-clipping norms specific to causal-conv or SSM architectures?

### Task 5 — Ensemble diversity diagnostic

V10 meta-ensemble's value depends on V1.x, V3, V4 producing **uncorrelated
errors**. Per WM_FINDINGS CC1 (pairwise V1.x ρ unmeasured); for V10 the
question extends to V1.x ↔ V3 ↔ V4 ρ.

Investigate:
1. What is published evidence that WaveNet vs Transformer vs Mamba produce
   uncorrelated errors on **financial time-series**?
2. If they're 0.95+ correlated, V10 ensemble = waste (B003 verdict on V1.x
   siblings). What 2024-2026 work measures inter-architecture correlation
   on identical data?

For each: paper, ρ measurement, applicability.

## Output format

Return one document with these sections:

1. **Executive verdict** (≤ 200 words). Action plan per active version (V3,
   V4) + decision on V2/V5 revival.
2. **V3 WaveNet upgrade inventory** (Task 1 deliverable).
3. **V4 Mamba-3 upgrade inventory + ShIC-stabilization fix** (Task 2).
4. **V2/V5 revival decision** (Task 3) — REVIVE-as-X / STAY-ARCHIVED with rationale.
5. **Cross-version invariant additions for V3/V4** (Task 4).
6. **Ensemble-diversity expectation** (Task 5) — predicted V1.x↔V3↔V4 ρ.
7. **Top 5 next retrains** for V3/V4 in priority order.

## Confidence tagging (mandatory per memory/feedback_search_reliability_protocol.md)

Every load-bearing numerical claim (IC, ShIC, Sharpe, %-improvement, dates,
param counts, money) MUST be tagged inline:

- `[VERIFIED]` — raw source fetched (arxiv abstract page, GitHub raw README,
  HuggingFace `/api/models/<id>` JSON, paper HTML/PDF body — NOT a
  summarized WebFetch response).
- `[REPORTED]` — sourced from a WebSearch snippet OR a summarized WebFetch;
  not yet re-checked against raw text.
- `[INFERRED]` — derived/computed/extrapolated; not directly stated.

End the response with a **Reliability ledger** counting VERIFIED vs
REPORTED vs INFERRED across all numerical claims. If REPORTED > 0 on any
decision-gating number (one that a V3/V4 retrain depends on), surface those
in a `## Caveats` section.

Never let a recommendation sound more confident than its verification
level.

## Operational constraints

- Hardware: 1× RTX 4060 (8.59 GB VRAM), i9 20 cores, 32 GB RAM.
- V3 retrain budget: ~2-3 GPU-h per variant.
- V4 retrain budget: ~5-7 GPU-h per variant (Mamba-3 SSD has 4× compute factor).
- ShIC > IC × 0.5 must hold; do NOT recommend anything that lifts IC while
  crashing ShIC ratio (V4 already on edge).
- No emojis in any output that touches Python files (Windows cp1252).
- Reference materials in this repo:
  - `CLAUDE.md` (cross-version invariants)
  - `docs/WM_FINDINGS_2026_04_29.md` (V3/V4 current state + scoring)
  - `memory/fix_logs/v3_0.md`, `memory/fix_logs/v4_0.md`
  - `src/wm/v{3,4}/v{N}_training/` (multiple sub-variants)

## Time / cost budget

- WebSearch calls: 8-12.
- WebFetch calls: 3-5 (pick papers most relevant to a 2-4M-param time-series
  predictor — NOT to LLMs).
- Output budget: 2000-3500 words.

## Stop conditions

- If 2024-2026 literature offers no measurable upgrade for V3 / V4 / V2-revive
  / V5-revive that beats their current architecture by ≥ +0.01 IC at the
  same param budget, drop the polite framing and recommend KEEP V3 + V4
  AS-IS / V2 + V5 STAY-ARCHIVED.
- If one architecture (e.g. xLSTM 2024) clearly dominates everything in
  the V2-V5 cohort, say so and recommend it as the only upgrade worth the
  retrain budget.
