# B002 — Frontier LLM lab approaches: which ones overlay onto our crypto WM?

> **Status:** OPEN  •  **Sent:** 2026-05-02  •  **Depends on:** B001
> (Kronos verdict + 3-prong stack adoption)
> **For:** an `@browser`-routed Claude Code session with WebSearch / WebFetch.
> **From:** the build state in `src/frontier_ml/` after B001 closed.
> **Tone:** /un — direct, evidence-first, ship-or-concede-clear.

## Mission framing

We are now committed to building **the project's equivalent of Opus
4.7 / GPT-5 / Gemini 2.5 / Llama 4 / DeepSeek-V3** — but for a single
domain (crypto trading) under one hardware constraint (1× RTX 4060 Laptop,
8.59 GB VRAM, i9 20 cores, 32 GB RAM) and one type of inference target
(time-series world model + price/return forecasting).

The user's framing (verbatim): *"have browser look at a broader literature
review step by step and investigate Gemini, Claude, DeepSeek, ChatGPT, etc.
approaches we can overlay onto the new frontier ML (expand scope to close
further gaps)."*

We have already adopted from B001:
- Causal multi-horizon next-token + lead-lag InfoNCE (foundation Prong 1)
- Distillation from V1.x ensemble + foundation (Prong 2)
- Multi-modal cross-attention adapter with lagged side channels (Prong 3)
- Adaptive log-spaced TwoHot bins (browser R3)

What we **may not** have adopted but COULD if it transfers:
- Mixture-of-experts (DeepSeek-V3, Llama 4 Maverick, GPT-4 Mixture)
- Multi-Token Prediction (DeepSeek-V3, Meta MTP papers)
- Speculative decoding / draft-target inference (Anthropic Sonnet, others)
- Constitutional AI / RLHF / DPO / KTO post-training (all four labs)
- Long-context attention tricks (Anthropic 1M, Gemini 2M, Mamba-Hybrid)
- Native-multimodal training (Gemini 1.5+, GPT-4o, Claude 3.7+)
- Tool-use post-training (Claude / GPT / Gemini agents)
- Compute-optimal Chinchilla scaling (DeepMind / DeepSeek)
- Test-time compute scaling (o1 / o3 / Gemini-thinking / DeepSeek-R1)
- Inference-time chain-of-thought / scratchpad (all reasoners)
- Self-play / synthetic data generation (DeepSeek-R1 GRPO)
- Speculative routing / sparse activation (Mixtral, DeepSeek)
- Layer-wise normalizers / GLU variants / RoPE / NTK-aware scaling

The job: figure out **which of these techniques transfer to a 31.7M-param
domain-specialist time-series WM on 4060/8GB**, and which are
LLM-specific and don't.

## Tasks (priority-ordered)

### Task 1 — Decompose each frontier lab's known training stack

For each of: **Anthropic Claude (3.7 / 4 / 4.6 / 4.7), OpenAI GPT-4o /
o1 / GPT-5 / GPT-OSS, Google Gemini 1.5 / 2.0 / 2.5, DeepSeek V2 / V3 /
R1, Meta Llama 3.1 / Llama 4 / V-JEPA**, list:

1. Their known *architectural* innovations (released in tech reports or
   open-source releases through 2026-04).
2. Their known *training* innovations (data mixing, curriculum, RLHF
   variants, post-training tricks).
3. Their known *inference* innovations (sampling, decoding, test-time scaling).

For each item, note: **(name) | (what it does) | (transferable to a
time-series WM y/n/maybe) | (compute cost on 4060) | (specific application
to our crypto pipeline)**.

### Task 2 — Map each transferable technique to one of our existing modules

We have these places to plug into:
- `foundation/backbone.py` — Mamba-3 + cross-asset attention
- `foundation/objectives.py` — causal multi-horizon TwoHot + lead-lag InfoNCE
- `foundation/pretrain.py` — AMP loop with stop-grad contrastive
- `distillation/student.py` — 4.3M / 10.6M Mamba student
- `distillation/distill_loss.py` — α·KL + β·L1 + γ·L2
- `multimodal/adapter.py` — cross-attention on frozen foundation
- `multimodal/channels.py` — funding/OI/ETF/macro lagged channels

For each transferable technique from Task 1, write:
- WHICH file it modifies / replaces
- WHAT the modification looks like (1-3 sentence sketch; no code)
- EXPECTED IC / ShIC delta (with citation if available, else honest "speculative")
- COMPUTE COST in GPU-hours

### Task 3 — Specific high-leverage candidates to investigate deeply

1. **Multi-Token Prediction (MTP)**: DeepSeek-V3 reports +0.5-2% loss
   improvement. Our current `pretrain.py` predicts h={1,4,16,64}
   independently. Would joint MTP-style prediction (each head predicts
   the NEXT k bars in sequence, sharing intermediate representations)
   lift IC?
2. **Speculative routing / MoE**: V11 (WaveNet+MoE) and V13 (TFT) already
   touch this. Has the field converged on a clean MoE recipe for
   time-series? Would it help our 31.7M Mamba?
3. **Test-time compute scaling**: o1 / R1 / Gemini-thinking spend more
   compute per inference for better answers. Could we run 30 Monte Carlo
   forward passes per bar, keep the median, expect +ShIC? (Some early
   evidence: Kronos uses sample_count=30 by default for distributional
   forecasts.)
4. **Synthetic data / self-distillation**: DeepSeek-R1 generates synthetic
   reasoning traces; AlphaGo self-play; AlphaProof. Could we generate
   synthetic future paths via diffusion (V14-style) and have the
   foundation learn to MATCH the diffusion's marginal? The "synthetic
   pretrain corpus" play.
5. **Native multi-modal pretraining**: Gemini and Claude train all
   modalities jointly. Our Prong 3 trains a frozen-foundation adapter.
   Would it be better to PRETRAIN the foundation jointly with the side
   channels (funding/OI/ETF) from step 0, instead of as a downstream
   adapter?
6. **Mamba-Hybrid (Mamba + attention layers)**: per Jamba / Bamba papers,
   1:7 ratio of attention to Mamba layers gives best of both. Our
   31.7M is pure Mamba SSD. Would replacing 1 of 8 Mamba layers with
   a transformer layer lift IC at fixed param count?

### Task 4 — Eval / scaling laws translated to our regime

The classic scaling result (Hoffmann et al. 2022 Chinchilla) prescribes
~20 tokens per param for compute-optimal training. We have:
- 31.7M params
- ~110M chimera_legacy bars (u100)
- ~50M effective training samples after 50/20/20/10 split

That's ~1.6 train-tokens-per-param, well below Chinchilla optimum.
Question: in 2025-2026, has the scaling-law field updated for *time-series*
foundation models specifically? Are smaller models more data-efficient
on this domain?

Find: any 2024-2026 paper that derives time-series scaling laws (token-per-
param ratio, optimal model size for fixed corpus). Apply the result to our
chimera_legacy corpus and recommend: should we train **smaller** than 31.7M?

### Task 5 — Project-specific question: agent-tier crypto in 2026-05

The B001 response (CryptoBench evidence) said LLM-trader paradigm is 1-2
years from frontier-tier on crypto. **Is that still true 1 month later?**
Specifically:
- Has any group released an LLM-trader system since 2026-04 with measured
  Sharpe ≥ 2.0 on real-money crypto?
- Has Grok-4's CryptoBench accuracy (44% per B001) been beaten?
- Has anyone released a crypto-specialist reasoning model (R1-style) for
  market prediction?

If yes to any: revisit our R2 (no agent prong) decision.

## Output format

Return one document with these sections:

1. **Executive verdict** (≤ 200 words). Top three techniques we should
   adopt + estimated compound IC lift if all three ship.
2. **Lab-by-lab decomposition** — Anthropic, OpenAI, Google, DeepSeek,
   Meta — bulleted technique inventory with transfer column.
3. **Module-by-module retrofit map** — table mapping each transferable
   technique to which `frontier_ml/` file it modifies, expected IC delta,
   GPU-hours cost.
4. **Top 5 Specific Candidates** — deep dive on the high-leverage 5 from
   Task 3. Honest assessment of which would and wouldn't work on our
   domain + scale.
5. **Scaling law re-estimate** — Chinchilla-equivalent for time-series in
   2025-2026; recommend optimal model size for our 110M-bar corpus.
6. **Agent-tier 2026-05 update** — any change to the R2 verdict.
7. **Top 5 next experiments** in priority order (max 5).

## Operational constraints

- Hardware: 1× RTX 4060 (8.59 GB VRAM), i9 20 cores, 32 GB RAM.
- Project-tier goal: Headline IC > 0.10 / ShIC > 0.05.
- Anti-fragile invariants non-negotiable: ShIC > IC × 0.5, walk-forward CV
  with purge gaps, DSR > 0 on OOS+UNSEEN.
- No emojis in any output that touches Python files (Windows cp1252).
- Specialist mindsets via Skill tool, not parallel Agent subagents
  (rate-limit constraint).

## Time / cost budget

- WebSearch calls: 8-15. Prioritize quality over volume.
- WebFetch calls: 3-6. Pick papers / repos that the closest analogs cite.
- Output budget: 2000-4000 words.

## Confidence tagging (mandatory per memory/feedback_search_reliability_protocol.md)

Every load-bearing numerical claim (param count, IC, Sharpe, %-improvement,
dates, money figures) MUST be tagged inline:

- `[VERIFIED]` — raw source fetched (arxiv abstract page, GitHub README via
  `raw.githubusercontent.com`, HuggingFace `/api/models/<id>` JSON, paper
  HTML/PDF body — NOT a summarized WebFetch response).
- `[REPORTED]` — sourced from a WebSearch snippet OR a summarized WebFetch;
  not yet re-checked against raw text.
- `[INFERRED]` — derived/computed/extrapolated; not directly stated.

End the response with a **Reliability ledger**: count of VERIFIED vs
REPORTED vs INFERRED across all numerical claims. If REPORTED > 0 on any
decision-gating number (i.e. one that an action item depends on), surface
that explicitly in a `## Caveats` section so the reader knows precisely
which numbers must be re-checked before acting.

Never let a recommendation sound more confident than its underlying
verification level: REPORTED foundations → REPORTED-grade recommendations,
not "this is the only sensible move."

## Stop conditions

- If ALL of the proposed techniques are LLM-specific and none transfer to
  31.7M time-series, say so directly. We accept the answer.
- If ONE technique dominates (e.g. "MTP is the single biggest free lunch"),
  say so without polite hedging and recommend it as R-the-only-thing.
- If the scaling-law analysis says we should *shrink* the foundation model
  rather than grow it, say so. We are not married to 31.7M.
