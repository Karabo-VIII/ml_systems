# Research Brief — Frontier-tier WM + Agent for Crypto (2026-05-02)

> **For:** a Claude Code session with WebSearch / WebFetch / browser tools
> available, routed via the project's `@browser` tag (see
> [docs/BROWSER_DIRECTIVE.md](../../docs/BROWSER_DIRECTIVE.md)).
> **From:** the project state in `src/frontier_ml/` as of 2026-05-02.
> **Goal:** establish whether our current Mamba-3 + distillation + multi-modal
> stack is genuinely frontier-tier for our use case, or whether SOTA work in
> 2024-2026 has shipped a paradigm we should adopt instead.

## Mission framing (do not skip)

We are building **the project's equivalent of Claude Opus 4.7 / GPT-5 / Gemini
Ultra / Llama 4** — but for a single domain (crypto trading) under a single
hardware constraint (1× RTX 4060 Laptop, 8.59 GB VRAM, i9 20 logical cores,
32 GB RAM). The scope is not "the next paper"; it is **the project-tier
frontier for our use case** — i.e. the best world-model + agent combination
that reliably ships on this hardware against real capital, with
anti-fragility guarantees (ShIC > IC × 0.5).

The user is not asking "are foundation models good?" — they are asking:
**"have we made the right architectural and training decisions, given what the
field now knows in 2026-05?"** Bring back specific evidence, not a textbook
review.

The answer should let us either:
1. **Validate** the current 3-prong design (foundation + distillation + multi-
   modal) and proceed to launch the 35-hour u100 pretrain, OR
2. **Pivot** to a paradigm that 2024-2026 SOTA has demonstrably overtaken our
   bet, with citations.

## What we have already built (read these before searching)

The current stack lives in [`src/frontier_ml/`](.) of this repo:

- `STATUS.md` — current build state, file map, empirical results
- `PLAN.md` — 3-prong design with literature-driven updates applied
- `LITERATURE.md` — 8 holes already poked and closed with citations
- `foundation/` — 31.7M-param Mamba-3 backbone + cross-asset attention,
  pretrained on causal multi-horizon TwoHot + lead-lag InfoNCE contrastive.
  u10 5K-step mini complete: linear-probe IC = +0.052, intrinsic IC = -0.032
  (head undertrained at 1.3% of full compute).
- `distillation/` — scaffold for distilling V1.x ensemble + foundation into
  4.3M / 10.6M deployable student via hybrid α·KL + β·L1 + γ·L2 loss.
- `multimodal/` — scaffold for cross-attention adapter on frozen foundation
  with funding/OI/ETF/macro side channels (explicit lag for hygiene).

Existing trained baselines (V1.x family, all under `src/wm/v*/`):

| Version | Architecture | Params | Best IC | Best ShIC | Status |
|---|---|---|---|---|---|
| V1.0 | Transformer + RSSM | 2.0M | 0.066 | 0.032 | reference baseline |
| V1.1 | + XD anti-memorization | 2.0M | **0.067** | 0.033 | record |
| V1.4 | + FeatureAttention (iTransformer) | 2.0M | 0.068 | 0.031 | record |
| V1.6 | + KL anneal + Gumbel + ATME | 2.0M | 0.062 | 0.033 | active |
| V3 | WaveNet | 1.9M | n/a | n/a | clean variant |
| V4 | Mamba-3 + RSSM | 3.5M | n/a | n/a | architecture validated |
| V6 | Transformer + JEPA discriminator | 3.1M | n/a | n/a | clean variant |
| V11 | WaveNet + MoE + Discriminator | 2.9M | n/a | n/a | combined |
| V12 | Cross-Asset Attention | 0.84M | n/a | n/a | xattn baseline |
| V13 | Temporal Fusion Transformer | 2.2M | n/a | n/a | feature selection |
| V14 | Diffusion return distribution | 2.4M | n/a | n/a | full-distribution |

**Tier ladder (CLAUDE.md mandate):**
```
Filter      IC > 0.015, ShIC > 0.015
Sizer       IC > 0.030, ShIC > 0.020
Trader      IC > 0.050, ShIC > 0.030     [V1.x is here]
Headline    IC > 0.10,  ShIC > 0.05      [PRIMARY TARGET]
Ambitious   IC > 0.13,  ShIC > 0.065
Capacity    IC > 0.20,  ShIC > 0.10      [requires V20 tick-level]
```

## What we have already concluded from prior research

From `memory/ml_upgrades_research_2026_04_22.md` and `LITERATURE.md`:

- **Chronos / TimesFM zero-shot don't beat XGBoost on crypto** (Wu 2024,
  Ozbulut & Ucar 2024). Specifically: TimesFM 200M zero-shot IC = 0.008 on
  BTC 1h vs XGB 0.031. → general-purpose foundation models lose to domain-
  specialized small models on narrow tasks.
- **MASTER (Huang 2024) reports IC 0.041 on crypto** — already below our
  V1.1 record (0.067).
- **GNNs for cross-asset show no lift** when xd_* features are already in
  the input panel (Cheng 2024).
- **FinCon contrastive** underperforms supervised XGB on crypto (IC 0.023 vs
  0.041).
- **DreamerV3 trading needs tick data + 24+ GB GPU** — out of scope.
- **LOB-level transformers** need L2 order book — we have aggTrades only.
- **Multi-task learning** adds ~0.003 IC marginally; we already have
  DIRECT_RETURN_WEIGHT=3.0 Huber.
- **Hawkes branching ratio** is reported +6-12% ShIC (Rambaldi 2024); not yet
  shipped in our backbone but already in our feature set.

What this means: most "obvious" upgrades to foundation-model approaches have
been ruled out empirically. The question is what remains.

## Tasks for the browser session (priority-ordered)

### Task 1 — Frontier WM/foundation work in time-series (2024-Q3 to 2026-Q2)

WebSearch for and synthesize:

1. **Foundation models for financial time-series specifically** that have been
   published or open-sourced **after** TimesFM (Q1 2024) and MOMENT (Q2 2024).
   Look for:
   - Anything from Bloomberg, JPMorgan, Two Sigma, Renaissance research.
   - Anything self-published on arxiv with "crypto" + "foundation" + "2025"/
     "2026" in title or abstract.
   - SSRN papers on sub-daily / tick / dollar-bar foundation pretrain.
   - "FinTSB", "FinChrono", "QuantFM", or anything that calls itself a
     finance-specialized foundation model.
2. **What is the current published IC ceiling on daily/hourly crypto?**
   We know XGB ≈ 0.031, V1.x ≈ 0.067, MASTER ≈ 0.041, TimesFM zero-shot
   ≈ 0.008. What does 2026-published SOTA report?
3. **State-space model upgrades to Mamba** — has Mamba-4 / Mamba-S / Mamba-Reg
   shipped and been benchmarked on time-series? Specifically: any work that
   beats Mamba-3 on long-sequence forecasting at < 50M params?
4. **JEPA variants for time-series** — V-JEPA went multi-modal in 2024;
   has anyone shipped a t-JEPA or trading-JEPA that we should compare against?

For each finding, return: **(paper / repo) | (claim re: IC or equivalent) |
(architecture / params) | (whether it can run on 4060/8GB) | (relevance to us)**.

### Task 2 — Frontier agentic systems for trading (2024-Q3 to 2026-Q2)

This is where we have the LEAST coverage. Our system has a PPO agent
(`src/agent/`) sitting in the backburner; the 3-prong plan does not currently
include an agent. Investigate:

1. **FinMem, FinCon, FinAgent, FinR1** and any 2025-2026 successor — have
   any of these matured into deployable trading agents? What does their
   reported live performance look like (not paper Sharpe, deployed P&L)?
2. **LLM-as-trader systems** — has any group successfully wired Claude /
   GPT / Gemini into a live crypto trading loop with non-trivial alpha?
   Specifically: anyone using tool-calling, reflection, memory systems, or
   self-reflection-style prompting to *outperform a non-LLM XGBoost
   baseline* on crypto 1d / 4h / 1h returns?
3. **Voyager-style skill libraries for trading** — has the "library of
   reusable skills" pattern from Voyager (Wang 2023) been ported to trading?
4. **Reasoning-trace chains for risk decisions** — chain-of-thought,
   tree-of-thought, ReAct, or Let's-Verify-Step-by-Step variants for
   position sizing or exit decisions.
5. **Multi-agent debate for crypto execution** — papers / open systems where
   multiple LLM agents argue over a position, with measurable IC lift.

For each: return **(system) | (architecture: backbone + tools + memory) |
(reported performance) | (open source y/n) | (relevance to a 4060 host)**.

### Task 3 — What "Opus 4.7-class" actually means for our domain

The user phrases the goal as "Opus 4.7-level WM and agents for crypto." That
is aspirational and not directly comparable. Reify it:

1. What are the **defining capabilities** of frontier LLMs (Opus 4.7, GPT-5,
   Gemini 2.5, Llama 4) that have transferred or COULD transfer to a
   time-series / trading WM? Specifically:
   - In-context learning (transferred? we don't use it)
   - Multi-modal grounding (transferred? we have side channels)
   - Tool use (transferred? our agent is PPO)
   - Long-context reasoning (we use seq_len 512 dollar bars)
   - RLHF / Constitutional AI / DPO (none of these apply but does an analog?)
   - Agentic planning (not yet implemented)
2. Which of these has been demonstrated to lift IC on financial data, and
   which has been shown to NOT transfer?
3. **What is the literal ceiling we should target?** "Headline IC > 0.10"
   is our ladder, but is there an externally-known ceiling for daily-bar
   crypto WMs? (We suspect ~0.13 by data dimensionality argument.)

### Task 4 — Hardware-realistic frontier on 4060/8GB

Our hard constraint: 1× RTX 4060 (8.59 GB), i9 20 cores, 32 GB RAM. Find:

1. The **best-published time-series result that ships on consumer GPU** — i.e.
   not "we trained on 8× A100." What is the IC ceiling under our hardware
   budget?
2. Any **gradient-checkpointing / FlashAttention-2 / Mamba-FFT** tricks that
   would let us scale to 100M+ params on 4060 at seq_len 512.
3. Any **distillation tricks** beyond Phuong & Lampert 2019 (KL+L1+L2) that
   2025-2026 work shows further closes the teacher-student gap.

### Task 5 — Project-specific gaps (must-check)

1. We use **TwoHot 255 bins over [-1, 1]** for return prediction. Most h=1
   5-min returns are < ±0.1%, so ~99% of mass concentrates on ~50 bins of
   the 255. Has 2025-2026 work proposed adaptive-bin TwoHot, log-bins, or
   ordinal regression replacements?
2. **Cross-asset same-timestamp positives** — we use lead-lag δ ∈ {0, 1, 3,
   12} bars. Has any subsequent work derived an OPTIMAL lag distribution
   for crypto?
3. **Walk-forward purge gap** — we use 400 bars across V1.x. Is this still
   the right gap given 2026 work on look-ahead diagnosis (e.g. backtest
   overfitting tests with finer purge resolution)?
4. **InfoNCE temperature** — we use 0.1 for contrastive. Any literature on
   crypto-specific contrastive temperature?

## Output format

Return one document with these sections, in this order:

1. **Executive verdict** (max 200 words). One of:
   - "Stack is on-trend; proceed to u100 pretrain"
   - "Stack misses paradigm X; pivot recommended"
   - "Stack is correct but missing component Y"
2. **Top 5 papers / repos / projects** that we should KNOW about and
   probably HAVEN'T cited in `LITERATURE.md` yet. Prioritize by relevance
   to our 4060 hardware + crypto domain + architecture decisions we still
   have headroom to change.
3. **Specific recommendations** (≤ 5 items) that are either:
   - "drop X from the design because Y just published evidence it doesn't work"
   - "add component Z because the field has converged on it"
   - "reconfigure parameter P from A to B per the latest published ablation"
4. **Headline-tier feasibility re-estimate** — given everything found, what
   is the realistic IC ceiling for our 4060 + chimera_legacy + 3-prong stack?
   And what would actually get us above 0.13 (Ambitious tier) if anything
   short of tick-level data can?
5. **Agent track verdict** — should we open a fourth prong for an LLM agent
   layer on top, OR is the field still 2-3 years from frontier-tier crypto
   trading agents being a thing? If the former, what specific architecture?
6. **Concrete experiments to run next**, in priority order (max 5).

## Operational constraints (from CLAUDE.md)

- **Headline target is the bar.** SHIP-tier (current V1.x) is a stepping
  stone, not the finish line.
- **30% of GPU budget allocated to aspirational runs.** This research brief
  is one such allocation; act accordingly.
- **No emojis in any output that touches Python files** (Windows cp1252).
- **Anti-fragile invariants are non-negotiable**: ShIC > IC × 0.5,
  walk-forward CV with purge gaps, DSR > 0 on OOS+UNSEEN.
- **Specialist mindsets via Skill tool**, NOT parallel Agent subagents
  (rate-limit constraint).

## Time / cost budget for the browser session

- **WebSearch calls:** 8-15 (be selective; quality > volume).
- **WebFetch calls:** 3-6 (prioritize papers / repos already cited as
  near-misses elsewhere).
- **Length budget for output:** 2000-4000 words.
- **No code generation in the output** — this is a research synthesis brief,
  not an implementation task. The next session (separate, no @browser tag)
  will do the code work based on this brief's recommendations.

## Stop conditions

If after Task 1 + Task 2 you find that the field has shipped *no* materially
new paradigm since LITERATURE.md was written (2026-05-01), say so directly
and stop early. We have an active build and don't need synthesized novelty
that isn't there.

If you find that one paradigm dominates everything we've considered (e.g.
"adaptive-bin TwoHot is now the norm and we should rewrite our heads"), say
that just as directly. Drop the polite hedging.

The user is operating under `/un` (unconstrained mode); the browser session
should match that tone — direct, evidence-first, ship-or-concede-clear.
