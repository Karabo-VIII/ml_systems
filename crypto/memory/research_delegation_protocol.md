# Research Delegation Protocol

> Layer 3 protocol — orchestrates multi-agent research/audit pipelines.
> Referenced by Layer 2 (`.claude/skills/LAYER2_UNCONSTRAINED.md`).

## Purpose

When a task requires breadth (read 10+ files, sweep 2024-2026 literature, audit
across all 9 model versions), Opus must NOT eat its own context budget on raw
ingestion. Sonnet sub-agents are the force multiplier: they read; Opus reasons.

This protocol formalizes the delegation pattern.

## The 4-Agent Protocol (for high-blast-radius decisions)

```
       ┌─────────────┐    ┌─────────────┐
       │  Sonnet     │    │  Sonnet     │
       │  Scout A    │    │  Scout B    │
       │ (recon)     │    │ (research)  │
       └──────┬──────┘    └──────┬──────┘
              │                  │
              └────────┬─────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  expert-oracle  │  ← optional, for first-principles pass
              │  (Sonnet model) │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ expert-auditor  │  ← optional, for adversarial pass
              │  (Sonnet model) │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Opus (you)     │
              │  Synthesis      │
              │  + Decisions    │
              │  + Edits        │
              └─────────────────┘
```

**Rules**:
- All sub-agents use `model: "sonnet"` by default. Opus sub-agents cost 50-100K
  context tokens on return — only use for genuine architectural ambiguity.
- MAX 2 Sonnet scouts in parallel.
- Scouts and oracle/auditor agents are READ-ONLY. They report; Opus decides.
- Scouts produce structured reports (see Scout Return Format below).

## When to invoke this protocol

| Task type | Use protocol? | Why |
|-----------|---------------|-----|
| Architecture decision touching 5+ files | YES | Multiple lenses needed |
| Literature review for novel technique | YES | Breadth via parallel search |
| Cross-version audit (V1-V14) | YES | Sonnet scouts read, Opus synthesizes |
| Single-bug fix | NO | Direct Opus is faster |
| Settings change in one file | NO | Direct Opus |
| Routine commit/push | NO | Direct Opus |

## Scout Return Format (mandatory)

When spawning a Sonnet scout, REQUIRE this format in your prompt:

```
For each finding, return:
  FILE: <path>:<line>
  FINDING: <one sentence>
  SEVERITY: CRITICAL / HIGH / MEDIUM / LOW
  EVIDENCE: <exact code snippet or doc quote>
  PROPOSED FIX: <one sentence, optional>

Cap response at <N> words. No fluff. No restating the question.
```

This lets Opus triage a 1500-word report in ~30 seconds without re-reading source.

## Scout Domain Decomposition Patterns

For codebase audits, decompose by LAYER not by file count:

- **Pipeline scout**: `src/pipeline/`, `data/processed/` schema, `make_dataset.py` (v51 primary) + `make_dataset_legacy.py` (v50)
- **Model/training scout**: `src/wm/v*/`, `memory/MEMORY.md`, `memory/fix_logs/`
- **Strategy/validation scout**: `src/strategy/`, `src/analysis/`, validation gates
- **Trading/execution scout**: `paper_trader.py`, OMS, live data feeds (when they exist)
- **Doc/protocol scout**: `CLAUDE.md`, `.claude/skills/`, `memory/`

For literature reviews, decompose by topic:
- **SOTA techniques scout**: 2024-2026 papers in the relevant domain
- **Implementation scout**: existing open-source implementations, repo quality
- **Critique scout**: known limitations, failure modes, contradictory results

## Synthesis Discipline (Opus's role)

After scouts return, Opus must:

1. **Identify convergence**: where do scouts agree? High-confidence findings.
2. **Identify divergence**: where do they disagree? Investigate yourself.
3. **Identify gaps**: what did neither scout cover? Your job to fill.
4. **Score severity**: CRITICAL/HIGH/MEDIUM/LOW — own the prioritization.
5. **Decide layer placement**: for any change, which layer (0/1/2/3) does it belong in? Justify.
6. **Execute**: only Opus edits files. Sub-agents are read-only by protocol.

## Anti-patterns

❌ **Spawning agents you don't need**: Direct tools (Grep/Glob/Read) are faster
for <5 files. Don't spawn an agent to read one file.

❌ **Spawning agents in series when independent**: Use a single message with
multiple Agent calls for parallel work. Serial is rate-limited and slow.

❌ **Asking Sonnet to make decisions**: Sonnet reports findings; Opus decides
which to act on. Decisions are Opus's responsibility.

❌ **Trusting a scout report without spot-checking**: For CRITICAL findings,
verify the scout's evidence yourself before acting.

❌ **Spawning Opus sub-agents by default**: 50-100K context cost on return.
Only use for genuine architectural ambiguity (e.g., apex Multi Mode debate).

## Examples from project history

**Strategy gap recon (2026-04-14)**: Spawned 3 parallel Sonnet scouts on
disjoint domains (pipeline, models, strategy/validation/trading). Each
returned a structured gap report. Opus synthesized and produced a
prioritized close-list. Result: 14 commits closing all V1-integration gaps.

**Pattern L conformal validation (2026-04-14)**: Used Sonnet scouts for
cross-asset BTC/ETH/SOL/DOGE conformal calibration runs (each ran in parallel
background). Opus aggregated the 4-asset panel and identified that HIGH_MAG
gating gives universal 2.17-3.09x IC lift while LOW_WIDTH (literature default)
fails at 0.14-0.72x.

**V4 Mamba-3 + V6 VIB upgrade (2026-04-12/13)**: Used `expert-oracle` and
`expert-auditor` (sonnet model) for independent architectural reviews.
Found magnitude-explosion bug in V4 SSD path that single-pass review missed.
