# Sonnet Integration Safety Protocol

> Per user directive 2026-05-13: "Sonnet models sometimes produce sub-par results,
> so be careful with their integration."

## The principle

Sonnet sub-agents are a context-budget force multiplier for BREADTH (parallel
file scanning, large-surface inventory, literature sweep). They are NOT a
substitute for Opus judgment on:

- Numeric correctness
- Architectural decisions
- Code changes
- Bug verdicts
- "Done" claims

**Rule**: every Sonnet output is treated as a HYPOTHESIS that Opus must verify.
Verification fails → discard the finding, re-do directly.

## When Sonnet is appropriate

| Task type | Sonnet OK? | Why |
|---|---|---|
| Parallel scan of 5+ files for keyword | YES | breadth, no judgment needed |
| Multi-source web research (5+ URLs) | YES | breadth + cost-efficient |
| Inventory all docs/memos matching a pattern | YES | enumeration |
| "Read these 10 files and report headlines" | YES | summarization where Opus must verify each cited number |
| Architecture-decision recommendation | NO | judgment-heavy; Opus does this |
| Code change | NO | Sonnet is read-only by policy |
| Bug fix verdict ("this is the bug") | NO | Opus must reproduce |
| "Yes, V7 wires correctly" | NO | Opus runs the validator |
| Single-file or <5-file lookup | NO | direct Grep/Glob/Read is faster |

## Verification protocol (mandatory after every Sonnet invocation)

For each claim in a Sonnet scout return:

1. **Numerical claims**: open the source file/memo; verify the number is there
   and means what Sonnet says it means
2. **File-path claims**: `ls` or Glob to confirm the path exists
3. **Pattern claims** ("X exists in N files"): `grep` to confirm the count
4. **Sharpe / ROI / hit-rate claims**: cross-check against the cited memo
5. **"Done / shipped" verdicts**: run the actual test (BlendComposer / py_compile /
   smoke script)
6. **Aggregations / summaries**: re-derive from raw data when stakes are high

If any verification fails → discard the entire return + re-do directly with
Opus. Don't try to "patch" a Sonnet hallucination.

## Observed Sonnet failure modes (running list)

Document Sonnet failures here as they happen, so future Opus instances know
the typical errors to look for:

| Date | Failure mode | Source memo | What Opus had to fix |
|---|---|---|---|
| 2026-05-13 | Sonnet scout reported `xsec K=5+5 FULL` as "model.pkl exists, NOT wired into paper_trader_v2". Reality: it IS wired — the production stack uses `xgb_ndcg_v1_u4h_v0_base_48feat` which is the K=5 ranker. `xsec_ranker_v1.pkl` is an older standalone file. | `meta` synthesis 2026-05-13 (this session) | Re-read production_blends.yaml directly; corrected synthesis claim |
| 2026-05-13 | Sonnet scout summary said "frontier subtree ORPHANED in production_blends.yaml". Reality: `V6_FRONTIER_v2026_05` was wired 2026-05-12 (the day before). | `meta_layer_protocol_2026_05_11.md` | Opus re-checked git log; updated synthesis. |

## When in doubt — skip Sonnet

For tasks under the **4-file / 4-URL** threshold, just use Grep/Glob/Read/WebFetch
directly. Faster than spawning a Sonnet agent, and no verification overhead.

## Integration with research_delegation_protocol.md

The 4-agent protocol in `memory/research_delegation_protocol.md` uses Sonnet
for scouts AND for the oracle/auditor stages. The Opus "Synthesis + Decisions
+ Edits" stage is where verification happens. Treat the oracle/auditor Sonnet
stages as HYPOTHESIS GENERATORS, not as authoritative findings. Opus is the
only decision-maker.

## Escalation

If Sonnet scouts repeatedly produce sub-par results in a session — STOP using
them for the rest of that session. Add the failure to the table above. Notify
user that Sonnet integration is being paused for this session due to quality.
