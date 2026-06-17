# Slash Router — curated skill set (rebuilt 2026-05-28)

When the user types `/<token>`, resolve it to one of the **13 curated skills**.
`.claude/skills/` holds **14 directories** — the 13 invokable skills (each has a `SKILL.md`)
plus the `_common/` shared-resources dir (not invokable).
Matcher primitive: `python scripts/fuzzy_slash_match.py <token>`.

## The 14 skills

| Skill | Role |
|---|---|
| `normal` | Vanilla direct mode — no persona, no overhead. Default fallback. |
| `apex` | Full-power / parallel-worker coordinator — hard, multi-domain, plateau-breaking work. |
| `pipeline` | Data / pipeline expert. |
| `architect` | World-model architecture expert. |
| `trainer` | Training / loss / anti-fragile expert. |
| `trader` | Trading / risk / sleeve-lifecycle / live-ops expert. |
| `validator` | Claim-evidence validator (single-number sanity checks). |
| `research` | Literature / SOTA / experiment-design expert. |
| `audit` | Adversarial review — RED-team + deep multi-file gap analysis. |
| `decide` | Reasoning / decision — oracle (first-principles) + dialectic (BULL/BEAR/NULL) + meta (decomposition). |
| `orc` | Orchestrator — DEFAULT autonomous operating model; launches/oversees the 3 loops (problem-solver + meta-agent + 3-hourly self-evolution). |
| `discover` | Strategy-discovery expert — find a tradeable per-asset edge from scratch (gap-diagnosis → conditioner search → robustness battery). Use BEFORE `trader`. |
| `narrate` | Market-narration expert — describe the WHAT of (asset, period, chart-type); descriptive, per-setup, entry-framing only. Feeds `discover`. |
| `quant` | Quant / math / statistics expert — inference design, multiple-comparisons correction, distributional & time-series econometrics, the adversarial statistical referee on any numeric edge. Complements `validator` (mechanical gates). |

## Former-name routing (archived skills → curated)

| Old `/command` | Now routes to |
|---|---|
| `/auditor`, `/red-team`, `/deep`, `/analyze` | `audit` |
| `/oracle`, `/consult`, `/dialectic`, `/dialect`, `/debate`, `/meta` | `decide` |
| `/orchestrator`, `/orchestrate`, `/autonomous`, `/auto` | `orc` (skill dir renamed `orchestrator` → `orc`) |
| `/nar`, `/read` | `narrate` |
| `/find`, `/mine`, `/edge` | `discover` |
| `/data` | `pipeline` |
| `/design` | `architect` |
| `/train` | `trainer` |
| `/trade` | `trader` |
| `/validate`, `/check` | `validator` |
| `/stats`, `/math`, `/statistics`, `/significance` | `quant` |
| `/sota` | `research` |
| `/max`, `/un`, `/unconstrained`, `/maxx`, `/team`, `/swarm`, `/three-team` | `apex` |

## Resolution thresholds (fuzzy match against canonical names + aliases)
- **≥ 0.90** — auto-invoke with a one-line correction note.
- **0.70–0.90** — ask once, default-yes.
- **0.55–0.70** — explicit ask.
- **< 0.55** — treat as plain text (no skill).

## Fallback
Any `/command` that cannot be classified to a curated skill → treat as `/normal` (Opus direct).
