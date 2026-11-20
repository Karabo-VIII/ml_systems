# Pre-Action Adversarial Debate Protocol

> RED TEAM audit is POST-hoc (after commit). This protocol adds PRE-action
> adversarial critique — adversary attacks the plan BEFORE execution. Differs
> from `docs/DOUBLE_AUDIT_PROTOCOL.md` which is post-commit RED TEAM.

## Trigger

Apply this protocol BEFORE any of:

- Diff touching > 5 files in one commit
- Any change to: `src/strategy/`, `src/wm/v*/`, `src/pipeline/`, `config/_invariants.yaml`, `config/production_blends.yaml`
- Anything irreversible: dropping data, force-pushing, deleting production artifacts
- Adding a new sleeve runner / blend / model architecture
- Deploying to live capital (real-money flag flips)
- ML training runs > 30 min on the GPU

## Steps

### 1. State the proposed action (1-3 sentences)
Write down what you're about to do. Include file paths, expected outcome, and
the success criterion that will tell you it worked.

### 2. Adopt the ADVERSARY lens explicitly
The adversary's stance: **every plan is presumed broken until proven otherwise.**
Ask:

- **Silent break vector**: How could this change produce wrong results that pass
  py_compile / smoke / CDAP but break downstream? Cite the specific downstream
  consumer.
- **Hidden constraint**: What invariant in CLAUDE.md, LAYER2_UNCONSTRAINED.md, or
  an existing fix-log might this violate without obvious symptom?
- **Wrong-base-rate**: What's the base rate of changes like this introducing
  silent bugs? (For example: cross-version settings propagation is ~30% buggy on
  first try per fix_logs/INDEX.md pattern D.)
- **Cost-realism**: For trading changes, do the assumed costs match `cost_model.py`?
  Will live execution match paper assumptions?
- **Capacity ceiling**: For trading sizing changes, does the size fit the asset's
  capacity (`pos_cap` in u100.yaml)?
- **Look-ahead vector**: For any prediction-touching code, can future info leak
  into present features?

### 3. Produce an explicit adversary verdict
One of three:

- **GREEN** — adversary finds no attack vector. Proceed.
- **YELLOW** — adversary finds plausible failure mode but mitigation is clear.
  Implement mitigation, then proceed.
- **RED** — adversary finds a fundamental flaw. Re-plan from scratch.

### 4. Document the debate in the work log
Single line in the user-facing report or in the commit message:
> "Pre-action debate: GREEN (no attack vectors found)" OR
> "Pre-action debate: YELLOW (xsec sleeve trim might dilute conviction; mitigation = caveat
>  added + multi-window v3 backfill required pre-promotion)"

### 5. Proceed with execution
After GREEN/YELLOW: execute. Post-action RED TEAM audit (per
`docs/DOUBLE_AUDIT_PROTOCOL.md`) still runs.

## How this differs from existing audits

| Stage | Layer | When | Stance |
|---|---|---|---|
| CDAP `check_invariants.py` | Pre-commit hook | Before each git commit | Mechanical invariant check |
| Pre-action adversarial debate | This protocol | Before EXECUTION (often before commit) | Adversary attacks the plan |
| RED TEAM audit | `docs/DOUBLE_AUDIT_PROTOCOL.md` | After commit | Adversary attacks the diff |
| Live monitor | `src/audit/` | Post-deploy | Empirical regression detection |

Pre-action debate is the **cheapest catch point**. Failing here costs minutes;
failing at live-monitor costs real capital.

## Anti-patterns

- "Adversary found nothing because I'm being adversary on my own plan" → fine,
  but be explicit about checking each item above. Don't skip steps.
- "Adversary found something but I'll fix it later" → fix BEFORE commit.
- "Pre-action debate is too slow for small changes" → triggers above are explicit.
  Single-file lookup doesn't trigger. >5-file diff does.

## Example (V7_FRONTIER wire, 2026-05-13 — done retroactively)

**Action**: Add `V7_FRONTIER_v2026_05_13` blend to production_blends.yaml.
**Adversary findings**:
- YELLOW: 5pp trim from xsec_4h might dilute proven conviction. Mitigation: caveat logged that multi-window v3 backfill required before promotion.
- YELLOW: whale + liq sleeves fire sparsely (z >= 1.5); 10pp could sit in implicit cash. Mitigation: caveat logged; fire-frequency check is part of backfill validation.
- GREEN: BlendComposer validation + smoke-test of each sleeve in isolation.

**Verdict**: YELLOW — proceed with caveats in the YAML itself, gated by
NEW_R9_FRONTIER_FULL deploy status (not GO).

## Escalation

If adversary finds RED → don't proceed. Surface to user with the adversary
findings; ask for re-scope or alternative approach.
