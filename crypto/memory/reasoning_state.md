# Reasoning State

> Episodic memory: HOW to think about this project. Read FIRST on session start.
> See MEMORY.md for FACTS (what is true). This file is for PATTERNS (how to think).

## Mental Models

### The Cost-Constrained Weak-Signal Problem

This system has IC ≈ 0.03 ungated and 0.24% round-trip cost. That's a 12x
signal-to-cost ratio — too thin for naive per-bar trading. The strategy is:

1. **Predict every bar** (model has signal to extract)
2. **Trade rarely on accumulated/filtered evidence** (cost-aware execution)

Two mechanisms achieve this:
- **HIGH_MAG conformal gate** (Pattern L, 2026-04-14): top-10% bars by |pred|
  give 2.17-3.09x IC lift cross-asset. Gated IC ≈ 0.04 = 17x signal-to-cost.
- **SPRT temporal accumulation**: enter only when evidence crosses Wald boundary.

### When Empirics Contradict Literature

Pattern L is the canonical example: Gibbs & Candes (2021) prescribes
LOW_WIDTH gating for conformal abstention; we tested empirically and found
0.14-0.72x lift (FAIL). HIGH_MAG (firm-call) gives 2-3x lift.

**Heuristic for ECL cases**:
1. Document the empirical finding with exact numbers + conditions
2. State the literature prediction + cite source
3. Propose 3 mechanistic explanations (e.g., "our models are correlated, paper assumed diversity")
4. Design the discriminating experiment
5. Update Pattern table in fix_logs/INDEX.md as the canonical example

### The Layering Defense Mechanism

Skills are layered to prevent capability erosion:
- **Layer 0**: Pure Claude — never restrict
- **Layer 1**: Skill-specific hard rules (e.g., "trader uses CostModel")
- **Layer 2**: Universal amplifier (`.claude/skills/LAYER2_UNCONSTRAINED.md`)
- **Layer 3**: Cross-skill orchestration (`memory/research_delegation_protocol.md`)

**Test for layer placement**: if a rule were accidentally placed one layer
HIGHER, would it prevent the model from doing something legitimate? If yes,
add a "Scope: Layer N only" annotation.

### The Completion Mandate (Layer 2)

When a user invokes a skill, they want END-TO-END completion at SOTA quality.
NOT a partial delivery with hidden pending items. NOT "I'll defer this to
next session". The user owns context (compaction is a tool); Opus owns
completeness.

## Decision Patterns

### Architecture Decisions
1. Read existing code (don't reason from memory)
2. Quantify: param count, VRAM, training time, expected impact
3. Adversarial review: what could go wrong?
4. Build it (don't just plan it). Convert insights into code.

### Numerical/Crash Bugs
1. **PROBE FIRST.** 4s synthetic OR 60s real-data probe BEFORE proposing fix.
2. Identify the EMPIRICALLY CONFIRMED source (not "the math says...")
3. Apply minimal fix to confirmed source only
4. Re-run probe to confirm zero NaN
5. Never stack speculative fixes

### Cross-Version Changes
1. Count affected files FIRST (typically 9-40 files)
2. sed/grep for mechanical propagation, Edit for logic changes
3. py_compile each modified file
4. Cross-grep to verify nothing missed

### Strategy Validation
1. NEVER trust single-OOS-window Sharpe
2. Walk-forward across 5+ windows minimum
3. Apply CSCV/PBO when sweeping >20 configs (multiple-testing bias)
4. Apply Deflated Sharpe before "this strategy is profitable" claim

## Anti-Patterns (DO NOT REPEAT)

### Reasoning from stale knowledge
- "Gumbel-Softmax overflows fp16" — WRONG. Empirically fp16-safe to mag 20+.
- "CuDNN disable is ~20% cost" — WRONG. Was 7x slower.
- "Conformal LOW_WIDTH gives 3x lift" — WRONG for our correlated models.
- **Rule**: WebSearch any factual claim about technique behavior FIRST.

### Surface-level diagnosis
- "NaN at line X → add fp32 cast" without tracing WHY.
- **Rule**: Trace the full path before fixing.

### Fix-creates-new-bug
- Reorder backward → inplace error.
- Disable CuDNN → 7x slowdown.
- **Rule**: Check ALL constraints simultaneously, not one at a time.

### Stale code execution
- Fix committed but process uses old import.
- **Rule**: Delete __pycache__, verify file timestamp before claiming "fixed".

### Signal-destroying stability fix
- RMSNorm on h_seq killed ShIC by 57%.
- **Rule**: Compare against archived working baseline before stabilizing.

### Not using available tools
- WebSearch / WebFetch available but never used for PyTorch issues.
- Sonnet scouts available but Opus reads 20 files directly (context burn).
- **Rule**: Search FIRST, reason SECOND. Delegate reads to Sonnet.

### Trusting metric without verification
- "V6 gate fail" interpreted as bug. Was actually old-process-cached-code.
- **Rule**: Check WHEN the data was generated relative to the fix.

### Premature deferral
- "I'll check this in next session" with context remaining.
- "Let me defer to documentation."
- **Rule**: Completion Mandate. End-to-end. User owns context.

## Project State Quick Reference

(See MEMORY.md for live state. This is the THINKING, not the FACTS.)

- 12 model versions, 4 with f34 PASS checkpoints (V1.0/V1.1/V1.4/V1.6)
- HIGH_MAG conformal gate cross-verified on 4 assets (BTC/ETH/SOL/DOGE)
- Strategy layer SOTA upgrade complete (5 modules + RiskController + EWA)
- Walk-forward + CSCV/PBO + Deflated Sharpe runnable end-to-end
- Cost model unified across 6 analysis scripts

## When You're Stuck

1. Re-read this file. The answer is often in an anti-pattern you're repeating.
2. WebSearch the specific error message.
3. Read the actual code (don't reason from memory).
4. Spawn a Sonnet scout for a second opinion (read-only).
5. If 1-4 don't help, ask the user with concrete options — don't generic-defer.

## Self-Updating

When a new lesson is learned that doesn't fit existing patterns, append it
here. This file is intentionally short — keep entries dense and actionable.
Don't write what's already in MEMORY.md (facts), CLAUDE.md (invariants),
or fix_logs/ (specific bugs).
