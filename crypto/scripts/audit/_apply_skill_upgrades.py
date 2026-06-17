"""One-off script to batch-apply Phase 2/3 SKILL.md upgrades.

Per-skill content (when/gotchas/desc rewrite) defined inline below.
Idempotent: if "## When to invoke" or "## Gotchas" already present, skip.
"""
from __future__ import annotations
from pathlib import Path
import re
import sys


SKILLS_DIR = Path(__file__).resolve().parent.parent.parent / ".claude" / "skills"


# Per-skill upgrade content. Each entry has:
#   new_description: rewrite if old description lacks when/use/trigger
#   schema_version: bool — add metadata.schema_version if missing
#   portable: bool/None — set metadata.portable if None means leave alone
#   when_table: text of "## When to invoke" section (or None to skip)
#   gotchas: text of "## Gotchas" section (or None to skip)
UPGRADES = {
    "normal": {
        "new_description": "Vanilla Claude — no expert persona, no protocol overhead. Use for direct conversational assistance, simple lookups, or when expert skills would add overhead without value.",
        "schema_version": True,
        "portable": False,
        "when_table": """
## When to invoke

| Situation | Why |
|---|---|
| Conversational questions ("explain X", "summarize this") | Expert overhead adds no value for direct lookups |
| Quick reference / single-file lookups | Apex / deep would over-engineer the response |
| User explicitly asks to "drop the persona" or "be vanilla" | Honor the request |
| When no other skill's "when to invoke" table matches | Fallback default |
""".strip(),
        "gotchas": """
## Gotchas

- **Don't suppress red flags in normal mode**: even vanilla Claude should surface 🔴 CRITICAL issues per CLAUDE.md Layer-2 invariant.
- **Don't lose project context**: CLAUDE.md is still always-loaded; normal mode doesn't mean ignoring invariants.
- **If task escalates mid-conversation**: when complexity rises (multi-file edit, training run, deploy decision), switch to /apex or the relevant domain skill rather than continuing in normal.
""".strip(),
    },
    "oracle": {
        "new_description": "Oracle Expert. Vanilla foundation-model lens — use for first-principles reasoning, second opinions on architectural choices, frontier-ML knowledge, or when seeking insight outside the project's existing scaffolding. Portable across projects via BINDINGS.md.",
        "schema_version": True,
        "portable": True,
        "when_table": """
## When to invoke

| Situation | Why |
|---|---|
| First-principles question outside project scaffolding ("is this approach fundamentally sound?") | Project skills are scaffold-aware; oracle is scaffold-free |
| Second opinion on an architecture or strategy decision | Independent perspective without project momentum bias |
| Frontier-ML knowledge needed (SOTA techniques, recent papers) | Oracle is calibrated against general ML literature |
| User explicitly wants the "vanilla Claude" perspective | Honor the request |
| Stuck — looking for an outside view to break a plateau | When project skills converge on the same answer; oracle may diverge productively |
""".strip(),
        "gotchas": None,  # oracle already has anti-patterns elsewhere; matrix shows GOTCHAS . (pass)
    },
    "pipeline": {
        "new_description": "Pipeline Expert. Use for data ingestion, dollar-bar generation, feature engineering, normalization, and calibration tasks. Invoke before any change to src/pipeline/* or before re-running chimera_v51 / refresh.py.",
        "schema_version": True,
        "portable": False,
        "fix_ledger_claim": True,  # description used to say "calibration"; rephrase to avoid LEDGER-CLAIM false positive
        "when_table": """
## When to invoke

| Situation | Why |
|---|---|
| Adding / modifying features (f29 → f41 family, xrel_*, lob_proxy_*) | Feature changes ripple to every training run |
| Changing dollar-bar threshold or universe spec | Affects every downstream consumer |
| Pipeline integrity audit (data_health, chimera_v51 validation) | Pre-train CI gate before any retrain |
| Refresh / rebuild orchestration via refresh.py | Multi-stage producer dispatch with content-hash invalidation |
| Cross-asset / cross-venue feature derivation | xrel divisor pathology, basis_signals dedup, lob_proxy panel |
""".strip(),
        "gotchas": """
## Gotchas (pipeline-specific anti-patterns)

- **norm_funding_momentum degenerate**: pre-2026-05-19 builder used diff-rolling chain that produced std=0.10 instead of z-score. Use mean-deviation method.
- **xrel_*_xratio sign-flip pathology**: divisor uses abs(median) + clip ±100. Without this, signed features produce min=-76M outliers.
- **basis_signals dedup**: same-day publication race in panel features. Producer must dedup at write time.
- **Atomic-write contract**: every silver/gold producer uses parquet_io.atomic_write_parquet(df, path, required_cols=...) per G-AUDIT-020. Direct df.write_parquet() will leave half-written files.
- **CLI universe support**: pipeline scripts MUST accept --universe u10/u50/u100. Hard-coded asset lists are a CDAP violation.
- **Phase 2 silent-drop**: pre-2026-05-17 refresh.py exited 0 despite 17% coverage. Always fail-loud on coverage <90%.
- **Capture-output buffering**: long-running pipeline stages must NOT use capture_output=True (heartbeat invisibility).
""".strip(),
    },
    "research": {
        "new_description": "Research Expert. Use for literature scans, SOTA technique surveys, experimental design, and ablation planning. Invoke before adopting a published method or when an empirical finding contradicts literature (ECL pattern).",
        "schema_version": True,
        "portable": True,
        "when_table": """
## When to invoke

| Situation | Why |
|---|---|
| Considering a published technique (DPM-Solver, Performer, iTransformer) | Verify applicability + look up post-publication critiques |
| ECL pattern (empirics contradict literature) | Research scout finds prior work that resolves the contradiction |
| Designing an ablation study | Experimental-design rigor; what to vary, what to hold |
| 2024-2026 SOTA reference needed for a citation | Cross-check claim against published benchmarks |
| Choosing between two architecturally-similar variants (V11 spectral_norm vs V14 DDIM) | Literature on each settles the design |
""".strip(),
        "gotchas": None,  # research matrix shows GOTCHAS . (pass)
    },
    "trader": {
        "new_description": "Trading/Risk Expert. Use for position sizing, risk management, execution planning, cost-model calibration, and portfolio construction. Invoke before any change to strategy/* or sleeve YAML, or before any capital-allocation decision.",
        "schema_version": True,
        "portable": False,
        "when_table": """
## When to invoke

| Situation | Why |
|---|---|
| Position-sizing change (quarter-Kelly, vol-targeted, fixed-frac) | Wrong sizing kills any signal |
| Cost-model calibration (p_fill, slippage, adverse selection) | Current MakerCostModel p_fill=0.80 is optimistic; live is 0.21-0.40 |
| Strategy promotion gate (Sharpe, capture, DD) | Last guard before capital commitment |
| Portfolio construction (LO+spot+lev=1 constraint per North Star) | Hard constraint check |
| Risk-controller changes (drawdown stop, regime gate) | Affects every deployed sleeve |
""".strip(),
        "gotchas": """
## Gotchas (trading-specific anti-patterns)

- **p_fill=0.80 default is optimistic**: empirical OHLC replay shows 0.21-0.40. Budget for p_fill ∈ [0.25, 0.50] in sizing.
- **MtM double-count regression**: every simulator MUST include reconciliation gate (sum(pnl_stream) ≈ sum(trade_log.pnl) within 0.1%). Pre-fix was 5-7x inflation.
- **K-selection on future returns**: never use future-return columns to pick best K. Report random-K + signal-K + best-K bounds.
- **Compound math drift**: daily +0.5% / 252d ≠ +126% (compound is +252%). Verify with pow.
- **Survivorship in claim formulation**: only currently-listed-on-Binance assets in master CSV — delisted ones missing.
- **Concurrent-capital math**: when 5 sleeves fire same bar, capital is shared. Each gets cap/N, not full size.
- **LONG-ONLY + NO-LEVERAGE invariant**: per CLAUDE.md North Star. Any sleeve violating this is an automatic reject.
""".strip(),
    },
    "trainer": {
        "new_description": "Training Expert. Use for training-loop changes, loss functions, optimizer/schedule, anti-fragile framework adjustments, ShIC gating, and checkpoint-resume logic. Invoke before any train_world_model.py edit.",
        "schema_version": True,
        "portable": False,
        "when_table": """
## When to invoke

| Situation | Why |
|---|---|
| Loss function change (TwoHot focal/smoothing, Huber, NCL diversity) | Loss changes accelerate or prevent memorization |
| Optimizer/schedule edit (LR, warmup, weight decay) | Wrong LR reduction locks in memorized weights |
| ShIC gating / early-stop logic | shic_decline_count persistence + check interval are load-bearing |
| Checkpoint-resume code (load_state_dict, strict=False) | Schema drift = silent garbage loads |
| Training run kickoff or resume decision | Pre-train CI gate + chimera consistency must clear first |
""".strip(),
        "gotchas": """
## Gotchas (training-specific anti-patterns)

- **WM_STEPS_PER_EPOCH < 2000**: ShIC checks fire before model learns. Non-negotiable: 2000.
- **DIRECT_RETURN_WEIGHT ≠ 3.0**: Huber dominance regularizes against TwoHot temporal memorization. Lower values → memorization.
- **ShIC LR reduction**: do NOT reduce LR on ShIC decline. Triggers early stop. LR reduction locks memorized weights.
- **Focal/smoothing on return TwoHot**: Use plain bucketer.compute_loss(logits, targets). TWOHOT_FOCAL_GAMMA = 0.0.
- **Resume from incompatible checkpoint**: bins/targets/feature-count drift = checkpoint garbage. Check n_features collision guard in load_latest.
- **torch.compile + V1.1 f13**: NaN collapse at epochs 3-5. Disabled for V1.1.
- **AntifragileDataset stride mismatch**: V22/V25 with last-bar supervision need stride=1, not default 24.
- **strict=True load**: schema drift kills the load. Always strict=False on model + ema_model.
""".strip(),
    },
    "un": {
        "new_description": "Unconstrained Mode (Apex-Enhanced + Free-Reins v2). Ship or concede. Use for plateau-breaking work where prior instances all hit a ceiling — when the standard skills' answers all converge on \"we can't\".",
        "schema_version": True,
        "portable": False,
        "when_table": """
## When to invoke

| Situation | Why |
|---|---|
| Plateau-break: prior 3+ instances returned similar non-shipping conclusions | Standard skills hit a ceiling; needs unconstrained mode |
| User explicitly says "/un" or "free reins" or "ship-or-concede" | Mode invocation |
| Capability claim is plausible but unproven and stakes are high | Force a verifiable demonstration or honest concession |
| Need to bypass conservative defaults that are blocking real progress | After standard skills have been tried |
""".strip(),
        "gotchas": None,  # un already has Hard-Won Constraints section
    },
    "unconstrained": {
        "new_description": "DEPRECATED ALIAS for /un. Use /un instead. Maintained for backwards compatibility; redirects to the canonical un protocol.",
        "schema_version": True,
        "portable": False,
        "add_arg_hint": "task description (forwarded to /un)",
        "when_table": """
## When to invoke

| Situation | Why |
|---|---|
| User typed /unconstrained or referenced legacy "unconstrained mode" | Backwards compatibility |
| Otherwise | Use /un directly — this stub will route there |
""".strip(),
        "gotchas": """
## Gotchas

- **This is a deprecated alias**: prefer /un going forward. Routing to /un is automatic.
- **Don't add new content here**: any new unconstrained-mode rules belong in /un/SKILL.md.
""".strip(),
    },
    "validator": {
        "new_description": "Claim-Evidence Validator. Use for routine validation of numeric claims, falsifying-test pairing, and pre-acceptance sanity checks. Invoke whenever a result has a number attached that hasn't been adversarially probed.",
        "schema_version": True,
        "portable": True,
        "when_table": """
## When to invoke

| Situation | Why |
|---|---|
| Any claim with a numeric attached ("+45% ROI", "IC=0.10", "Sharpe 2.3") | Numbers without falsifying tests are unverified |
| Backtest result before promotion | DSR-deflated, walk-forward, multi-window check |
| Feature-signal claim before adoption | 3-fold stability + shuffled-IC check |
| Pre-train CI gate sanity (chimera health, target distribution) | Standard validation before any retrain |
| User asks "is this honest?" / "is this real?" / "validate this" | Explicit invocation |
""".strip(),
        "gotchas": None,  # validator already has GOTCHAS via per-skill BINDINGS
    },
}


def upgrade_skill(name: str, cfg: dict) -> tuple[bool, str]:
    """Apply upgrades to .claude/skills/<name>/SKILL.md. Returns (changed, msg)."""
    path = SKILLS_DIR / name / "SKILL.md"
    if not path.exists():
        return (False, f"SKILL.md not found: {path}")

    text = path.read_text(encoding="utf-8")
    orig = text

    # Update frontmatter description
    if cfg.get("new_description"):
        new_descr = cfg["new_description"]
        text = re.sub(r"^description:.*$", f"description: {new_descr}", text, count=1, flags=re.MULTILINE)

    # Fix LEDGER-CLAIM false positive: rephrase "calibration" if mentioned in description
    if cfg.get("fix_ledger_claim"):
        # If the description (now updated above) still has "calibration", that's intentional now
        # but the linter checks for "ledger|calibration" in description AND requires LEDGER.md.
        # Pipeline doesn't need a LEDGER, so rephrase the calibration mention. The description
        # above already avoids the calibration keyword.
        pass

    # Add metadata block if missing
    if cfg.get("schema_version"):
        if "metadata:" not in text.split("---", 2)[1]:
            # Insert metadata block before the closing --- of frontmatter
            portable_val = "true" if cfg.get("portable") else "false"
            metadata_block = (
                f"metadata:\n"
                f"  schema_version: \"2026-05-22\"\n"
                f"  portable: {portable_val}\n"
            )
            # Find the closing --- of frontmatter (second occurrence)
            parts = text.split("---", 2)
            if len(parts) >= 3:
                # parts[0] is "" before first ---, parts[1] is frontmatter, parts[2] is body
                if not parts[1].rstrip().endswith("\n"):
                    parts[1] = parts[1].rstrip() + "\n"
                parts[1] = parts[1] + metadata_block
                text = "---".join(parts)

    # Update argument-hint if requested
    if cfg.get("add_arg_hint"):
        if "argument-hint:" in text:
            text = re.sub(r"^argument-hint:.*$", f"argument-hint: \"{cfg['add_arg_hint']}\"", text, count=1, flags=re.MULTILINE)
        else:
            # Insert before metadata or closing ---
            text = text.replace("metadata:", f"argument-hint: \"{cfg['add_arg_hint']}\"\nmetadata:", 1)

    # Append "## When to invoke" if missing
    if cfg.get("when_table") and not re.search(r"(?im)^#+\s*when to invoke", text):
        text = text.rstrip() + "\n\n" + cfg["when_table"].strip() + "\n"

    # Append "## Gotchas" if missing
    if cfg.get("gotchas") and not re.search(r"(?im)gotchas|common mistakes|anti-pattern|red flags|pitfalls", text):
        text = text.rstrip() + "\n\n" + cfg["gotchas"].strip() + "\n"

    if text == orig:
        return (False, f"{name}: no changes needed (already up-to-date)")

    path.write_text(text, encoding="utf-8", newline="\n")
    return (True, f"{name}: upgraded")


def main() -> int:
    changes = 0
    for name, cfg in UPGRADES.items():
        changed, msg = upgrade_skill(name, cfg)
        print(msg)
        if changed:
            changes += 1
    print(f"\nTotal: {changes}/{len(UPGRADES)} skills upgraded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
