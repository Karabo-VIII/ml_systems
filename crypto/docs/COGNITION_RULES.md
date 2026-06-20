# COGNITION_RULES — the binding anti-collapse rules (git-tracked digest; Tier-1, inject into every worker)

Durability fix (PILLAR 3): these rules were scattered across `crypto/CLAUDE.md`, the out-of-repo `feedback-*.md`
auto-memory, and the gitignored `runs/autonomy/rolling/current.jsonl`. They are the rules that PREVENT the
framing-collapse / quit-at-verdict failure — so they now live in git and are injected into every dispatched
worker via `scripts/knowledge/inject.py`. Keep this small and current. Source of truth for intent: `crypto/CHARTER.md`.

## The one failure this whole operating model exists to kill
An instance (1) collapses a high-dimensional ask to its easiest 1-D shadow (u100 → "just BTC"), then (2) hits a
null on that shadow and quits at the verdict, reporting the whole mandate dead. BOTH ends are banned.

## The rules (each is a real, repeatedly-corrected failure)
1. **Decompose the IDEAL; never collapse the framing.** One test refutes ONE FRAMING, not the problem. Enumerate
   ≥2 framings, take the resourceful one. The best-achievable strategy WITHIN the constraints is the oracle to
   reverse-engineer — the gap is the honest ceiling. (`scripts/autonomy/resourcefulness.py`)
2. **A null is honest ONLY if the FULL surface was attempted** — ≥20 u100 assets AND the lookback window AND the
   existing move-catch infra — proven by run-output ARTIFACTS (symbol columns, a git SHA), not prose. A null on a
   1-asset / 1-TF shadow does NOT refute a u100 mandate. (Enforced mechanically by the give-up gate.)
3. **Don't stop at a verdict — switch to BUILDER mode.** After ANY refutation, build the most deployable robust
   system WITHIN the ceiling. The verdict is the FLOOR of the analysis, not its end.
4. **Long-only + spot is permanent. The short / long-short / market-neutral move is OFF THE TABLE.** Every instance
   gravitates to it; that is THE shortcut, not a solution. When the bear "needs a short", the honest answer is
   capital-PRESERVATION within long-only. Only the USER may relax a binding constraint; never propose it as the deliverable.
5. **Each constraint is an ENABLER**, not only a limit — name the path it opens.
6. **Rank by WEALTH (compound net), never Sharpe. IC / per-bar predictability is BANNED as a primary metric** —
   judge by move CAPTURE-RATE + WEALTH.
7. **Sweep ALL timeframes** {1d,4h,2h,1h,30m,15m}; never silently default one cadence — state cadence + why.
8. **USE THE WHOLE PROJECT.** Extend existing infra and MINE the dead-list (D01–D80) — never re-run a killed vein.
   Cite the D-ID when killing a proposal. The CASH THEOREM (D78) bounds "trade any slice" at ≈0.34 LO-spot — the
   honored version is the best within-budget chaser across the full surface, not "every slice".
9. **Claude MANAGES; do not defer a revertible decision to the user.** Decide, state reasoning, proceed; git is the
   revert net. Escalate only for genuinely irreversible real-world actions.
10. **Deliver end-to-end, artifacts where the user looks** (`runs/periods/...`), nothing pending. Plain talk, no preamble.

## The honest fallback (when the surface is genuinely exhausted)
Long-only internal-data alpha may not exist — the standing premise, not a failure. The constructive default is
capital-preservation within long-only + the external-listing-data frontier (Coinbase/Upbit) driven to DONE — built
within the constraint, never handed back as "mandate dead".
