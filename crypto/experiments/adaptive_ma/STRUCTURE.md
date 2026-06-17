# Adaptive-MA workspace — isolation + integration

THREE autonomous loops, each ISOLATED in its own folder (no writing to src/ or to each other; src/strat +
chimera_loader are used READ-ONLY). This keeps the main project clean (no bloat) while staying integrable.

- experiments/adaptive_ma/expert/  -- rig-E (metaop --mode expert): builds the strategy.
- experiments/adaptive_ma/plain/   -- rig-P (metaop --mode plain):  builds the strategy (comparison arm).
- experiments/adaptive_ma/meta/    -- META loop (metaop): the orchestrator's autonomous meta-loop -- audits,
                                      researches, formulates adjacent objectives, writes GUIDANCE to the rigs.

## Shared (read by all, written by the overseer/meta)
- docs/ADAPTIVE_MA_BRIEF_2026_06_05.md   -- the binding spec (bounded goal: 2-5%+ net/move, 4h, hours-to-<7d).
- experiments/adaptive_ma/RESEARCHER_REPORT_*.md, OVERSEER_LOG.md, META_LOOP_LOG.md.
- runs/autonomy/learnings/{expert,plain,meta}.jsonl -- the learnings lanes (mid-run guidance flows here; plan AND
  reflect now read them, so [META]/[OVERSEER] guidance reaches a running rig).

## Integration path (later, by the overseer, reviewed)
A strategy is PROMOTED from experiments/ to src/strat/adaptive_ma/ ONLY when it is VERIFIED: held-out per-trade
expectancy 2-5%+ net, BEATS the regime-matched random-entry null, minimal-DOF, maximally audited (firewall +
positive_control + block-bootstrap p05 + jackknife + walk-forward UNSEEN). Until then it stays isolated.
