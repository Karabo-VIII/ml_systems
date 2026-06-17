# Parallel Researcher Report #1 (scout-strat) — 2026-06-05 ~23:28

Bidirectional loop: overseer's independent researcher, in parallel with the build-rigs. VERIFIED = from in-tree
JSON; INFERRED = derived; REPORTED = web-cited.

## CRITICAL EMPIRICAL RESULT (VERIFIED, from rig-E results_u100.json + falsifiers.json)
- **0/69 assets beat the random-entry firewall on held-out.** MA-cross timing adds ZERO above random → beta-in-disguise.
- **UNSEEN per-trade expectancy: −2.09% (adaptive) vs −2.50% (fixed) — BOTH NEGATIVE.** Neither clears the 2–5% target.
- **F-B: ALL 6 constituent fixed configs beat the adaptive switcher** on held-out. Adaptation HURTS (354 UNSEEN trades adaptive vs 227 fixed → more trades, lower quality).
- **F-C beta: only 2/25 beat beta-matched costless hold** → capturing market beta, not idiosyncratic timing alpha.
- Cost is NOT the primary killer (taker vs ideal ~2pp median); the **entry timing** is. 1d cross fires a bar late.
- → **The 1d adaptive-MA cross approach is REFUTED.** (Honest null — rig-E self-falsified rigorously.)

## CADENCE VERDICT (INFERRED): 4h is structurally correct
- Round-trip taker 0.24%; to net 2–5% needs 2.24–5.24% gross. 1d: needs a ~P70–P85 single-day move AND the cross
  fires a day late (enters into the existing move). 4h: a 3-ATR (~3%) move in 6–12 bars (1–2 days) nets ~+2.76% —
  clears the floor without a tail event, 4× temporal resolution for entry/exit, 4× UNSEEN trades (→ positive_control
  Lens A reachable). 1h: excessive whipsaw friction (~38% vs 14%). **Use 4h primary.**

## TOP ADJACENT OBJECTIVES (EV-ranked) — the redirect
1. **ER-GATED ENTRY (gate, not switch):** trade the MA cross ONLY when ER > ~0.4 (trending); SKIP chop entirely.
   1 DOF. Directly kills the chop false-crosses that cause the 0/69 firewall failure. (rig-E currently uses ER to
   pick wider/narrower windows but STILL trades every cross — RF-3.)
2. **4h + ATR-trail exit via SetupHarness (TP/SL/trail), NOT opposite-cross.** Opposite-cross exits into reversal
   at a loss (RF-4). ATR-trail banks the 2–5% move. Fits the hours-to-<7d hold.
3. **Breakout-confirmed entry:** require close > prior N-bar high AND fast>slow → the subset of crosses with real
   momentum (strict subset → fewer, higher-quality trades). Verify via SetupHarness.leak_guard().
4. **Vol-targeted sizing** (ATR-normalized), not full-capital (RF-5).

## OVERFIT BOUNDS (the anti-overfit mandate)
- The DOF risk is the **feature→regime boundaries** (252-bar per-asset percentile), not the map cells.
- **Minimal honest config = 3 DOF: ER threshold + 1 MA config + 1 exit policy** (= ER-gated fixed-MA). This is the
  null to beat BEFORE adding map cells. Current 6-cell map tests 6 configs before ANY has shown a timing edge —
  wrong order.
- Audit protocol (existing apparatus): regime-matched firewall (`regime_matched=True`); positive_control at 4h
  (verify the gate has POWER); block-bootstrap p05/p95 + jackknife on UNSEEN; multi-seed PCT_WIN ∈ {126,252,504};
  walk-forward UNSEEN split (2025-09-07); DSR/Holm at family_n.

## RED FLAGS in rig-E (HIGH unless noted)
- RF-1 adaptation increases trade count not quality; RF-2 all 6 fixed beat adaptive; RF-3 ER selects windows
  instead of SUPPRESSING entries (should gate); RF-4 opposite-cross exit misaligned with 2–5%/trade; RF-5
  full-capital sizing (MED); RF-6 252-bar warmup → first year trades from wrong bucket (MED); RF-8 no TP/SL exit (MED).

Sources: KAMA/Efficiency-Ratio guides, ATR crypto sizing, walk-forward/overfit arXiv (2512.12924, 2602.00080, 2603.09219).
