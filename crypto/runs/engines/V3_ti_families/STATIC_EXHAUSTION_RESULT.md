# Static Exhaustion — RESULT (2026-06-24)

**User's premise (correct):** don't rush to Phase 2 (dynamic). If static can't be made
to work, dynamic can't either — dynamic is just static-with-adaptive-params. So PROVE
the static design space is exhausted first.

## I was wrong earlier. The non-MA "null" verdict was UNDER-POWERED.

The first SOTA OOS sweep tested only **vote_threshold=0.5 with NO confirmation layer**.
That is NOT the architecture the MA family won with. The MA family uses a decoupled
**confirmation layer** (`signal & close>EMA_lookback & EMA rising`) + a fixed slice-hold,
and its threshold was tuned. The non-MA families never got those two static levers.

When I pull them (sweep threshold {0.5–0.8} × confirmation {off,on}), the picture changes:

### Threshold-robustness (the anti-noise signature)
Three (family, TF) cells beat always-long OOS at **every** threshold×confirm setting —
that robustness-to-the-knob is the opposite of the scattered single-TF wins I saw before:
- **MACD|1h + confirmation:** +3.1 to +3.5pp/wk, ~60% deploy
- **Bollinger|1d:** +2.3 to +2.7pp/wk
- **RSI|4h:** +0.8 to +1.9pp/wk

The confirmation layer consistently *raised* the edge (e.g. MACD|1h +1.95 → +3.5pp).

## But threshold-sweeping ON OOS risks selection bias. So I applied the GOLD bar.

The MA family cleared two tests the non-MA families never faced: **DEV/OOS consistency**
+ **random-null ≥95th pct**. Ran both on the three standout cells (n=200 null draws):

| Cell           | DEV edge | OOS edge | DEV/OOS consistent | Random-null | VERDICT |
|----------------|----------|----------|--------------------|-------------|---------|
| **Bollinger\|1d** | +0.51pp | +2.38pp | YES | 100th pct | **REAL EDGE** |
| **RSI\|4h**       | +1.23pp | +0.77pp | YES | 100th pct | **REAL EDGE** |
| MACD\|1h          | -0.62pp | +3.24pp | NO  | 100th pct | NOT confirmed (OOS-fit) |

- **Bollinger|1d & RSI|4h are genuine static engines.** Positive on BOTH windows (so
  selectable BEFORE seeing OOS — no selection bias) AND beat random-null at 100th pct
  (so it's the TIMING, not just being-long-under-the-gate).
- **MACD|1h is NOT confirmed.** Its huge OOS edge sits on a NEGATIVE DEV edge — it would
  not have been chosen pre-OOS. The 100th-pct null pass is real for that window but the
  DEV/OOS inconsistency means it's likely fitting the specific OOS regime. Reject for now.

## What this means for the static phase
1. Static is **NOT exhausted** — and it **DOES work** beyond the MA family. RSI|4h and
   Bollinger|1d join the MA family as validated static engines.
2. The earlier "non-MA null" was a measurement artifact: wrong (too-permissive) config,
   missing confirmation layer, single threshold. Corrected now.
3. The honest-baseline caveats still hold (compound inflated ~9.5x by EW-rebal; report
   linear/wk; the edge is vs always-long under the same gate).

## Remaining 5 families through the proper levers (2026-06-24)

Ran the SAME threshold×confirm sweep + DEV/OOS consistency on Stochastic / ADX /
Supertrend / Funding / OI (Funding/OI with REAL chimera features):

| Family     | Best cell                | DEV edge | OOS edge | Deploy | Random-null | Verdict |
|------------|--------------------------|----------|----------|--------|-------------|---------|
| ADX        | 4h thr0.6 (no confirm)   | +0.74pp  | +1.09pp  | 6.9%/asset | **93rd pct** | BORDERLINE (fails 95 bar) |
| Stochastic | 1d thr0.5 confirm        | +0.04pp  | +0.25pp  | ~62%   | not run     | too weak (only DEV/OOS-consistent cell, tiny) |
| Supertrend | — | — | — | ~98% | — | always-long (not selective) |
| Funding    | — | — | — | ~99% | — | always-long (not selective) |
| OI-div     | — | — | — | ~99% | — | always-long (not selective) |

- **ADX|4h is the most genuinely SELECTIVE candidate** (6.9% per-asset deploy, very
  picky). DEV/OOS consistent. But random-null = 93rd pct — JUST below the 95 bar. So it
  is BORDERLINE: probably a weak real edge, not confirmed at the MA family's standard.
  Worth a lookback-sweep revisit, but not deployable as-is.
- Stochastic/Supertrend/Funding/OI: no selective DEV/OOS-consistent edge. Confirmed null.

## FINAL STATIC SCORECARD (validated = DEV/OOS consistent + random-null >=95th pct)

| Engine        | Status        | OOS edge vs AL | Notes |
|---------------|---------------|----------------|-------|
| **MA family** | **VALIDATED** | +4.11%/wk      | gold standard, 100th-pct null |
| **RSI\|4h**    | **VALIDATED** | +0.77pp        | DEV+OOS, 100th-pct null |
| **Bollinger\|1d** | **VALIDATED** | +2.38pp     | DEV+OOS, 100th-pct null |
| ADX\|4h       | borderline    | +1.09pp        | 93rd pct — not quite |
| MACD\|1h      | rejected      | +3.24pp OOS    | DEV negative → OOS-fit |
| Stochastic/Supertrend/Funding/OI | null | — | not selective / no edge |

## Answer to "is there more to do on static?"
YES, and it was worth doing. Static is NOT exhausted in the sense of "nothing works" —
the opposite: pulling the proper levers REVEALED **two new validated static engines**
(RSI|4h, Bollinger|1d) on top of the MA family, and one borderline (ADX|4h). My earlier
"all non-MA null" verdict was a measurement error (under-powered config).

### Static work still genuinely remaining (do BEFORE Phase 2)
- [ ] Lookback/slice sensitivity sweep on the 3 validated cells (RSI|4h, Bollinger|1d, +MA)
      — overfit probe: edge surface must be SMOOTH, not a spike.
- [ ] ADX|4h lookback revisit (it's selective; a better lookback may push it past 95th).
- [ ] Combine validated static engines (MA + RSI|4h + Bollinger|1d) into an anchored
      ensemble and validate the COMBINATION OOS (diversification may lift the whole).
- [ ] THEN, with a working multi-engine static base, evaluate whether dynamic adds value.

The user's gate is satisfied in principle: static demonstrably works (3 validated engines),
so dynamic has a real foundation to build on. But the lookback/overfit probes + the
ensemble-combination test should close out the static phase first.
