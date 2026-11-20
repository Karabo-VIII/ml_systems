# Build-fix-upgrade-validate ARC -- canonical summary (2026-06-12)

The user directive ("run normal configs vs configs+(fixed and upgrades), build/fix/upgrade the FIXED
approach, work autonomously 3h") taken end-to-end through OOS validation, canonical robustness grading,
and a 7-axis frontier map. 13 committed blocks, all `src/strat/`, period-keyed in `runs/periods/`.
LONG-ONLY + spot + lev=1. UNSEEN (2025-12-31..2026-06-01) NEVER touched -- spending it is a user call.

## The leaderboard (FULL stack = family + trail10 + minhold12 + maker; held-out OOS, 4h unless noted)
| base / lever | OOS compound | OOS Sharpe | OOS maxDD | breadth | OOS-heldout p05 | robust? |
|---|---|---|---|---|---|---|
| NAIVE (run-everything MA) | -5.6 | -0.1 | -27.5 | 36% | -- | no |
| MA-cross FULL | +2.0 | 0.23 | -23.6 | 50% | -33.0 | NO |
| **Donchian breakout FULL (4h)** | **+11.0** | **0.72** | **-14.4** | **72%** | **-21.2** | NO (best base) |
| breakout @1h / @30m / @1d | +2.0 / +1.6 / -6.7 | 0.22 / 0.16 / -0.66 | | 52/50/32% | -28/-28/-21 | no |
| MA+breakout ensemble (same-beta) | +6.5 | 0.45 | -18.9 | 62% | -27.3 | no (just averages) |
| breakout+MR ensemble (orthogonal) | +9.5 | 0.70 | -13.8 | 66% | -17.4 OOS / WORSE on VAL | no (span-dependent) |

## What each axis taught (the 7-axis map)
1. **Family fix (pick robust 2MA-slow / slow-N family > run-everything): TRANSFERS** OOS (the one durable
   structural win; VAL 31 vs 20, OOS -12 vs -19 for MA).
2. **Overlays:** trail(10% loose) + min_hold(12) + maker = a RELATIVE drawdown reducer (cuts OOS loss vs
   naive) but min-hold/maker are FINE-cadence levers (no-op at 4h). Loose 10% trail protects clean crashes
   without whipsawing chop (tight 5% lost) -- resolves the exit-mechanism question.
3. **Archetype is the BIGGEST lever:** Donchian breakout beats MA-cross on EVERY held-out axis (Sharpe
   0.72 vs 0.23, breadth 72% vs 50%). The MA-centric framing was too narrow.
4. **Timeframe (swept, HARD RULE):** breakout @4h is the validated sweet spot (1d too coarse, 1h/30m pay
   cost). p05 ceiling holds at every cadence.
5. **Regime gates REFUTED OOS (dead-list D74):** BOTH price (BTC-SMA hysteresis) AND funding gates help
   ONE in-sample window but fail OOS. Self-referential gates fail hardest. Hysteresis fixing the 1h-bull
   flicker is the one kept apparatus technique.
6. **Same-beta ensemble (MA+breakout) does NOT diversify** -- just averages (both long-only trend beta).
7. **Orthogonal-beta diversification (breakout+MR) is SPAN-DEPENDENT** -- helped OOS p05 (4/4 seeds) but
   HURT on VAL (0/4 seeds; MR's -12.9% VAL loss dominated). The 2nd-span firewall caught a one-window lead.

## The honest verdict
**No internal-data long-only configuration produces a held-out-ROBUST positive book.** The best base is
**Donchian breakout @4h, family-fixed, drawdown-stacked** -- it is the best held-out performer (OOS +11%,
Sharpe 0.72, breadth 72%, maxDD -14%) and a legitimate RELATIVE risk improvement over naive -- but it
FAILS the North Star absolute-robustness bar (OOS-heldout block-bootstrap p05 = -21% < 0; PBO 0.71 FAIL;
bear still negative). Every structural lever (overlays, gates, ensembles, timeframe, orthogonal MR) was
tested with held-out + 2-span + multi-seed + PBO + breadth rigor; all converge on the same ceiling:
**the long-only crypto-trend BETA tail.** Structure reshapes RISK; it does not manufacture held-out alpha.
Converges with the project-wide no-active-alpha-at-4h/daily finding and the dead-list.

## The strategic fork (USER decision -- the only un-refuted forward levers)
Escaping the long-only-beta ceiling needs a genuinely RELIABLE ORTHOGONAL source (a weak regime-dependent
MR is not it):
- **A. External/leading data** (Coinglass liquidation heatmap, on-chain netflow, social/news) -- the
  dead-list (D71/D72/D74) repeatedly points here; Coinglass sign-off is PARKED in the campaign charter.
- **B. Shorting** (a bear sleeve) -- out of the current LONG-ONLY scope; bear-short sign-off PARKED.
- **C. Accept the ceiling** -- ship breakout @4h as a RELATIVE-risk-controlled beta sleeve (not alpha),
  inside the core+satellite book the charter describes, and stop chasing internal alpha.

Recommendation: this is genuinely the user's call (A/B reshape the mandate; C accepts the honest ceiling).
The internal frontier is thoroughly + honestly mapped; further internal mining is the over-mining trap.

Blocks: fixed_approach / fixed_stack / regime_gate_search / complete_stack / oos_confirm / grade_full_stack
/ funding_gate_test / breakout_arc / breakout_tf_sweep / ensemble_arc / mr_diversify / mr_robust (+ D74).
All RWYB-reproducible; per-block docs alongside this file.
