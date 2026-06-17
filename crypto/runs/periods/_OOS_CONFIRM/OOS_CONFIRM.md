# Out-of-sample confirmation — does the assembled stack transfer?

The fixed-approach arc (ladder -> keeper stack -> regime gate -> complete stack) was all TRAIN-era
IN-SAMPLE structural design. This is the honest litmus: run the variants on the full VAL and OOS spans
(data the design never saw) + a 2021 TRAIN-reference. Tool: `src/strat/oos_confirm.py`. Equal-weight u10
book, causal MtM. **UNSEEN (2025-12-31..2026-06-01) stays SEALED.** ROI% / maxDD% / Sharpe / %posCells.

| variant | what |
|---|---|
| NAIVE | all 120 configs, signal-flip, TAKER (un-fixed baseline) |
| FIXED | 2MA-slow(60-150) family, signal-flip, TAKER (the family fix) |
| FULL | FIXED + TRAIL(10%) + min_hold(12) + MAKER (the keeper stack) |
| FULL_GATE | FULL + BTC100-hysteresis market gate (the 4h regime overlay) |

## Transfer table (4h+1h mean book ROI%)
| span | NAIVE | FIXED | FULL | FULL_GATE | FULL−NAIVE | GATE−FULL |
|---|---|---|---|---|---|---|
| 2021 (bull ref) | 573.6 | 854.3 | 172.8 | 106.8 | -400.8 | -66.0 |
| **VAL** (24-05→25-03) | 20.3 | **31.0** | 17.9 | 17.4 | -2.4 | -0.6 |
| **OOS** (25-03→25-12) | -19.4 | -12.2 | **-6.5** | -7.3 | **+12.9** | -0.8 |

## What transfers, what doesn't (the honest read)
1. **The FAMILY fix TRANSFERS.** FIXED (2MA-slow) beats NAIVE (run-everything) on BOTH VAL (31.0 vs
   20.3) and OOS (-12.2 vs -19.4). Restricting to the robust family is a genuine, out-of-sample
   improvement over running every config -- the single most durable result of the whole arc.
2. **The FULL stack is a regime-dependent DRAWDOWN CONTROLLER, not a return-enhancer.** It HURTS in the
   bull (2021 173% vs FIXED 854%; VAL 17.9 vs 31.0) because the 10% trail + min-hold cut upside on
   pullbacks. But it HELPS exactly when it matters -- the hard OOS: FULL -6.5 vs NAIVE -19.4 (**+12.9pp**),
   and at 4h FULL is the ONLY positive variant (+2.0%, maxDD -25 vs naive -28, 53% breadth). On the OOS
   equity curve FULL ends highest with the lowest drawdown -- it gives back the least in the late-2025
   decline. It lowers variance across regimes: you pay upside in bulls to lose less in the bad tape.
3. **The BTC market gate does NOT transfer -- REFUTED as a general overlay.** FULL_GATE is ~= or slightly
   WORSE than FULL on every out-of-sample span (VAL -0.6, OOS -0.8). The clean 4h bear-softening I found
   on the single Jun-2022 window did NOT generalize. This is EXACTLY the caveat flagged at build time
   ("the 1-bear-window market-gate edge is the least-sampled claim -> MUST confirm VAL/OOS"). Confirmed:
   it was an in-sample artifact of one bear window. The gate's cash "steps" on the OOS curve show it
   sitting out roughly neutrally. Honest refutation -- drop the gate.

## The honest ceiling
**Long-only MA on crypto is flat-to-negative on the hard OOS span (2025-03..12) even with the best
stack.** Only 4h FULL is positive (+2.0%); everything at 1h is deeply negative (NAIVE -33%, FULL -15%).
The structural arc makes the book MORE ROBUST and LESS BAD (family fix + drawdown control cut the OOS
loss from -19% to -6%, and 4h to +2%) -- but it does NOT manufacture out-of-sample alpha on a losing
tape. This converges with the project-wide finding (no verified active alpha at 4h/daily; internal-data
ceiling): entries/exits/costs/regime-gates reshape the RISK profile; they do not turn a beta-negative
period positive for a long-only book.

## Verdict (the build-fix-upgrade arc, end to end)
- **KEEP: the family fix (2MA-slow).** Transfers OOS. The core win.
- **KEEP for risk, not return: the FULL stack (trail + min-hold + maker).** A robust drawdown controller
  -- cuts OOS losses ~13pp vs naive, best OOS ending equity + lowest DD; costs bull upside. Worth it iff
  the objective weights robustness/maxDD (it does -- North Star: robust compound, maxDD<30%).
- **DROP: the BTC market gate.** In-sample only; refuted on VAL/OOS.
- **The ceiling is honest:** robustness up, OOS loss down, but no manufactured OOS alpha. Long-only +
  spot + lev=1 reshapes risk; it does not beat a negative tape.

All numbers RWYB-reproducible: `python -m strat.oos_confirm`. UNSEEN never touched.
Chart: `charts/oos_confirm.png` (left: ROI by variant x span, 4h; right: OOS book equity by variant --
FULL highest + lowest DD, NAIVE worst, gate's cash steps visible).

## Robustness battery (canonical scorecard) -- IMPORTANT CORRECTION
The variant comparison above used POINT estimates. Running the FULL 4h stack through the canonical
`strat.scorecard.score_book` (`python -m strat.grade_full_stack`; series ends 2025-12-31 so **UNSEEN
stays sealed -- n=0, untouched**) tempers the "robust drawdown controller" claim:

| metric | value | North Star bar | pass? |
|---|---|---|---|
| SEL (TRAIN+VAL) compound | +636.7% (ann +46.9%, Sharpe 1.50, maxDD -25.5%) | -- | strong in-sample (bull-driven) |
| OOS compound | +1.99% (ann +2.5%, Sharpe 0.23, maxDD -23.6%) | >0 | barely |
| **OOS-heldout block-bootstrap p05** | **-32.97%** | **p05 > 0** | **NO** |
| **PBO (20-config grid)** | **0.71 (FAIL)** | < 0.5 | **NO** |
| **OOS per-asset breadth** | **5/10** (ETH+45 BNB+30 SOL+15 BTC+3 ADA+2; AVAX/DOGE/LINK/XRP/LTC neg) | majority >0 | borderline (exactly 50%, majors-concentrated) |
| full-cycle p05 | +135.1 | -- | positive but SEL-bull-dominated, NOT a robustness signal |

**Honest correction:** the FULL stack reduces loss RELATIVE to naive (the variant table is real), but it
is **NOT robust in absolute terms** -- OOS-heldout block-bootstrap p05 is -33% (deeply negative), PBO
FAILs at 0.71, and OOS breadth is exactly the coin-flip line with the wins concentrated in the majors.
So: "cuts the OOS loss vs running-everything" = TRUE; "a shippable robust positive-expectancy book" =
FALSE. The drawdown control is a relative risk improvement, not an absolute edge. This is the canonical
North Star bar (p05>0 AND breadth-majority) doing its job -- the +2% OOS point estimate was fragile.
Scorecard json: `full_stack_scorecard.json`. UNSEEN remains sealed for a future USER-gated test-once.
