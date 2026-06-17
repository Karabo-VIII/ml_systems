# Intraday oracle-decompose — FINDINGS (2026-06-11)

The honest test of the user's intraday-speculation thesis ("we can see trends in the sub-daily
charts → there are intraday opportunities to exploit"). Tool: `src/mining/intraday_oracle.py` —
event-clock breakout trend-ride (enter on the bar, ride up to 3 days, ONE position per asset at a
time = genuinely low-turnover, avoiding the bar-clock trailing-stop whipsaw that cost-walls
`finecadence` F1). Oracle ceiling vs causal capture vs random-entry null, canonical-scorecard graded,
SEL/OOS/UNSEEN. u50, taker + maker. **The user's exact frame: 5m–4h entry, 3-day max hold.**

## Result — every intraday cadence, u50 (taker unless noted)
| Cadence | events | OOS oracle/event | OOS capture | **UNSEEN causal mean±SE** | UNSEEN vs-null | UNSEEN jackknife | hold |
|---|---|---|---|---|---|---|---|
| 15m | 52,816 | **+6.89%** | −0.099 | **−0.248±0.056%** | −0.05pp | −0.78 | ~4h |
| 30m | 43,172 | +6.51% | −0.114 | −0.074±0.094% | +0.13pp | −0.79 | ~8h |
| 1h | 33,883 | +6.13% | −0.135 | −0.226±0.118% | +0.02pp | −1.08 | ~15h |
| 1h (maker) | 33,883 | +6.31% | −0.111 | −0.046±0.118% | +0.20pp | −0.90 | ~15h |
| 2h | 22,576 | +6.20% | −0.181 | −0.331±0.211% | +0.28pp | −1.55 | ~34h |
| 4h | 21,336 | +5.40% | −0.054 | −0.318±0.219% | +0.15pp | −1.59 | ~2d |

## The verdict — the wall holds at every cadence, BUT the prize is now quantified
1. **The oracle ceiling is REAL and large: +5.4 to +6.9% per event, net of cost.** The user is
   right — the intraday moves seen in the charts genuinely exist, are abundant (22k–53k events on
   u50), and are big. The opportunity is not the problem.
2. **The causal breakout entry captures a NEGATIVE fraction (−5 to −18%) at every cadence.** It loses
   money despite the +6% ceiling — the entry is structurally too late / whipsawed / on the wrong side
   (D55 direction-unpredictability, D67 capture-negative, both timeframe-agnostic).
3. **UNSEEN causal mean is negative at every cadence, with a negative jackknife everywhere.** The few
   cadences that marginally "beat null" do so via concentration, not a robust edge. **No cadence is a
   sliver** (none has UNSEEN mean>0 AND beats-null AND jk-positive).
4. **Maker does not rescue it** (1h maker −0.046%, still negative + jk-negative).

This is D67/D72 reproduced precisely at the user's 5m–4h / 3-day-hold / event-clock / taker+maker
frame, across the full cadence grid. The intraday wall is confirmed with the user's own parameters.

## The reframe this produces (the genuinely useful output)
The gap between **oracle +6%/event** and **causal −0.2%/event** is the **information wall, quantified**:
~6 percentage points of move exist per event, but a causal internal-data entry cannot be on the right
side of it. This sharpens the value of the one un-refuted path: a **trigger-time continuation signal
with OOS AUC ≥ 0.58** (the D72/thread-24 spec; internal data tops out at 0.52) would unlock a fraction
of that +6%/event across thousands of events. **The intraday prize is large and sitting there; the
only lock is the discriminator, and the only known key is external/leading data** (Coinglass liq-
heatmap proximity, on-chain netflow, news/social). The decompose turns "is intraday worth it?" into
"the ceiling is +6%/event — buy the key (external data) or accept the slow regime-gated book."

## What is NOT killed (honest residue)
- The **magnitude / volatility channel** — D55 says *direction* is dead at every TF but *magnitude*
  is predictable. A convexity/vol-targeted intraday strategy (bet move SIZE, not direction) is
  untested here and is the one internal channel with a non-dead prior.
- **5m** (not run here — no 5m chimera; u10-only via the 1m liq_subbar data) — prior is the same wall
  (finer = more whipsaw, D72 at 1m already null), low EV but unrun.
- **External-data discriminator** — the quantified, ready-to-test path (harness + sealed UNSEEN), the
  user's call (deferred pending returns; this decompose is the "returns" case FOR it).

Repro: `python -m mining.intraday_oracle --universe u50 --cadences 15m,30m,1h,2h,4h` (+ `--maker`).
Artifacts: `runs/mining/intraday_oracle_u50_2026061{1}_*.json`.
