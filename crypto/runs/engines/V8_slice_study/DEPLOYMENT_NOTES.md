# DEPLOYMENT & INSTRUMENT SELECTION — notes (2026-06-27)

## The question: given multiple simultaneous signals, FIFO or what?

ANSWER: NOT FIFO. The production allocator (firm/portfolio.py `allocate()` used by
portfolio_replay.run()) does SIMULTANEOUS risk-budgeted allocation, re-evaluated every bar:

1. ALL currently-signaling assets are candidates (no queue, no first-come-first-served).
2. Each gets a raw stake = inverse-volatility (risk parity) — quiet assets get more weight.
3. Book is VOL-TARGETED to 2%/period (correlation-aware via covariance, not sum-of-vols).
4. Caps bind: max_per_name=0.15 (15%/asset), max_gross=1.0 (fully invested, long-only).
5. If signals exceed budget, ALL weights shrink proportionally (names are NOT dropped).

So instrument "selection" = everyone gets in, sized by inverse-vol, capped per-name, book
shrunk to fit 100% gross. Egalitarian risk-parity, NOT ranked/FIFO.

## CRITICAL FIDELITY GAP (honest disclosure)
The V8 slice study + ma_prod_report_v2 numbers use NAIVE EQUAL-WEIGHT (1/n_held across
held assets), NOT the production allocate() machinery. Difference:

| dimension      | slice study (reports)     | portfolio_replay (PROD)        |
|----------------|---------------------------|--------------------------------|
| weighting      | naive EW (1/n)            | inverse-vol risk parity        |
| vol target     | none                      | 2%/period                      |
| per-name cap   | none (1 signal -> 100%)   | 15%                            |
| correlation    | ignored                   | covariance-aware               |
| gross cap      | implicit (EW sums to 1)   | explicit 1.0                   |

CONSEQUENCE: the +4%/wk headline was measured on naive EW. The DIRECTION is robust
(4h/1d work, rolling>fixed, whipsaw helps, 30m/15m dead) but the EXACT deployed return
will differ under the prod allocator (vol-target + per-name cap change the realized number).
Before any deploy claim, the candidates MUST be re-run through portfolio_replay.run() with
allocate() to get the true deployable expectation. TODO.

## Deployment options (decide deliberately)
A) Risk-parity equal-ish (prod allocate(), validated machinery) — RECOMMENDED to start.
   It's the machinery the committed infra uses; expectation transfers honestly.
B) Add signal-strength RANKING + concentrate into top-K — higher conviction but:
   - higher concentration risk
   - needs separate validation vs the EW baseline
   - DEAD-LIST WARNING D62: per-asset feature conditioning of MA capture FAILED OOS
     (1/47 above noise; "winner" collapsed date-demeaned = regime effect). A ranker is
     a research project, not a free win. Do NOT add until separately validated.
C) Tier the book by timeframe (1d/4h/2h sub-books with own budgets) — cleaner ops,
   different turnover/hold horizons per tier.

## Recommendation
Deploy via PROD allocate() (option A). Re-run the 1d/4h/2h candidates through
portfolio_replay.run() to get the true (vol-targeted, per-name-capped) numbers before
committing to an expectation. Ranking (B) is future research, gated by D62's warning.


## PLAY-OUT RESULT (2026-06-27) — the fidelity gap QUANTIFIED
Ran SMA|4h through the ACTUAL prod selection (firm/portfolio.allocate: inv-vol risk
parity + 2% vol-target + 15% per-name cap) over OOS-2024. ma_playout.py.

| metric | naive EW (slice study) | PROD selection (play-out) |
|--------|------------------------|----------------------------|
| linear/wk | +4.02% | **+1.20%** |
| compound (7.5mo OOS) | (high) | +33.0% |
| Sharpe | 2.12 | 1.06 |
| MaxDD | ~-35% | -36.2% |
| avg names held | n/a (1/n) | 27 |
| avg gross | ~100% | 88.6% |

THE GAP IS REAL AND LARGE: naive EW ~3.3x the prod number. WHY: in a broad bull, 27 names
signal at once; inv-vol + 15% cap spreads capital thin (~3-4%/name) and vol-targets the
book DOWN. The MA timing edge gets diversified away. The honest deployable expectation for
SMA|4h is ~+1.2%/wk (+33% over 7.5mo OOS), Sharpe ~1.0 — NOT +4%.

ACTION-LOG (how it plays out): day 1 starts with 4 names @ 15% (under-deployed, gross 60%),
ramps to 8 names @ 12% (gross 100%) by midday, then 12+ names @ ~8% — settles ~27 names.
Tearsheet: reports/MA_playout/run_*/tearsheet_SMA_4h.html (QuantStats, vs EW-u100 benchmark).

IMPLICATION: ALL prior slice-study / prod-report weekly numbers are naive-EW and ~3x
optimistic vs deployable. The DIRECTIONAL findings (4h/1d work, rolling>fixed, whipsaw helps,
30m/15m dead) STILL HOLD, but the magnitude must be quoted from the PLAY-OUT, not the EW study.
This is where instrument SELECTION (concentrate top-K by signal strength) could recover edge
— but gated by D62 (per-asset selection failed OOS). Next: play out 1d + 4h other types.
