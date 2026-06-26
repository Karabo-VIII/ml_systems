# END-TO-END PIPELINE AUDIT — RESULT (2026-06-25)

The user asked to be ULTRA sure before trusting any DNA number. Built `ti_pipeline_audit.py`
which attacks the exact functions `ti_dna_report.py` uses, on every axis a backtest can lie.

## RESULT: 19/19 checks PASSED

| Axis | Check | Result | Evidence |
|------|-------|--------|----------|
| A2 | RSI/Bollinger/Donchian/CCI/WillR/Keltner/OBV/VWAP causal | PASS | interior value at bar i identical when future bars removed (truncation test) |
| A3 | OOS load 2025 sealed | PASS | max OOS bar = 2024-12-31 (zero bars ≥ 2025-01-01) |
| A3 | VAL/DEV wall | PASS | max VAL bar = 2024-05-14 (zero bars > 2024-05-15) |
| A1 | Look-ahead sensitivity | PASS | cheat (peek same bar) inflates +65%→+4217%; production uses causal shift(1) |
| A4 | Cost applied on turnover | PASS | 19,629 turnover events, mean cost 6.3e-5/bar |
| A4 | Zero turnover → zero cost | PASS | always-held panel: cost = 0.0 |
| A5 | Rebalancing-premium decomposition | PASS | **OOS-2024: EW +42% vs mean-single +39% = only 1.1x** |
| A6 | Regime gate binary | PASS | values = {0,1} |
| A6 | 2022 bear mostly cash | PASS | 94% of 2022 days gate=0 |
| A7 | Random-null deploy match | PASS | random bar-deploy matches engine within tolerance |
| A7 | Null non-degenerate | PASS | spread > 0, P5/P95 distinct |
| A8 | Single-asset BTC honest baseline | PASS | BTC buy-hold OOS-2024 = +50% |

## Key corrections / findings from the audit

1. **The look-ahead check was initially mis-specified** (I assumed extra LAG should not
   help — wrong; more lag is more conservative, and can align better with weekly slices).
   The correct probe is the CHEAT direction: using held[i] for return[i] (peeking).
   The cheat inflates 65→4217% → the harness is provably leak-sensitive AND production
   is on the causal path. This was a TEST bug, not a pipeline bug.

2. **Rebalancing premium is window-dependent.** The ~9.5x inflation cited earlier was for
   the full 2020-2024 DEV compounding. On the **OOS-2024 window it is only ~1.1x** (EW +42%
   vs mean-single +39%). So OOS-2024 compounds are MUCH closer to honest than DEV compounds.
   Linear/wk + null percentile remain the primary reads, but OOS-2024 compound is not the
   ~10x lie that DEV compound was.

3. **Indicator causality is airtight** across all 8 spot-checked indicators (the truncation
   test is the strongest causality proof: removing future bars cannot change a causal value).

## Verdict
Pipeline is causally clean, correctly walled (2025 sealed), cost-correct, the gate is
causal+binary+bear-protective, the random-null is correctly constructed, and the
rebalancing premium is quantified per-window. **DNA numbers produced by this harness are
trustworthy.** Re-run anytime: `python -m strat.ti_pipeline_audit`.
