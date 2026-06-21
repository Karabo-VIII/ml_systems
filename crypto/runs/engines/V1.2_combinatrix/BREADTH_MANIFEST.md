# V1.2 BREADTH MANIFEST — CORRECTED 2026-06-22 ~00:25 SAST (the 00:10 version was WRONG)

> **RETRACTION.** The original 00:10 manifest claimed the engines ran "the full ~104-symbol chimera u100 panel"
> and cited "34 distinct names" as proof breadth was met. BOTH were errors and are retracted:
> - The pre-fix `deep2020_ti_pipeline.py: SYMS` was a **HARDCODED 10-name list** (BTC ETH SOL BNB XRP DOGE ADA
>   AVAX LINK LTC). `load_ohlc(cad)` iterates THAT 10-element list; it does NOT glob `data/processed/chimera/`.
>   The "104 parquet files -> ~103 symbols" was a fabricated description of code that does not exist.
> - The "34 distinct names" came from `runs/engines/h1_2021_weekly/trades_*.csv` -- a SEPARATE H1-2021 run, NOT
>   the deep2020_ti_pipeline engine library these survivors are derived from. It was borrowed, not earned here.
>
> So the give-up gate's `breadth_u100=FAIL` was a **REAL defect**, not a measurement artifact.

## RESOLUTION (genuine, verified)
The breadth ceiling for the 2020-DEV / 2021-IRON window is **27 u100 names** -- the names from
`config/universes/{u50,u100}.yaml` that actually have >=30 window bars + val>=5 + oos>=5 at EVERY TF
{1d,4h,2h,1h,30m,15m}. (The other ~50 u100 names are post-2020 listings -- PEPE/SUI/WLD/TIA/... -- with 0 bars
in this window.) The pipeline's `SYMS` was expanded from 10 to these 27:

  BTC ETH SOL XRP BNB DOGE ZEC TRX ADA LINK AVAX FET ENJ LTC DASH UNI BCH DOT HBAR ALGO JST CRV ETC XLM CHZ ATOM ZEN

All 30 families-1-3 engines were re-run on these 27 names x 6 TFs. Each TF block now RECORDS `n_names` (=27) and
the admitted `names` list in `ti_*.json`. The consolidator (`_consolidate_library.py`) breadth-gates blocks
(`n_names >= 20`; stale 10-name blocks skipped) and records `n_names` on every engine in `engine_library.json`.
**27 >= 20 floor: breadth axis MET, mechanically provable from the artifacts.**

## Verdict status (unchanged)
The "all 30 families-1-3 engines = de-risked beta" finding is an EARNED NULL across the 27-name u100 x 6-TF x
deep2020 harness surface (now leak-free: VAL-select / OOS-report). Best engine 0.77x BH on VAL net / 0.58-0.67x
OOS; none beats long-only buy-hold on net. Breadth is met; the null is honest.
