# TOP-10 CONFIGS per MA TYPE (2020) -- the granular leaderboard beneath MA_REMEDY

User /orc 2026-06-14: "augment the above results: give me the top 10 results of every MA type." This is the
config-level view under [`MA_REMEDY.md`](MA_REMEDY.md)'s cluster summary -- the 10 best individual configs per
MA type, base FULL stack (trail10 + min_hold12 + maker), VAL Jul-Sep / OOS Oct-Dec, reusing the CORRECTED remedy
machinery (fixed-EW alignment + VAL-only vol target). Tools: `deep2020_ma_top10.py` + `_render.py`. ALL numbers
**[VERIFIED-backtest, IN-SAMPLE 2020]**. UNSEEN untouched.

> **READ THIS FIRST -- the leaderboard is OOS-Sharpe ranked (DESCRIPTIVE), and that ranking is a TRAP if read
> naively.** Ranking by held-out OOS Sharpe surfaces FINE-TF (1h) configs with HIGH +drift -- they earned a big
> OOS (the strong Nov-2020 quarter) off a MODEST VAL, i.e. **OOS-LUCKY, not robust**. The honest tell is the
> `drift` column (OOS_net - VAL_net): small |drift| = strong in BOTH VAL and OOS = trustworthy; large +drift =
> a fine-TF config that happened to fire in the OOS quarter. The DEPLOYABLE pick is NOT the #1 by Sharpe -- it
> is the small-drift config (see "MOST ROBUST per MA" below) or the VAL-selected cluster ensemble in MA_REMEDY.

## MOST ROBUST config per MA type (smallest |drift| -- strong in BOTH VAL+OOS; the honest pick)
| MA | TF | config | VAL net | OOS net | Sharpe | maxDD | |drift| |
|---|---|---|---|---|---|---|---|
| EMA | 4h | EMA(5,75,208) | 33.5 | 33.2 | 3.12 | -10.2 | 0.4 |
| SMA | 4h | SMA(12,84,148) | 28.5 | 28.6 | 3.12 | -10.1 | 0.1 |
| WMA | 4h | WMA(19,132,148) | 27.9 | 28.4 | 3.08 | -9.9 | 0.4 |
| **HMA** | 4h | **HMA(18,128)** | 38.0 | 38.1 | 2.87 | -14.1 | 0.1 |
| DEMA | 4h | DEMA(18,33) | 39.3 | 40.4 | 2.95 | -11.2 | 1.1 |
| TEMA | 1d | TEMA(15,67,233) | 30.4 | 30.9 | 3.51 | -9.3 | 0.5 |
| KAMA | 4h | KAMA(4,26) | 33.3 | 33.2 | 2.80 | -10.2 | 0.1 |
| VIDYA | 4h | VIDYA(5,75,208) | 17.4 | 18.8 | 3.35 | -4.9 | 1.4 |

**The robust configs cluster at 4h** -- confirming MA_REMEDY's 4h-is-the-sweet-spot finding at the single-config
level. These held up VAL->OOS almost exactly (drift ~0); they are the configs you'd actually trust. DEMA(18,33)
and HMA(18,128) are the standouts: ~38-40% in BOTH windows. [VERIFIED-2020-OOS]

## BEST single config per MA type x timeframe (cell champion, by OOS Sharpe)
| MA | 1d | 4h | 2h | 1h |
|---|---|---|---|---|
| EMA | EMA(12,84,148) 3.53 | EMA(2,148,208) 4.15 | EMA(2,148,208) 4.00 | EMA(6,210) 3.94 |
| SMA | SMA(15,27,48) 3.76 | SMA(2,148,208) 3.99 | SMA(186,208,233) 3.82 | SMA(10,248) 4.34 |
| WMA | WMA(22,33) 3.17 | WMA(10,248) 3.58 | WMA(2,73) 3.49 | WMA(37,124) 4.25 |
| HMA | HMA(48,67,94) 3.25 | HMA(31,75) 4.00 | HMA(73,145) 4.06 | HMA(60,132,233) 4.24 |
| DEMA | DEMA(18,33) 3.56 | DEMA(5,143) 3.39 | DEMA(26,45) 3.79 | DEMA(15,27,48) 4.25 |
| TEMA | TEMA(22,33) 3.56 | TEMA(102,237) 3.65 | TEMA(4,48,84) 4.08 | TEMA(15,67,233) 4.87 |
| KAMA | KAMA(186,208,233) 4.34 | KAMA(86,170) 3.85 | KAMA(6,65) 3.36 | KAMA(48,67,94) 4.04 |
| VIDYA | VIDYA(2,5,8) 3.29 | VIDYA(4,48,84) 3.99 | VIDYA(8,132,233) 4.37 | VIDYA(15,67,233) 4.77 |

## TOP-10 per MA type, POOLED across cadences (by OOS Sharpe; TF | config | OOSnet | Sh | maxDD | drift)
Full tables in `ma_top10_1d_4h.json` + `ma_top10_2h_1h.json`. The fine-TF dominance + high +drift is the
OOS-luck signature -- read alongside the robust table above.

- **EMA:** 4h EMA(2,148,208) 38.9/4.15/-6.9/+17 | 4h EMA(8,132,233) 35.7/4.01/-7.1/+16 | 2h EMA(2,148,208)
  43.0/4.00/-7.9/+15 | 1h EMA(6,210) 54.0/3.94/-8.8/+37 | 1h EMA(37,124) 48.5/3.94/-7.4/+31 (+5 more in json)
- **SMA:** 1h SMA(10,248) 48.7/4.34/-6.3/+33 | 1h SMA(8,108) 59.3/4.24/-7.6/+44 | 1h SMA(6,210) 53.0/4.10/-6.3/+37
  | 4h SMA(2,148,208) 33.5/3.99/-7.0/+10 | 2h SMA(186,208,233) 36.8/3.82/-8.7/+15
- **WMA:** 1h WMA(37,124) 57.3/4.25/-7.2/+47 | 1h WMA(22,128) 56.7/4.07/-7.6/+45 | 1h WMA(48,67,94)
  45.0/3.94/-7.2/+32 | 1h WMA(12,151) 52.2/3.87/-7.8/+36 | 1h WMA(18,128) 52.5/3.87/-8.1/+37
- **HMA:** 1h HMA(60,132,233) 37.3/4.24/-5.2/+34 | 1h HMA(186,208,233) 42.3/4.16/-6.0/+27 | 2h HMA(73,145)
  51.2/4.06/-7.9/+22 | 1h HMA(31,239) 52.0/4.02/-9.2/+24 | 4h HMA(31,75) 55.4/4.00/-6.8/+31
- **DEMA:** 1h DEMA(15,27,48) 49.4/4.25/-8.3/+42 | 1h DEMA(73,145) 43.6/4.18/-5.7/+22 | 1h DEMA(62,172)
  44.2/4.02/-5.3/+21 | 1h DEMA(37,124) 44.3/3.97/-6.1/+23 | 1h DEMA(31,75) 49.6/3.87/-9.6/+28
- **TEMA:** 1h TEMA(15,67,233) 45.0/4.87/-4.2/+34 | 1h TEMA(24,53,148) 40.3/4.45/-8.0/+25 | 1h TEMA(30,43,148)
  40.2/4.42/-6.2/+26 | 2h TEMA(4,48,84) 47.1/4.08/-6.7/+27 | 1h TEMA(37,124) 49.4/4.06/-10.3/+30
- **KAMA:** 1d KAMA(186,208,233) 16.2/4.34/-2.4/+16 | 1h KAMA(48,67,94) 34.5/4.04/-4.8/+6 | 4h KAMA(86,170)
  29.2/3.85/-6.9/+7 | 1h KAMA(31,75) 40.5/3.79/-6.4/+20 | 1h KAMA(15,128) 45.7/3.67/-6.9/+30
- **VIDYA:** 1h VIDYA(15,67,233) 38.4/4.77/-3.7/+22 | 1h VIDYA(8,132,233) 36.7/4.75/-3.9/+21 | 1h VIDYA(2,148,208)
  37.5/4.60/-5.7/+19 | 1h VIDYA(10,248) 32.8/4.51/-3.8/+15 | 1h VIDYA(12,84,148) 37.9/4.45/-4.5/+21

## WHAT THE LEADERBOARD TEACHES
1. **The 4h slow/vslow configs are the robust core** (drift ~0, VAL~=OOS); they are the deployable single
   configs per MA type. HMA(18,128) and DEMA(18,33) are the cleanest (~38-40% both windows). [VERIFIED-2020-OOS]
2. **The 1h configs win OOS Sharpe but are OOS-lucky** (modest VAL, big +drift) -- do NOT pick them on this
   ranking; their high Sharpe is the strong Nov quarter + finer-bar smoothing, not a robust edge.
3. **The best param region per MA type is consistent with the clusters** (slow/vslow at finer TF, mid at 1d);
   adaptive types (VIDYA/KAMA/TEMA) post the highest Sharpe but also the thinnest time-in (most selective).
4. This is the granular input the ML harness can use alongside the cluster manifest: target the ROBUST configs
   (small drift), not the OOS-Sharpe leaders.

RWYB: `python -m strat.deep2020_ma_top10 --cadences 1d,4h` then `--cadences 2h,1h`; `python -m
strat.deep2020_ma_top10_render`. json: `ma_top10_1d_4h.json` + `ma_top10_2h_1h.json`.
