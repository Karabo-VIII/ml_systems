# 2021 PER-CONFIG 2MA/3MA LEADERBOARD + the WORKING BAND (6mo TRAIN / 3mo VAL / 3mo OOS)

STRICT LONG-ONLY + spot (held in {0,1}, no short/inverse anywhere). 2021 BAND ONLY. Causal/lag-1, maker cost. Ironed sleeve = MA-cross -> 10% trail -> min_hold(12). 6/3/3 split: TRAIN ('2021-01-01', '2021-07-01') / VAL ('2021-07-01', '2021-10-01') / OOS ('2021-10-01', '2022-01-01').

## HONEST FRAMING -- the BAND is the deliverable, NOT the exact #1

The prior investigation (D62 + the per-asset null) found per-config RANK is NOISE that does not transfer across regimes. So:

- **TRUST THE BAND** = the set of configs POSITIVE across TRAIN AND VAL AND OOS. This is the robust set the FAMILY ENSEMBLE actually rides. Reported per cell as a (fast, slow) parameter RANGE.
- **DO NOT TRUST the exact ordering** within a cell. The within-band #1 is regime-transient. The rank-stability number (Spearman (TRAIN+VAL) net vs OOS net + top-10 overlap) quantifies how little the ordering transfers. Low rho / small overlap = the ranking is noise.
- PRIMARY SORT below = FULL-2020 net (wealth over the most data = the most stable estimate). Per-split net/Sharpe/maxDD shown so you can see whether a high FULL rank actually TRANSFERS.

DATA CAVEATS: 2h is SYNTHESIZED from 1h (OHLC-resample). FIXED-EW (unlisted/missing bar = CASH, cadence-invariant -- NOT skipna). The OOS regime VARIES by year (check the per-TF buy-hold OOS net below): 2021 OOS = Oct-Dec. In a bull-OOS, under-participation vs buy-hold is EXPECTED (not a defect); in a down/flat-OOS, a de-risked book can 'beat' buy-hold by holding cash (EXPOSURE, not alpha) -- read the OOS buy-hold net + the config time-in before crediting any 'beat'.

Repro: `python -m strat.ma_2020_config_leaderboard --tfs 1d,4h,2h,1h,30m,15m`  git_sha=f92f5f5  cost=maker(0.0006)  trail=0.1  min_hold=12  split={'TRAIN': ('2021-01-01', '2021-07-01'), 'VAL': ('2021-07-01', '2021-10-01'), 'OOS': ('2021-10-01', '2022-01-01'), 'FULL': ('2021-01-01', '2022-01-01')}

All numbers are [MEASURED] from the run below (equal-weight u10, causal/lag-1, maker).


# Timeframe: 1d
_Benchmark (equal-weight u10 buy-hold, no cost): FULL-2020 net = 1418.7% (participation-tax reference)._

## 1d x EMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=-0.192**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,44], slow in [3,203] -> 27/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,30], slow in [22,233] -> 27/60 configs positive across TRAIN & VAL & OOS
- band members: EMA(2,3), EMA(2,5,118), EMA(3,6), EMA(4,5), EMA(3,8,43), EMA(2,19,22), EMA(3,5,186), EMA(3,19,43), EMA(6,22,24), EMA(5,24,34), EMA(8,39), EMA(12,14,75), EMA(12,33), EMA(15,28), EMA(2,30,84), EMA(18,23), EMA(3,30,166), EMA(10,55), EMA(2,73), EMA(22,28), EMA(3,102), EMA(5,86), EMA(6,77), EMA(4,30,67) (+30 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `EMA(2,3)` | 2MA |   382.0   3.69   -56.2 |    60.5   2.95   -30.4 |     0.5    0.3   -29.4 |   677.4   2.87   -56.2 | YES |
| 2 | `EMA(2,3,4)` | 3MA |   362.7   3.79   -54.4 |    58.5   3.01   -27.7 |    -0.7   0.22   -26.1 |   628.7   2.91   -54.4 | - |
| 3 | `EMA(2,5,118)` | 3MA |   405.8   4.16   -41.2 |    31.0   2.12   -28.1 |     3.8   0.57   -18.4 |   588.1   3.03   -41.2 | YES |
| 4 | `EMA(2,4,19)` | 3MA |   375.7   4.16   -41.5 |    47.8   2.84   -24.4 |    -7.3  -0.48   -24.8 |   552.1   3.01   -41.8 | - |
| 5 | `EMA(3,4,38)` | 3MA |   346.6   4.14   -35.5 |    45.7   2.78   -25.7 |    -0.2   0.17   -21.1 |   549.2    3.1   -35.5 | - |
| 6 | `EMA(2,10)` | 2MA |   316.7    4.3   -34.3 |    48.8   2.94   -22.4 |    -1.2    0.1   -18.8 |   512.9   3.17   -38.7 | - |
| 7 | `EMA(3,6)` | 2MA |   303.3   4.19   -38.1 |    47.4   2.92   -22.9 |     1.9   0.39   -21.3 |   505.6   3.12   -38.8 | YES |
| 8 | `EMA(4,5)` | 2MA |   286.7    4.1   -38.1 |    46.5   2.88   -22.7 |     1.9   0.39   -21.3 |   477.4   3.07   -38.8 | YES |
| 9 | `EMA(2,8,22)` | 3MA |   335.5    4.8   -26.0 |    33.0   2.39   -22.2 |    -2.5  -0.13   -19.3 |   464.7   3.32   -27.1 | - |
| 10 | `EMA(3,8,43)` | 3MA |   309.2   4.71   -23.4 |    32.1   2.48   -20.5 |     4.2   0.66   -14.6 |   463.0   3.42   -23.4 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `EMA(86,122)` | 2MA |    12.3   2.87    -1.8 |    -4.2  -1.29    -4.8 |     2.0   1.35    -1.8 |     9.8   1.07    -6.0 | - |
| 119 | `EMA(44,203)` | 2MA |     6.3   1.59    -3.2 |     1.9   0.56    -5.7 |     0.1   0.13    -1.4 |     8.5   0.88    -5.7 | YES |
| 120 | `EMA(62,105)` | 2MA |     8.5   1.93    -3.6 |    -0.9  -0.03    -7.8 |    -0.8  -0.72    -2.0 |     6.7   0.56    -8.3 | - |


## 1d x SMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.239**; TRAIN+VAL top-10 -> OOS top-10 overlap = **2/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,86], slow in [10,210] -> 33/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,75], slow in [14,233] -> 43/60 configs positive across TRAIN & VAL & OOS
- band members: SMA(4,5,75), SMA(8,9,75), SMA(2,10), SMA(5,12), SMA(4,8,17), SMA(10,11,233), SMA(12,14,75), SMA(5,10,14), SMA(3,8,43), SMA(8,14,38), SMA(12,13), SMA(5,12,208), SMA(4,10,208), SMA(6,14,15), SMA(6,13), SMA(3,12,27), SMA(2,5,118), SMA(3,5,186), SMA(6,22,24), SMA(2,19,22), SMA(26,32), SMA(37,38), SMA(5,24,34), SMA(3,19,43) (+52 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `SMA(4,5,75)` | 3MA |   620.1   4.23   -35.1 |    54.6   3.22   -21.8 |     9.2   1.09   -19.0 |  1116.2   3.38   -35.1 | YES |
| 2 | `SMA(8,9,75)` | 3MA |   630.5   4.51   -34.5 |    39.3    2.6   -23.9 |    11.2   1.45   -11.2 |  1030.8   3.48   -34.5 | YES |
| 3 | `SMA(2,10)` | 2MA |   451.1   4.07   -42.6 |    52.4   3.06   -24.3 |    13.8   1.39   -15.4 |   855.9    3.3   -42.6 | YES |
| 4 | `SMA(4,5)` | 2MA |   515.9   3.71   -49.2 |    60.8   3.06   -21.9 |    -4.2  -0.01   -33.9 |   848.8   2.85   -50.9 | - |
| 5 | `SMA(2,3)` | 2MA |   554.1   3.76   -48.2 |    36.6    2.1   -24.7 |    -4.4   0.03   -34.9 |   753.9   2.68   -53.4 | - |
| 6 | `SMA(5,12)` | 2MA |   386.6   4.09   -31.2 |    53.7   3.17   -17.8 |    11.1    1.3   -13.5 |   730.5   3.33   -33.1 | YES |
| 7 | `SMA(4,8,17)` | 3MA |   395.6   4.99   -25.7 |    62.2   4.01   -12.1 |     3.1   0.53   -17.6 |   728.5   3.92   -32.9 | YES |
| 8 | `SMA(5,6,34)` | 3MA |   435.2   3.75   -38.0 |    58.8   3.73   -17.9 |    -3.3  -0.12   -25.4 |   721.8   2.99   -38.0 | - |
| 9 | `SMA(2,3,4)` | 3MA |   446.4   3.52   -55.5 |    56.0    2.9   -24.6 |    -4.0  -0.04   -28.8 |   718.7   2.72   -55.5 | - |
| 10 | `SMA(10,11,233)` | 3MA |   522.6   4.25   -30.3 |    22.2    1.9   -15.2 |     4.0   0.59   -18.8 |   691.1   3.08   -30.3 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `SMA(31,124)` | 2MA |     3.5   0.81    -3.8 |     3.1   0.57   -11.8 |    -1.1  -1.71    -1.1 |     5.5   0.42   -12.8 | - |
| 119 | `SMA(52,89)` | 2MA |     2.7   0.52    -8.3 |     6.3   0.89   -12.7 |    -5.3  -1.96    -7.8 |     3.4   0.27   -17.3 | - |
| 120 | `SMA(186,208,233)` | 3MA |     5.7   1.64    -1.4 |    -1.6  -1.76    -2.3 |    -2.6  -1.72    -4.7 |     1.3   0.25    -4.7 | - |


## 1d x WMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.024**; TRAIN+VAL top-10 -> OOS top-10 overlap = **1/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [12,237] -> 31/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,75], slow in [14,233] -> 39/60 configs positive across TRAIN & VAL & OOS
- band members: WMA(4,5,75), WMA(2,4,19), WMA(8,14,38), WMA(2,5,118), WMA(8,9,75), WMA(6,13), WMA(2,12,14), WMA(5,12), WMA(5,10,14), WMA(3,12,27), WMA(10,11,233), WMA(4,15), WMA(4,8,17), WMA(12,14,75), WMA(5,12,208), WMA(3,5,186), WMA(4,10,208), WMA(3,19,43), WMA(8,16), WMA(12,13), WMA(4,15,208), WMA(5,24,34), WMA(15,22,48), WMA(8,22,60) (+46 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `WMA(4,5,75)` | 3MA |   562.4   4.28   -34.0 |    45.0   2.73   -24.6 |     3.2   0.52   -16.9 |   891.1   3.26   -34.0 | YES |
| 2 | `WMA(2,3,4)` | 3MA |   493.9   3.72   -51.3 |    56.5   2.89   -26.1 |    -0.2   0.27   -30.0 |   827.2   2.87   -51.3 | - |
| 3 | `WMA(2,4,19)` | 3MA |   429.4   4.52   -37.7 |    62.8   3.49   -19.8 |     2.1    0.4   -19.6 |   779.9    3.5   -39.1 | YES |
| 4 | `WMA(8,14,38)` | 3MA |   449.9    4.6   -22.8 |    48.7    3.4   -14.2 |     5.4    0.9   -11.7 |   761.5   3.64   -26.2 | YES |
| 5 | `WMA(2,5,118)` | 3MA |   483.9   3.79   -43.7 |    36.8   2.35   -26.2 |     7.3    0.9   -18.7 |   756.9   2.92   -43.7 | YES |
| 6 | `WMA(8,9,75)` | 3MA |   421.5   4.07   -29.2 |    44.4   3.03   -19.9 |    13.6   1.89    -6.9 |   755.2   3.34   -29.2 | YES |
| 7 | `WMA(3,6)` | 2MA |   466.5   3.65   -52.6 |    59.4   3.14   -24.1 |    -6.1  -0.17   -30.7 |   748.0    2.8   -52.6 | - |
| 8 | `WMA(4,5)` | 2MA |   465.8   3.64   -52.0 |    56.6   2.97   -24.3 |    -5.9  -0.17   -31.2 |   734.4   2.77   -52.0 | - |
| 9 | `WMA(6,13)` | 2MA |   384.3   4.14   -29.7 |    49.3   3.06   -18.1 |    10.5   1.24   -13.9 |   699.2   3.32   -30.8 | YES |
| 10 | `WMA(2,12,14)` | 3MA |   351.2   4.54   -33.6 |    64.4   3.99   -15.2 |     6.3   0.82   -16.6 |   688.2   3.68   -33.9 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `WMA(52,203)` | 2MA |     8.1   1.64    -3.6 |     2.1   0.45    -9.4 |    -2.1  -1.96    -3.1 |     8.0    0.6   -11.2 | - |
| 119 | `WMA(102,103)` | 2MA |     9.5   2.04    -3.6 |    -1.8  -0.12   -11.4 |     0.0    0.0     0.0 |     7.5   0.55   -11.4 | - |
| 120 | `WMA(73,145)` | 2MA |     6.0    1.3    -3.6 |     0.1   0.14    -9.7 |    -1.8  -2.62    -2.0 |     4.1   0.36   -11.5 | - |


## 1d x HMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.318**; TRAIN+VAL top-10 -> OOS top-10 overlap = **2/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,86], slow in [3,239] -> 51/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,60], slow in [4,233] -> 46/60 configs positive across TRAIN & VAL & OOS
- band members: HMA(2,12,14), HMA(4,8,17), HMA(2,4,19), HMA(5,10,14), HMA(3,5,186), HMA(6,9,24), HMA(2,8,22), HMA(4,10,208), HMA(3,12,27), HMA(10,19), HMA(10,17,233), HMA(2,26), HMA(2,19,22), HMA(12,22,118), HMA(5,12,208), HMA(3,44), HMA(8,16), HMA(3,6), HMA(5,6,34), HMA(6,13), HMA(3,19,43), HMA(12,14,75), HMA(2,3), HMA(4,15,208) (+73 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `HMA(2,12,14)` | 3MA |   703.8   4.36   -46.9 |    66.5   3.88   -17.7 |     5.7   0.68   -28.4 |  1314.9   3.47   -46.9 | YES |
| 2 | `HMA(4,8,17)` | 3MA |   679.4   4.49   -37.8 |    67.2   3.59   -19.5 |     6.4   0.74   -24.7 |  1286.7   3.53   -37.8 | YES |
| 3 | `HMA(2,4,19)` | 3MA |   743.1   4.54   -38.7 |    52.4   2.92   -23.4 |     7.8   0.84   -22.7 |  1285.4   3.46   -38.7 | YES |
| 4 | `HMA(5,10,14)` | 3MA |   531.5   4.03   -44.1 |    70.8    3.8   -17.9 |     4.2   0.58   -24.0 |  1023.9   3.26   -44.1 | YES |
| 5 | `HMA(3,5,186)` | 3MA |   550.1   4.32   -19.4 |    48.0   2.87   -22.0 |    13.0   1.68    -8.7 |   987.2   3.45   -22.0 | YES |
| 6 | `HMA(8,9,75)` | 3MA |   546.7    4.5   -32.0 |    67.0   3.85   -17.7 |    -0.1   0.19   -14.6 |   978.9   3.54   -38.9 | - |
| 7 | `HMA(6,9,24)` | 3MA |   478.8   4.79   -40.6 |    45.4   2.88   -21.9 |    18.5   1.65   -17.9 |   897.4    3.7   -40.6 | YES |
| 8 | `HMA(2,8,22)` | 3MA |   469.3   4.55   -39.1 |    58.9   3.32   -23.1 |     9.0   0.94   -23.9 |   886.3   3.53   -39.1 | YES |
| 9 | `HMA(4,10,208)` | 3MA |   477.8   4.93   -28.4 |    54.4   3.44   -18.6 |     7.7   1.08   -11.6 |   861.0   3.86   -28.4 | YES |
| 10 | `HMA(3,12,27)` | 3MA |   440.5   4.98   -30.3 |    60.1    3.8   -15.5 |    11.0   1.11   -18.8 |   860.2   3.88   -31.1 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `HMA(12,13)` | 2MA |    39.3   1.27   -43.3 |    -7.6  -0.41   -20.6 |    -0.6   0.07   -13.9 |    27.9   0.72   -49.3 | - |
| 119 | `HMA(186,208,233)` | 3MA |    19.9   2.01    -7.1 |    -4.0  -0.63   -10.1 |     1.4   0.91    -1.4 |    16.6   0.96   -13.0 | - |
| 120 | `HMA(102,103)` | 2MA |     1.5   0.34    -7.4 |    11.8   2.78    -5.4 |    -5.5  -2.87    -7.1 |     7.2   0.66    -9.6 | - |


## 1d x DEMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.367**; TRAIN+VAL top-10 -> OOS top-10 overlap = **1/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,73], slow in [3,239] -> 48/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,60], slow in [4,233] -> 51/60 configs positive across TRAIN & VAL & OOS
- band members: DEMA(3,12,27), DEMA(2,12,14), DEMA(2,19,22), DEMA(5,6,34), DEMA(6,13), DEMA(3,18), DEMA(2,5,118), DEMA(2,8,22), DEMA(2,10), DEMA(2,26), DEMA(5,12), DEMA(6,9,24), DEMA(4,10,208), DEMA(3,6), DEMA(2,3), DEMA(3,19,43), DEMA(8,9,75), DEMA(8,14,38), DEMA(5,12,208), DEMA(4,15), DEMA(5,24,34), DEMA(4,8,17), DEMA(2,3,4), DEMA(4,5) (+75 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `DEMA(3,12,27)` | 3MA |   369.4   5.26   -20.4 |    68.7   4.36   -14.7 |    10.9   1.31   -11.4 |   778.4   4.24   -27.5 | YES |
| 2 | `DEMA(2,12,14)` | 3MA |   352.9    4.5   -36.2 |    69.5   4.37   -14.1 |    13.0   1.53   -12.3 |   767.1   3.85   -41.3 | YES |
| 3 | `DEMA(2,19,22)` | 3MA |   336.5   4.95   -28.1 |    73.6   4.53   -14.4 |     6.8   0.95   -10.3 |   709.1   4.08   -35.4 | YES |
| 4 | `DEMA(5,6,34)` | 3MA |   353.9   4.71   -32.5 |    68.4   4.34   -11.6 |     4.0    0.6   -15.6 |   694.8   3.82   -35.9 | YES |
| 5 | `DEMA(6,13)` | 2MA |   346.9   4.47   -38.8 |    57.2   3.52   -19.7 |     9.9   1.03   -21.6 |   671.8   3.52   -38.8 | YES |
| 6 | `DEMA(3,18)` | 2MA |   328.3   4.02   -48.0 |    69.3   3.95   -17.4 |     6.2   0.74   -21.8 |   669.5   3.32   -48.0 | YES |
| 7 | `DEMA(2,5,118)` | 3MA |   403.5   4.43   -37.0 |    43.6   2.95   -20.6 |     6.2   0.83   -15.7 |   667.8   3.41   -37.0 | YES |
| 8 | `DEMA(2,8,22)` | 3MA |   300.1   4.06   -38.5 |    73.4    4.6   -12.0 |     9.0   1.11   -10.4 |   656.4   3.56   -42.7 | YES |
| 9 | `DEMA(2,10)` | 2MA |   363.3    3.7   -50.8 |    56.2    3.0   -25.6 |     3.5   0.52   -28.5 |   648.9    2.9   -51.7 | YES |
| 10 | `DEMA(2,26)` | 2MA |   329.9   4.22   -42.8 |    56.6   3.31   -15.7 |    10.7   1.21   -14.1 |   645.2   3.41   -47.1 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `DEMA(102,103)` | 2MA |     1.9   0.36    -7.5 |     2.7   0.61    -8.8 |    -0.5  -0.21    -3.5 |     4.1   0.35   -10.8 | - |
| 119 | `DEMA(86,122)` | 2MA |     1.9   0.36    -7.5 |     1.0   0.29   -10.4 |    -1.0  -0.43    -3.5 |     2.0    0.2   -12.8 | - |
| 120 | `DEMA(75,132,148)` | 3MA |     4.5   0.75    -5.7 |     1.6   0.51    -5.9 |    -4.5  -1.83    -6.6 |     1.5   0.18    -9.8 | - |


## 1d x TEMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.194**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 48/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,75], slow in [4,233] -> 56/60 configs positive across TRAIN & VAL & OOS
- band members: TEMA(2,5,118), TEMA(8,9,75), TEMA(4,8,17), TEMA(5,10,14), TEMA(4,15,208), TEMA(5,12,208), TEMA(2,19,22), TEMA(2,8,22), TEMA(10,11,233), TEMA(3,5,186), TEMA(3,19,43), TEMA(2,4,19), TEMA(3,12,27), TEMA(5,6,34), TEMA(2,26), TEMA(2,12,14), TEMA(8,22,60), TEMA(8,14,38), TEMA(12,13), TEMA(3,8,43), TEMA(5,24,34), TEMA(6,9,24), TEMA(6,22,24), TEMA(3,4,38) (+80 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `TEMA(6,13)` | 2MA |   697.8    4.6   -38.2 |    53.9   3.23   -25.3 |    -4.4  -0.09   -27.6 |  1074.3   3.39   -38.2 | - |
| 2 | `TEMA(2,5,118)` | 3MA |   564.5   4.47   -26.6 |    63.3   3.78   -14.6 |     4.2   0.66   -13.9 |  1031.0   3.59   -26.6 | YES |
| 3 | `TEMA(8,9,75)` | 3MA |   496.2   4.61   -14.0 |    70.9   4.58   -11.6 |     1.2   0.31   -13.7 |   931.3   3.78   -21.0 | YES |
| 4 | `TEMA(4,8,17)` | 3MA |   490.5   4.85   -36.9 |    52.5   3.17   -23.0 |     9.2   0.96   -24.1 |   884.0   3.66   -36.9 | YES |
| 5 | `TEMA(4,5,75)` | 3MA |   446.4    5.3   -26.4 |    72.6   4.23   -18.4 |     0.0   0.17   -15.9 |   843.6    4.1   -38.0 | - |
| 6 | `TEMA(5,10,14)` | 3MA |   434.9   4.71   -35.4 |    53.4   3.17   -25.0 |     8.9   0.96   -22.6 |   793.7   3.59   -35.4 | YES |
| 7 | `TEMA(4,15,208)` | 3MA |   485.6   4.28   -24.9 |    39.4   3.22   -15.8 |     8.7   1.52    -5.8 |   787.0   3.43   -24.9 | YES |
| 8 | `TEMA(4,15)` | 2MA |   448.2   3.83   -41.4 |    58.8   3.47   -24.4 |    -0.7   0.22   -27.0 |   764.8   3.01   -41.4 | - |
| 9 | `TEMA(5,12,208)` | 3MA |   488.2   4.31   -20.9 |    38.8   3.23   -13.6 |     5.4   0.93   -10.1 |   760.6    3.4   -21.0 | YES |
| 10 | `TEMA(2,19,22)` | 3MA |   347.4    4.6   -33.1 |    65.9   4.98   -11.8 |    13.7    1.5   -13.2 |   744.0   3.96   -34.6 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `TEMA(30,132,186)` | 3MA |    21.5   1.62    -8.1 |     3.7   0.89    -7.1 |     7.5   2.17    -4.0 |    35.4    1.5    -8.1 | YES |
| 119 | `TEMA(186,208,233)` | 3MA |     2.0    0.4    -5.2 |    12.4   4.17    -2.6 |    -3.5  -1.74    -6.4 |    10.7   0.99    -6.4 | - |
| 120 | `TEMA(102,237)` | 2MA |     2.0   0.38    -6.3 |    -3.2   -0.6    -8.5 |     2.7   1.22    -3.1 |     1.5   0.18   -10.1 | - |


## 1d x KAMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.184**; TRAIN+VAL top-10 -> OOS top-10 overlap = **2/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,86], slow in [3,169] -> 29/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,75], slow in [4,233] -> 42/60 configs positive across TRAIN & VAL & OOS
- band members: KAMA(4,5), KAMA(3,6), KAMA(2,3), KAMA(2,3,4), KAMA(4,5,75), KAMA(5,6,34), KAMA(2,12,14), KAMA(2,8,22), KAMA(2,10), KAMA(2,4,19), KAMA(5,10,14), KAMA(3,8,43), KAMA(12,13), KAMA(3,5,186), KAMA(4,8,17), KAMA(3,12,27), KAMA(5,24,34), KAMA(4,30,67), KAMA(2,5,118), KAMA(8,9,75), KAMA(26,32), KAMA(12,14,75), KAMA(6,9,24), KAMA(3,19,43) (+47 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `KAMA(4,5)` | 2MA |   581.1   4.42   -21.0 |    77.0   3.93   -18.6 |     8.2   0.98   -16.5 |  1204.4   3.64   -21.0 | YES |
| 2 | `KAMA(3,6)` | 2MA |   449.6   4.49   -21.1 |    80.6   4.59   -13.2 |     0.8   0.26   -15.0 |   900.1   3.72   -23.9 | YES |
| 3 | `KAMA(2,3)` | 2MA |   466.1   4.58   -30.9 |    37.9   2.37   -20.7 |     8.5   0.95   -19.4 |   746.8   3.37   -30.9 | YES |
| 4 | `KAMA(2,3,4)` | 3MA |   423.3   5.12   -27.6 |    51.9   3.41   -12.7 |     5.9    0.8   -14.8 |   741.8   3.88   -33.1 | YES |
| 5 | `KAMA(4,5,75)` | 3MA |   428.6   4.29   -17.0 |    36.6   3.04   -13.5 |     5.9   0.99   -10.6 |   664.8   3.37   -17.0 | YES |
| 6 | `KAMA(5,6,34)` | 3MA |   359.5   3.97   -31.6 |    57.6   4.46   -10.2 |     3.8   0.66   -12.2 |   651.5   3.35   -31.6 | YES |
| 7 | `KAMA(2,12,14)` | 3MA |   339.5   5.21   -16.5 |    44.5   3.85   -12.4 |     3.3    0.6   -11.8 |   556.1   4.02   -16.5 | YES |
| 8 | `KAMA(2,8,22)` | 3MA |   342.2    5.4   -16.1 |    42.6   3.72   -12.2 |     1.2   0.31   -15.4 |   538.3   4.04   -16.9 | YES |
| 9 | `KAMA(2,10)` | 2MA |   252.3    5.4   -16.4 |    63.7   4.33   -12.3 |     3.6   0.61   -11.4 |   497.5   4.18   -19.3 | YES |
| 10 | `KAMA(2,4,19)` | 3MA |   277.9   4.27   -28.7 |    55.9   3.64   -17.9 |     0.4   0.21   -15.0 |   491.6   3.39   -28.7 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `KAMA(52,203)` | 2MA |     8.6   0.91    -9.2 |     2.4   0.63    -4.6 |    -7.5  -2.94    -7.8 |     2.8   0.24   -14.4 | - |
| 119 | `KAMA(73,145)` | 2MA |    -4.9   -0.4   -14.5 |     0.7   0.28    -4.3 |     3.9   1.49    -5.3 |    -0.5   0.05   -14.5 | - |
| 120 | `KAMA(62,239)` | 2MA |    -2.2  -0.27    -9.7 |     0.7   0.27    -4.9 |     0.1   0.15    -1.1 |    -1.4  -0.06   -10.3 | - |


## 1d x VIDYA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.178**; TRAIN+VAL top-10 -> OOS top-10 overlap = **1/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,44], slow in [10,86] -> 21/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,38], slow in [14,233] -> 34/60 configs positive across TRAIN & VAL & OOS
- band members: VIDYA(2,4,19), VIDYA(2,10), VIDYA(4,8,17), VIDYA(5,6,34), VIDYA(2,12,14), VIDYA(2,8,22), VIDYA(3,18), VIDYA(6,13), VIDYA(5,10,14), VIDYA(3,8,43), VIDYA(4,15), VIDYA(5,12), VIDYA(10,11,233), VIDYA(4,10,208), VIDYA(5,12,208), VIDYA(4,15,208), VIDYA(8,16), VIDYA(12,13), VIDYA(6,14,15), VIDYA(6,9,24), VIDYA(3,44), VIDYA(2,60,208), VIDYA(8,9,75), VIDYA(2,30,84) (+31 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `VIDYA(2,3)` | 2MA |   122.6   4.02   -16.6 |    60.0   4.13   -11.6 |    -6.4  -0.71   -17.4 |   233.3    3.1   -22.0 | - |
| 2 | `VIDYA(2,3,4)` | 3MA |   128.4   4.14   -11.3 |    52.2   3.78   -11.6 |    -6.0  -0.69   -16.4 |   226.9   3.09   -19.0 | - |
| 3 | `VIDYA(2,4,19)` | 3MA |    85.0   3.47   -11.4 |    49.9   4.45    -7.4 |     2.6   0.56    -8.3 |   184.6   3.18   -11.4 | YES |
| 4 | `VIDYA(2,10)` | 2MA |    68.9    3.3   -10.2 |    57.7   4.65    -6.0 |     2.7   0.53    -8.9 |   173.5   3.12   -13.3 | YES |
| 5 | `VIDYA(4,5)` | 2MA |    62.9   2.88   -11.5 |    55.8   4.42    -6.6 |    -6.5  -0.92   -16.2 |   137.2    2.6   -16.2 | - |
| 6 | `VIDYA(3,6)` | 2MA |    63.8   2.92    -9.9 |    54.2    4.3    -7.9 |    -8.3  -1.23   -17.8 |   131.5   2.53   -17.8 | - |
| 7 | `VIDYA(2,5,118)` | 3MA |    68.6   3.07    -9.9 |    42.3   3.82    -8.4 |    -3.9  -0.55   -14.0 |   130.4   2.59   -14.0 | - |
| 8 | `VIDYA(4,8,17)` | 3MA |    52.1   2.92   -10.8 |    42.1   4.26    -4.9 |     5.7   1.07    -8.4 |   128.4   2.91   -11.5 | YES |
| 9 | `VIDYA(3,4,38)` | 3MA |    68.6   3.11    -9.9 |    34.9   3.68    -8.4 |    -1.4   -0.1   -11.7 |   124.2   2.61   -11.7 | - |
| 10 | `VIDYA(5,6,34)` | 3MA |    50.9   2.76   -10.4 |    35.7   4.03    -5.9 |     6.5   1.19    -8.4 |   118.1   2.76   -10.9 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `VIDYA(26,172)` | 2MA |     9.8   2.46    -1.4 |     0.0    0.0     0.0 |     0.0    0.0     0.0 |     9.8   1.73    -1.4 | - |
| 119 | `VIDYA(186,208,233)` | 3MA |     6.8   1.67    -2.5 |     0.0    0.0     0.0 |     0.0    0.0     0.0 |     6.8   1.17    -2.5 | - |
| 120 | `VIDYA(102,237)` | 2MA |     3.0   0.87    -2.5 |     0.0    0.0     0.0 |     0.0    0.0     0.0 |     3.0   0.61    -2.5 | - |


# Timeframe: 4h
_Benchmark (equal-weight u10 buy-hold, no cost): FULL-2020 net = 1201.3% (participation-tax reference)._

## 4h x EMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=-0.176**; TRAIN+VAL top-10 -> OOS top-10 overlap = **1/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [5,237] -> 23/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [4,233] -> 29/60 configs positive across TRAIN & VAL & OOS
- band members: EMA(2,3,4), EMA(2,5,118), EMA(6,14,15), EMA(4,5,75), EMA(2,8,22), EMA(8,14,38), EMA(6,22,24), EMA(12,14,75), EMA(3,5,186), EMA(8,9,75), EMA(8,22,60), EMA(2,73), EMA(19,27,38), EMA(15,22,48), EMA(10,19), EMA(3,6), EMA(3,75,132), EMA(2,60,208), EMA(5,86), EMA(2,10), EMA(22,28), EMA(26,32), EMA(3,102), EMA(4,86) (+28 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `EMA(2,3)` | 2MA |   441.9   3.88   -44.6 |    49.3   2.64   -23.9 |    -1.1   0.23   -32.9 |   700.3   2.87   -48.8 | - |
| 2 | `EMA(2,3,4)` | 3MA |   402.1   3.84   -46.6 |    28.5   1.84   -22.4 |     4.2   0.58   -31.8 |   572.9   2.75   -51.5 | YES |
| 3 | `EMA(2,5,118)` | 3MA |   269.7   3.93   -31.0 |    48.8   3.49   -10.1 |    13.1   1.63   -10.1 |   522.2   3.35   -35.6 | YES |
| 4 | `EMA(2,4,19)` | 3MA |   378.3   4.42   -34.0 |    42.1   2.77   -14.2 |   -13.7  -1.05   -33.6 |   486.8   3.01   -40.1 | - |
| 5 | `EMA(6,14,15)` | 3MA |   310.0   4.52   -29.7 |    29.3   2.19   -13.7 |     2.9   0.49   -22.1 |   445.7   3.19   -32.5 | YES |
| 6 | `EMA(4,5,75)` | 3MA |   257.6    3.9   -31.5 |    39.8   2.89   -11.4 |     7.3   0.99   -14.4 |   436.2   3.12   -34.0 | YES |
| 7 | `EMA(3,4,38)` | 3MA |   291.5   4.03   -33.5 |    40.4   2.82   -15.3 |    -4.2  -0.25   -26.1 |   426.7   2.98   -36.9 | - |
| 8 | `EMA(2,8,22)` | 3MA |   265.9   3.84   -32.5 |    35.7   2.51   -15.4 |     6.0    0.8   -18.6 |   426.4   2.96   -35.6 | YES |
| 9 | `EMA(3,19,43)` | 3MA |   278.1   4.48   -29.8 |    33.0   2.51   -11.7 |    -1.4   0.03   -18.2 |   395.9   3.19   -32.3 | - |
| 10 | `EMA(6,9,24)` | 3MA |   290.2   4.28   -31.2 |    26.4   2.07   -12.9 |    -1.8   0.02   -23.1 |   384.6   2.97   -35.8 | - |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `EMA(62,239)` | 2MA |    -0.4  -0.04    -5.4 |    30.9   3.73    -7.2 |     1.8   0.58    -4.0 |    32.6   1.66    -7.2 | - |
| 119 | `EMA(102,237)` | 2MA |     3.4   0.89    -4.4 |    24.7   3.38    -7.5 |     0.6   0.26    -4.1 |    29.7   1.71    -8.5 | YES |
| 120 | `EMA(37,203)` | 2MA |     2.8   0.45    -6.9 |    30.9   3.41    -7.5 |    -4.9   -1.0    -9.0 |    28.0   1.25   -12.0 | - |


## 4h x SMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=-0.056**; TRAIN+VAL top-10 -> OOS top-10 overlap = **2/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [5,239] -> 26/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,75], slow in [34,233] -> 25/60 configs positive across TRAIN & VAL & OOS
- band members: SMA(4,5,75), SMA(2,5,118), SMA(5,6,34), SMA(8,9,75), SMA(12,14,75), SMA(4,5), SMA(37,38), SMA(4,30,67), SMA(10,11,233), SMA(4,15), SMA(2,73), SMA(4,10,208), SMA(12,77), SMA(8,39), SMA(26,32), SMA(10,55), SMA(2,30,84), SMA(6,67,233), SMA(5,24,34), SMA(8,14,38), SMA(30,43,186), SMA(6,77), SMA(10,34,60), SMA(15,65) (+27 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `SMA(2,3)` | 2MA |   616.8   4.11   -49.3 |    57.0    2.8   -28.3 |    -2.2   0.19   -34.4 |  1000.5   3.05   -52.9 | - |
| 2 | `SMA(4,5,75)` | 3MA |   451.9   5.01   -31.7 |    59.4   3.78    -8.8 |     8.5   1.08   -14.5 |   854.7   3.98   -32.6 | YES |
| 3 | `SMA(2,5,118)` | 3MA |   393.0   4.67   -29.6 |    73.3    4.8   -11.1 |     0.8   0.27   -15.0 |   761.2   3.85   -35.9 | YES |
| 4 | `SMA(3,5,186)` | 3MA |   379.6   4.47   -30.7 |    62.0   4.16   -10.3 |    -2.7  -0.13   -19.3 |   656.2   3.55   -31.4 | - |
| 5 | `SMA(3,4,38)` | 3MA |   434.0   4.75   -36.2 |    42.5   2.83   -16.7 |    -6.1   -0.4   -24.2 |   614.6   3.37   -36.8 | - |
| 6 | `SMA(5,6,34)` | 3MA |   375.2   4.46   -39.5 |    28.3   2.07   -20.4 |     8.1   0.98   -25.0 |   559.1   3.26   -40.0 | YES |
| 7 | `SMA(2,3,4)` | 3MA |   408.0   3.84   -43.3 |    31.6   1.96   -26.1 |    -5.5  -0.06   -32.9 |   532.1   2.64   -47.2 | - |
| 8 | `SMA(8,9,75)` | 3MA |   337.5    4.5   -28.3 |    35.8   2.58   -12.8 |     4.7   0.68   -19.3 |   522.2   3.34   -31.7 | YES |
| 9 | `SMA(2,4,19)` | 3MA |   414.2   4.51   -35.5 |    26.6   1.96   -20.2 |    -5.8  -0.32   -22.9 |   512.7   3.04   -44.7 | - |
| 10 | `SMA(3,6)` | 2MA |   353.0   3.55   -44.3 |    33.2   2.01   -23.8 |    -2.9   0.12   -33.8 |   485.9   2.52   -45.7 | - |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `SMA(37,203)` | 2MA |    13.1    1.5    -9.0 |    33.6   3.53    -7.8 |    -4.7  -0.97   -10.6 |    44.1    1.7   -12.9 | - |
| 119 | `SMA(52,203)` | 2MA |     2.9   0.48    -8.4 |    37.1   3.94    -7.2 |    -5.4  -1.51    -7.3 |    33.6    1.5   -11.7 | - |
| 120 | `SMA(73,145)` | 2MA |     5.8   0.54   -15.4 |    21.5   2.37   -10.7 |    -9.5  -2.18   -13.6 |    16.3   0.68   -22.5 | - |


## 4h x WMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=-0.133**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,62], slow in [10,210] -> 25/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [38,233] -> 20/60 configs positive across TRAIN & VAL & OOS
- band members: WMA(2,5,118), WMA(8,9,75), WMA(3,8,43), WMA(2,10), WMA(4,30,67), WMA(12,14,75), WMA(3,19,43), WMA(10,11,233), WMA(5,12,208), WMA(4,10,208), WMA(15,22,48), WMA(4,15,208), WMA(12,77), WMA(8,91), WMA(15,65), WMA(18,55), WMA(19,27,38), WMA(12,43,75), WMA(10,34,60), WMA(4,86), WMA(22,65), WMA(44,75), WMA(31,53), WMA(37,89) (+21 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `WMA(3,5,186)` | 3MA |   417.6   4.79   -33.1 |    74.4   4.82   -10.0 |    -0.3   0.14   -17.7 |   799.9   3.92   -38.2 | - |
| 2 | `WMA(2,5,118)` | 3MA |   427.4    4.8   -34.7 |    69.3   4.22    -8.1 |     0.1    0.2   -16.5 |   793.5   3.81   -36.9 | YES |
| 3 | `WMA(2,3,4)` | 3MA |   497.3   4.11   -45.3 |    52.1   2.75   -23.8 |    -5.6  -0.03   -37.4 |   758.1   2.96   -49.3 | - |
| 4 | `WMA(4,5)` | 2MA |   453.0   3.96   -43.2 |    48.3   2.64   -19.3 |    -4.4   0.01   -32.6 |   683.9   2.87   -44.3 | - |
| 5 | `WMA(4,5,75)` | 3MA |   377.9   4.72   -36.9 |    51.0   3.43   -11.9 |    -1.0   0.09   -22.2 |   614.3   3.56   -42.2 | - |
| 6 | `WMA(3,4,38)` | 3MA |   437.4   4.81   -39.6 |    46.2   2.98   -15.3 |   -10.1  -0.73   -29.2 |   606.1   3.34   -43.5 | - |
| 7 | `WMA(2,3)` | 2MA |   392.2   3.41   -56.0 |    54.0   2.69   -30.1 |    -7.3  -0.13   -38.4 |   602.6   2.55   -62.0 | - |
| 8 | `WMA(2,4,19)` | 3MA |   478.5   4.61   -36.5 |    15.6   1.28   -21.6 |    -1.8   0.09   -26.1 |   556.7   3.02   -43.6 | - |
| 9 | `WMA(3,6)` | 2MA |   374.6   3.64   -47.6 |    39.4   2.27   -22.3 |    -4.1   0.03   -35.3 |   534.0   2.62   -50.7 | - |
| 10 | `WMA(2,26)` | 2MA |   355.8   4.27   -33.0 |    24.0   1.72   -17.4 |    -2.2   0.07   -28.7 |   452.5   2.85   -41.2 | - |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `WMA(75,132,148)` | 3MA |    39.8   2.34   -11.7 |    24.7   2.66   -10.0 |    -8.8  -1.88   -13.0 |    59.1   1.72   -17.5 | - |
| 119 | `WMA(86,122)` | 2MA |    34.4    1.9   -13.3 |    21.9   2.36    -9.5 |    -7.6   -1.2   -12.9 |    51.4   1.44   -20.7 | - |
| 120 | `WMA(102,237)` | 2MA |     3.1   0.53    -6.7 |    35.1   3.79    -7.2 |    -1.8  -0.37    -6.0 |    36.8   1.62   -10.2 | - |


## 4h x HMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=-0.401**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 26/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,75], slow in [15,233] -> 23/60 configs positive across TRAIN & VAL & OOS
- band members: HMA(6,13), HMA(5,12), HMA(4,15), HMA(4,8,17), HMA(2,5,118), HMA(10,19), HMA(2,8,22), HMA(2,4,19), HMA(6,9,24), HMA(6,210), HMA(4,37), HMA(2,169), HMA(8,210), HMA(3,44), HMA(15,151), HMA(10,128), HMA(4,30,67), HMA(5,37), HMA(2,3), HMA(8,34,233), HMA(4,199), HMA(5,6,34), HMA(3,8,43), HMA(5,199) (+25 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `HMA(6,13)` | 2MA |   628.4   4.47   -33.9 |    83.6   3.85   -22.9 |     8.2   0.82   -28.2 |  1347.3   3.58   -43.6 | YES |
| 2 | `HMA(5,12)` | 2MA |   525.7   4.15   -48.9 |    90.2   4.05   -19.8 |     7.5   0.78   -28.7 |  1179.7   3.42   -49.8 | YES |
| 3 | `HMA(4,15)` | 2MA |   421.2   3.79   -44.1 |    90.6   4.09   -20.7 |     2.7   0.48   -31.5 |   920.4   3.16   -50.7 | YES |
| 4 | `HMA(2,10)` | 2MA |   573.6   4.02   -45.0 |    57.0   2.74   -22.9 |    -8.3   -0.2   -35.0 |   870.0   2.91   -48.8 | - |
| 5 | `HMA(5,12,208)` | 3MA |   609.0   6.34   -21.2 |    41.4   3.03   -13.7 |    -9.2  -0.68   -28.2 |   810.2   4.16   -28.2 | - |
| 6 | `HMA(5,10,14)` | 3MA |   390.4   3.83   -50.7 |    91.2   4.41   -18.2 |   -11.1  -0.57   -32.6 |   733.7   3.07   -50.7 | - |
| 7 | `HMA(4,15,208)` | 3MA |   510.3   5.81   -17.6 |    38.8    3.0   -12.7 |    -9.8   -0.8   -27.6 |   664.3   3.88   -27.6 | - |
| 8 | `HMA(4,8,17)` | 3MA |   364.8   3.68   -50.9 |    52.4   2.98   -20.2 |     5.0   0.62   -28.3 |   643.6    2.9   -50.9 | YES |
| 9 | `HMA(2,12,14)` | 3MA |   315.4   3.46   -52.1 |    69.8   3.68   -20.7 |    -1.5   0.17   -30.7 |   595.0   2.83   -54.6 | - |
| 10 | `HMA(4,10,208)` | 3MA |   435.3   5.29   -24.7 |    38.8   2.83   -14.6 |   -11.2  -0.93   -25.8 |   560.1   3.53   -32.0 | - |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `HMA(12,13)` | 2MA |    55.5   1.65   -39.3 |    39.6   2.76   -14.7 |    -1.7   0.09   -29.1 |   113.4   1.57   -40.3 | - |
| 119 | `HMA(186,208,233)` | 3MA |    58.2   2.95   -16.7 |    28.2   2.93    -9.8 |    -1.2  -0.01   -13.2 |   100.5   2.26   -20.4 | - |
| 120 | `HMA(102,103)` | 2MA |    -1.3  -0.02   -18.8 |    -2.1  -0.35    -8.8 |    -6.9  -1.54   -10.8 |   -10.0  -0.44   -26.1 | - |


## 4h x DEMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=-0.464**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [5,210] -> 30/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,75], slow in [14,233] -> 31/60 configs positive across TRAIN & VAL & OOS
- band members: DEMA(4,5), DEMA(3,6), DEMA(2,10), DEMA(3,4,38), DEMA(6,77), DEMA(5,86), DEMA(4,15,208), DEMA(8,91), DEMA(2,4,19), DEMA(5,12), DEMA(5,12,208), DEMA(4,15), DEMA(4,10,208), DEMA(3,18), DEMA(2,60,208), DEMA(3,75,132), DEMA(24,43,60), DEMA(8,84,186), DEMA(6,67,233), DEMA(12,43,75), DEMA(26,75), DEMA(4,60,166), DEMA(15,34,186), DEMA(44,75) (+37 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `DEMA(2,3,4)` | 3MA |   493.2    3.9   -51.8 |    70.5    3.3   -21.9 |    -2.1   0.18   -32.6 |   890.7   3.02   -54.7 | - |
| 2 | `DEMA(2,3)` | 2MA |   444.2   3.55   -52.9 |    68.3   3.11   -23.1 |   -11.5   -0.4   -35.4 |   710.2   2.67   -57.8 | - |
| 3 | `DEMA(4,5)` | 2MA |   344.6   3.46   -53.6 |    74.6   3.58   -21.6 |     2.9   0.49   -33.6 |   698.4   2.85   -57.6 | YES |
| 4 | `DEMA(3,6)` | 2MA |   308.9   3.26   -55.0 |    73.1   3.52   -23.3 |     6.6   0.72   -31.0 |   654.6   2.77   -60.0 | YES |
| 5 | `DEMA(4,5,75)` | 3MA |   446.7   5.19   -32.9 |    28.3   2.18   -20.2 |    -0.0   0.23   -30.0 |   601.7   3.52   -36.1 | - |
| 6 | `DEMA(2,10)` | 2MA |   348.5   3.46   -50.1 |    51.9    2.8   -22.5 |     0.9   0.36   -32.1 |   587.6   2.68   -50.1 | YES |
| 7 | `DEMA(3,5,186)` | 3MA |   303.3   4.39   -35.7 |    64.7   3.97   -18.5 |    -4.5  -0.35   -20.5 |   534.2   3.45   -45.2 | - |
| 8 | `DEMA(3,4,38)` | 3MA |   366.6   4.24   -36.6 |    24.3    1.8   -23.3 |     1.4   0.35   -25.7 |   487.9   2.94   -44.3 | YES |
| 9 | `DEMA(6,77)` | 2MA |   326.0    4.8   -28.7 |    29.6   2.28   -15.0 |     1.6   0.37   -29.4 |   461.1   3.28   -33.2 | YES |
| 10 | `DEMA(2,73)` | 2MA |   354.1   4.57   -35.1 |    20.4   1.62   -19.5 |    -1.7   0.11   -30.3 |   437.8   2.98   -39.9 | - |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `DEMA(62,239)` | 2MA |    72.3    3.5    -9.9 |    13.9   2.16    -7.5 |    -0.1   0.08    -6.3 |    96.0    2.5   -11.8 | - |
| 119 | `DEMA(102,237)` | 2MA |    67.5   3.56    -7.7 |    11.8   2.95    -6.5 |    -4.7  -1.03    -8.3 |    78.6   2.49    -8.3 | - |
| 120 | `DEMA(186,208,233)` | 3MA |    35.4   2.33   -12.0 |    27.2   4.26    -5.1 |    -2.5  -0.68    -7.4 |    67.8    2.3   -12.0 | - |


## 4h x TEMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=-0.213**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [10,237] -> 25/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [14,233] -> 29/60 configs positive across TRAIN & VAL & OOS
- band members: TEMA(2,10), TEMA(4,5,75), TEMA(5,12), TEMA(2,73), TEMA(2,4,19), TEMA(6,13), TEMA(4,8,17), TEMA(4,15), TEMA(5,12,208), TEMA(3,102), TEMA(2,26), TEMA(3,18), TEMA(4,10,208), TEMA(5,10,14), TEMA(8,16), TEMA(15,151), TEMA(5,118,208), TEMA(2,12,14), TEMA(5,24,34), TEMA(12,13), TEMA(3,75,132), TEMA(2,8,22), TEMA(3,44), TEMA(12,178) (+30 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `TEMA(2,10)` | 2MA |   447.5   3.76   -48.2 |    71.8   3.39   -20.4 |     2.1   0.45   -31.1 |   860.6    3.0   -50.9 | YES |
| 2 | `TEMA(4,5)` | 2MA |   516.1    3.9   -48.9 |    54.9   2.73   -24.5 |    -2.1   0.16   -28.2 |   833.9   2.92   -54.8 | - |
| 3 | `TEMA(2,3,4)` | 3MA |   487.9   3.74   -52.1 |    65.8   3.05   -24.7 |    -4.7   0.04   -31.8 |   828.8   2.84   -57.2 | - |
| 4 | `TEMA(3,6)` | 2MA |   495.9   3.84   -46.9 |    58.5   2.86   -22.8 |    -5.4  -0.05   -31.1 |   793.2   2.86   -52.4 | - |
| 5 | `TEMA(4,5,75)` | 3MA |   482.8   4.96   -29.3 |    31.4   2.25   -17.7 |     1.3   0.34   -29.8 |   675.5   3.44   -35.4 | YES |
| 6 | `TEMA(5,12)` | 2MA |   308.7   3.33   -53.0 |    68.2    3.5   -15.4 |     3.5   0.52   -30.1 |   611.4   2.77   -53.6 | YES |
| 7 | `TEMA(2,73)` | 2MA |   412.4   4.69   -31.6 |    31.8   2.17   -16.9 |     2.2   0.43   -28.8 |   590.3   3.23   -35.4 | YES |
| 8 | `TEMA(2,3)` | 2MA |   323.6   3.05   -59.5 |    76.9   3.31   -25.3 |    -9.4  -0.24   -34.1 |   579.0   2.45   -64.5 | - |
| 9 | `TEMA(2,4,19)` | 3MA |   358.1   3.65   -48.2 |    15.5   1.21   -27.1 |    26.9   1.97   -21.5 |   571.4   2.76   -51.5 | YES |
| 10 | `TEMA(2,5,118)` | 3MA |   410.7   4.64   -38.8 |    27.7   2.07   -21.8 |    -5.7  -0.25   -28.8 |   515.0    3.1   -43.6 | - |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `TEMA(8,34,233)` | 3MA |   150.0   3.65   -37.0 |    -1.2    0.1   -22.0 |    -5.4  -0.59   -19.1 |   133.6   2.06   -45.0 | - |
| 119 | `TEMA(102,237)` | 2MA |    65.5   3.19   -12.9 |     8.1   2.13    -7.5 |     7.3   1.21    -7.4 |    91.9   2.47   -14.4 | YES |
| 120 | `TEMA(186,208,233)` | 3MA |    48.1    3.4   -13.6 |     4.9   1.45    -7.3 |     0.9   0.27    -6.9 |    56.7   2.27   -13.6 | YES |


## 4h x KAMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=-0.151**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [6,237] -> 21/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,75], slow in [14,233] -> 30/60 configs positive across TRAIN & VAL & OOS
- band members: KAMA(3,6), KAMA(3,4,38), KAMA(22,28), KAMA(4,8,17), KAMA(5,6,34), KAMA(8,9,75), KAMA(12,14,75), KAMA(6,22,24), KAMA(4,10,208), KAMA(12,13), KAMA(2,12,14), KAMA(37,38), KAMA(2,73), KAMA(6,34,94), KAMA(8,14,38), KAMA(6,9,24), KAMA(6,33), KAMA(6,77), KAMA(5,38,208), KAMA(5,24,34), KAMA(3,8,43), KAMA(10,67,148), KAMA(10,55), KAMA(10,34,60) (+27 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `KAMA(4,5)` | 2MA |   828.8   5.49   -28.4 |    60.9   3.18   -20.9 |    -1.4   0.14   -24.3 |  1373.7   3.94   -28.4 | - |
| 2 | `KAMA(4,5,75)` | 3MA |   444.3   5.79   -14.4 |    53.3   3.69    -8.9 |   -12.1  -1.38   -24.4 |   634.1   4.02   -24.4 | - |
| 3 | `KAMA(3,6)` | 2MA |   415.8   4.11   -32.0 |    32.6    2.1   -22.5 |     3.8   0.54   -28.2 |   609.6   2.98   -33.4 | YES |
| 4 | `KAMA(3,4,38)` | 3MA |   318.5   4.79   -29.8 |    43.4   3.21   -11.1 |     1.1    0.3   -19.0 |   506.5    3.6   -31.7 | YES |
| 5 | `KAMA(22,28)` | 2MA |   283.4   4.92   -16.3 |    43.5    3.5   -12.3 |    10.2   1.31   -16.8 |   505.9   3.87   -16.8 | YES |
| 6 | `KAMA(3,5,186)` | 3MA |   359.2   5.11   -17.4 |    34.0   2.73   -12.4 |    -5.2  -0.57   -18.8 |   483.5   3.59   -23.3 | - |
| 7 | `KAMA(4,8,17)` | 3MA |   353.3   5.09   -20.8 |    22.5   1.78   -19.3 |     3.7    0.6   -13.4 |   476.0   3.45   -21.3 | YES |
| 8 | `KAMA(10,11,233)` | 3MA |   290.9   4.94   -14.9 |    49.0   3.51   -12.3 |    -1.6  -0.07   -18.8 |   473.2   3.71   -18.9 | - |
| 9 | `KAMA(2,3)` | 2MA |   377.8   3.77   -42.0 |    41.5   2.44   -18.6 |   -18.2  -1.11   -37.7 |   453.1   2.53   -45.8 | - |
| 10 | `KAMA(5,6,34)` | 3MA |   358.8   5.16   -21.5 |    14.8   1.32   -19.4 |     4.9   0.77   -10.8 |   452.4   3.44   -28.1 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `KAMA(62,239)` | 2MA |    44.1   3.89    -4.4 |    29.9   3.21    -9.6 |    -2.8  -0.74    -7.1 |    81.9   2.71   -13.3 | - |
| 119 | `KAMA(102,237)` | 2MA |    34.8   3.28    -6.1 |    24.7   2.92    -8.9 |     0.7   0.25    -6.0 |    69.2   2.48    -8.9 | YES |
| 120 | `KAMA(186,208,233)` | 3MA |    41.3   2.61   -13.4 |    20.7   2.74    -7.5 |    -2.8  -0.76    -6.6 |    65.7   2.11   -13.4 | - |


## 4h x VIDYA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=-0.073**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,52], slow in [12,210] -> 20/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,60], slow in [14,233] -> 32/60 configs positive across TRAIN & VAL & OOS
- band members: VIDYA(2,12,14), VIDYA(5,10,14), VIDYA(6,13), VIDYA(6,9,24), VIDYA(4,8,17), VIDYA(5,12), VIDYA(3,8,43), VIDYA(6,14,15), VIDYA(4,15), VIDYA(2,26), VIDYA(8,9,75), VIDYA(3,18), VIDYA(12,13), VIDYA(8,16), VIDYA(2,30,84), VIDYA(5,12,208), VIDYA(6,22,24), VIDYA(4,15,208), VIDYA(5,24,34), VIDYA(10,11,233), VIDYA(12,14,75), VIDYA(10,19), VIDYA(4,30,67), VIDYA(4,37) (+28 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `VIDYA(3,6)` | 2MA |   289.1   4.51   -21.8 |    37.7   2.54   -16.9 |    -4.0  -0.17   -24.4 |   414.4   3.11   -24.4 | - |
| 2 | `VIDYA(4,5)` | 2MA |   269.2   4.38   -21.6 |    37.8   2.56   -16.7 |    -5.6  -0.33   -25.3 |   380.1   3.01   -25.3 | - |
| 3 | `VIDYA(2,12,14)` | 3MA |   208.4   4.64   -19.5 |    47.8   3.41    -9.7 |     1.7   0.37   -14.2 |   363.6   3.46   -23.1 | YES |
| 4 | `VIDYA(2,10)` | 2MA |   254.5   4.44   -22.7 |    36.4    2.5   -17.8 |    -4.3  -0.21   -24.6 |   362.7   3.02   -26.4 | - |
| 5 | `VIDYA(2,8,22)` | 3MA |   218.4   4.53   -22.1 |    45.5   3.34    -9.5 |    -0.7   0.08   -13.1 |   360.1   3.39   -24.9 | - |
| 6 | `VIDYA(2,3,4)` | 3MA |   307.3    4.4   -19.6 |    21.5   1.68   -16.0 |    -8.1  -0.63   -23.4 |   354.7   2.82   -25.1 | - |
| 7 | `VIDYA(5,10,14)` | 3MA |   209.3   4.83   -21.3 |    38.3   2.85   -12.2 |     3.7   0.58   -14.6 |   343.6   3.43   -24.1 | YES |
| 8 | `VIDYA(6,13)` | 2MA |   187.8   4.56   -22.4 |    43.1   3.06   -13.7 |     6.7   0.89   -16.9 |   339.6   3.39   -25.4 | YES |
| 9 | `VIDYA(6,9,24)` | 3MA |   215.6   5.01   -14.5 |    33.2   2.66   -14.8 |     4.1   0.65   -14.5 |   337.4   3.52   -18.6 | YES |
| 10 | `VIDYA(4,8,17)` | 3MA |   195.4   4.49   -23.1 |    45.7   3.27   -10.0 |     1.3   0.32   -15.7 |   336.0   3.35   -26.0 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `VIDYA(75,132,148)` | 3MA |     2.3   0.76    -2.8 |     3.9    1.0    -6.9 |    -0.4  -0.47    -1.8 |     5.8   0.64    -7.3 | - |
| 119 | `VIDYA(102,237)` | 2MA |     7.3   1.76    -3.9 |    -2.8  -0.69    -7.0 |    -0.6  -0.54    -1.6 |     3.7   0.43    -7.5 | - |
| 120 | `VIDYA(186,208,233)` | 3MA |     4.9   1.09    -4.4 |    -2.8  -1.73    -3.3 |    -2.4   -4.1    -2.4 |    -0.6  -0.04    -9.4 | - |


# Timeframe: 2h
_Benchmark (equal-weight u10 buy-hold, no cost): FULL-2020 net = 1222.5% (participation-tax reference)._

## 2h x EMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=-0.341**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,62], slow in [3,210] -> 21/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,60], slow in [4,233] -> 14/60 configs positive across TRAIN & VAL & OOS
- band members: EMA(2,3), EMA(3,5,186), EMA(4,5), EMA(2,3,4), EMA(2,5,118), EMA(3,6), EMA(2,19,22), EMA(4,10,208), EMA(4,15,208), EMA(10,34,60), EMA(5,12,208), EMA(5,38,208), EMA(3,30,166), EMA(3,75,132), EMA(10,128), EMA(2,169), EMA(30,43,186), EMA(3,102), EMA(26,75), EMA(12,178), EMA(5,199), EMA(62,105), EMA(4,199), EMA(37,89) (+11 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `EMA(2,3)` | 2MA |   324.9   3.32   -47.2 |    47.5   2.45   -21.6 |    14.9   1.24   -18.5 |   620.1   2.71   -47.2 | YES |
| 2 | `EMA(3,5,186)` | 3MA |   373.6   5.01   -22.6 |    46.2   3.26   -10.8 |     2.6   0.47   -11.8 |   610.7   3.78   -26.8 | YES |
| 3 | `EMA(4,5)` | 2MA |   401.0   4.02   -42.3 |    35.3   2.14   -18.5 |     1.2   0.35   -25.8 |   585.8    2.9   -43.4 | YES |
| 4 | `EMA(2,3,4)` | 3MA |   336.8    3.6   -42.1 |    40.1    2.3   -22.9 |    10.1   0.98   -19.2 |   573.9   2.79   -42.4 | YES |
| 5 | `EMA(2,5,118)` | 3MA |   346.5   4.74   -21.3 |    40.1   2.74   -17.2 |     7.4   0.98   -13.1 |   571.7   3.57   -24.3 | YES |
| 6 | `EMA(4,5,75)` | 3MA |   363.0   5.02   -24.6 |    45.7   3.19    -9.5 |    -0.6   0.12   -21.7 |   570.7   3.68   -24.6 | - |
| 7 | `EMA(2,10)` | 2MA |   369.7    3.9   -41.4 |    42.8   2.48   -17.5 |    -1.0   0.17   -27.6 |   563.6   2.87   -42.8 | - |
| 8 | `EMA(3,8,43)` | 3MA |   379.3   5.02   -26.0 |    44.4    3.1   -10.9 |    -5.0  -0.35   -25.0 |   557.2   3.58   -26.0 | - |
| 9 | `EMA(3,6)` | 2MA |   398.6   3.97   -41.5 |    26.7   1.72   -23.9 |     2.1   0.42   -23.8 |   545.1   2.79   -41.5 | YES |
| 10 | `EMA(5,6,34)` | 3MA |   357.8   4.79   -28.4 |    33.6   2.49   -12.9 |    -5.3  -0.35   -27.9 |   479.4    3.3   -30.5 | - |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `EMA(62,239)` | 2MA |    51.7   3.09    -7.2 |    25.3   2.87    -7.4 |    -1.8  -0.25    -8.8 |    86.7   2.37   -12.0 | - |
| 119 | `EMA(186,208,233)` | 3MA |    -1.2  -0.14    -5.1 |    31.1   3.74    -6.5 |    -0.6  -0.09    -5.1 |    28.7   1.43    -8.0 | - |
| 120 | `EMA(102,237)` | 2MA |     4.1   0.69    -4.3 |    23.7   2.96    -6.6 |    -5.6  -1.29   -10.2 |    21.5   1.09   -11.3 | - |


## 2h x SMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=-0.254**; TRAIN+VAL top-10 -> OOS top-10 overlap = **3/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [5,210] -> 21/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,48], slow in [4,233] -> 18/60 configs positive across TRAIN & VAL & OOS
- band members: SMA(4,5), SMA(10,11,233), SMA(3,6), SMA(2,10), SMA(2,3,4), SMA(3,5,186), SMA(4,10,208), SMA(5,12,208), SMA(8,9,75), SMA(4,5,75), SMA(12,22,118), SMA(6,34,94), SMA(4,15,208), SMA(10,17,233), SMA(102,103), SMA(15,84,148), SMA(10,67,148), SMA(2,169), SMA(8,34,233), SMA(31,124), SMA(24,106,118), SMA(22,151), SMA(26,172), SMA(12,106,148) (+15 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `SMA(4,5)` | 2MA |   427.9   3.59   -52.4 |    39.8   2.12   -22.7 |    14.3   1.19   -24.0 |   743.6   2.79   -52.4 | YES |
| 2 | `SMA(10,11,233)` | 3MA |   450.5    5.4   -26.8 |    39.1    2.9   -13.9 |     9.1   1.21   -10.7 |   735.7   4.05   -33.9 | YES |
| 3 | `SMA(3,6)` | 2MA |   429.0   3.86   -52.9 |    44.7   2.39   -21.9 |     9.1   0.89   -21.1 |   734.7   2.94   -52.9 | YES |
| 4 | `SMA(2,10)` | 2MA |   534.5   4.57   -41.4 |    29.6   1.88   -21.3 |     1.1   0.34   -23.9 |   731.5   3.15   -43.1 | YES |
| 5 | `SMA(2,3,4)` | 3MA |   424.8   3.79   -43.0 |    49.1   2.57   -22.0 |     0.1   0.29   -22.8 |   683.0   2.85   -43.0 | YES |
| 6 | `SMA(3,5,186)` | 3MA |   298.4   4.21   -29.3 |    64.1    4.0    -9.6 |     7.8   1.01   -12.5 |   604.8   3.55   -32.1 | YES |
| 7 | `SMA(4,10,208)` | 3MA |   376.8   5.33   -27.4 |    30.0    2.5   -12.2 |     2.5   0.47   -11.1 |   535.7    3.8   -35.1 | YES |
| 8 | `SMA(3,8,43)` | 3MA |   358.8   4.97   -30.4 |    43.9    3.2   -10.8 |    -9.6  -0.92   -23.8 |   496.9   3.48   -33.4 | - |
| 9 | `SMA(5,12,208)` | 3MA |   333.9   5.12   -25.1 |    33.6   2.77    -9.9 |     2.4   0.46   -14.0 |   493.7   3.73   -29.7 | YES |
| 10 | `SMA(2,3)` | 2MA |   265.5   2.83   -57.6 |    60.0   2.76   -28.0 |    -0.7   0.28   -30.2 |   480.8   2.31   -62.1 | - |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `SMA(19,132,233)` | 3MA |    38.3   2.12   -17.9 |    15.0   1.75   -12.2 |    -4.4   -0.6   -15.7 |    52.1   1.47   -21.6 | - |
| 119 | `SMA(186,208,233)` | 3MA |    31.7   2.22   -16.2 |    20.7   2.41    -9.5 |    -8.3  -1.76   -14.8 |    45.8   1.54   -22.4 | - |
| 120 | `SMA(102,237)` | 2MA |    23.1    1.6   -13.9 |    18.0    2.2    -9.2 |    -7.2  -1.38   -10.8 |    34.7   1.21   -19.9 | - |


## 2h x WMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=-0.37**; TRAIN+VAL top-10 -> OOS top-10 overlap = **2/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [5,239] -> 24/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,75], slow in [4,233] -> 15/60 configs positive across TRAIN & VAL & OOS
- band members: WMA(4,10,208), WMA(4,5), WMA(2,10), WMA(5,12,208), WMA(3,6), WMA(10,11,233), WMA(2,60,208), WMA(2,3,4), WMA(12,14,75), WMA(12,33), WMA(8,22,60), WMA(15,28), WMA(8,34,233), WMA(10,67,148), WMA(8,84,186), WMA(22,28), WMA(15,84,148), WMA(24,106,118), WMA(37,89), WMA(44,203), WMA(52,203), WMA(6,210), WMA(8,210), WMA(102,103) (+15 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `WMA(4,10,208)` | 3MA |   404.6   5.48   -18.8 |    55.5   3.78   -10.7 |     2.8   0.49   -13.3 |   706.2   4.16   -22.1 | YES |
| 2 | `WMA(4,5)` | 2MA |   372.1   3.55   -51.3 |    58.4   2.88   -18.3 |     6.7   0.74   -22.6 |   698.0   2.85   -51.3 | YES |
| 3 | `WMA(2,10)` | 2MA |   431.3   3.96   -43.8 |    27.6   1.74   -24.2 |     6.4   0.72   -18.9 |   621.0   2.84   -43.8 | YES |
| 4 | `WMA(5,12,208)` | 3MA |   378.2   5.43   -20.3 |    46.7   3.41    -8.7 |     1.2   0.31   -13.5 |   609.8   4.02   -23.2 | YES |
| 5 | `WMA(5,12)` | 2MA |   434.1   4.28   -48.1 |    37.6   2.31   -17.7 |    -8.4  -0.43   -29.4 |   573.0   2.94   -48.1 | - |
| 6 | `WMA(3,6)` | 2MA |   337.9   3.42   -48.1 |    46.0   2.43   -22.2 |     3.8   0.54   -21.0 |   563.3   2.64   -48.1 | YES |
| 7 | `WMA(4,15,208)` | 3MA |   365.2   5.35   -19.3 |    45.9   3.35    -9.7 |    -3.5  -0.27   -15.3 |   555.0   3.86   -22.3 | - |
| 8 | `WMA(4,5,75)` | 3MA |   397.5   4.94   -30.8 |    48.4   3.19   -12.4 |   -16.9  -1.61   -32.8 |   513.8   3.33   -32.8 | - |
| 9 | `WMA(3,5,186)` | 3MA |   313.7    4.4   -26.9 |    54.5   3.38   -15.5 |    -5.5  -0.39   -21.5 |   504.1   3.29   -30.6 | - |
| 10 | `WMA(10,11,233)` | 3MA |   364.7   5.34   -27.3 |    28.8   2.42   -10.8 |     0.7   0.24   -15.6 |   502.4   3.74   -32.5 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `WMA(102,237)` | 2MA |    52.9   2.82   -11.8 |    30.9   3.03   -10.8 |     0.7   0.24   -11.5 |   101.5   2.33   -16.1 | YES |
| 119 | `WMA(19,132,233)` | 3MA |    70.9    2.9   -17.3 |    26.2   2.55    -9.6 |    -7.1  -0.82   -18.4 |   100.4   2.04   -21.1 | - |
| 120 | `WMA(186,208,233)` | 3MA |    48.9   2.72   -14.6 |    18.9    2.3    -8.7 |    -8.1  -1.51   -11.9 |    62.8   1.81   -20.2 | - |


## 2h x HMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.225**; TRAIN+VAL top-10 -> OOS top-10 overlap = **3/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,31], slow in [6,210] -> 27/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [14,233] -> 24/60 configs positive across TRAIN & VAL & OOS
- band members: HMA(8,16), HMA(10,19), HMA(2,26), HMA(5,10,14), HMA(18,23), HMA(3,18), HMA(2,12,14), HMA(4,8,17), HMA(6,9,24), HMA(6,13), HMA(5,37), HMA(3,44), HMA(6,33), HMA(4,37), HMA(2,8,22), HMA(2,4,19), HMA(3,6), HMA(3,12,27), HMA(2,10), HMA(4,15), HMA(12,14,75), HMA(6,77), HMA(5,86), HMA(3,102) (+27 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `HMA(8,16)` | 2MA |   763.5   4.95   -40.5 |    51.3   2.65   -23.0 |    15.9   1.33   -16.2 |  1414.6   3.72   -40.5 | YES |
| 2 | `HMA(10,19)` | 2MA |   728.7   5.04   -46.6 |    32.1   1.94   -26.4 |     3.4   0.52   -18.5 |  1031.9   3.46   -47.5 | YES |
| 3 | `HMA(2,26)` | 2MA |   477.4   4.09   -42.2 |    33.5   1.98   -23.1 |    12.0   1.08   -18.0 |   763.5   3.02   -42.2 | YES |
| 4 | `HMA(5,10,14)` | 3MA |   437.8   3.96   -46.1 |    41.1   2.31   -25.2 |    12.3   1.12   -16.9 |   752.3   3.04   -46.1 | YES |
| 5 | `HMA(18,23)` | 2MA |   435.9   4.19   -49.2 |    40.1   2.44   -22.2 |    12.5   1.17   -20.6 |   744.3    3.2   -54.2 | YES |
| 6 | `HMA(3,18)` | 2MA |   383.8   3.52   -50.1 |    52.8   2.61   -23.9 |     9.2   0.89   -16.8 |   707.8    2.8   -50.1 | YES |
| 7 | `HMA(2,12,14)` | 3MA |   445.9   4.07   -51.2 |    33.3    2.0   -26.8 |     5.0   0.63   -18.8 |   664.0   2.94   -51.2 | YES |
| 8 | `HMA(4,8,17)` | 3MA |   369.1   3.69   -48.1 |    34.3   2.03   -25.6 |    19.7   1.59   -17.2 |   654.1   2.89   -48.1 | YES |
| 9 | `HMA(6,9,24)` | 3MA |   400.4   3.99   -52.2 |    35.0   2.18   -23.9 |     6.1   0.71   -18.4 |   617.1   2.95   -55.6 | YES |
| 10 | `HMA(2,3)` | 2MA |   360.6   3.23   -60.4 |    50.3   2.44   -27.2 |    -0.3   0.32   -32.1 |   590.4   2.48   -60.4 | - |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `HMA(4,5,75)` | 3MA |    37.6   1.28   -51.3 |    10.7   1.11   -15.3 |     8.0   0.96   -17.1 |    64.5   1.16   -51.3 | YES |
| 119 | `HMA(52,89)` | 2MA |    77.7   2.13   -44.3 |    -5.9  -0.23   -20.6 |   -11.3  -0.86   -27.2 |    48.3   0.98   -52.1 | - |
| 120 | `HMA(102,103)` | 2MA |    -5.9  -0.32   -22.8 |    -0.9  -0.03   -11.3 |     2.4   0.55   -11.2 |    -4.5  -0.06   -29.2 | - |


## 2h x DEMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=-0.134**; TRAIN+VAL top-10 -> OOS top-10 overlap = **4/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [5,239] -> 22/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [4,233] -> 18/60 configs positive across TRAIN & VAL & OOS
- band members: DEMA(2,10), DEMA(4,5), DEMA(3,6), DEMA(4,15), DEMA(5,12), DEMA(3,18), DEMA(6,13), DEMA(5,12,208), DEMA(2,3,4), DEMA(10,19), DEMA(2,73), DEMA(6,34,94), DEMA(10,34,60), DEMA(2,19,22), DEMA(12,178), DEMA(3,12,27), DEMA(24,106,118), DEMA(8,84,186), DEMA(6,33), DEMA(8,39), DEMA(15,34,186), DEMA(2,169), DEMA(6,22,24), DEMA(5,118,208) (+16 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `DEMA(2,10)` | 2MA |   441.6   3.85   -42.4 |    55.7   2.79   -21.6 |    14.6   1.23   -14.4 |   866.4   3.08   -42.4 | YES |
| 2 | `DEMA(4,5)` | 2MA |   322.2    3.3   -47.7 |    57.2    2.8   -24.4 |    13.5   1.16   -15.0 |   653.6   2.75   -50.9 | YES |
| 3 | `DEMA(3,6)` | 2MA |   297.1   3.15   -53.2 |    59.1   2.87   -26.4 |     8.8   0.86   -14.8 |   587.1   2.63   -55.6 | YES |
| 4 | `DEMA(4,15)` | 2MA |   314.2   3.59   -49.6 |    44.5   2.59   -17.4 |     9.8   0.96   -18.7 |   557.2   2.84   -49.6 | YES |
| 5 | `DEMA(5,12)` | 2MA |   290.7   3.42   -53.3 |    41.1   2.44   -17.6 |    16.9   1.44   -16.6 |   544.2   2.79   -53.3 | YES |
| 6 | `DEMA(4,5,75)` | 3MA |   352.9   4.63   -35.9 |    40.6   2.91   -17.4 |    -2.3  -0.01   -25.9 |   522.0   3.34   -38.4 | - |
| 7 | `DEMA(8,16)` | 2MA |   357.0   3.99   -49.6 |    39.0   2.52   -17.9 |    -5.4  -0.19   -31.1 |   500.6   2.85   -49.6 | - |
| 8 | `DEMA(3,18)` | 2MA |   281.8    3.4   -51.3 |    42.1   2.49   -18.3 |     8.1   0.85   -20.4 |   486.6   2.68   -51.3 | YES |
| 9 | `DEMA(2,26)` | 2MA |   306.1   3.52   -49.2 |    46.8   2.69   -19.9 |    -1.8   0.14   -26.5 |   485.9   2.67   -49.2 | - |
| 10 | `DEMA(4,10,208)` | 3MA |   305.9   4.96   -32.9 |    50.0   3.72   -13.0 |    -9.5  -0.95   -20.8 |   450.7   3.56   -36.3 | - |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `DEMA(75,132,148)` | 3MA |   107.3   3.91   -18.1 |    14.8   1.64   -11.5 |    -3.5  -0.27   -19.9 |   129.7   2.41   -21.2 | - |
| 119 | `DEMA(102,237)` | 2MA |    57.0   2.69   -19.0 |    21.8   2.31    -9.1 |     3.2   0.59   -11.4 |    97.4   2.15   -20.6 | YES |
| 120 | `DEMA(186,208,233)` | 3MA |    40.1   2.42   -12.6 |    10.9   1.68    -7.4 |     7.6   1.32    -7.7 |    67.2   1.99   -17.5 | YES |


## 2h x TEMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.325**; TRAIN+VAL top-10 -> OOS top-10 overlap = **3/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,37], slow in [3,210] -> 26/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,19], slow in [14,233] -> 23/60 configs positive across TRAIN & VAL & OOS
- band members: TEMA(5,12), TEMA(6,13), TEMA(4,15), TEMA(3,18), TEMA(18,23), TEMA(2,19,22), TEMA(2,26), TEMA(5,10,14), TEMA(4,8,17), TEMA(6,14,15), TEMA(2,12,14), TEMA(6,9,24), TEMA(10,19), TEMA(8,16), TEMA(12,13), TEMA(4,199), TEMA(5,199), TEMA(2,10), TEMA(6,22,24), TEMA(22,28), TEMA(6,210), TEMA(2,8,22), TEMA(3,12,27), TEMA(2,3) (+25 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `TEMA(5,12)` | 2MA |   481.2   4.13   -38.7 |    50.3   2.66   -22.6 |    16.8   1.38   -15.3 |   920.4   3.24   -38.7 | YES |
| 2 | `TEMA(6,13)` | 2MA |   493.6   4.23   -44.9 |    51.9   2.79   -20.9 |    11.4   1.05   -15.4 |   904.1   3.28   -45.9 | YES |
| 3 | `TEMA(4,15)` | 2MA |   384.2   3.72   -42.9 |    52.2   2.77   -20.9 |    16.9   1.39   -14.6 |   761.8   3.03   -42.9 | YES |
| 4 | `TEMA(3,18)` | 2MA |   393.0   3.74   -46.2 |    43.3   2.39   -20.8 |    16.2   1.34   -14.2 |   720.5   2.96   -46.2 | YES |
| 5 | `TEMA(18,23)` | 2MA |   398.2   4.44   -44.0 |    56.9   3.46   -14.8 |     2.5   0.44   -27.5 |   700.8   3.43   -44.0 | YES |
| 6 | `TEMA(2,19,22)` | 3MA |   450.5   4.86   -36.3 |    24.6   1.85   -17.5 |     6.3   0.77   -23.8 |   628.7    3.4   -36.3 | YES |
| 7 | `TEMA(2,26)` | 2MA |   365.5   3.72   -47.2 |    37.3   2.21   -20.2 |    12.6   1.13   -17.2 |   619.3   2.87   -47.2 | YES |
| 8 | `TEMA(5,10,14)` | 3MA |   390.6   4.03   -46.5 |    33.5   2.16   -19.3 |     8.3   0.88   -15.8 |   609.7    3.0   -50.8 | YES |
| 9 | `TEMA(4,8,17)` | 3MA |   381.0   3.93   -50.3 |    34.5   2.19   -19.2 |     7.3    0.8   -18.3 |   593.7   2.93   -52.4 | YES |
| 10 | `TEMA(6,14,15)` | 3MA |   317.7    3.9   -41.0 |    50.6   3.13   -13.8 |     9.0   0.95   -18.1 |   586.1   3.13   -42.9 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `TEMA(62,239)` | 2MA |   119.8    3.8   -18.0 |    12.8   1.41   -19.7 |    -9.3  -1.13   -23.2 |   125.0   2.23   -25.8 | - |
| 119 | `TEMA(186,208,233)` | 3MA |    84.9   3.66   -11.1 |    20.4   2.43    -6.7 |    -5.5  -0.67   -21.5 |   110.4   2.42   -21.5 | - |
| 120 | `TEMA(38,67,233)` | 3MA |   111.6   3.46   -17.8 |     4.0   0.62   -12.1 |    -9.0  -1.03   -21.4 |   100.3   1.92   -21.4 | - |


## 2h x KAMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=-0.392**; TRAIN+VAL top-10 -> OOS top-10 overlap = **1/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [5,102], slow in [23,210] -> 18/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [3,75], slow in [15,233] -> 18/60 configs positive across TRAIN & VAL & OOS
- band members: KAMA(18,23), KAMA(10,11,233), KAMA(3,5,186), KAMA(26,32), KAMA(10,17,233), KAMA(3,12,27), KAMA(102,103), KAMA(12,33), KAMA(22,28), KAMA(5,38,208), KAMA(3,75,132), KAMA(12,22,118), KAMA(6,14,15), KAMA(10,67,148), KAMA(4,15,208), KAMA(12,43,75), KAMA(15,65), KAMA(52,89), KAMA(19,27,38), KAMA(12,106,148), KAMA(18,128), KAMA(5,199), KAMA(24,106,118), KAMA(75,132,148) (+12 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `KAMA(4,5)` | 2MA |   421.7   4.03   -41.2 |    26.9   1.74   -28.2 |    -3.7   0.04   -28.4 |   537.8   2.73   -46.8 | - |
| 2 | `KAMA(18,23)` | 2MA |   306.4   4.92   -24.0 |    30.8   2.26   -13.3 |     4.3   0.62   -19.3 |   454.3   3.41   -24.0 | YES |
| 3 | `KAMA(2,3)` | 2MA |   381.5   3.78   -43.0 |    18.5   1.33   -27.4 |    -4.0   0.01   -35.8 |   448.0    2.5   -47.6 | - |
| 4 | `KAMA(10,11,233)` | 3MA |   278.0   5.26   -20.2 |    24.5    2.2    -9.2 |    14.5   1.94    -9.9 |   438.9   3.87   -24.0 | YES |
| 5 | `KAMA(8,9,75)` | 3MA |   300.7    5.2   -24.5 |    46.0   3.32    -8.4 |    -8.3  -0.83   -23.6 |   436.7   3.61   -27.4 | - |
| 6 | `KAMA(3,6)` | 2MA |   368.5   4.02   -37.9 |    30.8    2.0   -21.5 |   -12.8  -0.76   -33.6 |   434.3   2.64   -42.7 | - |
| 7 | `KAMA(2,19,22)` | 3MA |   350.7   5.53   -22.8 |    19.9   1.68   -16.7 |    -3.5  -0.27   -16.9 |   421.7   3.52   -22.8 | - |
| 8 | `KAMA(2,3,4)` | 3MA |   409.5   4.48   -38.1 |    13.9   1.22   -17.8 |   -10.2  -0.67   -32.0 |   421.4   2.79   -40.2 | - |
| 9 | `KAMA(5,6,34)` | 3MA |   362.0   5.14   -22.8 |    30.3   2.39   -19.0 |   -14.8  -1.61   -26.8 |   413.1   3.28   -30.3 | - |
| 10 | `KAMA(2,5,118)` | 3MA |   290.8   4.82   -20.4 |    33.0   2.56   -11.9 |    -1.7  -0.05   -19.6 |   410.9    3.4   -23.5 | - |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `KAMA(102,237)` | 2MA |    59.5   3.21    -7.1 |    22.6   2.74    -8.0 |    -0.3   0.07   -11.7 |    94.9   2.42   -11.7 | - |
| 119 | `KAMA(62,239)` | 2MA |    53.3   2.95   -12.8 |    17.7   1.99   -10.6 |    -1.9  -0.19   -11.2 |    77.0   2.02   -21.8 | - |
| 120 | `KAMA(62,105)` | 2MA |    52.9    2.6   -20.6 |    16.4   1.76   -12.7 |    -5.9   -0.7   -20.4 |    67.5   1.66   -26.6 | - |


## 2h x VIDYA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=-0.141**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,73], slow in [23,239] -> 25/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,60], slow in [19,233] -> 24/60 configs positive across TRAIN & VAL & OOS
- band members: VIDYA(2,4,19), VIDYA(4,5,75), VIDYA(5,6,34), VIDYA(3,8,43), VIDYA(2,5,118), VIDYA(3,12,27), VIDYA(3,44), VIDYA(5,37), VIDYA(8,39), VIDYA(4,60,166), VIDYA(2,60,208), VIDYA(8,14,38), VIDYA(15,28), VIDYA(2,19,22), VIDYA(12,33), VIDYA(3,75,132), VIDYA(18,23), VIDYA(15,34,186), VIDYA(8,34,233), VIDYA(12,43,75), VIDYA(10,55), VIDYA(2,73), VIDYA(19,27,38), VIDYA(3,102) (+25 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `VIDYA(2,4,19)` | 3MA |   279.5   4.76   -27.4 |    21.0   1.83   -11.8 |     0.9   0.28   -20.6 |   363.4   3.23   -30.2 | YES |
| 2 | `VIDYA(3,6)` | 2MA |   259.8   4.13   -30.2 |    29.2   2.09   -18.4 |    -3.1  -0.06   -28.7 |   350.5   2.81   -32.0 | - |
| 3 | `VIDYA(3,4,38)` | 3MA |   225.4   4.49   -27.5 |    36.3   2.86   -10.8 |    -2.2   -0.1   -20.0 |   333.9   3.24   -33.1 | - |
| 4 | `VIDYA(4,8,17)` | 3MA |   220.9   4.41   -24.2 |    37.2   2.75    -9.1 |    -3.8  -0.22   -22.3 |   323.7   3.09   -27.5 | - |
| 5 | `VIDYA(4,5)` | 2MA |   246.5   4.05   -31.1 |    22.1   1.68   -20.0 |    -2.9  -0.04   -27.9 |   310.9   2.67   -33.7 | - |
| 6 | `VIDYA(2,8,22)` | 3MA |   198.5   4.17   -28.0 |    45.0   3.27    -9.3 |    -5.3  -0.42   -20.4 |   310.2   3.07   -31.7 | - |
| 7 | `VIDYA(5,10,14)` | 3MA |   202.5   4.25   -24.4 |    39.1   2.88   -11.0 |    -3.4  -0.16   -23.8 |   306.7   3.02   -27.5 | - |
| 8 | `VIDYA(4,5,75)` | 3MA |   184.0   4.19   -28.2 |    30.5   2.57    -7.9 |     9.3   1.32    -9.0 |   305.0   3.24   -32.5 | YES |
| 9 | `VIDYA(2,12,14)` | 3MA |   184.3   3.97   -26.7 |    38.0   2.81    -8.9 |    -0.9   0.09   -20.9 |   288.7   2.93   -29.7 | - |
| 10 | `VIDYA(5,6,34)` | 3MA |   169.8   3.96   -29.3 |    37.8   2.86   -10.9 |     3.1   0.52   -16.3 |   283.0   3.02   -34.2 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `VIDYA(37,89)` | 2MA |    -1.1  -0.13    -5.8 |    24.7   3.13    -6.6 |    -7.8  -1.86   -11.5 |    13.7   0.77   -12.1 | - |
| 119 | `VIDYA(19,132,233)` | 3MA |     0.5   0.15    -4.9 |    12.1   2.44    -5.9 |    -0.8  -0.13    -6.4 |    11.8   0.81    -6.4 | - |
| 120 | `VIDYA(186,208,233)` | 3MA |     2.0   0.65    -2.7 |     9.3   2.48    -4.3 |    -0.3  -0.27    -1.9 |    11.1   1.22    -4.3 | - |


# Timeframe: 1h
_Benchmark (equal-weight u10 buy-hold, no cost): FULL-2020 net = 1247.4% (participation-tax reference)._

## 1h x EMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=-0.201**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 49/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,75], slow in [4,233] -> 42/60 configs positive across TRAIN & VAL & OOS
- band members: EMA(5,12), EMA(6,13), EMA(4,15), EMA(10,11,233), EMA(5,12,208), EMA(4,15,208), EMA(8,16), EMA(6,22,24), EMA(8,14,38), EMA(4,10,208), EMA(5,10,14), EMA(10,17,233), EMA(2,26), EMA(12,22,118), EMA(6,14,15), EMA(3,30,166), EMA(12,13), EMA(2,30,84), EMA(2,10), EMA(4,30,67), EMA(5,24,34), EMA(10,19), EMA(6,34,94), EMA(4,86) (+67 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `EMA(5,12)` | 2MA |   397.9   4.52   -22.6 |    32.8   2.17   -15.7 |     0.2   0.25   -20.7 |   562.3   3.15   -22.6 | YES |
| 2 | `EMA(6,13)` | 2MA |   404.6   4.61   -22.4 |    27.0    1.9   -16.9 |     1.1   0.32   -19.2 |   547.8   3.16   -22.5 | YES |
| 3 | `EMA(4,15)` | 2MA |   358.0   4.28   -25.0 |    29.1   1.98   -17.7 |     1.1   0.33   -21.2 |   498.1   2.99   -25.0 | YES |
| 4 | `EMA(10,11,233)` | 3MA |   306.7   5.49   -13.3 |    30.1   2.47    -8.8 |     4.1   0.68   -11.5 |   451.0   3.84   -18.5 | YES |
| 5 | `EMA(3,18)` | 2MA |   339.1   4.16   -24.6 |    22.5   1.63   -18.5 |    -1.9   0.07   -24.6 |   427.7   2.79   -25.0 | - |
| 6 | `EMA(5,12,208)` | 3MA |   274.0   5.06   -13.1 |    35.3   2.77    -8.3 |     3.7   0.62   -11.9 |   424.8   3.67   -17.0 | YES |
| 7 | `EMA(4,15,208)` | 3MA |   268.1   5.01   -15.2 |    34.5   2.71   -10.0 |     4.6   0.74   -11.4 |   418.0   3.64   -17.8 | YES |
| 8 | `EMA(8,16)` | 2MA |   294.4   4.12   -24.2 |    14.1   1.21   -17.6 |    12.4   1.29   -16.5 |   405.9   2.88   -25.2 | YES |
| 9 | `EMA(6,22,24)` | 3MA |   294.2   4.68   -19.5 |    23.7   1.93   -11.6 |     2.7   0.46   -21.7 |   400.4   3.21   -22.1 | YES |
| 10 | `EMA(8,14,38)` | 3MA |   303.2   4.87   -20.9 |    21.1   1.78   -14.5 |     1.4   0.33   -21.1 |   395.0   3.26   -24.6 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `EMA(102,237)` | 2MA |    53.1   3.37   -14.9 |    13.6   1.83    -8.5 |     6.6    1.0   -12.9 |    85.5   2.32   -18.7 | YES |
| 119 | `EMA(62,239)` | 2MA |    44.1   2.56   -22.4 |    16.6   1.98    -8.4 |     8.1   1.12   -13.1 |    81.5   2.04   -26.8 | YES |
| 120 | `EMA(186,208,233)` | 3MA |    42.5   3.35   -11.1 |    19.5   2.37   -10.4 |    -3.7  -0.54   -11.6 |    64.0    2.1   -18.5 | - |


## 1h x SMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=-0.123**; TRAIN+VAL top-10 -> OOS top-10 overlap = **1/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [6,239] -> 23/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [22,233] -> 16/60 configs positive across TRAIN & VAL & OOS
- band members: SMA(12,13), SMA(12,14,75), SMA(3,30,166), SMA(2,26), SMA(5,38,208), SMA(8,34,233), SMA(2,60,208), SMA(6,33), SMA(5,37), SMA(2,30,84), SMA(2,19,22), SMA(3,6), SMA(18,23), SMA(19,43,233), SMA(4,37), SMA(4,30,67), SMA(3,44), SMA(10,34,60), SMA(2,10), SMA(10,55), SMA(8,84,186), SMA(4,60,166), SMA(2,73), SMA(12,77) (+15 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `SMA(12,13)` | 2MA |   802.4   5.14   -35.1 |    21.2   1.46   -23.3 |    13.4   1.19   -20.8 |  1140.6   3.55   -35.1 | YES |
| 2 | `SMA(6,13)` | 2MA |   586.5   4.99   -31.2 |    58.1   3.29   -13.4 |   -13.7  -0.94   -24.7 |   837.0   3.48   -31.2 | - |
| 3 | `SMA(8,16)` | 2MA |   594.2   5.38   -26.8 |    39.8   2.55   -14.6 |    -6.6  -0.35   -19.0 |   806.7   3.63   -26.8 | - |
| 4 | `SMA(4,15)` | 2MA |   558.6   5.13   -25.7 |    40.4   2.53   -16.0 |    -9.2  -0.55   -19.5 |   739.6   3.44   -25.9 | - |
| 5 | `SMA(5,12)` | 2MA |   465.3   4.45   -32.9 |    52.0   2.97   -17.5 |    -8.7  -0.47   -21.7 |   684.1   3.17   -32.9 | - |
| 6 | `SMA(5,10,14)` | 3MA |   519.7   5.41   -20.3 |    25.5   1.93   -15.8 |   -13.5  -1.11   -22.6 |   572.4   3.38   -27.0 | - |
| 7 | `SMA(3,18)` | 2MA |   445.9    4.7   -25.5 |    26.9   1.86   -17.5 |    -3.5  -0.05   -18.3 |   568.6   3.13   -25.5 | - |
| 8 | `SMA(6,22,24)` | 3MA |   299.2   4.59   -38.0 |    48.3   3.42   -10.1 |    -3.9  -0.23   -23.2 |   468.7    3.4   -38.8 | - |
| 9 | `SMA(12,14,75)` | 3MA |   366.0   5.44   -17.1 |    16.2   1.53   -14.2 |     2.5   0.46   -15.8 |   455.2   3.59   -17.1 | YES |
| 10 | `SMA(3,30,166)` | 3MA |   307.4   5.63   -16.8 |    28.3   2.43    -8.7 |     6.2   0.96   -13.2 |   454.9   3.96   -17.9 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `SMA(73,145)` | 2MA |    92.6   3.29   -26.0 |    11.5   1.23   -16.9 |    -9.2  -0.87   -25.0 |    95.0   1.83   -28.8 | - |
| 119 | `SMA(75,132,148)` | 3MA |    72.5   2.93   -28.6 |    14.1   1.48   -16.3 |    -2.7  -0.18   -19.3 |    91.6   1.88   -33.3 | - |
| 120 | `SMA(102,237)` | 2MA |    78.9   3.71   -24.2 |     2.9    0.5   -18.3 |     0.5   0.23   -22.5 |    85.1   1.99   -27.1 | YES |


## 1h x WMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=-0.137**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [5,239] -> 22/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [4,233] -> 15/60 configs positive across TRAIN & VAL & OOS
- band members: WMA(3,18), WMA(2,26), WMA(6,33), WMA(4,37), WMA(5,38,208), WMA(5,37), WMA(15,34,186), WMA(3,30,166), WMA(18,23), WMA(19,43,233), WMA(3,4,38), WMA(3,44), WMA(2,3,4), WMA(4,5), WMA(8,39), WMA(12,33), WMA(30,43,186), WMA(5,86), WMA(5,118,208), WMA(4,5,75), WMA(4,86), WMA(2,73), WMA(3,6), WMA(38,67,233) (+13 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `WMA(8,16)` | 2MA |   590.2   5.22   -26.4 |    44.0   2.74   -14.9 |   -13.1  -0.92   -22.8 |   763.9   3.48   -26.4 | - |
| 2 | `WMA(10,19)` | 2MA |   514.4   5.18   -27.7 |    35.3   2.37   -15.3 |    -9.4  -0.64   -21.3 |   653.0   3.42   -27.7 | - |
| 3 | `WMA(12,13)` | 2MA |   485.5   4.78   -27.1 |    35.1   2.29   -17.4 |   -13.0  -0.92   -24.8 |   587.9   3.14   -27.6 | - |
| 4 | `WMA(3,18)` | 2MA |   404.7   4.25   -35.4 |    24.9   1.71   -19.3 |     1.7   0.39   -20.8 |   541.2   2.92   -36.7 | YES |
| 5 | `WMA(2,26)` | 2MA |   410.2   4.46   -28.3 |    22.0   1.58   -20.9 |     1.5   0.36   -21.1 |   531.9    3.0   -28.3 | YES |
| 6 | `WMA(4,8,17)` | 3MA |   432.2   4.77   -25.5 |    22.3   1.69   -16.6 |    -6.6  -0.37   -19.5 |   508.4    3.1   -29.3 | - |
| 7 | `WMA(6,14,15)` | 3MA |   446.6   5.03   -23.6 |    18.9   1.51   -15.6 |    -8.8  -0.61   -20.4 |   492.7   3.14   -27.2 | - |
| 8 | `WMA(6,13)` | 2MA |   364.8   3.97   -33.9 |    35.6   2.24   -19.5 |    -6.1  -0.24   -19.2 |   491.6   2.77   -33.9 | - |
| 9 | `WMA(6,33)` | 2MA |   367.6   4.53   -25.0 |    19.6   1.52   -15.5 |     4.1   0.58   -19.4 |   482.3   3.07   -27.4 | YES |
| 10 | `WMA(4,37)` | 2MA |   318.9   4.22   -22.7 |    18.5   1.44   -19.5 |    10.3    1.1   -18.7 |   447.5   2.96   -26.3 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `WMA(73,145)` | 2MA |   105.4   3.32   -22.4 |    10.4   1.07   -17.5 |    -7.2  -0.54   -28.4 |   110.4   1.87   -28.4 | - |
| 119 | `WMA(62,239)` | 2MA |    74.9   2.99   -22.2 |    15.8    1.6   -14.3 |     1.2   0.32   -23.8 |   105.0   1.98   -25.6 | YES |
| 120 | `WMA(102,237)` | 2MA |    57.7   2.72   -24.5 |    13.8   1.58   -14.1 |     3.1   0.52   -20.2 |    85.1   1.88   -27.0 | YES |


## 1h x HMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=-0.18**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,62], slow in [3,203] -> 25/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,48], slow in [4,233] -> 17/60 configs positive across TRAIN & VAL & OOS
- band members: HMA(26,32), HMA(2,3), HMA(4,5), HMA(31,53), HMA(3,102), HMA(5,86), HMA(4,86), HMA(10,128), HMA(6,77), HMA(8,91), HMA(12,77), HMA(3,44), HMA(26,75), HMA(2,73), HMA(12,14,75), HMA(2,3,4), HMA(3,5,186), HMA(15,151), HMA(24,43,60), HMA(4,5,75), HMA(12,178), HMA(18,128), HMA(4,199), HMA(5,199) (+18 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `HMA(26,32)` | 2MA |  1033.5   6.66   -27.1 |    38.2   2.46   -16.1 |     1.7   0.38   -15.9 |  1493.2   4.46   -27.1 | YES |
| 2 | `HMA(19,27,38)` | 3MA |   734.7   6.35   -28.9 |    38.9   2.71   -11.4 |    -0.9   0.13   -16.0 |  1048.4   4.31   -28.9 | - |
| 3 | `HMA(22,28)` | 2MA |   717.6   5.52   -26.9 |    50.0   2.96   -15.8 |   -11.2  -0.69   -25.0 |   989.5   3.74   -26.9 | - |
| 4 | `HMA(5,24,34)` | 3MA |   715.6   6.16   -28.4 |    29.8   2.13   -16.1 |    -3.2  -0.06   -17.1 |   924.7   4.02   -28.4 | - |
| 5 | `HMA(2,3)` | 2MA |   465.8   3.57   -56.3 |    68.1   2.92   -23.9 |     6.5   0.71   -30.0 |   913.3   2.85   -56.3 | YES |
| 6 | `HMA(15,22,48)` | 3MA |   672.7   6.04   -29.7 |    35.4   2.56   -16.1 |    -5.7  -0.34   -17.7 |   886.8   4.04   -29.7 | - |
| 7 | `HMA(4,5)` | 2MA |   506.0    3.9   -57.8 |    33.1   1.86   -25.1 |     1.1   0.37   -25.7 |   715.0   2.77   -57.8 | YES |
| 8 | `HMA(10,55)` | 2MA |   520.4   4.91   -32.1 |    36.2   2.36   -20.6 |    -7.2  -0.35   -18.0 |   684.4   3.32   -35.3 | - |
| 9 | `HMA(18,55)` | 2MA |   481.3   4.95   -33.2 |    39.1   2.59   -15.3 |    -5.2  -0.22   -18.1 |   666.0   3.41   -33.2 | - |
| 10 | `HMA(8,39)` | 2MA |   471.7   4.49   -28.8 |    40.2   2.41   -24.8 |    -5.6  -0.18   -17.2 |   656.6   3.11   -32.5 | - |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `HMA(6,14,15)` | 3MA |    -8.9   0.23   -55.8 |    25.3   1.71   -31.5 |    -2.9   0.04   -32.8 |    10.8   0.52   -58.1 | - |
| 119 | `HMA(102,103)` | 2MA |    -2.9  -0.03   -21.7 |     6.2   0.89   -13.4 |     5.2    0.8   -21.1 |     8.4   0.41   -21.7 | - |
| 120 | `HMA(10,11,233)` | 3MA |    -6.5  -0.03   -50.4 |    -7.0  -0.57   -16.6 |     7.9   1.18   -11.5 |    -6.2   0.06   -56.0 | - |


## 1h x DEMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=-0.049**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,52], slow in [18,210] -> 28/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [38,233] -> 15/60 configs positive across TRAIN & VAL & OOS
- band members: DEMA(5,37), DEMA(18,23), DEMA(12,33), DEMA(6,33), DEMA(22,28), DEMA(3,44), DEMA(10,55), DEMA(2,60,208), DEMA(4,60,166), DEMA(3,102), DEMA(19,27,38), DEMA(3,18), DEMA(6,77), DEMA(4,86), DEMA(19,43,233), DEMA(8,91), DEMA(26,32), DEMA(30,43,186), DEMA(2,169), DEMA(18,55), DEMA(5,86), DEMA(5,118,208), DEMA(8,84,186), DEMA(4,199) (+19 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `DEMA(12,13)` | 2MA |   450.9   4.63   -27.8 |    33.6    2.2   -21.1 |    -7.1  -0.33   -19.5 |   583.8    3.1   -30.9 | - |
| 2 | `DEMA(5,37)` | 2MA |   399.5   4.38   -30.4 |    28.7   2.01   -18.5 |     2.0   0.41   -20.2 |   556.1   3.07   -30.4 | YES |
| 3 | `DEMA(10,19)` | 2MA |   440.9   4.64   -31.1 |    30.2   2.06   -20.2 |    -7.9  -0.42   -23.0 |   548.4   3.07   -34.3 | - |
| 4 | `DEMA(8,16)` | 2MA |   379.6   4.16   -30.3 |    41.6   2.54   -20.3 |    -5.4  -0.18   -18.2 |   542.0   2.95   -30.3 | - |
| 5 | `DEMA(18,23)` | 2MA |   335.6   4.35   -35.5 |    36.7   2.55   -14.1 |     5.4   0.69   -17.7 |   527.8   3.21   -35.5 | YES |
| 6 | `DEMA(12,33)` | 2MA |   340.6   4.36   -34.6 |    30.1   2.19   -15.2 |     3.1   0.49   -17.9 |   491.0    3.1   -34.6 | YES |
| 7 | `DEMA(15,28)` | 2MA |   343.0    4.4   -35.3 |    34.0   2.41   -14.1 |    -0.5   0.19   -20.4 |   490.6   3.11   -35.3 | - |
| 8 | `DEMA(8,39)` | 2MA |   382.7   4.51   -32.1 |    25.0   1.88   -17.6 |    -2.3   0.04   -21.3 |   489.6   3.05   -32.1 | - |
| 9 | `DEMA(6,14,15)` | 3MA |   411.8   4.71   -33.0 |    22.5   1.72   -23.3 |    -7.6  -0.46   -22.2 |   479.0   3.04   -35.8 | - |
| 10 | `DEMA(6,13)` | 2MA |   281.7   3.44   -42.9 |    47.2   2.68   -23.7 |    -0.0   0.27   -15.7 |   462.0   2.64   -42.9 | - |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `DEMA(44,75)` | 2MA |   118.2   2.98   -37.1 |    -1.2   0.16   -19.8 |    -2.8  -0.05   -25.3 |   109.5   1.66   -43.0 | - |
| 119 | `DEMA(37,89)` | 2MA |    96.9   2.66   -34.8 |     3.5   0.53   -19.3 |    -1.5   0.07   -24.5 |   100.6   1.59   -40.7 | - |
| 120 | `DEMA(52,89)` | 2MA |    85.8   2.63   -26.9 |    10.5   1.05   -17.0 |    -5.7  -0.33   -26.5 |    93.5   1.58   -32.5 | - |


## 1h x TEMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=-0.062**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,52], slow in [3,210] -> 28/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,48], slow in [27,233] -> 21/60 configs positive across TRAIN & VAL & OOS
- band members: TEMA(2,3), TEMA(22,28), TEMA(18,55), TEMA(26,32), TEMA(15,22,48), TEMA(19,27,38), TEMA(5,86), TEMA(10,55), TEMA(12,77), TEMA(3,102), TEMA(3,12,27), TEMA(31,53), TEMA(37,38), TEMA(8,14,38), TEMA(4,5,75), TEMA(8,91), TEMA(22,65), TEMA(2,5,118), TEMA(3,6), TEMA(4,5), TEMA(5,24,34), TEMA(26,75), TEMA(4,199), TEMA(4,30,67) (+25 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `TEMA(18,23)` | 2MA |   482.5   4.92   -30.5 |    43.7   2.73   -19.4 |    -4.5  -0.13   -16.4 |   699.7   3.43   -34.2 | - |
| 2 | `TEMA(6,14,15)` | 3MA |   422.7   4.43   -27.4 |    61.3   3.51   -22.1 |   -10.0  -0.59   -22.3 |   658.7   3.24   -27.4 | - |
| 3 | `TEMA(15,28)` | 2MA |   440.8   4.73   -30.8 |    39.5   2.54   -19.6 |    -4.2  -0.11   -16.5 |   622.6   3.28   -34.0 | - |
| 4 | `TEMA(2,3)` | 2MA |   402.5   3.38   -54.3 |    31.9   1.74   -31.3 |     7.7   0.77   -27.3 |   613.3    2.5   -54.3 | YES |
| 5 | `TEMA(12,33)` | 2MA |   377.6   4.38   -29.7 |    42.5   2.67   -21.0 |    -2.6   0.03   -17.8 |   562.7   3.13   -34.0 | - |
| 6 | `TEMA(10,19)` | 2MA |   357.7   3.96   -31.6 |    58.5   3.24   -23.1 |    -9.7  -0.51   -18.0 |   555.1   2.92   -31.6 | - |
| 7 | `TEMA(22,28)` | 2MA |   327.6   4.18   -35.9 |    43.6    2.8   -17.3 |     6.5   0.77   -14.5 |   553.7   3.18   -38.5 | YES |
| 8 | `TEMA(18,55)` | 2MA |   296.5   4.14   -30.7 |    47.3   3.12   -13.6 |     6.1   0.75   -18.6 |   520.0   3.23   -30.7 | YES |
| 9 | `TEMA(15,65)` | 2MA |   334.2   4.37   -32.4 |    42.2   2.85   -14.0 |    -2.6   -0.0   -21.6 |   501.8   3.16   -32.4 | - |
| 10 | `TEMA(26,32)` | 2MA |   244.8    3.7   -35.4 |    51.0   3.25   -12.9 |    15.5   1.48   -15.9 |   501.2   3.13   -35.4 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `TEMA(75,132,148)` | 3MA |    66.5   2.64   -15.6 |     4.7   0.65   -13.2 |    -6.5  -0.51   -23.1 |    63.0   1.39   -23.1 | - |
| 119 | `TEMA(86,122)` | 2MA |    65.4   2.35   -28.8 |     2.5   0.45   -20.1 |    -8.4  -0.61   -24.7 |    55.4   1.18   -39.3 | - |
| 120 | `TEMA(102,103)` | 2MA |    64.8   2.35   -29.0 |     1.2   0.33   -21.0 |    -7.2   -0.5   -24.2 |    54.7   1.18   -40.2 | - |


## 1h x KAMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=-0.426**; TRAIN+VAL top-10 -> OOS top-10 overlap = **1/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,237] -> 16/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [22,233] -> 13/60 configs positive across TRAIN & VAL & OOS
- band members: KAMA(102,103), KAMA(18,23), KAMA(37,38), KAMA(2,3), KAMA(15,22,48), KAMA(19,27,38), KAMA(22,65), KAMA(3,102), KAMA(86,122), KAMA(44,75), KAMA(5,118,208), KAMA(12,22,118), KAMA(6,22,24), KAMA(2,19,22), KAMA(15,65), KAMA(52,89), KAMA(15,28), KAMA(18,128), KAMA(24,43,60), KAMA(30,132,186), KAMA(15,84,148), KAMA(19,132,233), KAMA(102,237), KAMA(12,43,75) (+5 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `KAMA(4,5)` | 2MA |   362.7   3.71   -48.1 |    35.1   2.18   -22.9 |    -7.4  -0.29   -36.2 |   478.6   2.61   -48.1 | - |
| 2 | `KAMA(102,103)` | 2MA |   274.2   4.65   -29.3 |    25.3   2.24   -16.3 |    13.6   1.44   -14.9 |   432.7   3.44   -29.3 | YES |
| 3 | `KAMA(12,13)` | 2MA |   427.5   4.43   -41.2 |    13.3   1.11   -22.1 |   -12.4   -0.8   -34.0 |   423.9   2.68   -48.0 | - |
| 4 | `KAMA(18,23)` | 2MA |   235.1   4.06   -22.7 |    25.5   1.92   -20.4 |    17.1   1.65   -21.9 |   392.2   3.05   -23.7 | YES |
| 5 | `KAMA(37,38)` | 2MA |   184.5   3.33   -39.0 |    54.8   3.27   -15.1 |     7.6   0.86   -22.5 |   373.7   2.81   -39.0 | YES |
| 6 | `KAMA(3,6)` | 2MA |   313.5   3.61   -32.8 |    23.9   1.73   -22.3 |   -12.0  -0.72   -37.6 |   350.9   2.38   -37.6 | - |
| 7 | `KAMA(2,3)` | 2MA |   214.4   2.82   -50.9 |    34.0   2.01   -22.7 |     4.2   0.57   -25.8 |   339.1    2.2   -51.3 | YES |
| 8 | `KAMA(8,22,60)` | 3MA |   224.9   4.73   -14.2 |    31.4   2.74   -13.0 |    -1.0   0.03   -17.5 |   322.9   3.36   -17.5 | - |
| 9 | `KAMA(5,10,14)` | 3MA |   285.8   4.75   -24.2 |    12.9   1.33   -14.8 |    -5.2  -0.44   -21.2 |   312.9   3.02   -30.2 | - |
| 10 | `KAMA(5,12)` | 2MA |   339.1   4.43   -26.4 |    11.5   1.07   -19.5 |   -18.4  -1.55   -33.7 |   299.5   2.52   -33.7 | - |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `KAMA(62,105)` | 2MA |    60.1   2.45   -35.0 |    14.1   1.42   -14.7 |    -4.1  -0.26   -22.6 |    75.2   1.56   -42.6 | - |
| 119 | `KAMA(22,151)` | 2MA |    59.4   2.42   -24.2 |     8.6   0.99   -17.7 |    -1.6   0.03   -24.6 |    70.4   1.49   -26.2 | - |
| 120 | `KAMA(73,145)` | 2MA |    51.2   2.47   -27.5 |    13.2    1.5   -11.1 |    -6.2  -0.66   -21.6 |    60.5   1.51   -30.7 | - |


## 1h x VIDYA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=-0.133**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,44], slow in [10,210] -> 38/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,60], slow in [14,233] -> 33/60 configs positive across TRAIN & VAL & OOS
- band members: VIDYA(2,5,118), VIDYA(4,5,75), VIDYA(3,5,186), VIDYA(4,8,17), VIDYA(2,10), VIDYA(2,12,14), VIDYA(4,15), VIDYA(5,10,14), VIDYA(4,10,208), VIDYA(6,13), VIDYA(5,12), VIDYA(8,14,38), VIDYA(2,19,22), VIDYA(3,18), VIDYA(6,14,15), VIDYA(8,16), VIDYA(4,15,208), VIDYA(6,22,24), VIDYA(12,13), VIDYA(2,30,84), VIDYA(2,26), VIDYA(10,11,233), VIDYA(12,14,75), VIDYA(3,44) (+47 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `VIDYA(2,3)` | 2MA |   336.5   4.42   -23.1 |    28.5   2.06   -16.6 |    -6.1  -0.33   -19.2 |   426.8   2.96   -24.9 | - |
| 2 | `VIDYA(2,5,118)` | 3MA |   249.2   5.02   -17.0 |    44.6   3.57    -7.2 |     2.4   0.48   -11.3 |   417.2   3.84   -21.2 | YES |
| 3 | `VIDYA(2,3,4)` | 3MA |   260.9   4.05   -26.2 |    33.2   2.44   -16.4 |    -7.2  -0.52   -21.6 |   346.3   2.81   -28.5 | - |
| 4 | `VIDYA(3,4,38)` | 3MA |   239.3   4.64   -23.5 |    33.1   2.73   -11.8 |    -4.3   -0.4   -18.2 |   332.2   3.26   -24.3 | - |
| 5 | `VIDYA(3,6)` | 2MA |   244.9   3.87   -28.9 |    24.3   1.87   -16.3 |    -0.1    0.2   -22.0 |   328.2    2.7   -33.6 | - |
| 6 | `VIDYA(2,4,19)` | 3MA |   247.9   4.55   -26.6 |    25.8   2.26   -10.3 |    -3.0  -0.17   -19.0 |   324.6   3.12   -28.6 | - |
| 7 | `VIDYA(4,5,75)` | 3MA |   198.9   4.44   -20.0 |    34.6   2.83   -10.4 |     2.6   0.48   -15.0 |   312.8    3.3   -27.0 | YES |
| 8 | `VIDYA(4,5)` | 2MA |   233.4    3.8   -28.9 |    26.1   1.98   -15.7 |    -2.8  -0.06   -23.6 |   308.4   2.64   -33.1 | - |
| 9 | `VIDYA(3,5,186)` | 3MA |   196.1   4.39   -18.3 |    31.5   2.77   -10.3 |     3.2   0.59   -10.4 |   302.1    3.3   -24.8 | YES |
| 10 | `VIDYA(2,8,22)` | 3MA |   231.3    4.5   -24.0 |    22.7   1.95   -14.0 |    -3.3  -0.22   -21.1 |   292.9   2.99   -24.6 | - |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `VIDYA(62,239)` | 2MA |     4.2   0.82    -3.7 |    12.0   2.14    -6.4 |    -2.9  -0.67    -7.2 |    13.3   0.88    -7.2 | - |
| 119 | `VIDYA(37,203)` | 2MA |     1.3   0.26    -6.2 |    16.3   2.41    -6.6 |    -3.9   -0.7   -10.2 |    13.2   0.75   -10.2 | - |
| 120 | `VIDYA(52,203)` | 2MA |     1.1   0.25    -5.7 |    16.5   2.61    -6.6 |    -4.4  -0.91    -9.9 |    12.6   0.78    -9.9 | - |


# Timeframe: 30m
_Benchmark (equal-weight u10 buy-hold, no cost): FULL-2020 net = 1312.6% (participation-tax reference)._

## 30m x EMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.156**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [12,239] -> 29/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [14,233] -> 26/60 configs positive across TRAIN & VAL & OOS
- band members: EMA(10,19), EMA(12,13), EMA(6,33), EMA(6,14,15), EMA(4,37), EMA(19,43,233), EMA(8,34,233), EMA(10,34,60), EMA(19,27,38), EMA(22,28), EMA(2,8,22), EMA(6,9,24), EMA(10,55), EMA(2,12,14), EMA(15,22,48), EMA(6,67,233), EMA(10,67,148), EMA(26,32), EMA(8,14,38), EMA(2,73), EMA(4,60,166), EMA(5,10,14), EMA(6,77), EMA(37,38) (+31 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `EMA(10,19)` | 2MA |   279.2   3.94   -23.2 |    30.2    2.1   -14.3 |     2.4   0.43   -18.1 |   405.4   2.83   -23.3 | YES |
| 2 | `EMA(12,13)` | 2MA |   277.6   3.87   -20.3 |    28.2   1.98   -15.7 |     1.2   0.33   -17.6 |   390.0   2.75   -23.7 | YES |
| 3 | `EMA(6,33)` | 2MA |   262.5   3.89   -22.1 |    29.3   2.05   -16.1 |     2.9   0.47   -18.4 |   382.3   2.79   -26.4 | YES |
| 4 | `EMA(15,28)` | 2MA |   294.1   4.28   -20.5 |    24.7   1.83   -14.5 |    -2.4   -0.0   -21.9 |   379.7   2.87   -24.4 | - |
| 5 | `EMA(8,39)` | 2MA |   298.3   4.27   -20.9 |    26.2   1.91   -15.2 |    -5.5  -0.29   -22.5 |   375.1   2.83   -25.2 | - |
| 6 | `EMA(12,33)` | 2MA |   284.0    4.2   -20.8 |    25.9    1.9   -14.7 |    -3.0  -0.05   -22.0 |   368.9   2.83   -24.3 | - |
| 7 | `EMA(18,23)` | 2MA |   277.2   4.14   -20.9 |    25.0   1.85   -15.2 |    -1.0   0.13   -19.9 |   366.8   2.82   -24.7 | - |
| 8 | `EMA(6,14,15)` | 3MA |   255.5    3.9   -23.5 |    24.7   1.88   -16.2 |     4.0   0.58   -15.2 |   361.1   2.79   -28.3 | YES |
| 9 | `EMA(4,37)` | 2MA |   263.3   3.86   -23.0 |    22.5   1.66   -18.1 |     2.4   0.43   -20.4 |   355.9   2.67   -28.5 | YES |
| 10 | `EMA(5,37)` | 2MA |   263.3   3.89   -19.8 |    21.4   1.61   -18.3 |    -3.8  -0.12   -20.0 |   324.2   2.58   -25.6 | - |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `EMA(3,6)` | 2MA |    57.2   1.43   -54.4 |     3.4   0.55   -27.9 |   -13.5  -0.74   -27.8 |    40.6   0.83   -60.8 | - |
| 119 | `EMA(2,3,4)` | 3MA |    87.5   1.75   -60.2 |    -3.8   0.18   -34.2 |   -25.0  -1.64   -35.1 |    35.2   0.79   -65.9 | - |
| 120 | `EMA(2,3)` | 2MA |    63.4   1.46   -66.5 |    -3.4   0.22   -37.1 |   -24.8  -1.48   -34.5 |    18.6   0.65   -72.0 | - |


## 30m x SMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=-0.181**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,37], slow in [12,169] -> 17/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,30], slow in [14,233] -> 10/60 configs positive across TRAIN & VAL & OOS
- band members: SMA(5,6,34), SMA(3,44), SMA(10,55), SMA(6,77), SMA(5,86), SMA(4,86), SMA(37,38), SMA(2,73), SMA(3,8,43), SMA(2,26), SMA(30,43,186), SMA(8,91), SMA(4,60,166), SMA(4,5,75), SMA(6,67,233), SMA(10,128), SMA(6,14,15), SMA(3,18), SMA(2,169), SMA(3,102), SMA(2,12,14), SMA(2,4,19), SMA(4,15), SMA(6,13) (+3 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `SMA(22,28)` | 2MA |   693.1   5.83   -24.7 |    25.5   1.84   -14.7 |   -17.6  -1.41   -27.9 |   719.8   3.53   -28.9 | - |
| 2 | `SMA(3,4,38)` | 3MA |   484.0   5.17   -22.0 |    22.3   1.75   -14.6 |    -9.3  -0.68   -21.1 |   547.8   3.29   -24.2 | - |
| 3 | `SMA(15,28)` | 2MA |   427.5   4.87   -24.9 |    51.1   3.16   -12.1 |   -19.9  -1.76   -29.8 |   538.3   3.22   -29.8 | - |
| 4 | `SMA(12,33)` | 2MA |   450.1   5.08   -24.1 |    34.4   2.36   -14.0 |   -13.9  -1.12   -22.6 |   536.9   3.27   -24.1 | - |
| 5 | `SMA(6,33)` | 2MA |   378.0   4.55   -21.8 |    39.1   2.56   -14.9 |    -7.0  -0.41   -20.8 |   518.6   3.14   -21.8 | - |
| 6 | `SMA(5,6,34)` | 3MA |   402.2   4.87   -25.2 |    15.7   1.36   -17.0 |     2.5   0.45   -16.9 |   495.9   3.23   -30.2 | YES |
| 7 | `SMA(5,37)` | 2MA |   380.5   4.61   -22.2 |    30.5   2.12   -14.1 |    -5.7  -0.29   -21.0 |   491.2   3.09   -23.3 | - |
| 8 | `SMA(8,39)` | 2MA |   378.3   4.68   -22.3 |    30.8   2.16   -13.9 |    -6.0  -0.34   -19.6 |   487.7   3.13   -22.3 | - |
| 9 | `SMA(4,37)` | 2MA |   341.5   4.33   -22.7 |    30.4   2.09   -13.8 |    -2.0   0.05   -20.9 |   464.0   2.98   -23.3 | - |
| 10 | `SMA(3,44)` | 2MA |   337.0   4.36   -23.3 |    21.2    1.6   -15.6 |     6.0   0.73   -17.4 |   461.2    3.0   -26.3 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `SMA(4,8,17)` | 3MA |    21.3   0.89   -54.8 |    -4.3  -0.02   -23.4 |     5.7   0.72   -12.1 |    22.7   0.64   -62.3 | - |
| 119 | `SMA(2,10)` | 2MA |    46.2   1.28   -57.4 |    -5.1    0.1   -29.8 |   -11.7   -0.6   -28.4 |    22.6   0.66   -66.6 | - |
| 120 | `SMA(5,10,14)` | 3MA |     4.6   0.53   -54.4 |     3.3   0.52   -27.1 |     9.5   1.01   -11.1 |    18.4   0.59   -65.1 | YES |


## 30m x WMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=-0.237**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,37], slow in [13,210] -> 22/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [4,38], slow in [15,233] -> 8/60 configs positive across TRAIN & VAL & OOS
- band members: WMA(18,55), WMA(6,77), WMA(5,37), WMA(12,77), WMA(3,102), WMA(4,86), WMA(5,86), WMA(4,37), WMA(8,91), WMA(6,67,233), WMA(38,67,233), WMA(10,128), WMA(12,106,148), WMA(5,118,208), WMA(15,151), WMA(2,169), WMA(18,128), WMA(19,132,233), WMA(6,210), WMA(12,178), WMA(5,199), WMA(4,30,67), WMA(37,203), WMA(4,199) (+6 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `WMA(26,32)` | 2MA |   443.5   5.08   -19.4 |    37.1   2.53   -13.4 |   -16.9  -1.44   -25.1 |   519.5   3.25   -26.3 | - |
| 2 | `WMA(18,55)` | 2MA |   331.5   4.52   -24.3 |    24.9   1.86   -14.4 |     1.4   0.35   -19.1 |   446.5   3.08   -26.6 | YES |
| 3 | `WMA(10,55)` | 2MA |   299.2   4.19   -22.2 |    39.5    2.6   -13.2 |    -4.2  -0.16   -20.5 |   433.5   2.98   -22.7 | - |
| 4 | `WMA(8,39)` | 2MA |   321.6   4.17   -21.0 |    32.0   2.18   -16.0 |    -4.3  -0.16   -21.3 |   432.6   2.88   -24.5 | - |
| 5 | `WMA(15,65)` | 2MA |   327.3    4.5   -23.1 |    23.5   1.78   -13.4 |    -0.1   0.21   -21.1 |   427.3   3.03   -24.8 | - |
| 6 | `WMA(22,28)` | 2MA |   349.7   4.44   -23.4 |    31.3    2.2   -14.3 |   -15.0  -1.21   -28.2 |   402.0   2.85   -28.2 | - |
| 7 | `WMA(12,33)` | 2MA |   305.8   4.08   -25.8 |    30.4   2.13   -16.3 |    -8.0  -0.51   -22.3 |   387.2   2.76   -27.8 | - |
| 8 | `WMA(19,27,38)` | 3MA |   350.9   4.95   -20.9 |    22.9   1.89   -13.2 |   -13.8  -1.31   -23.7 |   377.9   3.06   -23.7 | - |
| 9 | `WMA(3,44)` | 2MA |   294.1    3.9   -21.9 |    22.9   1.66   -16.3 |    -2.5   0.03   -19.5 |   372.1   2.63   -27.8 | - |
| 10 | `WMA(37,38)` | 2MA |   298.5   4.32   -24.2 |    21.0   1.64   -14.9 |    -5.1  -0.26   -17.6 |   357.8    2.8   -28.8 | - |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `WMA(3,6)` | 2MA |    65.2   1.49   -60.6 |   -13.4  -0.34   -39.8 |   -18.4  -1.01   -33.0 |    16.9   0.62   -67.2 | - |
| 119 | `WMA(2,10)` | 2MA |    83.1   1.71   -60.4 |   -20.7   -0.8   -38.9 |   -20.2  -1.22   -34.8 |    15.9   0.61   -69.6 | - |
| 120 | `WMA(4,5)` | 2MA |    67.1   1.51   -61.8 |   -10.6  -0.17   -38.8 |   -22.5  -1.27   -34.8 |    15.8   0.62   -68.2 | - |


## 30m x HMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.413**; TRAIN+VAL top-10 -> OOS top-10 overlap = **1/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 21/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [4,19], slow in [60,233] -> 10/60 configs positive across TRAIN & VAL & OOS
- band members: HMA(2,3), HMA(4,5,75), HMA(8,210), HMA(5,199), HMA(4,86), HMA(6,210), HMA(12,178), HMA(4,199), HMA(8,91), HMA(10,128), HMA(22,151), HMA(62,105), HMA(15,151), HMA(10,17,233), HMA(5,86), HMA(4,10,208), HMA(2,169), HMA(52,203), HMA(6,77), HMA(73,145), HMA(10,34,60), HMA(8,9,75), HMA(62,239), HMA(8,84,186) (+7 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `HMA(37,89)` | 2MA |   545.3   5.48   -26.2 |    55.4   3.38   -15.7 |    -4.9  -0.19   -17.4 |   853.3   3.86   -26.8 | - |
| 2 | `HMA(44,75)` | 2MA |   579.4   5.55   -27.3 |    46.2   2.95   -15.5 |    -5.3  -0.22   -15.8 |   840.4    3.8   -27.3 | - |
| 3 | `HMA(52,89)` | 2MA |   538.6   5.55   -30.3 |    43.3   2.87   -14.0 |    -2.8   -0.0   -15.3 |   790.0   3.81   -31.4 | - |
| 4 | `HMA(2,3)` | 2MA |   426.3   3.41   -65.7 |    41.5   2.06   -26.8 |     4.1   0.59   -36.0 |   675.1   2.53   -65.7 | YES |
| 5 | `HMA(4,5,75)` | 3MA |   344.9   4.82   -25.8 |    40.7   3.19   -16.1 |    11.1   1.25   -10.6 |   595.1   3.73   -26.7 | YES |
| 6 | `HMA(4,5)` | 2MA |   312.1   3.12   -59.0 |    66.4   2.99   -21.4 |    -0.4   0.32   -33.1 |   583.0   2.52   -61.1 | - |
| 7 | `HMA(26,75)` | 2MA |   318.0   4.07   -29.7 |    55.1   3.33   -17.5 |    -1.2   0.14   -13.8 |   540.8   3.12   -29.7 | - |
| 8 | `HMA(8,210)` | 2MA |   289.0   4.02   -26.1 |    36.3   2.41   -18.0 |    15.8   1.48   -18.9 |   514.0   3.12   -26.1 | YES |
| 9 | `HMA(31,124)` | 2MA |   341.5   4.42   -29.7 |    48.1   3.08   -15.2 |    -6.6  -0.36   -19.6 |   510.3   3.17   -33.2 | - |
| 10 | `HMA(5,199)` | 2MA |   259.4   3.73   -25.7 |    39.5   2.52   -16.0 |    16.8   1.52   -20.2 |   485.3   2.99   -25.7 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `HMA(3,4,38)` | 3MA |    20.2   0.87   -65.4 |   -17.8  -0.73   -33.6 |   -24.9  -1.62   -40.7 |   -25.9   0.06   -77.8 | - |
| 119 | `HMA(8,14,38)` | 3MA |    -3.4   0.36   -64.1 |   -12.0   -0.5   -28.1 |   -17.7  -1.18   -35.4 |   -30.1  -0.11   -74.1 | - |
| 120 | `HMA(6,22,24)` | 3MA |     8.3   0.64   -59.6 |   -28.7  -1.65   -34.8 |   -12.7  -0.79   -33.2 |   -32.6  -0.12   -75.5 | - |


## 30m x DEMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.417**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [33,210] -> 35/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [3,60], slow in [38,233] -> 14/60 configs positive across TRAIN & VAL & OOS
- band members: DEMA(26,75), DEMA(31,53), DEMA(18,55), DEMA(19,27,38), DEMA(37,38), DEMA(12,77), DEMA(15,65), DEMA(8,91), DEMA(15,22,48), DEMA(4,86), DEMA(5,86), DEMA(3,102), DEMA(2,73), DEMA(15,151), DEMA(44,75), DEMA(6,77), DEMA(37,89), DEMA(2,169), DEMA(8,210), DEMA(6,210), DEMA(8,39), DEMA(18,128), DEMA(5,199), DEMA(4,199) (+25 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `DEMA(26,75)` | 2MA |   381.2   4.85   -24.2 |    35.5    2.5   -13.2 |     0.7   0.29   -22.6 |   556.5   3.39   -24.2 | YES |
| 2 | `DEMA(26,32)` | 2MA |   395.6   4.68   -26.4 |    36.8   2.48   -17.7 |    -5.1  -0.21   -21.9 |   543.1   3.21   -32.2 | - |
| 3 | `DEMA(31,53)` | 2MA |   329.9   4.48   -31.0 |    43.7   2.92   -13.7 |     3.7   0.54   -20.2 |   540.8   3.33   -31.0 | YES |
| 4 | `DEMA(18,55)` | 2MA |   332.4   4.35   -24.6 |    39.9   2.68   -16.3 |     2.7   0.46   -20.8 |   521.2   3.19   -25.8 | YES |
| 5 | `DEMA(19,27,38)` | 3MA |   344.7   4.68   -25.2 |    37.8   2.78   -14.8 |     0.2   0.23   -15.7 |   513.8   3.38   -25.2 | YES |
| 6 | `DEMA(37,38)` | 2MA |   315.3   4.36   -29.2 |    30.1   2.19   -15.8 |     6.2   0.75   -18.8 |   473.9   3.13   -31.0 | YES |
| 7 | `DEMA(12,77)` | 2MA |   297.1    4.1   -25.5 |    33.9   2.36   -17.1 |     4.7   0.63   -20.7 |   456.7    3.0   -25.5 | YES |
| 8 | `DEMA(15,65)` | 2MA |   279.5   3.98   -24.1 |    35.1   2.42   -18.5 |     2.3   0.42   -20.2 |   424.1   2.91   -25.8 | YES |
| 9 | `DEMA(22,28)` | 2MA |   305.3   4.04   -26.2 |    37.8    2.5   -19.9 |    -6.2  -0.29   -20.0 |   423.5   2.83   -30.3 | - |
| 10 | `DEMA(22,65)` | 2MA |   284.0   4.12   -27.0 |    34.9   2.47   -14.2 |    -0.4    0.2   -22.8 |   416.2   2.95   -27.0 | - |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `DEMA(4,8,17)` | 3MA |    26.3   0.98   -58.5 |     0.1   0.33   -24.7 |   -26.9  -1.99   -37.8 |    -7.6   0.28   -67.3 | - |
| 119 | `DEMA(2,4,19)` | 3MA |    36.1   1.14   -58.1 |   -16.6  -0.65   -31.4 |   -24.8  -1.69   -39.5 |   -14.6   0.22   -72.9 | - |
| 120 | `DEMA(2,10)` | 2MA |    25.9   0.97   -68.1 |   -10.9  -0.19   -40.0 |   -29.3  -1.69   -41.2 |   -20.7   0.22   -75.8 | - |


## 30m x TEMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.352**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [4,102], slow in [32,239] -> 32/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,75], slow in [48,233] -> 25/60 configs positive across TRAIN & VAL & OOS
- band members: TEMA(31,124), TEMA(22,151), TEMA(18,55), TEMA(52,89), TEMA(26,75), TEMA(22,65), TEMA(37,89), TEMA(15,65), TEMA(26,172), TEMA(44,75), TEMA(18,128), TEMA(2,5,118), TEMA(15,151), TEMA(26,32), TEMA(62,105), TEMA(24,43,60), TEMA(10,128), TEMA(15,22,48), TEMA(4,199), TEMA(12,77), TEMA(4,30,67), TEMA(12,178), TEMA(8,91), TEMA(30,43,186) (+33 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `TEMA(37,38)` | 2MA |   410.0   4.88   -25.3 |    48.2   3.03   -17.1 |    -1.4   0.12   -12.8 |   644.9   3.48   -27.9 | - |
| 2 | `TEMA(31,124)` | 2MA |   342.0   4.61   -26.5 |    42.6   2.89   -13.8 |     0.8   0.29   -21.0 |   535.0   3.35   -26.5 | YES |
| 3 | `TEMA(31,53)` | 2MA |   381.0   4.74   -26.7 |    34.6   2.36   -18.3 |    -3.8  -0.08   -16.5 |   522.9   3.22   -33.4 | - |
| 4 | `TEMA(22,151)` | 2MA |   312.7   4.34   -23.9 |    38.5   2.63   -16.1 |     1.4   0.34   -16.9 |   479.2   3.15   -23.9 | YES |
| 5 | `TEMA(18,55)` | 2MA |   275.5   3.79   -25.9 |    42.9   2.73   -18.9 |     5.5   0.68   -10.6 |   465.7   2.92   -29.2 | YES |
| 6 | `TEMA(52,89)` | 2MA |   274.4   4.25   -32.1 |    43.5   2.95   -13.9 |     3.5   0.53   -21.3 |   456.0    3.2   -32.1 | YES |
| 7 | `TEMA(26,75)` | 2MA |   282.2   4.05   -27.2 |    34.8   2.38   -17.5 |     5.5   0.69   -17.2 |   443.6   2.99   -33.4 | YES |
| 8 | `TEMA(22,65)` | 2MA |   295.4   4.05   -25.6 |    30.6   2.14   -19.2 |     3.9   0.56   -16.1 |   436.6   2.91   -32.9 | YES |
| 9 | `TEMA(37,89)` | 2MA |   260.8   4.05   -30.2 |    35.7    2.5   -13.6 |     8.4   0.93   -17.9 |   431.1   3.05   -30.2 | YES |
| 10 | `TEMA(15,65)` | 2MA |   259.1   3.65   -26.1 |    36.1   2.37   -18.2 |     2.9   0.47   -13.6 |   402.8   2.72   -27.1 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `TEMA(2,12,14)` | 3MA |    26.7   0.98   -61.9 |   -10.5  -0.28   -33.4 |   -19.8  -1.23   -37.1 |    -9.0   0.28   -70.5 | - |
| 119 | `TEMA(12,13)` | 2MA |    15.7   0.79   -63.3 |   -10.6  -0.25   -32.0 |   -16.9  -0.99   -34.4 |   -14.0   0.22   -71.5 | - |
| 120 | `TEMA(6,14,15)` | 3MA |    23.7   0.93   -62.9 |   -18.9  -0.88   -34.5 |   -20.8  -1.49   -38.2 |   -20.5   0.09   -72.6 | - |


## 30m x KAMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=-0.147**; TRAIN+VAL top-10 -> OOS top-10 overlap = **2/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [6,210] -> 18/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [3,38], slow in [60,233] -> 12/60 configs positive across TRAIN & VAL & OOS
- band members: KAMA(37,38), KAMA(3,6), KAMA(102,103), KAMA(8,210), KAMA(12,33), KAMA(6,210), KAMA(4,199), KAMA(8,91), KAMA(31,53), KAMA(5,86), KAMA(2,169), KAMA(5,199), KAMA(10,17,233), KAMA(19,43,233), KAMA(30,43,186), KAMA(8,39), KAMA(8,34,233), KAMA(5,37), KAMA(3,75,132), KAMA(5,38,208), KAMA(86,122), KAMA(38,67,233), KAMA(4,15), KAMA(15,84,148) (+6 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `KAMA(37,38)` | 2MA |   216.7    3.4   -32.0 |    45.4   2.81   -18.0 |     3.1   0.49   -21.0 |   374.8   2.66   -35.6 | YES |
| 2 | `KAMA(26,32)` | 2MA |   335.2    4.4   -29.4 |    -4.8  -0.09   -25.8 |    14.4   1.37   -12.9 |   374.1   2.78   -33.0 | - |
| 3 | `KAMA(18,23)` | 2MA |   300.6   4.03   -24.3 |    33.1    2.4   -22.3 |   -16.4  -0.97   -33.2 |   345.7   2.58   -36.4 | - |
| 4 | `KAMA(3,6)` | 2MA |   279.1   3.47   -28.7 |     4.7   0.61   -26.6 |     0.6   0.31   -20.1 |   299.5   2.22   -35.5 | YES |
| 5 | `KAMA(102,103)` | 2MA |   258.3   3.99   -38.6 |     5.4   0.67   -18.5 |     1.5   0.37   -29.7 |   283.3    2.5   -38.6 | YES |
| 6 | `KAMA(19,27,38)` | 3MA |   210.1   4.46   -18.7 |    23.4   2.28   -14.4 |    -1.3   0.02   -17.1 |   277.9   3.08   -22.5 | - |
| 7 | `KAMA(10,19)` | 2MA |   249.1   3.87   -26.0 |    16.4   1.36   -15.5 |    -8.3  -0.39   -27.6 |   272.9   2.39   -28.2 | - |
| 8 | `KAMA(3,102)` | 2MA |   194.2   3.71   -18.4 |    21.2   1.64   -18.7 |    -0.0   0.21   -16.4 |   256.5    2.5   -23.8 | - |
| 9 | `KAMA(8,16)` | 2MA |   232.2   3.62   -26.1 |    11.4   1.05   -20.8 |    -5.5  -0.28   -24.6 |   249.7   2.29   -30.8 | - |
| 10 | `KAMA(2,3)` | 2MA |   224.6   2.85   -51.6 |    17.1   1.24   -27.2 |    -8.2  -0.26   -25.7 |   248.7   1.89   -53.1 | - |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `KAMA(62,239)` | 2MA |    75.5   2.71   -20.7 |     0.6   0.28   -22.5 |    -5.2  -0.27   -24.6 |    67.4   1.38   -27.5 | - |
| 119 | `KAMA(30,132,186)` | 3MA |    70.2   2.77   -16.6 |     0.9   0.29   -10.5 |    -4.5  -0.41   -16.2 |    64.0   1.46   -23.5 | - |
| 120 | `KAMA(24,106,118)` | 3MA |    25.8    1.2   -28.9 |    11.2   1.26   -15.1 |    -1.7  -0.04   -16.4 |    37.5   0.96   -31.7 | - |


## 30m x VIDYA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.03**; TRAIN+VAL top-10 -> OOS top-10 overlap = **1/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [5,210] -> 47/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,60], slow in [14,233] -> 50/60 configs positive across TRAIN & VAL & OOS
- band members: VIDYA(4,5), VIDYA(2,10), VIDYA(3,6), VIDYA(6,13), VIDYA(5,10,14), VIDYA(3,12,27), VIDYA(5,6,34), VIDYA(6,9,24), VIDYA(2,12,14), VIDYA(4,10,208), VIDYA(4,8,17), VIDYA(4,15), VIDYA(3,5,186), VIDYA(2,8,22), VIDYA(6,14,15), VIDYA(2,26), VIDYA(4,5,75), VIDYA(2,4,19), VIDYA(5,12), VIDYA(3,18), VIDYA(2,5,118), VIDYA(8,14,38), VIDYA(8,22,60), VIDYA(2,19,22) (+73 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `VIDYA(4,5)` | 2MA |   288.0   4.03   -21.2 |    30.3   2.16   -18.5 |     2.3   0.43   -15.1 |   417.4    2.9   -26.7 | YES |
| 2 | `VIDYA(2,10)` | 2MA |   281.4    4.0   -21.5 |    18.1   1.45   -18.8 |     2.9   0.47   -15.4 |   363.5   2.74   -29.7 | YES |
| 3 | `VIDYA(3,6)` | 2MA |   266.4   3.86   -22.8 |    26.0    1.9   -20.3 |     0.3   0.24   -15.8 |   362.7   2.71   -29.9 | YES |
| 4 | `VIDYA(6,13)` | 2MA |   212.7   3.72   -18.5 |    22.1   1.71   -15.2 |    10.6   1.17   -16.8 |   322.4   2.73   -22.0 | YES |
| 5 | `VIDYA(5,10,14)` | 3MA |   200.1   3.76   -17.7 |    34.4   2.51   -12.4 |     3.3   0.53   -20.7 |   316.7   2.84   -20.7 | YES |
| 6 | `VIDYA(3,12,27)` | 3MA |   195.4   3.97   -19.3 |    31.3   2.45   -11.5 |     5.1   0.74   -18.1 |   307.6   2.98   -19.3 | YES |
| 7 | `VIDYA(5,6,34)` | 3MA |   212.1   4.14   -21.9 |    24.2   2.02    -9.4 |     4.2   0.65   -16.8 |   303.8   2.97   -22.2 | YES |
| 8 | `VIDYA(6,9,24)` | 3MA |   188.6   3.79   -17.7 |    33.8   2.53   -10.9 |     1.8   0.38   -21.2 |   293.1   2.83   -21.2 | YES |
| 9 | `VIDYA(2,12,14)` | 3MA |   196.7   3.79   -20.3 |    25.0    2.0   -11.7 |     5.1   0.73   -16.5 |   289.9   2.78   -21.3 | YES |
| 10 | `VIDYA(4,10,208)` | 3MA |   155.6   3.89   -13.8 |    40.0   3.18    -9.0 |     6.7   1.02    -9.9 |   281.6   3.16   -16.8 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `VIDYA(102,237)` | 2MA |    24.5   2.32    -8.3 |     6.3   1.22    -8.3 |    -2.5  -0.39    -9.0 |    29.1   1.35   -14.1 | - |
| 119 | `VIDYA(60,67,233)` | 3MA |    17.5   1.32   -15.1 |     3.2   0.59   -10.3 |     3.8   0.64   -14.7 |    25.9   0.96   -23.5 | YES |
| 120 | `VIDYA(186,208,233)` | 3MA |     5.2   0.83    -9.0 |    10.9   2.09    -6.1 |    -6.6  -1.71   -10.4 |     8.9   0.62   -10.4 | - |


# Timeframe: 15m
_Benchmark (equal-weight u10 buy-hold, no cost): FULL-2020 net = 1346.0% (participation-tax reference)._

## 15m x EMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.665**; TRAIN+VAL top-10 -> OOS top-10 overlap = **1/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,44], slow in [33,210] -> 16/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [15,30], slow in [60,233] -> 4/60 configs positive across TRAIN & VAL & OOS
- band members: EMA(37,38), EMA(15,65), EMA(26,75), EMA(10,55), EMA(15,84,148), EMA(37,89), EMA(30,43,186), EMA(44,75), EMA(10,128), EMA(24,43,60), EMA(2,169), EMA(18,128), EMA(12,33), EMA(15,151), EMA(12,178), EMA(5,199), EMA(19,132,233), EMA(4,199), EMA(6,210), EMA(8,210)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `EMA(26,32)` | 2MA |   249.7    3.9   -21.4 |    31.0   2.23   -13.0 |    -0.0   0.22   -20.9 |   358.0   2.79   -22.3 | - |
| 2 | `EMA(22,28)` | 2MA |   229.7   3.69   -19.7 |    22.8   1.75   -16.6 |    -0.9   0.14   -20.2 |   301.2   2.55   -25.0 | - |
| 3 | `EMA(37,38)` | 2MA |   205.8   3.62   -20.1 |    28.0   2.08   -13.9 |     0.5   0.26   -19.3 |   293.3   2.59   -22.3 | YES |
| 4 | `EMA(15,65)` | 2MA |   199.2   3.53   -18.9 |    25.3   1.92   -15.2 |     0.6   0.27   -19.6 |   277.2   2.51   -22.9 | YES |
| 5 | `EMA(12,77)` | 2MA |   202.6   3.58   -18.2 |    23.9   1.83   -15.3 |    -0.4   0.18   -18.4 |   273.6    2.5   -22.7 | - |
| 6 | `EMA(26,75)` | 2MA |   176.7    3.4   -19.6 |    26.9   2.04   -14.5 |     2.0   0.39   -17.9 |   258.1   2.48   -20.4 | YES |
| 7 | `EMA(10,55)` | 2MA |   185.2    3.3   -21.3 |    21.7   1.68   -15.2 |     3.1    0.5   -18.8 |   258.0   2.37   -21.3 | YES |
| 8 | `EMA(31,53)` | 2MA |   189.5   3.49   -20.3 |    25.2   1.92   -14.1 |    -1.4   0.09   -22.1 |   257.3   2.44   -24.7 | - |
| 9 | `EMA(22,65)` | 2MA |   193.4   3.52   -19.3 |    25.7   1.95   -14.4 |    -3.2  -0.08   -22.7 |   257.2   2.44   -22.7 | - |
| 10 | `EMA(19,27,38)` | 3MA |   202.6   3.64   -21.3 |    23.1   1.86   -13.6 |    -4.3  -0.22   -18.9 |   256.4   2.48   -22.0 | - |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `EMA(3,6)` | 2MA |    77.7   1.68   -58.2 |   -32.1  -1.84   -44.6 |   -43.7  -3.51   -48.3 |   -32.2  -0.05   -82.3 | - |
| 119 | `EMA(2,4,19)` | 3MA |    34.6   1.15   -49.6 |   -27.0  -1.78   -35.4 |   -33.5  -3.05   -37.9 |   -34.6  -0.27   -73.6 | - |
| 120 | `EMA(2,10)` | 2MA |    51.5   1.36   -59.7 |   -30.8  -1.74   -44.0 |   -39.6  -3.18   -44.8 |   -36.7  -0.17   -81.0 | - |


## 15m x SMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.219**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,15], slow in [37,210] -> 15/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [5,5], slow in [208,208] -> 1/60 configs positive across TRAIN & VAL & OOS
- band members: SMA(6,77), SMA(5,86), SMA(4,86), SMA(8,91), SMA(3,102), SMA(12,178), SMA(2,169), SMA(10,128), SMA(15,151), SMA(5,118,208), SMA(8,210), SMA(5,199), SMA(4,199), SMA(6,210), SMA(8,39), SMA(4,37)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `SMA(6,77)` | 2MA |   277.1   4.09   -18.6 |    28.9   2.09   -15.0 |     1.6   0.36   -18.9 |   393.5   2.89   -19.1 | YES |
| 2 | `SMA(22,65)` | 2MA |   321.4   4.43   -25.5 |    26.1   1.97   -13.1 |   -14.5  -1.19   -23.8 |   354.2   2.78   -25.5 | - |
| 3 | `SMA(15,65)` | 2MA |   284.2   4.15   -26.3 |    32.3   2.31   -13.6 |   -11.7  -0.89   -22.7 |   349.0   2.75   -26.3 | - |
| 4 | `SMA(12,77)` | 2MA |   267.3   4.07   -22.2 |    26.1   1.97   -14.2 |    -3.6  -0.11   -20.4 |   346.8   2.76   -22.2 | - |
| 5 | `SMA(26,75)` | 2MA |   263.5   4.08   -25.6 |    30.6   2.23   -13.1 |    -8.5  -0.59   -19.3 |   334.3   2.74   -25.6 | - |
| 6 | `SMA(5,86)` | 2MA |   254.6   3.94   -20.2 |    21.3   1.66   -18.5 |     0.7   0.29   -20.6 |   333.1   2.69   -21.2 | YES |
| 7 | `SMA(4,86)` | 2MA |   236.4   3.77   -22.1 |    20.4   1.59   -19.2 |     2.4   0.43   -18.2 |   314.9    2.6   -22.1 | YES |
| 8 | `SMA(31,53)` | 2MA |   266.3   3.93   -30.4 |    33.8    2.4   -14.3 |   -15.9  -1.33   -27.9 |   312.1   2.58   -30.4 | - |
| 9 | `SMA(2,73)` | 2MA |   258.8   3.84   -21.8 |    19.3   1.49   -22.3 |    -5.5  -0.24   -17.3 |   304.7   2.49   -27.0 | - |
| 10 | `SMA(8,91)` | 2MA |   204.9   3.56   -21.5 |    22.2   1.73   -15.5 |     3.3   0.51   -21.0 |   284.8   2.53   -25.6 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `SMA(2,3)` | 2MA |    24.8   0.98   -75.2 |   -11.7  -0.21   -33.0 |   -44.0  -2.91   -50.9 |   -38.3   0.03   -85.8 | - |
| 119 | `SMA(2,19,22)` | 3MA |   -16.8  -0.14   -62.8 |   -22.3  -1.54   -23.8 |   -18.5  -1.65   -26.0 |   -47.3  -0.72   -75.1 | - |
| 120 | `SMA(2,3,4)` | 3MA |     7.0   0.67   -69.5 |    -3.3   0.18   -34.0 |   -55.4  -4.67   -58.6 |   -53.9   -0.4   -84.7 | - |


## 15m x WMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.511**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [4,26], slow in [128,210] -> 9/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [19,19], slow in [233,233] -> 1/60 configs positive across TRAIN & VAL & OOS
- band members: WMA(10,128), WMA(22,151), WMA(15,151), WMA(8,210), WMA(6,210), WMA(12,178), WMA(5,199), WMA(26,172), WMA(19,132,233), WMA(4,199)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `WMA(44,75)` | 2MA |   298.0   4.33   -21.7 |    31.0   2.26   -11.9 |   -12.8  -1.04   -21.7 |   354.8   2.81   -22.7 | - |
| 2 | `WMA(26,75)` | 2MA |   278.5   4.11   -24.7 |    32.1    2.3   -14.2 |   -11.4  -0.87   -23.5 |   343.1   2.73   -24.7 | - |
| 3 | `WMA(52,89)` | 2MA |   266.1   4.15   -26.8 |    21.6   1.71   -13.4 |    -5.6  -0.31   -19.3 |   320.2   2.71   -27.2 | - |
| 4 | `WMA(10,128)` | 2MA |   227.4   3.78   -19.2 |    23.6   1.81   -15.5 |     0.2   0.24   -21.2 |   305.7   2.62   -22.6 | YES |
| 5 | `WMA(37,89)` | 2MA |   240.0    3.9   -24.3 |    23.9   1.84   -13.6 |    -4.6  -0.21   -19.3 |   301.9   2.62   -24.3 | - |
| 6 | `WMA(18,128)` | 2MA |   215.6    3.7   -20.7 |    24.5   1.87   -14.7 |    -0.2    0.2   -22.6 |   292.2   2.58   -24.2 | - |
| 7 | `WMA(12,77)` | 2MA |   229.2   3.64   -20.9 |    19.7   1.57   -15.9 |    -1.9   0.05   -19.5 |   286.5   2.47   -25.4 | - |
| 8 | `WMA(31,124)` | 2MA |   201.0   3.59   -23.2 |    22.2   1.74   -14.1 |    -1.5   0.07   -22.4 |   262.3   2.47   -25.9 | - |
| 9 | `WMA(8,91)` | 2MA |   199.8   3.41   -20.2 |    20.0   1.58   -17.0 |    -0.8   0.15   -19.4 |   256.9   2.35   -22.1 | - |
| 10 | `WMA(22,65)` | 2MA |   206.6   3.47   -24.4 |    22.5   1.74   -17.1 |    -8.1  -0.54   -23.2 |   245.2    2.3   -29.7 | - |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `WMA(3,4,38)` | 3MA |    22.7   0.92   -51.1 |   -25.8   -1.7   -31.6 |   -27.2  -2.38   -32.4 |   -33.7  -0.26   -71.6 | - |
| 119 | `WMA(8,14,38)` | 3MA |   -12.7  -0.04   -56.8 |   -17.5  -1.18   -24.7 |    -9.7  -0.78   -19.7 |   -34.9  -0.42   -65.3 | - |
| 120 | `WMA(6,22,24)` | 3MA |   -11.6   0.02   -60.3 |   -25.4  -1.84   -27.4 |   -11.2  -0.88   -24.5 |   -41.5  -0.55   -72.1 | - |


## 15m x HMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.3**; TRAIN+VAL top-10 -> OOS top-10 overlap = **1/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [22,31], slow in [124,151] -> 2/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [10,186], slow in [118,233] -> 4/60 configs positive across TRAIN & VAL & OOS
- band members: HMA(10,11,233), HMA(30,132,186), HMA(22,151), HMA(31,124), HMA(186,208,233), HMA(48,67,118)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `HMA(75,132,148)` | 3MA |   424.3   5.69   -21.3 |    24.0    2.1   -11.5 |    -8.5  -0.72   -16.7 |   494.8   3.62   -21.3 | - |
| 2 | `HMA(10,11,233)` | 3MA |   364.9   6.08   -21.9 |    13.5   1.55   -12.9 |     4.4   0.68    -9.6 |   450.6   4.02   -24.2 | YES |
| 3 | `HMA(73,145)` | 2MA |   272.1   4.13   -28.9 |    58.6   3.64   -15.0 |    -8.8  -0.54   -18.8 |   438.3   3.06   -28.9 | - |
| 4 | `HMA(2,3)` | 2MA |   329.0   2.96   -53.9 |    49.2   2.33   -25.9 |   -17.3  -0.65   -40.8 |   429.2   2.11   -53.9 | - |
| 5 | `HMA(62,239)` | 2MA |   268.6   4.11   -29.7 |    36.2   2.54   -16.9 |    -4.0  -0.12   -19.7 |   381.9   2.88   -36.1 | - |
| 6 | `HMA(86,122)` | 2MA |   233.2   3.77   -30.2 |    59.1   3.65   -14.7 |   -13.0  -0.94   -21.9 |   361.5   2.79   -30.2 | - |
| 7 | `HMA(102,237)` | 2MA |   258.0   4.12   -33.9 |    26.1    2.0   -16.1 |    -0.3   0.21   -16.2 |   350.3   2.83   -37.7 | - |
| 8 | `HMA(52,203)` | 2MA |   229.0   3.69   -28.9 |    41.8   2.85   -18.1 |    -5.9  -0.28   -18.4 |   339.0   2.69   -30.3 | - |
| 9 | `HMA(44,203)` | 2MA |   208.3   3.48   -29.7 |    39.1   2.66   -18.5 |    -0.7   0.18   -17.3 |   326.1   2.62   -33.6 | - |
| 10 | `HMA(37,203)` | 2MA |   224.3   3.61   -29.2 |    28.7   2.09   -19.7 |    -2.2   0.05   -16.2 |   308.1   2.55   -33.6 | - |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `HMA(12,43,75)` | 3MA |   -17.1  -0.26   -57.2 |   -29.0  -2.55   -33.6 |   -13.3  -1.22   -25.4 |   -49.0  -0.93   -73.1 | - |
| 119 | `HMA(4,30,67)` | 3MA |    -4.8   0.21   -54.9 |   -37.2  -3.05   -40.1 |   -18.6   -1.7   -32.1 |   -51.3  -0.88   -76.7 | - |
| 120 | `HMA(24,43,60)` | 3MA |   -31.4  -0.79   -58.3 |   -35.0  -2.83   -39.4 |   -11.4  -0.95   -29.0 |   -60.5  -1.29   -75.4 | - |


## 15m x DEMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.691**; TRAIN+VAL top-10 -> OOS top-10 overlap = **2/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [4,102], slow in [38,239] -> 20/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [5,75], slow in [118,233] -> 6/60 configs positive across TRAIN & VAL & OOS
- band members: DEMA(62,105), DEMA(26,172), DEMA(52,89), DEMA(22,151), DEMA(37,89), DEMA(31,124), DEMA(37,38), DEMA(15,151), DEMA(37,203), DEMA(12,178), DEMA(6,210), DEMA(44,203), DEMA(8,210), DEMA(18,128), DEMA(52,203), DEMA(5,199), DEMA(62,239), DEMA(4,199), DEMA(24,106,118), DEMA(75,132,148), DEMA(10,128), DEMA(30,132,186), DEMA(102,237), DEMA(19,132,233) (+2 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `DEMA(62,105)` | 2MA |   250.5   4.02   -32.4 |    42.3   2.92   -14.4 |     3.7   0.55   -19.5 |   417.3   3.05   -32.4 | YES |
| 2 | `DEMA(26,172)` | 2MA |   280.6    4.2   -22.8 |    25.4   1.95   -15.5 |     4.5   0.62   -19.9 |   398.7   2.96   -22.8 | YES |
| 3 | `DEMA(52,89)` | 2MA |   220.1   3.74   -24.4 |    31.2   2.27   -14.8 |     7.6   0.87   -17.1 |   351.9   2.81   -26.5 | YES |
| 4 | `DEMA(22,151)` | 2MA |   223.8   3.68   -24.3 |    25.1   1.92   -17.0 |     8.5   0.94   -18.8 |   339.5   2.72   -24.3 | YES |
| 5 | `DEMA(31,53)` | 2MA |   213.1   3.47   -30.6 |    34.8   2.42   -19.2 |    -3.3  -0.04   -15.7 |   307.9   2.52   -30.7 | - |
| 6 | `DEMA(44,75)` | 2MA |   212.4   3.61   -26.4 |    32.9   2.34   -17.3 |    -2.4   0.02   -20.8 |   305.2   2.59   -32.4 | - |
| 7 | `DEMA(37,89)` | 2MA |   202.5   3.51   -25.4 |    32.6   2.33   -17.0 |     0.8   0.29   -21.1 |   304.3   2.58   -29.4 | YES |
| 8 | `DEMA(31,124)` | 2MA |   186.5   3.34   -25.6 |    33.7    2.4   -15.2 |     4.9   0.64   -19.1 |   301.6   2.57   -25.6 | YES |
| 9 | `DEMA(48,67,118)` | 3MA |   216.6   4.08   -22.1 |    35.7    2.9   -11.3 |    -7.5   -0.6   -20.1 |   297.6   2.88   -22.1 | - |
| 10 | `DEMA(102,103)` | 2MA |   235.2   3.99   -26.4 |    25.1   1.97   -16.3 |    -5.7   -0.3   -26.4 |   295.5   2.65   -26.4 | - |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `DEMA(10,19)` | 2MA |    -2.7   0.39   -67.2 |   -24.7  -1.42   -40.4 |   -27.9  -2.03   -36.8 |   -47.2  -0.45   -81.0 | - |
| 119 | `DEMA(2,8,22)` | 3MA |    -2.6   0.38   -69.6 |   -24.2   -1.5   -39.3 |   -35.0  -2.84   -43.3 |   -52.0  -0.62   -84.7 | - |
| 120 | `DEMA(2,26)` | 2MA |   -20.8   0.04   -71.8 |   -14.8  -0.57   -36.2 |   -38.3  -2.96   -43.4 |   -58.4  -0.61   -83.7 | - |


## 15m x TEMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.48**; TRAIN+VAL top-10 -> OOS top-10 overlap = **2/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [6,102], slow in [75,239] -> 14/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [5,186], slow in [118,233] -> 13/60 configs positive across TRAIN & VAL & OOS
- band members: TEMA(62,239), TEMA(44,75), TEMA(37,89), TEMA(52,203), TEMA(48,67,118), TEMA(37,203), TEMA(102,237), TEMA(44,203), TEMA(60,67,233), TEMA(75,132,148), TEMA(102,103), TEMA(86,122), TEMA(73,145), TEMA(26,172), TEMA(8,210), TEMA(22,151), TEMA(24,106,118), TEMA(30,132,186), TEMA(38,67,233), TEMA(6,210), TEMA(186,208,233), TEMA(15,84,148), TEMA(12,106,148), TEMA(19,132,233) (+3 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `TEMA(62,239)` | 2MA |   261.6   4.15   -27.6 |    36.3   2.61   -14.9 |     4.2   0.59   -18.8 |   413.6   3.07   -27.6 | YES |
| 2 | `TEMA(44,75)` | 2MA |   242.4   3.83   -28.8 |    42.7   2.78   -19.4 |     4.0   0.56   -12.1 |   408.3    2.9   -32.2 | YES |
| 3 | `TEMA(37,89)` | 2MA |   237.8   3.75   -27.7 |    44.9   2.89   -19.2 |     2.4   0.43   -11.9 |   401.1   2.86   -31.1 | YES |
| 4 | `TEMA(62,105)` | 2MA |   267.1   4.19   -24.8 |    32.4   2.32   -19.2 |    -0.3   0.21   -17.4 |   384.5   2.93   -32.3 | - |
| 5 | `TEMA(52,203)` | 2MA |   207.2   3.67   -25.5 |    31.1   2.29   -14.5 |    14.2   1.38   -16.3 |   359.7   2.87   -25.5 | YES |
| 6 | `TEMA(31,124)` | 2MA |   233.8   3.68   -27.4 |    38.8   2.61   -17.9 |    -4.7  -0.16   -15.0 |   341.5   2.66   -28.8 | - |
| 7 | `TEMA(52,89)` | 2MA |   204.3   3.57   -28.4 |    42.4   2.84   -17.3 |    -0.2   0.22   -12.3 |   332.3    2.7   -31.3 | - |
| 8 | `TEMA(48,67,118)` | 3MA |   246.0   4.47   -22.3 |    16.8   1.58   -18.8 |     5.6   0.75   -11.4 |   326.7   3.06   -31.2 | YES |
| 9 | `TEMA(37,203)` | 2MA |   179.4   3.29   -25.3 |    32.5   2.32   -15.8 |    13.4   1.32   -16.8 |   319.8   2.65   -26.0 | YES |
| 10 | `TEMA(102,237)` | 2MA |   196.9   3.69   -29.3 |    37.7   2.74   -16.6 |     2.2   0.41   -24.4 |   317.6   2.79   -29.3 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `TEMA(3,8,43)` | 3MA |   -23.6  -0.16   -68.4 |   -11.6  -0.51   -33.7 |   -28.8  -2.33   -35.5 |   -51.9  -0.61   -78.4 | - |
| 119 | `TEMA(3,44)` | 2MA |   -21.9   0.01   -68.8 |   -11.8  -0.39   -34.1 |   -33.2  -2.48   -39.6 |   -54.0   -0.5   -80.5 | - |
| 120 | `TEMA(22,28)` | 2MA |   -28.4  -0.37   -70.0 |   -19.1  -0.98   -38.4 |   -23.2  -1.74   -37.0 |   -55.5  -0.74   -80.4 | - |


## 15m x KAMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.469**; TRAIN+VAL top-10 -> OOS top-10 overlap = **4/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [4,73], slow in [23,239] -> 16/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [5,75], slow in [148,233] -> 6/60 configs positive across TRAIN & VAL & OOS
- band members: KAMA(6,210), KAMA(5,199), KAMA(12,178), KAMA(8,210), KAMA(4,199), KAMA(10,128), KAMA(12,77), KAMA(8,91), KAMA(19,132,233), KAMA(73,145), KAMA(26,172), KAMA(62,105), KAMA(6,77), KAMA(52,89), KAMA(62,239), KAMA(38,67,233), KAMA(8,84,186), KAMA(6,67,233), KAMA(18,23), KAMA(5,118,208), KAMA(15,28), KAMA(75,132,148)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `KAMA(6,210)` | 2MA |   199.2   3.77   -18.8 |    37.3   2.59   -17.8 |    11.4   1.22   -12.5 |   357.4   2.96   -18.8 | YES |
| 2 | `KAMA(5,199)` | 2MA |   193.5   3.68   -20.1 |    37.1   2.58   -16.7 |     5.0   0.67   -15.1 |   322.5    2.8   -20.1 | YES |
| 3 | `KAMA(12,178)` | 2MA |   187.1    3.7   -17.6 |    27.7   2.08   -19.2 |     2.6   0.45   -19.8 |   275.8   2.65   -19.8 | YES |
| 4 | `KAMA(8,210)` | 2MA |   173.5   3.53   -19.0 |    31.3   2.28   -17.4 |     4.5   0.63   -17.2 |   275.5   2.63   -19.0 | YES |
| 5 | `KAMA(26,32)` | 2MA |   179.7   3.03   -32.6 |    40.2   2.63   -17.8 |    -7.0  -0.35   -22.8 |   264.7   2.27   -32.6 | - |
| 6 | `KAMA(4,199)` | 2MA |   157.3   3.25   -23.0 |    31.6   2.27   -17.6 |     4.0   0.58   -15.7 |   252.3   2.47   -23.0 | YES |
| 7 | `KAMA(44,75)` | 2MA |   198.0   3.68   -25.6 |    17.8   1.52   -17.5 |    -2.3   -0.0   -22.5 |   242.9   2.44   -31.4 | - |
| 8 | `KAMA(10,128)` | 2MA |   149.3    3.1   -19.4 |    26.6   2.03   -15.1 |     3.9   0.57   -19.2 |   227.9   2.34   -19.4 | YES |
| 9 | `KAMA(15,151)` | 2MA |   154.5   3.29   -19.4 |    26.8   2.06   -15.0 |    -1.5   0.07   -22.1 |   217.9   2.34   -22.1 | - |
| 10 | `KAMA(22,28)` | 2MA |   197.5   3.25   -28.2 |    14.6   1.22   -22.9 |    -7.5  -0.46   -25.3 |   215.3   2.08   -28.2 | - |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `KAMA(2,3,4)` | 3MA |    26.7   0.99   -49.0 |   -29.7  -2.05   -32.3 |   -30.4  -2.85   -34.8 |   -38.0  -0.31   -71.8 | - |
| 119 | `KAMA(3,6)` | 2MA |   -10.0   0.25   -66.2 |   -13.3  -0.54   -26.7 |   -24.8  -1.87   -33.2 |   -41.3  -0.29   -74.9 | - |
| 120 | `KAMA(2,10)` | 2MA |    14.2   0.74   -61.1 |   -30.0  -1.94   -33.4 |   -28.5   -2.4   -34.4 |   -42.9  -0.41   -78.0 | - |


## 15m x VIDYA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.419**; TRAIN+VAL top-10 -> OOS top-10 overlap = **2/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [12,239] -> 43/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [14,233] -> 42/60 configs positive across TRAIN & VAL & OOS
- band members: VIDYA(12,13), VIDYA(8,16), VIDYA(10,19), VIDYA(6,13), VIDYA(6,33), VIDYA(6,14,15), VIDYA(5,12), VIDYA(4,15), VIDYA(5,37), VIDYA(6,9,24), VIDYA(4,37), VIDYA(12,14,75), VIDYA(18,23), VIDYA(3,18), VIDYA(15,22,48), VIDYA(8,14,38), VIDYA(10,17,233), VIDYA(8,34,233), VIDYA(8,22,60), VIDYA(5,38,208), VIDYA(5,24,34), VIDYA(19,27,38), VIDYA(8,39), VIDYA(12,33) (+61 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `VIDYA(12,13)` | 2MA |   172.9   3.26   -18.5 |    38.9   2.66   -15.4 |     5.0   0.66   -17.3 |   298.0    2.6   -18.5 | YES |
| 2 | `VIDYA(8,16)` | 2MA |   188.1   3.39   -19.7 |    34.1   2.41   -14.9 |     1.0   0.31   -18.2 |   290.3   2.55   -19.7 | YES |
| 3 | `VIDYA(10,19)` | 2MA |   168.2   3.26   -20.5 |    30.3   2.21   -14.4 |     7.1   0.85   -16.2 |   274.3   2.53   -20.5 | YES |
| 4 | `VIDYA(6,13)` | 2MA |   192.0   3.35   -22.0 |    22.9   1.77   -17.8 |     1.6   0.36   -16.4 |   264.7    2.4   -22.0 | YES |
| 5 | `VIDYA(6,33)` | 2MA |   142.5   3.02   -19.8 |    26.7   2.01   -15.5 |    12.1   1.29   -13.4 |   244.3   2.42   -20.5 | YES |
| 6 | `VIDYA(5,10,14)` | 3MA |   170.9   3.33   -21.4 |    26.6    2.1   -12.6 |    -1.1   0.08   -15.8 |   239.0   2.42   -21.4 | - |
| 7 | `VIDYA(6,14,15)` | 3MA |   160.9   3.28   -19.9 |    22.8   1.85   -14.9 |     4.5   0.64   -15.7 |   234.7   2.42   -20.8 | YES |
| 8 | `VIDYA(5,12)` | 2MA |   173.5   3.14   -20.6 |    16.6   1.38   -17.9 |     3.1    0.5   -16.8 |   228.8   2.22   -24.0 | YES |
| 9 | `VIDYA(4,15)` | 2MA |   174.6   3.17   -20.2 |    15.8   1.33   -19.3 |     2.8   0.47   -16.0 |   226.9   2.22   -21.8 | YES |
| 10 | `VIDYA(5,37)` | 2MA |   132.1   2.89   -20.0 |    18.9   1.54   -19.1 |    10.3   1.13   -14.3 |   204.3   2.21   -20.9 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `VIDYA(2,5,118)` | 3MA |    23.4   1.06   -38.6 |    -6.3  -0.37   -21.2 |   -13.1  -1.68   -21.5 |     0.5   0.24   -47.7 | - |
| 119 | `VIDYA(2,3,4)` | 3MA |     7.5   0.57   -51.7 |   -20.1  -1.26   -29.8 |   -13.0  -1.02   -25.3 |   -25.3  -0.12   -63.6 | - |
| 120 | `VIDYA(2,3)` | 2MA |    -6.4   0.25   -61.6 |   -28.3  -1.87   -35.7 |   -21.2   -1.7   -29.2 |   -47.1  -0.57   -76.3 | - |


# GLOBAL rank-stability summary (the transfer-noise headline)

| TF | MA-type | n_band(2MA/3MA) | Spearman rho (TV vs OOS) | top-10 overlap |
|---|---|---|---:|---:|
| 1d | EMA | 27/27 | -0.192 | 0/10 |
| 1d | SMA | 33/43 | 0.239 | 2/10 |
| 1d | WMA | 31/39 | 0.024 | 1/10 |
| 1d | HMA | 51/46 | 0.318 | 2/10 |
| 1d | DEMA | 48/51 | 0.367 | 1/10 |
| 1d | TEMA | 48/56 | 0.194 | 0/10 |
| 1d | KAMA | 29/42 | 0.184 | 2/10 |
| 1d | VIDYA | 21/34 | 0.178 | 1/10 |
| 4h | EMA | 23/29 | -0.176 | 1/10 |
| 4h | SMA | 26/25 | -0.056 | 2/10 |
| 4h | WMA | 25/20 | -0.133 | 0/10 |
| 4h | HMA | 26/23 | -0.401 | 0/10 |
| 4h | DEMA | 30/31 | -0.464 | 0/10 |
| 4h | TEMA | 25/29 | -0.213 | 0/10 |
| 4h | KAMA | 21/30 | -0.151 | 0/10 |
| 4h | VIDYA | 20/32 | -0.073 | 0/10 |
| 2h | EMA | 21/14 | -0.341 | 0/10 |
| 2h | SMA | 21/18 | -0.254 | 3/10 |
| 2h | WMA | 24/15 | -0.37 | 2/10 |
| 2h | HMA | 27/24 | 0.225 | 3/10 |
| 2h | DEMA | 22/18 | -0.134 | 4/10 |
| 2h | TEMA | 26/23 | 0.325 | 3/10 |
| 2h | KAMA | 18/18 | -0.392 | 1/10 |
| 2h | VIDYA | 25/24 | -0.141 | 0/10 |
| 1h | EMA | 49/42 | -0.201 | 0/10 |
| 1h | SMA | 23/16 | -0.123 | 1/10 |
| 1h | WMA | 22/15 | -0.137 | 0/10 |
| 1h | HMA | 25/17 | -0.18 | 0/10 |
| 1h | DEMA | 28/15 | -0.049 | 0/10 |
| 1h | TEMA | 28/21 | -0.062 | 0/10 |
| 1h | KAMA | 16/13 | -0.426 | 1/10 |
| 1h | VIDYA | 38/33 | -0.133 | 0/10 |
| 30m | EMA | 29/26 | 0.156 | 0/10 |
| 30m | SMA | 17/10 | -0.181 | 0/10 |
| 30m | WMA | 22/8 | -0.237 | 0/10 |
| 30m | HMA | 21/10 | 0.413 | 1/10 |
| 30m | DEMA | 35/14 | 0.417 | 0/10 |
| 30m | TEMA | 32/25 | 0.352 | 0/10 |
| 30m | KAMA | 18/12 | -0.147 | 2/10 |
| 30m | VIDYA | 47/50 | 0.03 | 1/10 |
| 15m | EMA | 16/4 | 0.665 | 1/10 |
| 15m | SMA | 15/1 | 0.219 | 0/10 |
| 15m | WMA | 9/1 | 0.511 | 0/10 |
| 15m | HMA | 2/4 | 0.3 | 1/10 |
| 15m | DEMA | 20/6 | 0.691 | 2/10 |
| 15m | TEMA | 14/13 | 0.48 | 2/10 |
| 15m | KAMA | 16/6 | 0.469 | 4/10 |
| 15m | VIDYA | 43/42 | 0.419 | 2/10 |

**Median Spearman rho across all cells = -0.059** (mean 0.038, n=48 cells). The closer to 0 / negative, the more the within-cell config RANK is noise that does NOT transfer TRAIN+VAL -> OOS. This is the empirical basis for 'trust the band, not the #1'.
