# 2020 PER-CONFIG 2MA/3MA LEADERBOARD + the WORKING BAND

STRICT LONG-ONLY + spot (held in {0,1}, no short/inverse anywhere). 2020 BAND ONLY. Causal/lag-1, maker cost. Ironed sleeve = MA-cross -> 10% trail -> min_hold(12).

## HONEST FRAMING -- the BAND is the deliverable, NOT the exact #1

The prior investigation (D62 + the per-asset null) found per-config RANK is NOISE that does not transfer across regimes. So:

- **TRUST THE BAND** = the set of configs POSITIVE across TRAIN AND VAL AND OOS. This is the robust set the FAMILY ENSEMBLE actually rides. Reported per cell as a (fast, slow) parameter RANGE.
- **DO NOT TRUST the exact ordering** within a cell. The within-band #1 is regime-transient. The rank-stability number (Spearman (TRAIN+VAL) net vs OOS net + top-10 overlap) quantifies how little the ordering transfers. Low rho / small overlap = the ranking is noise.
- PRIMARY SORT below = FULL-2020 net (wealth over the most data = the most stable estimate). Per-split net/Sharpe/maxDD shown so you can see whether a high FULL rank actually TRANSFERS.

DATA CAVEATS: 2h is SYNTHESIZED from 1h (OHLC-resample). SOL/AVAX have only 2020-H2 history (~Sep 2020 on) -> absent from TRAIN, present in VAL/OOS; the book averages over assets present per bar (skipna). 2020 OOS (Oct-Dec) is a clean BULL (~0% bear) -- these are PARTICIPATING-BETA long-only books; under-participation vs buy-hold is expected and is NOT a defect.

Repro: `python -m strat.ma_2020_config_leaderboard --tfs 1d,4h,2h,1h,30m,15m`  git_sha=c0e0a19  cost=maker(0.0006)  trail=0.1  min_hold=12  split={'TRAIN': ('2020-01-01', '2020-07-01'), 'VAL': ('2020-07-01', '2020-10-01'), 'OOS': ('2020-10-01', '2021-01-01'), 'FULL': ('2020-01-01', '2021-01-01')}

All numbers are [MEASURED] from the run below (equal-weight u10, causal/lag-1, maker).


# Timeframe: 1d
_Benchmark (equal-weight u10 buy-hold, no cost): FULL-2020 net = 140.2% (participation-tax reference)._

## 1d x EMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.374**; TRAIN+VAL top-10 -> OOS top-10 overlap = **2/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 59/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,24], slow in [4,233] -> 46/60 configs positive across TRAIN & VAL & OOS
- band members: EMA(3,4,38), EMA(2,4,19), EMA(2,12,14), EMA(5,10,14), EMA(2,8,22), EMA(2,19,22), EMA(3,8,43), EMA(6,13), EMA(5,6,34), EMA(6,9,24), EMA(4,8,17), EMA(5,12), EMA(3,12,27), EMA(4,15), EMA(4,37), EMA(4,5,75), EMA(2,3), EMA(3,18), EMA(8,9,75), EMA(6,22,24), EMA(6,33), EMA(5,24,34), EMA(6,14,15), EMA(2,5,118) (+81 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `EMA(3,4,38)` | 3MA |    27.1   1.68   -17.4 |    25.9   2.51   -17.7 |    39.9   2.68   -21.2 |   123.8   2.18   -21.2 | YES |
| 2 | `EMA(2,4,19)` | 3MA |    37.2   2.11   -16.4 |    27.0   2.62   -16.9 |    25.2   1.76   -22.0 |   118.1   2.03   -22.0 | YES |
| 3 | `EMA(2,12,14)` | 3MA |    30.2    1.8   -16.0 |    26.5   2.85   -15.3 |    32.4   2.22   -21.6 |   118.0   2.11   -21.6 | YES |
| 4 | `EMA(5,10,14)` | 3MA |    34.3   2.08   -14.9 |    26.9   2.94   -16.4 |    27.9   2.04   -21.6 |   118.0   2.19   -21.6 | YES |
| 5 | `EMA(2,8,22)` | 3MA |    28.2    1.7   -17.5 |    27.7    2.7   -17.8 |    30.3   2.14   -26.3 |   113.3   2.04   -26.3 | YES |
| 6 | `EMA(2,19,22)` | 3MA |    29.1   1.77   -18.6 |    25.0    2.6   -18.4 |    32.0   2.24   -22.8 |   113.0   2.07   -22.8 | YES |
| 7 | `EMA(3,8,43)` | 3MA |    22.2   1.51   -16.6 |    25.9   2.55   -18.9 |    38.4   2.66   -22.4 |   112.9   2.12   -22.4 | YES |
| 8 | `EMA(6,13)` | 2MA |    26.9   1.83   -14.1 |    25.9   2.67   -16.6 |    30.2   2.16   -19.7 |   108.1   2.07   -19.7 | YES |
| 9 | `EMA(5,6,34)` | 3MA |    19.6   1.26   -22.0 |    26.6   2.64   -17.9 |    35.9   2.53   -23.4 |   105.8   1.99   -23.4 | YES |
| 10 | `EMA(6,9,24)` | 3MA |    26.4    1.7   -19.2 |    28.2   2.93   -18.1 |    26.6   1.98   -21.3 |   105.1   2.03   -21.3 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `EMA(75,132,148)` | 3MA |    20.5   2.52    -8.0 |    -0.3  -0.07    -4.1 |     3.1   1.47    -4.2 |    23.9   1.78    -8.0 | - |
| 119 | `EMA(102,237)` | 2MA |    20.9   2.56    -7.6 |     0.2   0.14    -4.1 |     0.6   0.57    -1.5 |    21.9   1.71    -7.6 | YES |
| 120 | `EMA(186,208,233)` | 3MA |    21.8   2.64    -6.8 |    -0.8  -0.35    -4.3 |     0.6   2.77     0.0 |    21.5    1.7    -6.8 | - |


## 1d x SMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.479**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 49/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,24], slow in [4,233] -> 35/60 configs positive across TRAIN & VAL & OOS
- band members: SMA(8,16), SMA(6,14,15), SMA(2,8,22), SMA(18,23), SMA(2,12,14), SMA(2,4,19), SMA(3,12,27), SMA(10,19), SMA(3,18), SMA(2,19,22), SMA(2,26), SMA(5,10,14), SMA(6,9,24), SMA(4,8,17), SMA(12,13), SMA(15,22,48), SMA(4,5), SMA(3,4,38), SMA(5,6,34), SMA(3,19,43), SMA(12,14,75), SMA(15,28), SMA(8,14,38), SMA(4,15) (+60 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `SMA(8,16)` | 2MA |    34.8   2.07   -19.5 |    22.7    2.6   -14.3 |    43.5   2.95   -17.1 |   137.3   2.43   -19.5 | YES |
| 2 | `SMA(6,14,15)` | 3MA |    38.5    2.2   -18.0 |    26.1   3.38   -10.6 |    35.6   2.51   -21.9 |   136.9   2.45   -21.9 | YES |
| 3 | `SMA(2,8,22)` | 3MA |    36.0   2.12   -14.3 |    37.4   4.78    -9.4 |    26.5   1.97   -22.3 |   136.4   2.45   -22.3 | YES |
| 4 | `SMA(18,23)` | 2MA |    22.9   1.84    -8.6 |    33.3   4.79    -7.0 |    44.1   3.23   -16.8 |   136.1   2.83   -16.8 | YES |
| 5 | `SMA(2,12,14)` | 3MA |    44.5   2.36   -17.2 |    26.4   3.11   -12.8 |    26.5   1.98   -24.9 |   130.9   2.31   -24.9 | YES |
| 6 | `SMA(2,4,19)` | 3MA |    44.5   2.33   -18.8 |    25.0   2.77   -14.2 |    27.3   1.93   -21.8 |   129.8   2.21   -21.8 | YES |
| 7 | `SMA(3,12,27)` | 3MA |    34.7    2.1   -13.9 |    29.6   3.35   -14.8 |    31.3   2.34   -20.7 |   129.3   2.39   -20.7 | YES |
| 8 | `SMA(10,19)` | 2MA |    30.1   2.09   -15.2 |    25.8   3.33   -12.7 |    38.1   2.83   -15.9 |   126.0   2.53   -15.9 | YES |
| 9 | `SMA(3,18)` | 2MA |    33.5   1.83   -22.4 |    28.2    2.8   -17.1 |    31.6   2.21   -20.9 |   125.2   2.14   -22.4 | YES |
| 10 | `SMA(2,19,22)` | 3MA |    23.9   1.61   -16.7 |    27.5   2.91   -17.6 |    42.5    2.9   -19.4 |   125.1   2.31   -19.4 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `SMA(52,89)` | 2MA |   -15.5  -0.64   -27.0 |    -1.8  -1.63    -2.7 |    14.8   2.14   -10.7 |    -4.7   0.01   -27.0 | - |
| 119 | `SMA(44,203)` | 2MA |   -12.9  -1.09   -20.2 |    -0.8  -0.36    -4.1 |     1.3   0.75    -3.2 |   -12.5  -0.69   -20.2 | - |
| 120 | `SMA(52,203)` | 2MA |   -15.5  -0.68   -27.0 |    -0.0   0.04    -3.1 |    -0.7  -1.81    -1.1 |   -16.0   -0.5   -27.0 | - |


## 1d x WMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.572**; TRAIN+VAL top-10 -> OOS top-10 overlap = **2/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,73], slow in [3,210] -> 52/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,75], slow in [4,233] -> 54/60 configs positive across TRAIN & VAL & OOS
- band members: WMA(6,9,24), WMA(2,19,22), WMA(2,8,22), WMA(6,22,24), WMA(3,12,27), WMA(10,19), WMA(8,22,60), WMA(8,14,38), WMA(18,23), WMA(4,5), WMA(15,22,48), WMA(6,14,15), WMA(3,19,43), WMA(4,5,75), WMA(12,33), WMA(5,24,34), WMA(22,28), WMA(5,10,14), WMA(2,12,14), WMA(3,4,38), WMA(3,8,43), WMA(2,26), WMA(15,28), WMA(5,6,34) (+82 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `WMA(6,9,24)` | 3MA |    47.1   2.61   -17.0 |    32.2    3.9    -9.8 |    33.5   2.38   -20.1 |   159.7   2.68   -20.1 | YES |
| 2 | `WMA(2,19,22)` | 3MA |    48.3   2.52   -14.6 |    30.7   3.59   -11.8 |    32.2   2.32   -20.4 |   156.3   2.59   -20.4 | YES |
| 3 | `WMA(2,8,22)` | 3MA |    42.8   2.35   -17.3 |    32.8    3.4   -13.1 |    33.1   2.34   -18.5 |   152.4   2.52   -18.5 | YES |
| 4 | `WMA(6,22,24)` | 3MA |    42.2   2.45   -13.7 |    25.5   2.96   -15.2 |    36.9   2.69   -18.3 |   144.4   2.58   -18.3 | YES |
| 5 | `WMA(3,12,27)` | 3MA |    39.0   2.22   -15.9 |    29.5   3.73   -10.6 |    33.8   2.38   -20.0 |   140.9   2.48   -20.0 | YES |
| 6 | `WMA(10,19)` | 2MA |    36.2   2.19   -18.6 |    26.9   3.14   -12.4 |    38.4   2.77   -15.2 |   139.2   2.54   -18.6 | YES |
| 7 | `WMA(8,22,60)` | 3MA |    27.5   2.02   -11.7 |    30.2   3.65   -12.5 |    40.9   3.18   -15.1 |   133.9   2.73   -15.1 | YES |
| 8 | `WMA(8,14,38)` | 3MA |    33.7    2.1   -14.6 |    24.0   2.77   -15.0 |    36.5   2.66   -17.8 |   126.4   2.39   -17.8 | YES |
| 9 | `WMA(18,23)` | 2MA |    24.6   1.85   -12.6 |    28.8   3.84   -10.8 |    38.2   2.87   -15.2 |   121.8   2.55   -15.2 | YES |
| 10 | `WMA(4,5)` | 2MA |    43.5   1.95   -24.8 |    28.5   2.43   -19.0 |    20.2   1.44   -22.8 |   121.6   1.85   -25.3 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `WMA(86,122)` | 2MA |    18.7   1.97    -7.2 |    -1.4  -0.73    -4.2 |     0.3   0.16    -5.6 |    17.4   1.05   -12.2 | - |
| 119 | `WMA(102,237)` | 2MA |    15.6   2.01    -8.1 |     0.8   0.34    -7.5 |    -1.6  -0.63    -4.0 |    14.7    1.1   -11.5 | - |
| 120 | `WMA(186,208,233)` | 3MA |    14.2   1.86    -8.9 |    -2.8   -0.9    -7.2 |     2.6   0.98    -4.1 |    14.0   1.04   -15.3 | - |


## 1d x HMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.374**; TRAIN+VAL top-10 -> OOS top-10 overlap = **1/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 55/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,60], slow in [4,233] -> 53/60 configs positive across TRAIN & VAL & OOS
- band members: HMA(37,38), HMA(3,19,43), HMA(4,15), HMA(5,86), HMA(5,24,34), HMA(4,86), HMA(12,14,75), HMA(8,9,75), HMA(8,22,60), HMA(19,27,38), HMA(8,91), HMA(5,10,14), HMA(15,22,48), HMA(12,22,118), HMA(22,65), HMA(6,77), HMA(6,13), HMA(26,75), HMA(10,55), HMA(10,34,60), HMA(5,12), HMA(2,30,84), HMA(18,55), HMA(12,77) (+84 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `HMA(37,38)` | 2MA |    35.5   2.28   -11.1 |    33.9   4.45    -7.6 |    41.0    3.0   -18.6 |   155.8   2.88   -18.6 | YES |
| 2 | `HMA(3,19,43)` | 3MA |    40.2   2.26   -18.9 |    30.2   4.14    -8.7 |    37.8   2.69   -18.5 |   151.5   2.67   -18.9 | YES |
| 3 | `HMA(4,15)` | 2MA |    38.2   1.65   -29.5 |    31.1   2.75   -17.4 |    37.1   2.33   -16.4 |   148.5   2.08   -29.5 | YES |
| 4 | `HMA(5,86)` | 2MA |    33.9   2.08   -18.0 |    36.2   4.23   -10.3 |    33.6   2.42   -21.0 |   143.7   2.57   -21.0 | YES |
| 5 | `HMA(5,24,34)` | 3MA |    40.7   2.29   -17.8 |    29.5   3.82    -8.3 |    33.4   2.49   -21.8 |   143.0   2.58   -21.8 | YES |
| 6 | `HMA(4,86)` | 2MA |    35.2   2.16   -18.0 |    35.5   4.13   -10.7 |    32.6    2.4   -20.4 |   142.8   2.58   -20.4 | YES |
| 7 | `HMA(12,14,75)` | 3MA |    41.4   2.58   -15.7 |    30.8   4.54    -8.1 |    30.0   2.27   -18.6 |   140.5   2.68   -18.6 | YES |
| 8 | `HMA(8,9,75)` | 3MA |    31.1   1.94   -17.4 |    30.7   4.17   -10.4 |    39.9   2.76   -16.8 |   139.9   2.56   -17.4 | YES |
| 9 | `HMA(8,22,60)` | 3MA |    49.3    2.9   -10.4 |    30.3   4.36   -10.1 |    23.0   1.99   -18.3 |   139.3   2.76   -18.3 | YES |
| 10 | `HMA(19,27,38)` | 3MA |    50.1   2.91   -13.6 |    24.2   3.14    -9.8 |    28.0   2.15   -21.2 |   138.6   2.58   -21.2 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `HMA(12,33)` | 2MA |    -2.7   0.24   -43.0 |     9.9   1.12   -21.5 |    16.1   1.32   -20.5 |    24.1   0.69   -43.0 | - |
| 119 | `HMA(102,237)` | 2MA |    19.7   2.01    -9.7 |    -0.0   0.01    -4.7 |     3.4   0.67   -11.0 |    23.7   1.25   -11.0 | - |
| 120 | `HMA(186,208,233)` | 3MA |    10.6   1.36   -10.5 |    11.0   3.28    -3.2 |    -2.2  -0.59    -5.3 |    20.0   1.33   -10.5 | - |


## 1d x DEMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.585**; TRAIN+VAL top-10 -> OOS top-10 overlap = **1/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,237] -> 56/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,75], slow in [4,233] -> 58/60 configs positive across TRAIN & VAL & OOS
- band members: DEMA(8,14,38), DEMA(3,8,43), DEMA(6,9,24), DEMA(3,4,38), DEMA(6,14,15), DEMA(12,33), DEMA(2,19,22), DEMA(5,6,34), DEMA(18,23), DEMA(3,19,43), DEMA(2,3,4), DEMA(3,12,27), DEMA(15,28), DEMA(4,37), DEMA(10,19), DEMA(5,37), DEMA(6,22,24), DEMA(6,33), DEMA(8,39), DEMA(3,44), DEMA(10,55), DEMA(12,14,75), DEMA(22,28), DEMA(2,8,22) (+90 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `DEMA(8,14,38)` | 3MA |    43.5   2.59   -12.9 |    29.0   4.64    -6.0 |    39.7    3.0   -16.8 |   158.5   2.98   -16.8 | YES |
| 2 | `DEMA(3,8,43)` | 3MA |    41.2    2.4   -13.5 |    30.7   4.35   -10.2 |    33.2   2.36   -17.2 |   145.8   2.61   -17.2 | YES |
| 3 | `DEMA(6,9,24)` | 3MA |    45.5   2.64   -16.8 |    23.8   3.24   -11.4 |    34.3   2.46   -16.8 |   141.9   2.58   -16.8 | YES |
| 4 | `DEMA(3,4,38)` | 3MA |    36.4   2.14   -17.9 |    32.3   3.94   -10.9 |    32.9   2.24   -17.8 |   139.9   2.42   -17.9 | YES |
| 5 | `DEMA(6,14,15)` | 3MA |    47.7   2.89   -14.0 |    25.5   3.28   -12.2 |    29.2   2.16   -20.8 |   139.5   2.57   -20.8 | YES |
| 6 | `DEMA(12,33)` | 2MA |    23.1   1.73   -14.8 |    35.1   4.83    -6.9 |    43.7   3.21   -15.9 |   139.0    2.8   -15.9 | YES |
| 7 | `DEMA(2,19,22)` | 3MA |    42.3    2.5   -15.2 |    25.2   3.88    -6.9 |    33.2    2.6   -18.6 |   137.3   2.69   -18.6 | YES |
| 8 | `DEMA(5,6,34)` | 3MA |    45.0   2.56   -13.7 |    28.3   3.99   -10.6 |    27.3   2.01   -20.1 |   136.9   2.48   -20.1 | YES |
| 9 | `DEMA(18,23)` | 2MA |    22.8   1.77   -14.8 |    29.5   4.19    -6.0 |    46.5    3.4   -15.2 |   132.8   2.76   -15.2 | YES |
| 10 | `DEMA(3,19,43)` | 3MA |    37.4   2.33   -13.6 |    32.1   4.76    -7.7 |    27.7    2.2   -20.6 |   132.0   2.61   -20.6 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `DEMA(4,15)` | 2MA |    -3.8    0.2   -42.8 |    19.9   1.93   -19.6 |     8.1   0.81   -24.1 |    24.6   0.69   -42.8 | - |
| 119 | `DEMA(62,239)` | 2MA |    20.3   2.37    -9.1 |     4.0   1.57    -3.1 |    -0.6  -0.08    -7.6 |    24.2   1.54    -9.1 | - |
| 120 | `DEMA(186,208,233)` | 3MA |    20.8   2.52    -7.5 |    -1.2  -0.54    -4.6 |     3.4   1.62    -3.8 |    23.4   1.73    -7.5 | - |


## 1d x TEMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.649**; TRAIN+VAL top-10 -> OOS top-10 overlap = **1/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 55/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [4,233] -> 57/60 configs positive across TRAIN & VAL & OOS
- band members: TEMA(8,22,60), TEMA(4,5,75), TEMA(26,32), TEMA(15,22,48), TEMA(5,24,34), TEMA(22,28), TEMA(3,6), TEMA(2,73), TEMA(18,55), TEMA(19,27,38), TEMA(12,14,75), TEMA(2,10), TEMA(4,5), TEMA(2,30,84), TEMA(8,14,38), TEMA(6,77), TEMA(6,34,94), TEMA(6,22,24), TEMA(3,19,43), TEMA(3,4,38), TEMA(37,38), TEMA(15,65), TEMA(12,43,75), TEMA(18,23) (+88 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `TEMA(8,22,60)` | 3MA |    46.2   3.09    -8.0 |    27.5   4.89    -4.7 |    40.9   3.23   -13.5 |   162.7   3.29   -13.5 | YES |
| 2 | `TEMA(4,5,75)` | 3MA |    43.0    2.4   -13.8 |    33.8   4.63    -7.6 |    31.8   2.21   -18.1 |   152.2   2.59   -18.1 | YES |
| 3 | `TEMA(26,32)` | 2MA |    28.3   2.34    -9.3 |    30.6   4.12    -9.0 |    48.2   3.77   -14.2 |   148.3   3.15   -14.2 | YES |
| 4 | `TEMA(15,22,48)` | 3MA |    42.5    2.9    -8.0 |    27.3   4.49    -4.0 |    35.8   2.98   -14.5 |   146.1    3.1   -14.5 | YES |
| 5 | `TEMA(5,24,34)` | 3MA |    40.5   2.72   -10.4 |    28.9   4.29    -9.4 |    35.7   2.68   -16.8 |   145.9   2.86   -16.8 | YES |
| 6 | `TEMA(22,28)` | 2MA |    33.1   2.58    -8.1 |    24.7   3.08   -11.2 |    46.0   3.47   -15.2 |   142.2   2.91   -15.2 | YES |
| 7 | `TEMA(3,6)` | 2MA |    25.9   1.12   -35.6 |    38.0   3.36   -13.2 |    39.1    2.4   -16.9 |   141.7   1.89   -35.6 | YES |
| 8 | `TEMA(2,73)` | 2MA |    37.7   2.31   -16.6 |    30.7   3.56   -13.2 |    33.3   2.26   -21.3 |   139.9   2.44   -21.3 | YES |
| 9 | `TEMA(18,55)` | 2MA |    31.3   2.48    -8.4 |    26.8   4.09    -5.8 |    42.9   3.38   -15.0 |   137.9   3.01   -15.0 | YES |
| 10 | `TEMA(19,27,38)` | 3MA |    32.8   2.44    -9.2 |    30.3   4.63    -5.4 |    36.3   2.95   -16.2 |   135.8   2.94   -16.2 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `TEMA(186,208,233)` | 3MA |    20.1   2.33    -9.2 |     2.7   1.11    -4.1 |     3.2   1.55    -3.9 |    27.4   1.87    -9.2 | YES |
| 119 | `TEMA(10,19)` | 2MA |    -4.7   0.16   -40.6 |    21.0   2.19   -16.6 |     9.3    0.9   -20.3 |    26.0   0.72   -40.6 | - |
| 120 | `TEMA(6,14,15)` | 3MA |   -10.1  -0.04   -40.9 |    21.4   2.31   -15.1 |    14.4   1.22   -17.7 |    24.9   0.71   -40.9 | - |


## 1d x KAMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.457**; TRAIN+VAL top-10 -> OOS top-10 overlap = **1/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,237] -> 43/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,60], slow in [4,233] -> 31/60 configs positive across TRAIN & VAL & OOS
- band members: KAMA(2,12,14), KAMA(5,12), KAMA(6,14,15), KAMA(2,4,19), KAMA(2,10), KAMA(6,13), KAMA(2,8,22), KAMA(4,8,17), KAMA(18,23), KAMA(3,6), KAMA(3,18), KAMA(3,4,38), KAMA(2,3), KAMA(5,6,34), KAMA(3,12,27), KAMA(5,10,14), KAMA(4,15), KAMA(6,9,24), KAMA(2,19,22), KAMA(4,5), KAMA(2,26), KAMA(4,10,208), KAMA(3,44), KAMA(5,12,208) (+50 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `KAMA(2,12,14)` | 3MA |    29.6    2.1    -9.7 |    30.7   4.13   -11.0 |    33.0   2.91   -14.6 |   125.3   2.76   -14.6 | YES |
| 2 | `KAMA(5,12)` | 2MA |    23.7   1.89   -13.2 |    22.7   2.87   -12.2 |    44.8   3.51   -14.3 |   119.8   2.62   -15.3 | YES |
| 3 | `KAMA(6,14,15)` | 3MA |    23.8    2.0    -7.2 |    38.1   5.36    -5.7 |    27.0   2.56   -14.6 |   117.2   2.86   -14.6 | YES |
| 4 | `KAMA(2,4,19)` | 3MA |    21.4   1.51   -14.9 |    33.7   4.22    -9.1 |    33.0   2.46   -20.2 |   115.9   2.34   -20.2 | YES |
| 5 | `KAMA(2,10)` | 2MA |    33.9   1.89   -19.8 |    29.3   3.47   -12.5 |    20.7   1.65   -19.2 |   108.9   2.06   -19.8 | YES |
| 6 | `KAMA(6,13)` | 2MA |    19.4   1.66   -11.6 |    22.2   3.08   -10.2 |    42.3   3.71   -13.6 |   107.6   2.65   -13.6 | YES |
| 7 | `KAMA(2,8,22)` | 3MA |    22.7   1.68   -11.5 |    23.5   3.05   -12.9 |    34.7   2.77   -20.3 |   104.2   2.31   -20.3 | YES |
| 8 | `KAMA(4,8,17)` | 3MA |    18.1   1.42   -18.8 |    24.9    3.4    -9.5 |    35.5   3.15   -15.3 |    99.8   2.41   -18.8 | YES |
| 9 | `KAMA(18,23)` | 2MA |    28.4   2.21    -9.6 |    18.4   3.38    -5.7 |    31.0   2.88   -11.6 |    99.2   2.58   -11.6 | YES |
| 10 | `KAMA(3,6)` | 2MA |    22.4   1.15   -31.1 |    23.3   2.45   -16.9 |    31.4   2.39   -15.4 |    98.3   1.77   -31.1 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `KAMA(62,105)` | 2MA |     1.3   0.24    -8.1 |    -5.2  -2.06    -6.0 |    -0.7  -0.05    -9.0 |    -4.6   -0.2   -10.2 | - |
| 119 | `KAMA(52,89)` | 2MA |   -10.3   -1.2   -13.9 |     1.5   0.48    -6.3 |     1.4   0.37   -10.0 |    -7.7  -0.36   -13.9 | - |
| 120 | `KAMA(52,203)` | 2MA |   -10.3  -1.19   -13.9 |    -3.2  -1.35    -4.9 |     3.4    0.8    -8.0 |   -10.1  -0.58   -14.0 | - |


## 1d x VIDYA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.579**; TRAIN+VAL top-10 -> OOS top-10 overlap = **4/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,37], slow in [3,210] -> 42/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,24], slow in [4,233] -> 32/60 configs positive across TRAIN & VAL & OOS
- band members: VIDYA(2,3,4), VIDYA(2,3), VIDYA(3,4,38), VIDYA(2,5,118), VIDYA(2,4,19), VIDYA(3,6), VIDYA(4,5), VIDYA(2,10), VIDYA(4,8,17), VIDYA(4,5,75), VIDYA(3,5,186), VIDYA(3,18), VIDYA(5,10,14), VIDYA(2,26), VIDYA(2,8,22), VIDYA(4,15), VIDYA(4,10,208), VIDYA(5,6,34), VIDYA(6,9,24), VIDYA(6,13), VIDYA(5,12), VIDYA(2,12,14), VIDYA(6,14,15), VIDYA(3,12,27) (+50 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `VIDYA(2,3,4)` | 3MA |    26.1   1.89   -16.0 |    26.5   3.44   -11.4 |    43.0   3.26   -17.0 |   128.1   2.65   -17.0 | YES |
| 2 | `VIDYA(2,3)` | 2MA |    24.4   1.54   -22.5 |    27.2   3.44   -11.4 |    34.8   2.65   -17.5 |   113.3   2.26   -22.5 | YES |
| 3 | `VIDYA(3,4,38)` | 3MA |    12.3   1.17   -14.6 |    27.1   3.86    -9.6 |    34.2   3.14   -14.6 |    91.5   2.43   -14.6 | YES |
| 4 | `VIDYA(2,5,118)` | 3MA |    12.8   1.31   -12.9 |    22.9   3.37   -10.0 |    37.3   3.34   -14.6 |    90.4   2.46   -14.6 | YES |
| 5 | `VIDYA(2,4,19)` | 3MA |    13.1   1.12   -15.9 |    28.1   3.77    -9.9 |    30.3   2.62   -16.2 |    88.9   2.19   -16.2 | YES |
| 6 | `VIDYA(3,6)` | 2MA |    12.9   1.23   -13.1 |    12.8   2.03   -11.8 |    41.0    3.3   -15.2 |    79.6   2.11   -15.2 | YES |
| 7 | `VIDYA(4,5)` | 2MA |    13.4   1.28   -13.1 |    12.9   2.05   -11.8 |    40.1   3.25   -15.2 |    79.4   2.11   -15.2 | YES |
| 8 | `VIDYA(2,10)` | 2MA |    30.2   2.36    -9.9 |    12.8   2.97    -5.2 |    21.9   1.89   -20.9 |    79.0   2.08   -20.9 | YES |
| 9 | `VIDYA(4,8,17)` | 3MA |    21.4    2.2    -7.6 |    14.2   3.32    -5.4 |    25.4   2.48   -14.6 |    73.8   2.34   -14.6 | YES |
| 10 | `VIDYA(4,5,75)` | 3MA |    15.9   1.65   -12.4 |     9.2   1.59   -11.8 |    34.1   3.17   -14.6 |    69.7   2.12   -14.6 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `VIDYA(19,132,233)` | 3MA |    19.1   2.33    -8.6 |    -1.9  -0.82    -3.8 |     0.0    0.0     0.0 |    16.8   1.36    -8.6 | - |
| 119 | `VIDYA(102,237)` | 2MA |    15.6   2.13    -7.4 |    -0.6  -0.35    -2.3 |     0.0    0.0     0.0 |    14.9   1.36    -7.4 | - |
| 120 | `VIDYA(186,208,233)` | 3MA |    15.6   2.13    -7.4 |    -1.2  -0.56    -3.8 |     0.0    0.0     0.0 |    14.2   1.28    -8.0 | - |


# Timeframe: 4h
_Benchmark (equal-weight u10 buy-hold, no cost): FULL-2020 net = 134.0% (participation-tax reference)._

## 4h x EMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.437**; TRAIN+VAL top-10 -> OOS top-10 overlap = **3/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 60/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [4,233] -> 60/60 configs positive across TRAIN & VAL & OOS
- band members: EMA(2,26), EMA(8,16), EMA(12,13), EMA(3,18), EMA(10,19), EMA(2,3), EMA(6,13), EMA(5,12), EMA(6,14,15), EMA(5,10,14), EMA(4,15), EMA(2,3,4), EMA(6,33), EMA(5,37), EMA(4,8,17), EMA(3,12,27), EMA(3,102), EMA(3,44), EMA(4,37), EMA(8,39), EMA(2,30,84), EMA(2,19,22), EMA(8,84,186), EMA(4,86) (+96 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `EMA(2,26)` | 2MA |    44.6   2.41   -11.3 |    24.5   2.92   -12.7 |    52.5   3.47   -10.8 |   174.5   2.82   -16.1 | YES |
| 2 | `EMA(8,16)` | 2MA |    35.8   2.02   -13.7 |    33.3   3.89    -9.1 |    48.0   3.38   -10.1 |   167.8   2.82   -13.7 | YES |
| 3 | `EMA(12,13)` | 2MA |    34.1   1.93   -13.2 |    32.9   3.86    -9.3 |    45.6   3.27   -12.5 |   159.4   2.73   -13.2 | YES |
| 4 | `EMA(3,18)` | 2MA |    41.6   2.26   -12.3 |    29.9    3.4   -10.7 |    39.3   2.76   -12.6 |   156.3   2.62   -14.7 | YES |
| 5 | `EMA(10,19)` | 2MA |    32.0   1.86   -14.5 |    34.6   4.11    -8.0 |    41.1    3.0   -14.5 |   150.6   2.65   -14.5 | YES |
| 6 | `EMA(2,3)` | 2MA |    55.6   2.48   -21.4 |    17.1   1.79   -23.5 |    37.1   2.41   -17.7 |   150.0   2.26   -23.5 | YES |
| 7 | `EMA(6,13)` | 2MA |    33.9   1.89   -14.5 |    33.5    3.8    -9.4 |    37.7   2.68   -14.5 |   146.2    2.5   -14.5 | YES |
| 8 | `EMA(5,12)` | 2MA |    40.2    2.2   -12.0 |    29.5   3.41    -9.7 |    35.2   2.54   -12.0 |   145.5   2.51   -13.5 | YES |
| 9 | `EMA(6,14,15)` | 3MA |    34.7   2.02   -14.9 |    33.0    4.0    -8.1 |    37.1   2.86   -13.7 |   145.5   2.67   -14.9 | YES |
| 10 | `EMA(5,10,14)` | 3MA |    36.5    2.1   -14.7 |    29.3   3.58    -8.0 |    37.7   2.89   -11.1 |   143.1   2.63   -14.7 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `EMA(102,237)` | 2MA |    17.2   1.98    -8.0 |     5.9   2.42    -4.0 |    14.2   2.86    -7.8 |    41.6   2.26    -9.2 | YES |
| 119 | `EMA(62,239)` | 2MA |    17.2   2.17    -6.9 |     5.1   1.99    -5.1 |    14.1   2.63    -8.0 |    40.5   2.23    -8.0 | YES |
| 120 | `EMA(186,208,233)` | 3MA |    11.9   1.56    -9.4 |     5.5   2.34    -4.0 |    14.5   3.76    -5.4 |    35.3   2.25   -10.3 | YES |


## 4h x SMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.339**; TRAIN+VAL top-10 -> OOS top-10 overlap = **3/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 56/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,75], slow in [4,233] -> 59/60 configs positive across TRAIN & VAL & OOS
- band members: SMA(3,6), SMA(2,26), SMA(4,5), SMA(2,3,4), SMA(102,103), SMA(10,19), SMA(3,44), SMA(6,33), SMA(4,37), SMA(3,5,186), SMA(5,37), SMA(8,39), SMA(10,11,233), SMA(18,23), SMA(37,38), SMA(86,122), SMA(3,18), SMA(24,106,118), SMA(3,4,38), SMA(3,75,132), SMA(2,169), SMA(15,84,148), SMA(4,86), SMA(3,30,166) (+91 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `SMA(3,6)` | 2MA |    56.4   2.53   -14.6 |    23.7   2.45   -21.4 |    50.7    3.0   -16.6 |   191.6   2.62   -22.8 | YES |
| 2 | `SMA(2,26)` | 2MA |    41.9   2.25   -13.7 |    34.8   3.81   -11.7 |    51.2   3.45   -12.9 |   189.1   2.94   -13.7 | YES |
| 3 | `SMA(4,5)` | 2MA |    57.6   2.47   -20.5 |    23.7   2.29   -22.8 |    42.0   2.46   -19.9 |   176.7   2.37   -22.8 | YES |
| 4 | `SMA(2,3,4)` | 3MA |    64.7   2.93   -16.9 |    25.7   2.57   -21.4 |    26.9    1.9   -21.6 |   162.7   2.43   -21.6 | YES |
| 5 | `SMA(102,103)` | 2MA |    29.2   2.28   -14.1 |    31.4   4.56    -7.0 |    47.9    3.9   -17.3 |   151.0   3.26   -17.3 | YES |
| 6 | `SMA(10,19)` | 2MA |    35.6   1.91   -20.7 |    37.3   4.25    -9.7 |    29.7   2.25   -16.9 |   141.5   2.44   -20.7 | YES |
| 7 | `SMA(3,44)` | 2MA |    34.7   1.97   -14.4 |    28.9   3.44    -8.2 |    38.3   2.82   -17.1 |   140.1   2.52   -17.1 | YES |
| 8 | `SMA(6,33)` | 2MA |    28.1   1.66   -15.0 |    32.9   3.86    -7.7 |    40.9   3.06   -15.2 |   139.8   2.55   -15.2 | YES |
| 9 | `SMA(4,37)` | 2MA |    31.2   1.81   -13.2 |    32.4   3.79    -8.7 |    38.0    2.8   -17.8 |   139.7   2.51   -17.8 | YES |
| 10 | `SMA(3,5,186)` | 3MA |    32.0   2.03   -12.1 |    32.8   4.32    -7.6 |    32.5   2.57   -20.8 |   132.1   2.61   -20.8 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `SMA(12,14,75)` | 3MA |     6.6   0.58   -21.0 |    12.1   1.84   -13.1 |    10.5   1.12   -19.3 |    32.1    1.0   -21.5 | YES |
| 119 | `SMA(6,13)` | 2MA |   -19.1  -0.65   -47.8 |    20.7   2.41   -16.5 |    34.2   2.41   -14.4 |    31.0   0.82   -47.8 | - |
| 120 | `SMA(186,208,233)` | 3MA |    -4.2  -0.62   -11.3 |    13.6   3.53    -5.7 |    11.6   2.42    -9.5 |    21.4   1.38   -11.3 | - |


## 4h x WMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.237**; TRAIN+VAL top-10 -> OOS top-10 overlap = **7/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 56/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [4,233] -> 58/60 configs positive across TRAIN & VAL & OOS
- band members: WMA(4,5), WMA(3,6), WMA(5,37), WMA(3,44), WMA(15,28), WMA(4,37), WMA(6,33), WMA(12,33), WMA(2,26), WMA(8,39), WMA(18,23), WMA(5,6,34), WMA(2,73), WMA(10,55), WMA(2,3,4), WMA(2,169), WMA(4,199), WMA(22,28), WMA(6,77), WMA(5,12,208), WMA(5,199), WMA(10,19), WMA(12,106,148), WMA(6,22,24) (+90 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `WMA(4,5)` | 2MA |    68.2   2.87   -13.0 |    30.9   3.04   -18.2 |    42.5   2.57   -20.3 |   213.8   2.74   -20.3 | YES |
| 2 | `WMA(3,6)` | 2MA |    70.3    3.0   -11.0 |    25.6   2.62   -20.4 |    42.3    2.6   -20.3 |   204.4   2.71   -20.4 | YES |
| 3 | `WMA(5,37)` | 2MA |    39.6   2.17   -13.7 |    31.6   3.69    -8.4 |    48.8   3.34   -14.2 |   173.4   2.83   -14.2 | YES |
| 4 | `WMA(3,44)` | 2MA |    41.6   2.28   -13.0 |    29.2   3.41   -10.4 |    49.3   3.37   -12.8 |   173.1   2.83   -14.0 | YES |
| 5 | `WMA(15,28)` | 2MA |    31.0    1.8   -13.2 |    39.2   4.44    -8.1 |    48.5   3.51   -14.6 |   170.9   2.88   -14.6 | YES |
| 6 | `WMA(4,37)` | 2MA |    39.4   2.16   -13.0 |    33.0   3.83    -8.0 |    45.9   3.17   -12.8 |   170.4   2.79   -13.0 | YES |
| 7 | `WMA(6,33)` | 2MA |    37.0   2.03   -13.8 |    30.3   3.58    -8.5 |    45.0   3.16   -13.2 |   158.9   2.68   -13.8 | YES |
| 8 | `WMA(12,33)` | 2MA |    31.4   1.82   -13.7 |    36.3    4.2    -8.5 |    43.9   3.19   -14.3 |   157.7   2.73   -14.3 | YES |
| 9 | `WMA(2,26)` | 2MA |    50.4   2.59   -10.6 |    26.3   3.03   -11.9 |    33.8    2.5   -12.2 |   154.2    2.6   -15.3 | YES |
| 10 | `WMA(8,39)` | 2MA |    31.8   1.83   -13.4 |    30.7   3.58    -8.6 |    44.2   3.19   -13.8 |   148.5   2.61   -13.8 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `WMA(102,237)` | 2MA |     5.8   0.82   -13.3 |    16.9   4.27    -4.5 |    20.1   3.44    -7.6 |    48.5    2.4   -13.3 | YES |
| 119 | `WMA(5,10,14)` | 3MA |    -8.6   -0.2   -39.8 |    25.8   3.13   -12.5 |    26.3   2.04   -13.1 |    45.3   1.09   -39.8 | - |
| 120 | `WMA(62,239)` | 2MA |     7.8   1.03   -11.6 |    15.2   3.73    -5.4 |    12.5   2.09    -8.0 |    39.7   1.94   -11.6 | YES |


## 4h x HMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.332**; TRAIN+VAL top-10 -> OOS top-10 overlap = **6/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 54/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [14,233] -> 55/60 configs positive across TRAIN & VAL & OOS
- band members: HMA(4,8,17), HMA(5,86), HMA(4,86), HMA(8,16), HMA(6,13), HMA(8,91), HMA(10,128), HMA(4,15), HMA(18,128), HMA(6,77), HMA(5,10,14), HMA(6,9,24), HMA(5,12), HMA(10,19), HMA(31,124), HMA(15,151), HMA(12,77), HMA(2,12,14), HMA(2,169), HMA(37,89), HMA(52,89), HMA(22,151), HMA(3,102), HMA(62,105) (+85 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `HMA(4,8,17)` | 3MA |    71.2    3.2   -12.4 |    22.0   2.43   -19.4 |    42.6   2.85   -13.9 |   197.9   2.86   -21.5 | YES |
| 2 | `HMA(5,86)` | 2MA |    56.3   2.72   -16.5 |    30.9   3.49   -11.0 |    45.6   3.15   -11.0 |   197.8   2.97   -16.5 | YES |
| 3 | `HMA(4,86)` | 2MA |    54.6   2.64   -15.0 |    32.6   3.63   -10.6 |    43.4   3.03   -11.7 |   193.9   2.93   -15.0 | YES |
| 4 | `HMA(8,16)` | 2MA |    62.7    2.7   -16.1 |    23.0   2.44   -22.4 |    43.7   2.76   -16.7 |   187.7   2.62   -23.1 | YES |
| 5 | `HMA(6,13)` | 2MA |    58.7   2.43   -25.3 |    23.7   2.37   -20.1 |    40.5   2.58   -15.5 |   176.0   2.44   -25.3 | YES |
| 6 | `HMA(8,91)` | 2MA |    47.8    2.5   -13.4 |    26.7   3.12   -14.5 |    46.7   3.25   -13.0 |   174.5   2.83   -17.7 | YES |
| 7 | `HMA(10,128)` | 2MA |    36.9   2.22   -14.8 |    35.3   4.14    -8.3 |    44.2   3.33   -14.3 |   167.0   2.96   -14.8 | YES |
| 8 | `HMA(4,15)` | 2MA |    38.0   1.68   -33.3 |    26.5   2.59   -21.5 |    48.6   2.98   -15.7 |   159.5   2.26   -33.3 | YES |
| 9 | `HMA(18,128)` | 2MA |    34.8   2.11   -13.3 |    38.0   4.49    -6.8 |    38.1   2.94   -14.7 |   156.8   2.84   -14.7 | YES |
| 10 | `HMA(6,77)` | 2MA |    40.3   2.22   -16.0 |    24.9   2.96   -12.0 |    40.8   2.92   -13.3 |   146.8   2.56   -16.0 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `HMA(102,103)` | 2MA |     5.0   0.91    -7.4 |     7.0   2.07    -7.4 |     8.3   2.23    -6.2 |    21.7   1.59    -7.4 | YES |
| 119 | `HMA(15,22,48)` | 3MA |   -18.5  -0.69   -42.7 |    20.6   2.78   -12.5 |    21.7   1.95   -18.6 |    19.7   0.65   -42.7 | - |
| 120 | `HMA(2,3,4)` | 3MA |    -6.7  -0.12   -39.6 |    10.3   1.19   -22.8 |    14.5   1.24   -23.9 |    17.9    0.6   -39.6 | - |


## 4h x DEMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.346**; TRAIN+VAL top-10 -> OOS top-10 overlap = **1/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 56/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [4,233] -> 55/60 configs positive across TRAIN & VAL & OOS
- band members: DEMA(26,32), DEMA(15,28), DEMA(12,33), DEMA(8,39), DEMA(22,28), DEMA(18,23), DEMA(15,22,48), DEMA(10,55), DEMA(3,44), DEMA(19,27,38), DEMA(3,102), DEMA(4,86), DEMA(5,86), DEMA(3,6), DEMA(4,5), DEMA(18,55), DEMA(6,77), DEMA(4,37), DEMA(8,210), DEMA(2,10), DEMA(5,37), DEMA(6,210), DEMA(3,5,186), DEMA(5,199) (+87 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `DEMA(26,32)` | 2MA |    35.5    2.2   -12.5 |    45.8   5.26    -6.7 |    38.6   3.08   -12.4 |   173.9   3.11   -12.5 | YES |
| 2 | `DEMA(15,28)` | 2MA |    37.6    2.1   -17.4 |    38.3   4.41    -8.7 |    41.6   3.02   -11.2 |   169.5   2.83   -17.4 | YES |
| 3 | `DEMA(12,33)` | 2MA |    39.3   2.17   -17.2 |    33.8   3.98    -8.7 |    41.2   2.98   -11.9 |   163.1   2.76   -17.2 | YES |
| 4 | `DEMA(8,39)` | 2MA |    43.1   2.45   -16.9 |    29.0   3.43    -8.7 |    40.3   2.94   -10.5 |   158.8   2.77   -16.9 | YES |
| 5 | `DEMA(22,28)` | 2MA |    35.6   2.18   -16.0 |    38.4   4.29    -8.9 |    37.8   3.02    -9.9 |   158.5   2.89   -16.0 | YES |
| 6 | `DEMA(18,23)` | 2MA |    30.5   1.78   -18.8 |    36.1   4.22    -9.0 |    45.2   3.25   -11.1 |   157.8   2.73   -18.8 | YES |
| 7 | `DEMA(15,22,48)` | 3MA |    42.2   2.78   -10.2 |    30.4   4.17    -6.5 |    37.3   3.33   -10.3 |   154.5   3.21   -10.3 | YES |
| 8 | `DEMA(10,55)` | 2MA |    32.8   2.07   -12.9 |    38.4   4.33    -9.6 |    38.0   2.87   -10.6 |   153.8   2.79   -12.9 | YES |
| 9 | `DEMA(3,44)` | 2MA |    50.1   2.62   -16.1 |    31.6   3.51   -10.3 |    27.3   2.13   -11.5 |   151.4    2.6   -16.1 | YES |
| 10 | `DEMA(19,27,38)` | 3MA |    37.9   2.46   -10.6 |    36.3   4.61    -6.2 |    33.7   2.85   -11.5 |   151.3   2.99   -11.5 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `DEMA(6,9,24)` | 3MA |    -2.3    0.1   -38.0 |    15.6   2.14   -11.6 |    19.5   1.79   -12.5 |    35.0   0.97   -38.0 | - |
| 119 | `DEMA(10,19)` | 2MA |   -25.3  -1.13   -43.7 |    31.4   3.67    -9.0 |    36.1   2.74   -16.5 |    33.6   0.91   -43.7 | - |
| 120 | `DEMA(2,8,22)` | 3MA |   -21.2  -0.82   -49.6 |    19.3   2.52   -13.5 |    25.0   2.13   -17.2 |    17.5    0.6   -49.6 | - |


## 4h x TEMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.19**; TRAIN+VAL top-10 -> OOS top-10 overlap = **1/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 52/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [4,233] -> 55/60 configs positive across TRAIN & VAL & OOS
- band members: TEMA(6,13), TEMA(6,77), TEMA(26,32), TEMA(26,75), TEMA(2,4,19), TEMA(15,65), TEMA(3,18), TEMA(2,169), TEMA(4,15), TEMA(5,12), TEMA(12,77), TEMA(8,16), TEMA(2,26), TEMA(4,8,17), TEMA(18,55), TEMA(31,53), TEMA(5,86), TEMA(5,10,14), TEMA(3,5,186), TEMA(12,178), TEMA(10,34,60), TEMA(5,24,34), TEMA(19,27,38), TEMA(22,65) (+83 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `TEMA(6,13)` | 2MA |    47.7    2.3   -18.7 |    18.9   2.07   -18.7 |    48.1   3.16   -16.1 |   160.1   2.49   -18.7 | YES |
| 2 | `TEMA(6,77)` | 2MA |    36.5   2.08   -19.8 |    32.2   3.65   -10.7 |    37.3   2.79   -11.0 |   147.6   2.61   -19.8 | YES |
| 3 | `TEMA(26,32)` | 2MA |    22.9   1.48   -17.7 |    28.9   3.64   -10.1 |    55.9   3.93   -10.7 |   147.1   2.71   -17.7 | YES |
| 4 | `TEMA(26,75)` | 2MA |    26.0   1.85   -14.1 |    44.4   4.99    -7.2 |    33.8   2.87   -12.7 |   143.5   2.88   -14.1 | YES |
| 5 | `TEMA(2,4,19)` | 3MA |    63.9   3.01   -13.8 |    18.3   2.02   -16.1 |    25.3    1.9   -15.9 |   143.0   2.36   -17.4 | YES |
| 6 | `TEMA(15,65)` | 2MA |    23.6   1.52   -20.6 |    32.7   3.88    -9.2 |    47.2   3.45   -10.7 |   141.3   2.63   -20.6 | YES |
| 7 | `TEMA(3,18)` | 2MA |    49.1   2.25   -25.9 |    19.4   2.05   -21.3 |    35.3   2.38   -15.7 |   141.0   2.22   -25.9 | YES |
| 8 | `TEMA(2,169)` | 2MA |    33.2   2.32   -10.6 |    26.9   3.32   -13.8 |    39.8   2.99   -12.5 |   136.3    2.7   -13.8 | YES |
| 9 | `TEMA(4,15)` | 2MA |    46.4   2.06   -25.8 |    12.7   1.44   -23.5 |    40.0   2.66   -15.4 |   130.9   2.09   -25.8 | YES |
| 10 | `TEMA(5,12)` | 2MA |    40.3   1.93   -25.6 |    14.0   1.57   -23.1 |    43.9   2.85   -16.8 |   130.1   2.12   -25.6 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `TEMA(12,33)` | 2MA |   -20.5  -0.95   -39.6 |    31.2   3.53   -11.5 |    30.0   2.34   -19.2 |    35.6   0.96   -39.6 | - |
| 119 | `TEMA(18,23)` | 2MA |   -19.1  -0.89   -40.6 |    28.2   3.25   -12.6 |    24.1    2.0   -19.2 |    28.7   0.84   -40.6 | - |
| 120 | `TEMA(15,28)` | 2MA |   -19.7  -0.89   -41.6 |    29.0   3.31   -11.9 |    23.4   1.95   -18.8 |    27.8   0.81   -41.6 | - |


## 4h x KAMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.34**; TRAIN+VAL top-10 -> OOS top-10 overlap = **2/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 57/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [4,233] -> 54/60 configs positive across TRAIN & VAL & OOS
- band members: KAMA(4,5), KAMA(3,6), KAMA(3,44), KAMA(4,37), KAMA(3,18), KAMA(4,199), KAMA(2,26), KAMA(5,199), KAMA(5,37), KAMA(2,169), KAMA(2,3), KAMA(6,13), KAMA(18,55), KAMA(10,11,233), KAMA(4,15), KAMA(3,5,186), KAMA(4,5,75), KAMA(2,10), KAMA(4,15,208), KAMA(5,12), KAMA(5,6,34), KAMA(5,12,208), KAMA(3,102), KAMA(10,19) (+87 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `KAMA(4,5)` | 2MA |    33.8   1.67   -33.4 |    27.6   2.81   -22.1 |    43.9   3.01   -16.3 |   145.7   2.31   -33.4 | YES |
| 2 | `KAMA(3,6)` | 2MA |    42.4   2.05   -21.4 |    18.9   2.12   -19.7 |    38.2   2.67   -12.6 |   133.9   2.24   -21.4 | YES |
| 3 | `KAMA(3,44)` | 2MA |    24.1   1.64   -13.8 |    28.4    3.6   -12.8 |    38.6   3.18   -11.4 |   120.9   2.55   -13.8 | YES |
| 4 | `KAMA(4,37)` | 2MA |    30.0    2.0   -12.1 |    29.0   3.71   -11.9 |    30.3   2.59   -12.0 |   118.7   2.52   -12.1 | YES |
| 5 | `KAMA(3,18)` | 2MA |    27.2   1.67   -15.9 |    34.5    4.2    -8.3 |    26.6   2.13   -16.8 |   116.7   2.29   -16.8 | YES |
| 6 | `KAMA(4,199)` | 2MA |    26.3   2.15   -14.1 |    27.7   4.28   -10.3 |    34.0   3.24   -12.9 |   116.2   2.92   -14.1 | YES |
| 7 | `KAMA(2,26)` | 2MA |    28.7   1.73   -16.5 |    25.2   3.09   -14.7 |    33.1   2.58   -13.8 |   114.6   2.26   -16.5 | YES |
| 8 | `KAMA(5,199)` | 2MA |    28.1   2.28   -12.8 |    26.6   4.13   -10.9 |    28.9   3.02   -11.0 |   108.9   2.89   -12.8 | YES |
| 9 | `KAMA(5,37)` | 2MA |    27.1   1.84   -12.2 |    28.2    3.7   -11.4 |    27.9   2.42   -13.3 |   108.5   2.39   -13.3 | YES |
| 10 | `KAMA(2,169)` | 2MA |    24.2   1.85   -16.8 |    24.4   3.67   -12.7 |    34.9   3.19   -11.0 |   108.4   2.64   -16.8 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `KAMA(30,132,186)` | 3MA |    -8.8  -1.12   -16.4 |    13.5   3.19    -7.1 |    13.6   2.02   -12.7 |    17.5   0.94   -17.0 | - |
| 119 | `KAMA(19,132,233)` | 3MA |    -8.0  -1.15   -16.3 |    13.6   3.21    -6.6 |     9.4   1.52   -11.9 |    14.3   0.83   -16.3 | - |
| 120 | `KAMA(62,105)` | 2MA |   -12.1  -0.97   -26.6 |    10.9   2.12   -13.5 |     6.6    1.1   -15.6 |     4.0   0.29   -26.6 | - |


## 4h x VIDYA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.745**; TRAIN+VAL top-10 -> OOS top-10 overlap = **1/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 57/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,75], slow in [4,233] -> 58/60 configs positive across TRAIN & VAL & OOS
- band members: VIDYA(2,3,4), VIDYA(4,5), VIDYA(2,26), VIDYA(3,6), VIDYA(2,19,22), VIDYA(3,5,186), VIDYA(2,10), VIDYA(2,5,118), VIDYA(3,4,38), VIDYA(5,12,208), VIDYA(3,8,43), VIDYA(2,8,22), VIDYA(6,22,24), VIDYA(3,12,27), VIDYA(4,37), VIDYA(5,12), VIDYA(8,14,38), VIDYA(2,4,19), VIDYA(2,3), VIDYA(8,16), VIDYA(4,30,67), VIDYA(5,6,34), VIDYA(4,5,75), VIDYA(6,9,24) (+91 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `VIDYA(2,3,4)` | 3MA |    31.0   1.88   -15.1 |    32.8   4.02    -7.5 |    42.8   3.24   -10.3 |   148.4   2.74   -15.1 | YES |
| 2 | `VIDYA(4,5)` | 2MA |    22.9   1.44   -19.9 |    35.1   4.18    -8.5 |    36.1   2.74   -13.3 |   126.0   2.41   -19.9 | YES |
| 3 | `VIDYA(2,26)` | 2MA |    26.9   1.96   -18.9 |    27.1   3.87    -9.6 |    39.9    3.4    -9.5 |   125.5    2.8   -18.9 | YES |
| 4 | `VIDYA(3,6)` | 2MA |    24.0   1.49   -19.6 |    33.1    4.0    -8.5 |    36.0   2.75   -13.7 |   124.5   2.39   -19.6 | YES |
| 5 | `VIDYA(2,19,22)` | 3MA |    29.9   2.14   -11.2 |    27.6   4.19   -10.3 |    33.3   2.94   -16.4 |   121.0   2.76   -16.4 | YES |
| 6 | `VIDYA(3,5,186)` | 3MA |    13.7   1.19   -16.3 |    30.6   4.22    -9.3 |    46.9   4.23    -8.6 |   118.0   2.88   -16.3 | YES |
| 7 | `VIDYA(2,10)` | 2MA |    22.8   1.45   -20.9 |    30.0   3.75    -8.1 |    36.2    2.7   -14.2 |   117.4    2.3   -20.9 | YES |
| 8 | `VIDYA(2,5,118)` | 3MA |    18.1   1.49   -13.9 |    27.6   3.92    -9.2 |    44.0   4.09    -8.6 |   117.1   2.87   -13.9 | YES |
| 9 | `VIDYA(3,4,38)` | 3MA |    24.5   1.76   -13.9 |    29.7    4.2    -7.7 |    33.9   2.94   -13.2 |   116.2   2.62   -13.9 | YES |
| 10 | `VIDYA(5,12,208)` | 3MA |    18.8   1.65   -14.9 |    28.3   4.33    -8.3 |    41.7   3.85   -12.9 |   116.1   2.95   -14.9 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `VIDYA(37,203)` | 2MA |    16.7    2.2    -6.1 |     3.1   1.87    -2.6 |    -6.0  -3.29    -7.1 |    13.1   1.13    -9.8 | - |
| 119 | `VIDYA(44,203)` | 2MA |    11.3   1.68    -7.2 |     3.3   2.07    -2.6 |    -4.3  -2.95    -4.9 |    10.1   0.97    -8.6 | - |
| 120 | `VIDYA(52,203)` | 2MA |     9.8    1.5    -8.4 |     2.5    1.6    -2.6 |    -3.9  -2.92    -4.9 |     8.2   0.82    -9.6 | - |


# Timeframe: 2h
_Benchmark (equal-weight u10 buy-hold, no cost): FULL-2020 net = 141.3% (participation-tax reference)._

## 2h x EMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.689**; TRAIN+VAL top-10 -> OOS top-10 overlap = **5/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 57/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [4,233] -> 60/60 configs positive across TRAIN & VAL & OOS
- band members: EMA(15,28), EMA(4,37), EMA(18,23), EMA(12,33), EMA(8,39), EMA(6,33), EMA(3,44), EMA(5,37), EMA(22,28), EMA(26,32), EMA(19,27,38), EMA(10,55), EMA(2,73), EMA(37,38), EMA(12,13), EMA(15,22,48), EMA(8,16), EMA(22,65), EMA(2,169), EMA(6,77), EMA(6,22,24), EMA(18,55), EMA(10,19), EMA(30,43,186) (+93 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `EMA(15,28)` | 2MA |    38.7   2.06   -13.1 |    27.8   3.06    -7.4 |    52.5   3.02   -11.0 |   170.4    2.6   -13.1 | YES |
| 2 | `EMA(4,37)` | 2MA |    47.6   2.45   -12.4 |    32.4   3.32    -9.5 |    38.1   2.27   -11.5 |   169.7   2.55   -13.0 | YES |
| 3 | `EMA(18,23)` | 2MA |    39.7    2.1   -12.6 |    28.0   3.08    -7.2 |    49.4   2.88   -11.0 |   167.3   2.58   -12.6 | YES |
| 4 | `EMA(12,33)` | 2MA |    37.5   2.01   -13.2 |    29.4   3.21    -7.0 |    48.1    2.8   -11.7 |   163.5   2.53   -13.2 | YES |
| 5 | `EMA(8,39)` | 2MA |    35.4   1.91   -13.3 |    27.9   3.06    -7.7 |    50.3   2.83   -10.9 |   160.2   2.48   -13.3 | YES |
| 6 | `EMA(6,33)` | 2MA |    44.2    2.3   -12.5 |    31.4   3.24    -8.0 |    37.0   2.23   -11.2 |   159.5   2.46   -12.8 | YES |
| 7 | `EMA(3,44)` | 2MA |    45.3   2.35   -11.2 |    24.7   2.68   -11.5 |    40.9    2.4   -11.0 |   155.3   2.42   -13.9 | YES |
| 8 | `EMA(5,37)` | 2MA |    42.1   2.21   -12.2 |    29.2   3.09    -7.2 |    38.2   2.28   -10.7 |   153.7   2.41   -12.2 | YES |
| 9 | `EMA(22,28)` | 2MA |    37.3   2.02   -12.5 |    25.5   2.85    -8.9 |    46.0   2.72   -12.9 |   151.4   2.44   -12.9 | YES |
| 10 | `EMA(26,32)` | 2MA |    32.1   1.84   -13.3 |    27.7   3.15    -7.9 |    43.7   2.64   -13.4 |   142.5    2.4   -13.4 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `EMA(62,239)` | 2MA |     4.6   0.64    -9.3 |    11.8   2.67    -6.5 |    17.1   2.11    -7.2 |    36.9   1.64   -10.9 | YES |
| 119 | `EMA(102,237)` | 2MA |     3.4   0.55    -9.2 |    12.7    3.0    -6.6 |    16.1   2.12    -6.9 |    35.2    1.7   -11.1 | YES |
| 120 | `EMA(4,5)` | 2MA |   -10.9   -0.3   -37.7 |    21.9   2.19   -16.5 |    21.8   1.38   -19.1 |    32.3   0.78   -37.7 | - |


## 2h x SMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.475**; TRAIN+VAL top-10 -> OOS top-10 overlap = **5/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 56/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [15,233] -> 54/60 configs positive across TRAIN & VAL & OOS
- band members: SMA(2,73), SMA(8,91), SMA(10,55), SMA(6,77), SMA(12,77), SMA(15,65), SMA(5,86), SMA(3,44), SMA(4,86), SMA(18,55), SMA(3,102), SMA(6,33), SMA(8,9,75), SMA(5,37), SMA(4,37), SMA(8,39), SMA(22,65), SMA(4,199), SMA(186,208,233), SMA(18,23), SMA(5,199), SMA(6,13), SMA(31,53), SMA(19,43,233) (+86 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `SMA(2,73)` | 2MA |    46.9   2.41   -13.7 |    20.4    2.3   -12.3 |    48.7   2.75   -14.8 |   162.9    2.5   -14.8 | YES |
| 2 | `SMA(8,91)` | 2MA |    34.2   1.99   -14.2 |    28.0   3.18    -7.5 |    47.3   2.88   -13.9 |   152.8   2.56   -14.2 | YES |
| 3 | `SMA(10,55)` | 2MA |    34.0   1.89   -12.9 |    25.8   2.83    -8.9 |    49.4    2.9   -12.1 |   151.8   2.45   -12.9 | YES |
| 4 | `SMA(6,77)` | 2MA |    38.5   2.11   -13.1 |    25.4   2.87    -9.0 |    44.6   2.63   -14.7 |   151.2   2.45   -14.7 | YES |
| 5 | `SMA(12,77)` | 2MA |    30.1   1.81   -13.1 |    31.6   3.49    -8.0 |    45.7   2.74   -16.1 |   149.4    2.5   -16.1 | YES |
| 6 | `SMA(15,65)` | 2MA |    27.9   1.68   -14.2 |    32.0   3.54    -7.3 |    47.0   2.86   -12.2 |   148.4   2.49   -14.2 | YES |
| 7 | `SMA(5,86)` | 2MA |    40.8    2.2   -13.2 |    23.2   2.66    -9.7 |    42.5   2.55   -15.6 |   147.3   2.41   -15.6 | YES |
| 8 | `SMA(3,44)` | 2MA |    30.9   1.67   -20.0 |    33.2   3.38    -9.9 |    41.4   2.44   -11.6 |   146.7    2.3   -20.0 | YES |
| 9 | `SMA(4,86)` | 2MA |    42.2   2.26   -12.9 |    22.0   2.53   -10.8 |    39.2   2.36   -16.0 |   141.5   2.34   -16.0 | YES |
| 10 | `SMA(18,55)` | 2MA |    27.1    1.6   -13.6 |    30.9   3.46    -8.0 |    44.3   2.74   -12.7 |   140.0   2.39   -13.6 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `SMA(2,12,14)` | 3MA |   -12.6  -0.56   -29.0 |     9.2   1.15   -18.7 |    27.8    1.8   -14.5 |    22.0   0.65   -29.0 | - |
| 119 | `SMA(3,6)` | 2MA |    -6.1  -0.02   -37.8 |    22.0   2.01   -24.2 |     3.7   0.44   -25.1 |    18.8   0.56   -37.8 | - |
| 120 | `SMA(12,13)` | 2MA |   -17.9   -0.7   -38.4 |    12.5   1.34   -23.0 |    19.4   1.21   -18.6 |    10.3   0.41   -38.4 | - |


## 2h x WMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.526**; TRAIN+VAL top-10 -> OOS top-10 overlap = **5/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 58/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [4,233] -> 55/60 configs positive across TRAIN & VAL & OOS
- band members: WMA(2,73), WMA(4,86), WMA(6,77), WMA(5,86), WMA(3,102), WMA(12,77), WMA(8,91), WMA(31,53), WMA(22,65), WMA(15,65), WMA(18,55), WMA(26,75), WMA(37,38), WMA(10,55), WMA(18,128), WMA(10,128), WMA(8,39), WMA(3,44), WMA(37,89), WMA(15,151), WMA(44,75), WMA(22,28), WMA(26,32), WMA(19,27,38) (+89 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `WMA(2,73)` | 2MA |    52.0   2.61   -11.8 |    24.6    2.6   -13.3 |    53.1   2.93   -11.1 |   189.8   2.71   -14.7 | YES |
| 2 | `WMA(4,86)` | 2MA |    45.5   2.35   -12.7 |    27.1   2.95    -8.9 |    52.8   2.92   -13.4 |   182.6   2.67   -13.4 | YES |
| 3 | `WMA(6,77)` | 2MA |    47.1    2.4   -12.7 |    28.2   3.04    -8.4 |    47.3   2.69   -13.8 |   177.8   2.63   -13.8 | YES |
| 4 | `WMA(5,86)` | 2MA |    44.7   2.31   -12.7 |    28.3   3.09    -7.5 |    49.4   2.79   -13.7 |   177.3   2.63   -13.7 | YES |
| 5 | `WMA(3,102)` | 2MA |    50.8   2.58   -11.6 |    26.4   2.93    -8.8 |    43.6   2.52   -14.2 |   173.8   2.61   -14.2 | YES |
| 6 | `WMA(12,77)` | 2MA |    41.1   2.18   -12.2 |    27.5   3.02    -7.5 |    48.8   2.84   -12.6 |   167.8   2.58   -12.6 | YES |
| 7 | `WMA(8,91)` | 2MA |    43.8   2.29   -12.9 |    29.6   3.25    -7.2 |    40.0   2.37   -14.5 |   161.0    2.5   -14.5 | YES |
| 8 | `WMA(31,53)` | 2MA |    28.7   1.64   -16.3 |    34.1   3.56    -8.6 |    50.4   2.99   -11.4 |   159.8   2.53   -16.3 | YES |
| 9 | `WMA(22,65)` | 2MA |    32.6   1.83   -13.3 |    26.6   2.93    -8.5 |    53.9    3.1   -11.1 |   158.3   2.52   -13.3 | YES |
| 10 | `WMA(15,65)` | 2MA |    35.3   1.93   -13.7 |    24.2   2.68    -9.1 |    49.2   2.88   -13.4 |   150.8   2.42   -13.7 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `WMA(5,6,34)` | 3MA |    -4.1  -0.05   -21.9 |    22.8   2.58    -9.7 |     9.4    0.8   -18.3 |    28.8   0.79   -21.9 | - |
| 119 | `WMA(3,6)` | 2MA |   -12.7  -0.29   -48.4 |    11.2   1.16   -25.0 |    15.0   1.01   -21.4 |    11.7   0.44   -48.4 | - |
| 120 | `WMA(4,5)` | 2MA |   -18.9  -0.53   -52.7 |    16.2   1.55   -23.6 |    11.6   0.84   -22.0 |     5.2   0.34   -52.7 | - |


## 2h x HMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.141**; TRAIN+VAL top-10 -> OOS top-10 overlap = **1/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 51/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [4,233] -> 51/60 configs positive across TRAIN & VAL & OOS
- band members: HMA(12,178), HMA(8,210), HMA(6,210), HMA(2,169), HMA(37,203), HMA(26,172), HMA(4,199), HMA(44,203), HMA(22,151), HMA(5,199), HMA(15,151), HMA(62,239), HMA(12,33), HMA(10,55), HMA(18,55), HMA(15,34,186), HMA(75,132,148), HMA(19,27,38), HMA(15,28), HMA(52,203), HMA(18,23), HMA(2,60,208), HMA(22,28), HMA(31,124) (+78 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `HMA(12,178)` | 2MA |    40.9   2.16   -15.5 |    35.4   3.58   -10.0 |    53.8   3.07   -10.0 |   193.4   2.79   -15.5 | YES |
| 2 | `HMA(8,210)` | 2MA |    45.5   2.43   -17.5 |    32.2   3.26   -11.2 |    47.3   2.83   -11.7 |   183.4   2.76   -17.5 | YES |
| 3 | `HMA(6,210)` | 2MA |    43.3   2.36   -18.6 |    31.9   3.22   -10.3 |    42.7   2.59   -14.1 |   169.7   2.63   -18.6 | YES |
| 4 | `HMA(2,169)` | 2MA |    34.9   1.79   -16.0 |    33.0   3.25    -9.9 |    43.2   2.42   -11.3 |   157.1   2.32   -16.0 | YES |
| 5 | `HMA(37,203)` | 2MA |    29.0   1.89   -17.1 |    34.1   3.63    -9.7 |    47.0   2.87   -11.5 |   154.3   2.64   -17.1 | YES |
| 6 | `HMA(26,172)` | 2MA |    19.4   1.28   -16.3 |    32.0   3.42    -9.2 |    58.9   3.29   -10.8 |   150.3   2.49   -16.3 | YES |
| 7 | `HMA(4,199)` | 2MA |    34.4   1.87   -23.1 |    32.2   3.27   -13.3 |    36.5   2.27   -14.1 |   142.7   2.31   -23.1 | YES |
| 8 | `HMA(44,203)` | 2MA |    23.1   1.55   -18.0 |    38.4   4.02    -9.2 |    40.9   2.73    -9.0 |   140.1   2.54   -18.0 | YES |
| 9 | `HMA(22,151)` | 2MA |    17.1   1.15   -16.8 |    26.5   2.84   -10.0 |    60.4   3.41   -10.1 |   137.6   2.36   -16.8 | YES |
| 10 | `HMA(5,199)` | 2MA |    37.0   1.99   -22.5 |    31.1   3.08   -13.0 |    31.0   1.96   -14.6 |   135.3   2.21   -22.5 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `HMA(6,13)` | 2MA |    -9.1  -0.13   -49.0 |     9.0   0.95   -27.3 |     6.6   0.59   -25.7 |     5.6   0.34   -49.0 | - |
| 119 | `HMA(8,9,75)` | 3MA |   -20.4  -0.94   -39.2 |     5.9   0.82   -22.5 |    23.6   1.67   -14.8 |     4.2   0.28   -39.2 | - |
| 120 | `HMA(4,30,67)` | 3MA |   -16.0  -0.76   -38.2 |    -2.2  -0.14   -21.9 |    15.1   1.32   -12.7 |    -5.4   0.02   -38.2 | - |


## 2h x DEMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.222**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 52/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [4,233] -> 49/60 configs positive across TRAIN & VAL & OOS
- band members: DEMA(10,128), DEMA(44,75), DEMA(8,91), DEMA(6,77), DEMA(37,89), DEMA(12,77), DEMA(5,86), DEMA(18,128), DEMA(22,65), DEMA(12,13), DEMA(31,53), DEMA(15,151), DEMA(22,151), DEMA(12,178), DEMA(26,75), DEMA(4,86), DEMA(37,38), DEMA(4,199), DEMA(31,124), DEMA(5,199), DEMA(15,65), DEMA(48,67,118), DEMA(8,210), DEMA(24,43,60) (+77 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `DEMA(10,128)` | 2MA |    43.5   2.63   -10.3 |    34.8   3.58    -7.7 |    40.9   2.52   -13.6 |   172.6   2.78   -13.6 | YES |
| 2 | `DEMA(44,75)` | 2MA |    36.3    2.2   -13.7 |    40.8   4.34    -6.2 |    38.1   2.56   -13.8 |   165.0    2.8   -13.8 | YES |
| 3 | `DEMA(8,91)` | 2MA |    46.5   2.51   -12.4 |    34.9   3.53    -7.5 |    32.6   2.09   -12.1 |   162.1   2.57   -12.4 | YES |
| 4 | `DEMA(6,77)` | 2MA |    36.5   1.95   -17.1 |    33.4    3.4    -8.4 |    41.7    2.5   -11.2 |   158.0   2.45   -17.1 | YES |
| 5 | `DEMA(37,89)` | 2MA |    35.7   2.16   -13.7 |    39.0    4.2    -6.3 |    34.6   2.41   -10.6 |   153.8    2.7   -13.7 | YES |
| 6 | `DEMA(12,77)` | 2MA |    32.3   1.89   -16.8 |    33.2   3.47    -7.6 |    42.6   2.59   -11.4 |   151.3   2.48   -16.8 | YES |
| 7 | `DEMA(5,86)` | 2MA |    38.4   2.06   -15.8 |    34.5   3.44    -8.0 |    34.1   2.12   -13.6 |   149.6   2.38   -15.8 | YES |
| 8 | `DEMA(18,128)` | 2MA |    35.4   2.25   -11.9 |    33.7   3.69    -7.0 |    37.8   2.44   -11.9 |   149.6   2.63   -11.9 | YES |
| 9 | `DEMA(22,65)` | 2MA |    27.6   1.67   -14.6 |    30.6   3.35    -8.8 |    47.4   2.94    -8.9 |   145.5   2.49   -14.6 | YES |
| 10 | `DEMA(12,13)` | 2MA |    33.9   1.76   -13.6 |    23.5   2.51   -15.1 |    47.9   2.78   -13.6 |   144.6   2.27   -15.1 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `DEMA(3,12,27)` | 3MA |   -14.5  -0.59   -34.7 |     6.2   0.87   -19.5 |    29.1   2.01   -15.1 |    17.2   0.56   -34.7 | - |
| 119 | `DEMA(5,6,34)` | 3MA |   -20.5  -0.99   -35.5 |    16.1    1.9   -15.1 |    23.5   1.64   -17.3 |    14.0   0.49   -35.5 | - |
| 120 | `DEMA(3,4,38)` | 3MA |   -24.5  -1.13   -40.2 |    17.1   1.94   -17.0 |    26.4   1.75   -15.7 |    11.7   0.44   -40.2 | - |


## 2h x TEMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.154**; TRAIN+VAL top-10 -> OOS top-10 overlap = **1/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 51/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [4,233] -> 48/60 configs positive across TRAIN & VAL & OOS
- band members: TEMA(26,172), TEMA(10,128), TEMA(37,203), TEMA(15,151), TEMA(31,124), TEMA(48,67,118), TEMA(22,151), TEMA(37,89), TEMA(12,178), TEMA(44,75), TEMA(12,33), TEMA(52,89), TEMA(5,199), TEMA(44,203), TEMA(4,199), TEMA(8,210), TEMA(15,28), TEMA(62,105), TEMA(8,39), TEMA(18,128), TEMA(6,210), TEMA(22,28), TEMA(18,23), TEMA(73,145) (+75 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `TEMA(26,172)` | 2MA |    28.7   1.97   -17.1 |    32.8   3.56    -8.4 |    46.3   2.98    -8.1 |   150.0   2.71   -17.1 | YES |
| 2 | `TEMA(10,128)` | 2MA |    29.4   1.78   -15.8 |    36.9   3.68    -9.1 |    40.3   2.51   -13.1 |   148.5   2.47   -15.8 | YES |
| 3 | `TEMA(37,203)` | 2MA |    36.7   2.47   -11.5 |    32.2   3.53    -8.8 |    32.2   2.44    -7.6 |   138.9   2.71   -11.5 | YES |
| 4 | `TEMA(15,151)` | 2MA |    22.3    1.5   -21.4 |    37.1   3.75    -8.9 |    40.6   2.53   -11.2 |   135.9   2.39   -21.4 | YES |
| 5 | `TEMA(31,124)` | 2MA |    20.6   1.41   -15.3 |    31.0   3.38    -9.9 |    49.1   3.15   -10.1 |   135.4   2.49   -15.3 | YES |
| 6 | `TEMA(48,67,118)` | 3MA |    26.9   2.07    -9.8 |    30.0   3.77    -7.9 |    41.0   3.27    -6.3 |   132.5    2.9    -9.8 | YES |
| 7 | `TEMA(22,151)` | 2MA |    20.8   1.47   -18.1 |    31.5   3.36    -9.0 |    45.3   2.91   -10.1 |   130.9   2.44   -18.1 | YES |
| 8 | `TEMA(37,89)` | 2MA |    16.2   1.16   -14.9 |    34.9   3.83    -9.3 |    43.9   2.85   -10.1 |   125.6   2.37   -14.9 | YES |
| 9 | `TEMA(12,178)` | 2MA |    31.7   1.93   -21.4 |    31.9   3.28    -9.3 |    29.6   1.98   -13.9 |   125.1   2.25   -21.4 | YES |
| 10 | `TEMA(44,75)` | 2MA |    17.2   1.21   -13.6 |    31.8   3.62    -9.8 |    44.7   2.89   -10.9 |   123.7   2.35   -13.6 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `TEMA(3,19,43)` | 3MA |   -16.9  -0.77   -36.8 |     6.6   0.95   -18.7 |    28.2   2.06   -13.6 |    13.6   0.49   -36.8 | - |
| 119 | `TEMA(6,13)` | 2MA |   -12.9  -0.28   -47.2 |    14.8   1.46   -23.9 |    12.8   0.92   -20.8 |    12.7   0.46   -47.2 | - |
| 120 | `TEMA(2,4,19)` | 3MA |    -8.1  -0.11   -45.1 |    16.9   1.68   -22.0 |     4.3   0.47   -23.4 |    12.1   0.45   -45.1 | - |


## 2h x KAMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.494**; TRAIN+VAL top-10 -> OOS top-10 overlap = **3/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 57/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,75], slow in [4,233] -> 59/60 configs positive across TRAIN & VAL & OOS
- band members: KAMA(6,13), KAMA(18,23), KAMA(10,55), KAMA(15,28), KAMA(5,86), KAMA(2,73), KAMA(6,33), KAMA(6,77), KAMA(15,65), KAMA(3,44), KAMA(3,102), KAMA(10,19), KAMA(4,86), KAMA(8,91), KAMA(5,37), KAMA(8,16), KAMA(2,3), KAMA(5,12), KAMA(4,37), KAMA(8,39), KAMA(26,32), KAMA(5,199), KAMA(18,55), KAMA(10,128) (+92 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `KAMA(6,13)` | 2MA |    54.6   2.23   -20.1 |    16.5    1.8   -16.8 |    46.6   2.79   -11.6 |   164.1   2.31   -20.1 | YES |
| 2 | `KAMA(18,23)` | 2MA |    34.3    2.1   -11.0 |    25.8   2.89   -14.6 |    45.6   2.88   -10.3 |   146.0   2.56   -14.6 | YES |
| 3 | `KAMA(10,55)` | 2MA |    34.0   1.99   -14.6 |    26.4    3.1    -8.9 |    41.5   2.68   -12.1 |   139.6   2.46   -14.6 | YES |
| 4 | `KAMA(15,28)` | 2MA |    44.1   2.58   -12.1 |    28.0   3.16   -12.7 |    29.3   2.02   -11.9 |   138.5   2.48   -12.7 | YES |
| 5 | `KAMA(5,86)` | 2MA |    38.7   2.23   -15.0 |    23.4   2.72   -11.4 |    37.0   2.46   -13.6 |   134.4   2.41   -15.0 | YES |
| 6 | `KAMA(2,73)` | 2MA |    33.2   1.87   -21.3 |    27.7   2.95    -9.1 |    36.8   2.28   -14.9 |   132.6   2.24   -21.3 | YES |
| 7 | `KAMA(6,33)` | 2MA |    25.6   1.52   -18.3 |    30.9   3.29    -7.2 |    40.8   2.55   -11.0 |   131.5   2.26   -18.3 | YES |
| 8 | `KAMA(6,77)` | 2MA |    34.1   1.98   -17.6 |    26.2   2.94   -10.7 |    35.9    2.4   -13.4 |   129.9   2.33   -17.6 | YES |
| 9 | `KAMA(15,65)` | 2MA |    37.1    2.3   -12.8 |    29.8   3.43    -8.6 |    29.0   2.17   -11.0 |   129.7    2.5   -12.8 | YES |
| 10 | `KAMA(3,44)` | 2MA |    23.8   1.42   -19.7 |    32.8   3.44    -8.5 |    38.0   2.36   -11.6 |   126.9   2.18   -19.7 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `KAMA(73,145)` | 2MA |    14.2   1.39    -8.2 |     9.1   1.62   -12.4 |     2.9   0.47   -10.5 |    28.3   1.14   -15.1 | YES |
| 119 | `KAMA(75,132,148)` | 3MA |     7.5    0.9    -7.7 |    10.2   1.98    -6.7 |     2.8   0.43   -12.8 |    21.8   0.96   -14.9 | YES |
| 120 | `KAMA(3,6)` | 2MA |   -16.1   -0.6   -46.4 |    18.0   1.83   -19.5 |    17.5   1.19   -19.0 |    16.4   0.53   -46.4 | - |


## 2h x VIDYA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.807**; TRAIN+VAL top-10 -> OOS top-10 overlap = **4/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 57/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,60], slow in [4,233] -> 57/60 configs positive across TRAIN & VAL & OOS
- band members: VIDYA(6,13), VIDYA(12,13), VIDYA(5,12), VIDYA(8,16), VIDYA(3,18), VIDYA(4,15), VIDYA(12,14,75), VIDYA(2,73), VIDYA(10,19), VIDYA(4,86), VIDYA(8,22,60), VIDYA(6,77), VIDYA(10,17,233), VIDYA(4,30,67), VIDYA(5,10,14), VIDYA(3,30,166), VIDYA(8,9,75), VIDYA(2,10), VIDYA(8,14,38), VIDYA(2,26), VIDYA(2,60,208), VIDYA(5,24,34), VIDYA(10,11,233), VIDYA(6,14,15) (+90 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `VIDYA(6,13)` | 2MA |    31.3   1.75   -14.3 |    24.5   2.77    -8.7 |    46.8   2.73   -12.5 |   139.9   2.31   -14.3 | YES |
| 2 | `VIDYA(12,13)` | 2MA |    27.4   1.68   -17.6 |    27.0   3.17    -8.3 |    47.5   2.88   -11.6 |   138.8   2.42   -17.6 | YES |
| 3 | `VIDYA(5,12)` | 2MA |    35.1   1.91   -14.8 |    25.8   2.88    -8.1 |    40.3    2.4   -12.3 |   138.4   2.28   -14.8 | YES |
| 4 | `VIDYA(8,16)` | 2MA |    29.4   1.75   -15.6 |    26.0   3.03    -8.6 |    45.2   2.75   -12.8 |   136.8   2.38   -15.6 | YES |
| 5 | `VIDYA(3,18)` | 2MA |    30.0   1.69   -14.5 |    26.3   2.96    -9.3 |    43.0   2.52   -11.7 |   134.8   2.24   -14.5 | YES |
| 6 | `VIDYA(4,15)` | 2MA |    34.0   1.86   -14.6 |    25.3   2.84    -8.6 |    39.4   2.35   -12.8 |   134.0   2.23   -14.6 | YES |
| 7 | `VIDYA(12,14,75)` | 3MA |    26.7   1.85   -13.5 |    28.4   3.83    -6.7 |    43.1    3.0   -11.3 |   132.7   2.67   -13.5 | YES |
| 8 | `VIDYA(2,73)` | 2MA |    28.4   2.05   -13.7 |    29.1   4.02    -6.1 |    36.7   2.72    -8.7 |   126.7   2.69   -13.7 | YES |
| 9 | `VIDYA(10,19)` | 2MA |    23.0   1.48   -17.9 |    28.3   3.45    -8.7 |    43.2    2.7   -13.1 |   126.0   2.33   -17.9 | YES |
| 10 | `VIDYA(4,86)` | 2MA |    31.7    2.4   -10.5 |    26.1   4.15    -6.4 |    35.0   2.79    -9.7 |   124.1   2.87   -10.5 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `VIDYA(44,75)` | 2MA |    -3.2  -0.43    -9.6 |    14.4   3.44    -5.6 |     8.7   1.51    -6.6 |    20.4    1.2   -11.3 | - |
| 119 | `VIDYA(37,89)` | 2MA |    -4.0  -0.59   -10.5 |    15.8   3.78    -4.4 |     8.3   1.39    -7.1 |    20.3    1.2   -12.2 | - |
| 120 | `VIDYA(186,208,233)` | 3MA |    -1.2  -0.18    -8.2 |     3.2   1.71    -2.4 |    12.5    3.4    -3.2 |    14.6   1.29    -8.5 | - |


# Timeframe: 1h
_Benchmark (equal-weight u10 buy-hold, no cost): FULL-2020 net = 144.6% (participation-tax reference)._

## 1h x EMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.672**; TRAIN+VAL top-10 -> OOS top-10 overlap = **2/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [13,239] -> 53/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [14,233] -> 55/60 configs positive across TRAIN & VAL & OOS
- band members: EMA(8,91), EMA(6,77), EMA(12,77), EMA(37,38), EMA(18,55), EMA(10,55), EMA(31,53), EMA(15,65), EMA(26,32), EMA(5,86), EMA(22,65), EMA(4,86), EMA(24,43,60), EMA(10,128), EMA(2,73), EMA(3,102), EMA(12,33), EMA(6,210), EMA(18,128), EMA(19,27,38), EMA(22,28), EMA(52,89), EMA(26,75), EMA(4,199) (+84 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `EMA(8,91)` | 2MA |    46.7   2.49   -12.7 |    17.0   2.17   -14.0 |    50.6   3.62    -9.9 |   158.4   2.75   -14.1 | YES |
| 2 | `EMA(6,77)` | 2MA |    45.6   2.45   -11.7 |    22.5   2.67   -11.2 |    41.2   3.04   -11.1 |   151.8   2.65   -12.7 | YES |
| 3 | `EMA(12,77)` | 2MA |    42.5   2.32   -12.3 |    15.2   1.99   -15.6 |    52.9   3.75    -9.7 |   151.0   2.68   -15.6 | YES |
| 4 | `EMA(37,38)` | 2MA |    36.2   2.02   -13.0 |    17.0   2.19   -16.0 |    56.8   4.05   -10.0 |   149.9   2.68   -16.0 | YES |
| 5 | `EMA(18,55)` | 2MA |    40.2    2.2   -14.4 |    21.7   2.64   -12.8 |    43.7   3.23   -11.1 |   145.2   2.59   -14.4 | YES |
| 6 | `EMA(10,55)` | 2MA |    40.3   2.21   -11.7 |    23.1   2.73   -10.6 |    41.9   3.08   -11.5 |   145.0   2.57   -11.9 | YES |
| 7 | `EMA(31,53)` | 2MA |    39.0   2.15   -12.3 |    12.2   1.68   -17.8 |    54.6   3.95   -11.0 |   141.1    2.6   -17.8 | YES |
| 8 | `EMA(15,65)` | 2MA |    41.3   2.25   -14.3 |    17.4   2.22   -16.2 |    45.4   3.33   -10.8 |   141.0   2.56   -16.2 | YES |
| 9 | `EMA(26,32)` | 2MA |    36.3   2.03   -13.7 |    19.0   2.35   -13.1 |    46.5   3.36   -10.7 |   137.5    2.5   -13.7 | YES |
| 10 | `EMA(5,86)` | 2MA |    43.3   2.35   -12.2 |    15.8   1.99   -15.9 |    42.7   3.13   -11.7 |   137.0   2.49   -18.0 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `EMA(4,5)` | 2MA |   -13.8  -0.46   -47.1 |    -1.0   0.08   -25.3 |    14.0   1.21   -17.4 |    -2.7   0.17   -47.1 | - |
| 119 | `EMA(3,6)` | 2MA |   -13.7  -0.48   -46.3 |    -3.9  -0.21   -28.7 |    13.6   1.19   -17.3 |    -5.8    0.1   -46.3 | - |
| 120 | `EMA(2,3,4)` | 3MA |   -17.9  -0.64   -42.9 |     2.5   0.45   -29.0 |     8.2   0.83   -21.6 |    -9.0   0.05   -42.9 | - |


## 1h x SMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.534**; TRAIN+VAL top-10 -> OOS top-10 overlap = **1/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [5,239] -> 53/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [14,233] -> 53/60 configs positive across TRAIN & VAL & OOS
- band members: SMA(10,19), SMA(3,102), SMA(10,128), SMA(4,86), SMA(12,13), SMA(8,91), SMA(12,178), SMA(15,151), SMA(6,77), SMA(37,89), SMA(12,77), SMA(12,22,118), SMA(5,199), SMA(22,151), SMA(18,128), SMA(5,86), SMA(10,55), SMA(4,199), SMA(2,169), SMA(31,124), SMA(6,210), SMA(6,14,15), SMA(15,65), SMA(8,210) (+82 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `SMA(10,19)` | 2MA |    61.6   2.71   -13.5 |    31.3   3.38   -11.0 |    32.5   2.51   -16.1 |   181.1   2.76   -16.1 | YES |
| 2 | `SMA(3,102)` | 2MA |    50.6   2.66   -14.2 |    18.9   2.31   -16.1 |    48.9    3.5   -13.4 |   166.5   2.82   -16.5 | YES |
| 3 | `SMA(10,128)` | 2MA |    38.3   2.14   -13.2 |    22.2    2.8   -11.0 |    51.6   3.77   -12.0 |   156.4   2.77   -13.2 | YES |
| 4 | `SMA(4,86)` | 2MA |    39.8   2.18   -17.2 |    23.8   2.79   -11.0 |    47.5   3.43   -10.7 |   155.2   2.69   -17.2 | YES |
| 5 | `SMA(12,13)` | 2MA |    53.3   1.92   -32.9 |    28.5   2.66   -15.6 |    29.0   2.08   -22.8 |   153.9    2.1   -32.9 | YES |
| 6 | `SMA(8,91)` | 2MA |    40.5    2.2   -14.7 |    18.7   2.36   -14.0 |    51.8   3.75   -11.0 |   153.1    2.7   -14.8 | YES |
| 7 | `SMA(12,178)` | 2MA |    38.8   2.26   -13.3 |    22.2   2.87   -10.6 |    45.1   3.56   -11.2 |   146.0   2.77   -13.3 | YES |
| 8 | `SMA(15,151)` | 2MA |    38.0   2.18   -12.8 |    20.6   2.66   -12.4 |    47.1   3.63   -11.3 |   144.7   2.72   -12.8 | YES |
| 9 | `SMA(6,77)` | 2MA |    36.8   2.01   -14.5 |    23.0   2.77   -10.6 |    44.7   3.26   -11.6 |   143.5   2.54   -14.5 | YES |
| 10 | `SMA(37,89)` | 2MA |    29.2   1.73   -16.0 |    22.2   2.75   -15.7 |    52.0   3.94   -10.3 |   140.0   2.63   -16.0 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `SMA(2,3)` | 2MA |     9.7   0.63   -43.7 |    -0.7   0.17   -33.4 |     4.6    0.6   -28.0 |    13.9   0.52   -43.7 | - |
| 119 | `SMA(2,3,4)` | 3MA |    -8.6  -0.11   -44.6 |     5.5   0.72   -24.7 |     3.6   0.54   -25.3 |    -0.1   0.26   -44.6 | - |
| 120 | `SMA(2,10)` | 2MA |   -19.9  -0.65   -49.8 |    -0.0   0.19   -26.9 |    11.0   1.02   -20.6 |   -11.1   0.01   -49.8 | - |


## 1h x WMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.632**; TRAIN+VAL top-10 -> OOS top-10 overlap = **1/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 51/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [14,233] -> 58/60 configs positive across TRAIN & VAL & OOS
- band members: WMA(15,28), WMA(18,23), WMA(15,151), WMA(12,178), WMA(10,128), WMA(18,128), WMA(12,33), WMA(26,172), WMA(22,28), WMA(4,199), WMA(31,124), WMA(22,151), WMA(62,105), WMA(3,102), WMA(5,86), WMA(8,210), WMA(44,203), WMA(37,203), WMA(5,199), WMA(52,89), WMA(8,91), WMA(37,89), WMA(44,75), WMA(4,86) (+85 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `WMA(15,28)` | 2MA |    43.1   2.18   -12.5 |    33.5   3.69    -9.3 |    41.7    3.1   -14.4 |   170.7   2.77   -14.4 | YES |
| 2 | `WMA(18,23)` | 2MA |    42.1   2.12   -12.2 |    34.3   3.75   -11.0 |    40.6   3.03   -16.9 |   168.3   2.72   -16.9 | YES |
| 3 | `WMA(15,151)` | 2MA |    43.6   2.35   -12.5 |    16.2   2.12   -15.8 |    54.4   3.89   -10.8 |   157.5   2.77   -15.8 | YES |
| 4 | `WMA(12,178)` | 2MA |    46.7    2.5   -12.4 |    20.2   2.53   -12.5 |    44.5   3.33   -13.4 |   154.7   2.74   -13.4 | YES |
| 5 | `WMA(10,128)` | 2MA |    45.4   2.41   -12.9 |    17.5   2.21   -16.4 |    48.4   3.53   -11.3 |   153.5    2.7   -17.1 | YES |
| 6 | `WMA(18,128)` | 2MA |    44.3   2.36   -12.3 |    15.1   1.98   -17.2 |    52.5   3.78   -11.3 |   153.2   2.71   -17.2 | YES |
| 7 | `WMA(12,33)` | 2MA |    30.9   1.68   -14.6 |    30.0   3.34   -10.1 |    46.4    3.4   -12.5 |   149.2   2.55   -14.6 | YES |
| 8 | `WMA(26,172)` | 2MA |    42.6   2.33   -12.5 |    14.1   1.94   -15.7 |    50.2   3.75   -11.9 |   144.5   2.68   -15.7 | YES |
| 9 | `WMA(22,28)` | 2MA |    37.8    2.0   -12.5 |    27.5   3.09   -12.9 |    38.3   2.89   -12.1 |   142.9   2.49   -15.5 | YES |
| 10 | `WMA(4,199)` | 2MA |    45.0   2.44   -13.7 |    16.7   2.14   -14.1 |    43.4    3.2   -13.2 |   142.6   2.59   -15.6 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `WMA(5,12)` | 2MA |   -24.2   -1.0   -51.3 |     8.0    1.0   -23.0 |    15.2   1.31   -20.2 |    -5.7   0.11   -51.3 | - |
| 119 | `WMA(2,3,4)` | 3MA |   -18.0  -0.53   -48.2 |     3.1    0.5   -25.7 |     1.1   0.38   -28.5 |   -14.6  -0.04   -48.2 | - |
| 120 | `WMA(2,10)` | 2MA |   -22.2  -0.79   -49.6 |    -2.5  -0.05   -29.1 |    12.6    1.1   -18.1 |   -14.7  -0.07   -49.6 | - |


## 1h x HMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.723**; TRAIN+VAL top-10 -> OOS top-10 overlap = **2/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 42/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [4,233] -> 36/60 configs positive across TRAIN & VAL & OOS
- band members: HMA(37,38), HMA(31,53), HMA(26,32), HMA(22,65), HMA(26,75), HMA(4,5,75), HMA(22,151), HMA(18,55), HMA(19,27,38), HMA(26,172), HMA(37,89), HMA(18,128), HMA(44,75), HMA(4,5), HMA(15,22,48), HMA(31,124), HMA(8,22,60), HMA(24,43,60), HMA(15,65), HMA(12,77), HMA(4,199), HMA(10,128), HMA(6,210), HMA(8,210) (+54 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `HMA(37,38)` | 2MA |    81.9   3.42   -11.4 |    27.8   3.05   -14.2 |    40.2   3.07   -14.9 |   226.0    3.2   -15.4 | YES |
| 2 | `HMA(31,53)` | 2MA |    69.6    3.1    -9.6 |    25.7   2.89   -15.0 |    44.3   3.52   -12.3 |   207.7   3.16   -15.0 | YES |
| 3 | `HMA(26,32)` | 2MA |    50.6   1.98   -39.6 |    37.7   3.84   -13.2 |    37.5   2.94   -12.1 |   185.1   2.58   -39.6 | YES |
| 4 | `HMA(22,65)` | 2MA |    57.4   2.61   -10.8 |    20.9   2.43   -15.7 |    46.4   3.59   -12.5 |   178.5   2.84   -16.1 | YES |
| 5 | `HMA(26,75)` | 2MA |    52.9   2.56   -12.6 |    24.2    2.8   -13.8 |    43.9   3.48   -11.6 |   173.1   2.87   -13.8 | YES |
| 6 | `HMA(4,5,75)` | 3MA |    45.4    2.8    -8.6 |    19.8   2.94   -11.3 |    45.1   4.35    -9.2 |   152.8   3.28   -11.3 | YES |
| 7 | `HMA(22,151)` | 2MA |    40.2   2.16   -15.3 |    21.9   2.54   -14.9 |    47.7    3.6   -12.2 |   152.5   2.67   -15.3 | YES |
| 8 | `HMA(18,55)` | 2MA |    40.0   1.87   -25.0 |    25.9   2.89   -14.4 |    40.6   3.18   -11.5 |   147.9   2.44   -25.0 | YES |
| 9 | `HMA(19,27,38)` | 3MA |    55.7   2.56   -24.6 |    22.8    2.7   -12.4 |    27.0   2.56   -11.1 |   142.9   2.58   -24.6 | YES |
| 10 | `HMA(26,172)` | 2MA |    33.3   1.89   -14.4 |    21.9   2.57   -15.3 |    49.5   3.66   -11.2 |   142.8   2.58   -15.3 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `HMA(2,12,14)` | 3MA |   -30.5   -1.3   -45.7 |     1.5   0.35   -30.7 |     5.7   0.67   -27.4 |   -25.4  -0.37   -46.8 | - |
| 119 | `HMA(10,19)` | 2MA |   -25.5   -1.0   -47.7 |     4.9   0.67   -29.1 |    -5.7  -0.12   -29.3 |   -26.3  -0.38   -47.7 | - |
| 120 | `HMA(6,14,15)` | 3MA |   -23.5  -1.06   -39.1 |   -11.0   -1.2   -28.5 |   -11.0  -0.72   -30.5 |   -39.5  -0.98   -47.7 | - |


## 1h x DEMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.555**; TRAIN+VAL top-10 -> OOS top-10 overlap = **2/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 49/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [4,233] -> 46/60 configs positive across TRAIN & VAL & OOS
- band members: DEMA(12,178), DEMA(15,28), DEMA(26,172), DEMA(37,203), DEMA(22,151), DEMA(44,203), DEMA(18,23), DEMA(52,203), DEMA(8,210), DEMA(15,151), DEMA(12,33), DEMA(73,145), DEMA(102,103), DEMA(31,124), DEMA(22,28), DEMA(75,132,148), DEMA(62,105), DEMA(24,106,118), DEMA(86,122), DEMA(62,239), DEMA(6,210), DEMA(18,128), DEMA(5,199), DEMA(37,38) (+71 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `DEMA(12,178)` | 2MA |    56.7   3.01   -10.9 |    28.0   3.16    -9.2 |    37.4    2.9   -10.6 |   175.7   2.96   -10.9 | YES |
| 2 | `DEMA(15,28)` | 2MA |    43.0   2.16   -16.8 |    25.3   2.86   -15.0 |    53.0   3.95   -11.6 |   174.0   2.84   -16.8 | YES |
| 3 | `DEMA(26,172)` | 2MA |    55.7   3.05    -9.6 |    24.0   2.94   -11.9 |    41.4   3.29    -9.9 |   173.0   3.06   -11.9 | YES |
| 4 | `DEMA(37,203)` | 2MA |    50.5   2.94   -14.0 |    22.8    2.9   -11.9 |    43.3   3.58    -8.6 |   164.9    3.1   -14.0 | YES |
| 5 | `DEMA(22,151)` | 2MA |    42.4   2.38   -15.4 |    22.4   2.72   -11.6 |    51.8   3.82   -11.2 |   164.7   2.89   -15.4 | YES |
| 6 | `DEMA(44,203)` | 2MA |    51.6   3.02   -11.7 |    23.7   3.02   -10.5 |    41.0   3.43    -8.8 |   164.3   3.11   -11.7 | YES |
| 7 | `DEMA(18,23)` | 2MA |    36.3   1.87   -18.8 |    23.0   2.64   -15.3 |    57.4    4.2   -11.1 |   163.9   2.72   -18.8 | YES |
| 8 | `DEMA(52,203)` | 2MA |    46.3   2.86   -10.2 |    26.2   3.34    -8.9 |    41.2   3.56    -9.3 |   160.8   3.16   -10.2 | YES |
| 9 | `DEMA(8,210)` | 2MA |    46.3   2.66   -12.3 |    27.3   3.08    -9.8 |    38.5   2.97   -11.4 |   158.1   2.81   -12.3 | YES |
| 10 | `DEMA(15,151)` | 2MA |    39.3   2.15   -16.9 |    27.5   3.19    -9.3 |    43.9   3.35   -10.6 |   155.4   2.73   -16.9 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `DEMA(3,18)` | 2MA |   -21.8  -0.75   -49.4 |     4.9   0.68   -24.8 |     5.7   0.67   -26.9 |   -13.3  -0.04   -49.4 | - |
| 119 | `DEMA(3,6)` | 2MA |     2.9   0.37   -40.1 |    -2.2   0.01   -31.1 |   -17.5  -0.86   -36.2 |   -17.0  -0.09   -44.2 | - |
| 120 | `DEMA(2,10)` | 2MA |    -8.9   -0.1   -48.0 |    -2.2   0.01   -32.1 |   -12.3  -0.53   -32.4 |   -21.9   -0.2   -48.0 | - |


## 1h x TEMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.733**; TRAIN+VAL top-10 -> OOS top-10 overlap = **5/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 46/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [4,233] -> 39/60 configs positive across TRAIN & VAL & OOS
- band members: TEMA(37,38), TEMA(22,65), TEMA(31,53), TEMA(15,65), TEMA(22,28), TEMA(12,77), TEMA(18,55), TEMA(62,239), TEMA(26,75), TEMA(10,128), TEMA(15,151), TEMA(5,199), TEMA(26,32), TEMA(19,27,38), TEMA(37,203), TEMA(86,122), TEMA(44,203), TEMA(6,210), TEMA(15,84,148), TEMA(73,145), TEMA(186,208,233), TEMA(18,128), TEMA(75,132,148), TEMA(102,103) (+61 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `TEMA(37,38)` | 2MA |    50.9   2.62   -12.6 |    17.3   2.09   -15.8 |    44.4    3.5   -11.6 |   155.6   2.75   -15.8 | YES |
| 2 | `TEMA(22,65)` | 2MA |    46.1   2.43   -18.1 |    17.9   2.09   -17.8 |    48.1   3.73   -10.7 |   155.2   2.73   -18.1 | YES |
| 3 | `TEMA(31,53)` | 2MA |    45.7   2.38   -19.9 |    17.4   2.08   -16.3 |    45.4   3.58   -10.9 |   148.8   2.66   -19.9 | YES |
| 4 | `TEMA(15,65)` | 2MA |    40.9   2.17   -21.1 |    12.1   1.51   -19.7 |    51.7   3.83   -11.6 |   139.5   2.51   -21.1 | YES |
| 5 | `TEMA(22,28)` | 2MA |    31.9   1.71   -19.4 |    27.1   2.97   -16.2 |    42.2   3.37   -10.1 |   138.4   2.47   -19.4 | YES |
| 6 | `TEMA(12,77)` | 2MA |    41.3   2.18   -20.1 |    12.3    1.5   -20.2 |    48.1   3.58   -12.5 |   134.9   2.44   -20.2 | YES |
| 7 | `TEMA(18,55)` | 2MA |    36.3   1.98   -17.7 |    13.3   1.65   -17.6 |    49.9    3.8   -11.1 |   131.4   2.45   -17.7 | YES |
| 8 | `TEMA(62,239)` | 2MA |    30.8   1.96   -10.2 |    18.8   2.44   -12.5 |    47.0    4.0    -8.3 |   128.5   2.69   -12.5 | YES |
| 9 | `TEMA(26,75)` | 2MA |    37.8   2.07   -16.5 |    15.9   1.91   -14.2 |    42.3   3.35   -12.4 |   127.3   2.41   -16.5 | YES |
| 10 | `TEMA(10,128)` | 2MA |    31.9   1.75   -20.8 |    28.0   3.05   -13.4 |    33.0   2.64   -15.2 |   124.6    2.3   -20.8 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `TEMA(5,12)` | 2MA |   -21.4  -0.74   -47.7 |    -2.2   0.01   -31.5 |    -2.6   0.13   -33.7 |   -25.1  -0.31   -47.7 | - |
| 119 | `TEMA(6,13)` | 2MA |   -19.5  -0.64   -47.7 |     0.6   0.26   -29.7 |    -7.6  -0.25   -34.4 |   -25.1  -0.33   -47.7 | - |
| 120 | `TEMA(4,15)` | 2MA |   -13.0  -0.32   -44.7 |    -7.1  -0.49   -33.7 |   -12.3  -0.56   -36.9 |   -29.2  -0.42   -46.2 | - |


## 1h x KAMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.59**; TRAIN+VAL top-10 -> OOS top-10 overlap = **2/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [5,239] -> 55/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [4,233] -> 59/60 configs positive across TRAIN & VAL & OOS
- band members: KAMA(18,128), KAMA(4,199), KAMA(10,128), KAMA(2,169), KAMA(3,102), KAMA(15,151), KAMA(15,65), KAMA(6,210), KAMA(18,55), KAMA(5,199), KAMA(12,33), KAMA(15,28), KAMA(8,210), KAMA(12,178), KAMA(22,151), KAMA(18,23), KAMA(12,77), KAMA(22,65), KAMA(8,91), KAMA(37,89), KAMA(6,77), KAMA(4,86), KAMA(31,124), KAMA(2,73) (+90 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `KAMA(18,128)` | 2MA |    47.2   2.65   -12.4 |    18.5   2.41   -13.2 |    43.2   3.44   -12.1 |   149.8   2.82   -13.2 | YES |
| 2 | `KAMA(4,199)` | 2MA |    46.7   2.54   -13.4 |    21.1    2.6   -11.5 |    40.5   3.26   -14.1 |   149.6   2.75   -14.1 | YES |
| 3 | `KAMA(10,128)` | 2MA |    43.1   2.39   -12.4 |    17.9   2.31   -14.7 |    47.6   3.64   -12.0 |   149.0   2.74   -14.7 | YES |
| 4 | `KAMA(2,169)` | 2MA |    41.8   2.23   -14.4 |    23.9   2.81    -9.3 |    40.1   3.15   -13.5 |   146.1   2.62   -14.4 | YES |
| 5 | `KAMA(3,102)` | 2MA |    38.8   2.14   -15.3 |    29.5   3.39   -10.1 |    36.5   2.86   -13.8 |   145.4   2.61   -15.3 | YES |
| 6 | `KAMA(15,151)` | 2MA |    49.3    2.7   -12.7 |    14.5   1.96   -14.6 |    42.6    3.4   -11.6 |   143.8   2.73   -14.6 | YES |
| 7 | `KAMA(15,65)` | 2MA |    48.3   2.61   -11.1 |    15.0   1.99   -15.0 |    42.9   3.37   -10.0 |   143.7   2.68   -15.0 | YES |
| 8 | `KAMA(6,210)` | 2MA |    45.9   2.55   -12.9 |    19.6   2.44   -12.4 |    37.6   3.08   -13.9 |   140.1   2.66   -13.9 | YES |
| 9 | `KAMA(18,55)` | 2MA |    38.7   2.17   -12.4 |    16.3   2.22   -13.1 |    47.9   3.76    -9.0 |   138.6   2.65   -13.1 | YES |
| 10 | `KAMA(5,199)` | 2MA |    44.6   2.45   -14.9 |    17.3   2.19   -13.8 |    40.2   3.25   -14.2 |   137.8   2.62   -14.9 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `KAMA(2,8,22)` | 3MA |     5.2   0.49   -19.2 |     7.6   1.16   -12.9 |     8.6   0.99   -18.5 |    23.1   0.78   -19.2 | YES |
| 119 | `KAMA(102,103)` | 2MA |   -11.8  -0.45   -42.5 |    15.5   2.17   -14.1 |    18.8   1.86   -17.4 |    21.0    0.7   -42.5 | - |
| 120 | `KAMA(2,10)` | 2MA |   -12.5   -0.4   -42.4 |     8.7   1.15   -18.9 |     9.8   0.99   -20.7 |     4.5   0.32   -42.4 | - |


## 1h x VIDYA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.483**; TRAIN+VAL top-10 -> OOS top-10 overlap = **2/10**. Ordering does NOT transfer -- trust the band, not the #1.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 60/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [4,233] -> 60/60 configs positive across TRAIN & VAL & OOS
- band members: VIDYA(10,19), VIDYA(5,37), VIDYA(4,37), VIDYA(6,33), VIDYA(12,13), VIDYA(18,23), VIDYA(3,18), VIDYA(6,14,15), VIDYA(6,13), VIDYA(8,16), VIDYA(22,28), VIDYA(3,44), VIDYA(8,39), VIDYA(4,15), VIDYA(2,26), VIDYA(15,28), VIDYA(3,102), VIDYA(5,12), VIDYA(12,33), VIDYA(2,169), VIDYA(10,55), VIDYA(6,77), VIDYA(2,73), VIDYA(6,22,24) (+96 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `VIDYA(10,19)` | 2MA |    42.0   2.29   -12.7 |    12.3   1.69   -17.7 |    49.7   3.65   -11.3 |   138.7   2.57   -18.4 | YES |
| 2 | `VIDYA(5,37)` | 2MA |    38.1   2.14   -13.1 |    13.0   1.81   -17.9 |    49.1   3.61   -10.3 |   132.7   2.52   -17.9 | YES |
| 3 | `VIDYA(4,37)` | 2MA |    38.7   2.16   -13.1 |    13.9   1.88   -16.4 |    46.9   3.47   -11.0 |   132.2    2.5   -17.1 | YES |
| 4 | `VIDYA(6,33)` | 2MA |    37.7   2.12   -13.3 |    12.8   1.78   -17.9 |    48.9    3.6   -10.5 |   131.4    2.5   -18.6 | YES |
| 5 | `VIDYA(12,13)` | 2MA |    38.1    2.1   -13.8 |    11.9   1.63   -17.9 |    49.5   3.63   -10.9 |   130.9   2.47   -18.1 | YES |
| 6 | `VIDYA(18,23)` | 2MA |    35.1   2.07   -13.5 |    15.7    2.2   -16.2 |    46.7   3.59   -11.7 |   129.2   2.56   -16.2 | YES |
| 7 | `VIDYA(3,18)` | 2MA |    30.0   1.74   -16.6 |    23.4   2.77    -9.9 |    42.6   3.15   -12.6 |   128.9   2.39   -16.6 | YES |
| 8 | `VIDYA(6,14,15)` | 3MA |    29.0   1.75   -15.1 |    23.2   2.87   -10.8 |    42.7   3.28   -11.1 |   126.8   2.46   -15.1 | YES |
| 9 | `VIDYA(6,13)` | 2MA |    29.3   1.72   -15.1 |    25.4   2.99   -10.5 |    39.7   2.97   -12.5 |   126.6   2.37   -15.1 | YES |
| 10 | `VIDYA(8,16)` | 2MA |    33.6   1.89   -14.4 |    17.2   2.19   -15.8 |    44.7   3.35   -11.6 |   126.5   2.39   -15.8 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `VIDYA(62,239)` | 2MA |     8.9   1.32    -9.8 |     9.1   2.43    -8.8 |    11.1   2.73    -4.1 |    32.0   1.99    -9.8 | YES |
| 119 | `VIDYA(52,203)` | 2MA |     5.8   0.89    -8.5 |    13.4    3.3    -5.6 |     9.4   2.12    -6.3 |    31.2   1.86    -8.5 | YES |
| 120 | `VIDYA(73,145)` | 2MA |     3.4   0.55    -9.3 |    11.9   2.95    -6.5 |     9.5   2.23    -4.4 |    26.7   1.64   -11.3 | YES |


# Timeframe: 30m
_Benchmark (equal-weight u10 buy-hold, no cost): FULL-2020 net = 150.0% (participation-tax reference)._

## 30m x EMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.721**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [19,239] -> 48/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [15,233] -> 46/60 configs positive across TRAIN & VAL & OOS
- band members: EMA(12,178), EMA(15,151), EMA(73,145), EMA(8,210), EMA(37,89), EMA(62,105), EMA(26,75), EMA(5,199), EMA(44,75), EMA(10,128), EMA(6,210), EMA(86,122), EMA(18,128), EMA(52,89), EMA(102,103), EMA(75,132,148), EMA(22,151), EMA(31,124), EMA(44,203), EMA(22,65), EMA(37,203), EMA(102,237), EMA(26,172), EMA(2,169) (+70 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `EMA(12,178)` | 2MA |    39.9   2.16   -13.7 |    18.8   2.44   -13.3 |    32.3   2.74   -11.6 |   119.8   2.38   -14.8 | YES |
| 2 | `EMA(15,151)` | 2MA |    40.4   2.18   -14.4 |    20.0   2.54   -12.9 |    29.8   2.54   -11.9 |   118.6   2.34   -14.4 | YES |
| 3 | `EMA(73,145)` | 2MA |    35.5   1.99   -14.2 |    15.2   2.15   -15.6 |    40.1   3.44    -9.1 |   118.6   2.44   -15.6 | YES |
| 4 | `EMA(8,210)` | 2MA |    43.5   2.32   -12.0 |    13.4   1.85   -15.3 |    31.8    2.7   -12.5 |   114.6   2.31   -17.4 | YES |
| 5 | `EMA(37,89)` | 2MA |    34.7   1.93   -15.3 |    18.7    2.4   -12.8 |    34.2   2.88   -11.2 |   114.6    2.3   -15.3 | YES |
| 6 | `EMA(62,105)` | 2MA |    33.9   1.89   -13.9 |    11.7   1.67   -17.5 |    42.4   3.48   -10.1 |   113.0   2.31   -18.3 | YES |
| 7 | `EMA(26,75)` | 2MA |    35.5   1.99   -12.6 |    22.3   2.81    -9.6 |    28.1   2.41   -13.0 |   112.2   2.27   -13.0 | YES |
| 8 | `EMA(5,199)` | 2MA |    47.8   2.51   -12.1 |    12.7   1.75   -16.7 |    26.9   2.35   -13.5 |   111.5   2.27   -18.2 | YES |
| 9 | `EMA(44,75)` | 2MA |    34.7   1.94   -15.6 |    18.3   2.35   -13.5 |    32.4   2.76   -11.1 |   111.0   2.26   -15.6 | YES |
| 10 | `EMA(10,128)` | 2MA |    39.4   2.17   -14.0 |    22.7   2.81    -9.6 |    23.2   2.06   -14.0 |   110.7   2.24   -14.0 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `EMA(3,6)` | 2MA |    -2.0   0.15   -38.3 |   -17.5  -1.69   -38.7 |   -11.9  -0.63   -31.2 |   -28.7  -0.48   -45.2 | - |
| 119 | `EMA(2,3,4)` | 3MA |   -11.5  -0.27   -36.5 |   -21.8  -2.13   -39.5 |    -2.6    0.1   -29.9 |   -32.6  -0.56   -48.5 | - |
| 120 | `EMA(2,3)` | 2MA |   -24.1  -0.79   -42.3 |   -26.5   -2.6   -42.5 |     4.8   0.61   -29.9 |   -41.5  -0.75   -58.8 | - |


## 30m x SMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.592**; TRAIN+VAL top-10 -> OOS top-10 overlap = **2/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [28,239] -> 46/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [24,233] -> 44/60 configs positive across TRAIN & VAL & OOS
- band members: SMA(37,38), SMA(19,27,38), SMA(5,199), SMA(26,32), SMA(4,199), SMA(22,151), SMA(26,172), SMA(18,55), SMA(8,210), SMA(19,43,233), SMA(31,53), SMA(18,128), SMA(6,210), SMA(12,178), SMA(22,28), SMA(15,151), SMA(37,203), SMA(52,203), SMA(62,239), SMA(31,124), SMA(30,43,186), SMA(44,203), SMA(2,169), SMA(10,128) (+66 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `SMA(37,38)` | 2MA |    46.0   2.14   -11.4 |    28.1   2.86   -15.8 |    48.8   3.25   -16.7 |   178.4   2.62   -16.7 | YES |
| 2 | `SMA(19,27,38)` | 3MA |    52.0   2.76   -10.4 |    27.6   3.56    -7.5 |    26.5    2.5   -14.7 |   145.4   2.82   -14.7 | YES |
| 3 | `SMA(5,199)` | 2MA |    47.7   2.48   -13.2 |    20.4   2.53   -14.2 |    30.2   2.62   -11.4 |   131.6   2.52   -14.5 | YES |
| 4 | `SMA(26,32)` | 2MA |    43.0   2.12   -21.3 |    28.7   3.13   -12.0 |    24.6   2.04   -16.7 |   129.3    2.3   -21.3 | YES |
| 5 | `SMA(4,199)` | 2MA |    49.3   2.57   -13.4 |    18.3    2.3   -15.6 |    29.4   2.56   -11.6 |   128.5   2.49   -16.1 | YES |
| 6 | `SMA(22,151)` | 2MA |    36.1   1.96   -16.5 |    22.3    2.8   -10.6 |    37.0   3.05   -11.1 |   128.0   2.45   -16.5 | YES |
| 7 | `SMA(26,172)` | 2MA |    37.9   2.06   -14.8 |    14.8   1.99   -15.1 |    41.5   3.41    -9.6 |   124.0   2.44   -15.2 | YES |
| 8 | `SMA(18,55)` | 2MA |    30.5   1.71   -12.7 |    31.1   3.55    -9.8 |    29.3    2.5   -10.8 |   121.2   2.33   -12.7 | YES |
| 9 | `SMA(8,210)` | 2MA |    43.6    2.3   -13.5 |    13.7   1.84   -16.7 |    31.9   2.76   -10.7 |   115.3   2.32   -17.6 | YES |
| 10 | `SMA(19,43,233)` | 3MA |    28.9   2.05   -13.6 |    28.2   3.93    -7.6 |    30.2    3.0   -11.0 |   115.2   2.75   -13.6 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `SMA(4,8,17)` | 3MA |   -32.4   -1.9   -47.7 |     4.0   0.64   -22.0 |     3.9   0.56   -16.9 |   -27.0  -0.63   -47.7 | - |
| 119 | `SMA(2,12,14)` | 3MA |   -27.8  -1.55   -44.7 |    -0.4   0.11   -20.8 |    -0.2   0.21   -16.7 |   -28.2  -0.66   -44.7 | - |
| 120 | `SMA(2,4,19)` | 3MA |   -31.6  -1.83   -40.6 |    -3.3  -0.23   -23.0 |     1.5   0.37   -18.4 |   -32.8   -0.8   -46.9 | - |


## 30m x WMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.768**; TRAIN+VAL top-10 -> OOS top-10 overlap = **1/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [28,239] -> 41/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [38,233] -> 38/60 configs positive across TRAIN & VAL & OOS
- band members: WMA(37,38), WMA(31,53), WMA(37,203), WMA(62,239), WMA(44,203), WMA(8,210), WMA(18,55), WMA(6,210), WMA(26,172), WMA(12,178), WMA(52,203), WMA(22,65), WMA(26,75), WMA(102,237), WMA(5,199), WMA(22,151), WMA(15,151), WMA(30,43,186), WMA(73,145), WMA(4,199), WMA(30,132,186), WMA(18,128), WMA(37,89), WMA(38,67,233) (+55 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `WMA(37,38)` | 2MA |    50.5   2.47   -13.1 |    36.8   4.07   -11.7 |    36.3   2.87   -13.0 |   180.6   2.91   -13.1 | YES |
| 2 | `WMA(31,53)` | 2MA |    44.2   2.28   -11.3 |    36.8   4.11    -7.9 |    30.8   2.57   -11.8 |   158.1   2.74   -11.8 | YES |
| 3 | `WMA(37,203)` | 2MA |    38.7   2.11   -14.4 |    20.1   2.56   -12.1 |    38.2   3.15   -10.2 |   130.4   2.51   -14.4 | YES |
| 4 | `WMA(62,239)` | 2MA |    34.7   1.93   -13.5 |    14.7   1.98   -17.3 |    43.5   3.69    -9.0 |   121.7   2.44   -17.3 | YES |
| 5 | `WMA(44,203)` | 2MA |    36.3   2.01   -16.0 |    16.8   2.19   -14.4 |    39.2   3.22   -10.4 |   121.6    2.4   -16.0 | YES |
| 6 | `WMA(8,210)` | 2MA |    50.7   2.64   -11.9 |    17.8   2.26   -13.7 |    22.6   2.03   -13.3 |   117.6   2.34   -13.7 | YES |
| 7 | `WMA(18,55)` | 2MA |    35.9   1.94   -12.0 |    32.1    3.6   -11.8 |    20.5   1.82   -15.0 |   116.4   2.24   -15.0 | YES |
| 8 | `WMA(6,210)` | 2MA |    45.2    2.4   -13.6 |    23.5   2.82   -11.9 |    20.0   1.84   -14.3 |   115.2   2.29   -14.3 | YES |
| 9 | `WMA(26,172)` | 2MA |    35.4   1.97   -14.2 |    19.5    2.5   -11.7 |    31.9   2.71   -11.4 |   113.4   2.29   -14.2 | YES |
| 10 | `WMA(12,178)` | 2MA |    41.1   2.22   -14.0 |    22.2   2.76   -11.1 |    23.7   2.11   -13.3 |   113.3   2.28   -14.0 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `WMA(2,3,4)` | 3MA |   -17.5  -0.47   -42.0 |   -19.0  -1.74   -38.1 |     4.0   0.56   -24.4 |   -30.6  -0.44   -54.1 | - |
| 119 | `WMA(2,4,19)` | 3MA |   -26.1  -1.25   -40.7 |   -12.1  -1.25   -31.8 |    -8.5  -0.45   -28.0 |   -40.6   -1.0   -49.2 | - |
| 120 | `WMA(2,3)` | 2MA |   -27.7  -0.86   -49.2 |   -15.0  -1.17   -39.9 |    -4.9    0.0   -28.2 |   -41.6  -0.67   -61.7 | - |


## 30m x HMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.79**; TRAIN+VAL top-10 -> OOS top-10 overlap = **6/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 26/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [15,233] -> 22/60 configs positive across TRAIN & VAL & OOS
- band members: HMA(52,89), HMA(62,105), HMA(52,203), HMA(44,203), HMA(73,145), HMA(60,67,233), HMA(86,122), HMA(37,203), HMA(44,75), HMA(62,239), HMA(2,3), HMA(48,67,118), HMA(37,89), HMA(26,172), HMA(31,124), HMA(4,5), HMA(75,132,148), HMA(12,178), HMA(18,128), HMA(22,151), HMA(15,151), HMA(10,11,233), HMA(24,106,118), HMA(38,67,233) (+24 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `HMA(52,89)` | 2MA |    86.7   3.55   -14.2 |    38.0   4.08   -10.4 |    42.3   3.63   -10.4 |   266.8   3.68   -14.2 | YES |
| 2 | `HMA(62,105)` | 2MA |    80.1   3.51   -10.1 |    23.7   2.83   -14.9 |    40.9   3.47   -11.9 |   214.1   3.33   -14.9 | YES |
| 3 | `HMA(52,203)` | 2MA |    40.6   2.24   -12.0 |    19.7   2.37   -13.2 |    56.2   4.36   -10.6 |   162.8    2.9   -13.2 | YES |
| 4 | `HMA(44,203)` | 2MA |    45.6   2.45   -11.8 |    16.8   2.05   -14.5 |    48.6   3.91   -10.6 |   152.6   2.78   -14.5 | YES |
| 5 | `HMA(73,145)` | 2MA |    39.2   2.12   -13.6 |    20.6   2.48   -12.5 |    49.3   3.99   -10.3 |   150.8   2.74   -13.6 | YES |
| 6 | `HMA(60,67,233)` | 3MA |    58.2   3.65    -8.4 |    19.9   3.19    -8.2 |    30.3    3.6    -7.9 |   147.1   3.52    -8.4 | YES |
| 7 | `HMA(86,122)` | 2MA |    42.6   2.23   -14.8 |    20.2   2.42   -12.9 |    43.8   3.62   -10.3 |   146.5   2.67   -14.8 | YES |
| 8 | `HMA(37,203)` | 2MA |    48.2   2.55   -11.3 |    16.3    2.0   -14.7 |    41.0   3.39   -12.3 |   143.1   2.66   -14.7 | YES |
| 9 | `HMA(44,75)` | 2MA |    40.2   1.82   -32.1 |    27.8   3.11   -12.6 |    28.9   2.54   -13.0 |   130.9   2.27   -32.1 | YES |
| 10 | `HMA(62,239)` | 2MA |    43.8   2.35   -12.5 |    14.1   1.83   -14.7 |    39.3   3.28   -11.5 |   128.6    2.5   -14.7 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `HMA(3,8,43)` | 3MA |   -27.4  -1.18   -46.6 |    -8.3  -0.74   -30.1 |   -13.6  -0.92   -26.7 |   -42.5  -1.01   -51.5 | - |
| 119 | `HMA(37,38)` | 2MA |   -38.6  -2.32   -51.1 |     5.0   0.73   -21.9 |   -13.9  -1.07   -26.1 |   -44.5  -1.27   -54.6 | - |
| 120 | `HMA(3,18)` | 2MA |   -28.7  -0.91   -48.1 |   -20.4  -1.85   -40.3 |    -2.3   0.16   -26.6 |   -44.6  -0.79   -61.1 | - |


## 30m x DEMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.854**; TRAIN+VAL top-10 -> OOS top-10 overlap = **2/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [28,239] -> 39/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [3,186], slow in [34,233] -> 27/60 configs positive across TRAIN & VAL & OOS
- band members: DEMA(186,208,233), DEMA(31,53), DEMA(102,237), DEMA(26,75), DEMA(22,151), DEMA(37,38), DEMA(62,239), DEMA(30,132,186), DEMA(26,32), DEMA(31,124), DEMA(22,65), DEMA(44,75), DEMA(37,203), DEMA(18,128), DEMA(38,67,233), DEMA(18,55), DEMA(52,89), DEMA(15,65), DEMA(37,89), DEMA(52,203), DEMA(12,77), DEMA(26,172), DEMA(19,132,233), DEMA(15,151) (+42 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `DEMA(186,208,233)` | 3MA |    38.5   2.48   -12.7 |    28.4   3.67    -9.0 |    37.1   3.69    -6.3 |   143.9   3.11   -12.7 | YES |
| 2 | `DEMA(31,53)` | 2MA |    49.1   2.49   -14.7 |    16.2    2.0   -16.7 |    40.6   3.36   -11.5 |   143.6   2.62   -16.7 | YES |
| 3 | `DEMA(102,237)` | 2MA |    32.3   1.96   -12.0 |    22.7    3.0   -10.6 |    46.6   4.18    -8.0 |   138.2   2.83   -12.0 | YES |
| 4 | `DEMA(26,75)` | 2MA |    45.1   2.42   -14.2 |    21.6   2.54   -14.8 |    34.6   2.93   -11.7 |   137.5   2.59   -14.8 | YES |
| 5 | `DEMA(22,151)` | 2MA |    34.6   1.98   -16.9 |    25.4   2.95   -11.9 |    36.9   3.09   -13.4 |   131.1   2.53   -16.9 | YES |
| 6 | `DEMA(37,38)` | 2MA |    51.8   2.56   -12.9 |    13.3   1.68   -18.1 |    30.2   2.62   -11.7 |   123.9   2.37   -18.1 | YES |
| 7 | `DEMA(62,239)` | 2MA |    25.5   1.63   -12.7 |    26.9   3.28    -8.4 |    37.2   3.32    -9.5 |   118.5   2.51   -12.7 | YES |
| 8 | `DEMA(30,132,186)` | 3MA |    34.3   2.18   -11.3 |    21.6   3.09    -7.2 |    32.9    3.4    -7.4 |   116.9   2.73   -11.3 | YES |
| 9 | `DEMA(26,32)` | 2MA |    41.4   2.09   -18.0 |    20.8   2.45   -11.3 |    25.3   2.31   -11.6 |   114.0   2.23   -18.0 | YES |
| 10 | `DEMA(31,124)` | 2MA |    33.6   1.95   -16.2 |    14.3   1.82   -13.8 |    39.2   3.26   -12.0 |   112.6   2.31   -16.2 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `DEMA(3,6)` | 2MA |   -31.3  -1.09   -51.1 |   -14.5  -1.22   -37.5 |    -6.6  -0.14   -25.7 |   -45.2  -0.84   -61.6 | - |
| 119 | `DEMA(4,8,17)` | 3MA |   -34.1  -1.77   -47.9 |    -8.4  -0.79   -30.2 |   -17.8  -1.33   -29.1 |   -50.4  -1.42   -57.4 | - |
| 120 | `DEMA(2,10)` | 2MA |   -36.3  -1.37   -52.0 |   -20.9  -1.96   -41.0 |   -15.1  -0.77   -36.5 |   -57.2  -1.31   -68.0 | - |


## 30m x TEMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.73**; TRAIN+VAL top-10 -> OOS top-10 overlap = **3/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [38,239] -> 27/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [3,186], slow in [60,233] -> 15/60 configs positive across TRAIN & VAL & OOS
- band members: TEMA(62,105), TEMA(22,151), TEMA(31,124), TEMA(37,89), TEMA(26,172), TEMA(52,89), TEMA(44,75), TEMA(186,208,233), TEMA(15,151), TEMA(37,203), TEMA(26,75), TEMA(8,210), TEMA(102,103), TEMA(86,122), TEMA(44,203), TEMA(18,128), TEMA(52,203), TEMA(48,67,118), TEMA(12,178), TEMA(73,145), TEMA(24,43,60), TEMA(60,67,233), TEMA(62,239), TEMA(6,210) (+18 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `TEMA(62,105)` | 2MA |    52.2   2.76   -12.5 |    16.9   2.09   -14.1 |    34.0   3.03   -11.4 |   138.4   2.67   -14.1 | YES |
| 2 | `TEMA(22,151)` | 2MA |    53.0   2.73   -13.3 |    13.9   1.76   -16.9 |    31.6   2.73   -12.7 |   129.4   2.49   -16.9 | YES |
| 3 | `TEMA(31,124)` | 2MA |    51.3    2.7   -12.6 |     9.0   1.22   -19.0 |    35.2   3.02   -13.0 |   123.0   2.44   -19.0 | YES |
| 4 | `TEMA(37,89)` | 2MA |    50.0   2.57   -12.6 |    14.8   1.85   -19.2 |    28.6   2.61   -12.5 |   121.4    2.4   -19.2 | YES |
| 5 | `TEMA(26,172)` | 2MA |    42.0   2.27   -16.5 |    16.6   2.06   -11.1 |    31.5   2.68   -12.0 |   117.7   2.33   -16.5 | YES |
| 6 | `TEMA(52,89)` | 2MA |    48.2   2.57   -13.4 |     5.0   0.76   -20.5 |    37.4   3.24   -11.2 |   113.8   2.34   -20.5 | YES |
| 7 | `TEMA(44,75)` | 2MA |    47.8   2.51   -10.5 |    11.8   1.54   -19.4 |    28.5   2.63   -12.1 |   112.4   2.31   -21.2 | YES |
| 8 | `TEMA(186,208,233)` | 3MA |    21.7   1.52   -10.9 |    21.4   3.01    -9.2 |    41.3   4.01    -9.1 |   108.8   2.58   -10.9 | YES |
| 9 | `TEMA(15,151)` | 2MA |    40.3   2.18   -15.2 |    10.1   1.32   -16.5 |    32.6   2.73   -12.0 |   104.9   2.14   -16.5 | YES |
| 10 | `TEMA(37,203)` | 2MA |    36.1   2.07   -17.3 |    15.9   2.04   -13.1 |    28.5   2.56   -13.4 |   102.8    2.2   -17.3 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `TEMA(2,4,19)` | 3MA |   -26.6  -0.98   -46.1 |   -16.6  -1.56   -37.7 |   -11.0  -0.53   -28.0 |   -45.6  -0.96   -58.5 | - |
| 119 | `TEMA(3,18)` | 2MA |   -25.7  -0.86   -50.7 |   -18.7  -1.74   -37.4 |   -12.8  -0.63   -34.0 |   -47.3  -0.97   -62.3 | - |
| 120 | `TEMA(2,26)` | 2MA |   -16.4  -0.46   -45.6 |   -19.7  -1.83   -39.2 |   -21.7  -1.33   -39.2 |   -47.4   -1.0   -60.1 | - |


## 30m x KAMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.849**; TRAIN+VAL top-10 -> OOS top-10 overlap = **3/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [28,239] -> 45/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,75], slow in [34,233] -> 34/60 configs positive across TRAIN & VAL & OOS
- band members: KAMA(31,53), KAMA(44,203), KAMA(37,203), KAMA(52,203), KAMA(26,172), KAMA(26,75), KAMA(37,89), KAMA(22,65), KAMA(31,124), KAMA(18,128), KAMA(30,43,186), KAMA(44,75), KAMA(22,151), KAMA(6,210), KAMA(8,210), KAMA(15,151), KAMA(73,145), KAMA(24,43,60), KAMA(62,105), KAMA(62,239), KAMA(5,199), KAMA(19,132,233), KAMA(38,67,233), KAMA(48,67,118) (+55 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `KAMA(31,53)` | 2MA |    54.6   2.16   -22.2 |    12.5   1.72   -16.8 |    38.7   3.47   -11.0 |   141.1   2.37   -22.2 | YES |
| 2 | `KAMA(44,203)` | 2MA |    32.0   1.83   -14.2 |    21.8   2.88   -11.6 |    49.2   4.13    -7.9 |   139.9   2.72   -14.2 | YES |
| 3 | `KAMA(37,203)` | 2MA |    38.9   2.16   -13.1 |    18.3   2.44   -12.4 |    43.3   3.69   -10.3 |   135.6   2.66   -13.1 | YES |
| 4 | `KAMA(52,203)` | 2MA |    32.1   1.85   -13.3 |    15.4   2.12   -14.4 |    47.6   3.99    -7.7 |   124.9   2.53   -14.4 | YES |
| 5 | `KAMA(26,172)` | 2MA |    31.1    1.8   -14.2 |    15.3   2.12   -16.5 |    46.4   3.84   -10.0 |   121.2   2.47   -16.5 | YES |
| 6 | `KAMA(26,75)` | 2MA |    32.9   1.65   -17.0 |    15.9   2.18   -14.0 |    43.0   3.72   -11.7 |   120.1    2.3   -17.0 | YES |
| 7 | `KAMA(37,89)` | 2MA |    34.0   1.94   -12.0 |    11.0   1.62   -14.8 |    46.4   3.94   -10.6 |   117.9   2.46   -16.1 | YES |
| 8 | `KAMA(22,65)` | 2MA |    30.3   1.55   -20.3 |    17.4    2.3   -11.8 |    42.5   3.68   -12.4 |   117.8   2.26   -20.3 | YES |
| 9 | `KAMA(31,124)` | 2MA |    34.2   1.93   -13.4 |    16.5   2.25   -12.4 |    36.6   3.22    -9.8 |   113.6   2.37   -13.4 | YES |
| 10 | `KAMA(18,128)` | 2MA |    23.1   1.39   -18.0 |    21.4   2.71   -10.8 |    39.4   3.29   -10.7 |   108.3   2.23   -18.0 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `KAMA(2,4,19)` | 3MA |   -16.7   -0.9   -31.7 |    -6.4  -0.73   -21.7 |    -4.0  -0.16   -18.9 |   -25.2  -0.63   -35.6 | - |
| 119 | `KAMA(2,8,22)` | 3MA |   -25.8   -1.7   -36.3 |    -3.5  -0.35   -23.0 |     0.6   0.27   -18.6 |   -28.0  -0.78   -38.8 | - |
| 120 | `KAMA(2,10)` | 2MA |   -24.2  -1.13   -44.2 |    -7.9  -0.75   -26.8 |    -2.7   0.05   -24.1 |   -32.0  -0.68   -45.7 | - |


## 30m x VIDYA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.536**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [10,239] -> 57/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [14,233] -> 57/60 configs positive across TRAIN & VAL & OOS
- band members: VIDYA(5,37), VIDYA(6,33), VIDYA(4,37), VIDYA(12,33), VIDYA(19,27,38), VIDYA(3,44), VIDYA(10,19), VIDYA(37,38), VIDYA(12,77), VIDYA(15,28), VIDYA(8,39), VIDYA(15,65), VIDYA(26,32), VIDYA(18,55), VIDYA(8,91), VIDYA(31,53), VIDYA(18,23), VIDYA(10,55), VIDYA(22,65), VIDYA(22,28), VIDYA(26,75), VIDYA(15,22,48), VIDYA(6,77), VIDYA(5,86) (+90 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `VIDYA(5,37)` | 2MA |    39.5   2.14   -12.8 |    21.2   2.68   -10.6 |    27.5   2.36   -14.1 |   115.5    2.3   -14.1 | YES |
| 2 | `VIDYA(6,33)` | 2MA |    35.7   1.96   -14.3 |    21.1   2.68    -9.8 |    30.3   2.55   -13.2 |   114.1   2.27   -14.3 | YES |
| 3 | `VIDYA(4,37)` | 2MA |    38.2   2.09   -13.6 |    20.1   2.54   -10.7 |    28.8   2.44   -14.1 |   113.7   2.27   -14.1 | YES |
| 4 | `VIDYA(12,33)` | 2MA |    38.7   2.08   -14.3 |    18.2   2.37   -13.2 |    30.3   2.62   -11.9 |   113.7   2.28   -14.3 | YES |
| 5 | `VIDYA(19,27,38)` | 3MA |    40.3    2.2   -14.8 |    17.8   2.43   -13.8 |    28.0   2.63    -9.6 |   111.5   2.36   -14.8 | YES |
| 6 | `VIDYA(3,44)` | 2MA |    42.5   2.29   -13.3 |    19.2   2.45   -11.5 |    24.2   2.13   -14.0 |   111.0   2.24   -14.0 | YES |
| 7 | `VIDYA(10,19)` | 2MA |    29.0   1.67   -16.5 |    26.5   3.18    -8.5 |    28.7   2.45   -13.4 |   110.1   2.22   -16.5 | YES |
| 8 | `VIDYA(37,38)` | 2MA |    32.3   1.87   -14.3 |    20.4   2.76   -11.7 |    31.9   2.95    -9.0 |   109.9   2.36   -14.3 | YES |
| 9 | `VIDYA(12,77)` | 2MA |    34.5   1.95   -14.1 |    13.4   1.96   -16.6 |    37.2   3.22    -9.9 |   109.3   2.32   -17.3 | YES |
| 10 | `VIDYA(15,28)` | 2MA |    38.2   2.05   -14.0 |    17.1   2.24   -14.3 |    29.1   2.54   -11.4 |   108.8   2.22   -14.3 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `VIDYA(2,5,118)` | 3MA |    -1.4   0.02   -17.0 |     5.8   1.03   -11.4 |     4.1   0.61   -18.8 |     8.6   0.43   -18.8 | - |
| 119 | `VIDYA(2,3,4)` | 3MA |   -14.2  -0.68   -37.0 |     5.3   0.81   -18.9 |    12.8   1.27   -17.7 |     1.9   0.24   -37.0 | - |
| 120 | `VIDYA(2,3)` | 2MA |   -19.6  -0.95   -42.4 |     0.8   0.26   -21.5 |     8.8   0.94   -19.2 |   -11.8  -0.11   -42.4 | - |


# Timeframe: 15m
_Benchmark (equal-weight u10 buy-hold, no cost): FULL-2020 net = 157.2% (participation-tax reference)._

## 15m x EMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.885**; TRAIN+VAL top-10 -> OOS top-10 overlap = **2/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [28,239] -> 37/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [38,233] -> 28/60 configs positive across TRAIN & VAL & OOS
- band members: EMA(26,172), EMA(37,203), EMA(62,105), EMA(44,203), EMA(73,145), EMA(62,239), EMA(186,208,233), EMA(31,124), EMA(86,122), EMA(102,103), EMA(102,237), EMA(30,132,186), EMA(52,203), EMA(12,178), EMA(22,151), EMA(75,132,148), EMA(52,89), EMA(48,67,118), EMA(15,151), EMA(8,210), EMA(60,67,233), EMA(6,210), EMA(24,106,118), EMA(19,132,233) (+41 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `EMA(26,172)` | 2MA |    37.2   2.16   -14.2 |    31.3   3.76    -8.2 |    17.4   1.78   -14.4 |   111.4   2.39   -14.4 | YES |
| 2 | `EMA(37,203)` | 2MA |    35.9    2.1   -13.6 |    21.7   2.83   -10.8 |    25.6   2.45   -13.4 |   107.8   2.35   -13.6 | YES |
| 3 | `EMA(62,105)` | 2MA |    32.9   1.96   -15.9 |    27.8   3.41    -9.2 |    21.9   2.13   -13.8 |   106.9   2.32   -15.9 | YES |
| 4 | `EMA(44,203)` | 2MA |    37.0   2.14   -14.5 |    20.0   2.65   -12.9 |    25.4   2.43   -12.9 |   106.1   2.32   -14.5 | YES |
| 5 | `EMA(73,145)` | 2MA |    32.7   1.93   -15.8 |    20.7    2.7   -11.4 |    28.7    2.7   -12.0 |   106.0   2.32   -15.8 | YES |
| 6 | `EMA(62,239)` | 2MA |    32.9   1.96   -14.8 |    19.8   2.64   -12.5 |    26.7   2.59   -11.4 |   101.8   2.28   -14.8 | YES |
| 7 | `EMA(186,208,233)` | 3MA |    31.5   1.99   -14.0 |    15.1   2.27   -15.8 |    32.4   3.16    -9.2 |   100.2   2.38   -16.2 | YES |
| 8 | `EMA(31,124)` | 2MA |    29.6   1.79   -17.0 |    29.3   3.53    -9.6 |    19.3   1.91   -14.1 |    99.9    2.2   -17.0 | YES |
| 9 | `EMA(86,122)` | 2MA |    30.8   1.84   -15.9 |    18.7   2.48   -11.9 |    28.7    2.7   -12.9 |    99.9   2.22   -15.9 | YES |
| 10 | `EMA(102,103)` | 2MA |    29.9   1.79   -15.9 |    19.2   2.53   -12.0 |    29.1   2.72   -12.8 |    99.9   2.22   -15.9 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `EMA(4,5)` | 2MA |   -33.9  -1.64   -40.7 |   -20.7  -2.25   -34.6 |   -24.8   -1.9   -36.7 |   -60.6  -1.83   -65.8 | - |
| 119 | `EMA(2,3)` | 2MA |   -41.1  -1.93   -50.3 |   -24.8  -2.49   -40.5 |   -14.1  -0.76   -34.1 |   -61.9  -1.69   -71.6 | - |
| 120 | `EMA(2,3,4)` | 3MA |   -44.7  -2.34   -51.4 |   -23.7  -2.51   -36.8 |   -18.3  -1.21   -32.1 |   -65.5  -2.03   -72.1 | - |


## 15m x SMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.764**; TRAIN+VAL top-10 -> OOS top-10 overlap = **4/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [3,102], slow in [65,239] -> 29/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [60,233] -> 23/60 configs positive across TRAIN & VAL & OOS
- band members: SMA(37,89), SMA(44,75), SMA(52,89), SMA(26,75), SMA(44,203), SMA(52,203), SMA(37,203), SMA(62,105), SMA(62,239), SMA(31,124), SMA(22,65), SMA(8,210), SMA(8,91), SMA(38,67,233), SMA(18,128), SMA(15,65), SMA(12,77), SMA(60,67,233), SMA(15,151), SMA(22,151), SMA(30,132,186), SMA(26,172), SMA(102,237), SMA(6,210) (+28 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `SMA(37,89)` | 2MA |    39.7   2.18   -12.6 |    33.2   3.91    -9.4 |    31.0   2.76   -11.5 |   143.7   2.71   -12.6 | YES |
| 2 | `SMA(44,75)` | 2MA |    48.5   2.53   -11.1 |    40.8   4.59    -9.0 |    13.8   1.41   -15.4 |   137.9   2.61   -15.4 | YES |
| 3 | `SMA(52,89)` | 2MA |    37.4   2.14   -10.8 |    25.2   3.05   -14.8 |    35.6   3.08   -11.6 |   133.3   2.62   -14.8 | YES |
| 4 | `SMA(26,75)` | 2MA |    36.7   2.06   -15.0 |    35.0   3.94    -9.2 |    25.3   2.33   -12.0 |   131.2   2.55   -15.0 | YES |
| 5 | `SMA(44,203)` | 2MA |    36.1   2.08   -17.2 |    16.7   2.21   -15.1 |    43.6   3.76   -11.4 |   128.0    2.6   -17.2 | YES |
| 6 | `SMA(52,203)` | 2MA |    33.4   1.96   -17.3 |    16.5    2.2   -16.2 |    40.2   3.53   -11.8 |   118.0   2.47   -17.3 | YES |
| 7 | `SMA(37,203)` | 2MA |    35.0   2.01   -16.8 |    16.8   2.23   -14.1 |    37.6   3.34   -12.8 |   117.0   2.44   -16.8 | YES |
| 8 | `SMA(62,105)` | 2MA |    23.6   1.47   -12.7 |    24.1   2.88   -14.3 |    37.3   3.26   -10.8 |   110.6   2.32   -14.3 | YES |
| 9 | `SMA(62,239)` | 2MA |    23.3   1.48   -15.8 |    16.6   2.23   -13.8 |    38.0   3.42   -13.5 |    98.3   2.21   -15.8 | YES |
| 10 | `SMA(31,124)` | 2MA |    27.5   1.68   -16.7 |    22.9   2.79   -12.8 |    20.8   2.04   -11.2 |    89.3   2.03   -16.7 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `SMA(2,3)` | 2MA |   -41.9  -1.64   -52.6 |   -19.3  -1.64   -40.7 |   -10.8  -0.44   -27.0 |   -58.2  -1.29   -70.9 | - |
| 119 | `SMA(2,3,4)` | 3MA |   -47.3  -2.25   -53.4 |   -11.7  -1.05   -34.2 |   -13.9  -0.78   -26.4 |   -60.0  -1.58   -68.1 | - |
| 120 | `SMA(3,6)` | 2MA |   -42.9  -2.13   -50.7 |   -19.4  -1.87   -36.2 |   -16.2  -0.97   -32.8 |   -61.4  -1.72   -69.6 | - |


## 15m x WMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.863**; TRAIN+VAL top-10 -> OOS top-10 overlap = **5/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [4,102], slow in [28,239] -> 29/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [5,186], slow in [60,233] -> 17/60 configs positive across TRAIN & VAL & OOS
- band members: WMA(62,105), WMA(52,89), WMA(62,239), WMA(37,89), WMA(44,75), WMA(86,122), WMA(102,103), WMA(73,145), WMA(102,237), WMA(52,203), WMA(186,208,233), WMA(60,67,233), WMA(44,203), WMA(31,124), WMA(37,203), WMA(48,67,118), WMA(26,172), WMA(75,132,148), WMA(18,128), WMA(22,151), WMA(26,75), WMA(19,132,233), WMA(30,132,186), WMA(15,151) (+22 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `WMA(62,105)` | 2MA |    43.0   2.37   -11.1 |    33.5   3.96   -10.1 |    25.2   2.34   -12.2 |   139.1   2.69   -12.2 | YES |
| 2 | `WMA(52,89)` | 2MA |    40.1   2.23   -12.1 |    38.7   4.36    -7.3 |    20.0   1.92   -13.8 |   133.3   2.59   -13.8 | YES |
| 3 | `WMA(62,239)` | 2MA |    32.4   1.91   -18.0 |    20.1   2.59   -14.1 |    33.0   2.97   -13.0 |   111.5   2.37   -18.0 | YES |
| 4 | `WMA(37,89)` | 2MA |    32.7   1.89   -16.2 |    31.6   3.64    -8.1 |    20.9   1.97   -15.7 |   111.1   2.29   -16.2 | YES |
| 5 | `WMA(44,75)` | 2MA |    35.0   1.98   -17.4 |    29.2   3.38   -11.3 |    15.7   1.57   -16.9 |   101.8   2.16   -17.4 | YES |
| 6 | `WMA(86,122)` | 2MA |    28.4   1.72   -15.9 |    25.6   3.06   -13.1 |    24.2   2.32   -10.3 |   100.3    2.2   -15.9 | YES |
| 7 | `WMA(102,103)` | 2MA |    26.0    1.6   -16.1 |    28.5   3.36   -11.5 |    21.3   2.09   -11.3 |    96.4   2.14   -16.1 | YES |
| 8 | `WMA(73,145)` | 2MA |    29.5   1.76   -14.5 |    23.2   2.83   -13.0 |    22.0   2.12   -11.2 |    94.6    2.1   -14.5 | YES |
| 9 | `WMA(102,237)` | 2MA |    23.9   1.49   -19.5 |    13.0   1.78   -17.6 |    36.2   3.26   -12.7 |    90.8   2.07   -19.5 | YES |
| 10 | `WMA(52,203)` | 2MA |    25.8   1.59   -17.7 |    21.7   2.71   -11.9 |    23.6   2.26   -13.8 |    89.2   2.03   -17.7 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `WMA(2,10)` | 2MA |   -42.4  -2.15   -50.1 |   -21.7  -2.24   -37.7 |   -21.3  -1.47   -33.9 |   -64.5  -1.95   -71.0 | - |
| 119 | `WMA(3,6)` | 2MA |   -44.8  -2.24   -52.3 |   -24.7  -2.57   -38.2 |   -17.3  -1.06   -31.2 |   -65.6  -1.95   -72.2 | - |
| 120 | `WMA(2,3,4)` | 3MA |   -46.0  -2.13   -54.0 |   -29.3  -3.09   -41.1 |   -11.6  -0.58   -30.8 |   -66.2  -1.88   -74.0 | - |


## 15m x HMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.689**; TRAIN+VAL top-10 -> OOS top-10 overlap = **7/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [3,239] -> 12/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [5,186], slow in [148,233] -> 8/60 configs positive across TRAIN & VAL & OOS
- band members: HMA(75,132,148), HMA(102,237), HMA(4,5), HMA(10,11,233), HMA(62,239), HMA(186,208,233), HMA(52,203), HMA(2,3), HMA(44,203), HMA(30,132,186), HMA(37,203), HMA(73,145), HMA(19,132,233), HMA(86,122), HMA(5,118,208), HMA(26,172), HMA(12,106,148), HMA(22,151), HMA(38,67,233), HMA(31,53)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `HMA(75,132,148)` | 3MA |    62.3   3.51   -12.1 |    29.7   3.82    -7.4 |    13.5   1.66   -10.7 |   138.8   3.05   -12.1 | YES |
| 2 | `HMA(102,237)` | 2MA |    55.2   2.87   -11.8 |    23.1   2.82   -12.7 |    23.8   2.43    -9.1 |   136.5   2.73   -12.9 | YES |
| 3 | `HMA(4,5)` | 2MA |    32.0   1.37   -43.9 |    29.1    2.5   -22.0 |    37.1    2.4   -23.2 |   133.6   1.91   -43.9 | YES |
| 4 | `HMA(10,11,233)` | 3MA |    36.7   2.55   -11.9 |    30.0    5.0    -7.9 |    31.0   3.98    -7.5 |   132.9   3.45   -11.9 | YES |
| 5 | `HMA(62,239)` | 2MA |    36.4   2.04   -20.9 |    23.9   2.85   -10.3 |    25.4   2.51   -11.4 |   111.9   2.35   -20.9 | YES |
| 6 | `HMA(186,208,233)` | 3MA |    27.3   1.79   -13.9 |    16.5   2.22   -11.7 |    40.3   3.82   -11.0 |   108.0   2.48   -13.9 | YES |
| 7 | `HMA(52,203)` | 2MA |    24.6   1.45   -27.3 |    32.0   3.61    -9.6 |    24.4   2.39   -11.0 |   104.6    2.2   -27.3 | YES |
| 8 | `HMA(2,3)` | 2MA |    20.6   0.96   -48.6 |    11.9   1.19   -26.7 |    35.1   2.17   -26.3 |    82.2   1.35   -48.6 | YES |
| 9 | `HMA(44,203)` | 2MA |    21.8   1.32   -27.4 |    22.2   2.64   -10.8 |    18.1   1.83   -13.7 |    75.8   1.76   -27.4 | YES |
| 10 | `HMA(30,132,186)` | 3MA |    37.7   2.66   -13.7 |     6.9   1.23    -9.0 |     6.9   1.02    -9.7 |    57.3   1.85   -13.7 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `HMA(2,26)` | 2MA |   -48.8   -2.2   -55.6 |   -24.2  -2.48   -40.8 |   -17.1  -1.04   -28.1 |   -67.8  -1.92   -73.6 | - |
| 119 | `HMA(5,6,34)` | 3MA |   -43.4   -2.2   -50.3 |   -27.4   -3.3   -39.1 |   -24.4  -1.94   -33.9 |   -69.0  -2.33   -73.9 | - |
| 120 | `HMA(2,8,22)` | 3MA |   -42.2  -1.97   -51.4 |   -28.3   -3.2   -40.9 |   -30.6  -2.47   -36.1 |   -71.2  -2.34   -75.8 | - |


## 15m x DEMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.883**; TRAIN+VAL top-10 -> OOS top-10 overlap = **4/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [6,102], slow in [38,239] -> 24/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [10,186], slow in [60,233] -> 11/60 configs positive across TRAIN & VAL & OOS
- band members: DEMA(62,105), DEMA(102,103), DEMA(86,122), DEMA(73,145), DEMA(52,203), DEMA(52,89), DEMA(102,237), DEMA(75,132,148), DEMA(44,203), DEMA(62,239), DEMA(26,172), DEMA(37,203), DEMA(31,124), DEMA(44,75), DEMA(48,67,118), DEMA(22,151), DEMA(186,208,233), DEMA(37,89), DEMA(30,132,186), DEMA(8,210), DEMA(24,106,118), DEMA(18,128), DEMA(60,67,233), DEMA(15,151) (+11 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `DEMA(62,105)` | 2MA |    50.8   2.74   -13.1 |    20.0   2.44   -13.2 |    28.8   2.82   -10.9 |   133.2   2.68   -13.2 | YES |
| 2 | `DEMA(102,103)` | 2MA |    35.9   2.15   -14.2 |    18.9   2.33   -14.9 |    41.7    3.8   -10.2 |   128.9   2.67   -14.9 | YES |
| 3 | `DEMA(86,122)` | 2MA |    33.8   2.04   -15.4 |    18.3   2.27   -15.1 |    41.1   3.76   -10.2 |   123.3   2.59   -15.4 | YES |
| 4 | `DEMA(73,145)` | 2MA |    33.5   2.03   -16.1 |    18.0   2.24   -15.2 |    41.6   3.78    -9.9 |   123.0   2.58   -16.1 | YES |
| 5 | `DEMA(52,203)` | 2MA |    30.2   1.86   -19.5 |    19.1   2.37   -14.2 |    36.2   3.31   -12.5 |   111.1    2.4   -19.5 | YES |
| 6 | `DEMA(52,89)` | 2MA |    42.9   2.34   -16.3 |    17.9   2.22   -11.7 |    23.0   2.35    -9.6 |   107.2   2.31   -16.3 | YES |
| 7 | `DEMA(102,237)` | 2MA |    27.1   1.77   -17.8 |    16.5   2.13   -14.9 |    38.8    3.7   -13.4 |   105.5   2.41   -17.8 | YES |
| 8 | `DEMA(75,132,148)` | 3MA |    24.5   1.69   -16.4 |    11.0   1.58   -15.5 |    47.8   4.57    -9.1 |   104.3    2.5   -16.4 | YES |
| 9 | `DEMA(44,203)` | 2MA |    30.5   1.86   -19.9 |    18.7   2.33   -14.5 |    29.4   2.79   -12.5 |   100.5   2.24   -19.9 | YES |
| 10 | `DEMA(62,239)` | 2MA |    29.7   1.86   -18.7 |    14.3   1.87   -14.3 |    35.0   3.37   -13.2 |   100.2   2.29   -18.7 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `DEMA(3,18)` | 2MA |   -44.4  -2.26   -50.0 |   -24.0  -2.61   -40.1 |   -29.3  -2.32   -37.5 |   -70.1  -2.33   -74.8 | - |
| 119 | `DEMA(5,12)` | 2MA |   -43.0  -1.99   -50.4 |   -24.0  -2.63   -37.1 |   -32.2  -2.63   -41.1 |   -70.6  -2.28   -75.7 | - |
| 120 | `DEMA(2,4,19)` | 3MA |   -42.7  -2.19   -51.6 |   -31.8   -3.8   -42.1 |   -27.8  -2.33   -34.4 |   -71.8  -2.56   -77.0 | - |


## 15m x TEMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.639**; TRAIN+VAL top-10 -> OOS top-10 overlap = **7/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [26,102], slow in [89,239] -> 13/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [3,186], slow in [118,233] -> 7/60 configs positive across TRAIN & VAL & OOS
- band members: TEMA(102,237), TEMA(86,122), TEMA(62,239), TEMA(102,103), TEMA(73,145), TEMA(186,208,233), TEMA(52,203), TEMA(75,132,148), TEMA(62,105), TEMA(44,203), TEMA(60,67,233), TEMA(48,67,118), TEMA(37,203), TEMA(24,106,118), TEMA(52,89), TEMA(37,89), TEMA(26,172), TEMA(30,132,186), TEMA(31,124), TEMA(3,75,132)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `TEMA(102,237)` | 2MA |    49.5    2.8   -14.3 |    18.0   2.26   -12.9 |    36.1    3.5   -10.0 |   140.0   2.86   -14.3 | YES |
| 2 | `TEMA(86,122)` | 2MA |    44.2   2.46   -15.2 |    20.1   2.49   -14.2 |    23.6    2.4   -10.9 |   114.3   2.44   -15.2 | YES |
| 3 | `TEMA(62,239)` | 2MA |    52.7   2.88   -11.9 |    11.1    1.5   -16.9 |    24.1   2.51   -12.3 |   110.5   2.44   -16.9 | YES |
| 4 | `TEMA(102,103)` | 2MA |    42.4   2.37   -15.8 |    18.5    2.3   -15.1 |    21.4   2.22   -11.1 |   104.9    2.3   -15.8 | YES |
| 5 | `TEMA(73,145)` | 2MA |    42.6   2.41   -14.3 |    17.4   2.19   -15.1 |    19.3   2.05   -11.8 |    99.7   2.25   -15.1 | YES |
| 6 | `TEMA(186,208,233)` | 3MA |    20.9   1.45   -16.5 |     7.7   1.17   -16.5 |    52.3   4.94    -8.9 |    98.5   2.38   -16.5 | YES |
| 7 | `TEMA(52,203)` | 2MA |    33.7   1.96   -17.4 |    13.7   1.79   -15.5 |    23.9   2.46   -11.3 |    88.3   2.06   -17.4 | YES |
| 8 | `TEMA(75,132,148)` | 3MA |    33.8   2.27   -12.7 |    12.9   1.86   -11.4 |    16.2   1.95   -10.7 |    75.6   2.07   -12.7 | YES |
| 9 | `TEMA(62,105)` | 2MA |    20.6   1.32   -33.7 |    22.3   2.65   -11.1 |     9.5   1.14   -12.9 |    61.6   1.58   -33.7 | YES |
| 10 | `TEMA(44,203)` | 2MA |    27.5   1.66   -20.2 |     7.5   1.06   -16.6 |    14.1   1.59   -10.6 |    56.5   1.49   -20.2 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `TEMA(5,37)` | 2MA |   -36.9  -1.85   -51.7 |   -31.2   -3.7   -43.0 |   -25.2   -2.0   -36.1 |   -67.5  -2.28   -73.4 | - |
| 119 | `TEMA(5,10,14)` | 3MA |   -46.6  -2.45   -55.0 |   -19.0  -2.16   -34.9 |   -25.4  -2.07   -33.6 |   -67.7  -2.27   -72.9 | - |
| 120 | `TEMA(3,44)` | 2MA |   -36.9   -1.8   -51.8 |   -26.6  -2.99   -40.6 |   -30.8  -2.57   -37.0 |   -67.9  -2.26   -73.1 | - |


## 15m x KAMA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.901**; TRAIN+VAL top-10 -> OOS top-10 overlap = **7/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [4,102], slow in [23,239] -> 31/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [3,186], slow in [118,233] -> 16/60 configs positive across TRAIN & VAL & OOS
- band members: KAMA(62,105), KAMA(44,203), KAMA(73,145), KAMA(102,237), KAMA(52,203), KAMA(75,132,148), KAMA(37,203), KAMA(62,239), KAMA(48,67,118), KAMA(52,89), KAMA(19,132,233), KAMA(10,67,148), KAMA(30,132,186), KAMA(86,122), KAMA(22,151), KAMA(15,151), KAMA(60,67,233), KAMA(26,172), KAMA(18,128), KAMA(37,89), KAMA(12,106,148), KAMA(44,75), KAMA(10,128), KAMA(12,178) (+23 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `KAMA(62,105)` | 2MA |    50.8   2.17   -21.3 |    14.6   2.03   -14.7 |    35.8   3.66   -14.1 |   134.8   2.46   -21.3 | YES |
| 2 | `KAMA(44,203)` | 2MA |    23.7    1.4   -15.6 |    21.6   2.91    -9.9 |    46.9   4.28   -13.6 |   121.0   2.49   -15.6 | YES |
| 3 | `KAMA(73,145)` | 2MA |    35.6   2.15   -12.7 |    12.8   1.83   -17.3 |    44.5   4.27   -12.7 |   121.0   2.67   -17.3 | YES |
| 4 | `KAMA(102,237)` | 2MA |    29.7   1.92   -12.3 |     9.5   1.45   -18.3 |    49.7   4.51    -8.4 |   112.6   2.58   -18.6 | YES |
| 5 | `KAMA(52,203)` | 2MA |    20.9   1.26   -16.5 |    20.2   2.74   -11.1 |    41.6   3.84   -13.7 |   105.8   2.27   -16.5 | YES |
| 6 | `KAMA(75,132,148)` | 3MA |    33.6   2.59    -9.4 |     9.9   1.82   -11.4 |    39.0    5.0    -6.8 |   104.1   3.08   -11.4 | YES |
| 7 | `KAMA(37,203)` | 2MA |    21.6   1.37   -19.5 |    22.3   2.89   -10.2 |    35.9   3.47   -13.0 |   102.0   2.29   -19.5 | YES |
| 8 | `KAMA(62,239)` | 2MA |    22.9   1.47   -15.3 |    16.3   2.24   -12.6 |    37.2   3.51   -12.4 |    96.1   2.23   -15.3 | YES |
| 9 | `KAMA(48,67,118)` | 3MA |    32.9   2.56   -15.1 |    18.0   2.97    -6.9 |    23.6   2.95    -9.9 |    93.9   2.75   -15.1 | YES |
| 10 | `KAMA(52,89)` | 2MA |    17.3   1.12   -31.2 |    12.5    1.7   -15.8 |    45.2   4.14    -9.9 |    91.7   2.08   -31.2 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `KAMA(2,3,4)` | 3MA |   -23.6  -1.14   -35.6 |   -19.9   -2.5   -35.7 |   -19.6  -1.61   -33.6 |   -50.8  -1.55   -58.1 | - |
| 119 | `KAMA(3,6)` | 2MA |   -37.6  -2.02   -47.0 |   -13.5  -1.39   -32.7 |   -11.0  -0.64   -29.0 |   -52.0  -1.46   -59.3 | - |
| 120 | `KAMA(2,10)` | 2MA |   -31.3  -1.68   -41.0 |   -23.1  -2.91   -36.3 |   -13.8  -0.99   -30.0 |   -54.5  -1.72   -61.9 | - |


## 15m x VIDYA   (n_assets=10, n_configs=120)

**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = **rho=0.625**; TRAIN+VAL top-10 -> OOS top-10 overlap = **0/10**. Some ordering persists, but still prefer the band.

**WORKING BAND (positive across TRAIN & VAL & OOS):**
- 2MA: fast in [2,102], slow in [12,239] -> 54/60 configs positive across TRAIN & VAL & OOS
- 3MA: fast in [2,186], slow in [14,233] -> 48/60 configs positive across TRAIN & VAL & OOS
- band members: VIDYA(5,86), VIDYA(10,128), VIDYA(4,86), VIDYA(8,91), VIDYA(3,102), VIDYA(6,77), VIDYA(26,32), VIDYA(15,151), VIDYA(15,65), VIDYA(37,89), VIDYA(26,75), VIDYA(12,77), VIDYA(44,75), VIDYA(22,65), VIDYA(18,55), VIDYA(24,43,60), VIDYA(18,128), VIDYA(10,55), VIDYA(62,105), VIDYA(31,124), VIDYA(31,53), VIDYA(52,89), VIDYA(37,38), VIDYA(12,178) (+78 more)

| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | FULL net/Sh/DD | band? |
|---:|---|---|---|---|---|---|:---:|
| 1 | `VIDYA(5,86)` | 2MA |    42.9   2.42   -13.4 |    25.4   3.17    -8.4 |    22.3   2.16   -15.6 |   119.3   2.49   -15.6 | YES |
| 2 | `VIDYA(10,128)` | 2MA |    37.4   2.22   -13.6 |    15.6   2.25   -14.4 |    36.7   3.34   -11.6 |   117.2   2.55   -15.2 | YES |
| 3 | `VIDYA(4,86)` | 2MA |    40.7   2.33   -13.6 |    24.8    3.1    -8.8 |    23.5   2.23   -15.5 |   116.9   2.45   -15.5 | YES |
| 4 | `VIDYA(8,91)` | 2MA |    44.0   2.49   -12.0 |    24.7   3.12   -10.7 |    19.6   1.97   -15.1 |   114.8   2.45   -15.1 | YES |
| 5 | `VIDYA(3,102)` | 2MA |    47.0   2.61   -12.9 |    22.5   2.87   -11.5 |    19.2   1.88   -16.3 |   114.6   2.42   -16.3 | YES |
| 6 | `VIDYA(6,77)` | 2MA |    39.6   2.28   -14.5 |    25.3   3.14    -9.3 |    21.9   2.13   -15.4 |   113.3   2.41   -15.4 | YES |
| 7 | `VIDYA(26,32)` | 2MA |    33.0   1.97   -17.0 |    25.7   3.21   -10.9 |    25.6   2.44   -13.9 |   110.1   2.37   -17.0 | YES |
| 8 | `VIDYA(15,151)` | 2MA |    32.0   1.95   -14.1 |    14.7   2.19   -16.2 |    37.4   3.43   -10.1 |   107.9   2.43   -17.0 | YES |
| 9 | `VIDYA(15,65)` | 2MA |    38.7   2.26   -12.5 |    23.0   2.93   -11.7 |    21.7   2.13   -14.1 |   107.5   2.35   -14.1 | YES |
| 10 | `VIDYA(37,89)` | 2MA |    33.9   2.04   -13.9 |    14.3   2.15   -17.0 |    35.5   3.35    -9.2 |   107.3   2.44   -18.5 | YES |
| ... | _(107 configs omitted)_ |  |  |  |  |  |  |
| 118 | `VIDYA(2,4,19)` | 3MA |   -15.2  -0.94   -27.6 |    -6.7   -0.9   -20.0 |    -3.5  -0.17   -17.6 |   -23.7  -0.69   -33.5 | - |
| 119 | `VIDYA(2,3,4)` | 3MA |   -14.1  -0.73   -28.2 |   -13.5  -1.72   -29.7 |   -14.2  -1.16   -27.7 |   -36.2  -1.07   -44.4 | - |
| 120 | `VIDYA(2,3)` | 2MA |    -9.2  -0.37   -30.7 |   -16.8  -2.09   -32.4 |   -20.5  -1.76   -33.1 |   -40.0  -1.16   -49.3 | - |


# GLOBAL rank-stability summary (the transfer-noise headline)

| TF | MA-type | n_band(2MA/3MA) | Spearman rho (TV vs OOS) | top-10 overlap |
|---|---|---|---:|---:|
| 1d | EMA | 59/46 | 0.374 | 2/10 |
| 1d | SMA | 49/35 | 0.479 | 0/10 |
| 1d | WMA | 52/54 | 0.572 | 2/10 |
| 1d | HMA | 55/53 | 0.374 | 1/10 |
| 1d | DEMA | 56/58 | 0.585 | 1/10 |
| 1d | TEMA | 55/57 | 0.649 | 1/10 |
| 1d | KAMA | 43/31 | 0.457 | 1/10 |
| 1d | VIDYA | 42/32 | 0.579 | 4/10 |
| 4h | EMA | 60/60 | 0.437 | 3/10 |
| 4h | SMA | 56/59 | 0.339 | 3/10 |
| 4h | WMA | 56/58 | 0.237 | 7/10 |
| 4h | HMA | 54/55 | 0.332 | 6/10 |
| 4h | DEMA | 56/55 | 0.346 | 1/10 |
| 4h | TEMA | 52/55 | 0.19 | 1/10 |
| 4h | KAMA | 57/54 | 0.34 | 2/10 |
| 4h | VIDYA | 57/58 | 0.745 | 1/10 |
| 2h | EMA | 57/60 | 0.689 | 5/10 |
| 2h | SMA | 56/54 | 0.475 | 5/10 |
| 2h | WMA | 58/55 | 0.526 | 5/10 |
| 2h | HMA | 51/51 | 0.141 | 1/10 |
| 2h | DEMA | 52/49 | 0.222 | 0/10 |
| 2h | TEMA | 51/48 | 0.154 | 1/10 |
| 2h | KAMA | 57/59 | 0.494 | 3/10 |
| 2h | VIDYA | 57/57 | 0.807 | 4/10 |
| 1h | EMA | 53/55 | 0.672 | 2/10 |
| 1h | SMA | 53/53 | 0.534 | 1/10 |
| 1h | WMA | 51/58 | 0.632 | 1/10 |
| 1h | HMA | 42/36 | 0.723 | 2/10 |
| 1h | DEMA | 49/46 | 0.555 | 2/10 |
| 1h | TEMA | 46/39 | 0.733 | 5/10 |
| 1h | KAMA | 55/59 | 0.59 | 2/10 |
| 1h | VIDYA | 60/60 | 0.483 | 2/10 |
| 30m | EMA | 48/46 | 0.721 | 0/10 |
| 30m | SMA | 46/44 | 0.592 | 2/10 |
| 30m | WMA | 41/38 | 0.768 | 1/10 |
| 30m | HMA | 26/22 | 0.79 | 6/10 |
| 30m | DEMA | 39/27 | 0.854 | 2/10 |
| 30m | TEMA | 27/15 | 0.73 | 3/10 |
| 30m | KAMA | 45/34 | 0.849 | 3/10 |
| 30m | VIDYA | 57/57 | 0.536 | 0/10 |
| 15m | EMA | 37/28 | 0.885 | 2/10 |
| 15m | SMA | 29/23 | 0.764 | 4/10 |
| 15m | WMA | 29/17 | 0.863 | 5/10 |
| 15m | HMA | 12/8 | 0.689 | 7/10 |
| 15m | DEMA | 24/11 | 0.883 | 4/10 |
| 15m | TEMA | 13/7 | 0.639 | 7/10 |
| 15m | KAMA | 31/16 | 0.901 | 7/10 |
| 15m | VIDYA | 54/48 | 0.625 | 0/10 |

**Median Spearman rho across all cells = 0.587** (mean 0.574, n=48 cells). The closer to 0 / negative, the more the within-cell config RANK is noise that does NOT transfer TRAIN+VAL -> OOS. This is the empirical basis for 'trust the band, not the #1'.
