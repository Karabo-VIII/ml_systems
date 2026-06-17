# PEPE EMA/MA FINAL synthesis -- 2026-05-23

## All-4-window-positive candidates (deploy-grade)

| cadence   | ma_type   |   fast |   slow | filter                  |    TRAIN |     VAL |   OOS_pre |   UNSEEN |   TRAIN_n |   VAL_n |   OOS_pre_n |   UNSEEN_n |   worst_window |   mean_compound |
|:----------|:----------|-------:|-------:|:------------------------|---------:|--------:|----------:|---------:|----------:|--------:|------------:|-----------:|---------------:|----------------:|
| 1h        | SMA_cross |      9 |     21 | whale_net>median        | 2113.97  | 745.415 |  125.816  |  78.5321 |       160 |     128 |         131 |         56 |        78.5321 |         765.933 |
| 1h        | SMA_cross |      7 |     15 | whale_net>median        |  809.854 | 361.798 |  113.533  |  64.1406 |       223 |     172 |         182 |         72 |        64.1406 |         337.332 |
| 1h        | EMA_cross |      9 |     21 | pepe_bull_AND_whale>med | 1745.51  | 161.782 |  140.599  |  62.258  |        54 |      48 |          31 |         25 |        62.258  |         527.536 |
| 1h        | EMA_cross |     12 |     26 | pepe_bull_AND_whale>med | 2376.93  | 218.824 |  164.248  |  61.4391 |        47 |      39 |          25 |         23 |        61.4391 |         705.36  |
| 1h        | SMA_cross |      9 |     21 | pepe_bull_AND_whale>med | 1112.88  | 211.324 |  129.833  |  57.7971 |        67 |      45 |          34 |         31 |        57.7971 |         377.959 |
| 1h        | SMA_cross |      5 |     30 | pepe_bull_AND_whale>med | 1869.43  | 118.735 |  124.389  |  54.1397 |        55 |      49 |          33 |         29 |        54.1397 |         541.675 |
| 1h        | SMA_cross |      5 |     15 | whale_net>median        |  878.805 | 662.395 |   66.2163 |  48.8628 |       245 |     176 |         194 |         74 |        48.8628 |         414.07  |
| 1h        | EMA_cross |      7 |     15 | pepe_bull_AND_whale>med | 1284.08  | 149.909 |  102.607  |  48.2092 |        72 |      53 |          41 |         32 |        48.2092 |         396.201 |
| 1h        | SMA_cross |      7 |     15 | pepe_bull_AND_whale>med |  612.175 | 120.098 |   97.9384 |  45.7431 |        85 |      58 |          43 |         38 |        45.7431 |         218.989 |
| 4h        | SMA_cross |      9 |     21 | whale_net>median        | 3540.53  | 594.551 |  282.708  |  44.7381 |        63 |      60 |          52 |         24 |        44.7381 |        1115.63  |
| 1h        | EMA_cross |     12 |     26 | whale_net>median        | 3609.74  | 423.847 |  201.786  |  42.4988 |       117 |     108 |          96 |         41 |        42.4988 |        1069.47  |
| 1h        | SMA_cross |      5 |     30 | whale_net>median        | 2121.08  | 415.198 |  143.012  |  42.3372 |       152 |     121 |         127 |         57 |        42.3372 |         680.408 |
| 4h        | SMA_state |     30 |      0 | pepe_bull_AND_whale>med | 1247.04  | 298.726 |  129.289  |  41.2758 |        41 |      27 |          24 |         16 |        41.2758 |         429.084 |
| 1h        | EMA_cross |      9 |     21 | whale_net>median        | 2000.03  | 366.352 |  157.096  |  41.2705 |       137 |     127 |         114 |         51 |        41.2705 |         641.188 |
| 4h        | EMA_cross |     12 |     26 | pepe_bull_AND_whale>med | 1785.16  | 267.563 |  201.471  |  40.2627 |        27 |      24 |          14 |         15 |        40.2627 |         573.615 |
| 4h        | SMA_cross |      9 |     21 | pepe_bull_AND_whale>med | 1681.81  | 256.076 |  186.899  |  39.0828 |        28 |      23 |          17 |         12 |        39.0828 |         540.968 |
| 4h        | EMA_cross |     12 |     26 | whale_net>median        | 4229.28  | 319.282 |  326.597  |  38.6823 |        46 |      47 |          44 |         22 |        38.6823 |        1228.46  |
| 1h        | SMA_cross |      5 |     15 | pepe_bull_AND_whale>med |  699.442 | 170.066 |   83.167  |  37.8007 |        92 |      58 |          48 |         39 |        37.8007 |         247.619 |
| 1h        | EMA_cross |      7 |     15 | whale_net>median        |  958.381 | 527.418 |  143.64   |  37.3626 |       198 |     146 |         151 |         63 |        37.3626 |         416.7   |
| 4h        | EMA_cross |      9 |     21 | pepe_bull_AND_whale>med | 1433.74  | 285.396 |  161.118  |  33.3659 |        30 |      23 |          17 |         14 |        33.3659 |         478.405 |
| 1h        | SMA_state |     30 |      0 | pepe_bull_AND_whale>med |  990.167 | 251.081 |   95.3571 |  33.3155 |        95 |      65 |          62 |         51 |        33.3155 |         342.48  |
| 4h        | SMA_state |     30 |      0 | whale_net>median        | 1371.36  | 501.673 |  214.413  |  32.451  |        93 |      73 |          69 |         31 |        32.451  |         529.974 |
| 1h        | EMA_cross |      5 |     15 | pepe_bull_AND_whale>med |  868.175 | 238.412 |  112.166  |  31.9835 |        86 |      59 |          42 |         40 |        31.9835 |         312.684 |
| 4h        | EMA_cross |      9 |     21 | whale_net>median        | 3415.89  | 451.442 |  258.329  |  31.7727 |        53 |      49 |          50 |         24 |        31.7727 |        1039.36  |
| 4h        | EMA_cross |      5 |     15 | pepe_bull_AND_whale>med | 1715.21  | 321.194 |  148.427  |  31.6513 |        31 |      29 |          20 |         15 |        31.6513 |         554.122 |

## Methodology

- Family size N = 324 (config x filter combinations)
- Multi-test correction: at N=324, expected ~20 configs random-positive in 4 windows by chance
- Observed: 49 configs positive in 4 windows
- If observed >> expected, signal is genuine
