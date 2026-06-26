# TI × TIMEFRAME — FULL PERFORMANCE DNA (no exclusions)

Source: `ti_dna_20260625_183353.json` | VAL ['2023-07-01', '2024-05-15'] | OOS-2024 ['2024-05-15', '2024-12-31'] (2025 sealed)

Cost 0.0006 RT maker | gate BTC-SMA100 | EW u100 | null draws 60 | 130 cells

Pipeline audited 19/19 (PIPELINE_AUDIT_RESULT.md): causal, walled, cost-correct, null verified.


**Robustness score (0-6):** VALedge>0 + OOSedge>0 + VALnull≥95 + OOSnull≥95 + both-edges-positive(consistency) + selective(<70% both). Higher = more robust.


**Reads:** edge = linear %/wk vs always-long (honest). null = engine compound percentile vs random@same-deploy (≥95 = genuine timing). deploy = fraction of bars long (low = selective). Sharpe annualized. Compound shown for context (OOS-2024 rebalancing premium ≈1.1x per audit, so ~honest; VAL multi-month is mildly inflated).


## Full DNA table (ranked by robustness, then OOS edge)

| Rk | Robust | Family | TF | VAL edge/wk | VAL null | VAL dep | VAL Shrp | OOS edge/wk | OOS null | OOS dep | OOS Shrp | VAL comp | OOS comp | profile |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | HMA | 1d | +0.414 | 100 | 98% | 2.23 | +2.657 | 100 | 97% | 2.12 | +192% | +211% | robust tilt (non-sel) |
| 2 | 5 | WILLR | 1d | +0.338 | 99 | 99% | 2.21 | +1.295 | 100 | 100% | 2.10 | +183% | +127% | robust tilt (non-sel) |
| 3 | 5 | BOLLINGER | 1d | +0.422 | 100 | 98% | 2.39 | +1.290 | 100 | 99% | 1.92 | +199% | +120% | robust tilt (non-sel) |
| 4 | 5 | TEMA | 1d | +0.728 | 100 | 98% | 2.31 | +1.124 | 100 | 96% | 2.16 | +227% | +120% | robust tilt (non-sel) |
| 5 | 5 | VIDYA | 4h | +0.517 | 100 | 98% | 2.16 | +0.813 | 100 | 99% | 1.82 | +204% | +95% | robust tilt (non-sel) |
| 6 | 5 | SMA | 4h | +0.409 | 100 | 98% | 2.11 | +0.792 | 100 | 98% | 1.71 | +191% | +90% | robust tilt (non-sel) |
| 7 | 5 | VWAP | 1d | +0.456 | 100 | 98% | 2.38 | +0.630 | 100 | 100% | 1.77 | +202% | +86% | robust tilt (non-sel) |
| 8 | 5 | WILLR | 4h | +0.411 | 100 | 99% | 2.04 | +0.605 | 100 | 99% | 1.64 | +187% | +81% | robust tilt (non-sel) |
| 9 | 5 | MFI | 1d | +0.456 | 100 | 100% | 2.30 | +0.529 | 100 | 100% | 1.83 | +199% | +84% | robust tilt (non-sel) |
| 10 | 5 | WMA | 1h | +0.718 | 100 | 98% | 2.07 | +0.490 | 100 | 98% | 1.26 | +224% | +60% | robust tilt (non-sel) |
| 11 | 5 | SUPERTREND | 4h | +0.513 | 100 | 99% | 2.03 | +0.439 | 100 | 98% | 1.40 | +195% | +67% | robust tilt (non-sel) |
| 12 | 5 | TEMA | 4h | +1.290 | 100 | 99% | 2.47 | +0.392 | 100 | 97% | 1.44 | +313% | +67% | robust tilt (non-sel) |
| 13 | 5 | WMA | 4h | +0.763 | 100 | 98% | 2.12 | +0.361 | 100 | 98% | 1.46 | +226% | +67% | robust tilt (non-sel) |
| 14 | 5 | VIDYA | 1d | +0.700 | 100 | 100% | 2.48 | +0.275 | 100 | 100% | 1.61 | +233% | +69% | robust tilt (non-sel) |
| 15 | 5 | PSAR | 4h | +0.963 | 100 | 99% | 2.34 | +0.188 | 100 | 99% | 1.30 | +263% | +56% | robust tilt (non-sel) |
| 16 | 4 | SUPERTREND | 2h | +0.094 | 0 | 99% | 1.76 | +1.878 | 100 | 99% | 1.82 | +147% | +142% | consistent (weak null) |
| 17 | 4 | EMA | 4h | +0.120 | 0 | 98% | 1.89 | +1.456 | 100 | 98% | 2.15 | +155% | +137% | consistent (weak null) |
| 18 | 4 | DEMA | 1d | +0.040 | 0 | 99% | 1.80 | +1.411 | 100 | 99% | 2.29 | +139% | +138% | consistent (weak null) |
| 19 | 4 | KELTNER | 1d | +0.199 | 18 | 97% | 2.19 | +1.096 | 100 | 100% | 2.07 | +170% | +115% | consistent (weak null) |
| 20 | 4 | PSAR | 1d | +0.266 | 50 | 99% | 2.16 | +0.948 | 100 | 97% | 1.98 | +175% | +106% | consistent (weak null) |
| 21 | 4 | OBV | 1d | +0.071 | 0 | 99% | 2.03 | +0.465 | 100 | 99% | 1.79 | +153% | +80% | consistent (weak null) |
| 22 | 4 | VORTEX | 1d | +0.246 | 0 | 100% | 2.17 | +0.143 | 100 | 100% | 1.50 | +174% | +62% | consistent (weak null) |
| 23 | 4 | DEMA | 4h | +0.750 | 100 | 99% | 1.86 | +0.093 | 72 | 97% | 1.27 | +202% | +53% | consistent (weak null) |
| 24 | 4 | MACD | 1d | +0.487 | 100 | 98% | 2.26 | +0.089 | 77 | 97% | 1.38 | +200% | +57% | consistent (weak null) |
| 25 | 4 | DEMA | 2h | +0.570 | 100 | 99% | 2.00 | +0.084 | 0 | 98% | 1.13 | +201% | +46% | consistent (weak null) |
| 26 | 4 | RSI | 1d | +0.375 | 100 | 97% | 2.25 | +0.020 | 0 | 100% | 1.27 | +189% | +51% | consistent (weak null) |
| 27 | 2 | VIDYA | 2h | -0.178 | 0 | 97% | 1.74 | +2.887 | 100 | 98% | 2.89 | +129% | +269% | OOS-only (VAL≤0) |
| 28 | 2 | CCI | 1d | -0.658 | 0 | 96% | 1.47 | +2.868 | 100 | 97% | 2.15 | +84% | +224% | OOS-only (VAL≤0) |
| 29 | 2 | KAMA | 4h | -0.611 | 0 | 99% | 1.36 | +2.552 | 100 | 100% | 2.36 | +84% | +217% | OOS-only (VAL≤0) |
| 30 | 2 | HMA | 1h | -3.451 | 0 | 98% | -0.65 | +2.462 | 100 | 97% | 2.38 | -50% | +208% | OOS-only (VAL≤0) |
| 31 | 2 | HMA | 2h | -0.236 | 0 | 98% | 1.55 | +2.397 | 100 | 97% | 1.98 | +114% | +177% | OOS-only (VAL≤0) |
| 32 | 2 | SUPERTREND | 1d | -0.463 | 0 | 99% | 1.54 | +2.336 | 100 | 99% | 1.97 | +97% | +180% | OOS-only (VAL≤0) |
| 33 | 2 | AROON | 1h | -2.421 | 0 | 97% | 0.04 | +2.033 | 100 | 96% | 1.90 | -25% | +150% | OOS-only (VAL≤0) |
| 34 | 2 | PSAR | 1h | -0.924 | 0 | 99% | 1.00 | +1.800 | 100 | 98% | 2.02 | +53% | +148% | OOS-only (VAL≤0) |
| 35 | 2 | DONCHIAN | 4h | -0.194 | 0 | 98% | 1.63 | +1.630 | 100 | 98% | 2.19 | +120% | +147% | OOS-only (VAL≤0) |
| 36 | 2 | TEMA | 2h | -1.137 | 0 | 98% | 0.89 | +1.408 | 100 | 98% | 1.62 | +40% | +110% | OOS-only (VAL≤0) |
| 37 | 2 | SMA | 30m | -1.337 | 0 | 98% | 0.79 | +1.148 | 100 | 97% | 1.57 | +31% | +96% | OOS-only (VAL≤0) |
| 38 | 2 | VORTEX | 4h | -0.315 | 0 | 100% | 1.68 | +1.041 | 100 | 100% | 2.12 | +115% | +115% | OOS-only (VAL≤0) |
| 39 | 2 | EMA | 30m | -0.926 | 0 | 98% | 0.99 | +0.947 | 100 | 97% | 1.43 | +52% | +81% | OOS-only (VAL≤0) |
| 40 | 2 | DONCHIAN | 30m | -1.981 | 0 | 98% | 0.36 | +0.865 | 100 | 97% | 1.44 | -4% | +80% | OOS-only (VAL≤0) |
| 41 | 2 | MACD | 1h | -3.240 | 0 | 98% | -0.51 | +0.583 | 100 | 98% | 1.35 | -45% | +68% | OOS-only (VAL≤0) |
| 42 | 2 | AROON | 30m | -4.195 | 0 | 96% | -0.95 | +0.583 | 98 | 95% | 1.21 | -67% | +58% | OOS-only (VAL≤0) |
| 43 | 2 | OBV | 4h | -0.680 | 0 | 100% | 1.37 | +0.557 | 100 | 99% | 1.32 | +81% | +69% | OOS-only (VAL≤0) |
| 44 | 2 | RSI | 4h | -0.638 | 0 | 96% | 1.27 | +0.401 | 100 | 97% | 1.38 | +77% | +65% | OOS-only (VAL≤0) |
| 45 | 2 | VIDYA | 1h | -0.188 | 0 | 98% | 1.60 | +0.329 | 100 | 99% | 1.42 | +123% | +65% | OOS-only (VAL≤0) |
| 46 | 2 | KAMA | 2h | -0.878 | 0 | 99% | 1.17 | +0.323 | 100 | 99% | 1.50 | +64% | +67% | OOS-only (VAL≤0) |
| 47 | 2 | AROON | 4h | -1.208 | 0 | 97% | 0.83 | +0.295 | 100 | 95% | 1.34 | +34% | +60% | OOS-only (VAL≤0) |
| 48 | 2 | WILLR | 1h | -1.593 | 0 | 100% | 0.63 | +0.210 | 100 | 99% | 1.32 | +18% | +58% | OOS-only (VAL≤0) |
| 49 | 2 | EMA | 1d | +0.479 | 100 | 100% | 2.27 | -0.157 | 0 | 100% | 1.27 | +200% | +47% | VAL-only (OOS≤0) |
| 50 | 2 | CMF | 1d | +0.555 | 100 | 100% | 2.40 | -0.216 | 0 | 100% | 1.31 | +214% | +46% | VAL-only (OOS≤0) |
| 51 | 2 | FLOW | 1d | +0.256 | 100 | 100% | 2.21 | -0.320 | 0 | 100% | 1.13 | +176% | +39% | VAL-only (OOS≤0) |
| 52 | 2 | SMA | 1d | +0.365 | 100 | 98% | 2.24 | -0.387 | 0 | 100% | 1.11 | +187% | +37% | VAL-only (OOS≤0) |
| 53 | 2 | HMA | 4h | +1.835 | 100 | 99% | 2.70 | -0.529 | 0 | 98% | 0.78 | +417% | +21% | VAL-only (OOS≤0) |
| 54 | 2 | WMA | 2h | +0.597 | 100 | 99% | 2.01 | -0.579 | 0 | 98% | 0.75 | +205% | +19% | VAL-only (OOS≤0) |
| 55 | 2 | WILLR | 2h | +0.463 | 100 | 100% | 2.13 | -0.705 | 0 | 99% | 0.73 | +200% | +18% | VAL-only (OOS≤0) |
| 56 | 2 | SMA | 1h | +0.869 | 100 | 99% | 2.26 | -0.730 | 0 | 98% | 0.67 | +255% | +14% | VAL-only (OOS≤0) |
| 57 | 2 | EMA | 2h | +0.629 | 100 | 98% | 2.03 | -0.809 | 0 | 98% | 0.65 | +209% | +14% | VAL-only (OOS≤0) |
| 58 | 1 | ADX | 4h | -0.314 | 0 | 74% | 1.12 | +0.563 | 88 | 69% | 1.22 | +72% | +60% | null/bleed |
| 59 | 1 | DEMA | 1h | -2.209 | 0 | 98% | 0.19 | +0.148 | 0 | 98% | 1.02 | -13% | +40% | null/bleed |
| 60 | 1 | WMA | 30m | -2.107 | 0 | 98% | 0.27 | +0.095 | 0 | 97% | 1.00 | -11% | +38% | null/bleed |
| 61 | 1 | AROON | 1d | +0.311 | 63 | 96% | 2.13 | -0.076 | 0 | 96% | 1.23 | +177% | +47% | null/bleed |
| 62 | 1 | PSAR | 2h | +0.133 | 0 | 99% | 1.77 | -0.092 | 0 | 98% | 1.09 | +151% | +41% | null/bleed |
| 63 | 1 | DONCHIAN | 1d | +0.223 | 17 | 98% | 2.19 | -0.234 | 0 | 100% | 1.24 | +172% | +44% | null/bleed |
| 64 | 1 | MACD | 4h | +0.042 | 0 | 99% | 1.70 | -0.386 | 0 | 98% | 0.94 | +138% | +31% | null/bleed |
| 65 | 1 | KAMA | 1h | +0.055 | 0 | 99% | 1.76 | -0.392 | 0 | 99% | 0.90 | +148% | +28% | null/bleed |
| 66 | 1 | ADX | 1d | +0.773 | 67 | 70% | 1.69 | -0.847 | 0 | 71% | 0.58 | +183% | +9% | null/bleed |
| 67 | 1 | OBV | 2h | -2.435 | 0 | 0% | 0.00 | -1.713 | 0 | 0% | 0.00 | +0% | +0% | abstain (2h synth) |
| 68 | 1 | VWAP | 2h | -2.435 | 0 | 0% | 0.00 | -1.713 | 0 | 0% | 0.00 | +0% | +0% | abstain (2h synth) |
| 69 | 1 | CMF | 2h | -2.435 | 0 | 0% | 0.00 | -1.713 | 0 | 0% | 0.00 | +0% | +0% | abstain (2h synth) |
| 70 | 1 | MFI | 2h | -2.435 | 0 | 0% | 0.00 | -1.713 | 0 | 0% | 0.00 | +0% | +0% | abstain (2h synth) |
| 71 | 1 | FLOW | 2h | -2.435 | 0 | 0% | 0.00 | -1.713 | 0 | 0% | 0.00 | +0% | +0% | abstain (2h synth) |
| 72 | 1 | ADX | 1h | -8.595 | 0 | 67% | -2.95 | -4.230 | 0 | 65% | -1.14 | -96% | -70% | null/bleed |
| 73 | 1 | ADX | 30m | -14.629 | 0 | 65% | -5.70 | -6.370 | 0 | 65% | -2.13 | -100% | -85% | null/bleed |
| 74 | 0 | KAMA | 30m | -0.684 | 0 | 99% | 1.26 | -0.034 | 0 | 98% | 1.07 | +80% | +42% | null/bleed |
| 75 | 0 | WMA | 1d | -0.091 | 0 | 98% | 1.83 | -0.051 | 0 | 99% | 1.35 | +132% | +52% | null/bleed |
| 76 | 0 | VWAP | 4h | -0.813 | 0 | 98% | 1.16 | -0.077 | 0 | 98% | 1.11 | +65% | +42% | null/bleed |
| 77 | 0 | TEMA | 1h | -2.123 | 0 | 98% | 0.23 | -0.125 | 0 | 97% | 1.03 | -12% | +38% | null/bleed |
| 78 | 0 | KELTNER | 4h | -0.487 | 0 | 97% | 1.37 | -0.174 | 0 | 97% | 1.00 | +90% | +36% | null/bleed |
| 79 | 0 | MFI | 4h | -0.415 | 0 | 100% | 1.58 | -0.183 | 0 | 99% | 1.21 | +104% | +45% | null/bleed |
| 80 | 0 | DONCHIAN | 2h | -1.538 | 0 | 99% | 0.65 | -0.236 | 0 | 97% | 0.98 | +20% | +34% | null/bleed |
| 81 | 0 | SMA | 2h | -0.349 | 0 | 99% | 1.48 | -0.245 | 0 | 98% | 1.02 | +104% | +36% | null/bleed |
| 82 | 0 | CMF | 4h | -0.915 | 0 | 100% | 1.18 | -0.318 | 0 | 100% | 1.08 | +63% | +37% | null/bleed |
| 83 | 0 | KAMA | 1d | -0.060 | 0 | 99% | 1.97 | -0.322 | 0 | 99% | 1.12 | +140% | +38% | null/bleed |
| 84 | 0 | MACD | 2h | -0.669 | 0 | 99% | 1.25 | -0.374 | 0 | 98% | 0.92 | +76% | +30% | null/bleed |
| 85 | 0 | STOCHASTIC | 1d | -2.548 | 0 | 97% | -0.17 | -0.525 | 0 | 93% | 0.75 | -25% | +20% | null/bleed |
| 86 | 0 | ADX | 2h | -1.776 | 0 | 71% | 0.33 | -0.582 | 0 | 69% | 0.48 | -16% | -6% | null/bleed |
| 87 | 0 | VORTEX | 2h | -0.197 | 0 | 99% | 1.62 | -0.673 | 0 | 98% | 0.76 | +121% | +20% | null/bleed |
| 88 | 0 | FLOW | 4h | -0.220 | 0 | 100% | 1.62 | -0.692 | 0 | 100% | 0.79 | +118% | +21% | null/bleed |
| 89 | 0 | DONCHIAN | 1h | -0.231 | 0 | 98% | 1.51 | -0.806 | 0 | 97% | 0.58 | +114% | +9% | null/bleed |
| 90 | 0 | VIDYA | 30m | -0.168 | 0 | 98% | 1.54 | -0.856 | 0 | 98% | 0.61 | +122% | +11% | null/bleed |
| 91 | 0 | MFI | 1h | -1.262 | 0 | 100% | 0.92 | -0.898 | 0 | 99% | 0.65 | +41% | +14% | null/bleed |
| 92 | 0 | KELTNER | 2h | -1.069 | 0 | 98% | 0.91 | -0.919 | 0 | 97% | 0.53 | +43% | +7% | null/bleed |
| 93 | 0 | SUPERTREND | 1h | -1.490 | 0 | 99% | 0.68 | -1.018 | 0 | 98% | 0.47 | +22% | +4% | null/bleed |
| 94 | 0 | AROON | 2h | -0.376 | 0 | 97% | 1.37 | -1.105 | 0 | 95% | 0.40 | +96% | -0% | null/bleed |
| 95 | 0 | CCI | 2h | -1.252 | 0 | 97% | 0.79 | -1.111 | 0 | 95% | 0.35 | +31% | -5% | null/bleed |
| 96 | 0 | CCI | 4h | -0.335 | 0 | 97% | 1.41 | -1.143 | 0 | 95% | 0.35 | +98% | -4% | null/bleed |
| 97 | 0 | CCI | 1h | -4.778 | 0 | 97% | -1.43 | -1.235 | 0 | 95% | 0.28 | -74% | -10% | null/bleed |
| 98 | 0 | RSI | 2h | -0.828 | 0 | 97% | 1.07 | -1.320 | 0 | 95% | 0.25 | +59% | -7% | null/bleed |
| 99 | 0 | BOLLINGER | 2h | -0.126 | 0 | 99% | 1.56 | -1.375 | 0 | 98% | 0.23 | +121% | -8% | null/bleed |
| 100 | 0 | BOLLINGER | 4h | -0.082 | 0 | 98% | 1.65 | -1.393 | 0 | 98% | 0.22 | +128% | -7% | null/bleed |
| 101 | 0 | MFI | 30m | -2.424 | 0 | 100% | 0.09 | -1.504 | 0 | 99% | 0.19 | -15% | -7% | null/bleed |
| 102 | 0 | RSI | 1h | -1.205 | 0 | 97% | 0.80 | -1.537 | 0 | 95% | 0.12 | +32% | -18% | null/bleed |
| 103 | 0 | DEMA | 30m | -4.745 | 0 | 98% | -1.36 | -1.581 | 0 | 97% | 0.11 | -73% | -15% | null/bleed |
| 104 | 0 | VORTEX | 1h | -2.192 | 0 | 100% | 0.22 | -1.591 | 0 | 100% | 0.12 | -7% | -10% | null/bleed |
| 105 | 0 | EMA | 1h | -0.165 | 0 | 98% | 1.56 | -1.634 | 0 | 98% | 0.07 | +122% | -15% | null/bleed |
| 106 | 0 | BOLLINGER | 1h | -2.652 | 0 | 98% | -0.11 | -1.744 | 0 | 97% | 0.00 | -29% | -20% | null/bleed |
| 107 | 0 | PSAR | 30m | -4.244 | 0 | 98% | -1.05 | -1.753 | 0 | 98% | 0.01 | -66% | -22% | null/bleed |
| 108 | 0 | RSI | 30m | -2.905 | 0 | 97% | -0.21 | -1.826 | 0 | 95% | -0.03 | -40% | -26% | null/bleed |
| 109 | 0 | FLOW | 1h | -1.423 | 0 | 100% | 0.78 | -1.876 | 0 | 100% | -0.10 | +30% | -17% | null/bleed |
| 110 | 0 | CMF | 1h | -2.077 | 0 | 100% | 0.31 | -1.936 | 0 | 100% | -0.15 | -2% | -18% | null/bleed |
| 111 | 0 | VWAP | 1h | -1.579 | 0 | 98% | 0.61 | -2.235 | 0 | 97% | -0.30 | +16% | -32% | null/bleed |
| 112 | 0 | FLOW | 30m | -3.598 | 0 | 100% | -0.75 | -2.554 | 0 | 100% | -0.59 | -51% | -34% | null/bleed |
| 113 | 0 | KELTNER | 1h | -3.510 | 0 | 98% | -0.68 | -2.669 | 0 | 97% | -0.59 | -52% | -40% | null/bleed |
| 114 | 0 | WILLR | 30m | -4.353 | 0 | 99% | -1.25 | -2.689 | 0 | 99% | -0.65 | -66% | -38% | null/bleed |
| 115 | 0 | OBV | 1h | -3.840 | 0 | 100% | -1.01 | -3.194 | 0 | 99% | -1.10 | -56% | -46% | null/bleed |
| 116 | 0 | STOCHASTIC | 4h | -5.341 | 0 | 96% | -2.00 | -3.248 | 0 | 95% | -1.04 | -80% | -50% | null/bleed |
| 117 | 0 | CMF | 30m | -3.814 | 0 | 100% | -0.94 | -3.494 | 0 | 100% | -1.35 | -55% | -51% | null/bleed |
| 118 | 0 | VWAP | 30m | -5.917 | 0 | 98% | -2.17 | -4.090 | 0 | 97% | -1.39 | -84% | -63% | null/bleed |
| 119 | 0 | SUPERTREND | 30m | -6.942 | 0 | 99% | -2.91 | -4.112 | 0 | 98% | -1.43 | -90% | -63% | null/bleed |
| 120 | 0 | VORTEX | 30m | -4.654 | 0 | 100% | -1.52 | -4.153 | 0 | 99% | -1.79 | -70% | -61% | null/bleed |
| 121 | 0 | TEMA | 30m | -8.011 | 0 | 98% | -3.50 | -4.710 | 0 | 98% | -1.87 | -94% | -69% | null/bleed |
| 122 | 0 | MACD | 30m | -8.210 | 0 | 98% | -3.67 | -4.739 | 0 | 98% | -1.79 | -94% | -70% | null/bleed |
| 123 | 0 | KELTNER | 30m | -9.729 | 0 | 97% | -4.54 | -4.789 | 0 | 96% | -1.58 | -97% | -73% | null/bleed |
| 124 | 0 | BOLLINGER | 30m | -8.420 | 0 | 98% | -3.82 | -5.224 | 0 | 97% | -1.98 | -95% | -75% | null/bleed |
| 125 | 0 | HMA | 30m | -8.152 | 0 | 98% | -3.46 | -5.929 | 0 | 98% | -2.63 | -94% | -79% | null/bleed |
| 126 | 0 | CCI | 30m | -9.877 | 0 | 96% | -4.15 | -6.160 | 0 | 95% | -2.24 | -98% | -83% | null/bleed |
| 127 | 0 | STOCHASTIC | 2h | -7.817 | 0 | 96% | -3.42 | -7.532 | 0 | 96% | -4.42 | -94% | -87% | null/bleed |
| 128 | 0 | OBV | 30m | -8.692 | 0 | 100% | -4.58 | -7.532 | 0 | 99% | -4.44 | -95% | -87% | null/bleed |
| 129 | 0 | STOCHASTIC | 1h | -16.709 | 0 | 96% | -9.83 | -15.356 | 0 | 95% | -9.56 | -100% | -99% | null/bleed |
| 130 | 0 | STOCHASTIC | 30m | -33.628 | 0 | 97% | -20.17 | -24.673 | 0 | 96% | -13.88 | -100% | -100% | null/bleed |

## Best per timeframe (by OOS edge, among consistent cells)


**1d** (consistent cells, top 6):
  - HMA        OOS +2.657pp/wk, null 100, deploy 97%, Sharpe 2.12
  - DEMA       OOS +1.411pp/wk, null 100, deploy 99%, Sharpe 2.29
  - WILLR      OOS +1.295pp/wk, null 100, deploy 100%, Sharpe 2.10
  - BOLLINGER  OOS +1.290pp/wk, null 100, deploy 99%, Sharpe 1.92
  - TEMA       OOS +1.124pp/wk, null 100, deploy 96%, Sharpe 2.16
  - KELTNER    OOS +1.096pp/wk, null 100, deploy 100%, Sharpe 2.07

**4h** (consistent cells, top 6):
  - EMA        OOS +1.456pp/wk, null 100, deploy 98%, Sharpe 2.15
  - VIDYA      OOS +0.813pp/wk, null 100, deploy 99%, Sharpe 1.82
  - SMA        OOS +0.792pp/wk, null 100, deploy 98%, Sharpe 1.71
  - WILLR      OOS +0.605pp/wk, null 100, deploy 99%, Sharpe 1.64
  - SUPERTREND OOS +0.439pp/wk, null 100, deploy 98%, Sharpe 1.40
  - TEMA       OOS +0.392pp/wk, null 100, deploy 97%, Sharpe 1.44

**2h** (consistent cells, top 6):
  - SUPERTREND OOS +1.878pp/wk, null 100, deploy 99%, Sharpe 1.82
  - DEMA       OOS +0.084pp/wk, null 0, deploy 98%, Sharpe 1.13

## Honest summary

- 130 cells total. **26** are DEV/OOS-consistent (edge>0 both windows).
- **22** are consistent AND beat OOS random-null ≥95th pct (genuine timing).
- **7** are selective (deploy<70% both windows) — these are the rare ones.
- Most TI×TF cells deploy 95-100% (always-long-with-tilt under the bull gate): they refine WHICH longs, not WHETHER long. Consistent with the standing finding that the regime gate is the dominant value lever (D78 CASH THEOREM).
- VAL→OOS sign flips are common at sub-daily TFs (2h especially) — the noise signature. 1d and 4h carry the most VAL/OOS-consistent cells.
- Volume families (OBV/VWAP/CMF/MFI/FLOW) abstain at 2h (chimera volume features have no native-2h baseline → NaN, correctly treated as unavailable, not faked).