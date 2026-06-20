"""
gate_sweep_verify.py -- Sanity checks for the gate sweep results.

1. Verify BTC sma200 gate: BTC spent majority of 2022 below SMA200 -> book should be mostly flat.
2. Verify BTC sma100 has no look-ahead: it's purely C[t] > SMA100[t] computed on past data.
3. Spot check per-year: compare BTC gate vs per-asset gate on small sample.
4. Check if 2022 == 0.0 for btc_sma200 makes sense (zero = all-cash 2022?).
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")
import strat.mover_lab as ml

ind = ml.load("2020-01-01", "2026-06-01")
C = ind["C"]
sma100 = C.rolling(100, min_periods=100).mean()
sma200 = ind["sma200"]

btc = "BTCUSDT"
btc_above_200 = (C[btc] > sma200[btc]).fillna(False)
btc_above_100 = (C[btc] > sma100[btc]).fillna(False)

# How many 2022 days is BTC above each SMA?
mask_2022 = (C.index >= "2022-01-01") & (C.index < "2023-01-01")
print("=== BTC gate activity in 2022 ===")
print(f"Days BTC above SMA200 in 2022: {btc_above_200[mask_2022].sum()} / {mask_2022.sum()}")
print(f"Days BTC above SMA100 in 2022: {btc_above_100[mask_2022].sum()} / {mask_2022.sum()}")

# So btc_sma200: book is entirely in cash for 2022 -> 0.0% is correct (no trades = 0 return)
# btc_sma100: some exposure in early 2022 (Jan-March when BTC was still above SMA100)

# Verify causal: SMA100 computed on rolling(100) - purely past window, no future leak
# SMA at time t uses data [t-99 .. t], t-inclusive, which is historical close prices -> CAUSAL OK

# Check the high full-cycle of btc_sma100 is real (not a fluke from 2021 supercycle)
print("\n=== Year-by-year for BTC-sma100 gate (no per-asset gate) ===")
print("These should match the printed table:")
print("  2021 = +2560%, 2022 = -19.7%, 2023 = +127%, 2024 = +112%, 2025 = +8.3%")
print("  The 30671% full is 2021 dominance: 2560% x 10x on 2020's 204% base")

# Multi-year compounding check: approximate
approx_full = (1+2.047)*(1+25.605)*(1-0.197)*(1+1.272)*(1+1.123)*(1+0.083) - 1
print(f"\n  Approximate full compound check: {approx_full*100:.0f}%  (vs reported 30671%)")

# Why is sma100 better than sma200?
# SMA100 is a faster gate: re-enters bull market sooner after a bear bottom
# In 2022: SMA200 keeps out ALL year; SMA100 may have re-entered briefly in Q4 2022 (BTC rebounded)
print("\n=== BTC SMA crossings in 2022 ===")
cross_200 = btc_above_200[mask_2022]
cross_100 = btc_above_100[mask_2022]
dates_2022 = C.index[mask_2022]
# When does BTC first go below sma100/sma200 in 2022?
below_200_dates = dates_2022[~cross_200]
below_100_dates = dates_2022[~cross_100]
if len(below_200_dates):
    print(f"BTC first below SMA200 in 2022: {below_200_dates[0].date()}")
if len(below_100_dates):
    print(f"BTC first below SMA100 in 2022: {below_100_dates[0].date()}")

# Does BTC recover above sma100 in 2022?
above_100_in_2022 = dates_2022[cross_100]
print(f"Days BTC above SMA100 in 2022: {len(above_100_in_2022)} ({100*len(above_100_in_2022)/mask_2022.sum():.0f}%)")
if len(above_100_in_2022):
    print(f"  First: {above_100_in_2022[0].date()}  Last: {above_100_in_2022[-1].date()}")

# Check 2025 exposure and what happened
mask_2025 = (C.index >= "2025-01-01") & (C.index < "2026-01-01")
print(f"\n=== 2025 BTC gate activity ===")
print(f"Days BTC above SMA200 in 2025: {btc_above_200[mask_2025].sum()} / {mask_2025.sum()} = {100*btc_above_200[mask_2025].mean():.0f}%")
print(f"Days BTC above SMA100 in 2025: {btc_above_100[mask_2025].sum()} / {mask_2025.sum()} = {100*btc_above_100[mask_2025].mean():.0f}%")

# Buy-hold reference 2022 and 2025 (universe EW)
R = ind["R"].fillna(0.0)
bh = R.fillna(0.0).mean(axis=1)
def bh_year(s, e):
    m = (bh.index >= s) & (bh.index < e)
    return round((np.prod(1 + bh[m].values) - 1)*100, 1)

print("\n=== Buy-hold (universe EW) reference ===")
for yr in ["2020","2021","2022","2023","2024","2025"]:
    print(f"  {yr}: {bh_year(f'{yr}-01-01', f'{int(yr)+1}-01-01')}%")
print(f"  full: {bh_year('2020-01-01','2026-06-01')}%")
