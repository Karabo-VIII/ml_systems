"""Re-analyze funding CSV with CORRECTED hypothesis.

Original hypothesis (mean reversion) was backwards. The data shows that
extreme positive funding-rate z-scores predict CONTINUED POSITIVE returns
on 72h horizons -- classical trend continuation.

Test directly: LONG on z > +threshold, measure forward returns.
"""
import pandas as pd
import numpy as np
import math
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "logs" / "funding_extreme_events.csv"
TAKER_RT = 0.0016

ev = pd.read_csv(CSV)
print(f"[info] loaded {len(ev)} events, {ev['sym'].nunique()} assets")
print(f"[info] z range: p1={ev['z'].quantile(0.01):.2f} p99={ev['z'].quantile(0.99):.2f}")

print("\n" + "=" * 100)
print(f"LONG on z > +THRESH (momentum continuation hypothesis)")
print(f"{'thresh':>7} {'horizon':>8} {'n':>6} {'asset_mean':>11} {'long_net':>10} {'std':>7} {'t_long':>7} {'hit%':>6} verdict")
print("=" * 100)

ship_long = []
for z_thresh in [1.0, 1.5, 2.0, 2.5, 3.0]:
    for h_col in ["r_8h", "r_24h", "r_48h", "r_72h"]:
        mask = (ev["z"] > z_thresh) & ev[h_col].notna()
        rets = ev.loc[mask, h_col].values
        if len(rets) < 30:
            continue
        n = len(rets)
        mean_g = rets.mean()
        mean_n = mean_g - TAKER_RT
        std = rets.std(ddof=1) if n > 1 else 1e-9
        std = max(std, 1e-9)
        t_stat = (mean_n * math.sqrt(n)) / std
        hit = (rets > TAKER_RT).mean()
        flag = "  <<SHIP" if (n >= 100 and mean_n > 0.005 and t_stat > 2.0 and hit > 0.55) else ""
        print(f"{z_thresh:>6.1f} {h_col.replace('r_',''):>7} {n:>6} "
              f"{mean_g*100:>+9.2f}% {mean_n*100:>+8.2f}% {std*100:>6.2f}% "
              f"{t_stat:>6.2f} {hit*100:>5.1f}%{flag}")
        if flag:
            ship_long.append((z_thresh, h_col, n, mean_n, t_stat, hit))

print("\n" + "=" * 100)
print("SHORT on z < -THRESH (ensuring symmetry check)")
print(f"{'thresh':>7} {'horizon':>8} {'n':>6} {'asset_mean':>11} {'short_net':>10} {'std':>7} {'t_short':>7} {'hit%':>6} verdict")
print("=" * 100)

for z_thresh in [1.0, 1.5, 2.0, 2.5, 3.0]:
    for h_col in ["r_8h", "r_24h", "r_48h", "r_72h"]:
        mask = (ev["z"] < -z_thresh) & ev[h_col].notna()
        rets = ev.loc[mask, h_col].values
        if len(rets) < 30:
            continue
        n = len(rets)
        mean_g = rets.mean()
        # SHORT return = -asset_return, so short profits if asset DROPS
        short_gross = -mean_g
        mean_n = short_gross - TAKER_RT
        std = rets.std(ddof=1)
        std = max(std, 1e-9)
        t_stat = (mean_n * math.sqrt(n)) / std
        hit_short = (rets < -TAKER_RT).mean()
        flag = "  <<SHIP" if (n >= 100 and mean_n > 0.005 and t_stat > 2.0 and hit_short > 0.55) else ""
        print(f"{z_thresh:>6.1f} {h_col.replace('r_',''):>7} {n:>6} "
              f"{mean_g*100:>+9.2f}% {mean_n*100:>+8.2f}% {std*100:>6.2f}% "
              f"{t_stat:>6.2f} {hit_short*100:>5.1f}%{flag}")

# Per-year breakdown for top cell
print("\n" + "=" * 100)
print("TEMPORAL DECAY CHECK: top cell (z > +2, LONG 72h) by year")
print("=" * 100)

# Parse ts column to year
ev["year"] = (pd.to_datetime(ev["ts"], unit="ms")).dt.year
top_mask = (ev["z"] > 2.0) & ev["r_72h"].notna()
for year in sorted(ev["year"].unique()):
    sub = ev[top_mask & (ev["year"] == year)]
    if len(sub) < 20:
        continue
    rets = sub["r_72h"].values
    n = len(rets)
    mean_g = rets.mean()
    mean_n = mean_g - TAKER_RT
    std = rets.std(ddof=1)
    std = max(std, 1e-9)
    t_stat = (mean_n * math.sqrt(n)) / std
    hit = (rets > TAKER_RT).mean()
    print(f"  {year}: n={n:>4} mean_net={mean_n*100:+.2f}% t={t_stat:+5.2f} hit={hit*100:.1f}%")

# Per-asset breakdown for top cell
print("\n" + "=" * 100)
print("PER-ASSET BREAKDOWN: top cell (z > +2, LONG 72h)")
print("=" * 100)
rows = []
for sym in sorted(ev.loc[top_mask, "sym"].unique()):
    sub = ev[top_mask & (ev["sym"] == sym)]
    if len(sub) < 10:
        continue
    rets = sub["r_72h"].values
    n = len(rets)
    mean_g = rets.mean()
    mean_n = mean_g - TAKER_RT
    std = rets.std(ddof=1) if n > 1 else 1e-9
    std = max(std, 1e-9)
    t_stat = (mean_n * math.sqrt(n)) / std
    hit = (rets > TAKER_RT).mean()
    rows.append((sym, n, mean_n, t_stat, hit))
rows.sort(key=lambda r: r[3], reverse=True)  # sort by t
for sym, n, mean_n, t, hit in rows[:15]:
    print(f"  {sym:<12} n={n:>4} mean_net={mean_n*100:+6.2f}% t={t:+5.2f} hit={hit*100:.1f}%")

print("\n" + "=" * 100)
if ship_long:
    print(f"[SHIP] {len(ship_long)} LONG-on-high-funding cells pass strict criteria:")
    for z, h, n, m, t, hit in ship_long:
        horizon = h.replace("r_", "")
        print(f"  z > +{z} LONG {horizon}: n={n} mean_net={m*100:+.2f}% t={t:.2f} hit={hit*100:.1f}%")
else:
    print("[CONCEDE] no cells pass strict criteria")
