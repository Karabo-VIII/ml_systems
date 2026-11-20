"""Persist the Family B best variant (volZ2.0 s=2% t=5% h=5) as a seed."""
from __future__ import annotations

import pickle
import sys
import time
import warnings
from pathlib import Path

import glob
import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
SEEDS_DIR = ROOT / "logs" / "paper_trader_v2" / "seeds"

sys.path.insert(0, str(ROOT / "src" / "strategy"))
from universe import UNIVERSE_50_LIQUID
UNIVERSE = set(UNIVERSE_50_LIQUID)

MAKER_RT = 0.08
CAPITAL = 10000.0
TEST_START = "2025-01-01"
TEST_END = "2026-04-22"

print("[panel] building...")
t0 = time.time()
all_fps = sorted(glob.glob(str(DATA / "*_chimera.parquet")))
rows = []
for fp in all_fps:
    asset = Path(fp).stem.replace("usdt_v50_chimera", "").upper()
    if asset not in UNIVERSE:
        continue
    df = pl.read_parquet(fp, columns=["timestamp", "close", "high", "low", "open"]).to_pandas()
    if len(df) < 1000:
        continue
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
    d = df.groupby("date").agg({"close": "last", "high": "max", "low": "min", "open": "first"}).reset_index()
    d["ret_d"] = d["close"].pct_change()
    d["vol_1d"] = d["ret_d"].abs()
    d["vol_z30"] = (d["vol_1d"] - d["ret_d"].rolling(30).std()) / (d["ret_d"].rolling(30).std() + 1e-9)
    d["asset"] = asset
    rows.append(d)
panel = pd.concat(rows, ignore_index=True).dropna(subset=["ret_d"])
panel["date"] = pd.to_datetime(panel["date"])
print(f"[panel] {panel.shape} in {time.time()-t0:.1f}s")

# Family B entry: vol_z30 > 2.0, ret_d >= -0.02
# Stop: -2% init
# Trail: 5% from peak
# Max hold: 5d
# Max concurrent: 10, 10% per trade

def family_B_sim(panel, vz_thresh=2.0, stop_pct=0.02, trail_pct=0.05, max_hold=5,
                 max_concurrent=10, pct_per_trade=0.10, reentry_cooldown=2):
    test = panel[(panel["date"] >= TEST_START) & (panel["date"] <= TEST_END)]
    all_dates = sorted(test["date"].unique())
    lookups = {d: grp for d, grp in test.groupby("date")}

    cash = CAPITAL
    positions = {}
    last_exit = {}
    trades = []
    daily_eq = []

    for d in all_dates:
        lkup = lookups.get(d)
        if lkup is None: continue

        # Exits
        close_map = dict(zip(lkup["asset"], lkup["close"]))
        low_map = dict(zip(lkup["asset"], lkup["low"]))
        high_map = dict(zip(lkup["asset"], lkup["high"]))
        closed = []
        for asset, pos in list(positions.items()):
            if asset not in close_map: continue
            # Update peak / trail
            if high_map[asset] > pos["peak"]:
                pos["peak"] = high_map[asset]
                new_trail = pos["peak"] * (1 - trail_pct)
                if new_trail > pos["stop"]:
                    pos["stop"] = new_trail
            exit_price = None; reason = None
            if low_map[asset] <= pos["stop"]:
                exit_price = pos["stop"]; reason = "stop"
            elif (d - pos["entry_date"]).days >= max_hold:
                exit_price = close_map[asset]; reason = "time"
            if exit_price is not None:
                size = pos["size"]
                pnl = size * (exit_price / pos["entry_price"] - 1)
                pnl -= size * (MAKER_RT / 200.0)
                cash += size + pnl
                net = (exit_price / pos["entry_price"] - 1) * 100 - MAKER_RT
                trades.append({
                    "asset": asset, "entry_date": pos["entry_date"], "exit_date": d,
                    "hold_days": (d - pos["entry_date"]).days,
                    "entry_price": pos["entry_price"], "exit_price": exit_price,
                    "net_ret_pct": net, "exit_reason": reason,
                })
                closed.append(asset)
                last_exit[asset] = d
        for a in closed: del positions[a]

        # Entries
        if len(positions) < max_concurrent:
            cands = lkup[(lkup["vol_z30"] >= vz_thresh) & (lkup["ret_d"] >= -0.02)].copy()
            cands = cands.sort_values("vol_z30", ascending=False)
            for _, row in cands.iterrows():
                asset = row["asset"]
                if asset in positions: continue
                if asset in last_exit and (d - last_exit[asset]).days < reentry_cooldown:
                    continue
                entry = row["close"]
                size = cash * pct_per_trade
                if size > cash: break
                cash -= size + size * (MAKER_RT / 200.0)
                positions[asset] = {
                    "entry_date": d, "entry_price": entry,
                    "stop": entry * (1 - stop_pct), "peak": entry, "size": size,
                }
                if len(positions) >= max_concurrent: break

        pos_val = sum(p["size"] * (close_map.get(a, p["entry_price"]) / p["entry_price"])
                      for a, p in positions.items() if a in close_map)
        daily_eq.append({"date": d, "equity": cash + pos_val, "n_positions": len(positions)})

    return pd.DataFrame(daily_eq), pd.DataFrame(trades)


print("\n[family B] running best variant (volZ2.0 s=2% t=5% h=5)...")
eq_df, tr_df = family_B_sim(panel, vz_thresh=2.0, stop_pct=0.02, trail_pct=0.05, max_hold=5)

# Save
seed_dir = SEEDS_DIR / "pt_asym_vol_expansion"
seed_dir.mkdir(parents=True, exist_ok=True)
snap = eq_df.rename(columns={"equity": "total_equity"}).copy()
snap["bar_idx"] = np.arange(len(snap))
snap["bar_ts"] = [int(pd.Timestamp(d).timestamp() * 1000) for d in snap["date"]]
snap["swing_equity"] = snap["total_equity"]
snap["short_equity"] = 0.0
snap["total_ret_pct"] = (snap["total_equity"] / CAPITAL - 1) * 100
snap["swing_ret_pct"] = snap["total_ret_pct"]
snap["short_ret_pct"] = 0.0
snap["swing_open_positions"] = snap["n_positions"].astype(int)
snap[["date", "bar_idx", "bar_ts", "total_equity", "swing_equity", "short_equity",
       "total_ret_pct", "swing_ret_pct", "short_ret_pct", "swing_open_positions"]
     ].to_csv(seed_dir / "daily_snapshot.csv", index=False)
tr_df.to_csv(seed_dir / "trade_log.csv", index=False)

# Summary
r = tr_df["net_ret_pct"].values
wins = r[r > 0]; losses = r[r <= 0]
hit = len(wins) / len(r) if len(r) else 0
asym = wins.mean() / abs(losses.mean()) if len(wins) and len(losses) else float("inf")
kelly_g = (hit * np.log1p(wins.mean()/100 if len(wins) else 0)
           + (1 - hit) * np.log1p(losses.mean()/100 if len(losses) else 0))
eq = eq_df["equity"].values
total = (eq[-1] / CAPITAL - 1) * 100
days = (eq_df["date"].iloc[-1] - eq_df["date"].iloc[0]).days or 1
cagr = ((eq[-1] / CAPITAL) ** (365 / days) - 1) * 100
dr = np.diff(eq) / eq[:-1]
sharpe = dr.mean() / dr.std() * np.sqrt(365)
dd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min() * 100
print(f"\n[result] n_trades={len(r)}  hit={hit*100:.0f}%  "
      f"CAGR={cagr:+.2f}%  Sharpe={sharpe:+.2f}  DD={dd:+.2f}%")
print(f"         asym={asym:.2f}x  Kelly/trade={kelly_g:+.4f}")
print(f"[saved] {seed_dir}/daily_snapshot.csv + trade_log.csv")
