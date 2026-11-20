"""Regenerate asym_breakout and asym_vol_expansion seeds with CONSERVATIVE
exit order (low-first), replacing optimistic (trail-first) seeds.

Also saves U100 variants at the same time.

Overwrites:
    logs/paper_trader_v2/seeds/pt_asym_breakout/
    logs/paper_trader_v2/seeds/pt_asym_breakout_U50/
    logs/paper_trader_v2/seeds/pt_asym_breakout_U100/
    logs/paper_trader_v2/seeds/pt_asym_vol_expansion/
    logs/paper_trader_v2/seeds/pt_asym_vol_expansion_U50/
    logs/paper_trader_v2/seeds/pt_asym_vol_expansion_U100/
"""
from __future__ import annotations

import json
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

MAKER_RT = 0.08
CAPITAL = 10000.0
TEST_START = "2025-01-01"
TEST_END = "2026-04-22"


def build_panel(asset_set):
    rows = []
    for fp in sorted(glob.glob(str(DATA / "*_chimera.parquet"))):
        asset = Path(fp).stem.replace("usdt_v50_chimera", "").upper()
        if asset not in asset_set: continue
        df = pl.read_parquet(fp, columns=["timestamp", "close", "high", "low", "open"]).to_pandas()
        if len(df) < 1000: continue
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        d = df.groupby("date").agg({"close": "last", "high": "max", "low": "min", "open": "first"}).reset_index()
        d["ret_d"] = d["close"].pct_change()
        d["vol_1d"] = d["ret_d"].abs()
        d["vol_z30"] = (d["vol_1d"] - d["ret_d"].rolling(30).std()) / (d["ret_d"].rolling(30).std() + 1e-9)
        d["high_prev_10d"] = d["high"].shift(1).rolling(10).max()
        d["asset"] = asset
        rows.append(d)
    return pd.concat(rows, ignore_index=True).dropna(subset=["ret_d"]).assign(
        date=lambda x: pd.to_datetime(x["date"])).sort_values(["asset", "date"]).reset_index(drop=True)


def sim_conservative(panel, entry_fn, max_concurrent=10, pct_per_trade=0.10,
                     reentry_cooldown=2):
    """Conservative exit order: check low vs PREVIOUS day's stop first, then update trail."""
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
        close_map = dict(zip(lkup["asset"], lkup["close"]))
        low_map = dict(zip(lkup["asset"], lkup["low"]))
        high_map = dict(zip(lkup["asset"], lkup["high"]))

        # --- Exits (LOW FIRST: check stop that existed at start of day) ---
        closed = []
        for asset, pos in list(positions.items()):
            if asset not in close_map: continue
            low = low_map[asset]; high = high_map[asset]; close = close_map[asset]
            exit_price, reason = None, None
            # 1) Check low vs CURRENT stop (which was set yesterday end)
            if low <= pos["stop"]:
                exit_price = pos["stop"]; reason = "stop"
            elif (d - pos["entry_date"]).days >= pos["max_hold"]:
                exit_price = close; reason = "time"
            # 2) If survived, update trail for TOMORROW
            if exit_price is None:
                if high > pos["peak"]:
                    pos["peak"] = high
                    if pos.get("trail_pct"):
                        new_trail = pos["peak"] * (1 - pos["trail_pct"])
                        if new_trail > pos["stop"]:
                            pos["stop"] = new_trail
            else:
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
                closed.append(asset); last_exit[asset] = d
        for a in closed: del positions[a]

        # --- Entries ---
        if len(positions) < max_concurrent:
            cands = []
            for _, row in lkup.iterrows():
                asset = row["asset"]
                if asset in positions: continue
                if asset in last_exit and (d - last_exit[asset]).days < reentry_cooldown:
                    continue
                spec = entry_fn(row)
                if spec is None: continue
                cands.append((spec.get("strength", 0), asset, row["close"], spec))
            cands.sort(reverse=True)
            for strength, asset, close, spec in cands[:max_concurrent - len(positions)]:
                size = cash * pct_per_trade
                if size > cash: break
                cash -= size + size * (MAKER_RT / 200.0)
                positions[asset] = {
                    "entry_date": d, "entry_price": close, "peak": close,
                    "stop": close * (1 - spec.get("init_stop", 0.02)),
                    "trail_pct": spec.get("trail_pct", 0),
                    "max_hold": spec.get("max_hold", 15), "size": size,
                }

        pos_val = sum(p["size"] * (close_map.get(a, p["entry_price"]) / p["entry_price"])
                      for a, p in positions.items() if a in close_map)
        daily_eq.append({"date": d, "equity": cash + pos_val, "n_positions": len(positions)})

    return pd.DataFrame(daily_eq), pd.DataFrame(trades)


def breakout_entry(row):
    bh = row.get("high_prev_10d")
    if pd.isna(bh) or row["close"] <= bh: return None
    return {"init_stop": 0.02, "trail_pct": 0.05, "max_hold": 15,
            "strength": (row["close"] - bh) / bh}


def vol_exp_entry(row):
    if pd.isna(row.get("vol_z30")) or pd.isna(row.get("ret_d")): return None
    if row["vol_z30"] < 2.0: return None
    if row["ret_d"] < -0.02: return None
    return {"init_stop": 0.02, "trail_pct": 0.05, "max_hold": 5,
            "strength": row["vol_z30"]}


def save_seed(eq_df, tr_df, seed_name):
    d = SEEDS_DIR / seed_name
    d.mkdir(parents=True, exist_ok=True)
    snap = eq_df.rename(columns={"equity": "total_equity"}).copy()
    snap["bar_idx"] = np.arange(len(snap))
    snap["bar_ts"] = [int(pd.Timestamp(x).timestamp() * 1000) for x in snap["date"]]
    snap["swing_equity"] = snap["total_equity"]
    snap["short_equity"] = 0.0
    snap["total_ret_pct"] = (snap["total_equity"] / CAPITAL - 1) * 100
    snap["swing_ret_pct"] = snap["total_ret_pct"]
    snap["short_ret_pct"] = 0.0
    snap["swing_open_positions"] = snap["n_positions"].astype(int)
    cols = ["date", "bar_idx", "bar_ts", "total_equity", "swing_equity", "short_equity",
            "total_ret_pct", "swing_ret_pct", "short_ret_pct", "swing_open_positions"]
    snap[cols].to_csv(d / "daily_snapshot.csv", index=False)
    if len(tr_df): tr_df.to_csv(d / "trade_log.csv", index=False)


def stats(eq_df, tr_df, label=""):
    eq = eq_df["equity"].values
    total = (eq[-1] / CAPITAL - 1) * 100
    days = (eq_df["date"].iloc[-1] - eq_df["date"].iloc[0]).days or 1
    cagr = ((eq[-1] / CAPITAL) ** (365 / days) - 1) * 100
    dr = np.diff(eq) / eq[:-1]
    sharpe = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
    dd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min() * 100
    out = {"label": label, "n_days": len(eq_df), "cagr_pct": cagr, "sharpe": sharpe,
           "max_dd_pct": dd, "total_ret_pct": total}
    if len(tr_df):
        r = tr_df["net_ret_pct"].values
        wins = r[r > 0]; losses = r[r <= 0]
        hit = len(wins) / len(r)
        asym = wins.mean() / abs(losses.mean()) if len(wins) and len(losses) else float("inf")
        out.update({"n_trades": len(r), "hit_rate": hit,
                    "asymmetry_ratio": asym,
                    "mean_win_pct": wins.mean() if len(wins) else 0,
                    "mean_loss_pct": losses.mean() if len(losses) else 0})
    return out


U50 = set(UNIVERSE_50_LIQUID)
U100 = {Path(f).stem.replace("usdt_v50_chimera", "").upper()
        for f in glob.glob(str(DATA / "*_chimera.parquet"))}

print(f"[info] U50={len(U50)}, U100={len(U100)}")
print("[panel U50] building...")
t0 = time.time()
panel_u50 = build_panel(U50)
print(f"[panel U50] {panel_u50.shape} in {time.time()-t0:.1f}s")
print("[panel U100] building...")
t0 = time.time()
panel_u100 = build_panel(U100)
print(f"[panel U100] {panel_u100.shape} in {time.time()-t0:.1f}s")

print("\n" + "=" * 90)
print("REGEN WITH CONSERVATIVE (LOW-FIRST) EXIT ORDER")
print("=" * 90)
print(f"{'seed':<42} {'days':>4} {'CAGR%':>7} {'Sh':>5} {'DD%':>6} {'n_tr':>4} {'asym':>5}")
print("-" * 90)

results = {}
for strat, entry_fn, panels in [
    ("asym_breakout", breakout_entry, [("U50", panel_u50), ("U100", panel_u100)]),
    ("asym_vol_expansion", vol_exp_entry, [("U50", panel_u50), ("U100", panel_u100)]),
]:
    for univ, panel in panels:
        eq_df, tr_df = sim_conservative(panel, entry_fn)
        m = stats(eq_df, tr_df, label=f"{strat}_{univ}")
        seed = f"pt_{strat}_{univ}"
        save_seed(eq_df, tr_df, seed)
        # Also save _U50 to default path (no suffix) so aggregator uses it
        if univ == "U50":
            save_seed(eq_df, tr_df, f"pt_{strat}")
        results[f"{strat}_{univ}"] = m
        print(f"{seed:<42} {m['n_days']:>4} {m['cagr_pct']:>+6.2f} {m['sharpe']:>+4.2f} "
              f"{m['max_dd_pct']:>+5.2f} {m.get('n_trades', 0):>4} "
              f"{m.get('asymmetry_ratio', 0):>4.2f}")

from datetime import datetime, timezone
out_dir = ROOT / "logs" / "deployment" / str(datetime.now(timezone.utc).date())
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "regen_asym_conservative.json"
with open(out, "w") as f:
    json.dump({
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "exit_order": "low_first (conservative)",
        "results": results,
    }, f, indent=2, default=str)
print(f"\n[saved] {out}")
