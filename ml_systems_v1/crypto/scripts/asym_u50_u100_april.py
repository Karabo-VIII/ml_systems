"""Seed new asymmetric strategies on U50 and U100 (all-chimera) universes.

Runs full-history sim 2025-01-01 -> 2026-04-22 then slices April 1-22 window.
Reports both full and window metrics for:
    (1) asym_breakout   (Family A, portfolio)
    (2) asym_vol_expansion  (Family B)
    (3) xsec_K10_10_FULL_dneut (reference; already has U50 snapshot)

Both universes:
    U50  = UNIVERSE_50_LIQUID (50 audited)
    U100 = all-chimera (~53; effective U100 until more fetches land)

Snapshots saved as pt_<strat>_U{50,100}/daily_snapshot.csv for aggregator.
"""
from __future__ import annotations

import json
import subprocess
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
WIN_START = "2026-04-01"
WIN_END = "2026-04-22"

U50 = set(UNIVERSE_50_LIQUID)
U100_ALL = {Path(f).stem.replace("usdt_v50_chimera", "").upper()
            for f in glob.glob(str(DATA / "*_chimera.parquet"))}
print(f"[info] U50={len(U50)}, U100(all-chimera)={len(U100_ALL)}")
print(f"[info] delta (in U100 not U50): {sorted(U100_ALL - U50)}")


def build_panel(asset_set):
    rows = []
    for fp in sorted(glob.glob(str(DATA / "*_chimera.parquet"))):
        asset = Path(fp).stem.replace("usdt_v50_chimera", "").upper()
        if asset not in asset_set:
            continue
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


# =============================================================================
# Generic portfolio runner (entry_fn + exit_fn)
# =============================================================================
def run_portfolio(panel, entry_fn, exit_fn, max_concurrent=10, pct_per_trade=0.10,
                  reentry_cooldown=2):
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

        # Exits
        closed = []
        for asset, pos in list(positions.items()):
            if asset not in close_map: continue
            if high_map[asset] > pos["peak"]:
                pos["peak"] = high_map[asset]
                if pos.get("trail_pct"):
                    new_trail = pos["peak"] * (1 - pos["trail_pct"])
                    if new_trail > pos["stop"]:
                        pos["stop"] = new_trail
            exit_price, reason = exit_fn(pos, {"close": close_map[asset], "low": low_map[asset],
                                                 "high": high_map[asset]}, d)
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
                closed.append(asset); last_exit[asset] = d
        for a in closed: del positions[a]

        # Entries
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
                    "stop": spec["stop"], "trail_pct": spec.get("trail_pct"),
                    "max_hold": spec.get("max_hold", 30), "size": size,
                }

        pos_val = sum(p["size"] * (close_map.get(a, p["entry_price"]) / p["entry_price"])
                      for a, p in positions.items() if a in close_map)
        daily_eq.append({"date": d, "equity": cash + pos_val, "n_positions": len(positions)})

    return pd.DataFrame(daily_eq), pd.DataFrame(trades)


def trail_exit(pos, row, today):
    if row["low"] <= pos["stop"]:
        return pos["stop"], "stop"
    if (today - pos["entry_date"]).days >= pos["max_hold"]:
        return row["close"], "time"
    return None, None


def breakout_entry(row):
    bh = row["high_prev_10d"]
    if pd.isna(bh) or row["close"] <= bh:
        return None
    entry = row["close"]
    return {"stop": entry * (1 - 0.02), "trail_pct": 0.05, "max_hold": 15,
            "strength": (entry - bh) / bh}


def vol_exp_entry(row):
    if pd.isna(row.get("vol_z30")) or pd.isna(row.get("ret_d")):
        return None
    if row["vol_z30"] < 2.0: return None
    if row["ret_d"] < -0.02: return None
    entry = row["close"]
    return {"stop": entry * (1 - 0.02), "trail_pct": 0.05, "max_hold": 5,
            "strength": row["vol_z30"]}


def metrics(eq_df, tr_df, win_start=None, win_end=None):
    if len(eq_df) < 2:
        return {"status": "insufficient"}
    if win_start:
        eq_sub = eq_df[(eq_df["date"] >= win_start) & (eq_df["date"] <= win_end)]
        tr_sub = tr_df[(pd.to_datetime(tr_df["entry_date"]) >= win_start) &
                       (pd.to_datetime(tr_df["entry_date"]) <= win_end)] if len(tr_df) else tr_df
    else:
        eq_sub = eq_df; tr_sub = tr_df
    if len(eq_sub) < 2:
        return {"status": "insufficient_window"}
    eq = eq_sub["equity"].values
    total = (eq[-1] / eq[0] - 1) * 100
    days = (eq_sub["date"].iloc[-1] - eq_sub["date"].iloc[0]).days or 1
    cagr = ((eq[-1] / eq[0]) ** (365 / days) - 1) * 100
    dr = np.diff(eq) / eq[:-1]
    sharpe = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
    dd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min() * 100
    result = {"n_days": len(eq_sub), "cagr_pct": cagr, "sharpe": sharpe, "max_dd_pct": dd,
              "total_ret_pct": total}
    if len(tr_sub) > 0:
        r = tr_sub["net_ret_pct"].values
        wins = r[r > 0]; losses = r[r <= 0]
        hit = len(wins) / len(r)
        asym = wins.mean() / abs(losses.mean()) if len(wins) and len(losses) else float("inf")
        kelly = (hit * np.log1p(wins.mean()/100 if len(wins) else 0) +
                 (1 - hit) * np.log1p(losses.mean()/100 if len(losses) else 0))
        result.update({"n_trades": len(r), "hit_rate": hit,
                       "mean_win_pct": wins.mean() if len(wins) else 0,
                       "mean_loss_pct": losses.mean() if len(losses) else 0,
                       "asymmetry_ratio": asym, "kelly_log_g_per_trade": kelly})
    return result


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


# =============================================================================
# Run all 4 combinations
# =============================================================================
print("\n[panel U50] building...")
t0 = time.time()
panel_u50 = build_panel(U50)
print(f"[panel U50] {panel_u50.shape} in {time.time()-t0:.1f}s")

print("[panel U100] building...")
t0 = time.time()
panel_u100 = build_panel(U100_ALL)
print(f"[panel U100] {panel_u100.shape} in {time.time()-t0:.1f}s")

results = {}
print("\n" + "=" * 100)
print(f"{'strat':<25} {'univ':<5} {'full days':>9} {'full CAGR':>9} {'full Sh':>7} {'full DD':>7} "
      f"{'win ret':>7} {'win Sh':>6} {'win DD':>6} {'win n_tr':>8} {'asym':>5}")
print("-" * 100)

configs = [
    ("asym_breakout", breakout_entry, trail_exit, {"max_concurrent": 10, "pct_per_trade": 0.10}),
    ("asym_vol_expansion", vol_exp_entry, trail_exit, {"max_concurrent": 10, "pct_per_trade": 0.10}),
]

for strat_name, entry_fn, exit_fn, kwargs in configs:
    for univ_label, panel in [("U50", panel_u50), ("U100", panel_u100)]:
        eq_df, tr_df = run_portfolio(panel, entry_fn, exit_fn, **kwargs)
        m_full = metrics(eq_df, tr_df)
        m_win = metrics(eq_df, tr_df, WIN_START, WIN_END)
        seed = f"pt_{strat_name}_{univ_label}"
        save_seed(eq_df, tr_df, seed)
        key = f"{strat_name}_{univ_label}"
        results[key] = {"full": m_full, "window_april": m_win, "seed": seed}
        print(f"{strat_name:<25} {univ_label:<5} {m_full['n_days']:>9} "
              f"{m_full['cagr_pct']:>+8.2f} {m_full['sharpe']:>+6.2f} {m_full['max_dd_pct']:>+6.2f} "
              f"{m_win.get('total_ret_pct', 0):>+6.2f} "
              f"{m_win.get('sharpe', 0):>+5.2f} "
              f"{m_win.get('max_dd_pct', 0):>+5.2f} "
              f"{m_win.get('n_trades', 0):>8} "
              f"{m_full.get('asymmetry_ratio', 0):>4.2f}")

# Add xsec K10+10 reference (both universes -- already have U50 snapshot)
print("\n" + "=" * 100)
print("REFERENCE: xsec_K10_10_FULL_dneut (already-shipped; April window slice only)")
print("-" * 100)
for seed_name, univ in [("pt_xsec_K10_10_FULL_dneut_U50", "U50"),
                         ("pt_xsec_K10_10_FULL_dneut_U100", "U100")]:
    snap_path = SEEDS_DIR / seed_name / "daily_snapshot.csv"
    if not snap_path.exists():
        print(f"  {seed_name:<40} {univ:<5} MISSING")
        continue
    df = pd.read_csv(snap_path)
    df["date"] = pd.to_datetime(df["date"])
    sub = df[(df["date"] >= WIN_START) & (df["date"] <= WIN_END)]
    if len(sub) < 2:
        print(f"  {seed_name:<40} {univ:<5} INSUFFICIENT WINDOW")
        continue
    eq = sub["total_equity"].values
    total = (eq[-1] / eq[0] - 1) * 100
    dr = np.diff(eq) / eq[:-1]
    sharpe = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
    dd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min() * 100
    print(f"  {seed_name:<40} {univ:<5} n={len(sub):>3} "
          f"window_ret={total:>+6.2f}%  Sh={sharpe:>+5.2f}  DD={dd:>+5.2f}%")
    results[f"xsec_K10_10_FULL_dneut_{univ}"] = {
        "full": {"status": "pre-existing seed"},
        "window_april": {"n_days": len(sub), "total_ret_pct": total,
                          "sharpe": sharpe, "max_dd_pct": dd},
    }

# Save report
from datetime import datetime, timezone
out_dir = ROOT / "logs" / "deployment" / str(datetime.now(timezone.utc).date())
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "asym_u50_u100_april.json"
with open(out, "w") as f:
    json.dump({
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "u50_count": len(U50),
        "u100_count": len(U100_ALL),
        "u100_extras": sorted(U100_ALL - U50),
        "window_start": WIN_START, "window_end": WIN_END,
        "results": results,
    }, f, indent=2, default=str)
print(f"\n[saved] {out}")
