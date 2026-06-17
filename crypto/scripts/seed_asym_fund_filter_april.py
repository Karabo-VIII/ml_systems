"""Persist the funding-filtered breakout seed + rerun April window for all 3
asymmetric strategies (conservative exit order).

Produces:
    pt_asym_breakout_fund_filter_0/daily_snapshot.csv  (new seed)
    April 1-22 metrics for all 3 asym strats (conservative)
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
WIN_START = "2026-04-01"
WIN_END = "2026-04-22"

U50 = set(UNIVERSE_50_LIQUID)


def build_panel_with_funding(asset_set):
    rows = []
    for fp in sorted(glob.glob(str(DATA / "*_chimera.parquet"))):
        asset = Path(fp).stem.replace("usdt_v50_chimera", "").upper()
        if asset not in asset_set: continue
        cols = ["timestamp", "close", "high", "low", "open"]
        probe = pl.read_parquet(fp, n_rows=1).columns
        if "norm_funding" in probe:
            cols.append("norm_funding")
        df = pl.read_parquet(fp, columns=cols).to_pandas()
        if len(df) < 1000: continue
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        aggs = {"close": "last", "high": "max", "low": "min", "open": "first"}
        if "norm_funding" in df.columns:
            aggs["norm_funding"] = "last"
        d = df.groupby("date").agg(aggs).reset_index()
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
    test = panel[(panel["date"] >= TEST_START) & (panel["date"] <= TEST_END)]
    all_dates = sorted(test["date"].unique())
    lookups = {d: grp for d, grp in test.groupby("date")}
    cash = CAPITAL; positions = {}; last_exit = {}
    trades = []; daily_eq = []

    for d in all_dates:
        lkup = lookups.get(d)
        if lkup is None: continue
        close_map = dict(zip(lkup["asset"], lkup["close"]))
        low_map = dict(zip(lkup["asset"], lkup["low"]))
        high_map = dict(zip(lkup["asset"], lkup["high"]))

        closed = []
        for asset, pos in list(positions.items()):
            if asset not in close_map: continue
            exit_price, reason = None, None
            if low_map[asset] <= pos["stop"]:
                exit_price = pos["stop"]; reason = "stop"
            elif (d - pos["entry_date"]).days >= pos["max_hold"]:
                exit_price = close_map[asset]; reason = "time"
            if exit_price is None:
                if high_map[asset] > pos["peak"]:
                    pos["peak"] = high_map[asset]
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
                trades.append({"asset": asset, "entry_date": pos["entry_date"], "exit_date": d,
                                "hold_days": (d - pos["entry_date"]).days,
                                "entry_price": pos["entry_price"], "exit_price": exit_price,
                                "net_ret_pct": net, "exit_reason": reason})
                closed.append(asset); last_exit[asset] = d
        for a in closed: del positions[a]

        if len(positions) < max_concurrent:
            cands = []
            for _, row in lkup.iterrows():
                asset = row["asset"]
                if asset in positions: continue
                if asset in last_exit and (d - last_exit[asset]).days < reentry_cooldown: continue
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


def breakout_fund_filter_entry(fund_max=0.0):
    def fn(row):
        bh = row.get("high_prev_10d")
        if pd.isna(bh) or row["close"] <= bh: return None
        fnd = row.get("norm_funding")
        if not pd.isna(fnd) and fnd > fund_max: return None
        return {"init_stop": 0.02, "trail_pct": 0.05, "max_hold": 15,
                "strength": (row["close"] - bh) / bh}
    return fn


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


def slice_metrics(eq_df, tr_df, start_d, end_d):
    sub_eq = eq_df[(eq_df["date"] >= start_d) & (eq_df["date"] <= end_d)].copy()
    sub_tr = tr_df[(pd.to_datetime(tr_df["entry_date"]) >= start_d) &
                   (pd.to_datetime(tr_df["entry_date"]) <= end_d)] if len(tr_df) else tr_df
    if len(sub_eq) < 2: return {"status": "insufficient"}
    eq = sub_eq["equity"].values
    total = (eq[-1] / eq[0] - 1) * 100
    dr = np.diff(eq) / eq[:-1]
    sharpe = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
    dd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min() * 100
    out = {"n_days": len(sub_eq), "total_ret_pct": total, "sharpe": sharpe, "max_dd_pct": dd}
    if len(sub_tr):
        r = sub_tr["net_ret_pct"].values
        out["n_trades"] = len(r)
    return out


def full_metrics(eq_df, tr_df):
    eq = eq_df["equity"].values
    total = (eq[-1] / CAPITAL - 1) * 100
    days = (eq_df["date"].iloc[-1] - eq_df["date"].iloc[0]).days or 1
    cagr = ((eq[-1] / CAPITAL) ** (365 / days) - 1) * 100
    dr = np.diff(eq) / eq[:-1]
    sharpe = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
    dd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min() * 100
    out = {"n_days": len(eq_df), "cagr_pct": cagr, "sharpe": sharpe,
           "max_dd_pct": dd, "total_ret_pct": total}
    if len(tr_df):
        r = tr_df["net_ret_pct"].values
        wins = r[r > 0]; losses = r[r <= 0]
        hit = len(wins) / len(r)
        asym = wins.mean() / abs(losses.mean()) if len(wins) and len(losses) else float("inf")
        out.update({"n_trades": len(r), "hit_rate": hit, "asymmetry_ratio": asym})
    return out


print("[panel] building w/ funding...")
t0 = time.time()
panel = build_panel_with_funding(U50)
print(f"[panel] {panel.shape} in {time.time()-t0:.1f}s")

# Seed the funding-filtered breakout
print("\n[gen] asym_breakout_fund_filter_0 (conservative exit)...")
eq_df, tr_df = sim_conservative(panel, breakout_fund_filter_entry(0.0))
save_seed(eq_df, tr_df, "pt_asym_breakout_fund_filter_0")
m_full = full_metrics(eq_df, tr_df)
m_win = slice_metrics(eq_df, tr_df, WIN_START, WIN_END)
print(f"  full: CAGR {m_full['cagr_pct']:+.2f}% Sh {m_full['sharpe']:+.2f} DD {m_full['max_dd_pct']:+.2f}% "
      f"n={m_full['n_trades']} hit={m_full['hit_rate']*100:.0f}% asym={m_full['asymmetry_ratio']:.2f}")
print(f"  April window: ret {m_win.get('total_ret_pct', 0):+.2f}% Sh {m_win.get('sharpe', 0):+.2f} "
      f"DD {m_win.get('max_dd_pct', 0):+.2f}% n={m_win.get('n_trades', 0)}")

# Read the already-regenerated conservative seeds + slice April windows
print("\n[april window] all 3 asymmetric strategies (conservative seeds):")
print(f"{'strategy':<40} {'full CAGR%':>10} {'full Sh':>7} {'full DD':>7} "
      f"{'win ret%':>8} {'win Sh':>6} {'win DD%':>7}")
print("-" * 95)

summary = {}
asym_seeds = [
    ("asym_breakout", "pt_asym_breakout"),
    ("asym_vol_expansion", "pt_asym_vol_expansion"),
    ("asym_breakout_fund_filter_0", "pt_asym_breakout_fund_filter_0"),
]
for name, seed in asym_seeds:
    fp = SEEDS_DIR / seed / "daily_snapshot.csv"
    df = pd.read_csv(fp)
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "total_equity"]].rename(columns={"total_equity": "equity"})
    # Full metrics
    eq = df["equity"].values
    total = (eq[-1] / CAPITAL - 1) * 100
    days = (df["date"].iloc[-1] - df["date"].iloc[0]).days or 1
    cagr = ((eq[-1] / CAPITAL) ** (365 / days) - 1) * 100
    dr = np.diff(eq) / eq[:-1]
    sharpe = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
    dd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min() * 100
    # April window
    sub = df[(df["date"] >= WIN_START) & (df["date"] <= WIN_END)]
    if len(sub) >= 2:
        se = sub["equity"].values
        win_total = (se[-1] / se[0] - 1) * 100
        wdr = np.diff(se) / se[:-1]
        win_sh = wdr.mean() / wdr.std() * np.sqrt(365) if wdr.std() > 0 else 0
        win_dd = ((se - np.maximum.accumulate(se)) / np.maximum.accumulate(se)).min() * 100
    else:
        win_total = win_sh = win_dd = 0
    summary[name] = {"full": {"cagr_pct": cagr, "sharpe": sharpe, "max_dd_pct": dd},
                     "april_window": {"total_ret_pct": win_total, "sharpe": win_sh,
                                      "max_dd_pct": win_dd}}
    print(f"{name:<40} {cagr:>+9.2f} {sharpe:>+6.2f} {dd:>+6.2f} "
          f"{win_total:>+7.2f} {win_sh:>+5.2f} {win_dd:>+6.2f}")

# Save
from datetime import datetime, timezone
out_dir = ROOT / "logs" / "deployment" / str(datetime.now(timezone.utc).date())
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "asym_april_canonical.json"
with open(out, "w") as f:
    json.dump({
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "exit_order": "low_first (conservative, corrected)",
        "window": {"start": WIN_START, "end": WIN_END},
        "summary": summary,
    }, f, indent=2, default=str)
print(f"\n[saved] {out}")
