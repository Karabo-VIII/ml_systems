"""Round 4: remaining items after paranoid validation.

All experiments use CONSERVATIVE (low-first) exit order.

Items:
    B1. 4-sleeve blend using U100 variants (all-chimera panel)
    B2. Vol-expansion CAPITULATION bounce (SPOT-deployable: buy-the-dip after vol spike)
    B3. Funding-aware filter (avoid longs when funding is extreme-positive)
    B5. Leverage study: what if we size 2x/3x/5x the 4-sleeve blend?
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
RAW = ROOT / "data" / "raw"
SEEDS_DIR = ROOT / "logs" / "paper_trader_v2" / "seeds"

sys.path.insert(0, str(ROOT / "src" / "strategy"))
from universe import UNIVERSE_50_LIQUID

MAKER_RT = 0.08
CAPITAL = 10000.0
TEST_START = "2025-01-01"
TEST_END = "2026-04-22"

U50 = set(UNIVERSE_50_LIQUID)
U100 = {Path(f).stem.replace("usdt_v50_chimera", "").upper()
        for f in glob.glob(str(DATA / "*_chimera.parquet"))}


def build_panel(asset_set, include_funding=False):
    rows = []
    for fp in sorted(glob.glob(str(DATA / "*_chimera.parquet"))):
        asset = Path(fp).stem.replace("usdt_v50_chimera", "").upper()
        if asset not in asset_set: continue
        cols = ["timestamp", "close", "high", "low", "open"]
        if include_funding:
            df_probe = pl.read_parquet(fp, n_rows=1).columns
            if "norm_funding" in df_probe:
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
        d["low_3d"] = d["low"].rolling(3).min()
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
            elif pos.get("target") and high_map[asset] >= pos["target"]:
                exit_price = pos["target"]; reason = "target"
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
                    "target": spec.get("target"),
                    "trail_pct": spec.get("trail_pct", 0),
                    "max_hold": spec.get("max_hold", 15), "size": size,
                }

        pos_val = sum(p["size"] * (close_map.get(a, p["entry_price"]) / p["entry_price"])
                      for a, p in positions.items() if a in close_map)
        daily_eq.append({"date": d, "equity": cash + pos_val, "n_positions": len(positions)})

    return pd.DataFrame(daily_eq), pd.DataFrame(trades)


def stats(eq_df, tr_df, label=""):
    eq = eq_df["equity"].values
    if len(eq) < 2: return {"label": label, "status": "insufficient"}
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
        out.update({"n_trades": len(r), "hit_rate": hit, "asymmetry_ratio": asym,
                    "mean_win_pct": wins.mean() if len(wins) else 0,
                    "mean_loss_pct": losses.mean() if len(losses) else 0})
    return out


# =============================================================================
# B1: 4-sleeve blend using U100 variants for asym sleeves
# =============================================================================
print("\n" + "=" * 80)
print("B1. 4-SLEEVE BLEND USING U100 ASYM VARIANTS")
print("=" * 80)

# Temporarily register U100 variants in aggregator by reading snapshots directly
sleeves = [
    ("xsec_K10_10_FULL_dneut_U50", 0.40),
    ("frontier_dib_flow_both", 0.25),
    ("asym_breakout_U100", 0.20),
    ("asym_vol_expansion_U100", 0.15),
]
sleeve_snaps = {}
for profile, w in sleeves:
    fp = SEEDS_DIR / f"pt_{profile}" / "daily_snapshot.csv"
    if not fp.exists():
        print(f"  MISSING: {fp}")
        continue
    df = pd.read_csv(fp)
    df["date"] = pd.to_datetime(df["date"])
    df["eq_normalized"] = df["total_equity"] / df["total_equity"].iloc[0]
    sleeve_snaps[profile] = df

common = None
for p, df in sleeve_snaps.items():
    ds = set(df["date"].dt.date)
    common = ds if common is None else common & ds
common = sorted(common)
print(f"  common dates: {len(common)}  ({common[0]} to {common[-1]})")

total_w = sum(w for _, w in sleeves)
pe = np.zeros(len(common))
for profile, w in sleeves:
    df = sleeve_snaps[profile].copy()
    df["date_d"] = df["date"].dt.date
    df2 = pd.DataFrame({"date_d": common}).merge(df, on="date_d", how="left")
    df2["eq_normalized"] = df2["eq_normalized"].ffill().bfill()
    pe += (w / total_w) * df2["eq_normalized"].values
pe *= CAPITAL

total = (pe[-1] / CAPITAL - 1) * 100
days = (common[-1] - common[0]).days or 1
cagr = ((pe[-1] / CAPITAL) ** (365 / days) - 1) * 100
dr = np.diff(pe) / pe[:-1]
sharpe = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
dd = ((pe - np.maximum.accumulate(pe)) / np.maximum.accumulate(pe)).min() * 100
sortino_dr = dr[dr < 0]
sortino = dr.mean() / sortino_dr.std() * np.sqrt(365) if len(sortino_dr) > 0 and sortino_dr.std() > 0 else 0

b1_result = {"cagr_pct": cagr, "sharpe": sharpe, "sortino": sortino, "max_dd_pct": dd, "total_ret_pct": total,
              "n_days": len(common)}
print(f"\n  4-sleeve blend (U100 asym): CAGR {cagr:+.2f}%  Sharpe {sharpe:+.2f}  "
      f"Sortino {sortino:+.2f}  DD {dd:+.2f}%")
print(f"  Compare 4-sleeve blend (U50 asym):  CAGR +74.20%  Sharpe +6.17  Sortino +19.00  DD -2.01%")


# =============================================================================
# B2: Vol-expansion CAPITULATION bounce (buy dip after vol spike + down day)
# =============================================================================
print("\n" + "=" * 80)
print("B2. VOL-EXPANSION CAPITULATION BOUNCE (SPOT-deployable)")
print("=" * 80)


def capit_entry(vz_thresh=2.0, ret_max=-0.03):
    def fn(row):
        if pd.isna(row.get("vol_z30")) or pd.isna(row.get("ret_d")): return None
        if row["vol_z30"] < vz_thresh: return None
        if row["ret_d"] > ret_max: return None  # Must be down day (capitulation)
        return {"init_stop": 0.03, "trail_pct": 0.06, "max_hold": 5,
                "strength": row["vol_z30"]}
    return fn


panel_u50 = build_panel(U50)
print(f"[panel] {panel_u50.shape}")
print(f"{'config':<40} {'CAGR%':>7} {'Sh':>5} {'DD%':>6} {'n_tr':>4} {'hit%':>4} {'asym':>5}")
print("-" * 90)
b2_results = []
for vz, rmax, lbl in [(2.0, -0.03, "vz2.0_ret<-3"), (2.0, -0.05, "vz2.0_ret<-5"),
                       (2.5, -0.03, "vz2.5_ret<-3"), (2.5, -0.05, "vz2.5_ret<-5"),
                       (1.5, -0.03, "vz1.5_ret<-3")]:
    eq, tr = sim_conservative(panel_u50, capit_entry(vz, rmax))
    m = stats(eq, tr, label=lbl)
    b2_results.append(m)
    print(f"{lbl:<40} {m['cagr_pct']:>+6.2f} {m['sharpe']:>+4.2f} "
          f"{m['max_dd_pct']:>+5.2f} {m.get('n_trades', 0):>4} "
          f"{m.get('hit_rate', 0)*100:>3.0f} {m.get('asymmetry_ratio', 0):>4.2f}")


# =============================================================================
# B3: Funding-aware filter (avoid longs when funding > threshold)
# =============================================================================
print("\n" + "=" * 80)
print("B3. FUNDING-AWARE FILTER ON BREAKOUTS")
print("=" * 80)

panel_u50_fund = build_panel(U50, include_funding=True)
has_funding = "norm_funding" in panel_u50_fund.columns
print(f"[info] funding column available: {has_funding}")
if has_funding:
    fund_stats = panel_u50_fund["norm_funding"].dropna()
    print(f"[info] funding: n={len(fund_stats)}, mean={fund_stats.mean():.3f}, "
          f"std={fund_stats.std():.3f}, p95={fund_stats.quantile(0.95):.3f}")


def breakout_no_fund_filter(row):
    bh = row.get("high_prev_10d")
    if pd.isna(bh) or row["close"] <= bh: return None
    return {"init_stop": 0.02, "trail_pct": 0.05, "max_hold": 15,
            "strength": (row["close"] - bh) / bh}


def breakout_fund_filter(fund_max_z=1.0):
    def fn(row):
        bh = row.get("high_prev_10d")
        if pd.isna(bh) or row["close"] <= bh: return None
        fnd = row.get("norm_funding")
        # Skip if funding is extreme-positive (already crowded)
        if not pd.isna(fnd) and fnd > fund_max_z: return None
        return {"init_stop": 0.02, "trail_pct": 0.05, "max_hold": 15,
                "strength": (row["close"] - bh) / bh}
    return fn


print(f"{'variant':<40} {'CAGR%':>7} {'Sh':>5} {'DD%':>6} {'n_tr':>4} {'asym':>5}")
print("-" * 80)
b3_results = []
for lbl, fn in [
    ("breakout baseline (no fund filter)", breakout_no_fund_filter),
    ("breakout + fund<1.0 filter", breakout_fund_filter(1.0)),
    ("breakout + fund<0.5 filter", breakout_fund_filter(0.5)),
    ("breakout + fund<0.0 filter", breakout_fund_filter(0.0)),
    ("breakout + fund<-0.5 filter (contrarian)", breakout_fund_filter(-0.5)),
]:
    eq, tr = sim_conservative(panel_u50_fund, fn)
    m = stats(eq, tr, label=lbl)
    b3_results.append(m)
    print(f"{lbl:<40} {m['cagr_pct']:>+6.2f} {m['sharpe']:>+4.2f} "
          f"{m['max_dd_pct']:>+5.2f} {m.get('n_trades', 0):>4} "
          f"{m.get('asymmetry_ratio', 0):>4.2f}")


# =============================================================================
# B5: Leverage study on 4-sleeve blend
# =============================================================================
print("\n" + "=" * 80)
print("B5. LEVERAGE STUDY on recommended_4sleeve_alpha_stack")
print("=" * 80)
print("Model: daily_ret -> daily_ret * leverage (no leverage cost modeled).")
print("Note: real PERP funding/borrow cost would shave 10-20% of leveraged CAGR.")

# Read the current daily portfolio equity curve
agg_daily_path = ROOT / "logs" / "portfolio_aggregator" / "recommended_4sleeve_alpha_stack_daily.csv"
agg_df = pd.read_csv(agg_daily_path)
agg_df["date"] = pd.to_datetime(agg_df["date"])
eq = agg_df["portfolio_equity"].values
dr = np.diff(eq) / eq[:-1]

print(f"{'leverage':<10} {'CAGR%':>8} {'Sh':>5} {'DD%':>7} {'ending NAV':>12}")
print("-" * 50)
b5_results = []
for lev in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    lev_dr = dr * lev
    # Compound
    lev_eq = np.concatenate([[eq[0]], eq[0] * np.cumprod(1 + lev_dr)])
    total = (lev_eq[-1] / CAPITAL - 1) * 100
    days = (agg_df["date"].iloc[-1] - agg_df["date"].iloc[0]).days or 1
    cagr = ((lev_eq[-1] / CAPITAL) ** (365 / days) - 1) * 100
    lev_sharpe = lev_dr.mean() / lev_dr.std() * np.sqrt(365) if lev_dr.std() > 0 else 0
    lev_dd = ((lev_eq - np.maximum.accumulate(lev_eq)) / np.maximum.accumulate(lev_eq)).min() * 100
    b5_results.append({"leverage": lev, "cagr_pct": cagr, "sharpe": lev_sharpe,
                        "max_dd_pct": lev_dd, "ending_nav": lev_eq[-1]})
    print(f"{lev:>6.1f}x   {cagr:>+7.2f} {lev_sharpe:>+4.2f} {lev_dd:>+6.2f}  ${lev_eq[-1]:>10.2f}")


# =============================================================================
# Save
# =============================================================================
from datetime import datetime, timezone
out_dir = ROOT / "logs" / "deployment" / str(datetime.now(timezone.utc).date())
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "asym_round4_all_items.json"
with open(out, "w") as f:
    json.dump({
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "B1_4sleeve_U100_asym": b1_result,
        "B2_capit_bounce": b2_results,
        "B3_funding_filter": b3_results,
        "B5_leverage": b5_results,
    }, f, indent=2, default=str)
print(f"\n[saved] {out}")
