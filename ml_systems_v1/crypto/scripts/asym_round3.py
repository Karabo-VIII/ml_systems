"""Round-3 asymmetric experiments + blend aggregation.

Items:
    (1) Family B: vol expansion event-driven long
    (2) Family E v2: oversold bounce with +120% target (overshoot)
    (3) Breakout + meta-gate combination (v8 p_win filter on breakouts)
    (4) Aggregate recommended_3sleeve_diversified_v2 via portfolio_aggregator
"""
from __future__ import annotations

import json
import pickle
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
MODELS = ROOT / "models"
SEEDS_DIR = ROOT / "logs" / "paper_trader_v2" / "seeds"

sys.path.insert(0, str(ROOT / "src" / "strategy"))
from universe import UNIVERSE_50_LIQUID
UNIVERSE = set(UNIVERSE_50_LIQUID)

MAKER_RT = 0.08
CAPITAL = 10000.0
TEST_START = "2025-01-01"
TEST_END = "2026-04-22"

v8 = pickle.load(open(MODELS / "meta_labeler" / "v8_catboost.pkl", "rb"))


# =============================================================================
# Build daily panel (includes OHLC + features needed for each test)
# =============================================================================
print("[panel] building...")
t0 = time.time()
all_fps = sorted(glob.glob(str(DATA / "*_chimera.parquet")))
rows = []
for fp in all_fps:
    asset = Path(fp).stem.replace("usdt_v50_chimera", "").upper()
    if asset not in UNIVERSE:
        continue
    try:
        df = pl.read_parquet(fp).to_pandas()
    except Exception:
        continue
    if len(df) < 1000:
        continue
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
    agg = {"close": "last", "open": "first", "high": "max", "low": "min", "volume": "sum"}
    for c in df.columns:
        if c.startswith("norm_") or c.startswith("xd_") or c == "hurst_regime":
            agg[c] = "last"
    d = df.groupby("date").agg(agg).reset_index()
    d["ret_d"] = d["close"].pct_change()
    d["ret_3d"] = d["close"].pct_change(3)
    d["ret_7d"] = d["close"].pct_change(7)
    d["ret_14d"] = d["close"].pct_change(14)
    d["vol_1d"] = d["ret_d"].abs()  # absolute 1-day return, proxy for 1d realized vol
    d["vol_7d"] = d["ret_d"].rolling(7).std()
    d["vol_30d"] = d["ret_d"].rolling(30).std()
    # Vol z-score: how extreme is today's vol vs recent?
    d["vol_z30"] = (d["vol_1d"] - d["ret_d"].rolling(30).std()) / (d["ret_d"].rolling(30).std() + 1e-9)
    d["low_3d"] = d["low"].rolling(3, min_periods=1).min()
    d["high_prev_10d"] = d["high"].shift(1).rolling(10).max()
    d["high_prev_20d"] = d["high"].shift(1).rolling(20).max()
    d["hl"] = (d["high"] - d["low"]) / d["open"]
    d["asset"] = asset
    rows.append(d)
panel = pd.concat(rows, ignore_index=True).dropna(subset=["ret_d"])
panel["date"] = pd.to_datetime(panel["date"])

btc = pl.read_parquet(DATA / "btcusdt_v50_chimera.parquet", columns=["timestamp", "close"]).to_pandas()
btc["date"] = pd.to_datetime(btc["timestamp"], unit="ms").dt.date
btc_d = btc.groupby("date").agg({"close": "last"}).reset_index()
btc_d["btc_30d"] = btc_d["close"].pct_change(30)
btc_d["date"] = pd.to_datetime(btc_d["date"])
panel = panel.merge(btc_d[["date", "btc_30d"]], on="date", how="left")

# v8 meta-labeler p_win
def score_v8(df):
    feats = v8.get("feature_names") or v8.get("features", [])
    X = np.vstack([df[f].fillna(0).values if f in df.columns else np.zeros(len(df))
                   for f in feats]).T
    return v8["model"].predict_proba(X)[:, 1]

panel["p_v8"] = score_v8(panel)
print(f"[panel] {panel.shape} in {time.time()-t0:.1f}s")


def summarize_trades(trades_df, eq_df, label=""):
    if len(trades_df) == 0 or len(eq_df) < 2:
        return {"label": label, "status": "insufficient", "n_trades": len(trades_df)}
    r = trades_df["net_ret_pct"].values
    wins = r[r > 0]; losses = r[r <= 0]
    hit = len(wins) / len(r) if len(r) else 0
    asym = wins.mean() / abs(losses.mean()) if len(wins) and len(losses) else float("inf")
    kelly_g = (hit * np.log1p(wins.mean()/100 if len(wins) else 0)
               + (1 - hit) * np.log1p(losses.mean()/100 if len(losses) else 0))
    eq = eq_df["equity"].values
    dr = np.diff(eq) / eq[:-1]
    total = (eq[-1] / CAPITAL - 1) * 100
    days = (eq_df["date"].iloc[-1] - eq_df["date"].iloc[0]).days or 1
    cagr = ((eq[-1] / CAPITAL) ** (365 / days) - 1) * 100
    sharpe = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
    cm = np.maximum.accumulate(eq)
    dd = ((eq - cm) / cm).min() * 100
    return {
        "label": label, "n_trades": len(r), "n_days": len(eq_df),
        "cagr_pct": cagr, "sharpe": sharpe, "max_dd_pct": dd, "total_ret_pct": total,
        "hit_rate": hit, "mean_win_pct": wins.mean() if len(wins) else 0,
        "mean_loss_pct": losses.mean() if len(losses) else 0,
        "asymmetry_ratio": asym, "kelly_log_g_per_trade": kelly_g,
    }


# =============================================================================
# GENERIC PORTFOLIO SIM ENGINE  (used by all 3 experiments)
# =============================================================================
def run_portfolio(panel, entry_fn, exit_fn, max_concurrent=10, pct_per_trade=0.10,
                  reentry_cooldown_days=2):
    """entry_fn(row) -> (entry, stop, target, max_hold) or None
       exit_fn(pos, row) -> (exit_price, reason) or (None, None)
    """
    test = panel[(panel["date"] >= TEST_START) & (panel["date"] <= TEST_END)]
    all_dates = sorted(test["date"].unique())
    # Build date -> asset row lookup
    lookups = {d: grp for d, grp in test.groupby("date")}

    cash = CAPITAL
    positions = {}
    last_exit = {}
    trades = []
    daily_eq = []

    for d in all_dates:
        lkup = lookups.get(d)
        if lkup is None: continue

        # 1) MTM + exits
        closed = []
        for asset, pos in list(positions.items()):
            row = lkup[lkup["asset"] == asset]
            if len(row) == 0: continue
            row = row.iloc[0]
            exit_price, reason = exit_fn(pos, row, d)
            if exit_price is not None:
                size = pos["size"]
                pnl = size * (exit_price / pos["entry_price"] - 1)
                pnl -= size * (MAKER_RT / 200.0)
                cash += size + pnl
                net_pct = (exit_price / pos["entry_price"] - 1) * 100 - MAKER_RT
                trades.append({
                    "asset": asset, "entry_date": pos["entry_date"], "exit_date": d,
                    "hold_days": (d - pos["entry_date"]).days,
                    "entry_price": pos["entry_price"], "exit_price": exit_price,
                    "net_ret_pct": net_pct, "exit_reason": reason,
                })
                closed.append(asset)
                last_exit[asset] = d
        for a in closed: del positions[a]

        # 2) New entries
        if len(positions) < max_concurrent:
            # Score candidates
            cands = []
            for _, row in lkup.iterrows():
                asset = row["asset"]
                if asset in positions: continue
                if asset in last_exit and (d - last_exit[asset]).days < reentry_cooldown_days:
                    continue
                spec = entry_fn(row)
                if spec is None: continue
                strength = spec.get("strength", 0)
                cands.append((strength, asset, row["close"], spec))
            cands.sort(reverse=True)
            slots = max_concurrent - len(positions)
            for strength, asset, close, spec in cands[:slots]:
                size = cash * pct_per_trade
                if size > cash: break
                cash -= size + size * (MAKER_RT / 200.0)
                positions[asset] = {
                    "entry_date": d, "entry_price": close,
                    "stop": spec["stop"], "target": spec.get("target"),
                    "max_hold": spec.get("max_hold", 30),
                    "peak": close, "size": size,
                    "trail_pct": spec.get("trail_pct"),
                }

        # 3) MTM
        pos_val = 0.0
        if len(positions):
            close_map = dict(zip(lkup["asset"], lkup["close"]))
            for asset, pos in positions.items():
                if asset in close_map:
                    pos_val += pos["size"] * (close_map[asset] / pos["entry_price"])
        daily_eq.append({"date": d, "equity": cash + pos_val, "n_pos": len(positions)})

    return pd.DataFrame(daily_eq), pd.DataFrame(trades)


# =============================================================================
# (1) FAMILY B: vol expansion long
# =============================================================================
print("\n" + "=" * 80)
print("(1) FAMILY B: VOL EXPANSION LONG")
print("=" * 80)


def family_B_entry(vol_z_thresh, ret_sign_min=-0.02, stop_pct=0.03, trail_pct=0.06, max_hold=10):
    def fn(row):
        if pd.isna(row.get("vol_z30")) or pd.isna(row.get("ret_d")):
            return None
        if row["vol_z30"] < vol_z_thresh:
            return None
        # Require a positive or mild return (not puking)
        if row["ret_d"] < ret_sign_min:
            return None
        entry = row["close"]
        return {"stop": entry * (1 - stop_pct), "trail_pct": trail_pct,
                "max_hold": max_hold, "strength": row["vol_z30"]}
    return fn


def trail_exit(pos, row, today):
    # Update peak + trailing stop
    if row["high"] > pos["peak"]:
        pos["peak"] = row["high"]
        if pos.get("trail_pct"):
            new_trail = pos["peak"] * (1 - pos["trail_pct"])
            if new_trail > pos["stop"]:
                pos["stop"] = new_trail
    # Check: stop hit during day
    if row["low"] <= pos["stop"]:
        return pos["stop"], "stop"
    # Target?
    if pos.get("target") and row["high"] >= pos["target"]:
        return pos["target"], "target"
    # Max hold
    if (today - pos["entry_date"]).days >= pos["max_hold"]:
        return row["close"], "time"
    return None, None


print(f"{'config':<40} {'days':>4} {'CAGR%':>7} {'Sh':>5} {'DD%':>6} "
      f"{'n_tr':>4} {'hit%':>4} {'asym':>5} {'kelly':>7}")
print("-" * 100)
B_results = []
for (vz, stop, trail, mh, lbl) in [
    (1.5, 0.03, 0.06, 10, "volZ1.5_s3_t6_h10"),
    (2.0, 0.03, 0.06, 10, "volZ2.0_s3_t6_h10"),
    (2.5, 0.03, 0.06, 10, "volZ2.5_s3_t6_h10"),
    (2.0, 0.02, 0.05, 5, "volZ2.0_s2_t5_h5"),
    (2.0, 0.04, 0.10, 20, "volZ2.0_s4_t10_h20"),
    (3.0, 0.03, 0.08, 15, "volZ3.0_s3_t8_h15"),
]:
    eq_df, tr_df = run_portfolio(
        panel,
        entry_fn=family_B_entry(vol_z_thresh=vz, stop_pct=stop, trail_pct=trail, max_hold=mh),
        exit_fn=trail_exit,
    )
    s = summarize_trades(tr_df, eq_df, label=lbl)
    B_results.append(s)
    if s.get("status") == "insufficient":
        print(f"{lbl:<40} (insufficient, n={s['n_trades']})")
        continue
    print(f"{lbl:<40} {s['n_days']:>4} {s['cagr_pct']:>+6.2f} "
          f"{s['sharpe']:>+4.2f} {s['max_dd_pct']:>+5.2f} {s['n_trades']:>4} "
          f"{s['hit_rate']*100:>3.0f} {s['asymmetry_ratio']:>4.2f} "
          f"{s['kelly_log_g_per_trade']:>+6.4f}")


# =============================================================================
# (2) FAMILY E v2: bounce with +120% target (overshoot)
# =============================================================================
print("\n" + "=" * 80)
print("(2) FAMILY E v2: BOUNCE WITH +120% TARGET (OVERSHOOT)")
print("=" * 80)


def family_E_v2_entry(drop_thresh=-0.15, rev_min=0.02, target_pct=1.20):
    def fn(row):
        if pd.isna(row.get("ret_3d")) or pd.isna(row.get("ret_d")):
            return None
        if row["ret_3d"] > drop_thresh:
            return None
        if row["ret_d"] < rev_min:
            return None
        entry = row["close"]
        low3d = row["low_3d"]
        drop = entry - low3d
        if drop <= 0:
            return None
        stop = low3d * 0.995  # slightly below 3d low
        # Target = entry + target_pct * (entry - low3d)
        # E.g. target_pct=1.20 -> target overshoots by 20% of the drop magnitude
        target = entry + target_pct * drop
        return {"stop": stop, "target": target, "max_hold": 10,
                "strength": -row["ret_3d"]}  # more negative ret_3d = stronger signal
    return fn


def hard_target_stop_time_exit(pos, row, today):
    if row["low"] <= pos["stop"]:
        return pos["stop"], "stop"
    if pos.get("target") and row["high"] >= pos["target"]:
        return pos["target"], "target"
    if (today - pos["entry_date"]).days >= pos["max_hold"]:
        return row["close"], "time"
    return None, None


print(f"{'config':<40} {'days':>4} {'CAGR%':>7} {'Sh':>5} {'DD%':>6} "
      f"{'n_tr':>4} {'hit%':>4} {'asym':>5} {'kelly':>7}")
print("-" * 100)
E_results = []
for (dt, rm, tp, lbl) in [
    (-0.15, 0.02, 1.0, "drop15_rev2_tgt100"),
    (-0.15, 0.02, 1.2, "drop15_rev2_tgt120"),
    (-0.15, 0.02, 1.5, "drop15_rev2_tgt150"),
    (-0.20, 0.03, 1.2, "drop20_rev3_tgt120"),
    (-0.20, 0.03, 1.5, "drop20_rev3_tgt150"),
    (-0.10, 0.02, 1.0, "drop10_rev2_tgt100"),
    (-0.10, 0.01, 0.8, "drop10_rev1_tgt80"),
]:
    eq_df, tr_df = run_portfolio(
        panel,
        entry_fn=family_E_v2_entry(drop_thresh=dt, rev_min=rm, target_pct=tp),
        exit_fn=hard_target_stop_time_exit,
    )
    s = summarize_trades(tr_df, eq_df, label=lbl)
    E_results.append(s)
    if s.get("status") == "insufficient":
        print(f"{lbl:<40} (insufficient, n={s['n_trades']})")
        continue
    print(f"{lbl:<40} {s['n_days']:>4} {s['cagr_pct']:>+6.2f} "
          f"{s['sharpe']:>+4.2f} {s['max_dd_pct']:>+5.2f} {s['n_trades']:>4} "
          f"{s['hit_rate']*100:>3.0f} {s['asymmetry_ratio']:>4.2f} "
          f"{s['kelly_log_g_per_trade']:>+6.4f}")


# =============================================================================
# (3) BREAKOUT + META-GATE
# =============================================================================
print("\n" + "=" * 80)
print("(3) BREAKOUT + META-GATE v8 (p_v8 filter on breakouts)")
print("=" * 80)


def breakout_meta_entry(N_col, init_stop, trail_pct, max_hold, meta_thresh):
    def fn(row):
        if pd.isna(row.get(N_col)) or row["close"] <= row[N_col]:
            return None
        if meta_thresh > 0 and (pd.isna(row.get("p_v8")) or row["p_v8"] < meta_thresh):
            return None
        entry = row["close"]
        return {"stop": entry * (1 - init_stop), "trail_pct": trail_pct,
                "max_hold": max_hold,
                "strength": (row["close"] - row[N_col]) / row[N_col]}
    return fn


print(f"{'config':<40} {'days':>4} {'CAGR%':>7} {'Sh':>5} {'DD%':>6} "
      f"{'n_tr':>4} {'hit%':>4} {'asym':>5} {'kelly':>7}")
print("-" * 100)
BM_results = []
for (N_col, istop, tstop, mh, meta, lbl) in [
    ("high_prev_10d", 0.02, 0.05, 15, 0.0,  "brk10_no_meta_base"),
    ("high_prev_10d", 0.02, 0.05, 15, 0.40, "brk10_meta@0.40"),
    ("high_prev_10d", 0.02, 0.05, 15, 0.45, "brk10_meta@0.45"),
    ("high_prev_10d", 0.02, 0.05, 15, 0.50, "brk10_meta@0.50"),
    ("high_prev_20d", 0.03, 0.05, 30, 0.0,  "brk20_no_meta_base"),
    ("high_prev_20d", 0.03, 0.05, 30, 0.45, "brk20_meta@0.45"),
    ("high_prev_20d", 0.03, 0.05, 30, 0.50, "brk20_meta@0.50"),
]:
    eq_df, tr_df = run_portfolio(
        panel,
        entry_fn=breakout_meta_entry(N_col, istop, tstop, mh, meta),
        exit_fn=trail_exit,
    )
    s = summarize_trades(tr_df, eq_df, label=lbl)
    BM_results.append(s)
    if s.get("status") == "insufficient":
        print(f"{lbl:<40} (insufficient, n={s['n_trades']})")
        continue
    print(f"{lbl:<40} {s['n_days']:>4} {s['cagr_pct']:>+6.2f} "
          f"{s['sharpe']:>+4.2f} {s['max_dd_pct']:>+5.2f} {s['n_trades']:>4} "
          f"{s['hit_rate']*100:>3.0f} {s['asymmetry_ratio']:>4.2f} "
          f"{s['kelly_log_g_per_trade']:>+6.4f}")


# =============================================================================
# (4) Aggregate recommended_3sleeve_diversified_v2
# =============================================================================
print("\n" + "=" * 80)
print("(4) AGGREGATE recommended_3sleeve_diversified_v2")
print("=" * 80)
result = subprocess.run(
    [sys.executable, str(ROOT / "src" / "analysis" / "portfolio_aggregator.py"),
     "--blend", "recommended_3sleeve_diversified_v2", "--dd-halt", "-12"],
    capture_output=True, text=True, timeout=120,
    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
print(result.stdout[-3000:])
if result.returncode != 0:
    print(f"[ERR] aggregator exit={result.returncode}")
    print(result.stderr[-1000:])


# =============================================================================
# SAVE
# =============================================================================
from datetime import datetime, timezone
out_dir = ROOT / "logs" / "deployment" / str(datetime.now(timezone.utc).date())
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "asym_round3.json"
with open(out, "w") as f:
    json.dump({
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_B_vol_expansion": B_results,
        "experiment_E_v2_overshoot": E_results,
        "experiment_breakout_meta_gate": BM_results,
        "aggregator_rc": result.returncode,
    }, f, indent=2, default=str)
print(f"\n[saved] {out}")
