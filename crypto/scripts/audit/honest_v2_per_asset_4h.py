"""honest_v2_per_asset_4h.py -- 4h cadence per-asset OOS sim.

Tests Fix #2 (sub-day) — MA computed on 4h bars instead of 1d candles.

Sim mechanics:
  - Load chimera 4h (6 bars per calendar day)
  - Compute MA at 4h cadence per asset's cousin-set cells
  - At end of each calendar day, check MA status on the day's last 4h bar
  - Enter on close of last 4h bar of trading day T
  - Walk forward at 4h granularity: hold_max = 14d * 6 = 84 4h bars
  - Trail-stop applied per 4h bar (finer-grained risk control)
  - Same constraints: K=8, per_asset 10%, total 60%

Compare against 1d baseline + 1d + confirmation gate.
"""
from __future__ import annotations

import sys
import json
import glob
import yaml
from datetime import date as _date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
PROFILE_4H_PATH = ROOT / "data" / "processed" / "per_asset_ma_ema_profile_4h.parquet"
OWN_REGIME_PATH = ROOT / "data" / "processed" / "asset_own_regime_panel.parquet"
CHIMERA_4H = ROOT / "data" / "processed" / "chimera" / "4h"
U50_YAML = ROOT / "config" / "universes" / "u50.yaml"
U100_YAML = ROOT / "config" / "universes" / "u100.yaml"
OUT_DIR = ROOT / "runs" / "audit" / "MA_EMA_PROFILE_2026_05_20"

OOS_START = _date(2024, 5, 16)
OOS_END   = _date(2025, 3, 15)

K_MAX = 8
PER_ASSET_CAP = 0.10
TOTAL_DEPLOY_CAP = 0.60
COST_RT = 0.0030

# 4h-specific: 14d × 6 bars/day = 84 bars
HOLD_MAX_BARS = 84
HARD_STOP = -0.04
TRAIL_ARM = 0.05
TRAIL_DROP = 0.03

REGIME_TIER_MULT = {"bull": 1.20, "chop": 1.00, "bear": 0.70, "crash": 0.40}
BUCKET_VOL_MULT = {"BLUE": 0.6, "STEADY": 0.9, "VOLATILE": 1.15, "DEGEN": 1.30}
REGIME_QUALIFIES = {
    "REGIME_DEPENDENT": {"bull", "chop"},
    "BULL_AND_CHOP": {"bull", "chop"},
    "BULL_ONLY": {"bull"},
    "ALL_WEATHER": {"bull", "chop", "bear", "crash"},
    "BLOCK_OWN_CRASH": {"bull", "chop", "bear"},
    "BLOCK_OWN_BEAR": {"bull", "chop"},
}


def load_universe():
    asset_to_bucket = {}
    for path in (U50_YAML, U100_YAML):
        with open(path) as f:
            doc = yaml.safe_load(f)
        for a in doc.get("assets", []) + doc.get("extra_assets", []):
            if a.get("status", "ready") != "ready": continue
            sym = a["symbol"].replace("USDT", "")
            asset_to_bucket[sym] = a.get("dna", "VOLATILE")
    return asset_to_bucket


def compute_ma(closes, ma_type, fast, slow):
    s = pd.Series(closes)
    if ma_type == "SMA":
        return s.rolling(fast).mean().values, s.rolling(slow).mean().values
    return s.ewm(span=fast, adjust=False).mean().values, s.ewm(span=slow, adjust=False).mean().values


def walk_forward_exit(entry_price, fwd_closes, hold_max, stop, trail_arm, trail_drop, cost):
    peak = entry_price; armed = False
    for d, p in enumerate(fwd_closes, start=1):
        if p is None or not np.isfinite(p): continue
        ret = p / entry_price - 1
        if p > peak: peak = p
        if not armed and ret >= trail_arm: armed = True
        if ret <= stop: return stop - cost, d, "stop"
        if armed and p <= peak * (1 - trail_drop):
            return p / entry_price - 1 - cost, d, "trail"
        if d >= hold_max: return ret - cost, d, "max_hold"
    last = next((p for p in reversed(fwd_closes) if p is not None and np.isfinite(p)), None)
    if last is None: return -cost, 0, "no_data"
    return last / entry_price - 1 - cost, len(fwd_closes), "expire"


def load_chimera_4h():
    chim = {}
    for f in glob.glob(str(CHIMERA_4H / "*_v51_chimera_4h_*.parquet")):
        sym = Path(f).name.split("_")[0].upper().replace("USDT", "")
        try:
            df = pl.read_parquet(f, columns=["timestamp", "close"]).to_pandas()
        except Exception: continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["date"] = df["timestamp"].dt.normalize()
        df = df.sort_values("timestamp").reset_index(drop=True)
        chim[sym] = df
    return chim


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirmation-gate", action="store_true",
                    help="Require 2+ qualifying cells per asset")
    args = ap.parse_args()

    print("="*78)
    print(f"4h CADENCE OOS SIM {'(+ confirmation gate)' if args.confirmation_gate else ''}")
    print("="*78)

    profile = pd.read_parquet(PROFILE_4H_PATH)
    own_regime = pl.read_parquet(OWN_REGIME_PATH).to_pandas()
    own_regime["date"] = pd.to_datetime(own_regime["date"]).dt.normalize()
    asset_to_bucket = load_universe()
    print(f"  profile: {len(profile)} cells / {profile['asset'].nunique()} assets")
    print(f"  cousin members: {profile['is_cousin_set_member'].sum()}")

    print("Loading chimera 4h...")
    chimera = load_chimera_4h()
    print(f"  {len(chimera)} assets")

    own_lookup = {(r["asset"], r["date"]): r["asset_own_regime"]
                  for _, r in own_regime.iterrows()}

    cousins = profile[profile["is_cousin_set_member"]].copy()
    cousins_by_asset = {a: g.to_dict("records") for a, g in cousins.groupby("asset")}

    # Precompute MA-active per asset per cell (on 4h bars)
    print("Computing 4h MA matrices...")
    asset_cells = {}
    for asset, cells_list in cousins_by_asset.items():
        chim = chimera.get(asset)
        if chim is None: continue
        closes = chim["close"].values
        cells_compiled = []
        for c in cells_list:
            ma_f, ma_s = compute_ma(closes, c["ma_type"], int(c["fast"]), int(c["slow"]))
            cells_compiled.append({
                "source": "cousin", "regime_qualifies": REGIME_QUALIFIES.get(c["regime_tag"], set()),
                "ma_type": c["ma_type"], "fast": int(c["fast"]), "slow": int(c["slow"]),
                "sharpe": c["sharpe"], "regime_tag": c["regime_tag"],
                "active": ma_f > ma_s,
            })
        asset_cells[asset] = (chim, cells_compiled)
    print(f"  {len(asset_cells)} assets prepared")

    # Build per-asset (date → idx of last 4h bar of that day)
    date_to_last_4h_idx = {}
    for asset, (chim, _) in asset_cells.items():
        by_day = {}
        for i, row in chim.iterrows():
            d = row["date"]
            by_day[d] = i  # last seen idx per day (chim is sorted ascending)
        date_to_last_4h_idx[asset] = by_day

    # Calendar dates
    cur = OOS_START; oos_dates = []
    while cur <= OOS_END:
        oos_dates.append(cur); cur += timedelta(days=1)

    # Median sharpe across cousins
    all_sharpes = [c["sharpe"] for cells in cousins_by_asset.values() for c in cells]
    median_sharpe = float(np.median([s for s in all_sharpes if s > 0])) if all_sharpes else 0.1

    print(f"\nSimulating {len(oos_dates)} calendar days at 4h granularity...")
    portfolio_value = 1.0
    available_cash = 1.0
    open_positions = []
    trade_log = []
    daily_records = []

    for sim_date in oos_dates:
        sim_dt = pd.Timestamp(sim_date)

        # Close positions: walk forward 4h bars from entry, apply stop/trail
        new_open = []
        for pos in open_positions:
            chim, _ = asset_cells.get(pos["asset"], (None, []))
            if chim is None: new_open.append(pos); continue
            # Forward bars: from entry_idx + 1 to current sim_date's last 4h bar
            today_idx = date_to_last_4h_idx.get(pos["asset"], {}).get(sim_dt)
            if today_idx is None: new_open.append(pos); continue
            entry_idx = pos["entry_idx"]
            if today_idx <= entry_idx: new_open.append(pos); continue
            fwd_closes = chim["close"].values[entry_idx + 1: today_idx + 1].tolist()
            fwd_closes = [float(p) if np.isfinite(p) else None for p in fwd_closes]
            rret, d_held, reason = walk_forward_exit(
                pos["entry_price"], fwd_closes,
                hold_max=HOLD_MAX_BARS, stop=HARD_STOP,
                trail_arm=TRAIL_ARM, trail_drop=TRAIL_DROP, cost=COST_RT,
            )
            if reason in ("stop", "trail", "max_hold"):
                pnl = pos["bet_size"] * rret
                available_cash += pos["bet_size"] + pnl
                trade_log.append({
                    "asset": pos["asset"], "entry_date": pos["entry_date"],
                    "exit_date": sim_date, "bars_held": d_held,
                    "days_held_approx": d_held / 6.0,
                    "bet_size": pos["bet_size"], "realized_ret": rret,
                    "exit_reason": reason,
                })
            else:
                new_open.append(pos)
        open_positions = new_open

        # Identify candidates: end-of-day check of MA active
        open_assets = set(p["asset"] for p in open_positions)
        candidates = []
        for asset, (chim, cells) in asset_cells.items():
            if asset in open_assets: continue
            today_idx = date_to_last_4h_idx.get(asset, {}).get(sim_dt)
            if today_idx is None: continue
            own_r = own_lookup.get((asset, sim_dt))
            if own_r is None: continue

            qualifying = [c for c in cells
                          if own_r in c["regime_qualifies"] and c["active"][today_idx]]
            if not qualifying: continue
            if args.confirmation_gate and len(qualifying) < 2: continue

            best_cell = max(qualifying, key=lambda c: c["sharpe"])
            bucket = asset_to_bucket.get(asset, "VOLATILE")
            tier_mult = REGIME_TIER_MULT.get(own_r, 0.5)
            vol_mult = BUCKET_VOL_MULT.get(bucket, 1.0)
            quality_mult = float(np.clip(best_cell["sharpe"] / max(median_sharpe, 0.05), 0.7, 1.3))
            deploy_score = best_cell["sharpe"] * tier_mult * vol_mult * quality_mult
            candidates.append({
                "asset": asset, "bucket": bucket, "regime": own_r,
                "deploy_score": deploy_score,
                "tier_mult": tier_mult, "vol_mult": vol_mult, "quality_mult": quality_mult,
                "chim": chim, "today_idx": today_idx,
            })

        if candidates:
            candidates.sort(key=lambda x: -x["deploy_score"])
            slots = K_MAX - len(open_positions)
            budget = portfolio_value * TOTAL_DEPLOY_CAP - sum(p["bet_size"] for p in open_positions)
            for c in candidates:
                if slots <= 0 or budget <= 0.001 * portfolio_value: break
                base = (TOTAL_DEPLOY_CAP / K_MAX) * portfolio_value
                target = base * c["tier_mult"] * c["vol_mult"] * c["quality_mult"]
                hard_cap = portfolio_value * PER_ASSET_CAP
                actual = min(target, hard_cap, budget, available_cash)
                if actual < 0.005 * portfolio_value: continue
                ep = float(c["chim"].iloc[c["today_idx"]]["close"])
                if ep <= 0 or not np.isfinite(ep): continue
                available_cash -= actual; budget -= actual; slots -= 1
                open_positions.append({
                    "asset": c["asset"], "entry_date": sim_dt,
                    "entry_idx": c["today_idx"], "entry_price": ep, "bet_size": actual,
                })

        # MtM
        omtm = 0
        for pos in open_positions:
            chim, _ = asset_cells.get(pos["asset"], (None, []))
            if chim is None: omtm += pos["bet_size"]; continue
            today_idx = date_to_last_4h_idx.get(pos["asset"], {}).get(sim_dt)
            if today_idx is None: omtm += pos["bet_size"]; continue
            cp = float(chim.iloc[today_idx]["close"])
            if not np.isfinite(cp): omtm += pos["bet_size"]; continue
            omtm += pos["bet_size"] * (cp / pos["entry_price"])
        portfolio_value = available_cash + omtm
        daily_records.append({"date": sim_date, "portfolio_value": portfolio_value,
                              "n_open": len(open_positions),
                              "deployed_pct": (portfolio_value - available_cash) / portfolio_value * 100})

    daily_df = pd.DataFrame(daily_records)
    trades_df = pd.DataFrame(trade_log)

    pv = daily_df["portfolio_value"].values
    total_pct = (pv[-1] / pv[0] - 1) * 100
    window_days = (OOS_END - OOS_START).days
    ann_pct = ((1 + total_pct/100) ** (365/max(window_days,1)) - 1) * 100
    drs = pv[1:] / pv[:-1] - 1
    sortino = (drs.mean() / drs[drs < 0].std() * np.sqrt(252)) if (drs < 0).sum() and drs[drs<0].std() > 0 else 0
    sharpe = (drs.mean() / drs.std() * np.sqrt(252)) if drs.std() > 0 else 0
    cum = pv / pv[0]; cm = np.maximum.accumulate(cum)
    max_dd = ((cum / cm - 1) * 100).min()
    mean_deployed = daily_df["deployed_pct"].mean()

    print(f"\n=== RESULTS (4h cadence{', confirmation gate' if args.confirmation_gate else ''}) ===")
    print(f"  Total: {total_pct:+.2f}%  Annualized: {ann_pct:+.2f}%  Daily: {drs.mean()*100:+.4f}%")
    print(f"  Sortino: {sortino:+.3f}  Sharpe: {sharpe:+.3f}  Max DD: {max_dd:+.2f}%")
    print(f"  Trades: {len(trades_df)}  Mean deploy: {mean_deployed:.1f}%")
    if len(trades_df):
        wr = (trades_df["realized_ret"] > 0).mean() * 100
        aw = trades_df.loc[trades_df["realized_ret"]>0, "realized_ret"].mean()*100 if (trades_df["realized_ret"]>0).any() else 0
        al = trades_df.loc[trades_df["realized_ret"]<0, "realized_ret"].mean()*100 if (trades_df["realized_ret"]<0).any() else 0
        mw = trades_df["realized_ret"].max() * 100
        mh = trades_df["bars_held"].median()
        mhd = mh / 6.0
        print(f"  Win rate: {wr:.1f}%  Avg W/L: {aw:+.2f}% / {al:+.2f}%")
        print(f"  Max win: {mw:+.2f}%  Median hold: {mh:.0f} 4h bars ({mhd:.1f} days)")

    suffix = "_with_conf" if args.confirmation_gate else "_baseline"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"oos_4h{suffix}_metrics.json", "w") as f:
        json.dump({
            "mode": f"4h{suffix}", "total_pct": float(total_pct), "ann_pct": float(ann_pct),
            "daily_mean_pct": float(drs.mean() * 100) if len(drs) else 0,
            "sortino": float(sortino), "sharpe": float(sharpe),
            "max_dd_pct": float(max_dd), "n_trades": int(len(trades_df)),
            "mean_deployed_pct": float(mean_deployed),
            "win_rate_pct": float((trades_df["realized_ret"] > 0).mean() * 100) if len(trades_df) else 0,
        }, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
