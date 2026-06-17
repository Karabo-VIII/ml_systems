#!/usr/bin/env python3
"""MARKET RESEARCH r1+r2 -- the OPPORTUNITY SURFACE: how much do assets move, how often, after cost.

NOT a strategy. This characterizes the RAW forward-move distribution = the CEILING of opportunity IF you had
perfect entry+exit timing. Harvestability (needing a signal to time it) is a SEPARATE question, held open.

Per bar t (entering at close[t]):
  - MFE_H  = max over t+1..t+H of (high[k]-close[t])/close[t]   -> best up-move available within H bars
  - MAE_H  = min over t+1..t+H of (low[k]-close[t])/close[t]    -> worst drawdown within H bars
  - ret_H  = (close[t+H]-close[t])/close[t]                     -> realized H-bar return
Then: frequency of MFE_H >= thresholds, and NET-OF-COST (MFE_H - cost_rt) -- the harvestable ceiling.
Plus the universe 'movers/day' premise re-test (1d): per date, how many assets moved >= X% (|ret_1|).

Run: python scripts/research/move_distribution.py --cadence 1d --horizon 1 [--cost 0.0024]
No emoji (Windows cp1252).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _files(cadence):
    d = os.path.join(ROOT, "data", "processed", "chimera", cadence)
    return sorted(glob.glob(os.path.join(d, "*.parquet")))


def _asset(path):
    base = os.path.basename(path)
    return base.split("_v51_chimera_")[0].upper()


def _load_ohlc(path):
    try:
        df = pl.read_parquet(path, columns=["timestamp", "open", "high", "low", "close"])
        return df.drop_nulls().sort("timestamp")
    except Exception:
        return None


def per_asset_stats(df, H, cost, thresholds):
    c = df["close"].to_numpy().astype(float)
    h = df["high"].to_numpy().astype(float)
    l = df["low"].to_numpy().astype(float)
    n = len(c)
    if n < H + 5:
        return None
    # rolling forward MFE/MAE over H bars (vectorized via shifted windows)
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    for k in range(1, H + 1):
        fwd_h = np.concatenate([h[k:], np.full(k, np.nan)])
        fwd_l = np.concatenate([l[k:], np.full(k, np.nan)])
        up = (fwd_h - c) / c
        dn = (fwd_l - c) / c
        mfe = np.where(np.isnan(mfe), up, np.fmax(mfe, up))
        mae = np.where(np.isnan(mae), dn, np.fmin(mae, dn))
    valid = ~np.isnan(mfe)
    mfe = mfe[valid]
    if len(mfe) == 0:
        return None
    out = {"n_bars": int(len(mfe)), "mfe_median": float(np.median(mfe)), "mfe_p90": float(np.percentile(mfe, 90))}
    for t in thresholds:
        out[f"freq_mfe_ge_{t}"] = float(np.mean(mfe >= t))
        out[f"freq_net_ge_{t}"] = float(np.mean((mfe - cost) >= t))  # net-of-cost harvestable ceiling
    out["freq_net_positive"] = float(np.mean((mfe - cost) > 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadence", default="1d")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--cost", type=float, default=0.0024)
    ap.add_argument("--thresholds", default="0.02,0.05,0.10")
    ap.add_argument("--max-assets", type=int, default=0, help="0 = all")
    args = ap.parse_args()
    thr = [float(x) for x in args.thresholds.split(",")]

    files = _files(args.cadence)
    if args.max_assets:
        files = files[: args.max_assets]
    if not files:
        print(f"no files for cadence {args.cadence}")
        return 1

    per_asset = {}
    daily_ret = {}  # date -> {asset: ret_1} for movers/day
    for f in files:
        a = _asset(f)
        df = _load_ohlc(f)
        if df is None or len(df) < args.horizon + 5:
            continue
        s = per_asset_stats(df, args.horizon, args.cost, thr)
        if s:
            per_asset[a] = s
        # movers/day premise (1d): per-date |ret_1|
        c = df["close"].to_numpy().astype(float)
        ts = df["timestamp"].to_numpy()
        ret1 = np.concatenate([[np.nan], (c[1:] - c[:-1]) / c[:-1]])
        for i in range(len(ts)):
            if not np.isnan(ret1[i]):
                import datetime
                day = datetime.datetime.utcfromtimestamp(int(ts[i]) / 1000).date().isoformat()
                daily_ret.setdefault(day, {})[a] = float(abs(ret1[i]))

    # universe-wide aggregation
    if not per_asset:
        print("no usable assets")
        return 1
    agg = {"cadence": args.cadence, "horizon_bars": args.horizon, "cost_rt": args.cost,
           "n_assets": len(per_asset), "thresholds": thr}
    for t in thr:
        agg[f"univ_freq_mfe_ge_{t}"] = float(np.mean([s[f"freq_mfe_ge_{t}"] for s in per_asset.values()]))
        agg[f"univ_freq_net_ge_{t}"] = float(np.mean([s[f"freq_net_ge_{t}"] for s in per_asset.values()]))
    agg["univ_freq_net_positive"] = float(np.mean([s["freq_net_positive"] for s in per_asset.values()]))
    agg["univ_mfe_median"] = float(np.median([s["mfe_median"] for s in per_asset.values()]))

    # movers/day (using |ret_1| on whatever cadence; meaningful for 1d)
    movers = {}
    for t in thr:
        per_day_counts = [sum(1 for v in d.values() if v >= t) for d in daily_ret.values() if len(d) >= 10]
        days_with_mover = [1 if any(v >= t for v in d.values()) else 0 for d in daily_ret.values() if len(d) >= 10]
        if per_day_counts:
            movers[f"avg_movers_ge_{t}_per_period"] = float(np.mean(per_day_counts))
            movers[f"median_movers_ge_{t}_per_period"] = float(np.median(per_day_counts))
            movers[f"pct_periods_with_>=1_mover_{t}"] = float(np.mean(days_with_mover))
    agg["movers_premise"] = movers
    agg["n_periods_for_movers"] = len([d for d in daily_ret.values() if len(d) >= 10])

    # top movers (by net-positive freq) for the avenue map
    ranked = sorted(per_asset.items(), key=lambda kv: kv[1]["freq_net_positive"], reverse=True)
    agg["top10_assets_by_net_positive_mfe"] = [
        {"asset": a, "freq_net_positive": round(s["freq_net_positive"], 3),
         f"freq_net_ge_{thr[1]}": round(s.get(f"freq_net_ge_{thr[1]}", 0), 3),
         "mfe_median": round(s["mfe_median"], 4)} for a, s in ranked[:10]]

    print(json.dumps(agg, indent=2))
    out_dir = os.path.join(ROOT, "runs", "research")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"move_dist_{args.cadence}_H{args.horizon}.json"), "w", encoding="utf-8") as fh:
        json.dump({"agg": agg, "per_asset": per_asset}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
