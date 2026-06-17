#!/usr/bin/env python3
"""MARKET RESEARCH r4 -- the BETA-CONFOUND: how much of alt 'movement' is just BTC?

When BTC dumps 5%, dozens of alts dump ~5% -- that is ONE market event, not N independent opportunities. A
per-asset strategy needs IDIOSYNCRATIC movement (independent of BTC) to be a real per-asset edge. This computes,
per asset: beta to BTC + R^2 (how much of its variance BTC explains), then the RESIDUAL (idiosyncratic) daily
return = asset_ret - beta*btc_ret, and re-derives 'movers/day' on RESIDUALS vs RAW. The gap = the confound.

Run: python scripts/research/beta_confound.py --cadence 1d
No emoji (Windows cp1252).
"""
import argparse
import datetime
import glob
import json
import os

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _files(cadence):
    return sorted(glob.glob(os.path.join(ROOT, "data", "processed", "chimera", cadence, "*.parquet")))


def _asset(p):
    return os.path.basename(p).split("_v51_chimera_")[0].upper()


def _ret_by_date(path):
    try:
        df = pl.read_parquet(path, columns=["timestamp", "close"]).drop_nulls().sort("timestamp")
    except Exception:
        return {}
    c = df["close"].to_numpy().astype(float)
    ts = df["timestamp"].to_numpy()
    out = {}
    for i in range(1, len(c)):
        if c[i - 1] > 0:
            day = datetime.datetime.utcfromtimestamp(int(ts[i]) / 1000).date().isoformat()
            out[day] = (c[i] - c[i - 1]) / c[i - 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadence", default="1d")
    ap.add_argument("--thresholds", default="0.02,0.05,0.10")
    args = ap.parse_args()
    thr = [float(x) for x in args.thresholds.split(",")]

    files = _files(args.cadence)
    btc_path = next((f for f in files if _asset(f) == "BTCUSDT"), None)
    if not btc_path:
        print("no BTC file -- cannot compute beta")
        return 1
    btc = _ret_by_date(btc_path)

    rets = {}
    for f in files:
        a = _asset(f)
        if a == "BTCUSDT":
            continue
        rets[a] = _ret_by_date(f)

    betas = {}
    resid_by_date = {}  # date -> {asset: residual_ret}
    raw_by_date = {}
    for a, r in rets.items():
        days = [d for d in r if d in btc]
        if len(days) < 100:
            continue
        ar = np.array([r[d] for d in days])
        br = np.array([btc[d] for d in days])
        var_b = np.var(br)
        if var_b <= 0:
            continue
        beta = float(np.cov(ar, br)[0, 1] / var_b)
        corr = float(np.corrcoef(ar, br)[0, 1])
        betas[a] = {"beta": round(beta, 3), "r2": round(corr * corr, 3), "n": len(days)}
        for i, d in enumerate(days):
            resid_by_date.setdefault(d, {})[a] = float(ar[i] - beta * br[i])
            raw_by_date.setdefault(d, {})[a] = float(abs(ar[i]))

    # aggregate
    blist = [v["beta"] for v in betas.values()]
    r2list = [v["r2"] for v in betas.values()]
    agg = {
        "cadence": args.cadence, "n_assets": len(betas),
        "median_beta_to_btc": round(float(np.median(blist)), 3),
        "median_r2_btc_explains": round(float(np.median(r2list)), 3),
        "mean_r2_btc_explains": round(float(np.mean(r2list)), 3),
        "pct_assets_r2_ge_0.3": round(float(np.mean([x >= 0.3 for x in r2list])), 3),
    }
    movers = {}
    for t in thr:
        raw_counts = [sum(1 for v in d.values() if v >= t) for d in raw_by_date.values() if len(d) >= 10]
        res_counts = [sum(1 for v in d.values() if abs(v) >= t) for d in resid_by_date.values() if len(d) >= 10]
        movers[f"raw_movers_ge_{t}"] = round(float(np.mean(raw_counts)), 2) if raw_counts else None
        movers[f"idiosyncratic_movers_ge_{t}"] = round(float(np.mean(res_counts)), 2) if res_counts else None
        if raw_counts and res_counts and np.mean(raw_counts) > 0:
            movers[f"idiosyncratic_fraction_{t}"] = round(float(np.mean(res_counts) / np.mean(raw_counts)), 3)
    agg["movers_raw_vs_idiosyncratic"] = movers
    print(json.dumps(agg, indent=2))
    out_dir = os.path.join(ROOT, "runs", "research")
    os.makedirs(out_dir, exist_ok=True)
    json.dump({"agg": agg, "betas": betas}, open(os.path.join(out_dir, f"beta_confound_{args.cadence}.json"), "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
