#!/usr/bin/env python3
"""MARKET RESEARCH -- is the IDIOSYNCRATIC opportunity diversifiable, or does it cluster?

After removing BTC-beta, ~65% of >=5% movers survive as 'idiosyncratic'. But are those residual moves truly
INDEPENDENT across assets (diversifiable -> a per-asset edge spreads risk), or do they still cluster by theme/
sector (so 'idiosyncratic' is really sector-beta)? Computes the mean pairwise correlation of per-asset RESIDUAL
daily returns (asset_ret - beta*btc_ret) vs RAW returns. Low residual corr => diversifiable. Run:
python scripts/research/idiosyncratic_clustering.py
No emoji (Windows cp1252).
"""
import datetime
import glob
import os

import numpy as np
import pandas as pd
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _files():
    return sorted(glob.glob(os.path.join(ROOT, "data", "processed", "chimera", "1d", "*.parquet")))


def _asset(p):
    return os.path.basename(p).split("_v51_chimera_")[0].upper()


def _ret_series(path):
    try:
        df = pl.read_parquet(path, columns=["timestamp", "close"]).drop_nulls().sort("timestamp")
    except Exception:
        return None
    c = df["close"].to_numpy().astype(float); ts = df["timestamp"].to_numpy()
    days = [datetime.datetime.utcfromtimestamp(int(t) / 1000).date().isoformat() for t in ts]
    s = pd.Series(c, index=days)
    return s.pct_change().dropna()


def main():
    files = _files()
    btc = _ret_series(next(f for f in files if _asset(f) == "BTCUSDT"))
    raw = {}
    for f in files:
        a = _asset(f)
        if a == "BTCUSDT":
            continue
        s = _ret_series(f)
        if s is not None and len(s) > 200:
            raw[a] = s
    raw_df = pd.DataFrame(raw)
    # residual = asset - beta*btc (beta per asset on common dates)
    resid = {}
    for a in raw_df.columns:
        j = pd.concat([raw_df[a], btc], axis=1, join="inner").dropna()
        if len(j) < 200:
            continue
        ar, br = j.iloc[:, 0].values, j.iloc[:, 1].values
        beta = np.cov(ar, br)[0, 1] / np.var(br)
        resid[a] = pd.Series(ar - beta * br, index=j.index)
    resid_df = pd.DataFrame(resid)

    def mean_offdiag_corr(df):
        # restrict to a dense recent window for a fair panel
        dense = df.dropna(axis=0, thresh=int(0.5 * df.shape[1]))
        cm = dense.corr(min_periods=100).values
        n = cm.shape[0]
        off = cm[~np.eye(n, dtype=bool)]
        off = off[~np.isnan(off)]
        return float(np.mean(off)), int(n)

    raw_c, n1 = mean_offdiag_corr(raw_df)
    res_c, n2 = mean_offdiag_corr(resid_df)
    print("CROSS-ASSET CO-MOVEMENT (mean pairwise daily-return correlation):")
    print(f"  RAW returns        : {raw_c:.3f}   (high => everything moves together = BTC/market beta)")
    print(f"  RESIDUAL (idiosyn) : {res_c:.3f}   (low => idiosyncratic moves are DIVERSIFIABLE / independent)")
    print(f"  n_assets ~ {n2}")
    print(f"  read: residual corr {res_c:.2f} vs raw {raw_c:.2f} -- "
          + ("idiosyncratic opportunity is largely DIVERSIFIABLE (good)." if res_c < 0.15
             else "idiosyncratic moves still CLUSTER (sector-beta), less diversifiable than they look."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
