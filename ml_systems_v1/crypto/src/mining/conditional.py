"""Chimera mining P12 -- conditional structure (the 'etc' cuts that enrich the decomposition).

P12a seasonality            : time-of-day (UTC hour) + day-of-week mean return / mean |ret| (vol). Reveals
                              intraday/weekly periodic structure (funding-time, weekend effects).
P12b vol_expansion_setup    : does a vol-EXPANSION bar predict a larger NEXT move (vol momentum) -- and is the
                              direction still a coin flip? The descriptive bridge from "vol is predictable" to a
                              testable SETUP. Conditional next-bar |ret| and up-rate, expansion vs calm.
P12c regime_conditioned_ac  : within each regime_label state, is the market more mean-reverting or momentum?
                              AC1 of returns per regime.

DESCRIPTIVE. Memory-safe (stream + accumulate). No costs/gate. No emoji.
Run:  python src/mining/conditional.py --cadences 1d,4h,1h,30m,15m
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs" / "mining"


def _files(cad):
    return sorted(glob.glob(str(ROOT / "data" / "processed" / "chimera" / cad /
                                f"*_v51_chimera_{cad}_*.parquet")))


# -------------------------------------------------------- P12a seasonality
def seasonality(cad: str, sample_assets: int = 50) -> dict:
    files = _files(cad)
    if len(files) > sample_assets:
        idx = np.linspace(0, len(files) - 1, sample_assets).astype(int)
        files = [files[i] for i in idx]
    hour_ret, hour_mag, hour_n = {}, {}, {}
    dow_ret, dow_mag, dow_n = {}, {}, {}
    for f in files:
        try:
            df = pl.read_parquet(f, columns=["timestamp", "close"]).to_pandas()
        except Exception:
            continue
        if len(df) < 200:
            continue
        ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        ret = np.log(df["close"]).diff().to_numpy()
        hod = ts.dt.hour.to_numpy(); dow = ts.dt.dayofweek.to_numpy()
        for i in range(1, len(ret)):
            if not np.isfinite(ret[i]):
                continue
            h = int(hod[i]); d = int(dow[i])
            hour_ret[h] = hour_ret.get(h, 0.0) + ret[i]; hour_mag[h] = hour_mag.get(h, 0.0) + abs(ret[i]); hour_n[h] = hour_n.get(h, 0) + 1
            dow_ret[d] = dow_ret.get(d, 0.0) + ret[i]; dow_mag[d] = dow_mag.get(d, 0.0) + abs(ret[i]); dow_n[d] = dow_n.get(d, 0) + 1
    out = {"cadence": cad}
    if cad != "1d" and hour_n:
        out["hour_of_day_mean_ret_bps"] = {str(h): round(1e4 * hour_ret[h] / hour_n[h], 2) for h in sorted(hour_n)}
        out["hour_of_day_mean_absret_bps"] = {str(h): round(1e4 * hour_mag[h] / hour_n[h], 1) for h in sorted(hour_n)}
        # best/worst hour by mean ret
        hr = {h: hour_ret[h] / hour_n[h] for h in hour_n}
        out["best_hour_utc"] = int(max(hr, key=hr.get)); out["worst_hour_utc"] = int(min(hr, key=hr.get))
    dows = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    if dow_n:
        out["day_of_week_mean_ret_bps"] = {dows[d]: round(1e4 * dow_ret[d] / dow_n[d], 2) for d in sorted(dow_n)}
        out["day_of_week_mean_absret_bps"] = {dows[d]: round(1e4 * dow_mag[d] / dow_n[d], 1) for d in sorted(dow_n)}
    out["note"] = "UTC. bps = 1e4*mean log-return per bar. Descriptive periodicity (no costs)."
    return out


# -------------------------------------------------------- P12b vol-expansion setup
def vol_expansion_setup(cad: str, sample_assets: int = 50) -> dict:
    files = _files(cad)
    if len(files) > sample_assets:
        idx = np.linspace(0, len(files) - 1, sample_assets).astype(int)
        files = [files[i] for i in idx]
    nxt_exp, nxt_calm, up_exp, up_calm = [], [], [], []
    for f in files:
        try:
            close = pl.read_parquet(f, columns=["close"])["close"].to_numpy().astype(float)
        except Exception:
            continue
        if len(close) < 300:
            continue
        ret = np.zeros(len(close)); ret[1:] = np.diff(np.log(np.clip(close, 1e-12, None)))
        amag = np.abs(ret)
        rv = pd.Series(amag).rolling(20).mean().to_numpy()       # local vol
        med = pd.Series(rv).expanding(50).median().shift(1).to_numpy()  # past-only threshold
        for t in range(50, len(ret) - 1):
            if not (np.isfinite(rv[t]) and np.isfinite(med[t]) and med[t] > 0 and np.isfinite(ret[t + 1])):
                continue
            expansion = rv[t] > 1.5 * med[t]
            (nxt_exp if expansion else nxt_calm).append(abs(ret[t + 1]))
            (up_exp if expansion else up_calm).append(1.0 if ret[t + 1] > 0 else 0.0)
    if not nxt_exp or not nxt_calm:
        return {"cadence": cad, "error": "insufficient"}
    return {"cadence": cad,
            "next_absret_after_EXPANSION_bps": round(1e4 * float(np.mean(nxt_exp)), 1),
            "next_absret_after_CALM_bps": round(1e4 * float(np.mean(nxt_calm)), 1),
            "expansion_magnitude_ratio": round(float(np.mean(nxt_exp) / np.mean(nxt_calm)), 2),
            "next_uprate_after_EXPANSION": round(float(np.mean(up_exp)), 3),
            "next_uprate_after_CALM": round(float(np.mean(up_calm)), 3),
            "n_expansion": len(nxt_exp), "n_calm": len(nxt_calm),
            "note": "vol-expansion = local 20-bar |ret| > 1.5x its expanding-median (past-only). Tests vol MOMENTUM "
                    "(does a big-vol bar beget a big next move) vs DIRECTION (up-rate ~0.5 => still a coin flip)."}


# -------------------------------------------------------- P12c regime-conditioned AC
def regime_conditioned_ac(cad: str, sample_assets: int = 50) -> dict:
    files = _files(cad)
    if len(files) > sample_assets:
        idx = np.linspace(0, len(files) - 1, sample_assets).astype(int)
        files = [files[i] for i in idx]
    pairs = {0: ([], []), 1: ([], []), 2: ([], [])}   # regime -> (ret[t], ret[t+1])
    for f in files:
        try:
            df = pl.read_parquet(f, columns=["close", "regime_label"])
        except Exception:
            continue
        close = df["close"].to_numpy().astype(float)
        if len(close) < 200 or "regime_label" not in df.columns:
            continue
        ret = np.zeros(len(close)); ret[1:] = np.diff(np.log(np.clip(close, 1e-12, None)))
        rl = df["regime_label"].to_numpy()
        try:
            rl = rl.astype(int)
        except Exception:
            continue
        for t in range(1, len(ret) - 1):
            r = rl[t]
            if 0 <= r < 3 and np.isfinite(ret[t]) and np.isfinite(ret[t + 1]):
                pairs[r][0].append(ret[t]); pairs[r][1].append(ret[t + 1])
    out = {"cadence": cad}
    for r in (0, 1, 2):
        a, b = pairs[r]
        if len(a) > 200 and np.std(a) > 0 and np.std(b) > 0:
            out[f"regime{r}_ac1"] = round(float(np.corrcoef(a, b)[0, 1]), 4)
            out[f"regime{r}_n"] = len(a)
    out["note"] = "AC1 of returns within each regime_label state. <0 = mean-revert, >0 = momentum. regime 0/2 are "
    out["note"] += "below/above-MA per the signature analysis."
    return out


def run_cadence(cad: str) -> dict:
    res = {}
    for name, fn in [("seasonality", seasonality), ("vol_expansion", vol_expansion_setup),
                     ("regime_cond_ac", regime_conditioned_ac)]:
        t0 = time.time()
        try:
            r = fn(cad)
        except Exception as e:
            r = {"cadence": cad, "error": f"{type(e).__name__}: {e}"}
        r["_elapsed_s"] = round(time.time() - t0, 1)
        (OUT / f"cond_{name}_{cad}.json").write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
        res[name] = r
        print(f"  [{cad}] {name}: {r.get('_elapsed_s')}s {'ERROR '+str(r.get('error')) if 'error' in r else 'ok'}", flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadences", default="1d,4h,1h,30m,15m")
    a = ap.parse_args(argv)
    for cad in [c.strip() for c in a.cadences.split(",") if c.strip()]:
        print(f"\n##### CONDITIONAL {cad} #####", flush=True)
        run_cadence(cad)


if __name__ == "__main__":
    raise SystemExit(main())
