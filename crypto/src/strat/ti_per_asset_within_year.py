"""src/strat/ti_per_asset_within_year.py -- PER-INSTRUMENT within-year (2020/2021) profile of the candidate TIs.

USER /orc 2026-06-16: "different instruments had different performance profiles -- what do those look like?"
(answered IN SCOPE: 2020/2021 only, NO 2022). For each candidate TI, per asset, build the 6/3/3 robust band on
THAT asset (TRAIN&VAL>0), EW the band, report OOS net per year -> which coins CARRY the candidate vs DRAG it
(the concentration/firewall check the pooled u10 book hides).

Reuses ti_per_asset_profile.{_load_one, _asset_series} (the per-asset ironed stack) + the 6/3/3 split. Long-only
spot, fixed-EW within an asset's band, maker. NO look-ahead (band = TRAIN&VAL; OOS held out). No 2022. No emoji.

RWYB: python -m strat.ti_per_asset_within_year --tfs 4h,1d
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.deep2020_ti_pipeline as TI                                      # noqa: E402
from strat.deep2020_ti_pipeline import INDICATORS                           # noqa: E402
from strat.ti_per_asset_profile import _load_one, _asset_series             # noqa: E402

DEST = ROOT.parent / "runs" / "periods" / "TRAIN" / "2021" / "DEEP_DIVE"
CHARTS = DEST / "charts"
# candidate set (cross-year emergers, per family); profile their per-instrument dispersion
CANDIDATES = ["MACD", "SUPERTREND", "ADX", "PSAR", "TSI", "ROC", "KELTNER", "DONCHIAN", "VOLIMB", "OBV", "RSI", "CCI"]
YEARS = [2020, 2021]


def _split(year):
    return {"TRAIN": (f"{year}-01-01", f"{year}-07-01"), "VAL": (f"{year}-07-01", f"{year}-10-01"),
            "OOS": (f"{year}-10-01", f"{year + 1}-01-01")}


def _seg_net(daily, lo, hi):
    s = daily[(daily.index >= pd.Timestamp(lo)) & (daily.index < pd.Timestamp(hi))].dropna()
    return float(np.prod(1 + s.to_numpy()) - 1) * 100 if len(s) > 1 else None


def _asset_band_oos(sdf, year):
    """Per-asset: robust band = configs with TRAIN>0 AND VAL>0; EW the band; return OOS net (held out). None if no band."""
    sp = _split(year)
    band = []
    for c in sdf.columns:
        tr = _seg_net(sdf[c], *sp["TRAIN"]); va = _seg_net(sdf[c], *sp["VAL"])
        if tr is not None and va is not None and tr > 0 and va > 0:
            band.append(c)
    if not band:
        return None, 0
    ens = sdf[band].mean(axis=1)
    return _seg_net(ens, *sp["OOS"]), len(band)


def profile(ti_key, tf, year):
    """{sym: oos_net} for the per-asset band-ensemble in `year`."""
    TI.WIN = (f"{year}-01-01", f"{year + 1}-01-01"); TI.SPLIT = f"{year}-10-01"
    ind = INDICATORS[ti_key]
    want_vol = ind.get("loader") == "ohlcv"
    params = list(ind["grid"]())
    # universe vt
    _a, vt = (TI.load_ohlcv if want_vol else TI.load_ohlc)(tf)
    out = {}
    for sym in TI.SYMS:
        loaded = _load_one(sym, tf, want_vol)   # NOTE: _load_one spans SPAN(2020-2023); we slice per-year below
        if loaded is None:
            continue
        A, sym = loaded
        sdf = _asset_series(A, ind, params, vt)
        if sdf is None or sdf.shape[1] < 2:
            continue
        oos, nband = _asset_band_oos(sdf, year)
        out[sym] = oos
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.ti_per_asset_within_year")
    ap.add_argument("--tfs", default="4h")
    ap.add_argument("--candidates", default=",".join(CANDIDATES))
    a = ap.parse_args(argv)
    cands = [c.strip() for c in a.candidates.split(",") if c.strip()]
    allout = {}
    for tf in [t.strip() for t in a.tfs.split(",") if t.strip()]:
        print(f"\n================= per-instrument within-year @ {tf} =================")
        prof = {}                                                            # ti -> year -> {sym: oos}
        for ti in cands:
            prof[ti] = {y: profile(ti, tf, y) for y in YEARS}
        syms = sorted({s for ti in prof for y in YEARS for s in prof[ti][y]})
        print(f"   assets: {syms}\n")
        for ti in cands:
            for y in YEARS:
                p = prof[ti][y]
                vals = [v for v in p.values() if v is not None]
                if not vals:
                    print(f"   {ti:10} {y}: no band on any asset"); continue
                pos = sum(1 for v in vals if v > 0)
                best = max(p.items(), key=lambda kv: kv[1] if kv[1] is not None else -1e9)
                worst = min(p.items(), key=lambda kv: kv[1] if kv[1] is not None else 1e9)
                print(f"   {ti:10} {y}: OOS net per asset med {np.median(vals):.1f} / [{min(vals):.0f},{max(vals):.0f}] "
                      f"| positive {pos}/{len(vals)} | carry {best[0]}={best[1]} drag {worst[0]}={worst[1]}")
        allout[tf] = {ti: {str(y): prof[ti][y] for y in YEARS} for ti in cands}
        # chart: 2021-OOS net per (TI x asset) heatmap
        mat = np.full((len(cands), len(syms)), np.nan)
        for i, ti in enumerate(cands):
            for j, s in enumerate(syms):
                v = prof[ti][2021].get(s)
                if v is not None:
                    mat[i, j] = v
        fig, ax = plt.subplots(figsize=(max(10, len(syms)), len(cands) * 0.6 + 2))
        im = ax.imshow(mat, cmap="RdYlGn", vmin=-30, vmax=30, aspect="auto")
        ax.set_xticks(range(len(syms))); ax.set_xticklabels(syms, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(cands))); ax.set_yticklabels(cands, fontsize=9)
        for i in range(len(cands)):
            for j in range(len(syms)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"PER-INSTRUMENT 2021-OOS net % @ {tf} (per-asset band-ensemble, 6/3/3). GREEN=carries, "
                     f"RED=drags. Reveals concentration the pooled u10 book hides.", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.025)
        fig.tight_layout()
        CHARTS.mkdir(parents=True, exist_ok=True)
        pc = CHARTS / f"per_asset_within_year_2021oos_{tf}.png"
        fig.savefig(pc, dpi=110); plt.close(fig)
        print(f"   [chart] {pc}")
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    jp = ROOT.parent / "runs" / "periods" / "TRAIN" / "2021" / "DEEP_DIVE" / f"ti_per_asset_within_year_{stamp}.json"
    json.dump({"repro": {"git_sha": sha, "candidates": cands}, "by_tf": allout}, open(jp, "w"), indent=1, default=str)
    print(f"\n[persisted] {jp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
