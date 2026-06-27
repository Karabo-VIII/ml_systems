"""TOP-10 CONFIGS PER MA TYPE — individual config performance within each band.

For each MA type (at its deployable TF), evaluate EACH config in the frozen band
INDIVIDUALLY as a standalone single-config engine through the PROD-grade allocator
(inv-vol + 15% cap + regime gate + prod cost), on OOS-2024. Rank, show top 10.

This reveals the DISPERSION inside each band: is the band edge carried by a few strong
configs or broadly shared? And which specific fast/slow crossovers are the workhorses.

Strictly causal, 2025 sealed. Each config = one fast/slow (or 3MA) crossover, confirmed
+ slice-held exactly like the full engine, but voting alone (n=1).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRYPTO = ROOT.parent

import strat.fleet_lab as fl
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.ma_movecatch_engine import select_band_trainval
from strat.ma_type_upgrade import held_cross, _nums, _ema
from strat.canonical_exits import COST_PROD, resolve_held
from strat.ma_slice_engine import btc_regime, OOS_WALL_MS, OOS_START, OOS_2024_END

BARS_PER_DAY = fl.BARS_PER_DAY
OUTDIR = CRYPTO / "runs" / "engines" / "V10_config_breakdown"
TIER_CFG = {"1d": {"lookback": 14, "slice": 14}, "4h": {"lookback": 14, "slice": 7}}


def single_config_held(close, periods, ma_type, lookback_bars, slice_bars):
    """One config -> confirmed + slice-held array (n=1 vote = the raw cross itself)."""
    c = np.asarray(close, float); n = len(c)
    raw = held_cross(c, periods, ma_type).astype(np.int8)
    m = np.isfinite(c)
    conf = np.zeros(n, dtype=bool)
    if m.sum() >= max(5, lookback_bars):
        cseg = c[m]; ema = _ema(cseg, lookback_bars)
        above = cseg > ema; rising = np.concatenate([[False], ema[1:] >= ema[:-1]])
        cs = (raw[m].astype(bool)) & above & rising
        conf[np.where(m)[0]] = cs
    return resolve_held(c, conf, slice_bars)


def eval_panel(held_cols, close_cols, reg, bpd, cap=0.15):
    close = pd.DataFrame(close_cols).sort_index()
    heldp = pd.DataFrame(held_cols).reindex(close.index).fillna(0).astype(np.int8)
    ret = close.pct_change().fillna(0.0).to_numpy()
    H = heldp.to_numpy().astype(float)
    vol = close.pct_change().rolling(max(20, bpd*5), min_periods=10).std().to_numpy()
    dates = close.index; T, A = H.shape
    rg = np.ones(T); ridx = reg.index
    for k, dt in enumerate(dates):
        d1 = pd.Timestamp(dt).normalize(); prior = ridx[ridx <= d1]
        rg[k] = int(reg[prior[-1]]) if len(prior) else 0
    Hg = H * rg[:, None]
    iv = np.where((Hg > 0) & np.isfinite(vol) & (vol > 0), 1.0/np.where(vol > 0, vol, np.nan), 0.0)
    iv = np.nan_to_num(iv); s = iv.sum(axis=1, keepdims=True)
    w = np.where(s > 0, iv/np.maximum(s, 1e-12), 0.0); w = np.minimum(w, cap)
    Wl = np.vstack([np.zeros((1, A)), w[:-1]])
    gr = np.nansum(Wl*ret, axis=1)
    turn = np.nansum(np.abs(w - np.vstack([np.zeros((1, A)), w[:-1]])), axis=1)
    net = gr - turn*(COST_PROD/2)
    mask = (dates >= pd.Timestamp(OOS_START)) & (dates < pd.Timestamp(OOS_2024_END))
    nw = net[mask]; g = np.nansum(np.abs(w), axis=1)[mask]
    lin = float(np.mean(nw))*bpd*7*100
    sh = (float(np.mean(nw))*bpd*365)/(float(np.std(nw))*np.sqrt(bpd*365)) if np.std(nw) > 1e-12 else 0
    dep = float(np.mean((H[mask].sum(axis=1) > 0))) if mask.sum() else 0
    return round(lin, 3), round(sh, 2), round(dep, 3)


def run_type(ma_type, tf):
    cfg = TIER_CFG[tf]; bpd = BARS_PER_DAY[tf]
    lb, sl = cfg["lookback"]*bpd, cfg["slice"]*bpd
    band = select_band_trainval(ma_type, tf)[0]
    reg = btc_regime(OOS_WALL_MS)
    # load closes once
    closes = {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, OOS_WALL_MS)
        if rec is None:
            continue
        d, o, h, l, c = rec
        closes[sym] = (d, np.asarray(c, float))
    # evaluate each config individually
    rows = []
    for cname in band:
        periods = _nums(cname)
        held_cols, close_cols = {}, {}
        for sym, (d, c) in closes.items():
            held_cols[sym] = pd.Series(single_config_held(c, periods, ma_type, lb, sl), index=d)
            close_cols[sym] = pd.Series(c, index=d)
        lin, sh, dep = eval_panel(held_cols, close_cols, reg, bpd)
        rows.append({"config": cname, "periods": periods, "lin_wk": lin, "sharpe": sh, "deploy": dep})
    rows.sort(key=lambda r: -r["lin_wk"])
    return rows, len(band)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="SMA,EMA,WMA,VIDYA,KAMA,DEMA,HMA,TEMA")
    ap.add_argument("--tf", default="1d")
    a = ap.parse_args()
    types = [t.strip().upper() for t in a.types.split(",")]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    allout = {}
    for ma in types:
        rows, nband = run_type(ma, a.tf)
        allout[ma] = rows
        top = rows[:10]
        pos = sum(1 for r in rows if r["lin_wk"] > 0)
        print(f"\n=== {ma}|{a.tf} — TOP 10 of {nband} band configs ({pos} positive) ===")
        print(f"  {'rank':<5}{'config':<16}{'lin/wk':>8}{'Sharpe':>8}{'deploy':>8}")
        for i, r in enumerate(top, 1):
            print(f"  {i:<5}{r['config']:<16}{r['lin_wk']:>+7.2f}%{r['sharpe']:>8.2f}{r['deploy']:>7.0%}")
        band_lin = np.mean([r["lin_wk"] for r in rows])
        top10_lin = np.mean([r["lin_wk"] for r in top])
        print(f"  band-avg lin/wk {band_lin:+.2f}% | top10-avg {top10_lin:+.2f}% | "
              f"best {rows[0]['lin_wk']:+.2f}% worst {rows[-1]['lin_wk']:+.2f}%")
    with open(OUTDIR / f"config_breakdown_{a.tf}_{ts}.json", "w") as f:
        json.dump({"tf": a.tf, "by_type": allout, "ts": ts}, f, indent=2, default=str)
    print(f"\n  Saved config_breakdown_{a.tf}_{ts}.json")


if __name__ == "__main__":
    main()
