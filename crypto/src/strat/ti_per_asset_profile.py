"""src/strat/ti_per_asset_profile.py -- PER-INSTRUMENT all-weather profile of the candidate TIs.

USER /orc 2026-06-16: "different instruments had different performance profiles -- what do those look like?"
Everything so far is u10 fixed-EW (POOLED). This breaks the pool open: for each candidate TI, run the
rolling-from-band INDEPENDENTLY per asset (each asset selects its own config from its own trailing band) and
report per-asset net 2020/2021/2022 -> which coins CARRY the candidate vs DRAG it (concentration/firewall check).

Reuses deep2020_ti_pipeline.{INDICATORS, load_ohlc, load_ohlcv} (the loaded per-asset panels) + the SAME ironed
stack (trail10 -> min_hold -> lag -> vt -> maker) applied PER ASSET, + ti_band_rolling._rolling / _per_year.
NO look-ahead. Long-only spot, maker, 4h. UNSEEN sealed. No emoji.

RWYB: python -m strat.ti_per_asset_profile --tfs 4h
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

import glob                                                                 # noqa: E402
import strat.deep2020_ti_pipeline as TI                                      # noqa: E402
from strat.deep2020_ti_pipeline import INDICATORS                           # noqa: E402
from strat.portfolio_replay import apply_trail_stop, MAKER_RT               # noqa: E402
from strat.structural_fixes import min_hold                                 # noqa: E402
from strat.ma_2020_breakdown import _panel                                  # noqa: E402
from strat.ti_band_rolling import _rolling, _per_year, YEARS, SPAN, _net    # noqa: E402


def _load_one(sym, cad, want_vol):
    """Build ONE named asset dict (mirrors deep2020_ti_pipeline.load_ohlc/ohlcv per-asset construction) over SPAN.
    Returns (A, sym) or None. Keeps the ticker (the universe loader discards it -> can't profile per instrument)."""
    s_ms = pd.Timestamp(SPAN[0]).value // 10**6; e_ms = pd.Timestamp(SPAN[1]).value // 10**6
    vw = TI.VOLWIN[cad]
    if want_vol:
        fs = sorted(glob.glob(f"data/processed/chimera/{cad}/{sym.lower()}*.parquet"))
        if not fs:
            return None
        import polars as pl
        try:
            df = pl.read_parquet(fs[-1], columns=["timestamp", "open", "high", "low", "close",
                                                  "volume", "buy_vol", "sell_vol"]).sort("timestamp")
        except Exception:
            return None
        ms = df["timestamp"].to_numpy()
        e = int(np.searchsorted(ms, e_ms)); s0 = max(0, int(np.searchsorted(ms, s_ms)) - TI.WARMUP); sl = slice(s0, e)
        c2 = df["close"].to_numpy()[sl]; ms2 = ms[sl]
        if len(c2) < 40:
            return None
        win = ms2 >= s_ms
        if win.sum() < 30:
            return None
        ret = np.zeros(len(c2)); ret[1:] = c2[1:] / c2[:-1] - 1.0
        rv = pd.Series(ret).rolling(vw, min_periods=max(3, vw // 3)).std().shift(1).to_numpy()
        A = {"o": df["open"].to_numpy()[sl], "h": df["high"].to_numpy()[sl], "l": df["low"].to_numpy()[sl],
             "c": c2, "vol": df["volume"].to_numpy()[sl], "buy_vol": df["buy_vol"].to_numpy()[sl],
             "sell_vol": df["sell_vol"].to_numpy()[sl], "ret": ret, "win": win,
             "idx": pd.to_datetime(ms2[win], unit="ms"), "rv": rv}
        return A, sym
    try:
        o, h, l, c, ms = _panel(sym, cad)
    except Exception:
        return None
    e = int(np.searchsorted(ms, e_ms)); s0 = max(0, int(np.searchsorted(ms, s_ms)) - TI.WARMUP)
    o2, h2, l2, c2, ms2 = o[s0:e], h[s0:e], l[s0:e], c[s0:e], ms[s0:e]
    if len(c2) < 40:
        return None
    win = ms2 >= s_ms
    if win.sum() < 30:
        return None
    ret = np.zeros(len(c2)); ret[1:] = c2[1:] / c2[:-1] - 1.0
    rv = pd.Series(ret).rolling(vw, min_periods=max(3, vw // 3)).std().shift(1).to_numpy()
    return {"o": o2, "h": h2, "l": l2, "c": c2, "ret": ret, "win": win,
            "idx": pd.to_datetime(ms2[win], unit="ms"), "rv": rv}, sym

OUT = ROOT.parent / "runs" / "periods" / "ALL_WEATHER"
CHARTS = OUT / "charts"
# the candidates to profile per-asset: the 6 tier-A + best MA (one per family + the MA reference)
CANDIDATES = ["MACD", "PSAR", "TSI", "KELTNER", "MFI", "RSI"]
YRS = ["2020_bull", "2021_mixed", "2022_bear"]


def _asset_series(A, ind, params_list, vt):
    """All-config DAILY net for ONE asset (the ironed sleeve applied per asset). DataFrame [date x cfg]."""
    c2, ret, win, idx, rv = A["c"], A["ret"], A["win"], A["idx"], A["rv"]
    mh = ind.get("minhold", 12)
    cols = {}
    for p in params_list:
        held0 = np.asarray(ind["iron"](A, p)).astype(np.int8)
        held = min_hold(apply_trail_stop(held0.copy(), c2, 0.10)[0].astype(np.int8), mh).astype(np.float64)
        pos = np.zeros(len(c2)); pos[1:] = held[:-1]
        if vt is not None:
            pos = pos * np.clip(vt / (np.nan_to_num(rv, nan=vt) + 1e-12), 0.0, 1.0)
        flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
        net = (pos * ret - flips * (MAKER_RT / 2.0))[win]
        s = pd.Series(net, index=idx)
        daily = s.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()
        if len(daily) > 40:
            cols[ind["name"](p)] = daily
    return pd.DataFrame(cols).sort_index() if cols else None


def _asset_buyhold(A):
    s = pd.Series(A["ret"][A["win"]], index=A["idx"])
    return s.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()


def profile(ti_key, tf):
    """Per-asset rolling-from-band net per year + per-asset buy-hold. Returns {sym: {year: net}, '_bh': {...}}."""
    TI.WIN = SPAN; TI.SPLIT = "2022-10-01"
    ind = INDICATORS[ti_key]
    want_vol = ind.get("loader") == "ohlcv"
    _assets, vt = (TI.load_ohlcv if want_vol else TI.load_ohlc)(tf)          # universe-level vt (median rv)
    if not _assets:
        return None
    params_list = list(ind["grid"]())
    out = {}
    for sym in TI.SYMS:
        loaded = _load_one(sym, tf, want_vol)
        if loaded is None:
            continue
        A, sym = loaded
        sdf = _asset_series(A, ind, params_list, vt)
        if sdf is None or sdf.shape[1] < 2:
            continue
        rp, _ = _rolling(sdf, "pick")
        if rp is None:
            continue
        py = _per_year(rp)
        bh = _asset_buyhold(A)
        out[sym] = {y: py.get(y, {}).get("net") for y in YRS}
        out[sym]["_bh22"] = round(_net(bh[(bh.index >= pd.Timestamp(YEARS["2022_bear"][0])) &
                                          (bh.index < pd.Timestamp(YEARS["2022_bear"][1]))]), 1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.ti_per_asset_profile")
    ap.add_argument("--tfs", default="4h")
    ap.add_argument("--candidates", default=",".join(CANDIDATES))
    a = ap.parse_args(argv)
    cands = [c.strip() for c in a.candidates.split(",") if c.strip()]
    allout = {}
    for tf in [t.strip() for t in a.tfs.split(",") if t.strip()]:
        print(f"\n================= per-asset profile @ {tf} =================")
        prof = {}
        for ti in cands:
            p = profile(ti, tf)
            if p:
                prof[ti] = p
        # need the asset set (union)
        syms = sorted({s for ti in prof for s in prof[ti] if not s.startswith("_")})
        # print per-candidate: per-asset 2022-bear net + how many assets all-weather-positive
        print(f"   assets: {syms}\n")
        for ti in cands:
            if ti not in prof:
                continue
            p = prof[ti]
            n22 = {s: p[s].get("2022_bear") for s in syms if s in p}
            allw = sum(1 for s in n22 if n22[s] is not None and n22[s] >= -5 and
                       (p[s].get("2020_bull") or -1) > 0 and (p[s].get("2021_mixed") or -1) > 0)
            vals = [v for v in n22.values() if v is not None]
            carry = sorted(n22.items(), key=lambda kv: kv[1] if kv[1] is not None else -1e9, reverse=True)
            print(f"   {ti:9} 2022-bear net per asset: med {np.median(vals):.1f} / range [{min(vals):.0f},{max(vals):.0f}] "
                  f"| all-weather-positive assets: {allw}/{len(vals)} | best {carry[0][0]}={carry[0][1]} worst {carry[-1][0]}={carry[-1][1]}")
        allout[tf] = prof
        # heatmap: candidate (rows) x asset (cols), color = 2022-bear net
        mat = np.full((len(cands), len(syms)), np.nan)
        for i, ti in enumerate(cands):
            for j, s in enumerate(syms):
                if ti in prof and s in prof[ti]:
                    v = prof[ti][s].get("2022_bear")
                    if v is not None:
                        mat[i, j] = v
        fig, ax = plt.subplots(figsize=(max(10, len(syms) * 1.0), len(cands) * 0.7 + 2))
        im = ax.imshow(mat, cmap="RdYlGn", vmin=-30, vmax=15, aspect="auto")
        ax.set_xticks(range(len(syms))); ax.set_xticklabels(syms, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(cands))); ax.set_yticklabels(cands, fontsize=9)
        for i in range(len(cands)):
            for j in range(len(syms)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"PER-INSTRUMENT 2022-BEAR net % @ {tf} (rolling-from-band, run INDEPENDENTLY per asset). "
                     f"GREEN=preserved/positive, RED=bled. Shows which coins CARRY vs DRAG each candidate.", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.025)
        fig.tight_layout()
        pchart = CHARTS / f"per_asset_2022bear_{tf}.png"
        fig.savefig(pchart, dpi=110); plt.close(fig)
        print(f"   [chart] {pchart}")
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    jp = OUT / f"ti_per_asset_profile_{stamp}.json"
    json.dump({"repro": {"git_sha": sha, "span": SPAN, "candidates": cands}, "by_tf": allout},
              open(jp, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {jp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
