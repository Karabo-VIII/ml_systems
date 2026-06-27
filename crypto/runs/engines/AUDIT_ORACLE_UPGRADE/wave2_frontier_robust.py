"""WAVE 2 robustness gates on the concentration frontier -- is the de-leaked cap->EW lift a
STRUCTURAL recoverable edge, or window-specific concentration luck?

Two skeptical checks (both DE-LEAKED, prod cost, long-only):
 1. CROSS-WINDOW: run the same cap frontier on VAL (a different window than OOS-2024). The cap level
    is NOT a parameter selected on VAL (only the frozen band was), so EW-vs-cap is a clean relative
    test. If EW > cap on BOTH windows, "loosen the cap" is structural; if only on OOS, it's luck.
 2. FRAGILITY: decompose the cap15 -> EW return gap by single-name contribution. If the gap is driven
    by the top-few (name) total contributions, the extra return is concentration/tail-risk, not broad.

VAL is optimistically biased (band chosen VAL-positive), but that bias hits every cap equally, so the
RELATIVE cap-vs-EW comparison is unbiased. 2025 SEALED.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys, json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(ROOT))

import strat.fleet_lab as fl
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.ma_slice_engine import (held_rolling, _cached_band_held, btc_regime,
    OOS_WALL_MS, OOS_START, OOS_2024_END, VAL_WALL_MS, VAL_START, VAL_END)
from strat.canonical_exits import COST_PROD

BARS_PER_DAY = fl.BARS_PER_DAY
TIER_CFG = {"1d": {"lookback": 14, "slice": 14}, "4h": {"lookback": 14, "slice": 7}}
TYPES = ["SMA", "EMA", "WMA", "VIDYA", "KAMA"]
WIN = {"OOS": (OOS_WALL_MS, OOS_START, OOS_2024_END), "VAL": (VAL_WALL_MS, VAL_START, VAL_END)}


def build(tf, cap, window, want_contrib=False):
    wall, ws, we = WIN[window]
    cfg = TIER_CFG[tf]; bpd = BARS_PER_DAY[tf]
    lb, sl = cfg["lookback"]*bpd, cfg["slice"]*bpd
    reg = btc_regime(wall).shift(1).fillna(0).astype(int)   # DE-LEAKED
    closes = {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, wall)
        if rec is None:
            continue
        d, o, h, l, c = rec
        closes[sym] = (d, o, h, l, np.asarray(c, float))
    syms = list(closes.keys())
    close_df = pd.DataFrame({s: pd.Series(closes[s][4], index=closes[s][0]) for s in syms}).sort_index()
    if window == "OOS":
        assert close_df.index.max() < pd.Timestamp("2025-01-01"), "2025 LEAK"
    idx = close_df.index; T = len(idx); A = len(syms)
    conv = np.zeros((T, A))
    for ma in TYPES:
        hc = {}
        for s in syms:
            d, o, h, l, c = closes[s]
            bh = _cached_band_held(ma, tf, s, o, h, l, c, wall)
            hc[s] = pd.Series(held_rolling(c, bh, lb, sl), index=d)
        hd = pd.DataFrame(hc).reindex(idx).fillna(0).reindex(columns=close_df.columns).fillna(0).to_numpy()
        conv += hd
    ret = close_df.pct_change().fillna(0.0).to_numpy()
    vol = close_df.pct_change().rolling(max(20, bpd*5), min_periods=10).std().to_numpy()
    rg = np.ones(T); ridx = reg.index
    for k, dt in enumerate(idx):
        d1 = pd.Timestamp(dt).normalize(); prior = ridx[ridx <= d1]
        rg[k] = int(reg[prior[-1]]) if len(prior) else 0
    mult = (conv/len(TYPES)) * rg[:, None]
    iv = np.where((mult > 0) & np.isfinite(vol) & (vol > 0), 1.0/np.where(vol > 0, vol, np.nan), 0.0)
    iv = np.nan_to_num(iv) * mult
    s = iv.sum(axis=1, keepdims=True)
    w = np.minimum(np.where(s > 0, iv/np.maximum(s, 1e-12), 0.0), cap)
    Wl = np.vstack([np.zeros((1, A)), w[:-1]])
    contrib_bar = Wl * ret                       # per (bar,name) gross contribution
    gr = np.nansum(contrib_bar, axis=1)
    turn = np.nansum(np.abs(w - np.vstack([np.zeros((1, A)), w[:-1]])), axis=1)
    net = gr - turn*(COST_PROD/2)
    mask = (idx >= pd.Timestamp(ws)) & (idx < pd.Timestamp(we))
    bpd_ = bpd
    res = {"net": net[mask], "bpd": bpd_}
    if want_contrib:
        # total per-NAME gross contribution over the window (for fragility decomposition)
        res["name_contrib"] = np.nansum(contrib_bar[mask], axis=0)   # length A
        res["syms"] = syms
    return res


def met(net, bpd):
    lin = float(np.mean(net))*bpd*7*100
    cum = np.cumprod(1+net)
    dd = float(np.min((cum-np.maximum.accumulate(cum))/np.maximum.accumulate(cum)))*100 if len(cum)>1 else 0
    sh = (float(np.mean(net))*bpd*365)/(float(np.std(net))*np.sqrt(bpd*365)) if np.std(net)>1e-12 else 0
    return round(lin,3), round(sh,3), round(dd,1)


def main():
    caps = [0.15, 0.33, 0.50, 1.0]
    out = {"check": "WAVE2_frontier_robustness", "ts": datetime.now().strftime("%Y%m%d_%H%M%S"), "windows": {}}
    print("="*86)
    print("CROSS-WINDOW cap frontier (DE-LEAKED, prod) -- is loosen-the-cap structural or OOS-luck?")
    print("="*86)
    for tf in ["1d", "4h"]:
        out["windows"][tf] = {}
        for window in ["VAL", "OOS"]:
            print(f"\n  {tf} / {window}:   {'cap':>5}{'lin_wk':>9}{'Sharpe':>8}{'maxDD':>8}")
            cells = {}
            for cap in caps:
                r = build(tf, cap, window)
                lin, sh, dd = met(r["net"], r["bpd"])
                lbl = "EW" if cap >= 1.0 else f"{int(cap*100)}%"
                cells[lbl] = {"lin_wk": lin, "sharpe": sh, "maxdd": dd}
                print(f"            {lbl:>5}{lin:>+9.3f}{sh:>+8.2f}{dd:>+8.1f}")
            # EW vs cap15 lift on this window
            lift = round(cells["EW"]["lin_wk"] - cells["15%"]["lin_wk"], 3)
            print(f"            -> EW-minus-cap15 lift = {lift:+.3f} pp/wk")
            cells["EW_minus_cap15_lift_pp"] = lift
            out["windows"][tf][window] = cells

    # FRAGILITY: cap15 -> EW gap by single-name contribution (OOS, 1d)
    print("\n" + "="*86)
    print("FRAGILITY -- is the cap15->EW lift broad or a few-name concentration? (OOS-2024, 1d)")
    print("="*86)
    frag = {}
    for tf in ["1d", "4h"]:
        c15 = build(tf, 0.15, "OOS", want_contrib=True)
        cEW = build(tf, 1.0, "OOS", want_contrib=True)
        gap = cEW["name_contrib"] - c15["name_contrib"]      # extra gross contribution per name from un-capping
        syms = cEW["syms"]
        order = np.argsort(-gap)                              # most-positive extra contribution first
        tot = float(np.sum(gap[gap > 0]))                    # total positive extra return
        topk = {}
        for k in [1, 3, 5]:
            share = float(np.sum(gap[order[:k]])) / tot if tot > 1e-12 else 0
            topk[f"top{k}_share_of_positive_lift"] = round(share, 3)
        top5names = [(syms[i], round(float(gap[i])*100, 2)) for i in order[:5]]
        print(f"  {tf}: top1/top3/top5 share of the positive un-capping lift = "
              f"{topk['top1_share_of_positive_lift']:.0%}/{topk['top3_share_of_positive_lift']:.0%}/"
              f"{topk['top5_share_of_positive_lift']:.0%}  | top5 names (extra gross %): {top5names}")
        frag[tf] = {**topk, "top5_names_extra_gross_pct": top5names,
                    "n_names_with_positive_lift": int((gap > 1e-9).sum())}
    out["fragility"] = frag
    with open(HERE / "wave2_frontier_robust.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved {HERE/'wave2_frontier_robust.json'}")


if __name__ == "__main__":
    main()
