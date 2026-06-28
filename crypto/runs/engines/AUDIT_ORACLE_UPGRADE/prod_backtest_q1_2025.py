"""PROD-GRADE backtest -- BTC-core + breadth-gated-alt-satellite, primary window Q1-2025.

User (/orc 2026-06-28): "solve for this [BTC-core + breadth-gated-alt-satellite] + give me a
PROD-GRADE backtest for Q1 2025." Elevated "prod grade" to its SOTA meaning (not just cost):

  PROD-GRADE REALISM (what this harness enforces):
   - NO LOOK-AHEAD: breadth gate de-leaked (regime known at t applied to t+1), all signals causal.
   - NEXT-OPEN FILLS: decide at close[t], EXECUTE at open[t+1]; P&L measured OPEN-TO-OPEN (you act on
     the open after your signal, not the magic close). Close-to-close also reported as the reference.
   - POINT-IN-TIME universe: _load_asset returns only post-listing bars (no survivorship/pre-listing).
   - DIFFERENTIAL SLIPPAGE: BTC core fills tight (liquid); alt satellite pays wider spread (COST_PROD,
     + a conservative 1.5x variant) -- the liquidity-cost asymmetry without fabricating an ADV filter.
   - TURNOVER costed every rebalance. long-only spot lev=1.
   DATA CAVEAT: the volume feed is unreliable (BTC 1d shows ~26 "coins" = ~$2M, off ~4 orders) so a
   dollar-ADV liquidity filter is NOT built; the u100 top-100 list IS the liquidity screen. Flagged.

  STRUCTURE: total = w_btc * BTC-core + (1-w_btc) * alt-satellite.
   - BTC-core: HODL (always long BTC) OR breadth-gated BTC (to cash when gate off). Both tested.
   - Alt-satellite: the VALIDATED de-risk core = cross-type conv band + cross-sectional BREADTH gate +
     EW (cap33), NO momentum (momentum was refuted on the Q1-2025 unseal). Liquidity = u100 screen.
   Sweep w_btc in {0,.25,.5,.75,1}. Windows: Q1-2025 (primary) + VAL + OOS-2024. Benchmarks: BH BTC,
   BH alt-basket (EW u100). Ranked by COMPOUND wealth; maxDD reported (retail survivability).
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
from strat.ma_slice_engine import held_rolling, _cached_band_held
from strat.ma_slice_engine import OOS_START, OOS_2024_END, VAL_START, VAL_END
from strat.canonical_exits import COST_PROD, COST_TAKER

BARS_PER_DAY = fl.BARS_PER_DAY
TYPES = ["SMA", "EMA", "WMA", "VIDYA", "KAMA"]
LB_D, SL_D, CAP = 14, 14, 0.33
BTC_COST = COST_TAKER          # BTC core: liquid, taker (0.0024 RT)
ALT_COST = COST_PROD           # alt satellite: prod (taker+slip, 0.0028 RT); 1.5x variant tested too
# windows: (wall_ms, start, end)
WIN = {
    "VAL":  (int(pd.Timestamp(VAL_END).value//10**6), VAL_START, VAL_END),
    "OOS24":(int(pd.Timestamp("2025-01-01").value//10**6), OOS_START, OOS_2024_END),
    "Q1_25":(int(pd.Timestamp("2025-04-01").value//10**6), "2025-01-01", "2025-04-01"),
}


def hyst(sig, up, dn):
    """up/dn may be scalars or per-bar arrays (BTC vs its own moving SMA*1.02/0.98)."""
    n = len(sig); s = np.zeros(n, dtype=int); st = 0
    up = np.full(n, up, float) if np.isscalar(up) else np.asarray(up, float)
    dn = np.full(n, dn, float) if np.isscalar(dn) else np.asarray(dn, float)
    for i in range(n):
        if not (np.isfinite(sig[i]) and np.isfinite(up[i]) and np.isfinite(dn[i])): s[i] = st; continue
        if st == 0 and sig[i] > up[i]: st = 1
        elif st == 1 and sig[i] < dn[i]: st = 0
        s[i] = st
    return s


def build(window, tf="1d"):
    wall, ws, we = WIN[window]; bpd = BARS_PER_DAY[tf]; lb, sl = LB_D*bpd, SL_D*bpd; sw = 100*bpd
    closes, opens = {}, {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, wall)
        if rec is None: continue
        d, o, h, l, c = rec
        closes[sym] = pd.Series(np.asarray(c, float), index=d)
        opens[sym] = pd.Series(np.asarray(o, float), index=d)
    cdf = pd.DataFrame(closes).sort_index(); odf = pd.DataFrame(opens).reindex(cdf.index).reindex(columns=cdf.columns)
    idx = cdf.index; T = len(idx); A = len(cdf.columns)
    conv = np.zeros((T, A))
    for ma in TYPES:
        hc = {}
        for s in cdf.columns:
            rec = _load_asset(s, tf, wall)
            if rec is None: continue
            d, o, h, l, c = rec
            bh = _cached_band_held(ma, tf, s, o, h, l, c, wall)
            hc[s] = pd.Series(held_rolling(c, bh, lb, sl), index=d)
        conv += pd.DataFrame(hc).reindex(idx).fillna(0).reindex(columns=cdf.columns).fillna(0).to_numpy()
    C = cdf.to_numpy(); O = odf.to_numpy()
    vol = cdf.pct_change().rolling(max(20, bpd*5), min_periods=10).std().to_numpy()
    # OPEN-TO-OPEN return for bar i = open[i+1]/open[i]-1 (held from open[i] to open[i+1])
    roo = np.full((T, A), 0.0)
    roo[:-1] = O[1:]/np.where(O[:-1] > 0, O[:-1], np.nan) - 1.0
    roo = np.nan_to_num(roo)
    rcc = cdf.pct_change().fillna(0.0).to_numpy()   # close-to-close reference
    # breadth gate (de-leaked, causal: from closes up to t, applied via lag-1 in net)
    sall = cdf.rolling(sw, min_periods=sw//2).mean().to_numpy(); vc = np.isfinite(sall).sum(axis=1)
    brd = np.where(vc > 0, np.nansum((C > sall)*np.isfinite(sall), axis=1)/np.maximum(vc,1), np.nan)
    g_brd = hyst(brd, 0.5, 0.4)
    # btc gate (single-asset, de-leaked) for comparison
    btc = "BTCUSDT" if "BTCUSDT" in cdf.columns else cdf.columns[0]
    bc = cdf[btc]; bsma = bc.rolling(sw, min_periods=sw//2).mean().to_numpy()
    g_btc = hyst(bc.to_numpy(), bsma*1.02, bsma*0.98)
    j_btc = list(cdf.columns).index(btc)
    mask = (idx >= pd.Timestamp(ws)) & (idx < pd.Timestamp(we))
    return {"cdf":cdf,"odf":odf,"conv":conv,"vol":vol,"roo":roo,"rcc":rcc,"g_brd":g_brd,"g_btc":g_btc,
            "j_btc":j_btc,"mask":mask,"bpd":bpd,"A":A,"idx":idx}


def net_from_W(W, ret, cost):
    """lag-1: weight decided at close[t] applied to the open[t+1]->open[t+2] return = ret[t+1]."""
    Wl = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])
    gr = np.nansum(Wl*ret, axis=1)
    turn = np.nansum(np.abs(W - np.vstack([np.zeros((1, W.shape[1])), W[:-1]])), axis=1)
    return gr - turn*(cost/2)


def alt_sleeve_W(f, gate):
    conv, vol = f["conv"], f["vol"]
    mult = (conv/len(TYPES)) * (gate[:, None] > 0)
    iv = np.where((mult > 0) & np.isfinite(vol) & (vol > 0), 1.0/np.where(vol > 0, vol, np.nan), 0.0)
    iv = np.nan_to_num(iv)*mult; s = iv.sum(axis=1, keepdims=True)
    return np.minimum(np.where(s > 0, iv/np.maximum(s, 1e-12), 0.0), CAP)


def btc_sleeve_net(f, ret, gated):
    T, A = f["conv"].shape; W = np.zeros((T, A))
    g = f["g_brd"] if gated else np.ones(T, dtype=int)
    W[:, f["j_btc"]] = (g > 0).astype(float)
    return net_from_W(W, ret, BTC_COST)


def bh_net(f, ret, which):
    T, A = f["conv"].shape
    if which == "btc":
        W = np.zeros((T, A)); W[:, f["j_btc"]] = 1.0
        return net_from_W(W, ret, BTC_COST)
    live = np.isfinite(f["cdf"].to_numpy()); cnt = live.sum(axis=1, keepdims=True)
    W = np.where(live & (cnt > 0), 1.0/np.maximum(cnt, 1), 0.0)
    return net_from_W(W, ret, ALT_COST)


def metrics(n, f):
    n = n[f["mask"]]
    if len(n) == 0: return None
    comp = float(np.prod(1+n)-1)*100; lin = float(np.mean(n))*f["bpd"]*7*100
    cum = np.cumprod(1+n)
    dd = float(np.min((cum-np.maximum.accumulate(cum))/np.maximum.accumulate(cum)))*100 if len(cum)>1 else 0
    sh = (float(np.mean(n))*f["bpd"]*365)/(float(np.std(n))*np.sqrt(f["bpd"]*365)) if np.std(n)>1e-12 else 0
    return {"compound":round(comp,1),"lin_wk":round(lin,3),"sharpe":round(sh,2),"maxdd":round(dd,1)}


def main():
    out = {"check":"PROD_GRADE_btc_core_alt_satellite","ts":datetime.now().strftime("%Y%m%d_%H%M%S"),
           "exec":"next-open fills (open-to-open, lag-1)","cost":f"BTC {BTC_COST} / ALT {ALT_COST} RT",
           "caveat":"volume feed unreliable -> no ADV filter; u100 top-100 IS the liquidity screen","windows":{}}
    for window in ["VAL","OOS24","Q1_25"]:
        f = build(window)
        ret = f["roo"]   # PROD: open-to-open
        bh_btc = metrics(bh_net(f, ret, "btc"), f)
        bh_alt = metrics(bh_net(f, ret, "alt"), f)
        alt_brd = alt_sleeve_W(f, f["g_brd"])
        alt_net = net_from_W(alt_brd, ret, ALT_COST)
        rows = {}
        # benchmarks
        rows["BH BTC"] = bh_btc
        rows["BH alt-basket"] = bh_alt
        rows["alt-satellite alone (breadth)"] = metrics(alt_net, f)
        # BTC-core + alt-satellite sweep (BTC core: HODL and gated)
        for gated, gtag in [(False,"HODL"), (True,"gated")]:
            btc_net = btc_sleeve_net(f, ret, gated)
            for w in [0.0, 0.25, 0.5, 0.75, 1.0]:
                book = w*btc_net + (1-w)*alt_net
                rows[f"core{int(w*100)}%BTC({gtag})+alt"] = metrics(book, f)
        # alt at conservative 1.5x slippage (sensitivity)
        alt_net_hi = net_from_W(alt_brd, ret, ALT_COST*1.5)
        rows["alt-satellite (1.5x slip)"] = metrics(alt_net_hi, f)
        out["windows"][window] = rows
        wl = WIN[window]
        print(f"\n{'='*92}\n{window}  ({wl[1]}..{wl[2]})  PROD-GRADE (next-open fills, BTC {BTC_COST}/ALT {ALT_COST})  [by WEALTH]")
        print(f"{'='*92}")
        print(f"  {'config':<30}{'compound':>10}{'lin_wk':>8}{'Sharpe':>7}{'maxDD':>7}")
        print("  "+"-"*72)
        for name,m in sorted(rows.items(), key=lambda kv:-kv[1]["compound"]):
            print(f"  {name:<30}{m['compound']:>+9.1f}%{m['lin_wk']:>+8.2f}{m['sharpe']:>+7.2f}{m['maxdd']:>+7.0f}")
    with open(HERE/"prod_backtest_q1_2025.json","w") as fo:
        json.dump(out, fo, indent=2, default=str)
    print(f"\nSaved {HERE/'prod_backtest_q1_2025.json'}")


if __name__ == "__main__":
    main()
