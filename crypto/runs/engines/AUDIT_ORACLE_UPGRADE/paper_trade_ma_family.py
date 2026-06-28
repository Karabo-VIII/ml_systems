"""PAPER-TRADE the official per-MA-type candidates forward on the REST of OOS (2025-Q2 -> 2026-05).

User (/orc 2026-06-28): official wiring candidates PER MA TYPE, wired to a paper trader, run for
2025-Q2+ (the rest of the unseen OOS window; data runs to ~2026-05-28). The MA family on the FULL
u100 is the product (NOT BTC -- anti-collapse).

WIRING NOTE: the existing wealth_bot/bot paper trader is SINGLE-POSITION (one non-overlapping trade
at a time) -- built for the (TI,ASSET) dossier framework. The MA family is a CROSS-SECTIONAL u100
PORTFOLIO (8-30 concurrent names). Forcing it into the single-position runner would collapse it to
single-asset (the exact error just corrected). So this is the PORTFOLIO paper-trade runner -- it
reuses the wealth_bot prod CONVENTIONS (next-open fills, taker+slip cost, journal/telemetry, $-equity)
in the correct cross-sectional architecture.

OFFICIAL PER-TYPE CANDIDATE (canonical spec, locked): u100 full universe, 1d bars, 14d lookback /
7d trading window, ROLLING slices, cross-sectional BREADTH gate (frac>own 100d-SMA, hyst 0.5/0.4),
inv-vol weights @ cap33, EW within, NO momentum. One candidate per MA type {SMA,EMA,WMA,VIDYA,KAMA}
+ the cross-type CONVERGENCE book. PROD-GRADE: next-open fills (open-to-open), prod cost (0.0028 RT),
de-leaked gate, PIT listing universe, turnover costed. long-only spot lev=1. $10,000 start.
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
from strat.canonical_exits import COST_PROD

BARS_PER_DAY = fl.BARS_PER_DAY
TYPES = ["SMA", "EMA", "WMA", "VIDYA", "KAMA"]
LB_D, SL_D, CAP, TF = 14, 7, 0.33, "1d"
START_CAP = 10_000.0
FWD_START, FWD_END = "2025-04-01", "2026-07-01"      # Q2-2025 .. data end (~2026-05-28)
WALL = int(pd.Timestamp("2026-07-01").value // 10**6)


def hyst(sig, up, dn):
    n = len(sig); s = np.zeros(n, dtype=int); st = 0
    for i in range(n):
        if not np.isfinite(sig[i]): s[i] = st; continue
        if st == 0 and sig[i] > up: st = 1
        elif st == 1 and sig[i] < dn: st = 0
        s[i] = st
    return s


def build():
    bpd = BARS_PER_DAY[TF]; lb, sl = LB_D*bpd, SL_D*bpd; sw = 100*bpd
    closes, opens = {}, {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, TF, WALL)
        if rec is None: continue
        d, o, h, l, c = rec
        closes[sym] = pd.Series(np.asarray(c, float), index=d); opens[sym] = pd.Series(np.asarray(o, float), index=d)
    cdf = pd.DataFrame(closes).sort_index(); odf = pd.DataFrame(opens).reindex(cdf.index).reindex(columns=cdf.columns)
    idx = cdf.index; T = len(idx); A = len(cdf.columns)
    typ_held = {}; conv = np.zeros((T, A))
    for ma in TYPES:
        hc = {}
        for s in cdf.columns:
            rec = _load_asset(s, TF, WALL)
            if rec is None: continue
            d, o, h, l, c = rec
            bh = _cached_band_held(ma, TF, s, o, h, l, c, WALL)
            hc[s] = pd.Series(held_rolling(c, bh, lb, sl), index=d)
        H = pd.DataFrame(hc).reindex(idx).fillna(0).reindex(columns=cdf.columns).fillna(0).to_numpy()
        typ_held[ma] = H; conv += H
    C = cdf.to_numpy(); O = odf.to_numpy()
    vol = cdf.pct_change().rolling(max(20, bpd*5), min_periods=10).std().to_numpy()
    roo = np.full((T, A), 0.0); roo[:-1] = O[1:]/np.where(O[:-1] > 0, O[:-1], np.nan) - 1.0; roo = np.nan_to_num(roo)
    sall = cdf.rolling(sw, min_periods=sw//2).mean().to_numpy(); vc = np.isfinite(sall).sum(axis=1)
    brd = np.where(vc > 0, np.nansum((C > sall)*np.isfinite(sall), axis=1)/np.maximum(vc,1), np.nan)
    g_brd = hyst(brd, 0.5, 0.4)
    btc = "BTCUSDT" if "BTCUSDT" in cdf.columns else cdf.columns[0]; j_btc = list(cdf.columns).index(btc)
    mask = (idx >= pd.Timestamp(FWD_START)) & (idx < pd.Timestamp(FWD_END))
    return {"cols":list(cdf.columns),"cdf":cdf,"conv":conv,"typ":typ_held,"vol":vol,"roo":roo,
            "g_brd":g_brd,"j_btc":j_btc,"mask":mask,"bpd":bpd,"A":A,"idx":idx}


def weights(f, kind):
    g = (f["g_brd"][:, None] > 0)
    held = (f["conv"]/len(TYPES)) if kind == "conv" else f["typ"][kind].astype(float)
    mult = held * g; vol = f["vol"]
    iv = np.where((mult > 0) & np.isfinite(vol) & (vol > 0), 1.0/np.where(vol > 0, vol, np.nan), 0.0)
    iv = np.nan_to_num(iv)*mult; s = iv.sum(axis=1, keepdims=True)
    return np.minimum(np.where(s > 0, iv/np.maximum(s, 1e-12), 0.0), CAP)


def paper_trade(f, W, cost=COST_PROD):
    """Cross-sectional paper-trade: next-open fills (lag-1), turnover costed -> $ equity + journal."""
    ret = f["roo"]; idx = f["idx"]; cols = f["cols"]
    Wl = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])           # position over [t,t+1] decided at t-1 close
    gr = np.nansum(Wl*ret, axis=1)
    turn = np.nansum(np.abs(W - np.vstack([np.zeros((1, W.shape[1])), W[:-1]])), axis=1)
    net = gr - turn*(cost/2)
    m = f["mask"]; net_w = net[m]; dates_w = idx[m]; W_w = W[m]; Wl_w = Wl[m]
    eq = START_CAP * np.cumprod(1+net_w)
    # journal: rebalance events (held set changes) -- date, n_held, top-5 names+wt, turnover$, equity
    journal = []
    prev = set()
    for i in range(len(dates_w)):
        held_idx = np.where(W_w[i] > 1e-6)[0]
        cur = set(int(j) for j in held_idx)
        if cur != prev:
            top = sorted(held_idx, key=lambda j: -W_w[i][j])[:5]
            journal.append({"date": str(pd.Timestamp(dates_w[i]).date()), "n_held": int(len(held_idx)),
                            "top": [(cols[j], round(float(W_w[i][j]), 3)) for j in top],
                            "turnover": round(float(np.abs(W_w[i]-Wl_w[i]).sum()), 3),
                            "equity": round(float(eq[i]), 0)})
            prev = cur
    return net_w, dates_w, eq, journal


def metrics(net, eq, bpd):
    if len(net) == 0: return None
    comp = float(eq[-1]/START_CAP - 1)*100
    lin = float(np.mean(net))*bpd*7*100
    dd = float(np.min((eq-np.maximum.accumulate(eq))/np.maximum.accumulate(eq)))*100 if len(eq)>1 else 0
    sh = (float(np.mean(net))*bpd*365)/(float(np.std(net))*np.sqrt(bpd*365)) if np.std(net)>1e-12 else 0
    sb = 7*bpd; sl=[float(np.prod(1+net[i:i+sb])-1)*100 for i in range(0,len(net)-sb+1,sb)]
    pct7 = sum(1 for x in sl if x>0)/len(sl) if sl else 0
    return {"final_equity":round(float(eq[-1]),0),"compound_pct":round(comp,1),"lin_wk":round(lin,3),
            "sharpe":round(sh,2),"maxdd_pct":round(dd,1),"pct7d":round(pct7,3),"n_bars":len(net)}


def quarterly(net, dates):
    out = {}
    s = pd.Series(net, index=pd.DatetimeIndex(dates))
    for q, g in s.groupby(s.index.to_period("Q")):
        out[str(q)] = round(float(np.prod(1+g.to_numpy())-1)*100, 1)
    return out


def bench_net(f, which):
    ret = f["roo"]
    if which == "btc":
        W = np.zeros((len(f["idx"]), f["A"])); W[:, f["j_btc"]] = 1.0
    else:
        live = np.isfinite(f["cdf"].to_numpy()); cnt = live.sum(axis=1, keepdims=True)
        W = np.where(live & (cnt > 0), 1.0/np.maximum(cnt, 1), 0.0)
    Wl = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])
    gr = np.nansum(Wl*ret, axis=1); turn = np.nansum(np.abs(W-Wl), axis=1)
    net = (gr - turn*(COST_PROD/2))[f["mask"]]
    eq = START_CAP*np.cumprod(1+net)
    return net, eq


def main():
    f = build()
    n_bars = int(f["mask"].sum())
    print(f"{'='*98}\nPAPER-TRADE: MA family per-type, FORWARD on {FWD_START}..{str(pd.Timestamp(f['idx'][f['mask']][-1]).date())} "
          f"({n_bars} 1d bars, unseen OOS)  | $10k start, prod-grade")
    print(f"{'='*98}")
    out = {"check":"PAPER_TRADE_MA_FAMILY_2025Q2plus","spec":"u100, 1d, 14d/7d rolling, breadth-gated, cap33, EW inv-vol, NO momentum",
           "window":f"{FWD_START}..{str(pd.Timestamp(f['idx'][f['mask']][-1]).date())}","n_bars":n_bars,
           "exec":"next-open fills, prod cost 0.0028 RT, de-leaked, PIT, $10k start","ts":datetime.now().strftime("%Y%m%d_%H%M%S"),
           "candidates":{},"benchmarks":{},"journals":{}}
    # benchmarks
    for w in ["btc","alt"]:
        nb, eb = bench_net(f, w); out["benchmarks"][f"BH_{w}"] = metrics(nb, eb, f["bpd"])
    # per-type + conv candidates
    rows = {}
    for kind in TYPES + ["conv"]:
        W = weights(f, kind)
        net, dates, eq, journal = paper_trade(f, W)
        m = metrics(net, eq, f["bpd"]); m["quarterly"] = quarterly(net, dates); m["n_rebalances"] = len(journal)
        nm = ("CONV (cross-type)" if kind == "conv" else kind)
        rows[nm] = m; out["candidates"][nm] = m; out["journals"][nm] = journal[-12:]   # last 12 rebal events
    print(f"\n  {'candidate':<18}{'final$':>10}{'compound':>10}{'lin_wk':>8}{'Sharpe':>7}{'maxDD':>7}{'7dprof':>7}{'rebals':>7}")
    print("  "+"-"*78)
    for nm,m in sorted(rows.items(), key=lambda kv:-kv[1]["compound_pct"]):
        print(f"  {nm:<18}{m['final_equity']:>10.0f}{m['compound_pct']:>+9.1f}%{m['lin_wk']:>+8.2f}{m['sharpe']:>+7.2f}{m['maxdd_pct']:>+7.0f}{m['pct7d']*100:>6.0f}%{m['n_rebalances']:>7}")
    print(f"  {'-- BH alt-basket':<18}{out['benchmarks']['BH_alt']['final_equity']:>10.0f}{out['benchmarks']['BH_alt']['compound_pct']:>+9.1f}%{'':>8}{out['benchmarks']['BH_alt']['sharpe']:>+7.2f}{out['benchmarks']['BH_alt']['maxdd_pct']:>+7.0f}")
    print(f"  {'-- BH BTC (ctx)':<18}{out['benchmarks']['BH_btc']['final_equity']:>10.0f}{out['benchmarks']['BH_btc']['compound_pct']:>+9.1f}%{'':>8}{out['benchmarks']['BH_btc']['sharpe']:>+7.2f}{out['benchmarks']['BH_btc']['maxdd_pct']:>+7.0f}")
    # quarterly breakdown for CONV + best type
    print(f"\n  QUARTERLY compound (CONV cross-type): {out['candidates']['CONV (cross-type)']['quarterly']}")
    print(f"  QUARTERLY BH alt-basket: {quarterly(*[bench_net(f,'alt')[0], f['idx'][f['mask']]])}")
    with open(HERE/"paper_trade_ma_family.json","w") as fo:
        json.dump(out, fo, indent=2, default=str)
    print(f"\nSaved {HERE/'paper_trade_ma_family.json'}")


if __name__ == "__main__":
    main()
