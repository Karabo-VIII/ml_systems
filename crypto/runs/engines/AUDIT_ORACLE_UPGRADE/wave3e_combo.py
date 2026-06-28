"""WAVE 3E -- the SYNTHESIS: breadth gate (cuts drawdown) + momentum top-K inv-vol selection
(raises return-efficiency). Does stacking the two robust levers beat the current deployable on
BOTH windows AND both TFs, without compounding overfit?

2x2: gate {btc_sma100 (current), breadth} x selection {EW-all (current), mom top-5 inv-vol}.
Baseline = btc_sma100 + EW-all (the confirmed cap33 deployable). de-leaked, PROD, 2025 SEALED.
Win = combo beats baseline on return-per-drawdown on BOTH VAL+OOS, with DD no worse than EW-all.
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
from strat.ma_slice_engine import OOS_WALL_MS, OOS_START, OOS_2024_END, VAL_WALL_MS, VAL_START, VAL_END
from strat.canonical_exits import COST_PROD

BARS_PER_DAY = fl.BARS_PER_DAY
TYPES = ["SMA", "EMA", "WMA", "VIDYA", "KAMA"]
WIN = {"OOS": (OOS_WALL_MS, OOS_START, OOS_2024_END), "VAL": (VAL_WALL_MS, VAL_START, VAL_END)}
LB_D, SL_D, CAP, K = 14, 14, 0.33, 5


def hysteresis(sig, up, dn):
    n = len(sig); s = np.zeros(n, dtype=int); st = 0
    for i in range(n):
        if not (np.isfinite(sig[i]) and np.isfinite(up[i]) and np.isfinite(dn[i])):
            s[i] = st; continue
        if st == 0 and sig[i] > up[i]: st = 1
        elif st == 1 and sig[i] < dn[i]: st = 0
        s[i] = st
    return s


def build(window, tf):
    wall, ws, we = WIN[window]; bpd = BARS_PER_DAY[tf]
    lb, sl = LB_D*bpd, SL_D*bpd
    closes = {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, wall)
        if rec is None: continue
        d, o, h, l, c = rec
        closes[sym] = (d, o, h, l, np.asarray(c, float))
    syms = list(closes.keys())
    cdf = pd.DataFrame({s: pd.Series(closes[s][4], index=closes[s][0]) for s in syms}).sort_index()
    idx = cdf.index; T = len(idx); A = len(syms)
    conv = np.zeros((T, A))
    for ma in TYPES:
        hc = {}
        for s in syms:
            d, o, h, l, c = closes[s]
            bh = _cached_band_held(ma, tf, s, o, h, l, c, wall)
            hc[s] = pd.Series(held_rolling(c, bh, lb, sl), index=d)
        conv += pd.DataFrame(hc).reindex(idx).fillna(0).reindex(columns=cdf.columns).fillna(0).to_numpy()
    ret = cdf.pct_change().fillna(0.0).to_numpy()
    vol = cdf.pct_change().rolling(max(20, bpd*5), min_periods=10).std().to_numpy()
    mom = cdf.pct_change(lb).to_numpy()
    C = cdf.to_numpy(); sma_w = 100*bpd
    # gates
    btc = "BTCUSDT" if "BTCUSDT" in cdf.columns else cdf.columns[0]
    bc = cdf[btc]; sma = bc.rolling(sma_w, min_periods=sma_w//2).mean().to_numpy()
    g_btc = hysteresis(bc.to_numpy(), sma*1.02, sma*0.98)
    sma_all = cdf.rolling(sma_w, min_periods=sma_w//2).mean().to_numpy()
    above = (C > sma_all).astype(float); vc = np.isfinite(sma_all).sum(axis=1)
    breadth = np.where(vc > 0, np.nansum(above*np.isfinite(sma_all), axis=1)/np.maximum(vc,1), np.nan)
    g_brd = hysteresis(breadth, np.full(T,0.5), np.full(T,0.4))
    mask = (idx >= pd.Timestamp(ws)) & (idx < pd.Timestamp(we))
    return {"conv": conv, "ret": ret, "vol": vol, "mom": mom, "g_btc": g_btc, "g_brd": g_brd,
            "mask": mask, "bpd": bpd, "A": A}


def weights(f, gate, selection):
    conv, vol, mom = f["conv"], f["vol"], f["mom"]
    elig = (conv > 0) & (gate[:, None] > 0)
    T, A = conv.shape; W = np.zeros((T, A))
    if selection == "ew":
        mult = (conv/len(TYPES)) * (gate[:, None] > 0)
        iv = np.where((mult > 0) & np.isfinite(vol) & (vol > 0), 1.0/np.where(vol > 0, vol, np.nan), 0.0)
        iv = np.nan_to_num(iv)*mult; s = iv.sum(axis=1, keepdims=True)
        W = np.minimum(np.where(s > 0, iv/np.maximum(s, 1e-12), 0.0), CAP)
    else:  # mom top-K inv-vol
        sc = np.where(elig & np.isfinite(mom), mom, -np.inf)
        for t in range(T):
            e = np.where(np.isfinite(sc[t]))[0]
            if len(e) == 0: continue
            k = min(K, len(e)); top = e[np.argsort(-sc[t][e])[:k]]
            v = vol[t][top]; ivw = np.where(np.isfinite(v) & (v > 0), 1.0/v, 0.0)
            W[t, top] = ivw/ivw.sum() if ivw.sum() > 0 else 1.0/k
    return W


def net(W, ret, cost=COST_PROD):
    Wl = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])
    gr = np.nansum(Wl*ret, axis=1)
    turn = np.nansum(np.abs(W - np.vstack([np.zeros((1, W.shape[1])), W[:-1]])), axis=1)
    return gr - turn*(cost/2)


def met(n, bpd, mask):
    n = n[mask]
    if len(n) == 0: return None
    lin = float(np.mean(n))*bpd*7*100
    cum = np.cumprod(1+n)
    dd = float(np.min((cum-np.maximum.accumulate(cum))/np.maximum.accumulate(cum)))*100 if len(cum)>1 else 0
    sh = (float(np.mean(n))*bpd*365)/(float(np.std(n))*np.sqrt(bpd*365)) if np.std(n)>1e-12 else 0
    return {"lin_wk": round(lin,3), "sharpe": round(sh,3), "maxdd": round(dd,1),
            "ret_per_dd": round(lin/abs(dd),4) if abs(dd)>1e-9 else 0}


def main():
    out = {"check": "WAVE3E_combo_breadth_momentum", "ts": datetime.now().strftime("%Y%m%d_%H%M%S"), "tf": {}}
    combos = [("btc_sma100", "ew", "BASELINE btc100+EW"),
              ("btc_sma100", "mom", "btc100 + mom-invvol"),
              ("breadth", "ew", "breadth + EW"),
              ("breadth", "mom", "COMBO breadth + mom-invvol")]
    for tf in ["1d", "4h"]:
        fV, fO = build("VAL", tf), build("OOS", tf)
        res = {}
        print(f"\n{'='*100}\n{tf}  SYNTHESIS 2x2 (de-leaked, PROD, cap33, K5). Baseline = btc100 + EW-all.")
        print(f"{'='*100}")
        print(f"  {'config':<28}{'VAL_lin':>8}{'VAL_Sh':>7}{'VAL_DD':>7}{'VAL_r/DD':>9} | "
              f"{'OOS_lin':>8}{'OOS_Sh':>7}{'OOS_DD':>7}{'OOS_r/DD':>9} | {'verdict':>8}")
        base_v = base_o = None
        for gate, sel, label in combos:
            gv = fV["g_btc"] if gate == "btc_sma100" else fV["g_brd"]
            go = fO["g_btc"] if gate == "btc_sma100" else fO["g_brd"]
            mv = met(net(weights(fV, gv, sel), fV["ret"]), fV["bpd"], fV["mask"])
            mo = met(net(weights(fO, go, sel), fO["ret"]), fO["bpd"], fO["mask"])
            res[label] = {"VAL": mv, "OOS": mo}
            if "BASELINE" in label: base_v, base_o = mv, mo; verdict = "(base)"
            else:
                better_radj = mv["ret_per_dd"] > base_v["ret_per_dd"] and mo["ret_per_dd"] > base_o["ret_per_dd"]
                dd_ok = abs(mv["maxdd"]) <= abs(base_v["maxdd"])+2 and abs(mo["maxdd"]) <= abs(base_o["maxdd"])+2
                verdict = "WIN" if (better_radj and dd_ok) else ("r/DD+" if better_radj else "no")
            print(f"  {label:<28}{mv['lin_wk']:>+8.2f}{mv['sharpe']:>+7.2f}{mv['maxdd']:>+7.0f}{mv['ret_per_dd']:>+9.3f} | "
                  f"{mo['lin_wk']:>+8.2f}{mo['sharpe']:>+7.2f}{mo['maxdd']:>+7.0f}{mo['ret_per_dd']:>+9.3f} | {verdict:>8}")
        out["tf"][tf] = res
    with open(HERE / "wave3e_combo.json", "w") as fo:
        json.dump(out, fo, indent=2, default=str)
    print(f"\nSaved {HERE/'wave3e_combo.json'}")


if __name__ == "__main__":
    main()
