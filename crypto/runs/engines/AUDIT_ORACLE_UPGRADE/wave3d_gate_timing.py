"""WAVE 3D (Option B) -- is the deployment GATE (when to be in/out) sharp enough?

The gap-DNA's biggest component (esp. 4h, +31pp) is "wrong cash calls" = deployment TIMING. The
current gate is a SINGLE-asset 1d-BTC-SMA100 hysteresis. Stress-test it against principled
alternatives, applied to the SAME de-leaked cap33 conv inv-vol book (only the GATE changes):
  - btc_sma100  : current (1d BTC SMA100, hysteresis 1.02/0.98)
  - btc_sma50   : faster BTC trend gate
  - breadth     : cross-sectional -- frac of u100 names above own SMA100; hysteresis on>0.5/off<0.4
  - btc100&brd  : both must agree (conservative)
  - none        : ungated (regime gate removed -> shows the gate's whole contribution)
Win condition (overfit-guarded): an alt must improve maxDD AND keep/raise return-per-drawdown on
BOTH VAL AND OOS vs btc_sma100. de-leaked, PROD cost, 2025 SEALED, long-only spot lev=1.
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
LB_D, SL_D, CAP = 14, 14, 0.33


def hysteresis(sig, ref_up, ref_dn):
    """Generic hysteresis state machine on sig vs (ref_up, ref_dn). Returns 0/1 array."""
    n = len(sig); s = np.zeros(n, dtype=int); st = 0
    for i in range(n):
        if not (np.isfinite(sig[i]) and np.isfinite(ref_up[i]) and np.isfinite(ref_dn[i])):
            s[i] = st; continue
        if st == 0 and sig[i] > ref_up[i]:
            st = 1
        elif st == 1 and sig[i] < ref_dn[i]:
            st = 0
        s[i] = st
    return s


def build(window, tf="1d"):
    wall, ws, we = WIN[window]; bpd = BARS_PER_DAY[tf]
    lb, sl = LB_D*bpd, SL_D*bpd
    closes = {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, wall)
        if rec is None: continue
        d, o, h, l, c = rec
        closes[sym] = (d, o, h, l, np.asarray(c, float))
    syms = list(closes.keys())
    close_df = pd.DataFrame({s: pd.Series(closes[s][4], index=closes[s][0]) for s in syms}).sort_index()
    idx = close_df.index; T = len(idx); A = len(syms)
    conv = np.zeros((T, A))
    for ma in TYPES:
        hc = {}
        for s in syms:
            d, o, h, l, c = closes[s]
            bh = _cached_band_held(ma, tf, s, o, h, l, c, wall)
            hc[s] = pd.Series(held_rolling(c, bh, lb, sl), index=d)
        conv += pd.DataFrame(hc).reindex(idx).fillna(0).reindex(columns=close_df.columns).fillna(0).to_numpy()
    ret = close_df.pct_change().fillna(0.0).to_numpy()
    vol = close_df.pct_change().rolling(max(20, bpd*5), min_periods=10).std().to_numpy()
    C = close_df.to_numpy()
    sma_w = 100*bpd

    # --- gates on the engine grid (all causal: built from closes up to t; applied via lag-1 in net) ---
    gates = {}
    # BTC SMA100 / SMA50 hysteresis
    btc = "BTCUSDT" if "BTCUSDT" in close_df.columns else close_df.columns[0]
    bc = close_df[btc]
    for w, name in [(100*bpd, "btc_sma100"), (50*bpd, "btc_sma50")]:
        sma = bc.rolling(w, min_periods=w//2).mean().to_numpy()
        gates[name] = hysteresis(bc.to_numpy(), sma*1.02, sma*0.98)
    # breadth: frac of names with close > own SMA100
    sma_all = close_df.rolling(sma_w, min_periods=sma_w//2).mean().to_numpy()
    above = (C > sma_all).astype(float)
    valid_cnt = np.isfinite(sma_all).sum(axis=1)
    breadth = np.where(valid_cnt > 0, np.nansum(above*np.isfinite(sma_all), axis=1)/np.maximum(valid_cnt,1), np.nan)
    gates["breadth"] = hysteresis(breadth, np.full(T, 0.5), np.full(T, 0.4))
    gates["btc100&brd"] = (gates["btc_sma100"] & gates["breadth"]).astype(int)
    gates["none"] = np.ones(T, dtype=int)

    # --- the cap33 conv inv-vol book weights, UNGATED (gate applied separately) ---
    mult = conv/len(TYPES)
    iv = np.where((mult > 0) & np.isfinite(vol) & (vol > 0), 1.0/np.where(vol > 0, vol, np.nan), 0.0)
    iv = np.nan_to_num(iv)*mult
    s = iv.sum(axis=1, keepdims=True)
    w = np.minimum(np.where(s > 0, iv/np.maximum(s, 1e-12), 0.0), CAP)
    mask = (idx >= pd.Timestamp(ws)) & (idx < pd.Timestamp(we))
    return {"w": w, "ret": ret, "gates": gates, "mask": mask, "bpd": bpd}


def net_from_W(W, ret, cost=COST_PROD):
    Wl = np.vstack([np.zeros((1, W.shape[1])), W[:-1]])
    gr = np.nansum(Wl*ret, axis=1)
    turn = np.nansum(np.abs(W - np.vstack([np.zeros((1, W.shape[1])), W[:-1]])), axis=1)
    return gr - turn*(cost/2)


def metrics(n, bpd, mask, gate):
    nm = n[mask]
    if len(nm) == 0: return None
    lin = float(np.mean(nm))*bpd*7*100
    cum = np.cumprod(1+nm)
    dd = float(np.min((cum-np.maximum.accumulate(cum))/np.maximum.accumulate(cum)))*100 if len(cum)>1 else 0
    sh = (float(np.mean(nm))*bpd*365)/(float(np.std(nm))*np.sqrt(bpd*365)) if np.std(nm)>1e-12 else 0
    deploy = float(np.mean(gate[mask] > 0))
    return {"lin_wk": round(lin,3), "sharpe": round(sh,3), "maxdd": round(dd,1),
            "ret_per_dd": round(lin/abs(dd),4) if abs(dd)>1e-9 else 0, "deploy": round(deploy,3)}


def main():
    out = {"check": "WAVE3D_gate_timing", "ts": datetime.now().strftime("%Y%m%d_%H%M%S"), "cap": CAP, "tf": {}}
    for tf in ["1d", "4h"]:
        fV, fO = build("VAL", tf), build("OOS", tf)
        res = {}
        for gname in ["btc_sma100", "btc_sma50", "breadth", "btc100&brd", "none"]:
            mv = metrics(net_from_W(fV["w"]*fV["gates"][gname][:,None], fV["ret"]), fV["bpd"], fV["mask"], fV["gates"][gname])
            mo = metrics(net_from_W(fO["w"]*fO["gates"][gname][:,None], fO["ret"]), fO["bpd"], fO["mask"], fO["gates"][gname])
            res[gname] = {"VAL": mv, "OOS": mo}
        out["tf"][tf] = res
        base_v, base_o = res["btc_sma100"]["VAL"], res["btc_sma100"]["OOS"]
        print(f"\n{'='*104}\n{tf}  GATE TIMING on the cap33 conv book (de-leaked, PROD). Baseline = btc_sma100.")
        print(f"{'='*104}")
        print(f"  {'gate':<14}{'VAL_lin':>8}{'VAL_Sh':>7}{'VAL_DD':>7}{'VAL_r/DD':>9}{'VALdep':>7} | "
              f"{'OOS_lin':>8}{'OOS_Sh':>7}{'OOS_DD':>7}{'OOS_r/DD':>9}{'OOSdep':>7} | {'beats base':>10}")
        for g, r in res.items():
            mv, mo = r["VAL"], r["OOS"]
            beat = (mv["ret_per_dd"] >= base_v["ret_per_dd"] and mo["ret_per_dd"] >= base_o["ret_per_dd"]
                    and abs(mv["maxdd"]) <= abs(base_v["maxdd"]) and abs(mo["maxdd"]) <= abs(base_o["maxdd"]))
            tag = "(base)" if g == "btc_sma100" else ("YES" if beat else "no")
            print(f"  {g:<14}{mv['lin_wk']:>+8.2f}{mv['sharpe']:>+7.2f}{mv['maxdd']:>+7.0f}{mv['ret_per_dd']:>+9.3f}{mv['deploy']*100:>6.0f}% | "
                  f"{mo['lin_wk']:>+8.2f}{mo['sharpe']:>+7.2f}{mo['maxdd']:>+7.0f}{mo['ret_per_dd']:>+9.3f}{mo['deploy']*100:>6.0f}% | {tag:>10}")
    with open(HERE / "wave3d_gate_timing.json", "w") as fo:
        json.dump(out, fo, indent=2, default=str)
    print(f"\nSaved {HERE/'wave3d_gate_timing.json'}")


if __name__ == "__main__":
    main()
