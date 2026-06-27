"""CONVERGENCE-CONVICTION across MA TYPES (the user's idea, robust reframe).

User idea: deploy top candidates separately (no voting); grade signal quality by how many
CONVERGE per type, and size position by convergence. V11 showed convergence across
hand-picked CONFIGS is built on a weak base (top-K loses to full band OOS). This tests the
ROBUST reframe: convergence across MA TYPES — each type's FULL BAND produces its own held
signal; count how many of the N types are long on an asset = conviction (0..N); size by it.

Compares:
  A) FLAT: equal-weight any asset held by >=1 type (the "in or out" baseline)
  B) CONVERGENCE-WEIGHTED: weight proportional to (n_types_long / N) — more agreement = more size
  C) CONVERGENCE-GATED: only enter assets where >= M types agree (high-conviction only)

All through prod allocator (inv-vol base * conviction multiplier, 15% cap, regime gate,
prod cost). OOS-2024, 2025 sealed. Each type uses its FROZEN full band (the SOTA per V11).
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
from strat.ma_slice_engine import held_rolling, _cached_band_held, causal_regime, OOS_WALL_MS, OOS_START, OOS_2024_END
from strat.canonical_exits import COST_PROD

BARS_PER_DAY = fl.BARS_PER_DAY
OUTDIR = CRYPTO / "runs" / "engines" / "V12_convergence"
TIER_CFG = {"1d": {"lookback": 14, "slice": 14}, "4h": {"lookback": 14, "slice": 7}}
TYPES = ["SMA", "EMA", "WMA", "VIDYA", "KAMA"]


def build_type_holds(tf):
    """Per-type held panel (full frozen band). Returns dict[type] -> held_df, and close_df."""
    cfg = TIER_CFG[tf]; bpd = BARS_PER_DAY[tf]
    lb, sl = cfg["lookback"]*bpd, cfg["slice"]*bpd
    syms = _u100_symbols()
    closes = {}
    for sym in syms:
        rec = _load_asset(sym, tf, OOS_WALL_MS)
        if rec is None:
            continue
        d, o, h, l, c = rec
        closes[sym] = (d, o, h, l, np.asarray(c, float))
    type_held = {}
    for ma in TYPES:
        hc = {}
        for sym, (d, o, h, l, c) in closes.items():
            bh = _cached_band_held(ma, tf, sym, o, h, l, c, OOS_WALL_MS)
            hc[sym] = pd.Series(held_rolling(c, bh, lb, sl), index=d)
        type_held[ma] = hc
    close_df = pd.DataFrame({s: pd.Series(v[4], index=v[0]) for s, v in closes.items()}).sort_index()
    return type_held, close_df


def eval_conviction(type_held, close_df, tf, mode, M=2, cap=0.15):
    """mode: flat | convweight | convgate(M). Returns metrics."""
    bpd = BARS_PER_DAY[tf]
    reg = causal_regime(OOS_WALL_MS)   # de-leaked: bar on day D reads regime[D-1]
    idx = close_df.index
    # convergence count per asset-bar = sum of type held signals
    conv = None
    for ma in TYPES:
        hd = pd.DataFrame(type_held[ma]).reindex(idx).fillna(0).astype(np.int8)
        hd = hd.reindex(columns=close_df.columns).fillna(0)
        conv = hd.to_numpy().astype(float) if conv is None else conv + hd.to_numpy().astype(float)
    N = len(TYPES)
    ret = close_df.pct_change().fillna(0.0).to_numpy()
    vol = close_df.pct_change().rolling(max(20, bpd*5), min_periods=10).std().to_numpy()
    T, A = conv.shape
    rg = np.ones(T); ridx = reg.index
    for k, dt in enumerate(idx):
        d1 = pd.Timestamp(dt).normalize(); prior = ridx[ridx <= d1]
        rg[k] = int(reg[prior[-1]]) if len(prior) else 0

    if mode == "flat":
        active = (conv >= 1).astype(float)
        mult = active
    elif mode == "convweight":
        mult = conv / N                      # 0..1 conviction multiplier
    elif mode == "convgate":
        mult = (conv >= M).astype(float)     # only high-conviction
    else:
        raise ValueError(mode)
    mult = mult * rg[:, None]                # regime gate

    iv = np.where((mult > 0) & np.isfinite(vol) & (vol > 0), 1.0/np.where(vol > 0, vol, np.nan), 0.0)
    iv = np.nan_to_num(iv) * mult            # inv-vol scaled by conviction
    s = iv.sum(axis=1, keepdims=True)
    w = np.where(s > 0, iv/np.maximum(s, 1e-12), 0.0)
    w = np.minimum(w, cap)
    Wl = np.vstack([np.zeros((1, A)), w[:-1]])
    gr = np.nansum(Wl*ret, axis=1)
    turn = np.nansum(np.abs(w - np.vstack([np.zeros((1, A)), w[:-1]])), axis=1)
    net = gr - turn*(COST_PROD/2)
    mask = (idx >= pd.Timestamp(OOS_START)) & (idx < pd.Timestamp(OOS_2024_END))
    nw = net[mask]
    lin = float(np.mean(nw))*bpd*7*100
    comp = float(np.prod(1+nw)-1)*100
    cum = np.cumprod(1+nw); dd = float(np.min((cum-np.maximum.accumulate(cum))/np.maximum.accumulate(cum)))*100 if len(cum)>1 else 0
    sh = (float(np.mean(nw))*bpd*365)/(float(np.std(nw))*np.sqrt(bpd*365)) if np.std(nw) > 1e-12 else 0
    dep = float(np.mean((mult[mask] > 0).any(axis=1)))
    return {"lin_wk": round(lin,3), "compound": round(comp,1), "sharpe": round(sh,2), "maxdd": round(dd,1), "deploy_bars": round(dep,3)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1d")
    a = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"CONVERGENCE-CONVICTION across {len(TYPES)} MA TYPES — {a.tf}, OOS-2024, prod, 15% cap")
    print("=" * 84)
    type_held, close_df = build_type_holds(a.tf)
    rows = {}
    for label, kw in [("flat (>=1 type)", {"mode": "flat"}),
                      ("conv-weighted", {"mode": "convweight"}),
                      ("conv-gate >=2", {"mode": "convgate", "M": 2}),
                      ("conv-gate >=3", {"mode": "convgate", "M": 3}),
                      ("conv-gate >=4", {"mode": "convgate", "M": 4})]:
        m = eval_conviction(type_held, close_df, a.tf, **kw)
        rows[label] = m
        print(f"  {label:<18} lin/wk {m['lin_wk']:>+6.2f}%  comp {m['compound']:>+7.1f}%  "
              f"Sharpe {m['sharpe']:>5.2f}  MaxDD {m['maxdd']:>6.1f}%")
    print("=" * 84)
    print("conv-weighted/gate beating flat = type-convergence is a real conviction signal.")
    with open(OUTDIR / f"convergence_{a.tf}_{ts}.json", "w") as f:
        json.dump({"tf": a.tf, "rows": rows, "types": TYPES, "ts": ts}, f, indent=2, default=str)
    print(f"  Saved convergence_{a.tf}_{ts}.json")


if __name__ == "__main__":
    main()
