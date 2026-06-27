"""Batch play-out — run multiple candidates through PROD selection, report prod-grade table.

Reports each at the 15% cap (prudent floor) AND notes the EW ceiling, so the prod-grade
verdict is honest about the concentration dial.
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
OUTDIR = CRYPTO / "reports" / "MA_playout"
TIER_CFG = {
    "1d": {"lookback": 14, "slice": 14, "min_hold": 0, "cooldown": 0},
    "4h": {"lookback": 14, "slice": 7,  "min_hold": 0, "cooldown": 0},
    "2h": {"lookback": 7,  "slice": 7,  "min_hold": 3, "cooldown": 3},
}
CANDIDATES = [
    ("VIDYA", "1d"), ("SMA", "1d"), ("EMA", "1d"), ("DEMA", "1d"),
    ("SMA", "4h"), ("VIDYA", "4h"), ("WMA", "4h"), ("EMA", "4h"), ("KAMA", "4h"),
    ("SMA", "2h"), ("VIDYA", "2h"),
]


def panel(ma, tf):
    cfg = TIER_CFG[tf]; bpd = BARS_PER_DAY[tf]
    lb, sl = cfg["lookback"]*bpd, cfg["slice"]*bpd
    reg = causal_regime(OOS_WALL_MS)   # de-leaked: bar on day D reads regime[D-1]
    cc, hc = {}, {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, OOS_WALL_MS)
        if rec is None:
            continue
        d, o, h, l, c = rec
        bh = _cached_band_held(ma, tf, sym, o, h, l, c, OOS_WALL_MS)
        held = held_rolling(c, bh, lb, sl, min_hold_bars=cfg["min_hold"], cooldown_bars=cfg["cooldown"])
        cc[sym] = pd.Series(c, index=d); hc[sym] = pd.Series(held, index=d)
    close = pd.DataFrame(cc).sort_index()
    heldp = pd.DataFrame(hc).reindex(close.index).fillna(0).astype(np.int8)
    return close, heldp, reg, bpd


def evalcap(close, heldp, reg, bpd, cap):
    ret = close.pct_change().fillna(0.0).to_numpy()
    H = heldp.to_numpy().astype(float)
    vol = close.pct_change().rolling(max(20, bpd*5), min_periods=10).std().to_numpy()
    dates = close.index; T, A = H.shape
    rg = np.ones(T); ridx = reg.index
    for k, dt in enumerate(dates):
        d1 = pd.Timestamp(dt).normalize(); prior = ridx[ridx <= d1]
        rg[k] = int(reg[prior[-1]]) if len(prior) else 0
    Hg = H * rg[:, None]
    # inv-vol weights, capped, gross<=1
    iv = np.where((Hg > 0) & np.isfinite(vol) & (vol > 0), 1.0/np.where(vol > 0, vol, np.nan), 0.0)
    iv = np.nan_to_num(iv); s = iv.sum(axis=1, keepdims=True)
    w = np.where(s > 0, iv/np.maximum(s, 1e-12), 0.0)
    w = np.minimum(w, cap)
    Wl = np.vstack([np.zeros((1, A)), w[:-1]])
    gr = np.nansum(Wl*ret, axis=1)
    turn = np.nansum(np.abs(w - np.vstack([np.zeros((1, A)), w[:-1]])), axis=1)
    net = gr - turn*(COST_PROD/2)
    mask = (dates >= pd.Timestamp(OOS_START)) & (dates < pd.Timestamp(OOS_2024_END))
    nw = net[mask]
    g = np.nansum(np.abs(w), axis=1)[mask]
    lin = float(np.mean(nw))*bpd*7*100
    comp = float(np.prod(1+nw)-1)*100
    cum = np.cumprod(1+nw); dd = float(np.min((cum-np.maximum.accumulate(cum))/np.maximum.accumulate(cum)))*100
    sh = (float(np.mean(nw))*bpd*365)/(float(np.std(nw))*np.sqrt(bpd*365)) if np.std(nw) > 1e-12 else 0
    ag = float(np.mean(g[g > 0])) if (g > 0).any() else 0
    return lin, comp, sh, dd, ag


def main():
    print("PROD-GRADE PLAY-OUT — all top candidates, OOS-2024, prod cost, inv-vol + cap")
    print("=" * 96)
    print(f"  {'cand':<10}| {'15%cap lin/wk':>13}{'Sh':>6}{'MaxDD':>7} | {'50%cap lin/wk':>13}{'Sh':>6} | "
          f"{'EW lin/wk':>10}{'Sh':>6}")
    print("  " + "-" * 88)
    rows = []
    for ma, tf in CANDIDATES:
        close, heldp, reg, bpd = panel(ma, tf)
        l15, c15, s15, d15, g15 = evalcap(close, heldp, reg, bpd, 0.15)
        l50, c50, s50, d50, g50 = evalcap(close, heldp, reg, bpd, 0.50)
        lew, cew, sew, dew, gew = evalcap(close, heldp, reg, bpd, 1.00)
        verdict = "PROD" if (l15 > 0 and s15 > 0.8) else ("MARGINAL" if l15 > 0 else "FAIL")
        rows.append({"ma": ma, "tf": tf, "cap15_linwk": round(l15,3), "cap15_sharpe": round(s15,2),
                     "cap15_maxdd": round(d15,1), "cap50_linwk": round(l50,3), "cap50_sharpe": round(s50,2),
                     "ew_linwk": round(lew,3), "ew_sharpe": round(sew,2), "verdict": verdict})
        print(f"  {ma+'|'+tf:<10}| {l15:>+12.2f}%{s15:>6.2f}{d15:>6.0f}% | {l50:>+12.2f}%{s50:>6.2f} | "
              f"{lew:>+9.2f}%{sew:>6.2f}  [{verdict}]")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTDIR / f"prodgrade_verdict_{ts}.json", "w") as f:
        json.dump({"rows": rows, "ts": ts}, f, indent=2)
    print("=" * 96)
    print(f"  Saved prodgrade_verdict_{ts}.json")


if __name__ == "__main__":
    main()
