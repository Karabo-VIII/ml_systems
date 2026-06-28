"""DEPLOYABLE V3 -- the wealth-maximising long-only retail engine (Claude, 2026-06-28).

The campaign (runs/engines/AUDIT_ORACLE_UPGRADE/) audited Kiro V2-V15, fixed a same-day regime
leak, built the long-only oracle, and found TWO robust deployment levers that the static MA band
left on the table. V3 stacks them on the validated, DE-LEAKED scaffold:

  Layer 1  CROSS-SECTIONAL BREADTH GATE (replaces single-BTC-SMA100): deploy when the fraction of
           u100 names above their own 100-bar SMA crosses UP through 0.5, exit below 0.4 (hysteresis).
           Catches regime turns the BTC-only gate misses -> cuts drawdown at ~flat return.
  Layer 2  CROSS-TYPE CONVERGENCE BAND (SMA/EMA/WMA/VIDYA/KAMA, 14-bar lookback, confirmation):
           eligible set = names the band confirms as trending.
  Layer 3  MOMENTUM TOP-8 INV-VOL SELECTION: among the eligible set, hold the top-8 by trailing
           14-bar return, inverse-vol weighted. Tilts into the movers (cross-sectional momentum =
           the one causal selection signal that survives both windows + a random-selection null).

VALIDATION (de-leaked, PROD cost 0.0028 RT, long-only spot lev=1, 2025 SEALED):
  - Robust: 36/36 (K,cap,breadth-threshold) param-combos beat the baseline (btc100+EW cap33) on
    COMPOUND wealth on BOTH windows (VAL + OOS-2024), both 1d and 4h. NOT default-param-luck.
  - Wealth: 1d OOS-2024 compound ~+177% (median over params +148%) vs baseline +68% (~2.5x);
    VAL +466-564%; maxDD ~-29% (similar to baseline). 4h near-identical (+176%).
  - Momentum selection significant vs random under the breadth gate: 97th pctile OOS (1d) / 100th (4h).
  - cap is irrelevant at K8 inv-vol (never binds) -> K is the concentration control.
  FINAL GATE: the 2025 unseal is the last out-of-sample confirm (momentum is the one "active" bet).

This module reproduces the validated book AND emits the LATEST target portfolio (what to hold now).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.fleet_lab as fl
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.ma_slice_engine import held_rolling, _cached_band_held
from strat.ma_slice_engine import OOS_WALL_MS, OOS_START, OOS_2024_END, VAL_WALL_MS, VAL_START, VAL_END
from strat.canonical_exits import COST_PROD

BARS_PER_DAY = fl.BARS_PER_DAY
TYPES = ["SMA", "EMA", "WMA", "VIDYA", "KAMA"]
WIN = {"OOS": (OOS_WALL_MS, OOS_START, OOS_2024_END), "VAL": (VAL_WALL_MS, VAL_START, VAL_END)}

# --- LOCKED V3 PARAMETERS (validated; do not silently retune -- re-run wave3f robustness if changed) ---
LOOKBACK_D = 14          # band confirmation + momentum lookback (days)
SLICE_D = 14             # band slice (days), 1d tier
K = 8                    # momentum top-K (concentration control; cap irrelevant at K8 inv-vol)
BREADTH_ON, BREADTH_OFF = 0.50, 0.40   # breadth gate hysteresis
BREADTH_SMA_D = 100      # per-name trend SMA for breadth (days)


def _hysteresis(sig, up, dn):
    n = len(sig); s = np.zeros(n, dtype=int); st = 0
    for i in range(n):
        if not (np.isfinite(sig[i]) and np.isfinite(up) and np.isfinite(dn)):
            s[i] = st; continue
        if st == 0 and sig[i] > up: st = 1
        elif st == 1 and sig[i] < dn: st = 0
        s[i] = st
    return s


def build_book(tf="1d", window="OOS", wall_ms=None, win_start=None, win_end=None):
    """Build the V3 deployable book. Returns (net_stream, dates, target_weights_df, bpd).
    De-leaked + causal throughout: every gate/score is computed from closes up to each bar and
    applied via lag-1 (position over [t,t+1] uses decisions known at t's close)."""
    bpd = BARS_PER_DAY[tf]
    if wall_ms is None:
        wall_ms, win_start, win_end = WIN[window]
    lb, sl = LOOKBACK_D*bpd, SLICE_D*bpd
    sw = BREADTH_SMA_D*bpd
    closes = {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, wall_ms)
        if rec is None:
            continue
        d, o, h, l, c = rec
        closes[sym] = (d, o, h, l, np.asarray(c, float))
    syms = list(closes.keys())
    close_df = pd.DataFrame({s: pd.Series(closes[s][4], index=closes[s][0]) for s in syms}).sort_index()
    idx = close_df.index; T = len(idx); A = len(syms)

    # Layer 2: cross-type convergence band (eligible breadth)
    conv = np.zeros((T, A))
    for ma in TYPES:
        hc = {}
        for s in syms:
            d, o, h, l, c = closes[s]
            bh = _cached_band_held(ma, tf, s, o, h, l, c, wall_ms)
            hc[s] = pd.Series(held_rolling(c, bh, lb, sl), index=d)
        conv += pd.DataFrame(hc).reindex(idx).fillna(0).reindex(columns=close_df.columns).fillna(0).to_numpy()

    C = close_df.to_numpy()
    ret = close_df.pct_change().fillna(0.0).to_numpy()
    vol = close_df.pct_change().rolling(max(20, bpd*5), min_periods=10).std().to_numpy()
    mom = close_df.pct_change(lb).to_numpy()

    # Layer 1: cross-sectional breadth gate (de-leaked: built from closes up to t, applied via lag-1)
    sma_all = close_df.rolling(sw, min_periods=sw//2).mean().to_numpy()
    vc = np.isfinite(sma_all).sum(axis=1)
    breadth = np.where(vc > 0, np.nansum((C > sma_all)*np.isfinite(sma_all), axis=1)/np.maximum(vc, 1), np.nan)
    gate = _hysteresis(breadth, BREADTH_ON, BREADTH_OFF)

    # Layer 3: momentum top-K inv-vol among the gated + band-confirmed eligible set
    elig = (conv > 0) & (gate[:, None] > 0)
    sc = np.where(elig & np.isfinite(mom), mom, -np.inf)
    W = np.zeros((T, A))
    for t in range(T):
        e = np.where(np.isfinite(sc[t]))[0]
        if len(e) == 0:
            continue
        k = min(K, len(e)); top = e[np.argsort(-sc[t][e])[:k]]
        v = vol[t][top]; ivw = np.where(np.isfinite(v) & (v > 0), 1.0/v, 0.0)
        W[t, top] = ivw/ivw.sum() if ivw.sum() > 0 else 1.0/k

    Wl = np.vstack([np.zeros((1, A)), W[:-1]])
    gr = np.nansum(Wl*ret, axis=1)
    turn = np.nansum(np.abs(W - np.vstack([np.zeros((1, A)), W[:-1]])), axis=1)
    net = gr - turn*(COST_PROD/2)
    wdf = pd.DataFrame(W, index=idx, columns=close_df.columns)
    if win_start is not None:
        mask = (idx >= pd.Timestamp(win_start)) & (idx < pd.Timestamp(win_end))
        return net[mask], idx[mask], wdf.loc[mask], bpd
    return net, idx, wdf, bpd


def profile(net, bpd):
    comp = float(np.prod(1+net)-1)*100
    lin = float(np.mean(net))*bpd*7*100
    cum = np.cumprod(1+net)
    dd = float(np.min((cum-np.maximum.accumulate(cum))/np.maximum.accumulate(cum)))*100 if len(cum)>1 else 0
    sh = (float(np.mean(net))*bpd*365)/(float(np.std(net))*np.sqrt(bpd*365)) if np.std(net)>1e-12 else 0
    sb = 7*bpd; sl = [float(np.prod(1+net[i:i+sb])-1)*100 for i in range(0,len(net)-sb+1,sb)]
    pct7 = sum(1 for x in sl if x > 0)/len(sl) if sl else 0
    return {"compound_pct": round(comp,1), "lin_wk_pct": round(lin,3), "sharpe": round(sh,2),
            "maxdd_pct": round(dd,1), "pct_profitable_7d": round(pct7,3), "n_bars": len(net)}


def latest_targets(tf="1d"):
    """Emit the CURRENT target portfolio (names + weights to hold now) from the full OOS-wall data's
    last bar. NOTE: uses OOS_WALL (2025 sealed). For live use, pass a fresh wall_ms."""
    net, dates, wdf, bpd = build_book(tf=tf, window=None, wall_ms=OOS_WALL_MS, win_start=None, win_end=None)
    last = wdf.iloc[-1]
    held = last[last > 0].sort_values(ascending=False)
    return dates[-1], held


def main():
    print("="*84)
    print("DEPLOYABLE V3 -- breadth gate + cross-type band + momentum top-8 inv-vol (de-leaked, PROD)")
    print(f"LOCKED params: tf=1d, K={K}, breadth {BREADTH_ON}/{BREADTH_OFF} on {BREADTH_SMA_D}d SMA, "
          f"lookback {LOOKBACK_D}d  | long-only spot lev=1, 2025 SEALED")
    print("="*84)
    for tf in ["1d", "4h"]:
        print(f"\n  --- {tf} ---")
        for window in ["VAL", "OOS"]:
            net, dates, wdf, bpd = build_book(tf=tf, window=window)
            p = profile(net, bpd)
            print(f"  {window}: compound {p['compound_pct']:+.0f}%  lin {p['lin_wk_pct']:+.2f}%/wk  "
                  f"Sharpe {p['sharpe']:+.2f}  maxDD {p['maxdd_pct']:+.0f}%  7dprof {p['pct_profitable_7d']:.0%}  ({p['n_bars']} bars)")
    # latest target book (what to hold), 1d
    dt, held = latest_targets("1d")
    print(f"\n  LATEST 1d target book @ {pd.Timestamp(dt).date()} (top-{K} momentum inv-vol within breadth+band):")
    if len(held) == 0:
        print("    [CASH] -- breadth gate OFF or no band-confirmed names. Hold cash.")
    else:
        for sym, w in held.items():
            print(f"    {sym:<14} {w:6.1%}")


if __name__ == "__main__":
    main()
