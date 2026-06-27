"""Decompose the EW (+3.36%/wk) vs PROD (+1.20%/wk) gap — FAST, vectorized, correctly labeled.

Earlier version mislabeled: with vol_target=1e9 the allocator collapses to EW (all weights
hit the cap then shrink), so "inv-vol" never engaged. This version computes each weighting
scheme DIRECTLY and vectorized — no per-bar allocate() — so it's fast and the labels are honest.

Schemes (all gross<=1, long-only, regime-gated, PROD cost):
  EW             : w_i = 1/n
  EW + cap C     : w_i = min(1/n, C)  (under-deploys when n < 1/C -> cash drag)
  invvol + cap C : w_i ∝ 1/vol_i, capped at C, renormalized to gross<=1
Cap sweep over C in {0.15,0.20,0.25,0.33,0.50,1.00} to show the concentration/return curve.
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
from strat.ma_slice_engine import held_rolling, _cached_band_held, btc_regime, OOS_WALL_MS, OOS_START, OOS_2024_END
from strat.canonical_exits import COST_PROD

BARS_PER_DAY = fl.BARS_PER_DAY
MA, TF = "SMA", "4h"
bpd = BARS_PER_DAY[TF]
lb, sl = 14 * bpd, 7 * bpd
win_s, win_e = pd.Timestamp(OOS_START), pd.Timestamp(OOS_2024_END)

regime = btc_regime(OOS_WALL_MS)
close_cols, held_cols = {}, {}
for sym in _u100_symbols():
    rec = _load_asset(sym, TF, OOS_WALL_MS)
    if rec is None:
        continue
    dates, o, h, l, c = rec
    bh = _cached_band_held(MA, TF, sym, o, h, l, c, OOS_WALL_MS)
    held = held_rolling(c, bh, lb, sl)
    close_cols[sym] = pd.Series(c, index=dates)
    held_cols[sym] = pd.Series(held, index=dates)

close = pd.DataFrame(close_cols).sort_index()
heldp = pd.DataFrame(held_cols).reindex(close.index).fillna(0).astype(np.int8)
ret = close.pct_change().fillna(0.0).to_numpy()
H = heldp.to_numpy().astype(float)                       # [T x A] held
vol = pd.DataFrame(close_cols).sort_index().pct_change().rolling(max(20, bpd*5), min_periods=10).std().to_numpy()
dates = close.index
T, A = H.shape

# regime per bar
reg = np.ones(T)
ridx = regime.index
for k, dt in enumerate(dates):
    d1 = pd.Timestamp(dt).normalize(); prior = ridx[ridx <= d1]
    reg[k] = int(regime[prior[-1]]) if len(prior) else 0
Hg = H * reg[:, None]                                    # gated holds


def eval_W(W):
    """W [T x A] target weights. Lag 1 bar, MtM, PROD turnover cost, OOS window."""
    Wl = np.vstack([np.zeros((1, A)), W[:-1]])
    gross_ret = np.nansum(Wl * ret, axis=1)
    turn = np.nansum(np.abs(W - np.vstack([np.zeros((1, A)), W[:-1]])), axis=1)
    net = gross_ret - turn * (COST_PROD / 2)
    mask = (dates >= win_s) & (dates < win_e)
    nw = net[mask]
    g = np.nansum(np.abs(W), axis=1)[mask]
    lin = float(np.mean(nw)) * bpd * 7 * 100
    comp = float(np.prod(1 + nw) - 1) * 100
    sh = (float(np.mean(nw)) * bpd*365) / (float(np.std(nw))*np.sqrt(bpd*365)) if np.std(nw) > 1e-12 else 0
    ag = float(np.mean(g[g > 0])) if (g > 0).any() else 0
    return lin, comp, sh, ag


def W_ew():
    n = Hg.sum(axis=1, keepdims=True)
    return np.where(n > 0, Hg / np.maximum(n, 1), 0.0)


def W_ew_cap(C):
    n = Hg.sum(axis=1, keepdims=True)
    w = np.where(n > 0, Hg / np.maximum(n, 1), 0.0)
    return np.minimum(w, C * (Hg > 0))                   # cap each name at C (under-deploys if n<1/C)


def W_invvol_cap(C):
    iv = np.where((Hg > 0) & np.isfinite(vol) & (vol > 0), 1.0 / np.where(vol > 0, vol, np.nan), 0.0)
    iv = np.nan_to_num(iv)
    s = iv.sum(axis=1, keepdims=True)
    w = np.where(s > 0, iv / np.maximum(s, 1e-12), 0.0)  # inv-vol weights summing to 1
    # apply per-name cap with iterative water-filling (simple: clip then renormalize once)
    w = np.minimum(w, C)
    s2 = w.sum(axis=1, keepdims=True)
    # only renormalize UP to gross 1 if under (cap may leave gross<1 -> keep, that's cash)
    return w


print(f"\nGAP DECOMPOSITION — SMA|4h OOS-2024, PROD cost {COST_PROD} (FAST, correctly labeled)")
print("=" * 80)
for label, W in [("EW (1/n)", W_ew()),
                 ("EW + 15% cap", W_ew_cap(0.15)),
                 ("invvol + 15% cap", W_invvol_cap(0.15))]:
    lin, comp, sh, ag = eval_W(W)
    print(f"  {label:<20} lin/wk {lin:>+6.2f}%  comp {comp:>+7.1f}%  Sharpe {sh:>5.2f}  avg_gross {ag:>5.0%}")

print(f"\nPER-NAME CAP SWEEP (EW capped) — concentration vs return")
print("=" * 80)
for C in [0.15, 0.20, 0.25, 0.33, 0.50, 1.00]:
    lin, comp, sh, ag = eval_W(W_ew_cap(C))
    nm = int(round(1 / C))
    print(f"  cap {C:>4.0%}/name (>= {nm} names to fill)  lin/wk {lin:>+6.2f}%  comp {comp:>+7.1f}%  "
          f"Sharpe {sh:>5.2f}  avg_gross {ag:>5.0%}")
print("=" * 80)
print("cap=100% == EW (no limit). Lower cap forces diversification + cash drag when few signal.")
