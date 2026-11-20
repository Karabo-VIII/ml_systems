"""src/strat/fixed_approach.py -- build / fix / upgrade the FIXED MA approach, normal vs upgrades side-by-side.

WHY (user /orc 2026-06-12): "run the normal configs vs configs + (fixed and upgrades) and compare side by
side ... we are building, fixing, and upgrading the FIXED approach." The ML config-selector is REFUTED
(selecting config from conditions doesn't transfer) -- so we improve the FIXED config STRUCTURALLY. This
stacks the validated upgrades into a LADDER and measures each step's effect across regimes.

THE LADDER (each row = the previous + one change), as an equal-weight book over u10:
  L0 NAIVE_GRID     : all distinct configs (every speed), signal-flip, taker        <- the naive "run everything"
  L1 FIXED_SLOW     : restrict to 2MA-slow(60-150), cadence-matched, signal-flip    <- FIX: pick the robust family
  L2 +MIN_HOLD      : + min_hold(12) (kills whipsaw)
  L3 +TRAIL         : + 10% trailing stop after the min-hold (crash protection)
  L4 +REGIME_GATE   : + sit OUT (cash) when close < SMA(regime) -- the one selection door the ML work left open
  L5 +MAKER         : L4 priced at maker (0.06% rt) instead of taker (0.24%)

Run across PERIODS spanning regimes: Jan-2020 rally, Feb-2020 top/reversal, the COMBINED 2-month, a
2022 BEAR month, a 2024 BULL month. Side-by-side: ladder x period (book ROI, maxDD, Sharpe). An upgrade
is KEPT iff it improves the cross-period picture (esp. the bear/reversal), DROPPED honestly otherwise.

Book = equal-weight across (config, asset) of the causal MtM net daily returns. DESCRIPTIVE; TRAIN-era
data; outputs stored per the period store. RWYB: python -m strat.fixed_approach
No emoji (cp1252).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR
from strat.portfolio_replay import holding_state, apply_trail_stop, TAKER_RT, MAKER_RT
from strat.replay_distinct_grid import distinct_specs
from strat.ma_mechanics import _cached_panel
from strat.structural_fixes import min_hold

PERIODS = {  # label -> (start, end, regime-note)
    "Jan2020_rally": ("2020-01-07", "2020-02-07", "rally (slow-MA warmup-starved)"),
    "Feb2020_revsl": ("2020-02-07", "2020-03-07", "top + COVID-onset reversal"),
    "JanFeb_comb":   ("2020-01-07", "2020-03-07", "rally -> reversal (2mo)"),
    "Jun2022_bear":  ("2022-06-01", "2022-07-01", "sustained bear"),
    "Feb2024_bull":  ("2024-02-01", "2024-03-01", "sustained bull"),
}
CADENCES = ["4h", "1h"]
ANN = {"4h": 365 * 6, "1h": 365 * 24}


def _sma(c, n):
    if len(c) < n: return np.full(len(c), np.nan)
    cs = np.cumsum(np.insert(c, 0, 0.0))
    out = np.full(len(c), np.nan); out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def _held_for(name, o, c, layer):
    """held series after applying the layer's overlay + regime gate (NOT cost -- cost is in book_net)."""
    h = holding_state(name, o, c, c, c).astype(np.int8)
    if layer in ("L2", "L3", "L4", "L5"):
        h = min_hold(h, 12).astype(np.int8)
    if layer in ("L3", "L4", "L5"):
        h = apply_trail_stop(h.copy(), c, 0.10)[0].astype(np.int8)
    if layer in ("L4", "L5"):
        reg = c > _sma(c, 200)                      # macro-trend gate: long only above SMA200
        h = (h & np.nan_to_num(reg).astype(np.int8)).astype(np.int8)
    return h


def book_net_series(config_set, cadence, start, end, layer, cost):
    s_ms = pd.Timestamp(start).value // 10**6; e_ms = pd.Timestamp(end).value // 10**6
    syms = [a["symbol"] for a in yaml.safe_load(open(ROOT.parent / "config" / "universes" / "u10.yaml"))["assets"]]
    per_cell = []                                   # list of net-daily-return arrays aligned to the window grid
    idx_ref = None
    for sym in syms:
        try:
            o, h, l, c, ms = _cached_panel(sym, cadence)
        except Exception:
            continue
        keep = ms < e_ms; o, c, ms = o[keep], c[keep], ms[keep]
        wm = ms >= s_ms
        if wm.sum() < 10:
            continue
        ret = np.zeros(len(c)); ret[1:] = c[1:] / c[:-1] - 1.0
        for name in config_set:
            held = _held_for(name, o, c, layer)
            pos = np.zeros(len(c)); pos[1:] = held[:-1]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            net = pos * ret - flips * (cost / 2.0)
            per_cell.append(net[wm])
            if idx_ref is None: idx_ref = pd.to_datetime(ms[wm], unit="ms")
    if not per_cell:
        return None, None
    m = min(len(x) for x in per_cell)
    book = np.mean([x[:m] for x in per_cell], axis=0)   # equal-weight across (config, asset)
    return book, (idx_ref[:m] if idx_ref is not None else None)


def metrics(book, cadence):
    if book is None or len(book) < 5:
        return {}
    eq = np.cumprod(1 + book); peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min() * 100)
    sharpe = float(book.mean() / (book.std() + 1e-12) * np.sqrt(ANN.get(cadence, 365)))
    return {"roi": round(float(eq[-1] - 1) * 100, 1), "maxdd": round(dd, 1), "sharpe": round(sharpe, 2),
            "eq": eq}


def main() -> int:
    # config sets: naive = every distinct 2MA+3MA config; slow = the robust family (2MA, slow MA in [60,150))
    import re
    allcfg = {}
    for fam in ("2MA", "3MA"):
        allcfg.update(distinct_specs(fam, 0.15, max_n=60))
    PR.STRATS.update(allcfg)
    naive = list(allcfg)

    def _nums(n):
        return [int(x) for x in re.findall(r"\d+", n)]

    def _is_2ma(n):
        return len(_nums(n)) == 2                       # 2MA names = ema_a_b (2 ints); 3MA = ema_a_b_c (3 ints)

    slow = [n for n in allcfg if _is_2ma(n) and 60 <= max(_nums(n)) < 150]
    print(f"   config pool: {len(naive)} distinct (2MA+3MA); FIXED-slow family (2MA, slow in [60,150)): {len(slow)}")
    LAYERS = [("L0_NAIVE", naive, "L0", TAKER_RT), ("L1_FIXED", slow, "L1", TAKER_RT),
              ("L2_MINHOLD", slow, "L2", TAKER_RT), ("L3_TRAIL", slow, "L3", TAKER_RT),
              ("L4_REGIME", slow, "L4", TAKER_RT), ("L5_MAKER", slow, "L5", MAKER_RT)]

    results = {}
    for cad in CADENCES:
        print(f"\n########## CADENCE {cad} -- ladder x period (book ROI% / maxDD% / Sharpe) ##########")
        hdr = f"   {'layer':12}" + "".join(f"{p[:11]:>22}" for p in PERIODS)
        print(hdr)
        for lname, cset, layer, cost in LAYERS:
            row = f"   {lname:12}"
            for plabel, (s, e, _note) in PERIODS.items():
                book, _ = book_net_series(cset, cad, s, e, layer, cost)
                mt = metrics(book, cad)
                results[(cad, lname, plabel)] = mt
                row += f"{(str(mt.get('roi'))+'/'+str(mt.get('maxdd'))+'/'+str(mt.get('sharpe'))):>22}" if mt else f"{'--':>22}"
            print(row)

    # ---- chart: ladder ROI per period (4h) + combined-period equity curves of each layer ----
    fig, ax = plt.subplots(1, 2, figsize=(17, 6))
    lnames = [l[0] for l in LAYERS]; plabels = list(PERIODS)
    x = np.arange(len(plabels)); w = 0.13
    for i, ln in enumerate(lnames):
        vals = [results.get(("4h", ln, p), {}).get("roi", np.nan) for p in plabels]
        ax[0].bar(x + (i - 2.5) * w, vals, w, label=ln)
    ax[0].set_xticks(x); ax[0].set_xticklabels([p[:11] for p in plabels], fontsize=8, rotation=12)
    ax[0].axhline(0, color="k", lw=0.7); ax[0].set_ylabel("book ROI % (4h)"); ax[0].legend(fontsize=7)
    ax[0].set_title("Ladder x period -- 4h book ROI (naive -> fixed -> upgrades)")
    # combined-period equity per layer (4h)
    for ln, cset, layer, cost in LAYERS:
        book, idx = book_net_series(cset, "4h", *PERIODS["JanFeb_comb"][:2], layer, cost)
        mt = metrics(book, "4h")
        if mt and idx is not None:
            ax[1].plot(idx, mt["eq"], label=f"{ln} ({mt['roi']:+.0f}%)", lw=1.4)
    ax[1].axhline(1.0, color="grey", ls=":", lw=0.6); ax[1].legend(fontsize=7)
    ax[1].set_title("Combined Jan+Feb equity by layer (4h book)"); ax[1].set_ylabel("$1 book equity")
    fig.tight_layout()
    out = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "JAN_FEB_COMBINED" / "charts" / "fixed_approach_ladder.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"\n[figure] {out}")
    json.dump({f"{c}|{l}|{p}": {k: v for k, v in m.items() if k != 'eq'} for (c, l, p), m in results.items()},
              open(out.parent.parent / "analysis" / "fixed_approach_ladder.json", "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
