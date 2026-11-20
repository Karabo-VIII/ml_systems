"""src/strat/ma_type_tf_research.py -- PHASE 1a: the (MA-type x TF) TREND-system research grid (CONSTRUCTION).

USER (5h strategy-discovery-engine build, phase 1a): across ALL 8 MA types {EMA,SMA,WMA,HMA,DEMA,TEMA,
KAMA,VIDYA} x ALL fine timeframes {1d,4h,2h,1h,30m,15m}, find the best IRONED TREND book per (MA-type, TF)
on the 2020 BAND ONLY, and produce a leaderboard + equity-curve charts. This is the foundation of the
discovery engine (phase 1b adds complementarity/blends on top).

WHAT IT BUILDS (per (MA-type, TF)):
  - the IRONED trend book = slow-MA cross FAMILY ensemble (the distinct 2MA+3MA configs in the slow band
    60<=max_len<150), equal-weight u10, with the confirm/whipsaw filter + exit overlay SELECTED on TRAIN+VAL
    ONLY (per MA-type per TF), then confirmed ONCE on OOS. Maker cost, causal/lag-1.
  - METRICS on the within-2020 OOS: net% (=WEALTH, the primary rank), Sharpe, maxDD%, coverage%, turnover,
    breadth(#/10), vs VOLTGT_BH + BUYHOLD at that TF.

SPLIT (within-2020, NOT cross-year): ma_2020_breakdown.SPLIT -- TRAIN 2020-01..07 / VAL ..10 / OOS ..2021-01.
SELECT (confirm-K, exit overlay) on TRAIN+VAL; confirm ONCE on OOS. No look-ahead, positions lagged 1 bar.

REUSE (do NOT reinvent the MA math or the replay engine):
  - strat.ma_type_upgrade: MA_TYPES, _MA (the 8 MA functions), held_cross, _nums
  - strat.ma_2020_breakdown: _panel (native + synthetic-2h), SPLIT, YEAR, WARMUP
  - strat.structural_fixes: confirm, min_hold
  - strat.portfolio_replay: apply_trail_stop, MAKER_RT
  - strat.replay_distinct_grid: distinct_specs (the deduped slow-MA family configs)
  - strat.battery: block_bootstrap_p05_p95 (the OOS tail-robustness p05)

HONEST BAR (read FIRST): the 2020 OOS (Oct-Dec) is a clean BULL (~0% bear). Pure MA trend books are
PARTICIPATING BETA -- they sit out part of the bull, so net < buy-hold is EXPECTED (the participation tax).
We do NOT chase 'beat buy-hold net in the bull' (a bull artifact). We rank by NET (=wealth) but flag the
under-participation caveat and the whole-cycle payoff (bear-DD protection + cross-TF diversification) the
~0%-bear 2020 bull cannot show. Two-sided: MA-types that DON'T work are reported too.

OUTPUTS:
  runs/periods/TRAIN/2020/DEEP_DIVE/ma_type_tf_research.json    -- the full grid + per-TF winners
  runs/periods/TRAIN/2020/DEEP_DIVE/charts/ma_type_tf_heatmap.png
  runs/periods/TRAIN/2020/DEEP_DIVE/charts/best_matype_equity_per_tf.png
  runs/periods/TRAIN/2020/DEEP_DIVE/charts/matype_family_by_tf.png

RWYB: python -m strat.ma_type_tf_research [--tfs 1d,4h,2h,1h,30m,15m]
No emoji (Windows cp1252). Does NOT git commit (overseer commits).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# the adaptive-MA helpers (KAMA/VIDYA) divide by a guarded denominator (np.where masks the result);
# the divide still emits a benign RuntimeWarning. Silence it -- it is not a numerical defect.
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
np.seterr(invalid="ignore", divide="ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR                                    # noqa: E402
from strat.portfolio_replay import apply_trail_stop, MAKER_RT          # noqa: E402
from strat.replay_distinct_grid import distinct_specs                  # noqa: E402
from strat.ma_type_upgrade import _MA, _nums, MA_TYPES                 # noqa: E402
from strat.ma_2020_breakdown import _panel, SPLIT, YEAR, WARMUP        # noqa: E402
from strat.structural_fixes import min_hold, confirm                   # noqa: E402
from strat.battery import block_bootstrap_p05_p95                      # noqa: E402

OUT = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
CHARTS = OUT / "charts"
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
        "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
TFS = ["1d", "4h", "2h", "1h", "30m", "15m"]
# bars/year per cadence -- for the Sharpe annualization + vol-target realized-vol window
ANN = {"1d": 365, "4h": 365 * 6, "2h": 365 * 12, "1h": 365 * 24, "30m": 365 * 48, "15m": 365 * 96}
# realized-vol lookback for the vol-targeted benchmark (~2 weeks of bars per cadence)
VOLWIN = {"1d": 14, "4h": 84, "2h": 168, "1h": 168, "30m": 336, "15m": 672}
# MA-type FAMILY taxonomy for the ordering story (analysis #2)
FAMILY_TAX = {"adaptive": ["KAMA", "VIDYA"], "low_lag": ["HMA", "TEMA", "DEMA"],
              "simple": ["EMA", "SMA", "WMA"]}
# whipsaw confirm-K candidates + exit candidates selected on TRAIN+VAL (per MA-type per TF)
CONF_GRID = [0, 2, 3]
EXIT_GRID = ["none", "minhold", "trail10", "mh_trail15", "chandelier"]


# ===========================================================================
# data: per-asset arrays on the 2020 YEAR window (+ warmup), causal
# ===========================================================================
def _asset_arrays(sym, cad):
    """(c, h, l, ms, win_mask) on [2020-WARMUP .. 2021); win_mask = bars inside 2020. None if too short."""
    try:
        o, h, l, c, ms = _panel(sym, cad)
    except Exception:
        return None
    s_ms = pd.Timestamp(YEAR[0]).value // 10**6
    e_ms = pd.Timestamp(YEAR[1]).value // 10**6
    e_idx = int(np.searchsorted(ms, e_ms))
    s_idx = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
    c2, h2, l2, ms2 = c[s_idx:e_idx], h[s_idx:e_idx], l[s_idx:e_idx], ms[s_idx:e_idx]
    if len(c2) < 60:
        return None
    win = ms2 >= s_ms
    if win.sum() < 30:
        return None
    return c2, h2, l2, ms2, win


def _build_panels(cad, slow):
    """{sym: (c,h,l,ms,win, {ma_type:{period:ma_array}})}. MA computed ONCE per (asset, type, period)."""
    uniq = sorted({p for n in slow for p in _nums(n)})
    out = {}
    for sym in SYMS:
        a = _asset_arrays(sym, cad)
        if a is None:
            continue
        c, h, l, ms, win = a
        caches = {mt: {p: _MA[mt](c, p) for p in uniq} for mt in MA_TYPES}
        out[sym] = (c, h, l, ms, win, caches)
    return out


# ===========================================================================
# held-series builders (causal: held[t] uses close[:t+1] only)
# ===========================================================================
def _entry_held(periods, cache):
    mas = [cache[p] for p in periods]
    h = (mas[0] > mas[1]) if len(periods) == 2 else ((mas[0] > mas[1]) & (mas[1] > mas[2]))
    return np.nan_to_num(h).astype(np.int8)


def _chandelier(held, c, hi, lo, k=3.0, per=22):
    tr = np.maximum(hi - lo, np.abs(hi - np.concatenate([[c[0]], c[:-1]])))
    atr = pd.Series(tr).rolling(per, min_periods=1).mean().to_numpy()
    h = held.copy().astype(np.int8)
    d = np.diff(np.concatenate([[0], h, [0]]))
    starts = np.where(d == 1)[0]; ends = np.where(d == -1)[0]
    for s, e in zip(starts, ends):
        peak = c[s]
        for i in range(s, e):
            peak = max(peak, c[i])
            if c[i] <= peak - k * atr[i]:
                h[i + 1:e] = 0
                break
    return h


def _apply_exit(h0, c, hi, lo, exit_):
    """exit overlay applied to one held series (crease 4).
    'minhold'   participation-FORCING min_hold(12), no trail (rides wiggles).
    'trail10'   10% trailing stop on the raw signal.
    'mh_trail15' min_hold(12) then 15% trail (ride wiggles, cut the big reversal).
    'chandelier' 3xATR(22) chandelier trail from the peak."""
    h = h0.astype(np.int8)
    if exit_ == "none":
        return h
    if exit_ == "minhold":
        return min_hold(h, 12).astype(np.int8)
    if exit_ == "trail10":
        return apply_trail_stop(h.copy(), c, 0.10)[0].astype(np.int8)
    if exit_ == "mh_trail15":
        return apply_trail_stop(min_hold(h, 12).astype(np.int8).copy(), c, 0.15)[0].astype(np.int8)
    if exit_ == "chandelier":
        return _chandelier(h, c, hi, lo)
    return h


def _ret_of(c):
    ret = np.zeros(len(c)); ret[1:] = c[1:] / c[:-1] - 1.0
    return ret


# ===========================================================================
# the IRONED family book for one (MA-type, TF) under (conf_k, exit) -> (book net Series, exposure Series)
# ===========================================================================
def build_book(panels, slow, ma_type, conf_k, exit_):
    """equal-weight u10 book of bar-level net (maker), causal lag-1. family = all slow configs equal-weight.
    Returns (book_net Series, book_exposure Series). None if no assets."""
    net_cells, exp_cells = [], []
    for sym, (c, h, l, ms, win, caches) in panels.items():
        cache = caches[ma_type]
        ret = _ret_of(c)
        poss = []
        for name in slow:
            h0 = _entry_held(_nums(name), cache)
            if conf_k and conf_k > 1:
                h0 = confirm(h0, conf_k).astype(np.int8)
            h0 = _apply_exit(h0, c, h, l, exit_)
            pos = np.zeros(len(c)); pos[1:] = h0[:-1].astype(np.float64)   # lag 1 bar
            poss.append(pos)
        fpos = np.mean(poss, axis=0)                                       # family fraction-long
        flips = np.abs(np.diff(np.concatenate([[0.0], fpos])))
        net = fpos * ret - flips * (MAKER_RT / 2.0)
        idx = pd.to_datetime(ms[win], unit="ms")
        net_cells.append(pd.Series(net[win], index=idx))
        exp_cells.append(pd.Series(fpos[win], index=idx))
    if not net_cells:
        return None, None
    book = pd.concat(net_cells, axis=1).mean(axis=1, skipna=True)
    expo = pd.concat(exp_cells, axis=1).mean(axis=1, skipna=True)
    return book, expo


# ===========================================================================
# benchmarks: BUYHOLD + VOLTGT_BH at the TF (equal-weight u10, no per-trade cost)
# ===========================================================================
def benchmarks(panels, cad):
    rets, rvs = {}, {}
    for sym, (c, h, l, ms, win, caches) in panels.items():
        idx = pd.to_datetime(ms, unit="ms")
        r = pd.Series(c, index=idx).pct_change()
        rets[sym] = r[pd.Series(win, index=idx).values]
        rvs[sym] = r.rolling(VOLWIN[cad], min_periods=max(3, VOLWIN[cad] // 3)).std()[pd.Series(win, index=idx).values]
    R = pd.DataFrame(rets).sort_index()
    V = pd.DataFrame(rvs).sort_index()
    med = float(np.nanmedian(V.to_numpy()))
    bh = R.mean(axis=1)
    w = (med / (V.shift(1) + 1e-12)).clip(0, 1).fillna(0.0)
    vtg = (w * R).mean(axis=1)
    vtg_exp = w.mean(axis=1)
    return {"BUYHOLD": (bh, None), "VOLTGT_BH": (vtg, vtg_exp)}


# ===========================================================================
# metrics on a net-return Series over a window
# ===========================================================================
def _slice(s, lo, hi):
    return s[(s.index >= pd.Timestamp(lo)) & (s.index < pd.Timestamp(hi))].dropna()


def metrics(book, expo, cad, lo, hi):
    s = _slice(book, lo, hi)
    if len(s) < 5:
        return {}
    x = s.to_numpy()
    eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq)
    out = {
        "net": round(float(eq[-1] - 1) * 100, 1),
        "maxdd": round(float(((eq - pk) / pk).min() * 100), 1),
        "sharpe": round(float(np.mean(x) / (np.std(x) + 1e-12) * np.sqrt(ANN[cad])), 2),
        "n_bars": int(len(x)),
    }
    if expo is not None:
        e = _slice(expo, lo, hi)
        if len(e):
            daily_e = e.resample("1D").mean().dropna()
            out["coverage"] = round(float(np.mean(daily_e > 0.5)) * 100, 0)
            out["avg_exp"] = round(float(e.mean()), 3)
            out["turnover"] = round(float(np.abs(np.diff(e.to_numpy())).sum()), 1)
    daily = s.resample("1D").apply(lambda v: float(np.prod(1 + v) - 1)).dropna().to_numpy()
    if len(daily) > 8:
        out["p05"] = block_bootstrap_p05_p95(daily).get("p05")
    return out


def breadth_neff(panels, slow, ma_type, conf_k, exit_, lo, hi):
    """per-asset OOS net -> breadth (#/10 positive) + n_eff (Herfindahl on positive contributions)."""
    per = {}
    for sym in panels:
        sub = {sym: panels[sym]}
        book, _ = build_book(sub, slow, ma_type, conf_k, exit_)
        if book is None:
            continue
        s = _slice(book, lo, hi)
        if len(s) < 5:
            continue
        per[sym] = round(float(np.prod(1 + s.to_numpy()) - 1) * 100, 1)
    if not per:
        return {"breadth": 0, "n_assets": 0, "n_eff": 0.0, "per_asset": {}}
    vals = np.array(list(per.values()))
    pos = np.clip(vals, 0, None)
    neff = float(1.0 / np.sum((pos / pos.sum()) ** 2)) if pos.sum() > 0 else 0.0
    return {"breadth": int(np.sum(vals > 0)), "n_assets": len(per), "n_eff": round(neff, 2),
            "per_asset": per}


# ===========================================================================
# one (MA-type, TF): SELECT (conf_k, exit) on TRAIN+VAL, confirm ONCE on OOS
# ===========================================================================
def run_cell(panels, slow, ma_type, cad):
    sel_lo, sel_hi = SPLIT["TRAIN"][0], SPLIT["VAL"][1]
    oos = SPLIT["OOS"]

    def _selnet(conf_k, exit_):
        b, _ = build_book(panels, slow, ma_type, conf_k, exit_)
        s = _slice(b, sel_lo, sel_hi) if b is not None else None
        return (float(np.prod(1 + s.to_numpy()) - 1) * 100) if (s is not None and len(s) >= 5) else -1e9

    # CREASE 3: confirm-K on TRAIN+VAL
    conf_sel = max(CONF_GRID, key=lambda k: _selnet(k, "none"))
    # CREASE 4: exit on TRAIN+VAL (with the selected confirm-K)
    exit_sel = max(EXIT_GRID, key=lambda e: _selnet(conf_sel, e))

    book, expo = build_book(panels, slow, ma_type, conf_sel, exit_sel)
    m = metrics(book, expo, cad, *oos)
    bn = breadth_neff(panels, slow, ma_type, conf_sel, exit_sel, *oos)
    # also the OOS equity curve (for the chart) -- daily-resampled cum-return
    eq = None
    s = _slice(book, *oos)
    if len(s):
        daily = s.resample("1D").apply(lambda v: float(np.prod(1 + v) - 1)).dropna()
        eq = (np.cumprod(1 + daily.to_numpy()) - 1) * 100
        eq = {"dates": [d.strftime("%Y-%m-%d") for d in daily.index], "cum_pct": [round(float(v), 2) for v in eq]}
    return {"ma_type": ma_type, "cadence": cad, "selected_confirm_k": conf_sel, "selected_exit": exit_sel,
            "oos": m, "breadth_neff": bn, "oos_equity": eq}


# ===========================================================================
# CHARTS
# ===========================================================================
def chart_heatmap(grid, tfs, bench):
    """ma_type_tf_heatmap.png -- OOS net% (panel 1) + Sharpe (panel 2) across (MA-type x TF)."""
    net = np.full((len(MA_TYPES), len(tfs)), np.nan)
    shp = np.full((len(MA_TYPES), len(tfs)), np.nan)
    for i, mt in enumerate(MA_TYPES):
        for j, tf in enumerate(tfs):
            m = grid.get((mt, tf), {}).get("oos", {})
            if m:
                net[i, j] = m.get("net", np.nan)
                shp[i, j] = m.get("sharpe", np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, data, title, cmap in [(axes[0], net, "OOS net % (=wealth, primary rank)", "RdYlGn"),
                                  (axes[1], shp, "OOS Sharpe (caveat: rewards sitting out the bull)", "RdYlGn")]:
        vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(tfs))); ax.set_xticklabels(tfs)
        ax.set_yticks(range(len(MA_TYPES))); ax.set_yticklabels(MA_TYPES)
        ax.set_xlabel("timeframe"); ax.set_title(title, fontsize=10)
        for i in range(len(MA_TYPES)):
            for j in range(len(tfs)):
                if np.isfinite(data[i, j]):
                    ax.text(j, i, f"{data[i, j]:.0f}" if title.startswith("OOS net") else f"{data[i, j]:.2f}",
                            ha="center", va="center", fontsize=7,
                            color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # benchmark annotation
    btxt = "  ".join(f"{tf}: BH {bench[tf]['BUYHOLD']['net']:.0f}% / VOLTGT {bench[tf]['VOLTGT_BH']['net']:.0f}%"
                     for tf in tfs if tf in bench)
    fig.suptitle("2020 within-OOS (Oct-Dec, BULL): IRONED MA-trend book net & Sharpe by (MA-type x TF)\n"
                 "BENCHMARKS  " + btxt + "\n(bull => MA books are participating BETA: net < buy-hold is EXPECTED)",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    CHARTS.mkdir(parents=True, exist_ok=True)
    p = CHARTS / "ma_type_tf_heatmap.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    return p


def chart_best_equity(grid, tfs, bench_eq):
    """best_matype_equity_per_tf.png -- per-TF OOS equity of the best MA-type book vs VOLTGT_BH vs BUYHOLD."""
    ncol = 3; nrow = int(np.ceil(len(tfs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 4.2 * nrow), squeeze=False)
    for k, tf in enumerate(tfs):
        ax = axes[k // ncol][k % ncol]
        # winner by net
        cells = [(mt, grid[(mt, tf)]) for mt in MA_TYPES if grid.get((mt, tf), {}).get("oos")]
        if not cells:
            ax.set_title(f"{tf} (no data)"); continue
        win_mt, win = max(cells, key=lambda kv: kv[1]["oos"].get("net", -1e9))
        eq = win["oos_equity"]
        if eq:
            ax.plot(pd.to_datetime(eq["dates"]), eq["cum_pct"], lw=2, color="#1f77b4",
                    label=f"best: {win_mt} ({win['oos']['net']:.0f}%)")
        for label, color in [("BUYHOLD", "#888888"), ("VOLTGT_BH", "#2ca02c")]:
            be = bench_eq.get((tf, label))
            if be:
                ax.plot(pd.to_datetime(be["dates"]), be["cum_pct"], lw=1.6, color=color, ls="--",
                        label=f"{label} ({be['cum_pct'][-1]:.0f}%)")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{tf} -- best MA-type book vs benchmarks (OOS Oct-Dec 2020)", fontsize=10)
        ax.legend(fontsize=8); ax.set_ylabel("cum return %")
    for k in range(len(tfs), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Best IRONED MA-type trend book per TF vs VOLTGT_BH vs BUYHOLD -- 2020 within-OOS equity\n"
                 "(BULL window: trend books participate but pay the participation tax; value = risk-adjusted + "
                 "whole-cycle DD protection)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = CHARTS / "best_matype_equity_per_tf.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    return p


def chart_family_by_tf(grid, tfs):
    """matype_family_by_tf.png -- adaptive vs low-lag vs simple family-avg OOS net by TF (the ordering story)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(tfs)); w = 0.25
    colors = {"adaptive": "#d62728", "low_lag": "#1f77b4", "simple": "#7f7f7f"}
    for i, (fam, members) in enumerate(FAMILY_TAX.items()):
        vals = []
        for tf in tfs:
            nets = [grid[(mt, tf)]["oos"]["net"] for mt in members
                    if grid.get((mt, tf), {}).get("oos") and grid[(mt, tf)]["oos"].get("net") is not None]
            vals.append(np.mean(nets) if nets else np.nan)
        ax.bar(x + (i - 1) * w, vals, w, label=f"{fam} ({'/'.join(members)})", color=colors[fam])
    ax.set_xticks(x); ax.set_xticklabels(tfs); ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("timeframe"); ax.set_ylabel("family-avg OOS net %")
    ax.set_title("MA-type FAMILY ordering by TF: adaptive (KAMA/VIDYA) vs low-lag (HMA/TEMA/DEMA) vs simple "
                 "(EMA/SMA/WMA)\n2020 within-OOS net; does the ordering flip with TF (adaptive at fine TFs)?",
                 fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = CHARTS / "matype_family_by_tf.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    return p


# ===========================================================================
# MAIN
# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.ma_type_tf_research")
    ap.add_argument("--tfs", default=",".join(TFS))
    a = ap.parse_args(argv)
    tfs = [t.strip() for t in a.tfs.split(",")]

    print("## PHASE 1a -- the (MA-type x TF) IRONED TREND-system research grid, 2020 BAND ONLY")
    print(f"   split (within-2020): TRAIN {SPLIT['TRAIN']} / VAL {SPLIT['VAL']} / OOS {SPLIT['OOS']}")
    print(f"   SELECT (confirm-K {CONF_GRID}, exit {EXIT_GRID}) on TRAIN+VAL per (MA-type,TF); confirm ONCE on OOS")
    print(f"   cost maker (MAKER_RT={MAKER_RT}); causal lag-1; equal-weight u10; rank by NET (=wealth)")
    print(f"   MA types {MA_TYPES}; TFs {tfs}\n")

    # the slow-MA family (60<=max_len<150): the distinct 2MA+3MA configs
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=60))
    PR.STRATS.update(ma_cfg)
    slow = [n for n in ma_cfg if 60 <= max(_nums(n)) < 150]
    print(f"   slow-MA family: {len(slow)} distinct configs (2MA+3MA, 60<=max_len<150)\n")

    grid = {}            # (ma_type, tf) -> cell
    bench = {}           # tf -> {BUYHOLD:{...}, VOLTGT_BH:{...}}
    bench_eq = {}        # (tf, label) -> {dates, cum_pct}
    for tf in tfs:
        print(f"================================ {tf} ================================")
        panels = _build_panels(tf, slow)
        if len(panels) < 5:
            print(f"   [skip] only {len(panels)} assets with data")
            continue
        # benchmarks
        bm = benchmarks(panels, tf)
        bench[tf] = {}
        for label, (ser, exp) in bm.items():
            m = metrics(ser, exp if exp is not None else None, tf, *SPLIT["OOS"])
            if label == "BUYHOLD":
                m["coverage"] = 100.0; m["avg_exp"] = 1.0
            bench[tf][label] = m
            s = _slice(ser, *SPLIT["OOS"])
            if len(s):
                daily = s.resample("1D").apply(lambda v: float(np.prod(1 + v) - 1)).dropna()
                cum = (np.cumprod(1 + daily.to_numpy()) - 1) * 100
                bench_eq[(tf, label)] = {"dates": [d.strftime("%Y-%m-%d") for d in daily.index],
                                         "cum_pct": [round(float(v), 2) for v in cum]}
        print(f"   BENCH  BUYHOLD   OOS net {bench[tf]['BUYHOLD']['net']:>7}%  maxDD {bench[tf]['BUYHOLD']['maxdd']:>7}%  Sharpe {bench[tf]['BUYHOLD']['sharpe']:>5}")
        print(f"   BENCH  VOLTGT_BH OOS net {bench[tf]['VOLTGT_BH']['net']:>7}%  maxDD {bench[tf]['VOLTGT_BH']['maxdd']:>7}%  Sharpe {bench[tf]['VOLTGT_BH']['sharpe']:>5}  cov {bench[tf]['VOLTGT_BH'].get('coverage')}\n")

        print(f"   {'MA_type':8} {'conf':>4} {'exit':>11} {'OOSnet%':>8} {'Sharpe':>7} {'maxDD%':>7} {'cov%':>6} {'turn':>7} {'breadth':>8} {'n_eff':>6} {'p05':>8}")
        for mt in MA_TYPES:
            cell = run_cell(panels, slow, mt, tf)
            grid[(mt, tf)] = cell
            m = cell["oos"]; bn = cell["breadth_neff"]
            print(f"   {mt:8} {cell['selected_confirm_k']:>4} {cell['selected_exit']:>11} "
                  f"{str(m.get('net')):>8} {str(m.get('sharpe')):>7} {str(m.get('maxdd')):>7} "
                  f"{str(m.get('coverage')):>6} {str(m.get('turnover')):>7} "
                  f"{str(bn['breadth'])+'/'+str(bn['n_assets']):>8} {str(bn['n_eff']):>6} {str(m.get('p05')):>8}")
        # per-TF winner
        cells = [(mt, grid[(mt, tf)]) for mt in MA_TYPES if grid[(mt, tf)].get("oos")]
        if cells:
            wmt, w = max(cells, key=lambda kv: kv[1]["oos"].get("net", -1e9))
            print(f"   --> {tf} WINNER by net: {wmt} (net {w['oos']['net']}%, Sharpe {w['oos'].get('sharpe')}, "
                  f"DD {w['oos'].get('maxdd')}, cov {w['oos'].get('coverage')}%) vs VOLTGT_BH {bench[tf]['VOLTGT_BH']['net']}%\n")

    # ---- per-TF winners + family ordering summary ----
    winners = {}
    for tf in tfs:
        cells = [(mt, grid[(mt, tf)]) for mt in MA_TYPES if grid.get((mt, tf), {}).get("oos")]
        if cells:
            wmt, w = max(cells, key=lambda kv: kv[1]["oos"].get("net", -1e9))
            winners[tf] = {"ma_type": wmt, "oos_net": w["oos"]["net"], "sharpe": w["oos"].get("sharpe"),
                           "maxdd": w["oos"].get("maxdd"), "coverage": w["oos"].get("coverage"),
                           "confirm_k": w["selected_confirm_k"], "exit": w["selected_exit"],
                           "breadth": w["breadth_neff"]["breadth"], "n_eff": w["breadth_neff"]["n_eff"],
                           "p05": w["oos"].get("p05")}

    # family-avg net per TF (the ordering story)
    fam_by_tf = {}
    for tf in tfs:
        fam_by_tf[tf] = {}
        for fam, members in FAMILY_TAX.items():
            nets = [grid[(mt, tf)]["oos"]["net"] for mt in members
                    if grid.get((mt, tf), {}).get("oos") and grid[(mt, tf)]["oos"].get("net") is not None]
            fam_by_tf[tf][fam] = round(float(np.mean(nets)), 1) if nets else None

    print("\n" + "=" * 100)
    print("## PER-TF WINNER (by OOS net) + family ordering")
    print(f"   {'TF':5} {'winner':7} {'conf':>4} {'exit':>11} {'net%':>7} {'Sharpe':>7} {'DD%':>7} {'cov%':>6} "
          f"{'p05':>8} | {'VOLTGT%':>8} {'BH%':>7} | {'adaptive':>9} {'low_lag':>8} {'simple':>7}")
    for tf in tfs:
        if tf not in winners:
            continue
        wn = winners[tf]; fb = fam_by_tf[tf]
        print(f"   {tf:5} {wn['ma_type']:7} {wn['confirm_k']:>4} {wn['exit']:>11} {str(wn['oos_net']):>7} "
              f"{str(wn['sharpe']):>7} {str(wn['maxdd']):>7} {str(wn['coverage']):>6} {str(wn['p05']):>8} | "
              f"{str(bench[tf]['VOLTGT_BH']['net']):>8} {str(bench[tf]['BUYHOLD']['net']):>7} | "
              f"{str(fb['adaptive']):>9} {str(fb['low_lag']):>8} {str(fb['simple']):>7}")
    print("=" * 100)

    # ---- CHARTS ----
    p1 = chart_heatmap(grid, tfs, bench)
    p2 = chart_best_equity(grid, tfs, bench_eq)
    p3 = chart_family_by_tf(grid, tfs)
    print(f"\n[figure] {p1}\n[figure] {p2}\n[figure] {p3}")

    # ---- JSON ----
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "repro": {"command": "python -m strat.ma_type_tf_research --tfs " + ",".join(tfs),
                  "git_sha": sha, "cost_maker": MAKER_RT, "split": SPLIT, "warmup": WARMUP,
                  "family_n": len(slow), "conf_grid": CONF_GRID, "exit_grid": EXIT_GRID,
                  "ma_types": MA_TYPES, "tfs": tfs, "generated": stamp,
                  "honest_caveat": "2020 OOS (Oct-Dec) is a clean BULL (~0% bear); MA trend books are "
                                   "participating BETA, net < buy-hold is EXPECTED (participation tax); "
                                   "value is risk-adjusted + whole-cycle DD protection + cross-TF diversification "
                                   "the bull cannot show. Sharpe rewards sitting out the bull -- rank by NET."},
        "grid": {f"{mt}|{tf}": grid[(mt, tf)] for (mt, tf) in grid},
        "benchmarks": bench,
        "winners_by_tf": winners,
        "family_avg_net_by_tf": fam_by_tf,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    jp = OUT / "ma_type_tf_research.json"
    json.dump(payload, open(jp, "w", encoding="utf-8"), indent=1, default=str)
    print(f"[json] {jp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
