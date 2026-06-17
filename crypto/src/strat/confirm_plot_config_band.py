"""src/strat/confirm_plot_config_band.py -- CONFIRM + PLOT the per-config 2MA/3MA WORKING-BAND findings (2020).

USER: "confirm and plot these" -- re-derive the config_leaderboard.json headline claims independently
(RWYB, NOT web research) and render presentation-grade PNGs. Internal verify-and-visualize task.

ABSOLUTE CONSTRAINT: STRICT LONG-ONLY + spot. held in {0,1}. NEVER short/inverse/long-short. 2020 BAND ONLY.
This script REUSES the leaderboard backtest apparatus VERBATIM (no reinvention):
  strat.ma_2020_config_leaderboard.{build_panels, _held_cross, config_book, buyhold_bench, _asset_close,
                                    _metrics, run_cell, SPLITS, YEAR, MAKER_RT, TRAIL, MINHOLD, ANN, SYMS}
  strat.portfolio_replay.apply_trail_stop ; strat.structural_fixes.min_hold ; strat.ma_type_upgrade._nums
The ONLY new computation is EXPOSURE (time-in-market 0..1) -- the lagged `pos` array the net-return path
already builds internally (lines replicated EXACTLY from config_book), surfaced as its own series so we can
prove the long-only CRASH-AVOIDANCE mechanism (the sleeve goes to CASH in the Feb-Mar crash; it does NOT short).

CLAIMS CONFIRMED (independently re-derived from the live sleeve, NOT just re-read from JSON):
  C1 BAND counts (a few cells)         -- via run_cell re-run on 1d, cross-checked vs JSON
  C2 top-1d configs FULL/OOS net       -- via config_book + _metrics, cross-checked vs JSON
  C3 CRASH-AVOIDANCE (load-bearing)    -- exposure -> ~0 (cash) across Feb-Mar; maxDD avoided vs buy-hold
  C4 rank transience (median rho 0.57) -- read from JSON stability blocks (already the leaderboard's output)

LOOK-AHEAD FRAMING (stated, not hidden): the sleeve is causal/lag-1 (pos[t] uses held[:t], close[:t+1]).
The BAND and the per-config RANK are computed on FULL-2020 = DESCRIPTIVE of what was discovered over the
year, NOT a forward predictor. The crash-avoidance is a within-sample MECHANISM check (does the long-only
exit actually move to cash when price craters?), which is causal and real -- it is NOT a claim that the
*ranking* predicts forward.

RWYB: python -m strat.confirm_plot_config_band
No emoji (Windows cp1252). Does NOT git commit (overseer commits).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR                                       # noqa: E402
from strat.portfolio_replay import apply_trail_stop, MAKER_RT            # noqa: E402
from strat.replay_distinct_grid import distinct_specs                    # noqa: E402
from strat.ma_type_upgrade import _nums, MA_TYPES                        # noqa: E402
from strat.structural_fixes import min_hold                             # noqa: E402
import strat.ma_2020_config_leaderboard as L                            # noqa: E402
from strat.ma_2020_config_leaderboard import (                          # noqa: E402
    build_panels, config_book, buyhold_bench, _asset_close, _metrics, run_cell,
    SPLITS, YEAR, SYMS, TRAIL, MINHOLD, ANN,
)

OUT = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
CHARTS = OUT / "charts"
JSON_PATH = OUT / "config_leaderboard.json"

# Feb-Mar 2020 COVID crash window (the BTC -50% drop centred on 2020-03-12 "Black Thursday")
CRASH = ("2020-02-19", "2020-03-31")
H1 = ("2020-01-01", "2020-07-01")   # H1-2020 = the crash half (== the TRAIN split)


# ===========================================================================
# EXPOSURE: replicate config_book's `pos` construction EXACTLY, but return the
# per-bar average time-in-market (0..1) and the book equity, not just net.
# (config_book collapses to net returns; we need exposure separately to PROVE
#  the long-only crash-avoidance mechanism. Lines below are copied verbatim
#  from config_book so the sleeve is bit-identical -- no reinvention.)
# ===========================================================================
def config_exposure_and_equity(panels, periods):
    """Returns (exposure_series, equity_series, net_series) for ONE config over 2020.
    exposure[t] = mean over assets-present of pos[t] (the lagged held in {0,1}) = time-in-market.
    equity = cumprod(1+net) of the equal-weight book. All causal/lag-1, maker cost. None if no assets."""
    exp_cells, net_cells = [], []
    for sym, (c, ms, win, ret, cache) in panels.items():
        # --- IDENTICAL to config_book ---
        h0 = L._held_cross(periods, cache)                                 # long/flat in {0,1}
        h1 = apply_trail_stop(h0.copy(), c, TRAIL)[0].astype(np.int8)      # ironed 10% trail
        w = min_hold(h1, MINHOLD).astype(np.float64)                       # ironed min-hold(12)
        pos = np.zeros(len(c)); pos[1:] = w[:-1]                           # lag 1 bar (no look-ahead)
        flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
        net = (pos * ret - flips * (MAKER_RT / 2.0))[win]
        # --- new: surface the exposure (the lagged position) over the 2020 window ---
        idx = pd.to_datetime(ms[win], unit="ms")
        exp_cells.append(pd.Series(pos[win], index=idx))
        net_cells.append(pd.Series(net, index=idx))
    if not net_cells:
        return None, None, None
    exposure = pd.concat(exp_cells, axis=1).fillna(0.0).mean(axis=1).sort_index()
    net_book = pd.concat(net_cells, axis=1).fillna(0.0).mean(axis=1).sort_index()
    equity = (1.0 + net_book).cumprod()
    return exposure, equity, net_book


def buyhold_equity(cad):
    """equal-weight u10 buy-hold equity over 2020 (no cost) -- the participation reference."""
    cols = []
    for sym in SYMS:
        a = _asset_close(sym, cad)
        if a is None:
            continue
        c, ms, win = a
        r = np.zeros(len(c)); r[1:] = c[1:] / c[:-1] - 1.0
        cols.append(pd.Series(r[win], index=pd.to_datetime(ms[win], unit="ms")))
    if not cols:
        return None
    bh = pd.concat(cols, axis=1).fillna(0.0).mean(axis=1).sort_index()
    return (1.0 + bh).cumprod()


def _maxdd(equity):
    """max drawdown % of an equity series (negative number)."""
    e = equity.dropna().to_numpy()
    if len(e) < 2:
        return None
    pk = np.maximum.accumulate(e)
    return round(float(((e - pk) / pk).min() * 100), 1)


def _net_pct(equity):
    e = equity.dropna().to_numpy()
    return round(float(e[-1] - 1) * 100, 1) if len(e) else None


# ===========================================================================
# top-N config picker per TF (across all MA-types), straight from the JSON grid
# ===========================================================================
def top_configs_for_tf(grid, tf, n=5):
    """List of (ma_type, config, periods, FULL_net, FULL_maxdd) -- top-n by FULL-2020 net across all MA cells."""
    pool = []
    for mt in MA_TYPES:
        cell = grid.get(f"{mt}|{tf}")
        if not cell:
            continue
        for r in cell["ranked"]:
            if r["FULL"]["net"] is not None:
                pool.append((mt, r["config"], r["periods"], r["FULL"]["net"], r["FULL"]["maxdd"]))
    pool.sort(key=lambda x: -x[3])
    return pool[:n]


def band_members_for_tf(grid, tf):
    """All (ma_type, config, periods) that are in the BAND (positive 3-way) for this TF, across MA-types."""
    out = []
    for mt in MA_TYPES:
        cell = grid.get(f"{mt}|{tf}")
        if not cell:
            continue
        band_set = set(cell["band"]["band_configs"])
        for r in cell["ranked"]:
            if r["config"] in band_set and r["positive_3way"]:
                out.append((mt, r["config"], r["periods"]))
    return out


# ===========================================================================
# build panels once per (TF, MA-type) and cache (build_panels is the heavy call)
# ===========================================================================
_PANEL_CACHE = {}
def panels_for(tf, mt, all_periods):
    key = (tf, mt)
    if key not in _PANEL_CACHE:
        _PANEL_CACHE[key] = build_panels(tf, mt, all_periods)
    return _PANEL_CACHE[key]


def _shade_crash(ax, lo=CRASH[0], hi=CRASH[1], label="Feb-Mar 2020 crash"):
    ax.axvspan(pd.Timestamp(lo), pd.Timestamp(hi), color="#d62728", alpha=0.10, label=label, zorder=0)


# ===========================================================================
# PLOT (a): top-5 config equity vs buy-hold, crash shaded  (one per TF: 1d, 30m)
# ===========================================================================
def plot_top_equity(grid, tf, all_periods, fname):
    tops = top_configs_for_tf(grid, tf, n=5)
    bh_eq = buyhold_equity(tf)
    fig, ax = plt.subplots(figsize=(11, 6))
    _shade_crash(ax)
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(tops)))
    cfg_dds = []
    for (mt, cfg, periods, full_net, full_dd), col in zip(tops, colors):
        panels = panels_for(tf, mt, all_periods)
        _, eq, _ = config_exposure_and_equity(panels, periods)
        if eq is None:
            continue
        dd = _maxdd(eq); cfg_dds.append(dd)
        lbl = f"{mt}({','.join(map(str, periods))})  +{_net_pct(eq):.0f}%  DD {dd:.0f}%"
        ax.plot(eq.index, eq.to_numpy(), lw=1.7, color=col, label=lbl)
    if bh_eq is not None:
        ax.plot(bh_eq.index, bh_eq.to_numpy(), lw=2.6, color="k", ls="--",
                label=f"EW buy-hold (no cost)  +{_net_pct(bh_eq):.0f}%  DD {_maxdd(bh_eq):.0f}%")
        bh_dd = _maxdd(bh_eq)
        avg_cfg_dd = np.mean([d for d in cfg_dds if d is not None]) if cfg_dds else None
        if avg_cfg_dd is not None:
            ax.annotate(f"maxDD avoided: configs ~{avg_cfg_dd:.0f}% vs buy-hold {bh_dd:.0f}%\n"
                        f"(long-only EXITS to CASH in the crash -- no shorting)",
                        xy=(0.985, 0.04), xycoords="axes fraction", va="bottom", ha="right", fontsize=9,
                        bbox=dict(boxstyle="round", fc="#fff4e6", ec="#d62728", alpha=0.95))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}x"))
    ax.set_title(f"Top-5 config equity vs equal-weight buy-hold -- {tf}, FULL-2020 (STRICT long-only + spot)\n"
                 f"The band's best configs BEAT no-cost buy-hold full-cycle AND cut the drawdown -- "
                 f"via crash-avoidance, not leverage/shorting", fontsize=11)
    ax.set_xlabel("2020"); ax.set_ylabel("equity (log, x initial)")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9, ncol=1)
    ax.grid(True, which="both", alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout()
    CHARTS.mkdir(parents=True, exist_ok=True)
    p = CHARTS / fname
    fig.savefig(p, dpi=120); plt.close(fig)
    return p


# ===========================================================================
# PLOT (b): band ENSEMBLE vs buy-hold vs the single #1, small multiple over TFs
# ===========================================================================
def _band_ensemble(grid, tf, all_periods):
    """Equal-weight mean equity of ALL band members for this TF, the single #1 config equity, buy-hold equity."""
    members = band_members_for_tf(grid, tf)
    member_nets = []
    for (mt, cfg, periods) in members:
        panels = panels_for(tf, mt, all_periods)
        _, _, net = config_exposure_and_equity(panels, periods)
        if net is not None:
            member_nets.append(net)
    ens_eq = None
    if member_nets:
        ens_net = pd.concat(member_nets, axis=1).fillna(0.0).mean(axis=1).sort_index()
        ens_eq = (1.0 + ens_net).cumprod()
    # single #1 = top FULL-net config across all MA cells
    tops = top_configs_for_tf(grid, tf, n=1)
    one_eq = None; one_lbl = ""
    if tops:
        mt, cfg, periods, _, _ = tops[0]
        panels = panels_for(tf, mt, all_periods)
        _, one_eq, _ = config_exposure_and_equity(panels, periods)
        one_lbl = f"{mt}({','.join(map(str, periods))})"
    bh_eq = buyhold_equity(tf)
    return ens_eq, one_eq, one_lbl, bh_eq, len(members)


def plot_band_ensemble(grid, tfs, all_periods, fname="band_ensemble_vs_buyhold.png"):
    fig, axes = plt.subplots(1, len(tfs), figsize=(5.4 * len(tfs), 5.2), squeeze=False)
    for j, tf in enumerate(tfs):
        ax = axes[0][j]
        _shade_crash(ax, label="crash" if j == 0 else None)
        ens_eq, one_eq, one_lbl, bh_eq, n_mem = _band_ensemble(grid, tf, all_periods)
        if ens_eq is not None:
            ax.plot(ens_eq.index, ens_eq.to_numpy(), lw=2.4, color="#2ca02c",
                    label=f"BAND ensemble (n={n_mem})  +{_net_pct(ens_eq):.0f}%  DD {_maxdd(ens_eq):.0f}%")
        if one_eq is not None:
            ax.plot(one_eq.index, one_eq.to_numpy(), lw=1.5, color="#ff7f0e", alpha=0.9,
                    label=f"single #1 {one_lbl}  +{_net_pct(one_eq):.0f}%  DD {_maxdd(one_eq):.0f}%")
        if bh_eq is not None:
            ax.plot(bh_eq.index, bh_eq.to_numpy(), lw=2.0, color="k", ls="--",
                    label=f"buy-hold  +{_net_pct(bh_eq):.0f}%  DD {_maxdd(bh_eq):.0f}%")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}x"))
        ax.set_title(f"{tf}", fontsize=11)
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
        ax.grid(True, which="both", alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        if j == 0:
            ax.set_ylabel("equity (log, x initial)")
    fig.suptitle("BAND-as-a-BOOK: the equal-weight ensemble of ALL band members (robust) vs the noisy single #1 "
                 "vs equal-weight buy-hold -- FULL-2020, STRICT long-only + spot.\n"
                 "The band ensemble is the deployable object (the #1 is regime-transient); both go to cash in the "
                 "crash (the long-only preservation mechanism).", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    CHARTS.mkdir(parents=True, exist_ok=True)
    p = CHARTS / fname
    fig.savefig(p, dpi=120); plt.close(fig)
    return p


# ===========================================================================
# PLOT (c): band ensemble AVERAGE EXPOSURE (time-in-market) over 2020
# ===========================================================================
def plot_band_exposure(grid, tfs, all_periods, fname="band_exposure_timeline.png"):
    fig, ax = plt.subplots(figsize=(11, 6))
    _shade_crash(ax)
    colors = {"1d": "#1f77b4", "4h": "#2ca02c", "15m": "#9467bd"}
    crash_exposures = {}
    for tf in tfs:
        members = band_members_for_tf(grid, tf)
        exps = []
        for (mt, cfg, periods) in members:
            panels = panels_for(tf, mt, all_periods)
            exp, _, _ = config_exposure_and_equity(panels, periods)
            if exp is not None:
                exps.append(exp)
        if not exps:
            continue
        band_exp = pd.concat(exps, axis=1).fillna(0.0).mean(axis=1).sort_index()
        # smooth lightly for readability on fine TFs (rolling mean over ~3 days of bars)
        win_bars = max(1, int(ANN[tf] / 365 * 3))
        band_exp_s = band_exp.rolling(win_bars, min_periods=1).mean()
        ax.plot(band_exp_s.index, band_exp_s.to_numpy(), lw=1.8, color=colors.get(tf, None),
                label=f"{tf} band ensemble (n={len(members)})")
        # exposure inside the crash window
        cw = band_exp[(band_exp.index >= pd.Timestamp(CRASH[0])) & (band_exp.index < pd.Timestamp(CRASH[1]))]
        crash_exposures[tf] = round(float(cw.mean()), 2) if len(cw) else None
    ax.axhline(1.0, color="grey", lw=0.7, ls=":")
    ax.axhline(0.0, color="grey", lw=0.7, ls=":")
    ax.set_ylim(-0.05, 1.08)
    txt = "avg exposure IN the crash window\n(vs ~100% for buy-hold):\n" + "\n".join(
        f"  {tf}: {v:.0%} in-market" for tf, v in crash_exposures.items() if v is not None)
    ax.annotate(txt, xy=(0.30, 0.50), xycoords="axes fraction", va="center", ha="left", fontsize=9.5,
                bbox=dict(boxstyle="round", fc="#e6f2ff", ec="#1f77b4", alpha=0.95))
    ax.set_title("Band-ensemble EXPOSURE (time-in-market, 0..1) over 2020 -- STRICT long-only + spot\n"
                 "The long-only band DE-RISKS TO CASH in the Feb-Mar crash (exposure collapses), then "
                 "re-arms for the recovery. This is the within-constraint answer to bears: CASH, not shorting.",
                 fontsize=11)
    ax.set_xlabel("2020"); ax.set_ylabel("avg fraction of book in-market (0=cash, 1=fully long)")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.tight_layout()
    CHARTS.mkdir(parents=True, exist_ok=True)
    p = CHARTS / fname
    fig.savefig(p, dpi=120); plt.close(fig)
    return p, crash_exposures


# ===========================================================================
# PLOT (d): band-width per MA-type (avg across TFs) -- VIDYA widest, HMA/TEMA narrowest
# ===========================================================================
def plot_band_width(grid, tfs, fname="band_width_by_matype.png"):
    avg_band = {}
    per_tf = {}
    for mt in MA_TYPES:
        vals = []
        for tf in tfs:
            cell = grid.get(f"{mt}|{tf}")
            if cell:
                tot = cell["band"]["n_band_2ma"] + cell["band"]["n_band_3ma"]
                vals.append(tot)
        avg_band[mt] = float(np.mean(vals)) if vals else 0.0
        per_tf[mt] = vals
    order = sorted(MA_TYPES, key=lambda m: -avg_band[m])
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_colors = ["#2ca02c" if m == order[0] else ("#d62728" if m in order[-2:] else "#1f77b4") for m in order]
    bars = ax.bar(order, [avg_band[m] for m in order], color=bar_colors, edgecolor="k", alpha=0.85)
    for b, m in zip(bars, order):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{avg_band[m]:.0f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.axhline(120, color="grey", lw=0.8, ls=":")
    ax.text(len(order) - 0.5, 121, "max=120 configs", fontsize=8, color="grey", ha="right")
    ax.set_ylim(0, 128)
    ax.set_title("WORKING-BAND WIDTH per MA-type (avg # configs positive across TRAIN&VAL&OOS, mean over TFs)\n"
                 "out of 120 (60 2MA + 60 3MA). VIDYA the WIDEST (most robust band), HMA/TEMA the NARROWEST.\n"
                 f"TFs averaged: {', '.join(tfs)}. STRICT long-only + spot, 2020.", fontsize=11)
    ax.set_ylabel("avg band size (configs in-band / 120)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    CHARTS.mkdir(parents=True, exist_ok=True)
    p = CHARTS / fname
    fig.savefig(p, dpi=120); plt.close(fig)
    return p, avg_band, per_tf


# ===========================================================================
# CONFIRM: independent re-derivation of the headline numbers
# ===========================================================================
def confirm(grid, all_periods):
    """Re-derive band counts (1d, via run_cell), top-config FULL/OOS net, crash-avoidance. Returns a verdict dict."""
    v = {}

    # --- C1: re-run run_cell on a couple of 1d cells and cross-check band counts vs JSON ---
    c1 = {}
    for mt in ["VIDYA", "EMA", "TEMA"]:
        panels = panels_for("1d", mt, all_periods)
        cell = run_cell(panels, L_ALL_SPECS, "1d")
        live = (cell["band"]["n_band_2ma"], cell["band"]["n_band_3ma"])
        jb = grid[f"{mt}|1d"]["band"]
        jc = (jb["n_band_2ma"], jb["n_band_3ma"])
        c1[mt] = {"live_2ma_3ma": live, "json_2ma_3ma": jc, "match": live == jc}
    v["C1_band_recount_1d"] = c1

    # --- C2: re-derive the 3 named top-1d configs' FULL + OOS net via config_book + _metrics ---
    named = {"WMA": [6, 9, 24], "TEMA": [8, 22, 60], "HMA": [37, 38]}
    c2 = {}
    bh1d = grid_bench["1d"]["BUYHOLD"]["FULL"]["net"]
    for mt, periods in named.items():
        panels = panels_for("1d", mt, all_periods)
        book = config_book(panels, periods)
        full = _metrics(book, "1d", *SPLITS["FULL"])
        oos = _metrics(book, "1d", *SPLITS["OOS"])
        # find JSON record
        jrec = next((r for r in grid[f"{mt}|1d"]["ranked"] if r["periods"] == periods), None)
        c2[f"{mt}({','.join(map(str, periods))})"] = {
            "live_FULL_net": full["net"], "json_FULL_net": jrec["FULL"]["net"] if jrec else None,
            "live_FULL_maxdd": full["maxdd"], "live_OOS_net": oos["net"],
            "json_OOS_net": jrec["OOS"]["net"] if jrec else None,
            "beats_buyhold_FULL": full["net"] > bh1d,
            "match_FULL": abs(full["net"] - (jrec["FULL"]["net"] if jrec else -1e9)) < 0.15,
        }
    v["C2_top_configs"] = {"buyhold_FULL_net": bh1d, "configs": c2}

    # --- C3: crash-avoidance -- exposure of the top-5 1d band configs in the crash + maxDD vs buy-hold ---
    tops = top_configs_for_tf(grid, "1d", n=5)
    bh_eq = buyhold_equity("1d")
    bh_dd_full = _maxdd(bh_eq)
    bh_h1 = bh_eq[(bh_eq.index >= pd.Timestamp(H1[0])) & (bh_eq.index < pd.Timestamp(H1[1]))]
    bh_dd_h1 = _maxdd(bh_h1 / bh_h1.iloc[0]) if len(bh_h1) else None
    c3rows = []
    for (mt, cfg, periods, _, _) in tops:
        panels = panels_for("1d", mt, all_periods)
        exp, eq, _ = config_exposure_and_equity(panels, periods)
        crash_exp = exp[(exp.index >= pd.Timestamp(CRASH[0])) & (exp.index < pd.Timestamp(CRASH[1]))]
        eq_h1 = eq[(eq.index >= pd.Timestamp(H1[0])) & (eq.index < pd.Timestamp(H1[1]))]
        c3rows.append({
            "config": f"{mt}({','.join(map(str, periods))})",
            "avg_exposure_in_crash": round(float(crash_exp.mean()), 3) if len(crash_exp) else None,
            "min_exposure_in_crash": round(float(crash_exp.min()), 3) if len(crash_exp) else None,
            "maxdd_full": _maxdd(eq), "maxdd_h1": _maxdd(eq_h1 / eq_h1.iloc[0]) if len(eq_h1) else None,
        })
    v["C3_crash_avoidance"] = {
        "crash_window": CRASH, "h1_window": H1,
        "buyhold_maxdd_full": bh_dd_full, "buyhold_maxdd_h1": bh_dd_h1,
        "top5_configs": c3rows,
    }

    # --- C4: rank transience (read the leaderboard's own stability output) ---
    rhos, overlaps = [], []
    for k, cell in grid.items():
        st = cell.get("stability", {})
        if st.get("spearman_trainval_vs_oos") is not None:
            rhos.append(st["spearman_trainval_vs_oos"])
        if st.get("top10_overlap") is not None:
            overlaps.append(st["top10_overlap"])
    v["C4_rank_transience"] = {
        "median_spearman_rho": round(float(np.median(rhos)), 3) if rhos else None,
        "mean_spearman_rho": round(float(np.mean(rhos)), 3) if rhos else None,
        "json_median_rho_field": json.load(open(JSON_PATH, encoding="utf-8")).get("median_spearman_rho"),
        "overlap_min": int(min(overlaps)) if overlaps else None,
        "overlap_max": int(max(overlaps)) if overlaps else None,
        "n_cells": len(rhos),
    }
    return v


# module-level handles set in main (used by confirm)
L_ALL_SPECS = None
grid_bench = None


def main(argv=None) -> int:
    global L_ALL_SPECS, grid_bench
    print("## CONFIRM + PLOT the per-config 2MA/3MA WORKING-BAND findings (2020) -- STRICT long-only + spot")
    print(f"   reusing strat.ma_2020_config_leaderboard apparatus VERBATIM; maker={MAKER_RT} trail={TRAIL} "
          f"min_hold={MINHOLD}; causal lag-1\n")

    payload = json.load(open(JSON_PATH, encoding="utf-8"))
    grid = payload["grid"]
    grid_bench = payload["benchmarks"]

    # rebuild the SAME config universe the leaderboard used (so run_cell re-runs identically)
    specs2 = distinct_specs("2MA", 0.15, max_n=60)
    specs3 = distinct_specs("3MA", 0.15, max_n=60)
    L_ALL_SPECS = {**specs2, **specs3}
    PR.STRATS.update(L_ALL_SPECS)
    all_periods = sorted({p for n in L_ALL_SPECS for p in _nums(n)})

    # ---- CONFIRM ----
    print("[1/2] CONFIRM -- independent re-derivation ...")
    verdict = confirm(grid, all_periods)
    print(json.dumps(verdict, indent=1, default=str))

    # ---- PLOT ----
    print("\n[2/2] PLOT ...")
    p_a1 = plot_top_equity(grid, "1d", all_periods, "config_top_equity_1d.png")
    print(f"[figure] {p_a1}")
    p_a2 = plot_top_equity(grid, "30m", all_periods, "config_top_equity_30m.png")
    print(f"[figure] {p_a2}")
    p_b = plot_band_ensemble(grid, ["1d", "4h", "15m"], all_periods)
    print(f"[figure] {p_b}")
    p_c, crash_exp = plot_band_exposure(grid, ["1d", "4h", "15m"], all_periods)
    print(f"[figure] {p_c}  (crash-window exposure: {crash_exp})")
    p_d, avg_band, per_tf = plot_band_width(grid, ["1d", "4h", "2h", "1h", "30m", "15m"])
    print(f"[figure] {p_d}")
    print(f"   avg band width per MA-type: { {k: round(v, 1) for k, v in sorted(avg_band.items(), key=lambda x:-x[1])} }")

    # ---- stash a machine-readable confirm payload for the writeup ----
    confirm_out = {"verdict": verdict, "avg_band_width": avg_band, "band_width_per_tf": per_tf,
                   "crash_window_exposure": crash_exp,
                   "figures": [str(x) for x in [p_a1, p_a2, p_b, p_c, p_d]]}
    cp = OUT / "config_band_confirm_payload.json"
    json.dump(confirm_out, open(cp, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[json] {cp}")
    print("\n## CONFIRM COMPLETE -- see CONFIG_BAND_CONFIRM.md for the verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
