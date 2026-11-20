"""src/strat/ma_2020_config_leaderboard.py -- PER-CONFIG 2MA/3MA leaderboard + the WORKING BAND, 2020 band.

USER (deep-dive, extends ma_type_tf_research which only has FAMILY-ensemble winners_by_tf at the MA-TYPE
grain): for each (TIMEFRAME x MA-TYPE), rank the TOP-10 (winners first) and the BOTTOM-3 (losers last)
INDIVIDUAL 2MA AND 3MA configs on the 2020 band, show net/Sharpe/maxDD separately per TRAIN/VAL/OOS/FULL-2020
(does the rank TRANSFER?), identify the robust WORKING BAND (configs POSITIVE across TRAIN AND VAL AND OOS,
as a parameter range), and run the MANDATORY rank-stability/transfer check (Spearman (TRAIN+VAL) vs OOS +
top-10 overlap). The user's actual ask: "a band of working 2MA and 3MA configs that were discovered."

ABSOLUTE CONSTRAINT: STRICT LONG-ONLY + spot. held in {0,1}; NEVER short/inverse/long-short. 2020 BAND ONLY.

HONEST FRAMING (read FIRST -- band > rank): the whole prior investigation (D62 + the per-asset null)
found per-config rank is NOISE that does NOT transfer across regimes. So we TRUST THE BAND (the robust set
positive across all three splits = the set the FAMILY ENSEMBLE actually rides), NOT the exact #1. The
within-band ordering is regime-transient; the ENSEMBLE of the band is what is robust, not any single config.
The rank-stability number quantifies exactly how little the ordering transfers.

THE IRONED LONG-ONLY SLEEVE (identical to ma_2020_breakdown._cells, per-config not per-family):
  held = MA-cross (long while fast-MA > slow-MA [2MA] / fast>mid>slow [3MA])   -- strictly long/flat
       -> apply_trail_stop(0.10)  -> min_hold(12)  -> lag 1 bar  -> maker cost (MAKER_RT) on flips.
  equal-weight u10 book (mean across assets present per bar, skipna). Causal: value at t uses close[:t+1].

SPLIT (within-2020): ma_2020_breakdown.SPLIT -- TRAIN 2020-01..07 / VAL ..10 / OOS ..2021-01 ; FULL = the year.
CONFIG UNIVERSE: the FULL distinct grid (distinct_specs '2MA' + '3MA', max_n=60) -- NOT pre-restricted to the
slow family, so the working BAND is DISCOVERED as a (fast,slow) parameter range rather than assumed.

DATA CAVEATS (flagged in output): 2h is SYNTHESIZED from 1h (OHLC-resample, not native). SOL/AVAX have only
2020-H2 history (start ~Sep 2020) -> they are ABSENT from TRAIN, present in VAL/OOS; the book averages over
whatever assets exist per bar (skipna), so per-split breadth differs -- the band test still holds per split.

OUTPUTS:
  runs/periods/TRAIN/2020/DEEP_DIVE/CONFIG_LEADERBOARD.md      -- grouped TF -> MA-type; top-10+bottom-3 +
                                                                 BAND (param range) + rank-stability number
  runs/periods/TRAIN/2020/DEEP_DIVE/config_leaderboard.json    -- every config ranked per cell + band + stability
  runs/periods/TRAIN/2020/DEEP_DIVE/charts/config_band_heatmap.png   -- robust (fast,slow) regions per MA-type/TF
  runs/periods/TRAIN/2020/DEEP_DIVE/charts/rank_stability.png        -- (TRAIN+VAL) rank vs OOS rank (transfer noise)

RWYB: python -m strat.ma_2020_config_leaderboard [--tfs 1d,4h,2h,1h,30m,15m]
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
import yaml

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
from strat.structural_fixes import min_hold                           # noqa: E402

OUT = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
CHARTS = OUT / "charts"
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
        "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
TFS = ["1d", "4h", "2h", "1h", "30m", "15m"]
ANN = {"1d": 365, "4h": 365 * 6, "2h": 365 * 12, "1h": 365 * 24, "30m": 365 * 48, "15m": 365 * 96}
TRAIL = 0.10        # the ironed long-only trail (same as ma_2020_breakdown)
MINHOLD = 12        # the ironed min-hold

SPLITS = {"TRAIN": SPLIT["TRAIN"], "VAL": SPLIT["VAL"], "OOS": SPLIT["OOS"], "FULL": YEAR}


# ===========================================================================
# data: per-asset close arrays on [2020-WARMUP .. 2021), + the 2020 window mask
# ===========================================================================
def _asset_close(sym, cad):
    """(c, ms, win_mask) on [2020-WARMUP .. 2021). win_mask = bars inside 2020. None if too short."""
    try:
        o, h, l, c, ms = _panel(sym, cad)
    except Exception:
        return None
    s_ms = pd.Timestamp(YEAR[0]).value // 10**6
    e_ms = pd.Timestamp(YEAR[1]).value // 10**6
    e_idx = int(np.searchsorted(ms, e_ms))
    s_idx = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
    c2, ms2 = c[s_idx:e_idx], ms[s_idx:e_idx]
    if len(c2) < 40:
        return None
    win = ms2 >= s_ms
    if win.sum() < 20:
        return None
    return c2, ms2, win


def build_panels(cad, ma_type, all_periods):
    """{sym: (c, ms, win, ret, {period: ma_array})} -- each MA computed ONCE per (asset, period)."""
    maf = _MA[ma_type]
    out = {}
    for sym in SYMS:
        a = _asset_close(sym, cad)
        if a is None:
            continue
        c, ms, win = a
        ret = np.zeros(len(c)); ret[1:] = c[1:] / c[:-1] - 1.0
        cache = {p: maf(c, p) for p in all_periods}
        out[sym] = (c, ms, win, ret, cache)
    return out


# ===========================================================================
# the IRONED LONG-ONLY sleeve, per config -> equal-weight u10 book net Series (2020 only)
# ===========================================================================
def _held_cross(periods, cache):
    """LONG-ONLY cross: 2MA long while MA(fast)>MA(slow); 3MA long while fast>mid>slow. held in {0,1}."""
    mas = [cache[p] for p in periods]
    h = (mas[0] > mas[1]) if len(periods) == 2 else ((mas[0] > mas[1]) & (mas[1] > mas[2]))
    return np.nan_to_num(h).astype(np.int8)


def config_book(panels, periods):
    """equal-weight u10 book of bar-level net (maker, causal lag-1) for ONE config. LONG-ONLY ironed sleeve.
    Returns a pd.Series indexed by the 2020 bar timestamps. None if no assets."""
    cells = []
    for sym, (c, ms, win, ret, cache) in panels.items():
        h0 = _held_cross(periods, cache)                                   # long/flat in {0,1}
        h1 = apply_trail_stop(h0.copy(), c, TRAIL)[0].astype(np.int8)      # ironed 10% trail
        w = min_hold(h1, MINHOLD).astype(np.float64)                       # ironed min-hold(12)
        pos = np.zeros(len(c)); pos[1:] = w[:-1]                           # lag 1 bar (no look-ahead)
        flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
        net = (pos * ret - flips * (MAKER_RT / 2.0))[win]
        cells.append(pd.Series(net, index=pd.to_datetime(ms[win], unit="ms")))
    if not cells:
        return None
    # FIXED-EW (cadence-invariant): a missing/pre-listing bar = the slot is in CASH (0 return), NOT reweighted
    # to EW-of-present. skipna=True inflated fine-TF nets via thin-trading gaps + SOL/AVAX 2020 listing dates
    # (the cross-harness reconciliation MA_RECONCILIATION.md; 1d was bit-identical, 1h was +17pp inflated).
    return pd.concat(cells, axis=1).fillna(0.0).mean(axis=1).sort_index()


def _metrics(book, cad, lo, hi):
    """net%, sharpe (annualized), maxDD%, n_bars over [lo,hi)."""
    s = book[(book.index >= pd.Timestamp(lo)) & (book.index < pd.Timestamp(hi))].dropna()
    if len(s) < 5:
        return {"net": None, "sharpe": None, "maxdd": None, "n_bars": int(len(s))}
    x = s.to_numpy()
    eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq)
    return {
        "net": round(float(eq[-1] - 1) * 100, 1),
        "sharpe": round(float(np.mean(x) / (np.std(x) + 1e-12) * np.sqrt(ANN[cad])), 2),
        "maxdd": round(float(((eq - pk) / pk).min() * 100), 1),
        "n_bars": int(len(x)),
    }


# ===========================================================================
# rank-stability: Spearman of per-config (TRAIN+VAL) net vs OOS net + top-10 overlap
# ===========================================================================
def _spearman(a, b):
    """Spearman rho on paired arrays (rank-correlate, NaN-robust). None if <4 valid pairs."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 4:
        return None
    ra = pd.Series(a[m]).rank().to_numpy()
    rb = pd.Series(b[m]).rank().to_numpy()
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra @ ra) * (rb @ rb))
    return round(float(ra @ rb / den), 3) if den > 0 else None


# ===========================================================================
# one (MA-type, TF) cell: rank every config, find the band, compute stability
# ===========================================================================
def run_cell(panels, specs, cad):
    """specs = {name: (fam, params)} for this MA-type. Returns the full cell dict."""
    rows = []
    for name, (fam, params) in specs.items():
        periods = _nums(name)
        kind = "2MA" if len(periods) == 2 else "3MA"
        book = config_book(panels, periods)
        if book is None:
            continue
        per = {w: _metrics(book, cad, *rng) for w, rng in SPLITS.items()}
        if per["FULL"]["net"] is None:
            continue
        pos3 = (per["TRAIN"]["net"] is not None and per["TRAIN"]["net"] > 0 and
                per["VAL"]["net"] is not None and per["VAL"]["net"] > 0 and
                per["OOS"]["net"] is not None and per["OOS"]["net"] > 0)
        rows.append({"config": name, "kind": kind, "periods": periods,
                     "fast": periods[0], "slow": periods[-1],
                     "TRAIN": per["TRAIN"], "VAL": per["VAL"], "OOS": per["OOS"], "FULL": per["FULL"],
                     "positive_3way": bool(pos3)})
    # PRIMARY SORT: FULL-2020 net desc (winners first) -- wealth over the most data
    rows.sort(key=lambda r: (r["FULL"]["net"] if r["FULL"]["net"] is not None else -1e9), reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank_full"] = i

    # ---- the WORKING BAND: positive across TRAIN AND VAL AND OOS ----
    band = [r for r in rows if r["positive_3way"]]
    band_2ma = [r for r in band if r["kind"] == "2MA"]
    band_3ma = [r for r in band if r["kind"] == "3MA"]

    def _range(items, key):
        vals = [r[key] for r in items]
        return [min(vals), max(vals)] if vals else None

    band_summary = {
        "n_total_2ma": sum(1 for r in rows if r["kind"] == "2MA"),
        "n_total_3ma": sum(1 for r in rows if r["kind"] == "3MA"),
        "n_band_2ma": len(band_2ma), "n_band_3ma": len(band_3ma),
        "band_2ma_fast_range": _range(band_2ma, "fast"),
        "band_2ma_slow_range": _range(band_2ma, "slow"),
        "band_3ma_fast_range": _range(band_3ma, "fast"),
        "band_3ma_slow_range": _range(band_3ma, "slow"),
        "band_configs": [r["config"] for r in band],
    }

    # ---- rank-stability / transfer: Spearman (TRAIN+VAL net) vs OOS net + top-10 overlap ----
    tv, oo = [], []
    for r in rows:
        tn = r["TRAIN"]["net"]; vn = r["VAL"]["net"]; on = r["OOS"]["net"]
        # TRAIN+VAL combined score = sum of the two split nets (both available for all configs at this TF)
        tvn = (tn if tn is not None else np.nan) + (vn if vn is not None else np.nan)
        tv.append(tvn); oo.append(on if on is not None else np.nan)
    rho = _spearman(tv, oo)
    # top-10 overlap: of the TRAIN+VAL top-10, how many are in the OOS top-10
    order_tv = [i for i in np.argsort([-(t if np.isfinite(t) else -1e9) for t in tv])][:10]
    order_oo = [i for i in np.argsort([-(o if np.isfinite(o) else -1e9) for o in oo])][:10]
    overlap = len(set(order_tv) & set(order_oo))
    tv_top10_cfgs = [rows[i]["config"] for i in order_tv]
    oo_top10_cfgs = [rows[i]["config"] for i in order_oo]

    stability = {"spearman_trainval_vs_oos": rho, "top10_overlap": overlap,
                 "trainval_top10": tv_top10_cfgs, "oos_top10": oo_top10_cfgs,
                 "trainval_net": [round(float(t), 1) if np.isfinite(t) else None for t in tv],
                 "oos_net": [round(float(o), 1) if np.isfinite(o) else None for o in oo]}

    return {"ma_type": None, "cadence": cad, "n_assets": len(panels), "n_configs": len(rows),
            "ranked": rows, "band": band_summary, "stability": stability}


# ===========================================================================
# CHARTS
# ===========================================================================
def chart_band_heatmap(grid, tfs):
    """config_band_heatmap.png -- per (MA-type x TF), which (fast,slow) regions are in the robust BAND.
    One small scatter per cell: x=fast, y=slow, green=in-band (positive 3-way), grey=out. 2MA + 3MA(slow)."""
    nrows, ncols = len(MA_TYPES), len(tfs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 2.3 * nrows), squeeze=False)
    for i, mt in enumerate(MA_TYPES):
        for j, tf in enumerate(tfs):
            ax = axes[i][j]
            cell = grid.get((mt, tf))
            if not cell or not cell["ranked"]:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                for r in cell["ranked"]:
                    inb = r["positive_3way"]
                    mk = "o" if r["kind"] == "2MA" else "^"
                    ax.scatter(r["fast"], r["slow"], s=22 if inb else 10,
                               c=("#2ca02c" if inb else "#cccccc"), marker=mk,
                               edgecolors="k" if inb else "none", linewidths=0.4, alpha=0.9 if inb else 0.6)
                ax.set_yscale("log"); ax.set_xscale("log")
                ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(tf, fontsize=9)
            if j == 0:
                ax.set_ylabel(mt, fontsize=9)
    fig.suptitle("WORKING BAND map: (fast, slow) configs POSITIVE across TRAIN & VAL & OOS (2020) -- "
                 "GREEN=in-band (o=2MA, ^=3MA), grey=out. LONG-ONLY ironed sleeve. log-log axes.\n"
                 "Trust the BAND (the robust set the ensemble rides), NOT the exact #1 -- ordering is "
                 "regime-transient.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    CHARTS.mkdir(parents=True, exist_ok=True)
    p = CHARTS / "config_band_heatmap.png"
    fig.savefig(p, dpi=105); plt.close(fig)
    return p


def chart_rank_stability(grid, tfs):
    """rank_stability.png -- per (MA-type x TF): scatter of per-config (TRAIN+VAL net) vs (OOS net),
    annotated with Spearman rho + top-10 overlap. Shows the transfer noise: rank does NOT survive."""
    nrows, ncols = len(MA_TYPES), len(tfs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 2.3 * nrows), squeeze=False)
    for i, mt in enumerate(MA_TYPES):
        for j, tf in enumerate(tfs):
            ax = axes[i][j]
            cell = grid.get((mt, tf))
            if not cell or not cell["stability"]["oos_net"]:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                st = cell["stability"]
                tv = np.array([v if v is not None else np.nan for v in st["trainval_net"]], float)
                oo = np.array([v if v is not None else np.nan for v in st["oos_net"]], float)
                ax.scatter(tv, oo, s=10, c="#1f77b4", alpha=0.6)
                ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
                rho = st["spearman_trainval_vs_oos"]; ov = st["top10_overlap"]
                ax.text(0.04, 0.92, f"rho={rho}\novlp={ov}/10", transform=ax.transAxes,
                        fontsize=7, va="top", ha="left",
                        bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.8))
                ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(tf, fontsize=9)
            if j == 0:
                ax.set_ylabel(mt, fontsize=9)
            if i == nrows - 1:
                ax.set_xlabel("TRAIN+VAL net%", fontsize=7)
    fig.suptitle("RANK-STABILITY / TRANSFER: per-config (TRAIN+VAL net) vs (OOS net), per (MA-type x TF). "
                 "Spearman rho + top-10 overlap annotated.\nLow/negative rho + small overlap = the within-cell "
                 "config ORDERING does NOT transfer -> trust the BAND, not the #1.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    p = CHARTS / "rank_stability.png"
    fig.savefig(p, dpi=105); plt.close(fig)
    return p


# ===========================================================================
# MARKDOWN
# ===========================================================================
def _fmt_split(m):
    if not m or m.get("net") is None:
        return f"{'--':>7} {'--':>6} {'--':>7}"
    return f"{m['net']:>7} {str(m['sharpe']):>6} {m['maxdd']:>7}"


def _band_range_str(bs, kind):
    fr = bs.get(f"band_{kind.lower()}_fast_range"); sr = bs.get(f"band_{kind.lower()}_slow_range")
    n = bs.get(f"n_band_{kind.lower()}"); tot = bs.get(f"n_total_{kind.lower()}")
    if not fr or n == 0:
        return f"{kind}: BAND EMPTY (0/{tot} positive across all 3 splits)"
    return (f"{kind}: fast in [{fr[0]},{fr[1]}], slow in [{sr[0]},{sr[1]}] "
            f"-> {n}/{tot} configs positive across TRAIN & VAL & OOS")


def write_markdown(grid, tfs, bench, repro):
    yr = YEAR[0][:4]
    lines = []
    lines.append(f"# {yr} PER-CONFIG 2MA/3MA LEADERBOARD + the WORKING BAND (6mo TRAIN / 3mo VAL / 3mo OOS)")
    lines.append("")
    lines.append(f"STRICT LONG-ONLY + spot (held in {{0,1}}, no short/inverse anywhere). {yr} BAND ONLY. "
                 "Causal/lag-1, maker cost. Ironed sleeve = MA-cross -> 10% trail -> min_hold(12). "
                 f"6/3/3 split: TRAIN {SPLIT['TRAIN']} / VAL {SPLIT['VAL']} / OOS {SPLIT['OOS']}.")
    lines.append("")
    lines.append("## HONEST FRAMING -- the BAND is the deliverable, NOT the exact #1")
    lines.append("")
    lines.append("The prior investigation (D62 + the per-asset null) found per-config RANK is NOISE that "
                 "does not transfer across regimes. So:")
    lines.append("")
    lines.append("- **TRUST THE BAND** = the set of configs POSITIVE across TRAIN AND VAL AND OOS. This is "
                 "the robust set the FAMILY ENSEMBLE actually rides. Reported per cell as a (fast, slow) "
                 "parameter RANGE.")
    lines.append("- **DO NOT TRUST the exact ordering** within a cell. The within-band #1 is regime-transient. "
                 "The rank-stability number (Spearman (TRAIN+VAL) net vs OOS net + top-10 overlap) quantifies "
                 "how little the ordering transfers. Low rho / small overlap = the ranking is noise.")
    lines.append("- PRIMARY SORT below = FULL-2020 net (wealth over the most data = the most stable estimate). "
                 "Per-split net/Sharpe/maxDD shown so you can see whether a high FULL rank actually TRANSFERS.")
    lines.append("")
    lines.append(f"DATA CAVEATS: 2h is SYNTHESIZED from 1h (OHLC-resample). FIXED-EW (unlisted/missing bar = CASH, "
                 f"cadence-invariant -- NOT skipna). The OOS regime VARIES by year (check the per-TF buy-hold OOS "
                 f"net below): {yr} OOS = Oct-Dec. In a bull-OOS, under-participation vs buy-hold is EXPECTED (not a "
                 f"defect); in a down/flat-OOS, a de-risked book can 'beat' buy-hold by holding cash (EXPOSURE, not "
                 f"alpha) -- read the OOS buy-hold net + the config time-in before crediting any 'beat'.")
    lines.append("")
    lines.append(f"Repro: `{repro['command']}`  git_sha={repro['git_sha']}  cost=maker({MAKER_RT})  "
                 f"trail={TRAIL}  min_hold={MINHOLD}  split={SPLITS}")
    lines.append("")
    lines.append("All numbers are [MEASURED] from the run below (equal-weight u10, causal/lag-1, maker).")
    lines.append("")

    for tf in tfs:
        if not any((mt, tf) in grid for mt in MA_TYPES):
            continue
        bm = bench.get(tf, {})
        lines.append(f"\n# Timeframe: {tf}")
        bhn = bm.get("BUYHOLD", {}).get("FULL", {}).get("net")
        lines.append(f"_Benchmark (equal-weight u10 buy-hold, no cost): FULL-2020 net = {bhn}% "
                     f"(participation-tax reference)._")
        for mt in MA_TYPES:
            cell = grid.get((mt, tf))
            if not cell or not cell["ranked"]:
                continue
            st = cell["stability"]; bs = cell["band"]
            # display name: the period-tuple generator names every config 'ema_<periods>' regardless of the
            # MA-TYPE it is evaluated under -- render with the ACTUAL MA-type so the label is not misleading.
            disp = lambda c: f"{mt}({c.split('_', 1)[1].replace('_', ',')})" if '_' in str(c) else str(c)
            lines.append(f"\n## {tf} x {mt}   (n_assets={cell['n_assets']}, n_configs={cell['n_configs']})")
            lines.append("")
            lines.append(f"**Rank-stability (transfer):** Spearman (TRAIN+VAL net) vs (OOS net) = "
                         f"**rho={st['spearman_trainval_vs_oos']}**; TRAIN+VAL top-10 -> OOS top-10 overlap = "
                         f"**{st['top10_overlap']}/10**. "
                         + ("Ordering does NOT transfer -- trust the band, not the #1."
                            if (st['spearman_trainval_vs_oos'] is None or st['spearman_trainval_vs_oos'] < 0.5)
                            else "Some ordering persists, but still prefer the band."))
            lines.append("")
            lines.append(f"**WORKING BAND (positive across TRAIN & VAL & OOS):**")
            lines.append(f"- {_band_range_str(bs, '2MA')}")
            lines.append(f"- {_band_range_str(bs, '3MA')}")
            if bs["band_configs"]:
                show = bs["band_configs"][:24]
                more = "" if len(bs["band_configs"]) <= 24 else f" (+{len(bs['band_configs'])-24} more)"
                lines.append(f"- band members: {', '.join(disp(c) for c in show)}{more}")
            lines.append("")
            # table: top-10 + bottom-3
            lines.append("| rank | config | kind | TRAIN net/Sh/DD | VAL net/Sh/DD | OOS net/Sh/DD | "
                         "FULL net/Sh/DD | band? |")
            lines.append("|---:|---|---|---|---|---|---|:---:|")
            top = cell["ranked"][:10]
            bottom = cell["ranked"][-3:] if len(cell["ranked"]) > 13 else []
            for r in top:
                lines.append(f"| {r['rank_full']} | `{disp(r['config'])}` | {r['kind']} | "
                             f"{_fmt_split(r['TRAIN'])} | {_fmt_split(r['VAL'])} | {_fmt_split(r['OOS'])} | "
                             f"{_fmt_split(r['FULL'])} | {'YES' if r['positive_3way'] else '-'} |")
            if bottom:
                lines.append(f"| ... | _({cell['n_configs']-13} configs omitted)_ |  |  |  |  |  |  |")
                for r in bottom:
                    lines.append(f"| {r['rank_full']} | `{disp(r['config'])}` | {r['kind']} | "
                                 f"{_fmt_split(r['TRAIN'])} | {_fmt_split(r['VAL'])} | {_fmt_split(r['OOS'])} | "
                                 f"{_fmt_split(r['FULL'])} | {'YES' if r['positive_3way'] else '-'} |")
            lines.append("")

    # ---- global summary of rank-stability across all cells ----
    lines.append("\n# GLOBAL rank-stability summary (the transfer-noise headline)")
    lines.append("")
    lines.append("| TF | MA-type | n_band(2MA/3MA) | Spearman rho (TV vs OOS) | top-10 overlap |")
    lines.append("|---|---|---|---:|---:|")
    rhos = []
    for tf in tfs:
        for mt in MA_TYPES:
            cell = grid.get((mt, tf))
            if not cell or not cell["ranked"]:
                continue
            st = cell["stability"]; bs = cell["band"]
            if st["spearman_trainval_vs_oos"] is not None:
                rhos.append(st["spearman_trainval_vs_oos"])
            lines.append(f"| {tf} | {mt} | {bs['n_band_2ma']}/{bs['n_band_3ma']} | "
                         f"{st['spearman_trainval_vs_oos']} | {st['top10_overlap']}/10 |")
    if rhos:
        lines.append("")
        lines.append(f"**Median Spearman rho across all cells = {round(float(np.median(rhos)), 3)}** "
                     f"(mean {round(float(np.mean(rhos)), 3)}, n={len(rhos)} cells). "
                     f"The closer to 0 / negative, the more the within-cell config RANK is noise that does "
                     f"NOT transfer TRAIN+VAL -> OOS. This is the empirical basis for 'trust the band, not the #1'.")
    lines.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "CONFIG_LEADERBOARD.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p, (round(float(np.median(rhos)), 3) if rhos else None)


# ===========================================================================
# benchmark: equal-weight u10 buy-hold per split (no cost) -- the participation-tax reference
# ===========================================================================
def buyhold_bench(cad):
    cols = []
    for sym in SYMS:
        a = _asset_close(sym, cad)
        if a is None:
            continue
        c, ms, win = a
        r = np.zeros(len(c)); r[1:] = c[1:] / c[:-1] - 1.0
        cols.append(pd.Series(r[win], index=pd.to_datetime(ms[win], unit="ms")))
    if not cols:
        return {}
    bh = pd.concat(cols, axis=1).fillna(0.0).mean(axis=1).sort_index()  # FIXED-EW (cadence-invariant; see above)
    return {"BUYHOLD": {w: _metrics(bh, cad, *rng) for w, rng in SPLITS.items()}}


# ===========================================================================
# MAIN
# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.ma_2020_config_leaderboard")
    ap.add_argument("--tfs", default=",".join(TFS))
    ap.add_argument("--year", type=int, default=2020, help="within-YEAR 6/3/3 (default 2020; e.g. --year 2021)")
    a = ap.parse_args(argv)
    tfs = [t.strip() for t in a.tfs.split(",")]
    # YEAR override: same 6/3/3 (TRAIN 6mo / VAL 3mo / OOS 3mo) for any year, outputs under runs/periods/TRAIN/<year>.
    global YEAR, SPLIT, SPLITS, OUT, CHARTS
    yr = a.year
    YEAR = (f"{yr}-01-01", f"{yr + 1}-01-01")
    SPLIT = {"TRAIN": (f"{yr}-01-01", f"{yr}-07-01"), "VAL": (f"{yr}-07-01", f"{yr}-10-01"),
             "OOS": (f"{yr}-10-01", f"{yr + 1}-01-01")}
    SPLITS = {"TRAIN": SPLIT["TRAIN"], "VAL": SPLIT["VAL"], "OOS": SPLIT["OOS"], "FULL": YEAR}
    OUT = ROOT.parent / "runs" / "periods" / "TRAIN" / str(yr) / "DEEP_DIVE"
    CHARTS = OUT / "charts"

    # the FULL distinct config universe (NOT pre-restricted to the slow family)
    specs2 = distinct_specs("2MA", 0.15, max_n=60)
    specs3 = distinct_specs("3MA", 0.15, max_n=60)
    all_specs = {**specs2, **specs3}
    PR.STRATS.update(all_specs)
    all_periods = sorted({p for n in all_specs for p in _nums(n)})

    print("## PER-CONFIG 2MA/3MA LEADERBOARD + the WORKING BAND, 2020 BAND ONLY -- STRICT LONG-ONLY + spot")
    print(f"   split (within-2020): TRAIN {SPLIT['TRAIN']} / VAL {SPLIT['VAL']} / OOS {SPLIT['OOS']} / FULL {YEAR}")
    print(f"   ironed long-only sleeve: MA-cross -> trail({TRAIL}) -> min_hold({MINHOLD}); maker(MAKER_RT="
          f"{MAKER_RT}); causal lag-1; equal-weight u10")
    print(f"   config universe: {len(specs2)} 2MA + {len(specs3)} 3MA distinct configs; "
          f"MA types {MA_TYPES}; TFs {tfs}")
    print(f"   PRIMARY rank = FULL-2020 net (wealth). Band = positive across TRAIN&VAL&OOS. "
          f"Stability = Spearman(TV vs OOS)+top10 overlap.\n")

    grid = {}
    bench = {}
    for tf in tfs:
        print(f"================================ {tf} ================================")
        bench[tf] = buyhold_bench(tf)
        bhn = bench[tf].get("BUYHOLD", {}).get("FULL", {}).get("net")
        print(f"   BENCH buy-hold (u10, no cost): FULL-2020 net {bhn}%")
        for mt in MA_TYPES:
            panels = build_panels(tf, mt, all_periods)
            if len(panels) < 5:
                print(f"   {mt:8} [skip] only {len(panels)} assets with data")
                continue
            cell = run_cell(panels, all_specs, tf)
            cell["ma_type"] = mt
            grid[(mt, tf)] = cell
            st = cell["stability"]; bs = cell["band"]
            top = cell["ranked"][0] if cell["ranked"] else None
            _d = (lambda c: f"{mt}({c.split('_', 1)[1].replace('_', ',')})" if '_' in str(c) else str(c))
            top_str = (f"#1={_d(top['config'])} (FULL {top['FULL']['net']}%)" if top else "none")
            print(f"   {mt:8} cfgs={cell['n_configs']:3d}  band(2MA/3MA)={bs['n_band_2ma']}/{bs['n_band_3ma']}  "
                  f"rho={str(st['spearman_trainval_vs_oos']):>6}  top10_ovlp={st['top10_overlap']}/10  {top_str}")
        print()

    if not grid:
        print("[error] no cells produced -- check data availability")
        return 1

    # ---- charts ----
    p_band = chart_band_heatmap(grid, tfs)
    p_stab = chart_rank_stability(grid, tfs)
    print(f"[figure] {p_band}")
    print(f"[figure] {p_stab}")

    # ---- repro + JSON + markdown ----
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    repro = {"command": "python -m strat.ma_2020_config_leaderboard --tfs " + ",".join(tfs),
             "git_sha": sha, "generated": stamp, "cost_maker": MAKER_RT, "trail": TRAIL,
             "min_hold": MINHOLD, "split": SPLITS, "warmup": WARMUP,
             "n_2ma": len(specs2), "n_3ma": len(specs3), "ma_types": MA_TYPES, "tfs": tfs,
             "long_only": True, "short_logic": "NONE (strict long-only + spot, held in {0,1})",
             "caveats": ["2h synthesized from 1h", "SOL/AVAX 2020-H2 only -> absent from TRAIN",
                         "2020 OOS is a clean BULL -> participating-beta under-participation expected",
                         "per-config RANK is regime-transient noise -> trust the band, not the #1"]}

    p_md, median_rho = write_markdown(grid, tfs, bench, repro)

    payload = {
        "repro": repro,
        "benchmarks": bench,
        "median_spearman_rho": median_rho,
        "grid": {f"{mt}|{tf}": grid[(mt, tf)] for (mt, tf) in grid},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    jp = OUT / "config_leaderboard.json"
    json.dump(payload, open(jp, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[markdown] {p_md}")
    print(f"[json] {jp}")
    print(f"\n## HEADLINE: median Spearman rho (TRAIN+VAL vs OOS) across all cells = {median_rho} "
          f"-- the lower, the more config-rank is noise. Trust the BAND, not the #1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
