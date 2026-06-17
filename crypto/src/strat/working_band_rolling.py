"""src/strat/working_band_rolling.py -- the WORKING-BAND + ROLLING-SELECTION all-weather test (the user's strat).

USER /orc 2026-06-16 (verbatim spirit): "find the WORKING REGION/band within an MA type (e.g. for SMA-2MA returns
come from a fast>slow region; for EMA from a slow vs ultra-slow region). Express specific configs to trade, but
pick them with ROLLING knowledge + ROLLING performance, and see if THAT translates. I'm not saying (2,21) will
translate, but the BAND/cluster should. Chop season may need FASTER MAs to catch oscillations. And we must trade
ALL-WEATHER, not just depend on the bull (total returns look amazing in hindsight)."

WHAT THIS BUILDS (descriptive + a real walk-forward strat test):
  1. BAND RETURN-SURFACE per MA type: a (fast, slow) heatmap of per-year net for every 2MA config -> SHOWS the
     working region (where returns concentrate) + whether it is CONSISTENT across MA types and across years.
  2. ROLLING-FROM-BAND selection (the strat): walk-forward, each rebalance pick -- from the rolling BAND (configs
     positive over the trailing lookback) -- the recent-best config; trade it the next step; roll. This is "use
     rolling knowledge + rolling performance to pick one." Compared to: BAND-ENSEMBLE (EW the rolling band),
     STATIC-#1 (the naive all-time best -- the thing we KNOW doesn't transfer), and BUY-HOLD.
  3. CHOP-AWARE variant: when the slow MA is FLAT (chop), restrict the rolling band to FASTER configs (the user's
     "need faster MAs in chop to catch oscillations"). Tested vs the plain rolling pick.
  4. ALL-WEATHER: per-year net + maxDD for 2020 (bull) / 2021 (mega-bull -> Q4 decline) / 2022 (BEAR). The honest
     test: does rolling-from-band stay POSITIVE / preserve in the bear, or does it only work in the bull?

DISCIPLINE: STRICT long-only + spot; fixed-EW u10; MA-cross -> trail10 -> min_hold(12) -> lag1 -> maker (the same
ironed sleeve as the leaderboard). 2MA configs (clean fast/slow surface). NO look-ahead: the rolling band + pick
use ONLY trailing data; the traded window is strictly forward. UNSEEN (2025-26) untouched. No emoji.

RWYB: python -m strat.working_band_rolling --tfs 4h --matypes EMA,SMA  (then all)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.ma_2020_config_leaderboard as CL                                # noqa: E402
from strat.replay_distinct_grid import distinct_specs                        # noqa: E402
from strat.ma_type_upgrade import _nums, MA_TYPES, _ema                      # noqa: E402
import strat.portfolio_replay as PR                                          # noqa: E402

OUT = ROOT.parent / "runs" / "periods" / "ALL_WEATHER"
CHARTS = OUT / "charts"
OUT.mkdir(parents=True, exist_ok=True); CHARTS.mkdir(parents=True, exist_ok=True)
SPAN = ("2020-01-01", "2023-01-01")                                          # bull / mixed / BEAR
YEARS = {"2020_bull": ("2020-01-01", "2021-01-01"), "2021_mixed": ("2021-01-01", "2022-01-01"),
         "2022_bear": ("2022-01-01", "2023-01-01")}
LOOKBACK_D = 120                                                             # rolling "knowledge" window (days)
STEP_D = 30                                                                  # rebalance step (days)


def _2ma_series(ma_type, tf):
    """DataFrame [date x 'fast,slow'] of DAILY-compounded net for every 2MA config (fixed-EW u10 ironed sleeve),
    over SPAN. Reuses the leaderboard's build_panels/config_book by pointing CL.YEAR at the full span."""
    CL.YEAR = SPAN; CL.SPLIT = {"TRAIN": SPAN, "VAL": SPAN, "OOS": SPAN}; CL.SPLITS = {**CL.SPLIT, "FULL": SPAN}
    specs2 = distinct_specs("2MA", 0.15, max_n=60)
    PR.STRATS.update(specs2)
    periods_all = sorted({p for n in specs2 for p in _nums(n)})
    panels = CL.build_panels(tf, ma_type, periods_all)
    if len(panels) < 5:
        return None
    cols = {}
    for name in specs2:
        periods = _nums(name)
        if len(periods) != 2:
            continue
        book = CL.config_book(panels, periods)                              # bar-level net Series, fixed-EW u10
        if book is None or len(book) < 50:
            continue
        daily = book.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()
        cols[f"{periods[0]},{periods[1]}"] = daily
    if not cols:
        return None
    return pd.DataFrame(cols).sort_index()


def _net(s):
    s = s.dropna()
    return float(np.prod(1 + s.to_numpy()) - 1) * 100 if len(s) > 1 else 0.0


def _maxdd(s):
    s = s.dropna()
    if len(s) < 2:
        return 0.0
    eq = np.cumprod(1 + s.to_numpy()); pk = np.maximum.accumulate(eq)
    return float(((eq - pk) / pk).min() * 100)


def _buyhold_daily(tf):
    CL.YEAR = SPAN
    cols = []
    for sym in CL.SYMS:
        a = CL._asset_close(sym, tf)
        if a is None:
            continue
        c, ms, win = a
        r = np.zeros(len(c)); r[1:] = c[1:] / c[:-1] - 1.0
        cols.append(pd.Series(r[win], index=pd.to_datetime(ms[win], unit="ms")))
    bh = pd.concat(cols, axis=1).fillna(0.0).mean(axis=1).sort_index()
    return bh.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()


# =====================================================================================================
# ROLLING-FROM-BAND: walk-forward pick (no look-ahead)
# =====================================================================================================
def _rolling(series_df, mode="pick", chop_bh=None):
    """Walk-forward over SPAN. At each rebalance date t: rolling BAND = configs with trailing-LOOKBACK net > 0;
    `pick` = the band member with the best trailing-LOOKBACK net; `ensemble` = EW the band. Trade the chosen
    book over the next STEP days (strictly forward). chop_bh (daily buy-hold) optional -> in CHOP (trailing slow
    trend ~ flat) restrict the band to the FASTER half (smaller slow). Returns (stitched daily net, picks)."""
    idx = series_df.index
    start = idx.min() + pd.Timedelta(days=LOOKBACK_D)
    pieces, picks = [], []
    t = start
    cfgs = list(series_df.columns)
    slows = np.array([int(c.split(",")[1]) for c in cfgs])
    med_slow = np.median(slows)
    while t < idx.max():
        nxt = t + pd.Timedelta(days=STEP_D)
        look = series_df[(idx >= t - pd.Timedelta(days=LOOKBACK_D)) & (idx < t)]
        fwd = series_df[(idx >= t) & (idx < nxt)]
        if len(look) < 20 or len(fwd) < 2:
            t = nxt; continue
        look_net = (np.prod(1 + look.fillna(0.0).to_numpy(), axis=0) - 1) * 100   # trailing net per config
        band_mask = look_net > 0                                            # the rolling BAND (trailing-positive)
        # CHOP-AWARE (efficiency-ratio gate): ER = |net move| / total path over the trailing window. LOW ER =
        # lots of motion but little net progress = CHOP -> restrict the band to FASTER configs (catch oscillations);
        # HIGH ER = clean trend -> keep the full band. (ER is Kaufman's trend/chop discriminator -- a real regime
        # signal, unlike the prior crude |net|<15 flat-market proxy which did not help.)
        if chop_bh is not None:
            bh_look = chop_bh[(chop_bh.index >= t - pd.Timedelta(days=LOOKBACK_D)) & (chop_bh.index < t)]
            if len(bh_look) > 10:
                cum = np.cumprod(1 + bh_look.to_numpy())
                net_move = abs(cum[-1] - 1.0)
                path = float(np.sum(np.abs(np.diff(np.concatenate([[1.0], cum])))))
                er = net_move / (path + 1e-12)                             # 0=pure chop .. 1=pure trend
                if er < 0.30:                                              # choppy regime
                    band_mask = band_mask & (slows <= med_slow)            # -> faster half of the band
        if not band_mask.any():
            band_mask = look_net == look_net.max()                          # fallback: least-bad
        band_cfgs = [c for c, m in zip(cfgs, band_mask) if m]
        if mode == "pick":
            best = max(band_cfgs, key=lambda c: look_net[cfgs.index(c)])
            seg = fwd[best].dropna(); picks.append(best)
        else:  # ensemble
            seg = fwd[band_cfgs].mean(axis=1).dropna(); picks.append(f"EW{len(band_cfgs)}")
        if len(seg):
            pieces.append(seg)
        t = nxt
    if not pieces:
        return None, picks
    return pd.concat(pieces).sort_index(), picks


def _static_1(series_df):
    """The naive baseline: the all-time (full-SPAN) best 2MA config, traded the whole span (in-sample-peeked #1)."""
    full = (np.prod(1 + series_df.fillna(0.0).to_numpy(), axis=0) - 1)
    best = series_df.columns[int(np.argmax(full))]
    return series_df[best].dropna(), best


def _per_year(daily):
    out = {}
    for yk, (lo, hi) in YEARS.items():
        s = daily[(daily.index >= pd.Timestamp(lo)) & (daily.index < pd.Timestamp(hi))]
        out[yk] = {"net": round(_net(s), 1), "maxdd": round(_maxdd(s), 1), "n": int(len(s.dropna()))}
    return out


# =====================================================================================================
# CHARTS
# =====================================================================================================
def chart_band_surface(surfaces, tf):
    """(fast, slow) net heatmap per MA type per year -- WHERE returns concentrate (the working region)."""
    mts = list(surfaces.keys()); yrs = list(YEARS.keys())
    fig, axes = plt.subplots(len(mts), len(yrs), figsize=(4.2 * len(yrs), 3.1 * len(mts)), squeeze=False)
    for i, mt in enumerate(mts):
        for j, yk in enumerate(yrs):
            ax = axes[i][j]; pts = surfaces[mt][yk]
            if not pts:
                ax.text(0.5, 0.5, "no data", ha="center"); ax.set_xticks([]); ax.set_yticks([]); continue
            f = np.array([p[0] for p in pts]); s = np.array([p[1] for p in pts]); v = np.array([p[2] for p in pts])
            vmax = np.nanpercentile(np.abs(v), 95) or 1.0
            sc = ax.scatter(f, s, c=v, cmap="RdYlGn", vmin=-vmax, vmax=vmax, s=26, edgecolors="k", linewidths=0.3)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(yk, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{mt}\nslow", fontsize=8)
            if i == len(mts) - 1:
                ax.set_xlabel("fast", fontsize=7)
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=5)
    fig.suptitle(f"WORKING-BAND return surface @ {tf}: 2MA (fast, slow) net% per MA-type per year (GREEN=+, RED=-, "
                 f"log-log). Where the GREEN concentrates = the working region.\nIf the green zone is CONSISTENT "
                 f"across MA-types + persists across years, the BAND translates (even if the exact #1 doesn't).",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    p = CHARTS / f"band_return_surface_{tf}.png"
    fig.savefig(p, dpi=108); plt.close(fig); print(f"   [chart] {p}"); return p


def chart_rolling_equity(results, bh_daily, tf):
    """Per MA-type: rolling-pick vs band-ensemble vs static-#1 vs buy-hold equity over 2020-2022 (all-weather)."""
    mts = list(results.keys())
    ncol = 2; nrow = int(np.ceil(len(mts) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(8.5 * ncol, 3.0 * nrow), squeeze=False)
    bh_eq = (1 + bh_daily).cumprod()
    for k, mt in enumerate(mts):
        ax = axes[k // ncol][k % ncol]; r = results[mt]
        ax.plot(bh_eq.index, bh_eq.values, color="black", lw=2.0, label="buy-hold", zorder=2)
        for key, col in (("rolling_pick", "#1f77b4"), ("band_ensemble", "#2ca02c"),
                         ("rolling_pick_chop", "#9467bd"), ("static_1", "#d62728")):
            s = r.get(key + "_daily")
            if s is not None and len(s) > 2:
                eq = (1 + s).cumprod()
                ax.plot(eq.index, eq.values, color=col, lw=1.2, alpha=0.9, label=key)
        for yb in ("2021-01-01", "2022-01-01"):
            ax.axvline(pd.Timestamp(yb), color="grey", ls="--", lw=0.7, alpha=0.6)
        ax.set_yscale("log"); ax.set_title(f"{mt} @ {tf}", fontsize=9); ax.tick_params(labelsize=6)
        if k == 0:
            ax.legend(fontsize=6, loc="upper left")
        ax.grid(alpha=0.25)
    for k in range(len(mts), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"ROLLING-FROM-BAND vs band-ensemble vs static-#1 vs buy-hold @ {tf} (log-y, 2020|2021|2022). "
                 f"Walk-forward, NO look-ahead (static-#1 IS in-sample-peeked).\nThe all-weather question: does "
                 f"rolling-from-band stay positive / preserve in 2022 BEAR, or only ride the bull?", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = CHARTS / f"rolling_from_band_equity_{tf}.png"
    fig.savefig(p, dpi=108); plt.close(fig); print(f"   [chart] {p}"); return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.working_band_rolling")
    ap.add_argument("--tfs", default="4h")
    ap.add_argument("--matypes", default=",".join(MA_TYPES))
    a = ap.parse_args(argv)
    tfs = [t.strip() for t in a.tfs.split(",") if t.strip()]
    mts = [m.strip() for m in a.matypes.split(",") if m.strip()]
    allout = {}
    for tf in tfs:
        print(f"\n================= {tf} =================")
        bh_daily = _buyhold_daily(tf)
        surfaces, results = {}, {}
        for mt in mts:
            sdf = _2ma_series(mt, tf)
            if sdf is None:
                print(f"   {mt}: no series"); continue
            # band surface per year
            surfaces[mt] = {}
            for yk, (lo, hi) in YEARS.items():
                sub = sdf[(sdf.index >= pd.Timestamp(lo)) & (sdf.index < pd.Timestamp(hi))]
                pts = []
                for c in sdf.columns:
                    f, s = (int(x) for x in c.split(","))
                    pts.append([f, s, _net(sub[c])])
                surfaces[mt][yk] = pts
            # rolling strategies
            rp, picks_p = _rolling(sdf, "pick")
            re_, _ = _rolling(sdf, "ensemble")
            rpc, _ = _rolling(sdf, "pick", chop_bh=bh_daily)
            s1, s1cfg = _static_1(sdf)
            results[mt] = {"rolling_pick_daily": rp, "band_ensemble_daily": re_,
                           "rolling_pick_chop_daily": rpc, "static_1_daily": s1, "static_1_cfg": s1cfg,
                           "rolling_pick": _per_year(rp) if rp is not None else {},
                           "band_ensemble": _per_year(re_) if re_ is not None else {},
                           "rolling_pick_chop": _per_year(rpc) if rpc is not None else {},
                           "static_1": _per_year(s1), "n_distinct_picks": len(set(picks_p))}
            r = results[mt]
            print(f"   {mt:6} | rolling-pick {[r['rolling_pick'].get(y,{}).get('net') for y in YEARS]} "
                  f"| ensemble {[r['band_ensemble'].get(y,{}).get('net') for y in YEARS]} "
                  f"| chop {[r['rolling_pick_chop'].get(y,{}).get('net') for y in YEARS]} "
                  f"| static#1 {[r['static_1'].get(y,{}).get('net') for y in YEARS]}")
        bh_year = _per_year(bh_daily)
        print(f"   {'BUYHOLD':6} | per-year net {[bh_year[y]['net'] for y in YEARS]} "
              f"maxdd {[bh_year[y]['maxdd'] for y in YEARS]}")
        chart_band_surface(surfaces, tf)
        chart_rolling_equity(results, bh_daily, tf)
        # strip the daily series before persisting (keep per-year summaries)
        slim = {mt: {k: v for k, v in r.items() if not k.endswith("_daily")} for mt, r in results.items()}
        allout[tf] = {"buyhold_per_year": bh_year, "results": slim}
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT / f"working_band_rolling_{stamp}.json"
    json.dump({"repro": {"git_sha": sha, "span": SPAN, "lookback_d": LOOKBACK_D, "step_d": STEP_D,
                         "years": YEARS}, "by_tf": allout}, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
