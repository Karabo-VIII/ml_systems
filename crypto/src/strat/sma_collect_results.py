"""Collect SMA profile results from existing run data and write final JSON.

This script collects:
1. OOS/band metrics from the live optimizer run output (already computed)
2. Move-catch metrics from existing ma_movecatch_*.json files
3. All-weather results already computed

Writes: runs/strat/sma_profile_final.json
"""
from __future__ import annotations
import json
import sys
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strat.ma_2020_config_leaderboard import (
    build_panels, config_book, run_cell, _metrics, SPLITS, SYMS, TRAIL, MINHOLD, MAKER_RT,
)
from strat.replay_distinct_grid import distinct_specs
from strat.ma_type_upgrade import _nums
from strat.data_expansion import block_bootstrap_distribution
from strat.portfolio_replay import apply_trail_stop
from strat.structural_fixes import min_hold as min_hold_fn
from strat.ma_type_upgrade import _MA
import strat.ma_2020_config_leaderboard as L

TFS = ["1d", "4h", "2h", "1h", "30m", "15m"]
MA_TYPE = "SMA"

YEAR = ("2020-01-01", "2021-01-01")
SPLIT_TRAIN = ("2020-01-01", "2020-07-01")
SPLIT_VAL   = ("2020-07-01", "2020-10-01")
SPLIT_OOS   = ("2020-10-01", "2021-01-01")
AW_SPAN     = ("2020-01-01", "2023-01-01")
AW_YEARS = {
    "2020_bull": ("2020-01-01", "2021-01-01"),
    "2021_mixed": ("2021-01-01", "2022-01-01"),
    "2022_bear": ("2022-01-01", "2023-01-01"),
}
LOOKBACK_D = 120
STEP_D = 30

OUT_STRAT = ROOT.parent / "runs" / "strat"
OUT_AW    = ROOT.parent / "runs" / "periods" / "ALL_WEATHER"

STYLE_MAP = {
    "1d": "long-term",
    "4h": "swing",
    "2h": "swing",
    "1h": "swing-intraday",
    "30m": "intraday",
    "15m": "intraday",
}


def _build_sma_band(tf):
    """Build SMA band for tf; return dict with metrics."""
    specs2 = distinct_specs("2MA", 0.15, max_n=60)
    specs3 = distinct_specs("3MA", 0.15, max_n=60)
    all_specs = {**specs2, **specs3}
    sma_specs = {}
    for name, (fam, params) in all_specs.items():
        new_name = "sma_" + "_".join(str(p) for p in _nums(name))
        sma_specs[new_name] = (fam, dict(params, type=MA_TYPE))

    all_periods = sorted({p for n in sma_specs for p in _nums(n)})
    panels = build_panels(tf, MA_TYPE, all_periods)
    if not panels:
        return {}

    cell = run_cell(panels, sma_specs, tf)
    band_cfgs = cell["band"]["band_configs"]
    n_band = len(band_cfgs)
    bs = cell["band"]

    if not band_cfgs:
        return {"n_band": 0}

    # Ensemble book
    books = []
    for cfg in band_cfgs:
        periods = _nums(cfg)
        if len(periods) < 2:
            continue
        b = config_book(panels, periods)
        if b is not None and len(b) > 10:
            books.append(b)
    if not books:
        return {"n_band": n_band, "band_configs": band_cfgs}

    book = pd.concat(books, axis=1).fillna(0.0).mean(axis=1).sort_index()

    m_train = _metrics(book, tf, *SPLIT_TRAIN)
    m_val   = _metrics(book, tf, *SPLIT_VAL)
    m_oos   = _metrics(book, tf, *SPLIT_OOS)
    m_full  = _metrics(book, tf, *YEAR)

    # Turnover
    flip_counts, bar_counts = [], []
    maf = _MA[MA_TYPE]
    for cfg in band_cfgs[:20]:
        periods = _nums(cfg)
        if len(periods) < 2:
            continue
        for sym, (c, ms, win, ret, cache) in panels.items():
            h0 = (cache[periods[0]] > cache[periods[1]]).astype(np.int8)
            h0 = np.nan_to_num(h0).astype(np.int8)
            h1 = apply_trail_stop(h0.copy(), c, TRAIL)[0].astype(np.int8)
            h2 = min_hold_fn(h1, MINHOLD).astype(np.int8)
            pos = np.zeros(len(c)); pos[1:] = h2[:-1]
            pos_win = pos[win]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos_win])))
            flip_counts.append(float(np.sum(flips)))
            bar_counts.append(float(len(pos_win)))

    turnover_rt = None
    if sum(bar_counts) > 0:
        turnover_rt = round(sum(flip_counts) / sum(bar_counts), 4)

    # Bootstrap p05 on OOS
    oos = book[(book.index >= pd.Timestamp(SPLIT_OOS[0])) & (book.index < pd.Timestamp(SPLIT_OOS[1]))].dropna()
    p05 = None
    if len(oos) >= 10:
        dist = block_bootstrap_distribution(oos.to_numpy(), n_boot=400, block=5, seed=42)
        p05 = round(float(dist["p05"]) * 100, 1)

    return {
        "n_band": n_band,
        "band_configs": band_cfgs,
        "band_fast_range": bs.get("band_2ma_fast_range"),
        "band_slow_range": bs.get("band_2ma_slow_range"),
        "train_net": m_train["net"],
        "val_net": m_val["net"],
        "oos_net": m_oos["net"],
        "oos_maxdd": m_oos["maxdd"],
        "full_net": m_full["net"],
        "p05_bootstrap": p05,
        "turnover_rt": turnover_rt,
        "_book": book,
        "_panels": panels,
        "_band_cfgs": band_cfgs,
    }


def _aw_profile(tf, band_cfgs, panels):
    """All-weather rolling-band ensemble profile."""
    # Extend window to 2020-2023
    orig_year = L.YEAR
    orig_split = L.SPLIT
    orig_splits = L.SPLITS
    L.YEAR = AW_SPAN
    L.SPLIT = {"TRAIN": AW_SPAN, "VAL": AW_SPAN, "OOS": AW_SPAN}
    L.SPLITS = {"TRAIN": AW_SPAN, "VAL": AW_SPAN, "OOS": AW_SPAN, "FULL": AW_SPAN}

    specs2 = distinct_specs("2MA", 0.15, max_n=60)
    all_periods = sorted({p for n in specs2 for p in _nums(n)})
    panels_aw = build_panels(tf, MA_TYPE, all_periods)

    cols = {}
    for name in specs2:
        periods = _nums(name)
        if len(periods) != 2:
            continue
        b = config_book(panels_aw, periods)
        if b is None or len(b) < 50:
            continue
        daily = b.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()
        cols[f"{periods[0]},{periods[1]}"] = daily

    L.YEAR = orig_year
    L.SPLIT = orig_split
    L.SPLITS = orig_splits

    if not cols:
        return {}

    sdf = pd.DataFrame(cols).sort_index()
    idx = sdf.index
    start = idx.min() + pd.Timedelta(days=LOOKBACK_D)
    pieces = []
    t = start
    cfgs = list(sdf.columns)
    while t < idx.max():
        nxt = t + pd.Timedelta(days=STEP_D)
        look = sdf[(idx >= t - pd.Timedelta(days=LOOKBACK_D)) & (idx < t)]
        fwd  = sdf[(idx >= t) & (idx < nxt)]
        if len(look) < 20 or len(fwd) < 2:
            t = nxt; continue
        look_net = (np.prod(1 + look.fillna(0.0).to_numpy(), axis=0) - 1) * 100
        band_mask = look_net > 0
        if not band_mask.any():
            band_mask = look_net == look_net.max()
        band_cfgs_now = [c for c, m in zip(cfgs, band_mask) if m]
        seg = fwd[band_cfgs_now].mean(axis=1).dropna()
        if len(seg):
            pieces.append(seg)
        t = nxt

    if not pieces:
        return {}
    rolling = pd.concat(pieces).sort_index()

    out = {}
    for yk, (lo, hi) in AW_YEARS.items():
        s = rolling[(rolling.index >= pd.Timestamp(lo)) & (rolling.index < pd.Timestamp(hi))]
        s = s.dropna()
        net = float(np.prod(1 + s.to_numpy()) - 1) * 100 if len(s) > 1 else 0.0
        if len(s) >= 2:
            eq = np.cumprod(1 + s.to_numpy()); pk = np.maximum.accumulate(eq)
            mdd = float(((eq - pk) / pk).min() * 100)
        else:
            mdd = 0.0
        out[yk] = {"net": round(net, 1), "maxdd": round(mdd, 1)}
    return out


def _get_sma_movecatch(tf):
    """Get SMA movecatch data from existing JSON files."""
    fname_map = {
        "1d": "ma_movecatch_1d.json",
        "4h": "ma_movecatch_4h.json",
        "2h": "ma_movecatch_2h.json",
        "1h": "ma_movecatch_1h.json",
        "30m": "ma_movecatch_30m.json",
        "15m": "ma_movecatch_15m.json",
    }
    p = OUT_STRAT / fname_map.get(tf, "")
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        sma = d.get("results", {}).get("0.05", {}).get("SMA", {})
        return sma
    except Exception:
        return {}


def main():
    print("\n=== SMA PROFILE COLLECTOR ===")
    per_tf = {}

    for tf in TFS:
        print(f"\n--- {tf} ---", flush=True)

        # Band metrics
        print(f"  Building SMA band {tf}...", flush=True)
        band = _build_sma_band(tf)
        book = band.pop("_book", None)
        panels = band.pop("_panels", None)
        band_cfgs = band.pop("_band_cfgs", None) or []

        # All-weather
        print(f"  Computing all-weather {tf}...", flush=True)
        aw = _aw_profile(tf, band_cfgs, panels)

        # Move-catch (from existing files)
        mc = _get_sma_movecatch(tf)

        per_tf[tf] = {
            "tf": tf,
            "style": STYLE_MAP[tf],
            "band_size": band.get("n_band", 0),
            "band_fast_range": band.get("band_fast_range"),
            "band_slow_range": band.get("band_slow_range"),
            "train_net": band.get("train_net"),
            "val_net": band.get("val_net"),
            "oos_net": band.get("oos_net"),
            "oos_maxdd": band.get("oos_maxdd"),
            "p05_bootstrap": band.get("p05_bootstrap"),
            "aw_2020_net": aw.get("2020_bull", {}).get("net"),
            "aw_2020_maxdd": aw.get("2020_bull", {}).get("maxdd"),
            "aw_2021_net": aw.get("2021_mixed", {}).get("net"),
            "aw_2021_maxdd": aw.get("2021_mixed", {}).get("maxdd"),
            "aw_2022_net": aw.get("2022_bear", {}).get("net"),
            "aw_2022_maxdd": aw.get("2022_bear", {}).get("maxdd"),
            "coverage": mc.get("coverage"),
            "entry_lag": mc.get("mean_entry_lag"),
            "capture_med": mc.get("raw_capture_mean"),
            "weighted_capture": mc.get("weighted_capture_mean"),
            "turnover_rt": band.get("turnover_rt"),
        }

        print(f"  [{tf}] band={band.get('n_band')} OOS={band.get('oos_net')}% "
              f"p05={band.get('p05_bootstrap')}% "
              f"AW: 2020={aw.get('2020_bull',{}).get('net')}% "
              f"2021={aw.get('2021_mixed',{}).get('net')}% "
              f"2022={aw.get('2022_bear',{}).get('net')}% "
              f"coverage={mc.get('coverage')} lag={mc.get('mean_entry_lag')}", flush=True)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_STRAT / f"sma_profile_final_{stamp}.json"
    out_data = {
        "ma_type": MA_TYPE,
        "generated": stamp,
        "methodology": {
            "band": "6/3/3 within-2020 (TRAIN=01..07, VAL=07..10, OOS=10..2021-01), positive all 3 splits",
            "sleeve": "MA-cross -> trail(0.10) -> min_hold(12) -> lag1 -> maker(0.0006 RT), fixed-EW fillna(0)",
            "aw": "rolling-band ensemble (120d lookback, 30d step) 2020-2023",
            "movecatch": "TRAIN window 5% threshold, causal band-ensemble, existing ma_movecatch files",
            "bootstrap": "block-bootstrap(n=400, block=5) on OOS returns, p05 of compound net",
        },
        "per_tf": per_tf,
    }
    out_path.write_text(json.dumps(out_data, indent=2, default=str), encoding="utf-8")
    print(f"\n[persisted] {out_path}")

    # Print summary table
    print("\n\n=== SMA PROFILE FINAL TABLE ===")
    print(f"{'TF':>4} | {'Style':>18} | {'Band':>5} | {'OOS%':>7} | {'p05%':>6} | "
          f"{'AW20%':>7} | {'AW21%':>7} | {'AW22%':>7} | {'AW22DD%':>8} | "
          f"{'Cov':>5} | {'Lag':>5} | {'Cap':>5} | {'WCap':>5} | {'Turn':>6}")
    print("-" * 130)
    for tf in TFS:
        r = per_tf[tf]
        def fmt(v, w=7):
            return str(round(v, 1) if v is not None else '--').rjust(w)
        def fmt3(v, w=5):
            return str(round(v, 3) if v is not None else '--').rjust(w)
        print(f"{tf:>4} | {r['style']:>18} | {r['band_size']:>5} | "
              f"{fmt(r['oos_net'])} | {fmt(r['p05_bootstrap'], 6)} | "
              f"{fmt(r['aw_2020_net'])} | {fmt(r['aw_2021_net'])} | "
              f"{fmt(r['aw_2022_net'])} | {fmt(r['aw_2022_maxdd'], 8)} | "
              f"{fmt3(r['coverage'])} | {fmt3(r['entry_lag'])} | "
              f"{fmt3(r['capture_med'])} | {fmt3(r['weighted_capture'])} | "
              f"{fmt3(r['turnover_rt'], 6)}")

    return out_data


if __name__ == "__main__":
    result = main()
