"""src/strat/vidya_tf_profile.py -- VIDYA MA TYPE: full TF profile across all 6 cadences.

GOAL: optimised VIDYA profile per TF {1d,4h,2h,1h,30m,15m}.
  - Working band: configs positive across TRAIN+VAL+OOS within-2020
  - Ironed sleeve: trail(0.10)+min_hold(12)+lag1+MAKER cost
  - Fixed-EW book: fillna(0.0).mean(axis=1) -- NOT skipna
  - All-weather: 2020/2021/2022 compound net from band ensemble
  - OOS net + block-bootstrap p05
  - Move-catch: REUSES existing ma_movecatch_{1d,4h,2h,1h,30m}.json; runs 15m
  - Per-asset: does per-asset band beat pooled?

RWYB: python -m strat.vidya_tf_profile
No emoji (Windows cp1252). Does NOT git commit.
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
np.seterr(invalid="ignore", divide="ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR                                       # noqa
from strat.portfolio_replay import apply_trail_stop, MAKER_RT             # noqa
from strat.replay_distinct_grid import distinct_specs                     # noqa
from strat.ma_type_upgrade import _nums, _MA                              # noqa
from strat.ma_2020_breakdown import _panel, SPLIT, YEAR, WARMUP           # noqa
from strat.ma_2020_config_leaderboard import (                            # noqa
    build_panels, config_book, _metrics, SPLITS, SYMS, ANN, run_cell,
)
from strat.structural_fixes import min_hold                               # noqa
from strat.data_expansion import block_bootstrap_distribution             # noqa
import strat.ma_2020_config_leaderboard as L                             # noqa

OUT = ROOT.parent / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)

MA_TYPE = "VIDYA"
TFS = ["1d", "4h", "2h", "1h", "30m", "15m"]

STYLE_MAP = {
    "1d": "long-term",
    "4h": "swing",
    "2h": "swing",
    "1h": "swing",
    "30m": "intraday",
    "15m": "intraday",
}


# ============================================================
# BAND: get VIDYA working-band configs
# ============================================================

def _get_vidya_band(cad: str):
    """Build the VIDYA working band for one TF. Returns (band_configs, panels, band_meta)."""
    specs2 = distinct_specs("2MA", 0.15, max_n=60)
    specs3 = distinct_specs("3MA", 0.15, max_n=60)
    vidya_specs = {}
    for name, (fam, params) in {**specs2, **specs3}.items():
        new_name = f"vidya_" + "_".join(str(p) for p in _nums(name))
        vidya_specs[new_name] = (fam, dict(params, type=MA_TYPE))

    all_periods = sorted(set(p for n in vidya_specs for p in _nums(n)))
    panels = build_panels(cad, MA_TYPE, all_periods)
    if len(panels) < 3:
        return [], panels, {}

    cell = run_cell(panels, vidya_specs, cad)
    if not cell:
        return [], panels, {}

    band_cfgs = cell["band"]["band_configs"] if cell.get("band") else []
    n2 = cell["band"]["n_band_2ma"]
    n3 = cell["band"]["n_band_3ma"]
    n_total = cell["n_configs"]
    print(f"  [{cad}] VIDYA band: {len(band_cfgs)} configs ({n2} 2MA + {n3} 3MA) of {n_total} total")
    return band_cfgs, panels, cell.get("band", {})


# ============================================================
# ENSEMBLE BOOK: EW mean of band-configs net streams
# ============================================================

def _band_ensemble_book(panels, band_cfgs):
    """Build the EW ensemble book across band configs (2020 window). Returns pd.Series or None."""
    if not band_cfgs:
        return None
    cells = []
    for cfg in band_cfgs:
        periods = _nums(cfg)
        if len(periods) < 2:
            continue
        bk = config_book(panels, periods)
        if bk is not None and len(bk) > 0:
            cells.append(bk)
    if not cells:
        return None
    return pd.concat(cells, axis=1).fillna(0.0).mean(axis=1).sort_index()


# ============================================================
# ALL-WEATHER: build book over 2020-2022
# ============================================================

def _get_allweather_book(cad: str, band_cfgs: list):
    """Build the VIDYA all-weather book 2020-2022. Returns pd.Series or None."""
    if not band_cfgs:
        return None
    orig_year = L.YEAR
    orig_split = L.SPLIT
    orig_splits = L.SPLITS

    AW = ("2020-01-01", "2023-01-01")
    L.YEAR = AW
    L.SPLIT = {"TRAIN": AW, "VAL": AW, "OOS": AW}
    L.SPLITS = {**L.SPLIT, "FULL": AW}

    all_periods = sorted(set(p for cfg in band_cfgs for p in _nums(cfg)))
    aw_panels = build_panels(cad, MA_TYPE, all_periods)

    cells = []
    for cfg in band_cfgs:
        periods = _nums(cfg)
        if len(periods) < 2:
            continue
        bk = config_book(aw_panels, periods)
        if bk is not None and len(bk) > 0:
            cells.append(bk)

    L.YEAR = orig_year
    L.SPLIT = orig_split
    L.SPLITS = orig_splits

    if not cells:
        return None
    return pd.concat(cells, axis=1).fillna(0.0).mean(axis=1).sort_index()


def _slice_net(s, lo, hi):
    sub = s[(s.index >= pd.Timestamp(lo)) & (s.index < pd.Timestamp(hi))].dropna()
    if len(sub) < 5:
        return None
    return round(float(np.cumprod(1 + sub.to_numpy())[-1] - 1) * 100, 1)


def _slice_maxdd(s, lo, hi):
    sub = s[(s.index >= pd.Timestamp(lo)) & (s.index < pd.Timestamp(hi))].dropna()
    if len(sub) < 5:
        return None
    eq = np.cumprod(1 + sub.to_numpy()); pk = np.maximum.accumulate(eq)
    return round(float(((eq - pk) / pk).min() * 100), 1)


# ============================================================
# MOVE-CATCH: load existing JSON or run 15m
# ============================================================

def _load_movecatch(cad: str):
    p = OUT / f"ma_movecatch_{cad}.json"
    if not p.exists():
        return None
    with open(p) as f:
        data = json.load(f)
    results = data.get("results", {})
    thresh_data = results.get("0.05", results.get(0.05, {}))
    return thresh_data.get("VIDYA")


def _run_15m_movecatch():
    p = OUT / "ma_movecatch_15m.json"
    if p.exists():
        print("  [15m] movecatch JSON exists, loading VIDYA cell.")
        return _load_movecatch("15m")
    print("  [15m] Running 15m VIDYA movecatch (this may take ~5min) ...")
    try:
        from strat.ma_movecatch_decomp import analyze_cell
        cell = analyze_cell("15m", MA_TYPE, 0.05)
        out_data = {"cadence": "15m", "results": {"0.05": {MA_TYPE: cell}}}
        with open(p, "w") as f:
            json.dump(out_data, f, indent=2, default=str)
        print(f"  [15m] movecatch written -> {p}")
        return cell
    except Exception as e:
        print(f"  [15m] movecatch ERROR: {e}")
        return None


# ============================================================
# TURNOVER: approximate RT/year from ensemble
# ============================================================

def _turnover_rt_yr(panels, band_cfgs, cad, max_cfgs=20, max_syms=5):
    if not band_cfgs:
        return None
    flip_counts = []
    for sym, (c, ms, win, ret, cache) in list(panels.items())[:max_syms]:
        per_cfg_pos = []
        for cfg in band_cfgs[:max_cfgs]:
            periods = _nums(cfg)
            maf = _MA[MA_TYPE]
            mas = [maf(c, p) for p in periods]
            if len(periods) == 2:
                h0 = (mas[0] > mas[1]).astype(np.int8)
            else:
                h0 = ((mas[0] > mas[1]) & (mas[1] > mas[2])).astype(np.int8)
            h0 = np.nan_to_num(h0).astype(np.int8)
            h1 = apply_trail_stop(h0.copy(), c, 0.10)[0].astype(np.int8)
            h2 = min_hold(h1, 12).astype(np.float64)
            pos = np.zeros(len(c)); pos[1:] = h2[:-1]
            per_cfg_pos.append(pos[win].astype(float))
        if not per_cfg_pos:
            continue
        ens_pos = np.mean(per_cfg_pos, axis=0)
        flips = np.sum(np.abs(np.diff(np.concatenate([[0.0], ens_pos]))))
        n_bars = len(ens_pos)
        if n_bars > 0:
            flip_counts.append(flips / n_bars * ANN[cad])
    if not flip_counts:
        return None
    return round(float(np.mean(flip_counts)), 1)


# ============================================================
# PER-ASSET: does per-asset band-selection beat pooled?
# ============================================================

def _per_asset_oos_nets(panels, band_cfgs):
    per_asset = {}
    oos_lo, oos_hi = SPLITS["OOS"]
    for sym, (c, ms, win, ret, cache) in panels.items():
        sym_cells = []
        for cfg in band_cfgs:
            periods = _nums(cfg)
            if len(periods) < 2:
                continue
            maf = _MA[MA_TYPE]
            mas = [maf(c, p) for p in periods]
            if len(periods) == 2:
                h0 = (mas[0] > mas[1]).astype(np.int8)
            else:
                h0 = ((mas[0] > mas[1]) & (mas[1] > mas[2])).astype(np.int8)
            h0 = np.nan_to_num(h0).astype(np.int8)
            h1 = apply_trail_stop(h0.copy(), c, 0.10)[0].astype(np.int8)
            h2 = min_hold(h1, 12).astype(np.float64)
            pos = np.zeros(len(c)); pos[1:] = h2[:-1]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            net = pos * ret - flips * (MAKER_RT / 2.0)
            net_s = pd.Series(net[win], index=pd.to_datetime(ms[win], unit="ms"))
            sym_cells.append(net_s)
        if not sym_cells:
            continue
        bk = pd.concat(sym_cells, axis=1).fillna(0.0).mean(axis=1).sort_index()
        oos_s = bk[(bk.index >= pd.Timestamp(oos_lo)) & (bk.index < pd.Timestamp(oos_hi))].dropna()
        if len(oos_s) > 5:
            per_asset[sym] = round(float(np.cumprod(1 + oos_s.to_numpy())[-1] - 1) * 100, 1)
    return per_asset


# ============================================================
# MAIN
# ============================================================

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"VIDYA TF PROFILE {ts}\n")
    results = {}

    for cad in TFS:
        print(f"\n{'='*60}")
        print(f"TF: {cad}")

        band_cfgs, panels, band_meta = _get_vidya_band(cad)
        band_size = len(band_cfgs)

        if band_size == 0:
            print(f"  No band configs")
            results[cad] = {"tf": cad, "band_size": 0, "style": STYLE_MAP[cad], "note": "no band"}
            continue

        # --- 2020 OOS metrics ---
        ens_book = _band_ensemble_book(panels, band_cfgs)
        oos_net = oos_maxdd = p05_val = None
        if ens_book is not None:
            oos_lo, oos_hi = SPLITS["OOS"]
            oos_book = ens_book[(ens_book.index >= pd.Timestamp(oos_lo)) &
                                (ens_book.index < pd.Timestamp(oos_hi))].dropna()
            if len(oos_book) > 5:
                oos_net = round(float(np.cumprod(1 + oos_book.to_numpy())[-1] - 1) * 100, 1)
                eq = np.cumprod(1 + oos_book.to_numpy()); pk = np.maximum.accumulate(eq)
                oos_maxdd = round(float(((eq - pk) / pk).min() * 100), 1)
                try:
                    dist = block_bootstrap_distribution(oos_book.to_numpy(), n_bootstrap=1000, seed=42)
                    p05_val = round(float(np.percentile(dist, 5)) * 100, 1)
                except Exception as e:
                    print(f"  p05 error: {e}")
        print(f"  2020 OOS: net={oos_net}%  maxDD={oos_maxdd}%  p05={p05_val}")

        # --- All-weather (2020/2021/2022) ---
        print(f"  Building all-weather book ...")
        aw = _get_allweather_book(cad, band_cfgs)
        aw_2020 = aw_2021 = aw_2022 = None
        if aw is not None:
            aw_2020 = _slice_net(aw, "2020-01-01", "2021-01-01")
            aw_2021 = _slice_net(aw, "2021-01-01", "2022-01-01")
            aw_2022 = _slice_net(aw, "2022-01-01", "2023-01-01")
        print(f"  All-weather: 2020={aw_2020}%  2021={aw_2021}%  2022={aw_2022}%")

        # --- Move-catch ---
        if cad == "15m":
            mc = _run_15m_movecatch()
        else:
            mc = _load_movecatch(cad)
        coverage = entry_lag = raw_capture = wtd_capture = None
        if mc:
            coverage = mc.get("coverage")
            entry_lag = mc.get("mean_entry_lag")
            raw_capture = mc.get("raw_capture_mean")
            wtd_capture = mc.get("weighted_capture_mean")
        print(f"  Move-catch (5%): cov={coverage}  lag={entry_lag}  raw={raw_capture}  wtd={wtd_capture}")

        # --- Turnover ---
        rt_yr = _turnover_rt_yr(panels, band_cfgs, cad)
        print(f"  Turnover: ~{rt_yr} RT/yr")

        # --- Per-asset OOS ---
        try:
            pa = _per_asset_oos_nets(panels, band_cfgs)
            n_pos = sum(1 for v in pa.values() if v is not None and v > 0)
            n_tot = len(pa)
        except Exception as e:
            print(f"  Per-asset error: {e}")
            pa = {}; n_pos = n_tot = 0
        print(f"  Per-asset OOS: {n_pos}/{n_tot} positive (pooled={oos_net}%)")
        print(f"  Per-asset nets: {pa}")

        results[cad] = {
            "tf": cad,
            "band_size": band_size,
            "style": STYLE_MAP[cad],
            "oos_net": oos_net,
            "oos_maxdd": oos_maxdd,
            "p05_bootstrap": p05_val,
            "aw_2020": aw_2020,
            "aw_2021": aw_2021,
            "aw_2022": aw_2022,
            "coverage": coverage,
            "entry_lag": entry_lag,
            "raw_capture": raw_capture,
            "weighted_capture": wtd_capture,
            "turnover_rt_yr": rt_yr,
            "per_asset_n_positive_oos": n_pos,
            "per_asset_n_total": n_tot,
            "per_asset_nets": pa,
        }

    # Write
    out_path = OUT / f"vidya_tf_profile_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({"ma_type": MA_TYPE, "timestamp": ts, "per_tf": results}, f, indent=2, default=str)
    print(f"\n[json] {out_path}")

    # Summary
    print(f"\n{'='*105}")
    print(f"VIDYA PROFILE SUMMARY")
    print(f"{'='*105}")
    hdr = (f"{'TF':5} {'Style':12} {'Band':5} {'OOS%':7} {'DD%':7} {'p05':7} "
           f"{'2020%':7} {'2021%':7} {'2022%':7} {'Cov':6} {'Lag':6} {'WtdCap':7} {'RT/yr':6}")
    print(hdr); print("-" * 105)
    for cad in TFS:
        r = results.get(cad, {})
        if not r.get("band_size"):
            print(f"{cad:5}  no band"); continue
        def f(v, fmt="{:.1f}"):
            return fmt.format(v) if v is not None else "  --"
        print(
            f"{cad:5} {r['style']:12} {r['band_size']:5d} "
            f"{f(r.get('oos_net')):>7} {f(r.get('oos_maxdd')):>7} {f(r.get('p05_bootstrap')):>7} "
            f"{f(r.get('aw_2020')):>7} {f(r.get('aw_2021')):>7} {f(r.get('aw_2022')):>7} "
            f"{f(r.get('coverage'), '{:.3f}'):>6} {f(r.get('entry_lag'), '{:.3f}'):>6} "
            f"{f(r.get('weighted_capture'), '{:.3f}'):>7} {f(r.get('turnover_rt_yr')):>6}"
        )
    print("=" * 105)
    return results, out_path


if __name__ == "__main__":
    main()
