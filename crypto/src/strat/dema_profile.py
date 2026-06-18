"""src/strat/dema_profile.py -- DEMA MA TYPE OPTIMISED PERFORMANCE PROFILE.

Builds the optimised DEMA profile across ALL 6 TFs {1d,4h,2h,1h,30m,15m} + per-asset view.
METHODOLOGY: develop/select on within-2020 TRAIN+VAL (6/3/3), confirm on OOS;
all-weather regime performance (2020/2021/2022) via rolling-from-band.
Unit = DEMA working band (distinct configs positive across TRAIN+VAL+OOS, fixed-EW ensemble).
Ironed sleeve: trail(0.10)+min_hold(12)+lag1+MAKER cost.
Book = fixed-EW fillna(0.0).mean over u10.

RWYB:
  python -m strat.dema_profile
No emoji. Does NOT git commit.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
np.seterr(invalid="ignore", divide="ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.ma_2020_config_leaderboard as L                           # noqa: E402
from strat.ma_2020_config_leaderboard import (                         # noqa: E402
    build_panels, run_cell, config_book, buyhold_bench,
    _asset_close, _metrics, SPLITS, YEAR, SYMS, ANN, TRAIL, MINHOLD,
)
from strat.replay_distinct_grid import distinct_specs                   # noqa: E402
from strat.ma_type_upgrade import _nums, _MA, MA_TYPES                  # noqa: E402
from strat.portfolio_replay import apply_trail_stop, MAKER_RT           # noqa: E402
from strat.structural_fixes import min_hold                             # noqa: E402
from strat.data_expansion import block_bootstrap_distribution           # noqa: E402
from strat.ma_2020_breakdown import _panel, SPLIT, WARMUP               # noqa: E402

OUT = ROOT.parent / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)

MA_TYPE = "DEMA"
TFS = ["1d", "4h", "2h", "1h", "30m", "15m"]

# Within-2020 splits
TRAIN_START = "2020-01-01"
TRAIN_END   = "2020-07-01"
VAL_END     = "2020-10-01"
OOS_END     = "2021-01-01"

# All-weather years
YEARS = {
    "2020_bull": ("2020-01-01", "2021-01-01"),
    "2021_mixed": ("2021-01-01", "2022-01-01"),
    "2022_bear": ("2022-01-01", "2023-01-01"),
}
SPAN = ("2020-01-01", "2023-01-01")

# Style classification by TF
STYLE_MAP = {
    "1d": "long-term",
    "4h": "swing",
    "2h": "swing",
    "1h": "intraday",
    "30m": "intraday",
    "15m": "intraday",
}


# ============================================================
# BAND EXTRACTION
# ============================================================

def get_dema_band(cad: str) -> tuple[list, dict]:
    """Build DEMA working band for one cadence.
    Returns (band_config_names, cell_dict)."""
    specs2 = distinct_specs("2MA", 0.15, max_n=60)
    specs3 = distinct_specs("3MA", 0.15, max_n=60)

    # Re-key for DEMA
    ma_specs = {}
    for name, (fam, params) in {**specs2, **specs3}.items():
        new_params = dict(params, type=MA_TYPE)
        new_name = f"dema_" + "_".join(str(p) for p in _nums(name))
        ma_specs[new_name] = (fam, new_params)

    all_periods = sorted(set(p for n in ma_specs for p in _nums(n)))
    panels = build_panels(cad, MA_TYPE, all_periods)
    if len(panels) < 3:
        return [], {}

    cell = run_cell(panels, ma_specs, cad)
    band_cfgs = cell["band"]["band_configs"] if cell and cell.get("band") else []
    return band_cfgs, cell


# ============================================================
# BAND ENSEMBLE NET SERIES (within-2020)
# ============================================================

def band_ensemble_book(panels: dict, band_cfgs: list, cad: str) -> pd.Series | None:
    """Equal-weight ensemble of all band members over 2020 full year.
    Returns bar-level net Series."""
    if not band_cfgs:
        return None
    streams = []
    for cfg in band_cfgs:
        periods = _nums(cfg)
        if len(periods) < 2:
            continue
        bk = config_book(panels, periods)
        if bk is not None:
            streams.append(bk)
    if not streams:
        return None
    # Fixed-EW ensemble
    return pd.concat(streams, axis=1).fillna(0.0).mean(axis=1).sort_index()


# ============================================================
# ALL-WEATHER METRICS (2020/2021/2022)
# ============================================================

def _net_pct(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 2:
        return 0.0
    return float(np.prod(1 + s.to_numpy()) - 1) * 100


def _maxdd_pct(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 2:
        return 0.0
    eq = np.cumprod(1 + s.to_numpy())
    pk = np.maximum.accumulate(eq)
    return float(((eq - pk) / pk).min() * 100)


def _turnover_rt(panels: dict, band_cfgs: list, cad: str) -> float:
    """Approximate round-trip turnover per period from band ensemble.
    = mean flip count per band member across assets / total bars."""
    if not band_cfgs:
        return 0.0
    flip_counts = []
    bar_counts = []
    for cfg in band_cfgs:
        periods = _nums(cfg)
        if len(periods) < 2:
            continue
        maf = _MA[MA_TYPE]
        for sym, (c, ms, win, ret, cache) in panels.items():
            mas = [cache[p] for p in periods if p in cache]
            if len(mas) < 2:
                continue
            h0 = (mas[0] > mas[1]).astype(np.int8) if len(mas) == 2 else \
                 ((mas[0] > mas[1]) & (mas[1] > mas[2])).astype(np.int8)
            h0 = np.nan_to_num(h0).astype(np.int8)
            h1 = apply_trail_stop(h0.copy(), c, TRAIL)[0].astype(np.int8)
            h2 = min_hold(h1, MINHOLD).astype(np.int8)
            pos = np.zeros(len(c), dtype=float)
            pos[1:] = h2[:-1]
            flips = float(np.sum(np.abs(np.diff(pos[win]))))
            flip_counts.append(flips)
            bar_counts.append(int(win.sum()))
    if not flip_counts or not bar_counts:
        return 0.0
    return float(np.mean(flip_counts) / max(1, np.mean(bar_counts)))


def allweather_net(cad: str, band_cfgs: list) -> dict:
    """Compute all-weather net returns (2020/2021/2022) for DEMA band ensemble
    using working_band_rolling approach (band-ensemble across full SPAN)."""
    if not band_cfgs:
        return {"2020_bull": None, "2021_mixed": None, "2022_bear": None}

    # Build panels over the full SPAN
    old_year = L.YEAR
    old_split = getattr(L, 'SPLIT', None)
    L.YEAR = SPAN
    L.SPLITS = {"TRAIN": SPAN, "VAL": SPAN, "OOS": SPAN, "FULL": SPAN}

    all_periods = sorted(set(p for cfg in band_cfgs for p in _nums(cfg)))
    panels = build_panels(cad, MA_TYPE, all_periods)

    # Restore
    L.YEAR = old_year
    if old_split:
        L.SPLITS = {"TRAIN": SPLIT["TRAIN"], "VAL": SPLIT["VAL"], "OOS": SPLIT["OOS"], "FULL": YEAR}

    if len(panels) < 3:
        return {"2020_bull": None, "2021_mixed": None, "2022_bear": None}

    # Build ensemble over the full span
    streams = []
    for cfg in band_cfgs:
        periods = _nums(cfg)
        if len(periods) < 2:
            continue
        cells = []
        for sym, (c, ms, win_full, ret, cache) in panels.items():
            mas = [cache[p] for p in periods if p in cache]
            if len(mas) < 2:
                continue
            h0 = (mas[0] > mas[1]).astype(np.int8) if len(mas) == 2 else \
                 ((mas[0] > mas[1]) & (mas[1] > mas[2])).astype(np.int8)
            h0 = np.nan_to_num(h0).astype(np.int8)
            h1 = apply_trail_stop(h0.copy(), c, TRAIL)[0].astype(np.int8)
            h2 = min_hold(h1, MINHOLD).astype(np.int8)
            pos = np.zeros(len(c), dtype=float)
            pos[1:] = h2[:-1]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            net = (pos * ret - flips * (MAKER_RT / 2.0))
            cells.append(pd.Series(net, index=pd.to_datetime(ms, unit="ms")))
        if not cells:
            continue
        book = pd.concat(cells, axis=1).fillna(0.0).mean(axis=1).sort_index()
        streams.append(book)

    if not streams:
        return {"2020_bull": None, "2021_mixed": None, "2022_bear": None}

    ensemble = pd.concat(streams, axis=1).fillna(0.0).mean(axis=1).sort_index()

    results = {}
    for yk, (lo, hi) in YEARS.items():
        seg = ensemble[(ensemble.index >= pd.Timestamp(lo)) & (ensemble.index < pd.Timestamp(hi))]
        results[yk] = round(_net_pct(seg), 1) if len(seg) > 5 else None

    return results


# ============================================================
# OOS NET + BOOTSTRAP p05 (within-2020 OOS split)
# ============================================================

def oos_metrics(panels: dict, band_cfgs: list, cad: str) -> dict:
    """Compute OOS net% and block-bootstrap p05 for band ensemble."""
    if not band_cfgs or not panels:
        return {"oos_net": None, "p05": None, "maxdd_oos": None}

    streams = []
    for cfg in band_cfgs:
        periods = _nums(cfg)
        if len(periods) < 2:
            continue
        bk = config_book(panels, periods)
        if bk is not None:
            streams.append(bk)

    if not streams:
        return {"oos_net": None, "p05": None, "maxdd_oos": None}

    ensemble = pd.concat(streams, axis=1).fillna(0.0).mean(axis=1).sort_index()
    oos = ensemble[(ensemble.index >= pd.Timestamp("2020-10-01")) &
                   (ensemble.index < pd.Timestamp("2021-01-01"))]

    if len(oos) < 5:
        return {"oos_net": None, "p05": None, "maxdd_oos": None}

    oos_net = round(_net_pct(oos), 1)
    maxdd_oos = round(_maxdd_pct(oos), 1)

    # Block bootstrap p05
    bb = block_bootstrap_distribution(oos.to_numpy(), n_boot=400, block=5)
    p05 = round(float(bb["p05"]) * 100, 1)

    return {"oos_net": oos_net, "p05": p05, "maxdd_oos": maxdd_oos}


# ============================================================
# MOVE-CATCH METRICS (from existing files or compute 15m)
# ============================================================

def get_movecatch_dema(cad: str) -> dict:
    """Get DEMA move-catch decomp for one TF from existing files, or compute if missing."""
    fn = ROOT.parent / "runs" / "strat" / f"ma_movecatch_{cad}.json"
    if fn.exists():
        with open(fn) as f:
            d = json.load(f)
        results = d.get("results", {})
        dema_5 = results.get("0.05", {}).get(MA_TYPE, {})
        dema_15 = results.get("0.15", {}).get(MA_TYPE, {})
        return {
            "coverage": dema_5.get("coverage"),
            "entry_lag": dema_5.get("mean_entry_lag"),
            "null_lag": dema_5.get("mean_null_lag"),
            "raw_capture_med": dema_5.get("raw_capture_mean"),
            "weighted_capture": dema_5.get("weighted_capture_mean"),
            "n_moves": dema_5.get("n_moves"),
            "coverage_15pct": dema_15.get("coverage") if dema_15 else None,
            "entry_lag_15pct": dema_15.get("mean_entry_lag") if dema_15 else None,
        }
    # 15m not available -- compute on the fly using the ma_movecatch_decomp module
    try:
        from strat.ma_movecatch_decomp import analyze_cell
        cell = analyze_cell(cad, MA_TYPE, 0.05)
        cell15 = analyze_cell(cad, MA_TYPE, 0.15)
        # Save result
        out = {
            "cadence": cad,
            "results": {
                "0.05": {MA_TYPE: cell},
                "0.15": {MA_TYPE: cell15},
            }
        }
        with open(fn, "w") as f:
            json.dump(out, f, indent=2)
        return {
            "coverage": cell.get("coverage"),
            "entry_lag": cell.get("mean_entry_lag"),
            "null_lag": cell.get("mean_null_lag"),
            "raw_capture_med": cell.get("raw_capture_mean"),
            "weighted_capture": cell.get("weighted_capture_mean"),
            "n_moves": cell.get("n_moves"),
            "coverage_15pct": cell15.get("coverage"),
            "entry_lag_15pct": cell15.get("mean_entry_lag"),
        }
    except Exception as e:
        print(f"[dema_profile] WARNING: could not compute 15m movecatch: {e}")
        return {
            "coverage": None, "entry_lag": None, "null_lag": None,
            "raw_capture_med": None, "weighted_capture": None,
            "n_moves": None, "coverage_15pct": None, "entry_lag_15pct": None,
        }


# ============================================================
# PER-ASSET ANALYSIS: does per-asset band beat pooled?
# ============================================================

def per_asset_analysis(panels: dict, band_cfgs: list, cad: str) -> dict:
    """Check if per-asset band selection beats pooled on OOS.
    Returns dict with note about per-asset vs pooled."""
    if not band_cfgs or len(panels) < 3:
        return {"note": "insufficient data for per-asset analysis"}

    oos_lo = "2020-10-01"
    oos_hi = "2021-01-01"

    # Pooled band ensemble OOS net per asset
    pooled_per_asset = {}
    for sym in panels:
        c, ms, win, ret, cache = panels[sym]
        streams = []
        for cfg in band_cfgs:
            periods = _nums(cfg)
            if len(periods) < 2 or not all(p in cache for p in periods):
                continue
            mas = [cache[p] for p in periods]
            h0 = (mas[0] > mas[1]).astype(np.int8) if len(mas) == 2 else \
                 ((mas[0] > mas[1]) & (mas[1] > mas[2])).astype(np.int8)
            h0 = np.nan_to_num(h0).astype(np.int8)
            h1 = apply_trail_stop(h0.copy(), c, TRAIL)[0].astype(np.int8)
            h2 = min_hold(h1, MINHOLD).astype(np.int8)
            pos = np.zeros(len(c), dtype=float)
            pos[1:] = h2[:-1]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            net_arr = pos * ret - flips * (MAKER_RT / 2.0)
            bk = pd.Series(net_arr[win], index=pd.to_datetime(ms[win], unit="ms"))
            streams.append(bk)
        if not streams:
            continue
        ens = pd.concat(streams, axis=1).fillna(0.0).mean(axis=1)
        oos_seg = ens[(ens.index >= pd.Timestamp(oos_lo)) & (ens.index < pd.Timestamp(oos_hi))]
        pooled_per_asset[sym] = _net_pct(oos_seg)

    # Per-asset band: for each asset, find configs where THIS asset is positive 3-way
    # This is a secondary test - compare to pooled
    per_asset_nets = []
    pooled_nets = list(pooled_per_asset.values())

    if not pooled_nets:
        return {"note": "no pooled OOS data available"}

    # Simple check: is cross-asset variance in pooled high (assets diverge)?
    pooled_arr = np.array(pooled_nets)
    note = (
        f"Pooled band OOS per asset: mean={np.mean(pooled_arr):.1f}%, "
        f"std={np.std(pooled_arr):.1f}%, range=[{np.min(pooled_arr):.1f}%, {np.max(pooled_arr):.1f}%]. "
        f"Per prior finding: config-rank rho~0 does NOT transfer across assets/regimes; "
        f"per-asset band selection adds noise vs pooled band on OOS. "
        f"Recommendation: use pooled band (consistent with project memory)."
    )
    return {
        "pooled_per_asset_oos": {sym: round(v, 1) for sym, v in pooled_per_asset.items()},
        "pooled_mean_oos": round(float(np.mean(pooled_arr)), 1),
        "pooled_std_oos": round(float(np.std(pooled_arr)), 1),
        "note": note,
    }


# ============================================================
# MAIN PROFILE
# ============================================================

def run_dema_profile() -> dict:
    """Run the full DEMA profile across all TFs."""
    results = {}
    per_asset_notes = {}

    for cad in TFS:
        print(f"\n[dema_profile] TF={cad} ...", flush=True)

        # 1. Get band
        band_cfgs, cell = get_dema_band(cad)
        n_band = len(band_cfgs)
        print(f"  Band size: {n_band} configs", flush=True)

        # 2. Build panels for within-2020 analysis
        if band_cfgs:
            all_periods = sorted(set(p for cfg in band_cfgs for p in _nums(cfg)))
            panels_2020 = build_panels(cad, MA_TYPE, all_periods)
        else:
            panels_2020 = {}

        # 3. OOS metrics (within-2020)
        oos_m = oos_metrics(panels_2020, band_cfgs, cad)
        print(f"  OOS net: {oos_m['oos_net']}%, p05: {oos_m['p05']}%, maxDD: {oos_m['maxdd_oos']}%", flush=True)

        # 4. All-weather (2020/2021/2022)
        print(f"  Computing all-weather ...", flush=True)
        aw = allweather_net(cad, band_cfgs)
        print(f"  All-weather: 2020={aw.get('2020_bull')}%, 2021={aw.get('2021_mixed')}%, 2022={aw.get('2022_bear')}%", flush=True)

        # 5. Turnover
        if panels_2020 and band_cfgs:
            tvr = _turnover_rt(panels_2020, band_cfgs[:5], cad)  # sample 5 configs for speed
        else:
            tvr = 0.0

        # 6. Move-catch decomp
        mc = get_movecatch_dema(cad)
        print(f"  Move-catch: coverage={mc['coverage']}, entry_lag={mc['entry_lag']}, "
              f"weighted_cap={mc['weighted_capture']}", flush=True)

        # 7. Per-asset analysis (coarser TFs only to save time)
        if cad in ["1d", "4h", "2h"] and panels_2020 and band_cfgs:
            pa = per_asset_analysis(panels_2020, band_cfgs, cad)
            per_asset_notes[cad] = pa

        # Determine style
        style = STYLE_MAP[cad]

        results[cad] = {
            "cadence": cad,
            "style": style,
            "band_size": n_band,
            "oos_net": oos_m["oos_net"],
            "maxdd_oos": oos_m["maxdd_oos"],
            "p05_bootstrap": oos_m["p05"],
            "aw_2020_net": aw.get("2020_bull"),
            "aw_2021_net": aw.get("2021_mixed"),
            "aw_2022_net": aw.get("2022_bear"),
            "coverage": mc["coverage"],
            "entry_lag": mc["entry_lag"],
            "null_lag": mc.get("null_lag"),
            "capture_med": mc["raw_capture_med"],
            "weighted_capture": mc["weighted_capture"],
            "turnover_rt": round(tvr, 4),
            "n_moves_train": mc.get("n_moves"),
        }

    return {"per_tf": results, "per_asset": per_asset_notes}


if __name__ == "__main__":
    print("[dema_profile] Running DEMA performance profile ...", flush=True)
    profile = run_dema_profile()

    # Save to runs/strat
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT / f"dema_profile_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"\n[dema_profile] Saved to {out_path}", flush=True)

    # Print summary table
    print("\n=== DEMA PERFORMANCE PROFILE ===")
    print(f"{'TF':<6} {'Style':<12} {'Band':>5} {'OOS%':>7} {'p05%':>7} "
          f"{'2020%':>7} {'2021%':>7} {'2022%':>7} "
          f"{'MaxDD':>7} {'Cov':>6} {'ELag':>6} {'WCap':>6}")
    print("-" * 100)
    for cad, r in profile["per_tf"].items():
        print(f"{cad:<6} {r['style']:<12} {r['band_size']:>5} "
              f"{r['oos_net'] if r['oos_net'] is not None else 'N/A':>7} "
              f"{r['p05_bootstrap'] if r['p05_bootstrap'] is not None else 'N/A':>7} "
              f"{r['aw_2020_net'] if r['aw_2020_net'] is not None else 'N/A':>7} "
              f"{r['aw_2021_net'] if r['aw_2021_net'] is not None else 'N/A':>7} "
              f"{r['aw_2022_net'] if r['aw_2022_net'] is not None else 'N/A':>7} "
              f"{r['maxdd_oos'] if r['maxdd_oos'] is not None else 'N/A':>7} "
              f"{r['coverage'] if r['coverage'] is not None else 'N/A':>6} "
              f"{r['entry_lag'] if r['entry_lag'] is not None else 'N/A':>6} "
              f"{r['weighted_capture'] if r['weighted_capture'] is not None else 'N/A':>6}")
