"""src/strat/sma_profile_optimizer.py -- SMA-ONLY optimised profile across all 6 TFs.

GOAL: build the OPTIMISED SMA profile across {1d,4h,2h,1h,30m,15m}:
  1. SMA working band per TF (configs positive TRAIN+VAL+OOS within-2020, fixed-EW ironed sleeve)
  2. OOS net (within-2020, Oct-Dec)
  3. All-weather regime performance (2020 bull / 2021 mixed / 2022 bear) via rolling-from-band
  4. Move-catch profile per TF: capture, entry-lag, coverage (from existing JSON files; 15m run here)
  5. Block-bootstrap p05 on OOS returns
  6. Turnover (RT per period) + band_size

METHODOLOGY (6/3/3):
  TRAIN: 2020-01-01..2020-07-01  (+ 400-bar warmup)
  VAL:   2020-07-01..2020-10-01
  OOS:   2020-10-01..2021-01-01  <- confirm split (not touched until now)
  UNSEEN: 2025+ -- NOT TOUCHED.

UNIT = SMA WORKING BAND only. NO other MA type. No pooling.

RWYB:
  python -m strat.sma_profile_optimizer
"""
from __future__ import annotations

import json
import sys
import datetime as dt
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
np.seterr(invalid="ignore", divide="ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strat.ma_2020_config_leaderboard import (
    build_panels, config_book, run_cell, _metrics, _asset_close,
    SPLITS, SYMS, TRAIL, MINHOLD, MAKER_RT, ANN,
)
from strat.replay_distinct_grid import distinct_specs
from strat.ma_type_upgrade import _nums, _MA
from strat.data_expansion import block_bootstrap_distribution
from strat.ma_movecatch_decomp import analyze_cell, run_tf as movecatch_run_tf
import strat.ma_2020_config_leaderboard as L

# ---- All-weather imports (for the rolling-from-band regime test)
import strat.portfolio_replay as PR

OUT_STRAT = ROOT.parent / "runs" / "strat"
OUT_AW = ROOT.parent / "runs" / "periods" / "ALL_WEATHER"
OUT_STRAT.mkdir(parents=True, exist_ok=True)
OUT_AW.mkdir(parents=True, exist_ok=True)

TFS = ["1d", "4h", "2h", "1h", "30m", "15m"]
MA_TYPE = "SMA"

# 2020 6/3/3 splits
YEAR = ("2020-01-01", "2021-01-01")
SPLIT_TRAIN = ("2020-01-01", "2020-07-01")
SPLIT_VAL   = ("2020-07-01", "2020-10-01")
SPLIT_OOS   = ("2020-10-01", "2021-01-01")

# All-weather span
AW_SPAN = ("2020-01-01", "2023-01-01")
AW_YEARS = {
    "2020_bull": ("2020-01-01", "2021-01-01"),
    "2021_mixed": ("2021-01-01", "2022-01-01"),
    "2022_bear": ("2022-01-01", "2023-01-01"),
}

# Rolling-from-band parameters
LOOKBACK_D = 120
STEP_D = 30


# ===========================================================================
# STEP 1: SMA working band per TF (from config_leaderboard.json if available,
#         else compute fresh)
# ===========================================================================

def _build_sma_band(tf: str) -> dict:
    """Returns {'band_configs': [...], 'n_band': int, 'oos_net': float, 'oos_maxdd': float,
                'train_net': float, 'val_net': float, 'full_net': float, 'band_fast_range': [a,b],
                'band_slow_range': [a,b], 'turnover_rt': float}
    Builds the SMA band for this TF and computes the band-ensemble OOS metrics.
    """
    # Try loading from existing config_leaderboard.json
    lb_path = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE" / "config_leaderboard.json"
    if lb_path.exists():
        try:
            lb = json.loads(lb_path.read_text(encoding="utf-8"))
            cell_key = None
            # Find the SMA cell for this TF
            for entry in lb.get("grid", []):
                if isinstance(entry, dict) and entry.get("ma_type") == "SMA" and entry.get("cadence") == tf:
                    cell_key = entry
                    break
            # Alternatively the grid might be a dict
            grid = lb.get("grid", {})
            if isinstance(grid, dict):
                cell_key = grid.get((MA_TYPE, tf)) or grid.get(f"{MA_TYPE}_{tf}")
        except Exception:
            cell_key = None
    else:
        cell_key = None

    # Always compute fresh to ensure correctness (leaderboard JSON structure varies)
    print(f"  [{tf}] Building SMA band from scratch (fresh)...")
    specs2 = distinct_specs("2MA", 0.15, max_n=60)
    specs3 = distinct_specs("3MA", 0.15, max_n=60)
    all_specs = {**specs2, **specs3}

    # Re-key for SMA
    sma_specs = {}
    for name, (fam, params) in all_specs.items():
        new_params = dict(params, type=MA_TYPE)
        new_name = f"sma_" + "_".join(str(p) for p in _nums(name))
        sma_specs[new_name] = (fam, new_params)

    all_periods = sorted({p for n in sma_specs for p in _nums(n)})
    panels = build_panels(tf, MA_TYPE, all_periods)
    if not panels:
        return {"band_configs": [], "n_band": 0, "oos_net": None, "oos_maxdd": None,
                "train_net": None, "val_net": None, "full_net": None,
                "band_fast_range": None, "band_slow_range": None, "turnover_rt": None}

    cell = run_cell(panels, sma_specs, tf)
    band_cfgs = cell["band"]["band_configs"]
    n_band = len(band_cfgs)
    bs = cell["band"]

    if not band_cfgs:
        return {"band_configs": [], "n_band": 0, "oos_net": None, "oos_maxdd": None,
                "train_net": None, "val_net": None, "full_net": None,
                "band_fast_range": None, "band_slow_range": None, "turnover_rt": None}

    # Band-ensemble OOS metrics: mean of band member net streams
    book = _ensemble_book(panels, band_cfgs)
    if book is None:
        return {"band_configs": band_cfgs, "n_band": n_band, "oos_net": None, "oos_maxdd": None,
                "train_net": None, "val_net": None, "full_net": None,
                "band_fast_range": None, "band_slow_range": None, "turnover_rt": None}

    m_train = _metrics(book, tf, *SPLIT_TRAIN)
    m_val   = _metrics(book, tf, *SPLIT_VAL)
    m_oos   = _metrics(book, tf, *SPLIT_OOS)
    m_full  = _metrics(book, tf, *YEAR)

    # Turnover estimate: count flips on the ensemble position
    turnover_rt = _turnover_rt(panels, band_cfgs, tf)

    result = {
        "band_configs": band_cfgs,
        "n_band": n_band,
        "oos_net": m_oos["net"],
        "oos_maxdd": m_oos["maxdd"],
        "train_net": m_train["net"],
        "val_net": m_val["net"],
        "full_net": m_full["net"],
        "band_fast_range": bs.get("band_2ma_fast_range"),
        "band_slow_range": bs.get("band_2ma_slow_range"),
        "turnover_rt": turnover_rt,
        "_book": book,  # keep for bootstrap
    }
    print(f"  [{tf}] SMA band: {n_band} configs | OOS net={m_oos['net']}% | "
          f"TRAIN={m_train['net']}% | VAL={m_val['net']}%")
    return result


def _ensemble_book(panels, band_cfgs):
    """EW band-ensemble bar-level net series (fixed-EW, cadence-invariant)."""
    books = []
    for cfg in band_cfgs:
        periods = _nums(cfg)
        if len(periods) < 2:
            continue
        b = config_book(panels, periods)
        if b is not None and len(b) > 10:
            books.append(b)
    if not books:
        return None
    return pd.concat(books, axis=1).fillna(0.0).mean(axis=1).sort_index()


def _turnover_rt(panels, band_cfgs, tf) -> float:
    """Estimate RT per period (flips per bar averaged over band members x assets)."""
    from strat.portfolio_replay import apply_trail_stop
    from strat.structural_fixes import min_hold as min_hold_fn
    from strat.ma_type_upgrade import _MA

    flip_counts = []
    bar_counts = []
    maf = _MA[MA_TYPE]

    for cfg in band_cfgs[:20]:  # cap at 20 for speed
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

    if not bar_counts or sum(bar_counts) == 0:
        return None
    total_flips = sum(flip_counts)
    total_bars = sum(bar_counts)
    # RT per period: one period = one bar; round-trip cost = MAKER_RT per flip
    return round(total_flips / total_bars, 4)


# ===========================================================================
# STEP 2: Block-bootstrap p05 on OOS band-ensemble returns
# ===========================================================================

def _bootstrap_p05(book, tf) -> float | None:
    """Block-bootstrap the OOS returns: p05 of compound net."""
    oos = book[
        (book.index >= pd.Timestamp(SPLIT_OOS[0])) &
        (book.index <  pd.Timestamp(SPLIT_OOS[1]))
    ].dropna()
    if len(oos) < 10:
        return None
    dist = block_bootstrap_distribution(oos.to_numpy(), n_boot=400, block=5, seed=42)
    return round(float(dist["p05"]) * 100, 1)


# ===========================================================================
# STEP 3: All-weather rolling-from-band (2020 bull / 2021 mixed / 2022 bear)
# ===========================================================================

def _2ma_series_aw(tf):
    """All-weather daily-compounded net per (fast,slow) SMA config over 2020-2023."""
    # Temporarily extend the window
    orig_year = L.YEAR
    orig_split = L.SPLIT
    orig_splits = L.SPLITS

    L.YEAR = AW_SPAN
    L.SPLIT = {"TRAIN": AW_SPAN, "VAL": AW_SPAN, "OOS": AW_SPAN}
    L.SPLITS = {"TRAIN": AW_SPAN, "VAL": AW_SPAN, "OOS": AW_SPAN, "FULL": AW_SPAN}

    specs2 = distinct_specs("2MA", 0.15, max_n=60)
    all_periods = sorted({p for n in specs2 for p in _nums(n)})
    panels = build_panels(tf, MA_TYPE, all_periods)

    cols = {}
    for name in specs2:
        periods = _nums(name)
        if len(periods) != 2:
            continue
        book = config_book(panels, periods)
        if book is None or len(book) < 50:
            continue
        daily = book.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()
        cols[f"{periods[0]},{periods[1]}"] = daily

    # Restore
    L.YEAR = orig_year
    L.SPLIT = orig_split
    L.SPLITS = orig_splits

    if not cols:
        return None
    return pd.DataFrame(cols).sort_index()


def _net_s(s):
    s = s.dropna()
    return float(np.prod(1 + s.to_numpy()) - 1) * 100 if len(s) > 1 else 0.0


def _maxdd_s(s):
    s = s.dropna()
    if len(s) < 2:
        return 0.0
    eq = np.cumprod(1 + s.to_numpy()); pk = np.maximum.accumulate(eq)
    return float(((eq - pk) / pk).min() * 100)


def _per_year_aw(daily):
    out = {}
    for yk, (lo, hi) in AW_YEARS.items():
        s = daily[(daily.index >= pd.Timestamp(lo)) & (daily.index < pd.Timestamp(hi))]
        out[yk] = {"net": round(_net_s(s), 1), "maxdd": round(_maxdd_s(s), 1)}
    return out


def _rolling_band_aw(series_df):
    """Walk-forward rolling-band ensemble (no look-ahead) over AW_SPAN."""
    idx = series_df.index
    start = idx.min() + pd.Timedelta(days=LOOKBACK_D)
    pieces = []
    t = start
    cfgs = list(series_df.columns)
    while t < idx.max():
        nxt = t + pd.Timedelta(days=STEP_D)
        look = series_df[(idx >= t - pd.Timedelta(days=LOOKBACK_D)) & (idx < t)]
        fwd  = series_df[(idx >= t) & (idx < nxt)]
        if len(look) < 20 or len(fwd) < 2:
            t = nxt; continue
        look_net = (np.prod(1 + look.fillna(0.0).to_numpy(), axis=0) - 1) * 100
        band_mask = look_net > 0
        if not band_mask.any():
            band_mask = look_net == look_net.max()
        band_cfgs = [c for c, m in zip(cfgs, band_mask) if m]
        seg = fwd[band_cfgs].mean(axis=1).dropna()
        if len(seg):
            pieces.append(seg)
        t = nxt
    if not pieces:
        return None
    return pd.concat(pieces).sort_index()


def _aw_regime_profile(tf: str, band_result: dict) -> dict:
    """Compute all-weather regime performance for SMA @ tf."""
    print(f"  [{tf}] All-weather rolling-band (SMA)...")
    sdf = _2ma_series_aw(tf)
    if sdf is None:
        return {"note": "no series"}

    rolling = _rolling_band_aw(sdf)
    if rolling is None:
        return {"note": "no rolling result"}

    per_year = _per_year_aw(rolling)
    return {
        "aw_2020_net": per_year["2020_bull"]["net"],
        "aw_2020_maxdd": per_year["2020_bull"]["maxdd"],
        "aw_2021_net": per_year["2021_mixed"]["net"],
        "aw_2021_maxdd": per_year["2021_mixed"]["maxdd"],
        "aw_2022_net": per_year["2022_bear"]["net"],
        "aw_2022_maxdd": per_year["2022_bear"]["maxdd"],
    }


# ===========================================================================
# STEP 4: Move-catch profile per TF (from existing JSON files)
# ===========================================================================

MOVECATCH_FILES = {
    "1d": ROOT.parent / "runs" / "strat" / "ma_movecatch_1d.json",
    "4h": ROOT.parent / "runs" / "strat" / "ma_movecatch_4h.json",
    "2h": ROOT.parent / "runs" / "strat" / "ma_movecatch_2h.json",
    "1h": ROOT.parent / "runs" / "strat" / "ma_movecatch_1h.json",
    "30m": ROOT.parent / "runs" / "strat" / "ma_movecatch_30m.json",
}


def _load_movecatch_sma(tf: str) -> dict | None:
    """Load SMA move-catch stats from existing JSON (5% threshold, primary)."""
    p = MOVECATCH_FILES.get(tf)
    if p and p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return d.get("results", {}).get("0.05", {}).get("SMA")
        except Exception:
            return None
    return None


def _run_movecatch_15m() -> dict | None:
    """Run 15m SMA move-catch (5% threshold), reusing ma_movecatch_decomp."""
    out_path = OUT_STRAT / "ma_movecatch_15m.json"
    if out_path.exists():
        try:
            d = json.loads(out_path.read_text(encoding="utf-8"))
            sma_data = d.get("results", {}).get("0.05", {}).get("SMA")
            if sma_data is not None:
                print("  [15m] Loaded 15m SMA movecatch from existing file.")
                return sma_data
        except Exception:
            pass

    print("  [15m] Running 15m move-catch decomp (SMA only)...")
    # Only run SMA (we pass the full run_tf which handles all types, but we'll extract SMA)
    try:
        movecatch_run_tf("15m", thresholds=[0.05])
        d = json.loads(out_path.read_text(encoding="utf-8"))
        return d.get("results", {}).get("0.05", {}).get("SMA")
    except Exception as e:
        print(f"  [15m] Error running movecatch: {e}")
        # Fallback: compute directly
        try:
            cell = analyze_cell("15m", MA_TYPE, 0.05)
            return cell
        except Exception as e2:
            print(f"  [15m] Fallback also failed: {e2}")
            return None


# ===========================================================================
# STEP 5: STYLE TAG per TF
# ===========================================================================

STYLE_MAP = {
    "1d": "long-term",
    "4h": "swing",
    "2h": "swing",
    "1h": "swing-intraday",
    "30m": "intraday",
    "15m": "intraday",
}


# ===========================================================================
# MAIN: assemble the full SMA profile per TF
# ===========================================================================

def run_sma_profile():
    print(f"\n{'='*60}")
    print(f"SMA PROFILE OPTIMIZER  -- {MA_TYPE} only, all 6 TFs")
    print(f"{'='*60}")

    per_tf_results = {}

    for tf in TFS:
        print(f"\n--- TF: {tf} ---")

        # Step 1: SMA working band
        band_result = _build_sma_band(tf)
        book = band_result.pop("_book", None)

        # Step 2: Block-bootstrap p05 (OOS)
        p05 = None
        if book is not None:
            p05 = _bootstrap_p05(book, tf)
            print(f"  [{tf}] Bootstrap p05 OOS = {p05}%")

        # Step 3: All-weather regime
        aw = _aw_regime_profile(tf, band_result)

        # Step 4: Move-catch
        if tf == "15m":
            mc = _run_movecatch_15m()
        else:
            mc = _load_movecatch_sma(tf)

        if mc:
            coverage      = mc.get("coverage")
            entry_lag     = mc.get("mean_entry_lag")
            cap_med       = mc.get("raw_capture_mean")
            w_cap         = mc.get("weighted_capture_mean")
        else:
            coverage = entry_lag = cap_med = w_cap = None

        per_tf_results[tf] = {
            "tf": tf,
            "style": STYLE_MAP[tf],
            "band_size": band_result["n_band"],
            "band_fast_range": band_result.get("band_fast_range"),
            "band_slow_range": band_result.get("band_slow_range"),
            "train_net": band_result.get("train_net"),
            "val_net": band_result.get("val_net"),
            "oos_net": band_result.get("oos_net"),
            "oos_maxdd": band_result.get("oos_maxdd"),
            "p05_bootstrap": p05,
            "aw_2020_net": aw.get("aw_2020_net"),
            "aw_2020_maxdd": aw.get("aw_2020_maxdd"),
            "aw_2021_net": aw.get("aw_2021_net"),
            "aw_2021_maxdd": aw.get("aw_2021_maxdd"),
            "aw_2022_net": aw.get("aw_2022_net"),
            "aw_2022_maxdd": aw.get("aw_2022_maxdd"),
            "coverage": coverage,
            "entry_lag": entry_lag,
            "capture_med": cap_med,
            "weighted_capture": w_cap,
            "turnover_rt": band_result.get("turnover_rt"),
        }

        print(f"  [{tf}] SUMMARY: band={band_result['n_band']} | "
              f"OOS={band_result.get('oos_net')}% | p05={p05}% | "
              f"AW: 2020={aw.get('aw_2020_net')}% 2021={aw.get('aw_2021_net')}% "
              f"2022={aw.get('aw_2022_net')}% | "
              f"coverage={coverage} lag={entry_lag} cap={cap_med}")

    # Persist to runs/strat
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_STRAT / f"sma_profile_{stamp}.json"
    out_data = {
        "ma_type": MA_TYPE,
        "generated": stamp,
        "methodology": {
            "band": "TRAIN+VAL+OOS positive (6/3/3 within-2020)",
            "sleeve": "MA-cross -> trail(0.10) -> min_hold(12) -> lag1 -> maker(0.0006 RT)",
            "book": "fixed-EW fillna(0.0).mean over u10",
            "aw": "rolling-from-band ensemble (120d lookback, 30d step) over 2020-2023",
            "movecatch": "TRAIN window (2020-01..07), 5% threshold, causal band-ensemble",
        },
        "per_tf": per_tf_results,
    }
    out_path.write_text(json.dumps(out_data, indent=2, default=str), encoding="utf-8")
    print(f"\n[persisted] {out_path}")
    return out_data


if __name__ == "__main__":
    result = run_sma_profile()

    # Print final summary table
    print("\n\n=== SMA OPTIMISED PROFILE SUMMARY ===")
    print(f"{'TF':>4} | {'Style':>18} | {'Band':>5} | {'OOS%':>7} | {'p05%':>6} | "
          f"{'AW20%':>7} | {'AW21%':>7} | {'AW22%':>7} | {'MaxDD%':>7} | "
          f"{'Cov':>5} | {'Lag':>5} | {'Cap':>5} | {'Trn':>6}")
    print("-" * 120)
    for tf in TFS:
        r = result["per_tf"][tf]
        print(f"{tf:>4} | {r['style']:>18} | {r['band_size']:>5} | "
              f"{str(r['oos_net'] or '--'):>7} | {str(r['p05_bootstrap'] or '--'):>6} | "
              f"{str(r['aw_2020_net'] or '--'):>7} | {str(r['aw_2021_net'] or '--'):>7} | "
              f"{str(r['aw_2022_net'] or '--'):>7} | {str(r['aw_2022_maxdd'] or '--'):>7} | "
              f"{str(round(r['coverage'], 3) if r['coverage'] else '--'):>5} | "
              f"{str(round(r['entry_lag'], 3) if r['entry_lag'] else '--'):>5} | "
              f"{str(round(r['capture_med'], 3) if r['capture_med'] else '--'):>5} | "
              f"{str(r['turnover_rt'] or '--'):>6}")
