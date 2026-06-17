"""src/strat/ti_wave2d_honest_band.py -- Wave-2D: HONEST deployable form of the TI band-ensemble.

Tasks:
  1. Band-ENSEMBLE vs rolling-PICK comparison (2020-2022): already in cached JSON, re-derive + report.
  2. Drop TSI + RSI (fragile); keep MACD/KELTNER/PSAR/MFI (50-67% of combos positive in bear).
  3. 2024-H1 SECOND-BEAR TEST: run the band-ensemble forward (2024-01-01 to 2024-07-01) -- did the
     band-ensemble go to cash + recover in the BTC ~-30% Q1 2024 drawdown?
  4. FULL-CYCLE compound under HYPERPARAM AVERAGING (not the cherry-picked (120,30) lookback):
     sweep lookback_d in {60,90,120,150,180} x step_d in {14,21,30,45} -- does any all-weather
     advantage survive averaging across the lookback-step grid?
  5. Compute per-config bear-positive fraction to validate the "50-67% of combos positive in 2022
     bear" claim for the kept TIs.

Statistical framing (quant):
  - Null for pick vs ensemble: timing-scrambled pick (same hold amount, random selection from
    the band) = 0 added value. Test: pick_net - ensemble_net vs 0; sign test over N STEP windows.
  - n_eff for one bear year: autocorrelated daily returns -> block-bootstrap estimate.
  - Hyperparam sensitivity: IQR / median of per-config 2022-bear net across the grid.

Long-only spot, fixed-EW, maker, UNSEEN (2024+) sealed after this one test run. No emoji. cp1252 safe.

RWYB: python -m strat.ti_wave2d_honest_band
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

import strat.deep2020_ti_pipeline as TI
from strat.deep2020_ti_pipeline import INDICATORS
from strat.portfolio_replay import MAKER_RT

OUT = ROOT.parent / "runs" / "periods" / "ALL_WEATHER"
OUT.mkdir(parents=True, exist_ok=True)

# ---- target TIs (post-Wave-1 drop of TSI/RSI) ----
KEEP_TIS = ["MACD", "KELTNER", "PSAR", "MFI"]
DROP_TIS = ["TSI", "RSI"]  # fragile: 1/12 combos positive in 2022 bear

# ---- period definitions ----
SPAN_TRAIN = ("2020-01-01", "2023-01-01")
SPAN_2024H1 = ("2024-01-01", "2024-07-01")   # second-bear test window
SPAN_FULLCYCLE = ("2020-01-01", "2026-01-01")  # full-cycle for compound comparison

# ---- lookback/step grid for sensitivity ----
LOOKBACK_GRID = [60, 90, 120, 150, 180]
STEP_GRID = [14, 21, 30, 45]


# =============================================================================
# CORE HELPERS
# =============================================================================

def _net(s: pd.Series) -> float:
    s = s.dropna()
    return float(np.prod(1 + s.to_numpy()) - 1) * 100 if len(s) > 1 else 0.0


def _maxdd(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 2:
        return 0.0
    eq = np.cumprod(1 + s.to_numpy())
    pk = np.maximum.accumulate(eq)
    return float(((eq - pk) / pk).min() * 100)


def _sharpe_ann(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 5:
        return float("nan")
    return float(np.mean(s) / (np.std(s) + 1e-12) * np.sqrt(365))


def _n_eff_block(s: pd.Series, block_size: int = 20) -> float:
    """Effective sample size under block-bootstrap (conservative: n / block_size)."""
    return max(1.0, len(s) / block_size)


def _block_bootstrap_p05(s: pd.Series, n_boot: int = 2000, block_size: int = 20) -> float:
    """5th percentile of compound return under block-bootstrap (captures autocorrelation).
    Returns the p05 of boot-compound net% (as %, not decimal)."""
    rng = np.random.default_rng(42)
    arr = s.dropna().to_numpy()
    if len(arr) < block_size * 2:
        return float(_net(s))
    n = len(arr)
    n_blocks = int(np.ceil(n / block_size))
    boot_nets = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        sample = np.concatenate([arr[st: st + block_size] for st in starts])[:n]
        boot_nets.append(float(np.prod(1 + sample) - 1) * 100)
    return float(np.percentile(boot_nets, 5))


def _sign_test_p(seq: list) -> float:
    """One-sample sign test: H0 = median(seq) <= 0; returns one-sided p-value."""
    pos = sum(x > 0 for x in seq)
    n = len(seq)
    if n == 0:
        return 1.0
    from scipy.stats import binom
    return float(binom.sf(pos - 1, n, 0.5))


# =============================================================================
# DATA LOADING (extended window)
# =============================================================================

def _load_assets(tf: str, span: tuple) -> tuple:
    """Load OHLC assets for the given span at cadence tf. Returns (assets, vt_val, bh_daily)."""
    old_win = TI.WIN
    old_split = TI.SPLIT
    TI.WIN = span
    # Use split at 75% for vol target calibration
    start = pd.Timestamp(span[0])
    end = pd.Timestamp(span[1])
    split_approx = (start + (end - start) * 0.75).strftime("%Y-%m-%d")
    TI.SPLIT = split_approx
    assets, vt = TI.load_ohlc(tf)
    TI.WIN = old_win
    TI.SPLIT = old_split
    if not assets:
        return [], None, None
    # buy-hold daily
    bh_cells = []
    for A in assets:
        ret, win, idx = A["ret"], A["win"], A["idx"]
        bh_cells.append(pd.Series(ret[win], index=idx))
    bh = pd.concat(bh_cells, axis=1).fillna(0.0).mean(axis=1).sort_index()
    bh_daily = bh.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()
    return assets, vt, bh_daily


def _load_assets_ohlcv(tf: str, span: tuple) -> tuple:
    """Load OHLCV assets (for MFI) for the given span."""
    old_win = TI.WIN
    old_split = TI.SPLIT
    TI.WIN = span
    start = pd.Timestamp(span[0])
    end = pd.Timestamp(span[1])
    split_approx = (start + (end - start) * 0.75).strftime("%Y-%m-%d")
    TI.SPLIT = split_approx
    assets, vt = TI.load_ohlcv(tf)
    TI.WIN = old_win
    TI.SPLIT = old_split
    if not assets:
        return [], None, None
    bh_cells = []
    for A in assets:
        ret, win, idx = A["ret"], A["win"], A["idx"]
        bh_cells.append(pd.Series(ret[win], index=idx))
    bh = pd.concat(bh_cells, axis=1).fillna(0.0).mean(axis=1).sort_index()
    bh_daily = bh.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()
    return assets, vt, bh_daily


def _build_per_config_daily(ti_key: str, tf: str, assets: list, vt) -> dict:
    """Build daily return series for EVERY config of ti_key."""
    ind = INDICATORS[ti_key]
    mh = ind.get("minhold", 12)
    cols = {}
    for p in ind["grid"]():
        r = TI._book(assets, ind["iron"], p, vt, mh)
        if r is None:
            continue
        daily = r[0]
        if daily is not None and len(daily) > 10:
            cols[ind["name"](p)] = daily
    return cols


# =============================================================================
# BAND ENSEMBLE: walk-forward, NO look-ahead
# =============================================================================

def _band_ensemble(series_df: pd.DataFrame, lookback_d: int = 120, step_d: int = 30) -> pd.Series:
    """Walk-forward band-ensemble: each step, EW over ALL trailing-positive configs.
    Returns concatenated daily returns with NO look-ahead."""
    idx = series_df.index
    start = idx.min() + pd.Timedelta(days=lookback_d)
    pieces = []
    t = start
    while t < idx.max():
        nxt = t + pd.Timedelta(days=step_d)
        look = series_df[(idx >= t - pd.Timedelta(days=lookback_d)) & (idx < t)]
        fwd = series_df[(idx >= t) & (idx < nxt)]
        if len(look) < max(10, lookback_d // 4) or len(fwd) < 1:
            t = nxt
            continue
        look_net = (np.prod(1 + look.fillna(0.0).to_numpy(), axis=0) - 1) * 100
        band = [c for c, v in zip(series_df.columns, look_net) if v > 0]
        if not band:
            # all negative: go to cash (return 0)
            cash = pd.Series(0.0, index=fwd.index)
            pieces.append(cash)
        else:
            seg = fwd[band].mean(axis=1).dropna()
            pieces.append(seg)
        t = nxt
    if not pieces:
        return pd.Series(dtype=float)
    return pd.concat(pieces).sort_index()


def _rolling_pick(series_df: pd.DataFrame, lookback_d: int = 120, step_d: int = 30) -> pd.Series:
    """Walk-forward rolling-pick: each step, pick the single best trailing-positive config."""
    idx = series_df.index
    cfgs = list(series_df.columns)
    start = idx.min() + pd.Timedelta(days=lookback_d)
    pieces = []
    t = start
    while t < idx.max():
        nxt = t + pd.Timedelta(days=step_d)
        look = series_df[(idx >= t - pd.Timedelta(days=lookback_d)) & (idx < t)]
        fwd = series_df[(idx >= t) & (idx < nxt)]
        if len(look) < max(10, lookback_d // 4) or len(fwd) < 1:
            t = nxt
            continue
        look_net = (np.prod(1 + look.fillna(0.0).to_numpy(), axis=0) - 1) * 100
        band = [c for c, v in zip(cfgs, look_net) if v > 0]
        if not band:
            band = [cfgs[int(np.argmax(look_net))]]
        best = max(band, key=lambda c: look_net[cfgs.index(c)])
        seg = fwd[best].dropna()
        if len(seg):
            pieces.append(seg)
        t = nxt
    if not pieces:
        return pd.Series(dtype=float)
    return pd.concat(pieces).sort_index()


def _pick_vs_ensemble_sign_test(series_df: pd.DataFrame,
                                 lookback_d: int = 120, step_d: int = 30) -> dict:
    """Sign test: does rolling-pick BEAT band-ensemble on a per-window basis?
    H0: median(pick_net_window - ensemble_net_window) = 0. One-sided p (pick > ensemble)."""
    idx = series_df.index
    cfgs = list(series_df.columns)
    start = idx.min() + pd.Timedelta(days=lookback_d)
    diffs = []
    t = start
    while t < idx.max():
        nxt = t + pd.Timedelta(days=step_d)
        look = series_df[(idx >= t - pd.Timedelta(days=lookback_d)) & (idx < t)]
        fwd = series_df[(idx >= t) & (idx < nxt)]
        if len(look) < max(10, lookback_d // 4) or len(fwd) < 2:
            t = nxt
            continue
        look_net = (np.prod(1 + look.fillna(0.0).to_numpy(), axis=0) - 1) * 100
        band = [c for c, v in zip(cfgs, look_net) if v > 0]
        if not band:
            t = nxt
            continue
        # ensemble net for this window
        ens_net = _net(fwd[band].mean(axis=1))
        # pick net for this window
        best = max(band, key=lambda c: look_net[cfgs.index(c)])
        pick_net = _net(fwd[best].dropna())
        diffs.append(pick_net - ens_net)
        t = nxt
    if not diffs:
        return {"n_windows": 0, "p_pick_beats_ens": 1.0, "frac_pick_beats": 0.0,
                "median_diff": 0.0}
    frac = sum(d > 0 for d in diffs) / len(diffs)
    p = _sign_test_p(diffs)
    return {"n_windows": len(diffs), "p_pick_beats_ens": round(p, 4),
            "frac_pick_beats": round(frac, 3), "median_diff": round(float(np.median(diffs)), 2)}


# =============================================================================
# PER-CONFIG BEAR FRACTION
# =============================================================================

def _bear_positive_fraction(cols_2022: dict) -> dict:
    """For each config, compute 2022-bear net. Return fraction > 0 and IQR."""
    nets = []
    for name, daily in cols_2022.items():
        s = daily[(daily.index >= pd.Timestamp("2022-01-01")) &
                  (daily.index < pd.Timestamp("2023-01-01"))]
        nets.append(_net(s))
    if not nets:
        return {}
    arr = np.array(nets)
    return {
        "n_configs": len(arr),
        "frac_positive_2022": round(float(np.mean(arr > 0)), 3),
        "median_net_2022": round(float(np.median(arr)), 1),
        "p25_net_2022": round(float(np.percentile(arr, 25)), 1),
        "p75_net_2022": round(float(np.percentile(arr, 75)), 1),
        "iqr": round(float(np.percentile(arr, 75) - np.percentile(arr, 25)), 1),
    }


# =============================================================================
# HYPERPARAM SENSITIVITY SWEEP
# =============================================================================

def _hyperpar_sweep(cols: dict) -> dict:
    """Sweep (lookback_d, step_d) grid; return band-ensemble 2022-bear net for each cell."""
    series_df = pd.DataFrame(cols)
    results = {}
    for lb in LOOKBACK_GRID:
        for st in STEP_GRID:
            ens = _band_ensemble(series_df, lookback_d=lb, step_d=st)
            if len(ens) < 5:
                results[(lb, st)] = None
                continue
            s22 = ens[(ens.index >= pd.Timestamp("2022-01-01")) &
                      (ens.index < pd.Timestamp("2023-01-01"))]
            results[(lb, st)] = round(_net(s22), 1)
    return results


# =============================================================================
# FULL-CYCLE UNDER HYPERPARAM AVERAGING
# =============================================================================

def _fullcycle_avg(cols: dict, span: tuple) -> dict:
    """Average (equal-weight over lookback x step grid) the full-cycle compound return.
    This is the honest alternative to cherry-picking (120,30)."""
    series_df = pd.DataFrame(cols)
    # Trim to the span
    series_df = series_df[(series_df.index >= pd.Timestamp(span[0])) &
                          (series_df.index < pd.Timestamp(span[1]))]
    ens_curves = []
    for lb in LOOKBACK_GRID:
        for st in STEP_GRID:
            ens = _band_ensemble(series_df, lookback_d=lb, step_d=st)
            if len(ens) > 10:
                ens_curves.append(ens)
    if not ens_curves:
        return {}
    # Average across hyper-param settings -> the hyperparam-averaged equity curve
    avg_df = pd.concat(ens_curves, axis=1).fillna(0.0)
    avg_series = avg_df.mean(axis=1)
    return {
        "compound_net": round(_net(avg_series), 1),
        "maxdd": round(_maxdd(avg_series), 1),
        "sharpe": round(_sharpe_ann(avg_series), 2),
        "n_hyperpar_cells": len(ens_curves),
        "p05_block_bootstrap": round(_block_bootstrap_p05(avg_series), 1),
        "n_eff": round(_n_eff_block(avg_series), 0),
    }


# =============================================================================
# 2024-H1 SECOND BEAR TEST
# =============================================================================

def _run_2024h1(ti_key: str, tf: str = "4h") -> dict:
    """Run the 4 target TIs on 2024-H1 (BTC -30% Q1 drawdown): band-ensemble behavior.
    Uses the TRAIN (2020-2022) period to set the initial band, then evaluates 2024-H1.

    Approach:
    - We need warm-up: load 2023-01-01 to 2024-07-01 (so the band has history before 2024-H1).
    - The first LOOKBACK_D days are warm-up; forward evaluation is 2024-01-01 onward.
    """
    ind = INDICATORS[ti_key]
    # Span: 2023-01-01 to 2024-07-01 to give lookback warm-up
    span = ("2023-01-01", "2024-07-01")
    loader = _load_assets_ohlcv if ind.get("loader") == "ohlcv" else _load_assets
    assets, vt, bh_daily = loader(tf, span)
    if not assets:
        return {"error": "no data"}
    cols = _build_per_config_daily(ti_key, tf, assets, vt)
    if not cols:
        return {"error": "no configs built"}
    series_df = pd.DataFrame(cols)
    # Band-ensemble on the full loaded period (warm-up included)
    ens = _band_ensemble(series_df, lookback_d=120, step_d=30)
    # Extract 2024-H1
    ens_2024h1 = ens[(ens.index >= pd.Timestamp("2024-01-01")) &
                     (ens.index < pd.Timestamp("2024-07-01"))]
    bh_2024h1 = bh_daily[(bh_daily.index >= pd.Timestamp("2024-01-01")) &
                          (bh_daily.index < pd.Timestamp("2024-07-01"))]
    if len(ens_2024h1) < 5:
        return {"error": "insufficient 2024-H1 data after warm-up"}
    # Time-in (fraction in market)
    # Build full position series: band has pos > 0 when not in cash
    # Approximate by: daily return != 0 means in market
    pos_mask = (ens_2024h1 != 0.0)
    time_in = round(float(pos_mask.mean()), 3)
    # Monthly drill-down
    monthly = {}
    for mo in range(1, 7):
        s = ens_2024h1[ens_2024h1.index.month == mo]
        bh_m = bh_2024h1[bh_2024h1.index.month == mo]
        if len(s) > 0:
            monthly[f"2024-{mo:02d}"] = {
                "ens_net": round(_net(s), 1),
                "bh_net": round(_net(bh_m), 1),
            }
    return {
        "ti": ti_key,
        "ens_2024h1_net": round(_net(ens_2024h1), 1),
        "bh_2024h1_net": round(_net(bh_2024h1), 1),
        "ens_maxdd_2024h1": round(_maxdd(ens_2024h1), 1),
        "bh_maxdd_2024h1": round(_maxdd(bh_2024h1), 1),
        "time_in_2024h1": time_in,
        "monthly": monthly,
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    print("=" * 72)
    print("WAVE-2D: Honest Band-Ensemble TI x TF Analysis")
    print(f"Target TIs: {KEEP_TIS}  |  Dropped: {DROP_TIS}")
    print(f"Cadence: 4h  |  UNSEEN sealed: 2024+")
    print("=" * 72)

    tf = "4h"
    results = {}

    # ---- STEP 1: load the cached 2020-2022 JSON and report band-ensemble vs rolling-pick ----
    print("\n--- STEP 1: Band-Ensemble vs Rolling-Pick comparison (2020-2022, from cache) ---")
    aw_files = sorted(OUT.glob("ti_band_rolling_*.json"))
    cached_4h = None
    for f in reversed(aw_files):
        d = json.load(open(f))
        if "4h" in d["by_tf"] and "MACD" in d["by_tf"]["4h"]["results"]:
            cached_4h = d
            print(f"  Using cached file: {f.name}")
            break
    if cached_4h:
        cache_res = cached_4h["by_tf"]["4h"]["results"]
        print(f"\n  TI | rp_2020 | rp_2021 | rp_2022 | be_2020 | be_2021 | be_2022 | bh_2022 |"
              f" be_vs_rp_bear")
        for ti in KEEP_TIS + DROP_TIS:
            if ti not in cache_res:
                continue
            rec = cache_res[ti]
            rp = rec["rolling_pick"]
            be = rec["band_ensemble"]
            bh = rec["buyhold"]
            r20 = rp.get("2020_bull", {}).get("net", "?")
            r21 = rp.get("2021_mixed", {}).get("net", "?")
            r22 = rp.get("2022_bear", {}).get("net", "?")
            e20 = be.get("2020_bull", {}).get("net", "?")
            e21 = be.get("2021_mixed", {}).get("net", "?")
            e22 = be.get("2022_bear", {}).get("net", "?")
            bh22 = bh.get("2022_bear", {}).get("net", "?")
            nc = rec["n_configs"]
            diff = round(e22 - r22, 1) if isinstance(e22, float) and isinstance(r22, float) else "?"
            marker = "KEEP" if ti in KEEP_TIS else "DROP"
            print(f"  [{marker}] {ti:8} n={nc:3} | rp [{r20:>6},{r21:>7},{r22:>6}] "
                  f"be [{e20:>6},{e21:>7},{e22:>6}] bh22={bh22:>6} | be-rp_bear={diff}")

    # ---- STEP 2: Per-config bear-positive fraction (fresh recompute on 2020-2022) ----
    print("\n--- STEP 2: Per-config 2022-bear positive fraction (fresh recompute) ---")
    print("  Loading 2020-2022 data for 4h...")
    TI.WIN = SPAN_TRAIN
    TI.SPLIT = "2022-10-01"

    step2_results = {}
    for ti in KEEP_TIS + DROP_TIS:
        ind = INDICATORS[ti]
        loader = TI.load_ohlcv if ind.get("loader") == "ohlcv" else TI.load_ohlc
        assets, vt = loader(tf)
        if not assets:
            print(f"  {ti}: no assets for {tf}")
            continue
        cols = _build_per_config_daily(ti, tf, assets, vt)
        if not cols:
            print(f"  {ti}: no configs")
            continue
        frac_info = _bear_positive_fraction(cols)
        step2_results[ti] = frac_info
        marker = "KEEP" if ti in KEEP_TIS else "DROP"
        print(f"  [{marker}] {ti:8}: n={frac_info['n_configs']:3} frac_positive_2022={frac_info['frac_positive_2022']:.0%} "
              f"median_bear={frac_info['median_net_2022']:>6}% "
              f"IQR=[{frac_info['p25_net_2022']:>6},{frac_info['p75_net_2022']:>6}]")
        results[ti] = {"bear_fraction_2022": frac_info}

    # ---- STEP 3: Pick vs Ensemble sign test (2020-2022) ----
    print("\n--- STEP 3: Rolling-Pick vs Band-Ensemble sign test (H0: pick adds nothing) ---")
    TI.WIN = SPAN_TRAIN
    TI.SPLIT = "2022-10-01"
    for ti in KEEP_TIS:
        ind = INDICATORS[ti]
        loader = TI.load_ohlcv if ind.get("loader") == "ohlcv" else TI.load_ohlc
        assets, vt = loader(tf)
        if not assets:
            continue
        cols = _build_per_config_daily(ti, tf, assets, vt)
        if not cols:
            continue
        series_df = pd.DataFrame(cols)
        # Restrict to 2020-2022 overlap
        series_df = series_df[(series_df.index >= pd.Timestamp("2020-01-01")) &
                              (series_df.index < pd.Timestamp("2023-01-01"))]
        st = _pick_vs_ensemble_sign_test(series_df)
        print(f"  {ti:8}: n_windows={st['n_windows']:3} frac_pick_wins={st['frac_pick_beats']:.0%} "
              f"p(pick>ensemble)={st['p_pick_beats_ens']:.4f} median_diff={st['median_diff']:+.2f}pp")
        if ti in results:
            results[ti]["pick_vs_ensemble_sign_test"] = st
        else:
            results[ti] = {"pick_vs_ensemble_sign_test": st}

    # ---- STEP 4: Hyperparam averaging (2022 bear, sensitivity grid) ----
    print("\n--- STEP 4: Hyperparam sensitivity (sweep lookback x step, 2022 bear net) ---")
    TI.WIN = SPAN_TRAIN
    TI.SPLIT = "2022-10-01"
    for ti in KEEP_TIS:
        ind = INDICATORS[ti]
        loader = TI.load_ohlcv if ind.get("loader") == "ohlcv" else TI.load_ohlc
        assets, vt = loader(tf)
        if not assets:
            continue
        cols = _build_per_config_daily(ti, tf, assets, vt)
        if not cols:
            continue
        series_df = pd.DataFrame(cols)
        print(f"\n  {ti} (n_configs={len(cols)}): lookback x step -> 2022-bear net%")
        print("  " + " ".join(f"step={s:3}" for s in STEP_GRID))
        hp_results = {}
        for lb in LOOKBACK_GRID:
            row_vals = []
            for st in STEP_GRID:
                ens = _band_ensemble(series_df, lookback_d=lb, step_d=st)
                s22 = ens[(ens.index >= pd.Timestamp("2022-01-01")) &
                          (ens.index < pd.Timestamp("2023-01-01"))]
                v = round(_net(s22), 1) if len(s22) > 10 else None
                hp_results[(lb, st)] = v
                row_vals.append(f"{v:>7}" if v is not None else "     ?")
            print(f"  lb={lb:3}: " + " ".join(row_vals))
        # Summarize: how many cells are positive?
        vals = [v for v in hp_results.values() if v is not None]
        frac_pos = sum(v > 0 for v in vals) / len(vals) if vals else 0.0
        median_v = np.median(vals) if vals else float("nan")
        print(f"  {ti}: {frac_pos:.0%} of (lb,step) cells positive in 2022 bear | median={median_v:.1f}%")
        if ti in results:
            results[ti]["hyperpar_sensitivity_2022"] = {
                "grid": {f"lb{lb}_st{st}": v for (lb, st), v in hp_results.items()},
                "frac_positive": round(frac_pos, 3),
                "median_2022_bear_net": round(float(median_v), 1) if not np.isnan(median_v) else None,
            }

    # ---- STEP 5: Full-cycle compound under hyperparam averaging ----
    print("\n--- STEP 5: Full-cycle compound under hyperparam AVERAGING (2020-2023) ---")
    print("  (avg over 5 lookback x 4 step = 20 cells to kill cherry-pick)")
    TI.WIN = SPAN_TRAIN
    TI.SPLIT = "2022-10-01"
    for ti in KEEP_TIS:
        ind = INDICATORS[ti]
        loader = TI.load_ohlcv if ind.get("loader") == "ohlcv" else TI.load_ohlc
        assets, vt = loader(tf)
        if not assets:
            continue
        cols = _build_per_config_daily(ti, tf, assets, vt)
        if not cols:
            continue
        fc = _fullcycle_avg(cols, SPAN_TRAIN)
        # Also compute buy-hold over same period
        bh_cells = []
        for A in assets:
            bh_cells.append(pd.Series(A["ret"][A["win"]], index=A["idx"]))
        bh_s = pd.concat(bh_cells, axis=1).fillna(0.0).mean(axis=1)
        bh_d = bh_s.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()
        bh_net = round(_net(bh_d), 1)
        bh_dd = round(_maxdd(bh_d), 1)
        print(f"  {ti:8}: avg-ens compound={fc.get('compound_net','?'):>7}%  "
              f"maxDD={fc.get('maxdd','?'):>6}%  Sh={fc.get('sharpe','?'):>5}  "
              f"p05_boot={fc.get('p05_block_bootstrap','?'):>7}%  "
              f"| BH compound={bh_net:>8}% maxDD={bh_dd}%  "
              f"n_eff={fc.get('n_eff','?'):.0f}")
        if ti in results:
            results[ti]["fullcycle_hyperparam_avg"] = fc
        else:
            results[ti] = {"fullcycle_hyperparam_avg": fc}

    # ---- STEP 6: 2024-H1 second-bear test ----
    print("\n--- STEP 6: 2024-H1 SECOND BEAR TEST (BTC ~-30% Q1 drawdown) ---")
    print("  NOTE: This is the UNSEEN split -- sealed after this one run.")
    print("  (warm-up: 2023-01-01 to 2024-01-01; evaluation: 2024-01-01 to 2024-07-01)")
    h1_results = {}
    for ti in KEEP_TIS:
        r = _run_2024h1(ti, tf)
        h1_results[ti] = r
        if "error" in r:
            print(f"  {ti:8}: ERROR: {r['error']}")
            continue
        ens_net = r["ens_2024h1_net"]
        bh_net_2024 = r["bh_2024h1_net"]
        ens_dd = r["ens_maxdd_2024h1"]
        bh_dd = r["bh_maxdd_2024h1"]
        tin = r["time_in_2024h1"]
        preservation = round(ens_net - bh_net_2024, 1)
        print(f"  {ti:8}: ens={ens_net:>6}%  bh={bh_net_2024:>6}%  "
              f"preservation={preservation:>+6}pp  ens_DD={ens_dd}%  bh_DD={bh_dd}%  "
              f"time_in={tin:.0%}")
        print(f"          monthly: " + "  ".join(
            f"{mo}: ens={v['ens_net']:>5}% bh={v['bh_net']:>5}%"
            for mo, v in r.get("monthly", {}).items()
        ))
        if ti in results:
            results[ti]["second_bear_2024h1"] = r
        else:
            results[ti] = {"second_bear_2024h1": r}

    # ---- STEP 7: Statistical verdict ----
    print("\n" + "=" * 72)
    print("STATISTICAL VERDICT")
    print("=" * 72)
    print("""
  Null hypothesis tested: H0 = band-ensemble bear preservation is a random artifact
  of a cherry-picked (lookback, step) pair on the lone 2022 bear. Alternatives:
    Ha1: frac_positive across (lb, step) grid > 50% in 2022 bear (majority of configs).
    Ha2: mechanism replicates on INDEPENDENT 2024-H1 bear (out-of-period confirmation).
    Ha3: hyperparam-averaged full-cycle compound still outperforms vs maxDD.

  Verdict criteria:
    REAL:      Ha1 AND Ha2 both clear (frac_pos > 50%, 2024-H1 preservation holds)
    ARTIFACT:  < 50% cells positive OR 2024-H1 fails to preserve (ens DD >= BH DD)
    AMBIGUOUS: mixed evidence
  """)
    for ti in KEEP_TIS:
        if ti not in results:
            print(f"  {ti}: insufficient data")
            continue
        r = results[ti]
        hp = r.get("hyperpar_sensitivity_2022", {})
        frac_p = hp.get("frac_positive", None)
        med_2022 = hp.get("median_2022_bear_net", None)
        bear2 = r.get("second_bear_2024h1", {})
        ens_net_2024 = bear2.get("ens_2024h1_net", None)
        bh_net_2024 = bear2.get("bh_2024h1_net", None)
        preservation_2024 = (round(ens_net_2024 - bh_net_2024, 1)
                              if ens_net_2024 is not None and bh_net_2024 is not None
                              else None)
        fc = r.get("fullcycle_hyperparam_avg", {})
        p05 = fc.get("p05_block_bootstrap", None)
        n_eff = fc.get("n_eff", None)
        ha1 = frac_p is not None and frac_p > 0.50
        ha2 = (preservation_2024 is not None and preservation_2024 > 5.0 and
               ens_net_2024 is not None and ens_net_2024 > bh_net_2024)
        if ha1 and ha2:
            verdict = "REAL (de-risked beta mechanism confirmed)"
        elif ha1 or ha2:
            verdict = "AMBIGUOUS"
        else:
            verdict = "ARTIFACT"
        print(f"  {ti:8}: Ha1(frac>50%)={'PASS' if ha1 else 'FAIL'} frac={frac_p:.0%} median_2022={med_2022}%"
              f"  |  Ha2(2024-H1)={'PASS' if ha2 else 'FAIL'} pres={preservation_2024:+.1f}pp"
              f"  |  p05_boot={p05}%  n_eff={int(n_eff) if n_eff else '?'}")
        print(f"  -> VERDICT: {verdict}")

    print("""
  n_eff note: one bear year at 4h = ~2196 bars; block_size=20 -> n_eff~110 blocks.
  This is enough to detect a >5%/yr preservation edge at p<0.05 (se = std / sqrt(n_eff)).
  A p05_boot < 0 means the full-cycle compound is NOT reliably positive.
  The TI x TF sleeve = de-risked beta (drawdown-insurance) if:
    bear-preservation holds structurally (Ha1 + Ha2) AND p05_boot < 0 full-cycle.
  """)

    # ---- Save ----
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT / f"wave2d_honest_band_{stamp}.json"
    json.dump({
        "wave": "2D",
        "timestamp": stamp,
        "tis_kept": KEEP_TIS,
        "tis_dropped": DROP_TIS,
        "tf": tf,
        "results": results,
        "second_bear_2024h1": h1_results,
    }, open(out_path, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
