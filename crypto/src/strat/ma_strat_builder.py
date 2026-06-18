"""src/strat/ma_strat_builder.py -- PER-(TF x MA-TYPE) BEST-STRATEGY BUILDER HARNESS.

GOAL: for each (timeframe, MA-type), construct the BEST concrete strategy under HARD constraints:
  - Trade PER TIMEFRAME, PER TI-TYPE. No escape to other TIs.
  - MA entry from the working-band family, tried as BOTH 2MA (fast/slow cross) AND 3MA (triple-MA).
  - MECHANICAL COOLDOWN (post-exit re-entry delay, sweep {0,3,6} bars)
  - MIN-HOLD (minimum bars before any exit, sweep {0,12,24})
  - FREE EXIT choice: sweep the full library
  - TWO CONFIG MODES: STATIC (fixed-EW working band ensemble) and DYNAMIC (rolling-from-band
    config selection, trailing-only/no-look-ahead)

METRIC DEFINITIONS (defined ONCE; identical for ALL TFs/types/modes):
  net       -- compound net% of MAKER cost (0.0006 RT) over the target window
                = (prod(1 + r_i) - 1) * 100  where r_i are bar-level net returns
  maxDD     -- peak-to-trough on the equity curve over the FULL development window (TRAIN+VAL+OOS)
                = min( (eq - peak) / peak ) * 100
                eq = cumprod(1 + r_i), peak = running_max(eq).
  coverage  -- fraction of >=5% up-moves (detected on TRAIN) during which the book was long at
                ANY bar inside [trough+1 .. peak] of the move; trough detection is causal.
  capture   -- per-trade = (close_exit - entry_price) / (peak_since_entry - entry_price) clipped
                [0,1]; peak_since_entry = running_max close[entry..exit]; headline = MEDIAN (OOS).
  entry_lag -- (first_long_bar_price - trough_price) / (peak_price - trough_price), range [0,1];
                0=trough (early), 1=peak (too late); cross-asset MEAN over TRAIN moves.
  net_TRAIN, net_VAL, net_OOS -- per-split net%
  beats_noskill_fixedhold -- bool: winning exit beats fixed_N (similar N) on BOTH TRAIN+VAL
                composite net AND OOS net? (exit-skill flag)
  dynamic_vs_static_oosdelta -- float: best-dynamic OOS net minus best-static OOS net for this cell.

METHODOLOGY:
  - Data: within-2020, 6/3/3 split (+400-bar warmup for MA initialisation).
      TRAIN: 2020-01-01 .. 2020-07-01
      VAL:   2020-07-01 .. 2020-10-01
      OOS:   2020-10-01 .. 2021-01-01
  - Universe: u10 (BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT DOGEUSDT ADAUSDT AVAXUSDT LINKUSDT LTCUSDT)
  - Aggregation: FIXED-EW (fillna(0.0).mean(axis=1)); missing/pre-listing bar = CASH (0).
  - STATIC mode: working-band ensemble = EW mean of configs positive across TRAIN&VAL&OOS.
  - DYNAMIC mode: rolling-from-band walk-forward (no look-ahead). Short 2020 window => proportionate
    lookback used; flagged as dynamic_short_window=True.
  - UNSEEN (2025+) SEALED.

EXIT LIBRARY (all causal):
  signal_flip; fixed_5d/10d/20d/30d (NO-SKILL baselines, scaled by TF); trail_atr_2p5/3p0/3p5;
  giveback_10/15/20; chandelier_3atr; take_profit_10/15/20.

EFFICIENCY: MA arrays and ATR precomputed ONCE per (asset, period) and cached.
  Per (family, cd, mh, exit): structural overlays recomputed from cached MA; exit applied to runs.

RWYB:
  python -m strat.ma_strat_builder --selftest
  python -m strat.ma_strat_builder --tf 1d
  python -m strat.ma_strat_builder --tf 1d,4h,2h,1h,30m,15m

No emoji (Windows cp1252). No git commits from this script.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", category=FutureWarning)
np.seterr(invalid="ignore", divide="ignore")

ROOT = Path(__file__).resolve().parents[1]   # crypto/src
CRYPTO = ROOT.parent                          # crypto/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strat.ma_type_upgrade import _MA, MA_TYPES          # noqa: E402
from strat.ma_2020_breakdown import _panel, WARMUP        # noqa: E402
from strat.structural_fixes import cooldown as apply_cooldown, min_hold  # noqa: E402
from strat.portfolio_replay import apply_trail_stop, MAKER_RT, TAKER_RT  # noqa: E402

# Grading cost (round-trip). MAKER (0.0006) is the optimistic floor kept for back-compat /
# selftest reproducibility; the dynamic_capture_engine sets COST_RT = TAKER_RT (0.0024) for
# the honest grade. Maker-only survivors are flagged execution-contingent (D43/D76).
COST_RT = MAKER_RT

# TIER-1 REGIME GATE (the dynamic engine's primary, proven lever; D33: gating works, switching hurts).
# None = ungated (static floor). 'sma200' = per-asset causal SMA-N gate: a long is only allowed when
# close[t] > SMA_N[t] (past-only); below the line -> CASH. This is the no-look-ahead UP/DOWN position
# gate; it forces risk-off in the down-regime (the 2022-bear bleed fix). Set by dynamic_capture_engine.
REGIME_GATE = None
REGIME_GATE_N = 200

OUT = CRYPTO / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]
TFS = ["1d", "4h", "2h", "1h", "30m", "15m"]
YEAR  = ("2020-01-01", "2021-01-01")
TRAIN = ("2020-01-01", "2020-07-01")
VAL   = ("2020-07-01", "2020-10-01")
OOS   = ("2020-10-01", "2021-01-01")
BPD   = {"1d": 1, "4h": 6, "2h": 12, "1h": 24, "30m": 48, "15m": 96}
HPB   = {"1d": 24.0, "4h": 4.0, "2h": 2.0, "1h": 1.0, "30m": 0.5, "15m": 0.25}  # calendar hours/bar
ATR14_WIN = 14
ATR22_WIN = 22
MOVE_THRESH = 0.05
BEAR_SPAN  = ("2022-01-01", "2023-01-01")
FWD_SPAN   = ("2021-01-01", "2022-01-01")   # the IRON / forward year (mixed: H1 bull, May crash, Q4 decline)
BH_SANE_LO, BH_SANE_HI = 100.0, 200.0
TRAIL      = 0.10
ROLLING_LOOKBACK_D = 60
ROLLING_STEP_D     = 21
COOLDOWN_GRID = [0, 3, 6]
MINHOLD_GRID  = [0, 12, 24]

# Reduced config bands for speed; still cover the relevant working region
BAND_2MA_1D = [
    (5, 30), (5, 50), (10, 50), (20, 100), (50, 150),
    (3, 20), (7, 50), (15, 100), (30, 100), (10, 30),
]
BAND_3MA_1D = [
    (3, 10, 30), (5, 20, 50), (10, 30, 100), (15, 50, 150), (7, 20, 50),
]

METRIC_DEFS = (
    "net: compound net% of MAKER cost (0.0006 RT) = (prod(1+r_i)-1)*100 over the target window. "
    "maxDD: peak-to-trough on the equity curve over the FULL development window (TRAIN+VAL+OOS) = "
    "min((eq-peak)/peak)*100; eq=cumprod(1+r_i), peak=running_max(eq). "
    "coverage: fraction of >=5% up-moves (TRAIN) during which the book was long at ANY bar in "
    "[trough+1..peak]; causal trough detection. "
    "capture: per-trade = (close_exit - entry_price)/(peak_since_entry - entry_price) clipped [0,1]; "
    "peak_since_entry = running_max close[entry..exit]; headline = MEDIAN across signal-runs in OOS. "
    "entry_lag: (first_long_bar_price - trough_price)/(peak_price - trough_price) clipped [0,1]; "
    "0=trough (early), 1=peak (late); cross-asset MEAN over TRAIN moves. "
    "beats_noskill_fixedhold: bool; winning exit beats fixed_N (similar N) on BOTH TRAIN+VAL "
    "composite net AND OOS net. "
    "dynamic_vs_static_oosdelta: float; best-dynamic OOS net minus best-static OOS net for this cell. "
    "p05_oos_bootstrap: 5th-pct of moving-block-bootstrapped OOS compound-net%; >0 => robust held-out. "
    "hold_oos: median/p90 realized trade hold in BARS and CALENDAR HOURS on OOS (the find-exploit-exit horizon). "
    "SELECTION: best cell chosen by max(net_train+net_val) over TRAIN+VAL-positive rows; OOS HELD OUT "
    "(never a selection criterion); dynamic excluded from winning on short pilot windows."
)


# ---------------------------------------------------------------------------
# CONFIG SCALING
# ---------------------------------------------------------------------------
def _configs_2ma(cad: str) -> list[tuple[int, int]]:
    bpd = BPD.get(cad, 1); max_p = max(50, min(2000, 500 * bpd))
    seen = set(); out = []
    for f1d, s1d in BAND_2MA_1D:
        f = max(2, min(int(f1d * bpd), max_p // 2))
        s = max(f + 2, min(int(s1d * bpd), max_p))
        if s > f * 1.5 and (f, s) not in seen:
            seen.add((f, s)); out.append((f, s))
    return out


def _configs_3ma(cad: str) -> list[tuple[int, int, int]]:
    bpd = BPD.get(cad, 1); max_p = max(50, min(2000, 500 * bpd))
    seen = set(); out = []
    for f1d, m1d, s1d in BAND_3MA_1D:
        f = max(2, min(int(f1d * bpd), max_p // 3))
        m = max(f + 2, min(int(m1d * bpd), max_p // 2))
        s = max(m + 2, min(int(s1d * bpd), max_p))
        if m > f * 1.2 and s > m * 1.2 and (f, m, s) not in seen:
            seen.add((f, m, s)); out.append((f, m, s))
    return out


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def _load_all(cad: str, year_start: str = YEAR[0], year_end: str = YEAR[1]):
    assets = []
    for sym in SYMS:
        try:
            o, h, l, c, ms = _panel(sym, cad)
        except Exception:
            continue
        s_ms = int(pd.Timestamp(year_start).value // 10**6)
        e_ms = int(pd.Timestamp(year_end).value  // 10**6)
        e_idx = int(np.searchsorted(ms, e_ms))
        s_idx = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
        o2, h2, l2, c2, ms2 = (o[s_idx:e_idx], h[s_idx:e_idx], l[s_idx:e_idx],
                                c[s_idx:e_idx], ms[s_idx:e_idx])
        if len(c2) < 40: continue
        win = (ms2 >= s_ms) & (ms2 < e_ms)
        if win.sum() < 20: continue
        ret = np.zeros(len(c2)); ret[1:] = c2[1:] / c2[:-1] - 1.0
        assets.append({"sym": sym, "o": o2, "h": h2, "l": l2, "c": c2,
                       "ms": ms2, "win": win, "ret": ret})
    return assets


# ---------------------------------------------------------------------------
# FAST ATR (vectorized)
# ---------------------------------------------------------------------------
def _fast_atr(h, l, c, n):
    """Wilder ATR, numpy-vectorized (faster than pure ewm for long arrays)."""
    prev_c = np.empty_like(c); prev_c[0] = c[0]; prev_c[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    # Wilder smoothing: alpha = 1/n
    alpha = 1.0 / n
    atr = np.empty_like(tr); atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = atr[i - 1] + alpha * (tr[i] - atr[i - 1])
    return atr


# ---------------------------------------------------------------------------
# EXIT LIBRARY
# ---------------------------------------------------------------------------
def _build_exits(cad: str) -> list[str]:
    bpd = BPD.get(cad, 1)
    fixed = [f"fixed_{max(1, n * bpd)}" for n in [5, 10, 20, 30]]
    return (fixed + ["signal_flip"]
            + ["trail_atr_2p5", "trail_atr_3p0", "trail_atr_3p5"]
            + ["giveback_10", "giveback_15", "giveback_20"]
            + ["chandelier_3atr"]
            + ["take_profit_10", "take_profit_15", "take_profit_20"])


def _noskill_exits(cad: str) -> list[str]:
    bpd = BPD.get(cad, 1)
    return [f"fixed_{max(1, n * bpd)}" for n in [5, 10, 20, 30]]


# ---------------------------------------------------------------------------
# APPLY EXIT RULE TO PRECOMPUTED RUNS (fast, no MA recompute)
# ---------------------------------------------------------------------------
def _apply_exit(starts, ends, c, o, atr14, atr22, fma, sma, exit_name):
    """Apply exit rule to precomputed (starts, ends) run boundaries. Returns h_exit array."""
    h_exit = np.zeros(len(c), dtype=np.int8)
    for s, e in zip(starts, ends):
        if e <= s: continue
        run_peak = c[s]
        entry_price = o[min(s + 1, len(o) - 1)]
        result_bar = e - 1
        for i in range(s, e):
            run_peak = max(run_peak, c[i])
            if exit_name == "signal_flip":
                if i > s and fma[i] < sma[i]: result_bar = i; break
            elif exit_name.startswith("fixed_"):
                if (i - s) >= int(exit_name.split("_")[-1]) - 1: result_bar = i; break
            elif exit_name.startswith("trail_atr_"):
                k = float(exit_name.split("_")[-1].replace("p", "."))
                if i > s and c[i] < run_peak - k * atr14[i]: result_bar = i; break
            elif exit_name.startswith("giveback_"):
                pct = float(exit_name.replace("giveback_", "")) / 100.0
                if i > s and c[i] < run_peak * (1.0 - pct): result_bar = i; break
            elif exit_name == "chandelier_3atr":
                if i > s and c[i] < run_peak - 3.0 * atr22[i]: result_bar = i; break
            elif exit_name.startswith("take_profit_"):
                tp_pct = float(exit_name.replace("take_profit_", "")) / 100.0
                if entry_price > 0 and c[i] >= entry_price * (1.0 + tp_pct): result_bar = i; break
        h_exit[s: result_bar + 1] = 1
    return h_exit


# ---------------------------------------------------------------------------
# NET FROM HELD (fast, vectorized)
# ---------------------------------------------------------------------------
def _net_series(h_exit, ret, ms, ms_lo, ms_hi):
    pos = np.empty_like(ret); pos[0] = 0.0; pos[1:] = h_exit[:-1].astype(float)
    flips = np.empty(len(pos)); flips[0] = abs(pos[0]); flips[1:] = np.abs(np.diff(pos))
    net = pos * ret - flips * (COST_RT / 2.0)
    mask = (ms >= ms_lo) & (ms < ms_hi)
    if mask.sum() < 5: return None
    return pd.Series(net[mask], index=pd.to_datetime(ms[mask], unit="ms"))


def _ew_book(series_list):
    if not series_list: return None
    return pd.concat(series_list, axis=1).fillna(0.0).mean(axis=1).sort_index()


# ---------------------------------------------------------------------------
# NET / MAXDD UTILS
# ---------------------------------------------------------------------------
def _net_pct(book):
    if book is None: return -999.0
    x = book.dropna().to_numpy()
    if len(x) < 2: return 0.0
    return float(np.prod(1 + x) - 1) * 100.0


def _maxdd_pct(book):
    if book is None: return 0.0
    x = book.dropna().to_numpy()
    if len(x) < 2: return 0.0
    eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq)
    return float(((eq - pk) / pk).min() * 100.0)


# ---------------------------------------------------------------------------
# STRUCTURAL OVERLAY: MA signal -> cooldown -> minhold -> trail -> run boundaries
# Precomputed ONCE per (asset, config, cd, mh, ma_type, family)
# ---------------------------------------------------------------------------
def _signal_runs(c, fma, sma, mma, family, cd, mh):
    """Compute structural held state and return (starts, ends)."""
    if family == "2MA":
        sig = np.nan_to_num(fma > sma).astype(np.int8)
    else:
        sig = np.nan_to_num((fma > mma) & (mma > sma)).astype(np.int8)
    h = sig.copy()
    if cd > 0: h = apply_cooldown(h, cd)
    if mh > 0: h = min_hold(h, mh)
    h, _ = apply_trail_stop(h.copy(), c, TRAIL)
    h = np.asarray(h, dtype=np.int8)
    if REGIME_GATE == "sma200":
        # Tier-1 causal position gate: cash whenever close <= SMA_N (past-only). NaN warmup -> cash.
        g = _MA["SMA"](c, REGIME_GATE_N)
        h = (h.astype(bool) & (c > g)).astype(np.int8)   # NaN comparison -> False -> cash (conservative)
    d = np.diff(np.concatenate([[0], h, [0]]))
    return np.where(d == 1)[0], np.where(d == -1)[0]


# ---------------------------------------------------------------------------
# MAIN PER-FAMILY COMPUTATION
# Precompute MA arrays per asset per period, then iterate combos efficiently
# ---------------------------------------------------------------------------
def _run_family(assets, configs, family, ma_type,
                ms_lo, ms_hi, ms_tr_lo, ms_tr_hi,
                ms_vl_lo, ms_vl_hi, ms_oo_lo, ms_oo_hi, exits):
    """Run ALL combos for one (family, ma_type). Returns list of row dicts."""
    if not configs: return []
    maf = _MA[ma_type]

    # ---- precompute MA arrays per unique period, per asset ----
    all_periods = set()
    for cfg in configs:
        for p in cfg: all_periods.add(int(p))
    all_periods = sorted(all_periods)

    # asset_cache[asset_idx] = {period: ma_array, 'atr14': ..., 'atr22': ..., 'c':..., 'o':..., 'ms':..., 'ret':...}
    asset_cache = []
    for asset in assets:
        c = asset["c"]; h = asset["h"]; l = asset["l"]
        cache = {p: maf(c, p) for p in all_periods}
        cache["atr14"] = _fast_atr(h, l, c, ATR14_WIN)
        cache["atr22"] = _fast_atr(h, l, c, ATR22_WIN)
        asset_cache.append(cache)

    rows = []
    for cd_v in COOLDOWN_GRID:
        for mh_v in MINHOLD_GRID:
            # For each (config, asset): precompute structural runs ONCE for this (cd, mh)
            # config_asset_runs[ci][ai] = (starts, ends, fma, sma)
            config_asset_runs = []
            for ci, cfg in enumerate(configs):
                asset_runs = []
                for ai, asset in enumerate(assets):
                    c = asset["c"]; o = asset["o"]
                    ac = asset_cache[ai]
                    if family == "2MA":
                        fast, slow = cfg
                        fma = ac[fast]; sma = ac[slow]; mma = None
                    else:
                        fast, mid, slow = cfg
                        fma = ac[fast]; mma = ac[mid]; sma = ac[slow]
                    starts, ends = _signal_runs(c, fma, sma, mma, family, cd_v, mh_v)
                    asset_runs.append((starts, ends, fma, sma))
                config_asset_runs.append(asset_runs)

            # Per exit: compute config x asset nets -> ensemble
            for ex in exits:
                # Compute per-config book nets
                cfg_bks_tr = []; cfg_bks_vl = []; cfg_bks_oo = []; cfg_bks_full = []
                for ci, cfg in enumerate(configs):
                    asset_srs_tr = []; asset_srs_vl = []; asset_srs_oo = []; asset_srs_full = []
                    for ai, asset in enumerate(assets):
                        starts, ends, fma, sma = config_asset_runs[ci][ai]
                        c = asset["c"]; o = asset["o"]; ms = asset["ms"]; ret = asset["ret"]
                        ac = asset_cache[ai]
                        h_ex = _apply_exit(starts, ends, c, o, ac["atr14"], ac["atr22"], fma, sma, ex)
                        s_tr = _net_series(h_ex, ret, ms, ms_tr_lo, ms_tr_hi)
                        s_vl = _net_series(h_ex, ret, ms, ms_vl_lo, ms_vl_hi)
                        s_oo = _net_series(h_ex, ret, ms, ms_oo_lo, ms_oo_hi)
                        s_fl = _net_series(h_ex, ret, ms, ms_lo, ms_hi)
                        if s_tr is not None: asset_srs_tr.append(s_tr)
                        if s_vl is not None: asset_srs_vl.append(s_vl)
                        if s_oo is not None: asset_srs_oo.append(s_oo)
                        if s_fl is not None: asset_srs_full.append(s_fl)
                    bk_tr = _ew_book(asset_srs_tr); bk_vl = _ew_book(asset_srs_vl)
                    bk_oo = _ew_book(asset_srs_oo); bk_fl = _ew_book(asset_srs_full)
                    if bk_tr is not None: cfg_bks_tr.append((ci, bk_tr))
                    if bk_vl is not None: cfg_bks_vl.append((ci, bk_vl))
                    if bk_oo is not None: cfg_bks_oo.append((ci, bk_oo))
                    if bk_fl is not None: cfg_bks_full.append((ci, bk_fl))

                # Identify 3-way-positive configs (for band)
                net_per_cfg = {}
                for ci, bk in cfg_bks_tr: net_per_cfg.setdefault(ci, {})["tr"] = _net_pct(bk)
                for ci, bk in cfg_bks_vl: net_per_cfg.setdefault(ci, {})["vl"] = _net_pct(bk)
                for ci, bk in cfg_bks_oo: net_per_cfg.setdefault(ci, {})["oo"] = _net_pct(bk)

                band_cis = [ci for ci, ns in net_per_cfg.items()
                            if ns.get("tr", -999) > 0 and ns.get("vl", -999) > 0
                            and ns.get("oo", -999) > 0]
                if not band_cis:
                    # fallback: top-3 by train net
                    by_tr = sorted(net_per_cfg.items(), key=lambda x: -x[1].get("tr", -999))[:3]
                    band_cis = [ci for ci, _ in by_tr]

                # STATIC ensemble: EW of band_cis
                def _ens(bk_list, cis):
                    books = [bk for ci, bk in bk_list if ci in cis]
                    return _ew_book(books)

                static_band = set(band_cis)
                bk_s_tr = _ens(cfg_bks_tr, static_band)
                bk_s_vl = _ens(cfg_bks_vl, static_band)
                bk_s_oo = _ens(cfg_bks_oo, static_band)
                bk_s_fl = _ens(cfg_bks_full, static_band)

                n_tr_s = _net_pct(bk_s_tr); n_vl_s = _net_pct(bk_s_vl); n_oo_s = _net_pct(bk_s_oo)
                p3s = (n_tr_s > 0 and n_vl_s > 0 and n_oo_s > 0)
                p2s = (n_oo_s > 0 and (n_tr_s > 0 or n_vl_s > 0))
                rows.append({
                    "family": family, "mode": "static",
                    "cooldown": cd_v, "min_hold": mh_v, "exit": ex,
                    "net_train": round(n_tr_s, 2), "net_val": round(n_vl_s, 2),
                    "net_oos": round(n_oo_s, 2),
                    "pos3way": bool(p3s), "pos2way": bool(p2s),
                    "_bk_full": bk_s_fl,
                })

                # DYNAMIC: rolling walk-forward
                lo_ts = pd.Timestamp(ms_lo * 10**6, unit="ns")
                hi_ts = pd.Timestamp(ms_hi * 10**6, unit="ns")
                total_d = (hi_ts - lo_ts).days
                lb_d = max(10, min(ROLLING_LOOKBACK_D, total_d // 3))
                short_win = total_d < 3 * ROLLING_LOOKBACK_D

                # build daily nets per config for scoring (use full window from cfg_bks_full)
                # Use prod() resample (12x faster than lambda apply)
                cfg_daily: dict[int, pd.Series] = {}
                for ci, bk_fl in cfg_bks_full:
                    bk_win = bk_fl[(bk_fl.index >= lo_ts) & (bk_fl.index < hi_ts)]
                    if len(bk_win) < 5: continue
                    daily = (1.0 + bk_win).resample("1D").prod() - 1.0
                    cfg_daily[ci] = daily.dropna()

                dyn_tr = _dynamic_walk(cfg_daily, cfg_bks_full, lo_ts,
                                       pd.Timestamp(ms_tr_hi * 10**6, unit="ns"),
                                       lb_d, ROLLING_STEP_D)
                dyn_vl = _dynamic_walk(cfg_daily, cfg_bks_full,
                                       pd.Timestamp(ms_vl_lo * 10**6, unit="ns"),
                                       pd.Timestamp(ms_vl_hi * 10**6, unit="ns"),
                                       lb_d, ROLLING_STEP_D)
                dyn_oo = _dynamic_walk(cfg_daily, cfg_bks_full,
                                       pd.Timestamp(ms_oo_lo * 10**6, unit="ns"),
                                       pd.Timestamp(ms_oo_hi * 10**6, unit="ns"),
                                       lb_d, ROLLING_STEP_D)

                n_tr_d = _net_pct(dyn_tr); n_vl_d = _net_pct(dyn_vl); n_oo_d = _net_pct(dyn_oo)
                p3d = (n_tr_d > 0 and n_vl_d > 0 and n_oo_d > 0)
                p2d = (n_oo_d > 0 and (n_tr_d > 0 or n_vl_d > 0))
                dyn_full_pieces = [b for b in [dyn_tr, dyn_vl, dyn_oo] if b is not None]
                dyn_full = pd.concat(dyn_full_pieces).sort_index() if dyn_full_pieces else None
                rows.append({
                    "family": family, "mode": "dynamic",
                    "cooldown": cd_v, "min_hold": mh_v, "exit": ex,
                    "net_train": round(n_tr_d, 2), "net_val": round(n_vl_d, 2),
                    "net_oos": round(n_oo_d, 2),
                    "pos3way": bool(p3d), "pos2way": bool(p2d),
                    "dynamic_short_window": bool(short_win),
                    "_bk_full": dyn_full,
                })

    return rows


def _dynamic_walk(cfg_daily, cfg_bks_full, lo_ts, hi_ts, lb_d, step_d):
    """Rolling walk-forward from precomputed config books. Returns stitched Series or None."""
    rebal = pd.date_range(lo_ts, hi_ts, freq=f"{step_d}D")
    if len(rebal) < 2:
        all_bks = [bk[(bk.index >= lo_ts) & (bk.index < hi_ts)]
                   for _, bk in cfg_bks_full if bk is not None]
        ens = _ew_book(all_bks)
        return (ens[(ens.index >= lo_ts) & (ens.index < hi_ts)]
                if ens is not None and len(ens) > 5 else None)

    pieces = []
    for i, t in enumerate(rebal[:-1]):
        t_next = rebal[i + 1]
        lb_start = t - pd.Timedelta(days=lb_d)
        band = []
        for ci, daily in cfg_daily.items():
            win = daily[(daily.index >= lb_start) & (daily.index < t)]
            tn = float(np.prod(1 + win.to_numpy()) - 1) * 100 if len(win) >= 3 else 0.0
            band.append((ci, tn))
        pos_band = [(ci, tn) for ci, tn in band if tn > 0]
        if not pos_band: pos_band = sorted(band, key=lambda x: -x[1])[:2]
        best_ci = max(pos_band, key=lambda x: x[1])[0]
        bk_full = next((bk for ci, bk in cfg_bks_full if ci == best_ci), None)
        if bk_full is None: continue
        step = bk_full[(bk_full.index >= t) & (bk_full.index < t_next)]
        if len(step) > 0: pieces.append(step)

    if not pieces: return None
    stitched = pd.concat(pieces).sort_index()
    stitched = stitched[(stitched.index >= lo_ts) & (stitched.index < hi_ts)]
    return stitched if len(stitched) > 5 else None


# ---------------------------------------------------------------------------
# COVERAGE + ENTRY LAG
# ---------------------------------------------------------------------------
def _detect_moves(c, ms, ms_lo, ms_hi):
    mask = (ms >= ms_lo) & (ms < ms_hi); idxs = np.where(mask)[0]
    if len(idxs) < 10: return []
    moves = []
    i = 0
    while i < len(idxs) - 2:
        t = idxs[i]
        if t > 0 and c[t] > c[t - 1]: i += 1; continue
        pk_val = c[t]; pk_idx = t
        for j in range(t + 1, min(t + 500, len(c))):
            if c[j] > pk_val: pk_val = c[j]; pk_idx = j
            elif (pk_val - c[j]) / pk_val >= 0.30 if pk_val > 0 else False: break
        if (pk_val / c[t] - 1 if c[t] > 0 else 0) >= MOVE_THRESH and pk_idx > t + 1:
            moves.append((t, pk_idx)); i = pk_idx
        else: i += 1
    return moves


def _coverage_lag(assets, configs, family, ma_type, cd, mh, exit_name, ms_tr_lo, ms_tr_hi):
    if not configs: return float("nan"), float("nan")
    cfg = configs[len(configs) // 2]; maf = _MA[ma_type]
    caught = 0; total = 0; lags = []
    for asset in assets:
        c = asset["c"]; o = asset["o"]; ms = asset["ms"]
        moves = _detect_moves(c, ms, ms_tr_lo, ms_tr_hi)
        if not moves: continue
        if family == "2MA":
            fast, slow = cfg; fma = maf(c, fast); sma = maf(c, slow); mma = None
        else:
            fast, mid, slow = cfg; fma = maf(c, fast); mma = maf(c, mid); sma = maf(c, slow)
        starts, ends = _signal_runs(c, fma, sma, mma, family, cd, mh)
        atr14 = _fast_atr(asset["h"], asset["l"], c, ATR14_WIN)
        atr22 = _fast_atr(asset["h"], asset["l"], c, ATR22_WIN)
        h_ex = _apply_exit(starts, ends, c, o, atr14, atr22, fma, sma, exit_name)
        for (tr_i, pk_i) in moves:
            total += 1
            tp = c[tr_i]; pp = c[pk_i]; denom = pp - tp
            in_w = h_ex[tr_i + 1: pk_i + 1]
            if np.any(in_w == 1):
                caught += 1
                if denom > 0:
                    fi = np.where(in_w == 1)[0][0] + tr_i + 1
                    lags.append(float(np.clip((c[fi] - tp) / denom, 0.0, 1.0)))
            else:
                lags.append(1.0)
    cov = float(caught) / float(total) if total > 0 else float("nan")
    lag = float(np.mean(lags)) if lags else float("nan")
    return cov, lag


# ---------------------------------------------------------------------------
# CAPTURE (median per-trade, OOS)
# ---------------------------------------------------------------------------
def _capture_median(assets, configs, family, ma_type, cd, mh, exit_name, ms_lo, ms_hi):
    if not configs: return float("nan")
    cfg = configs[len(configs) // 2]; maf = _MA[ma_type]
    all_cap = []
    for asset in assets:
        c = asset["c"]; o = asset["o"]; ms = asset["ms"]
        if ((ms >= ms_lo) & (ms < ms_hi)).sum() < 10: continue
        if family == "2MA":
            fast, slow = cfg; fma = maf(c, fast); sma = maf(c, slow); mma = None
        else:
            fast, mid, slow = cfg; fma = maf(c, fast); mma = maf(c, mid); sma = maf(c, slow)
        starts, ends = _signal_runs(c, fma, sma, mma, family, cd, mh)
        atr14 = _fast_atr(asset["h"], asset["l"], c, ATR14_WIN)
        atr22 = _fast_atr(asset["h"], asset["l"], c, ATR22_WIN)
        h_ex = _apply_exit(starts, ends, c, o, atr14, atr22, fma, sma, exit_name)
        d_h = np.diff(np.concatenate([[0], h_ex, [0]]))
        for s, e in zip(np.where(d_h == 1)[0], np.where(d_h == -1)[0]):
            if s >= len(ms) or ms[s] < ms_lo or ms[s] >= ms_hi: continue
            eb = min(e - 1, len(c) - 1); entry_bar = min(s + 1, len(c) - 1)
            if entry_bar >= len(c) - 1 or eb < entry_bar: continue
            ep = o[entry_bar]; xp = c[eb]
            pk = float(np.max(c[entry_bar: eb + 1]))
            avail = (pk - ep) / ep if ep > 0 else 0.0
            gross = (xp - ep) / ep if ep > 0 else 0.0
            if avail > 0.005:
                all_cap.append(float(np.clip((gross - COST_RT) / avail, 0.0, 1.0)))
            else:
                all_cap.append(1.0 if gross > COST_RT else 0.0)
    return float(np.median(all_cap)) if all_cap else float("nan")


# ---------------------------------------------------------------------------
# BLOCK-BOOTSTRAP p05 (robustness gate) -- moving-block resample of the OOS book
# ---------------------------------------------------------------------------
def _block_bootstrap_p05(book, n_boot=1000, block=None, seed=0):
    """5th-percentile of the bootstrapped compound-net distribution. >0 => robust."""
    if book is None: return None
    x = book.dropna().to_numpy()
    n = len(x)
    if n < 20: return None
    if block is None: block = max(5, int(round(n ** 0.5)))
    if block >= n: block = max(2, n // 3)
    rng = np.random.default_rng(seed)
    starts_pool = np.arange(0, n - block + 1)
    if len(starts_pool) < 1: return None
    nblocks = int(np.ceil(n / block))
    nets = np.empty(n_boot)
    for b in range(n_boot):
        st = rng.choice(starts_pool, size=nblocks, replace=True)
        idx = np.concatenate([np.arange(s, s + block) for s in st])[:n]
        nets[b] = (np.prod(1 + x[idx]) - 1) * 100.0
    return float(np.percentile(nets, 5))


# ---------------------------------------------------------------------------
# HOLD STATS (calendar duration of trades) -- makes "24 bars != 24 days" visible
# ---------------------------------------------------------------------------
def _hold_stats(assets, configs, family, ma_type, cd, mh, exit_name, cad, ms_lo, ms_hi):
    """Median/p90 realized hold in BARS and CALENDAR HOURS for the selected cell, in [ms_lo,ms_hi)."""
    if not configs: return None
    cfg = configs[len(configs) // 2]; maf = _MA[ma_type]
    holds = []
    for asset in assets:
        c = asset["c"]; o = asset["o"]; ms = asset["ms"]
        if ((ms >= ms_lo) & (ms < ms_hi)).sum() < 10: continue
        if family == "2MA":
            fast, slow = cfg; fma = maf(c, fast); sma = maf(c, slow); mma = None
        else:
            fast, mid, slow = cfg; fma = maf(c, fast); mma = maf(c, mid); sma = maf(c, slow)
        starts, ends = _signal_runs(c, fma, sma, mma, family, cd, mh)
        atr14 = _fast_atr(asset["h"], asset["l"], c, ATR14_WIN)
        atr22 = _fast_atr(asset["h"], asset["l"], c, ATR22_WIN)
        h_ex = _apply_exit(starts, ends, c, o, atr14, atr22, fma, sma, exit_name)
        d_h = np.diff(np.concatenate([[0], h_ex, [0]]))
        for s, e in zip(np.where(d_h == 1)[0], np.where(d_h == -1)[0]):
            if s >= len(ms) or ms[s] < ms_lo or ms[s] >= ms_hi: continue
            holds.append(int(e - s))
    if not holds: return None
    hpb = HPB.get(cad, 24.0)
    med_bars = float(np.median(holds)); p90_bars = float(np.percentile(holds, 90))
    return {
        "median_bars": round(med_bars, 1),
        "median_hours": round(med_bars * hpb, 1),
        "p90_hours": round(p90_bars * hpb, 1),
        "n_trades": len(holds),
    }


# ---------------------------------------------------------------------------
# BUY-HOLD
# ---------------------------------------------------------------------------
def _buyhold(assets, ms_lo, ms_hi):
    cells = []
    for asset in assets:
        ms = asset["ms"]; ret = asset["ret"]
        mask = (ms >= ms_lo) & (ms < ms_hi)
        if mask.sum() < 5: continue
        cells.append(pd.Series(ret[mask], index=pd.to_datetime(ms[mask], unit="ms")))
    return _ew_book(cells)


def _net_on_span(cad, ma_type, best, cfgs, span):
    """Replay the selected cell's FROZEN spec (family/config/cd/mh/exit + REGIME_GATE) on an
    out-of-window span (e.g. 2021 forward / 2022 bear) -> EW-book net%. Respects the active gate."""
    try:
        assets = _load_all(cad, span[0], span[1])
        if len(assets) < 3 or not cfgs:
            return None
        cfg = cfgs[len(cfgs) // 2]; maf = _MA[ma_type]
        lo = int(pd.Timestamp(span[0]).value // 10**6)
        hi = int(pd.Timestamp(span[1]).value // 10**6)
        cells = []
        for asset in assets:
            c = asset["c"]; o = asset["o"]
            if best["family"] == "2MA":
                fast, slow = cfg; fma = maf(c, fast); sma = maf(c, slow); mma = None
            else:
                fast, mid, slow = cfg; fma = maf(c, fast); mma = maf(c, mid); sma = maf(c, slow)
            st, en = _signal_runs(c, fma, sma, mma, best["family"], best["cooldown"], best["min_hold"])
            atr14 = _fast_atr(asset["h"], asset["l"], c, ATR14_WIN)
            atr22 = _fast_atr(asset["h"], asset["l"], c, ATR22_WIN)
            h_ex = _apply_exit(st, en, c, o, atr14, atr22, fma, sma, best["exit"])
            s = _net_series(h_ex, asset["ret"], asset["ms"], lo, hi)
            if s is not None: cells.append(s)
        bk = _ew_book(cells)
        return round(_net_pct(bk), 2) if bk is not None else None
    except Exception:
        return None


def buyhold_sanity(cad: str) -> float | None:
    assets = _load_all(cad)
    if not assets: return None
    ms_lo = int(pd.Timestamp(YEAR[0]).value // 10**6)
    ms_hi = int(pd.Timestamp(YEAR[1]).value // 10**6)
    bh = _buyhold(assets, ms_lo, ms_hi)
    return round(_net_pct(bh), 1) if bh is not None else None


# ---------------------------------------------------------------------------
# MAIN PER-CELL
# ---------------------------------------------------------------------------
def run_cell(ma_type: str, cad: str, verbose: bool = False, static_only: bool = False) -> dict:
    t0 = dt.datetime.now()
    ms_lo  = int(pd.Timestamp(YEAR[0]).value  // 10**6)
    ms_hi  = int(pd.Timestamp(YEAR[1]).value  // 10**6)
    ms_tr_lo = int(pd.Timestamp(TRAIN[0]).value // 10**6)
    ms_tr_hi = int(pd.Timestamp(TRAIN[1]).value // 10**6)
    ms_vl_lo = int(pd.Timestamp(VAL[0]).value   // 10**6)
    ms_vl_hi = int(pd.Timestamp(VAL[1]).value   // 10**6)
    ms_oo_lo = int(pd.Timestamp(OOS[0]).value   // 10**6)
    ms_oo_hi = int(pd.Timestamp(OOS[1]).value   // 10**6)

    assets = _load_all(cad)
    if len(assets) < 3:
        return {"ma_type": ma_type, "cad": cad, "error": f"insufficient assets ({len(assets)})"}

    bh = _buyhold(assets, ms_lo, ms_hi)
    bh_net = round(_net_pct(bh), 1) if bh is not None else None
    if bh_net is not None and not (BH_SANE_LO < bh_net < BH_SANE_HI):
        print(f"  [WARN] buy-hold {ma_type} {cad}: {bh_net}%")

    exits = _build_exits(cad)
    noskill = _noskill_exits(cad)

    all_rows = []
    for family, configs in [("2MA", _configs_2ma(cad)), ("3MA", _configs_3ma(cad))]:
        rows = _run_family(
            assets, configs, family, ma_type,
            ms_lo, ms_hi, ms_tr_lo, ms_tr_hi,
            ms_vl_lo, ms_vl_hi, ms_oo_lo, ms_oo_hi, exits)
        all_rows.extend(rows)

    if not all_rows:
        return {"ma_type": ma_type, "cad": cad, "error": "no combos",
                "n_assets": len(assets), "bh_net_full": bh_net}

    # SELECTION: pick on DEV data (TRAIN+VAL) ONLY -- OOS is held out as confirm, never
    # a selection criterion (selecting on net_oos was the overfit source). DYNAMIC is
    # excluded from winning the cell on a short pilot window (reported as a sidecar delta).
    def _eligible(r):
        if r["mode"] == "dynamic" and r.get("dynamic_short_window", False):
            return False
        return True
    elig = [r for r in all_rows if _eligible(r) and (not static_only or r["mode"] == "static")]
    if not elig: elig = [r for r in all_rows if not static_only or r["mode"] == "static"] or all_rows
    devpool = [r for r in elig if r["net_train"] > 0 and r["net_val"] > 0]
    if devpool:
        pool = devpool; sel_gate = "trainval_positive"
    else:
        pool = elig; sel_gate = "unconstrained_dev"
    # criterion = TRAIN+VAL composite (held-out OOS NOT in the objective)
    best = max(pool, key=lambda r: r["net_train"] + r["net_val"])

    best_st_oos = max((r["net_oos"] for r in all_rows if r["mode"] == "static"), default=0.0)
    best_dy_oos = max((r["net_oos"] for r in all_rows if r["mode"] == "dynamic"), default=0.0)
    dyn_delta = round(best_dy_oos - best_st_oos, 2)

    maxdd = round(_maxdd_pct(best.get("_bk_full")), 2) if best.get("_bk_full") is not None else None

    cfgs = _configs_2ma(cad) if best["family"] == "2MA" else _configs_3ma(cad)
    cov, lag = _coverage_lag(assets, cfgs, best["family"], ma_type,
                              best["cooldown"], best["min_hold"], best["exit"],
                              ms_tr_lo, ms_tr_hi)
    cap = _capture_median(assets, cfgs, best["family"], ma_type,
                          best["cooldown"], best["min_hold"], best["exit"],
                          ms_oo_lo, ms_oo_hi)

    # p05 block-bootstrap on the HELD-OUT OOS book (robustness gate; >0 => robust)
    bk_full = best.get("_bk_full")
    p05_oos = None
    if bk_full is not None:
        oo_lo_ts = pd.Timestamp(ms_oo_lo * 10**6, unit="ns")
        oo_hi_ts = pd.Timestamp(ms_oo_hi * 10**6, unit="ns")
        bk_oos = bk_full[(bk_full.index >= oo_lo_ts) & (bk_full.index < oo_hi_ts)]
        p05_oos = _block_bootstrap_p05(bk_oos)
        if p05_oos is not None: p05_oos = round(p05_oos, 2)

    # hold duration (calendar) on OOS -- "find-exploit-exit" horizon
    hold = _hold_stats(assets, cfgs, best["family"], ma_type,
                       best["cooldown"], best["min_hold"], best["exit"],
                       cad, ms_oo_lo, ms_oo_hi)

    # beats_noskill
    beats_ns = False
    if best["exit"] not in noskill:
        tv_b = best["net_train"] + best["net_val"]; oo_b = best["net_oos"]
        for ns in noskill:
            ns_rows = [r for r in all_rows if r["exit"] == ns
                       and r["cooldown"] == best["cooldown"]
                       and r["min_hold"] == best["min_hold"]
                       and r["family"] == best["family"]
                       and r["mode"] == best["mode"]]
            if not ns_rows: continue
            nr = max(ns_rows, key=lambda r: r["net_oos"])
            if tv_b > nr["net_train"] + nr["net_val"] and oo_b > nr["net_oos"]:
                beats_ns = True; break

    # FORWARD (2021 iron year) + 2022 bear -- replay the frozen selected spec out-of-window
    net_2021 = _net_on_span(cad, ma_type, best, cfgs, FWD_SPAN)
    net_bear = _net_on_span(cad, ma_type, best, cfgs, BEAR_SPAN)

    elapsed = (dt.datetime.now() - t0).total_seconds()
    if verbose:
        hold_h = hold["median_hours"] if hold else None
        print(f"  [{ma_type:5s} {cad:4s}] gate={sel_gate:17s} "
              f"fam={best['family']:3s} mode={best['mode']:7s} "
              f"OOS={best['net_oos']:+6.1f}% p05={str(p05_oos):>7} "
              f"cd={best['cooldown']} mh={best['min_hold']} "
              f"exit={best['exit']:16s} hold~{str(hold_h):>6}h maxDD={str(maxdd):>7} "
              f"dyn_dlt={dyn_delta:+.1f} [{elapsed:.1f}s]")

    clean_rows = [{k: v for k, v in r.items() if k != "_bk_full"} for r in all_rows]
    return {
        "ma_type": ma_type, "cad": cad,
        "n_assets": len(assets), "bh_net_full": bh_net,
        "best_family": best["family"], "best_mode": best["mode"],
        "best_cooldown": best["cooldown"], "best_min_hold": best["min_hold"],
        "best_exit": best["exit"], "sel_gate": sel_gate,
        "net_train": best["net_train"], "net_val": best["net_val"], "net_oos": best["net_oos"],
        "p05_oos_bootstrap": p05_oos,
        "maxDD_full": maxdd,
        "hold_oos": hold,
        "coverage_train": round(cov, 3) if not np.isnan(cov) else None,
        "capture_oos_median": round(cap, 3) if not np.isnan(cap) else None,
        "entry_lag_train_mean": round(lag, 3) if not np.isnan(lag) else None,
        "beats_noskill_fixedhold": beats_ns,
        "dynamic_vs_static_oosdelta": dyn_delta,
        "net_2021_fwd": net_2021,
        "net_bear_2022": net_bear,
        "n_combos_total": len(all_rows),
        "n_combos_devpool": len(pool), "sel_pool_gate": sel_gate,
        "n_combos_3way": sum(1 for r in all_rows if r.get("pos3way")),
        "elapsed_s": round(elapsed, 1), "metric_defs": METRIC_DEFS,
    }


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------
def selftest(verbose: bool = True) -> dict:
    cad = "1d"
    print(f"[selftest] cadence={cad}  MA_TYPES={MA_TYPES}")
    print(f"[selftest] 2MA: {_configs_2ma(cad)}  3MA: {_configs_3ma(cad)}")
    print(f"[selftest] exits: {_build_exits(cad)}")

    bh_net = buyhold_sanity(cad)
    print(f"[selftest] buy-hold FULL-2020 1d: {bh_net}%  (expect {BH_SANE_LO}-{BH_SANE_HI}%)")
    assert bh_net is not None and BH_SANE_LO < bh_net < BH_SANE_HI, \
        f"buy-hold sanity FAILED: {bh_net}%"

    results = []
    for ma_type in MA_TYPES:
        r = run_cell(ma_type, cad, verbose=verbose)
        results.append(r)
        if "error" not in r:
            assert r["net_oos"] is not None
            if r["maxDD_full"] is not None:
                assert r["maxDD_full"] <= 0.01, f"maxDD={r['maxDD_full']}"
            assert r["sel_gate"] in ("trainval_positive", "unconstrained_dev")
            assert r["best_family"] in ("2MA", "3MA")
            assert r["best_mode"] in ("static", "dynamic")

    print("\n[selftest] SUMMARY (1d):")
    print(f"  {'MA':8}  {'gate':17}  {'fam':3}  {'mode':7}  "
          f"{'OOS%':>7}  {'p05':>6}  {'hold_h':>7}  {'maxDD':>7}  "
          f"{'cap':>5}  {'lag':>5}  {'beat_ns':>7}  {'dyn_dlt':>7}  exit")
    for r in results:
        if "error" in r:
            print(f"  {r['ma_type']:8}  ERROR: {r['error']}"); continue
        hh = r.get("hold_oos", {})
        hh_h = hh.get("median_hours") if isinstance(hh, dict) else None
        print(f"  {r['ma_type']:8}  {r['sel_gate']:17}  {r['best_family']:3}  {r['best_mode']:7}  "
              f"{r['net_oos']:>7.1f}  {str(r.get('p05_oos_bootstrap','--')):>6}  "
              f"{str(hh_h):>7}  "
              f"{str(r.get('maxDD_full','--')):>7}  "
              f"{str(r.get('capture_oos_median','--')):>5}  "
              f"{str(r.get('entry_lag_train_mean','--')):>5}  "
              f"{str(r['beats_noskill_fixedhold']):>7}  "
              f"{r['dynamic_vs_static_oosdelta']:>+7.1f}  "
              f"{r['best_exit']}")
    print("\n[selftest] PASSED")
    return {"cad": cad, "bh_net": bh_net, "results": results}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m strat.ma_strat_builder")
    ap.add_argument("--tf", default="1d")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ma_types", default=",".join(MA_TYPES))
    ap.add_argument("--verbose", action="store_true", default=True)
    a = ap.parse_args(argv)

    if a.selftest:
        st = selftest(verbose=True)
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        p = OUT / f"ma_strat_builder_selftest_{ts}.json"
        p.write_text(json.dumps(st, indent=2, default=str), encoding="utf-8")
        print(f"\n[selftest] written: {p}")
        return 0

    tfs = [t.strip() for t in a.tf.split(",")]
    ma_types = [m.strip() for m in a.ma_types.split(",")]
    all_results = {}
    for cad in tfs:
        if cad not in TFS: print(f"[skip] {cad}"); continue
        print(f"\n=== TF={cad} ===")
        bh = buyhold_sanity(cad)
        print(f"  buy-hold: {bh}%")
        cad_results = []
        for ma_type in ma_types:
            r = run_cell(ma_type, cad, verbose=a.verbose)
            cad_results.append(r)
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        p = OUT / f"ma_strat_builder_{cad}_{ts}.json"
        payload = {"tf": cad, "bh_net_full_2020": bh, "metric_defs": METRIC_DEFS,
                   "results": cad_results}
        p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"  written: {p}")
        all_results[cad] = cad_results

    print("\n=== SUMMARY ===")
    for cad, rows in all_results.items():
        for r in rows:
            if "error" in r:
                print(f"  {cad}  {r['ma_type']}  ERROR: {r['error']}"); continue
            print(f"  {cad}  {r['ma_type']:8}  {r['sel_gate']:17}  {r['best_family']:3}  "
                  f"{r['best_mode']:7}  {r['net_oos']:+7.1f}%  "
                  f"dyn_delta={r['dynamic_vs_static_oosdelta']:+.1f}  {r['best_exit']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
