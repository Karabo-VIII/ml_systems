"""hma_profile_builder.py -- FOCUSED HMA-ONLY all-TF profiler.

Builds the optimised HMA profile across all 6 TFs (1d,4h,2h,1h,30m,15m).
REUSES: ma_2020_config_leaderboard (band members), working_band_rolling logic
        (all-weather 2020/2021/2022), ma_movecatch_decomp (move-catch profile),
        data_expansion.block_bootstrap_distribution (OOS p05).

METHODOLOGY (aligned with brief):
  - Band = configs positive across TRAIN+VAL+OOS within 2020 (from existing leaderboard JSON).
  - OOS net = within-2020 OOS split (2020-10-01..2021-01-01), using the fixed-EW ensemble.
  - All-weather = 2020 bull / 2021 mixed / 2022 bear via rolling-from-band ensemble on SPAN.
  - Move-catch: reuses existing results for 1d/4h/2h/1h/30m, runs 15m fresh.
  - Block-bootstrap p05 on OOS.
  - Turnover = avg flips per RT period of the ensemble.

RWYB: python -m strat.hma_profile_builder
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

import strat.portfolio_replay as PR
from strat.portfolio_replay import apply_trail_stop, MAKER_RT
from strat.replay_distinct_grid import distinct_specs
from strat.ma_type_upgrade import _nums, _MA
from strat.ma_2020_breakdown import _panel, SPLIT, YEAR, WARMUP
from strat.ma_2020_config_leaderboard import (
    build_panels, config_book, _asset_close, _metrics,
    SPLITS, SYMS, ANN, TRAIL, MINHOLD,
)
from strat.structural_fixes import min_hold
from strat.data_expansion import block_bootstrap_distribution

MA_TYPE = "HMA"
TFS = ["1d", "4h", "2h", "1h", "30m", "15m"]

# All-weather span
SPAN = ("2020-01-01", "2023-01-01")
YEARS = {
    "2020_bull": ("2020-01-01", "2021-01-01"),
    "2021_mixed": ("2021-01-01", "2022-01-01"),
    "2022_bear": ("2022-01-01", "2023-01-01"),
}
LOOKBACK_D = 120
STEP_D = 30

OUT = ROOT.parent / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)

LB_PATH = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE" / "config_leaderboard.json"
MOVECATCH_DIR = ROOT.parent / "runs" / "strat"


def _net(s):
    s = s.dropna()
    return float(np.prod(1 + s.to_numpy()) - 1) * 100 if len(s) > 1 else 0.0


def _maxdd(s):
    s = s.dropna()
    if len(s) < 2:
        return 0.0
    eq = np.cumprod(1 + s.to_numpy())
    pk = np.maximum.accumulate(eq)
    return float(((eq - pk) / pk).min() * 100)


# ===========================================================================
# Load the existing config leaderboard + extract HMA band configs per TF
# ===========================================================================
def load_band_configs(tf):
    """Return list of period-tuples for the HMA working band at `tf`. [] if missing."""
    if not LB_PATH.exists():
        print(f"  WARNING: leaderboard not found at {LB_PATH}")
        return []
    with open(LB_PATH) as f:
        lb = json.load(f)
    grid = lb.get("grid", {})
    key = f"{MA_TYPE}|{tf}"
    cell = grid.get(key)
    if not cell:
        print(f"  WARNING: no leaderboard cell for {key}")
        return []
    band_cfgs = cell["band"]["band_configs"]
    return [_nums(c) for c in band_cfgs if len(_nums(c)) >= 2]


def band_size(tf):
    if not LB_PATH.exists():
        return 0
    with open(LB_PATH) as f:
        lb = json.load(f)
    cell = lb["grid"].get(f"{MA_TYPE}|{tf}", {})
    if not cell:
        return 0
    b = cell["band"]
    return b["n_band_2ma"] + b["n_band_3ma"]


# ===========================================================================
# OOS metrics from the leaderboard (within-2020: 2020-10-01..2021-01-01)
# ===========================================================================
def leaderboard_oos_net(tf):
    """Load the band-ensemble OOS net from the existing leaderboard cell (if available)."""
    if not LB_PATH.exists():
        return None, None
    with open(LB_PATH) as f:
        lb = json.load(f)
    cell = lb["grid"].get(f"{MA_TYPE}|{tf}", {})
    if not cell or not cell["ranked"]:
        return None, None
    # Rebuild the ensemble OOS metric: average of band-member OOS nets
    band_cfgs = cell["band"]["band_configs"]
    if not band_cfgs:
        return None, None
    oos_nets = []
    for r in cell["ranked"]:
        if r["config"] in band_cfgs and r["OOS"]["net"] is not None:
            oos_nets.append(r["OOS"]["net"])
    if not oos_nets:
        return None, None
    return float(np.mean(oos_nets)), float(np.std(oos_nets))


# ===========================================================================
# Build the per-bar ensemble net stream for a TF over a given date range.
# Uses the IRONED SLEEVE (trail+minhold+lag1+maker) with fixed-EW u10.
# This is the EXACT same sleeve as the leaderboard / config_book.
# ===========================================================================
def build_ensemble_stream(band_members, tf, lo, hi):
    """Build the EW band-ensemble bar-net Series over [lo, hi).
    band_members: list of period-tuples.
    Returns pd.Series or None."""
    if not band_members:
        return None

    # We need the full close arrays over [lo-WARMUP .. hi). We use _asset_close but
    # that is locked to 2020; for all-weather we need the raw _panel from ma_2020_breakdown.
    maf = _MA[MA_TYPE]
    all_periods = sorted({p for m in band_members for p in m})
    member_nets = []

    for periods in band_members:
        cells = []
        for sym in SYMS:
            try:
                o, h, l, c, ms = _panel(sym, tf)
            except Exception:
                continue
            s_ms = pd.Timestamp(lo).value // 10**6
            e_ms = pd.Timestamp(hi).value // 10**6
            # Include warmup bars before lo for MA computation
            s_idx = int(np.searchsorted(ms, s_ms))
            s_idx = max(0, s_idx - WARMUP)
            e_idx = int(np.searchsorted(ms, e_ms))
            if e_idx - s_idx < 40:
                continue
            c2 = c[s_idx:e_idx]
            ms2 = ms[s_idx:e_idx]
            win = ms2 >= s_ms
            if win.sum() < 10:
                continue
            ret = np.zeros(len(c2))
            ret[1:] = c2[1:] / c2[:-1] - 1.0

            # Build MA cross signal
            mas = [maf(c2, p) for p in periods]
            if len(periods) == 2:
                h0 = (mas[0] > mas[1]).astype(np.int8)
            else:
                h0 = ((mas[0] > mas[1]) & (mas[1] > mas[2])).astype(np.int8)
            h0 = np.nan_to_num(h0).astype(np.int8)

            # Iron: trail + min_hold + lag1
            h1 = apply_trail_stop(h0.copy(), c2, TRAIL)[0].astype(np.int8)
            h2 = min_hold(h1, MINHOLD).astype(np.int8)
            pos = np.zeros(len(c2))
            pos[1:] = h2[:-1]

            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            net = (pos * ret - flips * (MAKER_RT / 2.0))[win]
            cells.append(pd.Series(net, index=pd.to_datetime(ms2[win], unit="ms")))

        if not cells:
            continue
        book = pd.concat(cells, axis=1).fillna(0.0).mean(axis=1).sort_index()
        member_nets.append(book)

    if not member_nets:
        return None

    # Fixed-EW ensemble (fillna(0.0) = missing slot is cash, not reweighted)
    ensemble = pd.concat(member_nets, axis=1).fillna(0.0).mean(axis=1).sort_index()
    return ensemble


def _per_year_net(daily_stream):
    """Compute per-year net and maxDD from a (possibly sub-daily) bar stream."""
    out = {}
    for yk, (lo, hi) in YEARS.items():
        s = daily_stream[(daily_stream.index >= pd.Timestamp(lo)) &
                         (daily_stream.index < pd.Timestamp(hi))]
        out[yk] = {"net": round(_net(s), 1), "maxdd": round(_maxdd(s), 1), "n": int(len(s.dropna()))}
    return out


# ===========================================================================
# OOS band-ensemble net (within-2020: 2020-10-01..2021-01-01) via config_book
# We rebuild this for each TF to get the exact number.
# ===========================================================================
def compute_oos_ensemble(band_members, tf):
    """Compute OOS ensemble net using the same config_book as the leaderboard.
    This is the CANONICAL OOS metric aligned to the leaderboard methodology."""
    if not band_members:
        return None, None, None

    # Build panels over 2020 (the leaderboard window)
    all_periods = sorted({p for m in band_members for p in m})
    panels = build_panels(tf, MA_TYPE, all_periods)
    if not panels:
        return None, None, None

    member_nets = []
    for periods in band_members:
        bk = config_book(panels, periods)
        if bk is None:
            continue
        member_nets.append(bk)

    if not member_nets:
        return None, None, None

    ensemble = pd.concat(member_nets, axis=1).fillna(0.0).mean(axis=1).sort_index()

    # OOS split
    oos_lo, oos_hi = SPLITS["OOS"]
    oos_s = ensemble[(ensemble.index >= pd.Timestamp(oos_lo)) &
                     (ensemble.index < pd.Timestamp(oos_hi))]
    if len(oos_s) < 5:
        return None, None, None

    oos_net = round(_net(oos_s), 1)
    oos_dd = round(_maxdd(oos_s), 1)

    # Block-bootstrap p05 on OOS
    block_sz = max(3, len(oos_s) // 20)
    bbd_result = block_bootstrap_distribution(oos_s.to_numpy(), n_boot=2000, block=block_sz, stat="mean")
    p05 = round(float(bbd_result["p05"]) * 100, 2)

    # Coverage: fraction of time in-position (time-in)
    # Also compute TRAIN+VAL ensemble net for reference
    tv_lo, tv_hi = SPLITS["TRAIN"][0], SPLITS["VAL"][1]
    tv_s = ensemble[(ensemble.index >= pd.Timestamp(tv_lo)) &
                    (ensemble.index < pd.Timestamp(tv_hi))]
    tv_net = round(_net(tv_s), 1)

    # Turnover: avg round-trips per trading period
    # (estimate from one representative asset's position array)
    turnover_rt = _compute_turnover(band_members, tf)

    # Coverage: fraction of bars where the ensemble position > 0.5 (in-position)
    coverage = _compute_coverage(band_members, tf)

    return oos_net, oos_dd, p05, tv_net, turnover_rt, coverage, ensemble


def _compute_turnover(band_members, tf):
    """Rough estimate: average round-trips per 1000 bars (from BTC)."""
    try:
        o, h, l, c, ms = _panel("BTCUSDT", tf)
        s_ms = pd.Timestamp("2020-01-01").value // 10**6
        e_ms = pd.Timestamp("2021-01-01").value // 10**6
        s_idx = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
        e_idx = int(np.searchsorted(ms, e_ms))
        c2 = c[s_idx:e_idx]
        if len(c2) < 40:
            return None

        maf = _MA[MA_TYPE]
        positions = []
        for periods in band_members[:10]:  # sample up to 10 members
            mas = [maf(c2, p) for p in periods]
            if len(periods) == 2:
                h0 = (mas[0] > mas[1]).astype(np.int8)
            else:
                h0 = ((mas[0] > mas[1]) & (mas[1] > mas[2])).astype(np.int8)
            h0 = np.nan_to_num(h0).astype(np.int8)
            h1 = apply_trail_stop(h0.copy(), c2, TRAIL)[0].astype(np.int8)
            h2 = min_hold(h1, MINHOLD).astype(np.int8)
            pos = np.zeros(len(c2))
            pos[1:] = h2[:-1]
            positions.append(pos)

        if not positions:
            return None

        ensemble_pos = np.mean(np.stack(positions, axis=0), axis=0)
        # Count transitions (EW ensemble transitions > 0.1 threshold)
        transitions = np.sum(np.abs(np.diff(ensemble_pos)) > 0.1)
        n_bars = len(ensemble_pos)
        # RT per 100 bars
        return round(float(transitions / n_bars * 100), 2)
    except Exception as e:
        return None


def _compute_coverage(band_members, tf):
    """Fraction of bars the ensemble is in-position (>0.5 threshold)."""
    try:
        o, h, l, c, ms = _panel("BTCUSDT", tf)
        s_ms = pd.Timestamp("2020-01-01").value // 10**6
        e_ms = pd.Timestamp("2021-01-01").value // 10**6
        s_idx = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
        e_idx = int(np.searchsorted(ms, e_ms))
        c2 = c[s_idx:e_idx]
        ms2 = ms[s_idx:e_idx]
        win = ms2 >= s_ms
        if win.sum() < 10:
            return None

        maf = _MA[MA_TYPE]
        positions = []
        for periods in band_members[:20]:
            mas = [maf(c2, p) for p in periods]
            if len(periods) == 2:
                h0 = (mas[0] > mas[1]).astype(np.int8)
            else:
                h0 = ((mas[0] > mas[1]) & (mas[1] > mas[2])).astype(np.int8)
            h0 = np.nan_to_num(h0).astype(np.int8)
            h1 = apply_trail_stop(h0.copy(), c2, TRAIL)[0].astype(np.int8)
            h2 = min_hold(h1, MINHOLD).astype(np.int8)
            pos = np.zeros(len(c2))
            pos[1:] = h2[:-1]
            positions.append(pos[win].astype(float))

        if not positions:
            return None

        ens = np.mean(np.stack(positions, axis=0), axis=0)
        return round(float(np.mean(ens > 0.5)), 3)
    except Exception:
        return None


# ===========================================================================
# All-weather: build band-ensemble stream over 2020-2022 for a TF.
# Uses the same logic as working_band_rolling (band-ensemble mode).
# ===========================================================================
def compute_allweather(band_members, tf):
    """Compute all-weather per-year net for the fixed HMA band ensemble.
    Runs the ensemble over SPAN=(2020-2023) and slices per year.
    Returns dict or None."""
    print(f"  [allweather] Building HMA ensemble for {tf} over {SPAN}...")
    stream = build_ensemble_stream(band_members, tf, SPAN[0], SPAN[1])
    if stream is None or len(stream) < 50:
        print(f"  [allweather] WARN: stream too short for {tf}")
        return None
    return _per_year_net(stream)


# ===========================================================================
# Move-catch: load existing results or compute for 15m
# ===========================================================================
def load_movecatch_hma(tf):
    """Load HMA movecatch metrics for a TF. Returns dict or None."""
    path = MOVECATCH_DIR / f"ma_movecatch_{tf}.json"
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    results = d.get("results", {})
    # Primary threshold 0.05
    hma_05 = results.get("0.05", {}).get(MA_TYPE)
    hma_10 = results.get("0.1", {}).get(MA_TYPE)
    hma_15 = results.get("0.15", {}).get(MA_TYPE)
    return {"0.05": hma_05, "0.10": hma_10, "0.15": hma_15}


def run_15m_movecatch():
    """Run the 15m movecatch for HMA by calling ma_movecatch_decomp."""
    import subprocess
    print("  Running 15m movecatch for HMA...")
    cmd = [
        sys.executable, "-m", "strat.ma_movecatch_decomp",
        "--tf", "15m",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=str(ROOT))
    if result.returncode != 0:
        print(f"  15m movecatch FAILED:\n{result.stderr[-2000:]}")
        return None
    print("  15m movecatch done.")
    return load_movecatch_hma("15m")


# ===========================================================================
# STYLE TAG per TF
# ===========================================================================
STYLE_TAG = {
    "1d": "long-term",
    "4h": "swing",
    "2h": "swing",
    "1h": "intraday",
    "30m": "intraday",
    "15m": "intraday",
}


# ===========================================================================
# MAIN: build the HMA profile
# ===========================================================================
def main():
    profile = {}

    for tf in TFS:
        print(f"\n=== HMA @ {tf} ===")

        # 1. Band configs from leaderboard
        members = load_band_configs(tf)
        n_band = band_size(tf)
        print(f"  Band size: {n_band} configs ({len(members)} loaded)")

        if not members:
            print(f"  SKIP: no band members for {tf}")
            continue

        # 2. OOS metrics (within-2020)
        print(f"  Computing OOS ensemble metrics...")
        result = compute_oos_ensemble(members, tf)
        if result[0] is None:
            print(f"  WARN: OOS ensemble failed for {tf}")
            oos_net = None
            oos_dd = None
            p05 = None
            tv_net = None
            turnover_rt = None
            coverage = None
            ensemble_stream = None
        else:
            oos_net, oos_dd, p05, tv_net, turnover_rt, coverage, ensemble_stream = result
            print(f"  OOS net: {oos_net:+.1f}%, OOS maxDD: {oos_dd:.1f}%, p05: {p05:+.2f}%")
            print(f"  Turnover: {turnover_rt} RT/100bars, Coverage: {coverage:.2%}")

        # 3. All-weather (2020/2021/2022) -- rebuild over SPAN
        allweather = compute_allweather(members, tf)
        if allweather:
            for yk, v in allweather.items():
                print(f"  {yk}: net={v['net']:+.1f}%, maxdd={v['maxdd']:.1f}%")

        # 4. Move-catch
        mc = load_movecatch_hma(tf)
        if mc is None and tf == "15m":
            mc = run_15m_movecatch()

        mc05 = mc.get("0.05") if mc else None
        mc15 = mc.get("0.15") if mc else None

        if mc05:
            print(f"  Move-catch (5%): coverage={mc05.get('coverage'):.2f}, "
                  f"entry_lag={mc05.get('mean_entry_lag'):.3f}, "
                  f"weighted_capture={mc05.get('weighted_capture_mean'):.3f}")

        profile[tf] = {
            "tf": tf,
            "style": STYLE_TAG[tf],
            "n_band": n_band,
            "oos_net": oos_net,
            "oos_maxdd": oos_dd,
            "p05_bootstrap": p05,
            "tv_net": tv_net,
            "turnover_rt_per_100bars": turnover_rt,
            "coverage": coverage,
            "allweather": allweather,
            "movecatch_05": mc05,
            "movecatch_15": mc15,
        }

    # Save incremental output
    out_path = OUT / "hma_profile_all_tfs.json"
    with open(out_path, "w") as f:
        json.dump(profile, f, indent=2, default=str)
    print(f"\nProfile saved to {out_path}")

    # Print summary table
    print("\n=== HMA PROFILE SUMMARY ===")
    print(f"{'TF':6} {'Style':12} {'Band':5} {'OOS%':8} {'p05':8} {'Cov':6} {'MaxDD':7} "
          f"{'2020%':8} {'2021%':8} {'2022%':8} {'EntLag':7} {'WCap':6}")
    print("-" * 95)
    for tf in TFS:
        p = profile.get(tf, {})
        if not p:
            continue
        aw = p.get("allweather") or {}
        mc05 = p.get("movecatch_05") or {}
        print(f"{tf:6} {p.get('style',''):12} "
              f"{p.get('n_band',0):5} "
              f"{(str(round(p['oos_net'],1))+('%') if p.get('oos_net') is not None else 'N/A'):8} "
              f"{(str(p['p05_bootstrap'])+'%' if p.get('p05_bootstrap') is not None else 'N/A'):8} "
              f"{(str(round(p['coverage'],2)) if p.get('coverage') is not None else 'N/A'):6} "
              f"{(str(round(p['oos_maxdd'],1))+'%' if p.get('oos_maxdd') is not None else 'N/A'):7} "
              f"{(str(aw.get('2020_bull',{}).get('net','N/A'))+'%'):8} "
              f"{(str(aw.get('2021_mixed',{}).get('net','N/A'))+'%'):8} "
              f"{(str(aw.get('2022_bear',{}).get('net','N/A'))+'%'):8} "
              f"{(str(round(mc05.get('mean_entry_lag',0),3)) if mc05 else 'N/A'):7} "
              f"{(str(round(mc05.get('weighted_capture_mean',0),3)) if mc05 else 'N/A'):6}")

    return profile


if __name__ == "__main__":
    main()
