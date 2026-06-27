"""TI x TIMEFRAME — full performance DNA report (NO exclusions) + end-to-end AUDIT.

The user's reframe (2026-06-24):
  1. NOT "best engine" — report EVERY TI type x Timeframe with its full performance DNA
     and profile, winners and bleeders alike. Nothing excluded at this stage.
  2. Two windows ONLY: VAL (2023-07-01..2024-05-15) and OOS-2024 (2024-05-15..2024-12-31).
     DO NOT TOUCH 2025 — the data wall is 2025-01-01 so only 2024 bars load for OOS.
  3. Hyper-audit: verify the whole pipeline end-to-end (causality/look-ahead, cost,
     rebalancing-premium, gate logic, random-null construction) so every number is robust.

Canonical splits (crypto/config/data_config.yaml):
  TRAIN  <= 2023-07-01
  VAL     2023-07-01 .. 2024-05-15   (DEV ends at the 2024-05-15 wall)
  OOS     2024-05-15 .. 2025-03-15   (we use ONLY the 2024 portion -> .. 2024-12-31)
  UNSEEN >= 2025-03-15               (never touched)

Families covered (ALL on one identical harness):
  MA-cross ensemble: EMA SMA WMA HMA DEMA TEMA KAMA VIDYA  (8)
  Non-MA SOTA:       RSI MACD BOLLINGER STOCHASTIC ADX SUPERTREND  (close/ohlc)
  Volume:            OBV VWAP CMF MFI FLOW                  (real chimera volume)
  Misc:              DONCHIAN CCI WILLR PSAR AROON VORTEX KELTNER

NOTE on MA: the VALIDATED MA result came from the dedicated DECOUPLED BAND engine
(ma_wealth_driver / V2), NOT from this simple MA-cross ensemble. The MA-cross here is a
COMPARABLE baseline for the DNA table; the canonical V2 band numbers are cited separately
in the registry. Labeled "MA-cross" to avoid conflating the two.

Honest accounting: DEV/VAL compounds are inflated by the EW-rebalancing premium
(quantified in the audit). Linear/wk edge vs always-long + random-null percentile +
single-asset BTC are the honest reads. BOTH linear and compound are reported.
"""
from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
import json
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRYPTO = ROOT.parent

import strat.fleet_lab as fl
from strat.ma_wealth_driver import _load_asset, _u100_symbols, MAKER_RT
from strat.ma_type_upgrade import _MA, held_cross
from strat.ti_engine_sota_v11 import load_asset_with_features
from strat.ti_engine_extra import (
    ensemble_held, ALL_CONFIGS, VOLUME_FAMILIES, MISC_FAMILIES, _ema,
)
from strat.ti_engine_sota import (
    rsi_single_held, macd_single_held, bollinger_single_held, stochastic_single_held,
    adx_single_held, supertrend_single_held,
    RSI_CONFIGS, MACD_CONFIGS, BOLLINGER_CONFIGS, STOCHASTIC_CONFIGS,
    ADX_CONFIGS, SUPERTREND_CONFIGS,
)

BARS_PER_DAY = fl.BARS_PER_DAY

# ---- WINDOWS (binding) ----
VAL_START = "2023-07-01"
VAL_END = "2024-05-15"
OOS_START = "2024-05-15"
OOS_2024_END = "2024-12-31"           # <<< 2025 NEVER touched
DEV_WALL_MS = int(pd.Timestamp(VAL_END).value // 10**6)        # loads <= 2024-05-15 for VAL
OOS_WALL_MS = int(pd.Timestamp("2025-01-01").value // 10**6)   # loads <= 2024-12-31 for OOS
OUTDIR = CRYPTO / "runs" / "engines" / "V5_dna_report"

# ---- MA-cross ensemble config grid (fast,slow) period pairs, per MA type ----
MA_CROSS_CONFIGS = [(5, 20), (8, 21), (10, 30), (12, 26), (20, 50), (9, 26), (7, 25), (15, 40)]
MA_TYPES = list(_MA.keys())   # EMA SMA WMA HMA DEMA TEMA KAMA VIDYA

NONMA_CLOSE = {
    "RSI": (RSI_CONFIGS, lambda c, cfg: rsi_single_held(c, *cfg)),
    "MACD": (MACD_CONFIGS, lambda c, cfg: macd_single_held(c, *cfg)),
    "BOLLINGER": (BOLLINGER_CONFIGS, lambda c, cfg: bollinger_single_held(c, *cfg)),
}
NONMA_OHLC = {
    "STOCHASTIC": (STOCHASTIC_CONFIGS,
                   lambda o, h, l, c, cfg: stochastic_single_held(c, h, l, *cfg)),
    "ADX": (ADX_CONFIGS, lambda o, h, l, c, cfg: adx_single_held(c, *cfg)),
    "SUPERTREND": (SUPERTREND_CONFIGS,
                   lambda o, h, l, c, cfg: supertrend_single_held(c, h, l, *cfg)),
}


# --------------------------------------------------------------------------- #
# Regime gate (V1 BTC SMA100)
# --------------------------------------------------------------------------- #
def btc_regime(wall_ms: int) -> pd.Series | None:
    rec = _load_asset("BTCUSDT", "1d", wall_ms)
    if not rec:
        return None
    d, _, _, _, cl = rec
    sma = pd.Series(cl, index=d).rolling(100).mean()
    s = pd.Series(0, index=d, dtype=int); st = 0
    for i in range(100, len(d)):
        if np.isnan(sma.iloc[i]):
            continue
        if st == 0 and cl[i] > sma.iloc[i] * 1.02:
            st = 1
        elif st == 1 and cl[i] < sma.iloc[i] * 0.98:
            st = 0
        s.iloc[i] = st
    return s


def causal_regime(wall_ms: int) -> "pd.Series | None":
    """De-leaked 1d BTC regime for GATING in replay() (see V1.1 align_htf_on bug class).

    btc_regime[D] is BTC's day-D 1d CLOSE regime (~23:59). replay()'s broadcast
    (normalize-to-day + ffill) makes a day-D bar read regime[D] => SAME-DAY LOOK-AHEAD.
    Causal fix: a bar on day D reads the last CLOSED 1d bar = regime[D-1]; we shift the
    daily series forward one full 1d bar before broadcast
    (see runs/engines/AUDIT_ORACLE_UPGRADE/leakcheck.json).
    """
    s = btc_regime(wall_ms)
    if s is None:
        return None
    return s.shift(1).fillna(0).astype(int)


# --------------------------------------------------------------------------- #
# Unified held builder for ANY family
# --------------------------------------------------------------------------- #
def compute_held(family: str, o, h, l, c, vote_threshold, feats=None):
    """Return int8 held ensemble for one asset, dispatching by family group."""
    n = len(c)
    if family in MA_TYPES:
        votes = np.zeros(n, dtype=float)
        for periods in MA_CROSS_CONFIGS:
            votes += held_cross(np.asarray(c, float), periods, family).astype(float)
        return ((votes / len(MA_CROSS_CONFIGS)) > vote_threshold).astype(np.int8)
    if family in NONMA_CLOSE:
        configs, fn = NONMA_CLOSE[family]
        votes = np.zeros(n, dtype=float); nv = 0
        for cfg in configs:
            try:
                votes += fn(np.asarray(c, float), cfg).astype(float); nv += 1
            except Exception:
                continue
        return ((votes / max(nv, 1)) > vote_threshold).astype(np.int8) if nv else np.zeros(n, np.int8)
    if family in NONMA_OHLC:
        configs, fn = NONMA_OHLC[family]
        votes = np.zeros(n, dtype=float); nv = 0
        for cfg in configs:
            try:
                votes += fn(o, h, l, c, cfg).astype(float); nv += 1
            except Exception:
                continue
        return ((votes / max(nv, 1)) > vote_threshold).astype(np.int8) if nv else np.zeros(n, np.int8)
    if family in ALL_CONFIGS:   # volume + misc
        vol = bv = sv = None
        if feats is not None:
            vol = feats.get("volume"); bv = feats.get("buy_vol"); sv = feats.get("sell_vol")
        return ensemble_held(family, ALL_CONFIGS[family], o, h, l, c, vote_threshold,
                             volume=vol, buy_vol=bv, sell_vol=sv)
    raise ValueError(f"unknown family {family}")


def build_panel(family: str, tf: str, wall_ms: int, vote_threshold: float):
    need_vol = family in VOLUME_FAMILIES
    close_cols, held_cols = {}, {}
    for sym in _u100_symbols():
        feats = None
        if need_vol:
            rec = load_asset_with_features(sym, tf, wall_ms, ["volume", "buy_vol", "sell_vol"])
            if rec is None:
                continue
            dates, o, h, l, c, feats = rec
        else:
            rec = _load_asset(sym, tf, wall_ms)
            if rec is None:
                continue
            dates, o, h, l, c = rec
        held = compute_held(family, o, h, l, c, vote_threshold, feats=feats)
        close_cols[sym] = pd.Series(c, index=dates)
        held_cols[sym] = pd.Series(held, index=dates)
    if len(close_cols) < 10:
        return None, None
    close_df = pd.DataFrame(close_cols).sort_index()
    held_df = pd.DataFrame(held_cols).reindex(close_df.index).fillna(0).astype(np.int8)
    return close_df, held_df


# --------------------------------------------------------------------------- #
# Vectorized replay (returns full diagnostic bundle)
# --------------------------------------------------------------------------- #
def replay(close_df, held_df, regime, window_start, window_end, lookahead=False):
    """EW long-only replay. CAUSAL: position during return [i-1,i] = held[i-1]
    (decision known at prior close), implemented as held.shift(1).
    lookahead=True is the AUDIT CHEAT: position during [i-1,i] = held[i] (peeks at the
    bar that closes the return). A causal harness MUST show the cheat inflating returns;
    we use the causal path everywhere else."""
    rets = close_df.pct_change().to_numpy()[1:]
    shift_n = 0 if lookahead else 1
    held_lag = held_df.shift(shift_n).fillna(0).to_numpy()[1:]
    turn = held_df.shift(max(shift_n - 1, 0)).diff().abs().fillna(0).to_numpy()[1:]
    dates = close_df.index[1:]

    if regime is not None:
        reg_vals = np.zeros(len(dates)); ridx = regime.index
        # vectorized: map each date to last regime <= its normalized day
        norm_days = pd.DatetimeIndex(dates).normalize()
        reg_aligned = regime.reindex(ridx.union(norm_days)).ffill().reindex(norm_days).fillna(0)
        reg_vals = reg_aligned.to_numpy().astype(float)
    else:
        reg_vals = np.ones(len(dates))

    mask = np.ones(len(dates), dtype=bool)
    if window_start is not None:
        mask &= dates >= pd.Timestamp(window_start)
    if window_end is not None:
        mask &= dates < pd.Timestamp(window_end)

    valid = np.isfinite(rets)
    rets_f = np.where(valid, rets, 0.0)
    n_held = (held_lag * valid).sum(axis=1)
    eng_gross = np.where(n_held > 0, (rets_f * held_lag * valid).sum(axis=1) / np.maximum(n_held, 1), 0.0)
    cost = np.where(n_held > 0, (turn * valid).sum(axis=1) * (MAKER_RT / 2.0) / np.maximum(n_held, 1), 0.0)
    eng = (eng_gross - cost) * reg_vals
    n_all = valid.sum(axis=1)
    al = np.where(n_all > 0, (rets_f * valid).sum(axis=1) / np.maximum(n_all, 1), 0.0)

    return {"eng": eng[mask], "al": al[mask], "eng_gross": (eng_gross * reg_vals)[mask],
            "cost": cost[mask], "deploy": float(np.mean(n_held[mask] > 0)) if mask.sum() else 0.0,
            "rets_f": rets_f, "valid": valid, "reg_vals": reg_vals, "mask": mask,
            "n_held": n_held, "turn_total": float((turn * valid).sum())}


def profile(panel, bpd):
    eng, al = panel["eng"], panel["al"]
    if len(eng) == 0:
        return None
    lin_wk_e = float(np.mean(eng)) * bpd * 7 * 100
    lin_wk_a = float(np.mean(al)) * bpd * 7 * 100
    comp_e = float(np.prod(1 + eng) - 1) * 100
    comp_a = float(np.prod(1 + al) - 1) * 100
    cum = np.cumprod(1 + eng)
    maxdd = float(np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum))) * 100 if len(cum) > 1 else 0.0
    vol = float(np.std(eng)) * np.sqrt(bpd * 365) * 100
    sharpe = (float(np.mean(eng)) * bpd * 365) / (float(np.std(eng)) * np.sqrt(bpd * 365)) if np.std(eng) > 1e-12 else 0.0
    # 7d non-overlapping slices
    slice_bars = 7 * bpd
    sl = [float(np.prod(1 + eng[i:i+slice_bars]) - 1) * 100
          for i in range(0, len(eng) - slice_bars + 1, slice_bars)]
    pct_prof = (sum(1 for s in sl if s > 0) / len(sl)) if sl else 0.0
    return {
        "lin_wk": round(lin_wk_e, 4), "al_lin_wk": round(lin_wk_a, 4),
        "edge_pp": round(lin_wk_e - lin_wk_a, 4),
        "compound": round(comp_e, 1), "al_compound": round(comp_a, 1),
        "comp_edge": round(comp_e - comp_a, 1),
        "maxdd": round(maxdd, 1), "ann_vol": round(vol, 1), "sharpe": round(sharpe, 2),
        "deploy": round(panel["deploy"], 3), "n_slices": len(sl),
        "pct_slices_prof": round(pct_prof, 3),
        "cost_drag_wk": round(float(np.mean(panel["cost"])) * bpd * 7 * 100, 4),
    }


def random_null(panel, window_mask_only, n_draws, deploy, seed0=0, metric="compound"):
    """N random ensembles at the same deploy under the same gate. Percentile of the
    engine vs the random distribution, on the SAME window. metric: compound or linear."""
    rets_f = panel["rets_f"]; valid = panel["valid"]; reg_vals = panel["reg_vals"]; m = panel["mask"]
    T, A = rets_f.shape
    vals = []
    for s in range(n_draws):
        rng = np.random.default_rng(seed0 + s)
        rh = (rng.random((T, A)) < deploy).astype(np.int8)
        nh = (rh * valid).sum(axis=1)
        r = np.where(nh > 0, (rets_f * rh * valid).sum(axis=1) / np.maximum(nh, 1), 0.0) * reg_vals
        r = r[m]
        vals.append(float(np.prod(1 + r) - 1) * 100 if metric == "compound" else float(np.sum(r)) * 100)
    return np.array(vals)


# --------------------------------------------------------------------------- #
# DNA report
# --------------------------------------------------------------------------- #
ALL_FAMILIES = (MA_TYPES + list(NONMA_CLOSE) + list(NONMA_OHLC)
                + ["OBV", "VWAP", "CMF", "MFI", "FLOW"]
                + ["DONCHIAN", "CCI", "WILLR", "PSAR", "AROON", "VORTEX", "KELTNER"])


def run_dna(families, tfs, vote_threshold=0.5, n_null=200):
    reg_dev = causal_regime(DEV_WALL_MS)   # de-leaked: bar on day D reads regime[D-1]
    reg_oos = causal_regime(OOS_WALL_MS)   # de-leaked: bar on day D reads regime[D-1]
    rows = []
    hdr = (f"  {'family':<10} {'tf':<4} | {'VAL edge':>8} {'VALnull':>7} {'VALdep':>6} "
           f"{'VALshrp':>7} | {'OOSedge':>8} {'OOSnull':>7} {'OOSdep':>6} {'OOSshrp':>7} | "
           f"{'VALcomp':>9} {'OOScomp':>8}")
    for family in families:
        print(f"\n{'='*120}\n  {family}   [vote>{vote_threshold:.0%}, null={n_null}]\n{'='*120}")
        print(hdr); print("  " + "-" * 116)
        for tf in tfs:
            bpd = BARS_PER_DAY[tf]
            # VAL (load DEV wall, slice VAL window)
            cdf, hdf = build_panel(family, tf, DEV_WALL_MS, vote_threshold)
            if cdf is None:
                print(f"  {family:<10} {tf:<4} | INSUFFICIENT_ASSETS")
                continue
            pval = replay(cdf, hdf, reg_dev, VAL_START, VAL_END)
            mval = profile(pval, bpd)
            vnull = (float(np.mean(random_null(pval, True, n_null, pval["deploy"]) < mval["compound"])) * 100
                     if mval and pval["deploy"] > 1e-3 else 0.0)
            # OOS-2024 (load 2025 wall -> only 2024 bars, slice OOS window to 2024-12-31)
            cdfo, hdfo = build_panel(family, tf, OOS_WALL_MS, vote_threshold)
            poos = replay(cdfo, hdfo, reg_oos, OOS_START, OOS_2024_END)
            moos = profile(poos, bpd)
            onull = (float(np.mean(random_null(poos, True, n_null, poos["deploy"]) < moos["compound"])) * 100
                     if moos and poos["deploy"] > 1e-3 else 0.0)
            row = {"family": family, "tf": tf, "vote_threshold": vote_threshold,
                   "VAL": mval, "VAL_null_pct": round(vnull, 1),
                   "OOS2024": moos, "OOS_null_pct": round(onull, 1)}
            rows.append(row)
            if mval and moos:
                print(f"  {family:<10} {tf:<4} | {mval['edge_pp']:>+7.3f} {vnull:>6.0f} "
                      f"{mval['deploy']:>5.1%} {mval['sharpe']:>7.2f} | "
                      f"{moos['edge_pp']:>+7.3f} {onull:>6.0f} {moos['deploy']:>5.1%} {moos['sharpe']:>7.2f} | "
                      f"{mval['compound']:>+8.0f}% {moos['compound']:>+7.0f}%")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="all")
    ap.add_argument("--tfs", default="1d,4h,2h,1h,30m")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--null", type=int, default=200)
    a = ap.parse_args()
    if a.families == "all":
        fams = ALL_FAMILIES
    elif a.families == "ma":
        fams = MA_TYPES
    elif a.families == "volume":
        fams = list(VOLUME_FAMILIES)
    elif a.families == "misc":
        fams = list(MISC_FAMILIES)
    else:
        fams = [f.strip().upper() for f in a.families.split(",")]
    tfs = tuple(t.strip() for t in a.tfs.split(","))

    print("=" * 120)
    print("TI x TIMEFRAME — FULL PERFORMANCE DNA (no exclusions)")
    print("=" * 120)
    print(f"VAL  {VAL_START}..{VAL_END}   |   OOS-2024 {OOS_START}..{OOS_2024_END}   (2025 SEALED)")
    print(f"gate=BTC-SMA100 | cost={MAKER_RT} RT maker | EW u100 | null draws={a.null}")
    print(f"edge=linear %/wk vs always-long ; null=engine compound percentile vs random@same-deploy")

    t0 = time.time()
    rows = run_dna(fams, tfs, vote_threshold=a.threshold, n_null=a.null)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTDIR / f"ti_dna_{ts}.json"
    with open(out, "w") as f:
        json.dump({"windows": {"VAL": [VAL_START, VAL_END], "OOS2024": [OOS_START, OOS_2024_END]},
                   "cost_rt": MAKER_RT, "threshold": a.threshold, "n_null": a.null,
                   "rows": rows, "timestamp": ts}, f, indent=2, default=str)
    print(f"\n  Saved: {out}  ({time.time()-t0:.0f}s, {len(rows)} cells)")


if __name__ == "__main__":
    main()
