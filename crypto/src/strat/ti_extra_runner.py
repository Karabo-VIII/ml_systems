"""Gauntlet runner for the VOLUME + MISC TI families (ti_engine_extra.py).

Runs the SAME validation rigor the validated families passed, but VECTORIZED end
to end (pivot-matrix portfolio replay) so a full family x TF sweep finishes in
seconds, not the ~400s the per-bar dict loop took.

The gauntlet (per family x TF):
  1. DEV sweep   — 2020-01-01 .. 2024-05-15, ensemble + regime gate + maker cost
  2. OOS test    — 2024-05-15 .. 2025-01-01, SAME frozen configs (held-out)
  3. Consistency — DEV edge>0 AND OOS edge>0 vs always-long (selectable pre-OOS)
  4. Random-null — engine vs N random ensembles at the SAME deploy rate under the
                   SAME gate (isolates TI timing from being-long-under-the-gate)
  5. Selectivity — deploy < 70% (not always-long-with-tilt)

VALIDATED bar (the MA-family standard): DEV/OOS consistent AND random-null >= 95th pct
AND selective. Reports BOTH linear/wk (honest) and compound (inflated ~9.5x, context).
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
from strat.ti_engine_sota_v11 import load_asset_with_features
from strat.ti_engine_extra import (
    ensemble_held, ALL_CONFIGS, VOLUME_FAMILIES, MISC_FAMILIES,
)
from strat.ma_type_upgrade import _ema

BARS_PER_DAY = fl.BARS_PER_DAY
DEV_START = "2020-01-01"
DEV_END = "2024-05-15"
OOS_START = "2024-05-15"
OOS_END = "2025-01-01"
DEV_WALL_MS = int(pd.Timestamp(DEV_END).value // 10**6)
OOS_WALL_MS = int(pd.Timestamp(OOS_END).value // 10**6)
VOL_COLS = ["volume", "buy_vol", "sell_vol"]
OUTDIR = CRYPTO / "runs" / "engines" / "V4_ti_extra"


# --------------------------------------------------------------------------- #
# Regime gate (V1 BTC SMA100) — shared
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


# --------------------------------------------------------------------------- #
# Panel build: holds + close matrices for one family x TF (causal, walled)
# --------------------------------------------------------------------------- #
def build_panel(family: str, tf: str, wall_ms: int, vote_threshold: float,
                confirm: bool = False, lookback_days: int = 7, slice_days: int = 7):
    """Return (close_df, held_df) pivoted [dates x assets]. Holds computed on FULL
    causal history up to wall_ms. Volume families load real volume columns.

    confirm=True applies the MA-style decoupled confirmation + slice-hold lever:
    raw signal must ALSO have close>EMA(lookback) AND EMA rising, then holds for a
    fixed slice. This is the selectivity lever the MA family won with."""
    need_vol = family in VOLUME_FAMILIES
    configs = ALL_CONFIGS[family]
    bpd = BARS_PER_DAY[tf]
    lookback_bars = lookback_days * bpd
    slice_bars = slice_days * bpd
    close_cols, held_cols = {}, {}
    for sym in _u100_symbols():
        if need_vol:
            rec = load_asset_with_features(sym, tf, wall_ms, VOL_COLS)
            if rec is None:
                continue
            dates, o, h, l, c, feats = rec
            vol = feats.get("volume"); bv = feats.get("buy_vol"); sv = feats.get("sell_vol")
            held = ensemble_held(family, configs, o, h, l, c, vote_threshold,
                                 volume=vol, buy_vol=bv, sell_vol=sv)
        else:
            rec = _load_asset(sym, tf, wall_ms)
            if rec is None:
                continue
            dates, o, h, l, c = rec
            held = ensemble_held(family, configs, o, h, l, c, vote_threshold)
        if confirm:
            held = _apply_confirmation(np.asarray(c, float), held, lookback_bars, slice_bars)
        close_cols[sym] = pd.Series(c, index=dates)
        held_cols[sym] = pd.Series(held, index=dates)
    if len(close_cols) < 10:
        return None, None
    close_df = pd.DataFrame(close_cols).sort_index()
    held_df = pd.DataFrame(held_cols).reindex(close_df.index).fillna(0).astype(np.int8)
    return close_df, held_df


def _apply_confirmation(close, raw_held, lookback_bars, slice_bars):
    """MA-style decoupled confirmation + slice-hold (the architecture MA won with)."""
    fin = np.asarray(close, dtype=float)
    n = len(fin)
    m = np.isfinite(fin)
    out = np.zeros(n, dtype=np.int8)
    if m.sum() < max(5, lookback_bars // 4):
        return out
    cseg = fin[m]
    ema_lb = _ema(cseg, lookback_bars)
    above = cseg > ema_lb
    rising = np.concatenate([[True], ema_lb[1:] >= ema_lb[:-1]])
    rh = np.asarray(raw_held, dtype=bool)[m]
    confirmed = rh & above & rising
    held_seg = np.zeros(len(cseg), dtype=np.int8)
    i = 0; L = len(cseg)
    while i < L:
        if confirmed[i]:
            end = min(i + slice_bars, L)
            j = i
            while j < end and confirmed[j]:
                held_seg[j] = 1
                j += 1
            i = j
        else:
            i += 1
    out[np.where(m)[0]] = held_seg
    return out


# --------------------------------------------------------------------------- #
# Vectorized portfolio replay with regime gate + maker cost
# --------------------------------------------------------------------------- #
def replay(close_df, held_df, regime, tf, window_start=None, window_end=None):
    """Vectorized EW long-only replay. held lags 1 bar (causal). Cost on turnover.
    Returns dict of engine/AL arrays + deploy."""
    rets = close_df.pct_change().to_numpy()[1:]          # [T-1 x A]
    held_lag = held_df.shift(1).fillna(0).to_numpy()[1:]  # decision known at prior close
    turn = held_df.diff().abs().fillna(0).to_numpy()[1:]  # turnover events
    dates = close_df.index[1:]

    # regime mask aligned to dates (gate by prior daily close)
    if regime is not None:
        reg_vals = np.zeros(len(dates))
        ridx = regime.index
        for k, dt in enumerate(dates):
            d1 = pd.Timestamp(dt).normalize()
            prior = ridx[ridx <= d1]
            reg_vals[k] = int(regime[prior[-1]]) if len(prior) else 0
    else:
        reg_vals = np.ones(len(dates))

    # window slice
    mask = np.ones(len(dates), dtype=bool)
    if window_start is not None:
        mask &= dates >= pd.Timestamp(window_start)
    if window_end is not None:
        mask &= dates < pd.Timestamp(window_end)

    valid = np.isfinite(rets)
    rets_f = np.where(valid, rets, 0.0)

    n_held = (held_lag * valid).sum(axis=1)              # assets held per bar
    eng_gross = np.where(n_held > 0,
                         (rets_f * held_lag * valid).sum(axis=1) / np.maximum(n_held, 1),
                         0.0)
    # cost: maker RT/2 per turnover, averaged over held names
    cost = np.where(n_held > 0,
                    (turn * valid).sum(axis=1) * (MAKER_RT / 2.0) / np.maximum(n_held, 1),
                    0.0)
    eng = (eng_gross - cost) * reg_vals                  # CASH in bear

    n_all = valid.sum(axis=1)
    al = np.where(n_all > 0, (rets_f * valid).sum(axis=1) / np.maximum(n_all, 1), 0.0)

    eng = eng[mask]; al = al[mask]
    deploy = float(np.mean(n_held[mask] > 0)) if mask.sum() else 0.0
    return {"eng": eng, "al": al, "deploy": deploy, "held_lag": held_lag,
            "rets_f": rets_f, "valid": valid, "reg_vals": reg_vals, "mask": mask,
            "n_held": n_held}


def metrics(eng, al, bpd):
    if len(eng) == 0:
        return None
    lin_wk_e = float(np.mean(eng)) * bpd * 7 * 100
    lin_wk_a = float(np.mean(al)) * bpd * 7 * 100
    comp_e = float(np.prod(1 + eng) - 1) * 100
    comp_a = float(np.prod(1 + al) - 1) * 100
    cum = np.cumprod(1 + eng)
    maxdd = float(np.min((cum - np.maximum.accumulate(cum)) /
                         np.maximum.accumulate(cum))) * 100 if len(cum) > 1 else 0.0
    return {
        "lin_wk": round(lin_wk_e, 4), "al_lin_wk": round(lin_wk_a, 4),
        "edge_pp": round(lin_wk_e - lin_wk_a, 4),
        "compound": round(comp_e, 1), "al_compound": round(comp_a, 1),
        "maxdd": round(maxdd, 1),
    }


def random_null(panel, regime, tf, deploy, window_start, n_draws=200, seed0=0):
    """Engine-free: N random ensembles at the same deploy rate under the same gate.
    Returns percentile of engine compound vs random distribution (DEV window)."""
    rets_f = panel["rets_f"]; valid = panel["valid"]; reg_vals = panel["reg_vals"]
    dates_mask = panel["mask"]
    T, A = rets_f.shape
    randc = []
    for s in range(n_draws):
        rng = np.random.default_rng(seed0 + s)
        rh = (rng.random((T, A)) < deploy).astype(np.int8)
        nh = (rh * valid).sum(axis=1)
        r = np.where(nh > 0, (rets_f * rh * valid).sum(axis=1) / np.maximum(nh, 1), 0.0)
        r = r * reg_vals
        r = r[dates_mask]
        randc.append(float(np.prod(1 + r) - 1) * 100)
    return np.array(randc)


# --------------------------------------------------------------------------- #
# Full gauntlet for one family
# --------------------------------------------------------------------------- #
def run_family(family: str, tfs=("1d", "4h", "2h", "1h", "30m"),
               vote_threshold=0.5, n_null=200, confirm=False):
    t0 = time.time()
    reg_dev = btc_regime(DEV_WALL_MS)
    reg_oos = btc_regime(OOS_WALL_MS)
    rows = []
    print(f"\n{'='*94}\n  {family}  (configs={len(ALL_CONFIGS[family])}, vote>{vote_threshold:.0%}"
          f"{', CONFIRM' if confirm else ''})"
          f"  [{'VOLUME' if family in VOLUME_FAMILIES else 'MISC'}]\n{'='*94}")
    print(f"  {'TF':<5} {'DEVedge':>8} {'OOSedge':>8} {'deploy':>7} {'nullpct':>8} "
          f"{'DEVcomp':>10} {'OOScomp':>9}  verdict")
    print(f"  {'-'*84}")
    for tf in tfs:
        bpd = BARS_PER_DAY[tf]
        # DEV
        cdf, hdf = build_panel(family, tf, DEV_WALL_MS, vote_threshold, confirm=confirm)
        if cdf is None:
            print(f"  {tf:<5} INSUFFICIENT_ASSETS")
            continue
        pdev = replay(cdf, hdf, reg_dev, tf, window_start=DEV_START, window_end=DEV_END)
        mdev = metrics(pdev["eng"], pdev["al"], bpd)
        # OOS (extended wall, frozen configs, slice to OOS window)
        cdfo, hdfo = build_panel(family, tf, OOS_WALL_MS, vote_threshold, confirm=confirm)
        poos = replay(cdfo, hdfo, reg_oos, tf, window_start=OOS_START, window_end=OOS_END)
        moos = metrics(poos["eng"], poos["al"], bpd)
        # random-null on DEV
        if mdev and pdev["deploy"] > 0.001:
            randc = random_null(pdev, reg_dev, tf, pdev["deploy"], DEV_START, n_draws=n_null)
            nullpct = float(np.mean(randc < mdev["compound"])) * 100
        else:
            nullpct = 0.0
        # verdict
        consistent = bool(mdev and moos and mdev["edge_pp"] > 0 and moos["edge_pp"] > 0)
        selective = pdev["deploy"] < 0.70
        null_ok = nullpct >= 95.0
        if consistent and selective and null_ok:
            verdict = "*** VALIDATED"
        elif consistent and null_ok:
            verdict = "strong(not sel)"
        elif consistent:
            verdict = "consistent"
        elif selective and moos and moos["edge_pp"] > 0:
            verdict = "OOS-only"
        else:
            verdict = "null"
        rows.append({
            "family": family, "tf": tf, "vote_threshold": vote_threshold,
            "confirm": confirm,
            "dev_edge_pp": mdev["edge_pp"] if mdev else None,
            "oos_edge_pp": moos["edge_pp"] if moos else None,
            "deploy": round(pdev["deploy"], 3),
            "null_pctile": round(nullpct, 1),
            "dev_compound": mdev["compound"] if mdev else None,
            "oos_compound": moos["compound"] if moos else None,
            "dev_lin_wk": mdev["lin_wk"] if mdev else None,
            "oos_lin_wk": moos["lin_wk"] if moos else None,
            "dev_maxdd": mdev["maxdd"] if mdev else None,
            "oos_maxdd": moos["maxdd"] if moos else None,
            "consistent": consistent, "selective": selective, "null_ok": null_ok,
            "verdict": verdict,
        })
        de = mdev["edge_pp"] if mdev else 0
        oe = moos["edge_pp"] if moos else 0
        dc = mdev["compound"] if mdev else 0
        oc = moos["compound"] if moos else 0
        print(f"  {tf:<5} {de:>+7.3f} {oe:>+7.3f} {pdev['deploy']:>6.1%} {nullpct:>7.0f} "
              f"{dc:>+9.0f}% {oc:>+8.0f}%  {verdict}")
    print(f"  ({family} done in {time.time()-t0:.0f}s)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="all",
                    help="comma list or 'all' / 'volume' / 'misc'")
    ap.add_argument("--tfs", default="1d,4h,2h,1h,30m")
    ap.add_argument("--null", type=int, default=200)
    ap.add_argument("--thresholds", default="0.5",
                    help="comma list of vote thresholds to sweep (selectivity lever)")
    ap.add_argument("--confirm", action="store_true",
                    help="apply MA-style decoupled confirmation + slice-hold lever")
    a = ap.parse_args()

    if a.families == "all":
        fams = list(ALL_CONFIGS.keys())
    elif a.families == "volume":
        fams = list(VOLUME_FAMILIES)
    elif a.families == "misc":
        fams = list(MISC_FAMILIES)
    else:
        fams = [f.strip().upper() for f in a.families.split(",")]
    tfs = tuple(t.strip() for t in a.tfs.split(","))
    thresholds = [float(x) for x in a.thresholds.split(",")]

    print("=" * 94)
    print("V4 TI-EXTRA GAUNTLET — VOLUME + MISC families (SOTA rigor, vectorized)")
    print("=" * 94)
    print(f"DEV {DEV_START}..{DEV_END} | OOS {OOS_START}..{OOS_END} | "
          f"gate=BTC-SMA100 | cost={MAKER_RT} RT | null={a.null}")
    print(f"Vote-threshold sweep (selectivity lever): {thresholds}")
    print("VALIDATED = DEV/OOS-consistent AND random-null>=95pct AND selective(<70%)")

    all_rows = []
    for fam in fams:
        for thr in thresholds:
            all_rows.extend(run_family(fam, tfs=tfs, vote_threshold=thr,
                                       n_null=a.null, confirm=a.confirm))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTDIR / f"ti_extra_gauntlet_{ts}.json"
    with open(out, "w") as f:
        json.dump({"rows": all_rows, "dev": [DEV_START, DEV_END],
                   "oos": [OOS_START, OOS_END], "cost_rt": MAKER_RT,
                   "n_null": a.null, "timestamp": ts}, f, indent=2, default=str)

    # summary
    val = [r for r in all_rows if r["verdict"].startswith("***")]
    print(f"\n{'='*94}\n  VALIDATED CELLS: {len(val)}\n{'='*94}")
    for r in sorted(val, key=lambda x: -(x["oos_edge_pp"] or 0)):
        print(f"  {r['family']:<10}|{r['tf']:<4} thr{r['vote_threshold']} OOSedge {r['oos_edge_pp']:+.3f}pp "
              f"DEVedge {r['dev_edge_pp']:+.3f}pp deploy {r['deploy']:.1%} null {r['null_pctile']:.0f}pct")
    print(f"\n  Saved: {out}")
    return all_rows


if __name__ == "__main__":
    main()
