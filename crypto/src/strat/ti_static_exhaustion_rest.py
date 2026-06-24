"""Static Exhaustion — remaining families (Stochastic, ADX, Supertrend, Funding, OI).

Applies the SAME proper levers that revealed RSI/Bollinger as real static edges:
  - vote threshold sweep {0.5, 0.6, 0.7, 0.8}
  - MA-style decoupled confirmation layer {off, on}
  - regime gate + cost (always on)
to the five families NOT yet retested with proper levers.

Stochastic/ADX/Supertrend are price-based (close/OHLC). Funding/OI use the REAL
chimera features (norm_funding, norm_oi_price_divergence) via ti_engine_sota_v11.

For any cell that is SELECTIVE (deploy<70%) AND beats always-long OOS AND is DEV/OOS
consistent, we then run random-null (handled by a follow-up if winners appear).
"""
from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
import json
import sys
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
from strat.ti_engine_sota import (
    stochastic_single_held, adx_single_held, STOCHASTIC_CONFIGS, ADX_CONFIGS,
)
from strat.ti_engine_sota_v11 import (
    load_asset_with_features, supertrend_held_vec,
    funding_held_vec, oi_divergence_held_vec,
    FUNDING_REAL_CONFIGS, OI_DIV_REAL_CONFIGS,
)
from strat.ti_static_exhaustion import apply_confirmation, _btc_regime

BARS_PER_DAY = fl.BARS_PER_DAY
OOS_START = "2024-05-15"; OOS_END = "2025-01-01"
DEV_END = "2024-05-15"
OOS_WALL_MS = int(pd.Timestamp(OOS_END).value // 10**6)
DEV_WALL_MS = int(pd.Timestamp(DEV_END).value // 10**6)
OUTDIR = CRYPTO / "runs" / "engines" / "V3_ti_families" / "static_exhaustion"

SUPERTREND_CFG = [(7, 2.0), (7, 2.5), (10, 2.0), (10, 2.5), (10, 3.0), (14, 2.5), (14, 3.0), (20, 3.0)]

# families needing OHLC/close vs chimera feature
PRICE_FAMS = ["STOCHASTIC", "ADX", "SUPERTREND"]
FEAT_FAMS = {"FUNDING": "norm_funding", "OI_DIVERGENCE": "norm_oi_price_divergence"}


def votes_price(family, o, h, l, c):
    close = np.asarray(c, dtype=float); high = np.asarray(h, dtype=float); low = np.asarray(l, dtype=float)
    n = len(close); votes = np.zeros(n, dtype=float); nv = 0
    if family == "STOCHASTIC":
        for cfg in STOCHASTIC_CONFIGS:
            votes += stochastic_single_held(close, high, low, *cfg).astype(float); nv += 1
    elif family == "ADX":
        for cfg in ADX_CONFIGS:
            votes += adx_single_held(close, *cfg).astype(float); nv += 1
    elif family == "SUPERTREND":
        for cfg in SUPERTREND_CFG:
            votes += supertrend_held_vec(close, high, low, *cfg).astype(float); nv += 1
    return votes / max(1, nv)


def votes_feat(family, feat_arr):
    n = len(feat_arr); votes = np.zeros(n, dtype=float); nv = 0
    if family == "FUNDING":
        for z_long, z_exit in FUNDING_REAL_CONFIGS:
            votes += funding_held_vec(feat_arr, 7, z_long, z_exit).astype(float); nv += 1
    elif family == "OI_DIVERGENCE":
        for (thr,) in OI_DIV_REAL_CONFIGS:
            votes += oi_divergence_held_vec(feat_arr, thr).astype(float); nv += 1
    return votes / max(1, nv)


def run_cell(family, tf, threshold, confirm, wall_ms, win_start, win_end,
             slice_days=7, lookback_days=7):
    bpd = BARS_PER_DAY[tf]
    slice_bars = slice_days * bpd; lookback_bars = lookback_days * bpd
    btc_regime = _btc_regime(wall_ms)
    is_feat = family in FEAT_FAMS

    close_dict, held_dict = {}, {}
    for sym in _u100_symbols():
        if is_feat:
            rec = load_asset_with_features(sym, tf, wall_ms, [FEAT_FAMS[family]])
            if rec is None:
                continue
            dates, o, h, l, c, feats = rec
            frac = votes_feat(family, feats[FEAT_FAMS[family]])
        else:
            rec = _load_asset(sym, tf, wall_ms)
            if rec is None:
                continue
            dates, o, h, l, c = rec
            frac = votes_price(family, o, h, l, c)
        raw = (frac > threshold).astype(np.int8)
        held = apply_confirmation(c, raw, lookback_bars, slice_bars) if confirm else raw
        close_dict[sym] = pd.Series(c, index=dates)
        held_dict[sym] = pd.Series(held, index=dates)

    if len(close_dict) < 10:
        return None
    all_dates = sorted(set().union(*[s.index.tolist() for s in close_dict.values()]))
    ws = pd.Timestamp(win_start); we = pd.Timestamp(win_end)
    win = [d for d in all_dates if ws <= d < we]
    if len(win) < 10:
        return None

    eng, al = [], []
    for i in range(1, len(win)):
        dp, dc = win[i-1], win[i]
        er, ar = [], []
        for sym in close_dict:
            if dp in close_dict[sym].index and dc in close_dict[sym].index:
                cp = close_dict[sym][dp]; cc = close_dict[sym][dc]
                if cp > 0:
                    r = (cc/cp) - 1.0; ar.append(r)
                    if dp in held_dict[sym].index and held_dict[sym][dp] == 1:
                        er.append(r)
        e = float(np.mean(er)) if er else 0.0
        if btc_regime is not None:
            d1 = pd.Timestamp(dp).normalize()
            prior = btc_regime.index[btc_regime.index <= d1]
            if len(prior) and int(btc_regime[prior[-1]]) == 0:
                e = 0.0
        eng.append(e); al.append(float(np.mean(ar)) if ar else 0.0)
    eng = np.array(eng); al = np.array(al)
    return {
        "eng_linwk": round(float(np.mean(eng)) * bpd * 7 * 100, 3),
        "al_linwk": round(float(np.mean(al)) * bpd * 7 * 100, 3),
        "deploy": round(float(np.mean(eng != 0)), 3),
    }


def main():
    print("=" * 96)
    print("STATIC EXHAUSTION — remaining 5 families (proper levers: threshold x confirmation)")
    print("=" * 96)
    print("Feat families (Funding/OI) use REAL chimera features. 'WORKS' = DEV+OOS both beat AL.")

    thresholds = [0.5, 0.6, 0.7, 0.8]
    families = PRICE_FAMS + list(FEAT_FAMS)
    # Funding/OI unavailable at synth-2h; use native TFs
    tf_map = {"STOCHASTIC": ["4h", "1h", "1d"], "ADX": ["4h", "1h", "1d"],
              "SUPERTREND": ["4h", "1h", "1d"], "FUNDING": ["4h", "1h", "1d"],
              "OI_DIVERGENCE": ["4h", "1h", "1d"]}

    all_rows = []; winners = []
    for family in families:
        print(f"\n{'='*96}\n  {family}\n{'='*96}")
        print(f"  {'TF':<5} {'thr':>5} {'conf':>5} {'DEVedge':>9} {'OOSedge':>9} {'OOSdep':>8}  flag")
        print(f"  {'-'*70}")
        for tf in tf_map[family]:
            for thr in thresholds:
                for confirm in [False, True]:
                    dev = run_cell(family, tf, thr, confirm, DEV_WALL_MS, "2020-01-01", DEV_END)
                    oos = run_cell(family, tf, thr, confirm, OOS_WALL_MS, OOS_START, OOS_END)
                    if dev is None or oos is None:
                        continue
                    dev_edge = dev["eng_linwk"] - dev["al_linwk"]
                    oos_edge = oos["eng_linwk"] - oos["al_linwk"]
                    row = {"family": family, "tf": tf, "threshold": thr, "confirm": confirm,
                           "dev_edge_pp": round(dev_edge, 3), "oos_edge_pp": round(oos_edge, 3),
                           "oos_deploy": oos["deploy"]}
                    all_rows.append(row)
                    consistent = dev_edge > 0 and oos_edge > 0
                    selective = oos["deploy"] < 0.70
                    flag = ""
                    if consistent and selective:
                        flag = "*** WORKS"; winners.append(row)
                    elif consistent:
                        flag = "consistent(not sel)"
                    print(f"  {tf:<5} {thr:>5.1f} {str(confirm):>5} {dev_edge:>+8.3f} {oos_edge:>+8.3f} "
                          f"{oos['deploy']:>7.1%}  {flag}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTDIR / f"static_exhaustion_rest_{ts}.json", "w") as f:
        json.dump({"rows": all_rows, "winners": winners, "timestamp": ts}, f, indent=2)

    print(f"\n{'='*96}")
    print(f"  DEV/OOS-CONSISTENT + SELECTIVE cells: {len(winners)}")
    print(f"{'='*96}")
    for w in sorted(winners, key=lambda x: -x["oos_edge_pp"]):
        print(f"  {w['family']}|{w['tf']} thr{w['threshold']} conf{w['confirm']}: "
              f"DEV {w['dev_edge_pp']:+.2f}pp OOS {w['oos_edge_pp']:+.2f}pp deploy {w['oos_deploy']:.1%}")
    if not winners:
        print("  NONE — these 5 families have no DEV/OOS-consistent selective static edge.")
    print(f"\n  Saved: static_exhaustion_rest_{ts}.json")


if __name__ == "__main__":
    main()
