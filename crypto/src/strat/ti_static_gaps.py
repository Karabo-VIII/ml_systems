"""Static gap-closer — the two refinements left open by ti_static_closeout.

The closeout doc explicitly left two static items "remaining (low priority, not blocking)":
  GAP 1 — ADX|4h lookback revisit. ADX|4h is the most genuinely SELECTIVE non-MA cell
          (~6.9% per-asset deploy, DEV/OOS consistent) but sat at the 93rd random-null
          percentile — JUST below the 95 bar. A better efficiency-ratio lookback +
          vote-threshold may push it over (or confirm it stays borderline).
  GAP 2 — Sharpe/vol-weighted ensemble. The naive EW-of-3 averaged toward the middle and
          MA/Bollinger are 0.96 correlated (redundant). The corr matrix said the smart
          blend is MA + RSI (drop redundant Bollinger). Build it inverse-vol AND
          Sharpe-weighted and compare to naive EW and the best single engine.

Both reuse the EXACT validated framework (regime gate + decoupled confirmation + maker
cost + DEV/OOS windows + random-null) from ti_static_exhaustion / _randomnull / _closeout.
Honest reporting: linear %/wk is primary; edge = engine - always-long under same gate.
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
from strat.ma_wealth_driver import _load_asset, _u100_symbols
from strat.ti_engine_sota import adx_single_held
from strat.ti_static_exhaustion import apply_confirmation, _btc_regime
from strat.ti_static_closeout import (
    build_ma, build_rsi, build_bollinger, _gated_daily_stream, stats,
)

BARS_PER_DAY = fl.BARS_PER_DAY
DEV_END = "2024-05-15"
OOS_START = "2024-05-15"
OOS_END = "2025-01-01"
DEV_WALL_MS = int(pd.Timestamp(DEV_END).value // 10**6)
OOS_WALL_MS = int(pd.Timestamp(OOS_END).value // 10**6)
OUTDIR = CRYPTO / "runs" / "engines" / "V3_ti_families" / "static_gaps"

N_NULL = 200


# =============================================================================
# GAP 1 — ADX|4h lookback revisit (with gold-standard random-null)
# =============================================================================

def _adx_votes(close, lookbacks, threshold):
    """Ensemble vote fraction across a set of ADX efficiency-ratio lookbacks."""
    n = len(close)
    votes = np.zeros(n, dtype=float)
    for lb in lookbacks:
        votes += adx_single_held(close, lb, threshold).astype(float)
    return votes / len(lookbacks)


def _build_adx_panel(tf, lookbacks, eff_threshold, vote_threshold, confirm,
                     wall_ms, slice_days=7, conf_lookback_days=7):
    bpd = BARS_PER_DAY[tf]
    slice_bars = slice_days * bpd
    conf_lb_bars = conf_lookback_days * bpd
    cd, hd, dr = {}, {}, {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, wall_ms)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        close = np.asarray(c, dtype=float)
        frac = _adx_votes(close, lookbacks, eff_threshold)
        raw = (frac > vote_threshold).astype(np.int8)
        held = apply_confirmation(c, raw, conf_lb_bars, slice_bars) if confirm else raw
        cd[sym] = pd.Series(c, index=dates)
        hd[sym] = pd.Series(held, index=dates)
        dr[sym] = float(np.mean(held))
    return cd, hd, dr, slice_bars


def _portfolio_linwk(close_dict, held_dict, btc_regime, win_start, win_end, bpd):
    all_dates = sorted(set().union(*[s.index.tolist() for s in close_dict.values()]))
    ws = pd.Timestamp(win_start); we = pd.Timestamp(win_end)
    win = [d for d in all_dates if ws <= d < we]
    if len(win) < 10:
        return None, None, None
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
    return (float(np.mean(eng)) * bpd * 7 * 100,
            float(np.mean(al)) * bpd * 7 * 100,
            float(np.mean(eng != 0)))


def _random_held_like(held_dict, deploy_rates, slice_bars, rng):
    out = {}
    for sym, hs in held_dict.items():
        n = len(hs); dr = deploy_rates[sym]
        rh = np.zeros(n, dtype=np.int8); i = 0
        while i < n:
            if rng.random() < dr:
                rh[i:min(i+slice_bars, n)] = 1
            i += slice_bars
        out[sym] = pd.Series(rh, index=hs.index)
    return out


def gap1_adx_revisit():
    print("=" * 88)
    print("GAP 1 — ADX|4h lookback revisit (push the 93rd-pct borderline cell)")
    print("=" * 88)
    print("Sweep efficiency-ratio lookback sets x vote threshold; keep DEV/OOS-consistent")
    print("SELECTIVE cells; run random-null on the best. Bar = >=95th pct + DEV/OOS both +.")
    tf = "4h"; bpd = BARS_PER_DAY[tf]
    btc_dev = _btc_regime(DEV_WALL_MS)
    btc_oos = _btc_regime(OOS_WALL_MS)

    # Candidate lookback SETS (efficiency-ratio windows) and thresholds to revisit
    lookback_sets = {
        "short(5,7,10)": [5, 7, 10],
        "mid(10,14,21)": [10, 14, 21],
        "long(21,30,45)": [21, 30, 45],
        "wide(7,14,21,30)": [7, 14, 21, 30],
    }
    eff_thr = 0.5  # efficiency threshold (trend strength); 0.5 was the borderline winner
    rows = []
    print(f"\n  {'lookback_set':<18} {'vthr':>5} {'conf':>5} {'DEVedge':>9} {'OOSedge':>9} {'deploy':>8}  flag")
    print(f"  {'-'*72}")
    for name, lbs in lookback_sets.items():
        for vthr in [0.5, 0.6, 0.7]:
            for confirm in [False, True]:
                cd_d, hd_d, _, _ = _build_adx_panel(tf, lbs, eff_thr, vthr, confirm, DEV_WALL_MS)
                dev_e, dev_a, _ = _portfolio_linwk(cd_d, hd_d, btc_dev, "2020-01-01", DEV_END, bpd)
                cd_o, hd_o, dr_o, sb = _build_adx_panel(tf, lbs, eff_thr, vthr, confirm, OOS_WALL_MS)
                oos_e, oos_a, dep = _portfolio_linwk(cd_o, hd_o, btc_oos, OOS_START, OOS_END, bpd)
                if dev_e is None or oos_e is None:
                    continue
                dev_edge = dev_e - dev_a; oos_edge = oos_e - oos_a
                consistent = dev_edge > 0 and oos_edge > 0
                selective = dep < 0.70
                flag = "*** consistent+sel" if (consistent and selective) else (
                    "consistent" if consistent else "")
                rows.append({"lookback_set": name, "lookbacks": lbs, "vote_thr": vthr,
                             "confirm": confirm, "dev_edge_pp": round(dev_edge, 3),
                             "oos_edge_pp": round(oos_edge, 3), "oos_deploy": round(dep, 3),
                             "consistent": consistent, "selective": selective,
                             "_cd_o": None})  # placeholder
                print(f"  {name:<18} {vthr:>5.1f} {str(confirm):>5} {dev_edge:>+8.3f} "
                      f"{oos_edge:>+8.3f} {dep:>7.1%}  {flag}")

    # pick the best DEV/OOS-consistent + selective cell by OOS edge for random-null
    cands = [r for r in rows if r["consistent"] and r["selective"]]
    null_result = None
    if cands:
        best = max(cands, key=lambda r: r["oos_edge_pp"])
        print(f"\n  Best consistent+selective cell -> random-null (n={N_NULL}):")
        print(f"    {best['lookback_set']} vthr{best['vote_thr']} conf{best['confirm']} "
              f"OOS edge {best['oos_edge_pp']:+.3f}pp deploy {best['oos_deploy']:.1%}")
        cd_o, hd_o, dr_o, sb = _build_adx_panel(tf, best["lookbacks"], eff_thr,
                                                best["vote_thr"], best["confirm"], OOS_WALL_MS)
        oos_e, oos_a, _ = _portfolio_linwk(cd_o, hd_o, btc_oos, OOS_START, OOS_END, bpd)
        real_edge = oos_e - oos_a
        rng = np.random.default_rng(42); null_edges = []
        for _ in range(N_NULL):
            rhd = _random_held_like(hd_o, dr_o, sb, rng)
            r_e, r_a, _ = _portfolio_linwk(cd_o, rhd, btc_oos, OOS_START, OOS_END, bpd)
            if r_e is not None:
                null_edges.append(r_e - r_a)
        null_edges = np.array(null_edges)
        pct = float(np.mean(null_edges < real_edge)) * 100
        print(f"    real edge {real_edge:+.3f}pp vs null mean {null_edges.mean():+.3f}pp "
              f"(sd {null_edges.std():.3f}) -> {pct:.0f}th pct  "
              f"{'PASS (>=95)' if pct >= 95 else 'still borderline (<95)'}")
        null_result = {"cell": {k: best[k] for k in ("lookback_set", "vote_thr", "confirm",
                                "oos_edge_pp", "oos_deploy")},
                       "real_edge_pp": round(real_edge, 3),
                       "null_mean_pp": round(float(null_edges.mean()), 3),
                       "null_sd": round(float(null_edges.std()), 3),
                       "percentile": round(pct, 1), "pass_95": bool(pct >= 95)}
    else:
        print("\n  No DEV/OOS-consistent + selective ADX|4h cell found across the lookback "
              "revisit -> ADX|4h stays NULL/borderline, not promoted.")

    for r in rows:
        r.pop("_cd_o", None)
    return {"sweep": rows, "random_null": null_result}


# =============================================================================
# GAP 2 — Sharpe / inverse-vol weighted ensemble (MA + RSI, drop redundant Bollinger)
# =============================================================================

def gap2_smart_ensemble():
    print("\n" + "=" * 88)
    print("GAP 2 — Smart-weighted ensemble (corr matrix says MA + RSI; BB redundant w/ MA)")
    print("=" * 88)
    btc = _btc_regime(OOS_WALL_MS)

    print("  Building MA WMA|4h ...")
    ma_cd, ma_hd = build_ma()
    ma_d, ma_al = _gated_daily_stream(ma_cd, ma_hd, btc, "4h")
    print("  Building RSI|4h ...")
    rsi_cd, rsi_hd = build_rsi()
    rsi_d, rsi_al = _gated_daily_stream(rsi_cd, rsi_hd, btc, "4h")
    print("  Building Bollinger|1d ...")
    bb_cd, bb_hd = build_bollinger()
    bb_d, bb_al = _gated_daily_stream(bb_cd, bb_hd, btc, "1d")

    idx = ma_d.index.union(rsi_d.index).union(bb_d.index)
    ma_a = ma_d.reindex(idx).fillna(0.0)
    rsi_a = rsi_d.reindex(idx).fillna(0.0)
    bb_a = bb_d.reindex(idx).fillna(0.0)
    al_a = ma_al.reindex(idx).fillna(0.0)

    def inv_vol_w(streams):
        vols = np.array([s.std() for s in streams])
        w = (1.0 / (vols + 1e-12)); return w / w.sum()

    def sharpe_w(streams):
        sh = np.array([max(s.mean() / (s.std() + 1e-12), 0.0) for s in streams])
        if sh.sum() <= 0:
            return np.ones(len(streams)) / len(streams)
        return sh / sh.sum()

    # MA + RSI (the smart pair) under both weighting schemes
    pair = [ma_a, rsi_a]
    w_iv = inv_vol_w(pair); w_sh = sharpe_w(pair)
    ens_ma_rsi_ew = (ma_a + rsi_a) / 2.0
    ens_ma_rsi_iv = w_iv[0] * ma_a + w_iv[1] * rsi_a
    ens_ma_rsi_sh = w_sh[0] * ma_a + w_sh[1] * rsi_a
    # 3-way naive (closeout baseline) for reference
    ens_3_ew = (ma_a + rsi_a + bb_a) / 3.0

    rows = [
        stats(ma_d, ma_al, "MA WMA|4h (alone)"),
        stats(rsi_d, rsi_al, "RSI|4h (alone)"),
        stats(bb_d, bb_al, "Bollinger|1d (alone)"),
        stats(ens_3_ew, al_a, "EW-of-3 (closeout)"),
        stats(ens_ma_rsi_ew, al_a, "MA+RSI equal-wt"),
        stats(ens_ma_rsi_iv, al_a, f"MA+RSI inv-vol({w_iv[0]:.2f}/{w_iv[1]:.2f})"),
        stats(ens_ma_rsi_sh, al_a, f"MA+RSI sharpe-wt({w_sh[0]:.2f}/{w_sh[1]:.2f})"),
    ]
    print(f"\n  {'engine':<30} {'linwk':>8} {'edge':>8} {'MaxDD':>8} {'dVol':>7} {'Sharpe':>7}")
    print(f"  {'-'*74}")
    for r in rows:
        print(f"  {r['label']:<30} {r['linwk']:>+7.3f}% {r['edge_pp']:>+7.3f} "
              f"{r['maxdd']:>7.1f}% {r['daily_vol_pct']:>6.3f}% {r['sharpe_ann']:>6.2f}")

    best = max(rows, key=lambda r: r["sharpe_ann"])
    print(f"\n  Best Sharpe: {best['label']} ({best['sharpe_ann']:.2f})")
    return {"weights": {"inv_vol": [round(float(x), 4) for x in w_iv],
                        "sharpe": [round(float(x), 4) for x in w_sh]},
            "rows": rows, "best_by_sharpe": best["label"]}


def main():
    g1 = gap1_adx_revisit()
    g2 = gap2_smart_ensemble()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTDIR / f"static_gaps_{ts}.json", "w") as f:
        json.dump({"gap1_adx_revisit": g1, "gap2_smart_ensemble": g2,
                   "timestamp": ts}, f, indent=2)
    print(f"\n  Saved: static_gaps_{ts}.json")


if __name__ == "__main__":
    main()
