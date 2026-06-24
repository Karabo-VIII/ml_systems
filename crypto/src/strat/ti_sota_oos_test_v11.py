"""OOS validation for the V1.1 families 6-8 (Supertrend / Funding / OI-divergence).

Same held-out gauntlet as ti_sota_oos_test.py, but for the vectorized + real-chimera
families built in ti_engine_sota_v11.py. Funding/OI use the ACTUAL norm_funding /
norm_oi_price_divergence chimera columns (not the V1 price proxies).

Reports LINEAR/wk (honest read), compound (inflated, for completeness), edge vs
always-long, and the single-asset BTC baseline.

Note: at synthesized-2h the chimera norm features are unavailable (no native baseline),
so Funding/OI will show ~0 deploy at 2h by design — that's correct, not a bug.
"""
from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
import json
import sys
import time
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
from strat.ti_engine_sota_v11 import (
    load_asset_with_features,
    make_funding_provider, make_oi_provider, make_supertrend_provider,
)

BARS_PER_DAY = fl.BARS_PER_DAY
OOS_START = "2024-05-15"
OOS_END = "2025-01-01"
OOS_WALL_MS = int(pd.Timestamp(OOS_END).value // 10**6)
OUTDIR = CRYPTO / "runs" / "engines" / "V3_ti_families" / "oos_validation"

SUPERTREND_CONFIGS = [(7, 2.0), (7, 2.5), (10, 2.0), (10, 2.5), (10, 3.0),
                      (14, 2.5), (14, 3.0), (20, 3.0)]


def _btc_regime(wall_ms: int):
    btc_rec = _load_asset("BTCUSDT", "1d", wall_ms)
    if not btc_rec:
        return None
    btc_dates, _, _, _, btc_close = btc_rec
    sma100 = pd.Series(btc_close, index=btc_dates).rolling(100).mean()
    regime_s = pd.Series(0, index=btc_dates, dtype=int)
    state = 0
    for i in range(100, len(btc_dates)):
        s = sma100.iloc[i]; c = btc_close[i]
        if np.isnan(s):
            continue
        if state == 0 and c > s * 1.02:
            state = 1
        elif state == 1 and c < s * 0.98:
            state = 0
        regime_s.iloc[i] = state
    return regime_s


def _metrics(rets, slices, bpd, label):
    if len(rets) == 0:
        return {"label": label, "status": "EMPTY"}
    compound = float(np.prod(1 + rets) - 1) * 100
    linear_daily = float(np.mean(rets)) * bpd * 100
    return {
        "label": label,
        "compound_pct": round(compound, 2),
        "linear_sum_pct": round(float(np.sum(rets)) * 100, 2),
        "linear_weekly_pct": round(linear_daily * 7, 4),
        "n_slices": len(slices),
        "pct_slices_profitable": round(sum(1 for s in slices if s > 0) / len(slices), 3) if slices else 0,
        "median_slice_pct": round(float(np.median(slices)), 2) if slices else 0,
    }


# feature each family needs
FAM_FEATURE = {
    "SUPERTREND": None,
    "FUNDING": "norm_funding",
    "OI_DIVERGENCE": "norm_oi_price_divergence",
}


def run_oos_family_v11(family: str, tf: str, slice_days: int = 7) -> dict:
    t0 = time.time()
    bpd = BARS_PER_DAY[tf]
    slice_bars = slice_days * bpd
    oos_start_ts = pd.Timestamp(OOS_START)
    btc_regime = _btc_regime(OOS_WALL_MS)
    symbols = _u100_symbols()

    feat_col = FAM_FEATURE[family]

    if family == "SUPERTREND":
        provider = make_supertrend_provider(SUPERTREND_CONFIGS)
    elif family == "FUNDING":
        provider = make_funding_provider()
    elif family == "OI_DIVERGENCE":
        provider = make_oi_provider()

    close_dict, held_dict = {}, {}
    for sym in symbols:
        if family == "SUPERTREND":
            rec = _load_asset(sym, tf, OOS_WALL_MS)
            if rec is None:
                continue
            dates, o, h, l, c = rec
            held = provider(o, h, l, c)
        else:
            rec = load_asset_with_features(sym, tf, OOS_WALL_MS, [feat_col])
            if rec is None:
                continue
            dates, o, h, l, c, feats = rec
            held = provider(feats[feat_col])
        close_dict[sym] = pd.Series(c, index=dates)
        held_dict[sym] = pd.Series(held, index=dates)

    n_assets = len(close_dict)
    if n_assets < 10:
        return {"status": "INSUFFICIENT_ASSETS", "n_assets": n_assets, "family": family, "tf": tf}

    all_dates = sorted(set().union(*[s.index.tolist() for s in close_dict.values()]))
    oos_dates = [d for d in all_dates if d >= oos_start_ts]
    if len(oos_dates) < 10:
        return {"status": "INSUFFICIENT_OOS_DATA", "family": family, "tf": tf}

    engine_rets, al_rets, combined_rets = [], [], []
    for i in range(1, len(oos_dates)):
        dp, dc = oos_dates[i-1], oos_dates[i]
        eng_r, all_r = [], []
        n_transitions = 0
        for sym in close_dict:
            if dp in close_dict[sym].index and dc in close_dict[sym].index:
                cp = close_dict[sym][dp]; cc = close_dict[sym][dc]
                if cp > 0:
                    ret = (cc / cp) - 1.0
                    all_r.append(ret)
                    if dp in held_dict[sym].index and held_dict[sym][dp] == 1:
                        eng_r.append(ret)
                        loc = held_dict[sym].index.get_loc(dp)
                        if loc > 0 and held_dict[sym].iloc[loc-1] != held_dict[sym].iloc[loc]:
                            n_transitions += 1
        if eng_r:
            ew = float(np.mean(eng_r))
            cost = n_transitions * (MAKER_RT / 2.0) / max(1, len(eng_r))
            engine_rets.append(ew - cost)
        else:
            engine_rets.append(0.0)
        al_rets.append(float(np.mean(all_r)) if all_r else 0.0)

        if btc_regime is not None:
            d_1d = pd.Timestamp(dp).normalize()
            prior = btc_regime.index[btc_regime.index <= d_1d]
            regime_val = int(btc_regime[prior[-1]]) if len(prior) > 0 else 0
            combined_rets.append(engine_rets[-1] if regime_val == 1 else 0.0)
        else:
            combined_rets.append(engine_rets[-1])

    eng = np.array(engine_rets); al = np.array(al_rets); comb = np.array(combined_rets)
    n_bars = len(eng)

    def slc(arr):
        out = []; i = 0
        while i + slice_bars <= n_bars:
            out.append(float(np.prod(1 + arr[i:i+slice_bars]) - 1) * 100); i += slice_bars
        return out

    eng_m = _metrics(eng, slc(eng), bpd, "engine")
    al_m = _metrics(al, slc(al), bpd, "always-long")
    comb_m = _metrics(comb, slc(comb), bpd, "combined")
    deploy = float(np.mean(eng != 0))

    return {
        "status": "OK", "family": family, "tf": tf,
        "n_assets": n_assets, "n_oos_bars": n_bars,
        "engine": eng_m, "always_long": al_m, "combined": comb_m,
        "deploy_rate": round(deploy, 3),
        "combined_edge_vs_al_linwk_pp": round(comb_m["linear_weekly_pct"] - al_m["linear_weekly_pct"], 4),
        "elapsed_s": round(time.time() - t0, 1),
    }


def run_sweep(tfs=("4h", "1h", "1d", "30m")):
    print("=" * 84)
    print("V1.1 FAMILIES 6-8 — OOS VALIDATION (real chimera features, vectorized)")
    print("=" * 84)
    print("Funding=norm_funding, OI=norm_oi_price_divergence (REAL features, not proxies)")
    print("Honest read = linear/wk + edge vs AL. (2h omitted: chimera norm unavailable at synth-2h)")

    all_results = {}
    for family in ["SUPERTREND", "FUNDING", "OI_DIVERGENCE"]:
        print(f"\n{'='*84}\n  {family}\n{'='*84}")
        print(f"  {'TF':<5} {'Eng lin/wk':>11} {'Comb lin/wk':>12} {'AL lin/wk':>10} "
              f"{'edge pp':>8} {'Comb %Prof':>11} {'Deploy':>8}")
        print(f"  {'-'*70}")
        fam_results = []
        for tf in tfs:
            r = run_oos_family_v11(family, tf)
            fam_results.append(r)
            if r["status"] != "OK":
                print(f"  {tf:<5} {r['status']}")
                continue
            e, a, c = r["engine"], r["always_long"], r["combined"]
            print(f"  {tf:<5} {e['linear_weekly_pct']:>+10.3f}% {c['linear_weekly_pct']:>+11.3f}% "
                  f"{a['linear_weekly_pct']:>+9.3f}% {r['combined_edge_vs_al_linwk_pp']:>+7.3f} "
                  f"{c['pct_slices_profitable']:>10.0%} {r['deploy_rate']:>7.1%}")
        all_results[family] = fam_results

    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = OUTDIR / f"ti_oos_v11_results_{ts}.json"
    with open(outfile, "w") as f:
        json.dump({"results": all_results, "oos_window": f"{OOS_START} to {OOS_END}",
                   "cost_rt": MAKER_RT, "timestamp": ts,
                   "note": "Funding/OI use REAL chimera features"}, f, indent=2, default=str)
    print(f"\n  Saved: {outfile}")
    return all_results


if __name__ == "__main__":
    run_sweep()
