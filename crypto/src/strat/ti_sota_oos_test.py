"""OOS validation for the SOTA TI families (RSI / MACD / Bollinger).

Mirrors `oos_engine_test.py` (the MA family OOS pattern) but uses the SOTA
ensemble providers from `ti_engine_sota.py`.

The honest test:
1. Load with an EXTENDED wall (2025-01-01) so the OOS window is available.
2. Compute ensemble holds on the FULL causal history (configs are FROZEN — the
   config grids in ti_engine_sota were never fit on OOS data; they are fixed
   hand-chosen grids, so this is genuinely held-out).
3. Slice portfolio returns to ONLY the OOS window (2024-05-15 -> 2025-01-01).
4. Report BOTH linear and compound, engine vs always-long vs combined (gated).
5. Also report the single-asset BTC honest baseline (no rebalancing premium).

Per the forensic audit (FORENSIC_AUDIT_RESULT.md): the EW-rebalanced compound is
inflated ~9.5x vs linear. So linear/wk and the single-asset numbers are the
honest read; compound is reported for completeness only.
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
from strat.ti_engine_sota import (
    make_ensemble_provider,
    RSI_CONFIGS, MACD_CONFIGS, BOLLINGER_CONFIGS,
    macd_single_held, rsi_single_held, bollinger_single_held,
)

BARS_PER_DAY = fl.BARS_PER_DAY
OOS_START = "2024-05-15"
OOS_END = "2025-01-01"
OOS_WALL_MS = int(pd.Timestamp(OOS_END).value // 10**6)

OUTDIR = CRYPTO / "runs" / "engines" / "V3_ti_families" / "oos_validation"

FAMILIES = {
    "RSI": RSI_CONFIGS,
    "MACD": MACD_CONFIGS,
    "BOLLINGER": BOLLINGER_CONFIGS,
}


def _btc_regime(wall_ms: int) -> pd.Series | None:
    """V1 BTC SMA100 regime (1 = bull/deploy, 0 = bear/cash)."""
    btc_rec = _load_asset("BTCUSDT", "1d", wall_ms)
    if not btc_rec:
        return None
    btc_dates, _, _, _, btc_close = btc_rec
    sma100 = pd.Series(btc_close, index=btc_dates).rolling(100).mean()
    regime_s = pd.Series(0, index=btc_dates, dtype=int)
    state = 0
    for i in range(100, len(btc_dates)):
        s = sma100.iloc[i]
        c = btc_close[i]
        if np.isnan(s):
            continue
        if state == 0 and c > s * 1.02:
            state = 1
        elif state == 1 and c < s * 0.98:
            state = 0
        regime_s.iloc[i] = state
    # DE-LEAK (V1.1 align_htf_on bug class): regime[D] is the day-D 1d CLOSE (~23:59); the
    # per-bar prior[-1] gate broadcast on a day-D bar must read the last CLOSED 1d bar =
    # regime[D-1]. Shift forward one 1d bar. See runs/engines/AUDIT_ORACLE_UPGRADE/leakcheck.json.
    return regime_s.shift(1).fillna(0).astype(int)


def _metrics(rets: np.ndarray, slices: list, bpd: int, label: str) -> dict:
    if len(rets) == 0:
        return {"label": label, "status": "EMPTY"}
    compound = float(np.prod(1 + rets) - 1) * 100
    linear_sum = float(np.sum(rets)) * 100
    linear_daily = float(np.mean(rets)) * bpd * 100
    linear_weekly = linear_daily * 7
    cum = np.cumprod(1 + rets)
    maxdd = float(np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum))) * 100
    pct_prof = sum(1 for s in slices if s > 0) / len(slices) if slices else 0
    median_s = float(np.median(slices)) if slices else 0
    return {
        "label": label,
        "compound_pct": round(compound, 2),
        "linear_sum_pct": round(linear_sum, 2),
        "linear_daily_pct": round(linear_daily, 4),
        "linear_weekly_pct": round(linear_weekly, 4),
        "maxdd_pct": round(maxdd, 1),
        "n_slices": len(slices),
        "pct_slices_profitable": round(pct_prof, 3),
        "median_slice_pct": round(median_s, 2),
        "win_rate_bar": round(float(np.mean(rets > 0)), 3),
    }


def run_oos_family(family: str, tf: str, slice_days: int = 7,
                   vote_threshold: float = 0.5) -> dict:
    """Run one TI family on the OOS window at one timeframe."""
    t0 = time.time()
    configs = FAMILIES[family]
    bpd = BARS_PER_DAY[tf]
    slice_bars = slice_days * bpd
    provider = make_ensemble_provider(family, configs, vote_threshold=vote_threshold)

    oos_start_ts = pd.Timestamp(OOS_START)
    btc_regime = _btc_regime(OOS_WALL_MS)

    symbols = _u100_symbols()
    close_dict = {}
    held_dict = {}
    for sym in symbols:
        rec = _load_asset(sym, tf, OOS_WALL_MS)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        held = provider(sym, o, h, l, c)  # holds on FULL causal history
        close_dict[sym] = pd.Series(c, index=dates)
        held_dict[sym] = pd.Series(held, index=dates)

    n_assets = len(close_dict)
    if n_assets < 10:
        return {"status": "INSUFFICIENT_ASSETS", "n_assets": n_assets, "family": family, "tf": tf}

    all_dates = sorted(set().union(*[s.index.tolist() for s in close_dict.values()]))
    oos_dates = [d for d in all_dates if d >= oos_start_ts]
    if len(oos_dates) < 10:
        return {"status": "INSUFFICIENT_OOS_DATA", "n_oos_dates": len(oos_dates),
                "family": family, "tf": tf}

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
                        # cost on transition into the position
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

        # combined: apply regime gate
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
        out = []
        i = 0
        while i + slice_bars <= n_bars:
            out.append(float(np.prod(1 + arr[i:i+slice_bars]) - 1) * 100)
            i += slice_bars
        return out

    eng_m = _metrics(eng, slc(eng), bpd, "engine (no gate)")
    al_m = _metrics(al, slc(al), bpd, "always-long")
    comb_m = _metrics(comb, slc(comb), bpd, "combined (regime-gated)")

    # single-asset BTC honest baseline (no rebalancing premium)
    btc_honest = None
    if "BTCUSDT" in close_dict:
        bc = close_dict["BTCUSDT"]
        bc_oos = bc[bc.index >= oos_start_ts]
        bh = held_dict["BTCUSDT"]
        if len(bc_oos) > 5:
            br = []
            for i in range(1, len(bc_oos)):
                dp = bc_oos.index[i-1]
                r = (bc_oos.iloc[i] / bc_oos.iloc[i-1]) - 1.0
                held_v = bh[dp] if dp in bh.index else 0
                br.append(r if held_v == 1 else 0.0)
            br = np.array(br)
            bh_full = (bc_oos.iloc[-1] / bc_oos.iloc[0] - 1) * 100
            btc_honest = {
                "btc_engine_compound_pct": round(float(np.prod(1 + br) - 1) * 100, 2),
                "btc_buyhold_compound_pct": round(float(bh_full), 2),
                "btc_engine_linear_pct": round(float(np.sum(br)) * 100, 2),
                "btc_deploy_rate": round(float(np.mean([1 for x in br if x != 0]) / len(br)) if len(br) else 0, 3),
            }

    return {
        "status": "OK",
        "family": family, "tf": tf,
        "oos_window": f"{OOS_START} to {OOS_END}",
        "n_assets": n_assets, "n_oos_bars": n_bars,
        "n_oos_days": round(n_bars / bpd, 1),
        "engine": eng_m, "always_long": al_m, "combined": comb_m,
        "btc_single_asset": btc_honest,
        "combined_edge_vs_al_linwk_pp": round(comb_m["linear_weekly_pct"] - al_m["linear_weekly_pct"], 4),
        "elapsed_s": round(time.time() - t0, 1),
    }


def run_oos_sweep(tfs=("4h", "2h", "1h", "1d", "30m")):
    print("=" * 84)
    print("SOTA TI FAMILIES — OOS VALIDATION (2024-05-15 to 2025-01-01, held-out)")
    print("=" * 84)
    print("Configs FROZEN (hand-chosen grids, never fit on OOS). Reporting LINEAR + compound.")
    print("Linear/wk and single-asset BTC are the HONEST read (compound is inflated ~9.5x).")

    all_results = {}
    for family in FAMILIES:
        print(f"\n{'='*84}\n  {family}\n{'='*84}")
        print(f"  {'TF':<5} {'Eng lin/wk':>11} {'Comb lin/wk':>12} {'AL lin/wk':>10} "
              f"{'edge pp':>8} {'Comb %Prof':>11} {'BTC eng/BH':>16}")
        print(f"  {'-'*78}")
        fam_results = []
        for tf in tfs:
            r = run_oos_family(family, tf)
            fam_results.append(r)
            if r["status"] != "OK":
                print(f"  {tf:<5} {r['status']}")
                continue
            e, a, c = r["engine"], r["always_long"], r["combined"]
            btc = r["btc_single_asset"]
            btc_str = (f"{btc['btc_engine_compound_pct']:+.0f}/{btc['btc_buyhold_compound_pct']:+.0f}%"
                       if btc else "n/a")
            print(f"  {tf:<5} {e['linear_weekly_pct']:>+10.3f}% {c['linear_weekly_pct']:>+11.3f}% "
                  f"{a['linear_weekly_pct']:>+9.3f}% {r['combined_edge_vs_al_linwk_pp']:>+7.3f} "
                  f"{c['pct_slices_profitable']:>10.0%} {btc_str:>16}")
        all_results[family] = fam_results

    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = OUTDIR / f"ti_oos_results_{ts}.json"
    with open(outfile, "w") as f:
        json.dump({"results": all_results, "oos_window": f"{OOS_START} to {OOS_END}",
                   "cost_rt": MAKER_RT, "timestamp": ts}, f, indent=2, default=str)
    print(f"\n  Saved: {outfile}")
    return all_results


if __name__ == "__main__":
    run_oos_sweep()
