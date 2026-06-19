"""src/strat/mover_signal_id.py -- MOVE-CAPTURE: Can we IDENTIFY the movers?

OBJECTIVE (corrected from prior fold):
  - Across a 7d holding window with 14d lookback signals, do flagged assets
    out-move the universe over the next 7 days?
  - CAPTURE-RATE = flagged-mover 7d return vs universe mean (random-asset null at matched count)
  - Date-block permutation for significance (causal, NO lookahead)
  - No per-asset SMA200 gate (the proven failure mode)
  - MARKET-LEVEL circuit-breaker: scale total book exposure by BTC-trend breadth vol

SIGNALS TESTED (14d lookback, all causal):
  1. breakout:  close >= 14d-high (momentum breakout)
  2. vol_exp:   volume jump (vol_rel = vol / vol20avg > 1.5, proxy via price-vol proxy)
  3. mom_accel: mom7 > mom14 > 0 (momentum acceleration)
  4. range_pos: range_pos >= 0.8 (upper 80% of 14d range = near breakout)
  5. combo:     2+ of the above 4 signals

JUDGE ON:
  (a) CAPTURE-RATE: mean 7d fwd return of flagged group vs universe mean
  (b) Top-3 mover hit rate: fraction of periods where >=1 flagged asset is in top-3 actual movers
  (c) Date-block permutation p-value (500 permutations, block=30d)
  (d) Bear survival: 2022 full-year + drawdown stats via market circuit-breaker strategy

No emoji (cp1252). Does NOT git commit.
RWYB: python -m strat.mover_signal_id
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.mover_lab as lab
import strat.referee_harness as rh

COST = lab.COST
N_PERM = 500
BLOCK_DAYS = 30
OOS_START = "2022-01-01"
OOS_END = "2026-06-01"
FULL_START = "2020-01-01"
SEEDS = [11, 23, 42]
N_SLICES = 500
SLICE_DAYS = 7


# -------------------------------------------------------
# SIGNAL COMPUTATION (all causal, 14d lookback)
# -------------------------------------------------------
def compute_signals(ind: dict) -> dict:
    """Return dict of boolean DataFrames (date x asset), all causal."""
    C = ind["C"]
    hh14 = ind["hh14"]   # rolling 14d high
    ll14 = ind["ll14"]   # rolling 14d low
    mom7 = ind["mom7"]
    mom14 = ind["mom14"]

    # volume proxy: use ATR14 as a vol-of-price proxy (true volume not in ind)
    # vol_rel = atr14 / atr14.rolling(20).mean() -- price-range expansion
    atr14 = ind["atr14"]
    atr_avg = atr14.rolling(20, min_periods=10).mean()
    vol_rel = atr14 / (atr_avg + 1e-12)

    sig = {}
    # 1. breakout: close >= 14d-high (CURRENT close, all data <= today)
    sig["breakout"] = (C >= hh14).fillna(False)

    # 2. vol_exp: ATR expansion > 1.5x 20d average
    sig["vol_exp"] = (vol_rel >= 1.5).fillna(False)

    # 3. mom_accel: mom7 > mom14 > 0 (acceleration + positive trend)
    sig["mom_accel"] = ((mom7 > mom14) & (mom14 > 0)).fillna(False)

    # 4. range_pos: close in upper 20% of 14d range
    rng = hh14 - ll14
    rp = (C - ll14) / (rng + 1e-12)
    sig["range_pos"] = (rp >= 0.8).fillna(False)

    # 5. combo: 2+ of the above 4
    stack = sig["breakout"].astype(int) + sig["vol_exp"].astype(int) + \
            sig["mom_accel"].astype(int) + sig["range_pos"].astype(int)
    sig["combo"] = (stack >= 2)

    return sig


# -------------------------------------------------------
# CAPTURE-RATE ANALYSIS (per signal, per date)
# -------------------------------------------------------
def capture_rate_analysis(sig: dict, ind: dict, period_start: str, period_end: str) -> dict:
    """
    For each date d in [period_start, period_end):
      - flagged = assets where signal is True at d
      - 7d fwd return = C.shift(-7)/C - 1 at date d (LABEL ONLY, not used in signal)
      - capture_rate = mean(fwd_flagged) - mean(fwd_universe)
    Aggregate: mean capture rate, fraction where capture_rate > 0, top-3 hit rate.
    """
    C = ind["C"]
    fwd7 = C.shift(-7) / C - 1   # fwd return for label (not signal)

    idx = C.index
    mask = (idx >= pd.Timestamp(period_start)) & (idx < pd.Timestamp(period_end))
    dates = idx[mask]

    # exclude last 7 days (no valid fwd return)
    dates = dates[:-7]

    results = {}
    for sig_name, S in sig.items():
        cap_rates = []
        top3_hits = []
        flagged_counts = []
        flagged_means = []
        universe_means = []

        for d in dates:
            fwd_row = fwd7.loc[d].dropna()
            flag_row = S.loc[d].reindex(fwd_row.index).fillna(False)
            flagged_assets = flag_row[flag_row].index.tolist()
            n_flagged = len(flagged_assets)
            n_universe = len(fwd_row)

            if n_universe < 3:
                continue

            univ_mean = float(fwd_row.mean())
            universe_means.append(univ_mean)

            if n_flagged == 0:
                flagged_counts.append(0)
                flagged_means.append(np.nan)
                cap_rates.append(np.nan)
                top3_hits.append(False)
                continue

            flagged_mean = float(fwd_row[flagged_assets].mean())
            flagged_means.append(flagged_mean)
            flagged_counts.append(n_flagged)
            cap_rates.append(flagged_mean - univ_mean)

            # top-3 hit: is any flagged asset in top-3 actual movers?
            top3 = fwd_row.nlargest(3).index.tolist()
            top3_hits.append(bool(set(flagged_assets) & set(top3)))

        cap_arr = np.array([c for c in cap_rates if not np.isnan(c)])
        fm_arr = np.array([f for f in flagged_means if not np.isnan(f)])
        um_arr = np.array(universe_means)
        fc_arr = np.array(flagged_counts)
        th_arr = np.array(top3_hits)

        results[sig_name] = {
            "n_dates": len(cap_arr),
            "mean_capture_rate_pct": round(100 * float(cap_arr.mean()), 3) if len(cap_arr) else np.nan,
            "pos_capture_pct": round(100 * float((cap_arr > 0).mean()), 1) if len(cap_arr) else np.nan,
            "mean_flagged_7d_pct": round(100 * float(fm_arr.mean()), 3) if len(fm_arr) else np.nan,
            "mean_universe_7d_pct": round(100 * float(um_arr.mean()), 3) if len(um_arr) else np.nan,
            "top3_hit_rate_pct": round(100 * float(th_arr.mean()), 1) if len(th_arr) else np.nan,
            "mean_flagged_count": round(float(fc_arr[fc_arr > 0].mean()), 1) if (fc_arr > 0).any() else np.nan,
            "flag_rate_pct": round(100 * float((fc_arr > 0).mean()), 1) if len(fc_arr) else np.nan,
        }
    return results


# -------------------------------------------------------
# DATE-BLOCK PERMUTATION TEST
# -------------------------------------------------------
def block_perm_test(sig_bool: pd.DataFrame, ind: dict,
                    period_start: str, period_end: str,
                    n_perm: int = N_PERM, block_days: int = BLOCK_DAYS, seed: int = 42) -> dict:
    """
    Null: randomly permute the SIGNAL column across date-blocks (keeps autocorrelation structure).
    Returns p-value = fraction of permutations where mean capture rate >= observed.
    """
    C = ind["C"]
    fwd7 = C.shift(-7) / C - 1
    idx = C.index
    mask = (idx >= pd.Timestamp(period_start)) & (idx < pd.Timestamp(period_end))
    dates = idx[mask][:-7]

    def mean_cap(S_perm):
        caps = []
        for d in dates:
            fwd_row = fwd7.loc[d].dropna()
            flag_row = S_perm.loc[d].reindex(fwd_row.index).fillna(False)
            flagged = flag_row[flag_row].index.tolist()
            if len(flagged) == 0 or len(fwd_row) < 3:
                continue
            caps.append(float(fwd_row[flagged].mean()) - float(fwd_row.mean()))
        return np.mean(caps) if caps else 0.0

    observed = mean_cap(sig_bool)

    # Build date blocks
    rng = np.random.default_rng(seed)
    n = len(dates)
    n_blocks = max(1, n // block_days)
    block_size = n // n_blocks

    # Permute by shuffling blocks of the signal DataFrame (date axis)
    null_stats = []
    for _ in range(n_perm):
        # Create permuted signal: randomly shift blocks
        perm_idx = []
        block_starts = np.arange(0, n, block_size)
        perm_block_order = rng.permutation(len(block_starts))
        for bi in perm_block_order:
            bs = block_starts[bi]
            be = min(bs + block_size, n)
            perm_idx.extend(range(bs, be))
        perm_idx = perm_idx[:n]
        perm_dates = dates[perm_idx]
        S_perm = sig_bool.copy()
        S_perm.loc[dates] = sig_bool.loc[perm_dates].values
        null_stats.append(mean_cap(S_perm))

    null_arr = np.array(null_stats)
    p_val = float((null_arr >= observed).mean())
    return {
        "observed_cap_pct": round(100 * observed, 3),
        "null_mean_cap_pct": round(100 * float(null_arr.mean()), 3),
        "null_p95_cap_pct": round(100 * float(np.percentile(null_arr, 95)), 3),
        "p_value": round(p_val, 4),
        "significant_05": bool(p_val < 0.05),
        "n_perm": n_perm,
    }


# -------------------------------------------------------
# MARKET CIRCUIT-BREAKER STRATEGY (ungated per-asset)
# -------------------------------------------------------
def market_circuit_breaker_weight(sig_name: str, sig: dict, ind: dict,
                                  top_k: int = 3, rebal_days: int = 7) -> pd.DataFrame:
    """
    Build weight matrix using signal `sig_name`, NO per-asset SMA200 gate.
    Market circuit-breaker scales TOTAL book exposure by:
      exposure_scale = f(BTC_trend, breadth, vol)
    where:
      BTC above SMA200 + breadth >= 40%: full exposure (1.0)
      BTC above SMA200 + breadth 20-40%: 60% exposure
      BTC below SMA200 OR breadth < 20%: 20% exposure (minimal, not zero)

    Causal: positions lagged 1 bar; top-K flagged by signal score.
    """
    C = ind["C"]; sma200 = ind["sma200"]; sma50 = ind["sma50"]
    vol20 = ind["vol20"]; S = sig[sig_name]

    # Precompute market exposure scale per date (causal)
    breadth = (C > sma50).astype(float).mean(axis=1)
    btc_up = (C["BTCUSDT"] > sma200["BTCUSDT"]).fillna(False)

    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    last_rebal = -9999

    for i, d in enumerate(C.index):
        if i - last_rebal >= rebal_days:
            # Market circuit-breaker scale
            brc = float(breadth.iloc[i]) if not pd.isna(breadth.iloc[i]) else 0.5
            btc_t = bool(btc_up.iloc[i])
            if btc_t and brc >= 0.40:
                scale = 1.0
            elif btc_t and brc >= 0.20:
                scale = 0.60
            else:
                scale = 0.20  # bear circuit-breaker: minimal exposure, NOT zero

            # Pick top-K by signal (any asset, no per-asset gate)
            flag_row = S.iloc[i]
            flagged = flag_row[flag_row.astype(bool)].index.tolist()

            if len(flagged) == 0:
                # No signal: fallback to top-K by mom14 (momentum carry)
                mom_row = ind["mom14"].iloc[i].dropna()
                if len(mom_row) >= top_k:
                    flagged = mom_row.nlargest(top_k).index.tolist()
                else:
                    flagged = mom_row.index.tolist()

            if len(flagged) > top_k:
                # Tiebreak by mom7
                mom7_row = ind["mom7"].iloc[i]
                flagged = sorted(flagged, key=lambda s: -float(mom7_row.get(s, 0)))[:top_k]

            W.loc[d, :] = 0.0
            if flagged:
                w_per = scale / len(flagged)
                for s in flagged:
                    if s in W.columns:
                        W.loc[d, s] = w_per
            last_rebal = i
        elif i > 0:
            W.iloc[i] = W.iloc[i - 1]

    return W


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 76)
    print("MOVER SIGNAL IDENTIFICATION -- capture-rate lab")
    print(f"OOS: {OOS_START} -> {OOS_END} | FULL: {FULL_START}")
    print("SIGNALS: breakout / vol_exp / mom_accel / range_pos / combo")
    print("NO per-asset gate | market circuit-breaker for bear survival")
    print("=" * 76)

    ind = lab.load(FULL_START, OOS_END)
    C = ind["C"]
    print(f"Assets: {list(C.columns)}")
    print(f"Dates: {C.index[0].date()} -> {C.index[-1].date()}  ({len(C)} bars)")

    sig = compute_signals(ind)

    # --- SECTION 1: Capture-rate analysis (FULL period + OOS) ---
    print("\n[1] CAPTURE-RATE ANALYSIS")
    print("-" * 76)
    for period_label, pstart, pend in [
        ("FULL 2020-2026", FULL_START, OOS_END),
        ("OOS 2022-2026", OOS_START, OOS_END),
        ("BEAR 2022", "2022-01-01", "2023-01-01"),
        ("BULL 2020", "2020-01-01", "2021-01-01"),
    ]:
        print(f"\n  Period: {period_label}")
        res = capture_rate_analysis(sig, ind, pstart, pend)
        hdr = f"  {'Signal':<12} {'Cap%':>8} {'Pos%':>7} {'FlagRet%':>9} {'UnivRet%':>9} {'Top3Hit%':>9} {'FlagCnt':>7} {'FlagRate%':>9}"
        print(hdr)
        print("  " + "-" * 72)
        for sn, r in res.items():
            print(f"  {sn:<12} {r['mean_capture_rate_pct']:>8.3f} {r['pos_capture_pct']:>7.1f} "
                  f"{r['mean_flagged_7d_pct']:>9.3f} {r['mean_universe_7d_pct']:>9.3f} "
                  f"{r['top3_hit_rate_pct']:>9.1f} {r['mean_flagged_count']:>7.1f} "
                  f"{r['flag_rate_pct']:>9.1f}")

    # --- SECTION 2: Date-block permutation significance (OOS only) ---
    print("\n[2] DATE-BLOCK PERMUTATION SIGNIFICANCE (OOS 2022-2026, n_perm=500, block=30d)")
    print("-" * 76)
    perm_results = {}
    for sig_name, S in sig.items():
        print(f"  Permuting {sig_name}...", end=" ", flush=True)
        pr = block_perm_test(S, ind, OOS_START, OOS_END, n_perm=N_PERM, seed=42)
        perm_results[sig_name] = pr
        sig_marker = "**" if pr["significant_05"] else "  "
        print(f"obs={pr['observed_cap_pct']:+.3f}% null_mean={pr['null_mean_cap_pct']:+.3f}% "
              f"p={pr['p_value']:.4f} {sig_marker}")

    # --- SECTION 3: Random-7d-slice profitability via referee_harness ---
    print("\n[3] RANDOM-7d-SLICE PROFITABILITY (n=500 slices, OOS 2022-2026)")
    print("    Market circuit-breaker (NO per-asset gate), top-3 by signal, rebal 7d")
    print("-" * 76)

    bh_W = rh.bh_ew_weights(ind)
    bh_b = rh.book_daily_returns(bh_W, ind)
    bh_stats = {s: rh.bh_slice_stats(bh_b, OOS_START, OOS_END, N_SLICES, SLICE_DAYS, s) for s in SEEDS}
    bh_pr = [bh_stats[s]["pos_rate"] for s in SEEDS]
    bh_mn = [bh_stats[s]["mean_pct"] for s in SEEDS]
    print(f"  BH baseline:  pos_rate={round(np.mean(bh_pr),1)}% (seeds {bh_pr})  "
          f"mean={round(np.mean(bh_mn),2)}%")

    # Gated baseline (SMA200 per-asset -- the PROVEN FAILURE)
    gate_sma = ind["gate"].astype(float)
    gW = gate_sma.div(gate_sma.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    gated_b = rh.book_daily_returns(gW, ind)
    g_stats = [rh.slice_stats(gated_b, bh_b, OOS_START, OOS_END, N_SLICES, SLICE_DAYS, s) for s in SEEDS]
    g_pr = [s["pos_rate"] for s in g_stats]
    g_mn = [s["mean_pct"] for s in g_stats]
    g_bw = [s["beat_bh_pct"] for s in g_stats]
    print(f"  Gated-SMA200: pos_rate={round(np.mean(g_pr),1)}% (seeds {g_pr})  "
          f"mean={round(np.mean(g_mn),2)}%  beat_bh={round(np.mean(g_bw),1)}%  [BASELINE TO BEAT]")

    slice_results = {}
    print(f"\n  {'Signal':<12} {'PosRate%':>9} {'Mean%':>7} {'BeatBH%':>8} {'DownWkMn%':>10} {'Expo':>6}")
    print("  " + "-" * 60)
    for sig_name in sig.keys():
        W = market_circuit_breaker_weight(sig_name, sig, ind, top_k=3, rebal_days=7)
        b = rh.book_daily_returns(W, ind)
        sts = [rh.slice_stats(b, bh_b, OOS_START, OOS_END, N_SLICES, SLICE_DAYS, s) for s in SEEDS]
        pr = [s["pos_rate"] for s in sts]
        mn = [s["mean_pct"] for s in sts]
        bw = [s["beat_bh_pct"] for s in sts]
        dn = [s["down_wk_eng_mean"] for s in sts]
        expo = float((W.sum(axis=1) > 0).loc[C.index >= OOS_START].mean())
        slice_results[sig_name] = {
            "pos_rate": round(float(np.mean(pr)), 1),
            "pos_rate_seeds": pr,
            "mean_pct": round(float(np.mean(mn)), 2),
            "beat_bh": round(float(np.mean(bw)), 1),
            "down_wk_mean": round(float(np.mean([d for d in dn if d is not None])), 2) if any(d is not None for d in dn) else None,
            "avg_expo": round(expo, 2),
        }
        pr_str = f"{round(np.mean(pr),1):>9.1f}"
        mn_str = f"{round(np.mean(mn),2):>7.2f}"
        bw_str = f"{round(np.mean(bw),1):>8.1f}"
        dn_mean = round(float(np.mean([d for d in dn if d is not None])), 2) if any(d is not None for d in dn) else None
        dn_str = f"{dn_mean:>10.2f}" if dn_mean is not None else f"{'N/A':>10}"
        print(f"  {sig_name:<12} {pr_str} {mn_str} {bw_str} {dn_str} {expo:>6.2f}")

    # --- SECTION 4: Bear survival (2022 full year) ---
    print("\n[4] BEAR SURVIVAL -- full year 2022 compound returns")
    print("-" * 76)
    # BH 2022
    bh_ret_2022 = float((1 + bh_b.loc["2022-01-01":"2022-12-31"]).prod() - 1)
    print(f"  BH 2022: {100*bh_ret_2022:.1f}%")
    for sig_name in sig.keys():
        W = market_circuit_breaker_weight(sig_name, sig, ind, top_k=3, rebal_days=7)
        b = rh.book_daily_returns(W, ind)
        ret_2022 = float((1 + b.loc["2022-01-01":"2022-12-31"]).prod() - 1)
        ret_full = float((1 + b.loc[FULL_START:OOS_END]).prod() - 1)
        eq = np.cumprod(1 + b.values); pk = np.maximum.accumulate(eq)
        mdd = float(((eq - pk) / pk).min() * 100)
        print(f"  {sig_name:<12}: 2022={100*ret_2022:.1f}%  full-cycle={100*ret_full:.1f}%  maxDD={mdd:.1f}%")

    # --- SECTION 5: Signal flag rate + SMA200 exclusion audit ---
    print("\n[5] SMA200-GATE EXCLUSION AUDIT (verifying the failure mode)")
    print("-" * 76)
    # Identify top-1 actual mover per day
    fwd7 = C.shift(-7) / C - 1
    gate_sma200 = ind["gate"]  # per-asset SMA200 gate
    n_total = 0; n_top1_excluded = 0; n_top3_excluded_any = 0
    btc_top1_count = 0

    for d in C.index[:-7]:
        fwd_row = fwd7.loc[d].dropna()
        if len(fwd_row) < 3:
            continue
        n_total += 1
        top1 = fwd_row.idxmax()
        top3 = fwd_row.nlargest(3).index.tolist()
        gate_row = gate_sma200.loc[d]
        if not bool(gate_row.get(top1, True)):
            n_top1_excluded += 1
        if all(not bool(gate_row.get(a, True)) for a in top3):
            n_top3_excluded_any += 1
        if top1 == "BTCUSDT":
            btc_top1_count += 1

    print(f"  Total dates analyzed: {n_total}")
    print(f"  Top-1 mover EXCLUDED by SMA200 gate: {n_top1_excluded}/{n_total} = "
          f"{100*n_top1_excluded/n_total:.1f}%  [target: ~39%]")
    print(f"  All top-3 movers excluded: {n_top3_excluded_any}/{n_total} = "
          f"{100*n_top3_excluded_any/n_total:.1f}%  [target: ~38%]")
    print(f"  BTC is top-1 mover: {btc_top1_count}/{n_total} = "
          f"{100*btc_top1_count/n_total:.1f}%  [target: ~9%]")

    # --- SUMMARY ---
    runtime = round(time.time() - t0, 1)
    print(f"\n{'='*76}")
    print("SUMMARY VERDICT")
    print(f"{'='*76}")

    best_sig = max(slice_results.items(), key=lambda x: x[1]["mean_pct"])
    print(f"  Best signal by mean slice return: {best_sig[0]} ({best_sig[1]['mean_pct']:+.2f}%)")
    best_cap = max(
        ((sn, capture_rate_analysis({sn: sig[sn]}, ind, OOS_START, OOS_END)[sn])
         for sn in sig),
        key=lambda x: x[1].get("mean_capture_rate_pct", -99)
    )
    print(f"  Best signal by capture rate OOS: {best_cap[0]} ({best_cap[1]['mean_capture_rate_pct']:+.3f}%/7d)")
    sig_sigs = [sn for sn, pr in perm_results.items() if pr["significant_05"]]
    print(f"  Signals significant at p<0.05 (perm test): {sig_sigs if sig_sigs else 'NONE'}")
    print(f"  SMA200 gate failure confirmed: "
          f"top-1 excluded {100*n_top1_excluded/n_total:.1f}% | BTC top-1 only {100*btc_top1_count/n_total:.1f}%")

    # --- SAVE ---
    out = {
        "meta": {
            "oos": [OOS_START, OOS_END],
            "full": [FULL_START, OOS_END],
            "n_slices": N_SLICES, "seeds": SEEDS, "n_perm": N_PERM,
            "runtime_s": runtime
        },
        "capture_rate": {
            "FULL": capture_rate_analysis(sig, ind, FULL_START, OOS_END),
            "OOS": capture_rate_analysis(sig, ind, OOS_START, OOS_END),
            "BEAR_2022": capture_rate_analysis(sig, ind, "2022-01-01", "2023-01-01"),
            "BULL_2020": capture_rate_analysis(sig, ind, "2020-01-01", "2021-01-01"),
        },
        "perm_test_oos": perm_results,
        "slice_profitability": slice_results,
        "bh_baseline": {"pos_rate": round(float(np.mean(bh_pr)), 1), "mean_pct": round(float(np.mean(bh_mn)), 2)},
        "gated_baseline": {"pos_rate": round(float(np.mean(g_pr)), 1), "mean_pct": round(float(np.mean(g_mn)), 2),
                           "beat_bh": round(float(np.mean(g_bw)), 1)},
        "sma200_audit": {
            "n_total": n_total,
            "top1_excluded_pct": round(100*n_top1_excluded/n_total, 1),
            "top3_all_excluded_pct": round(100*n_top3_excluded_any/n_total, 1),
            "btc_top1_pct": round(100*btc_top1_count/n_total, 1),
        },
    }
    outp = ROOT.parent / "runs" / "strat" / f"mover_signal_id_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {outp}  ({runtime}s)")
    return out


if __name__ == "__main__":
    main()
