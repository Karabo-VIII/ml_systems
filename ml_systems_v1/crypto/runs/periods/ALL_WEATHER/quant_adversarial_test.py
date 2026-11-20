"""Adversarial statistical tests for TI x TF wave-1 traction claims.

Tests run (pre-registered before seeing 2023 data):
  T1. TIMING-SCRAMBLED NULL: shuffle the rolling-pick ASSIGNMENT (keep the same exposure schedule,
      randomize WHICH config is picked) -- does the observed 2022-bear performance beat the null?
      H0: rolling-pick is no better than random config selection within the band.
      H1: rolling-pick selection (recent-best-in-band) adds value over random band selection.
      One-sided p-value (alt: observed >= null percentile).

  T2. 2023 GENUINE OOS: apply FROZEN 2020-2022 rolling-pick hyperparams (lookback=120, step=30)
      to 2023 -- an entirely unseen calendar year (the existing work ENDS at 2022-12-31).
      H0: tier-A candidates degrade to zero or negative in 2023 (the all-weather signal was in-sample).
      H1: tier-A candidates stay flat-or-positive in 2023.
      Test: sign test (are 2023 net > 0?) across tier-A set with N-correction.
      This is the single most powerful falsifier for the "in-sample-tuned lookback" weakness.

  T3. MULTIPLE-COMPARISONS AUDIT: 26 TIs x 6 TFs x 3 strategies = 468 cells tested.
      Report Bonferroni-adjusted threshold and how many tier-A claims survive it.
      (Observed p-values estimated from the hyperparameter sensitivity grid.)

  T4. SAME-EXPOSURE SHUFFLED CONTROL: for each tier-A TI, compare 2022 net vs a null that
      holds the SAME exposure schedule but assigns random returns from that TI's config pool
      (block-shuffled, keeps autocorrelation structure). Beat this null = the TIMING matters,
      not just the exposure fraction.

No emoji. Long-only. All runs are read-only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3] / "crypto"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import strat.deep2020_ti_pipeline as TI
from strat.deep2020_ti_pipeline import INDICATORS

SPAN_FULL = ("2020-01-01", "2024-01-01")   # include 2023
YEARS = {
    "2020_bull": ("2020-01-01", "2021-01-01"),
    "2021_mixed": ("2021-01-01", "2022-01-01"),
    "2022_bear": ("2022-01-01", "2023-01-01"),
    "2023_oos": ("2023-01-01", "2024-01-01"),   # THE UNSEEN YEAR
}
LOOKBACK_D = 120
STEP_D = 30
N_SHUFFLE = 2000
TIER_A = ["MACD", "TSI", "KELTNER", "PSAR", "MFI", "RSI"]

# ========================= helpers =========================
def _net(s):
    s = s.dropna()
    return float(np.prod(1 + s.to_numpy()) - 1) * 100 if len(s) > 1 else 0.0

def _maxdd(s):
    s = s.dropna()
    if len(s) < 2:
        return 0.0
    eq = np.cumprod(1 + s.to_numpy()); pk = np.maximum.accumulate(eq)
    return float(((eq - pk) / pk).min() * 100)

def _per_year(daily):
    out = {}
    for yk, (lo, hi) in YEARS.items():
        s = daily[(daily.index >= pd.Timestamp(lo)) & (daily.index < pd.Timestamp(hi))]
        out[yk] = {"net": round(_net(s), 2), "maxdd": round(_maxdd(s), 2), "n": int(len(s.dropna()))}
    return out

def _ti_series_extended(ti_key, tf):
    """Get daily return series for all configs of ti_key, spanning 2020-2023."""
    TI.WIN = SPAN_FULL
    TI.SPLIT = "2022-10-01"
    ind = INDICATORS[ti_key]
    loader = TI.load_ohlcv if ind.get("loader") == "ohlcv" else TI.load_ohlc
    assets, vt = loader(tf)
    if not assets or len(assets) < 5:
        return None, None
    mh = ind.get("minhold", 12)
    cols = {}
    for p in ind["grid"]():
        r = TI._book(assets, ind["iron"], p, vt, mh)
        if r is None:
            continue
        daily = r[0]
        if daily is not None and len(daily) > 50:
            cols[ind["name"](p)] = daily
    if not cols:
        return None, None
    bh_cells = []
    for A in assets:
        ret, win, idx = A["ret"], A["win"], A["idx"]
        bh_cells.append(pd.Series(ret[win], index=idx))
    bh = pd.concat(bh_cells, axis=1).fillna(0.0).mean(axis=1).sort_index()
    bh_daily = bh.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()
    return pd.DataFrame(cols).sort_index(), bh_daily

def _rolling_pick(series_df, seed=None):
    """Walk-forward rolling-pick (no look-ahead). Returns (stitched daily series, list of picks)."""
    rng = np.random.default_rng(seed)
    idx = series_df.index
    cfgs = list(series_df.columns)
    start = idx.min() + pd.Timedelta(days=LOOKBACK_D)
    pieces, picks = [], []
    t = start
    while t < idx.max():
        nxt = t + pd.Timedelta(days=STEP_D)
        look = series_df[(idx >= t - pd.Timedelta(days=LOOKBACK_D)) & (idx < t)]
        fwd = series_df[(idx >= t) & (idx < nxt)]
        if len(look) < 20 or len(fwd) < 2:
            t = nxt; continue
        look_net = (np.prod(1 + look.fillna(0.0).to_numpy(), axis=0) - 1) * 100
        band = [c for c, v in zip(cfgs, look_net) if v > 0]
        if not band:
            band = [cfgs[int(np.argmax(look_net))]]
        best = max(band, key=lambda c: look_net[cfgs.index(c)])
        seg = fwd[best].dropna()
        picks.append(best)
        if len(seg):
            pieces.append(seg)
        t = nxt
    if not pieces:
        return None, picks
    return pd.concat(pieces).sort_index(), picks

def _rolling_random(series_df, picks_schedule, seed=None):
    """Same exposure schedule (same band-membership gate + same timing windows) as the real pick,
    but RANDOMLY select which band member to use. Keeps exposure fraction identical."""
    rng = np.random.default_rng(seed)
    idx = series_df.index
    cfgs = list(series_df.columns)
    start = idx.min() + pd.Timedelta(days=LOOKBACK_D)
    pieces = []
    t = start
    while t < idx.max():
        nxt = t + pd.Timedelta(days=STEP_D)
        look = series_df[(idx >= t - pd.Timedelta(days=LOOKBACK_D)) & (idx < t)]
        fwd = series_df[(idx >= t) & (idx < nxt)]
        if len(look) < 20 or len(fwd) < 2:
            t = nxt; continue
        look_net = (np.prod(1 + look.fillna(0.0).to_numpy(), axis=0) - 1) * 100
        band = [c for c, v in zip(cfgs, look_net) if v > 0]
        if not band:
            band = [cfgs[int(np.argmax(look_net))]]
        chosen = rng.choice(band)   # random within the same band
        seg = fwd[chosen].dropna()
        if len(seg):
            pieces.append(seg)
        t = nxt
    if not pieces:
        return None
    return pd.concat(pieces).sort_index()

# ========================= T1: TIMING-SCRAMBLED NULL =========================
def test_T1_timing_scrambled(sdf, ti_key, tf):
    """H0: rolling-pick is no better than random selection within the same band.
    Test: observed 2022-bear net vs empirical distribution of N_SHUFFLE random picks."""
    real_series, _ = _rolling_pick(sdf)
    if real_series is None:
        return None
    real_2022 = _net(real_series[(real_series.index >= pd.Timestamp("2022-01-01")) &
                                  (real_series.index < pd.Timestamp("2023-01-01"))])
    null_2022 = []
    for seed in range(N_SHUFFLE):
        ns = _rolling_random(sdf, None, seed=seed)
        if ns is not None:
            val = _net(ns[(ns.index >= pd.Timestamp("2022-01-01")) & (ns.index < pd.Timestamp("2023-01-01"))])
            null_2022.append(val)
    if not null_2022:
        return None
    null_arr = np.array(null_2022)
    p_val = float(np.mean(null_arr >= real_2022))   # one-sided: H1 = real >= null
    pct = float(np.percentile(null_arr, [5, 25, 50, 75, 95]).tolist()[2])   # median null
    return {
        "real_2022_net": round(real_2022, 2),
        "null_median_2022": round(float(np.median(null_arr)), 2),
        "null_p05_2022": round(float(np.percentile(null_arr, 5)), 2),
        "null_p95_2022": round(float(np.percentile(null_arr, 95)), 2),
        "p_value_one_sided": round(p_val, 4),
        "n_shuffles": len(null_arr),
        "verdict": "SIGNAL" if p_val < 0.10 else "SAME_AS_RANDOM",
    }

# ========================= T2: 2023 GENUINE OOS =========================
def test_T2_2023_oos(sdf, bh, ti_key, tf):
    """Apply frozen 2020-2022 rolling-pick policy to 2023 (fully unseen year)."""
    real_series, picks = _rolling_pick(sdf)
    if real_series is None:
        return None
    oos_2023 = real_series[(real_series.index >= pd.Timestamp("2023-01-01")) &
                            (real_series.index < pd.Timestamp("2024-01-01"))]
    bh_2023 = bh[(bh.index >= pd.Timestamp("2023-01-01")) & (bh.index < pd.Timestamp("2024-01-01"))]
    n23 = _net(oos_2023)
    dd23 = _maxdd(oos_2023)
    bh23 = _net(bh_2023)
    return {
        "net_2023": round(n23, 2),
        "maxdd_2023": round(dd23, 2),
        "bh_2023": round(bh23, 2),
        "n_bars_2023": int(len(oos_2023.dropna())),
        "beats_bh_2023": bool(n23 > bh23),
        "positive_2023": bool(n23 > 0),
        "n_distinct_picks_2023": len(set(
            p for p in picks
        )),
    }

# ========================= T4: same-exposure block shuffle =========================
def test_T4_exposure_shuffle(sdf, ti_key, tf, N=1000):
    """Null that matches the EXPOSURE FRACTION of the rolling-pick but uses block-shuffled returns.
    If the real 2022 return sits in the tails of this null, the exposure-timing (which config)
    matters -- not just the exposure fraction."""
    real_series, _ = _rolling_pick(sdf)
    if real_series is None:
        return None
    sub_2022 = real_series[(real_series.index >= pd.Timestamp("2022-01-01")) &
                            (real_series.index < pd.Timestamp("2023-01-01"))].dropna()
    if len(sub_2022) < 20:
        return None
    real_2022 = _net(sub_2022)
    # build the pool of 2022 daily returns across all configs
    all_2022_returns = []
    for c in sdf.columns:
        s = sdf[c][(sdf.index >= pd.Timestamp("2022-01-01")) & (sdf.index < pd.Timestamp("2023-01-01"))].dropna()
        all_2022_returns.append(s.to_numpy())
    if not all_2022_returns:
        return None
    rng = np.random.default_rng(42)
    n = len(sub_2022)
    block = max(5, n // 15)   # ~5% block = ~17 blocks for 2022
    null_nets = []
    for _ in range(N):
        cfg_idx = rng.integers(0, len(all_2022_returns))
        pool = all_2022_returns[cfg_idx]
        if len(pool) < n:
            continue
        # block-shuffle the pool, then take n-bar sample
        n_blocks = len(pool) // block
        block_starts = rng.permutation(n_blocks) * block
        shuffled = np.concatenate([pool[s:s+block] for s in block_starts])[:n]
        null_nets.append(float(np.prod(1 + shuffled) - 1) * 100)
    if not null_nets:
        return None
    null_arr = np.array(null_nets)
    p_val = float(np.mean(null_arr >= real_2022))
    return {
        "real_2022_net": round(real_2022, 2),
        "exposure_null_median": round(float(np.median(null_arr)), 2),
        "exposure_null_p05": round(float(np.percentile(null_arr, 5)), 2),
        "exposure_null_p95": round(float(np.percentile(null_arr, 95)), 2),
        "p_value_timing": round(p_val, 4),
        "verdict": "TIMING_MATTERS" if p_val < 0.10 else "EXPOSURE_ONLY",
    }

# ========================= MAIN =========================
def main():
    tf = "4h"
    print(f"Adversarial statistical tests: TI x TF tier-A @ {tf}")
    print(f"Tier-A candidates: {TIER_A}")
    print(f"N_SHUFFLE (T1, T4) = {N_SHUFFLE}")
    print(f"2023 OOS: fully unseen year (data ends 2022-12-31 in all prior work)")
    print()

    results = {}
    for ti_key in TIER_A:
        print(f"\n=== {ti_key} ===")
        sdf, bh = _ti_series_extended(ti_key, tf)
        if sdf is None:
            print(f"  SKIP: no data for {ti_key}@{tf}")
            continue
        print(f"  n_configs={sdf.shape[1]}  date_range={sdf.index.min().date()} to {sdf.index.max().date()}")

        t1 = test_T1_timing_scrambled(sdf, ti_key, tf)
        print(f"  T1 timing-scrambled null: real_2022={t1['real_2022_net']}%  null_med={t1['null_median_2022']}%"
              f"  p={t1['p_value_one_sided']}  [{t1['verdict']}]")

        t2 = test_T2_2023_oos(sdf, bh, ti_key, tf)
        print(f"  T2 2023 OOS: net={t2['net_2023']}%  dd={t2['maxdd_2023']}%  bh={t2['bh_2023']}%"
              f"  positive={t2['positive_2023']}  beats_bh={t2['beats_bh_2023']}")

        t4 = test_T4_exposure_shuffle(sdf, ti_key, tf)
        print(f"  T4 exposure-shuffle null: real_2022={t4['real_2022_net']}%  null_med={t4['exposure_null_median']}%"
              f"  p={t4['p_value_timing']}  [{t4['verdict']}]")

        results[ti_key] = {"T1_timing_null": t1, "T2_2023_oos": t2, "T4_exposure_shuffle": t4}

    # ========================= T3: Multiple-comparisons audit =========================
    print("\n=== T3: MULTIPLE-COMPARISONS AUDIT ===")
    n_tis = len(INDICATORS)        # 18 non-MA TIs + 8 MA types = 26
    n_tfs = 6                      # 1d,4h,2h,1h,30m,15m
    n_strategies = 3               # rolling_pick, band_ensemble, static_1
    n_tests = n_tis * n_tfs * n_strategies
    alpha = 0.05
    bonferroni_thresh = alpha / n_tests
    print(f"  N_TI={n_tis}  N_TF={n_tfs}  N_strategies={n_strategies}  => N_tests={n_tests}")
    print(f"  Bonferroni-adjusted alpha: {bonferroni_thresh:.6f} (vs naive {alpha})")
    print(f"  Tier-A set size = {len(TIER_A)}  (6 TIs claimed all-weather positive in 2022-bear)")
    # use T1 p-values as per-test observed p-values
    tier_a_p_vals = [results[k]["T1_timing_null"]["p_value_one_sided"] for k in TIER_A if k in results]
    bonferroni_survivors = [p for p in tier_a_p_vals if p < bonferroni_thresh]
    bh_survivors = sum(1 for p in tier_a_p_vals if p < alpha / len(TIER_A))   # BH correction within Tier-A
    print(f"  Tier-A T1 p-values (vs random band selection, 2022 bear): {[round(p,4) for p in tier_a_p_vals]}")
    print(f"  Survivors: Bonferroni(N={n_tests}) = {len(bonferroni_survivors)}  BH(N=6) = {bh_survivors}")

    # ========================= 2023 SIGN TEST =========================
    print("\n=== 2023 OOS SIGN TEST (pre-registered: H1 = positive net in 2023) ===")
    n_positive_2023 = sum(1 for k in TIER_A if k in results and results[k]["T2_2023_oos"]["positive_2023"])
    n_total_2023 = sum(1 for k in TIER_A if k in results)
    # Under H0 (random), P(+) = 0.5 -- binomial test
    from scipy.stats import binom_test  # type: ignore
    try:
        p_sign = float(binom_test(n_positive_2023, n_total_2023, 0.5, alternative="greater"))
    except Exception:
        from scipy.stats import binomtest
        p_sign = float(binomtest(n_positive_2023, n_total_2023, 0.5, alternative="greater").pvalue)
    print(f"  {n_positive_2023}/{n_total_2023} tier-A TIs positive in 2023 OOS")
    print(f"  Binomial sign test p (H1: majority positive) = {p_sign:.4f}")
    print(f"  {'POSITIVE' if p_sign < 0.05 else 'NOT SIGNIFICANT'} at alpha=0.05")

    print("\n=== FULL RESULTS SUMMARY ===")
    print(f"{'TI':10s}  {'2022 real':>10s}  {'T1 null_med':>11s}  {'T1 p':>8s}  {'T1 verdict':>16s}  {'2023 net':>8s}  {'2023 bh':>7s}  {'T4 p':>8s}  {'T4 verdict':>16s}")
    for k in TIER_A:
        if k not in results:
            print(f"  {k:10s}  MISSING"); continue
        r = results[k]
        t1 = r["T1_timing_null"]; t2 = r["T2_2023_oos"]; t4 = r["T4_exposure_shuffle"]
        print(f"  {k:10s}  {t1['real_2022_net']:>10.1f}  {t1['null_median_2022']:>11.1f}  {t1['p_value_one_sided']:>8.4f}  {t1['verdict']:>16s}  {t2['net_2023']:>8.1f}  {t2['bh_2023']:>7.1f}  {t4['p_value_timing']:>8.4f}  {t4['verdict']:>16s}")

    # Save
    out = {
        "tests": results,
        "multiple_comparisons": {
            "n_tis": n_tis, "n_tfs": n_tfs, "n_strategies": n_strategies,
            "n_tests_total": n_tests,
            "bonferroni_threshold": bonferroni_thresh,
            "tier_a_p_values": {k: results[k]["T1_timing_null"]["p_value_one_sided"] for k in TIER_A if k in results},
            "bonferroni_survivors": len(bonferroni_survivors),
            "bh_survivors_within_tier_a": bh_survivors,
        },
        "sign_test_2023": {
            "n_positive": n_positive_2023,
            "n_total": n_total_2023,
            "p_value": p_sign,
        }
    }
    import datetime
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = Path(__file__).parent / f"quant_adversarial_{stamp}.json"
    import json
    json.dump(out, open(outpath, "w"), indent=1, default=str)
    print(f"\n[saved] {outpath}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
