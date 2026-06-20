"""src/strat/cycle3_makeorbreak.py -- CYCLE-3 MAKE-OR-BREAK.

PRE-REGISTERED config: mom14, K=5, rebal=3, gate=per-asset-SMA200.
NO config sweep (the sweep DoF is what inflated/killed prior p-values).

Two honest tests:
  (A) MOVING-BLOCK BOOTSTRAP -- block length 21 trading days, 2000 resamples,
      on the daily (strategy - matched-exposure-random) return differential,
      OOS window 2023-01-01 to 2026-06-01.
      Reports p05 of the bootstrap distribution of mean(differential) and the
      one-sided p-value (fraction of bootstrap means <= 0, i.e. null is better).

  (B) LEAVE-ONE-YEAR-OUT WALK-FORWARD: for each test year in {2021, 2023, 2024, 2025},
      the pre-registered r3 config vs. a single random-gated-5 null drawn fresh per year.
      Reports whether the strategy compound > null compound and the margin.
      Simple comparison (no sweep): how many years does the strategy beat the null median?

VERDICT: REAL-ALPHA or DEAD.

RWYB: python -m strat.cycle3_makeorbreak
No emoji (cp1252). Does NOT git commit.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.mover_lab as ml

COST = ml.COST

# ---------------------------------------------------------------------------
# PRIMITIVES (self-contained; mirrors cycle2_referee but extended for block-boot)
# ---------------------------------------------------------------------------

def book_returns(W, ind):
    """Daily book returns (lag-1, taker cost). Returns Series."""
    R = ind["R"].reindex(index=W.index, columns=W.columns).fillna(0.0)
    pos = W.shift(1).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs()).sum(axis=1)
    bret = (pos * R).sum(axis=1) - turn * (COST / 2.0)
    return bret


def window_slice(bret, win_s, win_e):
    """Return the daily returns Series within [win_s, win_e)."""
    s = pd.Timestamp(win_s); e = pd.Timestamp(win_e)
    return bret[(bret.index >= s) & (bret.index < e)]


def comp_pct(arr):
    """Compound return in percent from a numpy array of daily returns."""
    return (np.prod(1.0 + arr) - 1.0) * 100.0


def maxdd_pct(arr):
    eq = np.cumprod(1.0 + arr); pk = np.maximum.accumulate(eq)
    return float(((eq - pk) / pk).min() * 100.0)


def random_gated_k(ind, K, rebal, rng):
    """Random-K pick among per-asset-SMA200-gated assets, EW, carried between rebals."""
    C = ind["C"]; g = ind["gate"]
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns); last = -999
    cols = list(C.columns)
    for i, d in enumerate(C.index):
        if i - last >= rebal:
            elig = [s for s in cols if bool(g.loc[d, s])]
            if elig:
                k = min(K, len(elig))
                pick = list(rng.choice(elig, size=k, replace=False))
                W.loc[d, :] = 0.0
                for s in pick:
                    W.loc[d, s] = 1.0 / k
            last = i
        elif i > 0:
            W.iloc[i] = W.iloc[i - 1]
    return W


# ---------------------------------------------------------------------------
# (A) MOVING-BLOCK BOOTSTRAP
# ---------------------------------------------------------------------------

def moving_block_bootstrap(diff_daily, block_len=21, n_boot=2000, rng=None):
    """
    Resample the daily differential series (strategy - null) using a moving-block bootstrap.
    Returns bootstrap distribution of the mean daily differential.
    One-sided p-value: fraction of bootstrap means <= 0 (null is as good or better).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    x = diff_daily.to_numpy()
    n = len(x)
    # Build all possible blocks of length block_len
    n_blocks_needed = int(np.ceil(n / block_len))
    max_start = n - block_len  # inclusive max start index
    if max_start <= 0:
        raise ValueError(f"Series length {n} too short for block_len {block_len}")

    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks_needed)
        resampled = np.concatenate([x[s : s + block_len] for s in starts])[:n]
        boot_means[b] = resampled.mean()

    true_mean = float(x.mean())
    p_val = float(np.mean(boot_means <= 0.0))  # one-sided: frac where bootstrap mean <= 0
    p05 = float(np.percentile(boot_means, 5))
    p95 = float(np.percentile(boot_means, 95))
    return {
        "true_mean_daily_diff": true_mean,
        "true_ann_diff_pct": true_mean * 365 * 100,
        "boot_p05_daily": p05,
        "boot_p95_daily": p95,
        "boot_p05_ann_pct": p05 * 365 * 100,
        "p_value_onesided": p_val,
        "n_boot": n_boot,
        "block_len": block_len,
        "n_days": n,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("=" * 72)
    print("CYCLE-3 MAKE-OR-BREAK: mom14 K=5 rebal=3 per-asset-SMA200-gate")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # LOAD (2020-2026 for full context, OOS = 2023+)
    # -----------------------------------------------------------------------
    print("\n[1] Loading data 2020-01-01 -> 2026-06-01 ...")
    ind = ml.load("2020-01-01", "2026-06-01")
    assets = list(ind["C"].columns)
    print(f"    Assets: {len(assets)} -> {assets}")
    print(f"    Date range: {ind['C'].index[0].date()} to {ind['C'].index[-1].date()}")

    # -----------------------------------------------------------------------
    # PRE-REGISTERED STRATEGY
    # -----------------------------------------------------------------------
    print("\n[2] Building pre-registered strategy: mom14 K=5 r=3 per-asset-gate ...")
    score = ind["mom14"]
    W_strat = ml.topk_weight(score, ind, K=5, gate=True, rebal=3)
    bret_strat = book_returns(W_strat, ind)

    # -----------------------------------------------------------------------
    # FULL-CYCLE SUMMARY (context; IS is 2020-2022, OOS is 2023-2026)
    # -----------------------------------------------------------------------
    print("\n[3] Full-cycle performance context:")
    print(f"{'Period':<12} {'Strategy %':>12} {'MaxDD %':>10}")
    windows = [
        ("2020", "2020-01-01", "2021-01-01"),
        ("2021", "2021-01-01", "2022-01-01"),
        ("2022", "2022-01-01", "2023-01-01"),
        ("IS(20-22)", "2020-01-01", "2023-01-01"),
        ("2023", "2023-01-01", "2024-01-01"),
        ("2024", "2024-01-01", "2025-01-01"),
        ("2025", "2025-01-01", "2026-01-01"),
        ("OOS(23-26)", "2023-01-01", "2026-06-01"),
        ("FULL", "2020-01-01", "2026-06-01"),
    ]
    for label, ws, we in windows:
        sl = window_slice(bret_strat, ws, we)
        if len(sl) < 3:
            print(f"  {label:<12} {'--':>12} {'--':>10}")
            continue
        c = comp_pct(sl.to_numpy())
        dd = maxdd_pct(sl.to_numpy())
        print(f"  {label:<12} {c:>12.1f} {dd:>10.1f}")

    # -----------------------------------------------------------------------
    # MATCHED RANDOM REFERENCE: median of N_NULL random-gated-5 runs
    # -----------------------------------------------------------------------
    N_NULL = 500
    print(f"\n[4] Building matched-exposure random-gated-5 null ({N_NULL} seeds) ...")
    null_bretes = []
    rng_null = np.random.default_rng(99999)
    for seed in range(N_NULL):
        rng_s = np.random.default_rng(seed)
        Wn = random_gated_k(ind, K=5, rebal=3, rng=rng_s)
        null_bretes.append(book_returns(Wn, ind))
    # Null median compound over OOS window
    oos_s, oos_e = "2023-01-01", "2026-06-01"
    null_oos_comps = []
    for br in null_bretes:
        sl = window_slice(br, oos_s, oos_e)
        null_oos_comps.append(comp_pct(sl.to_numpy()) if len(sl) > 2 else None)
    null_oos_comps = [c for c in null_oos_comps if c is not None]
    strat_oos = comp_pct(window_slice(bret_strat, oos_s, oos_e).to_numpy())
    print(f"  OOS(2023-2026) strategy:   {strat_oos:>8.1f}%")
    print(f"  OOS(2023-2026) null median:{np.median(null_oos_comps):>8.1f}%")
    print(f"  OOS(2023-2026) null p05:   {np.percentile(null_oos_comps, 5):>8.1f}%")
    print(f"  OOS(2023-2026) null p95:   {np.percentile(null_oos_comps, 95):>8.1f}%")
    p_raw = float(np.mean(np.array(null_oos_comps) >= strat_oos))
    print(f"  Naive p-value (frac null >= strat): {p_raw:.3f}  [NOTE: iid shuffle — underestimates variance]")

    # -----------------------------------------------------------------------
    # (A) MOVING-BLOCK BOOTSTRAP on the daily DIFFERENTIAL (strategy - null_median)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("TEST A: MOVING-BLOCK BOOTSTRAP (block=21 days, 2000 resamples)")
    print("=" * 72)

    # Build the differential daily series: strategy return - null_median_return
    # Use the MEDIAN of N_NULL nulls as the reference null series
    null_mat = pd.DataFrame({i: null_bretes[i] for i in range(N_NULL)})
    null_med_series = null_mat.median(axis=1)

    oos_mask = (bret_strat.index >= pd.Timestamp(oos_s)) & (bret_strat.index < pd.Timestamp(oos_e))
    diff_oos = (bret_strat - null_med_series)[oos_mask]

    print(f"  OOS daily differential: mean={diff_oos.mean()*100:.4f}% | std={diff_oos.std()*100:.4f}%")
    print(f"  OOS days in window: {len(diff_oos)}")

    boot_result = moving_block_bootstrap(diff_oos, block_len=21, n_boot=2000, rng=np.random.default_rng(777))

    print(f"\n  --- Block Bootstrap Results ---")
    print(f"  True mean daily differential:  {boot_result['true_mean_daily_diff']*100:>+8.4f}%/day")
    print(f"  True annualised differential:  {boot_result['true_ann_diff_pct']:>+8.2f}%/yr")
    print(f"  Bootstrap p05 (annualised):    {boot_result['boot_p05_ann_pct']:>+8.2f}%/yr")
    print(f"  Bootstrap p95 (annualised):    {boot_result['boot_p95_daily']*365*100:>+8.2f}%/yr")
    print(f"  One-sided p-value (H0: diff<=0): {boot_result['p_value_onesided']:.4f}")
    print(f"  n_boot={boot_result['n_boot']}, block_len={boot_result['block_len']}, n_days={boot_result['n_days']}")

    p_boot = boot_result["p_value_onesided"]
    boot_p05_ann = boot_result["boot_p05_ann_pct"]
    if p_boot < 0.05:
        boot_verdict = f"PASS (p={p_boot:.4f} < 0.05)"
    else:
        boot_verdict = f"FAIL (p={p_boot:.4f} >= 0.05)"
    print(f"\n  TEST A VERDICT: {boot_verdict}")

    # -----------------------------------------------------------------------
    # (B) LEAVE-ONE-YEAR-OUT WALK-FORWARD
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("TEST B: LEAVE-ONE-YEAR-OUT WALK-FORWARD")
    print("=" * 72)
    print("  Test years: 2021, 2023, 2024, 2025")
    print("  Comparison: strategy compound vs null-median compound (N=500 fixed-seed nulls per year)")
    print(f"  {'Year':<6} {'Strategy%':>10} {'Null_Med%':>10} {'Null_P05%':>10} {'Null_P95%':>10} {'p(naive)':>9} {'Beat?':>6}")
    print("  " + "-" * 63)

    wf_years = [2021, 2023, 2024, 2025]
    wf_results = []
    for y in wf_years:
        ws_y = f"{y}-01-01"
        we_y = f"{y+1}-01-01"
        s_sl = window_slice(bret_strat, ws_y, we_y)
        s_comp = comp_pct(s_sl.to_numpy()) if len(s_sl) > 2 else None

        # Per-year null compounds (reuse the same 500 null runs)
        yr_nulls = []
        for br in null_bretes:
            sl = window_slice(br, ws_y, we_y)
            if len(sl) > 2:
                yr_nulls.append(comp_pct(sl.to_numpy()))
        yr_nulls = np.array(yr_nulls)

        if s_comp is None or len(yr_nulls) == 0:
            print(f"  {y:<6} {'--':>10} {'--':>10} {'--':>10} {'--':>10} {'--':>9} {'--':>6}")
            continue

        null_med_y = float(np.median(yr_nulls))
        null_p05_y = float(np.percentile(yr_nulls, 5))
        null_p95_y = float(np.percentile(yr_nulls, 95))
        p_naive_y = float(np.mean(yr_nulls >= s_comp))
        beat = "YES" if s_comp > null_med_y else "no"

        print(f"  {y:<6} {s_comp:>10.1f} {null_med_y:>10.1f} {null_p05_y:>10.1f} {null_p95_y:>10.1f} {p_naive_y:>9.3f} {beat:>6}")
        wf_results.append({
            "year": y, "strat": s_comp, "null_med": null_med_y,
            "null_p05": null_p05_y, "null_p95": null_p95_y,
            "p_naive": p_naive_y, "beat": beat == "YES",
        })

    n_beat = sum(1 for r in wf_results if r["beat"])
    n_total = len(wf_results)
    print(f"\n  Walk-forward beat count: {n_beat}/{n_total} years strategy > null median")

    # Breakdown of which years drive the signal
    beat_years = [r["year"] for r in wf_results if r["beat"]]
    miss_years = [r["year"] for r in wf_results if not r["beat"]]
    print(f"  Beat years: {beat_years}")
    print(f"  Miss years: {miss_years}")

    if n_beat >= 3:
        wf_verdict = f"CONSISTENT ({n_beat}/{n_total} years beat null median)"
    elif n_beat == 2:
        wf_verdict = f"MIXED ({n_beat}/{n_total} years) -- check if driven by single year"
    else:
        wf_verdict = f"INCONSISTENT ({n_beat}/{n_total} years)"
    print(f"\n  TEST B VERDICT: {wf_verdict}")

    # -----------------------------------------------------------------------
    # FINAL VERDICT
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("FINAL VERDICT")
    print("=" * 72)
    print(f"  Test A (block bootstrap, OOS 2023-2026): {boot_verdict}")
    print(f"  Test B (walk-forward, {n_beat}/{n_total} years):            {wf_verdict}")
    print()

    # Decision logic
    test_a_pass = p_boot < 0.05
    test_b_pass = n_beat >= 3  # at least 3 of 4 years

    if test_a_pass and test_b_pass:
        final = "REAL-ALPHA: Both tests pass. Selection edge is real and consistent."
    elif test_a_pass and not test_b_pass:
        final = "CONDITIONAL: Block bootstrap passes but walk-forward inconsistent -- edge concentrated in subset of years. PROBABLY DEAD."
    elif not test_a_pass and test_b_pass:
        final = "WEAK: Walk-forward consistent but block bootstrap fails -- variance too high for aggregate claim. PROBABLY DEAD."
    else:
        final = "DEAD: Both tests fail. No selection edge over matched-exposure random-gated-5."

    print(f"  >> {final}")
    print()

    # Extra: check if 2024 is the sole driver
    r24 = next((r for r in wf_results if r["year"] == 2024), None)
    others = [r for r in wf_results if r["year"] != 2024]
    if r24:
        others_beat = sum(1 for r in others if r["beat"])
        print(f"  Single-year driver check: 2024 {'beat' if r24['beat'] else 'missed'} | "
              f"remaining years {others_beat}/{len(others)} beat null median")
        if r24 and r24["beat"] and others_beat == 0:
            print("  WARNING: Edge is 2024-only -- concentrated single-year, NOT a durable signal.")
        elif not r24["beat"]:
            print("  NOTE: 2024 is NOT driving the result (it actually missed).")

    elapsed = time.time() - t0
    print(f"\n[done in {elapsed:.1f}s]")


if __name__ == "__main__":
    main()
