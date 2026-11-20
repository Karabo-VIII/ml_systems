"""src/strat/subdaily_oracle_probe.py -- Sub-daily move-capture oracle probe.

TASK (orc sub-daily cycle, 2026-06-20):
  At 4h and 1h:
  1. BUILD the oracle (ex-post best-K selection per 7d slice) = CEILING.
  2. COMPUTE the engine's honest aggregate capture:
       sum(realized) / sum(oracle-available) dollar-weighted, NOT mean(eng/oracle).
  3. COMPUTE identification AUC: does the ex-ante mover-score predict oracle-top-K membership?
     KEY QUESTION: is AUC_1h > AUC_4h > AUC_1d(0.549)?

COST HONESTY: 0.0024 taker RT deducted from realized ROI. Oracle is GROSS (ceiling unreachable net).
DEV ONLY (<= 2024-05-15, hard-walled by load_wide).
SAME-EXPOSURE shuffle control: random-K same {K, hold, slice-dates} -> null distribution.
REGIME STRATIFICATION: bull / chop / bear (SMA200-based, forward-period proxy) to check if
  sub-daily selection signal is regime-conditional (the 1d failure: chop/bear p05 < 0).

No emoji. No git commits.
RWYB: python -m strat.subdaily_oracle_probe (runs from crypto/src)
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from strat.fleet_lab import load_wide, BARS_PER_DAY, DEV_END

try:
    from sklearn.metrics import roc_auc_score
    _has_sklearn = True
except ImportError:
    _has_sklearn = False

COST = 0.0024   # taker RT
N_SLICES = 300  # per-TF slice sample
SEED = 42
K = 5           # top-K oracle/engine
CALENDAR_HOLD_DAYS = 7   # fixed hold horizon in calendar days


# -----------------------------------------------------------------------
# Oracle helpers
# -----------------------------------------------------------------------

def oracle_pick(C: pd.DataFrame, di: int, hold_bars: int, K: int) -> list[str]:
    """Oracle: ex-post best K assets by raw hold-period return (no cost, it's the ceiling)."""
    if di + hold_bars >= len(C.index):
        return []
    fwd = {}
    for s in C.columns:
        c0, c1 = C[s].iloc[di], C[s].iloc[di + hold_bars]
        if pd.notna(c0) and pd.notna(c1) and c0 > 0:
            fwd[s] = c1 / c0 - 1
    if len(fwd) < K:
        return []
    return sorted(fwd, key=lambda s: -fwd[s])[:K]


def oracle_gross_return(C: pd.DataFrame, di: int, hold_bars: int, K: int) -> float | None:
    """Mean return of oracle top-K (GROSS, no cost -- this is the ceiling)."""
    picks = oracle_pick(C, di, hold_bars, K)
    if not picks:
        return None
    return float(np.mean([C[s].iloc[di + hold_bars] / C[s].iloc[di] - 1 for s in picks]))


def engine_score(F: dict, di: int, feats: list[str]) -> pd.Series:
    """Ex-ante composite z-score (the engine's mover signal)."""
    parts = []
    for f in feats:
        row = F[f].iloc[di]
        z = (row - row.mean()) / (row.std() + 1e-12)
        parts.append(z.fillna(0.0))
    return sum(parts) / max(1, len(parts))


def engine_realized_return(C: pd.DataFrame, F: dict, di: int, hold_bars: int, K: int,
                            feats: list[str]) -> float | None:
    """Engine selects top-K by ex-ante score -> realized net return after cost."""
    if di + hold_bars >= len(C.index):
        return None
    sc = engine_score(F, di, feats)
    elig = sc.dropna()
    if len(elig) < K:
        return None
    picks = elig.sort_values(ascending=False).index[:K]
    fwd = []
    for s in picks:
        c0, c1 = C[s].iloc[di], C[s].iloc[di + hold_bars]
        if pd.notna(c0) and pd.notna(c1) and c0 > 0:
            fwd.append(c1 / c0 - 1)
    if not fwd:
        return None
    return float(np.mean(fwd)) - COST


def shuffle_realized_return(C: pd.DataFrame, di: int, hold_bars: int, K: int,
                             rng: np.random.Generator) -> float | None:
    """Same-exposure null: random-K from eligible assets, same hold."""
    if di + hold_bars >= len(C.index):
        return None
    elig = [s for s in C.columns if pd.notna(C[s].iloc[di]) and pd.notna(C[s].iloc[di + hold_bars])]
    if len(elig) < K:
        return None
    picks = rng.choice(elig, K, replace=False)
    fwd = [C[s].iloc[di + hold_bars] / C[s].iloc[di] - 1 for s in picks]
    return float(np.mean(fwd)) - COST


# -----------------------------------------------------------------------
# Regime label (coarse: based on 200-bar SMA of equal-weight index)
# -----------------------------------------------------------------------

def compute_regime(C: pd.DataFrame, n: int = 200) -> pd.Series:
    """SMA200-based regime label: 'bull' / 'bear' / 'chop' per bar index position."""
    ew = C.ffill().mean(axis=1)
    sma = ew.rolling(n, min_periods=n).mean()
    rel = ew / (sma + 1e-12)
    regime = pd.Series("chop", index=C.index)
    regime[rel > 1.05] = "bull"
    regime[rel < 0.95] = "bear"
    return regime


# -----------------------------------------------------------------------
# Identification AUC
# -----------------------------------------------------------------------

def compute_identification_auc(C: pd.DataFrame, F: dict, slices: list[int],
                                hold_bars: int, K: int, feats: list[str]) -> float | None:
    """AUC of ex-ante engine score for predicting oracle-top-K membership.

    Label = 1 if asset is in oracle top-K at (di, hold_bars). Score = engine composite z.
    Aggregate across all (di, asset) pairs on the slice set.
    """
    if not _has_sklearn:
        return None
    labels, scores = [], []
    for di in slices:
        oracle_k = set(oracle_pick(C, di, hold_bars, K))
        if not oracle_k:
            continue
        sc = engine_score(F, di, feats)
        for s in C.columns:
            v = sc.get(s, np.nan) if hasattr(sc, "get") else (sc[s] if s in sc.index else np.nan)
            if pd.isna(v):
                continue
            labels.append(1 if s in oracle_k else 0)
            scores.append(float(v))
    if len(labels) < 50 or sum(labels) < 5:
        return None
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return None


# -----------------------------------------------------------------------
# Block-bootstrap p05
# -----------------------------------------------------------------------

def block_bootstrap_p05(excess: np.ndarray, n_boot: int = 2000, block: int = 20,
                         seed: int = 0) -> float:
    """Moving-block bootstrap p05 of mean excess (engine - shuffle)."""
    rng = np.random.default_rng(seed)
    n = len(excess)
    if n < block:
        return float(np.percentile(excess, 5))
    means = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n // block + 1)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]
        means.append(excess[idx].mean())
    return float(np.percentile(means, 5))


# -----------------------------------------------------------------------
# Main probe
# -----------------------------------------------------------------------

def run_tf_probe(tf: str, feats: list[str] | None = None) -> dict:
    bpd = BARS_PER_DAY[tf]
    hold_bars = CALENDAR_HOLD_DAYS * bpd
    print(f"\n=== TF={tf} ({bpd} bars/day, hold={hold_bars} bars = {CALENDAR_HOLD_DAYS}d) ===")
    t0 = time.time()

    # ----- load data -----
    min_bars = max(400, 200 * bpd)
    print(f"  loading DEV data (n=50, min_bars={min_bars})...")
    lab = load_wide(n=50, min_bars=min_bars, tf=tf)
    C, F, syms = lab["C"], lab["F"], lab["syms"]
    print(f"  {len(syms)} assets, range {C.index.min()} -> {C.index.max()}")

    if feats is None:
        # Default: broad composite of all available TI + chimera features
        feats = [k for k in F.keys() if k not in ("dvol",)]   # dvol noisy; include all TIs + chimera
    print(f"  features: {feats}")

    # ----- slice dates -----
    rng = np.random.default_rng(SEED)
    warm = max(40, 30 * bpd)   # warmup bars for indicators
    valid = [i for i in range(warm, len(C.index) - hold_bars - 1)]
    n_sl = min(N_SLICES, len(valid))
    slices = sorted(rng.choice(valid, n_sl, replace=False))
    print(f"  {n_sl} slices from {n_sl}/{len(valid)} valid bars")

    # ----- regime labels at each slice bar -----
    regime = compute_regime(C)
    regime_at = [str(regime.iloc[di]) if di < len(regime) else "chop" for di in slices]
    regime_counts = {r: regime_at.count(r) for r in ("bull", "chop", "bear")}
    print(f"  regime distribution: {regime_counts}")

    # ----- Oracle: compute ceiling -----
    oracle_gross = []
    for di in slices:
        g = oracle_gross_return(C, di, hold_bars, K)
        oracle_gross.append(g)
    oracle_gross_arr = np.array([x for x in oracle_gross if x is not None])
    print(f"  ORACLE (gross ceiling, K={K}): mean={100*np.nanmean(oracle_gross_arr):.2f}%  "
          f"n={len(oracle_gross_arr)}")

    # ----- Engine: ex-ante selection -----
    engine_net = []
    for di in slices:
        r = engine_realized_return(C, F, di, hold_bars, K, feats)
        engine_net.append(r)
    engine_arr = np.array([x for x in engine_net if x is not None])
    valid_mask = [x is not None for x in engine_net]
    print(f"  ENGINE (net, K={K}): mean={100*np.nanmean(engine_arr):.2f}%  n={len(engine_arr)}")

    # ----- Shuffle null: same-exposure -----
    N_SHUFFLE = 500
    shuffle_means = []
    for _ in range(N_SHUFFLE):
        sr = [shuffle_realized_return(C, slices[i], hold_bars, K, rng)
              for i in range(len(slices)) if valid_mask[i]]
        sr_arr = np.array([x for x in sr if x is not None])
        if len(sr_arr) > 0:
            shuffle_means.append(sr_arr.mean())
    shuffle_mean_dist = np.array(shuffle_means)
    null_mean = float(np.mean(shuffle_mean_dist))
    null_p95 = float(np.percentile(shuffle_mean_dist, 95))
    print(f"  SHUFFLE NULL (same-exposure): mean={100*null_mean:.2f}%  p95={100*null_p95:.2f}%")

    # Excess over shuffle
    eng_mean = float(np.nanmean(engine_arr))
    excess_mean = eng_mean - null_mean
    z_score = (eng_mean - null_mean) / (float(np.std(shuffle_mean_dist)) + 1e-12)
    print(f"  ENGINE vs SHUFFLE: excess={100*excess_mean:.3f}pp  z={z_score:.2f}")

    # Per-slice excess for bootstrap
    # align: use valid slices for both
    eng_valid = np.array([x for x in engine_net if x is not None])
    # shuffle per-slice (single run for bootstrap input)
    shuf_per_slice = []
    rng2 = np.random.default_rng(SEED + 1)
    for i, di in enumerate(slices):
        if not valid_mask[i]:
            continue
        sr = shuffle_realized_return(C, di, hold_bars, K, rng2)
        shuf_per_slice.append(sr if sr is not None else 0.0)
    shuf_arr = np.array(shuf_per_slice)
    if len(eng_valid) == len(shuf_arr):
        excess_arr = eng_valid - shuf_arr
        p05 = block_bootstrap_p05(excess_arr, n_boot=2000, block=max(10, bpd * 3))
        print(f"  BLOCK-BOOTSTRAP p05 (excess over shuffle): {100*p05:.3f}pp")
    else:
        p05 = None
        print(f"  BLOCK-BOOTSTRAP: length mismatch ({len(eng_valid)} vs {len(shuf_arr)}), skip")

    # ----- Aggregate capture rate (dollar-weighted) -----
    # sum(engine_net * entry_price) / sum(oracle_gross * entry_price)
    # Use equal entry price => sum(eng) / sum(oracle)  (dollar-weight by count is equivalent)
    valid_pairs = [(oracle_gross[i], engine_net[i])
                   for i in range(len(slices))
                   if oracle_gross[i] is not None and engine_net[i] is not None]
    if valid_pairs:
        sum_oracle = sum(p[0] for p in valid_pairs)
        sum_engine = sum(p[1] for p in valid_pairs)
        capture_rate = (sum_engine / sum_oracle) if abs(sum_oracle) > 1e-9 else None
        print(f"  AGGREGATE CAPTURE RATE: realized/oracle = {100*capture_rate:.1f}%  "
              f"(sum_eng={100*sum_engine:.1f}pp, sum_oracle={100*sum_oracle:.1f}pp)")
    else:
        capture_rate = None

    # ----- Identification AUC -----
    auc = compute_identification_auc(C, F, slices, hold_bars, K, feats)
    if auc is not None:
        print(f"  IDENTIFICATION AUC (oracle-top-K membership): {auc:.4f}  "
              f"  (1d baseline=0.549)")
    else:
        print(f"  IDENTIFICATION AUC: sklearn not available or insufficient data")

    # ----- Regime-stratified engine vs shuffle -----
    print(f"\n  REGIME STRATIFICATION (engine net mean):")
    for reg in ("bull", "chop", "bear"):
        reg_idx = [i for i, r in enumerate(regime_at)
                   if r == reg and i < len(engine_net) and engine_net[i] is not None]
        if not reg_idx:
            print(f"    {reg:5}: (no data)")
            continue
        e_reg = np.array([engine_net[i] for i in reg_idx])
        # shuffle null for same slice set
        rng3 = np.random.default_rng(SEED + 99)
        s_reg = []
        for i in reg_idx:
            sr = shuffle_realized_return(C, slices[i], hold_bars, K, rng3)
            s_reg.append(sr if sr is not None else 0.0)
        s_reg = np.array(s_reg)
        excess_reg = e_reg - s_reg
        p05_reg = block_bootstrap_p05(excess_reg, n_boot=1000,
                                       block=max(5, bpd * 2)) if len(excess_reg) >= 10 else None
        frac_pos = float(np.mean(excess_reg > 0))
        p05_str = f"{100*p05_reg:.3f}pp" if p05_reg is not None else "n/a"
        print(f"    {reg:5}: n={len(e_reg):4d}  engine={100*e_reg.mean():+.2f}%  "
              f"shuffle={100*s_reg.mean():+.2f}%  "
              f"excess={100*excess_reg.mean():+.3f}pp  p05={p05_str}  frac>0={frac_pos:.2f}")

    # ----- EW buy-hold (reference) -----
    ew_returns = []
    for di in slices:
        if di + hold_bars >= len(C.index):
            continue
        vals = [(C[s].iloc[di + hold_bars] / C[s].iloc[di] - 1)
                for s in C.columns
                if pd.notna(C[s].iloc[di]) and pd.notna(C[s].iloc[di + hold_bars])]
        if vals:
            ew_returns.append(np.mean(vals))
    ew_arr = np.array(ew_returns)
    print(f"\n  EW BUY-HOLD (reference, gross): mean={100*np.nanmean(ew_arr):.2f}%")

    elapsed = time.time() - t0
    print(f"  elapsed: {elapsed:.1f}s")

    result = {
        "tf": tf,
        "bpd": bpd,
        "hold_bars": hold_bars,
        "n_assets": len(syms),
        "n_slices": n_sl,
        "regime_counts": regime_counts,
        "oracle_gross_mean_pct": round(100 * float(np.nanmean(oracle_gross_arr)), 3),
        "engine_net_mean_pct": round(100 * float(np.nanmean(engine_arr)), 3),
        "shuffle_null_mean_pct": round(100 * null_mean, 3),
        "shuffle_null_p95_pct": round(100 * null_p95, 3),
        "excess_over_shuffle_pp": round(100 * excess_mean, 4),
        "excess_z": round(z_score, 3),
        "block_bootstrap_p05_pp": round(100 * p05, 4) if p05 is not None else None,
        "aggregate_capture_rate_pct": round(100 * capture_rate, 2) if capture_rate is not None else None,
        "identification_auc": round(auc, 4) if auc is not None else None,
        "ew_buyhold_gross_mean_pct": round(100 * float(np.nanmean(ew_arr)), 3),
        "feats": feats,
        "elapsed_s": round(elapsed, 1),
    }
    return result


def main():
    print("SUB-DAILY MOVE-CAPTURE ORACLE PROBE")
    print(f"DEV wall: <= {DEV_END}, K={K}, hold={CALENDAR_HOLD_DAYS}d calendar, cost={COST} taker RT")
    print(f"n_slices={N_SLICES}, n_bootstrap=2000, seed={SEED}")

    results = {}

    # ---- 4h probe ----
    results["4h"] = run_tf_probe("4h")

    # ---- 1h probe ----
    results["1h"] = run_tf_probe("1h")

    # ---- Cross-TF summary ----
    print("\n" + "=" * 70)
    print("CROSS-TF SUMMARY")
    print(f"{'Metric':45} {'4h':>10} {'1h':>10}")
    print("-" * 70)
    for k, label in [
        ("oracle_gross_mean_pct", "Oracle gross ceiling (%)"),
        ("engine_net_mean_pct", "Engine net selection (%)"),
        ("shuffle_null_mean_pct", "Shuffle null mean (%)"),
        ("excess_over_shuffle_pp", "Excess over shuffle (pp)"),
        ("excess_z", "Excess z-score"),
        ("block_bootstrap_p05_pp", "Block-bootstrap p05 (pp)"),
        ("aggregate_capture_rate_pct", "Aggregate capture rate (%)"),
        ("identification_auc", "Identification AUC"),
        ("ew_buyhold_gross_mean_pct", "EW buy-hold gross (%)"),
    ]:
        v4 = results.get("4h", {}).get(k)
        v1 = results.get("1h", {}).get(k)
        print(f"  {label:43} {str(v4):>10} {str(v1):>10}")

    print(f"\n  1d AUC baseline (from prior campaign): 0.549")
    for tf in ("4h", "1h"):
        auc = results.get(tf, {}).get("identification_auc")
        if auc is not None:
            delta = auc - 0.549
            print(f"  {tf} AUC={auc:.4f}  vs 1d: {'+' if delta>0 else ''}{delta:.4f} "
                  f"({'sub-daily BETTER' if delta > 0.01 else 'sub-daily FLAT/WORSE' if delta < -0.01 else 'sub-daily MARGINAL'})")

    # Save
    out_path = Path(__file__).resolve().parents[2] / "runs" / "strat" / "subdaily_oracle_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
