"""SmartCandidateGenerator -- replaces brute-force adjacent-period MA grid.

The brute-force grid (10K cells of (fast, slow) with fast in 1..100, slow > fast)
produces adjacent-period "winners" that measure the same signal (e.g., SMA(28,31)
vs SMA(29,32) correlate ~0.85+). Per the SD framework (doc #23), the right
substrate is:

  - Fibonacci-spaced: (1,2), (2,3), (3,5), (5,8), (8,13), (13,21), (21,34), (34,55), (55,89)
  - Golden-ratio: slow = phi * fast (phi = 1.618), capped at 100
  - Log-spaced: geometric progression covering decades
  - Adjacent-period anchor (sparse): a few (n, n+1) pairs as comparison

Then a decorrelation filter drops pairs whose signal vector correlates > 0.85
with another pair already kept.

This module is data-light: it just returns candidate (fast, slow) tuples.
The decorrelation step needs a reference close-series; we provide an
empirical_decorrelate(closes, candidates) helper that filters on a sample.

CONTRACT
  inputs: int max_period, int max_candidates
  outputs: list[(int fast, int slow)] sorted (fast, slow)
  invariants: fast < slow; fast >= 1; slow <= max_period; no duplicates
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

PHI = (1 + 5 ** 0.5) / 2  # 1.6180339887


__contract__ = {
    "kind": "candidate_generator",
    "owner": "strat-layer/smart_discovery",
    "inputs": "max_period (int), max_candidates (int)",
    "outputs": "list[(fast, slow)] tuples",
    "invariants": [
        "fast < slow",
        "fast >= 1",
        "slow <= max_period",
        "no duplicate pairs",
        "deterministic for given inputs",
    ],
}


def fibonacci_pairs(max_period: int = 100) -> list[tuple[int, int]]:
    """Adjacent Fibonacci-spaced pairs: (F_n, F_{n+1}) and a few (F_n, F_{n+2})."""
    fib = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    fib = [f for f in fib if f <= max_period]
    pairs = []
    for i in range(len(fib) - 1):
        pairs.append((fib[i], fib[i + 1]))
    for i in range(len(fib) - 2):
        if fib[i + 2] <= max_period:
            pairs.append((fib[i], fib[i + 2]))
    return pairs


def golden_ratio_pairs(max_period: int = 100, min_fast: int = 2) -> list[tuple[int, int]]:
    """slow = round(phi * fast). Stride fast 2,3,5,8,13,21,34."""
    fasts = [2, 3, 5, 8, 13, 21, 34, 55]
    pairs = []
    for f in fasts:
        if f < min_fast:
            continue
        s = int(round(PHI * f))
        if s > f and s <= max_period:
            pairs.append((f, s))
        # phi^2 ~= 2.618
        s2 = int(round(PHI * PHI * f))
        if s2 > f and s2 <= max_period:
            pairs.append((f, s2))
    return pairs


def log_spaced_pairs(max_period: int = 100, n_decades: int = 4) -> list[tuple[int, int]]:
    """Geometric pairs across decades. fast in {1, 3, 10, 30}; slow = fast * {2, 3, 5}."""
    fasts = [1, 2, 3, 5, 7, 10, 14, 21, 30, 50]
    fasts = [f for f in fasts if f < max_period]
    pairs = []
    for f in fasts:
        for mult in [2, 3, 5]:
            s = f * mult
            if s > f and s <= max_period:
                pairs.append((f, s))
    return pairs


def adjacent_anchor_pairs(max_period: int = 100) -> list[tuple[int, int]]:
    """A few adjacent-period pairs as sanity comparison (the family the
    brute-force grid picked). Sparse coverage."""
    pairs = []
    for f in [3, 7, 14, 20, 28, 40, 55, 75, 90]:
        if f + 1 <= max_period:
            pairs.append((f, f + 1))
    return pairs


def cousin_pairs(max_period: int = 100) -> list[tuple[int, int]]:
    """Slightly-spaced "cousin" pairs: (n, n+2), (n, n+3) at sparse n values.
    Captures the fast-cross-slow concept without adjacent-period collinearity."""
    pairs = []
    for f in [3, 5, 8, 13, 21, 34]:
        for delta in [2, 3, 4]:
            s = f + delta
            if s <= max_period:
                pairs.append((f, s))
    return pairs


def generate_raw_candidates(max_period: int = 100) -> list[tuple[int, int]]:
    """Union of all four families, deduplicated."""
    raw = set()
    raw.update(fibonacci_pairs(max_period))
    raw.update(golden_ratio_pairs(max_period))
    raw.update(log_spaced_pairs(max_period))
    raw.update(adjacent_anchor_pairs(max_period))
    raw.update(cousin_pairs(max_period))
    # Filter to fast < slow, both in [1, max_period]
    valid = [(f, s) for (f, s) in raw if f >= 1 and s > f and s <= max_period]
    return sorted(set(valid))


def empirical_decorrelate(
    closes: np.ndarray,
    candidates: list[tuple[int, int]],
    corr_threshold: float = 0.85,
    ma_type: str = "SMA",
) -> list[tuple[int, int]]:
    """Given a reference close series, compute the signal vector for each
    candidate (binary fire-or-not) and drop pairs whose signal vector
    correlates >= corr_threshold with an already-kept pair.

    Greedy: process in order of (fast, slow); keep the first one, then check
    each subsequent against the kept set.
    """
    if len(closes) < 200:
        return candidates  # too short to decorrelate meaningfully

    def sig_vec(fast: int, slow: int) -> np.ndarray:
        s = pd.Series(closes)
        if ma_type == "SMA":
            mf = s.rolling(fast).mean().values
            ml = s.rolling(slow).mean().values
        else:
            mf = s.ewm(span=fast, adjust=False).mean().values
            ml = s.ewm(span=slow, adjust=False).mean().values
        # Cross-up event: mf[i-1] <= ml[i-1] AND mf[i] > ml[i]
        cross = np.zeros(len(closes), dtype=np.int8)
        cross[1:] = ((mf[1:] > ml[1:]) & (mf[:-1] <= ml[:-1])).astype(np.int8)
        return cross

    kept: list[tuple[int, int]] = []
    kept_vecs: list[np.ndarray] = []
    for (f, s) in candidates:
        v = sig_vec(f, s)
        if v.sum() < 5:
            continue  # too few fires to be useful
        drop = False
        for kv in kept_vecs:
            # Correlation of two binary vectors
            if v.std() < 1e-9 or kv.std() < 1e-9:
                continue
            c = np.corrcoef(v, kv)[0, 1]
            if c >= corr_threshold:
                drop = True
                break
        if not drop:
            kept.append((f, s))
            kept_vecs.append(v)
    return kept


def smart_grid(
    max_period: int = 100,
    reference_closes: np.ndarray | None = None,
    corr_threshold: float = 0.85,
    ma_type: str = "SMA",
) -> list[tuple[int, int]]:
    """Top-level: generate raw, then decorrelate if reference series given."""
    raw = generate_raw_candidates(max_period)
    if reference_closes is None:
        return raw
    return empirical_decorrelate(reference_closes, raw, corr_threshold, ma_type)


def smoke_test():
    """Self-test: print candidate sets + sanity check."""
    print("=== Fibonacci pairs ===")
    fib = fibonacci_pairs()
    print(f"  count: {len(fib)}")
    print(f"  pairs: {fib}")

    print("\n=== Golden ratio pairs ===")
    gold = golden_ratio_pairs()
    print(f"  count: {len(gold)}")
    print(f"  pairs: {gold}")

    print("\n=== Log-spaced pairs ===")
    log_p = log_spaced_pairs()
    print(f"  count: {len(log_p)}")
    print(f"  pairs: {log_p}")

    print("\n=== Adjacent anchor (sparse) ===")
    adj = adjacent_anchor_pairs()
    print(f"  count: {len(adj)}")
    print(f"  pairs: {adj}")

    print("\n=== Cousin pairs ===")
    cous = cousin_pairs()
    print(f"  count: {len(cous)}")
    print(f"  pairs: {cous}")

    raw = generate_raw_candidates()
    print(f"\n=== RAW union ===")
    print(f"  count: {len(raw)}")

    # Decorrelation test with synthetic data
    rng = np.random.default_rng(42)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 1000)))
    kept = empirical_decorrelate(closes, raw, 0.85, "SMA")
    print(f"\n=== After decorrelation (synthetic 1000-bar closes, threshold 0.85) ===")
    print(f"  kept: {len(kept)} / {len(raw)} ({100*len(kept)/len(raw):.1f}%)")
    print(f"  pairs: {kept}")


if __name__ == "__main__":
    smoke_test()
