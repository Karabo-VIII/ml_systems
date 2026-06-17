"""
Transfer Entropy — Schreiber 2000, discrete binned estimator.

Definition:
  TE(X → Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})

Interpretation: bits of additional information that past X provides about
future Y, beyond what past Y already provides. Captures DIRECTIONAL
lead-lag (vs correlation which is symmetric).

Implementation: discrete binning of returns into K quantiles (default 3:
down/flat/up). Probabilities via counts. Numerically stable for our
~1500-day history with K=3 (27 joint cells × enough samples).

Also provides `te_matrix_from_returns()` to batch-compute TE for all pairs
in an asset universe.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def bin_series(x: np.ndarray, n_bins: int = 3,
                quantiles: Optional[np.ndarray] = None) -> np.ndarray:
    """Quantile-bin a 1-D series to integer classes [0, n_bins)."""
    if quantiles is None:
        q = np.linspace(0, 1, n_bins + 1)[1:-1]
        quantiles = np.quantile(x[np.isfinite(x)], q) if np.any(np.isfinite(x)) else np.zeros(n_bins - 1)
    return np.digitize(x, quantiles).astype(np.int32)


def transfer_entropy(x: np.ndarray, y: np.ndarray,
                      lag: int = 1, n_bins: int = 3) -> float:
    """Compute TE(X → Y) in nats.

    Args:
        x, y: aligned 1-D series (length T). y is the target.
        lag: lead lag (default 1 = x at t-1 predicts y at t)
        n_bins: quantile bins per series

    Returns:
        TE(X → Y) in nats (base e). Bounded [0, log(n_bins)].
    """
    T = len(y)
    if T < 50 or len(x) != T:
        return 0.0

    xb = bin_series(x, n_bins)
    yb = bin_series(y, n_bins)

    # Align: Y_t (t from lag+1 to T), Y_{t-1} from lag to T-1, X_{t-1} from lag to T-1
    y_t = yb[lag:]          # shape (T-lag,)
    y_prev = yb[lag - 1: -1] if lag > 0 else yb[:-1]  # one step back
    x_prev = xb[lag - 1: -1] if lag > 0 else xb[:-1]
    # Ensure alignment
    m = min(len(y_t), len(y_prev), len(x_prev))
    y_t = y_t[:m]
    y_prev = y_prev[:m]
    x_prev = x_prev[:m]

    # Joint counts
    K = n_bins
    p_yt_yprev_xprev = np.zeros((K, K, K))
    p_yprev_xprev = np.zeros((K, K))
    p_yt_yprev = np.zeros((K, K))
    p_yprev = np.zeros(K)

    for i in range(m):
        a, b, c = y_t[i], y_prev[i], x_prev[i]
        p_yt_yprev_xprev[a, b, c] += 1
        p_yprev_xprev[b, c] += 1
        p_yt_yprev[a, b] += 1
        p_yprev[b] += 1

    # Normalize
    p_yt_yprev_xprev /= m
    p_yprev_xprev /= m
    p_yt_yprev /= m
    p_yprev /= m

    # TE = sum p(y_t, y_prev, x_prev) * log( p(y_t|y_prev,x_prev) / p(y_t|y_prev) )
    # = sum p(y_t, y_prev, x_prev) * log( p(y_t,y_prev,x_prev) * p(y_prev)
    #                                      / (p(y_prev,x_prev) * p(y_t,y_prev)) )
    te = 0.0
    eps = 1e-12
    for a in range(K):
        for b in range(K):
            for c in range(K):
                p3 = p_yt_yprev_xprev[a, b, c]
                if p3 <= eps:
                    continue
                num = p3 * p_yprev[b]
                den = p_yprev_xprev[b, c] * p_yt_yprev[a, b]
                if den <= eps:
                    continue
                te += p3 * np.log(num / den)
    return float(max(0.0, te))


def _te_from_binned(xb: np.ndarray, yb: np.ndarray, lag: int, K: int) -> float:
    """Vectorized TE on already-binned series. ~50x faster than transfer_entropy().

    Same math as transfer_entropy() but the inner count phase uses
    np.bincount on raveled (a,b,c) joint indices instead of a Python loop.
    Caller is responsible for passing aligned binned arrays of equal length.
    """
    if len(yb) != len(xb) or len(yb) < lag + 50:
        return 0.0
    # Triple alignment: y_t, y_prev, x_prev
    y_t = yb[lag:]
    y_prev = yb[lag - 1: -1] if lag > 0 else yb[:-1]
    x_prev = xb[lag - 1: -1] if lag > 0 else xb[:-1]
    m = min(len(y_t), len(y_prev), len(x_prev))
    if m <= 0:
        return 0.0
    y_t, y_prev, x_prev = y_t[:m], y_prev[:m], x_prev[:m]

    # Joint index in [0, K^3): a*K*K + b*K + c
    joint3 = (y_t.astype(np.int64) * (K * K)
              + y_prev.astype(np.int64) * K
              + x_prev.astype(np.int64))
    p3 = np.bincount(joint3, minlength=K * K * K).astype(np.float64).reshape(K, K, K) / m
    p_yp_xp = p3.sum(axis=0)              # shape (K, K) -- marginalize y_t
    p_yt_yp = p3.sum(axis=2)              # shape (K, K) -- marginalize x_prev
    p_yp = p3.sum(axis=(0, 2))            # shape (K,)   -- marginalize y_t and x_prev

    # TE = sum p3 * log( p3 * p_yp / (p_yp_xp * p_yt_yp) )
    eps = 1e-12
    nonzero = p3 > eps
    # Broadcast factors for the log
    num = p3 * p_yp[None, :, None]
    den = p_yp_xp[None, :, :] * p_yt_yp[:, :, None]
    safe = nonzero & (den > eps)
    log_term = np.zeros_like(p3)
    log_term[safe] = np.log(num[safe] / den[safe])
    te = float((p3 * log_term).sum())
    return max(0.0, te)


def te_matrix_from_returns(returns: Dict[str, np.ndarray],
                             lag: int = 1, n_bins: int = 3
                             ) -> Tuple[List[str], np.ndarray]:
    """Compute full TE matrix for all asset pairs (vectorized + pre-binned).

    Speedup vs naive: bins each asset ONCE per window (not 2N times) and
    uses np.bincount for joint-count tally instead of a Python for-loop.
    Net: 50-100x faster than calling transfer_entropy() per pair on
    typical 90-day windows.

    Returns:
        (names, M) where M[i, j] = TE(names[i] → names[j])
    """
    names = sorted(returns.keys())
    N = len(names)
    M = np.zeros((N, N))
    if N < 2:
        return names, M

    # Bin each asset ONCE per window using its own quantiles.
    # Same per-asset binning policy as the per-pair function (no global bins
    # — preserves per-asset quantile semantics).
    binned: Dict[str, np.ndarray] = {}
    for n in names:
        arr = returns[n]
        if not isinstance(arr, np.ndarray) or len(arr) < lag + 50:
            continue
        binned[n] = bin_series(arr, n_bins)
    valid = [n for n in names if n in binned]
    name_idx = {n: i for i, n in enumerate(names)}

    for i_name in valid:
        for j_name in valid:
            if i_name == j_name:
                continue
            xb = binned[i_name]
            yb = binned[j_name]
            T = min(len(xb), len(yb))
            te = _te_from_binned(xb[:T], yb[:T], lag=lag, K=n_bins)
            M[name_idx[i_name], name_idx[j_name]] = te
    return names, M


def te_btc_anchored(returns: Dict[str, np.ndarray], btc_key: str = "BTCUSDT",
                     lag: int = 1, n_bins: int = 3
                     ) -> Dict[str, Tuple[float, float]]:
    """BTC-anchored TE only: O(2N) instead of O(N²).

    Returns dict {asset: (te_btc_to_asset, te_asset_to_btc)}.
    Use when te_in/te_out (max-over-pairs) are not required by downstream.
    """
    out: Dict[str, Tuple[float, float]] = {}
    if btc_key not in returns:
        return out
    btc_b = bin_series(returns[btc_key], n_bins) if len(returns[btc_key]) >= lag + 50 else None
    if btc_b is None:
        return out
    for name, arr in returns.items():
        if name == btc_key:
            out[name] = (0.0, 0.0)
            continue
        if len(arr) < lag + 50:
            continue
        ab = bin_series(arr, n_bins)
        T = min(len(btc_b), len(ab))
        te_in = _te_from_binned(btc_b[:T], ab[:T], lag=lag, K=n_bins)
        te_out = _te_from_binned(ab[:T], btc_b[:T], lag=lag, K=n_bins)
        out[name] = (te_in, te_out)
    return out


if __name__ == "__main__":
    # Smoke test: X leads Y with known causal link
    np.random.seed(0)
    T = 800
    x = np.random.randn(T)
    # Y depends on lagged X
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.5 * x[t - 1] + 0.3 * np.random.randn()
    te_xy = transfer_entropy(x, y, lag=1, n_bins=3)
    te_yx = transfer_entropy(y, x, lag=1, n_bins=3)
    print("TE(X->Y) = %.4f nats (should be large, X causes Y)" % te_xy)
    print("TE(Y->X) = %.4f nats (should be small)" % te_yx)
    print("Asymmetry TE(X->Y) - TE(Y->X) = %+.4f" % (te_xy - te_yx))
