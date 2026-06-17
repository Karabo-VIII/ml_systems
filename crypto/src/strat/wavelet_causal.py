"""src/strat/wavelet_causal.py -- a CAUSAL (past-only) multi-scale wavelet decomposition + features.

WHY (TSFM_WAVELET_WM_SURVEY_2026_06_09.md rec #1): the published wavelet-trading literature is
contaminated by look-ahead -- a DWT over the FULL series before the split, or non-causal SYMMETRIC
wavelets whose coefficient at time t depends on samples t+k. That is the same bug class as our
G-AUDIT-011 (look-ahead via full-history standardization). The ONLY honest way to adopt wavelets is a
strictly causal transform whose coefficient at index t uses ONLY samples <= t.

WHAT this is:
  - A causal a-trous (undecimated / stationary) wavelet transform with a Haar analysis filter, computed
    by LEFT-aligned (causal) convolution with LEFT-only edge padding. At scale j the filter taps are
    spread by 2**j ("a trous" = "with holes"), so band j captures structure at period ~2**j bars. We
    produce J detail bands d_1..d_J (high-freq -> low-freq) and a residual approximation a_J. Because the
    convolution is causal, coefficient[t] is a function of x[t], x[t-1], ... only -- NEVER x[t+1].

  - Per-bar, past-only FEATURES for the setup apparatus:
      scale_energy[j]    -- rolling (past-only) energy in detail band j (mean of squared coeffs over a
                            trailing window).
      dominant_scale     -- argmax_j scale_energy[j] (which time-scale holds the most energy right now).
      energy_expansion   -- current total high-freq energy / its PAST expanding median (shift(1) so the
                            threshold itself never sees the current bar). This is the multi-scale
                            generalization of the single-scale vol-expansion trigger in
                            src/mining/conditional.py:88-95 (local |ret| vol > 1.5x its expanding median).

THE LEAK TEST (`leak_test`, the single most important deliverable): proves features[t] is INVARIANT to
any change in x[t+1:]. If that ever fails the transform is non-causal and the probe is worthless.

Pure numpy. No pywt, no scipy.signal. Haar (and an optional length-4 Daubechies db2) implemented inline.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------------------------------
# Analysis filters (low-pass h, high-pass g). Coefficients ordered so that, under a LEFT-causal
# convolution y[t] = sum_k taps[k] * x[t - k*dilation], tap[0] multiplies the CURRENT sample x[t] and
# tap[k>0] multiplies PAST samples x[t-k*dilation]. This ordering is what makes the transform causal.
# --------------------------------------------------------------------------------------------------
_SQRT2 = np.sqrt(2.0)

# Haar: low-pass averages (x[t]+x[t-1])/sqrt2, high-pass differences (x[t]-x[t-1])/sqrt2.
_HAAR_LO = np.array([1.0, 1.0]) / _SQRT2
_HAAR_HI = np.array([1.0, -1.0]) / _SQRT2

# Daubechies db2 (length-4). Standard orthonormal coefficients; reversed for the same current-first tap
# ordering so the convolution stays causal.
_C0 = (1 + np.sqrt(3)) / (4 * _SQRT2)
_C1 = (3 + np.sqrt(3)) / (4 * _SQRT2)
_C2 = (3 - np.sqrt(3)) / (4 * _SQRT2)
_C3 = (1 - np.sqrt(3)) / (4 * _SQRT2)
_DB2_LO = np.array([_C0, _C1, _C2, _C3])           # tap0 = newest sample
_DB2_HI = np.array([_C3, -_C2, _C1, -_C0])         # QMF high-pass, same ordering

_FILTERS = {"haar": (_HAAR_LO, _HAAR_HI), "db2": (_DB2_LO, _DB2_HI)}


def _causal_conv(x: np.ndarray, taps: np.ndarray, dilation: int) -> np.ndarray:
    """Left-causal dilated convolution: y[t] = sum_k taps[k] * x[t - k*dilation].

    Indices t - k*dilation < 0 are EDGE-padded on the LEFT ONLY (clamp to x[0]). The output at index t
    therefore depends exclusively on x[0..t] -- never on any future sample. This is the property the leak
    test verifies."""
    n = len(x)
    y = np.zeros(n, dtype=float)
    for k, c in enumerate(taps):
        shift = k * dilation
        if shift == 0:
            y += c * x
        else:
            # shifted[t] = x[t-shift] for t>=shift, else x[0] (left edge-pad). No future samples used.
            shifted = np.empty(n, dtype=float)
            shifted[:shift] = x[0]
            shifted[shift:] = x[:n - shift]
            y += c * shifted
    return y


def causal_swt(x: np.ndarray, J: int = 4, wavelet: str = "haar") -> dict:
    """Causal a-trous / undecimated stationary wavelet transform.

    Returns dict with:
      details: list of J arrays (len n) -- detail coefficients d_1..d_J (fine -> coarse).
      approx:  array (len n) -- the residual approximation a_J after J low-pass stages.
    At stage j the low/high-pass taps are dilated by 2**(j-1) (a-trous). Every coefficient at index t is a
    causal function of x[0..t]."""
    x = np.asarray(x, dtype=float).ravel()
    if wavelet not in _FILTERS:
        raise ValueError(f"unknown wavelet {wavelet!r}; known={list(_FILTERS)}")
    lo, hi = _FILTERS[wavelet]
    details = []
    approx = x.copy()
    for j in range(J):
        dilation = 2 ** j
        d = _causal_conv(approx, hi, dilation)   # detail at this scale (past-only)
        a = _causal_conv(approx, lo, dilation)   # smoother approximation (past-only)
        details.append(d)
        approx = a
    return {"details": details, "approx": approx, "J": J, "wavelet": wavelet}


def _rolling_mean_pastonly(v: np.ndarray, win: int) -> np.ndarray:
    """Trailing mean over the last `win` samples INCLUDING the current one (past-only, no future bars).

    out[t] = mean(v[max(0,t-win+1) .. t]). Uses a cumulative sum; out[t] depends only on v[0..t]."""
    v = np.asarray(v, dtype=float)
    n = len(v)
    cs = np.concatenate([[0.0], np.cumsum(v)])
    out = np.empty(n, dtype=float)
    for t in range(n):
        lo = max(0, t - win + 1)
        out[t] = (cs[t + 1] - cs[lo]) / (t + 1 - lo)
    return out


def _expanding_median_shifted(v: np.ndarray, min_periods: int = 50) -> np.ndarray:
    """Expanding median of v[0..t-1] (SHIFTED by one bar so the threshold at t never sees v[t]).

    out[t] = median(v[0..t-1]) for t >= min_periods, else NaN. Strictly past-only (mirrors the
    expanding(50).median().shift(1) pattern in src/mining/conditional.py:91)."""
    v = np.asarray(v, dtype=float)
    n = len(v)
    out = np.full(n, np.nan)
    for t in range(min_periods, n):
        window = v[:t]                       # v[0..t-1] -- excludes the current bar
        finite = window[np.isfinite(window)]
        if finite.size >= min_periods:
            out[t] = float(np.median(finite))
    return out


def wavelet_features(x: np.ndarray, J: int = 4, wavelet: str = "haar",
                     energy_win: int = 20, expmed_min_periods: int = 50) -> dict:
    """Per-bar, past-only multi-scale features from the causal SWT of series x.

    Returns dict of arrays (all length n, every index t a causal function of x[0..t]):
      scale_energy : (n, J) -- rolling past-only energy (mean squared detail coeff) per band j.
      dominant_scale : (n,) int -- argmax_j scale_energy[t, j] (NaN-energy rows -> -1).
      total_hf_energy : (n,) -- sum over bands of scale_energy (the multi-scale "vol" proxy).
      energy_expansion : (n,) -- total_hf_energy[t] / expanding_median(total_hf_energy)[t]
                                 (shift(1) median; NaN until expmed_min_periods). The setup trigger.
    """
    swt = causal_swt(x, J=J, wavelet=wavelet)
    details = swt["details"]
    n = len(x)
    scale_energy = np.empty((n, J), dtype=float)
    for j in range(J):
        scale_energy[:, j] = _rolling_mean_pastonly(details[j] ** 2, energy_win)
    dominant_scale = np.full(n, -1, dtype=int)
    finite_rows = np.isfinite(scale_energy).all(axis=1)
    dominant_scale[finite_rows] = np.argmax(scale_energy[finite_rows], axis=1)
    total_hf = scale_energy.sum(axis=1)
    med = _expanding_median_shifted(total_hf, min_periods=expmed_min_periods)
    with np.errstate(invalid="ignore", divide="ignore"):
        energy_expansion = np.where(np.isfinite(med) & (med > 0), total_hf / med, np.nan)
    return {"scale_energy": scale_energy, "dominant_scale": dominant_scale,
            "total_hf_energy": total_hf, "energy_expansion": energy_expansion,
            "J": J, "wavelet": wavelet, "energy_win": energy_win}


# --------------------------------------------------------------------------------------------------
# THE LEAK TEST -- the single most important deliverable. Proves features[:t+1] are bit-identical when
# x[t+1:] is arbitrarily perturbed (noise added AND reversed). If this ever fails, the transform leaks
# the future and the probe is worthless.
# --------------------------------------------------------------------------------------------------
def leak_test(x: np.ndarray | None = None, J: int = 4, wavelet: str = "haar",
              energy_win: int = 20, n_cuts: int = 12, seed: int = 0, verbose: bool = True) -> dict:
    """For a set of cut indices t, perturb x[t+1:] (add noise + reverse it) and assert every feature is
    bit-identical on [:t+1]. Returns a dict; raises AssertionError on any leak."""
    rng = np.random.default_rng(seed)
    if x is None:
        # a non-trivial test series: trend + multi-scale oscillation + noise (so all bands are exercised)
        n = 600
        tt = np.arange(n)
        x = (0.001 * tt + np.sin(2 * np.pi * tt / 8) + 0.5 * np.sin(2 * np.pi * tt / 32)
             + rng.normal(0, 0.3, n))
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    base = wavelet_features(x, J=J, wavelet=wavelet, energy_win=energy_win)
    keys = ["scale_energy", "dominant_scale", "total_hf_energy", "energy_expansion"]

    cuts = sorted(set(int(c) for c in np.linspace(int(0.15 * n), n - 2, n_cuts)))
    results = []
    for t in cuts:
        xp = x.copy()
        future = xp[t + 1:]
        # perturb the future two ways at once: reverse it AND add large noise -> any future dependence shows.
        xp[t + 1:] = future[::-1] + rng.normal(0, 5.0, future.shape)
        feat = wavelet_features(xp, J=J, wavelet=wavelet, energy_win=energy_win)
        for k in keys:
            a, b = base[k][:t + 1], feat[k][:t + 1]
            # NaN-aware bit-identity: same NaN mask AND bit-identical on the finite entries.
            mask_a, mask_b = np.isnan(np.asarray(a, float)), np.isnan(np.asarray(b, float))
            assert np.array_equal(mask_a, mask_b), f"LEAK: NaN-mask differs in {k!r} at cut t={t}"
            fa, fb = np.asarray(a, float)[~mask_a], np.asarray(b, float)[~mask_b]
            assert np.array_equal(fa, fb), (
                f"LEAK: feature {k!r} at cut t={t} changed when x[t+1:] was perturbed "
                f"-> transform is NON-CAUSAL (max|delta|={np.max(np.abs(fa - fb)) if fa.size else 0})")
        results.append(t)
    out = {"passed": True, "n_cuts": len(cuts), "cuts": cuts, "series_len": n,
           "J": J, "wavelet": wavelet, "features_checked": keys}
    if verbose:
        print(f"[wavelet leak_test] PASS -- {len(cuts)} cut points, {len(keys)} features each "
              f"bit-identical on [:t+1] under future perturbation (reverse + N(0,5) noise). "
              f"wavelet={wavelet} J={J} n={n}")
    return out


def _self_check():
    """RWYB: leak test on both wavelets + a tiny sanity print. exit 0 == causal."""
    for wv in ("haar", "db2"):
        leak_test(wavelet=wv, J=4, verbose=True)
    # sanity: a sudden vol burst late in a quiet series should fire energy_expansion AFTER the burst, never before.
    n = 300
    x = np.concatenate([np.random.default_rng(1).normal(0, 0.2, 200),
                        np.random.default_rng(2).normal(0, 2.0, 100)])
    f = wavelet_features(x, J=4, wavelet="haar")
    ee = f["energy_expansion"]
    pre = np.nanmean(ee[150:199])
    post = np.nanmean(ee[205:260])
    print(f"[wavelet self-check] energy_expansion pre-burst~{pre:.2f}  post-burst~{post:.2f} "
          f"(post should exceed pre; causal so it reacts only AFTER the burst)")
    print("[wavelet_causal] self-check OK")


if __name__ == "__main__":
    _self_check()
