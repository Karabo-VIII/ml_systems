"""src/strat/data_expansion.py -- CANONICAL, GENERAL, REUSABLE limited-data DATA-EXPANSION toolkit.

When you must train / select / estimate on a TINY dataset (one month, one regime, a handful of
trades), the naive in-sample argmax over a search grid is dangerously optimistic. This module is the
canonical set of six techniques that make limited-data ESTIMATION robust and overfit-resistant. They
DO NOT manufacture signal -- they make the estimate of a (possibly absent) signal honest. The
keystone is James-Stein shrinkage: it collapses to the prior when the apparent winner is just noise,
and preserves the winner only when the spread genuinely clears the estimation-noise floor.

Provenance: extracted from src/strat/config_selector_jan2feb.py (the Jan->Feb 2020 period-level
regime->config selector), where these six correctly (a) AVOIDED overfitting calm-January data and
(b) correctly found ONLY a cost-avoidance effect into the COVID-crash month -- i.e. they did not
hallucinate an edge that was not there. That is the test of a sound estimation tool.

THE SIX TECHNIQUES (each is a public function; generic inputs -- numpy arrays / group labels / score
dicts; NO config objects, NO market loaders, NO project-specific types):
  1. cross_sectional_pool   -- treat each cross-section entity as an independent regime sample. THE
                               biggest N-multiplier. (Caveat: independence is an assumption -- corr
                               across entities deflates the TRUE effective N; reported as n_eff_naive.)
  2. subperiod_windows      -- split one series into overlapping sub-windows for more samples, with an
                               HONEST n_eff that down-weights for overlap + autocorrelation (does NOT
                               pretend overlapping windows are independent).
  3. block_bootstrap_distribution -- resample a return stream in blocks (preserve short-range autocorr)
                               -> a non-degenerate distribution + a ROBUST stat (median/p25), not the max.
  4. fit_regime_generator / simulate_regime_paths / score_across_paths -- fit drift+vol+AR(1), simulate
                               K synthetic future-like paths, score a scorer across them, take a robust
                               (median/p25) score. (Block-bootstrap-of-paths and IAAFT are alternative
                               generators; AR(1)+drift+vol is the simplest.)
  5. james_stein_shrink     -- shrink each candidate's score toward a prior (default = grand mean) by a
                               data-driven factor B in [0,1]. THE overfit-killer: B->0 (collapse to prior)
                               when the spread is just estimation noise; B->1 (keep the raw) when the
                               spread clears the noise floor = signal.
  6. regime_bucket / hierarchical_pool_by_bucket -- bucket samples by a (trend x vol) regime label and
                               pool WITHIN bucket (lower-dimensional, generalizable) instead of per-entity.

HONEST CAVEAT (binding, repeated in the doc): these techniques make a limited-data estimate ROBUST and
overfit-RESISTANT. They DO NOT create signal where none exists. If the underlying process has no edge,
a correct application of these tools returns "no confident pick" (shrinkage B~0, robust stats ~0) --
that is the tool working, not failing.

RWYB:
  python src/strat/data_expansion.py        # two-sided selftest (no market data); each technique proven sound

No emoji (Windows cp1252). Pure-numpy; no project imports. Does NOT git commit.
"""
from __future__ import annotations

from typing import Callable, Dict, Mapping, Sequence

import numpy as np

__contract__ = {
    "kind": "estimation_toolkit",
    "inputs": {
        "returns": "1-D numpy array of per-sample returns (decimal, e.g. 0.02 = +2%)",
        "per_entity_samples": "Mapping[entity -> 1-D numpy array of that entity's samples]",
        "scores": "Mapping[candidate_name -> scalar score] OR a 1-D numpy array of scores",
        "group_labels": "sequence of hashable bucket labels, parallel to a sample array",
    },
    "outputs": {
        "pooled_samples": "concatenated numpy array + an honest effective-N",
        "shrunk_scores": "scores pulled toward a prior + the shrinkage factor B in [0,1]",
        "robust_stat": "median / p25 of a resampled or simulated distribution",
    },
    "invariants": {
        "no_signal_manufacture": "these make a limited-data ESTIMATE robust; they do NOT create signal -- "
                                 "no edge in => no confident pick out (B~0, robust stats ~0)",
        "honest_effective_n": "overlapping sub-windows + correlated entities are DOWN-weighted, never "
                              "counted as independent",
        "shrinkage_is_overfit_killer": "james_stein_shrink collapses to the prior when spread ~ noise floor",
        "robust_not_max": "rank by median/p25 of a distribution, never the in-sample argmax/max",
        "preserve_autocorr": "block bootstrap + AR(1) paths preserve short-range autocorrelation",
        "pure_numpy_no_project_imports": "generic inputs only; decoupled from any caller's domain types",
        "deterministic_given_seed": "every randomized routine takes a seed and is reproducible",
    },
}


# ===========================================================================
# 1. CROSS-SECTIONAL POOLING -- the biggest N-multiplier
# ===========================================================================
def cross_sectional_pool(per_entity_samples: Mapping[object, Sequence[float]]):
    """Pool per-entity sample arrays into one stream, treating each cross-section entity as an
    INDEPENDENT regime sample. THE biggest effective-N multiplier in limited-data work: if you have
    7 assets each with 20 trades in one month, you have 140 pooled samples for estimating a config's
    central tendency (vs 20 for any single asset).

    WHAT       : concatenate {entity -> samples} into a single 1-D array; report n_eff.
    WHEN-TO-USE: you have several comparable entities (assets / instruments / symbols) observed over the
                 SAME limited period, and you want a population estimate of a statistic that you believe
                 is shared (e.g. "does this config survive cost on average across the cross-section").
    CAVEAT     : independence is an ASSUMPTION. Cross-section entities are usually CORRELATED (crypto
                 especially -- a market-wide move hits all of them), so the TRUE effective N is below the
                 naive pooled count. We return n_eff_naive = the pooled count AND n_eff_corr_adjusted =
                 a Kish-style down-adjustment using the mean pairwise correlation of the (overlap-aligned)
                 entity means IF >=2 entities share length; otherwise n_eff_corr_adjusted == n_eff_naive
                 with a flag. Do NOT report n_eff_naive as if entities were independent.

    Returns: dict with keys
        pooled            : 1-D np.ndarray of all samples concatenated
        n_entities        : number of contributing entities
        n_eff_naive       : len(pooled) (the optimistic count)
        n_eff_corr_adjusted : Kish-deflated effective N (<= n_eff_naive)
        rho_bar           : estimated mean pairwise correlation across entities (None if not estimable)
        note              : the independence caveat
    """
    arrays = {}
    for ent, s in per_entity_samples.items():
        a = np.asarray(s, float).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            arrays[ent] = a
    if not arrays:
        return {"pooled": np.array([]), "n_entities": 0, "n_eff_naive": 0,
                "n_eff_corr_adjusted": 0.0, "rho_bar": None,
                "note": "no finite samples in any entity"}
    pooled = np.concatenate(list(arrays.values()))
    n_naive = int(pooled.size)

    # Correlation-deflation: align entities on a common length (truncate to the shortest) and estimate
    # the mean off-diagonal pairwise correlation. Kish effective sample size for correlated streams:
    #   n_eff = n_naive / (1 + (m_bar - 1) * rho_bar)
    # where m_bar = mean samples-per-entity. rho_bar < 0 is floored at 0 (negative corr can only help).
    rho_bar = None
    n_eff_corr = float(n_naive)
    ents = list(arrays.values())
    common = min(a.size for a in ents)
    if len(ents) >= 2 and common >= 3:
        M = np.vstack([a[:common] for a in ents])         # n_entities x common
        # guard against zero-variance rows (corrcoef would emit nan)
        good = np.std(M, axis=1) > 1e-12
        if good.sum() >= 2:
            C = np.corrcoef(M[good])
            iu = np.triu_indices_from(C, k=1)
            offdiag = C[iu]
            offdiag = offdiag[np.isfinite(offdiag)]
            if offdiag.size:
                rho_bar = float(np.clip(np.mean(offdiag), 0.0, 0.999))
                m_bar = n_naive / len(ents)
                n_eff_corr = float(n_naive / (1.0 + (m_bar - 1.0) * rho_bar))
                n_eff_corr = float(min(n_eff_corr, n_naive))
    note = ("each entity treated as an independent regime sample (THE biggest N-multiplier); "
            "n_eff_corr_adjusted deflates for cross-entity correlation -- do NOT cite n_eff_naive as "
            "independent samples when entities co-move")
    return {"pooled": pooled, "n_entities": len(arrays), "n_eff_naive": n_naive,
            "n_eff_corr_adjusted": n_eff_corr, "rho_bar": rho_bar, "note": note}


# ===========================================================================
# 2. SUB-PERIOD WINDOWING -- more samples, HONESTLY down-weighted
# ===========================================================================
def subperiod_windows(index_or_len, window: int, step: int):
    """Split one series into (possibly overlapping) sub-windows -> more (sub-period) samples, with an
    HONEST effective-N that down-weights for the overlap + autocorrelation. This is the technique that
    is easiest to ABUSE: overlapping windows are NOT independent, and reporting their raw count inflates
    confidence. This function refuses to do that.

    WHAT       : produce a list of (start, stop) slices over [0, n) with the given window + step. Compute
                 n_eff = n_windows * (1 - overlap_fraction), the overlap-deflated effective count. With
                 step >= window (no overlap) n_eff == n_windows; with heavy overlap n_eff is much smaller.
    WHEN-TO-USE: a single time series is your only sample of a period and you want several sub-period
                 reads (e.g. weekly windows of a one-month series) as SUPPORTING evidence -- never as
                 independent primary samples.
    CAVEAT     : even non-overlapping windows of a time series are autocorrelated; n_eff here corrects
                 only for the explicit overlap, so treat it as an UPPER bound on independence. The honest
                 use is "supporting evidence", and the doc says so.

    Args:
        index_or_len : an int length, or any sized/array-like whose len() is the series length.
        window       : window length in samples (>=1).
        step         : stride between window starts (>=1). step < window => overlap.

    Returns: dict with keys
        slices       : list of (start, stop) integer tuples (stop exclusive)
        n_windows    : naive window count
        overlap_fraction : mean fraction of each window shared with its neighbour (0 if step>=window)
        n_eff        : overlap-deflated effective count (<= n_windows); HONEST, NOT the naive count
        note         : the autocorrelation caveat
    """
    n = int(index_or_len) if np.isscalar(index_or_len) else int(len(index_or_len))
    window = max(1, int(window))
    step = max(1, int(step))
    slices = []
    s = 0
    while s < n:
        e = min(s + window, n)
        slices.append((s, e))
        if e >= n:
            break
        s += step
    n_windows = len(slices)
    # overlap fraction: fraction of a window's span shared with the next window start.
    overlap = max(0, window - step)
    overlap_fraction = float(overlap / window) if window else 0.0
    overlap_fraction = float(np.clip(overlap_fraction, 0.0, 1.0))
    n_eff = float(n_windows * (1.0 - overlap_fraction))
    n_eff = float(min(n_eff, n_windows))
    note = ("sub-windows of ONE series are autocorrelated; n_eff corrects for explicit OVERLAP only "
            "(upper bound on independence). Use as SUPPORTING evidence, never as independent primary "
            "samples. Overlapping-window count is NOT a free N-multiplier.")
    return {"slices": slices, "n_windows": n_windows, "overlap_fraction": round(overlap_fraction, 4),
            "n_eff": round(n_eff, 3), "note": note}


# ===========================================================================
# 3. BLOCK BOOTSTRAP -- a distribution + a robust stat, autocorr-preserving
# ===========================================================================
def block_bootstrap_distribution(returns: Sequence[float], n_boot: int = 400, block: int = 3,
                                 stat: str = "median", seed: int = 0):
    """Block-bootstrap a return stream into a DISTRIBUTION of the compound outcome, preserving short-range
    autocorrelation by resampling contiguous blocks (not i.i.d. draws). Rank by a ROBUST stat (median or
    p25), never the in-sample max -- the max is the overfit trap.

    WHAT       : draw n_boot resamples of the same length as `returns`, each built from contiguous blocks
                 of length `block` (wrapping at the end), compound each ((prod(1+r)-1)), and summarize the
                 resulting distribution. Returns the distribution + median/p25/p05/mean and the requested
                 robust point stat.
    WHEN-TO-USE: you have a SHORT stream of per-event (per-trade) returns and want a robust central
                 estimate + a sense of downside spread, rather than the single (optimistic) realized
                 compound. The blocks preserve runs/streaks that i.i.d. bootstrap would destroy.
    CAVEAT     : block bootstrap preserves only SHORT-range dependence (set `block` to your autocorr
                 horizon). It does not invent new states -- a stream with no losers cannot produce a
                 loss; a degenerate (empty/length-1) stream returns a point mass at its own value.

    Args:
        returns : 1-D sequence of per-event returns (decimal). Empty -> degenerate zero result.
        n_boot  : number of bootstrap resamples.
        block   : block length (samples) -- set to the autocorrelation horizon.
        stat    : 'median' or 'p25' -- the robust point estimate to surface as `robust`.
        seed    : RNG seed (reproducible).

    Returns: dict with keys
        distribution : 1-D np.ndarray of the n_boot compound outcomes (decimal)
        median, p25, p05, mean : summary stats of the distribution (decimal)
        robust       : the chosen robust point stat (median or p25)
        n            : len(returns)
    """
    r = np.asarray(returns, float).ravel()
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {"distribution": np.array([0.0]), "median": 0.0, "p25": 0.0, "p05": 0.0,
                "mean": 0.0, "robust": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    L = min(max(1, int(block)), r.size)
    n_blocks = int(np.ceil(r.size / L))
    comps = np.empty(int(n_boot), float)
    for b in range(int(n_boot)):
        starts = rng.integers(0, r.size, size=n_blocks)
        pieces = []
        for st in starts:
            if st + L <= r.size:
                pieces.append(r[st:st + L])
            else:                                          # wrap around the end (circular block bootstrap)
                pieces.append(np.concatenate([r[st:], r[:(st + L) % r.size]]))
        samp = np.concatenate(pieces)[:r.size]
        comps[b] = np.prod(1.0 + samp) - 1.0
    median = float(np.median(comps))
    p25 = float(np.quantile(comps, 0.25))
    p05 = float(np.quantile(comps, 0.05))
    mean = float(np.mean(comps))
    robust = p25 if stat == "p25" else median
    return {"distribution": comps, "median": median, "p25": p25, "p05": p05, "mean": mean,
            "robust": float(robust), "n": int(r.size)}


# ===========================================================================
# 4. SYNTHETIC REGIME PATHS -- fit a generator, simulate, score robustly
# ===========================================================================
def fit_regime_generator(returns: Sequence[float]):
    """Fit the simplest regime-conditioned return generator: drift mu, volatility sigma, and an AR(1)
    coefficient phi (lag-1 autocorrelation) on a stream of (log-)returns. These three parameters capture
    the first-order dynamics needed to simulate plausible future-like paths.

    WHAT       : estimate {drift, vol, ar1} from a return series.
    WHEN-TO-USE: as the input to simulate_regime_paths -- when you want to stress a candidate across MANY
                 plausible continuations of the observed regime, not just the one realized path.
    CAVEAT     : AR(1)+drift+vol is a deliberately SIMPLE generator (Gaussian innovations, single lag). It
                 will NOT reproduce fat tails, vol clustering, or regime switches. Alternatives:
                 block-bootstrap-of-paths (preserves the empirical marginal + short autocorr) and IAAFT
                 (preserves the amplitude distribution + power spectrum). Use this as the cheap default.

    Args:
        returns : 1-D sequence of per-bar returns (use log-returns for multiplicative compounding).

    Returns: dict {drift, vol, ar1, n} (vol > 0; ar1 clipped to [-0.95, 0.95]); None if < 4 finite obs.
    """
    r = np.asarray(returns, float).ravel()
    r = r[np.isfinite(r)]
    if r.size < 4:
        return None
    drift = float(np.mean(r))
    vol = float(np.std(r) + 1e-12)
    if np.std(r) > 0:
        phi = float(np.corrcoef(r[:-1], r[1:])[0, 1])
        phi = phi if np.isfinite(phi) else 0.0
    else:
        phi = 0.0
    return {"drift": drift, "vol": vol, "ar1": float(np.clip(phi, -0.95, 0.95)), "n": int(r.size)}


def simulate_regime_paths(params: Mapping[str, float], n_bars: int, K: int = 40, seed: int = 0,
                          p0: float = 100.0):
    """Simulate K synthetic return/price paths from a regime generator's params (drift mu, vol sigma,
    AR(1) phi). The AR(1) recursion is vectorized ACROSS the K paths (the only sequential dependence is
    along the short bar axis). Returns BOTH the return matrix and a price matrix so callers can score
    either return-based or price-based logic.

    WHAT       : produce K x n_bars synthetic returns and the corresponding price paths from {drift,vol,ar1}.
    WHEN-TO-USE: feed score_across_paths to get a robust (median/p25) score of a candidate across many
                 plausible continuations of the fitted regime.
    CAVEAT     : Gaussian-innovation AR(1) -- no fat tails / vol clustering / regime switches (see
                 fit_regime_generator). The paths PRESERVE the input drift/vol/ar1 in expectation; verify
                 with the selftest if you change the recursion.

    Args:
        params : {drift, vol, ar1} (as from fit_regime_generator). Also accepts legacy keys mu/sigma/phi.
        n_bars : path length in bars.
        K      : number of paths.
        seed   : RNG seed (reproducible).
        p0     : starting price for the price matrix.

    Returns: dict with keys
        returns : K x n_bars np.ndarray of synthetic returns
        prices  : K x n_bars np.ndarray of synthetic prices (p0 * cumprod(1+returns)-ish via exp-cumsum)
        K, n_bars : echoed dimensions
    """
    mu = float(params.get("drift", params.get("mu", 0.0)))
    sigma = float(params.get("vol", params.get("sigma", 1e-9)))
    phi = float(params.get("ar1", params.get("phi", 0.0)))
    n_bars = max(1, int(n_bars))
    K = max(1, int(K))
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, (K, n_bars))
    r = np.empty((K, n_bars))
    prev = np.zeros(K)
    for i in range(n_bars):                                # short loop, vectorized over the K paths
        r[:, i] = mu + phi * prev + eps[:, i]
        prev = r[:, i] - mu                                # AR(1) on the demeaned innovation
    prices = p0 * np.exp(np.cumsum(r, axis=1))
    return {"returns": r, "prices": prices, "K": K, "n_bars": n_bars}


def score_across_paths(scorer_fn: Callable[[np.ndarray], float], paths: Mapping[str, np.ndarray],
                       on: str = "returns", stat: str = "p25"):
    """Score a candidate across all K synthetic paths and return a ROBUST (median/p25) aggregate. This is
    how you turn 'it worked on the one realized path' into 'it works across plausible continuations'.

    WHAT       : apply scorer_fn to each of the K rows of paths[on], summarize the K scores robustly.
    WHEN-TO-USE: after simulate_regime_paths, to rank candidates by their robust (not best-case) behaviour
                 across the synthetic distribution.
    CAVEAT     : the robustness is only as good as the generator (see fit_regime_generator caveat). p25 is
                 the recommended robust-pessimistic default; 'median' is the central default.

    Args:
        scorer_fn : callable taking a single 1-D path (a row of paths[on]) -> scalar score.
        paths     : dict from simulate_regime_paths (must contain key `on`).
        on        : 'returns' or 'prices' -- which matrix to feed the scorer.
        stat      : 'p25' (robust-pessimistic) | 'median' | 'mean'.

    Returns: dict {robust, median, p25, mean, K} where `robust` is the chosen stat.
    """
    M = np.asarray(paths[on], float)
    scores = np.array([float(scorer_fn(M[k])) for k in range(M.shape[0])], float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return {"robust": 0.0, "median": 0.0, "p25": 0.0, "mean": 0.0, "K": 0}
    median = float(np.median(scores))
    p25 = float(np.quantile(scores, 0.25))
    mean = float(np.mean(scores))
    robust = {"p25": p25, "median": median, "mean": mean}.get(stat, p25)
    return {"robust": float(robust), "median": median, "p25": p25, "mean": mean, "K": int(scores.size)}


# ===========================================================================
# 5. JAMES-STEIN SHRINKAGE -- THE overfit-killer
# ===========================================================================
def james_stein_shrink(scores, prior=None, noise_var=None):
    """Shrink each candidate's score toward a common prior by a DATA-DRIVEN factor B in [0,1]. This is the
    single most important limited-data tool: it refuses to crown the lucky in-sample maximum unless the
    cross-candidate spread genuinely clears the estimation-noise floor.

        shrunk = prior + B * (raw - prior)
        B      = 1 - (k - 2) * sigma2 / sum((raw - prior)^2)          (clipped to [0, 1])

    where k = number of candidates and sigma2 = the per-score ESTIMATION-NOISE variance (NOT the cross-
    candidate spread). The behaviour is the whole point:
        - spread >> noise  => B -> 1  => keep the raw scores (the differences are SIGNAL; trust the winner)
        - spread ~  noise  => B -> 0  => collapse to the prior (the differences are NOISE; pick nothing)

    WHAT       : pull a vector/dict of scores toward a prior by the James-Stein factor B; return both.
    WHEN-TO-USE: ALWAYS, before taking an argmax over candidate scores estimated on limited data (config
                 ranking, hyperparameter selection, per-asset picks). It is the overfit-killer.
    CAVEAT     : you must supply (or let it default) a sensible noise_var in the SAME units^2 as the
                 scores. The default below (a 1.0-unit^2 floor) assumes scores are in compound-% units --
                 if your scores are in a different scale, pass noise_var explicitly (e.g. the squared
                 standard error of each score, or the squared half-IQR of a per-candidate bootstrap).
                 B is a SCALAR applied to all candidates (classic James-Stein), not per-candidate.

    Args:
        scores   : Mapping[name -> score] OR a 1-D array-like of scores.
        prior    : the shrink target. None => the grand mean (cross-candidate / cross-sectional mean).
        noise_var: per-score estimation-noise variance sigma2 (same units^2 as scores). None => a 1.0
                   floor in compound-% units (conservative; separates wide-spread signal from i.i.d. noise
                   by absolute spread magnitude).

    Returns: (shrunk, B)
        shrunk : same container type as input (dict in -> dict out; array in -> np.ndarray out)
        B      : the scalar shrinkage factor in [0, 1] (0 = fully shrunk to prior; 1 = raw kept)
    """
    is_dict = isinstance(scores, Mapping)
    if is_dict:
        names = list(scores)
        raw = np.array([float(scores[n]) for n in names], float)
    else:
        raw = np.asarray(scores, float).ravel()
    k = raw.size
    if prior is None:
        prior = float(np.mean(raw)) if k else 0.0
    prior = float(prior)
    if k < 3:
        # James-Stein needs k >= 3 to be admissible; with < 3 candidates, no shrinkage (return raw, B=1).
        if is_dict:
            return {n: float(v) for n, v in zip(names, raw)}, 1.0
        return raw.copy(), 1.0
    ss = float(np.sum((raw - prior) ** 2)) + 1e-12
    if noise_var is None:
        sigma2 = 1.0 ** 2                                  # compound-% units default noise floor
    else:
        sigma2 = float(noise_var)
    B = 1.0 - (k - 2) * sigma2 / ss
    B = float(np.clip(B, 0.0, 1.0))
    shr = prior + B * (raw - prior)
    if is_dict:
        return {n: float(s) for n, s in zip(names, shr)}, B
    return shr, B


# ===========================================================================
# 6. REGIME BUCKETING + HIERARCHICAL POOLING -- generalize, don't memorize
# ===========================================================================
def vol_terciles(values: Sequence[float]):
    """Compute (q33, q67) tercile breakpoints of a set of volatilities (or any dispersion measure) so a
    new value can be labeled lo/mid/hi. Helper for regime_bucket. < 3 values -> (nan, nan) (all 'mid')."""
    v = np.asarray(values, float).ravel()
    v = v[np.isfinite(v)]
    if v.size < 3:
        return (float("nan"), float("nan"))
    q1, q2 = np.quantile(v, [1.0 / 3.0, 2.0 / 3.0])
    return (float(q1), float(q2))


def regime_bucket(returns: Sequence[float], vol_terciles_breaks=None):
    """Label a sample's regime as a (trend x vol) bucket from its return stream. This lower-dimensional
    label is what you pool over (technique 6) instead of memorizing a config per individual entity.

    WHAT       : compute a trend bucket {up, flat, down} (from the cumulative return / scaled drift) and a
                 vol bucket {lo, mid, hi} (from realized vol vs supplied tercile breaks) -> "trend_vol".
    WHEN-TO-USE: to assign each entity/period a coarse regime so you can hierarchical_pool_by_bucket and
                 learn a config PER REGIME (generalizes) rather than per entity (memorizes).
    CAVEAT     : trend thresholds are pre-registered constants (a Sharpe-like trend_strength of +/-0.5 and
                 a small cumulative-return guard). vol bucketing needs cross-sectional tercile breaks
                 (from vol_terciles); without them everything is 'mid'. The label is COARSE by design --
                 that coarseness is what makes it generalize.

    Args:
        returns           : 1-D per-bar return stream for this entity/period.
        vol_terciles_breaks : (q33, q67) from vol_terciles(...) over the cross-section's vols. None or
                              nan -> vol bucket is 'mid'.

    Returns: dict {bucket, trend, vol_bucket, trend_strength, vol}
    """
    r = np.asarray(returns, float).ravel()
    r = r[np.isfinite(r)]
    if r.size < 3:
        return {"bucket": "flat_mid", "trend": "flat", "vol_bucket": "mid",
                "trend_strength": 0.0, "vol": 0.0}
    vol = float(np.std(r))
    cum = float(np.prod(1.0 + r) - 1.0)
    # trend strength: cumulative return scaled by per-bar vol * sqrt(n) (annualization-free, scale-aware)
    trend_strength = cum / (vol * np.sqrt(r.size) + 1e-9)
    trend = "up" if (trend_strength > 0.5 and cum > 0) else (
            "down" if (trend_strength < -0.5 or cum < -0.05) else "flat")
    vol_bucket = "mid"
    if vol_terciles_breaks is not None:
        q1, q2 = vol_terciles_breaks
        if np.isfinite(q1) and np.isfinite(q2):
            vol_bucket = "lo" if vol <= q1 else ("hi" if vol > q2 else "mid")
    return {"bucket": f"{trend}_{vol_bucket}", "trend": trend, "vol_bucket": vol_bucket,
            "trend_strength": round(trend_strength, 4), "vol": round(vol, 6)}


def hierarchical_pool_by_bucket(per_entity_samples: Mapping[object, Sequence[float]],
                                bucket_fn: Callable[[object, np.ndarray], str]):
    """Pool samples WITHIN regime bucket instead of per-entity. Each entity is assigned a coarse regime
    label by bucket_fn, then all entities sharing a bucket are pooled. The result is a lower-dimensional,
    generalizable map (one estimate per regime) rather than a high-variance per-entity estimate.

    WHAT       : group entities by bucket_fn(entity, samples) and concatenate their samples per bucket.
    WHEN-TO-USE: technique 6 -- when you want a regime->X map that generalizes (learn per bucket), not a
                 per-entity map that memorizes the in-sample winner for each individual entity.
    CAVEAT     : pooling assumes entities within a bucket are exchangeable (same data-generating regime).
                 That is the whole bet of regime-conditioning; if a bucket has only one entity it is just
                 that entity (n_entities==1 flag). Buckets with few entities still over-fit -- check the
                 per-bucket n_entities before trusting a bucket's pick.

    Args:
        per_entity_samples : Mapping[entity -> 1-D sample array].
        bucket_fn          : callable (entity, samples_array) -> hashable bucket label.

    Returns: dict bucket_label -> {pooled, n_entities, n_samples, entities}
    """
    buckets: Dict[str, dict] = {}
    for ent, s in per_entity_samples.items():
        a = np.asarray(s, float).ravel()
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        b = bucket_fn(ent, a)
        d = buckets.setdefault(b, {"_arrays": [], "entities": []})
        d["_arrays"].append(a)
        d["entities"].append(ent)
    out = {}
    for b, d in buckets.items():
        pooled = np.concatenate(d["_arrays"])
        out[b] = {"pooled": pooled, "n_entities": len(d["entities"]),
                  "n_samples": int(pooled.size), "entities": list(d["entities"])}
    return out


# ===========================================================================
# TWO-SIDED SELFTEST -- each technique proven sound (no market data)
# ===========================================================================
def _selftest():
    print("## DATA-EXPANSION CANONICAL -- two-sided selftest (no market data)")
    results = {}

    # -- 5. JAMES-STEIN (the keystone): wide-spread REAL winner -> B high (trust the winner);
    #       i.i.d. NOISE -> B low (heavy shrink, pick nothing). Mirrors the config_selector selftest.
    pos_scores = {f"cfg{i}": float(i) for i in range(10)}          # cfg9 clearly best, wide spread
    shr_pos, B_pos = james_stein_shrink(pos_scores, prior=4.5)
    win = max(shr_pos, key=shr_pos.get)
    rng = np.random.default_rng(0)
    noise_scores = {f"cfg{i}": float(rng.normal(0, 0.01)) for i in range(12)}   # tiny i.i.d. noise
    _, B_neg = james_stein_shrink(noise_scores, prior=0.0)
    js_pass = (win == "cfg9" and B_pos > 0.5 and B_neg < 0.5)
    results["james_stein_shrink"] = js_pass
    print(f"  [5] james_stein_shrink: POSITIVE wide-spread winner -> B={B_pos:.2f} (expect >0.5), "
          f"argmax={win} (expect cfg9); NEGATIVE i.i.d. noise -> B={B_neg:.2f} (expect <0.5) "
          f"-> {'PASS' if js_pass else 'FAIL'}")

    # -- 3. BLOCK BOOTSTRAP: preserves the MARGINAL (mean/std within tolerance of a single-event-compound
    #       reference) while giving a NON-DEGENERATE spread. A 1-block bootstrap of i.i.d. draws reproduces
    #       the empirical marginal; check the mean of compounded blocks tracks the analytic expectation and
    #       the spread is non-zero.
    rng2 = np.random.default_rng(3)
    stream = rng2.normal(0.0, 0.02, 60)                            # 60 small i.i.d. returns
    bs = block_bootstrap_distribution(stream, n_boot=2000, block=1, stat="median", seed=7)
    # with block=1 each resample is an i.i.d. bootstrap of the same length -> E[compound] ~= realized-ish
    realized = float(np.prod(1.0 + stream) - 1.0)
    dist = bs["distribution"]
    marg_ok = abs(bs["mean"] - realized) < 0.02                    # mean of bootstrap near realized compound
    spread_ok = float(np.std(dist)) > 1e-4                         # non-degenerate
    bb_pass = marg_ok and spread_ok
    results["block_bootstrap_distribution"] = bb_pass
    print(f"  [3] block_bootstrap_distribution: boot_mean={bs['mean']:+.4f} vs realized={realized:+.4f} "
          f"(|diff|<0.02: {marg_ok}); spread std={np.std(dist):.4f} (non-degenerate: {spread_ok}) "
          f"-> {'PASS' if bb_pass else 'FAIL'}")

    # -- 4. SYNTHETIC PATHS: the simulated paths PRESERVE the input regime stats (drift/vol/ar1 within tol).
    true = {"drift": 0.0015, "vol": 0.012, "ar1": 0.30}
    paths = simulate_regime_paths(true, n_bars=400, K=200, seed=11)
    R = paths["returns"]
    emp_drift = float(np.mean(R))
    emp_vol = float(np.std(R))
    # empirical lag-1 autocorr across all paths (pooled)
    flat0 = R[:, :-1].ravel(); flat1 = R[:, 1:].ravel()
    emp_ar1 = float(np.corrcoef(flat0, flat1)[0, 1])
    drift_ok = abs(emp_drift - true["drift"]) < 0.0010
    vol_ok = abs(emp_vol - true["vol"]) < 0.0020
    ar1_ok = abs(emp_ar1 - true["ar1"]) < 0.06
    sp_pass = drift_ok and vol_ok and ar1_ok
    results["simulate_regime_paths"] = sp_pass
    print(f"  [4] simulate_regime_paths: drift {emp_drift:+.4f} vs {true['drift']:+.4f} ({drift_ok}); "
          f"vol {emp_vol:.4f} vs {true['vol']:.4f} ({vol_ok}); ar1 {emp_ar1:+.3f} vs {true['ar1']:+.3f} "
          f"({ar1_ok}) -> {'PASS' if sp_pass else 'FAIL'}")

    # -- 2. SUB-PERIOD WINDOWS: n_eff is DOWN-weighted vs the naive count under overlap (proves it does
    #       NOT fake independence). Overlapping windows (step < window) must give n_eff < n_windows;
    #       non-overlapping (step == window) must give n_eff == n_windows.
    ov = subperiod_windows(100, window=20, step=5)                 # heavy overlap
    no = subperiod_windows(100, window=20, step=20)                # no overlap
    overlap_deflated = ov["n_eff"] < ov["n_windows"]
    nonoverlap_exact = abs(no["n_eff"] - no["n_windows"]) < 1e-9
    sw_pass = overlap_deflated and nonoverlap_exact
    results["subperiod_windows"] = sw_pass
    print(f"  [2] subperiod_windows: OVERLAP n_windows={ov['n_windows']} n_eff={ov['n_eff']} "
          f"(deflated: {overlap_deflated}); NO-OVERLAP n_windows={no['n_windows']} n_eff={no['n_eff']} "
          f"(exact: {nonoverlap_exact}) -> {'PASS' if sw_pass else 'FAIL'}")

    # -- 1. CROSS-SECTIONAL POOL (supporting check): pooled count == sum of entity samples; correlated
    #       entities deflate n_eff below the naive count; independent entities keep it ~ naive.
    corr_seed = np.random.default_rng(5).normal(0, 1, 50)
    ent_corr = {f"e{i}": corr_seed + np.random.default_rng(100 + i).normal(0, 0.05, 50) for i in range(6)}
    pc_corr = cross_sectional_pool(ent_corr)
    ent_ind = {f"e{i}": np.random.default_rng(200 + i).normal(0, 1, 50) for i in range(6)}
    pc_ind = cross_sectional_pool(ent_ind)
    pooled_ok = pc_corr["n_eff_naive"] == 300 and pc_ind["n_eff_naive"] == 300
    corr_deflates = pc_corr["n_eff_corr_adjusted"] < pc_corr["n_eff_naive"]
    ind_keeps = pc_ind["n_eff_corr_adjusted"] > 0.7 * pc_ind["n_eff_naive"]
    cs_pass = pooled_ok and corr_deflates and ind_keeps
    results["cross_sectional_pool"] = cs_pass
    print(f"  [1] cross_sectional_pool: pooled={pc_corr['n_eff_naive']} (==300: {pooled_ok}); "
          f"correlated n_eff={pc_corr['n_eff_corr_adjusted']:.0f} (rho={pc_corr['rho_bar']:.2f}, "
          f"deflates: {corr_deflates}); independent n_eff={pc_ind['n_eff_corr_adjusted']:.0f} "
          f"(keeps: {ind_keeps}) -> {'PASS' if cs_pass else 'FAIL'}")

    # -- 6. REGIME BUCKET + HIERARCHICAL POOL (supporting check): an up-trend stream labels 'up_*', a
    #       down stream 'down_*'; pooling groups same-regime entities together.
    up_stream = np.full(40, 0.01) + np.random.default_rng(9).normal(0, 0.001, 40)
    dn_stream = np.full(40, -0.01) + np.random.default_rng(10).normal(0, 0.001, 40)
    rb_up = regime_bucket(up_stream)
    rb_dn = regime_bucket(dn_stream)
    label_ok = rb_up["trend"] == "up" and rb_dn["trend"] == "down"
    ents = {"a": up_stream, "b": up_stream * 1.0, "c": dn_stream, "d": dn_stream * 1.0}
    pooled = hierarchical_pool_by_bucket(ents, lambda e, a: regime_bucket(a)["trend"])
    pool_ok = pooled.get("up", {}).get("n_entities") == 2 and pooled.get("down", {}).get("n_entities") == 2
    rb_pass = label_ok and pool_ok
    results["regime_bucket"] = rb_pass
    print(f"  [6] regime_bucket/hierarchical_pool: up-stream='{rb_up['trend']}' down-stream='{rb_dn['trend']}' "
          f"(labels: {label_ok}); pooled up={pooled.get('up',{}).get('n_entities')} "
          f"down={pooled.get('down',{}).get('n_entities')} (groups: {pool_ok}) "
          f"-> {'PASS' if rb_pass else 'FAIL'}")

    overall = all(results.values())
    print(f"\n  PER-TECHNIQUE: " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in results.items()))
    print(f"  OVERALL SELFTEST {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
