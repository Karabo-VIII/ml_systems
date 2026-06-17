"""Econometric SIGNATURE -- the MATH/ECON lens of the decomposition harness.

Puts COMPUTED NUMBERS to the time-series theory that docs/CRYPTO_MARKET_UNDERSTANDING.md
section II only DOCUMENTS. This is the AGNOSTIC, WHOLE-SERIES characterization -- "what kind
of stochastic process is this asset's return stream" -- complementary to the per-window
chimera viewer (src/mining/decompose.py). Where decompose.py inspects engineered chimera
FEATURES over a window, this tool runs the CANONICAL econometric ESTIMATORS over the whole
return series (or a sub-window) and reconciles them against (a) the documented literature
values in section II and (b) the chimera PROXY features that try to capture the same thing.

Sections (mirrors decompose's family grouping):
  1. DISTRIBUTION & TAILS   -- moments, Hill tail-index (left+right), cubic-law check, Jarque-Bera
  2. DEPENDENCE & MEMORY    -- Ljung-Box(ret / |ret| / ret^2), AC1, Hurst (R/S + DFA) on ret and |ret|
  3. STATIONARITY           -- ADF + KPSS
  4. VOLATILITY PROCESS     -- GARCH(1,1)-t persistence/half-life/nu + GJR leverage gamma (sign)
  5. JUMPS                  -- Barndorff-Nielsen-Shephard RV vs BV jump fraction + threshold count

  RECONCILIATION           -- per metric: OUR estimate | section-II literature/[RWYB-OURS] | chimera proxy mean
                              with an AGREE/DISAGREE flag (does our engineered proxy track the canonical estimator?)

Run:
  python -m mining.econometric_signature --asset BTC --cadence 4h
  python -m mining.econometric_signature --asset ETH --cadence 1d --start 2024-01-01 --end 2025-01-01
  python -m mining.econometric_signature --asset SOL --cadence 4h --json
DEFAULT = WHOLE series (the signature is a whole-series property). No emoji (cp1252).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pipeline.chimera_loader import ChimeraLoader          # noqa: E402

OUT = ROOT / "runs" / "mining"
OUT.mkdir(parents=True, exist_ok=True)

# annualization: bars per year per cadence (factor = sqrt(bars_per_year))
BARS_PER_YEAR = {"1d": 365, "4h": 365 * 6, "1h": 365 * 24, "30m": 365 * 48, "15m": 365 * 96}

# small-sample floor: below this, GARCH/tail estimates are unreliable
SMALL_N = 500
# tractability cap: above this, use the most RECENT MAX_N returns (event-bars like BTC dollar have ~2.7M bars ->
# GARCH/Hurst on millions of points is impractical). Contiguous tail keeps GARCH valid; time-bar series are far below.
MAX_N = 250_000

# chimera proxy feature map: canonical metric -> chimera column whose mean we report alongside
PROXY_MAP = {
    "garch_persistence": "norm_vol_cluster",
    "hurst": "hurst_regime",
    "excess_kurtosis": "norm_return_kurtosis",
    "jump_fraction": "rv_jump_frac",
    "annualized_vol": "norm_yz_volatility",
    "predictability": "norm_perm_entropy",
    "kyle": "norm_kyle_lambda",
}


def _norm_sym(s: str) -> str:
    s = s.upper()
    return s if s.endswith("USDT") else s + "USDT"


# ----------------------------------------------------------------------------- estimators

def hill_tail_index(x: np.ndarray, side: str, frac: float = 0.05) -> float:
    """Hill estimator of the tail index alpha for one tail, using the top ~frac order statistics.

    For the RIGHT tail we use the largest positive values; for the LEFT tail we negate and use the
    most-negative values (magnitudes). alpha = 1 / mean(log(x_(i)/x_(k))) over the k tail exceedances.
    Lower alpha = heavier tail. Equity 'cubic law' alpha ~ 3.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if side == "left":
        mags = -x[x < 0]           # magnitudes of negative returns
    else:
        mags = x[x > 0]
    mags = np.sort(mags)
    n = len(mags)
    if n < 30:
        return float("nan")
    k = max(10, int(np.ceil(frac * n)))     # number of upper order statistics
    k = min(k, n - 1)
    top = mags[-(k + 1):]                    # k exceedances + the threshold
    thresh = top[0]
    if thresh <= 0:
        return float("nan")
    exceed = top[1:]
    logs = np.log(exceed / thresh)
    m = np.mean(logs)
    if m <= 0:
        return float("nan")
    return float(1.0 / m)


def hurst_rs(x: np.ndarray, min_chunk: int = 8) -> float:
    """Hurst exponent via classical Rescaled-Range (R/S) analysis.

    For a series of length N, split into non-overlapping windows of size n (a geometric ladder),
    compute the rescaled range R/S per window, average, then fit log(R/S) ~ H*log(n). H~0.5 = random
    walk; H>0.5 = persistent/long-memory; H<0.5 = anti-persistent.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    N = len(x)
    if N < 100:
        return float("nan")
    # geometric ladder of window sizes
    sizes = np.unique(np.floor(np.logspace(np.log10(min_chunk),
                                           np.log10(N // 2), 16)).astype(int))
    sizes = sizes[sizes >= min_chunk]
    logn, logrs = [], []
    for n in sizes:
        n = int(n)
        n_chunks = N // n
        if n_chunks < 1:
            continue
        rs_vals = []
        for i in range(n_chunks):
            seg = x[i * n:(i + 1) * n]
            mean = seg.mean()
            dev = np.cumsum(seg - mean)
            R = dev.max() - dev.min()
            S = seg.std(ddof=0)
            if S > 1e-12 and R > 0:
                rs_vals.append(R / S)
        if rs_vals:
            logn.append(np.log(n))
            logrs.append(np.log(np.mean(rs_vals)))
    if len(logn) < 4:
        return float("nan")
    H = np.polyfit(np.array(logn), np.array(logrs), 1)[0]
    return float(H)


def hurst_dfa(x: np.ndarray, min_chunk: int = 8) -> float:
    """Hurst exponent via Detrended Fluctuation Analysis (DFA).

    Integrate the mean-removed series, split into windows of size n, detrend each window with a
    linear fit, compute the RMS fluctuation F(n), then fit log F(n) ~ alpha*log(n). The DFA exponent
    alpha equals the Hurst exponent for a stationary series. ~0.5 random walk; >0.5 long memory.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    N = len(x)
    if N < 100:
        return float("nan")
    y = np.cumsum(x - x.mean())             # integrated (cumulative) profile
    sizes = np.unique(np.floor(np.logspace(np.log10(min_chunk),
                                           np.log10(N // 4), 16)).astype(int))
    sizes = sizes[sizes >= min_chunk]
    logn, logf = [], []
    for n in sizes:
        n = int(n)
        n_chunks = N // n
        if n_chunks < 1:
            continue
        idx = np.arange(n)
        fluct = []
        for i in range(n_chunks):
            seg = y[i * n:(i + 1) * n]
            coeff = np.polyfit(idx, seg, 1)
            trend = np.polyval(coeff, idx)
            fluct.append(np.mean((seg - trend) ** 2))
        F = np.sqrt(np.mean(fluct))
        if F > 1e-12:
            logn.append(np.log(n))
            logf.append(np.log(F))
    if len(logn) < 4:
        return float("nan")
    alpha = np.polyfit(np.array(logn), np.array(logf), 1)[0]
    return float(alpha)


def ljung_box_p(x: np.ndarray, lags: int = 20) -> float:
    """Ljung-Box test p-value at a given lag. p>0.05 => no remaining linear autocorrelation."""
    from statsmodels.stats.diagnostic import acorr_ljungbox
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < lags + 10:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = acorr_ljungbox(x, lags=[lags], return_df=True)
    return float(res["lb_pvalue"].iloc[-1])


def adf_test(x: np.ndarray) -> dict:
    """Augmented Dickey-Fuller. Null = unit root (non-stationary). Reject (p<0.05) => stationary."""
    from statsmodels.tsa.stattools import adfuller
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, *_ = adfuller(x, autolag="AIC")
    return {"stat": float(stat), "p": float(p),
            "verdict": "STATIONARY (reject unit root)" if p < 0.05 else "non-stationary (cannot reject)"}


def kpss_test(x: np.ndarray) -> dict:
    """KPSS. Null = stationary. p<0.05 => reject stationarity (complementary to ADF)."""
    from statsmodels.tsa.stattools import kpss
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, *_ = kpss(x, regression="c", nlags="auto")
    return {"stat": float(stat), "p": float(p),
            "verdict": "STATIONARY (cannot reject null)" if p > 0.05 else "non-stationary (reject null)"}


def jarque_bera(x: np.ndarray) -> dict:
    """Jarque-Bera normality test. p<0.05 => reject normality (the expected result for crypto)."""
    from scipy import stats
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    stat, p = stats.jarque_bera(x)
    return {"stat": float(stat), "p": float(p),
            "verdict": "NON-NORMAL (reject)" if p < 0.05 else "cannot reject normality"}


def fit_garch(ret: np.ndarray) -> dict:
    """GARCH(1,1) with Student-t innovations. Returns are scaled x100 for arch's optimizer.

    persistence = alpha1 + beta1 (near 1.0 = near-integrated, shocks decay slowly);
    half-life = log(0.5)/log(persistence) bars; nu = fitted Student-t dof (lower = fatter tail).
    """
    from arch import arch_model
    r = np.asarray(ret, float)
    r = r[np.isfinite(r)] * 100.0           # scale for optimizer stability (note: x100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
        res = am.fit(disp="off", show_warning=False)
    pr = res.params
    alpha = float(pr.get("alpha[1]", float("nan")))
    beta = float(pr.get("beta[1]", float("nan")))
    nu = float(pr.get("nu", float("nan")))
    persist = alpha + beta
    # half-life is only meaningful when persistence is strictly < 1 with margin; for a near-integrated
    # process (persist -> 1) the half-life diverges (shocks effectively never decay) -> report None.
    half_life = float(np.log(0.5) / np.log(persist)) if 0 < persist < 0.9999 else float("inf")
    return {"alpha": round(alpha, 5), "beta": round(beta, 5),
            "persistence": round(persist, 5),
            "half_life_bars": round(half_life, 2) if np.isfinite(half_life) else None,
            "nu_studentt_dof": round(nu, 3),
            "note": "returns scaled x100 for arch optimizer; persistence/nu are scale-invariant"}


def fit_gjr(ret: np.ndarray) -> dict:
    """GJR-GARCH (o=1) -- the leverage term gamma.

    arch parametrizes the asymmetry term gamma[1] on (negative-shock)^2: gamma>0 means NEGATIVE
    shocks raise vol MORE (the equity 'leverage effect'); gamma<=0 means positive shocks raise vol
    as much or more (the INVERTED/absent leverage that section II reports for BTC).
    """
    from arch import arch_model
    r = np.asarray(ret, float)
    r = r[np.isfinite(r)] * 100.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        am = arch_model(r, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t")
        res = am.fit(disp="off", show_warning=False)
    gamma = float(res.params.get("gamma[1]", float("nan")))
    if not np.isfinite(gamma):
        sign = "undetermined"
    elif gamma > 0.01:
        sign = "EQUITY sign (neg shocks raise vol more)"
    elif gamma < -0.01:
        sign = "INVERTED/crypto sign (pos shocks raise vol more)"
    else:
        sign = "ABSENT (~0, no leverage asymmetry)"
    return {"gamma": round(gamma, 5), "leverage_sign": sign}


def bns_jumps(ret: np.ndarray, k_sigma: float = 5.0, roll: int = 50) -> dict:
    """Barndorff-Nielsen-Shephard jump test + a simple threshold jump count.

    RV = sum r^2 (realized variance, includes jumps).
    BV = (pi/2) * sum |r_i| |r_{i-1}| (bipower variation, robust to jumps).
    jump_fraction = max(0, (RV-BV)/RV) -- share of variance attributable to jumps.
    Threshold count: bars with |r| > k_sigma * rolling_sigma, and their share of total variance.
    """
    r = np.asarray(ret, float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < roll + 5:
        return {"jump_fraction_bns": None, "rv": None, "bv": None,
                "threshold_jump_count": None, "threshold_jump_var_share": None}
    rv = float(np.sum(r ** 2))
    mu1 = np.sqrt(2.0 / np.pi)
    bv = float((1.0 / (mu1 ** 2)) * np.sum(np.abs(r[1:]) * np.abs(r[:-1])))
    jump_frac = max(0.0, (rv - bv) / rv) if rv > 0 else 0.0
    # threshold jumps vs a causal rolling sigma
    import polars as pl
    s = pl.Series(r)
    roll_sigma = s.rolling_std(window_size=roll, min_samples=roll // 2).to_numpy()
    valid = np.isfinite(roll_sigma) & (roll_sigma > 0)
    jump_mask = np.zeros(n, bool)
    jump_mask[valid] = np.abs(r[valid]) > k_sigma * roll_sigma[valid]
    n_jumps = int(jump_mask.sum())
    total_var = float(np.sum(r ** 2))
    jump_var = float(np.sum(r[jump_mask] ** 2))
    return {"jump_fraction_bns": round(jump_frac, 5),
            "rv": round(rv, 6), "bv": round(bv, 6),
            "threshold_jump_count": n_jumps,
            "threshold_jump_var_share": round(jump_var / total_var, 5) if total_var > 0 else None,
            "k_sigma": k_sigma, "roll": roll}


# ----------------------------------------------------------------------------- driver

def _proxy_means(df, mask: np.ndarray) -> dict:
    """Mean of each chimera proxy feature over the (masked) series."""
    out = {}
    for metric, col in PROXY_MAP.items():
        if col in df.columns:
            v = df[col].to_numpy().astype(float)[mask]
            v = v[np.isfinite(v)]
            out[metric] = float(np.mean(v)) if len(v) else None
        else:
            out[metric] = None
    return out


def compute_signature(sym: str, cadence: str, start=None, end=None) -> dict:
    df = ChimeraLoader().load(_norm_sym(sym), cadence=cadence).sort("date")
    d = df["date"].cast(str).to_numpy()
    mask = np.ones(len(d), bool)
    if start:
        mask &= (d >= start)
    if end:
        mask &= (d <= end)
    close_all = df["close"].to_numpy().astype(float)
    close = close_all[mask]; dts = d[mask]
    # log returns of close, dropping nulls/infs/non-positive prices
    valid_px = np.isfinite(close) & (close > 0)
    close = close[valid_px]; dts = dts[valid_px]
    ret = np.diff(np.log(close))
    dts_ret = dts[1:]                                   # ret[i] is dated at dts[i+1]
    fin = np.isfinite(ret)
    ret = ret[fin]; dts_ret = dts_ret[fin]
    n_full = len(ret)
    # CAP for tractability on high-count EVENT-bars (BTC dollar ~2.7M): use the most RECENT MAX_N (contiguous ->
    # GARCH valid). Time-bar series are far below MAX_N so they are untouched + bit-identical to before.
    capped_from = None
    if n_full > MAX_N:
        capped_from = n_full
        ret = ret[-MAX_N:]; dts_ret = dts_ret[-MAX_N:]
    n = len(ret)
    small = n < SMALL_N

    # annualization: known TIME cadences keep the fixed bars/year (preserves the verified section-II numbers); EVENT
    # bars (dollar/dib/range/runs_*/adaptive_vol) have NO fixed bars/day -> derive bars/year from the actual timestamps
    # of the (capped) sample, so the annualized vol is honest rather than implicitly assuming 1 bar/day.
    if cadence in BARS_PER_YEAR:
        bpy = float(BARS_PER_YEAR[cadence]); ann_src = "fixed (time cadence)"
    else:
        try:
            t0 = dt.date.fromisoformat(str(dts_ret[0])[:10]); t1 = dt.date.fromisoformat(str(dts_ret[-1])[:10])
            yrs = max((t1 - t0).days / 365.25, 1e-6)
            bpy = n / yrs; ann_src = f"derived (event-bar ~{round(bpy)} bars/yr)"
        except Exception:
            bpy = 365.0; ann_src = "fallback 365"
    af = float(np.sqrt(bpy))
    aret = np.abs(ret)
    ret2 = ret ** 2

    from scipy import stats
    mean = float(np.mean(ret)); std = float(np.std(ret, ddof=1))
    skew = float(stats.skew(ret)); exkurt = float(stats.kurtosis(ret, fisher=True))  # excess kurtosis
    hill_left = hill_tail_index(ret, "left")
    hill_right = hill_tail_index(ret, "right")
    ac1 = float(np.corrcoef(ret[:-1], ret[1:])[0, 1]) if n > 50 and std > 0 else float("nan")

    h_rs_ret = hurst_rs(ret); h_dfa_ret = hurst_dfa(ret)
    h_rs_abs = hurst_rs(aret); h_dfa_abs = hurst_dfa(aret)

    sig = {
        "asset": _norm_sym(sym), "cadence": cadence, "start": start, "end": end,
        "n_returns": n, "n_full_before_cap": capped_from, "small_n": small, "small_n_threshold": SMALL_N,
        "annualization_factor": round(af, 3), "bars_per_year": round(bpy, 1), "annualization_source": ann_src,

        "distribution_tails": {
            "mean": mean, "std": std,
            "mean_annualized": round(mean * bpy, 5),
            "std_annualized": round(std * af, 5),
            "skew": round(skew, 4), "excess_kurtosis": round(exkurt, 3),
            "hill_alpha_left": round(hill_left, 3) if np.isfinite(hill_left) else None,
            "hill_alpha_right": round(hill_right, 3) if np.isfinite(hill_right) else None,
            "cubic_law_check": _cubic_verdict(hill_left, hill_right),
            "jarque_bera": jarque_bera(ret),
        },
        "dependence_memory": {
            "ljung_box_ret_p": round(ljung_box_p(ret), 6),
            "ljung_box_absret_p": round(ljung_box_p(aret), 6),
            "ljung_box_ret2_p": round(ljung_box_p(ret2), 6),
            "ac1_ret": round(ac1, 5) if np.isfinite(ac1) else None,
            "hurst_ret_rs": round(h_rs_ret, 4) if np.isfinite(h_rs_ret) else None,
            "hurst_ret_dfa": round(h_dfa_ret, 4) if np.isfinite(h_dfa_ret) else None,
            "hurst_absret_rs": round(h_rs_abs, 4) if np.isfinite(h_rs_abs) else None,
            "hurst_absret_dfa": round(h_dfa_abs, 4) if np.isfinite(h_dfa_abs) else None,
            "interpretation": _memory_verdict(ljung_box_p(ret), ljung_box_p(aret),
                                              h_dfa_ret, h_dfa_abs, ac1),
        },
        "stationarity": {"adf": adf_test(ret), "kpss": kpss_test(ret)},
        "jumps": bns_jumps(ret),
    }
    # GARCH block (can fail to converge on pathological short samples)
    try:
        sig["volatility_process"] = {"garch11_t": fit_garch(ret), "gjr_garch": fit_gjr(ret)}
    except Exception as e:  # noqa: BLE001
        sig["volatility_process"] = {"error": f"GARCH fit failed: {type(e).__name__}: {e}"}

    sig["chimera_proxy_means"] = _proxy_means(df, mask)
    sig["reconciliation"] = _build_reconciliation(sig)
    if small:
        sig["WARNING"] = (f"n={n} < {SMALL_N} bars: GARCH and tail-index estimates are UNRELIABLE on "
                          f"this short sample; treat [SMALL-N]-tagged rows with caution.")
    return sig


def _cubic_verdict(left, right) -> str:
    vals = [v for v in (left, right) if np.isfinite(v)]
    if not vals:
        return "n/a"
    avg = np.mean(vals)
    if 2.0 <= avg <= 3.5:
        return f"YES -- alpha~{avg:.2f} in the crypto/cubic-law band [2,3.5] (fat-tailed, equity-like cubic regime)"
    if avg < 2.0:
        return f"HEAVIER than cubic -- alpha~{avg:.2f} < 2 (very fat tails)"
    return f"LIGHTER than cubic -- alpha~{avg:.2f} > 3.5 (tails thinner than equity cubic law)"


def _memory_verdict(lb_ret, lb_abs, h_ret, h_abs, ac1) -> str:
    parts = []
    if np.isfinite(lb_ret):
        if lb_ret > 0.05:
            parts.append("returns show NO linear predictability (efficient)")
        elif np.isfinite(ac1) and abs(ac1) < 0.05:
            parts.append("returns efficient in direction (LB rejects only on negligible bid-ask-bounce AC1, no edge)")
        else:
            parts.append("returns show MATERIAL linear autocorrelation")
    if np.isfinite(lb_abs):
        parts.append("STRONG volatility clustering in |ret|" if lb_abs < 0.05
                      else "no volatility clustering detected")
    if np.isfinite(h_ret):
        parts.append(f"Hurst(ret) DFA~{h_ret:.2f} ({'random walk' if 0.45 <= h_ret <= 0.55 else ('persistent' if h_ret > 0.55 else 'anti-persistent')})")
    if np.isfinite(h_abs):
        parts.append(f"Hurst(|ret|) DFA~{h_abs:.2f} ({'long memory in vol' if h_abs > 0.55 else 'no vol long-memory'})")
    if np.isfinite(ac1) and ac1 < -0.02:
        parts.append(f"AC1<0 ({ac1:.3f}) consistent with bid-ask bounce / microstructure reversal")
    return "; ".join(parts)


# ------------------------------------------------------------- reconciliation (the key new value)

# section-II documented reference values (literature / [RWYB-OURS]) -- quoted from
# docs/CRYPTO_MARKET_UNDERSTANDING.md section II
SECTION_II_REF = {
    "excess_kurtosis": {"ref": "6 to 26 (daily BTC; period-dependent)", "lit_lo": 6.0, "lit_hi": 26.0,
                        "kind": "high_positive"},
    "hill_alpha": {"ref": "tail index alpha ~ 2 to 3.5 (cubic-law band)", "lit_lo": 2.0, "lit_hi": 3.5,
                   "kind": "band"},
    "garch_persistence": {"ref": "alpha+beta ~ 1.0 (near-integrated; shocks decay slowly)",
                          "lit_lo": 0.95, "lit_hi": 1.0, "kind": "band"},
    "hurst_ret": {"ref": "~0.5 (raw returns ~ random walk; direction near-unpredictable)",
                  "lit_lo": 0.42, "lit_hi": 0.58, "kind": "band"},
    "hurst_absret": {"ref": ">0.5 (long memory in volatility)", "lit_lo": 0.55, "lit_hi": 1.0,
                     "kind": "gt"},
    "ljung_box_ret": {"ref": "insignificant (p>0.05; direction unpredictable / efficient)",
                      "kind": "p_gt_05"},
    "ljung_box_absret": {"ref": "significant (p<<0.05; volatility clustering)", "kind": "p_lt_05"},
    "leverage": {"ref": "inverted/absent in BTC (gamma<=0; pos shocks raise vol more)", "kind": "le_0"},
}


def _flag_band(val, lo, hi):
    if val is None or not np.isfinite(val):
        return "N/A"
    return "AGREE" if lo <= val <= hi else "DISAGREE"


def _build_reconciliation(sig: dict) -> list:
    """3-way table per metric: OUR estimate | section-II literature | chimera proxy mean + AGREE/DISAGREE."""
    dt = sig["distribution_tails"]; dm = sig["dependence_memory"]
    vp = sig.get("volatility_process", {})
    px = sig["chimera_proxy_means"]
    garch = vp.get("garch11_t", {}) if isinstance(vp, dict) else {}
    gjr = vp.get("gjr_garch", {}) if isinstance(vp, dict) else {}
    rows = []

    def add(metric, our_val, ref_key, proxy_key, agree, note=""):
        ref = SECTION_II_REF.get(ref_key, {})
        rows.append({
            "metric": metric,
            "our_computed": our_val,
            "section_II_reference": ref.get("ref", "n/a"),
            "chimera_proxy_feature": PROXY_MAP.get(proxy_key) if proxy_key else None,
            "chimera_proxy_mean": (round(px.get(proxy_key), 5) if proxy_key and px.get(proxy_key) is not None else None),
            "agree_with_section_II": agree,
            "note": note,
        })

    # excess kurtosis  <-> norm_return_kurtosis
    ek = dt["excess_kurtosis"]
    add("excess_kurtosis", ek, "excess_kurtosis", "excess_kurtosis",
        ("AGREE" if (ek is not None and ek >= 6.0) else ("PARTIAL (fat but <6)" if (ek is not None and ek > 1) else "DISAGREE")),
        "section II: 6-26 for daily BTC; high positive = fat tails")
    # Hill alpha (avg of both tails) <-> (no direct proxy)
    hl, hr = dt.get("hill_alpha_left"), dt.get("hill_alpha_right")
    havg = np.nanmean([v for v in (hl, hr) if v is not None]) if any(v is not None for v in (hl, hr)) else None
    add("hill_tail_alpha_avg", (round(float(havg), 3) if havg is not None and np.isfinite(havg) else None),
        "hill_alpha", None, _flag_band(havg, 2.0, 3.5),
        "section II cubic-law band [2,3.5]; lower alpha = heavier tail")
    # GARCH persistence <-> norm_vol_cluster
    persist = garch.get("persistence")
    add("garch_persistence", persist, "garch_persistence", "garch_persistence",
        _flag_band(persist, 0.95, 1.0),
        "section II: ~1.0 near-integrated; proxy norm_vol_cluster is z-scored (sign/level, not 1:1)")
    # Hurst(ret) DFA <-> hurst_regime
    hret = dm.get("hurst_ret_dfa")
    add("hurst_ret_dfa", hret, "hurst_ret", "hurst", _flag_band(hret, 0.42, 0.58),
        "section II: ~0.5 random walk; proxy hurst_regime tracks Hurst level")
    # Hurst(|ret|) DFA <-> hurst_regime (vol long-memory)
    habs = dm.get("hurst_absret_dfa")
    add("hurst_absret_dfa", habs, "hurst_absret", "hurst",
        ("AGREE" if (habs is not None and habs > 0.55) else "DISAGREE"),
        "section II: >0.5 long memory in vol")
    # Ljung-Box returns <-> norm_perm_entropy (predictability proxy)
    # NOTE: at large n (10k+ bars) the Ljung-Box test rejects on the tiny bid-ask-bounce AC1 (|AC1|~0.02-0.03),
    # which is statistically significant but ECONOMICALLY negligible -- this is exactly section II's "micro-scale
    # NEGATIVE AC (bid-ask bounce, Roll)" stylized fact, NOT directional predictability. So when LB rejects but
    # |AC1| is tiny and negative, that is AGREE-IN-SPIRIT (efficient-direction + microstructure), not a contradiction.
    lbret = dm.get("ljung_box_ret_p")
    ac1 = dm.get("ac1_ret")
    if lbret is not None and lbret > 0.05:
        lb_agree = "AGREE"
    elif ac1 is not None and abs(ac1) < 0.05:
        lb_agree = "AGREE-IN-SPIRIT (LB rejects at large-n on negligible bid-ask-bounce AC1; no directional edge)"
    else:
        lb_agree = "DISAGREE (material return autocorrelation)"
    add("ljung_box_ret_p", lbret, "ljung_box_ret", "predictability", lb_agree,
        f"section II: insignificant/efficient; AC1={ac1} (|AC1| tiny => microstructure, not predictability)")
    # Ljung-Box |returns| <-> norm_vol_cluster
    lbabs = dm.get("ljung_box_absret_p")
    add("ljung_box_absret_p", lbabs, "ljung_box_absret", "garch_persistence",
        ("AGREE" if (lbabs is not None and lbabs < 0.05) else "DISAGREE"),
        "section II: significant (vol clustering)")
    # jump fraction <-> rv_jump_frac
    jf = sig["jumps"].get("jump_fraction_bns")
    pj = px.get("jump_fraction")
    jagree = "N/A"
    if jf is not None and pj is not None:
        jagree = "AGREE" if (np.sign(jf - 0.05) == np.sign(pj - np.nanmedian([pj]))) or abs(jf - pj) < 0.2 else "CHECK"
    add("jump_fraction_bns", jf, None, "jump_fraction", jagree,
        "BNS RV-vs-BV jump share; proxy rv_jump_frac is the engineered counterpart")
    # leverage sign <-> (no direct proxy)
    gamma = gjr.get("gamma")
    add("leverage_gamma", gamma, "leverage", None,
        ("AGREE" if (gamma is not None and gamma <= 0.01) else "DISAGREE (equity-sign found)"),
        "section II: BTC leverage inverted/absent (gamma<=0)")
    # annualized vol <-> norm_yz_volatility (level vs z-score, informational only)
    add("annualized_vol", dt.get("std_annualized"), None, "annualized_vol", "INFO",
        "raw annualized sigma vs z-scored YZ-vol proxy (not a 1:1 comparison)")
    # kyle lambda proxy (informational; no whole-series canonical estimator here)
    add("kyle_lambda_proxy", None, None, "kyle", "INFO",
        "chimera microstructure proxy only; no whole-series canonical estimator computed")
    return rows


# ----------------------------------------------------------------------------- rendering

def render_text(sig: dict) -> str:
    if "error" in sig:
        return str(sig["error"])
    small = sig.get("small_n")
    tag = " [SMALL-N]" if small else ""
    L = []
    L.append(f"## ECONOMETRIC SIGNATURE -- {sig['asset']} -- {sig['cadence']} -- "
             f"{sig['start'] or 'FULL'} -> {sig['end'] or 'FULL'}  ({sig['n_returns']} returns)")
    if sig.get("WARNING"):
        L.append(f"!! WARNING: {sig['WARNING']}")
    L.append(f"annualization factor sqrt({sig['bars_per_year']}) = {sig['annualization_factor']}")

    dt = sig["distribution_tails"]
    L.append("\n[1] DISTRIBUTION & TAILS")
    L.append(f"  mean {dt['mean']:+.6f} (ann {dt['mean_annualized']:+.4f})   "
             f"std {dt['std']:.6f} (ann {dt['std_annualized']:.4f})")
    L.append(f"  skew {dt['skew']:+.3f}   EXCESS kurtosis {dt['excess_kurtosis']:+.3f}{tag}")
    L.append(f"  Hill tail-index alpha: LEFT {dt['hill_alpha_left']}  RIGHT {dt['hill_alpha_right']}{tag}")
    L.append(f"  cubic-law: {dt['cubic_law_check']}")
    jb = dt["jarque_bera"]
    L.append(f"  Jarque-Bera stat {jb['stat']:.1f} p {jb['p']:.3g} -> {jb['verdict']}")

    dm = sig["dependence_memory"]
    L.append("\n[2] DEPENDENCE & MEMORY")
    L.append(f"  Ljung-Box(20):  ret p={dm['ljung_box_ret_p']:.4g} (>0.05=no linear predictability)"
             f"   |ret| p={dm['ljung_box_absret_p']:.4g}   ret^2 p={dm['ljung_box_ret2_p']:.4g}  (<<0.05=vol clustering)")
    L.append(f"  AC1(ret) {dm['ac1_ret']}  (often <0 intraday = bid-ask bounce)")
    L.append(f"  Hurst(ret):   R/S {dm['hurst_ret_rs']}   DFA {dm['hurst_ret_dfa']}   (~0.5 = random walk)")
    L.append(f"  Hurst(|ret|): R/S {dm['hurst_absret_rs']}   DFA {dm['hurst_absret_dfa']}   (>0.5 = vol long memory)")
    L.append(f"  => {dm['interpretation']}")

    st = sig["stationarity"]
    L.append("\n[3] STATIONARITY")
    L.append(f"  ADF  stat {st['adf']['stat']:.3f}  p {st['adf']['p']:.3g}  -> {st['adf']['verdict']}")
    L.append(f"  KPSS stat {st['kpss']['stat']:.3f}  p {st['kpss']['p']:.3g}  -> {st['kpss']['verdict']}")

    vp = sig.get("volatility_process", {})
    L.append("\n[4] VOLATILITY PROCESS")
    if isinstance(vp, dict) and "garch11_t" in vp:
        g = vp["garch11_t"]
        L.append(f"  GARCH(1,1)-t: alpha {g['alpha']}  beta {g['beta']}  persistence {g['persistence']}{tag}  "
                 f"half-life {g['half_life_bars']} bars  nu(dof) {g['nu_studentt_dof']}")
        # CANONICAL CAVEAT: persistence at/near 1.0 is the IGARCH boundary -- near-integrated (>=0.99), NOT a precise
        # 1.0; the vol process is non-stationary on-sample (half-life > data). Risk meaning: a vol spike is the new
        # regime -- do NOT assume vol mean-reversion in sizing. (see docs/ECONOMETRIC_SIGNATURE.md canonical basis)
        if isinstance(g.get("persistence"), (int, float)) and g["persistence"] >= 0.999:
            L.append("    [IGARCH boundary] persistence ~1 = near-integrated; vol shocks ~permanent on-sample "
                     "(do NOT assume vol mean-reversion)")
        gj = vp["gjr_garch"]
        L.append(f"  GJR leverage gamma {gj['gamma']}  -> {gj['leverage_sign']}")
        L.append(f"  ({g['note']})")
    else:
        L.append(f"  {vp.get('error', 'GARCH unavailable')}")

    jm = sig["jumps"]
    L.append("\n[5] JUMPS (Barndorff-Nielsen-Shephard)")
    L.append(f"  RV {jm['rv']}  BV {jm['bv']}  -> jump fraction (RV-BV)/RV = {jm['jump_fraction_bns']}")
    L.append(f"  threshold jumps (|ret|>{jm.get('k_sigma')}*roll_sigma): count {jm['threshold_jump_count']}  "
             f"var-share {jm['threshold_jump_var_share']}")

    L.append("\n[RECONCILIATION]  canonical estimator vs section-II literature vs chimera engineered proxy")
    L.append(f"  {'metric':24s} {'OURS':>12s}  {'proxy(chimera)':>16s}  {'AGREE?':>10s}  section-II ref")
    for r in sig["reconciliation"]:
        ov = r["our_computed"]
        ovs = f"{ov:.4g}" if isinstance(ov, (int, float)) and ov is not None else str(ov)
        pm = r["chimera_proxy_mean"]
        pms = f"{pm:.4g}" if isinstance(pm, (int, float)) and pm is not None else "-"
        L.append(f"  {r['metric'][:24]:24s} {ovs:>12s}  {pms:>16s}  {r['agree_with_section_II']:>10s}  "
                 f"{r['section_II_reference'][:48]}")
    return "\n".join(L)


def selftest() -> int:
    """Estimator-correctness gate on KNOWN-property series (data-free, no chimera load). Proves the math is right,
    not just runnable. Returns 0 iff all checks pass. Re-run: `python -m mining.econometric_signature --selftest`."""
    rng = np.random.RandomState(7)
    fails = []

    def chk(name, cond, got):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {got}")
        if not cond:
            fails.append(name)

    # 1. iid Normal -> Hurst ~0.5 (random walk); thin-tailed
    z = rng.randn(8000)
    chk("iid-Normal Hurst(ret) ~0.5", abs(hurst_dfa(z) - 0.5) < 0.08, f"DFA={hurst_dfa(z):.3f} R/S={hurst_rs(z):.3f}")
    # 2. Student-t(3) -> Hill tail alpha ~3 (cubic)
    t3 = rng.standard_t(3, 8000)
    al, ar = hill_tail_index(t3, "left"), hill_tail_index(t3, "right")
    chk("Student-t(3) Hill alpha ~3", 2.2 <= np.mean([al, ar]) <= 3.6, f"L={al:.2f} R={ar:.2f}")
    # 3. MANUAL GARCH(1,1) sim, true persistence 0.08+0.90=0.98 -> recover + vol clustering + vol long-memory
    n = 12000; aa, bb, ww = 0.08, 0.90, 0.02
    s2 = np.zeros(n); r = np.zeros(n); s2[0] = ww / (1 - aa - bb)
    for i in range(1, n):
        s2[i] = ww + aa * r[i - 1] ** 2 + bb * s2[i - 1]; r[i] = np.sqrt(s2[i]) * rng.randn()
    g = fit_garch(r); persist = g.get("persistence")
    chk("GARCH sim recover persistence ~0.98", persist is not None and abs(persist - 0.98) < 0.03, f"got {persist}")
    chk("GARCH sim |ret| long memory >0.55", hurst_dfa(np.abs(r)) > 0.55, f"Hurst(|ret|) DFA={hurst_dfa(np.abs(r)):.3f}")
    chk("GARCH sim vol clustering (LB|ret| sig)", ljung_box_p(np.abs(r)) < 0.05, f"LB|ret| p={ljung_box_p(np.abs(r)):.2g}")
    # 4. jumps: spiky >> smooth
    smooth = rng.randn(5000) * 0.01
    spiky = smooth.copy(); spiky[::200] += rng.choice([-1, 1], 25) * 0.2
    jf_s, jf_k = bns_jumps(smooth)["jump_fraction_bns"], bns_jumps(spiky)["jump_fraction_bns"]
    chk("BNS jumps spiky >> smooth", jf_k > 0.2 and jf_s < 0.05, f"smooth={jf_s} spiky={jf_k}")

    print(f"\nSELFTEST: {'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}  "
          "(estimators verified on iid-Normal / Student-t(3) / simulated-GARCH(0.98) / jump series)")
    return 0 if not fails else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mining.econometric_signature",
                                 description="Econometric SIGNATURE -- canonical time-series estimators "
                                             "over the WHOLE return series, reconciled vs section-II literature "
                                             "+ chimera proxy features.")
    ap.add_argument("--asset", help="asset symbol (e.g. BTC, ETHUSDT)")
    ap.add_argument("--cadence", default="4h", help="timeframe: 1d|4h|1h|30m|15m")
    ap.add_argument("--start", help="ISO date window start (default: whole series)")
    ap.add_argument("--end", help="ISO date window end (default: whole series)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--selftest", action="store_true", help="run the estimator-correctness gate (data-free) + exit")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()
    if not a.asset:
        ap.error("--asset is required (or use --selftest)")

    sig = compute_signature(a.asset, a.cadence, a.start, a.end)
    tag = f"{_norm_sym(a.asset).replace('USDT','').lower()}_{a.cadence}"
    outpath = OUT / f"econ_signature_{tag}.json"
    outpath.write_text(json.dumps(sig, indent=2, default=str), encoding="utf-8")
    print(json.dumps(sig, indent=2, default=str) if a.json else render_text(sig))
    print(f"\n[written] {outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
