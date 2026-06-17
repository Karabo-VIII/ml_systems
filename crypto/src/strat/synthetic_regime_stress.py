"""src/strat/synthetic_regime_stress.py -- PHASE 3 LINCHPIN: the SYNTHETIC REGIME-STRESS test.

THE OPEN THREAD every prior phase flagged: the 2020 OOS (Oct-Dec) is a ~0%-bear monotone BULL. So:
  (1) the trend<->MR COMPLEMENTARITY was shown to DD-dampen, but could NOT be regime-tested -- in a bull
      the trend sleeve is structurally favored and never gets a sustained whipsaw to be rescued from.
  (2) the DYNAMIC allocation engine beat the static blend at only 1 of 6 TFs (30m -- near the
      multiple-comparisons chance rate) BECAUSE in a monotone bull there is nothing to time. A
      regime-conditional allocator can only EARN its complexity when there ARE multiple regimes.

This module builds the decisive test WITHOUT touching held-out real data: a SYNTHETIC generator
CALIBRATED ON 2020-BAND DATA ONLY, producing bull / bear / chop / crash regimes + a STITCHED
full-cycle path, then runs the EXACT deployable sleeve/blend/engine code on those synthetic panels and
reports DISTRIBUTIONS over >=20 seeds (mean +- spread + worst-case path -- never a cherry-picked seed).

HOW IT STAYS HONEST (the load-bearing caveats):
  - The generator is CALIBRATED to 2020 stylized facts ONLY (moments per regime extracted from real
    2020-band data: Mar-2020 crash = bear/crash exemplar; Apr-Jul-2020 = chop; Oct-Dec-2020 = bull).
    NO 2026 / no other real data is ever read. (calibrate_2020() is the only place real data is touched,
    and it is window-fenced to 2020-01-01..2021-01-01.)
  - The generator is VALIDATED against the 2020 stylized facts BEFORE its results are trusted: synthetic
    vs real-2020 return distribution + ACF + vol-clustering overlay, with a numeric match report. An
    UNCALIBRATED / UNVALIDATED generator proves nothing -- that is the binding caveat.
  - The generator reproduces the crypto stylized facts the simple AR(1) generator in data_expansion.py
    deliberately does NOT: FAT TAILS (Student-t innovations), VOL CLUSTERING (GARCH(1,1)-like), DISTINCT
    REGIMES (per-regime drift/vol), and CROSS-ASSET CORRELATION (a shared BTC-beta factor + idio).
  - Synthetic NULL controls + multiple seeds; we report mean +- across seeds AND the worst-case path.

THE DECISIVE QUESTIONS (two-sided; report distributions, not single paths):
  (a) Does complementarity DD-dampening HOLD / STRENGTHEN in bear + chop (where trend whipsaws and MR
      should win)? Quantify the blend's maxDD reduction vs trend-alone, PER REGIME.
  (b) Does the DYNAMIC engine finally beat the static blend when there ARE multiple regimes
      (the stitched bull->crash->chop->recovery path)? This is the make-or-break for the dynamic engine.
      If it STILL does not beat static, the honest verdict is: the STATIC blend is the deployable answer
      and dynamic timing is not worth the complexity.
  (c) Does the 30m dynamic candidate's timing skill survive a regime flip, or was it a 2020-bull level
      effect (a multiple-comparisons artifact)?
  (d) Which strategy is the MOST ROBUST across the full regime mix (the "profitable, complementary,
      dynamic across regimes" goal)?

CONSTRAINTS (user mandate, BINDING): 2020 BAND ONLY for calibration; synthetic IS the test surface;
charts via matplotlib (Agg); no emoji (cp1252); RWYB; do NOT git commit.

RWYB:
  python -m strat.synthetic_regime_stress --selftest                 # generator + null soundness (no calib)
  python -m strat.synthetic_regime_stress --calibrate-only           # just the 2020 calibration + validation
  python -m strat.synthetic_regime_stress --seeds 20 --cadences 1d,30m   # the full regime-stress
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The deployable sleeve / blend / engine code we are stress-testing (NOT reinvented).
import strat.ma_2020_breakdown as M2                                  # noqa: E402  (the shared _panel)
import strat.deep2020_complementarity as COMP                        # noqa: E402
import strat.deep2020_osc as OSC                                     # noqa: E402
import strat.dynamic_allocation_engine as DAE                        # noqa: E402
from strat.portfolio_replay import MAKER_RT, TAKER_RT                # noqa: E402

OUT = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
CHARTS = OUT / "charts"

__contract__ = {
    "kind": "synthetic_regime_stress_test",
    "inputs": {
        "calibration": "per-regime moments (drift/vol/kurtosis/skew/AR1/vol-cluster/cross-corr) extracted "
                       "from REAL 2020-BAND data ONLY (Mar-2020 crash=bear; Apr-Jul=chop; Oct-Dec=bull); "
                       "no 2026/other data is ever read",
        "generator": "Student-t-innovation GARCH(1,1)-like per-asset price paths with a shared BTC-beta "
                     "factor + idio, per-regime params -> u10-like synthetic OHLC panels",
        "scenarios": "pure bull / pure bear / pure chop / a STITCHED bull->crash->chop->recovery full cycle",
        "strategies": "trend-alone / MR-alone / STATIC complementary blend / DYNAMIC engine / VOLTGT_BH / "
                      "buy-hold -- run via the EXACT deployable code on synthetic panels",
    },
    "outputs": {
        "per_regime_per_strategy": "net + maxDD distribution over >=20 seeds (mean +- spread + worst path)",
        "complementarity_dd_by_regime": "blend maxDD reduction vs trend-alone, per regime (does it pay more "
                                        "in bear/chop?)",
        "dynamic_vs_static_multi_regime": "does the dynamic engine beat the static blend on the STITCHED "
                                          "multi-regime path -- the make-or-break test",
        "verdict": "two-sided: dynamic-earns-its-complexity vs static-blend-is-the-answer; most-robust strat",
    },
    "invariants": {
        "calibrate_2020_only": "real data read ONLY in calibrate_2020(), window-fenced to 2020 band; never "
                               "2026/other",
        "generator_validated_before_trusted": "synthetic vs real-2020 dist+ACF+vol-cluster match reported; an "
                                              "uncalibrated/unvalidated generator proves NOTHING",
        "fat_tails_vol_clustering_regimes_xcorr": "generator reproduces the crypto stylized facts the simple "
                                                 "AR(1) generator omits (Student-t + GARCH + per-regime + beta)",
        "exact_deployable_code": "synthetic panels flow through the REAL sleeve/blend/engine via a _panel "
                                 "monkeypatch -- we test the deployable path, not a reimplementation",
        "distributions_not_single_paths": ">=20 seeds; report mean +- spread + WORST path; never cherry-pick",
        "two_sided_honest": "static-blend-is-the-answer is a real, valuable finding -- not a failure to hide",
        "long_only_gap_fill_is_dd_dampening": "both sleeves long-only -> the realistic win is DD-dampening / "
                                              "risk-adjusted, not return-rescue alpha",
        "no_double_count_maker": "synthetic panels run through the same MtM-no-double-count sleeve code; maker",
    },
}

# ============================================================================================
# 0. CALIBRATION CONFIG -- the 2020 regime exemplars (the ONLY real-data touch points)
# ============================================================================================
CALIB_WINDOW = ("2020-01-01", "2021-01-01")     # HARD FENCE: real data is only ever read inside this band
SYMS = COMP.SYMS                                 # the u10 basket
# regime exemplar sub-periods of 2020 (each a calibration target for that regime's moments)
REGIME_PERIODS = {
    "bear":  ("2020-02-15", "2020-03-25"),       # COVID crash = bear / crash exemplar
    "chop":  ("2020-04-01", "2020-07-15"),       # recovery-then-sideways stretch = chop / range exemplar
    "bull":  ("2020-10-01", "2021-01-01"),       # H2-2020 run = bull / trend exemplar (== the OOS)
}
N_BARS_REGIME = 92                               # default single-regime length (~ the 2020 OOS daily count)
# Per-regime synthetic lengths matched to the calibration-period DURATION (so a synthetic bear is ~the
# ~38-bar 2020 crash, NOT an unrealistically long 92-bar plunge that would over-state the catastrophe).
# Honest: a regime is calibrated on a finite 2020 stretch -- its synthetic length should not exceed that
# stretch's realism. The standalone scenarios still use a common 92 bars for a fair cross-regime net
# comparison; the STITCHED path uses the per-regime durations to build a realistic full cycle.
def _regime_bars():
    out = {}
    for rg, (s, e) in REGIME_PERIODS.items():
        out[rg] = int((pd.Timestamp(e) - pd.Timestamp(s)).days)
    return out
REGIME_BARS = _regime_bars()                     # {bear:~39, chop:~105, bull:~92} from the calibration spans


# ============================================================================================
# 1. CALIBRATION -- extract per-regime moments from REAL 2020-BAND data ONLY
# ============================================================================================
def _daily_returns_2020(sym, period):
    """Real daily returns for `sym` over a 2020-band sub-period. The ONLY real-data read; window-fenced."""
    s_ms = pd.Timestamp(period[0]).value // 10**6
    e_ms = pd.Timestamp(period[1]).value // 10**6
    # hard fence: never read outside the 2020 calibration band
    fence_s = pd.Timestamp(CALIB_WINDOW[0]).value // 10**6
    fence_e = pd.Timestamp(CALIB_WINDOW[1]).value // 10**6
    assert s_ms >= fence_s and e_ms <= fence_e, "calibration period escapes the 2020 band -- FORBIDDEN"
    try:
        o, h, l, c, ms = M2._panel(sym, "1d")
    except Exception:
        return None
    m = (ms >= s_ms) & (ms < e_ms)
    if m.sum() < 6:
        return None
    cc = c[m]
    r = np.diff(cc) / cc[:-1]
    return r[np.isfinite(r)]


def _moments(r):
    if r is None or len(r) < 5:
        return None
    a = np.abs(r)
    ac1 = float(np.corrcoef(r[:-1], r[1:])[0, 1]) if len(r) > 3 and np.std(r) > 0 else 0.0
    acabs = float(np.corrcoef(a[:-1], a[1:])[0, 1]) if len(r) > 3 and np.std(a) > 0 else 0.0
    return {"mean": float(np.mean(r)), "std": float(np.std(r)), "kurt": float(pd.Series(r).kurt()),
            "skew": float(pd.Series(r).skew()), "ar1": ac1 if np.isfinite(ac1) else 0.0,
            "vol_cluster": acabs if np.isfinite(acabs) else 0.0, "n": int(len(r))}


def _student_t_df_from_kurt(excess_kurt):
    """Map an empirical EXCESS kurtosis to a Student-t degrees-of-freedom. For a t with df>4,
    excess kurt = 6/(df-4) -> df = 4 + 6/excess_kurt. Clip to [4.5, 30] (df<=4 has infinite kurt)."""
    ek = max(float(excess_kurt), 0.05)
    df = 4.0 + 6.0 / ek
    return float(np.clip(df, 4.5, 30.0))


def calibrate_2020():
    """Extract per-regime calibration parameters from REAL 2020-band data ONLY (cross-asset averaged).
    Returns {regime -> params} + the per-regime pooled REAL return samples (for the validation overlay)
    + cross-asset correlation / BTC-beta. This is the ONLY function that reads real market data."""
    calib = {}
    real_samples = {}                            # regime -> pooled real daily returns (for validation)
    for rg, period in REGIME_PERIODS.items():
        per_asset = []
        pooled = []
        for sym in SYMS:
            r = _daily_returns_2020(sym, period)
            if r is None:
                continue
            mo = _moments(r)
            if mo:
                per_asset.append(mo)
                pooled.append(r)
        if not per_asset:
            continue
        agg = {k: float(np.nanmean([m[k] for m in per_asset]))
               for k in ("mean", "std", "kurt", "skew", "ar1", "vol_cluster")}
        agg["n_assets"] = len(per_asset)
        agg["t_df"] = _student_t_df_from_kurt(agg["kurt"])
        calib[rg] = agg
        real_samples[rg] = np.concatenate(pooled) if pooled else np.array([])

    # cross-asset correlation + BTC-beta (from the BULL regime, the cleanest co-move read)
    bull = REGIME_PERIODS["bull"]
    sms = pd.Timestamp(bull[0]).value // 10**6
    ems = pd.Timestamp(bull[1]).value // 10**6
    cols = {}
    for sym in SYMS:
        try:
            o, h, l, c, ms = M2._panel(sym, "1d")
        except Exception:
            continue
        m = (ms >= sms) & (ms < ems)
        if m.sum() < 6:
            continue
        cols[sym] = pd.Series(np.diff(c[m]) / c[m][:-1])
    xcorr = 0.55
    btc_beta = 0.5
    if len(cols) >= 3:
        df = pd.DataFrame(cols).dropna()
        cc = df.corr().values
        iu = np.triu_indices_from(cc, 1)
        xcorr = float(np.nanmean(cc[iu]))
        if "BTCUSDT" in df.columns:
            # beta of each alt on BTC (avg slope) as the shared-factor loading proxy
            btc = df["BTCUSDT"].to_numpy()
            vb = np.var(btc) + 1e-12
            betas = [float(np.cov(df[s].to_numpy(), btc)[0, 1] / vb) for s in df.columns if s != "BTCUSDT"]
            btc_beta = float(np.nanmean(betas))
    calib["_xasset"] = {"mean_pairwise_corr": round(xcorr, 3), "mean_btc_beta": round(btc_beta, 3),
                        "n_assets": len(cols)}
    calib["_meta"] = {"calib_window": CALIB_WINDOW, "regime_periods": REGIME_PERIODS,
                      "note": "moments extracted from REAL 2020-band data ONLY -- no 2026/other data read"}
    return calib, real_samples


# ============================================================================================
# 2. THE GENERATOR -- Student-t-innovation GARCH(1,1)-like, per-regime, shared BTC-beta factor
# ============================================================================================
def _garch_t_path(n_bars, drift, base_vol, t_df, vol_cluster, ar1, rng, common_shock=None,
                  beta=0.0):
    """One asset's synthetic daily-return path with the crypto stylized facts:
       - FAT TAILS:        Student-t(t_df) innovations (scaled to unit variance)
       - VOL CLUSTERING:   GARCH(1,1)-like recursion sigma_t^2 = w + a*eps_{t-1}^2 + b*sigma_{t-1}^2,
                           with (a+b) set so the unconditional vol matches base_vol and the persistence
                           tracks the empirical |r| autocorrelation (vol_cluster).
       - AUTOCORR:         a small AR(1) on the mean (ar1) -- crypto daily AR1 is weak/negative.
       - SHARED FACTOR:    if common_shock is given, blend beta*common_shock + sqrt(1-beta^2)*idio so the
                           cross-asset correlation matches the calibrated BTC-beta.
    Returns the n_bars return array."""
    # GARCH persistence from the |r| autocorrelation: alpha+beta_g ~ clip(vol_cluster*1.6, 0, 0.95).
    persist = float(np.clip(abs(vol_cluster) * 1.6, 0.0, 0.92))
    a_g = float(np.clip(persist * 0.35, 0.0, 0.30))      # ARCH term
    b_g = float(np.clip(persist - a_g, 0.0, 0.90))       # GARCH term
    uncond_var = base_vol ** 2
    w = uncond_var * (1.0 - a_g - b_g)
    # Student-t scaled to unit variance (var of t = df/(df-2))
    t_scale = np.sqrt((t_df - 2.0) / t_df) if t_df > 2 else 1.0

    sig2 = uncond_var
    eps_prev = 0.0
    r = np.empty(n_bars)
    mean_prev = 0.0
    for i in range(n_bars):
        sig2 = w + a_g * (eps_prev ** 2) + b_g * sig2
        sig = np.sqrt(max(sig2, 1e-12))
        if common_shock is not None:
            # standardized innovation = beta*common standardized shock + sqrt(1-b^2)*idio t-shock
            idio = rng.standard_t(t_df) * t_scale
            z = beta * common_shock[i] + np.sqrt(max(1.0 - beta ** 2, 0.0)) * idio
        else:
            z = rng.standard_t(t_df) * t_scale
        eps = sig * z
        # AR(1) on the demeaned return + per-bar drift
        r[i] = drift + ar1 * mean_prev + eps
        mean_prev = r[i] - drift
        eps_prev = eps
    return r


WARMUP_SYNTH = 160        # MA-warming prefix bars (covers the deployable slow-MA span ~149) before scoring


def generate_regime_panels(regime, calib, seed, n_bars=N_BARS_REGIME, syms=None, start_ts=None,
                           warmup_bars=WARMUP_SYNTH):
    """Generate u10-like synthetic OHLC panels for one regime. Returns {sym -> (o,h,l,c,ms)} matching the
    _panel signature so the deployable sleeve code can consume it. Daily bars; OHLC synthesized from the
    daily return with a small intrabar wick (vol-scaled) so high/low/MR oscillators have something to read.
    A SHARED BTC-beta common factor induces the calibrated cross-asset correlation. A warmup_bars CHOP
    prefix (timestamped BEFORE start_ts) warms the slow MAs so even a short regime is scorable; only the
    regime bars at/after start_ts are graded by the windowed sleeves."""
    syms = syms or SYMS
    p = calib[regime]
    xa = calib["_xasset"]
    beta = float(np.sqrt(np.clip(xa["mean_pairwise_corr"], 0.0, 0.95)))   # loading s.t. pairwise corr ~ beta^2
    rng = np.random.default_rng(seed)
    t_df = p["t_df"]
    t_scale = np.sqrt((t_df - 2.0) / t_df) if t_df > 2 else 1.0
    # WARMUP PREFIX: the deployable trend sleeve uses slow MAs (up to ~149 periods). A short synthetic
    # regime (e.g. the ~39-bar 2020 crash) has too few bars for those MAs to even compute -> the sleeve
    # returns nothing. We prepend warmup_bars of CHOP-regime history (a neutral, MA-warming prefix) so the
    # MAs are warm at regime entry. The warmup is generated from the CHOP params (neutral drift) so it does
    # NOT inject the test regime's trend into the scored window; only the regime portion is timestamped
    # at/after the synthetic fence start, so only it is graded. (When called by stitch_panels, warmup_bars
    # is 0 for ri>0 since the prior regime already warmed the MAs.)
    n_total = warmup_bars + n_bars
    warm_p = calib.get("chop", p)                                          # neutral warming regime
    # the shared market shock spanning warmup + regime
    common = rng.standard_t(t_df, n_total) * t_scale
    start_ts = start_ts or pd.Timestamp("2020-01-01")
    # timestamps: warmup bars sit BEFORE start_ts (negative offset); regime bars at/after start_ts.
    ms = ((np.arange(n_total) - warmup_bars) * 86400000 + (start_ts.value // 10**6)).astype(np.int64)
    panels = {}
    for k, sym in enumerate(syms):
        ar = np.random.default_rng(seed * 1000 + k)
        # modest per-asset drift/vol dispersion (alts are higher-beta/higher-vol than BTC)
        dscale = 1.0 + 0.25 * (k - len(syms) / 2) / max(1, len(syms))
        vscale = 1.0 + 0.15 * (k % 3 - 1)
        if warmup_bars > 0:
            r_warm = _garch_t_path(warmup_bars, warm_p["mean"] * dscale, warm_p["std"] * vscale,
                                   warm_p["t_df"], warm_p["vol_cluster"], warm_p["ar1"], ar,
                                   common_shock=common[:warmup_bars], beta=beta)
        else:
            r_warm = np.array([])
        r_reg = _garch_t_path(n_bars, p["mean"] * dscale, p["std"] * vscale, t_df, p["vol_cluster"],
                              p["ar1"], ar, common_shock=common[warmup_bars:], beta=beta)
        r = np.concatenate([r_warm, r_reg]) if warmup_bars > 0 else r_reg
        close = 100.0 * np.cumprod(1.0 + r)
        o = np.empty(n_total); o[0] = 100.0; o[1:] = close[:-1]
        # intrabar wick scaled by the bar's own |return| + a vol floor (so H>=max(o,c), L<=min(o,c))
        wick = (np.abs(r) * 0.6 + p["std"] * 0.3) * close
        hi = np.maximum(o, close) + np.abs(ar.normal(0, 1, n_total)) * wick * 0.5
        lo = np.minimum(o, close) - np.abs(ar.normal(0, 1, n_total)) * wick * 0.5
        lo = np.clip(lo, 1e-6, None)
        panels[sym] = (o, hi, lo, close, ms.copy())
    return panels


def stitch_panels(seq_regimes, calib, seed, n_bars_each=None, syms=None):
    """Build a STITCHED multi-regime path (e.g. bull->crash->chop->recovery): concatenate per-regime
    synthetic panels into ONE continuous price series per asset (carrying the price level forward across
    regime joins so the equity curve is continuous). Each regime uses its CALIBRATION-PERIOD duration
    (REGIME_BARS) so the cycle is realistic -- a synthetic bear is ~the 2020-crash length, not an
    over-stated 92-bar plunge. n_bars_each (if given) overrides per-regime durations with a fixed length.
    Returns {sym -> (o,h,l,c,ms)} + the regime-boundary bar indices (for charting / per-regime attribution)."""
    syms = syms or SYMS
    acc = {s: {"o": [], "h": [], "l": [], "c": [], "ms": []} for s in syms}
    boundaries = []
    cursor = 0
    level = {s: 100.0 for s in syms}
    base_ts = pd.Timestamp("2020-01-01")
    for ri, rg in enumerate(seq_regimes):
        nb = n_bars_each if n_bars_each else REGIME_BARS.get(rg, N_BARS_REGIME)
        # warmup prefix only on the FIRST regime; later regimes inherit warm MAs from the prior segment
        wb = WARMUP_SYNTH if ri == 0 else 0
        pan = generate_regime_panels(rg, calib, seed=seed * 100 + ri, n_bars=nb, syms=syms,
                                     start_ts=base_ts + pd.Timedelta(days=cursor), warmup_bars=wb)
        for s in syms:
            o, h, l, c, ms = pan[s]
            scale = level[s] / 100.0                      # carry the price level forward across the join
            acc[s]["o"].append(o * scale); acc[s]["h"].append(h * scale)
            acc[s]["l"].append(l * scale); acc[s]["c"].append(c * scale)
            acc[s]["ms"].append(ms)
            level[s] = float(c[-1] * scale)
        boundaries.append((cursor, rg))
        cursor += nb
    out = {}
    for s in syms:
        out[s] = (np.concatenate(acc[s]["o"]), np.concatenate(acc[s]["h"]),
                  np.concatenate(acc[s]["l"]), np.concatenate(acc[s]["c"]),
                  np.concatenate(acc[s]["ms"]).astype(np.int64))
    return out, boundaries


# ============================================================================================
# 3. GENERATOR VALIDATION -- synthetic vs real-2020 (dist + ACF + vol-clustering) BEFORE trusting it
# ============================================================================================
def validate_generator(calib, real_samples, seed=0, n_paths=20):
    """Generate many synthetic regime paths and compare their pooled moments + ACF + vol-clustering vs the
    REAL 2020 pooled samples per regime. Returns a numeric match report. The generator is only trusted if
    the synthetic moments bracket the real ones (mean/std/kurt sign+rough magnitude, ar1, vol_cluster)."""
    report = {}
    synth_pools = {}
    for rg in ("bear", "chop", "bull"):
        if rg not in calib or rg not in real_samples or real_samples[rg].size < 5:
            continue
        real = real_samples[rg]
        rm = _moments(real)
        syn_all = []
        for s in range(n_paths):
            # warmup_bars=0: validate the REGIME moments only (the chop warmup prefix would contaminate
            # the pooled moments). Validation cares about the return distribution, not MA warmth.
            pan = generate_regime_panels(rg, calib, seed=seed + s, warmup_bars=0)
            # pool the per-asset synthetic returns (matches how real_samples is pooled)
            for sym, (o, h, l, c, ms) in pan.items():
                syn_all.append(np.diff(c) / c[:-1])
        syn = np.concatenate(syn_all)
        synth_pools[rg] = syn
        sm = _moments(syn)
        # match flags: sign of mean agrees; std within 40%; kurt both elevated (>0 excess) or both ~0;
        # ar1 same sign-ish; vol_cluster same sign
        def _within(a, b, frac):
            return abs(a - b) <= frac * (abs(b) + 1e-9)
        match = {
            "mean_sign": bool(np.sign(sm["mean"]) == np.sign(rm["mean"]) or abs(rm["mean"]) < 0.002),
            "std_within_40pct": bool(_within(sm["std"], rm["std"], 0.40)),
            "kurt_both_fat": bool((sm["kurt"] > 0.5) == (rm["kurt"] > 0.5)),
            "vol_cluster_sign": bool(np.sign(sm["vol_cluster"]) == np.sign(rm["vol_cluster"])
                                     or abs(rm["vol_cluster"]) < 0.08),
        }
        report[rg] = {"real": {k: round(rm[k], 4) for k in ("mean", "std", "kurt", "skew", "ar1", "vol_cluster")},
                      "synth": {k: round(sm[k], 4) for k in ("mean", "std", "kurt", "skew", "ar1", "vol_cluster")},
                      "match": match, "all_match": bool(all(match.values())),
                      "n_real": int(real.size), "n_synth": int(syn.size)}
    n_ok = sum(1 for rg in report if report[rg]["all_match"])
    report["_summary"] = {"regimes_validated": len(report) - (1 if "_summary" in report else 0),
                          "regimes_all_match": n_ok,
                          "verdict": ("VALIDATED (synthetic brackets real 2020 stylized facts)"
                                      if n_ok >= 2 else
                                      "PARTIAL (some moments diverge -- read results with caution)")}
    return report, synth_pools


# ============================================================================================
# 4. RUN THE DEPLOYABLE STRATEGIES ON SYNTHETIC PANELS (via a _panel monkeypatch)
# ============================================================================================
# the synthetic series is timestamped from 2020-01-01; a stitched 4x92-bar path runs into early 2021.
# The deployable sleeves window to fixed 2020 sub-periods (COMP.WIN, DAE.RUNWAY, COMP.SPLIT). When running
# SYNTHETIC panels we must WIDEN those windows so every synthetic bar is scored (the windows only gate which
# bars are graded -- widening them changes nothing about the deployable cost/MtM logic). We restore them on
# exit. The synthetic REGIME bars are timestamped at/after 2020-01-01; the MA-warmup prefix sits BEFORE it.
# The fence START = 2020-01-01 so the warmup prefix loads (for MA history) but is NOT scored; the END is
# generous so every regime bar (incl. the stitched cycle running into 2021) is graded.
_SYNTH_FENCE = ("2020-01-01", "2021-12-01")


class _synthetic_panel_context:
    """Context manager: patch M2._panel (the shared loader) so the deployable sleeve / blend / engine code
    consumes synthetic panels instead of real market data, AND widen the sleeve scoring windows to cover the
    full synthetic span. We patch the SHARED reference that every sleeve module imported (M2._panel) plus
    each module's local binding, and temporarily set COMP.WIN / COMP.SPLIT / DAE.RUNWAY to the synthetic
    fence so all synthetic bars are graded. All originals are restored on exit."""
    def __init__(self, panels):
        self.panels = panels
        self._saved = []
        self._saved_consts = []

    def _fake_panel(self, sym, cadence):
        if sym not in self.panels:
            raise Exception(f"synthetic panel missing for {sym}")
        return self.panels[sym]

    def __enter__(self):
        targets = [(M2, "_panel"), (COMP, "_panel"), (OSC, "_panel"), (DAE, "_panel")]
        for mod, name in targets:
            if hasattr(mod, name):
                self._saved.append((mod, name, getattr(mod, name)))
                setattr(mod, name, self._fake_panel)
        # widen the sleeve scoring windows to the synthetic fence (restored on exit)
        const_targets = [(COMP, "WIN", _SYNTH_FENCE), (COMP, "SPLIT", _SYNTH_FENCE[0]),
                         (DAE, "RUNWAY", _SYNTH_FENCE)]
        for mod, name, newval in const_targets:
            if hasattr(mod, name):
                self._saved_consts.append((mod, name, getattr(mod, name)))
                setattr(mod, name, newval)
        return self

    def __exit__(self, *exc):
        for mod, name, orig in self._saved:
            setattr(mod, name, orig)
        for mod, name, orig in self._saved_consts:
            setattr(mod, name, orig)
        return False


def _perf(x, ann=365):
    """net% / Sharpe / maxDD% / block-bootstrap p05 of a daily-net array (mirrors the engine's _perf)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return {"net": None, "sharpe": None, "maxdd": None, "p05": None}
    eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq)
    from strat.data_expansion import block_bootstrap_distribution
    bb = block_bootstrap_distribution(x, n_boot=400, block=5, seed=13)
    return {"net": round(float((eq[-1] - 1) * 100), 1),
            "sharpe": round(float(np.mean(x) / (np.std(x) + 1e-12) * np.sqrt(ann)), 2),
            "maxdd": round(float(((eq - pk) / pk).min() * 100), 1),
            "p05": round(float(bb["p05"]) * 100, 1)}


def _align(*series):
    df = pd.concat([s.rename(str(i)) for i, s in enumerate(series)], axis=1).dropna()
    return df


def run_strategies_on_panel(panels, cad, dyn_weight_fn=None):
    """Run trend-alone / MR-alone / STATIC blend / DYNAMIC engine / VOLTGT_BH / buy-hold on the synthetic
    panels for one cadence, via the deployable sleeve code. Returns daily-net arrays for each strategy
    (aligned on the common daily index) + per-strategy _perf. dyn_weight_fn (optional) is the dynamic
    weighting (built INSIDE the patched context from the same sleeve daily nets)."""
    with _synthetic_panel_context(panels):
        # sleeves over the FULL synthetic series (the sleeve builders window to RUNWAY/WIN; we widen below)
        tnet, texp = COMP._trend_sleeve(cad)
        mnet, mexp = COMP._mr_sleeve(cad)
        bh = DAE._voltgt_bh_daily(cad)
    if tnet is None or mnet is None:
        return None
    df = _align(tnet, mnet)
    if len(df) < 12:
        return None
    t = df["0"]; m = df["1"]
    idx = df.index
    # VOLTGT_BH uses a long vol-window (e.g. 336 bars at 30m); on a SHORT synthetic regime that window
    # cannot fill -> an all-zero (degenerate) series. Detect that and mark VOLTGT as unavailable for this
    # scenario (reported None / excluded) rather than a misleading flat 0% with 0 DD.
    bh_arr = (bh.reindex(idx).fillna(0.0).to_numpy() if bh is not None else (0.5 * t + 0.5 * m).to_numpy())
    voltgt_degenerate = bool(np.allclose(bh_arr, 0.0))

    # buy-hold: equal-weight u10 synthetic close-to-close daily compound
    bh_book = []
    for sym, (o, h, l, c, ms) in panels.items():
        r = np.diff(c) / c[:-1]
        ix = pd.to_datetime(ms[1:], unit="ms")
        bh_book.append(pd.Series(r, index=ix))
    bh_eqw = pd.concat(bh_book, axis=1).mean(axis=1).reindex(idx).fillna(0.0).to_numpy()

    t_arr = t.to_numpy(); m_arr = m.to_numpy()
    # STATIC complementary blend: the canonical 50/50 (the deployable static answer; min-var is in-sample-fit)
    w_static = 0.5
    static_arr = w_static * t_arr + (1 - w_static) * m_arr

    # DYNAMIC engine: a causal regime-conditional weight from past-only trend-strength + recent perf,
    # applied lag-1. Reuses the deployable Tier-A rule mechanics (trend_strength -> w_trend, clipped).
    dyn_arr, w_series = _dynamic_book(t, m, bh_arr, idx)

    out = {
        "n_days": int(len(idx)),
        "TREND_ALONE": {"arr": t_arr, "perf": _perf(t_arr)},
        "MR_ALONE": {"arr": m_arr, "perf": _perf(m_arr)},
        "STATIC": {"arr": static_arr, "perf": _perf(static_arr)},
        "DYNAMIC": {"arr": dyn_arr, "perf": _perf(dyn_arr)},
        "VOLTGT_BH": {"arr": bh_arr,
                      "perf": ({"net": None, "sharpe": None, "maxdd": None, "p05": None}
                               if voltgt_degenerate else _perf(bh_arr))},
        "BUYHOLD": {"arr": bh_eqw, "perf": _perf(bh_eqw)},
        "_dyn_weight": w_series,
    }
    return out


def _dynamic_book(tnet, mnet, bh_arr, idx, roll=7):
    """The DYNAMIC engine's realized daily book on a synthetic series, using the deployable Tier-A regime
    rule (DAE.tier_a_weights) on past-only causal features built over non-overlapping rolling windows.
    Weight decided from windows < i, applied to window i's realized returns (lag-1 causal)."""
    t = tnet.to_numpy(); m = mnet.to_numpy()
    n = len(idx)
    # non-overlapping rolling windows
    wins = [(s, min(s + roll, n)) for s in range(0, n - 1, roll)]
    w_full = np.full(n, 0.5)
    w_series = []
    for wi, (lo, hi) in enumerate(wins):
        # past-only trend_strength from the PRIOR window's buy-hold proxy; perf_spread from last 3 windows
        if wi >= 1:
            plo, phi = wins[wi - 1]
            seg = bh_arr[plo:phi]
            ts = float(np.clip(abs(np.sum(seg)) / (np.sum(np.abs(seg)) + 1e-12), 0.0, 1.0)) if len(seg) > 1 else 0.5
        else:
            ts = 0.5
        rt, rm = [], []
        for j in range(max(0, wi - 3), wi):
            jlo, jhi = wins[j]
            rt.append(float(np.prod(1 + t[jlo:jhi]) - 1) * 100)
            rm.append(float(np.prod(1 + m[jlo:jhi]) - 1) * 100)
        ps = (np.mean(rt) - np.mean(rm)) if rt else 0.0
        # deployable Tier-A rule (same construction as DAE.tier_a_weights)
        X = np.array([[ts, ps]]); fnames = ["trend_strength", "perf_spread"]
        w = float(DAE.tier_a_weights(X, fnames)[0])
        w_full[lo:hi] = w
        w_series.append((idx[lo], w, ts))
    dyn = w_full * t + (1 - w_full) * m
    return dyn, w_series


# ============================================================================================
# 5. THE STRESS RUN -- per regime + stitched, over many seeds, distributions not single paths
# ============================================================================================
STITCH_SEQUENCE = ["bull", "bear", "chop", "bull"]    # bull -> crash -> chop -> recovery (full cycle)
STRATS = ["TREND_ALONE", "MR_ALONE", "STATIC", "DYNAMIC", "VOLTGT_BH", "BUYHOLD"]


def run_stress(cadences, seeds, n_bars=None):
    """For each cadence: generate {bull,bear,chop,stitched} synthetic panels over `seeds` seeds, run all
    strategies, collect net/maxDD/Sharpe/p05 distributions. Returns the full results dict.
    Standalone bull/bear/chop each use their CALIBRATION-PERIOD duration (REGIME_BARS) for realism (a
    synthetic bear is ~the 2020-crash length, not over-stated); the stitched path chains those durations.
    n_bars (if given) overrides with a fixed length for all scenarios."""
    results = {}
    scenarios = ["bull", "bear", "chop", "stitched"]
    for cad in cadences:
        print(f"\n########## CADENCE {cad} -- synthetic regime-stress ({len(seeds)} seeds) ##########")
        cad_res = {sc: {st: {"net": [], "maxdd": [], "sharpe": [], "p05": []} for st in STRATS}
                   for sc in scenarios}
        cad_res["_dyn_weight_example"] = None
        cad_res["_stitch_boundaries"] = None
        cad_res["_scenario_nbars"] = {sc: (n_bars if n_bars else REGIME_BARS.get(sc, N_BARS_REGIME))
                                      for sc in ("bull", "bear", "chop")}
        cad_res["_equity_example"] = {}
        for si, seed in enumerate(seeds):
            for sc in scenarios:
                if sc == "stitched":
                    panels, bounds = stitch_panels(STITCH_SEQUENCE, _CALIB, seed, n_bars_each=n_bars)
                    if si == 0:
                        cad_res["_stitch_boundaries"] = bounds
                else:
                    nb = n_bars if n_bars else REGIME_BARS.get(sc, N_BARS_REGIME)
                    panels = generate_regime_panels(sc, _CALIB, seed=seed, n_bars=nb)
                res = run_strategies_on_panel(panels, cad)
                if res is None:
                    continue
                for st in STRATS:
                    p = res[st]["perf"]
                    if p["net"] is not None:
                        cad_res[sc][st]["net"].append(p["net"])
                        cad_res[sc][st]["maxdd"].append(p["maxdd"])
                        cad_res[sc][st]["sharpe"].append(p["sharpe"])
                        cad_res[sc][st]["p05"].append(p["p05"])
                # stash one example equity (first seed) for charts
                if si == 0:
                    cad_res["_equity_example"][sc] = {
                        st: list(np.cumprod(1 + np.asarray(res[st]["arr"])) * 100 - 100) for st in STRATS}
                    if sc == "stitched":
                        cad_res["_dyn_weight_example"] = res["_dyn_weight"]
            print(f"   seed {seed} done ({si + 1}/{len(seeds)})", end="\r")
        print()
        results[cad] = _summarize_cadence(cad_res, scenarios)
        _print_cadence_table(cad, results[cad], scenarios)
    return results


def _dist(vals):
    """mean +- std + worst (min for net/sharpe/p05; the most-negative maxdd is worst) + median + n."""
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return {"mean": None, "std": None, "worst": None, "median": None, "n": 0}
    return {"mean": round(float(np.mean(v)), 1), "std": round(float(np.std(v)), 1),
            "worst": round(float(np.min(v)), 1), "median": round(float(np.median(v)), 1),
            "p25": round(float(np.percentile(v, 25)), 1), "n": int(v.size)}


def _summarize_cadence(cad_res, scenarios):
    summary = {"_equity_example": cad_res["_equity_example"],
               "_dyn_weight_example": cad_res["_dyn_weight_example"],
               "_stitch_boundaries": cad_res["_stitch_boundaries"],
               "_scenario_nbars": cad_res.get("_scenario_nbars")}
    for sc in scenarios:
        summary[sc] = {}
        for st in STRATS:
            d = cad_res[sc][st]
            summary[sc][st] = {"net": _dist(d["net"]), "maxdd": _dist(d["maxdd"]),
                               "sharpe": _dist(d["sharpe"]), "p05": _dist(d["p05"])}
        # complementarity DD-reduction: blend maxDD vs trend-alone maxDD (paired, per seed)
        tdd = np.asarray([x for x in cad_res[sc]["TREND_ALONE"]["maxdd"] if x is not None], float)
        sdd = np.asarray([x for x in cad_res[sc]["STATIC"]["maxdd"] if x is not None], float)
        if tdd.size and sdd.size and tdd.size == sdd.size:
            # reduction = how much LESS-negative the blend DD is (positive = blend dampens DD)
            red = sdd - tdd
            summary[sc]["_dd_reduction_static_vs_trend"] = {
                "mean": round(float(np.mean(red)), 1), "std": round(float(np.std(red)), 1),
                "worst": round(float(np.min(red)), 1), "frac_seeds_blend_dampens": round(float(np.mean(red > 0)), 2),
                "n": int(red.size)}
        # dynamic vs static (paired net + Sharpe per seed)
        dnet = np.asarray([x for x in cad_res[sc]["DYNAMIC"]["net"] if x is not None], float)
        snet = np.asarray([x for x in cad_res[sc]["STATIC"]["net"] if x is not None], float)
        dsh = np.asarray([x for x in cad_res[sc]["DYNAMIC"]["sharpe"] if x is not None], float)
        ssh = np.asarray([x for x in cad_res[sc]["STATIC"]["sharpe"] if x is not None], float)
        if dnet.size and snet.size and dnet.size == snet.size and dnet.size >= 2:
            net_d = dnet - snet
            sh_d = (dsh - ssh) if (dsh.size == ssh.size and dsh.size) else None
            # one-sided paired SIGN TEST vs the 50% null (is dyn>static more often than a coin flip?) +
            # a one-sided paired t on the differences (effect-size-aware). Both must clear for "BEATS".
            p_sign_net = _sign_test_p(net_d)
            p_t_net = _paired_t_p(net_d)
            summary[sc]["_dynamic_vs_static"] = {
                "net_diff_mean": round(float(np.mean(net_d)), 1),
                "net_diff_std": round(float(np.std(net_d)), 1),
                "frac_seeds_dyn_beats_static_net": round(float(np.mean(net_d > 0)), 2),
                "sign_test_p_net": round(p_sign_net, 4),
                "paired_t_p_net": round(p_t_net, 4),
                "sharpe_diff_mean": round(float(np.mean(sh_d)), 2) if sh_d is not None else None,
                "frac_seeds_dyn_beats_static_sharpe": round(float(np.mean(sh_d > 0)), 2) if sh_d is not None else None,
                "sign_test_p_sharpe": round(_sign_test_p(sh_d), 4) if sh_d is not None else None,
                # "BEATS" only if BOTH the sign test AND the paired-t clear 0.05 one-sided AND the mean
                # advantage is materially positive (>1pp net) -- guards against a noise-driven verdict
                "significant_net": bool(p_sign_net < 0.05 and p_t_net < 0.05 and np.mean(net_d) > 1.0),
                "n": int(dnet.size)}
    return summary


def _sign_test_p(diffs):
    """One-sided paired SIGN TEST: P(>= observed #wins | coin-flip null p=0.5), via the binomial survival
    function. Ties dropped. Returns 1.0 if no non-tie pairs. This is the right small-sample gate -- a
    '67% of 3 seeds' beat rate is NOT significant (p=0.5), and this test says so."""
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d) & (d != 0)]
    n = d.size
    if n == 0:
        return 1.0
    wins = int(np.sum(d > 0))
    # binomial survival: sum_{k=wins}^{n} C(n,k) 0.5^n  (one-sided, H1: dyn beats static)
    from math import comb
    p = sum(comb(n, k) for k in range(wins, n + 1)) * (0.5 ** n)
    return float(min(1.0, p))


def _paired_t_p(diffs):
    """One-sided paired t-test p-value (H1: mean diff > 0), effect-size-aware. Uses a normal approx to the
    t-survival for dependency-free portability (no scipy). Returns 1.0 for degenerate input."""
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2 or np.std(d, ddof=1) < 1e-12:
        return 1.0 if np.mean(d) <= 0 else 0.0
    t = np.mean(d) / (np.std(d, ddof=1) / np.sqrt(n))
    # one-sided p via the standard-normal survival (n>=20 -> t~z; conservative-ish for small n)
    from math import erf
    p = 0.5 * (1.0 - erf(t / np.sqrt(2.0)))
    return float(np.clip(p, 0.0, 1.0))


def _print_cadence_table(cad, summ, scenarios):
    for sc in scenarios:
        print(f"   --- {sc.upper()} ---")
        print(f"     {'strategy':12} {'net% mean+-sd':>16} {'net worst':>10} {'maxDD mean':>11} {'maxDD worst':>12} {'Sharpe':>8}")
        for st in STRATS:
            e = summ[sc][st]
            net = e["net"]; dd = e["maxdd"]; sh = e["sharpe"]
            print(f"     {st:12} {str(net['mean'])+' +- '+str(net['std']):>16} {str(net['worst']):>10} "
                  f"{str(dd['mean']):>11} {str(dd['worst']):>12} {str(sh['mean']):>8}")
        ddr = summ[sc].get("_dd_reduction_static_vs_trend")
        dvs = summ[sc].get("_dynamic_vs_static")
        if ddr:
            print(f"     COMPLEMENTARITY: blend DD vs trend-alone = {ddr['mean']:+}pp (worst {ddr['worst']:+}pp); "
                  f"blend dampens DD in {ddr['frac_seeds_blend_dampens']:.0%} of seeds")
        if dvs:
            sig = "SIG" if dvs.get("significant_net") else "n.s."
            print(f"     DYNAMIC vs STATIC: net diff {dvs['net_diff_mean']:+}pp +-{dvs['net_diff_std']}; "
                  f"beats static net in {dvs['frac_seeds_dyn_beats_static_net']:.0%} of {dvs['n']} seeds "
                  f"(sign-p={dvs.get('sign_test_p_net')}, t-p={dvs.get('paired_t_p_net')}) -> {sig}")


# ============================================================================================
# 6. VERDICT (two-sided, honest)
# ============================================================================================
def build_verdict(results, validation):
    lines = []
    val_ok = validation["_summary"]["regimes_all_match"] >= 2
    lines.append(f"GENERATOR VALIDATION: {validation['_summary']['verdict']} "
                 f"({validation['_summary']['regimes_all_match']}/{validation['_summary']['regimes_validated']} "
                 f"regimes' synthetic moments match real-2020).")
    if not val_ok:
        lines.append("  >> CAVEAT: generator only PARTIALLY validated -- the results below are SUGGESTIVE, "
                     "not load-bearing. An uncalibrated generator proves nothing.")

    # (a) complementarity DD-dampening by regime (aggregate the frac-of-seeds blend dampens)
    comp_by_regime = {}
    for cad in results:
        for sc in ("bull", "bear", "chop", "stitched"):
            ddr = results[cad][sc].get("_dd_reduction_static_vs_trend")
            if ddr:
                comp_by_regime.setdefault(sc, []).append(ddr["mean"])
    lines.append("")
    lines.append("Q(a) DOES COMPLEMENTARITY DD-DAMPENING HOLD/STRENGTHEN IN BEAR + CHOP?")
    for sc in ("bull", "bear", "chop", "stitched"):
        if sc in comp_by_regime:
            vals = comp_by_regime[sc]
            lines.append(f"   {sc:9}: mean blend-DD-vs-trend-alone = {np.mean(vals):+.1f}pp across cadences "
                         f"({'DAMPENS' if np.mean(vals) > 0.3 else 'neutral/worse'})")

    # (b)/(c) dynamic vs static on the STITCHED multi-regime path -- the make-or-break.
    # GATE (statistical, NOT a loose fraction): "BEATS" requires a one-sided paired SIGN TEST p<0.05 AND a
    # paired-t p<0.05 AND a material mean advantage (>1pp net) -- a '67% of 3 seeds' beat rate is NOT a
    # result (sign-test p=0.5). This is the multiple-comparisons / small-sample discipline the prior
    # phases' raw '1-of-6' lacked.
    lines.append("")
    lines.append("Q(b)/(c) DOES THE DYNAMIC ENGINE BEAT THE STATIC BLEND WHEN REGIMES VARY (stitched)?")
    lines.append("   GATE: sign-test p<0.05 AND paired-t p<0.05 AND mean net advantage >1pp (NOT a raw beat-frac)")
    dyn_wins_stitched = []
    for cad in results:
        dvs = results[cad]["stitched"].get("_dynamic_vs_static")
        if dvs:
            beat = dvs["frac_seeds_dyn_beats_static_net"]
            sig = dvs.get("significant_net", False)
            verdict_c = "BEATS (sig)" if sig else "does NOT beat (n.s.)"
            if sig:
                dyn_wins_stitched.append(cad)
            lines.append(f"   {cad:5}: dyn net diff {dvs['net_diff_mean']:+}pp +-{dvs['net_diff_std']}; "
                         f"beats static net in {beat:.0%} of {dvs['n']} seeds; sign-p={dvs.get('sign_test_p_net')} "
                         f"t-p={dvs.get('paired_t_p_net')} -> {verdict_c}")

    # (d) most-robust strategy across the full regime mix (min net over scenarios, worst-seed)
    lines.append("")
    lines.append("Q(d) MOST ROBUST STRATEGY ACROSS THE FULL REGIME MIX (worst-scenario worst-seed net):")
    robust_rank = {}
    for cad in results:
        for st in STRATS:
            worst = min((results[cad][sc][st]["net"]["worst"]
                         for sc in ("bull", "bear", "chop", "stitched")
                         if results[cad][sc][st]["net"]["worst"] is not None), default=None)
            if worst is not None:
                robust_rank.setdefault(st, []).append(worst)
    rob = {st: float(np.mean(v)) for st, v in robust_rank.items() if v}
    for st in sorted(rob, key=lambda s: -rob[s]):
        lines.append(f"   {st:12}: mean worst-scenario worst-seed net = {rob[st]:+.1f}%")
    most_robust = max(rob, key=lambda s: rob[s]) if rob else None

    # HEADLINE
    n_cad = len(results)
    if dyn_wins_stitched and val_ok:
        headline = (f"DYNAMIC-EARNS-ITS-COMPLEXITY (partial): on the STITCHED multi-regime path the dynamic "
                    f"engine beat the static blend at {len(dyn_wins_stitched)}/{n_cad} cadences "
                    f"({dyn_wins_stitched}). Regime-timed weighting added value once regimes actually VARIED -- "
                    f"consistent with the 2020-bull-only result being a 'nothing to time' artifact. Verify on "
                    f"more seeds / real multi-regime data before deploying the dynamic layer.")
    elif dyn_wins_stitched:
        headline = (f"DYNAMIC-SUGGESTIVE (generator only partially validated): dynamic beat static on the "
                    f"stitched path at {dyn_wins_stitched}, but the generator did not fully match 2020 -- "
                    f"treat as a hypothesis, not a result.")
    else:
        headline = (f"STATIC-BLEND-IS-THE-ANSWER: even with DISTINCT bull/bear/chop/stitched regimes, the "
                    f"dynamic engine did NOT reliably beat the static 50/50 complementary blend on net or "
                    f"Sharpe at any of {n_cad} cadences. The 2020-bull '1-of-6' result was NOT merely a "
                    f"'nothing to time' artifact -- the engine's causal regime detection does not add reliable "
                    f"risk-adjusted value even when regimes vary. SHIP THE STATIC BLEND; dynamic timing is not "
                    f"worth the complexity. (Honest two-sided result -- a real finding, not a failure.)")
    lines.insert(0, f"HEADLINE: {headline}")
    lines.insert(1, "")

    lines.append("")
    lines.append("CAVEATS (binding): (1) SYNTHETIC data calibrated to 2020 stylized facts ONLY -- it is a "
                 "STRESS surface, not real future data; a generator can only reproduce the facts it was "
                 "calibrated on. (2) Long-only sleeves -> gap-fill is DD-DAMPENING, not return rescue; in a "
                 "deep synthetic bear BOTH sleeves lose (the realistic finding). (3) The dynamic engine here "
                 "uses the deployable Tier-A causal regime rule; (4) maker cost; (5) >=20 seeds, distributions "
                 "reported (mean +- spread + worst path), NO seed cherry-picked. (6) The 30m '1-of-6' 2020 "
                 "result is re-tested under regime variation here (Q(c)).")
    return {"headline": headline, "generator_validated": bool(val_ok),
            "dynamic_beats_static_stitched_cadences": dyn_wins_stitched,
            "complementarity_dd_by_regime": {k: round(float(np.mean(v)), 1) for k, v in comp_by_regime.items()},
            "most_robust_strategy": most_robust, "robustness_rank": rob, "lines": lines}


# ============================================================================================
# 7. CHARTS
# ============================================================================================
def chart_synthetic_regimes(calib, real_samples, validation, synth_pools, seed=0):
    """Chart 1: example bull/bear/chop/stitched synthetic price paths + the generator-validation overlay
    (synthetic vs real-2020 return distribution per regime)."""
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.1, 1.1, 1.0])

    # row 1: example synthetic price paths (BTC-proxy = first sym) per regime + stitched
    # warmup_bars=0 -> show only the REGIME portion (the chop-warming prefix is not part of the regime)
    for j, rg in enumerate(["bull", "bear", "chop"]):
        ax = fig.add_subplot(gs[0, j])
        pan = generate_regime_panels(rg, calib, seed=seed, warmup_bars=0)
        for k, (sym, (o, h, l, c, ms)) in enumerate(pan.items()):
            ax.plot(c, lw=1.0, alpha=0.55, color=plt.cm.viridis(k / 10))
        ax.set_title(f"synthetic {rg.upper()}\n(u10-like, calibrated to 2020 {rg})", fontsize=9)
        ax.set_ylabel("price (start=100)"); ax.axhline(100, color="k", lw=0.5, alpha=0.4)
    ax = fig.add_subplot(gs[0, 3])
    stitched, bounds = stitch_panels(STITCH_SEQUENCE, calib, seed)
    for k, (sym, (o, h, l, c, ms)) in enumerate(stitched.items()):
        ax.plot(c, lw=1.0, alpha=0.55, color=plt.cm.viridis(k / 10))
    for b, rg in bounds:
        ax.axvline(b, color="#d62728", ls="--", lw=0.8, alpha=0.6)
        ax.text(b, ax.get_ylim()[1] * 0.98, rg, fontsize=7, rotation=90, va="top", color="#d62728")
    ax.set_title("STITCHED full cycle\nbull->crash->chop->recovery", fontsize=9)
    ax.set_ylabel("price")

    # row 2: generator validation -- synthetic vs real-2020 return distribution per regime
    for j, rg in enumerate(["bull", "bear", "chop"]):
        ax = fig.add_subplot(gs[1, j])
        if rg in real_samples and real_samples[rg].size and rg in synth_pools:
            real = real_samples[rg] * 100
            syn = synth_pools[rg] * 100
            bins = np.linspace(min(real.min(), np.percentile(syn, 1)),
                               max(real.max(), np.percentile(syn, 99)), 40)
            ax.hist(real, bins=bins, density=True, alpha=0.55, color="#1f77b4", label="REAL 2020")
            ax.hist(syn, bins=bins, density=True, alpha=0.45, color="#ff7f0e", label="SYNTHETIC")
            v = validation.get(rg, {})
            rm = v.get("real", {}); sm = v.get("synth", {})
            ax.set_title(f"{rg}: real std={rm.get('std')} kurt={rm.get('kurt')}\n"
                         f"synth std={sm.get('std')} kurt={sm.get('kurt')} "
                         f"[{'MATCH' if v.get('all_match') else 'partial'}]", fontsize=8)
            ax.legend(fontsize=7); ax.set_xlabel("daily return %")
    # row 2 last: ACF / vol-clustering validation
    ax = fig.add_subplot(gs[1, 3])
    lags = range(1, 11)
    for rg, col in [("bear", "#d62728"), ("bull", "#2ca02c")]:
        if rg in real_samples and real_samples[rg].size > 12 and rg in synth_pools:
            real = real_samples[rg]; syn = synth_pools[rg][:len(real) * 5]
            def _acf_abs(x):
                a = np.abs(x - x.mean())
                return [float(np.corrcoef(a[:-k], a[k:])[0, 1]) if len(a) > k + 2 else 0 for k in lags]
            ax.plot(list(lags), _acf_abs(real), "-o", color=col, ms=3, label=f"{rg} real |r| ACF")
            ax.plot(list(lags), _acf_abs(syn), "--s", color=col, ms=3, alpha=0.6, label=f"{rg} synth |r| ACF")
    ax.axhline(0, color="k", lw=0.5); ax.set_title("vol-clustering: |r| ACF\n(real vs synthetic)", fontsize=8)
    ax.set_xlabel("lag"); ax.legend(fontsize=6)

    # row 3: the calibration table + validation verdict (text)
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    txt = ["GENERATOR CALIBRATION (from REAL 2020-band data ONLY) + VALIDATION:", ""]
    txt.append(f"{'regime':8} {'mean/d':>9} {'std/d':>8} {'kurt':>7} {'skew':>7} {'AR1':>7} {'volclust':>9} {'t_df':>6}")
    for rg in ("bull", "bear", "chop"):
        p = calib.get(rg, {})
        txt.append(f"{rg:8} {p.get('mean',0)*100:>8.2f}% {p.get('std',0)*100:>7.2f}% {p.get('kurt',0):>7.1f} "
                   f"{p.get('skew',0):>7.2f} {p.get('ar1',0):>7.2f} {p.get('vol_cluster',0):>9.2f} {p.get('t_df',0):>6.1f}")
    xa = calib.get("_xasset", {})
    txt.append(f"\ncross-asset: mean pairwise corr={xa.get('mean_pairwise_corr')}, mean BTC-beta={xa.get('mean_btc_beta')}")
    txt.append(f"VALIDATION VERDICT: {validation['_summary']['verdict']}")
    ax.text(0.01, 0.98, "\n".join(txt), fontsize=8.5, family="monospace", va="top", transform=ax.transAxes)

    fig.suptitle("SYNTHETIC REGIME GENERATOR -- example paths + 2020-calibration validation\n"
                 "calibrated to 2020 stylized facts ONLY (fat tails / vol clustering / per-regime / cross-corr); "
                 "an UNvalidated generator proves nothing", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = CHARTS / "synthetic_regimes_example.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"[figure] {p}")


def chart_strategy_by_regime(results, cadences):
    """Chart 2: THE KEY RESULT -- each strategy's net + maxDD across {bull,bear,chop,stitched}, mean+-seed
    spread, per cadence. Does dynamic > static when regimes vary?"""
    scenarios = ["bull", "bear", "chop", "stitched"]
    cs = [c for c in cadences if c in results]
    if not cs:
        return
    nrow = len(cs)
    fig, axes = plt.subplots(nrow, 2, figsize=(15, 4.2 * nrow), squeeze=False)
    colors = {"TREND_ALONE": "#1f77b4", "MR_ALONE": "#ff7f0e", "STATIC": "#2ca02c",
              "DYNAMIC": "#d62728", "VOLTGT_BH": "#9467bd", "BUYHOLD": "#7f7f7f"}
    for ri, cad in enumerate(cs):
        axn, axd = axes[ri][0], axes[ri][1]
        x = np.arange(len(scenarios)); width = 0.13
        for si, st in enumerate(STRATS):
            means = [results[cad][sc][st]["net"]["mean"] for sc in scenarios]
            stds = [results[cad][sc][st]["net"]["std"] or 0 for sc in scenarios]
            means = [m if m is not None else 0 for m in means]
            axn.bar(x + (si - 2.5) * width, means, width, yerr=stds, capsize=2,
                    color=colors[st], label=st, alpha=0.9)
            ddm = [results[cad][sc][st]["maxdd"]["mean"] if results[cad][sc][st]["maxdd"]["mean"] is not None else 0
                   for sc in scenarios]
            axd.bar(x + (si - 2.5) * width, ddm, width, color=colors[st], label=st, alpha=0.9)
        axn.set_xticks(x); axn.set_xticklabels(scenarios); axn.axhline(0, color="k", lw=0.6)
        axn.set_ylabel("net % (mean +- seed sd)"); axn.set_title(f"{cad}: NET by regime", fontsize=10)
        if ri == 0:
            axn.legend(fontsize=6, ncol=2)
        axd.set_xticks(x); axd.set_xticklabels(scenarios); axd.axhline(0, color="k", lw=0.6)
        axd.set_ylabel("maxDD % (mean)"); axd.set_title(f"{cad}: maxDD by regime (less-negative=better)", fontsize=10)
    fig.suptitle("STRATEGY PERFORMANCE BY REGIME (synthetic, >=20 seeds, mean +- spread) -- THE KEY RESULT\n"
                 "does DYNAMIC (red) beat STATIC (green) on the STITCHED multi-regime path? (long-only -> the "
                 "honest win is DD-dampening / risk-adjusted, not net alpha)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = CHARTS / "strategy_by_regime_perf.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"[figure] {p}")


def chart_complementarity_dd(results, cadences):
    """Chart 3: the blend's maxDD-reduction vs trend-alone, by regime (does complementarity pay MORE in
    bear/chop than in bull?)."""
    scenarios = ["bull", "bear", "chop", "stitched"]
    cs = [c for c in cadences if c in results]
    if not cs:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    x = np.arange(len(scenarios)); width = 0.8 / max(1, len(cs))
    for ci, cad in enumerate(cs):
        red = [(results[cad][sc].get("_dd_reduction_static_vs_trend") or {}).get("mean", 0) for sc in scenarios]
        frac = [(results[cad][sc].get("_dd_reduction_static_vs_trend") or {}).get("frac_seeds_blend_dampens", 0)
                for sc in scenarios]
        ax1.bar(x + (ci - len(cs) / 2) * width, red, width, label=cad, alpha=0.9)
        ax2.bar(x + (ci - len(cs) / 2) * width, frac, width, label=cad, alpha=0.9)
    ax1.set_xticks(x); ax1.set_xticklabels(scenarios); ax1.axhline(0, color="k", lw=0.6)
    ax1.set_ylabel("blend maxDD - trend-alone maxDD (pp)\n(positive = blend dampens DD)")
    ax1.set_title("COMPLEMENTARITY DD-dampening by regime\n(does the blend cut DD MORE in bear/chop?)", fontsize=10)
    ax1.legend(fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(scenarios); ax2.axhline(0.5, color="k", ls="--", lw=0.6)
    ax2.set_ylabel("fraction of seeds where blend dampens DD"); ax2.set_ylim(0, 1)
    ax2.set_title("fraction of seeds the blend DAMPENS DD vs trend-alone\n(>0.5 = reliable dampening)", fontsize=10)
    ax2.legend(fontsize=8)
    fig.suptitle("DOES COMPLEMENTARITY PAY MORE IN BEAR + CHOP? (synthetic, >=20 seeds)\n"
                 "the open thread: a 2020-bull OOS could not test this -- here trend whipsaws and MR should fill", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = CHARTS / "complementarity_dd_by_regime.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print(f"[figure] {p}")


# ============================================================================================
# 8. MAIN
# ============================================================================================
_CALIB = None        # filled by main() after calibrate_2020()


def main(argv=None):
    global _CALIB
    ap = argparse.ArgumentParser(prog="python -m strat.synthetic_regime_stress")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--calibrate-only", action="store_true", dest="calibrate_only")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--cadences", default="1d,30m")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    CHARTS.mkdir(parents=True, exist_ok=True)
    print("## SYNTHETIC REGIME-STRESS -- PHASE 3 linchpin (calibrate on 2020 ONLY; synthetic is the test)")
    print(f"   calibration window (HARD FENCE) = {CALIB_WINDOW} | regime exemplars = "
          f"{ {k: v for k, v in REGIME_PERIODS.items()} }")

    # 1. CALIBRATE on 2020 ONLY
    print("\n## CALIBRATING the generator on REAL 2020-band data ONLY ...")
    _CALIB, real_samples = calibrate_2020()
    for rg in ("bull", "bear", "chop"):
        p = _CALIB.get(rg, {})
        print(f"   {rg:6}: mean {p.get('mean',0)*100:+.2f}%/d  std {p.get('std',0)*100:.2f}%  kurt {p.get('kurt',0):.1f}  "
              f"skew {p.get('skew',0):+.2f}  AR1 {p.get('ar1',0):+.2f}  vol-clust {p.get('vol_cluster',0):+.2f}  "
              f"t_df {p.get('t_df',0):.1f}")
    xa = _CALIB["_xasset"]
    print(f"   cross-asset: mean pairwise corr {xa['mean_pairwise_corr']}, mean BTC-beta {xa['mean_btc_beta']}")

    # 2. VALIDATE the generator BEFORE trusting it
    print("\n## VALIDATING the generator vs real-2020 stylized facts ...")
    # validation uses a FIXED >=30 synthetic paths (decoupled from the run seed count) so the synthetic
    # moments are stable -- a tiny n_paths makes the moment match noisy (a small-sample artifact, not a
    # generator defect). The generator's validity does not depend on how many stress seeds we run.
    validation, synth_pools = validate_generator(_CALIB, real_samples, seed=0, n_paths=max(30, a.seeds))
    for rg in ("bull", "bear", "chop"):
        if rg in validation:
            v = validation[rg]
            print(f"   {rg:6}: real(std={v['real']['std']},kurt={v['real']['kurt']},vc={v['real']['vol_cluster']}) "
                  f"vs synth(std={v['synth']['std']},kurt={v['synth']['kurt']},vc={v['synth']['vol_cluster']}) "
                  f"-> {'MATCH' if v['all_match'] else 'partial'} {v['match']}")
    print(f"   VALIDATION: {validation['_summary']['verdict']}")

    if a.calibrate_only:
        chart_synthetic_regimes(_CALIB, real_samples, validation, synth_pools)
        print("\n[calibrate-only] done.")
        return 0

    # 3. THE STRESS RUN
    seeds = list(range(1, a.seeds + 1))
    cadences = [c.strip() for c in a.cadences.split(",") if c.strip()]
    print(f"\n## RUNNING the regime-stress over {len(seeds)} seeds x {len(cadences)} cadences "
          f"x 4 scenarios (bull/bear/chop/stitched) ...")
    results = run_stress(cadences, seeds)

    # 4. VERDICT
    verdict = build_verdict(results, validation)
    print("\n" + "=" * 100)
    print("## AGGREGATE VERDICT")
    for line in verdict["lines"]:
        print(f"   {line}")
    print("=" * 100)

    # 5. CHARTS
    chart_synthetic_regimes(_CALIB, real_samples, validation, synth_pools)
    chart_strategy_by_regime(results, cadences)
    chart_complementarity_dd(results, cadences)

    # 6. PERSIST
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    export = {
        "repro": {"command": "python -m strat.synthetic_regime_stress " + " ".join(argv or sys.argv[1:]),
                  "git_sha": sha, "cost_maker": MAKER_RT, "cost_taker": TAKER_RT,
                  "calib_window": CALIB_WINDOW, "regime_periods": REGIME_PERIODS,
                  "n_seeds": a.seeds, "cadences": cadences, "stitch_sequence": STITCH_SEQUENCE,
                  "n_bars_regime": N_BARS_REGIME, "universe": "u10",
                  "constraint": "CALIBRATE ON 2020 BAND ONLY; synthetic is the test surface; never touch 2026/other"},
        "calibration": {k: v for k, v in _CALIB.items() if not k.startswith("_") or k in ("_xasset", "_meta")},
        "generator_validation": validation,
        "results": _strip_arrays(results),
        "verdict": verdict,
    }
    p = OUT / "synthetic_regime_stress.json"
    json.dump(export, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    return 0


def _strip_arrays(results):
    """Drop the heavy example-equity arrays from the persisted JSON (keep the distributions + summaries)."""
    out = {}
    for cad, summ in results.items():
        out[cad] = {k: v for k, v in summ.items() if not k.startswith("_equity")}
        # keep a compact dyn-weight example + boundaries
    return out


# ============================================================================================
# 9. SELFTEST -- generator soundness + null controls (NO real-data calibration)
# ============================================================================================
def selftest():
    """Two-sided generator soundness (synthetic, no real-data calibration):
    POSITIVE: the generator REPRODUCES its input regime moments (drift sign, vol magnitude, fat tails from
              low t_df, vol-clustering from GARCH persistence, cross-asset corr from the shared factor).
    NEGATIVE: a NULL regime (zero drift, no clustering, near-Gaussian) must NOT exhibit manufactured
              trend / fat-tails / clustering -- the generator does not invent stylized facts."""
    print("## SYNTHETIC-REGIME-STRESS SELFTEST (two-sided generator soundness; no real-data calib)")
    ok = True

    # synthetic calibration (planted, NOT from real data)
    calib = {
        "bull": {"mean": 0.006, "std": 0.045, "kurt": 3.0, "skew": 0.3, "ar1": -0.02, "vol_cluster": 0.20,
                 "t_df": _student_t_df_from_kurt(3.0)},
        "bear": {"mean": -0.011, "std": 0.094, "kurt": 8.0, "skew": -1.5, "ar1": -0.30, "vol_cluster": 0.25,
                 "t_df": _student_t_df_from_kurt(8.0)},
        "null": {"mean": 0.0, "std": 0.02, "kurt": 0.0, "skew": 0.0, "ar1": 0.0, "vol_cluster": 0.0,
                 "t_df": 30.0},
        "_xasset": {"mean_pairwise_corr": 0.49, "mean_btc_beta": 0.5, "n_assets": 10},
    }

    # POSITIVE: bull drift > 0, bear drift < 0; bear vol > bull vol; bear kurt elevated
    # warmup_bars=0 -> the moment checks see the REGIME returns only (no chop-warming prefix contamination)
    def _pool(rg, seeds=12):
        allr = []
        for s in range(seeds):
            pan = generate_regime_panels(rg, calib, seed=s, n_bars=92, warmup_bars=0)
            for sym, (o, h, l, c, ms) in pan.items():
                allr.append(np.diff(c) / c[:-1])
        return np.concatenate(allr)

    bull = _pool("bull"); bear = _pool("bear"); null = _pool("null")
    bull_drift_ok = bull.mean() > 0
    bear_drift_ok = bear.mean() < 0
    vol_order_ok = bear.std() > bull.std() > null.std()
    bear_fat_ok = pd.Series(bear).kurt() > 1.5
    null_thin_ok = pd.Series(null).kurt() < 3.0
    print(f"  POSITIVE drift: bull {bull.mean()*100:+.2f}%/d (>0: {bull_drift_ok}), "
          f"bear {bear.mean()*100:+.2f}%/d (<0: {bear_drift_ok})")
    print(f"  POSITIVE vol order: bear {bear.std()*100:.1f}% > bull {bull.std()*100:.1f}% > null {null.std()*100:.1f}% "
          f"-> {vol_order_ok}")
    print(f"  POSITIVE fat tails: bear kurt {pd.Series(bear).kurt():.1f} (>1.5: {bear_fat_ok}); "
          f"NULL kurt {pd.Series(null).kurt():.1f} (<3: {null_thin_ok})")
    ok &= bull_drift_ok and bear_drift_ok and vol_order_ok and bear_fat_ok

    # vol-clustering: bull |r| AR1 should be positive (GARCH on); null ~ 0
    def _vc(x):
        a = np.abs(x - x.mean())
        return float(np.corrcoef(a[:-1], a[1:])[0, 1])
    vc_bull = _vc(bull); vc_null = _vc(null)
    vc_ok = vc_bull > 0.03 and abs(vc_null) < 0.15
    print(f"  POSITIVE vol-clustering: bull |r| AR1 {vc_bull:+.2f} (>0.03), null {vc_null:+.2f} (~0) -> {vc_ok}")
    ok &= vc_ok

    # cross-asset correlation: a single bull panel's assets should correlate ~ pairwise_corr target
    pan = generate_regime_panels("bull", calib, seed=3, n_bars=120, warmup_bars=0)
    rets = pd.DataFrame({s: np.diff(c) / c[:-1] for s, (o, h, l, c, ms) in pan.items()})
    cc = rets.corr().values; iu = np.triu_indices_from(cc, 1)
    xcorr = float(np.nanmean(cc[iu]))
    xcorr_ok = 0.20 <= xcorr <= 0.75      # target ~0.49; allow generous band (finite-sample + GARCH noise)
    print(f"  POSITIVE cross-asset corr: mean pairwise {xcorr:.2f} (target ~{calib['_xasset']['mean_pairwise_corr']}, "
          f"band [0.20,0.75]) -> {xcorr_ok}")
    ok &= xcorr_ok

    # NEGATIVE: the null regime produces NO trend (cumulative return near 0, both signs across seeds)
    cums = []
    for s in range(30):
        pan = generate_regime_panels("null", calib, seed=100 + s, n_bars=92, warmup_bars=0)
        c = list(pan.values())[0][3]
        cums.append(float(c[-1] / c[0] - 1))
    cums = np.array(cums)
    null_no_trend = abs(np.mean(cums)) < 0.05 and (cums > 0).mean() > 0.25 and (cums < 0).mean() > 0.25
    print(f"  NEGATIVE null no-trend: mean cum {np.mean(cums)*100:+.1f}% (|.|<5%), "
          f"pos-frac {(cums>0).mean():.2f} (both signs present) -> {null_no_trend}")
    ok &= null_no_trend

    # generator-validation logic sanity: a generator calibrated to bull should self-validate vs its own pool
    real_like = {"bull": bull, "bear": bear}
    rep, _ = validate_generator(calib, real_like, seed=0, n_paths=12)
    val_logic_ok = rep["bull"]["match"]["mean_sign"] and rep["bull"]["match"]["std_within_40pct"]
    print(f"  VALIDATION-LOGIC: synth-vs-own-pool bull match flags {rep['bull']['match']} -> {val_logic_ok}")
    ok &= val_logic_ok

    # STAT-GATE soundness (the multiple-comparisons / small-sample discipline): the sign-test must NOT
    # call '2 of 3 wins' significant (p=0.5), MUST call a clean 18-of-20 advantage significant, and a
    # NULL (zero-mean noise differences) must NOT be flagged significant.
    p_3 = _sign_test_p(np.array([1.0, 1.0, -1.0]))           # 2 of 3 wins
    p_18 = _sign_test_p(np.array([1.0] * 18 + [-1.0] * 2))   # 18 of 20 wins
    rng = np.random.default_rng(7)
    null_diffs = rng.normal(0, 5.0, 30)                       # zero-mean noise -> should be n.s.
    sig_null = (_sign_test_p(null_diffs) < 0.05 and _paired_t_p(null_diffs) < 0.05)
    real_diffs = rng.normal(4.0, 3.0, 25)                     # genuine +4pp advantage -> should be sig
    sig_real = (_sign_test_p(real_diffs) < 0.05 and _paired_t_p(real_diffs) < 0.05 and np.mean(real_diffs) > 1.0)
    stat_ok = (p_3 > 0.4) and (p_18 < 0.01) and (not sig_null) and sig_real
    print(f"  STAT-GATE: sign-p(2of3)={p_3:.3f} (>0.4, NOT sig), sign-p(18of20)={p_18:.4f} (<0.01); "
          f"null-diffs flagged sig={sig_null} (expect False); real-advantage flagged sig={sig_real} "
          f"(expect True) -> {'PASS' if stat_ok else 'FAIL'}")
    ok &= stat_ok

    print(f"\n  SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
