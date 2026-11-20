"""src/strat/core_satellite_book.py -- THE DEPLOYABLE CORE+SATELLITE BOOK (turnkey).

WHAT THIS IS (2026-06-13): the project's most-robust deployable book, assembled from its TWO
verified components into ONE turnkey system. NOT a discovery -- both pieces are already validated:

  CORE (beta -- the daily-return engine):
    src/strat/daily_engine.py ENGINE net = long-only u10 vol-target buy-hold + a causal regime
    DEFENSIVE overlay (trend/chop/down -> exposure scalar). Full-cycle compound ~+1970%, Sharpe
    ~1.4, maxDD ~-48%. HONEST: a BETA engine (no held-out alpha; the internal-data ceiling holds).

  SATELLITE (market-neutral carry -- the project's FIRST verified held-out positive):
    the cross-sectional funding-DISPERSION dollar-neutral carry (long low/neg-funding, short
    high-funding perps; gross1/net0). Net DEPLOYABLE OOS +7.9% / UNSEEN +10.3% compound, beta~0,
    Sharpe ~4.1, ann-vol ~2%, maxDD ~-1.85%. ~ZERO-correlated with the core (pearson ~0.015).
    HONEST: a MODEST yield sleeve (~3-10%/yr steady-state, decay-risk UNCONFIRMED) -- it
    diversifies + lifts risk-adjusted return; it does NOT 100x the book.

THE KEY DISCIPLINE (the assessment's blend_sketch was WRONG -- do NOT repeat it):
  funding_satellite_assessment vol-matched the satellite to the core => 5.4x-21.6x LEVERAGE =>
  absurd compound (27000% .. 39,000,000%). A market-neutral funding carry CANNOT be leveraged 21x.
  This book sizes by a REALISTIC RISK BUDGET with a HARD LEVERAGE CAP:

    combined_daily_return = w_core * core + w_sat * (L_sat * satellite)

  where (w_core, w_sat) is a CAPITAL split (sums to 1) and L_sat is the satellite's GROSS leverage,
  CAPPED at SAT_MAX_LEVERAGE (default 3.0 -- the realistic max for a market-neutral perp carry on
  a small, decay-risk-flagged sleeve). We do NOT lever the satellite to vol-match the core; we cap
  it and accept that, at a sane risk budget, it adds a modest, uncorrelated carry sleeve.

  We report a CAPITAL-SPLIT family {core-only, 85/15, 70/30} and a RISK-BUDGET family (the satellite
  contributes a TARGET fraction of total book risk, with the implied leverage CLAMPED to the cap and
  the ACTUAL realized risk-share reported). The recommended deployable book is chosen on the
  diversification benefit (Sharpe up, maxDD down) at a leverage the sleeve can actually hold.

CAUSAL / honest: both legs are causal net streams (the daily_engine is lag-1 MtM-costed; the
satellite is the deployable frictions config). We ALIGN on common dates and combine the net streams.
No look-ahead is introduced by the blend (a static leverage cap + capital split is not fit on the
eval span; the risk-budget leverage uses the FULL-overlap vol, a deployment constant, NOT a forward).

MODES (turnkey):
  python -m strat.core_satellite_book                 # full backtest: core-alone vs the blend family
  python -m strat.core_satellite_book --today         # today's combined allocation (core book + sat legs)
  python -m strat.core_satellite_book --sat-leverage 2.0
  python -m strat.core_satellite_book --selftest      # two-sided synthetic soundness
  python -m strat.core_satellite_book --maker         # maker cost on the core leg

No emoji (Windows cp1252). Does NOT git commit (overseer commits).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "pipeline") not in sys.path:
    sys.path.insert(0, str(ROOT / "pipeline"))

OUT = ROOT.parent / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)
ANN = 365.0

# ----------------------------- book config (declarative) -----------------------------
# the realistic GROSS-leverage ceiling for the market-neutral funding-carry satellite.
# 3.0 is the deployment max for a SMALL, decay-risk-flagged neutral perp sleeve: perp shorts
# + borrow + tiered taker make >3x both operationally fragile (margin/liq risk on the short legs)
# AND imprudent on an edge whose magnitude is UNCONFIRMED forward. We cap and accept a modest sleeve.
SAT_MAX_LEVERAGE = 3.0
# the hard cap on the satellite's CAPITAL fraction in any risk-budget split. An all-satellite book is
# absurd for a small decay-risk-flagged neutral sleeve; we never let the risk-budget solver push the
# satellite past 40% of capital even when its tiny vol "wants" more risk-share.
SAT_FRAC_CAP = 0.40
# the capital-split family we report (satellite capital fraction). 0 = core-only baseline.
CAPITAL_SPLITS = [0.0, 0.15, 0.30]
# the risk-budget family: the TARGET fraction of total book VOL the satellite should contribute.
# (the implied leverage is CLAMPED to SAT_MAX_LEVERAGE; the ACTUAL realized risk-share is reported.)
RISK_BUDGET_TARGETS = [0.15, 0.30]
# the RECOMMENDED deployable allocation (capital split, satellite-leverage). 70/30 capital with the
# satellite at its 3x cap is the default -- justified empirically below (max diversification benefit
# the sleeve can actually hold without vol-matching madness).
RECO_CORE_FRAC = 0.70
RECO_SAT_LEVERAGE = SAT_MAX_LEVERAGE


# ===========================================================================
# 1. the two net streams (reuse the validated builders verbatim)
# ===========================================================================
def core_net_stream(cost_rt=None):
    """CORE = daily_engine ENGINE net (long-only u10 vol-target + regime defensive overlay)."""
    from strat.daily_engine import load_close_panel, build_book
    from strat.portfolio_replay import TAKER_RT
    crt = TAKER_RT if cost_rt is None else cost_rt
    panel = load_close_panel()
    bk = build_book(panel, core="voltgt", use_overlay=True, cost_rt=crt)
    s = bk["net"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="last")]
    return s, panel


def satellite_net_stream(universe="u50", k=5):
    """SATELLITE = funding-dispersion DEPLOYABLE net (1x-neutral, frictions config). Reuses the
    validated builder in funding_satellite_assessment verbatim -- this is the RAW 1x-neutral book
    (we apply the leverage cap ourselves in the blend; the builder is NOT pre-levered)."""
    from strat.funding_satellite_assessment import satellite_net_stream as _sat
    return _sat(universe=universe, k=k)


# ===========================================================================
# 2. metrics on a daily net stream
# ===========================================================================
def stream_stats(daily, ann=ANN):
    """compound / CAGR / ann-vol / Sharpe / maxDD / daily-pos on a daily net array or Series."""
    d = np.asarray(daily, float)
    d = d[np.isfinite(d)]
    if len(d) < 5:
        return {"n_days": int(len(d)), "error": "too short"}
    eq = np.cumprod(1 + d)
    peak = np.maximum.accumulate(eq)
    maxdd = float(((eq - peak) / peak).min() * 100)
    nyr = len(d) / ann
    cagr = float((eq[-1] ** (1 / nyr) - 1) * 100) if eq[-1] > 0 else -100.0
    return {
        "n_days": int(len(d)),
        "compound_pct": round(float((eq[-1] - 1) * 100), 2),
        "cagr_pct": round(cagr, 2),
        "ann_vol_pct": round(float(np.std(d) * np.sqrt(ann) * 100), 2),
        "sharpe": round(float(np.mean(d) / (np.std(d) + 1e-12) * np.sqrt(ann)), 2),
        "maxdd_pct": round(maxdd, 2),
        "daily_pos_rate_pct": round(float((d > 0).mean() * 100), 1),
        "daily_mean_bps": round(float(np.mean(d) * 1e4), 2),
    }


# ===========================================================================
# 3. blend the two streams at a SANE risk budget (leverage-capped, NOT vol-matched)
# ===========================================================================
def blend_capital_split(core, sat, core_frac, sat_leverage):
    """combined = core_frac * core + (1-core_frac) * (sat_leverage * sat).
    A CAPITAL split (core_frac + sat_frac = 1) with the satellite at a CAPPED gross leverage.
    Returns the combined daily Series on the aligned (common) index."""
    sat_frac = 1.0 - core_frac
    return core_frac * core + sat_frac * (sat_leverage * sat)


def _risk_budget_split(core, sat, target_risk_share, sat_leverage, cap):
    """CAPITAL-CONSTRAINED risk-budget sizing (weights sum to 1 -- NO >100% notional, the discipline).
    Hold the satellite at a CAPPED gross leverage L=min(sat_leverage, cap); then solve for the CAPITAL
    split (core_frac + sat_frac = 1) such that the satellite's vol contribution = target_risk_share of
    total book vol. Using the uncorrelated-legs sd proxy on the post-capital legs:
        sat_leg_sd = sat_frac * L * sd_sat ,  core_leg_sd = core_frac * sd_core
        share = sat_leg_sd / (core_leg_sd + sat_leg_sd)
    Solving for sat_frac at target share s (with f = sat_frac):
        s = f*L*sd_s / ((1-f)*sd_c + f*L*sd_s)  =>  f = s*sd_c / (L*sd_s*(1-s) + s*sd_c)
    Because the satellite vol is TINY, hitting a large s would demand sat_frac -> ~1 (an all-satellite
    book, absurd for a decay-risk-flagged sleeve), so sat_frac is itself CAPPED at SAT_FRAC_CAP. Returns
    (core_frac, sat_frac, L, capped_by_satfrac)."""
    L = min(float(sat_leverage), float(cap))
    sd_c = float(np.std(core))
    sd_s = float(np.std(sat))
    if sd_s <= 0 or L <= 0 or target_risk_share <= 0:
        return 1.0, 0.0, L, False
    s = min(float(target_risk_share), 0.999)
    f = (s * sd_c) / (L * sd_s * (1.0 - s) + s * sd_c + 1e-18)
    capped = False
    if f > SAT_FRAC_CAP:
        f = SAT_FRAC_CAP
        capped = True
    return round(1.0 - f, 4), round(f, 4), L, capped


def _realized_risk_share(core_leg, sat_leg):
    """The ACTUAL fraction of combined daily vol attributable to the satellite leg (post-leverage),
    using the uncorrelated-legs sd proxy: sd_sat / (sd_core + sd_sat). Honest realized number."""
    sc = float(np.std(core_leg))
    ss = float(np.std(sat_leg))
    return round(float(ss / (sc + ss + 1e-12)), 3) if (sc + ss) > 0 else 0.0


# ===========================================================================
# 4. the full backtest: core-alone vs the blend family
# ===========================================================================
def backtest(core, sat, start=None, end=None, sat_leverage=RECO_SAT_LEVERAGE):
    """Align core+sat on common dates; report core-alone + the capital-split family + the risk-budget
    family. Every blend caps the satellite at SAT_MAX_LEVERAGE. Returns a structured dict."""
    df = pd.concat({"core": core, "sat": sat}, axis=1)
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index < pd.Timestamp(end)]
    overlap = df.dropna()
    out = {
        "core_span": [str(core.index.min().date()), str(core.index.max().date())],
        "sat_span": [str(sat.index.min().date()), str(sat.index.max().date())],
        "overlap_days": int(len(overlap)),
        "overlap_span": ([str(overlap.index.min().date()), str(overlap.index.max().date())]
                         if len(overlap) else None),
        "sat_max_leverage": SAT_MAX_LEVERAGE,
    }
    if len(overlap) < 30:
        out["error"] = "insufficient overlap"
        return out
    c = overlap["core"]
    s = overlap["sat"]
    pear = float(np.corrcoef(c.to_numpy(), s.to_numpy())[0, 1])
    rc = c.rank().to_numpy(); rs = s.rank().to_numpy()
    spear = float(np.corrcoef(rc, rs)[0, 1])
    out["correlation"] = {"pearson": round(pear, 4), "spearman": round(spear, 4)}
    out["core_stats_overlap"] = stream_stats(c)
    out["sat_stats_overlap_1x"] = stream_stats(s)

    blends = {}

    # ---- CORE-ALONE baseline (no satellite) ----
    blends["CORE_ALONE"] = {**stream_stats(c), "core_frac": 1.0, "sat_frac": 0.0,
                            "sat_leverage": 0.0, "sat_realized_risk_share": 0.0,
                            "sizing": "core-only baseline"}

    # ---- CAPITAL-SPLIT family (sat capped at the requested leverage) ----
    for sat_frac in CAPITAL_SPLITS:
        if sat_frac == 0.0:
            continue
        core_frac = 1.0 - sat_frac
        sat_leg = sat_frac * (sat_leverage * s)
        core_leg = core_frac * c
        combined = core_leg + sat_leg
        st = stream_stats(combined)
        st.update({"core_frac": round(core_frac, 2), "sat_frac": round(sat_frac, 2),
                   "sat_leverage": round(float(sat_leverage), 2),
                   "sat_realized_risk_share": _realized_risk_share(core_leg, sat_leg),
                   "sizing": f"capital {int(core_frac*100)}/{int(sat_frac*100)} @ {sat_leverage:g}x sat"})
        blends[f"CAP_{int(core_frac*100)}_{int(sat_frac*100)}"] = st

    # ---- RISK-BUDGET family (CAPITAL-CONSTRAINED: weights sum to 1, sat capped) ----
    # target a sat risk-SHARE, solve for the CAPITAL split at the capped leverage (no >100% notional).
    for tgt in RISK_BUDGET_TARGETS:
        core_frac, sat_frac, L, satfrac_capped = _risk_budget_split(
            c.to_numpy(), s.to_numpy(), tgt, sat_leverage, SAT_MAX_LEVERAGE)
        core_leg = core_frac * c
        sat_leg = sat_frac * (L * s)
        combined = core_leg + sat_leg
        st = stream_stats(combined)
        rs = _realized_risk_share(core_leg, sat_leg)
        st.update({"core_frac": core_frac, "sat_frac": sat_frac,
                   "sat_leverage": round(float(L), 2),
                   "satfrac_capped": bool(satfrac_capped),
                   "target_risk_share": tgt,
                   "sat_realized_risk_share": rs,
                   "sizing": f"risk-budget {int(tgt*100)}% sat-vol -> capital "
                             f"{int(core_frac*100)}/{int(sat_frac*100)} @ {L:g}x sat"
                             f"{' (satfrac CAPPED)' if satfrac_capped else ''}"})
        blends[f"RISK_{int(tgt*100)}pct"] = st

    out["blends"] = blends

    # ---- the RECOMMENDED deployable book ----
    reco_core_leg = RECO_CORE_FRAC * c
    reco_sat_leg = (1.0 - RECO_CORE_FRAC) * (RECO_SAT_LEVERAGE * s)
    reco_combined = reco_core_leg + reco_sat_leg
    reco_st = stream_stats(reco_combined)
    core_st = blends["CORE_ALONE"]
    out["recommended"] = {
        "allocation": f"capital {int(RECO_CORE_FRAC*100)}/{int((1-RECO_CORE_FRAC)*100)} "
                      f"core/satellite, satellite @ {RECO_SAT_LEVERAGE:g}x (CAPPED)",
        "core_frac": RECO_CORE_FRAC, "sat_frac": round(1 - RECO_CORE_FRAC, 2),
        "sat_leverage": RECO_SAT_LEVERAGE,
        "sat_realized_risk_share": _realized_risk_share(reco_core_leg, reco_sat_leg),
        "stats": reco_st,
        "diversification_vs_core_alone": {
            "sharpe_core_alone": core_st["sharpe"],
            "sharpe_blend": reco_st["sharpe"],
            "sharpe_delta": round(reco_st["sharpe"] - core_st["sharpe"], 2),
            "maxdd_core_alone": core_st["maxdd_pct"],
            "maxdd_blend": reco_st["maxdd_pct"],
            "maxdd_delta_pp": round(reco_st["maxdd_pct"] - core_st["maxdd_pct"], 2),
            "compound_core_alone": core_st["compound_pct"],
            "compound_blend": reco_st["compound_pct"],
        },
    }
    return out


# ===========================================================================
# 5. --today: the combined allocation to hold today
# ===========================================================================
def today_book(core_frac=RECO_CORE_FRAC, sat_leverage=RECO_SAT_LEVERAGE, universe="u50", k=5):
    """The combined allocation for the latest day: the core book (per-asset long weights, scaled by
    its capital fraction) + the satellite's neutral legs (long/short perp names, scaled by its capital
    fraction * leverage). Reuses daily_engine.book_for_date for the core legs and the funding-dispersion
    tooling for the satellite's current long/short legs."""
    from strat.daily_engine import load_close_panel, book_for_date
    sat_frac = 1.0 - core_frac

    panel = load_close_panel()
    core_bk = book_for_date(panel, core="voltgt", use_overlay=True)
    core_legs = {a: round(w * core_frac, 4) for a, w in core_bk.get("weights", {}).items()}

    # satellite legs: the current funding-dispersion long/short selection (gross 1, net 0), scaled.
    sat_long, sat_short, sat_asof = _satellite_legs_today(universe=universe, k=k)
    sat_scale = sat_frac * sat_leverage   # the satellite's gross capital * leverage
    sat_legs = {
        "long_low_funding": {a: round(w * sat_scale, 4) for a, w in sat_long.items()},
        "short_high_funding": {a: round(-abs(w) * sat_scale, 4) for a, w in sat_short.items()},
    }
    return {
        "date": core_bk.get("date"),
        "allocation": f"capital {int(core_frac*100)}/{int(sat_frac*100)} core/satellite, "
                      f"satellite @ {sat_leverage:g}x (CAPPED at {SAT_MAX_LEVERAGE:g}x)",
        "core_regime": core_bk.get("regime"),
        "core_gross_exposure_scaled": round(core_bk.get("gross_exposure", 0.0) * core_frac, 3),
        "core_long_legs": dict(sorted(core_legs.items(), key=lambda kv: -kv[1])),
        "satellite_asof": sat_asof,
        "satellite_legs": sat_legs,
        "satellite_gross_scaled": round(sat_scale, 3),
        "data_quality_flags": core_bk.get("data_quality_flags", []),
        "notes": ["Satellite legs are PERP positions (net-zero, gross 1 before scaling). "
                  "Market-neutral execution: enter both sides; net dollar exposure ~0.",
                  "Satellite is a MODEST carry sleeve (decay-risk UNCONFIRMED) -- size conservatively."],
    }


def _satellite_legs_today(universe="u50", k=5, lookback=7, lag=1):
    """The funding-dispersion long/short selection on the LATEST available signal day. Returns
    (long_weights, short_weights, asof_date). Long = lowest-funding k names, short = highest-funding
    k names, equal-weight within each side (gross 1 per side before scaling)."""
    import mining.funding_dispersion_frictions as FF
    panel = FF.load_daily_panel(universe)
    dates, assets, ret, fund_daily, vol, oi = FF.to_wide(panel)
    fund_8h_sum, fund_settles = FF.load_funding_8h(assets, dates)
    sig_8h = FF.funding_signal(fund_8h_sum, lookback=lookback, lag=lag)
    # the latest row with a finite signal across enough names
    t = len(dates) - 1
    while t >= 0 and np.isfinite(sig_8h[t, :]).sum() < 2 * k:
        t -= 1
    if t < 0:
        return {}, {}, None
    row = sig_8h[t, :]
    finite = np.where(np.isfinite(row))[0]
    order = finite[np.argsort(row[finite])]
    lo_idx = order[:k]               # lowest funding -> LONG
    hi_idx = order[-k:]              # highest funding -> SHORT
    wl = 1.0 / max(1, len(lo_idx))
    ws = 1.0 / max(1, len(hi_idx))
    longs = {str(assets[i]): round(wl, 4) for i in lo_idx}
    shorts = {str(assets[i]): round(ws, 4) for i in hi_idx}
    asof = str(np.datetime64(dates[t], "D"))
    return longs, shorts, asof


# ===========================================================================
# 6. selftest -- two-sided soundness (synthetic, no market)
# ===========================================================================
def selftest():
    """POSITIVE: a positive-Sharpe, ~zero-correlated synthetic satellite IMPROVES the blend's Sharpe
    over core-alone (a real diversifier lifts risk-adjusted return). NEGATIVE: a ZERO-return satellite
    leaves the blend stats UNCHANGED vs core-alone (no phantom lift). Plus: the leverage cap binds
    (a vol-match risk-budget request is clamped to SAT_MAX_LEVERAGE)."""
    print("## CORE+SATELLITE-BOOK SELFTEST (two-sided)")
    ok = True
    rng = np.random.default_rng(0)
    idx = pd.date_range("2021-01-01", periods=1500, freq="D")

    # core: a noisy beta-like stream (positive drift, high vol)
    core = pd.Series(rng.normal(0.0015, 0.025, len(idx)), index=idx)

    # ---- POSITIVE: a positive-Sharpe, low-vol, ~uncorrelated satellite -> blend Sharpe UP ----
    sat_pos = pd.Series(rng.normal(0.0004, 0.0015, len(idx)), index=idx)  # high-Sharpe low-vol carry
    corr = float(np.corrcoef(core.to_numpy(), sat_pos.to_numpy())[0, 1])
    bt = backtest(core, sat_pos, sat_leverage=SAT_MAX_LEVERAGE)
    sh_core = bt["blends"]["CORE_ALONE"]["sharpe"]
    sh_reco = bt["recommended"]["stats"]["sharpe"]
    print(f"  POSITIVE: corr={corr:+.3f} | core-alone Sh={sh_core} -> blend Sh={sh_reco} "
          f"(expect blend > core-alone)")
    ok &= (sh_reco > sh_core)

    # ---- NEGATIVE: a ZERO-return satellite -> blend Sharpe == core-alone Sharpe. Under a capital
    #      split combined = core_frac*core + sat_frac*0 = core_frac*core; scaling a stream by a
    #      constant leaves Sharpe invariant (mean & std both scale). So zero sat -> no phantom lift. --
    sat_zero = pd.Series(np.zeros(len(idx)), index=idx)
    btz = backtest(core, sat_zero, sat_leverage=SAT_MAX_LEVERAGE)
    sh_core_z = btz["blends"]["CORE_ALONE"]["sharpe"]
    sh_cap_z = btz["blends"]["CAP_70_30"]["sharpe"]       # capital split, zero sat -> Sharpe unchanged
    sh_risk_z = btz["blends"]["RISK_15pct"]["sharpe"]
    print(f"  NEGATIVE: zero-return satellite | core-alone Sh={sh_core_z} vs 70/30 Sh={sh_cap_z} "
          f"vs risk-15 Sh={sh_risk_z} (expect all ~equal -- no phantom Sharpe lift)")
    ok &= (abs(sh_cap_z - sh_core_z) < 1e-6 and abs(sh_risk_z - sh_core_z) < 1e-6)

    # ---- SATFRAC CAP: a tiny-vol satellite "wants" a huge capital fraction to hit 30% risk-share ->
    #      capital fraction CLAMPED to SAT_FRAC_CAP (never an all-satellite book) ----
    sat_tiny = pd.Series(rng.normal(0.0002, 0.0008, len(idx)), index=idx)
    cf, sf, L, capped = _risk_budget_split(core.to_numpy(), sat_tiny.to_numpy(), 0.30,
                                           SAT_MAX_LEVERAGE, SAT_MAX_LEVERAGE)
    print(f"  CAP: 30% risk-budget on a tiny-vol satellite -> capital {cf:.2f}/{sf:.2f} L={L:.2f}x "
          f"satfrac_capped={capped} (expect sat_frac==cap {SAT_FRAC_CAP} and capped=True)")
    ok &= (capped and abs(sf - SAT_FRAC_CAP) < 1e-9)

    # ---- realized risk share is in [0,1] and monotonic in leverage ----
    rs_lo = _realized_risk_share(core, 1.0 * sat_pos)
    rs_hi = _realized_risk_share(core, 3.0 * sat_pos)
    print(f"  RISK-SHARE: sat realized risk-share at 1x={rs_lo} < 3x={rs_hi} (expect monotone up, in [0,1])")
    ok &= (0 <= rs_lo <= 1 and 0 <= rs_hi <= 1 and rs_hi > rs_lo)

    print(f"\n  SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ===========================================================================
# 7. CLI
# ===========================================================================
def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def _print_blend_table(blends):
    print(f"   {'book':16} {'compound%':>13} {'CAGR%':>8} {'annVol%':>8} {'Sharpe':>7} "
          f"{'maxDD%':>8} {'dPos%':>7} {'satLev':>7} {'satRisk%':>9}")
    order = (["CORE_ALONE"] + [k for k in blends if k.startswith("CAP_")]
             + [k for k in blends if k.startswith("RISK_")])
    for k in order:
        m = blends.get(k)
        if not m or "error" in m:
            continue
        rs = m.get("sat_realized_risk_share", 0.0)
        print(f"   {k:16} {m.get('compound_pct'):>13} {m.get('cagr_pct'):>8} {m.get('ann_vol_pct'):>8} "
              f"{m.get('sharpe'):>7} {m.get('maxdd_pct'):>8} {m.get('daily_pos_rate_pct'):>7} "
              f"{m.get('sat_leverage'):>7} {round(rs*100,1):>9}")


def _print_today(book):
    print(f"   DATE {book.get('date')} | {book.get('allocation')}")
    print(f"   CORE (regime={book.get('core_regime')}, gross-scaled={book.get('core_gross_exposure_scaled')}):")
    for a, w in book.get("core_long_legs", {}).items():
        print(f"     LONG  {a:10} {w:>7.4f}")
    sl = book.get("satellite_legs", {})
    print(f"   SATELLITE (asof {book.get('satellite_asof')}, gross-scaled={book.get('satellite_gross_scaled')}):")
    for a, w in sl.get("long_low_funding", {}).items():
        print(f"     LONG  {a:10} {w:>7.4f}  (low funding)")
    for a, w in sl.get("short_high_funding", {}).items():
        print(f"     SHORT {a:10} {w:>7.4f}  (high funding)")
    for f in book.get("data_quality_flags", []):
        print(f"   [data-quality] {f}")
    for n in book.get("notes", []):
        print(f"   [note] {n}")


def run_backtest_mode(core, sat, panel, lo, hi, sat_leverage, argv):
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"## CORE+SATELLITE BOOK -- BACKTEST {lo} .. {hi} -- satellite capped at {SAT_MAX_LEVERAGE:g}x")
    bt = backtest(core, sat, start=lo, end=hi, sat_leverage=sat_leverage)
    print(f"   overlap: {bt['overlap_days']} days {bt.get('overlap_span')}")
    if "error" in bt:
        print(f"   {bt['error']}")
        return 2
    cor = bt["correlation"]
    print(f"   correlation core/satellite: pearson={cor['pearson']:+.4f} spearman={cor['spearman']:+.4f} "
          f"(~zero = a genuine diversifier)\n")
    print(f"   satellite 1x-neutral (overlap): comp={bt['sat_stats_overlap_1x']['compound_pct']}% "
          f"vol={bt['sat_stats_overlap_1x']['ann_vol_pct']}% Sh={bt['sat_stats_overlap_1x']['sharpe']} "
          f"maxDD={bt['sat_stats_overlap_1x']['maxdd_pct']}%\n")
    _print_blend_table(bt["blends"])
    r = bt["recommended"]
    dv = r["diversification_vs_core_alone"]
    print(f"\n   --- RECOMMENDED: {r['allocation']} ---")
    print(f"   sat realized risk-share: {round(r['sat_realized_risk_share']*100,1)}% of book vol")
    print(f"   diversification vs CORE-ALONE: Sharpe {dv['sharpe_core_alone']} -> {dv['sharpe_blend']} "
          f"(delta {dv['sharpe_delta']:+}) | maxDD {dv['maxdd_core_alone']}% -> {dv['maxdd_blend']}% "
          f"(delta {dv['maxdd_delta_pp']:+}pp)")
    print(f"   compound {dv['compound_core_alone']}% -> {dv['compound_blend']}% (a modest carry lift, "
          f"NOT a moonshot)")
    # today's book
    print(f"\n   --- TODAY'S COMBINED ALLOCATION ---")
    tb = today_book(core_frac=RECO_CORE_FRAC, sat_leverage=sat_leverage)
    _print_today(tb)
    # persist
    p = OUT / f"core_satellite_book_{stamp}.json"
    out = {
        "repro": {"command": "python -m strat.core_satellite_book " + " ".join(argv),
                  "git_sha": _git_sha(), "window": [lo, hi],
                  "sat_max_leverage": SAT_MAX_LEVERAGE, "sat_leverage": sat_leverage,
                  "reco_core_frac": RECO_CORE_FRAC, "reco_sat_leverage": RECO_SAT_LEVERAGE},
        "backtest": bt,
        "today_book": tb,
    }
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n   [persisted] {p}")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="python -m strat.core_satellite_book")
    ap.add_argument("--backtest", default=None, help="START:END (YYYY-MM-DD:YYYY-MM-DD)")
    ap.add_argument("--today", action="store_true", help="print today's combined allocation")
    ap.add_argument("--sat-leverage", type=float, default=RECO_SAT_LEVERAGE,
                    help=f"satellite gross leverage (capped at SAT_MAX_LEVERAGE={SAT_MAX_LEVERAGE})")
    ap.add_argument("--universe", default="u50", help="satellite universe (funding dispersion)")
    ap.add_argument("--k", type=int, default=5, help="satellite long/short basket size")
    ap.add_argument("--maker", action="store_true", help="maker cost on the core leg")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()

    # cap the leverage (the discipline -- never vol-match)
    sat_leverage = min(float(a.sat_leverage), SAT_MAX_LEVERAGE)
    if a.sat_leverage > SAT_MAX_LEVERAGE:
        print(f"   [cap] requested sat leverage {a.sat_leverage} > cap {SAT_MAX_LEVERAGE} -> using {sat_leverage}")

    from strat.portfolio_replay import TAKER_RT, MAKER_RT
    cost_rt = MAKER_RT if a.maker else TAKER_RT

    if a.today:
        tb = today_book(core_frac=RECO_CORE_FRAC, sat_leverage=sat_leverage,
                        universe=a.universe, k=a.k)
        print(f"## CORE+SATELLITE BOOK -- TODAY'S ALLOCATION")
        _print_today(tb)
        return 0

    print("   [1/2] building core stream (daily_engine ENGINE)...")
    sys.stdout.flush()
    core, panel = core_net_stream(cost_rt=cost_rt)
    print(f"         core: {len(core)} days {core.index.min().date()}..{core.index.max().date()}")
    print("   [2/2] building satellite stream (funding-dispersion deployable)...")
    sys.stdout.flush()
    sat = satellite_net_stream(universe=a.universe, k=a.k)
    print(f"         satellite: {len(sat)} days {sat.index.min().date()}..{sat.index.max().date()}\n")

    if a.backtest:
        try:
            lo, hi = a.backtest.split(":")
        except ValueError:
            print("--backtest expects START:END (e.g. 2020-01-01:2026-01-01)")
            return 2
        return run_backtest_mode(core, sat, panel, lo, hi, sat_leverage, argv)

    return run_backtest_mode(core, sat, panel, "2020-01-01", "2026-06-01", sat_leverage, argv)


if __name__ == "__main__":
    raise SystemExit(main())
