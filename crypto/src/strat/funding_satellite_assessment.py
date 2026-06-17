"""funding_satellite_assessment.py -- ASSESS the cross-sectional funding-DISPERSION
carry (the other instance's first held-out positive, src/mining/funding_dispersion_*)
as a SATELLITE to pair with daily_engine's long-only vol-target BETA core.

This is an ASSESSMENT tool (not a production combiner): it extracts the two daily
NET-return streams, aligns them on the common date range, and computes the EMPIRICAL
diversification evidence the integration spec needs:
  (1) CORE   = daily_engine ENGINE net (long-only u10 vol-target + regime overlay).
  (2) SATELLITE = funding-dispersion DEPLOYABLE net (short-liquid + 8h + tiered taker
                  + borrow + hysteresis), the config funding_dispersion_frictions ships.
  (3) Pearson + Spearman correlation of the two daily streams (overlap).
  (4) A risk-budget sizing sketch: a core + risk-scaled-satellite blend, with the
      combined vol/Sharpe/maxDD vs the core alone (does the satellite REDUCE blended
      risk / lift risk-adjusted return, the test of a real diversifier).

HONEST: this re-uses the deployable funding config as-is. It does NOT re-validate the
funding edge (the other instance did + flagged DECAY-RISK UNCONFIRMED). It quantifies
the OVERLAP + the sizing math so the integration spec is grounded, not hand-waved.

Read-only on data. Standalone. Does NOT commit.

Run:
  python -m strat.funding_satellite_assessment
  python -m strat.funding_satellite_assessment --start 2021-01-01 --end 2026-01-01
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


# ===========================================================================
# 1. CORE stream: daily_engine ENGINE net (long-only vol-target + regime overlay)
# ===========================================================================
def core_net_stream():
    """The daily_engine ENGINE net-return Series (taker), full u10 history."""
    from strat.daily_engine import load_close_panel, build_book
    from strat.portfolio_replay import TAKER_RT
    panel = load_close_panel()
    bk = build_book(panel, core="voltgt", use_overlay=True, cost_rt=TAKER_RT)
    s = bk["net"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="last")]
    return s


# ===========================================================================
# 2. SATELLITE stream: funding-dispersion DEPLOYABLE net (frictions config)
# ===========================================================================
def satellite_net_stream(universe="u50", k=5, lookback=7, lag=1):
    """The funding-dispersion DEPLOYABLE daily net stream (the 3c config from
    funding_dispersion_frictions: short-liquid + 8h clock + tiered taker + borrow +
    hysteresis). Returns a daily-indexed Series of net returns."""
    import mining.funding_dispersion_frictions as FF  # noqa: E402
    panel = FF.load_daily_panel(universe)
    dates, assets, ret, fund_daily, vol, oi = FF.to_wide(panel)
    fund_8h_sum, fund_settles = FF.load_funding_8h(assets, dates)
    sig_8h = FF.funding_signal(fund_8h_sum, lookback=lookback, lag=lag)
    short_floor = 1.0 - FF.LIQ_TOP_FRACTION
    recs = FF.simulate(dates, ret, fund_8h_sum, fund_settles, vol, oi, sig_8h, k,
                       FF.make_weight_fn(short_floor, None),
                       tiered_cost=True, borrow_asym=True, hysteresis_band=2.0,
                       short_floor=short_floor, long_floor=None)
    idx = pd.to_datetime([np.datetime64(r[0], "D") for r in recs])
    s = pd.Series([r[4] for r in recs], index=idx)
    s.index = s.index.tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="last")]
    return s


# ===========================================================================
# 3. metrics + correlation + sizing sketch
# ===========================================================================
def _stats(daily):
    d = np.asarray(daily, float)
    if len(d) < 5:
        return {}
    eq = np.cumprod(1 + d); peak = np.maximum.accumulate(eq)
    maxdd = float(((eq - peak) / peak).min() * 100)
    comp = float((eq[-1] - 1) * 100)
    vol = float(np.std(d) * np.sqrt(ANN) * 100)
    sharpe = float(np.mean(d) / (np.std(d) + 1e-12) * np.sqrt(ANN))
    return {"n_days": int(len(d)), "compound_pct": round(comp, 2),
            "ann_vol_pct": round(vol, 2), "sharpe": round(sharpe, 2),
            "maxdd_pct": round(maxdd, 2), "daily_mean_bps": round(float(np.mean(d) * 1e4), 2)}


def assess(core, sat, start=None, end=None):
    """Align the two streams on their common dates, compute correlation + a risk-budget
    blend sketch. Returns a structured dict."""
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
    }
    if len(overlap) < 30:
        out["error"] = "insufficient overlap"
        return out
    c = overlap["core"].to_numpy(); s = overlap["sat"].to_numpy()
    pear = float(np.corrcoef(c, s)[0, 1])
    rc = pd.Series(c).rank().to_numpy(); rs = pd.Series(s).rank().to_numpy()
    spear = float(np.corrcoef(rc, rs)[0, 1])
    out["correlation"] = {"pearson": round(pear, 4), "spearman": round(spear, 4)}
    out["core_stats_overlap"] = _stats(c)
    out["sat_stats_overlap"] = _stats(s)

    # ---- risk-budget sizing sketch ----------------------------------------
    # Scale the satellite to a target vol fraction of the core, then blend.
    # We report blends at a few risk weights so the spec is grounded.
    vc = float(np.std(c)); vs = float(np.std(s))
    blends = {}
    for w_sat in (0.0, 0.25, 0.5, 1.0):
        # size the satellite so its risk contribution = w_sat * core daily vol
        scale = (w_sat * vc / (vs + 1e-12)) if vs > 0 else 0.0
        blended = c + scale * s
        st = _stats(blended)
        st["sat_vol_scale"] = round(float(scale), 3)
        st["sat_risk_weight"] = w_sat
        blends[f"core+{w_sat:g}xVolSat"] = st
    out["blend_sketch"] = blends

    # diversification verdict bits
    core_st = _stats(c)
    best_key = max(blends, key=lambda kk: blends[kk]["sharpe"])
    out["diversification"] = {
        "low_correlation": bool(abs(pear) < 0.2),
        "core_sharpe_overlap": core_st["sharpe"],
        "best_blend": best_key,
        "best_blend_sharpe": blends[best_key]["sharpe"],
        "sharpe_lift_vs_core": round(blends[best_key]["sharpe"] - core_st["sharpe"], 2),
        "best_blend_maxdd": blends[best_key]["maxdd_pct"],
        "core_maxdd": core_st["maxdd_pct"],
    }
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="python -m strat.funding_satellite_assessment")
    ap.add_argument("--universe", default="u50")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    a = ap.parse_args(argv)

    print("## FUNDING-DISPERSION SATELLITE ASSESSMENT (vs daily_engine BETA core)")
    print("   core = daily_engine ENGINE (long-only u10 vol-target + regime overlay, taker)")
    print(f"   satellite = funding-dispersion DEPLOYABLE (short-liquid+8h+tiered+borrow+hyst, "
          f"u={a.universe} k={a.k})\n")
    print("   [1/2] building core stream (daily_engine)...")
    sys.stdout.flush()
    core = core_net_stream()
    print(f"         core: {len(core)} days {core.index.min().date()}..{core.index.max().date()}")
    print("   [2/2] building satellite stream (funding_dispersion deployable)...")
    sys.stdout.flush()
    sat = satellite_net_stream(universe=a.universe, k=a.k)
    print(f"         satellite: {len(sat)} days {sat.index.min().date()}..{sat.index.max().date()}\n")

    res = assess(core, sat, a.start, a.end)
    print("=" * 78)
    print("ASSESSMENT")
    print("=" * 78)
    print(f"  overlap: {res['overlap_days']} days  {res.get('overlap_span')}")
    if "error" in res:
        print(f"  {res['error']}")
        return 0
    cor = res["correlation"]
    print(f"  correlation: pearson={cor['pearson']:+.4f}  spearman={cor['spearman']:+.4f}")
    print(f"  core   (overlap): comp={res['core_stats_overlap']['compound_pct']}%  "
          f"vol={res['core_stats_overlap']['ann_vol_pct']}%  Sh={res['core_stats_overlap']['sharpe']}  "
          f"maxDD={res['core_stats_overlap']['maxdd_pct']}%")
    print(f"  sat    (overlap): comp={res['sat_stats_overlap']['compound_pct']}%  "
          f"vol={res['sat_stats_overlap']['ann_vol_pct']}%  Sh={res['sat_stats_overlap']['sharpe']}  "
          f"maxDD={res['sat_stats_overlap']['maxdd_pct']}%")
    print("\n  --- risk-budget blend sketch (satellite sized to a vol fraction of the core) ---")
    print(f"  {'blend':22} {'comp%':>8} {'vol%':>7} {'Sharpe':>7} {'maxDD%':>8} {'satScale':>9}")
    for kk, st in res["blend_sketch"].items():
        print(f"  {kk:22} {st['compound_pct']:>8} {st['ann_vol_pct']:>7} {st['sharpe']:>7} "
              f"{st['maxdd_pct']:>8} {st['sat_vol_scale']:>9}")
    dv = res["diversification"]
    print(f"\n  diversification: low_corr={dv['low_correlation']} | core Sh={dv['core_sharpe_overlap']} "
          f"-> best blend {dv['best_blend']} Sh={dv['best_blend_sharpe']} "
          f"(lift {dv['sharpe_lift_vs_core']:+}) maxDD {dv['core_maxdd']}->{dv['best_blend_maxdd']}")

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT / f"funding_satellite_assessment_{stamp}.json"
    json.dump({"repro": {"command": "python -m strat.funding_satellite_assessment " + " ".join(argv),
                         "git_sha": sha, "universe": a.universe, "k": a.k},
               "assessment": res}, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
