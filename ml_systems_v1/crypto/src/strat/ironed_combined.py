"""src/strat/ironed_combined.py -- THE COMBINATION / PARTICIPATION LAYER (CONSTRUCTION, not refutation).

USER /orc 2026-06-13 (CORE THESIS verbatim): "combine and build these to get max opportunity
participation ... a crossover in one asset might miss, but across 50 it will hit somewhere ... combine
these to get max opportunity participation, and that should solve for ALL 4 aspects if we solve the
weaknesses right."

Two per-TF IRONED MA TREND systems were built+verified on the 2020 deep-dive (within-2020 split):
  - COARSE (ironed_coarse.py): 1d=DEPLOY (family-only +49.8% OOS, maxDD -14.1, cov 87%, p05 +0.58);
    4h=DEPLOY (full ironed +40.8%, maxDD -16.0, p05 +0.16). 2h=NOT-YET (cost-bound, excluded).
  - FINE (deep2020_ironed_fine.py): 1h/30m/15m de-risked sleeves (VIDYA + whipsaw filter); the best
    by net is 15m nogate_voltgt (+67.0% maker / maxDD -8.6); best DD is 15m half-gate (+62.1 / -6.9).

THIS LAYER combines the per-TF sleeves into ONE BOOK and MEASURES whether combination delivers the
participation + diversification the thesis expects. We REUSE the verified builders (import + call) and
RECONSTRUCT each recommended sleeve's net stream deterministically (NOT from JSON write timing).

PRE-REGISTERED (a DEPLOYMENT CONSTANT, NOT fit on OOS):
  - SLEEVE SET (the participating trend CORE): {1d family-only, 4h full-ironed, 15m nogate_voltgt}.
    1d+4h are the two DEPLOY sleeves; 15m is the best fine sleeve (adds a faster, lower-DD participant).
    A 1d+4h-only book is also reported (the two strict DEPLOY sleeves alone).
  - CAPITAL WEIGHTS: TWO pre-registered rules, BOTH reported -- (a) EQUAL-WEIGHT (1/N), (b) INVERSE-VOL
    (w_i proportional to 1/sd_i, sd on the FULL OOS overlap = a deployment-constant level, NOT a forward
    timing signal -- same convention as core_satellite_book's risk-budget vol). Weights sum to 1 (a
    CAPITAL split: the book is a weighted average of the sleeve net streams, gross = 1, no extra leverage).
  - ORTHOGONAL CARRY SATELLITE: the funding-DISPERSION dollar-neutral carry (the project's first verified
    held-out positive, ~zero-correlated to beta). Sized per core_satellite_book: gross-leverage cap 3x,
    capital fraction cap 40%. We report the SOTA combined book = CORE(per-TF trend) + SATELLITE(carry).

MEASURES (the thesis tests):
  1 PARTICIPATION / COVERAGE UNION: fraction of OOS days >=1 sleeve is profitably engaged, combined vs
    each sleeve alone (the "across TFs it hits somewhere" claim, quantified).
  2 CROSS-TF DIVERSIFICATION: cross-TF net-stream correlation matrix; combined n_eff, Sharpe, maxDD vs
    the BEST single sleeve. Does combining RAISE Sharpe + LOWER maxDD, or are sleeves correlated beta?
  3 BOOK-LEVEL NET: combined OOS net / ann(INDICATIVE) / maxDD / Sharpe / p05(block-bootstrap) vs
    VOLTGT_BH and vs the best single-TF sleeve. Is the WHOLE > the parts?
  4 THE ORTHOGONAL DIVERSIFIER: CORE+SATELLITE(carry) vs CORE-alone on Sharpe/maxDD (the genuine
    multiplier vs the correlated-beta sleeves).
  5 "PROFIT DAILY" HONESTY: fraction of OOS 1d/3d windows the combined book is positive (charter
    soft-bench). Honest: even the all-mover basket is +53% of windows (one-factor market) -- report REAL.

DISCIPLINE: RWYB (actually run it); claim-tag every number (VERIFIED iff run); flag the 3mo-bull-OOS
limitation (the de-risk/diversification value is partly a whole-cycle product the ~0%-bear 2020 OOS
cannot fully show); no look-ahead (weights pre-registered, OOS confirmed once). Cost maker, causal/lag-1.

RWYB: python -m strat.ironed_combined
JSON: runs/periods/TRAIN/2020/DEEP_DIVE/ironed_combined.json ; MD: IRONED_COMBINED.md (hand-written).
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

import strat.portfolio_replay as PR                                       # noqa: E402
from strat.portfolio_replay import MAKER_RT                              # noqa: E402
from strat.replay_distinct_grid import distinct_specs                    # noqa: E402
from strat.ma_type_upgrade import _nums                                  # noqa: E402
from strat.ma_2020_breakdown import SPLIT                                # noqa: E402
from strat.battery import block_bootstrap_p05_p95                        # noqa: E402
import strat.ironed_coarse as IC                                         # noqa: E402
import strat.deep2020_ironed_fine as IF                                  # noqa: E402

OUT = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
OOS = SPLIT["OOS"]                                                       # ("2020-10-01", "2021-01-01")
ANN_DAILY = 365.0
# satellite sizing discipline (per core_satellite_book; the only sane budget for a small neutral carry)
SAT_MAX_LEVERAGE = 3.0
SAT_FRAC_CAP = 0.40
RECO_CORE_FRAC = 0.70                                                    # 70/30 core/satellite (the default)


# ===========================================================================
# 1. build the slow EMA family (shared by both builders) ONCE
# ===========================================================================
def _slow_family():
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=60))
    PR.STRATS.update(ma_cfg)
    slow = [n for n in ma_cfg if 60 <= max(_nums(n)) < 150]
    slow2 = [n for n in slow if len(_nums(n)) == 2]
    fam_set = slow2 if len(slow2) >= 5 else slow
    return slow, fam_set


# ===========================================================================
# 2. reconstruct each RECOMMENDED sleeve's DAILY net stream (deterministic, from the spec kwargs)
# ===========================================================================
def _to_daily(book):
    """resample a bar-level book net Series to a daily compound net Series over the OOS window."""
    oos = book[(book.index >= pd.Timestamp(OOS[0])) & (book.index < pd.Timestamp(OOS[1]))].dropna()
    daily = oos.resample("1D").apply(lambda v: float(np.prod(1 + v) - 1)).dropna()
    daily.index = daily.index.normalize()
    return daily


def coarse_sleeve(cad, kw):
    """reconstruct a coarse recommended sleeve (1d family-only or 4h full-ironed). Returns daily net."""
    slow, fam_set = _slow_family()
    closes = IC._closes(cad)
    panel_df = IC._book_close_panel(closes, cad)
    regime, _th = IC.market_regime(panel_df, cad, SPLIT["TRAIN"][0], SPLIT["TRAIN"][1])
    book, _expo = IC.build_stack(closes, panel_df, regime, cad, slow=fam_set, **kw)
    return _to_daily(book)


def fine_sleeve(cad, *, best_type="VIDYA", whip=(8, 96, 48), exit_="none", gate=0, half=False):
    """reconstruct a fine recommended sleeve (default 15m nogate_voltgt). Returns daily net (maker)."""
    slow, _fam = _slow_family()
    IF._CUR_CAD = cad
    IF.VOLWIN_CUR = IF.VOLWIN[cad]
    panels = IF._build_caches(cad, slow)
    rvs = []
    for sym, (c, h, l, ms, win, caches) in panels.items():
        rv = pd.Series(IF._ret_of(c)).rolling(IF.VOLWIN_CUR).std().to_numpy()
        rvs.append(np.nanmedian(rv))
    voltgt = float(np.nanmedian(rvs))
    ap, _exp = IF._family_with_exit(panels, slow, best_type, *whip, exit_, gate, voltgt, half_gate=half)
    bk, _, _ = IF._book_from_positions(ap, cad, MAKER_RT)
    return _to_daily(bk)


def benchmark_voltgt_bh_1d():
    """the 1d VOLTGT_BH benchmark (the established 'best' in the bull), as a daily net stream."""
    slow, _fam = _slow_family()
    closes = IC._closes("1d")
    bm = IC.benchmarks(closes, "1d")
    return _to_daily(bm["VOLTGT_BH"]), _to_daily(bm["BUYHOLD"])


def carry_satellite():
    """the funding-dispersion DEPLOYABLE daily net stream (the orthogonal carry), sliced to OOS."""
    from strat.funding_satellite_assessment import satellite_net_stream
    s = satellite_net_stream(universe="u50", k=5)
    s.index = s.index.normalize()
    oos = s[(s.index >= pd.Timestamp(OOS[0])) & (s.index < pd.Timestamp(OOS[1]))].dropna()
    return oos


# ===========================================================================
# 3. metrics on a daily net stream
# ===========================================================================
def stats(daily, ann=ANN_DAILY):
    d = np.asarray(daily, float)
    d = d[np.isfinite(d)]
    if len(d) < 5:
        return {"n_days": int(len(d)), "error": "too short"}
    eq = np.cumprod(1 + d)
    peak = np.maximum.accumulate(eq)
    maxdd = float(((eq - peak) / peak).min() * 100)
    nyr = len(d) / ann
    comp = float((eq[-1] - 1) * 100)
    ann_ind = float(((1 + comp / 100) ** (1 / nyr) - 1) * 100) if nyr > 0 and eq[-1] > 0 else float("nan")
    sh = float(np.mean(d) / (np.std(d) + 1e-12) * np.sqrt(ann))
    out = {"n_days": int(len(d)), "compound": round(comp, 1), "ann_indicative": round(ann_ind, 1),
           "ann_vol_pct": round(float(np.std(d) * np.sqrt(ann) * 100), 1), "sharpe": round(sh, 2),
           "maxdd": round(maxdd, 1), "daily_pos_rate_pct": round(float((d > 0).mean() * 100), 1)}
    if len(d) > 8:
        bb = block_bootstrap_p05_p95(d)
        out["p05"] = bb.get("p05")
        out["p95"] = bb.get("p95")
    return out


# ===========================================================================
# 4. combine sleeve daily streams at PRE-REGISTERED capital weights (sum to 1)
# ===========================================================================
def _align(sleeves):
    """align sleeve daily Series on the OUTER date union (a sleeve flat on a missing day = 0 net).
    Returns a DataFrame (cols = sleeve names) reindexed to the union, NaN where a sleeve has no bar."""
    df = pd.DataFrame(sleeves)
    return df.sort_index()


def equal_weight(df):
    """1/N capital weights -- the book net on each day = mean of the sleeves PRESENT that day (skipna)."""
    return df.mean(axis=1, skipna=True)


def inverse_vol_weights(df):
    """pre-registered inverse-vol weights: w_i ~ 1/sd_i on the FULL overlap (a deployment-constant level,
    NOT a forward). Computed on the COMMON-date overlap so the sds are comparable. Returns (weights dict,
    weighted daily book over the union)."""
    overlap = df.dropna()
    sds = overlap.std()
    inv = 1.0 / (sds + 1e-12)
    w = (inv / inv.sum()).to_dict()
    # apply the fixed weights to the union, renormalizing per-day over the PRESENT sleeves (so a missing
    # sleeve does not silently shrink the book to <1 gross; the present sleeves carry the full capital).
    wser = pd.Series(w)
    mask = df.notna()
    wmat = mask.mul(wser, axis=1)
    wmat = wmat.div(wmat.sum(axis=1).replace(0, np.nan), axis=0)         # renormalize present sleeves to 1
    book = (df.fillna(0.0) * wmat).sum(axis=1)
    book = book[mask.any(axis=1)]                                        # keep days with >=1 sleeve
    return {k: round(float(v), 3) for k, v in w.items()}, book


# ===========================================================================
# 5. participation / coverage union
# ===========================================================================
def coverage_union(sleeves_engaged):
    """sleeves_engaged = {name: daily bool Series 'profitably engaged that day' = sleeve net > 0}.
    Returns the fraction of OOS days >=1 sleeve is profitably engaged (the UNION), plus each-alone."""
    df = pd.DataFrame(sleeves_engaged).sort_index()
    any_engaged = df.fillna(False).any(axis=1)
    union_days = df.index[df.notna().any(axis=1)]
    cov_union = float(any_engaged.reindex(union_days).fillna(False).mean()) * 100
    per_sleeve = {}
    for c in df.columns:
        s = df[c].dropna()
        per_sleeve[c] = round(float(s.mean()) * 100, 1) if len(s) else None
    return round(cov_union, 1), per_sleeve, len(union_days)


def in_market_union(sleeves_inmkt):
    """sleeves_inmkt = {name: daily bool 'in market that day' (nonzero exposure proxy: |net|>0 OR the
    sleeve had ANY position). We proxy 'engaged' by net != 0 (a flat sleeve nets exactly 0). Returns the
    fraction of OOS days >=1 sleeve is IN MARKET (engaged at all, win or lose) -- the participation union."""
    df = pd.DataFrame(sleeves_inmkt).sort_index()
    any_in = df.fillna(False).any(axis=1)
    union_days = df.index[df.notna().any(axis=1)]
    cov = float(any_in.reindex(union_days).fillna(False).mean()) * 100
    per = {c: round(float(df[c].dropna().mean()) * 100, 1) if len(df[c].dropna()) else None
           for c in df.columns}
    return round(cov, 1), per, len(union_days)


# ===========================================================================
# 6. cross-TF diversification: correlation matrix + n_eff
# ===========================================================================
def corr_and_neff(df):
    """cross-TF correlation matrix (on the common-date overlap) + n_eff from the mean pairwise corr."""
    overlap = df.dropna()
    if len(overlap) < 10 or overlap.shape[1] < 2:
        return None, None, None
    corr = overlap.corr()
    n = corr.shape[0]
    off = corr.to_numpy()[~np.eye(n, dtype=bool)]
    mean_corr = float(np.mean(off))
    neff = float(n / (1 + (n - 1) * mean_corr)) if (1 + (n - 1) * mean_corr) > 0 else float(n)
    return {a: {b: round(float(corr.loc[a, b]), 3) for b in corr.columns} for a in corr.index}, \
           round(mean_corr, 3), round(neff, 2)


# ===========================================================================
# 7. rolling-window positivity (the 'profit daily' honesty)
# ===========================================================================
def window_positivity(daily, win):
    """fraction of rolling `win`-day windows (compound) that are positive."""
    d = daily.dropna().to_numpy()
    if len(d) < win + 1:
        return None
    comp = np.array([np.prod(1 + d[i:i + win]) - 1 for i in range(len(d) - win + 1)])
    return round(float((comp > 0).mean()) * 100, 1)


# ===========================================================================
# 8. satellite blend (CORE + carry), leverage-capped capital split
# ===========================================================================
def blend_core_sat(core_daily, sat_daily, core_frac, sat_leverage):
    """combined = core_frac*core + (1-core_frac)*(L*sat) on the common dates. L capped at SAT_MAX_LEVERAGE."""
    L = min(float(sat_leverage), SAT_MAX_LEVERAGE)
    df = pd.concat({"core": core_daily, "sat": sat_daily}, axis=1).dropna()
    if len(df) < 10:
        return None, None, 0
    c = df["core"]; s = df["sat"]
    combined = core_frac * c + (1.0 - core_frac) * (L * s)
    return combined, df, len(df)


def realized_risk_share(core_leg, sat_leg):
    sc = float(np.std(core_leg)); ss = float(np.std(sat_leg))
    return round(float(ss / (sc + ss + 1e-12)), 3) if (sc + ss) > 0 else 0.0


# ===========================================================================
# MAIN
# ===========================================================================
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="python -m strat.ironed_combined")
    ap.add_argument("--sat-leverage", type=float, default=SAT_MAX_LEVERAGE)
    ap.add_argument("--core-frac", type=float, default=RECO_CORE_FRAC)
    a = ap.parse_args(argv)

    print("## IRONED COMBINED -- the COMBINATION / PARTICIPATION layer (2020 deep-dive, within-2020 OOS)")
    print(f"   OOS {OOS} (3mo bull tail); cost maker {MAKER_RT}; causal/lag-1; weights pre-registered")
    print("   reconstructing the recommended per-TF sleeves (deterministic, from spec kwargs)...")
    sys.stdout.flush()

    # ---- reconstruct the recommended sleeves (the DAILY net streams) ----
    sleeves = {}
    sleeves["1d_family"] = coarse_sleeve("1d", dict(family=True, exit_="none", conf_k=0,
                                                    gate=False, voltgt=False))
    sleeves["4h_ironed"] = coarse_sleeve("4h", dict(family=True, exit_="none", conf_k=2,
                                                    gate=True, voltgt=True))
    sleeves["15m_nogate"] = fine_sleeve("15m", whip=(8, 96, 48), exit_="none", gate=0, half=False)
    # also build the 15m half-gate (best-DD) as an ALTERNATIVE fine sleeve (reported, not in the core book)
    sleeve_15m_half = fine_sleeve("15m", whip=(8, 96, 48), exit_="none", gate=168, half=True)

    for k, v in sleeves.items():
        m = stats(v)
        print(f"   sleeve {k:12}: OOS {m['compound']}% maxDD {m['maxdd']} Sh {m['sharpe']} "
              f"p05 {m.get('p05')} dPos {m['daily_pos_rate_pct']}% n={m['n_days']}")

    # ---- benchmarks ----
    voltgt_bh, buyhold = benchmark_voltgt_bh_1d()
    vtg_m = stats(voltgt_bh); bh_m = stats(buyhold)
    print(f"   BENCH VOLTGT_BH(1d): {vtg_m['compound']}% maxDD {vtg_m['maxdd']} Sh {vtg_m['sharpe']} | "
          f"BUYHOLD(1d): {bh_m['compound']}% maxDD {bh_m['maxdd']}")

    # =====================================================================
    # CORE BOOK families: (A) the two strict DEPLOY sleeves {1d,4h};
    #                     (B) the full participating core {1d,4h,15m}.
    # =====================================================================
    book_sets = {
        "deploy_1d_4h": {"1d_family": sleeves["1d_family"], "4h_ironed": sleeves["4h_ironed"]},
        "core_1d_4h_15m": {"1d_family": sleeves["1d_family"], "4h_ironed": sleeves["4h_ironed"],
                           "15m_nogate": sleeves["15m_nogate"]},
    }

    results = {"sleeves_oos": {k: stats(v) for k, v in sleeves.items()},
               "sleeve_15m_halfgate_oos": stats(sleeve_15m_half),
               "benchmarks_oos": {"VOLTGT_BH_1d": vtg_m, "BUYHOLD_1d": bh_m},
               "books": {}}

    for bname, sset in book_sets.items():
        print(f"\n========== CORE BOOK: {bname} ({len(sset)} sleeves) ==========")
        df = _align(sset)

        # ---- MEASURE 1: participation / coverage union ----
        engaged = {k: (v > 0) for k, v in sset.items()}              # profitably engaged = net > 0
        inmkt = {k: (v != 0) for k, v in sset.items()}               # in market = nonzero net (flat=0)
        cov_profit, per_profit, ndays = coverage_union(engaged)
        cov_inmkt, per_inmkt, _ = in_market_union(inmkt)
        print(f"   [1 PARTICIPATION] OOS days={ndays} | UNION >=1 sleeve profitably engaged: {cov_profit}% "
              f"(per-sleeve {per_profit}) | UNION >=1 in-market: {cov_inmkt}% (per-sleeve {per_inmkt})")

        # ---- MEASURE 2: cross-TF diversification ----
        cmat, mean_corr, neff = corr_and_neff(df)
        print(f"   [2 DIVERSIFICATION] mean pairwise corr {mean_corr} -> n_eff {neff} of {len(sset)} sleeves")

        # ---- combine at the two pre-registered weighting rules ----
        eq_book = equal_weight(df)
        iv_w, iv_book = inverse_vol_weights(df)
        eq_m = stats(eq_book); iv_m = stats(iv_book)
        # best single sleeve by compound
        best_sleeve = max(sset, key=lambda k: stats(sset[k])["compound"])
        best_m = stats(sset[best_sleeve])
        print(f"   [3 BOOK NET] EQUAL-WEIGHT: {eq_m['compound']}% maxDD {eq_m['maxdd']} Sh {eq_m['sharpe']} "
              f"p05 {eq_m.get('p05')} | INVERSE-VOL {iv_w}: {iv_m['compound']}% maxDD {iv_m['maxdd']} "
              f"Sh {iv_m['sharpe']} p05 {iv_m.get('p05')}")
        print(f"   [3 vs PARTS] best single sleeve = {best_sleeve} ({best_m['compound']}% maxDD "
              f"{best_m['maxdd']} Sh {best_m['sharpe']} p05 {best_m.get('p05')}); "
              f"VOLTGT_BH {vtg_m['compound']}% Sh {vtg_m['sharpe']}")

        # ---- MEASURE 5: 'profit daily' honesty (rolling-window positivity) on the equal-weight book ----
        pos_1d = window_positivity(eq_book, 1)
        pos_3d = window_positivity(eq_book, 3)
        bh_pos_1d = window_positivity(buyhold, 1); bh_pos_3d = window_positivity(buyhold, 3)
        print(f"   [5 PROFIT-DAILY HONESTY] EQ book: 1d-window +{pos_1d}% / 3d-window +{pos_3d}% positive "
              f"(BUYHOLD 1d +{bh_pos_1d}% / 3d +{bh_pos_3d}% -- the one-factor-market floor)")

        results["books"][bname] = {
            "sleeves": list(sset.keys()),
            "participation": {"oos_days": ndays, "union_profitably_engaged_pct": cov_profit,
                              "per_sleeve_profit_days_pct": per_profit,
                              "union_in_market_pct": cov_inmkt, "per_sleeve_in_market_pct": per_inmkt},
            "diversification": {"corr_matrix": cmat, "mean_pairwise_corr": mean_corr, "n_eff": neff},
            "book_net": {"equal_weight": eq_m, "inverse_vol": {**iv_m, "weights": iv_w},
                         "best_single_sleeve": {"name": best_sleeve, **best_m}},
            "profit_daily": {"eq_window_pos_1d_pct": pos_1d, "eq_window_pos_3d_pct": pos_3d,
                             "buyhold_window_pos_1d_pct": bh_pos_1d, "buyhold_window_pos_3d_pct": bh_pos_3d},
        }

    # =====================================================================
    # MEASURE 4: THE ORTHOGONAL DIVERSIFIER -- CORE(trend) + SATELLITE(carry)
    # =====================================================================
    print(f"\n========== MEASURE 4: ORTHOGONAL CARRY SATELLITE ==========")
    print("   building the funding-dispersion carry satellite (OOS slice)...")
    sys.stdout.flush()
    sat = carry_satellite()
    sat_m = stats(sat)
    print(f"   satellite (funding-dispersion 1x-neutral, OOS): {sat_m['compound']}% maxDD {sat_m['maxdd']} "
          f"Sh {sat_m['sharpe']} annVol {sat_m['ann_vol_pct']}% n={sat_m['n_days']}")

    sat_blend = {}
    # CORE = the equal-weight core_1d_4h_15m book (the participating trend core)
    core_book_eq = equal_weight(_align(book_sets["core_1d_4h_15m"]))
    sat_lev = min(float(a.sat_leverage), SAT_MAX_LEVERAGE)
    core_frac = float(a.core_frac)
    for label, cf in [("core_alone", 1.0), (f"core{int(core_frac*100)}_sat{int((1-core_frac)*100)}", core_frac)]:
        if cf == 1.0:
            df = pd.concat({"core": core_book_eq, "sat": sat}, axis=1).dropna()
            combined = df["core"]
            csm = stats(combined)
            corr = float(np.corrcoef(df["core"].to_numpy(), df["sat"].to_numpy())[0, 1]) if len(df) > 2 else None
            sat_blend[label] = {**csm, "overlap_days": int(len(df)), "core_sat_pearson": round(corr, 4) if corr is not None else None}
        else:
            combined, df, n = blend_core_sat(core_book_eq, sat, cf, sat_lev)
            if combined is None:
                continue
            csm = stats(combined)
            core_leg = cf * df["core"]; sat_leg = (1 - cf) * (sat_lev * df["sat"])
            rs = realized_risk_share(core_leg, sat_leg)
            corr = float(np.corrcoef(df["core"].to_numpy(), df["sat"].to_numpy())[0, 1])
            sat_blend[label] = {**csm, "overlap_days": int(n), "core_frac": cf, "sat_leverage": sat_lev,
                                "sat_realized_risk_share": rs, "core_sat_pearson": round(corr, 4)}

    ca = sat_blend["core_alone"]
    bl_key = [k for k in sat_blend if k != "core_alone"][0]
    bl = sat_blend[bl_key]
    print(f"   CORE-ALONE (eq core_1d_4h_15m, overlap w/ sat): {ca['compound']}% maxDD {ca['maxdd']} "
          f"Sh {ca['sharpe']} p05 {ca.get('p05')}")
    print(f"   CORE+SAT ({bl_key}, sat capped {sat_lev}x, corr {bl['core_sat_pearson']}): {bl['compound']}% "
          f"maxDD {bl['maxdd']} Sh {bl['sharpe']} p05 {bl.get('p05')} | sat risk-share "
          f"{round(bl.get('sat_realized_risk_share',0)*100,1)}%")
    print(f"   => Sharpe {ca['sharpe']}->{bl['sharpe']} ({bl['sharpe']-ca['sharpe']:+.2f}) | "
          f"maxDD {ca['maxdd']}->{bl['maxdd']} ({bl['maxdd']-ca['maxdd']:+.1f}pp) | "
          f"p05 {ca.get('p05')}->{bl.get('p05')}")

    results["orthogonal_satellite"] = {
        "satellite_oos_1x": sat_m,
        "core_def": "equal-weight core_1d_4h_15m (the participating trend core)",
        "blends": sat_blend,
        "sat_max_leverage": SAT_MAX_LEVERAGE, "sat_frac_cap": SAT_FRAC_CAP,
        "diversification_vs_core_alone": {
            "sharpe_core_alone": ca["sharpe"], "sharpe_blend": bl["sharpe"],
            "sharpe_delta": round(bl["sharpe"] - ca["sharpe"], 2),
            "maxdd_core_alone": ca["maxdd"], "maxdd_blend": bl["maxdd"],
            "maxdd_delta_pp": round(bl["maxdd"] - ca["maxdd"], 1),
            "p05_core_alone": ca.get("p05"), "p05_blend": bl.get("p05"),
            "compound_core_alone": ca["compound"], "compound_blend": bl["compound"]},
    }

    # ---- persist ----
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "repro": {"command": "python -m strat.ironed_combined " + " ".join(argv),
                  "git_sha": sha, "cost_maker": MAKER_RT, "split": SPLIT, "oos": OOS,
                  "sat_max_leverage": SAT_MAX_LEVERAGE, "sat_frac_cap": SAT_FRAC_CAP,
                  "reco_core_frac": RECO_CORE_FRAC, "generated": stamp,
                  "weighting_rules": ["equal_weight (1/N)",
                                      "inverse_vol (w~1/sd on full OOS overlap, a deployment constant)"]},
        "results": results,
    }
    p = OUT / "ironed_combined.json"
    json.dump(payload, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[json] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
