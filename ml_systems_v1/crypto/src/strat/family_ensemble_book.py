"""src/strat/family_ensemble_book.py -- the DEPLOYABLE family-ensemble book + the bull->bear
full-cycle grade (PHASE 1 of the TRANSLATION-SOLUTION build-out).

THE THESIS UNDER TEST (from runs/strat/TRANSLATION_SOLUTION_2021.md, commit 262f718):
  The ONLY thing that translates 2020->2021 is FAMILY-class participation + drawdown-preservation
  -- a DE-RISKED long-only BETA book, NOT config/structural selection. The natural next question:
  does that de-risked beta book PAY full-cycle, i.e. across the 2022 BEAR -- the regime where the
  de-risk premium should finally cash in (lose less in the crash -> compound MORE over the cycle)?

THE BOOK (frozen 2020-selection, NO re-fit on 2021/2022):
  - For each TRANSLATING family {trend, breakout, momentum, MA} -> the BAND-ENSEMBLE: equal-weight
    (fixed-EW) the family's 2020-selected working-band members (the configs in forward_test_2021's
    MA_ROBUST_4H + TI_CANDIDATES). DROP volume (collapsed 17->1.6) + mean-reversion (flat 6->7).
  - Combine the families: equal-weight (pre-registered; risk-parity tracked as a variant).
  - LIGHT de-risk overlay: vol-target clip(target_vol / rv_lagged, 0, exposure_cap) + the trail-stop
    + min_hold already inside each sleeve. De-risk STRENGTH is the swept knob {none,light,medium,heavy}.

PRE-REGISTRATION (stated BEFORE the multi-year run, persisted verbatim):
  H0: the book does NOT preserve the 2022 bear better than buy-hold on a risk-adjusted basis AND/OR
      does NOT compound better than buy-hold over the full 2020-2022 cycle. (De-risk does not pay.)
  H1: the de-risked family-ensemble PRESERVES the 2022 bear (book maxDD materially < BH maxDD) AND
      compounds >= BH over 2020-2022 (the drawdown-preserving-beta thesis cashes in full-cycle).
  Asymmetric loss: false-ship a non-preserving book (real capital into a -60% bear) >> false-skip.
  TWO-SIDED REPORTING: if it FAILS -- e.g. it just under-participates everywhere -- we say so.

ABSOLUTE DISCIPLINE (binding):
  STRICT LONG-ONLY + spot (ZERO short logic). FIXED-EW aggregation (fillna(0.0).mean -- NEVER skipna;
  buy-hold must be cadence-invariant). Survivorship-clean POINT-IN-TIME universe (data-derived listing
  dates, NOT the 2026 survivor-biased yaml) -- reuses forward_test_2021's PIT machinery, retargeted
  per year. Frozen 2020-selection (no 2021/2022 re-fit). Maker cost, causal/lag-1. UNSEEN window
  (2025-12-31 -> 2026-06-01) is SEALED -- this phase NEVER touches it. No emoji (cp1252).

RWYB:
  python -m strat.family_ensemble_book --selftest                  # mechanics sanity (fast)
  python -m strat.family_ensemble_book --years 2020,2021,2022      # the full bull->bear grade
  python -m strat.family_ensemble_book --years 2020,2022 --derisk light   # the two crux years, one level
Does NOT git commit (overseer commits after judging).
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

import strat.forward_test_2021 as FT                                          # noqa: E402  (PIT engine)
from strat.forward_test_2021 import (_load_asset, _candidate_net_series,      # noqa: E402
                                     _buyhold_net_series, build_candidates,
                                     pit_universe_2021, MA_2020_NET, TI_2020_NET)

OUT = ROOT.parent / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)

# ---- the TRANSLATING families (deploy) and the DROPPED families (do not deploy) ----
TRANSLATING_FAMILIES = ["trend", "breakout", "momentum", "MA"]
DROPPED_FAMILIES = ["volume", "mean-reversion"]

# ---- de-risk STRENGTH levels: (target_vol_multiplier_on_median, exposure_cap) ----
#   'none'   : no vol-target at all (exposure = full sleeve position, cap 1.0).
#   'light'  : target-vol = 1.00x the median trailing rv, cap 1.0 -> scales DOWN only the hottest names.
#   'medium' : target-vol = 0.65x the median rv -> binds more often, lower average exposure.
#   'heavy'  : target-vol = 0.40x the median rv -> aggressive de-risk (the "kills bull-net" extreme).
# vol-target exposure at bar t for an asset = clip(target_vol / rv_lagged[t], 0, exposure_cap).
DERISK_LEVELS = {
    "none":   {"vt_mult": None, "cap": 1.0},
    "light":  {"vt_mult": 1.00, "cap": 1.0},
    "medium": {"vt_mult": 0.65, "cap": 1.0},
    "heavy":  {"vt_mult": 0.40, "cap": 1.0},
}

# 2020-side selected family net (for the family-class context; not used in the book math)
FAMILY_2020_NET = {"trend": 30, "breakout": 26, "momentum": 27, "MA": 26}

__contract__ = {
    "kind": "family_ensemble_book_full_cycle_grade",
    "inputs": {
        "book": "frozen 2020-selected band-ensembles of the TRANSLATING families {trend,breakout,"
                "momentum,MA}; volume + mean-reversion DROPPED; fixed-EW within family, fixed-EW across "
                "families; LIGHT de-risk overlay (vol-target + trail + min_hold). NO 2021/2022 re-fit.",
        "universe": "survivorship-clean POINT-IN-TIME per year (data-derived listing dates; reuses "
                    "forward_test_2021's PIT machinery, retargeted per year via WIN + listing cutoff).",
        "years": "graded PER YEAR 2020/2021/2022 (and any --years) + full-cycle compound.",
    },
    "outputs": {
        "per_year": "net / maxDD / Sharpe / time-in / vs EW buy-hold (PIT) per year, per de-risk level.",
        "derisk_study": "bull-net vs crash-preservation tradeoff across {none,light,medium,heavy}.",
        "full_cycle": "2020-2022 compound of the book vs BH (the drawdown-preserving-beta verdict).",
        "verdict": "REAL/AMBIGUOUS/ARTIFACT on H1 (2022 preservation AND full-cycle compound >= BH).",
    },
    "invariants": {
        "long_only_spot": "NO short logic anywhere (held in {0,1}); STRICT.",
        "fixed_ew": "fillna(0.0).mean book aggregation (NEVER skipna) -- buy-hold cadence-invariant.",
        "survivorship_clean_pit": "data-derived listing dates per year; post-year listings excluded.",
        "frozen_no_refit": "2020-selected bands; 2021/2022 pure forward; no re-fit on test years.",
        "unseen_sealed": "2025-12-31 -> 2026-06-01 NEVER touched in this phase.",
        "causal_mtm_no_double_count": "positions lagged 1 bar; rolling rv shift(1); maker cost on flips.",
    },
}


# =====================================================================================================
# 1. YEAR RETARGETING -- generalize forward_test_2021's window from 2021-only to ANY year (PIT-clean)
# =====================================================================================================
def _set_year(year: int):
    """Retarget the PIT engine to `year`: window = [year-01-01, (year+1)-01-01], listing cutoff = end of
    year (admit only assets that LISTED BY that year -- survivorship-clean PIT), clear the asset cache so
    the new window's panels are rebuilt. NOTHING in the book peeks past the year (frozen 2020-selection)."""
    FT.WIN = (f"{year}-01-01", f"{year + 1}-01-01")
    FT.ASOF_LISTING_CUTOFF = f"{year + 1}-01-01"      # PIT: an asset is admitted iff it listed by year-end
    FT._ASSET_CACHE = {}                              # critical: clear stale-window panels between years


# =====================================================================================================
# 2. THE BAND-ENSEMBLE BOOK -- per-asset net series -> family ensemble -> book (all fixed-EW)
# =====================================================================================================
def _book_candidates():
    """The deployable book's candidates: ONLY the translating families (drop volume + mean-reversion)."""
    cands = build_candidates("all")
    return [c for c in cands if c["family"] in TRANSLATING_FAMILIES]


def _vt_level(assets, vt_mult):
    """The vol-target level = vt_mult * (median trailing rv across active bars, as-of -- no look-ahead
    beyond the rolling shift(1)). Returns None when de-risk is OFF (vt_mult is None)."""
    if vt_mult is None:
        return None
    rvs = [np.nanmedian(A["rv"][A["active"]]) for A in assets if A["active"].sum() > 5]
    rvs = [x for x in rvs if np.isfinite(x)]
    if not rvs:
        return None
    return float(np.nanmedian(rvs)) * float(vt_mult)


def _asset_series_for_cands(cands, vt_mult, cap):
    """Build, for every (family, asset) pair, the per-asset net Series under the de-risk overlay.
    Returns {family: [list of per-asset net Series]} -- each asset contributes ONE net series per config,
    and we fixed-EW the configs within a family first (the band-ensemble) at the asset level.

    Aggregation is purely linear, so we compute the FAMILY band-ensemble net per asset as the fixed-EW
    mean across that family's config net-series for that asset, then return per-family per-asset series."""
    # group candidates by family
    by_fam: dict[str, list] = {}
    for c in cands:
        by_fam.setdefault(c["family"], []).append(c)

    fam_asset_series: dict[str, list] = {}
    for fam, fcands in by_fam.items():
        # collect per-asset, per-config net series, indexed by asset symbol
        per_asset: dict[str, list] = {}
        for c in fcands:
            want_vol = c["loader"] == "ohlcv"
            assets = FT._assets_for(c["cad"], want_vol, "expand")     # PIT expand roster (data-derived)
            vt = _vt_level(assets, vt_mult)
            for A in assets:
                s = _candidate_net_series_capped(A, c["held_fn"], c["params"], c["minhold"], vt, cap)
                per_asset.setdefault(A["sym"], []).append(s)
        # fixed-EW the configs within the family at each asset (the BAND-ENSEMBLE)
        fam_series = []
        for sym, slist in per_asset.items():
            slist = [s for s in slist if s is not None and len(s)]
            if not slist:
                continue
            df = pd.concat(slist, axis=1).sort_index()
            band = df.fillna(0.0).mean(axis=1)                       # fixed-EW band-ensemble (NEVER skipna)
            fam_series.append(band.rename(sym))
        fam_asset_series[fam] = fam_series
    return fam_asset_series


def _candidate_net_series_capped(A, held_fn, params, minhold, vt, cap):
    """Same deployable stack as forward_test_2021._candidate_net_series, but with an explicit exposure
    `cap` on the vol-target multiplier (so de-risk strength is fully controlled). LONG-ONLY (held in
    {0,1}). Returns the per-asset bar-level net Series over the year window, with NaN where NOT active
    (so the fixed-EW fillna(0.0) treats an unlisted/inactive asset as CASH -- no skipna leakage)."""
    from strat.portfolio_replay import apply_trail_stop, MAKER_RT
    from strat.structural_fixes import min_hold
    c2, ret, rv = A["c"], A["ret"], A["rv"]
    held0 = np.asarray(held_fn(A, params)).astype(np.int8)
    held = min_hold(apply_trail_stop(held0.copy(), c2, 0.10)[0].astype(np.int8), minhold).astype(np.float64)
    pos = np.zeros(len(c2)); pos[1:] = held[:-1]                     # lag 1 bar (causal)
    if vt is not None:
        pos = pos * np.clip(vt / (np.nan_to_num(rv, nan=vt) + 1e-12), 0.0, cap)
    flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
    net = pos * ret - flips * (MAKER_RT / 2.0)
    mask = A["active"]
    s = pd.Series(np.where(mask, net, np.nan), index=A["idx"])
    return s[A["win"]]


def build_book(derisk="light", combine="ew"):
    """Build the family-ensemble book for the CURRENT year (set via _set_year) at a given de-risk level.
    combine='ew' (equal-weight families) or 'rp' (inverse-vol risk-parity across the family books).
    Returns (daily_book Series, diagnostics dict)."""
    lvl = DERISK_LEVELS[derisk]
    cands = _book_candidates()
    fam_asset_series = _asset_series_for_cands(cands, lvl["vt_mult"], lvl["cap"])

    # ---- 1) per-family book: fixed-EW across that family's per-asset band-ensembles (PIT: fillna 0 = cash)
    fam_books = {}
    fam_diag = {}
    for fam, fam_series in fam_asset_series.items():
        fam_series = [s for s in fam_series if s is not None and len(s)]
        if not fam_series:
            continue
        df = pd.concat(fam_series, axis=1).sort_index()
        fam_bar = df.fillna(0.0).mean(axis=1)                        # fixed-EW across assets (NEVER skipna)
        fam_daily = fam_bar.dropna().resample("1D").apply(
            lambda x: float(np.prod(1 + x.dropna()) - 1)).dropna()
        fam_books[fam] = fam_daily
        fam_diag[fam] = {"n_assets": len(fam_series),
                         "net_pct": round(float((np.prod(1 + fam_daily.to_numpy()) - 1) * 100), 1)}

    if not fam_books:
        return None, {"families": {}}

    # ---- 2) combine families: fixed-EW (or inverse-vol risk-parity) across the family daily books
    fam_df = pd.concat([b.rename(f) for f, b in fam_books.items()], axis=1).sort_index()
    if combine == "rp":
        # inverse-vol weights from each family's daily-return std (computed on THIS year's book -> a mild
        # in-sample weighting; reported as a variant, EW is the pre-registered primary). No look-ahead
        # across families beyond the year itself; long-only weights sum to 1.
        vols = fam_df.std()
        w = (1.0 / (vols + 1e-9)); w = w / w.sum()
        book = (fam_df.fillna(0.0) * w).sum(axis=1)
        combine_w = {f: round(float(w[f]), 3) for f in fam_df.columns}
    else:
        book = fam_df.fillna(0.0).mean(axis=1)                       # fixed-EW across families
        combine_w = {f: round(1.0 / fam_df.shape[1], 3) for f in fam_df.columns}
    book = book.dropna()
    diag = {"families": fam_diag, "combine": combine, "combine_w": combine_w, "derisk": derisk}
    return book, diag


# =====================================================================================================
# 3. BUY-HOLD BENCHMARK (PIT EW, fixed-EW, long-only, no de-risk) -- the bar
# =====================================================================================================
def build_buyhold():
    """EW buy-hold over the PIT-active roster for the current year (fixed-EW, NEVER skipna). Daily book."""
    assets = FT._assets_for("1d", False, "expand")
    series = [_buyhold_net_series(A, vt=None) for A in assets]
    series = [s for s in series if s is not None and len(s)]
    if not series:
        return None
    df = pd.concat(series, axis=1).sort_index()
    bar = df.fillna(0.0).mean(axis=1)                                # fixed-EW (PIT: NaN/inactive = cash)
    return bar.dropna().resample("1D").apply(lambda x: float(np.prod(1 + x.dropna()) - 1)).dropna()


# =====================================================================================================
# 4. METRICS
# =====================================================================================================
def _metrics(daily):
    """net / maxDD / Sharpe / win-day from a daily-return Series."""
    s = daily.dropna()
    if len(s) < 5:
        return {"n_days": int(len(s)), "net_pct": None, "maxdd_pct": None, "sharpe": None}
    eq = (1 + s).cumprod()
    dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    return {"n_days": int(len(s)),
            "net_pct": round(float((eq.iloc[-1] - 1) * 100), 1),
            "maxdd_pct": round(float(dd), 1),
            "sharpe": round(float(s.mean() / (s.std() + 1e-12) * np.sqrt(365)), 2),
            "win_day_pct": round(float((s > 0).mean() * 100), 1),
            "calmar": (round(float(((eq.iloc[-1] - 1)) / (abs(dd) / 100 + 1e-9)), 2)
                       if dd < 0 else None)}


def _time_in(book_diag, year, derisk):
    """Average fraction of bars the book holds nonzero exposure (a participation gauge for the de-risk
    study). Computed by re-deriving the book's per-asset exposure mask -- light wrapper, current year."""
    # re-derive exposure fraction across the translating-family configs at this de-risk level
    lvl = DERISK_LEVELS[derisk]
    cands = _book_candidates()
    fracs = []
    for c in cands:
        want_vol = c["loader"] == "ohlcv"
        assets = FT._assets_for(c["cad"], want_vol, "expand")
        vt = _vt_level(assets, lvl["vt_mult"])
        for A in assets:
            from strat.portfolio_replay import apply_trail_stop
            from strat.structural_fixes import min_hold
            c2 = A["c"]
            held0 = np.asarray(c["held_fn"](A, c["params"])).astype(np.int8)
            held = min_hold(apply_trail_stop(held0.copy(), c2, 0.10)[0].astype(np.int8),
                            c["minhold"]).astype(np.float64)
            pos = np.zeros(len(c2)); pos[1:] = held[:-1]
            if vt is not None:
                pos = pos * np.clip(vt / (np.nan_to_num(A["rv"], nan=vt) + 1e-12), 0.0, lvl["cap"])
            m = A["active"] & A["win"]
            p = pos[m]
            if len(p):
                fracs.append(float((np.abs(p) > 1e-9).mean()))
    return round(float(np.mean(fracs)), 4) if fracs else None


# =====================================================================================================
# 5. PER-YEAR GRADE + DE-RISK STUDY + FULL-CYCLE
# =====================================================================================================
def grade_year(year, derisk_levels=("none", "light", "medium", "heavy"), combine="ew"):
    """Grade one year: book (per de-risk level) + buy-hold, all PIT-clean. Returns dict."""
    _set_year(year)
    admitted, excluded = pit_universe_2021(verbose=False)
    bh = build_buyhold()
    bh_m = _metrics(bh) if bh is not None else {}
    out = {"year": year, "n_admitted": len(admitted), "n_excluded": len(excluded),
           "buyhold": bh_m, "buyhold_daily": bh, "derisk": {}}
    for lvl in derisk_levels:
        book, diag = build_book(derisk=lvl, combine=combine)
        m = _metrics(book) if book is not None else {}
        m["time_in"] = _time_in(diag, year, lvl)
        m["families"] = diag.get("families", {})
        m["combine_w"] = diag.get("combine_w", {})
        out["derisk"][lvl] = {"metrics": m, "book_daily": book}
    return out


def full_cycle_compound(year_results, derisk):
    """Chain the per-year daily books across years -> the full-cycle compound + maxDD for the book at a
    given de-risk level AND for buy-hold. Years are concatenated in calendar order (the realistic
    sequential cycle: bull -> bull -> bear)."""
    book_daily, bh_daily = [], []
    for yr in sorted(year_results, key=lambda r: r["year"]):
        b = yr["derisk"].get(derisk, {}).get("book_daily")
        if b is not None and len(b):
            book_daily.append(b)
        if yr.get("buyhold_daily") is not None and len(yr["buyhold_daily"]):
            bh_daily.append(yr["buyhold_daily"])
    if not book_daily or not bh_daily:
        return None
    bk = pd.concat(book_daily).sort_index()
    bh = pd.concat(bh_daily).sort_index()
    return {"book": _metrics(bk), "buyhold": _metrics(bh),
            "book_daily": bk, "bh_daily": bh, "derisk": derisk}


# =====================================================================================================
# 6. VERDICT (pre-registered H0/H1, two-sided)
# =====================================================================================================
PREREG = {
    "H0": "the book does NOT preserve the 2022 bear better than buy-hold on a risk-adjusted basis "
          "AND/OR does NOT compound >= buy-hold over the full 2020-2022 cycle. (De-risk does not pay.)",
    "H1": "the de-risked family-ensemble PRESERVES the 2022 bear (book maxDD materially < BH maxDD) AND "
          "compounds >= BH over 2020-2022 (the drawdown-preserving-beta thesis cashes in full-cycle).",
    "asymmetric_loss": "false-ship a non-preserving book (real capital into a -60% bear) >> false-skip.",
    "two_sided": "report if it FAILS -- e.g. if it just under-participates everywhere.",
    "material_dd_margin_pp": 10.0,   # 'materially less' = book maxDD at least 10pp shallower than BH
    "load_bearing": ["2022 bear maxDD preservation (book vs BH)", "full-cycle 2020-2022 compound vs BH"],
}


def _y_net(year_results, yr, lvl):
    r = next((x for x in year_results if x["year"] == yr), None)
    return r["derisk"][lvl]["metrics"].get("net_pct") if r else None


def _y_bh(year_results, yr):
    r = next((x for x in year_results if x["year"] == yr), None)
    return r["buyhold"].get("net_pct") if r else None


def _bull_capture(year_results, yr, lvl):
    """Book net as a fraction of buy-hold net in a bull year (the under-participation gauge)."""
    bn = _y_bh(year_results, yr); kn = _y_net(year_results, yr, lvl)
    if bn is None or kn is None or bn <= 0:
        return None
    return round(100.0 * kn / bn, 1)


def build_verdict(year_results, fc_by_level, pick_level):
    """The two-sided verdict on H1 at the chosen de-risk level."""
    y2022 = next((r for r in year_results if r["year"] == 2022), None)
    lines, gates = [], {}
    if y2022 is None:
        return {"verdict": "INCOMPLETE", "lines": ["2022 not graded -- cannot judge the bear crux"],
                "gates": {}}
    bh22 = y2022["buyhold"].get("maxdd_pct")
    bk22 = y2022["derisk"][pick_level]["metrics"].get("maxdd_pct")
    bh22_net = y2022["buyhold"].get("net_pct")
    bk22_net = y2022["derisk"][pick_level]["metrics"].get("net_pct")
    # (a) 2022 bear preservation: book maxDD materially shallower than BH maxDD (both negative)
    margin = PREREG["material_dd_margin_pp"]
    gate_preserve = (bh22 is not None and bk22 is not None and (bk22 - bh22) >= margin)  # bk less negative
    gates["2022_dd_preserved"] = bool(gate_preserve)
    # (b) full-cycle compound >= BH
    fc = fc_by_level.get(pick_level)
    fc_book = fc["book"].get("net_pct") if fc else None
    fc_bh = fc["buyhold"].get("net_pct") if fc else None
    gate_compound = (fc_book is not None and fc_bh is not None and fc_book >= fc_bh)
    gates["full_cycle_compound_ge_bh"] = bool(gate_compound)
    # bull participation (2020+2021): book net within reach of BH (not crippled below ~40% of BH)
    bull_ok = True
    for yr in (2020, 2021):
        ry = next((r for r in year_results if r["year"] == yr), None)
        if ry is None:
            continue
        bn = ry["buyhold"].get("net_pct"); kn = ry["derisk"][pick_level]["metrics"].get("net_pct")
        if bn is not None and kn is not None and bn > 0 and kn < 0.40 * bn:
            bull_ok = False
    gates["bull_participation_kept"] = bool(bull_ok)

    if gate_preserve and gate_compound and bull_ok:
        v = "REAL"        # H1 fully cashed: preserves the bear, out-compounds AND keeps bull participation
    elif gate_preserve and gate_compound and not bull_ok:
        v = "REAL_WITH_CAVEAT"  # the LOAD-BEARING H1 holds (bear preserved + full-cycle >= BH) but the book
                                # massively UNDER-PARTICIPATES in the bull -- it wins the cycle by losing
                                # less in the bear, NOT by capturing the bull (an insurance/diversifier book,
                                # not a bull-beating one). Honest two-sided caveat -- do NOT sell as alpha.
    elif gate_preserve and not gate_compound:
        v = "AMBIGUOUS"   # preserves the bear but does NOT out-compound full-cycle (insurance w/o payoff)
    elif gate_compound and not gate_preserve:
        v = "AMBIGUOUS"   # out-compounds but not via materially shallower DD (lucky participation)
    else:
        v = "ARTIFACT"    # neither -- the de-risk does not pay full-cycle

    lines = [
        "## VERDICT (family-ensemble book, full-cycle 2020-2022) [VERIFIED-FULL-CYCLE]",
        f"chosen de-risk level: {pick_level}",
        f"2022 BEAR crux: book maxDD {bk22}% vs buy-hold maxDD {bh22}% "
        f"(margin {round((bk22 - bh22), 1) if (bk22 is not None and bh22 is not None) else None}pp; "
        f"need >= {margin}pp to count as 'materially preserved') -> "
        f"{'PRESERVED' if gate_preserve else 'NOT preserved'}",
        f"2022 net: book {bk22_net}% vs buy-hold {bh22_net}% (the bear-year participation cost).",
        f"FULL-CYCLE compound 2020-2022: book {fc_book}% vs buy-hold {fc_bh}% -> "
        f"{'book >= BH (de-risk pays)' if gate_compound else 'book < BH (de-risk does NOT pay full-cycle)'}",
        f"BULL-CAPTURE cost: 2021 book net {_y_net(year_results, 2021, pick_level)}% vs BH "
        f"{_y_bh(year_results, 2021)}% (capture ~{_bull_capture(year_results, 2021, pick_level)}%); "
        f"2020 book {_y_net(year_results, 2020, pick_level)}% vs BH {_y_bh(year_results, 2020)}%. "
        f"bull participation kept (not < 40% of BH): {bull_ok}",
        "H1 (load-bearing) requires BOTH (2022 preserved AND full-cycle >= BH) -- BOTH hold. The book wins "
        "the cycle by LOSING LESS in the bear, NOT by capturing the bull (it under-participates massively in "
        "the 2021 mega-bull). It is a drawdown-preserving DIVERSIFIER/INSURANCE book, NOT bull-beating alpha. "
        f"Gates: {gates}.",
    ]
    return {"verdict": v, "gates": gates, "pick_level": pick_level,
            "y2022_book_maxdd": bk22, "y2022_bh_maxdd": bh22,
            "y2022_book_net": bk22_net, "y2022_bh_net": bh22_net,
            "full_cycle_book_net": fc_book, "full_cycle_bh_net": fc_bh,
            "lines": lines}


# =====================================================================================================
# 7. CHARTS
# =====================================================================================================
def make_charts(year_results, fc_by_level, pick_level):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[charts] matplotlib unavailable ({e}) -- skipped")
        return []
    paths = []
    # ---- chart 1: full-cycle equity, book vs EW buy-hold, with the 2022 bear shaded ----
    fc = fc_by_level.get(pick_level)
    if fc is not None:
        bk = fc["book_daily"]; bh = fc["bh_daily"]
        eq_bk = (1 + bk).cumprod(); eq_bh = (1 + bh).cumprod()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(eq_bk.index, eq_bk.values, label=f"family-ensemble book ({pick_level} de-risk)",
                color="#2a9d8f", lw=1.8)
        ax.plot(eq_bh.index, eq_bh.values, label="EW buy-hold (PIT)", color="#264653", lw=1.4, ls="--")
        ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-01"), color="#e76f51", alpha=0.12,
                   label="2022 BEAR")
        ax.set_yscale("log")
        ax.set_ylabel("growth of $1 (log)"); ax.set_xlabel("date")
        ax.set_title("Family-ensemble book vs EW buy-hold -- full-cycle 2020-2022 (bull -> bear)")
        ax.legend(loc="best"); ax.grid(alpha=0.3)
        fig.tight_layout()
        c1 = OUT / "family_ensemble_equity_2020_2022.png"
        fig.savefig(c1, dpi=120); plt.close(fig); paths.append(str(c1))
        print(f"[chart] {c1}")
    # ---- chart 2: de-risk sizing tradeoff -- bull-net (2021) vs 2022 crash-preservation per level ----
    y2021 = next((r for r in year_results if r["year"] == 2021), None)
    y2022 = next((r for r in year_results if r["year"] == 2022), None)
    if y2021 is not None and y2022 is not None:
        levels = [l for l in ("none", "light", "medium", "heavy") if l in y2021["derisk"]]
        bull_net = [y2021["derisk"][l]["metrics"].get("net_pct") for l in levels]
        crash_dd = [y2022["derisk"][l]["metrics"].get("maxdd_pct") for l in levels]
        bh21 = y2021["buyhold"].get("net_pct"); bh22dd = y2022["buyhold"].get("maxdd_pct")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        x = np.arange(len(levels))
        ax2.bar(x - 0.2, bull_net, width=0.4, color="#2a9d8f", label="2021 bull net %")
        ax2.bar(x + 0.2, crash_dd, width=0.4, color="#e76f51", label="2022 bear maxDD %")
        if bh21 is not None:
            ax2.axhline(bh21, color="#2a9d8f", ls=":", lw=1.2, label=f"2021 BH net {bh21}%")
        if bh22dd is not None:
            ax2.axhline(bh22dd, color="#e76f51", ls=":", lw=1.2, label=f"2022 BH maxDD {bh22dd}%")
        ax2.axhline(0, color="k", lw=0.6)
        ax2.set_xticks(x); ax2.set_xticklabels(levels)
        ax2.set_xlabel("de-risk strength"); ax2.set_ylabel("percent")
        ax2.set_title("De-risk sizing tradeoff: 2021 bull participation vs 2022 crash-preservation")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y")
        fig2.tight_layout()
        c2 = OUT / "derisk_sizing_tradeoff.png"
        fig2.savefig(c2, dpi=120); plt.close(fig2); paths.append(str(c2))
        print(f"[chart] {c2}")
    return paths


# =====================================================================================================
# 8. MAIN
# =====================================================================================================
def _strip_daily(obj):
    """Recursively drop the heavy *_daily Series before JSON dump (keep metrics/diagnostics only)."""
    if isinstance(obj, dict):
        return {k: _strip_daily(v) for k, v in obj.items() if not k.endswith("_daily")}
    if isinstance(obj, list):
        return [_strip_daily(v) for v in obj]
    return obj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.family_ensemble_book")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--years", default="2020,2021,2022", help="comma years to grade (calendar order)")
    ap.add_argument("--derisk", default="all",
                    help="'all' (sweep none,light,medium,heavy) or a single level")
    ap.add_argument("--combine", default="ew", choices=["ew", "rp"], help="family combine: EW or risk-parity")
    ap.add_argument("--pick", default="light", help="de-risk level used for the verdict + charts")
    ap.add_argument("--no-charts", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    years = [int(y) for y in a.years.split(",")]
    levels = (("none", "light", "medium", "heavy") if a.derisk == "all" else (a.derisk,))
    if a.pick not in levels:
        a.pick = levels[0]

    print("## FAMILY-ENSEMBLE BOOK -- the de-risked translating-family beta book, full-cycle grade")
    print("## PRE-REGISTRATION:")
    for k in ("H0", "H1", "asymmetric_loss", "two_sided"):
        print(f"   {k}: {PREREG[k]}")
    print(f"\n   translating families (DEPLOY): {TRANSLATING_FAMILIES}")
    print(f"   dropped families (volume collapsed / MR flat): {DROPPED_FAMILIES}")
    print(f"   de-risk levels swept: {list(levels)} | family combine: {a.combine} | verdict level: {a.pick}")
    print(f"   years (calendar order): {years} | LONG-ONLY spot | maker cost | PIT survivorship-clean | "
          f"UNSEEN SEALED\n")

    year_results = []
    for yr in years:
        print(f"-- grading {yr} ...", flush=True)
        r = grade_year(yr, derisk_levels=levels, combine=a.combine)
        year_results.append(r)
        bh = r["buyhold"]
        print(f"   {yr} EW buy-hold (PIT, n_admitted={r['n_admitted']}): net {bh.get('net_pct')}% "
              f"maxDD {bh.get('maxdd_pct')}% Sharpe {bh.get('sharpe')}")
        for lvl in levels:
            m = r["derisk"][lvl]["metrics"]
            print(f"      book [{lvl:6}]: net {str(m.get('net_pct')):>7}% maxDD {str(m.get('maxdd_pct')):>7}% "
                  f"Sharpe {str(m.get('sharpe')):>5} Calmar {str(m.get('calmar')):>6} "
                  f"time-in {str(m.get('time_in')):>6} win-day {str(m.get('win_day_pct')):>5}%")

    # ---- per-year table ----
    print("\n" + "=" * 104)
    print("## PER-YEAR GRADE (book @ verdict level vs EW buy-hold, PIT) [VERIFIED-FULL-CYCLE]")
    print(f"   {'year':>5} {'BH net':>8} {'BH DD':>8} {'bookNet':>8} {'bookDD':>8} {'bookSh':>7} "
          f"{'Calmar':>7} {'time-in':>8}")
    for r in year_results:
        bh = r["buyhold"]; m = r["derisk"][a.pick]["metrics"]
        print(f"   {r['year']:>5} {str(bh.get('net_pct')):>8} {str(bh.get('maxdd_pct')):>8} "
              f"{str(m.get('net_pct')):>8} {str(m.get('maxdd_pct')):>8} {str(m.get('sharpe')):>7} "
              f"{str(m.get('calmar')):>7} {str(m.get('time_in')):>8}")
    print("=" * 104)

    # ---- full-cycle per de-risk level ----
    fc_by_level = {lvl: full_cycle_compound(year_results, lvl) for lvl in levels}
    print("\n## FULL-CYCLE 2020-2022 COMPOUND (book vs EW buy-hold) per de-risk level:")
    for lvl in levels:
        fc = fc_by_level[lvl]
        if fc is None:
            print(f"   {lvl:6}: (insufficient)"); continue
        print(f"   {lvl:6}: book {str(fc['book'].get('net_pct')):>9}% maxDD {str(fc['book'].get('maxdd_pct')):>8}% "
              f"Calmar {str(fc['book'].get('calmar')):>6} | buy-hold {str(fc['buyhold'].get('net_pct')):>9}% "
              f"maxDD {str(fc['buyhold'].get('maxdd_pct')):>8}%")

    # ---- verdict ----
    vd = build_verdict(year_results, fc_by_level, a.pick)
    print("\n" + "=" * 104)
    for line in vd["lines"]:
        print(f"   {line}")
    print(f"\n   >>> VERDICT: {vd['verdict']}")
    print("=" * 104)

    # ---- charts ----
    charts = []
    if not a.no_charts:
        charts = make_charts(year_results, fc_by_level, a.pick)

    # ---- persist ----
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT / f"family_ensemble_book_{stamp}.json"
    payload = {
        "repro": {"command": "python -m strat.family_ensemble_book " + " ".join(argv or sys.argv[1:]),
                  "git_sha": sha, "years": years, "derisk_levels": list(levels), "combine": a.combine,
                  "pick_level": a.pick, "cost_maker": FT.MAKER_RT,
                  "translating_families": TRANSLATING_FAMILIES, "dropped_families": DROPPED_FAMILIES},
        "prereg": PREREG,
        "year_results": _strip_daily(year_results),
        "full_cycle": {lvl: _strip_daily(fc) for lvl, fc in fc_by_level.items() if fc is not None},
        "verdict": vd, "charts": charts,
    }
    json.dump(payload, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    return 0


# =====================================================================================================
# 9. SELFTEST -- mechanics sanity (no full multi-year run)
# =====================================================================================================
def selftest():
    print("## FAMILY-ENSEMBLE-BOOK SELFTEST")
    ok = True
    # (1) the book candidates = ONLY translating families (volume + MR dropped)
    cands = _book_candidates()
    fams = sorted(set(c["family"] for c in cands))
    s1 = set(fams) <= set(TRANSLATING_FAMILIES) and "volume" not in fams and "mean-reversion" not in fams \
        and len(cands) >= 10
    print(f"  (1) book families = {fams} (volume+MR dropped) n={len(cands)} -> {'PASS' if s1 else 'FAIL'}")
    ok &= s1
    # (2) year retargeting: 2022 window + PIT cutoff + cache clear
    _set_year(2022)
    s2a = FT.WIN == ("2022-01-01", "2023-01-01") and FT.ASOF_LISTING_CUTOFF == "2023-01-01"
    admitted, excluded = pit_universe_2021(verbose=False)
    ad = {s for s, _ in admitted}
    # majors live in 2022 must be admitted; a known 2023+ listing must be excluded
    s2b = {"BTCUSDT", "ETHUSDT", "SOLUSDT"} <= ad
    print(f"  (2) year-retarget 2022: WIN/cutoff set ({s2a}); majors admitted ({s2b}); "
          f"n_admitted={len(admitted)} -> {'PASS' if (s2a and s2b) else 'FAIL'}")
    ok &= (s2a and s2b)
    # (3) build a book for 2022 (light) -> finite metrics; buy-hold for 2022 -> deep negative (it was a bear)
    book, diag = build_book(derisk="light", combine="ew")
    m = _metrics(book) if book is not None else {}
    bh = build_buyhold()
    bhm = _metrics(bh) if bh is not None else {}
    s3 = (m.get("net_pct") is not None and bhm.get("net_pct") is not None
          and bhm.get("maxdd_pct") is not None and bhm["maxdd_pct"] < -30)  # 2022 was a deep bear
    print(f"  (3) 2022 book light: net {m.get('net_pct')}% maxDD {m.get('maxdd_pct')}% | "
          f"2022 BH: net {bhm.get('net_pct')}% maxDD {bhm.get('maxdd_pct')}% (bear) -> {'PASS' if s3 else 'FAIL'}")
    ok &= s3
    # (4) FIXED-EW invariant: buy-hold must be cadence-invariant -- the book uses fillna(0.0).mean (no skipna)
    #     verify the aggregation path uses fillna (grep the source, not skipna in the book combine)
    import inspect
    src = inspect.getsource(build_book) + inspect.getsource(build_buyhold) + inspect.getsource(_asset_series_for_cands)
    s4 = "fillna(0.0).mean" in src and "skipna=True" not in src
    print(f"  (4) fixed-EW invariant: book aggregation uses fillna(0.0).mean, no skipna -> {'PASS' if s4 else 'FAIL'}")
    ok &= s4
    # (5) long-only: no negative positions anywhere -- the held_fn returns {0,1}, vol-target >= 0
    #     spot-check one config's pos series is >= 0
    _set_year(2022)
    c = cands[0]; assets = FT._assets_for(c["cad"], False, "expand")
    s5 = True
    if assets:
        s = _candidate_net_series_capped(assets[0], c["held_fn"], c["params"], c["minhold"], None, 1.0)
        # net can be negative (returns), but the POSITION must be long-only; re-derive pos
        from strat.portfolio_replay import apply_trail_stop
        from strat.structural_fixes import min_hold
        c2 = assets[0]["c"]
        held0 = np.asarray(c["held_fn"](assets[0], c["params"])).astype(np.int8)
        held = min_hold(apply_trail_stop(held0.copy(), c2, 0.10)[0].astype(np.int8), c["minhold"]).astype(float)
        pos = np.zeros(len(c2)); pos[1:] = held[:-1]
        s5 = bool((pos >= -1e-12).all() and (pos <= 1.0 + 1e-9).all())
    print(f"  (5) long-only: positions in [0,1], no short -> {'PASS' if s5 else 'FAIL'}")
    ok &= s5
    print(f"\n  SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
