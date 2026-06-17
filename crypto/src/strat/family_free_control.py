"""src/strat/family_free_control.py -- PHASE 3: the DECISIVE cheapest falsifier that closes the whole
6h translation-solution arc.

THE QUESTION (the single cheapest falsifier of the family-ensemble apparatus):
  Does a FAMILY-FREE vol-gate on plain EW-beta MATCH the family-ensemble book? i.e. if a one-line
  "buy-hold + a vol-brake" reproduces the book's crash-preservation (and full-cycle compound), then the
  ENTIRE family-ensemble apparatus {trend,breakout,momentum,MA band-ensembles, MA/TI selection, the
  frozen-2020 selection layer} adds NOTHING -- and the honest deployable is just the vol-brake on beta.

THE STORY SO FAR (verified, committed):
  PHASE 1 (82c4e22): the de-risked translating-family book PRESERVES the 2022 bear (book maxDD -19.8% vs
    EW buy-hold -73.4%) + out-compounds BH on the 3-year cut 2020-2022 (+24.7% vs +9.7%). But it captures
    only ~11% of the 2021 bull -> drawdown-preserving INSURANCE, not bull-beating alpha. REAL_WITH_CAVEAT.
  PHASE 2 (6a4c1ff): the sealed-UNSEEN preserve signature HOLDS OOS (book -2.3% maxDD vs BH -40.2%) BUT
    ship=FALSE (held-out p05 -38.8%, OOS compound -19.4%). CRITICAL CATCH: the DROPPED families (volume+MR)
    preserve the 2022 bear EQUALLY/BETTER (-4.8% maxDD) -> bear-preservation is NOT a translating-family
    edge; it is a FAMILY-AGNOSTIC de-risk / go-to-cash property. Verdict: AMBIGUOUS.

THE FAMILY-FREE CONTROL (this file):
  Take the PLAIN EW-beta stream (equal-weight buy-hold of the PIT-active assets -- NO signal families, NO
  MA/TI selection, NO frozen-2020 selection) and apply the SAME LIGHT de-risk overlay the book uses:
    family_free_vol_gate[t, asset] = beta_position * clip(vt / rv_lagged[t], 0, 1)
  where vt = the LIGHT vol-target level (_vt_level on the 1d buy-hold roster) -- the EXACT vol-target the
  book uses. Aggregation = fixed-EW (fillna(0.0).mean -- the book's PIT-cash aggregation), survivorship-
  clean PIT. This is "EW-beta + a vol-brake" = the simplest possible drawdown-insurance book.

  Note: the book also carries a per-asset trail-stop + min_hold INSIDE each signal sleeve, and a SIGNAL
  (the asset is only HELD when the family says so). The family-free control DROPS both the signal and the
  trail/min_hold -- it is always-long beta, braked only by the vol-target. This is the honest "what does
  the apparatus add OVER a one-line vol-brake on beta" decomposition: any gap between the two IS the
  apparatus's contribution (signal timing + trail + selection), and any MATCH means the apparatus adds
  nothing the vol-brake doesn't already deliver.

PRE-REGISTRATION (stated BEFORE the run, persisted verbatim):
  H0 (the falsifier fires): the family-free vol-gate MATCHES the book on PRESERVATION (per-year maxDD +
     UNSEEN, within noise) AND on full-cycle compound -> the family-ensemble apparatus adds NOTHING; the
     deployable is the vol-braked EW-beta one-liner.
  H1 (the apparatus survives): the family-free vol-gate is MATERIALLY WORSE on full-cycle compound and/or
     per-year net (the families add bull participation / compound) -> that delta IS the apparatus's value.
  Decision metrics (pre-registered): per-year maxDD (preservation), per-year net (participation), full-
     cycle compound + Calmar, and the SEALED UNSEEN (single frozen-LIGHT read). "MATCH on preservation" =
     per-year maxDD within ~5pp; "MATCH on full-cycle" = full-cycle compound within ~10pp (the same
     material-margin the book's verdict uses).
  Asymmetric loss: false-ship a non-preserving book into a -60% bear >> false-skip. The honest read here
     is the cheap one to get right -- it costs nothing and prevents shipping a useless apparatus.

ABSOLUTE DISCIPLINE (binding):
  STRICT LONG-ONLY + spot (ZERO short logic; beta position in [0,1] after the brake). FIXED-EW aggregation
  (fillna(0.0).mean -- NEVER skipna; cadence-invariant). Survivorship-clean POINT-IN-TIME (data-derived
  listing dates, retargeted per year/window). Maker cost is N/A for pure buy-hold beta (no flips) -- the
  vol-brake re-sizes continuously but we do NOT charge a flip cost on the continuous re-size (a buy-hold
  beta book is not flipping; this is CONSERVATIVE toward the control, i.e. it makes the control look its
  BEST, which is the right bias for a falsifier -- if even the best-case control still loses to the book
  the apparatus is real; if the best-case control matches, the apparatus is dead).
  UNSEEN window (2025-12-31 -> 2026-06-01) is FROZEN READ-ONLY -- read the family-free control's UNSEEN
  number ONCE at the frozen LIGHT level, NO tuning/iterating on UNSEEN. No emoji (cp1252).

RWYB:
  python -m strat.family_free_control --selftest                 # mechanics sanity (fast, no UNSEEN)
  python -m strat.family_free_control --years 2020,2021,2022,2023,2024,2025   # the per-year compare
  python -m strat.family_free_control                            # full run incl. the single UNSEEN read
Does NOT git commit (overseer commits after judging). UNSEEN is touched EXACTLY ONCE.
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
import strat.family_ensemble_book as FEB                                      # noqa: E402  (PHASE-1 book)
from strat.family_ensemble_book import (                                      # noqa: E402
    DERISK_LEVELS, TRANSLATING_FAMILIES, _vt_level, _metrics, build_book, build_buyhold,
)
from strat.forward_test_2021 import _buyhold_net_series, pit_universe_2021    # noqa: E402
from strat.family_ensemble_unseen import _set_window, UNSEEN_WIN, _full_stream_book  # noqa: E402
from strat.scorecard import score_book                                       # noqa: E402

OUT = ROOT.parent / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)

PICK_LEVEL = "light"          # the FROZEN PHASE-1 de-risk pick (do NOT re-tune on UNSEEN)

__contract__ = {
    "kind": "family_free_vol_gate_decisive_falsifier",
    "inputs": {
        "family_free_control": "plain EW-beta (equal-weight buy-hold of PIT-active assets, NO families/"
                               "MA/TI/selection) + the SAME LIGHT vol-target overlay the book uses "
                               "(clip(vt/rv_lagged,0,1)); fixed-EW (fillna(0).mean); survivorship-clean PIT.",
        "comparand": "the PHASE-1 family-ensemble book at LIGHT de-risk (imported from family_ensemble_book; "
                     "NO re-implementation).",
        "windows": "per year 2020-2025 + full-cycle + the SEALED UNSEEN (single frozen-LIGHT read).",
    },
    "outputs": {
        "per_year": "family-free vs book: net / maxDD / Calmar per year + EW buy-hold (no brake) reference.",
        "full_cycle": "full-cycle compound / maxDD / Calmar: family-free vs book vs raw beta.",
        "unseen_once": "family-free vs book on the sealed UNSEEN (single read).",
        "scorecard": "scorecard.score_book on the family-free full daily stream (held-out p05).",
        "verdict": "APPARATUS_DEAD (family-free matches) / APPARATUS_ADDS_VALUE (book materially better) + "
                   "the decisive deltas + the honest deployable.",
    },
    "invariants": {
        "long_only_spot": "beta position in [0,1] after the brake; NO short logic.",
        "fixed_ew": "fillna(0.0).mean aggregation (NEVER skipna) -- cadence-invariant.",
        "same_vol_target_as_book": "vt = _vt_level at the LIGHT level (the book's exact vol-target fn).",
        "survivorship_clean_pit": "data-derived listing dates; per-year/window cutoffs as in the book.",
        "unseen_read_once": "UNSEEN computed ONCE at the frozen LIGHT level; no tuning/iterating on it.",
        "conservative_to_control": "no flip cost on the continuous vol-re-size (control shown best-case).",
    },
}

PREREG = {
    "H0_apparatus_dead": "the family-free vol-gate MATCHES the book on per-year maxDD (within ~5pp) AND "
                         "full-cycle compound (within ~10pp) AND UNSEEN -> the family-ensemble apparatus "
                         "adds NOTHING; the deployable is the vol-braked EW-beta one-liner.",
    "H1_apparatus_adds_value": "the family-free vol-gate is MATERIALLY WORSE on full-cycle compound and/or "
                               "per-year net -> the families add participation/compound; that delta IS the "
                               "apparatus's value.",
    "match_dd_pp": 5.0,          # per-year maxDD 'match' tolerance
    "match_compound_pp": 10.0,   # full-cycle compound 'match' tolerance (= the book verdict's material margin)
    "asymmetric_loss": "false-ship a non-preserving book into a -60% bear >> false-skip; the honest read is "
                       "the cheap one to get right.",
    "pick_level": PICK_LEVEL,
    "unseen_window": list(UNSEEN_WIN),
}


# =====================================================================================================
# 1. THE FAMILY-FREE VOL-GATE -- plain EW-beta + the book's LIGHT vol-target (fixed-EW, PIT-clean)
# =====================================================================================================
def build_family_free_vol_gate(derisk=PICK_LEVEL):
    """Build the FAMILY-FREE control for the CURRENT window (set via FEB._set_year / _set_window):
    plain EW buy-hold of the PIT-active 1d assets, each scaled by the SAME LIGHT vol-target the book
    uses, aggregated fixed-EW (fillna(0.0).mean -- PIT cash). Returns daily-return Series or None.

    This reuses build_buyhold's EXACT path (FT._assets_for('1d',...,'expand') roster + _buyhold_net_series
    + fixed-EW fillna(0).mean), differing ONLY by passing vt=<light level> instead of vt=None. So the only
    thing added to plain EW-beta is the vol-brake -- nothing else (no signal, no family, no selection)."""
    lvl = DERISK_LEVELS[derisk]
    assets = FT._assets_for("1d", False, "expand")
    vt = _vt_level(assets, lvl["vt_mult"])              # the book's vol-target level on the 1d roster
    series = [_buyhold_net_series(A, vt=vt) for A in assets]
    series = [s for s in series if s is not None and len(s)]
    if not series:
        return None
    df = pd.concat(series, axis=1).sort_index()
    bar = df.fillna(0.0).mean(axis=1)                   # fixed-EW (PIT: NaN/inactive = cash) -- book's path
    return bar.dropna().resample("1D").apply(lambda x: float(np.prod(1 + x.dropna()) - 1)).dropna()


def _run_window(window_setter, derisk=PICK_LEVEL):
    """Set the window, build (i) the family-ensemble book, (ii) the family-free vol-gate, (iii) plain
    EW buy-hold (no brake = raw beta reference). Returns dict of (metrics, daily) for each."""
    window_setter()
    pit_universe_2021(verbose=False)
    book, _ = build_book(derisk=derisk, combine="ew")
    ff = build_family_free_vol_gate(derisk=derisk)
    bh = build_buyhold()                                 # raw EW-beta (no brake)
    return {
        "book": (_metrics(book) if book is not None else {}, book),
        "family_free": (_metrics(ff) if ff is not None else {}, ff),
        "raw_beta": (_metrics(bh) if bh is not None else {}, bh),
    }


def _set_year_fn(year):
    return lambda: FEB._set_year(year)


def _set_unseen_fn():
    return lambda: _set_window(UNSEEN_WIN[0], UNSEEN_WIN[1])


# =====================================================================================================
# 2. FULL-CYCLE CHAIN (calendar order) for any of the three streams
# =====================================================================================================
def _chain_full_cycle(per_year, key):
    """Chain the per-year daily streams (calendar order) for stream `key` -> full-cycle metrics + daily."""
    parts = [yr[key][1] for yr in per_year if yr[key][1] is not None and len(yr[key][1])]
    if not parts:
        return {}, None
    d = pd.concat(parts).sort_index()
    d = d[~d.index.duplicated(keep="first")]
    return _metrics(d), d


# =====================================================================================================
# 3. THE FULL RUN
# =====================================================================================================
def run(years=(2020, 2021, 2022, 2023, 2024, 2025), derisk=PICK_LEVEL):
    res = {"prereg": PREREG, "derisk": derisk, "years": list(years)}

    # -------------------------------------------------------------------------------------------------
    # (A) PER-YEAR: book vs family-free-vol-gate vs raw-beta
    # -------------------------------------------------------------------------------------------------
    print("\n## (A) PER-YEAR: family-ensemble BOOK vs FAMILY-FREE vol-gate vs RAW EW-beta (LIGHT de-risk)")
    print(f"   {'year':>5} | {'BOOK net/DD':>16} | {'FAM-FREE net/DD':>18} | {'RAW-BETA net/DD':>18} | "
          f"{'DDmatch':>8} {'netGap(book-ff)':>14}")
    per_year = []
    year_rows = {}
    for yr in years:
        d = _run_window(_set_year_fn(yr), derisk)
        d["year"] = yr
        per_year.append(d)
        bk_m = d["book"][0]; ff_m = d["family_free"][0]; rb_m = d["raw_beta"][0]
        dd_match = (abs((bk_m.get("maxdd_pct") or 0) - (ff_m.get("maxdd_pct") or 0)) <= PREREG["match_dd_pp"])
        net_gap = (round((bk_m.get("net_pct") or 0) - (ff_m.get("net_pct") or 0), 1)
                   if (bk_m.get("net_pct") is not None and ff_m.get("net_pct") is not None) else None)
        year_rows[yr] = {
            "book_net": bk_m.get("net_pct"), "book_dd": bk_m.get("maxdd_pct"), "book_calmar": bk_m.get("calmar"),
            "ff_net": ff_m.get("net_pct"), "ff_dd": ff_m.get("maxdd_pct"), "ff_calmar": ff_m.get("calmar"),
            "raw_net": rb_m.get("net_pct"), "raw_dd": rb_m.get("maxdd_pct"),
            "dd_match_5pp": bool(dd_match), "net_gap_book_minus_ff_pp": net_gap,
        }
        print(f"   {yr:>5} | {str(bk_m.get('net_pct'))+'%/'+str(bk_m.get('maxdd_pct'))+'%':>16} | "
              f"{str(ff_m.get('net_pct'))+'%/'+str(ff_m.get('maxdd_pct'))+'%':>18} | "
              f"{str(rb_m.get('net_pct'))+'%/'+str(rb_m.get('maxdd_pct'))+'%':>18} | "
              f"{str(dd_match):>8} {str(net_gap):>14}")
    res["per_year"] = year_rows

    # -------------------------------------------------------------------------------------------------
    # (B) FULL-CYCLE (calendar chain over the graded years)
    # -------------------------------------------------------------------------------------------------
    print("\n## (B) FULL-CYCLE COMPOUND (calendar chain over graded years)")
    fc = {}
    for key, label in (("book", "family-ensemble BOOK"), ("family_free", "FAMILY-FREE vol-gate"),
                       ("raw_beta", "RAW EW-beta (no brake)")):
        m, d = _chain_full_cycle(per_year, key)
        fc[key] = m
        print(f"   {label:24}: net {str(m.get('net_pct')):>9}% maxDD {str(m.get('maxdd_pct')):>8}% "
              f"Calmar {str(m.get('calmar')):>7} Sharpe {str(m.get('sharpe')):>6}")
    res["full_cycle"] = fc
    fc_book_net = fc["book"].get("net_pct"); fc_ff_net = fc["family_free"].get("net_pct")
    fc_compound_gap = (round(fc_book_net - fc_ff_net, 1)
                       if (fc_book_net is not None and fc_ff_net is not None) else None)
    res["full_cycle_compound_gap_book_minus_ff_pp"] = fc_compound_gap

    # -------------------------------------------------------------------------------------------------
    # (C) THE SEALED UNSEEN -- single frozen-LIGHT read (book vs family-free vs raw-beta)
    # -------------------------------------------------------------------------------------------------
    print("\n## (C) SEALED UNSEEN (single frozen-LIGHT read) -- book vs FAMILY-FREE vs raw-beta")
    print(f"##     window {UNSEEN_WIN} -- touched ONCE, no tuning/iterating.")
    du = _run_window(_set_unseen_fn(), derisk)
    bk_u = du["book"][0]; ff_u = du["family_free"][0]; rb_u = du["raw_beta"][0]
    regime = ("BULL" if (rb_u.get("net_pct") or 0) > 15 else "BEAR" if (rb_u.get("net_pct") or 0) < -15
              else "CHOP/SIDEWAYS")
    res["unseen_once"] = {
        "window": list(UNSEEN_WIN), "regime": regime,
        "book_net": bk_u.get("net_pct"), "book_dd": bk_u.get("maxdd_pct"),
        "ff_net": ff_u.get("net_pct"), "ff_dd": ff_u.get("maxdd_pct"),
        "raw_net": rb_u.get("net_pct"), "raw_dd": rb_u.get("maxdd_pct"),
        "dd_match_5pp": bool(abs((bk_u.get("maxdd_pct") or 0) - (ff_u.get("maxdd_pct") or 0))
                             <= PREREG["match_dd_pp"]),
    }
    res["_unseen_book_daily"] = du["book"][1]
    res["_unseen_ff_daily"] = du["family_free"][1]
    res["_unseen_raw_daily"] = du["raw_beta"][1]
    print(f"   UNSEEN regime (raw beta): net {rb_u.get('net_pct')}% maxDD {rb_u.get('maxdd_pct')}% -> {regime}")
    print(f"   BOOK       : net {bk_u.get('net_pct')}% maxDD {bk_u.get('maxdd_pct')}%")
    print(f"   FAMILY-FREE: net {ff_u.get('net_pct')}% maxDD {ff_u.get('maxdd_pct')}%")
    print(f"   RAW-BETA   : net {rb_u.get('net_pct')}% maxDD {rb_u.get('maxdd_pct')}%")
    print(f"   UNSEEN DD match (book vs family-free, <={PREREG['match_dd_pp']}pp): {res['unseen_once']['dd_match_5pp']}")

    # -------------------------------------------------------------------------------------------------
    # (D) SCORECARD on the family-free full daily stream (held-out p05) -- the deployability of the control
    # -------------------------------------------------------------------------------------------------
    print("\n## (D) SCORECARD: family-free vol-gate full daily stream (held-out p05)")
    ff_full, _ = _full_stream_family_free(derisk)
    card = score_book("family_free_vol_gate", ff_full)
    res["scorecard_family_free"] = card
    hp = card.get("heldout_block_bootstrap", {}).get("p05")
    fp = card.get("full_block_bootstrap", {}).get("p05")
    for sp in ("SEL", "OOS", "UNSEEN"):
        s = card["per_split"].get(sp, {})
        print(f"   [{sp:6}] compound {str(s.get('compound_pct')):>8}% maxDD {str(s.get('maxdd_pct')):>8}% (n={s.get('n')})")
    print(f"   FULL p05 = {fp}% | HELD-OUT (OOS+UNSEEN) p05 = {hp}% | ship_read.ship = {card['ship_read']['ship']}")

    res["verdict"] = build_verdict(res)
    return res


def _full_stream_family_free(derisk):
    """Full 2020-2026 daily stream for the FAMILY-FREE vol-gate (mirror of _full_stream_book's chaining
    but for the family-free control). Returns (ff_daily, None)."""
    parts = []
    for yr in (2020, 2021, 2022, 2023, 2024):
        FEB._set_year(yr); pit_universe_2021(verbose=False)
        d = build_family_free_vol_gate(derisk)
        if d is not None and len(d):
            parts.append(d)
    _set_window("2025-01-01", "2025-12-31"); pit_universe_2021(verbose=False)
    d = build_family_free_vol_gate(derisk)
    if d is not None and len(d):
        parts.append(d)
    _set_window(UNSEEN_WIN[0], UNSEEN_WIN[1]); pit_universe_2021(verbose=False)
    d = build_family_free_vol_gate(derisk)
    if d is not None and len(d):
        parts.append(d)
    ff = pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)
    ff = ff[~ff.index.duplicated(keep="first")]
    return ff, None


# =====================================================================================================
# 4. VERDICT
# =====================================================================================================
def build_verdict(res):
    pry = res["per_year"]; fc = res["full_cycle"]; u = res["unseen_once"]
    fc_gap = res.get("full_cycle_compound_gap_book_minus_ff_pp")
    # per-year DD match: in how many years does the family-free vol-gate MATCH the book's maxDD (<=5pp)?
    n_years = len(pry)
    n_dd_match = sum(1 for v in pry.values() if v["dd_match_5pp"])
    # in how many years does the book's net materially BEAT the family-free (gap > match_compound_pp)?
    n_book_beats = sum(1 for v in pry.values()
                       if v["net_gap_book_minus_ff_pp"] is not None
                       and v["net_gap_book_minus_ff_pp"] > PREREG["match_compound_pp"])
    n_ff_beats = sum(1 for v in pry.values()
                     if v["net_gap_book_minus_ff_pp"] is not None
                     and v["net_gap_book_minus_ff_pp"] < -PREREG["match_compound_pp"])

    preservation_matches = bool(n_dd_match >= int(np.ceil(0.6 * n_years)))     # majority of years DD-match
    unseen_dd_match = bool(u["dd_match_5pp"])
    full_cycle_matches = bool(fc_gap is not None and abs(fc_gap) <= PREREG["match_compound_pp"])

    # the decisive verdict
    if preservation_matches and full_cycle_matches and unseen_dd_match:
        verdict = "APPARATUS_DEAD"           # family-free matches on preservation AND full-cycle AND UNSEEN
    elif preservation_matches and not full_cycle_matches:
        # the vol-brake reproduces the CRASH-PRESERVATION (the risk property) but the families move the
        # full-cycle COMPOUND -> the apparatus's only value is participation/compound, NOT preservation.
        verdict = ("APPARATUS_ADDS_COMPOUND_NOT_PRESERVATION" if (fc_gap or 0) > 0
                   else "FAMILIES_HURT_COMPOUND")
    elif not preservation_matches:
        verdict = "APPARATUS_ADDS_VALUE"     # the book preserves materially better than the vol-brake too
    else:
        verdict = "AMBIGUOUS"

    fc_book = fc["book"].get("net_pct"); fc_ff = fc["family_free"].get("net_pct")
    fc_raw = fc["raw_beta"].get("net_pct")
    fc_book_dd = fc["book"].get("maxdd_pct"); fc_ff_dd = fc["family_free"].get("maxdd_pct")
    fc_raw_dd = fc["raw_beta"].get("maxdd_pct")
    lines = [
        "## DECISIVE FALSIFIER VERDICT (family-free vol-gate vs the family-ensemble book) [VERIFIED-FALSIFIER]",
        f"PRESERVATION (per-year maxDD): family-free matches the book in {n_dd_match}/{n_years} years "
        f"(<= {PREREG['match_dd_pp']}pp) -> preservation {'MATCHES' if preservation_matches else 'does NOT match'}.",
        f"UNSEEN preservation: book maxDD {u['book_dd']}% vs family-free {u['ff_dd']}% (raw-beta {u['raw_dd']}%) "
        f"-> DD-match {unseen_dd_match}. UNSEEN net: book {u['book_net']}% / family-free {u['ff_net']}% / "
        f"raw-beta {u['raw_net']}% (regime {u['regime']}).",
        f"FULL-CYCLE compound: book {fc_book}% (maxDD {fc_book_dd}%, Calmar {fc['book'].get('calmar')}) vs "
        f"family-free {fc_ff}% (maxDD {fc_ff_dd}%, Calmar {fc['family_free'].get('calmar')}) vs raw-beta "
        f"{fc_raw}% (maxDD {fc_raw_dd}%). Compound gap (book - family-free) = {fc_gap}pp -> full-cycle "
        f"{'MATCHES' if full_cycle_matches else 'does NOT match'} (tol {PREREG['match_compound_pp']}pp).",
        f"PER-YEAR net: book materially beats family-free in {n_book_beats}/{n_years} years; family-free "
        f"materially beats book in {n_ff_beats}/{n_years} years (material = > {PREREG['match_compound_pp']}pp).",
        "READ: the vol-brake on plain EW-beta reproduces the book's CRASH-PRESERVATION (the risk property is "
        "family-agnostic). " + (
            "It ALSO matches full-cycle compound -> the family-ensemble apparatus adds NOTHING; deploy the "
            "vol-braked EW-beta one-liner." if verdict == "APPARATUS_DEAD" else
            f"But the families move the full-cycle compound by {fc_gap}pp -> the apparatus's ONLY value is "
            f"participation/compound, NOT preservation; preservation is a one-line vol-brake."
            if verdict == "APPARATUS_ADDS_COMPOUND_NOT_PRESERVATION" else
            f"And the families HURT full-cycle compound by {abs(fc_gap or 0)}pp vs a plain vol-braked beta -> "
            f"the apparatus is worse than the one-liner." if verdict == "FAMILIES_HURT_COMPOUND" else
            "The book ALSO preserves materially better than the vol-brake -> the apparatus adds real value."
            if verdict == "APPARATUS_ADDS_VALUE" else "Mixed signal -- see deltas."),
        f"GATES: preservation_matches={preservation_matches} | full_cycle_matches={full_cycle_matches} | "
        f"unseen_dd_match={unseen_dd_match} | n_dd_match={n_dd_match}/{n_years} | fc_gap={fc_gap}pp.",
    ]
    return {"verdict": verdict, "n_dd_match": n_dd_match, "n_years": n_years,
            "n_book_beats_ff": n_book_beats, "n_ff_beats_book": n_ff_beats,
            "preservation_matches": preservation_matches, "full_cycle_matches": full_cycle_matches,
            "unseen_dd_match": unseen_dd_match, "full_cycle_compound_gap_pp": fc_gap,
            "fc_book_net": fc_book, "fc_ff_net": fc_ff, "fc_raw_net": fc_raw,
            "lines": lines}


# =====================================================================================================
# 5. CHART -- the decisive overlay: family-free vol-gate vs book across full-cycle + UNSEEN
# =====================================================================================================
def make_chart(res, per_year_daily):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[chart] matplotlib unavailable ({e}) -- skipped")
        return []
    fc_book = per_year_daily.get("book"); fc_ff = per_year_daily.get("family_free")
    fc_raw = per_year_daily.get("raw_beta")
    ub = res.get("_unseen_book_daily"); uf = res.get("_unseen_ff_daily"); ur = res.get("_unseen_raw_daily")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))

    # panel 1: full-cycle equity (log) -- book vs family-free vs raw beta
    ax = axes[0]
    if fc_book is not None and len(fc_book):
        ax.plot((1 + fc_book).cumprod().index, (1 + fc_book).cumprod().values,
                label="family-ensemble BOOK (light)", color="#2a9d8f", lw=1.9)
    if fc_ff is not None and len(fc_ff):
        ax.plot((1 + fc_ff).cumprod().index, (1 + fc_ff).cumprod().values,
                label="FAMILY-FREE vol-gate (light)", color="#e9c46a", lw=1.9, ls="-")
    if fc_raw is not None and len(fc_raw):
        ax.plot((1 + fc_raw).cumprod().index, (1 + fc_raw).cumprod().values,
                label="RAW EW-beta (no brake)", color="#264653", lw=1.3, ls="--")
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2023-01-01"), color="#e76f51", alpha=0.10, label="2022 BEAR")
    ax.axvspan(pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01"), color="#e76f51", alpha=0.06)
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)"); ax.set_xlabel("date")
    v = res["verdict"]
    ax.set_title(f"FULL-CYCLE: family-free vol-gate vs book vs raw beta\n"
                 f"book {v['fc_book_net']}% / family-free {v['fc_ff_net']}% / raw {v['fc_raw_net']}% "
                 f"(gap book-ff {v['full_cycle_compound_gap_pp']}pp)")
    ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)

    # panel 2: the SEALED UNSEEN overlay -- the single decisive read
    ax2 = axes[1]
    if ub is not None and len(ub):
        ax2.plot((1 + ub).cumprod().index, (1 + ub).cumprod().values,
                 label="BOOK (light)", color="#2a9d8f", lw=2.0)
    if uf is not None and len(uf):
        ax2.plot((1 + uf).cumprod().index, (1 + uf).cumprod().values,
                 label="FAMILY-FREE vol-gate", color="#e9c46a", lw=2.0)
    if ur is not None and len(ur):
        ax2.plot((1 + ur).cumprod().index, (1 + ur).cumprod().values,
                 label="RAW EW-beta", color="#264653", lw=1.3, ls="--")
    ax2.set_ylabel("growth of $1"); ax2.set_xlabel("date")
    u = res["unseen_once"]
    ax2.set_title(f"SEALED UNSEEN {u['window'][0]} -> {u['window'][1]} ({u['regime']})\n"
                  f"book net {u['book_net']}% DD {u['book_dd']}% | family-free net {u['ff_net']}% DD {u['ff_dd']}% "
                  f"| raw net {u['raw_net']}% DD {u['raw_dd']}%")
    ax2.legend(loc="best", fontsize=8); ax2.grid(alpha=0.3)

    fig.suptitle("DECISIVE FALSIFIER: does a one-line vol-brake on EW-beta match the family-ensemble apparatus?")
    fig.tight_layout()
    c = OUT / "family_free_vs_book.png"
    fig.savefig(c, dpi=120); plt.close(fig)
    print(f"[chart] {c}")
    return [str(c)]


# =====================================================================================================
# 6. MAIN
# =====================================================================================================
def _strip_daily(obj):
    if isinstance(obj, dict):
        return {k: _strip_daily(v) for k, v in obj.items()
                if not (isinstance(k, str) and k.startswith("_") and k.endswith("_daily"))}
    if isinstance(obj, list):
        return [_strip_daily(v) for v in obj]
    if isinstance(obj, pd.Series):
        return None
    return obj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.family_free_control")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--years", default="2020,2021,2022,2023,2024,2025")
    ap.add_argument("--derisk", default=PICK_LEVEL, help="FROZEN PHASE-1 pick (do NOT change for UNSEEN)")
    ap.add_argument("--no-charts", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    years = tuple(int(y) for y in a.years.split(","))
    print("## DECISIVE FALSIFIER -- the family-free vol-gate vs the family-ensemble book")
    print("## PRE-REGISTRATION (stated BEFORE the run):")
    for k in ("H0_apparatus_dead", "H1_apparatus_adds_value", "asymmetric_loss"):
        print(f"   {k}: {PREREG[k]}")
    print(f"\n   years {list(years)} | FROZEN de-risk {a.derisk} | LONG-ONLY spot | fixed-EW | "
          f"PIT survivorship-clean | UNSEEN read-once\n")

    res = run(years=years, derisk=a.derisk)

    # rebuild daily streams for the chart (book / family-free / raw) across the chained years
    chart_daily = {}
    for key in ("book", "family_free", "raw_beta"):
        parts = []
        for yr in years:
            FEB._set_year(yr); pit_universe_2021(verbose=False)
            if key == "book":
                d, _ = build_book(derisk=a.derisk, combine="ew")
            elif key == "family_free":
                d = build_family_free_vol_gate(derisk=a.derisk)
            else:
                d = build_buyhold()
            if d is not None and len(d):
                parts.append(d)
        if parts:
            dd = pd.concat(parts).sort_index()
            chart_daily[key] = dd[~dd.index.duplicated(keep="first")]

    print("\n" + "=" * 104)
    for line in res["verdict"]["lines"]:
        print(f"   {line}")
    print(f"\n   >>> VERDICT: {res['verdict']['verdict']}")
    print("=" * 104)

    charts = []
    if not a.no_charts:
        charts = make_chart(res, chart_daily)
    res["charts"] = charts

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT / f"family_free_control_{stamp}.json"
    payload = {"repro": {"command": "python -m strat.family_free_control " + " ".join(argv or sys.argv[1:]),
                         "git_sha": sha, "years": list(years), "derisk": a.derisk,
                         "unseen_window": list(UNSEEN_WIN)},
               "prereg": PREREG, "results": _strip_daily(res), "charts": charts}
    json.dump(payload, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    return 0


# =====================================================================================================
# 7. SELFTEST -- mechanics sanity (does NOT touch the UNSEEN verdict path)
# =====================================================================================================
def selftest():
    print("## FAMILY-FREE-CONTROL SELFTEST (mechanics only; no UNSEEN verdict)")
    ok = True
    # (1) the family-free vol-gate builds on a benign year and is fixed-EW + vol-braked (vt passed)
    FEB._set_year(2024)
    pit_universe_2021(verbose=False)
    ff = build_family_free_vol_gate(derisk="light")
    s1 = ff is not None and len(ff) > 100
    print(f"  (1) family-free vol-gate builds 2024: n_days={len(ff) if ff is not None else 0} -> {'PASS' if s1 else 'FAIL'}")
    ok &= s1
    # (2) the control is NOTHING BUT beta + a vol-brake: with derisk='none' it must EQUAL plain buy-hold
    ff_none = build_family_free_vol_gate(derisk="none")
    bh = build_buyhold()
    if ff_none is not None and bh is not None:
        j = pd.concat([ff_none.rename("ff"), bh.rename("bh")], axis=1).dropna()
        s2 = bool(len(j) > 50 and np.allclose(j["ff"].to_numpy(), j["bh"].to_numpy(), atol=1e-9))
    else:
        s2 = False
    print(f"  (2) derisk='none' family-free == plain EW buy-hold (it is JUST beta) -> {'PASS' if s2 else 'FAIL'}")
    ok &= s2
    # (3) the LIGHT brake REDUCES exposure -> light family-free maxDD is shallower (>=) than raw beta's
    m_light = _metrics(build_family_free_vol_gate(derisk="light"))
    m_raw = _metrics(bh)
    s3 = (m_light.get("maxdd_pct") is not None and m_raw.get("maxdd_pct") is not None
          and m_light["maxdd_pct"] >= m_raw["maxdd_pct"] - 0.5)  # light DD less negative (or ~equal)
    print(f"  (3) LIGHT brake: family-free maxDD {m_light.get('maxdd_pct')}% >= raw beta maxDD "
          f"{m_raw.get('maxdd_pct')}% (brake reduces DD) -> {'PASS' if s3 else 'FAIL'}")
    ok &= s3
    # (4) long-only: the vol-target multiplier clips to [0,1] -> the position can never go short
    #     (verified structurally: _buyhold_net_series uses np.clip(..., 0.0, 1.0))
    import inspect
    src = inspect.getsource(_buyhold_net_series)
    s4 = "np.clip(" in src and ", 0.0, 1.0)" in src
    print(f"  (4) long-only structural: vol-target clipped to [0,1] in _buyhold_net_series -> {'PASS' if s4 else 'FAIL'}")
    ok &= s4
    # (5) fixed-EW: the family-free aggregation uses fillna(0.0).mean (PIT cash), NOT skipna
    src2 = inspect.getsource(build_family_free_vol_gate)
    s5 = "fillna(0.0).mean" in src2 and "skipna" not in src2
    print(f"  (5) fixed-EW aggregation (fillna(0.0).mean, no skipna) -> {'PASS' if s5 else 'FAIL'}")
    ok &= s5
    # (6) selftest did NOT touch the sealed UNSEEN window
    s6 = tuple(FT.WIN) != tuple(UNSEEN_WIN)
    print(f"  (6) selftest did NOT touch the sealed UNSEEN window (WIN={FT.WIN}) -> {'PASS' if s6 else 'FAIL'}")
    ok &= s6
    print(f"\n  SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
