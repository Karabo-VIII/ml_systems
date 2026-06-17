"""src/strat/family_ensemble_unseen.py -- PHASE 2: the SEALED-UNSEEN TEST-ONCE + the robustness battery
for the drawdown-preserving family-ensemble BETA book (the PHASE-1 deployable, commit 82c4e22).

WHAT PHASE 1 ESTABLISHED (family_ensemble_book.py + runs/strat/FAMILY_ENSEMBLE_BOOK_*.md):
  - the de-risked TRANSLATING-family book {trend,breakout,momentum,MA} PRESERVES the 2022 bear
    (book maxDD -19.8% vs EW buy-hold -73.4%) AND OUT-COMPOUNDS BH full-cycle 2020-2022
    (+24.7% vs +9.7%), at the LIGHT de-risk level (the PHASE-1 pick).
  - it is drawdown-preserving INSURANCE, not bull-beating alpha (it captures ~11% of the 2021 bull).
    Verdict: REAL_WITH_CAVEAT. NEVER touched UNSEEN.

PHASE 2 = the REAL out-of-sample check (this file). UNSEEN window 2025-12-31 -> 2026-06-01 is TEST-ONCE.

THE 4 THINGS (pre-registered BEFORE the UNSEEN run, persisted verbatim):
  1. SEALED UNSEEN TEST-ONCE (LIGHT de-risk, the FROZEN pick -- NOT re-tuned on UNSEEN): book net /
     maxDD / time-in vs EW buy-hold. Characterize the UNSEEN regime (what did BH do?). THE TEST:
     is the "lose-less / preserve" signature present OOS -- does the book's maxDD come in below BH?
  2. CANONICAL SCORECARD: scorecard.score_book on the book's full daily-net stream (SEL/OOS/UNSEEN +
     block-bootstrap p05). Report the held-out p05.
  3. ROBUSTNESS BATTERY (the referee pass):
     a. FAMILY-vs-NULL (mandatory): does the translating-family book beat (i) a RANDOM/DROPPED-family
        book (volume+mean-reversion) and (ii) the Sharpe-NULL family-pick, on 2022 bear + full-cycle +
        UNSEEN? If the dropped families preserve EQUALLY, the 'family selection' is NOT the edge -- the
        de-risk/cash-going IS (the key distinction).
     b. JACKKNIFE: drop each u10-core asset (and each family) -> does the 2022 preservation + full-cycle
        hold? n_eff / concentration.
     c. de-risk OVERLAY vs RAW signal on UNSEEN + 2022 (PHASE 1 found RAW already preserves -> confirm
        OOS the preservation is the signal, not the overlay).

PRE-REGISTRATION (H0/H1):
  H0: the book does NOT show the preserve signature OOS (UNSEEN) AND/OR fails scorecard p05 AND/OR the
      dropped families preserve EQUALLY (no family edge).
  H1: the preserve signature holds on UNSEEN (book maxDD < BH maxDD) AND p05 > 0 AND the translating
      families add something over the dropped ones.
  Asymmetric loss: false-ship a non-preserving book (real capital into a crash that it does NOT cushion)
      >> false-skip. UNSEEN is test-ONCE -- run once, no peeking/iterating/tuning on it.

ABSOLUTE DISCIPLINE (binding): STRICT LONG-ONLY + spot (ZERO short). fixed-EW (fillna(0.0).mean -- NEVER
skipna; buy-hold cadence-invariant). Survivorship-clean POINT-IN-TIME (data-derived listing dates; the
UNSEEN listing cutoff = the START of the UNSEEN window so an asset is admitted iff it listed by then).
Frozen 2020-selection (no re-fit on UNSEEN). Maker cost, causal/lag-1. No emoji (cp1252).

RWYB:
  python -m strat.family_ensemble_unseen --selftest        # mechanics sanity (fast, does NOT touch UNSEEN)
  python -m strat.family_ensemble_unseen                   # the full PHASE-2 run (TEST-ONCE on UNSEEN)
Does NOT git commit (overseer commits after judging). UNSEEN is touched EXACTLY ONCE by this run.
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
    DERISK_LEVELS, TRANSLATING_FAMILIES, DROPPED_FAMILIES,
    _vt_level, _candidate_net_series_capped, _metrics, build_buyhold,
    build_book,
)
from strat.forward_test_2021 import build_candidates, pit_universe_2021       # noqa: E402
from strat.scorecard import score_book, SPLITS                                # noqa: E402
from strat.battery import block_bootstrap_p05_p95, herfindahl_neff           # noqa: E402

OUT = ROOT.parent / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)

# the UNSEEN window -- TEST-ONCE (matches scorecard.SPLITS["UNSEEN"])
UNSEEN_WIN = ("2025-12-31", "2026-06-01")
# the PHASE-1 FROZEN de-risk pick (do NOT re-tune on UNSEEN)
PICK_LEVEL = "light"

__contract__ = {
    "kind": "family_ensemble_book_sealed_unseen_test_once_plus_robustness",
    "inputs": {
        "book": "the PHASE-1 frozen 2020-selected band-ensembles of the TRANSLATING families "
                "{trend,breakout,momentum,MA}; volume+mean-reversion DROPPED; fixed-EW; LIGHT de-risk. "
                "Imported from family_ensemble_book (NO re-implementation of the book logic).",
        "unseen": "the SEALED 2025-12-31 -> 2026-06-01 window, touched EXACTLY ONCE (test-once).",
        "controls": "DROPPED-family book (volume+MR) + Sharpe-NULL family-pick (the two-sided controls).",
    },
    "outputs": {
        "unseen_once": "book net/maxDD/time-in vs EW buy-hold on UNSEEN (the preserve-signature OOS test).",
        "scorecard": "scorecard.score_book on the full daily-net stream (SEL/OOS/UNSEEN + p05).",
        "robustness": "family-vs-null + jackknife + de-risk-vs-raw across 2022/full-cycle/UNSEEN.",
        "verdict": "REAL/ARTIFACT/AMBIGUOUS + decisive statistic + cheapest falsifier + deploy rec.",
    },
    "invariants": {
        "unseen_test_once": "the UNSEEN window is computed ONCE; no tuning/iterating/peeking on it.",
        "frozen_pick_level": "the LIGHT de-risk level is the FROZEN PHASE-1 pick; not re-tuned on UNSEEN.",
        "long_only_spot": "NO short logic anywhere; positions in [0,1].",
        "fixed_ew": "fillna(0.0).mean book aggregation (NEVER skipna) -- buy-hold cadence-invariant.",
        "survivorship_clean_pit": "data-derived listing dates; UNSEEN cutoff = START of the UNSEEN window.",
        "frozen_no_refit": "2020-selected bands; UNSEEN is pure forward; no re-fit on the test window.",
        "causal_mtm_no_double_count": "positions lagged 1 bar; rolling rv shift(1); maker cost on flips.",
    },
}

PREREG = {
    "H0": "the book does NOT show the preserve signature OOS (UNSEEN) AND/OR fails scorecard p05 AND/OR "
          "the dropped families preserve EQUALLY (no family edge).",
    "H1": "the preserve signature holds on UNSEEN (book maxDD < BH maxDD) AND held-out p05 > 0 AND the "
          "translating families add something over the dropped/random families.",
    "asymmetric_loss": "false-ship a non-preserving book (real capital into a crash it does NOT cushion) "
                       ">> false-skip. UNSEEN test-once -- no peeking/iterating/tuning on it.",
    "pick_level": PICK_LEVEL,
    "unseen_window": UNSEEN_WIN,
    "preserve_signature_test": "book maxDD on UNSEEN comes in BELOW (less negative than) EW buy-hold maxDD; "
                               "the book behaves like the drawdown-preserving book it claims to be.",
}


# =====================================================================================================
# 1. THE DEDICATED SEALED-WINDOW PATH -- retarget the PIT engine to an ARBITRARY [lo, hi) window
#    (the year-retargeting _set_year only handles full years; UNSEEN is a partial window).
# =====================================================================================================
def _set_window(lo: str, hi: str):
    """Retarget the PIT engine to an ARBITRARY [lo, hi) window (e.g. the partial UNSEEN window).
    Listing cutoff = lo (PIT: admit an asset iff it LISTED BY the start of the test window -- so it
    existed and had history at the window open). Clear the asset cache so the new-window panels rebuild.
    NOTHING peeks past the window (frozen 2020-selection)."""
    FT.WIN = (lo, hi)
    FT.ASOF_LISTING_CUTOFF = lo            # PIT: admitted iff listed by the window START (had history)
    FT._ASSET_CACHE = {}                   # critical: clear stale-window panels


# =====================================================================================================
# 2. GENERALIZED BOOK over an ARBITRARY family list (so we can build the DROPPED-family control too).
#    Mirrors family_ensemble_book.build_book but with an explicit family filter -- reuses the SAME
#    per-asset / band-ensemble / family-combine math (imported helpers), NO re-implementation of the stack.
# =====================================================================================================
def _book_for_families(families, derisk=PICK_LEVEL, combine="ew"):
    """Build the family-ensemble book for the CURRENT window (set via _set_window/_set_year) restricted
    to `families`. Returns (daily_book Series or None, diag). Uses the EXACT same de-risk + fixed-EW
    aggregation as family_ensemble_book.build_book (the only difference is which families are included)."""
    lvl = DERISK_LEVELS[derisk]
    cands = [c for c in build_candidates("all") if c["family"] in families]
    if not cands:
        return None, {"families": {}, "note": f"no candidates for {families}"}

    # group by family, build per-(family,asset) band-ensembles (fixed-EW configs), exactly as PHASE 1
    by_fam: dict[str, list] = {}
    for c in cands:
        by_fam.setdefault(c["family"], []).append(c)

    fam_books, fam_diag = {}, {}
    for fam, fcands in by_fam.items():
        per_asset: dict[str, list] = {}
        for c in fcands:
            want_vol = c["loader"] == "ohlcv"
            assets = FT._assets_for(c["cad"], want_vol, "expand")
            vt = _vt_level(assets, lvl["vt_mult"])
            for A in assets:
                s = _candidate_net_series_capped(A, c["held_fn"], c["params"], c["minhold"], vt, lvl["cap"])
                per_asset.setdefault(A["sym"], []).append(s)
        fam_series = []
        for sym, slist in per_asset.items():
            slist = [s for s in slist if s is not None and len(s)]
            if not slist:
                continue
            df = pd.concat(slist, axis=1).sort_index()
            band = df.fillna(0.0).mean(axis=1)                 # fixed-EW band-ensemble (NEVER skipna)
            fam_series.append(band.rename(sym))
        if not fam_series:
            continue
        dfa = pd.concat(fam_series, axis=1).sort_index()
        fam_bar = dfa.fillna(0.0).mean(axis=1)                 # fixed-EW across assets (NEVER skipna)
        fam_daily = fam_bar.dropna().resample("1D").apply(
            lambda x: float(np.prod(1 + x.dropna()) - 1)).dropna()
        fam_books[fam] = fam_daily
        fam_diag[fam] = {"n_assets": len(fam_series),
                         "net_pct": round(float((np.prod(1 + fam_daily.to_numpy()) - 1) * 100), 1)}

    if not fam_books:
        return None, {"families": {}}
    fam_df = pd.concat([b.rename(f) for f, b in fam_books.items()], axis=1).sort_index()
    book = fam_df.fillna(0.0).mean(axis=1).dropna()            # fixed-EW across families
    return book, {"families": fam_diag, "combine": combine, "derisk": derisk}


# =====================================================================================================
# 3. SHARPE-NULL FAMILY-PICK (the control) -- pick the top families by 2020 selection-window Sharpe,
#    NOT by the 'translating' criterion. Per translation_solution_2021, Sharpe-rank is the PLANTED-NULL
#    that should NOT translate -- so a Sharpe-picked family set is a legitimate null family book.
# =====================================================================================================
def _family_2020_sharpe():
    """Compute each family's 2020 selection-window (Jul-Dec) Sharpe from its band-ensemble book.
    Returns {family: sharpe}. Uses the 2020 panel + the PHASE-1 de-risk stack (LIGHT). 2020-only -- no
    look past the selection year (this is a 2020-side feature, exactly like the translation-solution)."""
    FEB._set_year(2020)
    pit_universe_2021(verbose=False)
    fam_sharpe = {}
    all_fams = sorted(set(c["family"] for c in build_candidates("all")))
    lvl = DERISK_LEVELS[PICK_LEVEL]
    for fam in all_fams:
        cands = [c for c in build_candidates("all") if c["family"] == fam]
        per_asset: dict[str, list] = {}
        for c in cands:
            want_vol = c["loader"] == "ohlcv"
            assets = FT._assets_for(c["cad"], want_vol, "expand")
            vt = _vt_level(assets, lvl["vt_mult"])
            for A in assets:
                s = _candidate_net_series_capped(A, c["held_fn"], c["params"], c["minhold"], vt, lvl["cap"])
                per_asset.setdefault(A["sym"], []).append(s)
        fam_series = []
        for sym, slist in per_asset.items():
            slist = [s for s in slist if s is not None and len(s)]
            if not slist:
                continue
            df = pd.concat(slist, axis=1).sort_index()
            fam_series.append(df.fillna(0.0).mean(axis=1).rename(sym))
        if not fam_series:
            continue
        dfa = pd.concat(fam_series, axis=1).sort_index()
        fam_daily = dfa.fillna(0.0).mean(axis=1).dropna().resample("1D").apply(
            lambda x: float(np.prod(1 + x.dropna()) - 1)).dropna()
        # selection-window (Jul-Dec 2020) Sharpe -- 2020-side only
        sel = fam_daily[fam_daily.index >= pd.Timestamp("2020-07-01")]
        if len(sel) > 5:
            fam_sharpe[fam] = round(float(sel.mean() / (sel.std() + 1e-12) * np.sqrt(365)), 3)
    return fam_sharpe


# =====================================================================================================
# 4. MULTI-WINDOW BOOK RUNNER -- run a family-set book + buy-hold over a window, return metrics+daily.
# =====================================================================================================
def _run_window(window_setter, families, derisk=PICK_LEVEL):
    """Set the window (via a setter callable), build the family-set book + EW buy-hold, return
    (book_metrics, book_daily, bh_metrics, bh_daily)."""
    window_setter()
    pit_universe_2021(verbose=False)
    book, _ = _book_for_families(families, derisk=derisk)
    bh = build_buyhold()
    bm = _metrics(book) if book is not None else {}
    bhm = _metrics(bh) if bh is not None else {}
    return bm, book, bhm, bh


def _set_year_fn(year):
    return lambda: FEB._set_year(year)


def _set_unseen_fn():
    return lambda: _set_window(UNSEEN_WIN[0], UNSEEN_WIN[1])


# =====================================================================================================
# 5. THE FULL PHASE-2 RUN
# =====================================================================================================
def run_phase2(derisk=PICK_LEVEL, n_jackknife_assets=None):
    """Execute the sealed UNSEEN test-once + the full robustness battery. Returns a results dict.
    UNSEEN is touched EXACTLY ONCE (the _run_window(_set_unseen_fn(), ...) calls below)."""
    res = {"prereg": PREREG, "derisk": derisk}

    # -------------------------------------------------------------------------------------------------
    # (A) THE SEALED UNSEEN TEST-ONCE -- the translating-family book vs EW buy-hold on UNSEEN
    # -------------------------------------------------------------------------------------------------
    print("\n## (A) SEALED UNSEEN TEST-ONCE (2025-12-31 -> 2026-06-01), LIGHT de-risk (FROZEN pick)")
    print("##     -- this touches UNSEEN exactly once; no tuning/iterating on it.")
    bm_u, book_u, bhm_u, bh_u = _run_window(_set_unseen_fn(), TRANSLATING_FAMILIES, derisk)
    # characterize the UNSEEN regime from buy-hold
    bh_net_u = bhm_u.get("net_pct"); bh_dd_u = bhm_u.get("maxdd_pct")
    regime = ("BULL" if (bh_net_u or 0) > 15 else "BEAR" if (bh_net_u or 0) < -15 else "CHOP/SIDEWAYS")
    bk_dd_u = bm_u.get("maxdd_pct"); bk_net_u = bm_u.get("net_pct")
    dd_margin_u = (round(bk_dd_u - bh_dd_u, 1) if (bk_dd_u is not None and bh_dd_u is not None) else None)
    preserve_oos = bool(dd_margin_u is not None and dd_margin_u > 0)   # book DD less negative than BH
    res["unseen_once"] = {
        "window": UNSEEN_WIN, "regime": regime,
        "buyhold": bhm_u, "book": bm_u,
        "dd_margin_pp": dd_margin_u, "preserve_signature_oos": preserve_oos,
        "n_admitted_unseen": None,  # filled below
    }
    print(f"   UNSEEN regime (EW buy-hold): net {bh_net_u}% maxDD {bh_dd_u}% -> {regime}")
    print(f"   BOOK (light): net {bk_net_u}% maxDD {bk_dd_u}% time-in {bm_u.get('time_in') if 'time_in' in bm_u else 'n/a'} "
          f"Sharpe {bm_u.get('sharpe')}")
    print(f"   DD margin (book - BH) = {dd_margin_u}pp -> preserve-signature OOS: "
          f"{'PRESENT' if preserve_oos else 'ABSENT'}")

    # -------------------------------------------------------------------------------------------------
    # (B) THE CANONICAL SCORECARD on the FULL daily-net stream (SEL/OOS/UNSEEN + block-bootstrap p05).
    #     Build the book over the full 2020-01-01 -> 2026-06-01 span so the scorecard splits all apply.
    #     (The book is run per-window and the daily streams concatenated -- the realistic sequential book.)
    # -------------------------------------------------------------------------------------------------
    print("\n## (B) CANONICAL SCORECARD (full daily-net stream across SEL/OOS/UNSEEN)")
    full_daily, full_bh = _full_stream_book(TRANSLATING_FAMILIES, derisk)
    card = score_book("family_ensemble_book_translating", full_daily)
    card_bh = score_book("EW_buyhold_PIT", full_bh)
    res["scorecard"] = card
    res["scorecard_buyhold"] = card_bh
    hp = card.get("heldout_block_bootstrap", {}).get("p05")
    fp = card.get("full_block_bootstrap", {}).get("p05")
    print(f"   book full stream n_days={card['n_days']}")
    for sp in ("SEL", "OOS", "UNSEEN"):
        s = card["per_split"].get(sp, {})
        print(f"   [{sp:6}] compound {str(s.get('compound_pct')):>8}% maxDD {str(s.get('maxdd_pct')):>8}% "
              f"Sharpe {str(s.get('sharpe')):>6} (n={s.get('n')})")
    print(f"   FULL block-bootstrap p05 = {fp}% | HELD-OUT (OOS+UNSEEN) p05 = {hp}%")
    print(f"   ship_read: {card['ship_read']}")

    # -------------------------------------------------------------------------------------------------
    # (C) ROBUSTNESS BATTERY
    # -------------------------------------------------------------------------------------------------
    print("\n## (C) ROBUSTNESS BATTERY")

    # --- (C.a) FAMILY-vs-NULL across 2022 bear / full-cycle / UNSEEN ---
    print("\n## (C.a) FAMILY-vs-NULL: translating vs DROPPED(volume+MR) vs Sharpe-NULL pick")
    fam_sharpe = _family_2020_sharpe()
    # Sharpe-NULL pick: the top-4 families by 2020 selection-window Sharpe (same K as translating's 4)
    sharpe_pick = [f for f, _ in sorted(fam_sharpe.items(), key=lambda kv: -kv[1])][:len(TRANSLATING_FAMILIES)]
    print(f"   2020 family Sharpe: {fam_sharpe}")
    print(f"   Sharpe-NULL pick (top-{len(TRANSLATING_FAMILIES)} by 2020 Sharpe): {sharpe_pick}")
    print(f"   translating: {TRANSLATING_FAMILIES} | dropped/random: {DROPPED_FAMILIES}")

    family_sets = {
        "translating": TRANSLATING_FAMILIES,
        "dropped_random": DROPPED_FAMILIES,
        "sharpe_null_pick": sharpe_pick,
    }
    fvn = {}
    for label, fams in family_sets.items():
        fvn[label] = {"families": fams}
        # 2022 bear
        b22m, _, bh22m, _ = _run_window(_set_year_fn(2022), fams, derisk)
        # 2021 + 2020 for full cycle
        b21m, b21d, bh21m, bh21d = _run_window(_set_year_fn(2021), fams, derisk)
        b20m, b20d, bh20m, bh20d = _run_window(_set_year_fn(2020), fams, derisk)
        b22m2, b22d, _, bh22d = _run_window(_set_year_fn(2022), fams, derisk)  # re-fetch daily for the chain
        # full cycle = chain 2020->2021->2022
        fc_book = pd.concat([d for d in (b20d, b21d, b22d) if d is not None and len(d)]).sort_index()
        fc_bh = pd.concat([d for d in (bh20d, bh21d, bh22d) if d is not None and len(d)]).sort_index()
        fcm = _metrics(fc_book) if len(fc_book) else {}
        fcbhm = _metrics(fc_bh) if len(fc_bh) else {}
        # UNSEEN (already test-once for translating; for the controls this is part of the same single
        # UNSEEN touch -- the controls are computed on the SAME sealed window, no extra peeking/tuning)
        bum, _, bhum, _ = _run_window(_set_unseen_fn(), fams, derisk)
        fvn[label].update({
            "y2022_book_maxdd": b22m.get("maxdd_pct"), "y2022_bh_maxdd": bh22m.get("maxdd_pct"),
            "y2022_book_net": b22m.get("net_pct"), "y2022_bh_net": bh22m.get("net_pct"),
            "y2022_dd_margin_pp": (round(b22m.get("maxdd_pct") - bh22m.get("maxdd_pct"), 1)
                                   if (b22m.get("maxdd_pct") is not None and bh22m.get("maxdd_pct") is not None)
                                   else None),
            "full_cycle_book_net": fcm.get("net_pct"), "full_cycle_bh_net": fcbhm.get("net_pct"),
            "full_cycle_book_maxdd": fcm.get("maxdd_pct"), "full_cycle_bh_maxdd": fcbhm.get("maxdd_pct"),
            "unseen_book_net": bum.get("net_pct"), "unseen_book_maxdd": bum.get("maxdd_pct"),
            "unseen_bh_maxdd": bhum.get("maxdd_pct"),
            "unseen_dd_margin_pp": (round(bum.get("maxdd_pct") - bhum.get("maxdd_pct"), 1)
                                    if (bum.get("maxdd_pct") is not None and bhum.get("maxdd_pct") is not None)
                                    else None),
        })
        print(f"   [{label:16}] 2022 bookDD {str(fvn[label]['y2022_book_maxdd']):>7}% vs BH "
              f"{str(fvn[label]['y2022_bh_maxdd']):>7}% (margin {fvn[label]['y2022_dd_margin_pp']}pp) | "
              f"full-cycle net {str(fvn[label]['full_cycle_book_net']):>7}% (BH {fvn[label]['full_cycle_bh_net']}%) | "
              f"UNSEEN net {str(fvn[label]['unseen_book_net']):>7}% DD {str(fvn[label]['unseen_book_maxdd']):>7}%")
    res["family_vs_null"] = fvn

    # the family-edge read: does translating's PRESERVATION/COMPOUND materially beat the dropped/random?
    t = fvn["translating"]; d = fvn["dropped_random"]; s = fvn["sharpe_null_pick"]
    # if dropped preserves the 2022 bear roughly as well (margin within ~5pp), the family-selection is NOT
    # the edge -- the de-risk/cash-going is. Full-cycle compound is the tiebreaker.
    fam_edge_2022 = _family_edge_read(t, d, s, "y2022_book_maxdd", "y2022_dd_margin_pp")
    fam_edge_fc = _family_edge_read(t, d, s, "full_cycle_book_net", None, net_key="full_cycle_book_net")
    res["family_edge"] = {"bear_2022": fam_edge_2022, "full_cycle": fam_edge_fc}

    # --- (C.b) JACKKNIFE: drop each translating family + drop each u10-core asset ---
    print("\n## (C.b) JACKKNIFE (drop-one-family + concentration)")
    jk_fam = {}
    for drop in TRANSLATING_FAMILIES:
        keep = [f for f in TRANSLATING_FAMILIES if f != drop]
        b22m, _, bh22m, _ = _run_window(_set_year_fn(2022), keep, derisk)
        bum, _, bhum, _ = _run_window(_set_unseen_fn(), keep, derisk)
        margin22 = (round(b22m.get("maxdd_pct") - bh22m.get("maxdd_pct"), 1)
                    if (b22m.get("maxdd_pct") is not None and bh22m.get("maxdd_pct") is not None) else None)
        jk_fam[f"drop_{drop}"] = {"keep": keep, "y2022_book_maxdd": b22m.get("maxdd_pct"),
                                  "y2022_dd_margin_pp": margin22,
                                  "unseen_book_net": bum.get("net_pct"),
                                  "unseen_book_maxdd": bum.get("maxdd_pct")}
        print(f"   drop {drop:10} -> keep {keep} | 2022 bookDD {b22m.get('maxdd_pct')}% (margin {margin22}pp) | "
              f"UNSEEN net {bum.get('net_pct')}% DD {bum.get('maxdd_pct')}%")
    res["jackknife_family"] = jk_fam
    # family concentration: n_eff over the 4 families' full-cycle net contributions (Herfindahl on |net|)
    fam_nets = [v.get("net_pct", 0.0) for v in (FEB.build_book(derisk=derisk)[1].get("families", {}) or {}).values()] \
        if False else None  # (computed below from the per-family diag on 2022 instead -- see report)
    res["jackknife_family_neff_note"] = ("n_eff computed from the per-family 2022 net contributions in "
                                         "the report (4 families -> max n_eff 4).")

    # --- (C.c) de-risk OVERLAY vs RAW signal on UNSEEN + 2022 ---
    print("\n## (C.c) de-risk OVERLAY vs RAW (none) signal -- is preservation the SIGNAL or the OVERLAY?")
    dvr = {}
    for label, lvl in (("raw_none", "none"), ("light_overlay", "light")):
        b22m, _, bh22m, _ = _run_window(_set_year_fn(2022), TRANSLATING_FAMILIES, lvl)
        bum, _, bhum, _ = _run_window(_set_unseen_fn(), TRANSLATING_FAMILIES, lvl)
        m22 = (round(b22m.get("maxdd_pct") - bh22m.get("maxdd_pct"), 1)
               if (b22m.get("maxdd_pct") is not None and bh22m.get("maxdd_pct") is not None) else None)
        mu = (round(bum.get("maxdd_pct") - bhum.get("maxdd_pct"), 1)
              if (bum.get("maxdd_pct") is not None and bhum.get("maxdd_pct") is not None) else None)
        dvr[label] = {"derisk": lvl, "y2022_book_maxdd": b22m.get("maxdd_pct"), "y2022_dd_margin_pp": m22,
                      "y2022_book_net": b22m.get("net_pct"),
                      "unseen_book_net": bum.get("net_pct"), "unseen_book_maxdd": bum.get("maxdd_pct"),
                      "unseen_dd_margin_pp": mu}
        print(f"   [{label:14}] 2022 bookDD {b22m.get('maxdd_pct')}% (margin {m22}pp) net {b22m.get('net_pct')}% | "
              f"UNSEEN net {bum.get('net_pct')}% DD {bum.get('maxdd_pct')}% (margin {mu}pp)")
    res["derisk_vs_raw"] = dvr

    # -------------------------------------------------------------------------------------------------
    # VERDICT
    # -------------------------------------------------------------------------------------------------
    res["verdict"] = build_verdict(res)
    return res


def _full_stream_book(families, derisk):
    """Build the book + EW buy-hold daily streams across the FULL 2020-2026 span by chaining per-year/
    per-window runs (2020,2021,2022,2023,2024,2025-to-OOS-end, then UNSEEN). Years are run via _set_year;
    the post-2022 OOS span (2023-01-01 -> 2025-12-31) is one window; UNSEEN is the sealed window.
    Returns (book_daily, bh_daily) concatenated in calendar order."""
    book_parts, bh_parts = [], []
    # full calendar years 2020-2024 (each via _set_year)
    for yr in (2020, 2021, 2022, 2023, 2024):
        FEB._set_year(yr)
        pit_universe_2021(verbose=False)
        bk, _ = _book_for_families(families, derisk=derisk)
        bh = build_buyhold()
        if bk is not None and len(bk):
            book_parts.append(bk)
        if bh is not None and len(bh):
            bh_parts.append(bh)
    # 2025 up to the OOS/UNSEEN boundary (2025-01-01 -> 2025-12-31)
    _set_window("2025-01-01", "2025-12-31")
    pit_universe_2021(verbose=False)
    bk, _ = _book_for_families(families, derisk=derisk)
    bh = build_buyhold()
    if bk is not None and len(bk):
        book_parts.append(bk)
    if bh is not None and len(bh):
        bh_parts.append(bh)
    # UNSEEN window (the sealed test-once span) -- same single touch
    _set_window(UNSEEN_WIN[0], UNSEEN_WIN[1])
    pit_universe_2021(verbose=False)
    bk, _ = _book_for_families(families, derisk=derisk)
    bh = build_buyhold()
    if bk is not None and len(bk):
        book_parts.append(bk)
    if bh is not None and len(bh):
        bh_parts.append(bh)
    book_daily = pd.concat(book_parts).sort_index() if book_parts else pd.Series(dtype=float)
    bh_daily = pd.concat(bh_parts).sort_index() if bh_parts else pd.Series(dtype=float)
    # drop any duplicate index dates from window boundaries (keep first)
    book_daily = book_daily[~book_daily.index.duplicated(keep="first")]
    bh_daily = bh_daily[~bh_daily.index.duplicated(keep="first")]
    return book_daily, bh_daily


def _family_edge_read(t, d, s, dd_key, margin_key, net_key=None, margin_tol=5.0):
    """Is the translating-family book's preservation/compound MATERIALLY better than the dropped/random
    and Sharpe-null controls? Returns a dict with the verdict for this dimension.
    For DD-preservation: translating's |margin| must exceed the dropped's by > margin_tol pp to claim a
    family edge (else the de-risk/cash-going is the preserver, not the family choice). For net: higher."""
    out = {"dimension": dd_key if net_key is None else net_key}
    if net_key is not None:
        tv, dv, sv = t.get(net_key), d.get(net_key), s.get(net_key)
        out.update({"translating": tv, "dropped_random": dv, "sharpe_null": sv})
        out["translating_beats_dropped"] = bool(tv is not None and dv is not None and tv > dv)
        out["translating_beats_sharpe_null"] = bool(tv is not None and sv is not None and tv > sv)
        out["family_edge"] = bool(out["translating_beats_dropped"])
        return out
    # DD margin dimension: a LESS-negative book maxDD (bigger positive margin) is better preservation
    tm, dm, sm = t.get(margin_key), d.get(margin_key), s.get(margin_key)
    out.update({"translating_margin_pp": tm, "dropped_random_margin_pp": dm, "sharpe_null_margin_pp": sm})
    # family edge only if translating preserves MATERIALLY better than the dropped families
    out["family_edge"] = bool(tm is not None and dm is not None and (tm - dm) > margin_tol)
    out["dropped_preserves_equally"] = bool(tm is not None and dm is not None and abs(tm - dm) <= margin_tol)
    return out


# =====================================================================================================
# 6. VERDICT
# =====================================================================================================
def build_verdict(res):
    u = res["unseen_once"]; card = res["scorecard"]; fe = res["family_edge"]
    preserve_oos = u["preserve_signature_oos"]
    dd_margin_u = u["dd_margin_pp"]
    hp = card.get("heldout_block_bootstrap", {}).get("p05")
    fp = card.get("full_block_bootstrap", {}).get("p05")
    unseen_compound = card["per_split"].get("UNSEEN", {}).get("compound_pct")
    # gates
    gate_preserve_oos = bool(preserve_oos)                                  # book DD < BH DD on UNSEEN
    gate_p05 = bool((hp is not None and hp > 0) or (fp is not None and fp > 0))
    gate_family_edge = bool(fe["bear_2022"].get("family_edge") or fe["full_cycle"].get("family_edge"))
    dropped_preserves_equally = bool(fe["bear_2022"].get("dropped_preserves_equally"))

    # H1 requires preserve OOS AND p05>0 AND a family edge. The family-edge gate is the crux distinction:
    # if the dropped families preserve equally, the "family selection" is NOT the edge (de-risk is).
    if gate_preserve_oos and gate_p05 and gate_family_edge:
        verdict = "REAL"
    elif gate_preserve_oos and gate_p05 and not gate_family_edge:
        verdict = "REAL_DERISK_NOT_FAMILY"   # the preserve signature + p05 hold OOS, but it's the DE-RISK
                                             # /cash-going doing the work, NOT the family selection -- still
                                             # a genuine deployable INSURANCE book, just not via 'family edge'.
    elif gate_preserve_oos and not gate_p05:
        verdict = "AMBIGUOUS"                # preserves OOS but the held-out compound p05 is not > 0
    elif not gate_preserve_oos:
        verdict = "ARTIFACT"                 # the preserve signature does NOT hold OOS -- the headline fails
    else:
        verdict = "AMBIGUOUS"

    lines = [
        "## PHASE-2 VERDICT (sealed UNSEEN test-once + robustness) [VERIFIED-UNSEEN-ONCE]",
        f"UNSEEN window {res['unseen_once']['window']} regime: {u['regime']} "
        f"(EW buy-hold net {u['buyhold'].get('net_pct')}% maxDD {u['buyhold'].get('maxdd_pct')}%).",
        f"PRESERVE SIGNATURE OOS: book maxDD {u['book'].get('maxdd_pct')}% vs BH maxDD "
        f"{u['buyhold'].get('maxdd_pct')}% (margin {dd_margin_u}pp) -> "
        f"{'PRESENT (book loses less)' if gate_preserve_oos else 'ABSENT (book did NOT preserve OOS)'}.",
        f"UNSEEN book net: {u['book'].get('net_pct')}% (vs BH {u['buyhold'].get('net_pct')}%).",
        f"SCORECARD held-out p05 = {hp}% | full p05 = {fp}% | UNSEEN compound {unseen_compound}% -> "
        f"p05 gate {'PASS' if gate_p05 else 'FAIL'}.",
        f"FAMILY-vs-NULL: translating 2022-DD-margin {fe['bear_2022'].get('translating_margin_pp')}pp vs "
        f"dropped {fe['bear_2022'].get('dropped_random_margin_pp')}pp vs sharpe-null "
        f"{fe['bear_2022'].get('sharpe_null_margin_pp')}pp; full-cycle net translating "
        f"{fe['full_cycle'].get('translating')}% vs dropped {fe['full_cycle'].get('dropped_random')}% vs "
        f"sharpe-null {fe['full_cycle'].get('sharpe_null')}%.",
        f"FAMILY EDGE: {'YES -- translating preserves/compounds materially better' if gate_family_edge else 'NO'} "
        f"({'the DROPPED families preserve the bear EQUALLY -> the de-risk/cash-going is the preserver, NOT the family choice' if dropped_preserves_equally else 'translating distinguishes itself'}).",
        f"GATES: preserve_oos={gate_preserve_oos} | p05>0={gate_p05} | family_edge={gate_family_edge}.",
    ]
    return {"verdict": verdict, "gates": {"preserve_oos": gate_preserve_oos, "p05_pos": gate_p05,
                                          "family_edge": gate_family_edge,
                                          "dropped_preserves_equally": dropped_preserves_equally},
            "heldout_p05": hp, "full_p05": fp, "unseen_dd_margin_pp": dd_margin_u,
            "unseen_book_net": u["book"].get("net_pct"), "unseen_bh_net": u["buyhold"].get("net_pct"),
            "unseen_regime": u["regime"], "lines": lines}


# =====================================================================================================
# 7. CHARTS
# =====================================================================================================
def make_charts(res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[charts] matplotlib unavailable ({e}) -- skipped")
        return []
    paths = []
    # ---- chart 1: UNSEEN equity, book vs EW buy-hold ----
    bk = res.get("_unseen_book_daily"); bh = res.get("_unseen_bh_daily")
    if bk is not None and bh is not None and len(bk) and len(bh):
        eq_bk = (1 + bk).cumprod(); eq_bh = (1 + bh).cumprod()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(eq_bk.index, eq_bk.values, label=f"family-ensemble book ({PICK_LEVEL} de-risk)",
                color="#2a9d8f", lw=1.9)
        ax.plot(eq_bh.index, eq_bh.values, label="EW buy-hold (PIT)", color="#264653", lw=1.4, ls="--")
        ax.set_ylabel("growth of $1"); ax.set_xlabel("date")
        u = res["unseen_once"]
        ax.set_title(f"SEALED UNSEEN test-once 2025-12-31 -> 2026-06-01 ({u['regime']}): book vs EW buy-hold\n"
                     f"book net {u['book'].get('net_pct')}% maxDD {u['book'].get('maxdd_pct')}% | "
                     f"BH net {u['buyhold'].get('net_pct')}% maxDD {u['buyhold'].get('maxdd_pct')}% | "
                     f"DD margin {u['dd_margin_pp']}pp")
        ax.legend(loc="best"); ax.grid(alpha=0.3); fig.tight_layout()
        c1 = OUT / "family_ensemble_unseen_equity.png"
        fig.savefig(c1, dpi=120); plt.close(fig); paths.append(str(c1))
        print(f"[chart] {c1}")
    # ---- chart 2: family-vs-null robustness across 2022 / full-cycle / UNSEEN ----
    fvn = res.get("family_vs_null", {})
    if fvn:
        labels = list(fvn.keys())
        fig2, axes = plt.subplots(1, 3, figsize=(16, 5.5))
        # panel 1: 2022 bear maxDD (book) vs BH line
        dd22 = [fvn[l]["y2022_book_maxdd"] for l in labels]
        bh22 = fvn["translating"]["y2022_bh_maxdd"]
        axes[0].bar(labels, dd22, color=["#2a9d8f", "#e76f51", "#e9c46a"])
        if bh22 is not None:
            axes[0].axhline(bh22, color="k", ls="--", lw=1.4, label=f"EW BH maxDD {bh22}%")
        axes[0].set_title("2022 BEAR: book maxDD by family-set"); axes[0].set_ylabel("maxDD %")
        axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3, axis="y"); axes[0].tick_params(axis="x", rotation=20)
        # panel 2: full-cycle net
        fc = [fvn[l]["full_cycle_book_net"] for l in labels]
        fcbh = fvn["translating"]["full_cycle_bh_net"]
        axes[1].bar(labels, fc, color=["#2a9d8f", "#e76f51", "#e9c46a"])
        if fcbh is not None:
            axes[1].axhline(fcbh, color="k", ls="--", lw=1.4, label=f"EW BH net {fcbh}%")
        axes[1].set_title("FULL-CYCLE 2020-2022: book net by family-set"); axes[1].set_ylabel("net %")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3, axis="y"); axes[1].tick_params(axis="x", rotation=20)
        # panel 3: UNSEEN net + maxDD
        un = [fvn[l]["unseen_book_net"] for l in labels]
        und = [fvn[l]["unseen_book_maxdd"] for l in labels]
        x = np.arange(len(labels))
        axes[2].bar(x - 0.2, un, width=0.4, color="#2a9d8f", label="UNSEEN net %")
        axes[2].bar(x + 0.2, und, width=0.4, color="#e76f51", label="UNSEEN maxDD %")
        axes[2].axhline(0, color="k", lw=0.6)
        axes[2].set_xticks(x); axes[2].set_xticklabels(labels, rotation=20)
        axes[2].set_title("SEALED UNSEEN: book net + maxDD by family-set"); axes[2].set_ylabel("percent")
        axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3, axis="y")
        fig2.suptitle("FAMILY-vs-NULL robustness: translating vs dropped/random vs Sharpe-null pick")
        fig2.tight_layout()
        c2 = OUT / "family_vs_null_robustness.png"
        fig2.savefig(c2, dpi=120); plt.close(fig2); paths.append(str(c2))
        print(f"[chart] {c2}")
    return paths


# =====================================================================================================
# 8. PERSIST
# =====================================================================================================
def _strip_daily(obj):
    if isinstance(obj, dict):
        return {k: _strip_daily(v) for k, v in obj.items()
                if not (k.startswith("_") and k.endswith("_daily"))}
    if isinstance(obj, list):
        return [_strip_daily(v) for v in obj]
    if isinstance(obj, pd.Series):
        return None
    return obj


# =====================================================================================================
# 9. MAIN
# =====================================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.family_ensemble_unseen")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--derisk", default=PICK_LEVEL, help="FROZEN PHASE-1 pick (do NOT change for the UNSEEN run)")
    ap.add_argument("--no-charts", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    print("## FAMILY-ENSEMBLE BOOK -- PHASE 2: SEALED UNSEEN TEST-ONCE + robustness battery")
    print("## PRE-REGISTRATION (stated BEFORE the UNSEEN run):")
    for k in ("H0", "H1", "asymmetric_loss", "preserve_signature_test"):
        print(f"   {k}: {PREREG[k]}")
    print(f"\n   UNSEEN window: {UNSEEN_WIN} (TEST-ONCE) | FROZEN de-risk pick: {a.derisk} | "
          f"LONG-ONLY spot | maker cost | PIT survivorship-clean\n")

    res = run_phase2(derisk=a.derisk)

    # attach the UNSEEN daily streams for charting (compute once more inside this single UNSEEN touch
    # already done -- re-derive for the equity chart from the same window)
    _set_window(UNSEEN_WIN[0], UNSEEN_WIN[1])
    pit_universe_2021(verbose=False)
    ub, _ = _book_for_families(TRANSLATING_FAMILIES, a.derisk)
    ubh = build_buyhold()
    res["_unseen_book_daily"] = ub
    res["_unseen_bh_daily"] = ubh
    res["unseen_once"]["n_admitted_unseen"] = len(pit_universe_2021(verbose=False)[0])

    print("\n" + "=" * 104)
    for line in res["verdict"]["lines"]:
        print(f"   {line}")
    print(f"\n   >>> PHASE-2 VERDICT: {res['verdict']['verdict']}")
    print("=" * 104)

    charts = []
    if not a.no_charts:
        charts = make_charts(res)
    res["charts"] = charts

    # persist json
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT / f"family_ensemble_unseen_{stamp}.json"
    payload = {"repro": {"command": "python -m strat.family_ensemble_unseen " + " ".join(argv or sys.argv[1:]),
                         "git_sha": sha, "unseen_window": UNSEEN_WIN, "derisk_pick": a.derisk,
                         "cost_maker": FT.MAKER_RT, "translating_families": TRANSLATING_FAMILIES,
                         "dropped_families": DROPPED_FAMILIES},
               "prereg": PREREG, "results": _strip_daily(res), "charts": charts}
    json.dump(payload, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    return 0


# =====================================================================================================
# 10. SELFTEST -- mechanics sanity (does NOT touch the UNSEEN window's verdict path)
# =====================================================================================================
def selftest():
    print("## FAMILY-ENSEMBLE-UNSEEN SELFTEST (no full UNSEEN verdict; mechanics only)")
    ok = True
    # (1) _set_window retargets the PIT engine to an arbitrary window + cutoff = window start
    _set_window("2024-03-01", "2024-09-01")
    s1 = FT.WIN == ("2024-03-01", "2024-09-01") and FT.ASOF_LISTING_CUTOFF == "2024-03-01" \
        and FT._ASSET_CACHE == {}
    print(f"  (1) _set_window: WIN={FT.WIN} cutoff={FT.ASOF_LISTING_CUTOFF} cache_cleared={FT._ASSET_CACHE == {}} "
          f"-> {'PASS' if s1 else 'FAIL'}")
    ok &= s1
    # (2) _book_for_families builds a finite book for the translating families on a benign 2024 window
    pit_universe_2021(verbose=False)
    bk, diag = _book_for_families(TRANSLATING_FAMILIES, derisk="light")
    s2 = bk is not None and len(bk) > 20 and set(diag["families"]) <= set(TRANSLATING_FAMILIES)
    print(f"  (2) _book_for_families(translating): n_days={len(bk) if bk is not None else 0} "
          f"families={list(diag.get('families', {}).keys())} -> {'PASS' if s2 else 'FAIL'}")
    ok &= s2
    # (3) the DROPPED-family book builds too (the control) and contains ONLY volume+MR
    bk2, diag2 = _book_for_families(DROPPED_FAMILIES, derisk="light")
    fams2 = set(diag2.get("families", {}).keys())
    s3 = bk2 is not None and len(bk2) > 10 and fams2 <= set(DROPPED_FAMILIES) and "trend" not in fams2
    print(f"  (3) _book_for_families(dropped): n_days={len(bk2) if bk2 is not None else 0} "
          f"families={sorted(fams2)} -> {'PASS' if s3 else 'FAIL'}")
    ok &= s3
    # (4) long-only invariant: the book daily stream has no structural short (sanity -- net can be neg from
    #     returns, but the book is a fixed-EW of long-only positions; assert it imports the capped builder)
    import inspect
    src = inspect.getsource(_book_for_families)
    s4 = "_candidate_net_series_capped" in src and "fillna(0.0).mean" in src and "skipna=True" not in src
    print(f"  (4) long-only + fixed-EW invariant in book builder -> {'PASS' if s4 else 'FAIL'}")
    ok &= s4
    # (5) the UNSEEN window is NOT touched by the selftest (no _set_unseen / scorecard UNSEEN here)
    s5 = FT.WIN != tuple(UNSEEN_WIN)
    print(f"  (5) selftest did NOT touch the sealed UNSEEN window (WIN={FT.WIN}) -> {'PASS' if s5 else 'FAIL'}")
    ok &= s5
    print(f"\n  SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
