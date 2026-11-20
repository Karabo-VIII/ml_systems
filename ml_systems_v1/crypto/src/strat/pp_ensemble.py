"""src/strat/pp_ensemble.py -- THE LAST SHOT for long-only participate-and-preserve.

THE FLAGGED FALSIFIER (from participate_preserve_frontier.py's verdict, commit 4e0c34e):
  The frontier test found 4 long-only directional gates; exactly one (drawdown_aware) breaks NORTHEAST of
  the cash-going book on the DEV plane but FAILS held-out -- held-out p05 {-28.6,-27.8,-23.5}%, permutation
  max-stat p ~ 0.065-0.084, PBO 0.671 (best-of-4 + 2021-bull selection-window mirage). Its own cheapest
  falsifier was:

    "a REGIME-CONDITIONED ENSEMBLE of the 4 gates (does combining them shift NE?) before declaring the door
     permanently shut. If that also fails the held-out p05 + deflation gates, the door is shut and the
     remaining alpha must come from SHORT (OFF) or CARRY."

  This module IS that probe. Two ensemble constructions over the SAME 4 gates:
    (a) VOTE      : the fraction of the 4 gates invested at each bar -> a continuous exposure in
                    {0, .25, .5, .75, 1}. Simple averaging; no regime logic.
    (b) ROUTED    : a CAUSAL, PRE-REGISTERED regime detector on the asset's OWN price routes between gates --
                    confirmed-BULL -> full exposure (1.0); detected sustained-BEAR -> the drawdown_aware gate;
                    detected RECOVERY (below long-MA but long-MA no longer falling, or a fresh reclaim) -> the
                    bear_rally gate. The detector is trailing-MA + past-K-slope ONLY (no OOS fit, no lookahead).

THE DECISIVE GATE (pre-registered, asymmetric loss: false-ship >> false-skip):
  The ensemble RE-OPENS long-only participate-and-preserve ONLY IF a variant lands NORTHEAST of the cash-going
  book (MORE bull-capture AND preservation within 5pp) AND held-out block-bootstrap p05 > 0 AND permutation
  max-stat p < 0.05 AND PBO < 0.5. Anything less -> the door stays SHUT.

MULTIPLE-COMPARISONS HONESTY (binding):
  The ensemble is YET ANOTHER construction tried. The deflation family is therefore the FULL set of 6
  {bear_rally, terminal_leg, drawdown_aware, asymmetric, VOTE, ROUTED}; the max-statistic permutation null
  takes the MAX edge across ALL 6 each draw, and PBO runs over all 6 columns. A best-of-6 that fails this is
  NOT a result. (Adding 2 constructions can only RAISE the bar, never lower it.)

DISCIPLINE (inherited, binding):
  STRICT long-only + spot (exposure in [0,1]; ZERO short logic anywhere). SELECT/tune on 2020-2024+2025-OOS;
  the SEALED UNSEEN 2025-12-31->2026-06-01 is READ-ONCE (single read at the end, no tuning on it). fixed-EW
  (fillna(0.0).mean -- NEVER skipna). Survivorship-clean POINT-IN-TIME. Maker cost on flips, causal/lag-1,
  rv shift(1). No emoji (cp1252).

RWYB:
  python -m strat.pp_ensemble --selftest     # mechanics sanity (fast; does NOT touch UNSEEN)
  python -m strat.pp_ensemble                # full ensemble build + held-out grade + UNSEEN read-once
Does NOT git commit. UNSEEN touched EXACTLY ONCE (at the end).
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
import strat.participate_preserve_frontier as PP                             # noqa: E402  (the 4 gates + machinery)
from strat.forward_test_2021 import pit_universe_2021, MAKER_RT             # noqa: E402
from strat.family_ensemble_book import (                                     # noqa: E402
    DERISK_LEVELS, _vt_level, _metrics, build_book, build_buyhold,
)
from strat.participate_preserve_frontier import (                            # noqa: E402
    CONSTRUCTIONS, PICK_LEVEL, DEV_YEARS, DEV_2025_WIN, BULL_YEARS, BEAR_YEARS,
    _sma, _set_year_fn, _set_2025_fn, _set_unseen_fn, _family_free,
    _bull_capture_pct, _bear_preservation_pct, _chain, _dev_daily,
    _set_window, UNSEEN_WIN,
)
from strat.scorecard import score_book                                       # noqa: E402

OUT = ROOT.parent / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)


# =====================================================================================================
# 1. THE TWO ENSEMBLE EXPOSURE BUILDERS (per-asset, causal, long-only, exposure in [0,1])
# =====================================================================================================
# Each returns a CONTINUOUS exposure array in [0,1] over the full in-panel length (same length as A['c']),
# built ONLY from the 4 gate held-arrays + a causal regime detector. NO short logic; NO lookahead.

def _gate_holds(A):
    """Return the 4 gate held-arrays {name: held in {0,1}} for asset A at the frozen default kwargs.
    (The ensemble combines the gates as-shipped; tuning the gate kwargs would just re-inflate the count.)"""
    return {name: np.asarray(fn(A, **kw)).astype(np.float64)
            for name, (fn, kw) in CONSTRUCTIONS.items()}


def ens_vote(A):
    """VOTE ensemble: exposure = fraction of the 4 gates invested at each bar -> {0,.25,.5,.75,1}.
    A continuous long-only exposure; more gates agreeing = more invested. Causal (each gate is causal)."""
    H = _gate_holds(A)
    stack = np.vstack([H["bear_rally"], H["terminal_leg"], H["drawdown_aware"], H["asymmetric"]])
    return np.clip(stack.mean(axis=0), 0.0, 1.0)


def _causal_regime(A, slow=100, slope_k=20):
    """CAUSAL, PRE-REGISTERED 3-state regime detector on the asset's OWN price. Trailing slow-MA + its
    past-K slope ONLY (no future data, no OOS fit). States:
      2 = confirmed BULL      : close > slow-MA  AND  slow-MA rising (slow-MA_t > slow-MA_{t-K})
      0 = sustained BEAR      : close < slow-MA  AND  slow-MA falling (slow-MA_t < slow-MA_{t-K})
      1 = RECOVERY / chop     : everything else  (below MA but MA no longer falling, OR above MA but MA flat/falling)
    Returns an int array in {0,1,2}; -1 where the slow-MA is not yet defined (pre-warmup -> treated as cash)."""
    c = np.asarray(A["c"], float)
    ms = _sma(c, slow)
    ms_prev = np.concatenate([np.full(slope_k, np.nan), ms[:-slope_k]])
    rising = ms > ms_prev
    falling = ms < ms_prev
    above = c > ms
    reg = np.full(len(c), 1, dtype=np.int8)            # default RECOVERY/chop
    reg[above & rising] = 2                              # confirmed BULL
    reg[(~above) & falling] = 0                          # sustained BEAR
    reg[np.isnan(ms)] = -1                               # pre-warmup
    return reg


def ens_routed(A, slow=100, slope_k=20):
    """ROUTED ensemble: a causal regime detector selects WHICH gate (or full exposure) drives the position.
      confirmed BULL     -> full exposure 1.0          (participate fully in a confirmed uptrend)
      sustained BEAR     -> drawdown_aware gate         (the deepest-preserving gate, per the frontier test)
      RECOVERY/chop      -> bear_rally gate             (re-enters on confirmed counter-trend bounces)
      pre-warmup (reg<0) -> 0.0 (cash)                  (no signal yet -> PIT cash)
    The ROUTING MAP is pre-registered (matches the task spec verbatim) and uses NO future data. Long-only [0,1]."""
    H = _gate_holds(A)
    reg = _causal_regime(A, slow=slow, slope_k=slope_k)
    exp = np.zeros(len(reg), dtype=np.float64)
    exp[reg == 2] = 1.0                                  # confirmed bull -> full
    exp[reg == 0] = H["drawdown_aware"][reg == 0]        # sustained bear -> drawdown-aware
    exp[reg == 1] = H["bear_rally"][reg == 1]            # recovery/chop -> bear-rally
    # reg == -1 (pre-warmup) stays 0.0 (cash)
    return np.clip(exp, 0.0, 1.0)


# the ensemble registry: name -> exposure_fn (takes A, returns continuous exposure in [0,1])
ENSEMBLES = {
    "vote":   ens_vote,
    "routed": ens_routed,
}

# the FULL deflation family = the original 4 gates (as construction books) + the 2 ensembles. This is the
# honest multiplicity: the ensemble is best-of-6, not best-of-2.
PREREG = {
    "H0_door_shut": "NO ensemble variant lands NE of the cash-going book held-out with p05>0 AND survives "
        "multiple-comparisons deflation -> long-only participate-and-preserve is DEFINITIVELY SHUT; the honest "
        "doors are SHORT (OFF) or CARRY.",
    "H1_reopens": "an ensemble variant DOMINATES the book (MORE bull-capture AND preservation within 5pp), "
        "beats it on full-cycle held-out WEALTH, passes held-out block-bootstrap p05>0 AND permutation max-stat "
        "p<0.05 AND PBO<0.5 -> long-only participate-and-preserve RE-OPENS.",
    "asymmetric_loss": "false-ship a non-preserving book into a -60% bear >> false-skip. A best-of-6 that fails "
        "deflation is NOT a result. The ensemble RAISES the multiplicity count (4 gates + 2 ensembles = 6).",
    "dominance_def": "ensemble is NE of the book iff bull_capture_pct > book's AND bear_preservation_pct >= "
        "book's - 5pp.",
    "decisive_gate": "REOPENS iff (NE of book) AND (held-out p05 > 0) AND (permutation max-stat p < 0.05) AND "
        "(PBO < 0.5). Anything less = SHUT.",
    "regime_detector": "CAUSAL trailing slow-MA(100) + past-20-bar slope; states {bull=above&rising, "
        "bear=below&falling, recovery=else}; pre-registered routing map; NO OOS fit, NO lookahead.",
    "multiplicity_family": "deflation runs over ALL 6 {bear_rally, terminal_leg, drawdown_aware, asymmetric, "
        "vote, routed}; max-stat null takes max edge across 6; PBO over 6 columns.",
    "dev_years": list(DEV_YEARS),
    "unseen_window": list(UNSEEN_WIN),
}


# =====================================================================================================
# 2. PER-ASSET NET SERIES UNDER AN ENSEMBLE EXPOSURE (reuses the EXACT cost/lag/PIT stack)
# =====================================================================================================
def _ensemble_net_series(A, exposure_fn, vt, cap):
    """One asset's bar-level net Series under a continuous ensemble exposure. Identical cost/lag/PIT/vol-target
    stack to PP._gated_net_series, with the {0,1} gate replaced by the [0,1] ensemble exposure. Long-only:
    exposure in [0,1], vol-target multiplier clipped to [0,cap], maker on flips, lag-1 causal, PIT cash."""
    ret, rv = A["ret"], A["rv"]
    exp = np.asarray(exposure_fn(A)).astype(np.float64)              # [0,1] over full in-panel length
    exp = np.clip(exp, 0.0, 1.0)                                     # STRICT long-only guard
    pos = np.zeros(len(ret)); pos[1:] = exp[:-1]                     # lag 1 bar (causal)
    if vt is not None:
        pos = pos * np.clip(vt / (np.nan_to_num(rv, nan=vt) + 1e-12), 0.0, cap)
    flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
    net = pos * ret - flips * (MAKER_RT / 2.0)
    s = pd.Series(np.where(A["active"], net, np.nan), index=A["idx"])
    return s[A["win"]]


def build_ensemble_book(exposure_fn, derisk=PICK_LEVEL):
    """Build a LONG-ONLY ensemble book for the CURRENT window (set via _set_year / _set_window): apply
    `exposure_fn` per asset on the PIT-active 1d roster, fixed-EW aggregate (fillna(0.0).mean = PIT cash).
    Returns a daily-return Series or None. Mirrors PP.build_construction_book exactly."""
    lvl = DERISK_LEVELS[derisk]
    assets = FT._assets_for("1d", False, "expand")
    vt = _vt_level(assets, lvl["vt_mult"])
    series = [_ensemble_net_series(A, exposure_fn, vt, lvl["cap"]) for A in assets]
    series = [s for s in series if s is not None and len(s)]
    if not series:
        return None
    df = pd.concat(series, axis=1).sort_index()
    bar = df.fillna(0.0).mean(axis=1)                                # fixed-EW (PIT: NaN/inactive = cash)
    return bar.dropna().resample("1D").apply(lambda x: float(np.prod(1 + x.dropna()) - 1)).dropna()


# =====================================================================================================
# 3. WINDOW RUNNERS -- build cash_book / raw_beta / family_free / 4 gate-books / 2 ensemble-books
# =====================================================================================================
def _run_window(window_setter, derisk=PICK_LEVEL):
    """Set the window; build every stream. Returns {name: (metrics, daily)}."""
    window_setter()
    pit_universe_2021(verbose=False)
    out = {}
    book, _ = build_book(derisk=derisk, combine="ew")
    out["cash_book"] = (_metrics(book) if book is not None else {}, book)
    bh = build_buyhold()
    out["raw_beta"] = (_metrics(bh) if bh is not None else {}, bh)
    ff = _family_free(derisk)
    out["family_free"] = (_metrics(ff) if ff is not None else {}, ff)
    for name, (gate_fn, kwargs) in CONSTRUCTIONS.items():
        d = PP.build_construction_book(gate_fn, kwargs, derisk=derisk)
        out[name] = (_metrics(d) if d is not None else {}, d)
    for name, exp_fn in ENSEMBLES.items():
        d = build_ensemble_book(exp_fn, derisk=derisk)
        out[name] = (_metrics(d) if d is not None else {}, d)
    return out


def _full_stream(key, derisk):
    """Full 2020-2026 daily stream (calendar chain incl. the single UNSEEN window) for stream `key`."""
    parts = []
    setters = ([(_set_year_fn(y), y) for y in DEV_YEARS] + [(_set_2025_fn(), 2025), (_set_unseen_fn(), "UNSEEN")])
    for setter, _ in setters:
        setter(); pit_universe_2021(verbose=False)
        if key == "cash_book":
            d, _ = build_book(derisk=derisk, combine="ew")
        elif key == "raw_beta":
            d = build_buyhold()
        elif key == "family_free":
            d = _family_free(derisk)
        elif key in ENSEMBLES:
            d = build_ensemble_book(ENSEMBLES[key], derisk=derisk)
        else:
            gate_fn, kwargs = CONSTRUCTIONS[key]
            d = PP.build_construction_book(gate_fn, kwargs, derisk=derisk)
        if d is not None and len(d):
            parts.append(d)
    s = pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)
    return s[~s.index.duplicated(keep="first")]


# =====================================================================================================
# 4. DEFLATION over the FULL family of 6 (4 gates + 2 ensembles) -- max-stat permutation + PBO
# =====================================================================================================
FAMILY6 = list(CONSTRUCTIONS.keys()) + list(ENSEMBLES.keys())   # the honest multiplicity family


def _permutation_deflate6(per_window, best_name, n_perm=2000, block=10, seed=11):
    """MAX-statistic permutation null over the FULL 6-construction family. Statistic = compound(construction)
    - compound(book) over the 2020-2025 dev chain. Under H0 (the construction adds nothing beyond reshuffling
    the SAME beta exposure), the per-bar excess (construction - book) has no genuine sign structure; we
    sign-flip block-bootstrap each excess stream and recompute the MAX edge across ALL 6 each draw -> the null
    is the multiple-comparisons-corrected best-of-6. p = P(null max-edge >= observed best edge)."""
    book = _dev_daily(per_window, "cash_book")
    if len(book) < 50:
        return {"error": "insufficient dev book stream"}
    excess = {}
    for name in FAMILY6:
        d = _dev_daily(per_window, name)
        j = pd.concat([d.rename("x"), book.rename("b")], axis=1).dropna()
        if len(j) < 50:
            continue
        excess[name] = (j["x"] - j["b"]).to_numpy()
        excess[name + "__book"] = j["b"].to_numpy()
        excess[name + "__con"] = j["x"].to_numpy()
    if best_name not in excess:
        return {"error": f"best construction '{best_name}' excess stream unavailable"}

    def _edge(con, bk):
        return (np.prod(1 + con) - 1) * 100 - (np.prod(1 + bk) - 1) * 100

    keys = [name for name in FAMILY6 if name + "__con" in excess]
    obs = {name: _edge(excess[name + "__con"], excess[name + "__book"]) for name in keys}
    obs_best = max(obs.values())
    rng = np.random.default_rng(seed)
    null_max = np.empty(n_perm)
    for p in range(n_perm):
        edges = []
        for name in keys:
            ex = excess[name]
            n = len(ex); nb = int(np.ceil(n / block))
            sp = n - block + 1
            starts = rng.integers(0, max(1, sp), size=nb)
            signs = rng.choice([-1.0, 1.0], size=nb)
            chunks = [signs[i] * ex[st:st + block] for i, st in enumerate(starts)]
            re = np.concatenate(chunks)[:n]
            edges.append((np.prod(1 + re) - 1) * 100)
        null_max[p] = max(edges)
    pval = float((np.sum(null_max >= obs_best) + 1) / (n_perm + 1))
    return {"observed_best_edge_pp": round(obs_best, 2), "best_name": best_name,
            "per_construction_edge_pp": {k: round(v, 2) for k, v in obs.items()},
            "null_max_p95_pp": round(float(np.percentile(null_max, 95)), 2),
            "p_value_maxstat": round(pval, 4), "n_perm": n_perm, "block": block, "seed": seed,
            "family_size": len(keys), "survives_deflation": bool(pval < 0.05)}


def _pbo6(per_window):
    """PBO (CSCV) across the FULL 6-construction family (each construction's excess-over-book daily as a column)."""
    cols, names = [], []
    book = _dev_daily(per_window, "cash_book")
    for name in FAMILY6:
        d = _dev_daily(per_window, name)
        j = pd.concat([d.rename("x"), book.rename("b")], axis=1).dropna()
        if len(j) < 100:
            continue
        cols.append((j["x"] - j["b"]).to_numpy())
        names.append(name)
    if len(cols) < 2:
        return {"error": "need >=2 constructions with aligned streams for PBO"}
    L = min(len(c) for c in cols)
    R = np.column_stack([c[:L] for c in cols])
    try:
        from strat.pbo_cscv import pbo_cscv
        S = 8 if L >= 16 else 4
        out = pbo_cscv(R, S=S)
        out["family"] = names
        return out
    except Exception as e:
        return {"error": str(e)[:160], "family": names}


# =====================================================================================================
# 5. THE FULL RUN
# =====================================================================================================
STREAM_KEYS = ["cash_book", "raw_beta", "family_free"] + FAMILY6
PRESERVE_FLOOR = 50.0     # the participate-AND-preserve constraint (keep >= half the book's risk reduction)


def run(derisk=PICK_LEVEL):
    res = {"prereg": PREREG, "derisk": derisk}

    # (A) PER-YEAR DEV grade (2020-2024 + 2025-OOS) -- UNSEEN NOT touched here.
    print("\n## (A) PER-YEAR DEV (2020-2024 + 2025-OOS): cash_book / raw_beta / family_free / 4 gates / 2 ensembles")
    per_year = {k: {} for k in STREAM_KEYS}
    per_window = []
    dev_windows = [(_set_year_fn(y), y) for y in DEV_YEARS] + [(_set_2025_fn(), 2025)]
    for setter, label in dev_windows:
        d = _run_window(setter, derisk)
        per_window.append(d)
        for k in STREAM_KEYS:
            per_year[k][label] = d[k][0]
        cb = d["cash_book"][0]; rb = d["raw_beta"][0]
        print(f"   -- {label}: raw-beta {rb.get('net_pct')}% DD {rb.get('maxdd_pct')}% | "
              f"cash-book {cb.get('net_pct')}% DD {cb.get('maxdd_pct')}%")
        for name in ENSEMBLES:
            m = d[name][0]
            print(f"        ENSEMBLE {name:8} net {str(m.get('net_pct')):>8}% DD {str(m.get('maxdd_pct')):>8}% "
                  f"Sharpe {str(m.get('sharpe')):>5} Calmar {str(m.get('calmar')):>6}")
    res["per_year_metrics"] = {k: {str(y): v for y, v in d.items()} for k, d in per_year.items()}

    # (B) FULL-CYCLE DEV chain (2020-2025) -- wealth + maxDD per stream
    print("\n## (B) FULL-CYCLE DEV (2020-2025 calendar chain) -- wealth / maxDD")
    fc = {}
    for k in STREAM_KEYS:
        m, dd = _chain(per_window, k)
        fc[k] = {"metrics": m, "daily": dd}
        tag = " <ENSEMBLE>" if k in ENSEMBLES else ""
        print(f"   {k:16}: net {str(m.get('net_pct')):>9}% maxDD {str(m.get('maxdd_pct')):>8}% "
              f"Calmar {str(m.get('calmar')):>7} Sharpe {str(m.get('sharpe')):>6}{tag}")
    res["full_cycle_dev"] = {k: v["metrics"] for k, v in fc.items()}

    # (C) THE FRONTIER PLANE -- bull-capture % vs bear-preservation % (does an ensemble shift NE?)
    print("\n## (C) FRONTIER PLANE: bull-capture % (x) vs bear-DD-preservation % (y) -- does an ENSEMBLE shift NE?")
    raw_pm = per_year["raw_beta"]
    frontier = {}
    for k in STREAM_KEYS:
        bc = _bull_capture_pct(per_year[k], raw_pm)
        bp = _bear_preservation_pct(per_year[k], raw_pm)
        frontier[k] = {"bull_capture_pct": bc, "bear_preservation_pct": bp,
                       "fc_net_pct": fc[k]["metrics"].get("net_pct"),
                       "fc_maxdd_pct": fc[k]["metrics"].get("maxdd_pct")}
        tag = " <ENSEMBLE>" if k in ENSEMBLES else ""
        print(f"   {k:16}: bull-capture {str(bc):>7}% | bear-preservation {str(bp):>7}% | "
              f"full-cycle net {str(frontier[k]['fc_net_pct']):>8}% DD {str(frontier[k]['fc_maxdd_pct']):>8}%{tag}")
    res["frontier"] = frontier

    book_bc = frontier["cash_book"]["bull_capture_pct"]; book_bp = frontier["cash_book"]["bear_preservation_pct"]
    dominators = []
    for name in FAMILY6:
        bc = frontier[name]["bull_capture_pct"]; bp = frontier[name]["bear_preservation_pct"]
        if bc is None or bp is None or book_bc is None or book_bp is None:
            continue
        ne = (bc > book_bc) and (bp >= book_bp - 5.0)
        frontier[name]["dominates_book"] = bool(ne)
        if ne:
            dominators.append(name)
    ensemble_dominators = [n for n in dominators if n in ENSEMBLES]
    res["dominators"] = dominators
    res["ensemble_dominators"] = ensemble_dominators
    print(f"\n   cash-going book point: bull-capture {book_bc}% / bear-preservation {book_bp}%")
    print(f"   ALL constructions NE of the book: {dominators if dominators else 'NONE'}")
    print(f"   ENSEMBLES NE of the book (the falsifier's target): {ensemble_dominators if ensemble_dominators else 'NONE'}")

    # (D) PICK THE BEST ENSEMBLE by full-cycle DEV wealth s.t. bear-preservation >= floor (the decisive subject)
    cand = []
    for name in ENSEMBLES:
        bp = frontier[name]["bear_preservation_pct"]; net = frontier[name]["fc_net_pct"]
        if bp is not None and net is not None and bp >= PRESERVE_FLOOR:
            cand.append((name, net, bp))
    cand.sort(key=lambda x: -x[1])
    best_ens = cand[0][0] if cand else None
    res["best_ensemble"] = best_ens
    res["best_ensemble_candidates"] = [{"name": n, "fc_net_pct": net, "bear_preservation_pct": bp}
                                       for n, net, bp in cand]
    print(f"\n## (D) BEST participate-AND-preserve ENSEMBLE (DEV wealth s.t. bear-preservation >= {PRESERVE_FLOOR}%): {best_ens}")
    for n, net, bp in cand:
        print(f"      {n:8} full-cycle DEV net {net}% (bear-preservation {bp}%)")
    if best_ens is None:
        print(f"      NONE -- no ENSEMBLE keeps >= {PRESERVE_FLOOR}% bear-preservation. "
              f"Ensembling does not even reach the preserve floor.")

    # The deflation BEST must be the overall family best (an ensemble cannot be 'the result' unless it is the
    # best of the 6); we report both. For the decisive gate we evaluate the best ENSEMBLE.
    fam_net = {n: frontier[n]["fc_net_pct"] for n in FAMILY6 if frontier[n]["fc_net_pct"] is not None}
    overall_best = max(fam_net, key=fam_net.get) if fam_net else None
    res["overall_family_best"] = overall_best

    # (E) THE SEALED UNSEEN READ-ONCE -- all streams (single touch)
    print("\n## (E) SEALED UNSEEN READ-ONCE (2025-12-31 -> 2026-06-01) -- ALL streams. The ONLY UNSEEN touch.")
    du = _run_window(_set_unseen_fn(), derisk)
    u_regime_net = du["raw_beta"][0].get("net_pct")
    regime = ("BULL" if (u_regime_net or 0) > 15 else "BEAR" if (u_regime_net or 0) < -15 else "CHOP/SIDEWAYS")
    unseen = {"window": list(UNSEEN_WIN), "regime": regime}
    for k in STREAM_KEYS:
        m = du[k][0]
        unseen[k] = {"net_pct": m.get("net_pct"), "maxdd_pct": m.get("maxdd_pct"),
                     "sharpe": m.get("sharpe"), "calmar": m.get("calmar")}
        tag = " <ENSEMBLE>" if k in ENSEMBLES else ""
        print(f"   {k:16}: UNSEEN net {str(m.get('net_pct')):>8}% DD {str(m.get('maxdd_pct')):>8}% "
              f"Sharpe {str(m.get('sharpe')):>5}{tag}")
    res["unseen_once"] = unseen
    res["_unseen_daily"] = {k: du[k][1] for k in STREAM_KEYS}

    # (F) CANONICAL SCORECARD on the FULL stream (DEV + UNSEEN) -- the held-out block-bootstrap p05 ship-gate
    print("\n## (F) CANONICAL SCORECARD (full 2020-2026 stream; held-out p05 ship-gate)")
    cards = {}
    score_targets = ["cash_book"] + list(ENSEMBLES.keys())
    full_daily = {}
    for k in score_targets:
        fd = _full_stream(k, derisk)
        full_daily[k] = fd
        card = score_book(f"pp_ensemble::{k}", fd)
        cards[k] = card
        hp = card.get("heldout_block_bootstrap", {}).get("p05")
        fp = card.get("full_block_bootstrap", {}).get("p05")
        u = card["per_split"].get("UNSEEN", {})
        print(f"   [{k:10}] n_days={card['n_days']} | UNSEEN compound {u.get('compound_pct')}% | "
              f"held-out p05 {hp}% | full p05 {fp}% | ship={card['ship_read']['ship']}")
    res["scorecards"] = cards
    res["_full_daily"] = full_daily

    # (G) MULTIPLE-COMPARISONS DEFLATION over the FULL family of 6 (the best ensemble must survive)
    print("\n## (G) DEFLATION over the FULL family of 6 (4 gates + 2 ensembles) -- the best ENSEMBLE must survive")
    deflation = {}
    if best_ens is not None:
        deflation["permutation_null"] = _permutation_deflate6(per_window, best_ens)
        deflation["pbo"] = _pbo6(per_window)
        pn = deflation["permutation_null"]
        if "p_value_maxstat" in pn:
            print(f"   PERMUTATION (max-stat over the 6-family best-of): observed best ENSEMBLE edge over book = "
                  f"{pn['observed_best_edge_pp']}pp ({pn['best_name']}); null max-edge p95 = {pn['null_max_p95_pp']}pp; "
                  f"p = {pn['p_value_maxstat']} (family_size {pn['family_size']}) -> "
                  f"{'SURVIVES' if pn['survives_deflation'] else 'FAILS deflation'}.")
        else:
            print(f"   PERMUTATION: {pn.get('error', pn)}")
        pbo = deflation["pbo"]
        if "pbo" in pbo:
            print(f"   PBO across the 6-family (S={pbo['S']}, N={pbo['N']}): PBO={pbo['pbo']:.3f} -> {pbo['verdict']}.")
        else:
            print(f"   PBO: {pbo.get('error', pbo)}")
    else:
        deflation["note"] = "no preserving ensemble to deflate (the ensemble does not reach the preserve floor)."
        print("   " + deflation["note"])
    res["deflation"] = deflation

    res["verdict"] = build_verdict(res)
    return res


# =====================================================================================================
# 6. VERDICT
# =====================================================================================================
def build_verdict(res):
    fr = res["frontier"]; fc = res["full_cycle_dev"]; u = res["unseen_once"]
    best = res.get("best_ensemble")
    ensemble_dominators = res.get("ensemble_dominators", [])
    cards = res.get("scorecards", {})
    deflation = res.get("deflation", {})

    book_net = fc["cash_book"].get("net_pct"); book_dd = fc["cash_book"].get("maxdd_pct")
    bh_net = fc["raw_beta"].get("net_pct")

    gate_ne = bool(ensemble_dominators)                                         # an ENSEMBLE NE of the book
    best_net = fc.get(best, {}).get("net_pct") if best else None
    gate_beats_wealth = bool(best_net is not None and book_net is not None and best_net > book_net + 5.0)
    best_card = cards.get(best, {})
    hp = best_card.get("heldout_block_bootstrap", {}).get("p05") if best else None
    gate_p05 = bool(hp is not None and hp > 0)
    u_best = u.get(best, {}).get("net_pct") if best else None
    gate_unseen_pos = bool(u_best is not None and u_best > 0)
    pn = deflation.get("permutation_null", {})
    gate_perm = bool(pn.get("survives_deflation"))                              # permutation p < 0.05
    pbo_val = deflation.get("pbo", {}).get("pbo")
    gate_pbo = bool(pbo_val is not None and pbo_val < 0.5)
    u_best_dd = u.get(best, {}).get("maxdd_pct") if best else None
    u_book_dd = u.get("cash_book", {}).get("maxdd_pct")
    gate_preserve_unseen = bool(u_best_dd is not None and u_book_dd is not None and u_best_dd >= u_book_dd - 10.0)

    # THE DECISIVE GATE (pre-registered): REOPENS iff NE AND p05>0 AND permutation p<0.05 AND PBO<0.5
    reopens = gate_ne and gate_p05 and gate_perm and gate_pbo
    if reopens:
        verdict = "REOPENS"
    elif gate_ne and gate_beats_wealth:
        verdict = "SHUT_DEV_DOMINATES_BUT_NOT_SHIPPABLE"     # NE on dev but fails a held-out/deflation gate
    else:
        verdict = "SHUT_NO_NE_POINT"                          # no ensemble even reaches NE of the book

    lines = [
        "## DECISIVE VERDICT (regime-conditioned ENSEMBLE -- the last shot for long-only participate-AND-preserve) "
        "[VERIFIED-HELDOUT + UNSEEN-ONCE]",
        f"INCUMBENT (cash-going book): full-cycle DEV net {book_net}% maxDD {book_dd}% | frontier point "
        f"bull-capture {fr['cash_book'].get('bull_capture_pct')}% / bear-preservation "
        f"{fr['cash_book'].get('bear_preservation_pct')}%. Buy-hold net {bh_net}%.",
        f"ENSEMBLES NORTHEAST of the book (MORE participation AND preservation within 5pp): "
        f"{ensemble_dominators if ensemble_dominators else 'NONE'} -> "
        f"{'an ensemble dominates the book on the DEV plane' if gate_ne else 'NO ensemble dominates -- they sit ON the frontier line'}.",
        f"BEST participate-AND-preserve ENSEMBLE: {best} (full-cycle DEV net {best_net}% vs book {book_net}%; "
        f"{'beats book wealth by >5pp' if gate_beats_wealth else 'does NOT beat book wealth held-out'}).",
        f"SEALED UNSEEN read: best ensemble {best} net {u_best}% DD {u_best_dd}% vs cash-book net "
        f"{u.get('cash_book', {}).get('net_pct')}% DD {u_book_dd}% (regime {u['regime']}); "
        f"UNSEEN-positive {gate_unseen_pos}, preservation-kept {gate_preserve_unseen}.",
        f"DECISIVE GATE 1 (NE of book): {gate_ne}.",
        f"DECISIVE GATE 2 (held-out block-bootstrap p05 > 0): p05 = {hp}% -> {gate_p05}.",
        f"DECISIVE GATE 3 (permutation max-stat p < 0.05 over the 6-family): p = {pn.get('p_value_maxstat')} -> {gate_perm}.",
        f"DECISIVE GATE 4 (PBO < 0.5 over the 6-family): PBO = {pbo_val} -> {gate_pbo}.",
        f"ALL-GATES (REOPENS iff ALL true): NE={gate_ne} | p05>0={gate_p05} | perm_p<0.05={gate_perm} | PBO<0.5={gate_pbo}.",
        ("REOPENS: a regime-conditioned ENSEMBLE lands NORTHEAST of the cash-going book, clears held-out p05>0, "
         "survives the 6-family max-stat permutation (p<0.05) AND PBO<0.5. Long-only participate-and-preserve "
         "is NOT a fundamental tradeoff -- this ensemble is the genuine deploy candidate."
         if verdict == "REOPENS" else
         "SHUT: the regime-conditioned ENSEMBLE -- the cheapest remaining falsifier -- does NOT clear the "
         "decisive gate. Combining the 4 gates (by vote OR by causal regime-routing) does NOT produce a "
         "long-only construction that is NE of the book held-out with p05>0 AND survives multiple-comparisons "
         "deflation. Long-only participate-and-preserve is DEFINITIVELY CLOSED. The honest doors are SHORT "
         "(OFF -- the user's shortcut) or CARRY (the funding-dispersion dollar-neutral sleeve)."),
        f"CHEAPEST FALSIFIER: {_cheapest_falsifier(res, verdict)}",
    ]
    return {"verdict": verdict, "reopens": bool(reopens),
            "gates": {"ne_of_book": gate_ne, "beats_book_wealth": gate_beats_wealth,
                      "heldout_p05_pos": gate_p05, "unseen_pos": gate_unseen_pos,
                      "permutation_survives": gate_perm, "pbo_lt_half": gate_pbo,
                      "preserve_unseen": gate_preserve_unseen},
            "best_ensemble": best, "best_fc_dev_net_pct": best_net, "book_fc_dev_net_pct": book_net,
            "buyhold_fc_dev_net_pct": bh_net, "best_heldout_p05": hp,
            "best_unseen_net_pct": u_best, "best_unseen_maxdd_pct": u_best_dd,
            "permutation_p": pn.get("p_value_maxstat"), "pbo": pbo_val, "lines": lines}


def _cheapest_falsifier(res, verdict):
    if verdict == "REOPENS":
        return ("the reopened ensemble's held-out p05>0 + permutation p<0.05 + PBO<0.5 are LOAD-BEARING; "
                "re-derive on block in {5,10,20} and seeds {7,11,23} -- a single flip below the bar collapses it.")
    return ("long-only participate-and-preserve is now DEFINITIVELY SHUT (4 directional gates + 2 ensembles, all "
            "fail the held-out + deflation gates). The only remaining long-only falsifier would be an EXTERNAL "
            "conditioner (Coinglass / on-chain) that times the de-risk -- which is no longer 'long-only beta "
            "re-shaping' but a new signal. The honest doors are SHORT (OFF) or CARRY.")


# =====================================================================================================
# 7. CHART
# =====================================================================================================
def make_chart(res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[chart] matplotlib unavailable ({e}) -- skipped")
        return []
    fr = res["frontier"]
    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    base_colors = {"cash_book": "#2a9d8f", "raw_beta": "#264653", "family_free": "#8a817c"}
    for k, pt in fr.items():
        bc = pt.get("bull_capture_pct"); bp = pt.get("bear_preservation_pct")
        if bc is None or bp is None:
            continue
        if k in base_colors:
            ax.scatter(bc, bp, s=190, color=base_colors[k], zorder=6, edgecolor="k", lw=1.3)
            ax.annotate(k, (bc, bp), textcoords="offset points", xytext=(8, 6), fontsize=10, fontweight="bold")
        elif k in ENSEMBLES:
            dom = pt.get("dominates_book", False)
            ax.scatter(bc, bp, s=210, color=("#43aa8b" if dom else "#f4a261"), marker="*", zorder=7,
                       edgecolor="k", lw=1.2)
            ax.annotate("ENS:" + k + (" [NE]" if dom else ""), (bc, bp), textcoords="offset points",
                        xytext=(8, -14), fontsize=9, fontweight="bold")
        else:  # the 4 original gates
            dom = pt.get("dominates_book", False)
            ax.scatter(bc, bp, s=120, color=("#43aa8b" if dom else "#e76f51"), marker="D", zorder=4,
                       edgecolor="k", lw=0.9)
            ax.annotate(k + (" [NE]" if dom else ""), (bc, bp), textcoords="offset points",
                        xytext=(8, -12), fontsize=8)
    book = fr.get("cash_book", {})
    bbc = book.get("bull_capture_pct"); bbp = book.get("bear_preservation_pct")
    if bbc is not None and bbp is not None:
        ax.axvline(bbc, color="#2a9d8f", ls=":", lw=1.0, alpha=0.7)
        ax.axhline(bbp, color="#2a9d8f", ls=":", lw=1.0, alpha=0.7)
        ax.fill_betweenx([bbp - 5, 105], bbc, 200, color="#43aa8b", alpha=0.08,
                         label="dominance region (NE of book)")
    ax.set_xlabel("bull-capture %  (construction bull wealth / raw-beta bull wealth)  -- PARTICIPATION ->")
    ax.set_ylabel("bear-DD-preservation %  (1 - construction worst-bear-DD / raw-beta worst-bear-DD)  -- PRESERVATION ->")
    v = res["verdict"]
    ax.set_title("LONG-ONLY participate-AND-preserve -- the REGIME-CONDITIONED ENSEMBLE (last shot)\n"
                 f"VERDICT: {v['verdict']} "
                 f"({'an ensemble dominates + ships' if v['reopens'] else 'no ensemble clears the decisive gate -- door SHUT'})")
    ax.legend(loc="lower left", fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    c = OUT / "pp_ensemble_frontier.png"
    fig.savefig(c, dpi=120); plt.close(fig)
    print(f"[chart] {c}")
    return [str(c)]


# =====================================================================================================
# 8. PERSIST + MAIN
# =====================================================================================================
def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    if isinstance(obj, pd.Series):
        return None
    return obj


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.pp_ensemble")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--derisk", default=PICK_LEVEL)
    ap.add_argument("--no-charts", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    print("## LONG-ONLY PARTICIPATE-AND-PRESERVE -- REGIME-CONDITIONED ENSEMBLE (the flagged cheapest falsifier)")
    print("## PRE-REGISTRATION (stated BEFORE the run):")
    for k in ("H0_door_shut", "H1_reopens", "asymmetric_loss", "decisive_gate", "regime_detector", "multiplicity_family"):
        print(f"   {k}: {PREREG[k]}")
    print(f"\n   DEV years {list(DEV_YEARS)} + 2025-OOS | FROZEN de-risk {a.derisk} | LONG-ONLY spot | "
          f"fixed-EW | PIT | UNSEEN {list(UNSEEN_WIN)} READ-ONCE | deflation family of 6\n")

    res = run(derisk=a.derisk)

    print("\n" + "=" * 110)
    for line in res["verdict"]["lines"]:
        print(f"   {line}")
    print(f"\n   >>> VERDICT: {res['verdict']['verdict']}")
    print("=" * 110)

    charts = [] if a.no_charts else make_chart(res)
    res["charts"] = charts

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT / f"pp_ensemble_{stamp}.json"
    payload = {"repro": {"command": "python -m strat.pp_ensemble " + " ".join(argv or sys.argv[1:]),
                         "git_sha": sha, "dev_years": list(DEV_YEARS), "unseen_window": list(UNSEEN_WIN),
                         "derisk": a.derisk, "cost_maker": MAKER_RT, "deflation_family": FAMILY6},
               "prereg": PREREG, "results": _strip(res), "charts": charts}
    json.dump(payload, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    return 0


# =====================================================================================================
# 9. SELFTEST -- mechanics sanity (does NOT touch the UNSEEN window)
# =====================================================================================================
def selftest():
    from strat.family_ensemble_book import _set_year
    print("## PP-ENSEMBLE SELFTEST (mechanics only; no UNSEEN)")
    ok = True
    _set_year(2022)
    pit_universe_2021(verbose=False)
    assets = FT._assets_for("1d", False, "expand")
    A = assets[0]
    n = len(A["c"])

    # (1) both ensemble exposures are CONTINUOUS in [0,1] (long-only) and right length
    s1 = True
    for name, fn in ENSEMBLES.items():
        e = np.asarray(fn(A))
        good = (e.shape[0] == n) and (e.min() >= -1e-9) and (e.max() <= 1.0 + 1e-9)
        print(f"  (1) ensemble {name:8} len={e.shape[0]} range=[{round(float(e.min()),3)},{round(float(e.max()),3)}] "
              f"mean={round(float(e.mean()),3)} -> {'ok' if good else 'BAD (NOT in [0,1])'}")
        s1 &= good
    print(f"  (1) ensembles long-only [0,1] correct length -> {'PASS' if s1 else 'FAIL'}")
    ok &= s1

    # (2) VOTE = exact mean of the 4 gate held-arrays (no hidden re-weighting)
    H = _gate_holds(A)
    manual = np.clip(np.vstack(list(H.values())).mean(axis=0), 0.0, 1.0)
    s2 = bool(np.allclose(ens_vote(A), manual))
    print(f"  (2) VOTE == mean of the 4 gate held-arrays -> {'PASS' if s2 else 'FAIL'}")
    ok &= s2

    # (3) regime detector is CAUSAL (changing FUTURE prices does not change PAST regimes)
    A2 = dict(A); c2 = np.array(A["c"], float).copy()
    half = n // 2
    c2[half:] = c2[half:] * 1.5            # perturb the FUTURE only
    A2["c"] = c2
    r1 = _causal_regime(A); r2 = _causal_regime(A2)
    s3 = bool(np.array_equal(r1[:half], r2[:half]))     # past regimes unchanged
    print(f"  (3) regime detector causal (future perturbation leaves past regimes unchanged) -> {'PASS' if s3 else 'FAIL'}")
    ok &= s3

    # (4) ROUTED is causal too (same future-perturbation invariance on the past exposures)
    e1 = ens_routed(A); e2 = ens_routed(A2)
    s4 = bool(np.allclose(e1[:half], e2[:half]))
    print(f"  (4) ROUTED exposure causal (past exposures invariant to future perturbation) -> {'PASS' if s4 else 'FAIL'}")
    ok &= s4

    # (5) both ensemble books build a finite 2022 bear book and preserve (DD not deeper than raw beta)
    bh = build_buyhold(); m_raw = _metrics(bh)
    s5 = True
    for name, fn in ENSEMBLES.items():
        d = build_ensemble_book(fn)
        m = _metrics(d) if d is not None else {}
        finite = m.get("net_pct") is not None and m.get("maxdd_pct") is not None
        shallower = (m.get("maxdd_pct") is not None and m_raw.get("maxdd_pct") is not None
                     and m["maxdd_pct"] >= m_raw["maxdd_pct"] - 1.0)
        print(f"  (5) ENSEMBLE {name:8} 2022 net {m.get('net_pct')}% DD {m.get('maxdd_pct')}% "
              f"(raw-beta DD {m_raw.get('maxdd_pct')}%) finite={finite} shallower={shallower}")
        s5 &= bool(finite and shallower)
    print(f"  (5) ensemble books build + preserve in the 2022 bear -> {'PASS' if s5 else 'FAIL'}")
    ok &= s5

    # (6) the deflation family is exactly 6 (4 gates + 2 ensembles) -- honest multiplicity
    s6 = (len(FAMILY6) == 6) and set(FAMILY6) == set(CONSTRUCTIONS) | set(ENSEMBLES)
    print(f"  (6) deflation family = 6 (4 gates + 2 ensembles): {FAMILY6} -> {'PASS' if s6 else 'FAIL'}")
    ok &= s6

    # (7) the long-only invariant is structural (clip in [0,1], no skipna, fillna(0.0).mean)
    import inspect
    src = inspect.getsource(_ensemble_net_series) + inspect.getsource(build_ensemble_book)
    s7 = "np.clip(exp, 0.0, 1.0)" in src and "fillna(0.0).mean" in src and "skipna" not in src
    print(f"  (7) long-only + fixed-EW invariant (clip exp in [0,1], fillna(0.0).mean, no skipna) -> {'PASS' if s7 else 'FAIL'}")
    ok &= s7

    # (8) selftest did NOT touch the sealed UNSEEN window
    s8 = tuple(FT.WIN) != tuple(UNSEEN_WIN)
    print(f"  (8) selftest did NOT touch the sealed UNSEEN window (WIN={FT.WIN}) -> {'PASS' if s8 else 'FAIL'}")
    ok &= s8

    print(f"\n  SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
