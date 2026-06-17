"""src/strat/forward_ensemble_2021.py -- TRANSLATE the 2020-ROBUST CONFIG SETS (the working band per TI, NOT
the single #1) into 2021 as forward ENSEMBLES, and test whether the ENSEMBLE transfers where the single-#1
collapsed (rank-transfer ~ 0, ADX 2020-best -> 2021-worst).

USER /orc 2026-06-16: "find a way to translate the 2020 results into 2021 ... we didn't just select 1 config,
but MULTIPLE ones per TI. I suspect this requires extensive research." The 2021 single-#1 forward test
(forward_test_2021.py) found rank-transfer ~ 0 -- the single 2020-best config does NOT carry. But the 2020 work
selected a SET of robust configs per TI (the working band). This harness translates the SET.

RESEARCH-GROUNDED METHOD (scout brief 2026-06-16; DeMiguel 2009 1/N, Bailey/Lopez-de-Prado PBO, Pardo/Zakamulin
parameter-plateau, Timmermann/Wang forecast-combination puzzle):
  - The single backtest #1 is NOISE out-of-sample (PBO); the stable REGION/band is the signal (plateau).
  - EQUAL-WEIGHT (1/N) the robust band is the correct cross-regime translation -- 1/N provably beats any
    performance re-weighting OOS, ESPECIALLY across a regime shift (small sample + unstable + OOS-differs-from-IS
    => all three conditions favour 1/N). Re-weighting on 2020-IS picks up the same noise that overfit the configs.
  - Aggregate the RETURN STREAMS / signals (average), NOT majority-vote (homogeneous configs).
  - CAVEAT: the ensemble reduces within-class CONCENTRATION risk; it does NOT rescue the asset-class ceiling
    (long-only TI = de-risked beta). Expect: the ensemble TRANSFERS more ROBUSTLY than the #1 (fewer blow-ups,
    higher worst-case, better crash-preservation), not that it beats buy-hold.

PRIMARY = EW the 2020-robust band per TI (1/N over member daily books). CHALLENGER = recency-60d re-weight
(arxiv 2602.11708 adaptive, prospective) -- must beat EW OOS, not IS.

The TEST: per TI, does the 2020-robust-band ENSEMBLE on 2021 (a) avoid the catastrophic single-#1 transfer
failures (ADX -18%), (b) have a higher WORST-CASE / lower dispersion across TIs, (c) preserve in the May crash.

REUSES forward_test_2021 (PIT survivorship-clean universe, _candidate_net_series stack, _ew_book, _equity_metrics,
_regime_metrics, _arm_ti). 2021-only, frozen 2020 configs, maker fills, UNSEEN sealed. No emoji.

RWYB:
  python -m strat.forward_ensemble_2021 --selftest
  python -m strat.forward_ensemble_2021                 # core + expand, EW + recency
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

import strat.forward_test_2021 as FT                                          # noqa: E402
from strat.forward_test_2021 import (                                        # noqa: E402
    _assets_for, _candidate_net_series, _ew_book, _equity_metrics, _regime_metrics,
    _arm_ti, _spearman, MAKER_RT, TAKER_RT, WIN)
import strat.deep2020_ti_pipeline as TI                                      # noqa: E402

BASE = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
OUT = ROOT.parent / "runs" / "strat"
RECENCY_BARS = {"1d": 60, "4h": 360, "2h": 720, "1h": 1440, "30m": 2880, "15m": 5760}   # ~60d per cadence

__contract__ = {
    "kind": "forward_ensemble_2021_translation",
    "inputs": {
        "robust_sets": "per TI: the 2020-robust config SET (working band) from ti_registry top10_ironed_by_tf "
                       "filtered robust=True (multiple configs, not the #1)",
        "method": "EW (1/N over member daily books) PRIMARY; recency-60d re-weight CHALLENGER",
    },
    "outputs": {
        "transfer": "per TI: ENSEMBLE 2021 fwd net/maxDD/crash vs the single-#1 2021 fwd; aggregate -- does the "
                    "band transfer more robustly (higher worst-case, fewer blow-ups) than the #1?",
    },
    "invariants": {
        "ew_is_primary": "equal-weight the robust band (1/N) -- literature: 1/N beats re-weighting OOS across regime shift",
        "frozen_2020_set": "the robust SET is the 2020 selection; NO 2021 re-selection",
        "no_majority_vote": "aggregate return streams (average), not majority-vote (homogeneous configs)",
        "reuses_pit_universe": "same survivorship-clean PIT 2021 universe + stack as forward_test_2021",
        "unseen_sealed": "2021-only; UNSEEN 2025-26 untouched",
    },
}


# =====================================================================================================
# 1. THE 2020-ROBUST SETS per TI (the working band -- multiple configs we actually selected)
# =====================================================================================================
def load_robust_sets():
    """Per indicator: (best_ironed_tf, [robust cfg strings net-sorted], single_#1 cfg). From ti_registry."""
    reg = json.load(open(BASE / "ti_registry.json"))
    sets = {}
    for fam, inds in reg.items():
        for ind, d in inds.items():
            bt = d["profile"].get("best_ironed_tf")
            if not bt:
                continue
            t10 = d.get("top10_ironed_by_tf", {}).get(bt, [])
            robust = [r["cfg"] for r in t10 if r.get("robust")]               # net-sorted (top10 is net-desc)
            if len(robust) >= 1:
                sets[ind] = {"family": fam, "tf": bt, "robust_cfgs": robust, "n_robust": len(robust),
                             "single_1": robust[0]}
    return sets


# =====================================================================================================
# 2. MEMBER BOOKS + ENSEMBLE
# =====================================================================================================
def _vt_for(assets):
    rvs = [np.nanmedian(A["rv"][A["active"]]) for A in assets if A["active"].sum() > 5]
    rvs = [x for x in rvs if np.isfinite(x)]
    return float(np.nanmedian(rvs)) if rvs else None


def _member_books(ind, cfgs, cad, universe):
    """Daily book per robust-set member (over the PIT universe). Returns {cfg: daily Series}."""
    books = {}
    for cfg in cfgs:
        armed = _arm_ti(ind, cfg)
        if armed is None:
            continue
        held_fn, params, minhold, loader = armed
        want_vol = loader == "ohlcv"
        assets = _assets_for(cad, want_vol, universe)
        if not assets:
            continue
        vt = _vt_for(assets)
        series = [_candidate_net_series(A, held_fn, params, minhold, vt) for A in assets]
        book = _ew_book(series, universe)
        if book is not None and len(book) > 5:
            books[cfg] = book
    return books


def _ensemble_ew(member_books):
    """1/N equal-weight over member daily books (the literature-preferred translation)."""
    if not member_books:
        return None
    df = pd.concat(member_books.values(), axis=1).sort_index()
    return df.mean(axis=1, skipna=True).dropna()


def _ensemble_recency(member_books, cad):
    """CHALLENGER: recency-60d re-weight (prospective). Each member weighted by its trailing-~60d mean return
    (shifted 1 bar, clipped >=0, normalized); falls back to EW when all trailing weights are non-positive.
    Tests whether prospective adaptation beats 1/N OOS (arxiv 2602.11708) -- must beat EW, not IS."""
    if not member_books:
        return None
    df = pd.concat(member_books.values(), axis=1).sort_index()
    w = RECENCY_BARS.get(cad, 60)
    trail = df.rolling(w, min_periods=max(5, w // 4)).mean().shift(1)          # past-only trailing mean return
    wt = trail.clip(lower=0.0)
    s = wt.sum(axis=1)
    wt = wt.div(s.where(s > 0, np.nan), axis=0)                               # normalize where any positive
    ew = pd.DataFrame(1.0 / df.shape[1], index=df.index, columns=df.columns)  # EW fallback
    wt = wt.fillna(ew)
    return (df * wt).sum(axis=1).dropna()


# =====================================================================================================
# 3. PER-TI ENSEMBLE-vs-#1 TRANSFER on 2021
# =====================================================================================================
def run_ti(ind, spec, universe):
    """Build the per-TI member books once, then EW + recency ensembles + the single-#1, with 2021 metrics."""
    cad = spec["tf"]; cfgs = spec["robust_cfgs"]
    books = _member_books(ind, cfgs, cad, universe)
    if not books:
        return None
    ew = _ensemble_ew(books)
    rec = _ensemble_recency(books, cad)
    one = books.get(spec["single_1"])
    if one is None:
        one = next(iter(books.values()))
    def pack(b):
        if b is None or len(b) < 5:
            return None
        m = _equity_metrics(b, cad); m["regimes"] = _regime_metrics(b); return m
    return {"ind": ind, "family": spec["family"], "tf": cad, "n_members": len(books),
            "ENSEMBLE_EW": pack(ew), "ENSEMBLE_RECENCY": pack(rec), "SINGLE_1": pack(one),
            "single_1_cfg": spec["single_1"]}


def _crash(m):
    return (m or {}).get("regimes", {}).get("May_crash") if m else None


def build_verdict(rows, universe):
    valid = [r for r in rows if r and r.get("ENSEMBLE_EW") and r.get("SINGLE_1")]
    if not valid:
        return {"lines": [f"[{universe}] no valid TIs"]}
    ew_nets = [r["ENSEMBLE_EW"]["final_pct"] for r in valid]
    one_nets = [r["SINGLE_1"]["final_pct"] for r in valid]
    rec_nets = [r["ENSEMBLE_RECENCY"]["final_pct"] for r in valid if r.get("ENSEMBLE_RECENCY")]
    ew_wins = [r["ind"] for r in valid if r["ENSEMBLE_EW"]["final_pct"] > r["SINGLE_1"]["final_pct"]]
    ew_blowups = [r["ind"] for r in valid if r["ENSEMBLE_EW"]["final_pct"] < 0]
    one_blowups = [r["ind"] for r in valid if r["SINGLE_1"]["final_pct"] < 0]
    # crash-preservation: ensemble vs #1 in the May crash
    ew_better_crash = [r["ind"] for r in valid
                       if _crash(r["ENSEMBLE_EW"]) is not None and _crash(r["SINGLE_1"]) is not None
                       and _crash(r["ENSEMBLE_EW"]) > _crash(r["SINGLE_1"])]
    rec_beats_ew = [r["ind"] for r in valid if r.get("ENSEMBLE_RECENCY")
                    and r["ENSEMBLE_RECENCY"]["final_pct"] > r["ENSEMBLE_EW"]["final_pct"] + 0.5]
    n = len(valid)
    lines = [
        f"", f"## [{universe}] ENSEMBLE-vs-#1 TRANSFER to 2021 (n={n} TIs with a robust band) [VERIFIED-2021-FORWARD]",
        f"net (2021 fwd): EW-ensemble  min {min(ew_nets):.1f} / median {np.median(ew_nets):.1f} / max {max(ew_nets):.1f}",
        f"                single-#1    min {min(one_nets):.1f} / median {np.median(one_nets):.1f} / max {max(one_nets):.1f}",
        f"ROBUSTNESS (the point): EW-ensemble blow-ups (net<0) = {len(ew_blowups)}/{n} {ew_blowups}; "
        f"single-#1 blow-ups = {len(one_blowups)}/{n} {one_blowups}. "
        f"WORST-CASE: EW min {min(ew_nets):.1f}% vs #1 min {min(one_nets):.1f}%.",
        f"net WINS: EW-ensemble > #1 at {len(ew_wins)}/{n} TIs (but the value is the WORST-CASE/blow-up "
        f"reduction, not median net -- per robust_ma_runners + the 1/N literature).",
        f"CRASH-PRESERVATION (May-2021): EW-ensemble lost LESS than #1 at {len(ew_better_crash)}/{n} TIs.",
        f"CHALLENGER (recency-60d re-weight): beats EW at {len(rec_beats_ew)}/{n} TIs "
        f"(median rec {np.median(rec_nets):.1f}% vs EW {np.median(ew_nets):.1f}%) -- "
        f"{'recency adds OOS value' if len(rec_beats_ew) > n / 2 else 'NO -- 1/N EW remains the robust default (as the literature predicts across a regime shift)'}.",
    ]
    return {"n": n, "ew_wins": ew_wins, "ew_blowups": ew_blowups, "one_blowups": one_blowups,
            "ew_better_crash": ew_better_crash, "rec_beats_ew": rec_beats_ew,
            "ew_min": round(float(min(ew_nets)), 1), "one_min": round(float(min(one_nets)), 1),
            "ew_median": round(float(np.median(ew_nets)), 1), "one_median": round(float(np.median(one_nets)), 1),
            "lines": lines}


# =====================================================================================================
# 4. MAIN
# =====================================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.forward_ensemble_2021")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--universes", default="core,expand")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    sets = load_robust_sets()
    multi = {k: v for k, v in sets.items() if v["n_robust"] >= 2}             # ensembles need >=2 members
    print("## FORWARD-ENSEMBLE 2021 -- translate the 2020-ROBUST BANDS (multiple configs/TI) forward as 1/N ensembles")
    print(f"   robust bands: {len(sets)} TIs have >=1 robust config; {len(multi)} have >=2 (ensemble-able)")
    print(f"   method = EW (1/N) PRIMARY + recency-60d CHALLENGER | PIT 2021 universe | frozen 2020 sets | {WIN}\n")
    for ind, v in multi.items():
        print(f"   {v['family']:14} {ind:10} @{v['tf']:3}: {v['n_robust']} robust members (band={v['robust_cfgs'][:5]}{'...' if v['n_robust']>5 else ''})")

    allres = {}
    for universe in [u.strip() for u in a.universes.split(",")]:
        print(f"\n========== UNIVERSE: {universe} ==========")
        rows = []
        for ind, spec in multi.items():
            r = run_ti(ind, spec, universe)
            rows.append(r)
            if r and r.get("ENSEMBLE_EW") and r.get("SINGLE_1"):
                ew, one = r["ENSEMBLE_EW"], r["SINGLE_1"]
                print(f"   {r['family']:14} {ind:10} @{r['tf']:3} ({r['n_members']}m) | "
                      f"EW-ens net {ew['final_pct']:>7}% (DD {ew['maxdd_pct']}, crash {_crash(ew)}) | "
                      f"#1 net {one['final_pct']:>7}% (DD {one['maxdd_pct']}, crash {_crash(one)})")
        v = build_verdict(rows, universe)
        for line in v["lines"]:
            print(f"   {line}")
        allres[universe] = {"rows": rows, "verdict": v}

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT / f"forward_ensemble_2021_{stamp}.json"
    json.dump({"repro": {"command": "python -m strat.forward_ensemble_2021 " + " ".join(argv or sys.argv[1:]),
                         "git_sha": sha, "window": WIN, "cost_maker": MAKER_RT,
                         "method": "EW 1/N primary + recency-60d challenger", "robust_sets": sets},
               "results": allres}, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    return 0


def selftest():
    print("## FORWARD-ENSEMBLE-2021 SELFTEST")
    ok = True
    sets = load_robust_sets()
    multi = {k: v for k, v in sets.items() if v["n_robust"] >= 2}
    s1 = len(multi) >= 6
    print(f"  (1) robust bands loaded: {len(sets)} TIs, {len(multi)} with >=2 members -> {'PASS' if s1 else 'FAIL'}")
    ok &= s1
    # (2) EW ensemble of 2 synthetic member books = their average
    idx = pd.date_range("2021-01-01", periods=30, freq="1D")
    b1 = pd.Series(np.full(30, 0.01), index=idx); b2 = pd.Series(np.full(30, -0.01), index=idx)
    ew = _ensemble_ew({"a": b1, "b": b2})
    s2 = ew is not None and abs(float(ew.mean())) < 1e-9
    print(f"  (2) EW(+1%,-1%) ~ 0 mean: {float(ew.mean()):.5f} -> {'PASS' if s2 else 'FAIL'}")
    ok &= s2
    # (3) recency re-weight tilts toward the recently-better member
    up = pd.Series(np.concatenate([np.full(15, -0.01), np.full(15, 0.02)]), index=idx)
    dn = pd.Series(np.concatenate([np.full(15, 0.01), np.full(15, -0.02)]), index=idx)
    rec = _ensemble_recency({"up": up, "dn": dn}, "1d")
    s3 = rec is not None and float(rec.iloc[-1]) > float((up.iloc[-1] + dn.iloc[-1]) / 2)
    print(f"  (3) recency tilts to recent-winner: last {float(rec.iloc[-1]):.4f} > EW {float((up.iloc[-1]+dn.iloc[-1])/2):.4f} "
          f"-> {'PASS' if s3 else 'FAIL'}")
    ok &= s3
    print(f"\n  SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
