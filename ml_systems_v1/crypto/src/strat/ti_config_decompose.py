"""src/strat/ti_config_decompose.py -- DECOMPOSE the TI canonical list into its DISTINCT config set.

WHY (user /orc 2026-06-11): *"use that new callable function [canonicalize_grid] for us to have a set
of DISTINCT strats/configs for each individual strat and strat family in our TI canonical list. Can
you decompose per timeframe? Or is the decomposition timeframe-agnostic? Just run it and tell me the
findings."*

WHAT IT DOES. For every indicator in config/ti_master_catalog.yaml, it (a) generates the RAW candidate
parameter grid at a standard granularity, (b) runs canonicalize_grid (the near-dup eliminator) to
collapse practically-identical configs (MA(28,29) ~= MA(27,30)) to mutually-separated representatives,
and (c) reports raw -> DISTINCT per indicator, rolled up per family + grand total. The DISTINCT count
is the honest size of the TI strategy-config search space (the denominator for multiple-comparisons).

THE TIMEFRAME QUESTION (answered): canonicalize_grid operates on PARAMETER TUPLES, not on price data,
so the DISTINCT-CONFIG SET IS TIMEFRAME-AGNOSTIC -- the same set of distinct (fast,slow) pairs applies
at 1d, 4h, 15m, anywhere. What IS per-timeframe is (1) the WALL-CLOCK MEANING of a length (200 bars =
200d at 1d but ~33d at 4h) and (2) the OPTIMAL pick FROM that set (PER_CADENCE_CONFIG: 1d->EMA50/100,
4h->EMA50/200). So: the SEARCH SPACE is decomposed once, timeframe-agnostic; the WINNER is selected
per-cadence. This module proves (1) by running the same canonicalization with no timeframe input.

CONFIG-KIND taxonomy (each indicator's natural parameterization; an approximation, documented):
  single_length  one window/period L                  (RSI, ATR, EMA-as-filter, z-score window, ...)
  ma_cross       (fast, slow) cross                    (SMA/EMA/WMA/... used as a 2-MA cross)
  ma_triple      (a, b, c) 3-MA alignment
  macd           (fast, slow, signal)
  bollinger_2p   (period, std_mult)                    (also Keltner)
  stoch_2p       (k_period, d_period)
  supertrend_2p  (atr_period, atr_mult)
  fixed          parameter-free                        (VWAP session, pivots, prior-period H/L) -> 1

RWYB:
    python src/strat/ti_config_decompose.py                 # full report, rel_tol=0.15, writes JSON
    python src/strat/ti_config_decompose.py --rel-tol 0.10  # finer (more distinct configs survive)
    python src/strat/ti_config_decompose.py --prove-tf-agnostic   # show the set is identical w/o a TF
No emoji (Windows cp1252).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from framework.discovery_contract import canonicalize_grid  # the new callable

import yaml

CATALOG = ROOT.parent / "config" / "ti_master_catalog.yaml"


# ---- raw parameter-grid generators (the "standard granularity" of the sweep) -----------------
def _single_length():
    return [(L,) for L in range(2, 201)]                      # period 2..200


def _ma_cross():
    return [(f, s) for f in range(2, 121) for s in range(f + 1, 251)]   # fast<slow


def _ma_triple():
    # form triples from a fine log-spaced candidate-length set (bounds the combinatorics)
    cand, L = [], 2.0
    while L <= 250:
        cand.append(int(round(L))); L *= 1.12
    cand = sorted(set(cand))
    return [(a, b, c) for i, a in enumerate(cand) for j, b in enumerate(cand[i + 1:], i + 1)
            for c in cand[j + 1:]]


def _macd():
    return [(f, s, sig) for f in range(4, 25) for s in range(15, 55) for sig in range(4, 16) if f < s]


def _bollinger_2p():
    mults = [round(1.0 + 0.1 * k, 1) for k in range(0, 31)]   # 1.0..4.0
    return [(p, m) for p in range(5, 61) for m in mults]


def _stoch_2p():
    return [(k, d) for k in range(3, 31) for d in range(2, 13)]


def _supertrend_2p():
    mults = [round(1.0 + 0.25 * k, 2) for k in range(0, 17)]  # 1.0..5.0
    return [(p, m) for p in range(5, 41) for m in mults]


def _fixed():
    return [(1,)]


KIND_GEN = {
    "single_length": _single_length, "ma_cross": _ma_cross, "ma_triple": _ma_triple,
    "macd": _macd, "bollinger_2p": _bollinger_2p, "stoch_2p": _stoch_2p,
    "supertrend_2p": _supertrend_2p, "fixed": _fixed,
}

# indicator name -> kind (default single_length). Only the non-default ones are listed.
KIND_OF = {
    # trend
    "SMA": "ma_cross", "EMA": "ma_cross", "WMA": "ma_cross", "HMA": "ma_cross", "DEMA": "ma_cross",
    "TEMA": "ma_cross", "KAMA": "ma_cross", "ZLMA": "ma_cross", "ALMA": "ma_cross",
    "MACD": "macd", "Supertrend": "supertrend_2p",
    # momentum
    "Stochastic": "stoch_2p", "StochRSI": "stoch_2p", "AwesomeOsc": "ma_cross", "UltimateOsc": "ma_triple",
    # volatility
    "Bollinger": "bollinger_2p", "Keltner": "bollinger_2p",
    # volume
    "VWAP": "fixed", "VolumeProfilePOC": "fixed",
    # structure
    "PivotPoints": "fixed", "Fibonacci": "fixed", "PriorPeriodHL": "fixed",
}


def decompose(rel_tol: float = 0.15) -> dict:
    cat = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    fams = [k for k in cat if k != "meta" and isinstance(cat[k], list)]
    # compute distinct count per kind ONCE (timeframe-agnostic; same for every indicator of that kind)
    kind_stat = {}
    for kind, gen in KIND_GEN.items():
        raw = gen()
        c = canonicalize_grid(raw, rel_tol)
        kind_stat[kind] = {"n_raw": c.n_raw, "n_distinct": c.n_effective, "reps_sample": c.representatives[:8]}
    out = {"rel_tol": rel_tol, "kind_stat": kind_stat, "families": {}, "total_distinct": 0,
           "total_indicators": 0}
    for fam in fams:
        inds = []
        fam_distinct = 0
        for e in cat[fam]:
            name = e.get("name")
            kind = KIND_OF.get(name, "single_length")
            d = kind_stat[kind]["n_distinct"]
            inds.append({"name": name, "kind": kind, "distinct_configs": d,
                         "dead": e.get("dead"), "have": e.get("have")})
            fam_distinct += d
        out["families"][fam] = {"n_indicators": len(inds), "family_distinct_configs": fam_distinct,
                                "indicators": inds}
        out["total_distinct"] += fam_distinct
        out["total_indicators"] += len(inds)
    return out


def render(out: dict) -> str:
    L = []
    rt = out["rel_tol"]
    L.append(f"## TI CONFIG DECOMPOSITION -- distinct configs per strat/family (canonicalize_grid, rel_tol={rt})")
    L.append("")
    L.append("PER CONFIG-KIND (raw sweep -> DISTINCT after near-dup elimination; TIMEFRAME-AGNOSTIC):")
    L.append(f"   {'kind':16} {'raw':>7} {'distinct':>9} {'compression':>12}")
    for k, s in out["kind_stat"].items():
        comp = 100.0 * (1 - s["n_distinct"] / s["n_raw"]) if s["n_raw"] else 0.0
        L.append(f"   {k:16} {s['n_raw']:>7} {s['n_distinct']:>9} {comp:>11.0f}%")
    L.append("")
    L.append("PER FAMILY (sum of each indicator's distinct configs at its natural parameterization):")
    L.append(f"   {'family':20} {'#ind':>5} {'distinct configs':>16}")
    for fam, fd in out["families"].items():
        L.append(f"   {fam:20} {fd['n_indicators']:>5} {fd['family_distinct_configs']:>16}")
    L.append(f"   {'-'*44}")
    L.append(f"   {'TOTAL':20} {out['total_indicators']:>5} {out['total_distinct']:>16}")
    L.append("")
    L.append("INTERPRETATION:")
    L.append("  - The DISTINCT-config set is TIMEFRAME-AGNOSTIC (canonicalize_grid sees params, not data).")
    L.append("  - Per-timeframe is only: (1) wall-clock meaning of a length, (2) which distinct config WINS")
    L.append("    (PER_CADENCE_CONFIG: 1d->EMA50/100, 4h->EMA50/200) -- same set, different pick.")
    return "\n".join(L)


def prove_tf_agnostic(rel_tol: float = 0.15) -> str:
    """Demonstrate the set does not depend on a timeframe: canonicalize the SAME grid 'as if' for each
    cadence -- identical output every time (there is no cadence input)."""
    raw = _ma_cross()
    sets = {}
    for tf in ("1d", "4h", "1h", "15m"):
        c = canonicalize_grid(raw, rel_tol)   # no tf argument exists -> identical by construction
        sets[tf] = (c.n_effective, tuple(c.representatives))
    distinct_n = {tf: v[0] for tf, v in sets.items()}
    all_same = len({v[1] for v in sets.values()}) == 1
    return (f"prove-tf-agnostic (ma_cross, rel_tol={rel_tol}): distinct-N per cadence = {distinct_n}; "
            f"ALL CADENCES IDENTICAL = {all_same}  -> the distinct-config SET is timeframe-agnostic.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.ti_config_decompose")
    ap.add_argument("--rel-tol", type=float, default=0.15)
    ap.add_argument("--prove-tf-agnostic", action="store_true")
    ap.add_argument("--json", action="store_true", default=True)
    a = ap.parse_args(argv)
    if a.prove_tf_agnostic:
        print(prove_tf_agnostic(a.rel_tol))
        return 0
    out = decompose(a.rel_tol)
    print(render(out))
    print("")
    print(prove_tf_agnostic(a.rel_tol))
    if a.json:
        outdir = ROOT.parent / "runs" / "strat"
        outdir.mkdir(parents=True, exist_ok=True)
        p = outdir / f"ti_config_decompose_reltol{a.rel_tol}.json"
        json.dump(out, open(p, "w", encoding="utf-8"), indent=2, default=str)
        print(f"\n[persisted] {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
