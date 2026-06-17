"""src/strat/deep2020_family.py -- BLOCK C: the FAMILY-OF-STRATS thesis ('one config can't win them all').

User /orc 2026-06-13: "one config can't win them all, so a family of strats per timeframe might be the
winner or per instrument." This tests it rigorously. For each timeframe (and per instrument) it compares:
  BEST-SINGLE   the config picked on VAL (causal) -> its OOS                (the 'bet on one' approach)
  AVG-SINGLE    the mean OOS across all configs                            (an average single bet)
  FAMILY        the equal-weight book of ALL configs                       (the user's thesis)
on net%, Sharpe, maxDD, and -- the why -- the DIVERSIFICATION MATH: mean pairwise config correlation,
diversification ratio (avg-config-vol / family-vol), effective N. The thesis wins if the FAMILY has higher
Sharpe + lower maxDD + more robust VAL->OOS than betting on one config (which overfits VAL / can collapse).
Also a DIVERSITY LADDER: 1 cfg -> config-family(1 MA type) -> multi-MA family (all 8 types) -- does MORE
diversity help? RWYB: python -m strat.deep2020_family --cadences <tf>. No emoji (cp1252).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR
from strat.replay_distinct_grid import distinct_specs
from strat.ma_type_upgrade import _nums, MA_TYPES
import strat.ma_2020_breakdown as B

SPLIT = B.SPLIT
VAL = B.SPLIT["VAL"]; OOS = B.SPLIT["OOS"]
CADENCES = ["1d", "4h", "2h", "1h", "30m", "15m"]
ANN = {"1d": 365, "4h": 365 * 6, "2h": 365 * 12, "1h": 365 * 24, "30m": 365 * 48, "15m": 365 * 96}


def _cfg_books(cells, names):
    """per-config book = mean across assets -> {name: daily-return Series}."""
    books = {}
    for name in names:
        cols = [v for (c, s), v in cells.items() if c == name]
        if cols:
            b4 = pd.concat(cols, axis=1).mean(axis=1, skipna=True)
            books[name] = b4.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()
    return books


def _comp(s, lo, hi):
    x = s[(s.index >= lo) & (s.index < hi)]
    return float(np.prod(1 + x.to_numpy()) - 1) * 100 if len(x) else np.nan


def _sharpe(s, lo, hi, cad):
    x = s[(s.index >= lo) & (s.index < hi)].to_numpy()
    return float(np.mean(x) / (np.std(x) + 1e-12) * np.sqrt(365)) if len(x) > 3 else np.nan


def _maxdd(s, lo, hi):
    x = s[(s.index >= lo) & (s.index < hi)].to_numpy()
    if len(x) < 3:
        return np.nan
    eq = np.cumprod(1 + x); pk = np.maximum.accumulate(eq); return float(((eq - pk) / pk).min() * 100)


def main() -> int:
    global CADENCES
    if "--cadences" in sys.argv:
        CADENCES = sys.argv[sys.argv.index("--cadences") + 1].split(",")
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=60))
    PR.STRATS.update(ma_cfg)
    slow = [n for n in ma_cfg if 60 <= max(_nums(n)) < 150]
    print(f"BLOCK C family thesis: {len(slow)} configs x {len(MA_TYPES)} MA types x {len(CADENCES)} TF; VAL->OOS 2020\n")

    out = {}
    for cad in CADENCES:
        # build per-config books for EACH MA type (config-family), pooled across assets
        all_books = {}   # (ma_type, name) -> daily series
        for mt in MA_TYPES:
            cells = B._cells(slow, mt, cad)
            for name, s in _cfg_books(cells, slow).items():
                all_books[(mt, name)] = s
        if not all_books:
            continue
        idx = sorted(set().union(*[s.index for s in all_books.values()]))
        M = pd.DataFrame({k: s for k, s in all_books.items()}).reindex(idx).fillna(0.0)

        def stats(cols, label):
            sub = M[cols]
            fam = sub.mean(axis=1)
            # best single by VAL, avg single
            valc = {c: _comp(M[c], *VAL) for c in cols}
            best = max(valc, key=valc.get)
            best_oos = _comp(M[best], *OOS)
            avg_oos = float(np.nanmean([_comp(M[c], *OOS) for c in cols]))
            fam_oos = _comp(fam, *OOS)
            # diversification math on OOS window
            sub_oos = sub[(sub.index >= pd.Timestamp(OOS[0])) & (sub.index < pd.Timestamp(OOS[1]))]
            vols = sub_oos.std().to_numpy()
            corr = sub_oos.corr().to_numpy()
            mean_corr = float((corr.sum() - len(cols)) / (len(cols) * (len(cols) - 1))) if len(cols) > 1 else 1.0
            fam_vol = float(fam[(fam.index >= pd.Timestamp(OOS[0])) & (fam.index < pd.Timestamp(OOS[1]))].std())
            div_ratio = float(np.mean(vols) / (fam_vol + 1e-12))
            eff_n = 1.0 / (mean_corr + (1 - mean_corr) / len(cols)) if len(cols) > 1 else 1.0
            r = {"n": len(cols), "best_single_oos": round(best_oos, 1), "avg_single_oos": round(avg_oos, 1),
                 "family_oos": round(fam_oos, 1), "family_sharpe": round(_sharpe(fam, *OOS, cad), 2),
                 "family_maxdd": round(_maxdd(fam, *OOS), 1), "mean_cfg_corr": round(mean_corr, 2),
                 "div_ratio": round(div_ratio, 2), "eff_n": round(eff_n, 1),
                 "family_beats_best": fam_oos >= best_oos, "family_beats_avg": fam_oos >= avg_oos}
            out[(cad, label)] = r
            return r

        print(f"########## {cad} -- FAMILY vs BEST-SINGLE vs AVG-SINGLE (OOS) + diversification math ##########")
        print(f"   {'family':16} {'n':>4} {'best1%':>7} {'avg1%':>7} {'FAMILY%':>8} {'Shrp':>5} {'maxDD':>7} {'corr':>5} {'divR':>5} {'effN':>5}")
        # diversity ladder: EMA-only family -> all-MA family
        ema_cols = [(mt, n) for (mt, n) in all_books if mt == "EMA"]
        all_cols = list(all_books)
        for cols, lab in [(ema_cols, "EMA-family"), (all_cols, "all-MA-family")]:
            r = stats(cols, lab)
            print(f"   {lab:16} {r['n']:>4} {r['best_single_oos']:>7} {r['avg_single_oos']:>7} {r['family_oos']:>8} "
                  f"{r['family_sharpe']:>5} {r['family_maxdd']:>7} {r['mean_cfg_corr']:>5} {r['div_ratio']:>5} {r['eff_n']:>5}")
        print()

    op = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
    op.mkdir(parents=True, exist_ok=True)
    jt = "_".join(CADENCES)
    json.dump({f"{c}|{l}": r for (c, l), r in out.items()}, open(op / f"family_{jt}.json", "w"), indent=1, default=str)
    print(f"[json] {op / f'family_{jt}.json'}")
    print("\nTHESIS CHECK: family_beats_best / family_beats_avg per (cad,family):")
    for (c, l), r in out.items():
        print(f"   {c:4} {l:14} beats_best={r['family_beats_best']}  beats_avg={r['family_beats_avg']}  "
              f"(FAM {r['family_oos']}% vs best1 {r['best_single_oos']}% vs avg1 {r['avg_single_oos']}%; Sharpe {r['family_sharpe']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
