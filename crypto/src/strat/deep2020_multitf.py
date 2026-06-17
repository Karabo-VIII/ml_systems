"""src/strat/deep2020_multitf.py -- BLOCK K: does a family ACROSS TIMEFRAMES diversify? (the other half of C)

User thesis: "a family per timeframe OR PER INSTRUMENT might win." Block C tested a within-TF CONFIG family
-> eff N ~1.2 (no diversification; configs ~0.9 correlated). This tests the orthogonal cut: are the
EMA-family BOOKS at DIFFERENT timeframes (1d/4h/1h/15m) less correlated than configs within one TF? If so,
a MULTI-TIMEFRAME family genuinely diversifies (higher eff N, higher Sharpe) where a config family did not.
Per instrument + pooled, 2020 H2. RWYB: python -m strat.deep2020_multitf. No emoji (cp1252).
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
from strat.ma_type_upgrade import _nums
import strat.ma_2020_breakdown as B

TFS = ["1d", "4h", "1h", "15m"]
OOS = B.SPLIT["OOS"]


def _tf_book(slow, cad):
    """EMA-family book (mean across configs x assets) as a DAILY return series, on 2020 H2."""
    cells = B._cells(slow, "EMA", cad)
    if not cells:
        return None
    b4 = pd.concat(list(cells.values()), axis=1).mean(axis=1, skipna=True)
    return b4.resample("1D").apply(lambda x: float(np.prod(1 + x) - 1)).dropna()


def _sharpe(s):
    x = s.to_numpy(); return float(np.mean(x) / (np.std(x) + 1e-12) * np.sqrt(365))


def main() -> int:
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=60))
    PR.STRATS.update(ma_cfg)
    slow = [n for n in ma_cfg if 60 <= max(_nums(n)) < 150]
    print(f"BLOCK K multi-timeframe family: EMA-family book at {TFS}; 2020 H2\n")

    books = {}
    for cad in TFS:
        b = _tf_book(slow, cad)
        if b is not None:
            books[cad] = b[b.index >= pd.Timestamp(OOS[0])] if False else b
    if len(books) < 2:
        print("not enough TF books"); return 1
    # align on common daily index (OOS window for the corr/sharpe)
    df = pd.DataFrame(books).dropna()
    oos = df[(df.index >= pd.Timestamp(OOS[0])) & (df.index < pd.Timestamp(OOS[1]))]

    print("## cross-TIMEFRAME book correlation (OOS daily returns):")
    corr = oos.corr()
    print("        " + "".join(f"{c:>7}" for c in corr.columns))
    for r in corr.index:
        print(f"   {r:5}" + "".join(f"{corr.loc[r,c]:>7.2f}" for c in corr.columns))
    offdiag = (corr.values.sum() - len(corr)) / (len(corr) * (len(corr) - 1))
    print(f"   mean cross-TF correlation = {offdiag:.2f}  (vs within-TF config corr ~0.90 from Block C)")

    print("\n## single-TF vs MULTI-TF family (OOS):")
    print(f"   {'book':12} {'Sharpe':>7} {'net%':>8}")
    for cad in books:
        o = oos[cad]
        print(f"   {cad:12} {_sharpe(o):>7.2f} {float(np.prod(1+o.to_numpy())-1)*100:>8.1f}")
    multi = oos.mean(axis=1)
    print(f"   {'MULTI-TF':12} {_sharpe(multi):>7.2f} {float(np.prod(1+multi.to_numpy())-1)*100:>8.1f}")
    avg_single_sharpe = float(np.mean([_sharpe(oos[c]) for c in books]))
    eff_n = 1.0 / (offdiag + (1 - offdiag) / len(books))
    print(f"\n   avg single-TF Sharpe {avg_single_sharpe:.2f} -> MULTI-TF Sharpe {_sharpe(multi):.2f} "
          f"(lift {_sharpe(multi)-avg_single_sharpe:+.2f}); effective N across TFs = {eff_n:.1f}")
    verdict = "DIVERSIFIES (cross-TF less correlated than configs)" if offdiag < 0.7 else "barely (TFs still highly correlated -- same drift-beta)"
    print(f"   VERDICT: {verdict}")

    op = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
    op.mkdir(parents=True, exist_ok=True)
    json.dump({"cross_tf_corr": round(float(offdiag), 3), "eff_n_tf": round(float(eff_n), 2),
               "multi_sharpe": round(_sharpe(multi), 2), "avg_single_sharpe": round(avg_single_sharpe, 2)},
              open(op / "multitf_family.json", "w"), indent=1, default=str)
    print(f"\n[json] {op / 'multitf_family.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
