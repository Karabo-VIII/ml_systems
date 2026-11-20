"""src/strat/deep2020_ti_top10.py -- TOP-10 configs per indicator x TF, BASE vs IRONED, wealth-ranked.

The explicit "top 10 selection (base config vs ironed out config)" deliverable for the TI families. Reads
ti_<IND>_<cads>.json and prints, per (indicator, TF), the 10 best configs by WEALTH (ironed OOS net), each with
its BASE and IRONED metrics side by side + robust flag. Writes ti_top10.json. 2020 fixed-EW long-only,
DESCRIPTIVE in-sample. RWYB: python -m strat.deep2020_ti_top10. No emoji (cp1252).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
ORDER = ["MACD", "SUPERTREND", "PSAR", "ROC", "DONCHIAN", "KELTNER", "RSI", "STOCH", "BBPCT", "CCI", "WILLR", "OBV", "MFI", "VOLIMB", "CMF", "VORTEX", "ADX", "TSI"]
CADS = ["1d", "4h", "2h", "1h", "30m", "15m"]


def main() -> int:
    data = {}
    for jf in BASE.glob("ti_*.json"):
        ind = jf.stem.split("_")[1]
        for cad, payload in json.load(open(jf)).items():
            data.setdefault(ind, {})[cad] = payload
    if not data:
        print("no ti_*.json found"); return 1

    print("# TOP-10 configs per indicator x TF -- BASE vs IRONED, WEALTH-ranked (2020 fixed-EW, long-only)\n")
    print("Ranked by IRONED OOS net (wealth). cols: config | BASE net/Sh/maxDD | IRONED net/Sh/maxDD/drift/robust.")
    print("Robust := |drift|<=10 (delivers in BOTH VAL+OOS). DESCRIPTIVE in-sample-2020.\n")
    export = {}
    for ind in [i for i in ORDER if i in data]:
        for cad in CADS:
            cell = data[ind].get(cad)
            if not cell or not cell.get("rows"):
                continue
            bh = cell["buyhold"]; rows = sorted(cell["rows"], key=lambda r: -r["iron"]["net"])[:10]
            print(f"## {ind} @ {cad}   (buy-hold net {bh['net']}%, maxDD {bh['maxdd']}%)")
            print(f"   {'#':>2} {'config':22} {'BASE n/Sh/DD':>18} {'IRONED n/Sh/DD/dr':>22} {'R':>2} {'xBH':>5}")
            for i, r in enumerate(rows, 1):
                b, ir = r["base"], r["iron"]
                nbh = round(ir["net"] / bh["net"], 2) if bh["net"] else None
                print(f"   {i:>2} {r['cfg']:22} {str(b['net'])+'/'+str(b['sharpe'])+'/'+str(b['maxdd']):>18} "
                      f"{str(ir['net'])+'/'+str(ir['sharpe'])+'/'+str(ir['maxdd'])+'/'+str(ir['drift']):>22} "
                      f"{'R' if ir['robust'] else '-':>2} {str(nbh):>5}")
            export[f"{ind}|{cad}"] = [{"cfg": r["cfg"], "base": r["base"], "iron": r["iron"]} for r in rows]
            print()
    out = BASE / "ti_top10.json"
    json.dump(export, open(out, "w"), indent=1, default=str)
    print(f"[json] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
