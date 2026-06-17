"""src/strat/deep2020_ma_remedy_render.py -- consolidate the MA-remedy JSONs into the SIDE-BY-SIDE answer.

Reads ma_remedy_*.json (per-cadence remedy ladders) and renders the cross-the-board view that answers the
user's question: "if MA x Timeframe were your ONLY approach, what is the best build with the LEAST weaknesses?"

Recommended least-weakness 2020 build = the "+VOLTGT" variant (= R1 confirm-band + R2 cluster-ensemble +
R4 vol-target). "FULL-REM" (adds the R3 regime gate) is the ALL-WEATHER variant: it costs return in the 2020
bull (no bear to gate out) but is the bear-protection the literature says is the real out-of-sample edge.
Cross-cadence NET is NOT comparable (equal-weight rebalancing premium inflates finer cadences AND their
buy-hold) -> the honest cross-TF metric is net / same-cadence-buyhold (capture ratio) + Sharpe + maxDD + drift.
RWYB: python -m strat.deep2020_ma_remedy_render. No emoji (cp1252).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
MA_TYPES = ["EMA", "SMA", "WMA", "HMA", "DEMA", "TEMA", "KAMA", "VIDYA"]
CADS = ["1d", "4h", "2h", "1h"]
REC = "+VOLTGT"           # the recommended least-weakness 2020 build


def main() -> int:
    data = {}
    for jf in BASE.glob("ma_remedy_*.json"):
        for k, v in json.load(open(jf)).items():
            data[k] = v
    if not data:
        print("no ma_remedy_*.json found"); return 1

    print("# MA x TIMEFRAME -- the LEAST-WEAKNESS build per MA kind (2020, VAL-select / OOS-report, maker)\n")
    print("Recommended build = +VOLTGT (confirm-band + cluster-ensemble + vol-target). net/BH = OOS net vs")
    print("SAME-cadence buy-hold (the only honest cross-TF metric; raw net is rebalancing-inflated at fine TF).\n")

    allrec = []
    for cad in CADS:
        bench = data.get(f"{cad}|_BENCH")
        if not bench:
            continue
        bh = bench["buyhold"]; vbh = bench["voltgt_bh"]
        print(f"########## {cad}   BUYHOLD net {bh['net']}% Sh {bh['sharpe']} DD {bh['maxdd']}%   |   "
              f"VOLTGT_BH net {vbh['net']}% Sh {vbh['sharpe']} DD {vbh['maxdd']}% ##########")
        print(f"   {'MA':5} {'cluster':14} {'net':>6} {'net/BH':>7} {'Sharpe':>7} {'maxDD':>7} {'drift':>7} "
              f"{'turn':>6}  best-variant-by-Sharpe")
        for mt in MA_TYPES:
            cell = data.get(f"{cad}|{mt}")
            if not cell or REC not in cell.get("variants", {}):
                continue
            r = cell["variants"][REC]
            nbh = round(r["net"] / bh["net"], 2) if bh["net"] else None
            bestv = max(cell["variants"], key=lambda v: cell["variants"][v]["sharpe"])
            bsh = cell["variants"][bestv]["sharpe"]
            print(f"   {mt:5} {cell['winning_cluster']:14} {r['net']:>6} {str(nbh):>7} {r['sharpe']:>7} "
                  f"{r['maxdd']:>7} {r['drift']:>7} {r['turnover']:>6}  {bestv}(Sh{bsh})")
            allrec.append({"cad": cad, "mt": mt, "cluster": cell["winning_cluster"], "net": r["net"],
                           "net_bh": nbh, "sharpe": r["sharpe"], "maxdd": r["maxdd"], "drift": r["drift"],
                           "turn": r["turnover"], "bh_net": bh["net"]})
        print()

    # ---- rankings (the answer) ----
    print("=" * 90)
    print("## TOP-8 least-weakness builds by SHARPE (risk-adjusted leader; the +VOLTGT build):")
    for x in sorted(allrec, key=lambda a: -a["sharpe"])[:8]:
        print(f"   {x['mt']:5} {x['cad']:3} {x['cluster']:14} -> Sh {x['sharpe']}  net {x['net']}% "
              f"(={x['net_bh']}xBH)  maxDD {x['maxdd']}%  drift {x['drift']}")

    print("\n## TOP-8 by net/BH CAPTURE (how much of same-cadence buy-hold the de-risked book keeps):")
    for x in sorted(allrec, key=lambda a: -(a["net_bh"] or 0))[:8]:
        print(f"   {x['mt']:5} {x['cad']:3} {x['cluster']:14} -> net/BH {x['net_bh']}x  net {x['net']}% "
              f"Sh {x['sharpe']}  maxDD {x['maxdd']}%  drift {x['drift']}")

    print("\n## MOST ROBUST by |drift| (smallest VAL->OOS give-back = least selection-fragility):")
    for x in sorted(allrec, key=lambda a: abs(a["drift"]))[:8]:
        print(f"   {x['mt']:5} {x['cad']:3} {x['cluster']:14} -> |drift| {abs(x['drift'])}  net {x['net']}% "
              f"Sh {x['sharpe']}  maxDD {x['maxdd']}%")

    print("\n## CHAMPION per MA KIND (best cadence for each MA, by Sharpe; the per-kind least-weakness pick):")
    for mt in MA_TYPES:
        rows = [x for x in allrec if x["mt"] == mt]
        if not rows:
            continue
        b = max(rows, key=lambda a: a["sharpe"])
        print(f"   {mt:5} -> {b['cad']:3} {b['cluster']:14}  Sh {b['sharpe']}  net {b['net']}% "
              f"(={b['net_bh']}xBH)  maxDD {b['maxdd']}%  drift {b['drift']}")

    # overall least-weakness winner: high Sharpe, shallow DD, small drift, decent capture
    def lw(a):
        return a["sharpe"] - 0.04 * abs(a["drift"]) - 0.03 * abs(a["maxdd"])      # transparent composite
    best = max(allrec, key=lw)
    print(f"\n## OVERALL LEAST-WEAKNESS build (Sharpe - 0.04|drift| - 0.03|maxDD|): "
          f"{best['mt']} {best['cad']} {best['cluster']} -- Sh {best['sharpe']}, net {best['net']}% "
          f"(={best['net_bh']}xBH), maxDD {best['maxdd']}%, drift {best['drift']}")
    print("   NOTE: even the best build's net < same-cadence buy-hold -- the remedies MINIMIZE weaknesses")
    print("   (selection-drift, maxDD, whipsaw) but the net CEILING is the drift-beta (buy-hold). The MA book's")
    print("   edge over buy-hold is RISK (Sharpe/maxDD), not return -- exactly the Faber/cross-check verdict.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
