"""src/strat/deep2020_ti_render.py -- consolidate the TI pipeline JSONs into the cross-indicator view.

Reads ti_<IND>_<cads>.json (per indicator) and renders: per (indicator x TF) the BEST ironed config by WEALTH
+ its base counterpart + the IRON effect (dNet/dMaxDD/d|drift|) + robust count + vs buy-hold; the wealth-ROBUST
champion per indicator; and the per-FAMILY iron-effectiveness verdict. WEALTH (OOS compound) ranked, fixed-EW,
2020. DESCRIPTIVE in-sample. RWYB: python -m strat.deep2020_ti_render. No emoji (cp1252).
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
CADS = ["1d", "4h", "2h", "1h"]
FAMILY = {"MACD": "trend", "SUPERTREND": "trend", "PSAR": "trend", "ROC": "momentum",
          "RSI": "mean-reversion", "STOCH": "mean-reversion", "BBPCT": "mean-reversion",
          "CCI": "mean-reversion", "WILLR": "mean-reversion", "DONCHIAN": "breakout", "KELTNER": "breakout", "OBV": "volume", "MFI": "volume", "VOLIMB": "volume", "CMF": "volume", "VORTEX": "trend", "ADX": "trend", "TSI": "momentum"}


def main() -> int:
    data = {}                                                   # ind -> cad -> {buyhold, voltgt_bh, rows}
    for jf in BASE.glob("ti_*.json"):
        nm = jf.stem                                            # ti_MACD_1d_4h
        parts = nm.split("_")
        ind = parts[1]
        for cad, payload in json.load(open(jf)).items():
            data.setdefault(ind, {})[cad] = payload
    if not data:
        print("no ti_*.json found"); return 1

    print("# TI x TIMEFRAME -- per-config base vs IRONED, WEALTH-ranked (2020 fixed-EW, long-only)\n")
    print("Per (indicator, TF): the BEST ironed config by WEALTH (OOS net), its BASE counterpart, the IRON")
    print("effect, robust count, and net/BH. ROBUST := |drift|<=10. Ceiling reference = same-cadence buy-hold.\n")

    champs = {}
    for ind in [i for i in FAMILY if i in data]:
        print(f"## {ind}  ({FAMILY[ind]})")
        print(f"   {'TF':>3} {'best ironed cfg':22} {'base->IRON net':>16} {'net/BH':>7} {'Sh':>5} {'maxDD':>7} "
              f"{'drift':>6} {'robust':>8}  iron(dNet/dDD/d|drift|)")
        best_overall = None
        for cad in CADS:
            cell = data[ind].get(cad)
            if not cell or not cell.get("rows"):
                continue
            bh = cell["buyhold"]; rows = cell["rows"]
            rob = [r for r in rows if r["iron"]["robust"]]
            pool = rob if rob else rows
            best = max(pool, key=lambda r: r["iron"]["net"])    # WEALTH among robust ironed
            i, b = best["iron"], best["base"]
            nbh = round(i["net"] / bh["net"], 2) if bh["net"] else None
            dN = st.median([r["iron"]["net"] - r["base"]["net"] for r in rows])
            dD = st.median([r["iron"]["maxdd"] - r["base"]["maxdd"] for r in rows])
            dDr = st.median([abs(r["iron"]["drift"]) - abs(r["base"]["drift"]) for r in rows])
            print(f"   {cad:>3} {best['cfg']:22} {str(b['net'])+'->'+str(i['net']):>16} {str(nbh):>7} "
                  f"{i['sharpe']:>5} {i['maxdd']:>7} {i['drift']:>6} {str(len(rob))+'/'+str(len(rows)):>8}  "
                  f"{dN:+.1f}/{dD:+.1f}/{dDr:+.1f}")
            cand = {"cad": cad, **best, "net_bh": nbh}
            if best_overall is None or i["net"] > best_overall["iron"]["net"]:
                best_overall = cand
        champs[ind] = best_overall
        print()

    print("=" * 96)
    print("## WEALTH-ROBUST champion per indicator (best ironed config across TFs):")
    for ind, c in champs.items():
        if not c:
            continue
        i = c["iron"]
        print(f"   {ind:9} {FAMILY[ind]:15} -> {c['cad']:>3} {c['cfg']:22} net {i['net']}% (={c['net_bh']}xBH) "
              f"Sh {i['sharpe']} maxDD {i['maxdd']} drift {i['drift']} {'ROBUST' if i['robust'] else 'NON-ROBUST'}")

    # per-family iron verdict (median iron effect across that family's indicators x TFs)
    print("\n## PER-FAMILY iron effectiveness (median over the family's indicators x TFs):")
    fam_eff = {}
    for ind in data:
        fam = FAMILY.get(ind)
        for cad, cell in data[ind].items():
            if not cell.get("rows"):
                continue
            rows = cell["rows"]
            fam_eff.setdefault(fam, {"dN": [], "dD": [], "dDr": [], "rob": []})
            fam_eff[fam]["dN"].append(st.median([r["iron"]["net"] - r["base"]["net"] for r in rows]))
            fam_eff[fam]["dD"].append(st.median([r["iron"]["maxdd"] - r["base"]["maxdd"] for r in rows]))
            fam_eff[fam]["dDr"].append(st.median([abs(r["iron"]["drift"]) - abs(r["base"]["drift"]) for r in rows]))
            fam_eff[fam]["rob"].append(sum(1 for r in rows if r["iron"]["robust"]) / len(rows))
    for fam, e in fam_eff.items():
        print(f"   {fam:15} dNet {st.median(e['dN']):+.1f}pp  dMaxDD {st.median(e['dD']):+.1f}pp  "
              f"d|drift| {st.median(e['dDr']):+.1f}pp  robust-frac {st.mean(e['rob'])*100:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
