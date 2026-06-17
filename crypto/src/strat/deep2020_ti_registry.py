"""src/strat/deep2020_ti_registry.py -- the CANONICAL TI store: per-TI performance profile (all layers).

User /orc 2026-06-15: "store all of our discoveries canonically -- raw runs, ironed runs, mechanical fixes,
top-10 per TI within a family, + all info to know the performance profile of each TI."

Reads every layer and emits ONE canonical store:
  - RAW (base config) best per TF                  <- ti_<IND>_<cads>.json rows[].base
  - IRONED best per TF (+ xBH, drift, robust)       <- ti_<IND>_<cads>.json rows[].iron
  - RECOVERED (fine-TF mechanical fighter) per TF   <- ti_recover_*.json best_robust
  - TOP-10 ironed configs per TF                    <- ti_<IND>_<cads>.json
  - PROFILE: family, best deployable, deployable band, ceiling-xBH, robust-frac, verdict
Outputs ti_registry.json (machine) + TI_REGISTRY.md (human dossier, one card per TI).
2020 fixed-EW long-only. RWYB: python -m strat.deep2020_ti_registry. No emoji.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
FAMILY = {"MACD": "trend", "SUPERTREND": "trend", "PSAR": "trend", "VORTEX": "trend", "ADX": "trend",
          "ROC": "momentum", "TSI": "momentum", "DONCHIAN": "breakout", "KELTNER": "breakout",
          "RSI": "mean-reversion", "STOCH": "mean-reversion", "BBPCT": "mean-reversion", "CCI": "mean-reversion",
          "WILLR": "mean-reversion", "OBV": "volume", "MFI": "volume", "VOLIMB": "volume", "CMF": "volume"}
FAM_ORDER = ["trend", "momentum", "breakout", "mean-reversion", "volume"]
CADS = ["1d", "4h", "2h", "1h", "30m", "15m"]


def _load():
    raw, recov = {}, {}                                          # ind -> cad -> cell ; ind|cad -> recover
    bh = {}
    for jf in BASE.glob("ti_*.json"):
        st = jf.stem
        if st == "ti_top10" or st == "ti_registry":
            continue
        if st.startswith("ti_recover"):
            for k, v in json.load(open(jf)).items():
                recov[k] = v
            continue
        if st.startswith("ti_fighter") or st.startswith("ti_15m_fighter"):
            continue
        ind = st.split("_")[1]
        for cad, cell in json.load(open(jf)).items():
            if isinstance(cell, dict) and cell.get("rows"):
                raw.setdefault(ind, {})[cad] = cell
                bh[cad] = cell["buyhold"]["net"]
    return raw, recov, bh


def main() -> int:
    raw, recov, bh = _load()
    if not raw:
        print("no ti_*.json"); return 1
    reg = {}
    for ind in [i for i in FAMILY if i in raw]:
        fam = FAMILY[ind]
        rawb, ironb, recb, top10 = {}, {}, {}, {}
        for cad in CADS:
            cell = raw[ind].get(cad)
            if not cell:
                continue
            rows = cell["rows"]; b = cell["buyhold"]["net"]
            br = max(rows, key=lambda r: r["base"]["net"])
            rawb[cad] = {"cfg": br["cfg"], "net": br["base"]["net"], "sharpe": br["base"]["sharpe"], "maxdd": br["base"]["maxdd"]}
            rob = [r for r in rows if r["iron"]["robust"]]
            ib = max(rob if rob else rows, key=lambda r: r["iron"]["net"])
            ironb[cad] = {"cfg": ib["cfg"], "net": ib["iron"]["net"], "xBH": round(ib["iron"]["net"] / b, 2) if b else None,
                          "sharpe": ib["iron"]["sharpe"], "maxdd": ib["iron"]["maxdd"], "drift": ib["iron"]["drift"],
                          "robust": ib["iron"]["robust"], "robust_frac": round(len(rob) / len(rows), 2)}
            top10[cad] = [{"cfg": r["cfg"], "net": r["iron"]["net"], "sharpe": r["iron"]["sharpe"],
                           "robust": r["iron"]["robust"]} for r in sorted(rows, key=lambda r: -r["iron"]["net"])[:10]]
        # RECOVERED (fine-TF fighter) -- INDEPENDENT of raw cell (the recovery sweep covers TIs with no raw fine-TF run)
        for cad in ("15m", "30m"):
            rc = recov.get(f"{ind}|{cad}")
            if rc and rc.get("best_robust"):
                f = rc["best_robust"]["fighter"]; b = bh.get(cad)
                recb[cad] = {"cfg": rc["best_robust"]["cfg"], "net": f["net"], "sharpe": f["sharpe"],
                             "maxdd": f["maxdd"], "drift": f["drift"], "xBH": round(f["net"] / b, 2) if b else None,
                             "n_robust": rc.get("n_robust"), "n_total": rc.get("n_total")}
        # profile -- combine IRONED (1d-1h) + RECOVERED (fine-TF fighter) for the deployable picture
        cand = {}                                                # (tf, layer) -> cell (robust + positive only)
        for c, v in ironb.items():
            if v["robust"] and v["net"] > 0:
                cand[(c, "ironed")] = v
        for c, v in recb.items():                                # recovered best_robust is already robust+positive
            cand[(c, "recovered")] = v
        bestk = max(cand, key=lambda k: cand[k]["net"]) if cand else None
        best = cand[bestk] if bestk else None
        best_tf = f"{bestk[0]} ({bestk[1]})" if bestk else None
        ceiling = max((v["xBH"] for v in cand.values() if v.get("xBH") is not None), default=None)
        deployable = [f"{k[0]}/{k[1]}" for k, v in cand.items() if v.get("xBH") and v["xBH"] >= 0.5]
        best_ironed_tf = max(ironb, key=lambda c: ironb[c]["net"]) if ironb else None   # plain TF for top-10 lookup
        reg.setdefault(fam, {})[ind] = {
            "family": fam, "raw_best_by_tf": rawb, "ironed_best_by_tf": ironb,
            "recovered_best_by_tf": recb, "top10_ironed_by_tf": top10,
            "profile": {"best_deployable_loc": best_tf, "best_deployable": best, "ceiling_xBH": ceiling,
                        "deployable_band": deployable, "best_ironed_tf": best_ironed_tf,
                        "verdict": ("de-risked beta (<=buy-hold; iron cuts DD+robustifies)" if fam != "mean-reversion"
                                    else "weak-but-robust diversifier (<=0.5x BH, lowest DD, most robust)")}}

    json.dump(reg, open(BASE / "ti_registry.json", "w"), indent=1, default=str)

    # ---- human dossier ----
    L = ["# TI REGISTRY -- canonical performance profile of every technical indicator (2020)\n"]
    L.append("The single canonical store of all TI discoveries: RAW (base) / IRONED / RECOVERED (fine-TF "
             "mechanical fighter) best per TF, TOP-10 ironed per TF, + the deployable profile. 2020 band, "
             "fixed-EW, long-only spot, VAL Jul-Sep / OOS Oct-Dec. ALL **[VERIFIED-backtest, IN-SAMPLE 2020 "
             "OOS]**. net = OOS compound; xBH = vs same-cadence buy-hold; robust = |drift|<=10 (both windows). "
             "Layers: deep2020_ti_pipeline (raw/ironed) + deep2020_ti_recover (fighter) + this registry.\n")
    L.append("**Buy-hold net% per TF:** 1d 47.4 / 4h 47.8 / 2h 50.2 / 1h 51.6 / 30m 53.2 / 15m 54.8.\n")
    for fam in FAM_ORDER:
        if fam not in reg:
            continue
        L.append(f"\n# {fam.upper()} family\n")
        for ind, d in reg[fam].items():
            pr = d["profile"]
            bi = pr["best_deployable"]
            L.append(f"## {ind}")
            L.append(f"- **profile:** best DEPLOYABLE (ironed or fighter-recovered) = {bi['cfg'] if bi else '--'} @ "
                     f"{pr['best_deployable_loc']} (net {bi['net'] if bi else '--'}%, {bi['xBH'] if bi else '--'}xBH, "
                     f"Sh {bi['sharpe'] if bi else '--'}, maxDD {bi['maxdd'] if bi else '--'}); ceiling "
                     f"{pr['ceiling_xBH']}xBH; deployable cells {pr['deployable_band'] or '(none robust>=0.5x)'}; "
                     f"verdict: {pr['verdict']}.")
            L.append(f"\n  | TF | RAW best (net/Sh/DD) | IRONED best (net/xBH/Sh/DD/drift/R) | RECOVERED fine-TF (net/Sh/DD) | robust-frac |")
            L.append(f"  |---|---|---|---|---|")
            for cad in CADS:
                rb = d["raw_best_by_tf"].get(cad); ib = d["ironed_best_by_tf"].get(cad); rc = d["recovered_best_by_tf"].get(cad)
                if not ib and not rc:
                    continue
                rs = f"{rb['cfg']} {rb['net']}/{rb['sharpe']}/{rb['maxdd']}" if rb else "--"
                isr = (f"{ib['cfg']} {ib['net']}/{ib['xBH']}x/{ib['sharpe']}/{ib['maxdd']}/{ib['drift']}/{'R' if ib['robust'] else '-'}"
                       if ib else "-- (no raw run at this TF)")
                rcs = (f"{rc['cfg']} {rc['net']}/{rc['xBH']}x/{rc['sharpe']}/{rc['maxdd']} ({rc['n_robust']}/{rc['n_total']} rob)"
                       if rc else ("--" if cad not in ("15m", "30m") else "(none)"))
                rf = ib["robust_frac"] if ib else "--"
                L.append(f"  | {cad} | {rs} | {isr} | {rcs} | {rf} |")
            # top-10 at the best TF
            bt = pr["best_ironed_tf"]
            if bt and bt in d["top10_ironed_by_tf"]:
                t10 = d["top10_ironed_by_tf"][bt]
                L.append(f"\n  - **top-10 ironed @ {bt}:** " + " | ".join(
                    f"{r['cfg']}({r['net']}%{'R' if r['robust'] else ''})" for r in t10))
            L.append("")
    (BASE / "TI_REGISTRY.md").write_text("\n".join(L), encoding="utf-8")
    nti = sum(len(v) for v in reg.values())
    nrec = sum(1 for fam in reg.values() for d in fam.values() if d["recovered_best_by_tf"])
    print(f"[json] {BASE / 'ti_registry.json'}\n[md] {BASE / 'TI_REGISTRY.md'}  ({nti} TIs, {nrec} with recovered layer)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
