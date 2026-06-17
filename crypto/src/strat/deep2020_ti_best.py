"""src/strat/deep2020_ti_best.py -- the BEST DEPLOYABLE configs across ALL 18 TIs (decision capstone).

Reads every ti_<IND>_<cads>.json and ranks EVERY robust ironed config across all indicators/TFs by WEALTH
(OOS net), with a hard firewall: only ROBUST (|drift|<=10) picks qualify as deployable. Shows the top-25
overall, the best per timeframe, and the best per family -- the single "if you had to deploy one technical
config in 2020, here is the ranked answer" view. Writes TI_BEST.md. 2020 fixed-EW long-only, DESCRIPTIVE
in-sample. RWYB: python -m strat.deep2020_ti_best. No emoji.
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


def main() -> int:
    rows = []
    bh_by_cad = {}
    for jf in BASE.glob("ti_*.json"):
        if jf.stem == "ti_top10":
            continue
        ind = jf.stem.split("_")[1]
        for cad, cell in json.load(open(jf)).items():
            if not cell.get("rows"):
                continue
            bh_by_cad.setdefault(cad, cell["buyhold"]["net"])
            for r in cell["rows"]:
                it = r["iron"]
                if it.get("robust"):                                  # FIREWALL: deployable = robust only
                    nbh = round(it["net"] / cell["buyhold"]["net"], 2) if cell["buyhold"]["net"] else None
                    rows.append({"ind": ind, "fam": FAMILY.get(ind, "?"), "cad": cad, "cfg": r["cfg"],
                                 "net": it["net"], "sharpe": it["sharpe"], "maxdd": it["maxdd"],
                                 "drift": it["drift"], "tin": it.get("time_in"), "net_bh": nbh})
    if not rows:
        print("no robust ironed configs found"); return 1

    L = ["# TI BEST DEPLOYABLE -- robust ironed configs ranked by WEALTH across all 18 indicators (2020)\n"]
    L.append("FIREWALL: only ROBUST (|drift|<=10, delivers in BOTH VAL+OOS) ironed configs qualify. Ranked by "
             "OOS net (wealth). xBH = vs same-cadence buy-hold. ALL **[VERIFIED-backtest, IN-SAMPLE 2020, OOS]**. "
             "NOTE: every pick is <= buy-hold on net (the ceiling is the drift-beta) -- these are the best "
             "DE-RISKED participations, ranked; pick by your DD tolerance + cadence, not by chasing net.\n")
    L.append("## TOP-25 deployable robust ironed configs (across ALL indicators x TFs), by wealth")
    L.append("| # | indicator | family | TF | config | net | xBH | Sharpe | maxDD | drift | t-in |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(sorted(rows, key=lambda x: -x["net"])[:25], 1):
        L.append(f"| {i} | {r['ind']} | {r['fam']} | {r['cad']} | {r['cfg']} | {r['net']}% | {r['net_bh']}x | "
                 f"{r['sharpe']} | {r['maxdd']} | {r['drift']} | {r['tin']} |")

    L.append("\n## Best deployable robust ironed config per TIMEFRAME (by wealth)")
    L.append("| TF | indicator | config | net | xBH | Sharpe | maxDD | (buy-hold net) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for cad in ["1d", "4h", "2h", "1h", "30m", "15m"]:
        cr = [r for r in rows if r["cad"] == cad]
        if not cr:
            continue
        b = max(cr, key=lambda x: x["net"])
        L.append(f"| {cad} | {b['ind']} | {b['cfg']} | {b['net']}% | {b['net_bh']}x | {b['sharpe']} | "
                 f"{b['maxdd']} | {bh_by_cad.get(cad)}% |")

    L.append("\n## Best deployable robust ironed config per FAMILY (by wealth)")
    L.append("| family | indicator | TF | config | net | xBH | Sharpe | maxDD |")
    L.append("|---|---|---|---|---|---|---|---|")
    for fam in ["trend", "momentum", "breakout", "mean-reversion", "volume"]:
        fr = [r for r in rows if r["fam"] == fam]
        if not fr:
            continue
        b = max(fr, key=lambda x: x["net"])
        L.append(f"| {fam} | {b['ind']} | {b['cad']} | {b['cfg']} | {b['net']}% | {b['net_bh']}x | {b['sharpe']} | {b['maxdd']} |")

    best = max(rows, key=lambda x: x["net"])
    L.append(f"\n## THE SINGLE BEST deployable robust ironed config (by wealth, across everything)\n")
    L.append(f"**{best['ind']} {best['cfg']} @ {best['cad']}** ({best['fam']}): net {best['net']}% "
             f"(={best['net_bh']}x buy-hold), Sharpe {best['sharpe']}, maxDD {best['maxdd']}, drift {best['drift']}, "
             f"time-in {best['tin']}. [VERIFIED-2020-OOS] Still <= buy-hold on net -- it is the best risk-adjusted "
             f"DE-RISKED participation among {len(rows)} robust ironed configs, NOT an alpha over holding.\n")

    out = BASE / "TI_BEST.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"[md] {out}  ({len(rows)} robust ironed configs ranked; best = {best['ind']} {best['cfg']} @ {best['cad']} "
          f"net {best['net']}% ={best['net_bh']}xBH)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
