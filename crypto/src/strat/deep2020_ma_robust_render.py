"""src/strat/deep2020_ma_robust_render.py -- render the ROBUST vs NON-ROBUST split, ranked by WEALTH.

Reads ma_robust_*.json and, per MA type (pooled across cadences), shows TWO kept lists:
  ROBUST     (|drift|<=10: consistent VAL<->OOS) ranked by WEALTH = OOS compound net (the binding objective)
  NON-ROBUST (kept, labeled LUCKY if OOS>>VAL else OVERFIT if VAL>>OOS) ranked by OOS net too
plus the WEALTH-vs-SHARPE contrast (the wealth-leader is NOT the Sharpe-leader) and the wealth-ROBUST champion
per MA = max worst-window min(VAL,OOS) (the most wealth you'd have made in your WORSE window). DESCRIPTIVE,
in-sample-2020. RWYB: python -m strat.deep2020_ma_robust_render. No emoji (cp1252).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
MA_TYPES = ["EMA", "SMA", "WMA", "HMA", "DEMA", "TEMA", "KAMA", "VIDYA"]
CADS = ["1d", "4h", "2h", "1h"]
TOPN = 8


def _row(r):
    return (f"   {r['cad']:>3} {r['cfg']:20} {r['val_net']:>7} {r['net']:>7} {r['worst']:>7} "
            f"{r['sharpe']:>7} {r['maxdd']:>7} {r['drift']:>7}")


def main() -> int:
    data = {}
    for jf in BASE.glob("ma_robust_*.json"):
        for k, v in json.load(open(jf)).items():
            data[k] = v
    if not data:
        print("no ma_robust_*.json found"); return 1

    print("# ROBUST vs NON-ROBUST configs per MA type -- ranked by WEALTH (OOS compound), 2020 (in-sample)\n")
    print("OBJECTIVE = WEALTH (held-out compound net), NOT Sharpe. ROBUST := |drift|<=10 (delivers in BOTH")
    print("VAL+OOS). worst = min(VAL,OOS) = compound in your WORSE window (the wealth-robust metric). Both lists")
    print("KEPT. NON-ROBUST tagged LUCKY (OOS>>VAL, e.g. the Nov quarter) or OVERFIT (VAL>>OOS, fragile).\n")

    hdr = f"   {'TF':>3} {'config':20} {'VALnet':>7} {'OOSnet':>7} {'worst':>7} {'Sharpe':>7} {'maxDD':>7} {'drift':>7}"
    champs = []
    for mt in MA_TYPES:
        recs = []
        for cad in CADS:
            for r in data.get(f"{cad}|{mt}", []):
                recs.append({**r, "cad": cad})
        if not recs:
            continue
        robust = sorted([r for r in recs if r["robust"]], key=lambda r: -r["net"])
        nonrob = sorted([r for r in recs if not r["robust"]], key=lambda r: -r["net"])
        print(f"## {mt}  ({len(robust)} robust / {len(nonrob)} non-robust)")
        print(f"   -- ROBUST, top {TOPN} by WEALTH (OOS net) --")
        print(hdr)
        for r in robust[:TOPN]:
            print(_row(r))
        print(f"   -- NON-ROBUST, top {TOPN} by WEALTH (kept; tag) --")
        print(hdr + "   tag")
        for r in nonrob[:TOPN]:
            tag = "LUCKY" if r["drift"] > 0 else "OVERFIT"
            print(_row(r) + f"   {tag}")
        # wealth vs sharpe (within robust), and the worst-window champ
        if robust:
            w_lead = robust[0]
            s_lead = max(robust, key=lambda r: r["sharpe"])
            wr_champ = max(robust, key=lambda r: r["worst"])
            champs.append((mt, w_lead, s_lead, wr_champ))
            print(f"   -> WEALTH-leader {w_lead['cad']} {w_lead['cfg']} (net {w_lead['net']}, Sh {w_lead['sharpe']}) "
                  f"vs SHARPE-leader {s_lead['cad']} {s_lead['cfg']} (Sh {s_lead['sharpe']}, net {s_lead['net']})")
        print()

    print("=" * 96)
    print("## WEALTH vs SHARPE -- the deployable ROBUST pick per MA type (they DIFFER):")
    print(f"   {'MA':6} {'WEALTH-leader (max OOS net, robust)':45} {'SHARPE-leader (max Sh, robust)':40}")
    for mt, w, s, _ in champs:
        wl = f"{w['cad']} {w['cfg']} net {w['net']}% Sh {w['sharpe']}"
        sl = f"{s['cad']} {s['cfg']} Sh {s['sharpe']} net {s['net']}%"
        print(f"   {mt:6} {wl:45} {sl:40}")

    print("\n## WEALTH-ROBUST champion per MA type (max worst-window min(VAL,OOS) -- most wealth in the WORSE window):")
    for mt, _, _, c in champs:
        print(f"   {mt:6} -> {c['cad']:>3} {c['cfg']:20} worst {c['worst']}%  (VAL {c['val_net']} / OOS {c['net']}, "
              f"Sh {c['sharpe']}, maxDD {c['maxdd']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
