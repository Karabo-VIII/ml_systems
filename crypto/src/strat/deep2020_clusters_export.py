"""src/strat/deep2020_clusters_export.py -- merge per-TF cluster JSONs into the ML-HARNESS TARGET manifest.

Reads clusters_*.json (per TF) and builds: (1) the BEST-IN-CLASS table (per MA type x timeframe: winning
cluster + mean+-std stats + top config); (2) the ML-target manifest -- per (MA, TF) the best cluster +
top-K configs the ML harness should learn to REPLICATE (each config spec regenerates its causal positions
via the existing tools). RWYB: python -m strat.deep2020_clusters_export. No emoji (cp1252).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
MA_TYPES = ["EMA", "SMA", "WMA", "HMA", "DEMA", "TEMA", "KAMA", "VIDYA"]
CADS = ["1d", "4h", "2h", "1h", "30m", "15m"]


def main() -> int:
    data = {}
    for jf in BASE.glob("clusters_*.json"):
        for k, v in json.load(open(jf)).items():
            data[k] = v
    if not data:
        print("no clusters_*.json found"); return 1

    print("# BEST-IN-CLASS per MA type x timeframe (winning cluster + top config) -- 2020 OOS, FULL stack\n")
    print(f"   {'MA':6} " + "".join(f"{c:>20}" for c in CADS))
    manifest = {}
    for mt in MA_TYPES:
        row = f"   {mt:6} "
        for cad in CADS:
            key = f"{cad}|{mt}"
            if key not in data:
                row += f"{'--':>20}"; continue
            cl = data[key]["clusters"]; best = max(cl, key=lambda c: cl[c]["mean_sharpe"])
            top = data[key]["top5"][0]
            row += f"{(best.replace('20-60','').replace('60-150','-sl').replace('150+','-vsl').replace('<20','-f')[:11]+' '+str(top['sharpe'])):>20}"
            manifest[key] = {"best_cluster": best, "cluster_stats": cl[best], "top5": data[key]["top5"],
                             "ml_target_config": top["cfg"]}
        print(row)

    # per-MA winning-cluster summary (which param region wins for each MA type, across TFs)
    print("\n## winning CLUSTER per MA type, per timeframe (struct-speed):")
    print(f"   {'MA':6} " + "".join(f"{c:>14}" for c in CADS))
    for mt in MA_TYPES:
        row = f"   {mt:6} "
        for cad in CADS:
            key = f"{cad}|{mt}"
            if key in manifest:
                row += f"{manifest[key]['best_cluster']:>14}"
            else:
                row += f"{'--':>14}"
        print(row)

    # the top-3 ML targets per (MA, TF) with full stats
    print("\n## ML-TARGET manifest sample (top-3 configs per MA x TF to REPLICATE):")
    for cad in CADS:
        for mt in MA_TYPES:
            key = f"{cad}|{mt}"
            if key not in data:
                continue
            t3 = data[key]["top5"][:3]
            print(f"   {cad:4} {mt:6} -> " + " | ".join(f"{t['cfg']} (Sh{t['sharpe']}/{t['net']}%)" for t in t3))

    out = BASE / "clusters_ML_TARGET.json"
    json.dump(manifest, open(out, "w"), indent=1, default=str)
    print(f"\n[json] {out}  -- the ML-harness target manifest (best cluster + top-5 + ml_target_config per MA x TF)")
    print("ML HARNESS USAGE: for each (MA, TF) the harness learns to REPLICATE ml_target_config's causal positions")
    print("(regenerate via the config spec + FULL stack), or to SELECT within the winning cluster from market state.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
