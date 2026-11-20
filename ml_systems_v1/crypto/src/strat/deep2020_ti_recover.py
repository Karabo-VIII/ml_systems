"""src/strat/deep2020_ti_recover.py -- RECOVER the fine-TF strats: full-grid fixed-fighter, best robust per TI.

User /orc 2026-06-15: "recover them" -- apply the proven turnover-fighter (fixed level conf4/hold48/cd48) to
the FULL config grid (not just canonical) at the fine TFs (15m/30m), to find the best ROBUST recovered config
per indicator. This is the "mechanical-fix / recovered runs" layer for the canonical TI store.

For each (indicator, fine TF), every config: ironed signal -> trail10 -> entry-CONFIRM(4) -> min_hold(48) ->
cooldown(48) -> lag -> vol-target -> maker; record net/val_net/drift/Sharpe/maxDD/robust. Per TI: top-10 by
wealth + best ROBUST. fixed-EW, 2020. RWYB: python -m strat.deep2020_ti_recover --cadences 15m,30m. No emoji.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strat.deep2020_ti_pipeline import INDICATORS, load_ohlc, load_ohlcv
from strat.deep2020_ti_15m_fighter import _fighter_book

BASE = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE"
FIX = (4, 48, 48)            # the honest fixed fighter level (conf C, min_hold M, cooldown K)


def main() -> int:
    cads = ["15m", "30m"]
    if "--cadences" in sys.argv:
        cads = sys.argv[sys.argv.index("--cadences") + 1].split(",")
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    C, M, K = FIX
    export = {}
    cache = {}
    for ind_key, ind in INDICATORS.items():
        if only and ind_key not in only:
            continue
        loader = "ohlcv" if ind.get("loader") == "ohlcv" else "ohlc"
        for cad in cads:
            ck = (loader, cad)
            if ck not in cache:
                cache[ck] = (load_ohlcv if loader == "ohlcv" else load_ohlc)(cad)
            assets, vt = cache[ck]
            if not assets:
                continue
            rows = []
            for p in ind["grid"]():
                m = _fighter_book(assets, vt, ind["iron"], p, C, M, K)
                if m:
                    rows.append({"cfg": ind["name"](p), "fighter": m})
            if not rows:
                continue
            rows.sort(key=lambda r: -r["fighter"]["net"])
            robust = [r for r in rows if r["fighter"]["val_net"] > 0 and r["fighter"]["net"] > 0]
            best_rob = robust[0] if robust else None
            export[f"{ind_key}|{cad}"] = {"top10": rows[:10], "best_robust": best_rob,
                                          "n_robust": len(robust), "n_total": len(rows)}
            br = (f"best-robust {best_rob['cfg']} net {best_rob['fighter']['net']}% Sh "
                  f"{best_rob['fighter']['sharpe']}" if best_rob else "NO robust recovery")
            print(f"   {ind_key:10} {cad:4}: {len(robust)}/{len(rows)} robust; top net {rows[0]['fighter']['net']}%; {br}")
    out = BASE / ("ti_recover_" + "_".join(cads) + ".json")
    json.dump(export, open(out, "w"), indent=1, default=str)
    print(f"\n[json] {out}  ({len(export)} (TI x fine-TF) recovered)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
