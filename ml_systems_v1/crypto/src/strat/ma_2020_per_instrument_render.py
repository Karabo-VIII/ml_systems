"""src/strat/ma_2020_per_instrument_render.py -- render the per-instrument MA breakdown (VAL+OOS, 2020 H2).

Reads per_instrument_*.json (coarse+fine) and produces: (1) per-TF instrument x MA-class OOS heatmaps;
(2) the VAL->OOS SELECTION-TRANSFER analysis -- does picking the best MA per instrument ON VAL transfer to
OOS? (regret = best-possible-OOS minus VAL-selected-OOS); (3) per-instrument breadth (who carries 2020 H2).
RWYB: python -m strat.ma_2020_per_instrument_render. No emoji (cp1252).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "MA_2020_BREAKDOWN"
CADENCES = ["1d", "4h", "2h", "1h", "30m", "15m"]
MA_TYPES = ["EMA", "SMA", "WMA", "HMA", "DEMA", "TEMA", "KAMA", "VIDYA"]
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]


def main() -> int:
    data = {}   # (cad, sym) -> {ma: {val, oos}}
    for jf in sorted(BASE.glob("per_instrument_*.json")):
        d = json.load(open(jf))
        for k, v in d.items():
            cad, sym = k.split("|"); data[(cad, sym)] = v

    def oos(cad, sym, ma):
        v = data.get((cad, sym), {}).get(ma, {}).get("oos")
        return float(v) if v is not None else np.nan

    def val(cad, sym, ma):
        v = data.get((cad, sym), {}).get(ma, {}).get("val")
        return float(v) if v is not None else np.nan

    # ---- per-instrument best-MA (VAL-selected) -> OOS, per TF + transfer regret ----
    print("# 2020 H2 per-instrument: best MA selected ON VAL -> OOS (causal, no look-ahead)\n")
    print(f"   {'instrument':11}" + "".join(f"{c:>14}" for c in CADENCES))
    transfer = {}
    for sym in SYMS:
        row = f"   {sym:11}"
        for cad in CADENCES:
            d = data.get((cad, sym), {})
            valid = {mt: d[mt] for mt in MA_TYPES if d.get(mt, {}).get("val") is not None}
            if not valid:
                row += f"{'--':>14}"; continue
            bestVAL = max(valid, key=lambda mt: valid[mt]["val"])
            o = valid[bestVAL]["oos"]
            row += f"{(bestVAL+':'+(str(o) if o is not None else '-')):>14}"
        print(row)

    # transfer: per (cad), mean OOS of VAL-best vs mean of TRUE OOS-best (regret of VAL selection)
    print("\n## VAL->OOS SELECTION TRANSFER -- does picking the best MA per instrument on VAL work OOS?")
    print(f"   {'cad':5} {'VALbest_OOS':>12} {'OOSbest_OOS':>12} {'random_OOS':>11} {'regret':>8} {'verdict':>10}")
    for cad in CADENCES:
        vb, ob, rnd = [], [], []
        for sym in SYMS:
            d = data.get((cad, sym), {})
            valid = {mt: d[mt] for mt in MA_TYPES if d.get(mt, {}).get("val") is not None and d[mt].get("oos") is not None}
            if not valid:
                continue
            bestVAL = max(valid, key=lambda mt: valid[mt]["val"])
            vb.append(valid[bestVAL]["oos"])
            ob.append(max(valid[mt]["oos"] for mt in valid))         # hindsight OOS-best (ceiling)
            rnd.append(float(np.mean([valid[mt]["oos"] for mt in valid])))  # random/avg MA
        if not vb:
            continue
        vbm, obm, rndm = float(np.mean(vb)), float(np.mean(ob)), float(np.mean(rnd))
        regret = obm - vbm
        verdict = "TRANSFERS" if vbm >= rndm + 1 else ("~random" if abs(vbm - rndm) <= 2 else "WORSE-rnd")
        transfer[cad] = {"valbest_oos": round(vbm, 1), "oosbest_oos": round(obm, 1), "random_oos": round(rndm, 1), "regret": round(regret, 1)}
        print(f"   {cad:5} {vbm:>12.1f} {obm:>12.1f} {rndm:>11.1f} {regret:>8.1f} {verdict:>10}")

    # ---- per-instrument breadth: OOS sign under each instrument's MAs (who carries 2020 H2) ----
    print("\n## Per-instrument BREADTH -- mean OOS across MA classes (who carries 2020 H2; pooled over TFs)")
    print(f"   {'instrument':11} {'meanOOS':>8} {'%MA>0':>7} {'note':>22}")
    for sym in SYMS:
        vals = [oos(c, sym, mt) for c in CADENCES for mt in MA_TYPES]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            print(f"   {sym:11} {'--':>8} {'--':>7} {'no 2020-H2 data':>22}"); continue
        mo = float(np.mean(vals)); pp = 100 * float(np.mean(np.array(vals) > 0))
        note = "carrier" if mo > 20 else ("weak/new" if mo < 5 else "mid")
        print(f"   {sym:11} {mo:>8.1f} {pp:>6.0f}% {note:>22}")

    # ---- figure: per-TF instrument x MA OOS heatmaps ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, cad in zip(axes.ravel(), CADENCES):
        M = np.array([[oos(cad, s, mt) for mt in MA_TYPES] for s in SYMS])
        vmax = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1
        im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(MA_TYPES))); ax.set_xticklabels(MA_TYPES, fontsize=7, rotation=40)
        ax.set_yticks(range(len(SYMS))); ax.set_yticklabels([s.replace("USDT", "") for s in SYMS], fontsize=7)
        for i in range(len(SYMS)):
            for j in range(len(MA_TYPES)):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.0f}", ha="center", va="center", fontsize=6)
        ax.set_title(f"{cad} -- instrument x MA OOS%")
    fig.suptitle("2020 OOS (Oct-Dec) compound % per INSTRUMENT x MA class, per timeframe "
                 "(green=positive; SOL/AVAX = new-2020 listings)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = BASE / "charts" / "ma_2020_per_instrument.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"\n[figure] {out}")
    json.dump({"transfer": transfer}, open(BASE / "per_instrument_transfer.json", "w"), indent=1, default=str)
    print(f"[json] {BASE / 'per_instrument_transfer.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
