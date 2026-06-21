"""scripts/de_leak_eval.py -- V1.1 HONEST de-leaked evaluation (after the align_htf_on look-ahead fix).

The audit proved the htf_entry_gate had look-ahead (it read the not-yet-closed coarser bar). align_htf_on is now
fixed (close-time alignment). This re-evaluates the fine-TF engine HONESTLY on 2021 (OOS), per cell, comparing:
  - fine_plain  : band_off + entry_none           (the fine engine, no HTF -- the churn-bound V1-style baseline)
  - fine_htf    : band_off + htf_entry_gate        (the V1.1 universal rule, NOW de-leaked)
  - always_long : every admitted name held every bar (the dumb 2021-bull rider -- the fair slice7d yardstick)
  - coarse      : the next-coarser-TF engine alone (band_off + entry_none -- the redundancy check)
Question: de-leaked, does fine_htf BEAT always_long (slice7d) and BEAT the coarse engine (net)? Or is it just
drawdown-insurance redundant with V1's coarse engine? 2022+ NEVER loaded. No emoji. No commit.
"""
from __future__ import annotations
import io, json, sys, glob
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import strat.fleet_lab as fl
from strat import ma_wealth_driver as wd
from strat.exit_harness import build_exit_fn, NEXT_COARSER_TF
from sweep_finetf_exits import build_htf_lookup, slice7d_metrics

S, E = "2021-01-01", "2022-01-01"
OUT = ROOT / "runs" / "engines" / "V1.1_finetf_exits" / "deleaked_2021"
ENGINES, TFS = ["EMA", "HMA", "TEMA", "KAMA"], ["2h", "1h", "30m", "15m"]


def _always_long_fn(sym, o, h, l, c, base_mask):
    return np.ones_like(np.asarray(base_mask, dtype=np.int8))


def _run(ma, tf, fn, bpd):
    r = wd.run_wealth(ma, tf, "7d", exit_fn=fn, eval_start=S, eval_end=E, verbose=False)
    if "error" in r:
        return {"err": r["error"]}
    sm = slice7d_metrics(r.get("engine_equity"), r.get("engine_net_stream"), bpd)
    return {"slice7d": sm.get("slice7d_profit_rate"), "net": round(r.get("engine_compound_pct") or 0, 1),
            "maxdd": r.get("maxdd_engine_pct")}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ma in ENGINES:
        for tf in TFS:
            bpd = fl.BARS_PER_DAY[tf]
            floor_ms = int(pd.Timestamp(S).value // 10 ** 6)
            wall_ms = int(pd.Timestamp(E).value // 10 ** 6)
            look = build_htf_lookup(ma, tf, floor_ms, wall_ms)
            plain = _run(ma, tf, build_exit_fn(exit_type="band_off", entry_type="entry_none", bars_per_day=bpd), bpd)
            htf = _run(ma, tf, build_exit_fn(exit_type="band_off", entry_type="htf_entry_gate",
                                             bars_per_day=bpd, htf_on_lookup=look), bpd)
            al = _run(ma, tf, _always_long_fn, bpd)
            ctf = NEXT_COARSER_TF.get(tf)
            coarse = _run(ma, ctf, build_exit_fn(exit_type="band_off", entry_type="entry_none",
                                                 bars_per_day=fl.BARS_PER_DAY[ctf]), fl.BARS_PER_DAY[ctf]) if ctf else {}
            # honest verdicts (de-leaked, OOS 2021)
            beats_al = (htf.get("slice7d") or 0) > (al.get("slice7d") or 0)          # vs dumb bull-rider
            beats_coarse_net = (htf.get("net") or -1e9) > (coarse.get("net") or 1e9)  # adds wealth over coarse?
            dd_better_al = (htf.get("maxdd") or 0) > (al.get("maxdd") or 0)           # less-negative maxDD
            row = {"cell": f"{ma}|{tf}", "ma": ma, "tf": tf, "coarser": ctf,
                   "fine_plain": plain, "fine_htf": htf, "always_long": al, "coarse_engine": coarse,
                   "htf_beats_alwayslong_slice7d": beats_al,
                   "htf_beats_coarse_net": beats_coarse_net,
                   "htf_dd_better_than_alwayslong": dd_better_al}
            rows.append(row)
            json.dump(row, open(OUT / f"{ma}_{tf}.json", "w"), indent=1, default=str)
            print(f"  {ma+'|'+tf:9} htf[s7d={htf.get('slice7d')} net={htf.get('net')} dd={htf.get('maxdd')}]  "
                  f"alwaysLong[s7d={al.get('slice7d')} net={al.get('net')} dd={al.get('maxdd')}]  "
                  f"coarse({ctf})[net={coarse.get('net')} dd={coarse.get('maxdd')}]  "
                  f":: beatAL_s7d={beats_al} beatCoarse_net={beats_coarse_net}")
    nbAL = sum(r["htf_beats_alwayslong_slice7d"] for r in rows)
    nbC = sum(r["htf_beats_coarse_net"] for r in rows)
    nDD = sum(r["htf_dd_better_than_alwayslong"] for r in rows)
    summ = {"n": len(rows), "htf_beats_alwayslong_slice7d": f"{nbAL}/{len(rows)}",
            "htf_beats_coarse_net": f"{nbC}/{len(rows)}", "htf_dd_better_than_alwayslong": f"{nDD}/{len(rows)}",
            "rows": rows}
    json.dump(summ, open(OUT / "_summary.json", "w"), indent=1, default=str)
    print(f"\n[de-leak] DONE. htf beats always-long(slice7d): {nbAL}/{len(rows)} | "
          f"beats coarse(net): {nbC}/{len(rows)} | dd better than always-long: {nDD}/{len(rows)}")


if __name__ == "__main__":
    main()
