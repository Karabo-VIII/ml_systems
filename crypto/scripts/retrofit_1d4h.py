"""scripts/retrofit_1d4h.py -- V1.1: do EXITS improve the ORIGINAL 1d/4h engines? (the user's "rework the originals").

The fine-TF exit-harness was a null (de-leaked). Separate question: can the exit harness IMPROVE the working V1
coarse engines (1d/4h) -- better net or maxDD than the plain band_off engine? Prior art says probably not (dead-list
D61: smart exits on DAILY breakouts were a null, a hold-length artifact). We test it honestly anyway, de-leaked.

Per (engine, tf in {1d,4h}): on 2020 SELECT, sweep band_off (V1 baseline) + churn/giveback exits [+ htf_entry_gate
for 4h, coarser=1d, NOW de-leaked]; pick best by slice7d. On 2021 OOS, run band_off baseline + the 2020-best, and
report whether best beats band_off on net / maxDD / slice7d. 2022+ NEVER loaded. No emoji. No commit.
"""
from __future__ import annotations
import io, json, sys
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

OUT = ROOT / "runs" / "engines" / "V1.1_finetf_exits" / "retrofit_1d4h"
ENGINES, TFS = ["EMA", "HMA", "TEMA", "KAMA"], ["1d", "4h"]
COMBOS = [  # (label, exit kwargs)
    ("band_off", dict(exit_type="band_off")),
    ("band_off_confirm_M2", dict(exit_type="band_off_confirm", exit_params={"M": 2})),
    ("trail_pct_15", dict(exit_type="trail_pct", exit_params={"X": 15.0})),
    ("chandelier_atr_3", dict(exit_type="chandelier_atr", exit_params={"k": 3.0})),
    ("profit_giveback_t10z50", dict(exit_type="profit_giveback", exit_params={"target": 10.0, "Z": 50.0})),
    ("time_stop_14d", dict(exit_type="time_stop", exit_params={"N_days": 14})),
]


def _run(ma, tf, kw, look, bpd, s, e):
    fn = build_exit_fn(bars_per_day=bpd, htf_on_lookup=look, **kw)
    r = wd.run_wealth(ma, tf, "7d", exit_fn=fn, eval_start=s, eval_end=e, verbose=False)
    if "error" in r:
        return {"err": r["error"]}
    sm = slice7d_metrics(r.get("engine_equity"), r.get("engine_net_stream"), bpd)
    return {"slice7d": sm.get("slice7d_profit_rate"), "net": round(r.get("engine_compound_pct") or 0, 1),
            "maxdd": r.get("maxdd_engine_pct")}


def cell(ma, tf):
    bpd = fl.BARS_PER_DAY[tf]
    f20, w20 = int(pd.Timestamp("2020-01-01").value // 10**6), int(pd.Timestamp("2021-01-01").value // 10**6)
    f21, w21 = int(pd.Timestamp("2021-01-01").value // 10**6), int(pd.Timestamp("2022-01-01").value // 10**6)
    look20 = build_htf_lookup(ma, tf, f20, w20)
    look21 = build_htf_lookup(ma, tf, f21, w21)
    combos = list(COMBOS)
    if NEXT_COARSER_TF.get(tf):  # 4h has a coarser TF (1d) -> add the de-leaked HTF gate
        combos = combos + [("band_off__htf_entry_gate", dict(exit_type="band_off", entry_type="htf_entry_gate"))]
    # 2020 select
    res20 = {lab: _run(ma, tf, kw, look20, bpd, "2020-01-01", "2021-01-01") for lab, kw in combos}
    base20 = res20["band_off"]
    ok = {k: v for k, v in res20.items() if "err" not in v and v.get("slice7d") is not None}
    best_lab = max(ok, key=lambda k: (ok[k]["slice7d"], -(ok[k]["maxdd"] or 0)))
    best_kw = dict(combos)[best_lab]
    # 2021 OOS: baseline vs the 2020-best
    base21 = _run(ma, tf, dict(exit_type="band_off"), None, bpd, "2021-01-01", "2022-01-01")
    best21 = _run(ma, tf, best_kw, look21, bpd, "2021-01-01", "2022-01-01")
    improves = (best_lab != "band_off"
                and (best21.get("net") or -1e9) > (base21.get("net") or 1e9)
                and (best21.get("slice7d") or 0) >= (base21.get("slice7d") or 0))
    out = {"cell": f"{ma}|{tf}", "ma": ma, "tf": tf, "best_2020": best_lab,
           "base_2020": base20, "best_2020_metrics": res20[best_lab],
           "base_2021": base21, "best_2021": best21,
           "best_improves_baseline_oos": bool(improves)}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / f"{ma}_{tf}.json", "w"), indent=1, default=str)
    print(f"  {ma+'|'+tf:8} best2020={best_lab:24} | 2021 base[net={base21.get('net')} dd={base21.get('maxdd')} "
          f"s7d={base21.get('slice7d')}] best[net={best21.get('net')} dd={best21.get('maxdd')} "
          f"s7d={best21.get('slice7d')}] :: improves_OOS={improves}")
    return out


def main():
    rows = [cell(ma, tf) for ma in ENGINES for tf in TFS]
    n = sum(r["best_improves_baseline_oos"] for r in rows)
    summ = {"n_cells": len(rows), "exits_improve_baseline_oos": f"{n}/{len(rows)}", "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(summ, open(OUT / "_summary.json", "w"), indent=1, default=str)
    print(f"\n[retrofit 1d/4h] DONE. exits improve V1 baseline OOS: {n}/{len(rows)} "
          f"(0 == exits do NOT help the originals, consistent with D61).")


if __name__ == "__main__":
    main()
