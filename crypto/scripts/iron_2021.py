"""scripts/iron_2021.py -- V1.1 Wave 2: IRON the 2020-selected fine-TF exit winners on 2021 (OUT-OF-SAMPLE).

The honest OOS test. For each (engine, TF) cell, FREEZE the combo selected on 2020 (from sweep_2020/<cell>.json)
and run it UNCHANGED on 2021 (eval_start=2021-01-01, eval_end=2022-01-01 -> 2022+ incl. all of it NEVER loaded).
Compare to the band_off baseline on 2021. The question: does the 2020 fix HOLD on 2021, or was it 2020-overfit?

Reuses the verified path: ma_wealth_driver.run_wealth (eval-window + engine_equity), exit_harness.build_exit_fn,
and sweep_finetf_exits.{build_htf_lookup, slice7d_metrics, trades_per_week}. NO hand-rolled wealth, NO re-selection
on 2021 (that would be in-sample). 2020 is the SELECT set; 2021 is the HELD-OUT iron. No emoji (cp1252).
"""
from __future__ import annotations
import io
import json
import sys
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd  # noqa: E402
import strat.fleet_lab as fl  # noqa: E402
from strat import ma_wealth_driver as wd  # noqa: E402
from strat.exit_harness import build_exit_fn  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
from sweep_finetf_exits import (  # noqa: E402
    build_htf_lookup, slice7d_metrics, trades_per_week, EXIT_COMBOS, ENTRY_COMBOS,
)

# canonical label -> build_exit_fn kwargs (the SAME definitions the sweep used; best_exit stores the LABEL,
# e.g. "trail_pct_8", whose canonical exit_type is "trail_pct" + exit_params {"X":8} -- look it up, never guess).
EXIT_MAP = dict(EXIT_COMBOS)
ENTRY_MAP = dict(ENTRY_COMBOS)


def _kwargs_from_label(best_label: str) -> dict:
    exit_label, entry_label = best_label.split("__", 1)
    return {**EXIT_MAP[exit_label], **ENTRY_MAP[entry_label]}

SWEEP_2020 = ROOT / "runs" / "engines" / "V1.1_finetf_exits" / "sweep_2020"
OUT_DIR = ROOT / "runs" / "engines" / "V1.1_finetf_exits" / "sweep_2021"
IRON_START = "2021-01-01"
IRON_END = "2022-01-01"   # 2022+ (incl. all of it) NEVER loaded -> sealed OOS per the user


def _tpw_from_stream(net_stream, bpd) -> float | None:
    """Deployment-entry proxy for trades/week (same idea as run_cell's _trades_per_week, recomputed here since
    run_wealth does not inject it): count False->True edges of (|net|>0) = re-entries, divide by elapsed weeks."""
    if not net_stream:
        return None
    import numpy as np
    dep = (np.abs(np.asarray([v for _, v in net_stream], dtype=float)) > 1e-12).astype(int)
    if len(dep) < 2:
        return None
    entries = int(np.sum((dep[1:] == 1) & (dep[:-1] == 0))) + int(dep[0] == 1)
    weeks = len(dep) / (bpd * 7.0)
    return round(entries / weeks, 3) if weeks > 0 else None


def _eval(ma_type, tf, fn_kwargs, lookup, bpd):
    fn = build_exit_fn(bars_per_day=bpd, htf_on_lookup=lookup, **fn_kwargs)
    r = wd.run_wealth(ma_type, tf, "7d", exit_fn=fn, eval_start=IRON_START, eval_end=IRON_END, verbose=False)
    if "error" in r:
        return {"status": "error", "error": r["error"]}
    sm = slice7d_metrics(r.get("engine_equity"), r.get("engine_net_stream"), bpd)
    return {
        "status": "ok",
        "net_compound_pct": r.get("engine_compound_pct"),
        "maxdd_pct": r.get("maxdd_engine_pct"),
        "trades_per_week": _tpw_from_stream(r.get("engine_net_stream"), bpd),
        "mtm_reconciled": r.get("mtm_reconciled"),
        "data_range": r.get("data_range"),
        **sm,
    }


def iron_cell(cell_json: Path) -> dict:
    j = json.load(open(cell_json))
    ma_type, tf = j["ma_type"], j["tf"]
    bpd = fl.BARS_PER_DAY[tf]
    dev_wall = int(pd.Timestamp(fl.DEV_END).value // 10 ** 6)
    floor_ms = int(pd.Timestamp(IRON_START).value // 10 ** 6)
    wall_ms = min(int(pd.Timestamp(IRON_END).value // 10 ** 6), dev_wall)
    lookup = build_htf_lookup(ma_type, tf, floor_ms, wall_ms)

    # frozen 2020 winner -- reconstruct the EXACT build_exit_fn kwargs from the canonical label (best_exit stores
    # the label, not the exit_type). HTF rules need the lookup; harmless for others.
    win_kw = _kwargs_from_label(j["best_label"])
    win = _eval(ma_type, tf, win_kw, lookup, bpd)
    # baseline band_off+entry_none on 2021
    base = _eval(ma_type, tf, {"exit_type": "band_off", "entry_type": "entry_none"}, None, bpd)

    wr = win.get("slice7d_profit_rate")
    wn = win.get("net_compound_pct")
    holds = (win.get("status") == "ok" and wr is not None and wr >= 0.50 and (wn or -1) > 0)
    verdict = "HOLDS_OOS" if holds else (
        "PARTIAL_OOS" if (win.get("status") == "ok" and wr is not None and wr >= 0.45) else "FAILS_OOS")
    out = {
        "cell": f"{ma_type}|{tf}", "ma_type": ma_type, "tf": tf,
        "frozen_from_2020": {"exit": j["best_exit"], "entry": j["best_entry"],
                             "exit_params": j.get("best_exit_params"), "entry_params": j.get("best_entry_params"),
                             "label_2020": j.get("best_label"),
                             "slice7d_2020": j.get("best_slice7d_profit_rate")},
        "iron_window": {"start": IRON_START, "end": IRON_END, "note": "2022+ NEVER loaded"},
        "baseline_2021": base,
        "winner_2021": win,
        "lift_slice7d_pp_2021": (round((win.get("slice7d_profit_rate") or 0) - (base.get("slice7d_profit_rate") or 0), 4)
                                 if win.get("status") == "ok" and base.get("status") == "ok" else None),
        "verdict_oos": verdict,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_DIR / f"{ma_type}_{tf}.json", "w"), indent=1, default=str)
    print(f"  [{ma_type}|{tf}] 2020-win={j.get('best_label')} slice7d_2020={j.get('best_slice7d_profit_rate')} "
          f"-> 2021 slice7d={win.get('slice7d_profit_rate')} net={win.get('net_compound_pct')}% "
          f"maxDD={win.get('maxdd_pct')} tpw={win.get('trades_per_week')} :: {verdict}")
    return out


def main():
    cells = sorted(glob.glob(str(SWEEP_2020 / "*.json")))
    cells = [c for c in cells if "_summary" not in c]
    print(f"[iron 2021] {len(cells)} cells frozen from 2020 -> run OOS on 2021 [{IRON_START},{IRON_END})")
    results = []
    for c in cells:
        try:
            results.append(iron_cell(Path(c)))
        except Exception as e:
            print(f"  [ERR] {c}: {e}")
    # summary
    from collections import Counter
    vc = Counter(r["verdict_oos"] for r in results)
    summ = {"n_cells": len(results), "verdict_counts": dict(vc),
            "cells": [{"cell": r["cell"], "verdict": r["verdict_oos"],
                       "slice7d_2021": r["winner_2021"].get("slice7d_profit_rate"),
                       "net_2021": r["winner_2021"].get("net_compound_pct"),
                       "tpw_2021": r["winner_2021"].get("trades_per_week")} for r in results]}
    json.dump(summ, open(OUT_DIR / "_summary.json", "w"), indent=1, default=str)
    print(f"\n[iron 2021] DONE {len(results)} cells. verdicts={dict(vc)} -> {OUT_DIR/'_summary.json'}")


if __name__ == "__main__":
    main()
