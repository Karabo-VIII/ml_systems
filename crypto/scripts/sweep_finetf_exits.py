"""scripts/sweep_finetf_exits.py -- V1.1 FINE-TF EXIT/ENTRY SWEEP over 2020 (ONE in-process deterministic run).

WHY (meta /orc 2026-06-21). The prior 16-agent LLM fan-out of this sweep RATE-LIMITED (a deterministic-compute job
dressed as an agent fan-out -- the mistake). This is ONE synchronous Python process. Engines {EMA,HMA,TEMA,KAMA} x
TFs {2h,1h,30m,15m} (16 cells), over 2020 ONLY (eval_start=2020-01-01, eval_end=2021-01-01 so 2021+ incl. ALL of
2022+ is NEVER loaded), u100. Curated exit x entry grid prioritizing the CHURN-FIXERS the diagnosis flagged
(round-trip cost from churn is the single fine-TF killer; gross is +100%+ at every TF; capture flat ~0.38).

THE ACID METRIC (per combo, from the dated engine_equity that run_wealth now returns):
  slice7d_profit_rate = fraction of DISJOINT 7-day windows over 2020 whose engine NET return > 0.
We SELECT the best combo per cell by slice7d_profit_rate (tie-break: lower loss-slice rate, then lower maxDD).
Also reported per combo: median 7d-slice net %, loss-slice rate (net < -1%), net compound %, maxDD %, trades/week
(turnover proxy), when-deployed profit rate (over slices where the engine actually held > 0 exposure).

WEALTH path = ma_wealth_driver.run_wealth ONLY (MtM-no-double-count, lag-1, maker 0.0006 RT, TRUE-HODL, DEV wall +
the new two-sided eval window). Exit/entry masks via exit_harness.build_exit_fn. NO hand-rolled MtM (2 fabrications
were caught because workers hand-rolled it). Incremental per-cell JSON is written as we go (salvageable on stall).

RWYB: python scripts/sweep_finetf_exits.py --engines EMA --tfs 1h --quick     # 1 cell smoke
      python scripts/sweep_finetf_exits.py                                    # the full 16-cell 2020 sweep
No emoji (cp1252). No git commit (overseer commits / meta reviews).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # crypto/
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import strat.fleet_lab as fl                                          # noqa: E402  BARS_PER_DAY, DEV_END
from strat import ma_wealth_driver as wd                              # noqa: E402  THE wealth path
from strat.exit_harness import (                                     # noqa: E402  the mask transforms
    build_exit_fn, align_htf_on, NEXT_COARSER_TF,
)
from strat.ma_movecatch_engine import select_band_trainval, make_hold_provider  # noqa: E402

OUT_DIR = ROOT / "runs" / "engines" / "V1.1_finetf_exits" / "sweep_2020"
EVAL_START = "2020-01-01"
EVAL_END = "2021-01-01"                                              # 2021+ (incl. ALL of 2022+) NEVER loaded
SLICE_DAYS = 7
LOSS_SLICE_THR = -0.01                                               # net < -1% over a 7d slice = a "loss slice"

ENGINES = ["EMA", "HMA", "TEMA", "KAMA"]
TFS = ["2h", "1h", "30m", "15m"]

# Curated exit x entry grid (churn-fixers first). Each entry: (label, kwargs-for-build_exit_fn).
EXIT_COMBOS = [
    ("band_off",                dict(exit_type="band_off")),                                   # baseline
    ("band_off_confirm_M2",     dict(exit_type="band_off_confirm", exit_params={"M": 2})),
    ("band_off_confirm_M3",     dict(exit_type="band_off_confirm", exit_params={"M": 3})),
    ("trail_pct_8",             dict(exit_type="trail_pct", exit_params={"X": 8.0})),
    ("trail_pct_15",            dict(exit_type="trail_pct", exit_params={"X": 15.0})),
    ("chandelier_atr_3",        dict(exit_type="chandelier_atr", exit_params={"k": 3.0})),
    ("time_stop_7d",            dict(exit_type="time_stop", exit_params={"N_days": 7})),
    ("time_stop_14d",           dict(exit_type="time_stop", exit_params={"N_days": 14})),
    ("profit_giveback_t10z50",  dict(exit_type="profit_giveback", exit_params={"target": 10.0, "Z": 50.0})),
    ("htf_confirm_exit",        dict(exit_type="htf_confirm_exit")),                            # HTF
]
ENTRY_COMBOS = [
    ("entry_none",      dict(entry_type="entry_none")),                                         # baseline
    ("entry_debounce_N2", dict(entry_type="entry_debounce", entry_params={"N": 2})),
    ("entry_debounce_N3", dict(entry_type="entry_debounce", entry_params={"N": 3})),
    ("htf_entry_gate",  dict(entry_type="htf_entry_gate")),                                     # HTF
]
HTF_LABELS = {"htf_confirm_exit", "htf_entry_gate"}                  # combos needing the coarser-TF band-on


# ---------------------------------------------------------------------------
# HTF wiring: build a per-symbol next-coarser-TF band-on aligned to the BASE grid.
# ---------------------------------------------------------------------------
def build_htf_lookup(ma_type: str, base_tf: str, floor_ms: int, wall_ms: int):
    """Return htf_on_lookup(sym) -> int8 next-coarser-TF band-on aligned to that sym's BASE-TF dates (or None if
    the coarser data is unavailable). CAUSAL: the coarser band-on is forward-filled onto the base grid via
    align_htf_on (a coarser bar is only usable once CLOSED). The coarser band ENSEMBLE is the SAME committed infra
    (select_band_trainval(coarser_tf) + make_hold_provider) -- never ranked[0]. The same 2020 eval window
    (floor_ms, wall_ms) is applied to the coarser load so HTF never peeks outside 2020 either.

    The exit_fn signature is (sym,o,h,l,c,base_mask) with NO base dates, so we recover the base dates here by
    RE-LOADING the base TF for that sym with the identical _load_asset recipe -- the dates are deterministic, so the
    base_mask the caller passes aligns 1:1 with these base dates by construction (same loader, same window)."""
    coarser_tf = NEXT_COARSER_TF.get(base_tf)
    if coarser_tf is None:
        return None
    # coarser band ensemble (committed infra, TRAIN+VAL only)
    try:
        c_band, _ = select_band_trainval(ma_type, coarser_tf)
    except Exception:
        return None
    if len(c_band) < 2:
        return None
    c_provider = make_hold_provider(c_band, ma_type)

    cache: dict[str, np.ndarray | None] = {}

    def htf_on_lookup(sym: str):
        if sym in cache:
            return cache[sym]
        # base dates (deterministic; identical to what run_wealth loaded for this sym)
        base_rec = wd._load_asset(sym, base_tf, wall_ms, floor_ms=floor_ms)
        coarse_rec = wd._load_asset(sym, coarser_tf, wall_ms, floor_ms=floor_ms)
        if base_rec is None or coarse_rec is None:
            cache[sym] = None
            return None
        base_dates = base_rec[0]
        c_dates, c_o, c_h, c_l, c_c = coarse_rec
        # dedup base dates the same way run_wealth does (keep last), so alignment length matches the base_mask
        bs = pd.Series(np.arange(len(base_dates)), index=base_dates)
        bs = bs[~bs.index.duplicated(keep="last")].sort_index()
        base_dates_dedup = bs.index
        c_on = c_provider(sym, c_o, c_h, c_l, c_c)
        # dedup coarse dates likewise
        cs = pd.DataFrame({"on": c_on}, index=c_dates)
        cs = cs[~cs.index.duplicated(keep="last")].sort_index()
        aligned = align_htf_on(base_dates_dedup, cs.index, cs["on"].to_numpy().astype(np.int8))
        cache[sym] = aligned
        return aligned

    return htf_on_lookup


# ---------------------------------------------------------------------------
# 7d-slice acid metric from the dated engine equity curve.
# ---------------------------------------------------------------------------
def slice7d_metrics(engine_equity, engine_net_stream, bpd: int) -> dict:
    """From run_wealth's dated engine_equity ([iso, value]) + engine_net_stream ([iso, net]), compute the
    DISJOINT 7-day-slice acid metrics. Slices are non-overlapping calendar-anchored 7d windows over the panel.
    A slice's NET return = prod(1+net within the slice) - 1 (compounds the per-bar net stream within the window).
    when-deployed = slice has ANY bar with non-zero engine net (a proxy for 'engine actually held this week')."""
    if not engine_equity or not engine_net_stream:
        return {"n_slices": 0}
    dates = pd.to_datetime([d for d, _ in engine_net_stream])
    net = np.asarray([v for _, v in engine_net_stream], dtype=float)
    df = pd.DataFrame({"net": net}, index=dates).sort_index()
    if len(df) < 2:
        return {"n_slices": 0}
    t0 = df.index[0].normalize()
    # disjoint 7d slice id per bar
    slice_id = ((df.index - t0).total_seconds() // (SLICE_DAYS * 86400)).astype(int)
    slice_rets = []
    deployed_flags = []
    for sid, g in df.groupby(slice_id):
        sret = float(np.prod(1.0 + g["net"].to_numpy()) - 1.0)
        slice_rets.append(sret)
        deployed_flags.append(bool(np.any(np.abs(g["net"].to_numpy()) > 1e-12)))
    slice_rets = np.asarray(slice_rets, dtype=float)
    deployed_flags = np.asarray(deployed_flags, dtype=bool)
    n = len(slice_rets)
    if n == 0:
        return {"n_slices": 0}
    profit_rate = float(np.mean(slice_rets > 0))
    loss_rate = float(np.mean(slice_rets < LOSS_SLICE_THR))
    median_net = float(np.median(slice_rets) * 100)
    dep = deployed_flags
    when_deployed_rate = (float(np.mean(slice_rets[dep] > 0)) if dep.any() else None)
    return {
        "n_slices": int(n),
        "slice7d_profit_rate": round(profit_rate, 4),
        "loss_slice_rate": round(loss_rate, 4),
        "median_7d_net_pct": round(median_net, 3),
        "when_deployed_profit_rate": (round(when_deployed_rate, 4) if when_deployed_rate is not None else None),
        "n_slices_deployed": int(dep.sum()),
    }


def trades_per_week(r: dict, bpd: int) -> float | None:
    """Turnover proxy: engine avg gross exposure / hold-state churn is not directly in the output, so we use the
    n_bars + a coarse round-trip count is unavailable here; instead use the engine_net_stream non-zero-runs as an
    in/out proxy. Computed in run_cell from the stream (passed through here as r['_trades_per_week'])."""
    return r.get("_trades_per_week")


# ---------------------------------------------------------------------------
# ONE CELL: sweep all exit x entry combos for (engine, tf) over 2020.
# ---------------------------------------------------------------------------
def run_cell(ma_type: str, tf: str, *, defer_htf: bool = False, verbose: bool = True) -> dict:
    bpd = fl.BARS_PER_DAY[tf]
    dev_wall_ms = int(pd.Timestamp(fl.DEV_END).value // 10 ** 6)
    wall_ms = min(int(pd.Timestamp(EVAL_END).value // 10 ** 6), dev_wall_ms)
    floor_ms = int(pd.Timestamp(EVAL_START).value // 10 ** 6)

    # HTF lookup (built once per cell; None if coarser unavailable)
    htf_lookup = None
    htf_err = None
    try:
        htf_lookup = build_htf_lookup(ma_type, tf, floor_ms, wall_ms)
    except Exception as e:                                            # noqa: BLE001
        htf_err = f"{type(e).__name__}: {e}"
        htf_lookup = None

    combos = []
    for ex_label, ex_kw in EXIT_COMBOS:
        for en_label, en_kw in ENTRY_COMBOS:
            label = f"{ex_label}__{en_label}"
            is_htf = (ex_label in HTF_LABELS) or (en_label in HTF_LABELS)
            combos.append((label, ex_label, en_label, ex_kw, en_kw, is_htf))

    results = {}
    t_cell = time.time()
    for (label, ex_label, en_label, ex_kw, en_kw, is_htf) in combos:
        if is_htf and (defer_htf or htf_lookup is None):
            results[label] = {"status": "deferred",
                              "reason": ("defer_htf flag" if defer_htf else
                                         f"htf_lookup unavailable ({htf_err or 'coarser band <2 / no data'})")}
            continue
        try:
            kw = {**ex_kw, **en_kw, "bars_per_day": bpd}
            if is_htf:
                kw["htf_on_lookup"] = htf_lookup
            fn = build_exit_fn(**kw)
            r = wd.run_wealth(ma_type, tf, "7d", exit_fn=fn,
                              eval_start=EVAL_START, eval_end=EVAL_END, verbose=False)
            if "error" in r:
                results[label] = {"status": "error", "error": r["error"]}
                continue
            sm = slice7d_metrics(r.get("engine_equity"), r.get("engine_net_stream"), bpd)
            # trades/week proxy: count in->out transitions in the net stream's deployed mask, / n_weeks
            net = np.asarray([v for _, v in r.get("engine_net_stream", [])], dtype=float)
            deployed = (np.abs(net) > 1e-12).astype(int)
            transitions = int(np.sum(np.abs(np.diff(deployed)) == 1)) if len(deployed) > 1 else 0
            round_trips = transitions / 2.0
            n_weeks = max(1e-9, r["n_bars"] / (bpd * 7.0))
            tpw = round(round_trips / n_weeks, 3)
            rec = {
                "status": "ok",
                "exit": ex_label, "entry": en_label,
                "exit_params": ex_kw.get("exit_params", {}), "entry_params": en_kw.get("entry_params", {}),
                "net_compound_pct": r["engine_compound_pct"],
                "maxdd_pct": r["maxdd_engine_pct"],
                "hodl_compound_pct": r["hodl_compound_pct"],
                "vs_hodl_pp": r["vs_hodl_pp"],
                "mtm_reconciled": r["mtm_reconciled"],
                "n_assets": r["n_assets"], "n_bars": r["n_bars"],
                "trades_per_week": tpw,
                **sm,
            }
            results[label] = rec
            if verbose:
                print(f"    [{ma_type}|{tf}] {label:42s} "
                      f"slice7d={sm.get('slice7d_profit_rate')}  "
                      f"loss={sm.get('loss_slice_rate')}  net={rec['net_compound_pct']}%  "
                      f"maxDD={rec['maxdd_pct']}%  tpw={tpw}")
        except Exception as e:                                       # noqa: BLE001
            results[label] = {"status": "error", "error": f"{type(e).__name__}: {e}",
                              "trace": traceback.format_exc()[-600:]}
            if verbose:
                print(f"    [{ma_type}|{tf}] {label:42s} ERROR {type(e).__name__}: {e}")

    # ---- SELECT best combo by slice7d_profit_rate (tie-break: lower loss_slice_rate, then lower maxDD) ----
    ok = {k: v for k, v in results.items() if v.get("status") == "ok" and "slice7d_profit_rate" in v}

    def sort_key(kv):
        k, v = kv
        return (-(v.get("slice7d_profit_rate") or 0.0),
                (v.get("loss_slice_rate") if v.get("loss_slice_rate") is not None else 1.0),
                (v.get("maxdd_pct") if v.get("maxdd_pct") is not None else -999.0))  # less-negative maxDD better

    best_label, best = (None, None)
    if ok:
        # maxDD: less negative (closer to 0) is better -> for tie-break we want the LARGEST (closest to 0) maxdd_pct.
        ranked = sorted(ok.items(), key=lambda kv: (
            -(kv[1].get("slice7d_profit_rate") or 0.0),
            (kv[1].get("loss_slice_rate") if kv[1].get("loss_slice_rate") is not None else 1.0),
            -(kv[1].get("maxdd_pct") if kv[1].get("maxdd_pct") is not None else -999.0),
        ))
        best_label, best = ranked[0]

    baseline = results.get("band_off__entry_none")
    base_rate = baseline.get("slice7d_profit_rate") if baseline and baseline.get("status") == "ok" else None
    best_rate = best.get("slice7d_profit_rate") if best else None
    lift_pp = (round((best_rate - base_rate) * 100, 2) if (best_rate is not None and base_rate is not None) else None)

    verdict = "UNFIXED"
    if best is not None:
        if (best.get("slice7d_profit_rate") or 0) >= 0.50 and (best.get("net_compound_pct") or -1) > 0:
            verdict = "FIXED"
        elif base_rate is not None and best_rate is not None and best_rate > base_rate:
            verdict = "IMPROVED"

    return {
        "cell": f"{ma_type}|{tf}",
        "ma_type": ma_type, "tf": tf, "bars_per_day": bpd,
        "eval_window": {"eval_start": EVAL_START, "eval_end": EVAL_END},
        "n_combos": len(combos),
        "n_ok": len(ok),
        "htf_available": htf_lookup is not None, "htf_err": htf_err,
        "baseline_label": "band_off__entry_none",
        "baseline_slice7d_profit_rate": base_rate,
        "baseline_net_compound_pct": (baseline.get("net_compound_pct") if baseline and baseline.get("status") == "ok" else None),
        "baseline_maxdd_pct": (baseline.get("maxdd_pct") if baseline and baseline.get("status") == "ok" else None),
        "best_label": best_label,
        "best_exit": (best.get("exit") if best else None),
        "best_entry": (best.get("entry") if best else None),
        "best_exit_params": (best.get("exit_params") if best else None),
        "best_entry_params": (best.get("entry_params") if best else None),
        "best_slice7d_profit_rate": best_rate,
        "best_net_compound_pct": (best.get("net_compound_pct") if best else None),
        "best_maxdd_pct": (best.get("maxdd_pct") if best else None),
        "lift_pp": lift_pp,
        "verdict": verdict,
        "runtime_s": round(time.time() - t_cell, 1),
        "results": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser(prog="python scripts/sweep_finetf_exits.py")
    ap.add_argument("--engines", nargs="*", default=ENGINES)
    ap.add_argument("--tfs", nargs="*", default=TFS)
    ap.add_argument("--quick", action="store_true", help="smoke: defer HTF combos (run non-HTF grid only)")
    ap.add_argument("--defer-htf", action="store_true", help="mark HTF combos deferred (skip the coarser wiring)")
    a = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    defer_htf = a.defer_htf or a.quick
    t_all = time.time()
    summary_cells = []
    print(f"[sweep] engines={a.engines}  tfs={a.tfs}  eval=[{EVAL_START},{EVAL_END})  defer_htf={defer_htf}")
    print(f"[sweep] OUT_DIR={OUT_DIR}")

    for eng in a.engines:
        for tf in a.tfs:
            print(f"\n=== CELL {eng}|{tf} ===")
            try:
                cell = run_cell(eng, tf, defer_htf=defer_htf, verbose=True)
            except Exception as e:                                   # noqa: BLE001
                cell = {"cell": f"{eng}|{tf}", "status": "CELL_ERROR",
                        "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1200:]}
                print(f"  CELL ERROR {type(e).__name__}: {e}")
            # incremental per-cell write (salvageable)
            cpath = OUT_DIR / f"{eng}_{tf}.json"
            json.dump(cell, open(cpath, "w"), indent=1, default=str)
            print(f"  [json] {cpath}  verdict={cell.get('verdict')}  "
                  f"baseline={cell.get('baseline_slice7d_profit_rate')} -> best={cell.get('best_slice7d_profit_rate')} "
                  f"({cell.get('best_label')})")
            summary_cells.append({k: cell.get(k) for k in (
                "cell", "ma_type", "tf", "verdict", "baseline_slice7d_profit_rate", "best_slice7d_profit_rate",
                "lift_pp", "best_label", "best_exit", "best_entry", "best_exit_params", "best_entry_params",
                "baseline_net_compound_pct", "best_net_compound_pct", "baseline_maxdd_pct", "best_maxdd_pct",
                "htf_available", "n_ok", "runtime_s")})
            # rewrite the running summary after every cell (salvageable)
            _write_summary(summary_cells, t_all, a, defer_htf)

    _write_summary(summary_cells, t_all, a, defer_htf, final=True)
    print(f"\n[sweep] DONE {len(summary_cells)} cells in {round(time.time()-t_all,1)}s -> {OUT_DIR/'_summary.json'}")
    return 0


def _write_summary(cells, t_all, args, defer_htf, final=False):
    # most-effective lever per TF (the combo most often selected as best, per TF)
    by_tf_lever = {}
    for c in cells:
        tf = c.get("tf")
        if tf is None or not c.get("best_label"):
            continue
        by_tf_lever.setdefault(tf, []).append(c.get("best_label"))
    tf_top_lever = {}
    for tf, labels in by_tf_lever.items():
        vals, counts = np.unique(np.array(labels), return_counts=True)
        tf_top_lever[tf] = {"top_best_combo": str(vals[int(np.argmax(counts))]),
                            "count": int(counts.max()), "of": len(labels)}
    summary = {
        "sweep": "V1.1 fine-TF exit/entry sweep over 2020 (ONE in-process run)",
        "eval_window": {"eval_start": EVAL_START, "eval_end": EVAL_END, "note": "2021+ incl. ALL 2022+ NEVER loaded"},
        "acid_metric": "slice7d_profit_rate = fraction of DISJOINT 7d windows over 2020 with engine net return > 0",
        "select_rule": "max slice7d_profit_rate; tie-break lower loss_slice_rate then less-negative maxDD",
        "verdict_rule": "FIXED if best slice7d>=0.50 AND net>0; IMPROVED if best>baseline; else UNFIXED",
        "defer_htf": defer_htf,
        "engines": list(args.engines), "tfs": list(args.tfs),
        "n_cells": len(cells),
        "tf_top_lever": tf_top_lever,
        "cells": cells,
        "elapsed_s": round(time.time() - t_all, 1),
        "final": final,
        "wealth_path": "ma_wealth_driver.run_wealth (MtM-no-double-count, lag-1, maker 0.0006 RT, TRUE-HODL, 2020 eval window)",
    }
    json.dump(summary, open(OUT_DIR / "_summary.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    sys.exit(main())
