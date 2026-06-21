"""scripts/_audit_v11_oos.py -- RED-TEAM audit of the V1.1 fine-TF exit/htf-gate OOS claim.

Attacks the 7 vectors the campaign flagged as un-verified. Reuses the VERIFIED wealth path
(ma_wealth_driver.run_wealth) and the exact harness/sweep primitives -- NO hand-rolled MtM.
Writes evidence to runs/engines/V1.1_finetf_exits/_audit_oos.json. No emoji (cp1252).
"""
from __future__ import annotations
import io, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import strat.fleet_lab as fl  # noqa: E402
from strat import ma_wealth_driver as wd  # noqa: E402
from strat.exit_harness import build_exit_fn, align_htf_on, NEXT_COARSER_TF  # noqa: E402
from sweep_finetf_exits import build_htf_lookup, slice7d_metrics  # noqa: E402

IRON_START, IRON_END = "2021-01-01", "2022-01-01"
OUT = ROOT / "runs" / "engines" / "V1.1_finetf_exits" / "_audit_oos.json"
report = {}


def _bpd(tf): return fl.BARS_PER_DAY[tf]


# ===========================================================================
# VECTOR 1 -- LEAK in HTF alignment (causality of align_htf_on / searchsorted).
# ===========================================================================
def vector1_htf_causality():
    out = {"vector": "1_HTF_alignment_leak"}
    cases = []
    # synthetic: coarser bars close at 00:00 and 04:00 with values [1, 0].
    base = pd.date_range("2020-01-01 00:00", periods=8, freq="1h")
    htf = pd.date_range("2020-01-01 00:00", periods=2, freq="4h")
    on = np.array([1, 0], dtype=np.int8)
    al = align_htf_on(base, htf, on)
    # CAUSAL EXPECTATION: a coarser bar timestamped t is the bar that OPENS at t and CLOSES at t+4h.
    # The harness treats the coarser ts as the close-time and forward-fills the LAST coarser ts <= base_t.
    # So base 00:00..03:00 -> coarser_bar0 value (1); base 04:00..07:00 -> coarser_bar1 value (0).
    exp = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int8)
    cases.append({"name": "forward_fill_matches_searchsorted_right_minus_1",
                  "aligned": al.tolist(), "expected": exp.tolist(), "pass": bool(np.array_equal(al, exp))})

    # THE REAL QUESTION: is the coarser ts the OPEN-time or CLOSE-time of the coarser bar?
    # In the chimera loader, df['timestamp'] is the bar's OPEN/label time. A bar labeled 00:00 on a 4h grid
    # spans [00:00, 04:00) and is only COMPLETE at 04:00. If the harness forward-fills value@00:00 onto base
    # bars 00:00..03:00, it is using a bar that has NOT closed yet -> LOOK-AHEAD.
    # Test: a base bar at 01:00 must NOT see the coarser-bar-labeled-00:00 value if that bar closes at 04:00.
    # searchsorted(ht=[00:00,04:00], bt=01:00, 'right')-1 = index 0 -> uses coarser_bar0 (labeled 00:00).
    leaks = []
    for bt_iso, idx_used, coarse_open, coarse_close in [
        ("2020-01-01 01:00", int(np.searchsorted(htf.asi8, pd.Timestamp("2020-01-01 01:00").value, "right") - 1),
         "2020-01-01 00:00", "2020-01-01 04:00")]:
        # If the coarse ts is an OPEN label, bar0 closes at 04:00 > base 01:00 -> using it is look-ahead.
        leak = pd.Timestamp(coarse_close) > pd.Timestamp(bt_iso)
        leaks.append({"base_bar": bt_iso, "coarse_idx_used": idx_used,
                      "coarse_bar_open": coarse_open, "coarse_bar_close": coarse_close,
                      "is_lookahead_if_ts_is_OPEN_label": bool(leak)})
    out["synthetic_cases"] = cases
    out["timestamp_semantics_probe"] = leaks
    return out


# ===========================================================================
# VECTOR 2 -- 2022 SEAL. Prove the iron never loaded a 2022+ bar.
# ===========================================================================
def vector2_2022_seal(cells):
    out = {"vector": "2_2022_seal"}
    rows = []
    for ma, tf in cells:
        bpd = _bpd(tf)
        floor_ms = int(pd.Timestamp(IRON_START).value // 10 ** 6)
        wall_ms = min(int(pd.Timestamp(IRON_END).value // 10 ** 6),
                      int(pd.Timestamp(fl.DEV_END).value // 10 ** 6))
        lookup = build_htf_lookup(ma, tf, floor_ms, wall_ms)
        fn = build_exit_fn(bars_per_day=bpd, htf_on_lookup=lookup, exit_type="band_off", entry_type="entry_none")
        r = wd.run_wealth(ma, tf, "7d", exit_fn=fn, eval_start=IRON_START, eval_end=IRON_END, verbose=False)
        pmax = pd.Timestamp(r["eval_window"]["panel_max"])
        pmin = pd.Timestamp(r["eval_window"]["panel_min"])
        rows.append({"cell": f"{ma}|{tf}", "panel_min": str(pmin), "panel_max": str(pmax),
                     "max_lt_2022": bool(pmax < pd.Timestamp("2022-01-01")),
                     "min_ge_2021": bool(pmin >= pd.Timestamp("2021-01-01"))})
    out["per_cell"] = rows
    out["all_sealed"] = bool(all(x["max_lt_2022"] and x["min_ge_2021"] for x in rows))
    return out


# ===========================================================================
# VECTOR 3 -- lag-1 / causality of the chain (the lag-0 cheat must EXPLODE compound).
# ===========================================================================
def vector3_lag_causality():
    out = {"vector": "3_lag1_causality"}
    # Re-implement run_wealth's compound at lag-0 vs lag-1 on ONE cell to prove the test has power.
    ma, tf = "TEMA", "30m"
    bpd = _bpd(tf)
    floor_ms = int(pd.Timestamp(IRON_START).value // 10 ** 6)
    wall_ms = min(int(pd.Timestamp(IRON_END).value // 10 ** 6), int(pd.Timestamp(fl.DEV_END).value // 10 ** 6))
    lookup = build_htf_lookup(ma, tf, floor_ms, wall_ms)
    fn = build_exit_fn(bars_per_day=bpd, htf_on_lookup=lookup,
                       exit_type="profit_giveback", exit_params={"target": 10.0, "Z": 50.0},
                       entry_type="htf_entry_gate")
    r = wd.run_wealth(ma, tf, "7d", exit_fn=fn, eval_start=IRON_START, eval_end=IRON_END, verbose=False)
    eng_net = np.asarray([v for _, v in r["engine_net_stream"]], dtype=float)
    lag1_compound = float((np.prod(1.0 + eng_net) - 1.0) * 100)
    # reconstruct lag-0 by re-running compound with W (not W.shift(1)) -- emulate by shifting net forward 1
    # (net_t uses W_{t-1}.R_t; lag-0 would be W_t.R_t == shifting the engine net stream backward by 1 bar).
    lag0_net = np.empty_like(eng_net); lag0_net[:-1] = eng_net[1:]; lag0_net[-1] = 0.0
    lag0_compound = float((np.prod(1.0 + lag0_net) - 1.0) * 100)
    out["cell"] = f"{ma}|{tf}"
    out["lag1_compound_pct"] = round(lag1_compound, 2)
    out["lag0_emulated_compound_pct"] = round(lag0_compound, 2)
    out["lag0_inflates"] = bool(lag0_compound > lag1_compound * 1.5)
    out["matches_reported"] = bool(abs(lag1_compound - r["engine_compound_pct"]) < 1.0)
    out["mtm_reconciled"] = r["mtm_reconciled"]
    out["note"] = ("lag0 emulation shifts the net stream back 1 bar; a true lag-0 peek is not a clean shift but "
                   "this is a directional power check -- run_wealth shifts W by 1 bar (W.shift(1)) which is lag-1.")
    return out


# ===========================================================================
# VECTOR 4 -- BULL-YEAR ARTIFACT: always-long slice7d on 2021 vs engine.
# ===========================================================================
def vector4_bull_artifact(cells):
    out = {"vector": "4_bull_year_artifact"}
    rows = []
    for ma, tf, winner_kw, win_label in cells:
        bpd = _bpd(tf)
        floor_ms = int(pd.Timestamp(IRON_START).value // 10 ** 6)
        wall_ms = min(int(pd.Timestamp(IRON_END).value // 10 ** 6), int(pd.Timestamp(fl.DEV_END).value // 10 ** 6))
        lookup = build_htf_lookup(ma, tf, floor_ms, wall_ms)
        # engine (winner)
        fn = build_exit_fn(bars_per_day=bpd, htf_on_lookup=lookup, **winner_kw)
        r = wd.run_wealth(ma, tf, "7d", exit_fn=fn, eval_start=IRON_START, eval_end=IRON_END, verbose=False)
        eng_sm = slice7d_metrics(r.get("engine_equity"), r.get("engine_net_stream"), bpd)
        # band_off baseline
        fnb = build_exit_fn(bars_per_day=bpd, exit_type="band_off", entry_type="entry_none")
        rb = wd.run_wealth(ma, tf, "7d", exit_fn=fnb, eval_start=IRON_START, eval_end=IRON_END, verbose=False)
        base_sm = slice7d_metrics(rb.get("engine_equity"), rb.get("engine_net_stream"), bpd)
        # ALWAYS-LONG slice7d: the always-long rebalanced EW index net stream is what run_wealth uses internally
        # as the timing control. Reconstruct it from a forced always-on mask (entry_none + a mask of all 1s is
        # not exposed; instead use the alwayslong leg by setting exit_fn that returns all-ones).
        fn_all = build_exit_fn(bars_per_day=bpd, exit_type="band_off", entry_type="entry_none")

        # always-long = force the hold mask to 1 everywhere the asset is listed -> exit_fn returning ones.
        def ones_fn(sym, o, h, l, c, base_mask):
            return np.ones(len(base_mask), dtype=np.int8)
        r_al = wd.run_wealth(ma, tf, "7d", exit_fn=ones_fn, eval_start=IRON_START, eval_end=IRON_END, verbose=False)
        al_sm = slice7d_metrics(r_al.get("engine_equity"), r_al.get("engine_net_stream"), bpd)
        rows.append({
            "cell": f"{ma}|{tf}", "winner": win_label,
            "engine_slice7d": eng_sm.get("slice7d_profit_rate"),
            "engine_net_pct": r["engine_compound_pct"],
            "engine_maxdd_pct": r["maxdd_engine_pct"],
            "band_off_slice7d": base_sm.get("slice7d_profit_rate"),
            "band_off_net_pct": rb["engine_compound_pct"],
            "always_long_slice7d": al_sm.get("slice7d_profit_rate"),
            "always_long_net_pct": r_al["engine_compound_pct"],
            "always_long_maxdd_pct": r_al["maxdd_engine_pct"],
            "lift_vs_bandoff_pp": round((eng_sm.get("slice7d_profit_rate", 0) - base_sm.get("slice7d_profit_rate", 0)) * 100, 2),
            "lift_vs_alwayslong_pp": round((eng_sm.get("slice7d_profit_rate", 0) - al_sm.get("slice7d_profit_rate", 0)) * 100, 2),
        })
    out["per_cell"] = rows
    return out


# ===========================================================================
# VECTOR 5 -- DOES FINE TF ADD VALUE OVER THE COARSE ENGINE?
# ===========================================================================
def vector5_fine_vs_coarse(triples):
    out = {"vector": "5_fine_vs_coarse_value"}
    rows = []
    for ma, fine_tf, coarse_tf, winner_kw, win_label in triples:
        # fine + htf winner
        bpd_f = _bpd(fine_tf)
        floor_ms = int(pd.Timestamp(IRON_START).value // 10 ** 6)
        wall_ms = min(int(pd.Timestamp(IRON_END).value // 10 ** 6), int(pd.Timestamp(fl.DEV_END).value // 10 ** 6))
        lookup = build_htf_lookup(ma, fine_tf, floor_ms, wall_ms)
        fn = build_exit_fn(bars_per_day=bpd_f, htf_on_lookup=lookup, **winner_kw)
        rf = wd.run_wealth(ma, fine_tf, "7d", exit_fn=fn, eval_start=IRON_START, eval_end=IRON_END, verbose=False)
        sf = slice7d_metrics(rf.get("engine_equity"), rf.get("engine_net_stream"), bpd_f)
        # coarse engine alone (band_off, entry_none) at the coarser TF
        bpd_c = _bpd(coarse_tf)
        fnc = build_exit_fn(bars_per_day=bpd_c, exit_type="band_off", entry_type="entry_none")
        rc = wd.run_wealth(ma, coarse_tf, "7d", exit_fn=fnc, eval_start=IRON_START, eval_end=IRON_END, verbose=False)
        sc = slice7d_metrics(rc.get("engine_equity"), rc.get("engine_net_stream"), bpd_c)
        rows.append({
            "fine_cell": f"{ma}|{fine_tf}+htf", "coarse_cell": f"{ma}|{coarse_tf} band_off",
            "winner": win_label,
            "fine_slice7d": sf.get("slice7d_profit_rate"), "coarse_slice7d": sc.get("slice7d_profit_rate"),
            "fine_net_pct": rf["engine_compound_pct"], "coarse_net_pct": rc["engine_compound_pct"],
            "fine_maxdd_pct": rf["maxdd_engine_pct"], "coarse_maxdd_pct": rc["maxdd_engine_pct"],
            "fine_beats_coarse_slice7d": bool((sf.get("slice7d_profit_rate") or 0) > (sc.get("slice7d_profit_rate") or 0)),
            "fine_beats_coarse_net": bool((rf["engine_compound_pct"] or 0) > (rc["engine_compound_pct"] or 0)),
        })
    out["per_cell"] = rows
    return out


# ===========================================================================
# VECTOR 6 -- UNDER-PARTICIPATION: hold fraction / exposure must be non-trivial.
# ===========================================================================
def vector6_participation(cells):
    out = {"vector": "6_participation"}
    rows = []
    for ma, tf, winner_kw, win_label in cells:
        bpd = _bpd(tf)
        floor_ms = int(pd.Timestamp(IRON_START).value // 10 ** 6)
        wall_ms = min(int(pd.Timestamp(IRON_END).value // 10 ** 6), int(pd.Timestamp(fl.DEV_END).value // 10 ** 6))
        lookup = build_htf_lookup(ma, tf, floor_ms, wall_ms)
        fn = build_exit_fn(bars_per_day=bpd, htf_on_lookup=lookup, **winner_kw)
        r = wd.run_wealth(ma, tf, "7d", exit_fn=fn, eval_start=IRON_START, eval_end=IRON_END, verbose=False)
        net = np.asarray([v for _, v in r["engine_net_stream"]], dtype=float)
        deployed_frac = float(np.mean(np.abs(net) > 1e-12))
        rows.append({"cell": f"{ma}|{tf}", "winner": win_label,
                     "avg_gross_exposure": r["engine_avg_gross"],
                     "deployed_bar_fraction": round(deployed_frac, 4),
                     "n_slices_deployed": slice7d_metrics(r.get("engine_equity"), r.get("engine_net_stream"), bpd).get("n_slices_deployed"),
                     "n_slices": slice7d_metrics(r.get("engine_equity"), r.get("engine_net_stream"), bpd).get("n_slices")})
    out["per_cell"] = rows
    return out


# ===========================================================================
# VECTOR 7 -- CURVE-FIT: a SINGLE universal combo vs per-cell-tuned winners on 2021.
# ===========================================================================
def vector7_universal(cells):
    out = {"vector": "7_universal_vs_tuned"}
    rows = []
    universal_kw = dict(exit_type="band_off", entry_type="htf_entry_gate")  # simplest htf-gate rule, no exit tuning
    for ma, tf, winner_kw, win_label in cells:
        bpd = _bpd(tf)
        floor_ms = int(pd.Timestamp(IRON_START).value // 10 ** 6)
        wall_ms = min(int(pd.Timestamp(IRON_END).value // 10 ** 6), int(pd.Timestamp(fl.DEV_END).value // 10 ** 6))
        lookup = build_htf_lookup(ma, tf, floor_ms, wall_ms)
        fn_u = build_exit_fn(bars_per_day=bpd, htf_on_lookup=lookup, **universal_kw)
        ru = wd.run_wealth(ma, tf, "7d", exit_fn=fn_u, eval_start=IRON_START, eval_end=IRON_END, verbose=False)
        su = slice7d_metrics(ru.get("engine_equity"), ru.get("engine_net_stream"), bpd)
        fn_t = build_exit_fn(bars_per_day=bpd, htf_on_lookup=lookup, **winner_kw)
        rt = wd.run_wealth(ma, tf, "7d", exit_fn=fn_t, eval_start=IRON_START, eval_end=IRON_END, verbose=False)
        st = slice7d_metrics(rt.get("engine_equity"), rt.get("engine_net_stream"), bpd)
        rows.append({
            "cell": f"{ma}|{tf}", "tuned_winner": win_label,
            "universal_rule": "band_off__htf_entry_gate",
            "universal_slice7d": su.get("slice7d_profit_rate"), "tuned_slice7d": st.get("slice7d_profit_rate"),
            "universal_net_pct": ru["engine_compound_pct"], "tuned_net_pct": rt["engine_compound_pct"],
            "universal_maxdd_pct": ru["maxdd_engine_pct"], "tuned_maxdd_pct": rt["maxdd_engine_pct"],
            "universal_holds_oos": bool((su.get("slice7d_profit_rate") or 0) >= 0.50 and (ru["engine_compound_pct"] or -1) > 0),
            "tuning_gain_slice7d_pp": round(((st.get("slice7d_profit_rate") or 0) - (su.get("slice7d_profit_rate") or 0)) * 100, 2),
        })
    out["per_cell"] = rows
    out["universal_holds_all"] = bool(all(x["universal_holds_oos"] for x in rows))
    return out


def main():
    # winner kwargs per representative cell (from sweep_2021 frozen_from_2020)
    HMA_1h = dict(exit_type="profit_giveback", exit_params={"target": 10.0, "Z": 50.0}, entry_type="htf_entry_gate")
    TEMA_30m = dict(exit_type="profit_giveback", exit_params={"target": 10.0, "Z": 50.0}, entry_type="htf_entry_gate")
    EMA_2h = dict(exit_type="time_stop", exit_params={"N_days": 7}, entry_type="htf_entry_gate")

    print("[v1] HTF causality ..."); report["vector1"] = vector1_htf_causality()
    seal_cells = [("HMA", "1h"), ("TEMA", "30m"), ("EMA", "2h"), ("KAMA", "15m")]
    print("[v2] 2022 seal ..."); report["vector2"] = vector2_2022_seal(seal_cells)
    print("[v3] lag-1 causality ..."); report["vector3"] = vector3_lag_causality()
    bull_cells = [("HMA", "1h", HMA_1h, "profit_giveback_t10z50__htf_entry_gate"),
                  ("TEMA", "30m", TEMA_30m, "profit_giveback_t10z50__htf_entry_gate"),
                  ("EMA", "2h", EMA_2h, "time_stop_7d__htf_entry_gate")]
    print("[v4] bull-year artifact ..."); report["vector4"] = vector4_bull_artifact(bull_cells)
    triples = [("HMA", "1h", "4h", HMA_1h, "profit_giveback_t10z50__htf_entry_gate"),
               ("TEMA", "30m", "2h", TEMA_30m, "profit_giveback_t10z50__htf_entry_gate"),
               ("EMA", "2h", "1d", EMA_2h, "time_stop_7d__htf_entry_gate")]
    print("[v5] fine vs coarse ..."); report["vector5"] = vector5_fine_vs_coarse(triples)
    print("[v6] participation ..."); report["vector6"] = vector6_participation(bull_cells)
    uni_cells = bull_cells + [("KAMA", "1h", dict(exit_type="band_off", entry_type="htf_entry_gate"), "band_off__htf_entry_gate")]
    print("[v7] universal vs tuned ..."); report["vector7"] = vector7_universal(uni_cells)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=1, default=str)
    print(f"\n[json] {OUT}")
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
