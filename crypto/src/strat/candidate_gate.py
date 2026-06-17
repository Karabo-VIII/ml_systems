"""src/strat/candidate_gate.py -- the foundation's SINGLE reusable validation GATE.

Wires the apparatus into one callable: harness (taker cost) -> leak-probe -> firewall -> battery ->
consolidated verdict. Every future candidate (any AVENUE_MAP avenue) flows through this.

PROVENANCE: ported 2026-06-05 from runs/staging/candidate_eval_2026_06_04.py. Hardened against the
2026-06-05 apparatus red-audit (docs/APPARATUS_AUDIT_2026_06_05.md):
  - F11 (MEDIUM, FIXED): the integrated gate used the ADVISORY, cadence-sensitive
    `shift_sensitivity_test` (fixed-pp thresholds over-trigger on coarse bars) as a hard gate. Now uses
    the cadence-robust `relative_leak_test` against an AUTO-BUILT known-clean twin (a no-filter
    past-only crossover on the same df/windows) -- the validated design (RWYB: leaked-filter ratio 2.6
    vs clean ratio 1.0). Leak vector tested = the (exogenous) filter column; the fast/slow columns are
    past-only by construction (sma_past_only/wma_past_only).
  - F9 (MEDIUM, ENFORCED): StrategySpec defaults to maker cost (0.0010). The gate now WARNS loudly if a
    candidate is run below taker (0.0024) without explicit opt-in -- surfaced in `cost_warning`.
  - F2 (CRITICAL->HIGH, FIXED upstream): uses the firewall's own hardened `beats_held` flag (zero-trade
    windows now count as a FAIL), instead of re-deriving it with the buggy `is not None` filter.
"""
from __future__ import annotations

TAKER_COST_RT = 0.0024  # spot taker round-trip; the project's honest baseline


def build_clean_reference(harness):
    """A known-past-only TWIN of the candidate for relative_leak_test: same past-only fast/slow columns,
    NO filter (drops the exogenous leak vector), same windows/cost/df. Shares the candidate's cadence
    noise floor so it cancels in the leak ratio."""
    from wealth_bot.harness import CanonicalHarness, StrategySpec
    s = harness.spec
    ref_spec = StrategySpec(
        fast_col=s.fast_col, slow_col=s.slow_col, signal=s.signal,
        filter_col=None, filter_op=s.filter_op, filter_val=s.filter_val,
        exit_policy="signal_flip_or_filter",
        cost_rt=s.cost_rt, use_funding=s.use_funding, funding_col=s.funding_col,
        funding_scale=s.funding_scale, max_hold_bars=s.max_hold_bars, max_hold_ext_bars=s.max_hold_ext_bars,
    )
    return CanonicalHarness(harness.df, ref_spec, harness.windows,
                            chimera_path=str(harness.chimera_path) + "::clean_ref")


def evaluate_candidate(harness, family_n=None, n_books: int = 200, require_taker: bool = True) -> dict:
    """Run the full foundation gate on a constructed CanonicalHarness. Returns a consolidated verdict.
    SHIP requires: battery Lens A (strict) AND firewall beats-null on held-out AND no leak (relative)."""
    from wealth_bot.leak_probe import relative_leak_test
    try:  # package import (normal) vs script run (python src/strat/candidate_gate.py)
        from .firewall import random_entry_null
        from .battery import evaluate
        from .benchmark import benchmark_excess
    except ImportError:
        from strat.firewall import random_entry_null
        from strat.battery import evaluate
        from strat.benchmark import benchmark_excess

    res = harness.run()
    W = harness.WINDOWS
    comps = {w: res.window_stats[w].compound_pct for w in W}
    uns = [t["net_pnl"] for t in res.trades if t["window"] == "UNSEEN"]
    uns_pairs = [(t["entry_ts"], t["net_pnl"]) for t in res.trades if t["window"] == "UNSEEN"]
    uns_dd = res.window_stats["UNSEEN"].max_dd_pct

    # F11: cadence-robust relative leak verdict vs an auto-built clean twin
    ref = build_clean_reference(harness)
    if harness.spec.filter_col:
        leak = relative_leak_test(harness, ref)
        leak_verdict = leak["verdict"]
        leak_detail = {"ratio": leak["ratio"], "cand_pp": leak["candidate_delta_pp"], "ref_pp": leak["reference_delta_pp"]}
    else:
        leak_verdict = "PAST_ONLY_OK"  # no exogenous filter -> no leak vector; fast/slow are past-only by construction
        leak_detail = {"note": "no filter_col; relative test is vacuous"}

    fw = random_entry_null(harness, n_books=n_books)            # LD-4 firewall (F2-hardened)
    beats_held = bool(fw.get("beats_held"))                     # F2: firewall's hardened flag
    bench = benchmark_excess(harness)                           # STEP 5: beta-matched per-regime excess (incl. bear)
    beats_beta = bool(bench.get("beats_beta_held"))
    bat = evaluate(uns, comps, uns_dd, entry_pnl_pairs=uns_pairs, family_n=family_n, all_4_positive=res.all_4_positive)

    # F9: surface a cost warning if run below taker without opt-in
    cost_warning = None
    if require_taker and float(harness.spec.cost_rt) < TAKER_COST_RT - 1e-9:
        cost_warning = (f"cost_rt={harness.spec.cost_rt} is BELOW taker {TAKER_COST_RT} -- optimistic. "
                        "Verdict is NOT honest-cost. Pass cost_rt>=0.0024 or require_taker=False to silence.")

    ship = bool(bat["lens_A_strict"] and beats_held and beats_beta
                and leak_verdict == "PAST_ONLY_OK" and not cost_warning)
    return {
        "comps": {w: round(comps[w], 1) for w in W},
        "battery": {"verdict": bat["verdict"], "n": bat["n"], "n_eff": bat["n_eff"],
                    "jk3": bat["jk3"], "p05": bat["p05"], "concentration_flag": bat["concentration_flag"]},
        "firewall_beats_held_out": beats_held,
        "benchmark_excess": {"beats_beta_held": beats_beta, "bear_windows": bench.get("bear_windows"),
                             "bear_preserved": bench.get("bear_preserved"),
                             "per_window_excess_pp": {w: bench["per_window"][w].get("excess_pp") for w in W}},
        "leak_probe": {"verdict": leak_verdict, **leak_detail},
        "cost_warning": cost_warning,
        "CONSOLIDATED": "SHIP-TIER" if ship else (
            f"NOT-SHIP ({bat['verdict']}; firewall_held={beats_held}; beats_beta={beats_beta}; leak={leak_verdict})"),
    }


def _rwyb():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import json
    import numpy as np
    import pandas as pd
    from pipeline.chimera_loader import ChimeraLoader
    from wealth_bot.harness import CanonicalHarness, StrategySpec, WindowSpec, sma_past_only

    g = ChimeraLoader().load("PEPEUSDT", cadence="dollar"); d = g.to_dict(as_series=False)
    _raw = np.asarray(d["date"])
    _dt = pd.to_datetime(_raw, unit="ms") if np.issubdtype(_raw.dtype, np.number) else pd.to_datetime(_raw)
    df = pd.DataFrame({
        "date": _dt,
        "open": np.asarray(d["open"], float), "high": np.asarray(d.get("high", d["close"]), float),
        "low": np.asarray(d.get("low", d["close"]), float), "close": np.asarray(d["close"], float),
        "wh": np.asarray(d["wh_whale_net_usd"], float)})
    n = len(df); step = max(1, n // 6676); df["grp"] = np.arange(n) // step
    a = df.groupby("grp").agg(date=("date", "last"), open=("open", "first"), high=("high", "max"),
                              low=("low", "min"), close=("close", "last"), wh=("wh", "sum")).reset_index(drop=True)
    a = a.rename(columns={"wh": "wh_whale_net_usd"})
    a["sma_fast"] = sma_past_only(a["close"], 30); a["sma_slow"] = sma_past_only(a["close"], 50)
    spec = StrategySpec(fast_col="sma_fast", slow_col="sma_slow", signal="crossover",
                        filter_col="wh_whale_net_usd", filter_op="gt", filter_val=0.0,
                        exit_policy="signal_flip_or_filter", cost_rt=TAKER_COST_RT, use_funding=False,
                        funding_col="fund_rate_mean", funding_scale=0.0, max_hold_bars=18, max_hold_ext_bars=42)
    win = WindowSpec(train_end="2024-05-15", val_end="2025-03-15", oos_end="2025-12-31", unseen_end="2026-05-22")
    h = CanonicalHarness(a, spec, win, chimera_path="cand_gate_rwyb")
    print("[candidate_gate RWYB] PEPE coarse whale-gated slow-SMA, full hardened gate, family_n=6000:")
    print(json.dumps(evaluate_candidate(h, family_n=6000), indent=2, default=str))


if __name__ == "__main__":
    _rwyb()
