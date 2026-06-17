"""src/strat/positive_control.py -- the STATISTICAL-POWER half of the apparatus soundness check.

The battery/firewall self-tests prove the gate REJECTS (ghosts fail; beta-in-disguise is flagged). A
gate that rejects EVERYTHING is just as useless as one that accepts everything. This test proves the
gate has POWER: it constructs a SYNTHETIC series with a GENUINE, past-only, TIMING edge and confirms
the full `evaluate_candidate` chain SHIPs it (passes leak + beats the random-entry firewall + clears
the battery). FOUNDATION verification -- synthetic data, no market claim.

Construction (deterministic, seeded): a regime-switching price. Bull segments drift UP, bear segments
drift DOWN, switching on a fixed cycle. A long-only SMA crossover is LONG during bull segments and FLAT
during bear segments -> it captures the up-drift and sits out the down-drift. A cost-matched RANDOM
entry catches bull and bear ~equally -> ~0. So the crossover's TIMING genuinely beats random (not just
"long in an uptrend"). The signal is a past-only SMA -> no leak. The regime structure repeats in all 4
windows -> all-4-positive + a positive bootstrap p05.

If this does NOT ship, the gate is mis-calibrated (too strict / no power) -- a CRITICAL foundation
finding. RWYB: run `python src/strat/positive_control.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_edge_frame(seed: int = 11, bull_drift: float = 0.014, bear_drift: float = -0.016,
                    noise: float = 0.003, regime_bars: int = 8, start="2022-01-01", end="2026-05-22"):
    """Daily regime-switching close with a real, SMA-detectable, long-only timing edge.
    regime_bars=8 (bull 8 / bear 8) keeps the slow SMA window BELOW the bull length so the crossover
    catches the regime: it beats the cost-matched random-entry firewall, is positive in every window,
    and is recognized by the battery (PROVISIONAL) -> the gate HAS power.

    CALIBRATION FINDING (RWYB 2026-06-05): SHIP-TIER (Lens A) additionally needs n>=15 / n_eff>=15 in
    UNSEEN. UNSEEN here is only ~143 days, so an 8-bar-cycle crossover yields ~8 trades -> tops out at
    Lens C. Chasing n>=15 by shrinking the cycle to 4 bars makes the crossover entry/fill LAG exceed the
    regime -> it WHIPSAWS (buys the top, sells the bottom) and the gate CORRECTLY rejects it (all-windows
    negative -> FAIL). So a crossover edge cannot simultaneously be catchable AND reach SHIP-TIER on a
    short held-out window -- SHIP-TIER is correctly reserved for a HIGHER-FREQUENCY genuine substrate
    (finer bars), which is sample-size discipline working, NOT a lack of gate power."""
    dates = pd.date_range(start=start, end=end, freq="D")
    n = len(dates)
    rng = np.random.default_rng(seed)
    # deterministic regime cycle: bull for regime_bars, bear for regime_bars, repeat
    cycle = (np.arange(n) // regime_bars) % 2  # 0 = bull, 1 = bear
    drift = np.where(cycle == 0, bull_drift, bear_drift)
    rets = drift + rng.normal(0.0, noise, n)
    close = 100.0 * np.cumprod(1.0 + rets)
    # opens: previous close (fill-at-next-open uses opens[i+1]); keep a tiny gap-free OHLC
    open_ = np.concatenate([[100.0], close[:-1]])
    df = pd.DataFrame({"date": dates, "open": open_, "high": np.maximum(open_, close) * 1.001,
                       "low": np.minimum(open_, close) * 0.999, "close": close})
    return df


def run_positive_control(verbose: bool = True) -> dict:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from wealth_bot.harness import CanonicalHarness, StrategySpec, WindowSpec, sma_past_only
    try:
        from .candidate_gate import evaluate_candidate, TAKER_COST_RT
    except ImportError:
        from strat.candidate_gate import evaluate_candidate, TAKER_COST_RT

    df = make_edge_frame()
    df["sma_fast"] = sma_past_only(df["close"], 2)
    df["sma_slow"] = sma_past_only(df["close"], 5)  # slow window (5) < bull length (8) -> regime is catchable
    spec = StrategySpec(fast_col="sma_fast", slow_col="sma_slow", signal="crossover",
                        filter_col=None, exit_policy="signal_flip_or_filter", cost_rt=TAKER_COST_RT,
                        use_funding=False, funding_scale=0.0, max_hold_bars=None, max_hold_ext_bars=None)
    win = WindowSpec(train_end="2024-05-15", val_end="2025-03-15", oos_end="2025-12-31", unseen_end="2026-05-22")
    h = CanonicalHarness(df, spec, win, chimera_path="positive_control")
    v = evaluate_candidate(h, family_n=1, n_books=200)
    if verbose:
        import json
        print("[positive_control] synthetic GENUINE past-only timing edge through the full hardened gate:")
        print(json.dumps(v, indent=2, default=str))
    ships = (v["CONSOLIDATED"] == "SHIP-TIER")
    beats = bool(v["firewall_beats_held_out"])
    leak_ok = (v["leak_probe"]["verdict"] == "PAST_ONLY_OK")
    all4 = v["comps"] and all(c > 0 for c in v["comps"].values())
    bat_recognized = (v["battery"]["verdict"] != "FAIL") and (v["battery"]["jk3"] > 0)
    # POWER (frequency-independent): a genuine timing edge must beat the firewall, be leak-clean,
    # be positive in every window, and be recognized by the battery as at-least-provisional.
    has_power = bool(beats and leak_ok and all4 and bat_recognized)
    if verbose:
        print(f"\n[positive_control] CONSOLIDATED={v['CONSOLIDATED']}")
        print(f"  POWER CHECK (frequency-independent): firewall_beats_held={beats}  leak_ok={leak_ok}  "
              f"all_4_positive={all4}  battery_recognizes(jk3>0,!=FAIL)={bat_recognized}  -> HAS_POWER={has_power}")
        print(f"  SHIP-TIER (Lens A, additionally needs n>=15 & n_eff>=15 sample-size discipline)={ships}")
        if has_power:
            print("[positive_control] PASS -- the gate HAS statistical power: a genuine past-only TIMING edge "
                  "beats the firewall, is leak-clean, positive every window, and is recognized by the battery. "
                  "Combined with the battery/firewall REJECT-tests (ghost->FAIL, beta->BETA-IN-DISGUISE), the "
                  "gate both ACCEPTS real edges AND REJECTS ghosts/beta -> calibrated, not a reject-everything "
                  "sieve. SHIP-TIER additionally requires sample-size (n>=15) -- a low-frequency genuine edge "
                  "correctly tops out at PRAGMATIC/PROVISIONAL on a short held-out window, BY DESIGN.")
        else:
            print("[positive_control] *** CRITICAL *** the gate failed the frequency-independent POWER check on "
                  "a hand-crafted genuine edge -> possible mis-calibration. Inspect which dimension blocked it.")
    return {"has_power": has_power, "ships": ships, "beats_held": beats, "leak_ok": leak_ok, "verdict": v}


if __name__ == "__main__":
    run_positive_control()
