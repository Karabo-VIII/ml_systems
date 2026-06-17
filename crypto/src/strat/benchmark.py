"""src/strat/benchmark.py -- STEP 5 of the gate chain: benchmark-EXCESS per regime (incl. bear).

Built 2026-06-05 to close the foundation completeness gap G-1 (the gate chain listed STEP 5 in every
avenue spec but no callable existed -- only a BTC-1d bespoke script). This is generic: it wraps ANY
CanonicalHarness and asks the question the firewall does NOT: does the candidate beat a BETA-MATCHED
PASSIVE HOLD in each window -- especially does it PRESERVE capital in a bear window?

beta-matched static = hold the asset at the candidate's own time-in-market fraction f (rest in cash),
COSTLESS (a static position barely trades). The candidate's compound is NET (after taker cost). If the
candidate (net) beats a costless f-exposure passive hold, it is adding value beyond passive beta. The
firewall tests "is it timing or random?"; this tests "is it better than just holding f of the asset?".

Bear clause: a window where buy-and-hold is negative is a BEAR window. There the bar is capital
PRESERVATION -- the candidate must lose LESS than buy-and-hold (excess > 0 still works, since a smaller
loss beats the passive f-exposure loss). The verdict reports beats_beta per window + a bear flag.
"""
from __future__ import annotations

import numpy as np


def benchmark_excess(harness) -> dict:
    """Per-window candidate-net vs beta-matched-passive (costless) compound. Returns per-window excess +
    a held-out verdict. beta-matched uses the candidate's own time-in-market fraction f per window."""
    res = harness.run()
    df = harness.df
    n = len(df)
    close = df["close"].to_numpy(float)
    ret_bar = np.zeros(n)
    ret_bar[1:] = close[1:] / close[:-1] - 1.0
    import pandas as pd
    wlab = np.array([harness._window_label(pd.Timestamp(df["date"].iloc[i])) for i in range(n)])

    # per-bar in-position mask from the candidate's trades (entry_fill_idx .. exit_idx)
    pos = np.zeros(n, bool)
    for t in res.trades:
        a, b = int(t["entry_fill_idx"]), int(t["exit_idx"])
        if 0 <= a < b <= n:
            pos[a:b] = True

    windows = list(harness.WINDOWS)
    out = {}
    for w in windows:
        idx = np.where(wlab == w)[0]
        if idx.size < 2:
            out[w] = {"cand_pct": None, "beta_matched_pct": None, "buyhold_pct": None,
                      "excess_pp": None, "exposure_f": None, "is_bear": None, "beats_beta": None}
            continue
        r = ret_bar[idx]
        f = float(pos[idx].mean())                     # candidate time-in-market fraction
        buyhold = float((np.prod(1.0 + r) - 1.0) * 100)
        beta_matched = float((np.prod(1.0 + f * r) - 1.0) * 100)  # costless static f-exposure
        cand = float(res.window_stats[w].compound_pct)
        excess = cand - beta_matched
        is_bear = buyhold < 0.0
        out[w] = {"cand_pct": round(cand, 2), "beta_matched_pct": round(beta_matched, 2),
                  "buyhold_pct": round(buyhold, 2), "excess_pp": round(excess, 2),
                  "exposure_f": round(f, 3), "is_bear": bool(is_bear), "beats_beta": bool(excess > 0)}

    held = ["OOS", "UNSEEN"]
    beats_beta_held = all(out[w].get("beats_beta") is True for w in held)
    bear_windows = [w for w in windows if out[w].get("is_bear")]
    bear_preserved = all(out[w].get("beats_beta") is True for w in bear_windows) if bear_windows else None
    return {"per_window": out, "beats_beta_held": beats_beta_held,
            "bear_windows": bear_windows, "bear_preserved": bear_preserved,
            "note": "beta-matched = costless static hold at the candidate's own time-in-market fraction f"}


def _rwyb():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from .positive_control import make_edge_frame
    except ImportError:
        from strat.positive_control import make_edge_frame
    from wealth_bot.harness import CanonicalHarness, StrategySpec, WindowSpec, sma_past_only
    import json

    # the genuine synthetic edge (should beat beta-matched on held-out: it times the bull, sits out bear)
    df = make_edge_frame()
    df["sma_fast"] = sma_past_only(df["close"], 2)
    df["sma_slow"] = sma_past_only(df["close"], 5)
    spec = StrategySpec(fast_col="sma_fast", slow_col="sma_slow", signal="crossover", filter_col=None,
                        exit_policy="signal_flip_or_filter", cost_rt=0.0024, use_funding=False,
                        funding_scale=0.0, max_hold_bars=None, max_hold_ext_bars=None)
    win = WindowSpec(train_end="2024-05-15", val_end="2025-03-15", oos_end="2025-12-31", unseen_end="2026-05-22")
    h = CanonicalHarness(df, spec, win, chimera_path="benchmark_rwyb")
    print("[benchmark RWYB] genuine synthetic timing edge vs beta-matched passive (STEP 5):")
    print(json.dumps(benchmark_excess(h), indent=2, default=str))


if __name__ == "__main__":
    _rwyb()
