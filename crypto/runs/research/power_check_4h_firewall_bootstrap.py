"""runs/research/power_check_4h_firewall_bootstrap.py -- POWER CHECK of the 4h harness.

TASK: a 4h null is only trustworthy once the apparatus is shown CAPABLE of detecting a true edge at
this cadence. So inject a KNOWN-detectable signal (positive control) and confirm BOTH halves of the gate
FLAG it as significant:
  (1) the FIREWALL  -- cost-matched random-ENTRY null (regime-matched), compound + per-trade-expectancy
  (2) the BOOTSTRAP -- stationary block-bootstrap of the held-out trade returns (battery p05 > 0)

This REUSES the real apparatus functions (no reimplementation):
  - make_positive_control_4h / make_harness / per_trade_expectancy_null  <- firewall_4h_regime_matched.py
  - random_entry_null                                                    <- src/strat/firewall.py
  - block_bootstrap_p05_p95 / evaluate                                   <- src/strat/battery.py

Two-sided: also runs a PURE-NOISE 4h negative control through the SAME chain -- the firewall must NOT
beat it and the bootstrap p05 must NOT be > 0. A gate that flags noise as significant is broken too.

NO commit / NO deploy. RWYB: `python runs/research/power_check_4h_firewall_bootstrap.py`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))  # import the existing 4h firewall module as a library

from firewall_4h_regime_matched import (  # the REAL positive-control + harness + per-trade null
    make_positive_control_4h, make_harness, per_trade_expectancy_null, WIN, HELD, N_BOOKS, SEED,
)
from strat.firewall import random_entry_null
from strat.battery import block_bootstrap_p05_p95, evaluate, herfindahl_neff, jackknife


def _make_noise_4h(seed: int = 31) -> pd.DataFrame:
    """PURE-NOISE 4h OHLC: SAME background and ER/EMA machinery, but NO planted impulse edge.
    The ER gate still fires on random runs of same-sign noise, so the harness still trades -- but there is
    no within-gate timing structure to capture. The negative control."""
    dates = pd.date_range(start="2020-01-07", end="2026-05-28", freq="4h")
    n = len(dates)
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.006, n)          # zero-drift noise, comparable scale to the control background
    close = 100.0 * np.cumprod(1.0 + rets)
    open_ = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.0015, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.0015, n)))
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close})


def _held_returns(harness):
    """Per-window + pooled held-out (OOS+UNSEEN) NET per-trade returns from the REAL harness run."""
    res = harness.run()
    by_w = {w: [t["net_pnl"] for t in res.trades if t["window"] == w] for w in harness.WINDOWS}
    held = [x for w in HELD for x in by_w[w]]
    comps = {w: round(res.window_stats[w].compound_pct, 3) for w in harness.WINDOWS}
    dd = {w: round(res.window_stats[w].max_dd_pct, 2) for w in harness.WINDOWS}
    return res, by_w, held, comps, dd


def evaluate_chain(df, tag):
    h = make_harness(df)
    res, by_w, held, comps, dd = _held_returns(h)

    # ---- (1) FIREWALL: compound random-entry null + per-trade-expectancy null (both regime-matched) ----
    fw = random_entry_null(h, n_books=300, seed=SEED, regime_matched=True)
    pte = per_trade_expectancy_null(h, n_books=N_BOOKS, seed=SEED, regime_matched=True)

    # ---- (2) BOOTSTRAP: stationary block-bootstrap of the held-out trade returns (battery p05) ----
    bb_unseen = block_bootstrap_p05_p95(by_w["UNSEEN"])
    bb_held = block_bootstrap_p05_p95(held)
    # full battery verdict on UNSEEN (the held-out gate surface)
    bat = evaluate(by_w["UNSEEN"], comps, dd["UNSEEN"], family_n=1)

    n_counts = {w: len(by_w[w]) for w in h.WINDOWS}
    return {
        "tag": tag,
        "n_trades_per_window": n_counts,
        "compound_per_window": comps,
        "unseen_maxdd_pct": dd["UNSEEN"],
        # firewall outputs
        "firewall_compound_beats_held": bool(fw["beats_held"]),
        "firewall_compound_pos_held": bool(fw["pos_held"]),
        "firewall_compound_verdict": fw["verdict"],
        "firewall_pte_PASS": bool(pte["PASS_per_trade_expectancy"]),
        "firewall_pte_combined": pte["combined_held_out"],
        # bootstrap outputs
        "bootstrap_unseen": bb_unseen,
        "bootstrap_held_OOS+UNSEEN": bb_held,
        "bootstrap_held_p05_gt_0": (bb_held["p05"] is not None and bb_held["p05"] > 0),
        "bootstrap_unseen_p05_gt_0": (bb_unseen["p05"] is not None and bb_unseen["p05"] > 0),
        # battery verdict + concentration diagnostics
        "battery_verdict": bat["verdict"],
        "battery_p05": bat["p05"], "battery_n_eff": bat["n_eff"], "battery_jk3": bat["jk3"],
        "battery_n_unseen": bat["n"],
    }


def fmt(d):
    print(f"\n===== {d['tag']} =====")
    print(f"  n_trades/window      : {d['n_trades_per_window']}")
    print(f"  compound/window      : {d['compound_per_window']}")
    print(f"  FIREWALL  compound   : beats_held={d['firewall_compound_beats_held']} "
          f"pos_held={d['firewall_compound_pos_held']} :: {d['firewall_compound_verdict']}")
    c = d["firewall_pte_combined"]
    if c:
        print(f"  FIREWALL  per-trade  : PASS={d['firewall_pte_PASS']}  held real_exp={c['real_exp_pct']:+.4f}% "
              f"vs null p95={c['null_p95_pct']:+.4f}%  pctile_rank={c['pctile_rank']}  beats_p95={c['beats_null_p95']}")
    print(f"  BOOTSTRAP held(O+U)  : p05={d['bootstrap_held_OOS+UNSEEN']['p05']} "
          f"p50={d['bootstrap_held_OOS+UNSEEN']['p50']} p95={d['bootstrap_held_OOS+UNSEEN']['p95']}  "
          f"=> p05>0 (significant)={d['bootstrap_held_p05_gt_0']}")
    print(f"  BOOTSTRAP UNSEEN     : p05={d['bootstrap_unseen']['p05']} p50={d['bootstrap_unseen']['p50']} "
          f"p95={d['bootstrap_unseen']['p95']}  => p05>0={d['bootstrap_unseen_p05_gt_0']}")
    print(f"  BATTERY (UNSEEN)     : verdict={d['battery_verdict']} p05={d['battery_p05']} "
          f"n={d['battery_n_unseen']} n_eff={d['battery_n_eff']} jk3={d['battery_jk3']}")


def main():
    print("=" * 96)
    print("POWER CHECK @4h -- a KNOWN-detectable signal must be FLAGGED by BOTH the firewall AND the bootstrap")
    print("=" * 96)

    pos = evaluate_chain(make_positive_control_4h(), "POSITIVE CONTROL (planted within-gate 4h timing edge)")
    neg = evaluate_chain(_make_noise_4h(), "NEGATIVE CONTROL (pure-noise 4h, same gate machinery)")
    fmt(pos); fmt(neg)

    # ---- soundness verdicts ----
    firewall_flags_pos = pos["firewall_compound_beats_held"] and pos["firewall_pte_PASS"]
    bootstrap_flags_pos = pos["bootstrap_held_p05_gt_0"]
    firewall_clears_neg = not (neg["firewall_compound_beats_held"] and neg["firewall_pte_PASS"])
    bootstrap_clears_neg = not neg["bootstrap_held_p05_gt_0"]

    print("\n" + "=" * 96)
    print("SOUNDNESS (two-sided):")
    print(f"  POS: firewall FLAGS the planted edge (compound beats_held AND pte PASS) : {firewall_flags_pos}")
    print(f"  POS: bootstrap FLAGS the planted edge (held-out p05 > 0)                : {bootstrap_flags_pos}")
    print(f"  NEG: firewall does NOT flag pure noise                                  : {firewall_clears_neg}")
    print(f"  NEG: bootstrap does NOT flag pure noise (held-out p05 <= 0)             : {bootstrap_clears_neg}")
    power_confirmed = firewall_flags_pos and bootstrap_flags_pos
    two_sided = power_confirmed and firewall_clears_neg and bootstrap_clears_neg
    print("\n" + "=" * 96)
    print(f"POWER CONFIRMED (firewall + bootstrap both flag the genuine 4h edge): {power_confirmed}")
    print(f"TWO-SIDED (also reject pure noise)                                  : {two_sided}")
    print("=" * 96)

    out = {"power_confirmed_firewall_and_bootstrap": bool(power_confirmed), "two_sided": bool(two_sided),
           "positive_control": pos, "negative_control": neg}
    outpath = HERE / "power_check_4h_firewall_bootstrap_result.json"
    outpath.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[wrote] {outpath}")
    return out


if __name__ == "__main__":
    main()
