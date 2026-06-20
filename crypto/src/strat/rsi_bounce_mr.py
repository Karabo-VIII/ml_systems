"""src/strat/rsi_bounce_mr.py -- Mean-Reversion Bounce: buy oversold-in-uptrend, exit on bounce.

LANE: RSI oversold (< threshold) AND gate (C > sma200) -> enter; exit when RSI>50 OR
      take-profit +X% OR time-stop (N days), whichever fires first.

Variants grid:
  rsi_thresh: {25, 30, 35}
  tp:         {5%, 8%, 12%, None} (None = no hard TP, rely on rsi50 or time-stop)
  time_stop:  {3, 5, 7} days
  K:          {3, 5}   (max concurrent positions)

CAUSALITY: W.loc[d] uses only ind[...].loc[d] or earlier. Harness lags W by 1 bar.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import strat.mover_lab as ml


def build_mr_weights(
    ind: dict,
    rsi_thresh: float = 30.0,
    tp: float | None = 0.08,
    time_stop: int = 5,
    K: int = 3,
) -> pd.DataFrame:
    """
    Per-bar position matrix.

    Entry: at close d, rsi14[d] < rsi_thresh AND gate[d]==True.
    Hold: up to time_stop bars from entry.
    Exit (whichever fires first, checked at each bar's close):
      1) rsi14 >= 50  (bounce signal)
      2) cumulative return from entry >= tp  (take-profit)
      3) bars_held >= time_stop  (time-stop)

    We track each asset's 'hold counter' and 'entry price' day by day.
    W is EW over currently held positions, capped at K per rebal cycle.
    """
    C   = ind["C"]
    rsi = ind["rsi14"]
    gate = ind["gate"]

    dates  = C.index
    assets = C.columns
    n_dates = len(dates)

    # State per asset: hold_counter (0 = flat), entry_price
    hold  = {s: 0 for s in assets}
    epx   = {s: np.nan for s in assets}

    W = pd.DataFrame(0.0, index=dates, columns=assets)

    for i, d in enumerate(dates):
        if i == 0:
            continue  # skip first row (no prior close for entry)

        # --- update existing positions ---
        for s in assets:
            if hold[s] == 0:
                continue
            # check exits using today's close
            c_now   = C.loc[d, s]
            r_now   = rsi.loc[d, s]
            ret_cum = (c_now / epx[s] - 1) if not np.isnan(epx[s]) else 0.0
            should_exit = False
            if not np.isnan(r_now) and r_now >= 50.0:
                should_exit = True  # RSI bounce exit
            if tp is not None and ret_cum >= tp:
                should_exit = True  # take-profit
            if hold[s] >= time_stop:
                should_exit = True  # time-stop
            if should_exit:
                hold[s] = 0
                epx[s]  = np.nan
            else:
                hold[s] += 1

        # --- consider new entries (only if not already holding this asset) ---
        in_pos = [s for s in assets if hold[s] > 0]
        slots  = K - len(in_pos)
        if slots > 0:
            cands = []
            for s in assets:
                if hold[s] > 0:
                    continue  # already holding
                r_val = rsi.loc[d, s]
                g_val = gate.loc[d, s]
                if (not np.isnan(r_val)) and bool(g_val) and r_val < rsi_thresh:
                    cands.append((r_val, s))  # sort by lowest RSI first (most oversold)
            cands.sort(key=lambda x: x[0])
            for _, s in cands[:slots]:
                hold[s]  = 1
                epx[s]   = C.loc[d, s]

        # --- write today's weight (will be lagged by harness) ---
        held = [s for s in assets if hold[s] > 0]
        if held:
            w = 1.0 / len(held)
            for s in held:
                W.loc[d, s] = w

    return W


def run_all_variants(ind: dict) -> list[dict]:
    results = []

    # Reference: gated-beta (EW over all gated assets)
    gate = ind["gate"].astype(float)
    beta = gate.div(gate.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    m = ml.evaluate(beta, ind, label="gated-beta [REF]")
    results.append(m)

    # Grid
    rsi_thresholds = [25, 30, 35]
    tps            = [None, 0.05, 0.08, 0.12]
    time_stops     = [3, 5, 7]
    Ks             = [3, 5]

    total = len(rsi_thresholds) * len(tps) * len(time_stops) * len(Ks)
    done  = 0
    for rsi_t in rsi_thresholds:
        for tp_v in tps:
            for ts in time_stops:
                for K in Ks:
                    tp_tag = f"tp{int(tp_v*100)}%" if tp_v is not None else "noTP"
                    label  = f"MR_rsi{rsi_t}_{tp_tag}_ts{ts}_K{K}"
                    W      = build_mr_weights(ind, rsi_thresh=rsi_t, tp=tp_v, time_stop=ts, K=K)
                    m      = ml.evaluate(W, ind, label=label)
                    results.append(m)
                    done += 1
                    if done % 10 == 0:
                        print(f"  ...{done}/{total} done")

    return results


def _fmt(v):
    if v is None:
        return "    --"
    return f"{v:+7.1f}"


def print_table(results: list[dict]):
    header = (
        f"{'Config':<38} | {'2020':>7} | {'2021':>7} | {'2022':>7} | {'Full':>7} | "
        f"{'maxDD':>7} | {'gr21':>5} | {'grAll':>5} | {'expo':>5}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for m in results:
        print(
            f"{m['label']:<38} | {_fmt(m['comp_2020'])} | {_fmt(m['comp_2021'])} | "
            f"{_fmt(m['comp_2022'])} | {_fmt(m['comp_full'])} | "
            f"{_fmt(m['maxDD'])} | "
            f"{'--' if m['green_2021'] is None else int(m['green_2021']):>5} | "
            f"{int(m['green_all']):>5} | "
            f"{m['avg_expo']:>5.2f}"
        )
    print("=" * len(header))


def analyse(results: list[dict]):
    mr_only = [r for r in results if r["label"].startswith("MR_")]
    ref     = next(r for r in results if "REF" in r["label"])

    # best by comp_full
    best_full = max(mr_only, key=lambda r: r["comp_full"] or -9999)
    # best 2021
    best_2021 = max(mr_only, key=lambda r: r["comp_2021"] or -9999)
    # best 2022
    best_2022 = max(mr_only, key=lambda r: r["comp_2022"] or -9999)
    # best green_2021
    best_gr21 = max(mr_only, key=lambda r: r["green_2021"] or -9999)
    # greediest (highest avg_expo)
    greediest = max(mr_only, key=lambda r: r["avg_expo"])

    print("\n--- SUMMARY ---")
    print(f"Reference (gated-beta): full={ref['comp_full']:+.1f}%, 2021={ref['comp_2021']:+.1f}%, 2022={ref['comp_2022']:+.1f}%, maxDD={ref['maxDD']:.1f}%, green21={ref['green_2021']}")
    print(f"\nBEST by full-cycle: {best_full['label']}")
    print(f"  full={best_full['comp_full']:+.1f}%, 2021={best_full['comp_2021']:+.1f}%, 2022={best_full['comp_2022']:+.1f}%, maxDD={best_full['maxDD']:.1f}%, green21={best_full['green_2021']}")
    print(f"\nBEST 2021 (bull): {best_2021['label']}")
    print(f"  full={best_2021['comp_full']:+.1f}%, 2021={best_2021['comp_2021']:+.1f}%, 2022={best_2021['comp_2022']:+.1f}%, maxDD={best_2021['maxDD']:.1f}%")
    print(f"\nBEST 2022 (bear): {best_2022['label']}")
    print(f"  full={best_2022['comp_full']:+.1f}%, 2021={best_2022['comp_2021']:+.1f}%, 2022={best_2022['comp_2022']:+.1f}%, maxDD={best_2022['maxDD']:.1f}%")
    print(f"\nBEST green_2021: {best_gr21['label']} -> {best_gr21['green_2021']}%")
    print(f"\nGREEDIEST (highest expo): {greediest['label']} -> expo={greediest['avg_expo']:.2f}")
    print(f"  full={greediest['comp_full']:+.1f}%, 2021={greediest['comp_2021']:+.1f}%, 2022={greediest['comp_2022']:+.1f}%, maxDD={greediest['maxDD']:.1f}%")


if __name__ == "__main__":
    print("Loading data...")
    ind = ml.load()
    print(f"  assets: {list(ind['C'].columns)}, dates: {ind['C'].index[0].date()} -> {ind['C'].index[-1].date()}")

    print("\nRunning variant grid...")
    results = run_all_variants(ind)

    print_table(results)
    analyse(results)
