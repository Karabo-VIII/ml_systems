"""src/strat/exit_policy_sweep.py -- EXIT POLICY SWEEP on fixed entry (mom14 top3 gated, rebal 3d).

LANE: THE EXIT IS THE EDGE.
Entry is FIXED: mom14 top-3 gated assets, EW, rebalance every 3 days.
We sweep EXIT/HOLDING policies and quantify the bag-profit-vs-let-run tradeoff.

Exit policies:
  (1) flush3d          -- flush every 3d (baseline, equivalent to topk_weight rebal=3)
  (2) letrun           -- let winners run: hold while STILL gated AND still top-K, no time cap
  (3) tp10/tp20/tp30   -- take-profit at +10/+20/+30% from entry, then cash
  (4) atr2/atr3/atr4   -- ATR trailing stop (k * atr14 below running peak)
  (5) stop3d/stop7d/stop14d -- time-stop: exit after 3/7/14 bars from entry
  (6) sigflip          -- exit when mom14 turns negative (signal-flip)

CAUSALITY: W.loc[d] uses only ind[...].loc[d] or earlier. Harness lags W by 1 bar.
RWYB: python -m strat.exit_policy_sweep
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.mover_lab as lab

# ---- constants ----
K = 3       # top-3 by mom14
REBAL = 3   # entry signal checked every 3 days
GATE = True


# ============================================================
# Helper: build a base entry schedule (which assets are chosen each rebal day)
# ============================================================
def _entry_schedule(ind):
    """Returns a dict: {date_index_position -> [list of chosen syms]}."""
    C = ind["C"]
    g = ind["gate"]
    score = ind["mom14"]
    schedule = {}
    last = -999
    current_picks = []
    for i, d in enumerate(C.index):
        if i - last >= REBAL:
            elig = [s for s in C.columns if bool(g.loc[d, s]) and pd.notna(score.loc[d, s])]
            if elig:
                current_picks = sorted(elig, key=lambda s: -score.loc[d, s])[:K]
            else:
                current_picks = []
            schedule[i] = current_picks
            last = i
        else:
            schedule[i] = None  # carry
    return schedule


# ============================================================
# (1) FLUSH EVERY 3d -- just use topk_weight with rebal=3 (baseline)
# ============================================================
def w_flush3d(ind):
    return lab.topk_weight(ind["mom14"], ind, K=K, gate=GATE, rebal=3)


# ============================================================
# (2) LET-WINNERS-RUN: hold while still gated AND still in top-K, no time cap.
#     On rebal days, drop assets no longer gated or no longer top-K; add new top-K.
# ============================================================
def w_letrun(ind):
    C = ind["C"]
    g = ind["gate"]
    score = ind["mom14"]
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    held = set()
    last = -999
    for i, d in enumerate(C.index):
        if i - last >= REBAL:
            elig = [s for s in C.columns if bool(g.loc[d, s]) and pd.notna(score.loc[d, s])]
            # keep currently held that are still gated
            still_held = {s for s in held if s in elig}
            # top-K from eligible
            top_k = set(sorted(elig, key=lambda s: -score.loc[d, s])[:K])
            # union: carry winners + new top-K picks (cap at 2K to avoid dilution blow-up)
            new_held = (still_held | top_k)
            # if over K, keep the highest scorers
            if len(new_held) > K:
                new_held = set(sorted(new_held, key=lambda s: -score.loc[d, s])[:K])
            held = new_held
            last = i
        else:
            # carry day: drop any no longer gated
            held = {s for s in held if bool(g.loc[d, s]) and pd.notna(score.loc[d, s])}

        if held:
            for s in held:
                W.loc[d, s] = 1.0 / len(held)
    return W


# ============================================================
# (3) TAKE-PROFIT at threshold T% from entry price
#     Track entry price per asset; once price >= entry*(1+T), go to cash until next rebal.
# ============================================================
def w_takeprofit(ind, threshold):
    C = ind["C"]
    g = ind["gate"]
    score = ind["mom14"]
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    entry_price = {}   # sym -> entry close price
    in_profit_exit = set()  # syms currently in post-TP cash
    held = set()
    last = -999
    dates = list(C.index)
    for i, d in enumerate(dates):
        if i - last >= REBAL:
            # re-enter: clear TP-exits, re-score
            elig = [s for s in C.columns if bool(g.loc[d, s]) and pd.notna(score.loc[d, s])]
            top_k = sorted(elig, key=lambda s: -score.loc[d, s])[:K]
            new_held = set(top_k)
            # clear TP status for new picks; re-record entry prices
            for s in new_held:
                if s not in held:
                    entry_price[s] = float(C.loc[d, s])
                    in_profit_exit.discard(s)
            # drop departed positions
            in_profit_exit -= (in_profit_exit - new_held)
            held = new_held
            last = i
        else:
            pass  # carry day

        # apply TP: if price >= entry*(1+T), move to cash
        for s in list(held):
            if s in in_profit_exit:
                continue
            if s in entry_price and pd.notna(C.loc[d, s]):
                if C.loc[d, s] >= entry_price[s] * (1 + threshold):
                    in_profit_exit.add(s)

        active = {s for s in held if s not in in_profit_exit and bool(g.loc[d, s])}
        if active:
            for s in active:
                W.loc[d, s] = 1.0 / len(active)
    return W


# ============================================================
# (4) ATR TRAILING STOP: exit when price < (running_peak - k * atr14)
#     Per asset: track peak since entry. If price drops below trail, exit until next rebal.
# ============================================================
def w_atr_trail(ind, k):
    C = ind["C"]
    atr = ind["atr14"]
    g = ind["gate"]
    score = ind["mom14"]
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    peak_since_entry = {}  # sym -> float
    trail_exit = set()     # syms stopped out
    held = set()
    last = -999
    for i, d in enumerate(C.index):
        if i - last >= REBAL:
            elig = [s for s in C.columns if bool(g.loc[d, s]) and pd.notna(score.loc[d, s])]
            top_k = sorted(elig, key=lambda s: -score.loc[d, s])[:K]
            new_held = set(top_k)
            # reset stops for new/re-entering positions
            for s in new_held:
                if s not in held or s in trail_exit:
                    peak_since_entry[s] = float(C.loc[d, s])
            trail_exit -= (trail_exit - new_held)  # clear exits for positions no longer held
            held = new_held
            last = i

        # update peaks and check trail
        for s in list(held):
            if s in trail_exit:
                continue
            if pd.notna(C.loc[d, s]):
                peak_since_entry[s] = max(peak_since_entry.get(s, float(C.loc[d, s])), float(C.loc[d, s]))
                atr_val = atr.loc[d, s] if pd.notna(atr.loc[d, s]) else 0.0
                trail_level = peak_since_entry[s] - k * atr_val
                if float(C.loc[d, s]) < trail_level:
                    trail_exit.add(s)

        active = {s for s in held if s not in trail_exit and bool(g.loc[d, s])}
        if active:
            for s in active:
                W.loc[d, s] = 1.0 / len(active)
    return W


# ============================================================
# (5) TIME-STOP: exit after N bars from entry (regardless of performance)
#     Re-enter at next rebal if still top-K gated.
# ============================================================
def w_timestop(ind, N):
    C = ind["C"]
    g = ind["gate"]
    score = ind["mom14"]
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    entry_bar = {}   # sym -> bar index when entered
    time_exit = set()
    held = set()
    last = -999
    for i, d in enumerate(C.index):
        if i - last >= REBAL:
            elig = [s for s in C.columns if bool(g.loc[d, s]) and pd.notna(score.loc[d, s])]
            top_k = sorted(elig, key=lambda s: -score.loc[d, s])[:K]
            new_held = set(top_k)
            for s in new_held:
                if s not in held or s in time_exit:
                    entry_bar[s] = i
            time_exit -= (time_exit - new_held)
            held = new_held
            last = i

        # check time-stop
        for s in list(held):
            if s in time_exit:
                continue
            if i - entry_bar.get(s, i) >= N:
                time_exit.add(s)

        active = {s for s in held if s not in time_exit and bool(g.loc[d, s])}
        if active:
            for s in active:
                W.loc[d, s] = 1.0 / len(active)
    return W


# ============================================================
# (6) SIGNAL-FLIP: exit when mom14 turns negative for that asset
# ============================================================
def w_sigflip(ind):
    C = ind["C"]
    g = ind["gate"]
    score = ind["mom14"]
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    held = set()
    sig_exit = set()
    last = -999
    for i, d in enumerate(C.index):
        if i - last >= REBAL:
            elig = [s for s in C.columns if bool(g.loc[d, s]) and pd.notna(score.loc[d, s])]
            top_k = sorted(elig, key=lambda s: -score.loc[d, s])[:K]
            # re-admit if signal is now positive again
            new_held = set(top_k)
            sig_exit -= (sig_exit - new_held)
            held = new_held
            last = i

        # check signal flip: exit if mom14 < 0
        for s in list(held):
            if s in sig_exit:
                continue
            v = score.loc[d, s]
            if pd.notna(v) and v < 0:
                sig_exit.add(s)

        active = {s for s in held if s not in sig_exit and bool(g.loc[d, s])}
        if active:
            for s in active:
                W.loc[d, s] = 1.0 / len(active)
    return W


# ============================================================
# MAIN: build all variants, evaluate, print ranked table
# ============================================================
def main():
    print("Loading mover_lab data...")
    ind = lab.load()
    print(f"  Universe: {list(ind['C'].columns)}")
    print(f"  Date range: {ind['C'].index[0].date()} -> {ind['C'].index[-1].date()}")
    print()

    variants = []

    print("Building weight matrices...")

    # (1) Flush every 3d
    W = w_flush3d(ind)
    variants.append(("flush3d", W))

    # (2) Let winners run
    W = w_letrun(ind)
    variants.append(("letrun", W))

    # (3) Take-profit
    for t in [0.10, 0.20, 0.30]:
        W = w_takeprofit(ind, t)
        label = f"tp{int(t*100)}"
        variants.append((label, W))

    # (4) ATR trailing stop
    for k in [2, 3, 4]:
        W = w_atr_trail(ind, k)
        variants.append((f"atr_trail_k{k}", W))

    # (5) Time-stop
    for N in [3, 7, 14]:
        W = w_timestop(ind, N)
        variants.append((f"stop_{N}d", W))

    # (6) Signal flip
    W = w_sigflip(ind)
    variants.append(("sigflip", W))

    print(f"  Built {len(variants)} variants.")
    print()

    # Evaluate all
    results = []
    for label, W in variants:
        m = lab.evaluate(W, ind, H=3, label=label)
        results.append(m)
        print(f"  Evaluated {label}: comp_full={m['comp_full']:.1f}% green_2021={m['green_2021']}")

    print()

    # ---- Print ranked table ----
    print("=" * 110)
    print("EXIT POLICY SWEEP -- mom14 top3 gated, rebal 3d, sweep EXIT")
    print("Reference (gated-beta): 2021 +651% full +385% maxDD -80% green2021 61%")
    print("=" * 110)

    # Sort by comp_full descending
    results_sorted = sorted(results, key=lambda r: r["comp_full"] or -9999, reverse=True)

    HDR = f"{'EXIT':<18} {'2020%':>7} {'2021%':>8} {'2022%':>8} {'FULL%':>8} {'maxDD%':>8} {'grn21':>7} {'grnALL':>8} {'expo':>6} {'turn':>7}"
    print(HDR)
    print("-" * 110)
    for r in results_sorted:
        row = (
            f"{r['label']:<18} "
            f"{r['comp_2020'] or 'n/a':>7} "
            f"{r['comp_2021'] or 'n/a':>8} "
            f"{r['comp_2022'] or 'n/a':>8} "
            f"{r['comp_full']:>8.1f} "
            f"{r['maxDD']:>8.1f} "
            f"{r['green_2021'] or 'n/a':>7} "
            f"{r['green_all']:>8.1f} "
            f"{r['avg_expo']:>6.2f} "
            f"{r['avg_turnover']:>7.4f}"
        )
        print(row)

    print("=" * 110)
    print()

    # ---- Analysis ----
    best = results_sorted[0]
    # "greediest" = highest green_all (most consistently profitable windows)
    greediest = max(results, key=lambda r: r["green_all"] or 0)
    # Worst in bear
    worst_bear = min(results, key=lambda r: r["comp_2022"] or 9999)
    best_bear = max(results, key=lambda r: r["comp_2022"] or -9999)

    print(f"BEST (by comp_full):    {best['label']}  "
          f"full={best['comp_full']:.1f}%  2021={best['comp_2021']}%  maxDD={best['maxDD']}%")
    print(f"GREEDIEST (green_all):  {greediest['label']}  "
          f"green_all={greediest['green_all']:.0f}%  comp_full={greediest['comp_full']:.1f}%")
    print(f"WORST BEAR (2022):      {worst_bear['label']}  2022={worst_bear['comp_2022']}%")
    print(f"BEST BEAR (2022):       {best_bear['label']}  2022={best_bear['comp_2022']}%")
    print()

    # ---- Bag-profit vs Let-run comparison ----
    print("BAG-PROFIT vs LET-RUN TRADEOFF:")
    print(f"{'EXIT':<18} {'FULL%':>8} {'2021%':>8} {'grn21':>7} {'grnALL':>8} {'ASSESSMENT'}")
    tp_variants = [r for r in results if r["label"].startswith("tp") or r["label"] in ("flush3d", "letrun", "sigflip")]
    tp_sorted = sorted(tp_variants, key=lambda r: r["comp_full"] or -9999, reverse=True)
    for r in tp_sorted:
        full = r["comp_full"] or 0
        if r["label"] == "letrun":
            note = "Carry winners; max participation"
        elif r["label"] == "flush3d":
            note = "Baseline: forced flush"
        elif r["label"] == "sigflip":
            note = "Exit on signal reversal"
        elif r["label"].startswith("tp"):
            note = f"Bag profit at {r['label'][2:]}%"
        else:
            note = ""
        print(f"  {r['label']:<16} {full:>8.1f}  {r['comp_2021'] or 'n/a':>8}  {r['green_2021'] or 'n/a':>7}  {r['green_all']:>8.1f}  {note}")

    print()
    print("LESSON:")
    # Derive lesson from data
    flush = next(r for r in results if r["label"] == "flush3d")
    letrun = next(r for r in results if r["label"] == "letrun")
    tp10 = next(r for r in results if r["label"] == "tp10")

    atr_variants = [r for r in results if r["label"].startswith("atr")]
    best_atr = max(atr_variants, key=lambda r: r["comp_full"] or -9999)

    # Compare flush vs letrun
    if (letrun["comp_full"] or 0) > (flush["comp_full"] or 0):
        letrun_verdict = f"letting winners run (+{(letrun['comp_full'] or 0) - (flush['comp_full'] or 0):.0f}pp vs flush)"
    else:
        letrun_verdict = f"flush beats letrun (flush={flush['comp_full']:.0f}% > letrun={letrun['comp_full']:.0f}%)"

    if (tp10["comp_full"] or 0) > (flush["comp_full"] or 0):
        tp_verdict = f"early bag-profit (tp10) HELPS (+{(tp10['comp_full'] or 0) - (flush['comp_full'] or 0):.0f}pp)"
    else:
        tp_verdict = f"early bag-profit (tp10) HURTS ({tp10['comp_full']:.0f}% vs flush {flush['comp_full']:.0f}%)"

    atr_word = "help" if (best_atr["comp_full"] or 0) > (flush["comp_full"] or 0) else "hurt"
    print(f"  Given mom14 top3 gated entry (rebal 3d): {letrun_verdict}; {tp_verdict}; "
          f"best dynamic exit = {best['label']} ({best['comp_full']:.0f}% full, "
          f"green_all={best['green_all']:.0f}%). "
          f"ATR trails {atr_word} "
          f"(best ATR={best_atr['label']} {best_atr['comp_full']:.0f}%). "
          f"Bear (2022) best exit = {best_bear['label']} ({best_bear['comp_2022']}%).")

    return results


if __name__ == "__main__":
    main()
