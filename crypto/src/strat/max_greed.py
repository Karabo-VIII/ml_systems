"""src/strat/max_greed.py -- MAX GREED style sweep.

Variants:
  - TOP-1  : single best mom14 name (maximum concentration, gated, rebal=3)
  - TOP-3  : top-3 EW mom14 (baseline reference)
  - TOP-5  : top-5 EW mom14
  - EW-ALL : equal-weight all gated assets (naive gated-beta)
  - CONV-3 : conviction-scaled weights (softmax over mom14, top-3 gated, rebal=3)
  - CONV-1 : conviction-scaled single winner (full weight to highest mom14 gated)
  - PYRAMID: add weight to assets that keep rising; trim those that stall (momentum carry)
  - PYRAMID-STRICT: pyramid with sma50 trend filter (stricter gate)

Objective: compound return (bull) + survival (2022 bear). Find 'max greed that survives'.
RWYB: python -m strat.max_greed
No emoji.
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


# ---- helpers ----

def _topk_ew(score, ind, K, gate=True, rebal=3):
    """Standard EW top-K by score, gated, rebalance every rebal days."""
    return lab.topk_weight(score, ind, K, gate=gate, rebal=rebal)


def _conv_scaled(score, ind, K, gate=True, rebal=3, temperature=1.0):
    """
    Conviction-scaled: top-K by score, softmax weights proportional to score.
    temperature < 1 -> more concentrated; temperature > 1 -> more uniform.
    """
    C = ind["C"]
    g = ind["gate"] if gate else pd.DataFrame(True, index=C.index, columns=C.columns)
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    last = -999
    for i, d in enumerate(C.index):
        if i - last >= rebal:
            elig = [s for s in C.columns if bool(g.loc[d, s]) and pd.notna(score.loc[d, s])]
            if elig:
                pick = sorted(elig, key=lambda s: -score.loc[d, s])[:K]
                scores = np.array([float(score.loc[d, s]) for s in pick])
                # shift to positive, then softmax
                scores = scores - scores.min() + 1e-6
                scores = scores ** (1.0 / temperature)
                scores = scores / scores.sum()
                W.loc[d, :] = 0.0
                for s, w in zip(pick, scores):
                    W.loc[d, s] = float(w)
            last = i
        elif i > 0:
            W.iloc[i] = W.iloc[i - 1]
    return W


def _pyramid(ind, gate=True, base_K=5, add_thresh=0.02, trim_thresh=-0.01,
             max_weight=0.50, rebal=3):
    """
    Pyramid: start EW in top-K gated names. Each rebal:
      - if asset rose > add_thresh since last entry -> increase weight (add 1/K increment)
      - if asset fell < trim_thresh since last entry -> reduce weight (cut by half)
      - normalize to row-sum=1 if >0 positions, else go to cash.
    add_thresh / trim_thresh are % return since last rebal.
    """
    C = ind["C"]
    g = ind["gate"] if gate else pd.DataFrame(True, index=C.index, columns=C.columns)
    mom = ind["mom14"]
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    last_idx = -999
    last_prices = None  # prices at last rebal

    for i, d in enumerate(C.index):
        if i - last_idx >= rebal:
            cur_prices = C.loc[d]
            elig = [s for s in C.columns if bool(g.loc[d, s]) and pd.notna(mom.loc[d, s])]

            if i == 0 or last_prices is None:
                # cold start: EW top-K
                if elig:
                    pick = sorted(elig, key=lambda s: -mom.loc[d, s])[:base_K]
                    W.loc[d, :] = 0.0
                    for s in pick:
                        W.loc[d, s] = 1.0 / len(pick)
            else:
                prev_w = W.iloc[i - 1].copy()
                new_w = prev_w.copy()

                # adjust weights based on performance since last rebal
                for s in C.columns:
                    if s not in elig:
                        new_w[s] = 0.0
                        continue
                    if pd.isna(last_prices.get(s)) or last_prices[s] <= 0:
                        continue
                    ret_since = float(cur_prices[s]) / float(last_prices[s]) - 1
                    unit = 1.0 / base_K
                    if ret_since > add_thresh:
                        new_w[s] = min(new_w[s] + unit, max_weight)
                    elif ret_since < trim_thresh:
                        new_w[s] = max(new_w[s] * 0.5, 0.0)

                # add new top-K names that aren't held yet
                top_names = sorted(elig, key=lambda s: -mom.loc[d, s])[:base_K]
                for s in top_names:
                    if new_w[s] == 0.0:
                        new_w[s] = 1.0 / base_K

                # remove non-gated
                for s in C.columns:
                    if s not in elig:
                        new_w[s] = 0.0

                total = new_w.sum()
                if total > 0:
                    # cap individual weight
                    new_w = new_w.clip(upper=max_weight)
                    total = new_w.sum()
                    new_w = new_w / total  # normalize to 1.0
                W.loc[d, :] = new_w.values

            last_prices = cur_prices.to_dict()
            last_idx = i
        elif i > 0:
            W.iloc[i] = W.iloc[i - 1]
    return W


def _pyramid_strict(ind, base_K=5, add_thresh=0.02, trim_thresh=-0.01, max_weight=0.50, rebal=3):
    """Pyramid with sma50 + sma200 dual gate (stricter)."""
    C = ind["C"]
    sma50 = ind["sma50"]
    sma200 = ind["sma200"]
    # override gate: must be above BOTH sma50 and sma200
    strict_gate = ((C > sma50) & (C > sma200)).fillna(False)

    # build a modified ind with the strict gate
    ind2 = dict(ind)
    ind2["gate"] = strict_gate
    return _pyramid(ind2, gate=True, base_K=base_K, add_thresh=add_thresh,
                    trim_thresh=trim_thresh, max_weight=max_weight, rebal=rebal)


def _ew_gated(ind):
    """Equal-weight all gated assets each day (naive gated-beta)."""
    C = ind["C"]
    g = ind["gate"].astype(float)
    n = g.sum(axis=1).replace(0, np.nan)
    W = g.div(n, axis=0).fillna(0.0)
    return W


def main():
    print("Loading data...")
    ind = lab.load()

    score = ind["mom14"]

    # --- build all weight matrices ---
    print("Building strategies...")

    W_top1 = _topk_ew(score, ind, K=1, gate=True, rebal=3)
    W_top3 = _topk_ew(score, ind, K=3, gate=True, rebal=3)
    W_top5 = _topk_ew(score, ind, K=5, gate=True, rebal=3)
    W_ew   = _ew_gated(ind)

    # conviction-scaled (temperature=0.5 = more concentrated than EW)
    W_conv3_hot  = _conv_scaled(score, ind, K=3, gate=True, rebal=3, temperature=0.5)
    W_conv3_warm = _conv_scaled(score, ind, K=3, gate=True, rebal=3, temperature=1.5)
    W_conv1      = _conv_scaled(score, ind, K=1, gate=True, rebal=3, temperature=0.5)

    # conviction-scaled with different rebal frequencies
    W_top1_r1  = _topk_ew(score, ind, K=1, gate=True, rebal=1)   # daily rebal top-1 (max churn greed)
    W_top1_r7  = _topk_ew(score, ind, K=1, gate=True, rebal=7)   # weekly hold top-1

    # pyramid
    W_pyr    = _pyramid(ind, gate=True, base_K=5, add_thresh=0.02, trim_thresh=-0.01, max_weight=0.50, rebal=3)
    W_pyr_s  = _pyramid_strict(ind, base_K=5, add_thresh=0.02, trim_thresh=-0.01, max_weight=0.50, rebal=3)

    # pyramid tighter: lower add threshold, stronger trim
    W_pyr2   = _pyramid(ind, gate=True, base_K=3, add_thresh=0.03, trim_thresh=-0.015, max_weight=0.60, rebal=3)

    strategies = [
        ("EW-gated (beta ref)",     W_ew),
        ("TOP-5 mom14 r3",          W_top5),
        ("TOP-3 mom14 r3",          W_top3),
        ("TOP-1 mom14 r3",          W_top1),
        ("TOP-1 mom14 r1 (daily)",  W_top1_r1),
        ("TOP-1 mom14 r7 (weekly)", W_top1_r7),
        ("CONV-3 temp=0.5 (hot)",   W_conv3_hot),
        ("CONV-3 temp=1.5 (warm)",  W_conv3_warm),
        ("CONV-1 (full-conviction)",W_conv1),
        ("PYRAMID K5 r3",           W_pyr),
        ("PYRAMID K5 strict-gate",  W_pyr_s),
        ("PYRAMID K3 tighter r3",   W_pyr2),
    ]

    print("\nEvaluating strategies...")
    results = []
    for label, W in strategies:
        m = lab.evaluate(W, ind, H=3, label=label)
        results.append(m)

    # --- print results table ---
    header = f"{'Config':<30} {'2020':>7} {'2021':>7} {'2022':>7} {'Full':>8} {'maxDD':>7} {'Grn21':>6} {'GrnAll':>6} {'Expo':>5} {'Turn':>6}"
    print()
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for m in results:
        c20  = f"{m['comp_2020']:.0f}%" if m['comp_2020'] is not None else "  n/a"
        c21  = f"{m['comp_2021']:.0f}%" if m['comp_2021'] is not None else "  n/a"
        c22  = f"{m['comp_2022']:.0f}%" if m['comp_2022'] is not None else "  n/a"
        cful = f"{m['comp_full']:.0f}%"
        mdd  = f"{m['maxDD']:.0f}%"
        g21  = f"{m['green_2021']:.0f}%" if m['green_2021'] is not None else " n/a"
        gall = f"{m['green_all']:.0f}%"
        expo = f"{m['avg_expo']:.2f}"
        turn = f"{m['avg_turnover']:.3f}"
        print(f"{m['label']:<30} {c20:>7} {c21:>7} {c22:>7} {cful:>8} {mdd:>7} {g21:>6} {gall:>6} {expo:>5} {turn:>6}")

    print("=" * len(header))

    # --- analysis ---
    # Sort by comp_2021 descending (bull greedy)
    sorted_by_bull = sorted(results, key=lambda m: m['comp_2021'] or -999, reverse=True)
    best_bull = sorted_by_bull[0]

    # Sort by comp_full descending
    sorted_by_full = sorted(results, key=lambda m: m['comp_full'], reverse=True)
    best_full = sorted_by_full[0]

    # 'survives 2022' = comp_2022 > -60% (less than catastrophic) and maxDD > -90
    survivors = [m for m in results if (m['comp_2022'] is not None and m['comp_2022'] > -60.0) and m['maxDD'] > -90.0]
    if survivors:
        max_greed_survivor = max(survivors, key=lambda m: m['comp_2021'] or -999)
    else:
        max_greed_survivor = None

    # Greediest = smallest avg_expo or top-1 (highest concentration)
    # Actually greediest = lowest avg_expo (all-in on fewest names)
    greediest = min(results, key=lambda m: m['avg_expo'])

    print()
    print("ANALYSIS")
    print("-" * 60)
    print(f"Best 2021 (bull):      {best_bull['label']} -> {best_bull['comp_2021']:.0f}% 2021, {best_bull['maxDD']:.0f}% DD")
    print(f"Best full-cycle:       {best_full['label']} -> {best_full['comp_full']:.0f}% full, maxDD {best_full['maxDD']:.0f}%")
    print(f"Greediest (min expo):  {greediest['label']} -> expo={greediest['avg_expo']:.2f}, 2021={greediest['comp_2021']:.0f}%")
    if max_greed_survivor:
        print(f"Max-greed-that-survives: {max_greed_survivor['label']}")
        print(f"  2020={max_greed_survivor['comp_2020']:.0f}% 2021={max_greed_survivor['comp_2021']:.0f}% 2022={max_greed_survivor['comp_2022']:.0f}% full={max_greed_survivor['comp_full']:.0f}% maxDD={max_greed_survivor['maxDD']:.0f}%")
    else:
        print("Max-greed-that-survives: NONE pass the 2022 survival threshold at these settings")

    # Concentration vs compound frontier
    print()
    print("CONCENTRATION vs COMPOUND FRONTIER (sorted by 2021 return):")
    print(f"  {'Config':<30} {'2021':>7} {'2022':>7} {'maxDD':>7} {'Expo':>5}")
    for m in sorted_by_bull[:8]:
        c21 = f"{m['comp_2021']:.0f}%" if m['comp_2021'] is not None else "  n/a"
        c22 = f"{m['comp_2022']:.0f}%" if m['comp_2022'] is not None else "  n/a"
        mdd = f"{m['maxDD']:.0f}%"
        print(f"  {m['label']:<30} {c21:>7} {c22:>7} {mdd:>7} {m['avg_expo']:>5.2f}")


if __name__ == "__main__":
    main()
