"""
strat/momentum_rotation_flex.py -- Momentum Rotation x Flexible Holding Lab

LANE: MOMENTUM ROTATION x FLEXIBLE HOLDING
Signal: mom14 (+ mom7/mom30 variants)
Variants:
  (a) Flush-and-rebalance every {3,7,14}d top-K
  (b) Carry-winners-cut-losers: keep held names still in top-K+gated, replace drop-outs only
  (c) Bag-profit lock: trim/exit a name after it gains >+X% then redeploy
  (d) K in {1, 3, 5}

RWYB: python -m strat.momentum_rotation_flex
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.mover_lab as ml


# ---------------------------------------------------------------------------
# Strategy builders
# ---------------------------------------------------------------------------

def flush_rebal(score, ind, K, rebal):
    """(a) Plain flush-and-rebalance every `rebal` days, top-K by score, gated."""
    return ml.topk_weight(score, ind, K=K, gate=True, rebal=rebal)


def carry_winners_cut_losers(score, ind, K, rebal):
    """
    (b) Carry-winners-cut-losers:
        At each rebal checkpoint, keep held names that are STILL in top-K AND gated.
        Replace drop-outs with the highest-scoring eligible non-held names.
        EW over current holdings (rescale at each rebal).
    """
    C = ind["C"]
    g = ind["gate"]
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    held = set()
    last = -999

    for i, d in enumerate(C.index):
        if i - last >= rebal:
            elig = [s for s in C.columns if bool(g.loc[d, s]) and pd.notna(score.loc[d, s])]
            if elig:
                ranked = sorted(elig, key=lambda s: -score.loc[d, s])
                top_k_set = set(ranked[:K])

                # Keep winners: held names still in top-K and still gated
                survivors = held & top_k_set
                # How many slots to fill?
                n_fill = K - len(survivors)
                # Fill from top-K ranked, excluding survivors already counted
                new_picks = [s for s in ranked[:K] if s not in survivors][:n_fill]
                held = survivors | set(new_picks)

                W.loc[d, :] = 0.0
                for s in held:
                    W.loc[d, s] = 1.0 / len(held)
            else:
                held = set()
                W.loc[d, :] = 0.0
            last = i
        elif i > 0:
            W.iloc[i] = W.iloc[i - 1]

    return W


def bag_profit_lock(score, ind, K, rebal, profit_thresh):
    """
    (c) Bag-profit lock:
        At each bar, track entry price for each held name.
        If a name gained > profit_thresh% from entry, exit it and mark as 'cashed'.
        At next rebal checkpoint, replace cashed + gated-out names with fresh top-K picks.
        EW over active holdings; cashed slot goes to cash until next rebal.
    """
    C = ind["C"]
    g = ind["gate"]
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)

    held = {}      # sym -> entry_price
    cashed = set() # names that hit profit target (locked); freed at next rebal
    last = -999

    for i, d in enumerate(C.index):
        if i - last >= rebal:
            # Free cashed names
            cashed = set()

            elig = [s for s in C.columns if bool(g.loc[d, s]) and pd.notna(score.loc[d, s])]
            if elig:
                ranked = sorted(elig, key=lambda s: -score.loc[d, s])
                # Keep survivors among previously held that are still in top-K and gated
                survivors = {s: held[s] for s in held if s in set(ranked[:K]) and bool(g.loc[d, s])}
                n_fill = K - len(survivors)
                new_syms = [s for s in ranked[:K] if s not in survivors][:n_fill]
                held = {**survivors, **{s: float(C.loc[d, s]) for s in new_syms}}
            else:
                held = {}

            W.loc[d, :] = 0.0
            if held:
                w = 1.0 / len(held)
                for s in held:
                    W.loc[d, s] = w
            last = i

        elif i > 0:
            W.iloc[i] = W.iloc[i - 1].copy()

            # Check bag-profit condition for each held name
            exited_any = False
            to_exit = []
            for s in list(held.keys()):
                entry = held[s]
                cur = float(C.loc[d, s])
                gain = (cur - entry) / entry if entry > 0 else 0.0
                if gain >= profit_thresh:
                    to_exit.append(s)

            if to_exit:
                for s in to_exit:
                    del held[s]
                    cashed.add(s)
                    W.loc[d, s] = 0.0
                exited_any = True

            # Rescale remaining
            if held and exited_any:
                # Redeploy evenly among remaining (not refilling until next rebal)
                w = 1.0 / len(held)
                W.loc[d, :] = 0.0
                for s in held:
                    W.loc[d, s] = w

    return W


def momentum_signal(ind, kind):
    """Return the momentum signal DataFrame for the given kind."""
    return ind[kind]


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run():
    print("Loading data...")
    ind = ml.load()

    results = []

    # Reference: gated-beta (baseline)
    gate = ind["gate"].astype(float)
    gate_norm = gate.div(gate.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    m = ml.evaluate(gate_norm, ind, label="gated-beta[REF]")
    results.append(m)

    # -----------------------------------------------------------------------
    # (a) Flush-and-rebalance: mom14 / mom7 / mom30 x K in {1,3,5} x rebal in {3,7,14}
    # -----------------------------------------------------------------------
    for sig_name in ["mom14", "mom7", "mom30"]:
        score = momentum_signal(ind, sig_name)
        for K in [1, 3, 5]:
            for rebal in [3, 7, 14]:
                label = f"flush|{sig_name}|K{K}|r{rebal}"
                W = flush_rebal(score, ind, K, rebal)
                m = ml.evaluate(W, ind, label=label)
                results.append(m)
                print(f"  {label}: 2021={m['comp_2021']}% full={m['comp_full']}% DD={m['maxDD']}% green21={m['green_2021']}%")

    # -----------------------------------------------------------------------
    # (b) Carry-winners-cut-losers: mom14 / mom7 x K in {1,3,5} x rebal in {3,7,14}
    # -----------------------------------------------------------------------
    for sig_name in ["mom14", "mom7"]:
        score = momentum_signal(ind, sig_name)
        for K in [1, 3, 5]:
            for rebal in [3, 7, 14]:
                label = f"carry|{sig_name}|K{K}|r{rebal}"
                W = carry_winners_cut_losers(score, ind, K, rebal)
                m = ml.evaluate(W, ind, label=label)
                results.append(m)
                print(f"  {label}: 2021={m['comp_2021']}% full={m['comp_full']}% DD={m['maxDD']}% green21={m['green_2021']}%")

    # -----------------------------------------------------------------------
    # (c) Bag-profit lock: mom14 x K in {1,3,5} x profit_thresh in {0.10,0.20,0.30} x rebal in {3,7}
    # -----------------------------------------------------------------------
    score_m14 = momentum_signal(ind, "mom14")
    for K in [1, 3, 5]:
        for profit_thresh in [0.10, 0.20, 0.30]:
            for rebal in [3, 7]:
                label = f"baglock|mom14|K{K}|p{int(profit_thresh*100)}|r{rebal}"
                W = bag_profit_lock(score_m14, ind, K, rebal, profit_thresh)
                m = ml.evaluate(W, ind, label=label)
                results.append(m)
                print(f"  {label}: 2021={m['comp_2021']}% full={m['comp_full']}% DD={m['maxDD']}% green21={m['green_2021']}%")

    return results


def print_table(results):
    cols = ["label", "comp_2020", "comp_2021", "comp_2022", "comp_full", "maxDD", "green_2021", "green_all", "avg_expo"]
    # Header
    hdr = f"{'Config':<38} {'2020':>7} {'2021':>7} {'2022':>7} {'Full':>8} {'MaxDD':>7} {'Gr21':>6} {'GrAll':>6} {'Expo':>5}"
    print("\n" + "="*95)
    print(hdr)
    print("-"*95)
    for r in results:
        label = r["label"]
        row = (
            f"{label:<38}"
            f" {str(r.get('comp_2020','N/A')):>7}"
            f" {str(r.get('comp_2021','N/A')):>7}"
            f" {str(r.get('comp_2022','N/A')):>7}"
            f" {str(r.get('comp_full','N/A')):>8}"
            f" {str(r.get('maxDD','N/A')):>7}"
            f" {str(r.get('green_2021','N/A')):>6}"
            f" {str(r.get('green_all','N/A')):>6}"
            f" {str(r.get('avg_expo','N/A')):>5}"
        )
        print(row)
    print("="*95)


def analyse(results):
    """Print best/greedy highlights and lessons."""
    # Filter out reference
    strats = [r for r in results if r["label"] != "gated-beta[REF]"]
    ref = next(r for r in results if r["label"] == "gated-beta[REF]")

    # Best by comp_full
    best_full = max(strats, key=lambda r: r["comp_full"] or -9999)
    # Best by comp_2021
    best_2021 = max(strats, key=lambda r: r["comp_2021"] or -9999)
    # Greedy: top-1 (K=1)
    greedy = [r for r in strats if "|K1|" in r["label"]]
    best_greedy = max(greedy, key=lambda r: r["comp_full"] or -9999) if greedy else None
    # Best 2022 preservation
    best_2022 = max(strats, key=lambda r: r["comp_2022"] or -9999)
    # Best green_2021
    best_green21 = max(strats, key=lambda r: r["green_2021"] or -9999)

    # Flush vs carry comparison (mom14, K3, r7)
    flush_ref = next((r for r in strats if r["label"] == "flush|mom14|K3|r7"), None)
    carry_ref = next((r for r in strats if r["label"] == "carry|mom14|K3|r7"), None)

    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    print(f"\nREFERENCE (gated-beta): 2021={ref['comp_2021']}% full={ref['comp_full']}% DD={ref['maxDD']}% green21={ref['green_2021']}%")
    print(f"\nBEST by comp_full:  {best_full['label']}")
    print(f"  2020={best_full['comp_2020']}%  2021={best_full['comp_2021']}%  2022={best_full['comp_2022']}%  full={best_full['comp_full']}%  DD={best_full['maxDD']}%  green21={best_full['green_2021']}%")
    print(f"\nBEST by comp_2021:  {best_2021['label']}")
    print(f"  2020={best_2021['comp_2020']}%  2021={best_2021['comp_2021']}%  2022={best_2021['comp_2022']}%  full={best_2021['comp_full']}%  DD={best_2021['maxDD']}%")
    print(f"\nBEST 2022 (preservation): {best_2022['label']}")
    print(f"  2022={best_2022['comp_2022']}%  full={best_2022['comp_full']}%  DD={best_2022['maxDD']}%")
    print(f"\nBEST green_2021:  {best_green21['label']}")
    print(f"  green21={best_green21['green_2021']}%  green_all={best_green21['green_all']}%  2021={best_green21['comp_2021']}%")
    if best_greedy:
        print(f"\nGREEDIEST (K=1, best full): {best_greedy['label']}")
        print(f"  2020={best_greedy['comp_2020']}%  2021={best_greedy['comp_2021']}%  2022={best_greedy['comp_2022']}%  full={best_greedy['comp_full']}%  DD={best_greedy['maxDD']}%")
    if flush_ref and carry_ref:
        print(f"\nFlush vs Carry (mom14,K3,r7):")
        print(f"  flush: 2021={flush_ref['comp_2021']}%  full={flush_ref['comp_full']}%  DD={flush_ref['maxDD']}%  green21={flush_ref['green_2021']}%")
        print(f"  carry: 2021={carry_ref['comp_2021']}%  full={carry_ref['comp_full']}%  DD={carry_ref['maxDD']}%  green21={carry_ref['green_2021']}%")

    # Where each policy type wins and loses
    flush_strats = [r for r in strats if r["label"].startswith("flush")]
    carry_strats = [r for r in strats if r["label"].startswith("carry")]
    baglock_strats = [r for r in strats if r["label"].startswith("baglock")]

    def avg_metric(lst, key):
        vals = [r[key] for r in lst if r[key] is not None]
        return round(np.mean(vals), 1) if vals else None

    print("\n--- Policy averages ---")
    for name, lst in [("flush", flush_strats), ("carry", carry_strats), ("baglock", baglock_strats)]:
        print(f"  {name:<10} avg_full={avg_metric(lst,'comp_full'):>8}%  avg_2021={avg_metric(lst,'comp_2021'):>8}%  avg_2022={avg_metric(lst,'comp_2022'):>7}%  avg_DD={avg_metric(lst,'maxDD'):>7}%  avg_green21={avg_metric(lst,'green_2021'):>5}%")

    print("\n--- K breakdown (flush|mom14 only) ---")
    for K in [1, 3, 5]:
        sub = [r for r in flush_strats if f"|mom14|K{K}|" in r["label"]]
        print(f"  K={K}: avg_full={avg_metric(sub,'comp_full'):>8}%  avg_2021={avg_metric(sub,'comp_2021'):>8}%  avg_2022={avg_metric(sub,'comp_2022'):>7}%  avg_DD={avg_metric(sub,'maxDD'):>7}%")

    print("\n--- Rebal breakdown (flush|mom14|K3 only) ---")
    for rebal in [3, 7, 14]:
        sub = [r for r in flush_strats if f"|mom14|K3|r{rebal}" in r["label"]]
        print(f"  r={rebal}: avg_full={avg_metric(sub,'comp_full'):>8}%  avg_2021={avg_metric(sub,'comp_2021'):>8}%  avg_2022={avg_metric(sub,'comp_2022'):>7}%  avg_DD={avg_metric(sub,'maxDD'):>7}%")

    print("\n" + "="*60)


if __name__ == "__main__":
    results = run()
    print_table(results)
    analyse(results)
