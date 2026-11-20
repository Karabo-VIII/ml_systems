"""engine_contrarian_ablation.py -- per-signal ablation + cash-only baseline.

Runs 4 single-signal variants + cash-only + composite to diagnose which components help/hurt.
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
from strat.engine_contrarian import (
    _build_features, _rank_norm, build_weights,
    _slice_return, _buyhold_slice, W_REVERSAL, W_RANGE_POS, W_VOL_BRK, W_RSI
)

COST = lab.COST


def build_single_signal_W(ind, sig_name, K=3, rebal_days=7):
    rev, rng_s, vbrk, rsi_s = _build_features(ind)
    sigs = {
        "reversal":   rev,
        "range_pos":  rng_s,
        "vol_brk":    vbrk,
        "rsi_band":   rsi_s,
    }
    score = _rank_norm(sigs[sig_name])
    return lab.topk_weight(score, ind, K=K, gate=True, rebal=rebal_days)


def cash_W(ind):
    C = ind["C"]
    return pd.DataFrame(0.0, index=C.index, columns=C.columns)


def gated_beta_W(ind):
    C = ind["C"]
    gate = ind["gate"]
    return gate.astype(float).div(gate.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def run_ablation(n_slices=300, K=3, rebal_days=7, seed=42):
    rng = np.random.default_rng(seed)
    ind = lab.load(start="2020-01-01", end="2026-06-01")
    C   = ind["C"]
    R   = ind["R"]
    gate = ind["gate"]

    # build all weight matrices
    Ws = {
        "EW_BuyHold":       None,       # special: computed per-slice
        "Cash":             cash_W(ind),
        "GatedBeta":        gated_beta_W(ind),
        "Reversal-only":    build_single_signal_W(ind, "reversal",   K, rebal_days),
        "RangePos-only":    build_single_signal_W(ind, "range_pos",  K, rebal_days),
        "VolBreakout-only": build_single_signal_W(ind, "vol_brk",    K, rebal_days),
        "RSIBand-only":     build_single_signal_W(ind, "rsi_band",   K, rebal_days),
        "Composite-4sig":   build_weights(ind, K, rebal_days),
    }

    test_start_date = pd.Timestamp("2022-01-01")
    test_end_date   = pd.Timestamp("2026-05-01")
    idx = C.index
    valid_starts = np.where((idx >= test_start_date) & (idx < test_end_date - pd.Timedelta(days=7)))[0]
    sampled = rng.choice(valid_starts, size=min(n_slices, len(valid_starts)), replace=False)
    sampled = np.sort(sampled)

    bh_rets = np.array([_buyhold_slice(R, gate, si, H=7) for si in sampled])

    results = []
    for name, W in Ws.items():
        if name == "EW_BuyHold":
            rets = bh_rets
        else:
            rets = np.array([_slice_return(W, R, si, H=7) for si in sampled])

        down_mask = bh_rets < 0
        up_mask   = bh_rets >= 0
        results.append({
            "engine": name,
            "pos_rate_%": round(100 * np.mean(rets > 0), 1),
            "mean_%": round(100 * np.mean(rets), 2),
            "p05_%": round(100 * np.percentile(rets, 5), 2),
            "pr_down_%": round(100 * np.mean(rets[down_mask] > 0), 1) if down_mask.sum() else "n/a",
            "pr_up_%": round(100 * np.mean(rets[up_mask] > 0), 1) if up_mask.sum() else "n/a",
        })

    return results


def main():
    print("=" * 70)
    print("ABLATION -- Per-signal & baseline comparison (300 random 7d slices)")
    print("=" * 70)
    rows = run_ablation()
    cols = ["engine", "pos_rate_%", "mean_%", "p05_%", "pr_down_%", "pr_up_%"]
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        print("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    print("=" * 70)


if __name__ == "__main__":
    raise SystemExit(main())
