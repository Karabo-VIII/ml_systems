"""experiments/adaptive_ma/expert/ablation_2x2_4h.py -- WHERE IS THE EDGE? 2x2 ablation @ 4h.

AUDITOR RED-team task: the ER-gated fixed-MA @ 4h (ATR-trail exit) was already REFUTED vs a regime-matched
null (0/77, see ER_GATE_4H_FALSIFIER_REPORT.md). This script ISOLATES which component the (non-)edge lives
in by ablating the ENTRY GATE against the EXIT POLICY with MINIMAL added DOF -- just on/off switches, no new
params. All structural constants are LOCKED to the expert rig (8/21 EMA, ER>0.4, ATR14 x3.0, 42-bar cap,
taker 0.0024) so the ONLY thing that changes between the four cells is two binary switches.

2x2 design (all else locked, same 4h chimera data):

                       EXIT = ATR-trail x3 + 42cap     EXIT = opposite-cross (fast<slow) + 42cap
   ENTRY ER-gate ON    A  (= the redirect strategy)    B
   ENTRY ER-gate OFF   C                               D

  ENTRY state  : fast>slow  [AND er>0.4 when gate ON]     (the only entry switch = the `& er>0.4` term)
  EXIT switch  : ATR-trail x3.0 (+42-bar time cap)  vs  opposite-cross fast<slow (+42-bar time cap)

ATTRIBUTION:
  * ENTRY-GATE effect  = A vs C (under ATR-trail) and B vs D (under opp-cross): does adding ER>0.4 help?
  * EXIT-POLICY effect = A vs B (under ER-on)     and C vs D (under ER-off):   does ATR-trail beat opp-cross?
  * NULL test (project firewall = cost-matched RANDOM-ENTRY + passive hold of the cell's OWN hold-dist):
      - ER-on cells (A,B): REGIME-MATCHED null (random entries drawn only from er>0.4 bars) -> isolates
        within-regime entry TIMING (matches the prior falsifier).
      - ER-off cells (C,D): PLAIN null (random entries from all bars).
  THE DECISIVE CLAIM under test (from the task): "if ER-gate-OFF + ATR-trail (cell C) ALREADY beats the
  null, the conditioning premise is wrong and credit belongs to the exit policy."

All past-only: MA/ER/cross at close-of-bar t, filled at open[t+1]; ATR read as atr[j-1]. No emoji (cp1252).

RWYB:  python experiments/adaptive_ma/expert/ablation_2x2_4h.py [--quick] [--probe BTCUSDT]
SAFETY: analysis + JSON only under experiments/. No commit/push/deploy/capital.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.chimera_loader import ChimeraLoader  # noqa: E402
from pipeline.universe_loader import UniverseLoader  # noqa: E402
from wealth_bot.harness import WindowSpec, ema_past_only  # noqa: E402
from strat.setup_harness import SetupHarness, ExitPolicy  # noqa: E402
from strat.firewall import random_entry_null  # noqa: E402

# reuse the EXACT locked primitives + constants from the expert rig (one source of truth)
from er_gate_4h import (  # noqa: E402
    kaufman_er, atr_past, load_4h,
    FAST, SLOW, ER_WIN, ER_THRESH, ATR_WIN, ATR_TRAIL_MULT, MAX_HOLD, TAKER, WIN,
)

WINDOWS = ["TRAIN", "VAL", "OOS", "UNSEEN"]
HELD = ["OOS", "UNSEEN"]

# the four cells: (entry_gate_on, exit_kind)
CELLS = {
    "A_ERon_ATRtrail":  (True,  "atr"),
    "B_ERon_oppcross":  (True,  "xcross"),
    "C_ERoff_ATRtrail": (False, "atr"),
    "D_ERoff_oppcross": (False, "xcross"),
}


def build_cols(df: pd.DataFrame) -> pd.DataFrame:
    """All columns needed by the 2x2. close-of-bar past-only (filled next-open by the harness)."""
    out = df.copy().reset_index(drop=True)
    close = out["close"].astype(float)
    out["er"] = kaufman_er(close, ER_WIN)
    f = ema_past_only(close, length=FAST, shift=0).to_numpy()
    s = ema_past_only(close, length=SLOW, shift=0).to_numpy()
    out["fast"] = f
    out["slow"] = s
    fast, slow = out["fast"], out["slow"]
    gate = out["er"] > ER_THRESH
    out["entry_on"] = ((fast > slow) & gate).fillna(False).astype(int)    # ER-gate ON state
    out["entry_off"] = (fast > slow).fillna(False).astype(int)            # ER-gate OFF state
    out["xdn"] = (fast < slow).fillna(False).astype(int)                  # opposite-cross exit signal (state)
    out["atr"] = atr_past(out, ATR_WIN)
    return out


def make_policy(exit_kind: str) -> ExitPolicy:
    if exit_kind == "atr":
        return ExitPolicy(atr_trail_mult=ATR_TRAIL_MULT, atr_col="atr", max_hold_bars=MAX_HOLD)
    elif exit_kind == "xcross":
        return ExitPolicy(exit_signal_col="xdn", max_hold_bars=MAX_HOLD)
    raise ValueError(exit_kind)


def run_cell(cols: pd.DataFrame, gate_on: bool, exit_kind: str, n_books: int, seed: int) -> dict | None:
    entry_col = "entry_on" if gate_on else "entry_off"
    if int(cols[entry_col].sum()) < 4:
        return None
    policy = make_policy(exit_kind)
    # regime_match_on_entry=False: we set the null mode explicitly below.
    h = SetupHarness(cols, entry_col, policy, WIN, cost_rt=TAKER, regime_match_on_entry=False)
    if gate_on:
        # REGIME-MATCHED null: random entries drawn only from er>0.4 bars (within-regime timing test).
        h.spec.filter_col = "er"
        h.spec.filter_op = "gt"
        h.spec.filter_val = ER_THRESH
        fw = random_entry_null(h, n_books=n_books, seed=seed, regime_matched=True)
    else:
        # PLAIN null: random entries from all bars (no regime to match).
        fw = random_entry_null(h, n_books=n_books, seed=seed, regime_matched=False)
    res = h.run()
    return {
        "n_entries": int(cols[entry_col].sum()),
        "regime_mode": fw["regime_mode"],
        "windows": {w: {"comp": round(res.window_stats[w].compound_pct, 2),
                        "n": res.window_stats[w].n_trades,
                        "wr": round(res.window_stats[w].win_rate, 3)} for w in WINDOWS},
        "firewall": {w: {"real": fw["per_window"][w]["real"],
                         "null_p50": fw["per_window"][w]["null_p50"],
                         "null_p95": fw["per_window"][w]["null_p95"],
                         "beats_null": fw["per_window"][w]["beats_null"],
                         "n": fw["per_window"][w]["n_trades"]} for w in WINDOWS},
        "beats_held": bool(fw["beats_held"]),
        "pos_held": bool(fw["pos_held"]),
    }


def run_asset(df: pd.DataFrame, n_books: int, seed: int) -> dict | None:
    cols = build_cols(df)
    rec = {}
    for name, (gate_on, exit_kind) in CELLS.items():
        try:
            rec[name] = run_cell(cols, gate_on, exit_kind, n_books, seed)
        except Exception as e:  # noqa: BLE001
            rec[name] = {"error": repr(e)[:160]}
    return rec


def _median(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.median(xs)), 2) if xs else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 2) if xs else None


def aggregate(per_asset: dict) -> dict:
    agg = {}
    for name in CELLS:
        recs = [r[name] for r in per_asset.values()
                if name in r and r[name] is not None and "error" not in r[name] and "beats_held" in r[name]]
        n_eval = len(recs)
        oos = [r["windows"]["OOS"]["comp"] for r in recs]
        uns = [r["windows"]["UNSEEN"]["comp"] for r in recs]
        oos_null = [r["firewall"]["OOS"]["null_p50"] for r in recs]
        uns_null = [r["firewall"]["UNSEEN"]["null_p50"] for r in recs]
        beat_and_pos = [r for r in recs if r["beats_held"] and r["pos_held"]]
        pos_held = [r for r in recs if r["pos_held"]]
        agg[name] = {
            "n_eval": n_eval,
            "n_beat_null_AND_pos_held": len(beat_and_pos),
            "n_pos_held": len(pos_held),
            "OOS_real_median": _median(oos), "OOS_real_mean": _mean(oos),
            "UNSEEN_real_median": _median(uns), "UNSEEN_real_mean": _mean(uns),
            "OOS_null_p50_median": _median(oos_null),
            "UNSEEN_null_p50_median": _median(uns_null),
        }
    return agg


def main(quick: bool, probe: str | None):
    loader = ChimeraLoader()
    if probe:
        df = load_4h(loader, probe)
        print(f"[probe {probe}] bars={0 if df is None else len(df)}")
        if df is None:
            return
        rec = run_asset(df, n_books=300, seed=7)
        print(json.dumps(rec, indent=2, default=str))
        return

    syms = UniverseLoader.load().list("u100")
    if quick:
        syms = syms[:25]
    print(f"[2x2 ablation 4h] u100 4h | assets={len(syms)} | taker={TAKER} | MA={FAST}/{SLOW}EMA | "
          f"ER>{ER_THRESH} | ATRx{ATR_TRAIL_MULT} | cap={MAX_HOLD}bar", flush=True)

    per_asset = {}
    n_books = 200
    for k, s in enumerate(syms, 1):
        df = load_4h(loader, s)
        if df is None or len(df) < 1000:
            continue
        rec = run_asset(df, n_books=n_books, seed=7)
        if rec is not None:
            per_asset[s] = rec
        if k % 10 == 0:
            print(f"[run] {k}/{len(syms)} processed, {len(per_asset)} evaluated", flush=True)

    agg = aggregate(per_asset)
    out = {"config": {"cadence": "4h", "fast": FAST, "slow": SLOW, "ma": "ema", "er_win": ER_WIN,
                      "er_thresh": ER_THRESH, "atr_win": ATR_WIN, "atr_trail_mult": ATR_TRAIL_MULT,
                      "max_hold": MAX_HOLD, "taker": TAKER, "cells": {k: {"gate_on": v[0], "exit": v[1]}
                                                                      for k, v in CELLS.items()},
                      "null": "ERon=regime_matched(er>0.4), ERoff=plain_all_bars", "windows": WIN.__dict__},
           "aggregate": agg, "per_asset": per_asset}

    print("\n" + "=" * 92)
    print("2x2 ABLATION @ 4h  --  ENTRY-gate (ER>0.4) x EXIT (ATR-trail vs opposite-cross)")
    print("=" * 92)
    hdr = f"{'cell':<20}{'n':>4}{'OOS med':>10}{'UNSEEN med':>12}{'OOS null':>10}{'UNS null':>10}{'beat&pos':>10}{'pos_held':>9}"
    print(hdr)
    print("-" * 92)
    for name in CELLS:
        a = agg[name]
        print(f"{name:<20}{a['n_eval']:>4}{str(a['OOS_real_median']):>10}{str(a['UNSEEN_real_median']):>12}"
              f"{str(a['OOS_null_p50_median']):>10}{str(a['UNSEEN_null_p50_median']):>10}"
              f"{a['n_beat_null_AND_pos_held']:>5}/{a['n_eval']:<4}{a['n_pos_held']:>5}/{a['n_eval']:<3}")
    print("=" * 92)
    # attribution deltas (median held-out compound)
    def med_held(name):
        o, u = agg[name]["OOS_real_median"], agg[name]["UNSEEN_real_median"]
        return None if (o is None or u is None) else round(o + u, 2)
    A, B, C, D = (med_held(n) for n in ["A_ERon_ATRtrail", "B_ERon_oppcross", "C_ERoff_ATRtrail", "D_ERoff_oppcross"])
    print(f"  median held-out (OOS+UNSEEN) compound: A={A}  B={B}  C={C}  D={D}")
    if None not in (A, B, C, D):
        print(f"  ENTRY-GATE effect  (ER on - off): ATR-trail A-C={round(A-C,2)}   opp-cross B-D={round(B-D,2)}")
        print(f"  EXIT-POLICY effect (ATR - oppcross): ER-on A-B={round(A-B,2)}   ER-off C-D={round(C-D,2)}")
    print(f"  DECISIVE: cell C (ER-OFF + ATR-trail) beats-null&pos on "
          f"{agg['C_ERoff_ATRtrail']['n_beat_null_AND_pos_held']}/{agg['C_ERoff_ATRtrail']['n_eval']} assets")
    print("=" * 92)

    outpath = Path(__file__).resolve().parent / ("ablation_2x2_4h_quick.json" if quick else "ablation_2x2_4h_u100.json")
    outpath.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"[saved] {outpath}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="first 25 u100 assets only")
    ap.add_argument("--probe", type=str, default=None, help="single-asset probe (e.g. BTCUSDT)")
    args = ap.parse_args()
    main(quick=args.quick, probe=args.probe)
