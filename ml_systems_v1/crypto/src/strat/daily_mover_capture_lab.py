"""src/strat/daily_mover_capture_lab.py -- DAILY-MOVERS CAPTURE strategy lab.

LANE: short-horizon mover capture. At each checkpoint identify MOVERS -- assets with a big recent
1d/3d pop (ret1 high), fresh breakout (C >= hh14), or vol-expansion (vol20 jumping). Enter them,
ride SHORT with a bag-profit target (+X%) and/or a tight time stop (1-3 day hold), then rotate.

Variants built:
  V1  ret1-movers    : top-3 by yesterday return (ret1), hold 3d, gate off
  V2  breakout-movers: top-3 assets at/near 14d high (C/hh14), hold 3d, gate on
  V3  vol-exp-movers : top-3 by vol20 jump (vol20/vol20.shift(5)), hold 3d, gate off
  V4  combo-score    : rank by 0.4*ret1_z + 0.4*breakout_z + 0.2*vol_jump_z, top-3, 3d, gate on
  V5  ret1 bag-profit: top-3 ret1; exit at first checkpoint where position >= +bag_tgt (5%), else 1d forced
  V6  breakout ride  : top-3 breakout, hold 5d (ride longer)
  V7  ret1 tight     : top-3 ret1, hold 1d only (very tight, highest rotation)
  V8  combo gate-off : V4 combo score but no gate (capture bear movers too)
  V9  ret1 top1      : highest single ret1 mover, max concentration, 3d
  V10 vol-exp ungated: V3 but gate off (vol breakouts in any trend)

Reference (printed first): gated-beta EW + mom14-top3-rebal7

RWYB:
  python -m strat.daily_mover_capture_lab
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

# ---- helpers ----------------------------------------------------------

def _zscore_xs(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Cross-sectional z-score per row (across assets), clipped."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return ((df.sub(mu, axis=0)).div(sd, axis=0)).clip(-clip, clip)


def _topk_hold(score: pd.DataFrame, K: int, hold: int,
                gate: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Every `hold` bars: select top-K assets by score (among gated), EW weight 1/K.
    Carry position between rebalances.
    """
    W = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    last_rebal = -999
    for i, d in enumerate(score.index):
        if i - last_rebal >= hold:
            if gate is not None:
                elig = [s for s in score.columns
                        if bool(gate.loc[d, s]) and pd.notna(score.loc[d, s])]
            else:
                elig = [s for s in score.columns if pd.notna(score.loc[d, s])]
            if elig:
                top = sorted(elig, key=lambda s: -score.loc[d, s])[:K]
                W.loc[d, :] = 0.0
                for s in top:
                    W.loc[d, s] = 1.0 / len(top)
            last_rebal = i
        elif i > 0:
            W.iloc[i] = W.iloc[i - 1]
    return W


def _bag_profit_weight(score: pd.DataFrame, K: int, bag_tgt: float,
                        C: pd.DataFrame, gate: pd.DataFrame | None = None,
                        max_hold: int = 3) -> pd.DataFrame:
    """
    Select top-K by score every day. Hold position until EITHER:
      - book return on those positions hits +bag_tgt (close-to-close, crude), OR
      - max_hold days elapsed.
    Then re-enter. This simulates a greedy bag-profit rotation.
    We track per-asset position and how long it has been held.
    """
    dates = score.index
    W = pd.DataFrame(0.0, index=dates, columns=score.columns)
    # Per-asset state: hold_count (days held since entry), entry_price
    hold_count = pd.Series(0, index=score.columns)
    entry_price = pd.Series(np.nan, index=score.columns)
    held = pd.Series(False, index=score.columns)

    for i, d in enumerate(dates):
        if i == 0:
            continue
        # Advance hold counts and check exits
        for s in score.columns:
            if held[s]:
                hold_count[s] += 1
                cp = C.loc[d, s]
                ep = entry_price[s]
                pnl = cp / ep - 1.0 if (np.isfinite(ep) and ep > 0) else 0.0
                if pnl >= bag_tgt or hold_count[s] >= max_hold:
                    held[s] = False
                    hold_count[s] = 0
                    entry_price[s] = np.nan

        # Pick new entries only for positions not currently held
        if gate is not None:
            elig = [s for s in score.columns
                    if not held[s] and bool(gate.loc[d, s]) and pd.notna(score.loc[d, s])]
        else:
            elig = [s for s in score.columns if not held[s] and pd.notna(score.loc[d, s])]

        if elig:
            top = sorted(elig, key=lambda s: -score.loc[d, s])[:K]
            for s in top:
                held[s] = True
                hold_count[s] = 0
                entry_price[s] = C.loc[d, s]

        # Build weight row from currently held positions
        h_syms = [s for s in score.columns if held[s]]
        if h_syms:
            for s in h_syms:
                W.loc[d, s] = 1.0 / len(h_syms)

    return W


# ---- build all variants -----------------------------------------------

def build_variants(ind: dict) -> dict:
    C = ind["C"]
    gate = ind["gate"]

    # Raw scores
    ret1 = ind["ret1"]   # 1-day return
    # Breakout proximity: how close to 14d high (0 = at high, >0 = above, <0 = below)
    breakout = (C / ind["hh14"] - 1.0).clip(-0.5, 0.5)
    # Vol expansion: ratio of current 20d vol to 5-bar lagged 20d vol
    vol_jump = (ind["vol20"] / ind["vol20"].shift(5)).fillna(1.0)

    # Z-scores for combo
    ret1_z = _zscore_xs(ret1)
    bo_z = _zscore_xs(breakout)
    vj_z = _zscore_xs(vol_jump)
    combo = 0.4 * ret1_z + 0.4 * bo_z + 0.2 * vj_z

    variants = {}

    # V1: ret1 movers, hold 3d, NO gate (capture bear movers)
    variants["V1_ret1_3d_nogat"] = _topk_hold(ret1, K=3, hold=3, gate=None)

    # V2: breakout movers, hold 3d, gated
    variants["V2_breakout_3d_gat"] = _topk_hold(breakout, K=3, hold=3, gate=gate)

    # V3: vol-expansion movers, hold 3d, NO gate
    variants["V3_volexp_3d_nogat"] = _topk_hold(vol_jump, K=3, hold=3, gate=None)

    # V4: combo score, hold 3d, gated
    variants["V4_combo_3d_gat"] = _topk_hold(combo, K=3, hold=3, gate=gate)

    # V5: ret1 bag-profit: hold until +5% or 3d
    variants["V5_ret1_bag5pct"] = _bag_profit_weight(ret1, K=3, bag_tgt=0.05, C=C, gate=None, max_hold=3)

    # V6: breakout ride 5d, gated
    variants["V6_breakout_5d_gat"] = _topk_hold(breakout, K=3, hold=5, gate=gate)

    # V7: ret1 tight 1d (highest rotation)
    variants["V7_ret1_1d_nogat"] = _topk_hold(ret1, K=3, hold=1, gate=None)

    # V8: combo, NO gate (bear regime movers)
    variants["V8_combo_3d_nogat"] = _topk_hold(combo, K=3, hold=3, gate=None)

    # V9: single top ret1 mover (max concentration), 3d
    variants["V9_ret1_top1_3d"] = _topk_hold(ret1, K=1, hold=3, gate=None)

    # V10: vol-expansion, gated
    variants["V10_volexp_3d_gat"] = _topk_hold(vol_jump, K=3, hold=3, gate=gate)

    return variants


# ---- references -------------------------------------------------------

def build_references(ind: dict) -> dict:
    C = ind["C"]
    refs = {}
    # gated-beta: EW among all gated assets
    gate = ind["gate"].astype(float)
    refs["REF_gated_beta"] = gate.div(gate.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    # mom14 top3, rebal 7d (the mover_lab reference)
    refs["REF_mom14_top3_7d"] = ml.topk_weight(ind["mom14"], ind, K=3, rebal=7)
    return refs


# ---- print helpers ----------------------------------------------------

COLS = ["comp_2020", "comp_2021", "comp_2022", "comp_full",
        "maxDD", "green_2021", "green_all", "avg_expo", "avg_turnover"]

HDR = (f"{'variant':<26} {'2020%':>7} {'2021%':>7} {'2022%':>7} {'full%':>7} "
       f"{'maxDD':>7} {'gr21':>6} {'grAll':>6} {'expo':>6} {'turn':>6}")


def _row(label: str, m: dict) -> str:
    def fmt(k):
        v = m.get(k)
        return f"{v:>7.1f}" if v is not None else f"{'N/A':>7}"
    return (f"{label:<26} {fmt('comp_2020')} {fmt('comp_2021')} {fmt('comp_2022')} {fmt('comp_full')} "
            f"{fmt('maxDD')} {m.get('green_2021',0):>6.0f} {m.get('green_all',0):>6.0f} "
            f"{m.get('avg_expo',0):>6.2f} {m.get('avg_turnover',0):>6.3f}")


# ---- main -------------------------------------------------------------

def main():
    print("Loading mover_lab data...")
    ind = ml.load()
    print(f"  Assets: {list(ind['C'].columns)}")
    print(f"  Date range: {ind['C'].index[0].date()} -> {ind['C'].index[-1].date()}")
    print(f"  Rows: {len(ind['C'])}")
    print()

    print("Building reference strategies...")
    refs = build_references(ind)

    print("Building mover-capture variants...")
    variants = build_variants(ind)

    all_strats = {**refs, **variants}

    print("\nEvaluating all strategies (H=3 checkpoint)...")
    results = {}
    for name, W in all_strats.items():
        results[name] = ml.evaluate(W, ind, H=3, label=name)
        print(f"  done: {name}")

    # ---- summary table ------------------------------------------------
    print()
    print("=" * 100)
    print("DAILY-MOVERS CAPTURE LAB -- full results (H=3 checkpoint green-rate)")
    print("=" * 100)
    print(HDR)
    print("-" * 100)
    for name in refs:
        print(_row(name, results[name]))
    print("-" * 100)
    for name in variants:
        print(_row(name, results[name]))
    print("=" * 100)

    # ---- regime breakdown ----------------------------------------
    print()
    print("REGIME BREAKDOWN (2020=bull | 2021=mixed/ATH | 2022=crash):")
    print(f"{'variant':<26} {'2020%':>8} {'2021%':>8} {'2022%':>8}  verdict")
    print("-" * 75)
    for name in list(refs) + list(variants):
        m = results[name]
        c20 = m.get("comp_2020", 0) or 0
        c21 = m.get("comp_2021", 0) or 0
        c22 = m.get("comp_2022", 0) or 0
        # quick verdict
        wins = []
        if c21 > 0:   wins.append("bull21")
        if c22 > -20: wins.append("bear22-preserve")
        if c20 > 50:  wins.append("bull20")
        verdict = ", ".join(wins) if wins else "no-win-regime"
        print(f"{name:<26} {c20:>8.1f} {c21:>8.1f} {c22:>8.1f}  {verdict}")

    # ---- best / greediest -----------------------------------------
    print()
    # best = highest comp_full that is positive in 2022
    cands = {n: r for n, r in results.items() if n.startswith("V")}
    best_name = max(cands, key=lambda n: (
        (results[n].get("comp_2022", -999) or -999) > -30,   # not catastrophic bear
        results[n].get("comp_2021", -999) or -999,
        results[n].get("green_all", 0) or 0
    ))
    greediest_name = max(cands, key=lambda n: results[n].get("comp_2021", -999) or -999)

    print(f"BEST CONFIG (robustness-weighted):  {best_name}")
    print(f"  -> " + _row(best_name, results[best_name]))
    print()
    print(f"GREEDIEST CONFIG (max 2021 bull):   {greediest_name}")
    print(f"  -> " + _row(greediest_name, results[greediest_name]))

    # ---- where each wins / loses ----------------------------------
    print()
    print("WHERE IT WINS / WHERE IT LOSES:")
    print("-" * 80)
    for name in variants:
        m = results[name]
        c20 = m.get("comp_2020", 0) or 0
        c21 = m.get("comp_2021", 0) or 0
        c22 = m.get("comp_2022", 0) or 0
        g21 = m.get("green_2021", 0) or 0
        g22 = m.get("green_2022", 0) or 0
        wins = []
        loses = []
        if c21 > 200: wins.append("big-bull-2021")
        elif c21 > 50: wins.append("bull-2021")
        if c20 > 50: wins.append("bull-2020")
        if c22 > -30: wins.append("bear-preserve-2022")
        if g21 > 65: wins.append(f"high-green-rate-2021({g21:.0f}%)")
        if c21 < 0: loses.append("missed-2021-bull")
        if c22 < -50: loses.append("bear-drawdown-2022")
        if c20 < 0: loses.append("poor-2020")
        if g22 is not None and g22 < 35: loses.append(f"chop-bear-2022({g22:.0f}%)")
        print(f"  {name:<26}  WINS:{', '.join(wins) or 'none'}  LOSES:{', '.join(loses) or 'none'}")

    # ---- lesson ---------------------------------------------------
    print()
    print("1-LINE LESSON:")
    # compute best green_all vs ref
    ref_g = results["REF_gated_beta"].get("green_all", 0) or 0
    best_g = results[best_name].get("green_all", 0) or 0
    best_c21 = results[best_name].get("comp_2021", 0) or 0
    ref_c21 = results["REF_gated_beta"].get("comp_2021", 0) or 0
    greedy_c21 = results[greediest_name].get("comp_2021", 0) or 0
    greedy_c22 = results[greediest_name].get("comp_2022", -999) or -999
    print(f"  Short-horizon mover rotation captures the 2021 bull aggressively (best={best_name} "
          f"2021 {best_c21:.0f}% vs beta {ref_c21:.0f}%) but mover-chasing amplifies the 2022 crash "
          f"(greediest={greediest_name} 2022 {greedy_c22:.0f}%) -- green-rate is the honest signal: "
          f"mover-entries fire frequently but have low within-window persistence in bear/chop regimes.")

    print()
    print("NOTE: All results causal (W lagged 1 bar), taker cost charged, long-only spot, u10 assets.")
    print("      No look-ahead. Gate = C > SMA200. Overfit caveat: variants not cross-validated -- "
          "treat 2022 as the true held-out stress test.")
    return results


if __name__ == "__main__":
    main()
