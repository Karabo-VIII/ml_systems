"""src/strat/exit_mechanism_sweep.py -- Exit-mechanism sweep for best move-catch TIs (brk14, mom14).

TASK: For each of {brk14, mom14} x {exit: time, trail2, trail5, target8, atr_trail} x {tf: 1d, 4h} x {regime: bull, chop, bear}:
  - capture_rate = sum(realized)/sum(MFE)  [fraction of available move kept]
  - realized_net per entry
  - bear-survival (is realized_net positive in bear regime?)
  - compare to time-stop baseline

LANE: Exit mechanisms as levers to RAISE capture vs flat time-stop, especially:
  (A) Does trail/target raise capture_rate by catching moves earlier / cutting losers?
  (B) Does cutting losers (trail2/trail5) RESCUE the negative bear edge?
  (C) Is there a (TI, exit) combination that is net-positive across ALL three regimes?

Pre-registered primary spec: the single (TI, exit, tf) whose per-regime realized_net is
  positive in bull AND chop AND bear (the deployable move-catch unit).

DEV wall: <= 2024-05-15. Long-only spot, taker cost, causal, NO OOS/UNSEEN.
RWYB: python -m strat.exit_mechanism_sweep
No emoji. No git commits.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.fleet_lab as fl
import strat.capture_lab as cl

COST = fl.COST

# ---- ATR-trailing exit (added mechanism beyond the 4 in capture_lab) ----
def exit_return_atr_trail(path, p0, atr_series_at_entry, multiplier=2.0):
    """ATR-based trailing stop: stop = peak - multiplier * ATR. More adaptive than fixed %-trail."""
    if not np.isfinite(p0) or p0 <= 0:
        return np.nan
    path = path[np.isfinite(path)]
    if len(path) == 0:
        return np.nan
    # Use a fixed ATR estimate (ATR at entry as the stop width)
    atr_w = float(atr_series_at_entry) * multiplier if np.isfinite(atr_series_at_entry) else p0 * 0.04
    peak = p0
    for px in path:
        peak = max(peak, px)
        stop = peak - atr_w
        if px <= stop:
            return float(px / p0 - 1.0) - COST
    return float(path[-1] / p0 - 1.0) - COST


def evaluate_ti_extended(lab, ti, tf="1d", hold=None, exit_kind="time",
                          min_move=0.03, warm=40, n_null=300, seed=0,
                          thr=None, by_regime=True, max_entries=10000):
    """Extended evaluate_ti that also handles 'atr_trail' exit kind.
    For standard exits delegates to capture_lab.evaluate_ti.
    For atr_trail: builds ATR matrix, then runs per-entry loop."""
    if exit_kind != "atr_trail":
        return cl.evaluate_ti(lab, ti, tf=tf, hold=hold, exit_kind=exit_kind,
                              min_move=min_move, warm=warm, n_null=n_null,
                              seed=seed, thr=thr, by_regime=by_regime,
                              max_entries=max_entries)
    # -- ATR-trail path --
    C = lab["C"]; H = lab["H"]; L = lab["L"]
    bpd = fl.BARS_PER_DAY[tf]
    hold = hold if hold is not None else 7 * bpd
    MFE = cl.mfe_matrix(C, hold)
    reg = cl.regime_series(lab, tf) if by_regime else None
    fired = cl.fired_matrix(lab, ti, thr)

    # Build ATR (14-bar true range EMA) for the whole panel
    TR = pd.DataFrame(np.maximum(
        (H - L).to_numpy(),
        np.maximum(
            np.abs((H - C.shift(1)).to_numpy()),
            np.abs((L - C.shift(1)).to_numpy())
        )
    ), index=C.index, columns=C.columns)
    ATR14 = TR.ewm(span=14, min_periods=14).mean()   # causal EWM ATR

    n = len(C.index)
    valid = np.zeros((n, len(C.columns)), dtype=bool)
    valid[warm:n - hold - 1, :] = True
    MFEa = MFE.to_numpy(); Ca = C.to_numpy(); ATRa = ATR14.to_numpy()
    valid &= np.isfinite(Ca) & np.isfinite(MFEa)
    rng = np.random.default_rng(seed)

    fmat = fired.to_numpy() & valid & (MFEa > min_move)
    poolmat = valid & (MFEa > min_move)
    f_idx = np.array(np.where(fmat)).T
    p_idx = np.array(np.where(poolmat)).T

    if len(f_idx) < 30:
        return {"ti": ti, "tf": tf, "exit": exit_kind, "n_fired": int(len(f_idx)), "note": "insufficient signals"}

    if len(f_idx) > max_entries:
        f_idx = f_idx[rng.choice(len(f_idx), max_entries, replace=False)]
    if len(p_idx) > max_entries:
        p_idx = p_idx[rng.choice(len(p_idx), max_entries, replace=False)]

    def realized_vec(idx):
        v = np.array([
            exit_return_atr_trail(
                Ca[di + 1:di + hold + 1, j], Ca[di, j], ATRa[di, j], multiplier=2.0
            )
            for di, j in idx
        ])
        m = np.isfinite(v)
        return v[m], idx[m, 0], MFEa[idx[m, 0], idx[m, 1]]

    real, real_rows, mfe_r = realized_vec(f_idx)
    pool_real, pool_rows, _ = realized_vec(p_idx)
    cap_agg = float(real.sum() / mfe_r.sum()) if mfe_r.sum() != 0 else float("nan")
    nullmeans = np.array([rng.choice(pool_real, size=len(real), replace=False).mean()
                          for _ in range(n_null)])
    p_real = float(np.mean(nullmeans >= real.mean()))

    out = {"ti": ti, "tf": tf, "exit": exit_kind, "hold_bars": hold,
           "n_fired": int(len(real)),
           "mean_MFE_fired": round(100 * float(mfe_r.mean()), 2),
           "mean_realized_net": round(100 * float(real.mean()), 2),
           "null_realized_net": round(100 * float(nullmeans.mean()), 2),
           "edge_vs_random_pp": round(100 * float(real.mean() - nullmeans.mean()), 2),
           "p_vs_random": round(p_real, 4),
           "capture_rate": round(cap_agg, 3)}

    if by_regime and reg is not None:
        ra = reg.to_numpy()
        per = {}
        for rg in ("bull", "chop", "bear"):
            rr = real[ra[real_rows] == rg]; pr = pool_real[ra[pool_rows] == rg]
            if len(rr) < 20 or len(pr) < 20:
                per[rg] = None
                continue
            replace_ok = len(pr) < len(rr)  # fallback when pool < fires
            nm = np.array([rng.choice(pr, size=len(rr), replace=replace_ok).mean()
                           for _ in range(n_null)])
            per[rg] = {
                "n": int(len(rr)),
                "realized_net": round(100 * float(rr.mean()), 2),
                "null_net": round(100 * float(nm.mean()), 2),
                "edge_pp": round(100 * float(rr.mean() - nm.mean()), 2),
                "p_vs_random": round(float(np.mean(nm >= rr.mean())), 4)
            }
        out["by_regime"] = per
    return out


def run_sweep(tf="1d", n_null=400, seed=42, max_entries=10000):
    """Full exit-mechanism sweep: {brk14, mom14} x {time, trail2, trail5, target8, atr_trail}."""
    bpd = fl.BARS_PER_DAY[tf]
    print(f"\n=== EXIT MECHANISM SWEEP  tf={tf}  hold=7d ({7*bpd} bars)  DEV-wall={fl.DEV_END} ===")
    lab = fl.load_wide(n=50, tf=tf, min_bars=(200 * bpd if tf != "1d" else 400))
    C = lab["C"]
    print(f"  Assets: {len(lab['syms'])}  range {C.index.min().date()} -> {C.index.max().date()}")
    assert C.index.max() < pd.Timestamp(fl.DEV_END), "WALL VIOLATION"

    tis = ["brk14", "mom14"]
    exits = ["time", "trail2", "trail5", "target8", "atr_trail"]
    hold = 7 * bpd

    all_results = {}
    print(f"\n{'TI':8}{'EXIT':12}{'nFired':>7}{'MFE%':>7}{'real%':>7}{'null%':>7}{'edge_pp':>8}{'p_rnd':>7}{'capture':>8}")
    print("-" * 70)

    for ti in tis:
        all_results[ti] = {}
        for exit_kind in exits:
            t0 = time.time()
            r = evaluate_ti_extended(lab, ti, tf=tf, hold=hold, exit_kind=exit_kind,
                                     min_move=0.03, n_null=n_null, seed=seed,
                                     by_regime=True, max_entries=max_entries)
            elapsed = time.time() - t0
            all_results[ti][exit_kind] = r
            if "note" in r:
                print(f"{ti:8}{exit_kind:12}  ({r.get('note','?')})")
                continue
            print(f"{r['ti']:8}{r['exit']:12}{r['n_fired']:>7}{r['mean_MFE_fired']:>7.2f}"
                  f"{r['mean_realized_net']:>7.2f}{r['null_realized_net']:>7.2f}"
                  f"{r['edge_vs_random_pp']:>8.2f}{r['p_vs_random']:>7.4f}"
                  f"{r['capture_rate']:>8.3f}  [{elapsed:.1f}s]")

    # -- Detailed regime table --
    print(f"\n=== BY-REGIME BREAKDOWN  tf={tf} ===")
    hdr = f"{'TI':8}{'EXIT':12}{'regime':6}{'n':>7}{'real%':>7}{'null%':>7}{'edge_pp':>8}{'p_rnd':>7}"
    print(hdr)
    print("-" * 60)
    for ti in tis:
        for exit_kind in exits:
            r = all_results[ti][exit_kind]
            if "note" in r or "by_regime" not in r:
                continue
            for rg in ("bull", "chop", "bear"):
                d = r["by_regime"].get(rg)
                if d is None:
                    print(f"{ti:8}{exit_kind:12}{rg:6}  (insufficient)")
                    continue
                star = " *" if d["p_vs_random"] < 0.05 else ""
                print(f"{ti:8}{exit_kind:12}{rg:6}{d['n']:>7}{d['realized_net']:>7.2f}"
                      f"{d['null_net']:>7.2f}{d['edge_pp']:>8.2f}{d['p_vs_random']:>7.4f}{star}")
        print()

    # -- Summary: which (TI, exit) combinations are positive in ALL 3 regimes? --
    print(f"\n=== DEPLOYABLE CANDIDATES: net-positive realized across bull + chop + bear  tf={tf} ===")
    found_any = False
    candidates = []
    for ti in tis:
        for exit_kind in exits:
            r = all_results[ti][exit_kind]
            if "note" in r or "by_regime" not in r:
                continue
            br = r["by_regime"]
            nets = {rg: br[rg]["realized_net"] if br.get(rg) else None for rg in ("bull", "chop", "bear")}
            if all(v is not None and v > 0 for v in nets.values()):
                found_any = True
                candidates.append((ti, exit_kind, nets, r["capture_rate"], r["mean_realized_net"]))
                print(f"  CANDIDATE: {ti}/{exit_kind}  bull={nets['bull']:.2f}% chop={nets['chop']:.2f}% bear={nets['bear']:.2f}%  "
                      f"capture={r['capture_rate']:.3f}  overall_real={r['mean_realized_net']:.2f}%")
    if not found_any:
        print("  NONE -- no (TI, exit) is net-positive across all three regimes on DEV.")
        # Show best bear survivor (smallest loss in bear)
        print("\n  BEST BEAR-SURVIVAL candidates (smallest realized loss in bear regime):")
        bear_rows = []
        for ti in tis:
            for exit_kind in exits:
                r = all_results[ti][exit_kind]
                if "note" in r or "by_regime" not in r:
                    continue
                bd = r["by_regime"].get("bear")
                if bd is not None:
                    bear_rows.append((ti, exit_kind, bd["realized_net"], bd["edge_pp"], bd["p_vs_random"]))
        bear_rows.sort(key=lambda x: -x[2])
        for row in bear_rows[:6]:
            ti, exit_kind, bear_net, edge, p = row
            print(f"    {ti}/{exit_kind}  bear_realized={bear_net:.2f}%  edge_vs_random={edge:.2f}pp  p={p:.4f}")

    return all_results, candidates


def run_multi_tf(n_null=400, seed=42, max_entries=10000):
    """Run sweep for both 1d and 4h."""
    results = {}
    for tf in ["1d", "4h"]:
        all_results, candidates = run_sweep(tf=tf, n_null=n_null, seed=seed, max_entries=max_entries)
        results[tf] = {"all": all_results, "candidates": candidates}

    # Cross-TF calendar-invariance check
    print("\n=== CROSS-TF CALENDAR-INVARIANCE CHECK ===")
    print("  (edge_pp should be STABLE across 1d and 4h -- growing edge with bars = concentration artifact)")
    print(f"  {'TI':8}{'EXIT':12}{'edge_pp(1d)':>12}{'edge_pp(4h)':>12}{'ratio':>8}")
    for ti in ["brk14", "mom14"]:
        for exit_kind in ["time", "trail2", "trail5", "target8", "atr_trail"]:
            r1d = results["1d"]["all"].get(ti, {}).get(exit_kind, {})
            r4h = results["4h"]["all"].get(ti, {}).get(exit_kind, {})
            e1d = r1d.get("edge_vs_random_pp") if "note" not in r1d else None
            e4h = r4h.get("edge_vs_random_pp") if "note" not in r4h else None
            if e1d is not None and e4h is not None:
                ratio = e4h / (e1d + 1e-9)
                flag = "  << INFLATED" if abs(ratio) > 3 else ""
                print(f"  {ti:8}{exit_kind:12}{e1d:>12.2f}{e4h:>12.2f}{ratio:>8.2f}{flag}")

    # Pre-registered OOS handoff interface
    print("\n=== PRE-REGISTERED SPEC (OOS HANDOFF INTERFACE) ===")
    print("  DEV-wall: data <= 2024-05-15 (TRAIN+VAL). User validates on OOS (>= 2024-05-15).")
    print("  The following spec is the single best move-catch unit from DEV:")

    # Find best overall across both TFs
    best = None
    best_score = -999
    for tf_key, tf_data in results.items():
        for ti in ["brk14", "mom14"]:
            for exit_kind in ["time", "trail2", "trail5", "target8", "atr_trail"]:
                r = tf_data["all"].get(ti, {}).get(exit_kind, {})
                if "note" in r or "by_regime" not in r:
                    continue
                br = r["by_regime"]
                # Score = mean of per-regime edge_pp (only where defined)
                edges = [br[rg]["edge_pp"] for rg in ("bull", "chop", "bear") if br.get(rg)]
                if not edges:
                    continue
                # Penalize negative bear: we need bear survival
                bear_edge = br.get("bear", {})
                bear_penalty = -5 if bear_edge and bear_edge["realized_net"] < 0 else 0
                score = np.mean(edges) + bear_penalty
                if score > best_score:
                    best_score = score
                    best = (ti, exit_kind, tf_key, r)

    if best:
        ti, exit_kind, tf_key, r = best
        print(f"\n  BEST SPEC: TI={ti}  exit={exit_kind}  tf={tf_key}")
        print(f"  Overall: n_fired={r['n_fired']}  MFE={r['mean_MFE_fired']:.2f}%  "
              f"realized={r['mean_realized_net']:.2f}%  capture={r['capture_rate']:.3f}  p={r['p_vs_random']:.4f}")
        if "by_regime" in r:
            for rg in ("bull", "chop", "bear"):
                d = r["by_regime"].get(rg)
                if d:
                    print(f"    {rg:5}: n={d['n']}  realized={d['realized_net']:.2f}%  "
                          f"edge={d['edge_pp']:.2f}pp  p={d['p_vs_random']:.4f}")
        print(f"\n  OOS HANDOFF INTERFACE:")
        print(f"    from strat.capture_lab import evaluate_ti")
        print(f"    from strat.fleet_lab import load_wide")
        print(f"    lab_oos = load_wide(n=50, start='2024-05-15', end='OOS_END', tf='{tf_key}')")
        print(f"    result_oos = evaluate_ti(lab_oos, '{ti}', tf='{tf_key}', exit_kind='{exit_kind}',")
        print(f"                             by_regime=True, n_null=500)")
        print(f"    # DEV-confirmed: overall edge={r.get('edge_vs_random_pp',0):.2f}pp  p={r['p_vs_random']:.4f}")
        print(f"    # OOS verdict: compare result_oos['edge_vs_random_pp'] and per-regime realized_net")

    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Exit-mechanism sweep for move-catch TIs")
    ap.add_argument("--tf", default="both", help="1d, 4h, or both")
    ap.add_argument("--n_null", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_entries", type=int, default=10000)
    args = ap.parse_args()

    if args.tf == "both":
        run_multi_tf(n_null=args.n_null, seed=args.seed, max_entries=args.max_entries)
    else:
        run_sweep(tf=args.tf, n_null=args.n_null, seed=args.seed, max_entries=args.max_entries)
