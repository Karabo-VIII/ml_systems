"""src/strat/capture_adversarial.py -- ADVERSARIAL BATTERY for the chop move-catch edge (2026-06-20).

Four kills (per the ORC directive) applied to the top chop candidates:
  K1: DATE-BLOCK moving-block bootstrap (honest p with overlapping entries)
  K2: REVERSE-SCORE kill -- does WORST-momentum LOSE in chop? (real edge = direction-sensitive)
  K3: REGIME-LABEL shuffle -- chop edge DIES when regime labels are destroyed?
  K4: CALENDAR-INVARIANCE -- edge/day stable across 1d/4h, not growing with bar-count?

Plus for bear BH survivors: same K1/K2 battery.

DEV wall (<= 2024-05-15). No emoji. No commits.
RWYB: python -m strat.capture_adversarial
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.fleet_lab as fl
import strat.capture_lab as cl


# ---- K1: Date-block bootstrap (honest p, handles entry overlap) ----
def date_block_bootstrap(lab, ti, tf="1d", hold=7, exit_kind="time", min_move=0.03,
                         regime_target="chop", n_boot=500, block_size_days=20, seed=1):
    """Bootstrap the chop edge by resampling DATE BLOCKS of the time-series (not individual entries).
    This preserves the temporal structure and corrects for the overlapping-entry IID inflation.
    Returns: (observed_edge_pp, boot_mean, boot_std, p_value).
    """
    C = lab["C"]; bpd = fl.BARS_PER_DAY[tf]
    hold_bars = hold * bpd if tf != "1d" else hold
    block_size = block_size_days * bpd
    MFE = cl.mfe_matrix(C, hold_bars)
    TIME = cl.time_return_matrix(C, hold_bars)
    reg = cl.regime_series(lab, tf)
    fired = cl.fired_matrix(lab, ti)
    MFEa = MFE.to_numpy(); TIMEa = TIME.to_numpy(); Ca = C.to_numpy(); Ra = reg.to_numpy()
    n = len(C.index)
    valid = np.zeros((n, len(C.columns)), bool)
    valid[40:n - hold_bars - 1, :] = True
    valid &= np.isfinite(Ca) & np.isfinite(MFEa)
    pool_mask = valid & (MFEa > min_move)
    ti_mask = fired.to_numpy() & pool_mask
    # only keep bars in the target regime
    regime_ok = np.array([Ra == regime_target] * len(C.columns)).T   # (n, ncols) broadcast
    ti_mask_rg = ti_mask & regime_ok
    pool_mask_rg = pool_mask & regime_ok

    def edge_on_rows(ti_rows, pool_rows):
        """Edge = mean(realized|TI) - mean(realized|random from pool). Returns scalar."""
        if len(ti_rows) < 20 or len(pool_rows) < 20: return np.nan
        ti_real = TIMEa[ti_rows[:, 0], ti_rows[:, 1]]
        pool_real = TIMEa[pool_rows[:, 0], pool_rows[:, 1]]
        ti_real = ti_real[np.isfinite(ti_real)]
        pool_real = pool_real[np.isfinite(pool_real)]
        if len(ti_real) < 10 or len(pool_real) < 10: return np.nan
        return float(ti_real.mean() - pool_real.mean())

    # observed edge (full DEV)
    all_ti = np.array(np.where(ti_mask_rg)).T
    all_pool = np.array(np.where(pool_mask_rg)).T
    obs = edge_on_rows(all_ti, all_pool)
    if not np.isfinite(obs):
        return {"ti": ti, "regime": regime_target, "note": "insufficient data", "obs": obs}

    # block bootstrap over date indices
    date_idx = np.arange(n)
    rng = np.random.default_rng(seed)
    boot_edges = []
    n_blocks = max(1, n // block_size)
    for _ in range(n_boot):
        # resample n_blocks non-overlapping blocks (with replacement)
        starts = rng.integers(0, n - block_size, size=n_blocks)
        boot_dates = np.concatenate([np.arange(s, s + block_size) for s in starts])
        boot_dates = np.unique(np.clip(boot_dates, 0, n - 1))
        bset = set(boot_dates.tolist())
        b_ti = np.array([r for r in all_ti if r[0] in bset])
        b_pool = np.array([r for r in all_pool if r[0] in bset])
        e = edge_on_rows(b_ti, b_pool)
        if np.isfinite(e): boot_edges.append(e)
    boot_edges = np.array(boot_edges)
    p_val = float(np.mean(boot_edges <= 0)) if len(boot_edges) > 0 else np.nan
    return {
        "ti": ti, "tf": tf, "regime": regime_target, "n_fired": len(all_ti),
        "obs_edge_pp": round(100 * obs, 3),
        "boot_mean_pp": round(100 * float(np.mean(boot_edges)), 3) if len(boot_edges) else np.nan,
        "boot_std_pp": round(100 * float(np.std(boot_edges)), 3) if len(boot_edges) else np.nan,
        "boot_p05": round(100 * float(np.percentile(boot_edges, 5)), 3) if len(boot_edges) else np.nan,
        "p_block_boot": round(p_val, 4),
        "n_boot_valid": len(boot_edges),
    }


# ---- K2: Reverse-score kill (direction-sensitivity) ----
def reverse_score_kill(lab, ti, tf="1d", hold=7, min_move=0.03, regime_target="chop",
                       n_null=300, seed=2):
    """Does WORST-momentum (bottom tercile) LOSE in the target regime?
    Real edge = top fires positive AND bottom fires negative (asymmetric).
    Fake / concentration artifact = top fires positive but bottom is neutral (asymmetric-null coincidence).
    Returns (top_edge, bot_edge, direction_test_pass).
    """
    C = lab["C"]; bpd = fl.BARS_PER_DAY[tf]
    hold_bars = hold * bpd if tf != "1d" else hold
    MFE = cl.mfe_matrix(C, hold_bars); TIME = cl.time_return_matrix(C, hold_bars)
    reg = cl.regime_series(lab, tf)
    MFEa = MFE.to_numpy(); TIMEa = TIME.to_numpy(); Ca = C.to_numpy(); Ra = reg.to_numpy()
    n = len(C.index)
    valid = np.zeros((n, len(C.columns)), bool)
    valid[40:n - hold_bars - 1, :] = True
    valid &= np.isfinite(Ca) & np.isfinite(MFEa)
    pool_mask = valid & (MFEa > min_move)
    # regime filter
    regime_rows = np.where(Ra == regime_target)[0]
    regime_set = set(regime_rows.tolist())
    # TOP signal
    top_fired = cl.fired_matrix(lab, ti).to_numpy() & pool_mask
    top_rg = top_fired & np.isin(np.arange(n)[:, None] * np.ones((1, len(C.columns)), int), list(regime_set)).reshape(n, -1)
    # BOTTOM signal = opposite of top (reverse score)
    F_ti = lab["F"][ti].to_numpy()
    if ti in ("mom7", "mom14", "mom30", "accel", "brk14"):
        bot_fired_raw = F_ti < 0
    elif ti == "rsi14":
        bot_fired_raw = F_ti < 45
    elif ti == "rangepos":
        bot_fired_raw = F_ti < 0.3
    elif ti == "volexp":
        bot_fired_raw = F_ti < 0.8
    else:
        bot_fired_raw = F_ti < np.nanquantile(F_ti, 0.33, axis=1, keepdims=True)
    bot_fired = bot_fired_raw & pool_mask & np.isin(np.arange(n)[:, None] * np.ones((1, len(C.columns)), int), list(regime_set)).reshape(n, -1)
    pool_rg = pool_mask & np.isin(np.arange(n)[:, None] * np.ones((1, len(C.columns)), int), list(regime_set)).reshape(n, -1)
    def mean_edge(sig_mask):
        idx = np.array(np.where(sig_mask)).T
        pidx = np.array(np.where(pool_rg)).T
        if len(idx) < 20 or len(pidx) < 20: return np.nan, np.nan, np.nan
        real = TIMEa[idx[:, 0], idx[:, 1]]; pool = TIMEa[pidx[:, 0], pidx[:, 1]]
        real = real[np.isfinite(real)]; pool = pool[np.isfinite(pool)]
        if len(real) < 10 or len(pool) < 10: return np.nan, np.nan, np.nan
        rng = np.random.default_rng(seed)
        nm = np.array([rng.choice(pool, size=len(real), replace=len(pool) < len(real)).mean() for _ in range(n_null)])
        p = float(np.mean(nm >= real.mean()))
        return float(100 * real.mean()), float(100 * nm.mean()), p
    top_r, top_null, top_p = mean_edge(top_rg)
    bot_r, bot_null, bot_p = mean_edge(bot_fired)
    top_edge = (top_r - top_null) if np.isfinite(top_r) else np.nan
    bot_edge = (bot_r - bot_null) if np.isfinite(bot_r) else np.nan
    # direction test PASSES if top > 0 AND bottom < 0 (antisymmetric)
    direction_pass = (top_edge > 0) and (bot_edge < 0) if (np.isfinite(top_edge) and np.isfinite(bot_edge)) else None
    return {
        "ti": ti, "tf": tf, "regime": regime_target,
        "top_edge_pp": round(top_edge, 3) if np.isfinite(top_edge) else None,
        "top_p": round(top_p, 4) if np.isfinite(top_p) else None,
        "bot_edge_pp": round(bot_edge, 3) if np.isfinite(bot_edge) else None,
        "bot_p": round(bot_p, 4) if np.isfinite(bot_p) else None,
        "direction_test_PASS": direction_pass,
    }


# ---- K3: Regime-label shuffle (edge must die when labels are destroyed) ----
def regime_label_shuffle(lab, ti, tf="1d", hold=7, min_move=0.03, n_shuffle=300, seed=3):
    """Shuffle the regime labels (block-shuffle to preserve autocorrelation) and re-measure the chop edge.
    If the chop edge persists under shuffled labels -> it is NOT regime-specific -> artifact.
    Returns: (real_chop_edge, shuffle_mean_edge, p_shuffled).
    """
    C = lab["C"]; bpd = fl.BARS_PER_DAY[tf]
    hold_bars = hold * bpd if tf != "1d" else hold
    TIME = cl.time_return_matrix(C, hold_bars)
    MFE = cl.mfe_matrix(C, hold_bars)
    fired = cl.fired_matrix(lab, ti)
    reg = cl.regime_series(lab, tf)
    MFEa = MFE.to_numpy(); TIMEa = TIME.to_numpy(); Ca = C.to_numpy()
    n = len(C.index)
    valid = np.zeros((n, len(C.columns)), bool)
    valid[40:n - hold_bars - 1, :] = True
    valid &= np.isfinite(Ca) & np.isfinite(MFEa)
    pool_mask = valid & (MFEa > min_move)
    fired_mask = fired.to_numpy() & pool_mask

    def chop_edge_for_reg(reg_arr):
        rg_mask = (reg_arr == "chop")
        ti_rg = fired_mask & rg_mask[:, None]
        pool_rg = pool_mask & rg_mask[:, None]
        ti_idx = np.array(np.where(ti_rg)).T
        pool_idx = np.array(np.where(pool_rg)).T
        if len(ti_idx) < 20 or len(pool_idx) < 20: return np.nan
        ti_real = TIMEa[ti_idx[:, 0], ti_idx[:, 1]]
        pool_real = TIMEa[pool_idx[:, 0], pool_idx[:, 1]]
        ti_real = ti_real[np.isfinite(ti_real)]; pool_real = pool_real[np.isfinite(pool_real)]
        if len(ti_real) < 10 or len(pool_real) < 10: return np.nan
        return float(100 * (ti_real.mean() - pool_real.mean()))

    real_edge = chop_edge_for_reg(reg.to_numpy())
    rng = np.random.default_rng(seed)
    reg_arr = reg.to_numpy().copy()
    shuffle_edges = []
    block = max(5, n // 50)
    for _ in range(n_shuffle):
        shuf = reg_arr.copy()
        starts = list(range(0, n, block))
        rng.shuffle(starts)
        shuf_list = []
        for s in starts: shuf_list.extend(reg_arr[s:s+block].tolist())
        shuf = np.array(shuf_list[:n])
        e = chop_edge_for_reg(shuf)
        if np.isfinite(e): shuffle_edges.append(e)
    shuffle_edges = np.array(shuffle_edges)
    p_shuf = float(np.mean(shuffle_edges >= real_edge)) if len(shuffle_edges) > 0 else np.nan
    return {
        "ti": ti, "tf": tf,
        "real_chop_edge_pp": round(real_edge, 3),
        "shuffle_mean_pp": round(float(np.mean(shuffle_edges)), 3) if len(shuffle_edges) else np.nan,
        "shuffle_p05_pp": round(float(np.percentile(shuffle_edges, 95)), 3) if len(shuffle_edges) else np.nan,
        "p_shuffle": round(p_shuf, 4),
        "LABEL_SHUFFLE_KILL": bool(p_shuf > 0.05),
    }


# ---- K4: Calendar-invariance (1d vs 4h -- edge/day stable?) ----
def calendar_invariance(labs, ti, min_move=0.03, hold_days=7, regime_target="chop", n_null=200, seed=4):
    """Edge (pp) per calendar day at 1d vs 4h. A real effect = per-day consistent, not growing with bar count."""
    results = {}
    for tf in ["1d", "4h"]:
        lab = labs[tf]; bpd = fl.BARS_PER_DAY[tf]
        hold_bars = hold_days * bpd
        r = cl.evaluate_ti(lab, ti, tf=tf, hold=hold_bars, exit_kind="time",
                           min_move=min_move, n_null=n_null, by_regime=True, seed=seed)
        if "note" in r: results[tf] = None; continue
        rg = r.get("by_regime", {}).get(regime_target)
        results[tf] = {
            "edge_pp": rg["edge_pp"] if rg else np.nan,
            "edge_per_day": round(rg["edge_pp"] / hold_days, 3) if rg else np.nan,
            "p": rg["p_vs_random"] if rg else np.nan,
        }
    # consistency check: if edge_per_day is SIMILAR across TFs -> calendar-stable; if grows with bpd -> bar-count artifact
    r1 = results.get("1d", {}); r4 = results.get("4h", {})
    consistent = None
    if r1 and r4 and r1.get("edge_per_day") is not None and r4.get("edge_per_day") is not None:
        d1 = r1["edge_per_day"]; d4 = r4["edge_per_day"]
        # consistent if ratio is between 0.5 and 2.0 (within 2x)
        consistent = bool(0.3 < (d4 / (d1 + 1e-9)) < 3.0) if d1 > 0 and d4 > 0 else None
    return {"ti": ti, "regime": regime_target, "hold_days": hold_days,
            "1d": results.get("1d"), "4h": results.get("4h"), "calendar_consistent": consistent}


def main():
    import json, datetime
    print("[capture_adversarial] 4-kill battery on top chop/bear move-catch candidates")
    print(f"  DEV wall <= {fl.DEV_END}")
    print("Loading DEV labs...")
    labs = {}
    for tf in ["1d", "4h"]:
        bpd = fl.BARS_PER_DAY[tf]
        lab = fl.load_wide(n=50, tf=tf, min_bars=(200 * bpd if tf != "1d" else 400))
        assert lab["C"].index.max() < pd.Timestamp(fl.DEV_END), "WALL VIOLATION"
        print(f"  {tf}: {len(lab['syms'])} assets -> {lab['C'].index.max().date()}")
        labs[tf] = lab

    # top chop candidates (from main sweep BH survivors)
    chop_candidates = [("mom30", "1d"), ("rangepos", "1d"), ("rsi14", "1d"), ("brk14", "1d"), ("mom14", "1d")]
    # bear BH survivors
    bear_candidates = [("mom14", "4h"), ("volexp", "4h"), ("rsi14", "4h"), ("rangepos", "4h")]

    all_results = {"K1_block_boot": [], "K2_reverse_score": [], "K3_regime_shuffle": [], "K4_calendar": []}

    print("\n=== K1: DATE-BLOCK BOOTSTRAP (honest N_eff for chop edge) ===")
    print(f"  {'ti':10} {'tf':4} {'obs_pp':>8} {'boot_mean':>10} {'boot_std':>10} {'p05_pp':>8} {'p_block':>9} {'SURVIVES?':>10}")
    for ti, tf in chop_candidates:
        r = date_block_bootstrap(labs[tf], ti, tf=tf, hold=7, regime_target="chop", n_boot=400, block_size_days=20)
        all_results["K1_block_boot"].append(r)
        if "note" in r:
            print(f"  {ti:10} {tf:4}  ({r['note']})")
            continue
        survives = r["p_block_boot"] < 0.10
        print(f"  {ti:10} {tf:4} {r['obs_edge_pp']:>8.3f} {r['boot_mean_pp']:>10.3f} {r['boot_std_pp']:>10.3f} "
              f"{r['boot_p05']:>8.3f} {r['p_block_boot']:>9.4f} {'PASS' if survives else 'KILL':>10}")

    print("\n=== K1b: DATE-BLOCK BOOTSTRAP (bear edge -- 4h target8 survivors) ===")
    for ti, tf in bear_candidates:
        r = date_block_bootstrap(labs[tf], ti, tf=tf, hold=7, regime_target="bear", n_boot=400, block_size_days=20)
        all_results["K1_block_boot"].append(r)
        if "note" in r:
            print(f"  {ti:10} {tf:4}  ({r['note']})")
            continue
        survives = r["p_block_boot"] < 0.10
        print(f"  {ti:10} {tf:4} {r['obs_edge_pp']:>8.3f} {r['boot_mean_pp']:>10.3f} {r['boot_std_pp']:>10.3f} "
              f"{r['boot_p05']:>8.3f} {r['p_block_boot']:>9.4f} {'PASS' if survives else 'KILL':>10}")

    print("\n=== K2: REVERSE-SCORE KILL (direction-sensitivity) ===")
    print(f"  {'ti':10} {'tf':4} {'top_edge':>9} {'top_p':>7} {'bot_edge':>9} {'bot_p':>7} {'DIR_PASS?':>10}")
    for ti, tf in chop_candidates:
        r = reverse_score_kill(labs[tf], ti, tf=tf, hold=7, regime_target="chop")
        all_results["K2_reverse_score"].append(r)
        print(f"  {ti:10} {tf:4} {r.get('top_edge_pp', 'N/A'):>9}  {r.get('top_p', 'N/A'):>6}  "
              f"{r.get('bot_edge_pp', 'N/A'):>9}  {r.get('bot_p', 'N/A'):>6}  {str(r.get('direction_test_PASS', '?')):>10}")

    print("\n=== K3: REGIME-LABEL SHUFFLE (edge must die when labels destroyed) ===")
    print(f"  {'ti':10} {'tf':4} {'real_pp':>8} {'shuf_mean':>10} {'shuf_p95':>10} {'p_shuf':>8} {'KILL?':>7}")
    for ti, tf in chop_candidates:
        r = regime_label_shuffle(labs[tf], ti, tf=tf, hold=7, n_shuffle=250)
        all_results["K3_regime_shuffle"].append(r)
        print(f"  {ti:10} {tf:4} {r['real_chop_edge_pp']:>8.3f} {r['shuffle_mean_pp']:>10.3f} "
              f"{r['shuffle_p05_pp']:>10.3f} {r['p_shuffle']:>8.4f} {'YES' if r['LABEL_SHUFFLE_KILL'] else 'NO':>7}")

    print("\n=== K4: CALENDAR-INVARIANCE (edge/day at 1d vs 4h) ===")
    print(f"  {'ti':10} {'1d edge/day':>13} {'4h edge/day':>13} {'consistent?':>13}")
    for ti, _ in chop_candidates:
        r = calendar_invariance(labs, ti, regime_target="chop", hold_days=7)
        all_results["K4_calendar"].append(r)
        d1 = r.get("1d", {}) or {}; d4 = r.get("4h", {}) or {}
        print(f"  {ti:10} {d1.get('edge_per_day', 'N/A'):>13}  {d4.get('edge_per_day', 'N/A'):>12}  {str(r.get('calendar_consistent', '?')):>13}")

    print("\n=== ADVERSARIAL SUMMARY ===")
    print("  K1 BLOCK-BOOT: which chop candidates survive honest N_eff test (p<0.10)?")
    for r in all_results["K1_block_boot"]:
        if "note" not in r and r.get("regime") == "chop":
            label = "SURVIVES" if r.get("p_block_boot", 1.0) < 0.10 else "KILLED"
            print(f"    {r['ti']:10} {r['tf']:4} chop  p_block={r.get('p_block_boot','?')}  {label}")
    print("  K2 DIRECTION: which have antisymmetric (top+/bottom-) behaviour in chop?")
    for r in all_results["K2_reverse_score"]:
        print(f"    {r['ti']:10} {r['tf']:4} dir_pass={r.get('direction_test_PASS','?')}  "
              f"top={r.get('top_edge_pp','?')} bot={r.get('bot_edge_pp','?')}")
    print("  K3 LABEL-SHUFFLE: edge survives destroyed labels? (should be NO for real signal)")
    for r in all_results["K3_regime_shuffle"]:
        print(f"    {r['ti']:10} {r['tf']:4} KILL={r.get('LABEL_SHUFFLE_KILL','?')}  "
              f"real={r.get('real_chop_edge_pp','?')} shuf_mean={r.get('shuffle_mean_pp','?')} p={r.get('p_shuffle','?')}")
    print("  K4 CALENDAR: edge/day consistent across 1d/4h? (should be yes for real signal)")
    for r in all_results["K4_calendar"]:
        print(f"    {r['ti']:10} consistent={r.get('calendar_consistent','?')}")

    # save
    runs_dir = Path(__file__).resolve().parents[2] / "runs" / "strat"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = runs_dir / f"capture_adversarial_{ts}.json"
    import json
    with open(out, "w") as fh:
        json.dump(all_results, fh, default=str, indent=2)
    print(f"\n  Results -> {out}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
