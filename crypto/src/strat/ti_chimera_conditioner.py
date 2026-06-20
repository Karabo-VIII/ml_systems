"""src/strat/ti_chimera_conditioner.py -- TI x Chimera conditioner probe (fleet agent lane).

QUESTION: does a chimera feature as a CONDITIONER on a price-TI signal improve move-catch (capture-rate) vs
unconditioned TI and vs random entry, across regimes?

APPROACH:
  Conditioned trigger = TI fires AND chimera-condition(s) met.
  1 price-TI (mom14, strongest prior result) x {1, 2, 3 chimera} conditioner combinations:
    - Chimera features: vpin (low toxicity tercile), ofi (buy pressure > 0), dev (deviation), fdclose
    - Conditions: vpin LOW (<= tercile-33), ofi > 0, dev >= median, fdclose >= median
  Each conditioned trigger evaluated vs:
    (A) unconditioned TI (mom14 alone)
    (B) random-entry null (churn-immune)

ADVERSARIAL BATTERY (re the chop edge from prior run):
  1. DATE-BLOCK bootstrap (blocks of bars, not IID resampling) -> honest p_block
  2. REVERSE-SCORE: does WORST-momentum LOSE? real edge is direction-sensitive
  3. REGIME-LABEL shuffle: chop edge must DIE when regime labels destroyed
  4. CALENDAR-invariance: sweep 1d/4h -> edge/bar stable, not growing with bar-count

NOTE: Chimera standalone was DEAD (p=0.33/0.98). The question is whether chimera as a GATE/FILTER
adds to the price-TI edge (raises edge_pp / capture / p).

DEV-walled: <= 2024-05-15. No emoji (cp1252). Long-only spot, taker.
RWYB: python -m strat.ti_chimera_conditioner
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import strat.fleet_lab as fl
from strat.capture_lab import (
    mfe_matrix, time_return_matrix, regime_series, fired_matrix, COST
)

# ============================================================
# CONDITIONED TRIGGER BUILDER
# ============================================================

def _tercile_threshold(F_series: pd.DataFrame, frac: float = 0.33) -> pd.Series:
    """Row-wise (cross-sectional) tercile threshold per date. frac=0.33 -> 33rd pctile."""
    return F_series.quantile(frac, axis=1)  # shape: (n_dates,)


def conditioned_fired(lab, ti, conditions: list[tuple[str, str]], thr=None) -> pd.DataFrame:
    """Boolean matrix: base TI fires AND all chimera conditions satisfied (causal, per-asset).

    conditions: list of (chimera_feature, side) where side in {'low_tercile', 'high_tercile', 'pos', 'neg'}.
    'low_tercile' => feature <= cross-sectional 33rd pct (low toxicity for vpin)
    'high_tercile' => feature >= cross-sectional 67th pct
    'pos' => feature > 0
    'neg' => feature < 0
    """
    F = lab["F"]
    base = fired_matrix(lab, ti, thr)           # base TI signal
    gate = base.copy()
    for feat, side in conditions:
        if feat not in F:
            print(f"  WARN: chimera feature '{feat}' not in lab features, skip condition")
            continue
        Fx = F[feat]
        if side == "low_tercile":
            thr_row = _tercile_threshold(Fx, 0.33)
            cond = Fx.le(thr_row, axis=0)       # <= 33rd pctile cross-section
        elif side == "high_tercile":
            thr_row = _tercile_threshold(Fx, 0.67)
            cond = Fx.ge(thr_row, axis=0)
        elif side == "pos":
            cond = Fx > 0
        elif side == "neg":
            cond = Fx < 0
        elif side == "above_median":
            med = _tercile_threshold(Fx, 0.50)
            cond = Fx.ge(med, axis=0)
        elif side == "below_median":
            med = _tercile_threshold(Fx, 0.50)
            cond = Fx.le(med, axis=0)
        else:
            raise ValueError(f"Unknown side: {side}")
        gate = gate & cond.reindex_like(gate).fillna(False)
    return gate


# ============================================================
# EVALUATE CONDITIONED TI (vs random + vs unconditioned baseline)
# ============================================================

def evaluate_conditioned(lab, ti, conditions, tf="1d", hold=None, min_move=0.03, warm=40,
                          n_null=400, seed=0, thr=None, by_regime=True, block_size=30):
    """Capture-rate for a conditioned TI trigger.

    Returns dict with:
      - base metrics (n_fired, MFE, realized, null, edge_pp, p_iid, p_block, capture_rate)
      - by_regime breakdown
      - reverse_score test (WORST-mom fires: does it LOSE?)
      - regime_shuffle test (regime labels scrambled: edge must die)
      - condition_fire_rate (what fraction of base TI signals survive the condition)
    """
    C = lab["C"]
    bpd = fl.BARS_PER_DAY[tf]
    hold = hold if hold is not None else 7 * bpd
    MFE = mfe_matrix(C, hold)
    TIME = time_return_matrix(C, hold)
    reg = regime_series(lab, tf)
    MFEa = MFE.to_numpy(); TIMEa = TIME.to_numpy(); Ca = C.to_numpy()
    n = len(C.index)
    # valid universe mask
    valid = np.zeros((n, len(C.columns)), dtype=bool)
    valid[warm:n - hold - 1, :] = True
    valid &= np.isfinite(Ca) & np.isfinite(MFEa)

    # --- conditioned fire matrix ---
    cond_mat = conditioned_fired(lab, ti, conditions, thr)
    base_mat = fired_matrix(lab, ti, thr)   # unconditioned
    poolmat = valid & (MFEa > min_move)     # random-entry pool (same move-size gate)
    fmat_cond = cond_mat.to_numpy() & valid & (MFEa > min_move)
    fmat_base = base_mat.to_numpy() & valid & (MFEa > min_move)

    f_cond = np.array(np.where(fmat_cond)).T   # (n_fired, 2)
    f_base = np.array(np.where(fmat_base)).T
    p_idx = np.array(np.where(poolmat)).T

    # fire-rate: fraction of base signals surviving the chimera gate
    fire_rate = float(len(f_cond)) / max(1, len(f_base))

    if len(f_cond) < 20:
        return {"ti": ti, "conditions": conditions, "tf": tf,
                "n_fired": int(len(f_cond)), "note": "insufficient conditioned signals (<20)",
                "fire_rate": round(fire_rate, 3)}

    # realized returns (time exit: vectorized)
    real_cond = TIMEa[f_cond[:, 0], f_cond[:, 1]]
    mfe_cond  = MFEa[f_cond[:, 0], f_cond[:, 1]]
    real_base = TIMEa[f_base[:, 0], f_base[:, 1]]
    pool_real = TIMEa[p_idx[:, 0], p_idx[:, 1]]

    ok_c = np.isfinite(real_cond); real_cond = real_cond[ok_c]; mfe_cond = mfe_cond[ok_c]; rows_c = f_cond[ok_c, 0]
    ok_b = np.isfinite(real_base); real_base = real_base[ok_b]; rows_b = f_base[ok_b, 0]
    ok_p = np.isfinite(pool_real); pool_real = pool_real[ok_p]; rows_p = p_idx[ok_p, 0]

    rng = np.random.default_rng(seed)

    # --- IID null (pool resampled to len(real_cond)) ---
    nullmeans_iid = np.array([rng.choice(pool_real, size=len(real_cond), replace=False).mean()
                               for _ in range(n_null)])
    p_iid = float(np.mean(nullmeans_iid >= real_cond.mean()))

    # --- DATE-BLOCK bootstrap (block_size bars) to get honest p_block ---
    # We permute BLOCK indices so that entries overlapping in time stay together
    ra_full = reg.to_numpy()
    cond_dates = rows_c                          # bar-row indices for conditioned entries
    def block_bootstrap_p(real_target, rows_target, pool_real_b, n_boot=300):
        """Resample pool by DATE BLOCKS of block_size bars, draw matched N, compare means."""
        date_blocks = np.arange(0, n) // block_size       # block ID per bar
        unique_blocks = np.unique(date_blocks)
        pool_by_bar = {b: pool_real_b[rows_p // 1 == b] for b in unique_blocks} if False else None
        # simpler: tile pool into blocks, permute blocks, draw
        pool_rows_array = rows_p        # bar-row for each pool entry
        pool_block_id = pool_rows_array // block_size
        unique_pblocks = np.unique(pool_block_id)
        null_list = []
        for _ in range(n_boot):
            chosen_blocks = rng.choice(unique_pblocks, size=len(unique_pblocks), replace=True)
            samp = np.concatenate([pool_real_b[pool_block_id == b] for b in chosen_blocks if (pool_block_id == b).any()])
            if len(samp) < len(real_target): continue
            null_list.append(rng.choice(samp, size=len(real_target), replace=False).mean())
        if not null_list: return float("nan")
        null_arr = np.array(null_list)
        return float(np.mean(null_arr >= real_target.mean()))

    p_block = block_bootstrap_p(real_cond, rows_c, pool_real, n_boot=n_null)

    # --- REVERSE-SCORE test: worst-mom (anti-signal) should LOSE in same-regime windows ---
    # Reverse the TI condition: mom14 <= 0 (worst momentum) = anti-signal
    F = lab["F"]
    anti_mat = (F[ti] <= 0).to_numpy() & valid & (MFEa > min_move)
    anti_idx = np.array(np.where(anti_mat)).T
    if len(anti_idx) >= 20:
        real_anti = TIMEa[anti_idx[:, 0], anti_idx[:, 1]]
        ok_a = np.isfinite(real_anti); real_anti = real_anti[ok_a]; rows_a = anti_idx[ok_a, 0]
        rev_edge = round(100 * (real_cond.mean() - real_anti.mean()), 2)  # cond beats anti?
        rev_p = float(np.mean([rng.choice(real_anti, size=len(real_cond), replace=True).mean()
                                 >= real_cond.mean() for _ in range(n_null)]))
    else:
        rev_edge = float("nan"); rev_p = float("nan"); real_anti = np.array([])

    # --- REGIME-LABEL SHUFFLE: scramble regime, do per-chop edges die? ---
    reg_arr = ra_full.copy()
    reg_shuf = reg_arr.copy(); rng.shuffle(reg_shuf)
    def regime_edge(reg_labels, rg, real_v, rows_v, pool_v, rows_pool_v):
        mask = reg_labels[rows_v] == rg
        pool_mask = reg_labels[rows_pool_v] == rg
        rv = real_v[mask]; pv = pool_v[pool_mask]
        if len(rv) < 10 or len(pv) < 10: return None
        nm = np.array([rng.choice(pv, size=len(rv), replace=False).mean() for _ in range(n_null)])
        return {"n": int(len(rv)), "realized": round(100*float(rv.mean()),2),
                "null": round(100*float(nm.mean()),2), "edge_pp": round(100*(rv.mean()-nm.mean()),2),
                "p": round(float(np.mean(nm >= rv.mean())), 4)}

    # --- BY_REGIME (real labels) ---
    by_regime_out = {}
    for rg in ("bull", "chop", "bear"):
        r = regime_edge(reg_arr, rg, real_cond, rows_c, pool_real, rows_p)
        by_regime_out[rg] = r

    # regime-shuffle: same calc but with scrambled labels -> edge should be ~0
    by_regime_shuf = {}
    for rg in ("bull", "chop", "bear"):
        r = regime_edge(reg_shuf, rg, real_cond, rows_c, pool_real, rows_p)
        by_regime_shuf[rg] = r

    # aggregate capture rate
    cap_agg = float(real_cond.sum() / mfe_cond.sum()) if mfe_cond.sum() != 0 else float("nan")

    return {
        "ti": ti, "conditions": conditions, "tf": tf, "hold_bars": hold,
        "fire_rate": round(fire_rate, 3),
        "n_fired_base": int(len(real_base)),
        "n_fired_cond": int(len(real_cond)),
        "mean_MFE_cond": round(100 * float(mfe_cond.mean()), 2),
        "mean_realized_net_cond": round(100 * float(real_cond.mean()), 2),
        "mean_realized_net_base": round(100 * float(real_base.mean()), 2),
        "null_realized_net": round(100 * float(nullmeans_iid.mean()), 2),
        "edge_vs_random_pp": round(100 * (real_cond.mean() - nullmeans_iid.mean()), 2),
        "edge_vs_base_pp": round(100 * (real_cond.mean() - real_base.mean()), 2),
        "p_iid": round(p_iid, 4),
        "p_block": round(p_block, 4) if not np.isnan(p_block) else "nan",
        "capture_rate": round(cap_agg, 3),
        "reverse_test": {"cond_vs_anti_pp": rev_edge, "p_direction": round(rev_p, 4)
                          if not np.isnan(rev_p) else "nan",
                         "n_anti": int(len(real_anti))},
        "by_regime": by_regime_out,
        "regime_shuffle_check": by_regime_shuf,
    }


# ============================================================
# CALENDAR-INVARIANCE PROBE  (1d vs 4h)
# ============================================================

def calendar_invariance_probe(conditions, ti="mom14", n_null=300):
    """Check if edge/bar is stable across TFs (not just bar-count artifact)."""
    results = {}
    for tf in ("1d", "4h"):
        bpd = fl.BARS_PER_DAY[tf]
        print(f"  Loading {tf}...")
        lab = fl.load_wide(n=50, tf=tf, min_bars=200 * bpd if tf != "1d" else 400)
        r = evaluate_conditioned(lab, ti, conditions, tf=tf, n_null=n_null, by_regime=True)
        results[tf] = r
    # edge/bar = edge_pp / hold_bars (normalised) -- should be similar between 1d and 4h
    for tf, r in results.items():
        if "note" not in r:
            bpd = fl.BARS_PER_DAY[tf]
            edge_per_bar = r["edge_vs_random_pp"] / r["hold_bars"]
            results[tf]["edge_per_bar"] = round(edge_per_bar, 4)
    return results


# ============================================================
# MAIN SWEEP
# ============================================================

COND_COMBOS = [
    # 1-chimera
    ("vpin_low",          [("vpin", "low_tercile")]),           # low toxicity
    ("ofi_pos",           [("ofi", "pos")]),                    # buy pressure
    ("dev_high",          [("dev", "high_tercile")]),           # high deviation (trending?)
    ("fdclose_high",      [("fdclose", "high_tercile")]),       # fd close high
    # 2-chimera
    ("vpin_low+ofi_pos",  [("vpin", "low_tercile"), ("ofi", "pos")]),
    ("vpin_low+dev_high", [("vpin", "low_tercile"), ("dev", "high_tercile")]),
    ("ofi_pos+dev_high",  [("ofi", "pos"), ("dev", "high_tercile")]),
    # 3-chimera
    ("vpin_low+ofi_pos+dev_high", [("vpin", "low_tercile"), ("ofi", "pos"), ("dev", "high_tercile")]),
]

BASE_TI = "mom14"     # the lead TI from prior run


def run_sweep(tf="1d", n_null=400):
    bpd = fl.BARS_PER_DAY[tf]
    print(f"\n=== TI x CHIMERA CONDITIONER SWEEP ({tf}, n_null={n_null}) ===")
    print(f"    DEV wall: <= {fl.DEV_END}")
    print(f"    Base TI: {BASE_TI}  |  Chimera conditioner combos: {len(COND_COMBOS)}")
    t0 = time.time()
    lab = fl.load_wide(n=50, tf=tf, min_bars=200 * bpd if tf != "1d" else 400)
    C = lab["C"]
    print(f"    Assets: {len(lab['syms'])}  |  Range: {C.index.min()} -> {C.index.max()}")
    assert C.index.max() < pd.Timestamp(fl.DEV_END), "WALL VIOLATION"

    # Baseline: unconditioned mom14
    print(f"\n--- BASELINE: {BASE_TI} (unconditioned) ---")
    base_r = evaluate_conditioned(lab, BASE_TI, [], tf=tf, n_null=n_null, by_regime=True)
    _print_result("BASELINE (no cond)", base_r)

    results = {}
    for name, conds in COND_COMBOS:
        print(f"\n--- {name} ---")
        r = evaluate_conditioned(lab, BASE_TI, conds, tf=tf, n_null=n_null, by_regime=True)
        _print_result(name, r)
        results[name] = r

    print(f"\n=== SUMMARY TABLE ({tf}) ===")
    _print_summary(BASE_TI, base_r, results, tf)

    print(f"\n=== ADVERSARIAL BATTERY: CALENDAR-INVARIANCE (1d vs 4h) ===")
    print("  Testing best conditioner combo (vpin_low+ofi_pos) across TFs...")
    best_conds = [("vpin", "low_tercile"), ("ofi", "pos")]
    inv = calendar_invariance_probe(best_conds, ti=BASE_TI, n_null=n_null)
    print(f"\n  {'TF':6} {'edge_pp':>9} {'p_iid':>7} {'p_block':>8} {'edge/bar':>10} {'nFired':>8}")
    for tf_k, rv in inv.items():
        if "note" in rv:
            print(f"  {tf_k:6}  ({rv['note']})")
        else:
            print(f"  {tf_k:6} {rv['edge_vs_random_pp']:>9.2f} {rv['p_iid']:>7.4f} {str(rv['p_block']):>8} "
                  f"{rv.get('edge_per_bar',float('nan')):>10.4f} {rv['n_fired_cond']:>8}")

    print(f"\n[done in {time.time()-t0:.1f}s]")
    return base_r, results, inv


def _print_result(name, r):
    if "note" in r:
        print(f"  SKIP ({r['note']}) -- fire_rate={r.get('fire_rate','?')}")
        return
    print(f"  fire_rate={r['fire_rate']:.3f}  "
          f"n_fired_cond={r['n_fired_cond']} / n_fired_base={r['n_fired_base']}")
    print(f"  realized_cond={r['mean_realized_net_cond']:+.2f}%  "
          f"vs base={r['mean_realized_net_base']:+.2f}%  "
          f"vs null={r['null_realized_net']:+.2f}%")
    print(f"  edge_vs_random_pp={r['edge_vs_random_pp']:+.2f}pp  "
          f"edge_vs_base_pp={r['edge_vs_base_pp']:+.2f}pp  "
          f"p_iid={r['p_iid']:.4f}  p_block={r['p_block']}")
    print(f"  capture_rate={r['capture_rate']:.3f}")
    # by-regime
    br = r.get("by_regime", {})
    for rg in ("bull", "chop", "bear"):
        d = br.get(rg)
        if d:
            print(f"    {rg:5} n={d['n']:>5} real={d['realized']:+.2f}% null={d['null']:+.2f}% "
                  f"edge={d['edge_pp']:+.2f}pp p={d['p']:.4f}")
    # reverse test
    rev = r.get("reverse_test", {})
    if rev:
        print(f"  [REVERSE-SCORE] cond vs anti_pp={rev.get('cond_vs_anti_pp','?'):+.2f}  "
              f"p_direction={rev.get('p_direction','?')}")
    # regime-shuffle spot-check (chop)
    rs = r.get("regime_shuffle_check", {})
    chop_shuf = rs.get("chop")
    if chop_shuf:
        print(f"  [REGIME-SHUFFLE] chop edge after shuffle={chop_shuf['edge_pp']:+.2f}pp  "
              f"p={chop_shuf['p']:.4f}  (should be ~0 if real)")


def _print_summary(base_ti, base_r, results, tf):
    hdr = f"  {'combo':30} {'fire_rt':>8} {'edge_rand':>10} {'edge_base':>10} {'p_iid':>7} {'p_block':>8} {'cap':>7} {'chop_edge':>10} {'chop_p':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    def _row(name, r):
        if "note" in r:
            return f"  {name:30}  SKIP ({r['note']})"
        chop = r.get("by_regime", {}).get("chop")
        ce = f"{chop['edge_pp']:+.2f}" if chop else "  n/a"
        cp = f"{chop['p']:.4f}" if chop else "   n/a"
        return (f"  {name:30} {r['fire_rate']:>8.3f} {r['edge_vs_random_pp']:>+10.2f} "
                f"{r['edge_vs_base_pp']:>+10.2f} {r['p_iid']:>7.4f} {str(r['p_block']):>8} "
                f"{r['capture_rate']:>7.3f} {ce:>10} {cp:>8}")

    print(_row("BASELINE (no cond)", base_r))
    for name, r in results.items():
        print(_row(name, r))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="TI x Chimera conditioner move-catch probe")
    ap.add_argument("--tf", default="1d", help="Timeframe: 1d or 4h")
    ap.add_argument("--n_null", type=int, default=400, help="Bootstrap null samples")
    ap.add_argument("--calendar_inv", action="store_true", help="Run calendar-invariance 1d+4h sweep only")
    a = ap.parse_args()
    if a.calendar_inv:
        best_conds = [("vpin", "low_tercile"), ("ofi", "pos")]
        inv = calendar_invariance_probe(best_conds, ti=BASE_TI, n_null=a.n_null)
    else:
        run_sweep(tf=a.tf, n_null=a.n_null)
