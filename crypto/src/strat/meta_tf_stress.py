"""src/strat/meta_tf_stress.py -- STRESS the sub-daily chop/bear "unlock" before believing it.

The meta-audit found sub-daily (4h,1h) chop/bear selection alpha that survives the moving-block bootstrap
(1h chop +13.9pp NET, bear +6.4pp NET). A sub-daily unlock is EXACTLY the multiple-comparisons / slicing-artifact
mirage this project kills. Five adversarial stress tests:

  S1. HORIZON-SCALING SANITY. A REAL per-calendar-week selection edge should be ~calendar-invariant across TF.
      If alpha scales ~linearly with bars-per-slice (6x more bars @1h -> ~Nx bigger alpha), it is a COMPOUNDING-OF-
      MANY-BARS artifact, not a per-period edge. Report alpha / (slice_bars) and alpha per 7-calendar-days.

  S2. PER-BAR DECOMPOSITION. Decompose the 7d slice spread into per-bar real-vs-ctrl mean return. If the edge is a
      tiny per-bar cross-sectional tilt that COMPOUNDS, the per-bar alpha is small + the 7d spread is geometric
      compounding of it. Report mean per-bar (real - ctrl) and (1+perbar)^slice_bars - 1 reconstruction.

  S3. NON-OVERLAPPING / TURNOVER REALITY. Re-run with NON-OVERLAPPING consecutive 7d blocks (no overlapping-window
      inflation of n) AND report what fraction of the gross edge the realized taker cost eats at this turnover.
      Also: HOLD-TO-MATURITY variant -- pick top-K at slice start, HOLD them the whole 7d (no per-bar reselection,
      so cost = one round-trip per slice). Does the chop/bear edge SURVIVE when you can't re-pick every bar?

  S4. SIGN/PLACEBO. Shuffle the slice-start regime labels (break the regime->alpha link). A real regime-conditional
      edge dies; a pooled artifact survives. And REVERSE the score (pick the WORST K) -- a real momentum edge flips
      sign; an alignment/leak artifact may not.

  S5. SHIFT-2 LEAK PROBE. Re-run with positions lagged 2 bars instead of 1 (extra 1-bar gap). A causal edge decays
      gracefully; a 1-bar look-ahead/alignment leak (which gets MORE bars to leak at fine TF) collapses.

DEV-walled. No emoji. No commit.
RWYB: C:/.../.venv/Scripts/python.exe -m strat.meta_tf_stress
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
import strat.meta_tf_invariance_audit as M

COST = fl.COST
DEV_END = fl.DEV_END
BPD = fl.BARS_PER_DAY
BLEND = {"mom14": 0.35, "brk14": 0.25, "volexp": 0.20, "accel": 0.20}


def book_returns_lag(W, R, lag=1):
    Ral = R.reindex(index=W.index, columns=W.columns).fillna(0.0)
    pos = W.shift(lag).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs()).sum(axis=1)
    net = (pos * Ral).sum(axis=1) - turn * (COST / 2.0)
    gross = (pos * Ral).sum(axis=1)
    return net, gross


def build_W_hold(C, comp, expo, K, warmup, slice_bars, random_seed=None):
    """HOLD-TO-MATURITY: pick top-K (or random-K) only at slice-start bars (every slice_bars), HOLD slice_bars.
       This caps turnover at ~1 RT per slice (no per-bar reselection)."""
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    rng = np.random.default_rng(random_seed) if random_seed is not None else None
    cols = list(C.columns)
    e_arr = expo.reindex(C.index).values
    i = warmup
    n = len(C.index)
    while i < n:
        row = comp.iloc[i]
        valid = [s for s in cols if pd.notna(row[s])]
        if len(valid) >= K:
            if rng is not None:
                picks = list(rng.choice(valid, size=K, replace=False))
            else:
                picks = sorted(valid, key=lambda s: -row[s])[:K]
            e = float(e_arr[i]); w = e / len(picks)
            for j in range(i, min(i + slice_bars, n)):
                d = C.index[j]
                for s in picks:
                    W.loc[d, s] = w
        i += slice_bars
    return W


def regime_split_slices(real_all, ctrl_all, reg_all):
    out = {}
    for r in ["bull", "chop", "bear"]:
        m = reg_all == r
        out[r] = (real_all[m], ctrl_all[m])
    return out


def run_tf_stress(tf, n_slices=600, real_seeds=(11, 23, 42), ctrl_seeds=tuple(range(101, 121)),
                  K=5, boot=4000, dev_oos_start="2020-09-01"):
    bpd = BPD[tf]
    def days(d):
        return max(1, int(round(d * bpd)))
    slice_bars = days(7); warmup = days(200); block = days(7)
    lab = fl.load_wide(n=50, tf=tf, min_bars=days(200) + 50)
    C = lab["C"]; R = lab["R"]
    F = M.calendar_features(lab, bpd)
    cb = M.calendar_circuit_breaker(C, bpd)
    cut = C.index[int(len(C.index) * 0.6)]
    vol_hi = float(cb["btc_vol"][C.index < cut].dropna().quantile(0.80))
    expo = M.exposure_series(cb, vol_hi)
    regimes = M.regime_labels(cb)
    idx_dev = C.index[(C.index >= pd.Timestamp(dev_oos_start)) & (C.index < pd.Timestamp(DEV_END))]
    warm_date = C.index[min(warmup, len(C.index) - 1)]
    idx_dev = idx_dev[idx_dev >= warm_date]
    n_avail = len(idx_dev)
    reg_at_start = regimes.reindex(idx_dev).values
    comp = M.mover_score_panel(C, F, BLEND)

    # standard per-bar-reselect real + controls, lag1 and lag2
    W_real = M.build_W(C, comp, expo, K=K, warmup=warmup, random_seed=None)
    W_ctrls = [M.build_W(C, comp, expo, K=K, warmup=warmup, random_seed=cs) for cs in ctrl_seeds]
    b_real_l1, _ = book_returns_lag(W_real, R, 1)
    b_real_l2, _ = book_returns_lag(W_real, R, 2)
    ctrl_l1 = [book_returns_lag(Wc, R, 1)[0] for Wc in W_ctrls]
    ctrl_l2 = [book_returns_lag(Wc, R, 2)[0] for Wc in W_ctrls]

    # HOLD-TO-MATURITY real + controls (one RT per slice)
    W_real_h = build_W_hold(C, comp, expo, K, warmup, slice_bars, random_seed=None)
    W_ctrl_h = [build_W_hold(C, comp, expo, K, warmup, slice_bars, random_seed=cs) for cs in ctrl_seeds]
    b_real_h, _ = book_returns_lag(W_real_h, R, 1)
    ctrl_h = [book_returns_lag(Wc, R, 1)[0] for Wc in W_ctrl_h]

    starts_by_seed = {s: M.sample_starts(n_avail, n_slices, slice_bars, s) for s in real_seeds}

    def pooled(bret_real, bret_ctrls):
        ra, ca, reg = [], [], []
        for s in real_seeds:
            st = starts_by_seed[s]
            rr = M.slice_returns(bret_real, idx_dev, st, slice_bars)
            cc = np.vstack([M.slice_returns(bc, idx_dev, st, slice_bars) for bc in bret_ctrls]).mean(axis=0)
            ra.append(rr); ca.append(cc); reg.append(reg_at_start[st])
        return np.concatenate(ra), np.concatenate(ca), np.concatenate(reg)

    real_l1, ctrl_c1, reg_all = pooled(b_real_l1, ctrl_l1)
    real_l2, ctrl_c2, _ = pooled(b_real_l2, ctrl_l2)
    real_h, ctrl_ch, _ = pooled(b_real_h, ctrl_h)
    rng = np.random.default_rng(7)

    res = {"tf": tf, "bpd": bpd, "slice_bars": slice_bars}

    # ---- S1 horizon scaling + S2 per-bar decomposition ----
    s1s2 = {}
    for r in ["bull", "chop", "bear"]:
        m = reg_all == r
        alpha = float((real_l1 - ctrl_c1)[m].mean())   # 7d slice net alpha
        # per-bar equivalent: (1+alpha)^(1/slice_bars)-1 is NOT right for a spread; instead decompose real & ctrl
        real_7d = float(real_l1[m].mean()); ctrl_7d = float(ctrl_c1[m].mean())
        # geometric per-bar rate implied by the 7d compounded book return
        def per_bar(x):
            return (1 + x) ** (1.0 / slice_bars) - 1 if x > -1 else float("nan")
        s1s2[r] = {
            "alpha_7d_net_pp": round(100 * alpha, 3),
            "alpha_per_bar_bp": round(1e4 * (per_bar(real_7d) - per_bar(ctrl_7d)), 3),
            "alpha_per_calendar_day_pp": round(100 * alpha / 7.0, 4),
            "real_per_bar_bp": round(1e4 * per_bar(real_7d), 3),
            "ctrl_per_bar_bp": round(1e4 * per_bar(ctrl_7d), 3),
        }
    res["S1_S2_scaling"] = s1s2

    # ---- S3 hold-to-maturity (cost-honest, 1 RT/slice) ----
    s3 = {}
    for r in ["bull", "chop", "bear"]:
        m = reg_all == r
        diff = (real_h - ctrl_ch)[m]
        s3[r] = M.moving_block_boot(diff, block, boot, rng)
    res["S3_hold_to_maturity"] = s3

    # ---- S4a placebo: shuffle regime labels (break regime->alpha) ----
    rng2 = np.random.default_rng(13)
    reg_shuf = reg_all.copy(); rng2.shuffle(reg_shuf)
    s4a = {}
    for r in ["bull", "chop", "bear"]:
        m = reg_shuf == r
        diff = (real_l1 - ctrl_c1)[m]
        s4a[r] = M.moving_block_boot(diff, block, boot, rng2)
    res["S4a_regime_label_shuffle"] = s4a

    # ---- S4b reverse score: pick WORST-K (momentum edge should flip sign negative) ----
    comp_rev = -comp
    W_rev = M.build_W(C, comp_rev, expo, K=K, warmup=warmup, random_seed=None)
    b_rev, _ = book_returns_lag(W_rev, R, 1)
    real_rev, ctrl_crev, _ = pooled(b_rev, ctrl_l1)
    s4b = {}
    for r in ["bull", "chop", "bear"]:
        m = reg_all == r
        diff = (real_rev - ctrl_crev)[m]
        s4b[r] = {"alpha_7d_net_pp": round(100 * float(diff.mean()), 3)}
    res["S4b_reverse_score"] = s4b

    # ---- S5 shift-2 leak probe ----
    s5 = {}
    for r in ["bull", "chop", "bear"]:
        m = reg_all == r
        a1 = float((real_l1 - ctrl_c1)[m].mean())
        a2 = float((real_l2 - ctrl_c2)[m].mean())
        s5[r] = {"alpha_lag1_pp": round(100 * a1, 3), "alpha_lag2_pp": round(100 * a2, 3),
                 "retention_pct": round(100 * a2 / a1, 1) if abs(a1) > 1e-9 else None}
    res["S5_shift2_leak"] = s5
    return res


def main():
    t0 = time.time()
    TFS = ["1d", "4h", "1h"]
    print("=" * 100)
    print("STRESS the sub-daily chop/bear selection 'unlock' (5 adversarial probes)")
    print("=" * 100)
    out = {"cost": COST, "tfs": {}}
    for tf in TFS:
        print(f"\n{'#'*100}\nTF = {tf}\n{'#'*100}")
        r = run_tf_stress(tf)
        out["tfs"][tf] = r
        print("  [S1/S2] horizon-scaling + per-bar decomposition (is the 7d edge just compounded per-bar tilt?):")
        for rg in ["bull", "chop", "bear"]:
            d = r["S1_S2_scaling"][rg]
            print(f"    {rg:5s}: 7d-alpha={d['alpha_7d_net_pp']:+.3f}pp  per-bar-alpha={d['alpha_per_bar_bp']:+.3f}bp  "
                  f"per-cal-day={d['alpha_per_calendar_day_pp']:+.4f}pp  (real/ctrl per-bar={d['real_per_bar_bp']:+.2f}/{d['ctrl_per_bar_bp']:+.2f}bp)")
        print("  [S3] HOLD-TO-MATURITY (1 RT/slice, no per-bar reselect) -- does the edge survive honest turnover?:")
        for rg in ["bull", "chop", "bear"]:
            d = r["S3_hold_to_maturity"][rg]
            if "note" in d: print(f"    {rg:5s}: {d['note']}"); continue
            print(f"    {rg:5s}: alpha={d['alpha_mean_pp']:+.3f}pp  p05={d['boot_p05_pp']:+.3f}pp  "
                  f"frac>0={d['boot_frac_gt0']:.3f}  -> {d['verdict']}")
        print("  [S4a] REGIME-LABEL SHUFFLE placebo (real regime edge should DIE; pooled artifact survives):")
        for rg in ["bull", "chop", "bear"]:
            d = r["S4a_regime_label_shuffle"][rg]
            if "note" in d: print(f"    {rg:5s}: {d['note']}"); continue
            print(f"    {rg:5s}: alpha={d['alpha_mean_pp']:+.3f}pp  p05={d['boot_p05_pp']:+.3f}pp -> {d['verdict']}")
        print("  [S4b] REVERSE-SCORE (worst-K) -- momentum edge should flip NEGATIVE:")
        for rg in ["bull", "chop", "bear"]:
            print(f"    {rg:5s}: reverse-alpha={r['S4b_reverse_score'][rg]['alpha_7d_net_pp']:+.3f}pp")
        print("  [S5] SHIFT-2 leak probe (causal edge retains; 1-bar leak collapses):")
        for rg in ["bull", "chop", "bear"]:
            d = r["S5_shift2_leak"][rg]
            print(f"    {rg:5s}: lag1={d['alpha_lag1_pp']:+.3f}pp lag2={d['alpha_lag2_pp']:+.3f}pp "
                  f"retention={d['retention_pct']}%")

    out["runtime_s"] = round(time.time() - t0, 1)
    outp = ROOT.parent / "runs" / "strat" / f"meta_tf_stress_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    outp.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {outp}  ({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    main()
