"""src/strat/fleet_mover_dev_adversarial.py -- K=3 INDEPENDENT derivations of the decisive claim.

The headline (fleet_mover_dev.py): z=4.7-6.1 mover-selection vs random-same-exposure REPLICATES on DEV,
but regime-stratified the alpha is BULL-ONLY (chop/bear ~0 or negative). Before believing the verdict
'bull-beta artifact, not regime-robust skill', re-derive 3 independent ways:

  D1. EXPOSURE-IDENTITY proof: real and control carry the EXACT same daily total exposure (so the
      z cannot be a market-timing artifact). Print max abs daily exposure diff.
  D2. BLOCK-BOOTSTRAP (not the parametric t): within each regime, block-bootstrap the per-slice
      (real - control) advantage; report p05 and the fraction of bootstrap means > 0. Returns
      autocorrelate -> iid t overstates significance; the honest test is a moving-block bootstrap.
  D3. BTC-BETA / pure-cross-section decontamination: is the bull 'alpha' just buying high-beta names
      in a rising tape? Control #2 = EW-of-ALL-assets at the SAME exposure (no concentration). And
      regress per-slice real return on the same-slice BH(EW) return; report the residual alpha
      (intercept) and beta IN BULL. If residual alpha ~0 once we remove market beta -> beta artifact.

DEV-walled (<= 2024-05-15). No emoji. No git commit.
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
import strat.fleet_mover_dev as fm

COST = fl.COST
DEV_END = fl.DEV_END


def main():
    t0 = time.time()
    N_SLICES = 600
    SLICE_DAYS = 7
    REAL_SEEDS = [11, 23, 42]
    CTRL_SEEDS = list(range(101, 121))   # 20 random-pick controls
    DEV_OOS_START = "2020-09-01"
    K = 5
    BLEND = {"mom14": 0.35, "brk14": 0.25, "volexp": 0.20, "accel": 0.20}   # the strongest blend (plus_accel K=5)
    BOOT = 5000
    BLOCK = 5   # ~ one trading week block for the moving-block bootstrap

    print("=" * 80)
    print("ADVERSARIAL re-derivation -- 3 independent checks on the BULL-ONLY selection alpha")
    print(f"DEV WALL <= {DEV_END} | K={K} | blend={BLEND}")
    print("=" * 80)

    lab = fl.load_wide(n=50)
    C = lab["C"]; R = lab["R"]
    assert C.index.max() < pd.Timestamp(DEV_END), "WALL VIOLATION"

    cb = fm.circuit_breaker_inputs(C)
    cut = C.index[int(len(C.index) * 0.6)]
    vol_hi = float(cb["btc_vol"][C.index < cut].dropna().quantile(0.80))
    expo = fm.exposure_series(cb, vol_hi)
    regimes = fm.regime_labels(cb)
    idx_dev = C.index[(C.index >= pd.Timestamp(DEV_OOS_START)) & (C.index < pd.Timestamp(DEV_END))]
    n_avail = len(idx_dev)
    reg_at_start = regimes.reindex(idx_dev).values

    comp = fm.mover_score_panel(lab, BLEND)
    W_real = fm.build_W(lab, comp, expo, K=K, random_seed=None)
    b_real = fm.book_daily_returns(W_real, R)
    bh_b = fm.bh_ew_returns(C, R)

    ctrl_bs = [fm.book_daily_returns(fm.build_W(lab, comp, expo, K=K, random_seed=cs), R) for cs in CTRL_SEEDS]
    W_ctrls = [fm.build_W(lab, comp, expo, K=K, random_seed=cs) for cs in CTRL_SEEDS]

    out = {"K": K, "blend": BLEND, "vol_hi": round(vol_hi, 4)}

    # ============================================================
    # D1. EXPOSURE-IDENTITY: real vs each control must carry identical daily total exposure
    # ============================================================
    print("\n[D1] EXPOSURE IDENTITY (real vs control daily total book exposure):")
    er = W_real.sum(axis=1)
    max_diffs = []
    for cs, Wc in zip(CTRL_SEEDS, W_ctrls):
        ec = Wc.sum(axis=1)
        md = float((er - ec).abs().max())
        max_diffs.append(md)
    out["D1_max_exposure_diff"] = round(float(np.max(max_diffs)), 8)
    print(f"  max |exposure_real - exposure_ctrl| over all 20 controls, all bars = {out['D1_max_exposure_diff']:.2e}")
    print(f"  --> {'IDENTICAL exposure CONFIRMED (timing held constant)' if out['D1_max_exposure_diff'] < 1e-9 else 'WARNING: exposure differs -- timing NOT held constant'}")

    # ============================================================
    # Build per-slice arrays (pooled over real seeds), regime-tagged
    # control representative per slice = mean over the 20 random picks (the expected random book)
    # ============================================================
    starts_by_seed = {s: fm.sample_starts(n_avail, N_SLICES, SLICE_DAYS, s) for s in REAL_SEEDS}
    real_all, ctrl_all, bh_all, reg_all = [], [], [], []
    for s in REAL_SEEDS:
        starts = starts_by_seed[s]
        rr = fm.slice_returns(b_real, idx_dev, starts, SLICE_DAYS)
        bb = fm.slice_returns(bh_b, idx_dev, starts, SLICE_DAYS)
        cc = np.vstack([fm.slice_returns(bc, idx_dev, starts, SLICE_DAYS) for bc in ctrl_bs]).mean(axis=0)
        real_all.append(rr); ctrl_all.append(cc); bh_all.append(bb)
        reg_all.append(reg_at_start[starts])
    real_all = np.concatenate(real_all); ctrl_all = np.concatenate(ctrl_all)
    bh_all = np.concatenate(bh_all); reg_all = np.concatenate(reg_all)

    # ============================================================
    # D2. MOVING-BLOCK BOOTSTRAP of (real - control) within each regime (honest, autocorr-aware)
    # ============================================================
    print("\n[D2] MOVING-BLOCK BOOTSTRAP of per-slice (real - control), within regime:")
    print(f"     (iid t overstates significance on autocorrelated returns; block={BLOCK} slices, {BOOT} resamples)")
    rng = np.random.default_rng(7)
    out["D2_regime_bootstrap"] = {}
    for r in ["bull", "chop", "bear"]:
        m = reg_all == r
        diff = (real_all - ctrl_all)[m]
        n = len(diff)
        if n < 30:
            out["D2_regime_bootstrap"][r] = {"n": int(n), "note": "insufficient"}
            print(f"  {r:5s}: n={n} insufficient")
            continue
        # moving-block bootstrap of the mean
        nblocks = int(np.ceil(n / BLOCK))
        boot_means = np.empty(BOOT)
        for bI in range(BOOT):
            starts = rng.integers(0, n - BLOCK + 1, size=nblocks)
            idxs = (starts[:, None] + np.arange(BLOCK)[None, :]).ravel()[:n]
            boot_means[bI] = diff[idxs].mean()
        p05 = float(np.percentile(boot_means, 5))
        frac_pos = float((boot_means > 0).mean())
        out["D2_regime_bootstrap"][r] = {
            "n": int(n),
            "alpha_mean_pp": round(100 * float(diff.mean()), 3),
            "boot_p05_pp": round(100 * p05, 3),
            "boot_frac_gt0": round(frac_pos, 4),
            "verdict": "REAL" if p05 > 0 else ("ARTIFACT" if frac_pos < 0.95 else "AMBIGUOUS"),
        }
        ro = out["D2_regime_bootstrap"][r]
        print(f"  {r:5s}: n={n:4d} alpha={ro['alpha_mean_pp']:+.3f}pp  block-boot p05={ro['boot_p05_pp']:+.3f}pp  "
              f"frac>0={ro['boot_frac_gt0']:.3f}  -> {ro['verdict']}")

    # ============================================================
    # D3a. CONTROL #2 = EW-of-ALL at same exposure (no concentration) -- pure cross-section
    # ============================================================
    print("\n[D3a] CONTROL #2: EW-of-ALL-present-assets at the SAME circuit-breaker exposure (no concentration):")
    present = C.notna().astype(float)
    nP = present.sum(axis=1).replace(0, np.nan)
    W_ewall = present.div(nP, axis=0).mul(expo.reindex(C.index), axis=0).fillna(0.0)
    b_ewall = fm.book_daily_returns(W_ewall, R)
    ewall_all = []
    for s in REAL_SEEDS:
        ewall_all.append(fm.slice_returns(b_ewall, idx_dev, starts_by_seed[s], SLICE_DAYS))
    ewall_all = np.concatenate(ewall_all)
    out["D3a_vs_ewall"] = {}
    for r in ["bull", "chop", "bear"]:
        m = reg_all == r
        d = (real_all - ewall_all)[m]
        if len(d) < 30:
            out["D3a_vs_ewall"][r] = {"n": int(len(d)), "note": "insufficient"}; continue
        t = float(d.mean()) / (float(d.std(ddof=1)) / np.sqrt(len(d)) + 1e-12)
        out["D3a_vs_ewall"][r] = {"n": int(len(d)), "alpha_vs_ewall_pp": round(100 * float(d.mean()), 3),
                                  "t": round(t, 2)}
        print(f"  {r:5s}: real - EWall(same expo) = {100*float(d.mean()):+.3f}pp  t={t:+.2f}  "
              f"(real_pos={100*float((real_all[m]>0).mean()):.0f}% ewall_pos={100*float((ewall_all[m]>0).mean()):.0f}%)")

    # ============================================================
    # D3b. BTC-BETA decontamination: regress per-slice real on same-slice BH (market), IN BULL
    # ============================================================
    print("\n[D3b] MARKET-BETA decontamination (regress real-slice on BH-slice), per regime:")
    out["D3b_market_regression"] = {}
    for r in ["bull", "chop", "bear"]:
        m = reg_all == r
        y = real_all[m]; x = bh_all[m]
        if len(y) < 30:
            out["D3b_market_regression"][r] = {"n": int(len(y)), "note": "insufficient"}; continue
        X = np.vstack([np.ones_like(x), x]).T
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        se = float(np.sqrt((resid @ resid) / (len(y) - 2) / (len(y) * x.var() + 1e-12)))  # SE of slope approx
        # intercept SE
        s2 = float((resid @ resid) / (len(y) - 2))
        XtX_inv = np.linalg.inv(X.T @ X)
        se_int = float(np.sqrt(s2 * XtX_inv[0, 0]))
        t_int = float(beta[0]) / (se_int + 1e-12)
        out["D3b_market_regression"][r] = {
            "n": int(len(y)),
            "market_beta": round(float(beta[1]), 3),
            "residual_alpha_pp": round(100 * float(beta[0]), 3),
            "t_alpha": round(t_int, 2),
        }
        rr = out["D3b_market_regression"][r]
        print(f"  {r:5s}: beta_mkt={rr['market_beta']:+.3f}  residual_alpha(intercept)={rr['residual_alpha_pp']:+.3f}pp  "
              f"t_alpha={rr['t_alpha']:+.2f}")

    # also: vs same-slice BH directly (does real beat BH in bull/chop/bear?)
    print("\n[D3c] real - BH(EW, full exposure) per regime (does selection beat plain buy-hold?):")
    out["D3c_vs_bh"] = {}
    for r in ["bull", "chop", "bear"]:
        m = reg_all == r
        d = (real_all - bh_all)[m]
        out["D3c_vs_bh"][r] = {"n": int(m.sum()), "real_minus_bh_pp": round(100 * float(d.mean()), 3),
                               "beat_bh_rate": round(100 * float((real_all[m] > bh_all[m]).mean()), 1)}
        rr = out["D3c_vs_bh"][r]
        print(f"  {r:5s}: real - BH = {rr['real_minus_bh_pp']:+.3f}pp  beat_bh_rate={rr['beat_bh_rate']}%")

    out["runtime_s"] = round(time.time() - t0, 1)
    outp = ROOT.parent / "runs" / "strat" / f"fleet_mover_dev_adversarial_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    outp.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n{'='*80}\nSaved: {outp}  ({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    main()
