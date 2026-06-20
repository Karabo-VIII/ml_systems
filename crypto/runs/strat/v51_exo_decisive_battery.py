"""v51_exo_decisive_battery.py -- THE DECISIVE BATTERY: do v51 EXOGENOUS features break the de-risked-beta wall?

Quant referee lane. For each promising exogenous v51 feature, run BOTH trigger DIRECTIONS (top AND bottom tercile --
exo signals can be CONTRARIAN: a liq spike / funding extreme can precede a REVERSAL), and for each (feature, direction)
run the full hardened battery WITHIN each regime (bull/chop/bear):
  (1) DATE-BLOCK moving-block bootstrap of the edge (overlap-aware p -- the HONEST p, not the iid p_vs_random).
  (2) REGIME-LABEL SHUFFLE (circular rotation; SAME rotated labels for fired AND pool) -- is the edge REGIME-CONDITIONAL?
      real_edge must sit in the RIGHT tail of the rotated-label edge distribution (p_shuf_ge_real < 0.05).
  (3) Holm-Bonferroni FWER correction across the whole sweep (feature x direction x regime), because this is best-of-N.

VERDICT per (feature, direction, regime): is there a BEAR-POSITIVE (bear block_p_le0 < 0.05) or REGIME-CONDITIONAL
(p_shuf_ge_real < 0.05) move-catch edge -- after Holm correction? If NONE survive, the wall is exogenous-invariant.

DEV-walled (<= 2024-05-15). Long-only spot, taker cost. Causal (trailing-z features, end-of-day -> next-day entry).
No emoji (cp1252). RWYB: python v51_exo_decisive_battery.py
"""
from __future__ import annotations
import sys, json, datetime
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(ROOT))
import strat.v51_feature_lab as v51
import strat.capture_lab as cl
import strat.fleet_lab as fl

# the promising exogenous features (per the ORC first-read) + the bottom-tercile (contrarian) twins
FEATURES = ["norm_funding", "s3_smart_vs_retail_z", "liq_capitulation", "liq_short_panic",
            "bs_basis_xsec_z", "liq_delta_z30", "norm_oi_change"]
REGIMES = ("bull", "chop", "bear")
MIN_MOVE = 0.03
WARM = 40
HOLD_DAYS = 7


# ---- fired masks: BOTH directions. The v51 features are cross-sectional z-scores or flags. ----
def fired_mask(lab, feat, direction, valid):
    """Boolean (n, ncols) where the feature fires. direction in {top, bottom}.
    top    = high-value tercile (continuation prior); bottom = low-value tercile (contrarian/reversal prior).
    Flags (liq_capitulation / liq_short_panic) are binary {0,1}: top = flag==1, bottom = flag==0 (the non-event
    baseline -- a degenerate 'direction' that we report but down-weight; the flag's real test is top vs pool)."""
    X = lab["F"][feat]
    Xa = X.to_numpy()
    is_flag = feat in ("liq_capitulation", "liq_short_panic")
    if is_flag:
        if direction == "top":
            return (Xa >= 0.5) & valid
        return (Xa < 0.5) & valid   # baseline non-event (reported, not the headline)
    # continuous z-score: cross-sectional terciles per date (matches capture_lab top-tercile convention)
    top_thr = X.quantile(0.66, axis=1).to_numpy()[:, None]
    bot_thr = X.quantile(0.34, axis=1).to_numpy()[:, None]
    if direction == "top":
        return (Xa > top_thr) & valid & np.isfinite(Xa)
    return (Xa < bot_thr) & valid & np.isfinite(Xa)


def fired_pool_arrays(lab, feat, direction, hold):
    """Per-entry realized net (time-exit), date-row index, for FIRED set and POOL. Mirrors capture_lab universe."""
    C = lab["C"]
    MFE = cl.mfe_matrix(C, hold).to_numpy()
    TIME = cl.time_return_matrix(C, hold).to_numpy()
    Ca = C.to_numpy()
    n = len(C.index)
    valid = np.zeros((n, len(C.columns)), dtype=bool)
    valid[WARM:n - hold - 1, :] = True
    valid &= np.isfinite(Ca) & np.isfinite(MFE)
    poolmat = valid & (MFE > MIN_MOVE)
    fmat = fired_mask(lab, feat, direction, valid) & (MFE > MIN_MOVE)
    fi = np.array(np.where(fmat)).T
    pi = np.array(np.where(poolmat)).T
    if len(fi) == 0 or len(pi) == 0:
        return fi, np.array([]), pi, np.array([])
    fr = TIME[fi[:, 0], fi[:, 1]]; mf = np.isfinite(fr); fi, fr = fi[mf], fr[mf]
    pr = TIME[pi[:, 0], pi[:, 1]]; mp = np.isfinite(pr); pi, pr = pi[mp], pr[mp]
    return fi, fr, pi, pr


def block_bootstrap_edge(fi, fr, pi, pr, reg_arr, regime, n_dates, block_days=21, n_boot=2000, seed=1):
    """Honest moving-block bootstrap of WITHIN-REGIME edge (fired realized - same-regime-pool realized).
    Resamples by CALENDAR-DATE blocks (respects 7d-hold + cross-asset overlap). block_p_le0 = the HONEST verdict p."""
    rng = np.random.default_rng(seed)
    fm = reg_arr[fi[:, 0]] == regime; pm = reg_arr[pi[:, 0]] == regime
    fr_r, fdate = fr[fm], fi[fm, 0]
    pr_r, pdate = pr[pm], pi[pm, 0]
    if len(fr_r) < 20 or len(pr_r) < 20:
        return None
    obs = fr_r.mean() - pr_r.mean()
    fd = defaultdict(list); pdd = defaultdict(list)
    for v, d in zip(fr_r, fdate): fd[int(d)].append(v)
    for v, d in zip(pr_r, pdate): pdd[int(d)].append(v)
    n_blocks = max(1, n_dates // block_days)
    boots = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, n_dates - block_days + 1), size=n_blocks)
        fv = []; pv = []
        for s in starts:
            for d in range(int(s), int(s) + block_days):
                if d in fd: fv.extend(fd[d])
                if d in pdd: pv.extend(pdd[d])
        if len(fv) >= 5 and len(pv) >= 5:
            boots.append(np.mean(fv) - np.mean(pv))
    boots = np.array(boots)
    if len(boots) == 0:
        return None
    return {"regime": regime, "n_fired": int(len(fr_r)), "n_pool": int(len(pr_r)),
            "n_eff_fired_dates": int(len(np.unique(fdate))),
            "obs_edge_pp": round(100 * obs, 3),
            "block_p05_pp": round(100 * float(np.percentile(boots, 5)), 3),
            "block_p50_pp": round(100 * float(np.percentile(boots, 50)), 3),
            "block_p_le0": round(float(np.mean(boots <= 0.0)), 4),
            "n_boot_valid": int(len(boots))}


def regime_shuffle(fi, fr, pi, pr, reg_arr, regime, n_shuf=1000, block_days=21, seed=7):
    """HONEST regime-label shuffle: circularly ROTATE the regime-label series by a random offset (preserves
    composition + autocorr, breaks feature<->regime alignment). The same-regime POOL is re-drawn under the
    rotated labels too (both fired & pool use the SAME rotated labels). regime-conditional <=> real edge sits
    in the RIGHT tail (p_shuf_ge_real = frac of rotated edges >= real)."""
    rng = np.random.default_rng(seed)
    n = len(reg_arr)
    fm = reg_arr[fi[:, 0]] == regime; pm = reg_arr[pi[:, 0]] == regime
    if fm.sum() < 20 or pm.sum() < 20:
        return None
    obs = fr[fm].mean() - pr[pm].mean()
    shuf = []
    for _ in range(n_shuf):
        off = int(rng.integers(block_days, n - block_days))
        rr = np.roll(reg_arr, off)
        fmm = rr[fi[:, 0]] == regime; pmm = rr[pi[:, 0]] == regime
        if fmm.sum() < 20 or pmm.sum() < 20:
            continue
        shuf.append(fr[fmm].mean() - pr[pmm].mean())
    shuf = np.array(shuf)
    if len(shuf) == 0:
        return None
    p_ge = float(np.mean(shuf >= obs))
    return {"regime": regime, "obs_edge_pp": round(100 * obs, 3),
            "shuf_mean_pp": round(100 * float(shuf.mean()), 3),
            "shuf_p95_pp": round(100 * float(np.percentile(shuf, 95)), 3),
            "p_shuf_ge_real": round(p_ge, 4), "n_shuf": int(len(shuf)),
            "regime_conditional": bool(p_ge < 0.05)}


def holm_bonferroni(pvals, alpha=0.05):
    """Holm-Bonferroni step-down FWER. pvals: list of (key, p). Returns dict key -> {p, p_adj, reject}."""
    items = [(k, p) for k, p in pvals if p is not None]
    items_sorted = sorted(items, key=lambda kv: kv[1])
    m = len(items_sorted)
    out = {}
    prev_adj = 0.0
    for i, (k, p) in enumerate(items_sorted):
        adj = min(1.0, (m - i) * p)
        adj = max(adj, prev_adj)   # enforce monotonicity
        prev_adj = adj
        out[k] = {"p": round(p, 4), "p_holm": round(adj, 4), "reject_at_05": bool(adj < alpha)}
    return out


def main():
    print("[v51_exo_decisive_battery] DEV-walled exogenous move-catch -- both directions, block-boot + regime-shuffle")
    print(f"  loading v51 daily lab (DEV wall <= {v51.DEV_END}) ...", flush=True)
    lab = v51.load_v51_daily(n=50)
    C = lab["C"]
    assert C.index.max() < pd.Timestamp(v51.DEV_END), "WALL VIOLATION"
    avail = [f for f in FEATURES if f in lab["F"] and lab["F"][f].notna().sum().sum() > 500]
    print(f"  {len(lab['syms'])} assets; range {C.index.min().date()} -> {C.index.max().date()}")
    print(f"  features with data: {avail}", flush=True)

    bpd = 1  # 1d
    hold = HOLD_DAYS * bpd
    reg = cl.regime_series(lab, "1d"); reg_arr = reg.to_numpy()
    n_dates = len(C.index)
    reg_counts = {rg: int((reg_arr == rg).sum()) for rg in REGIMES}
    print(f"  regime bar counts: {reg_counts}", flush=True)

    results = {"meta": {"date_max": str(C.index.max().date()), "n_assets": len(lab["syms"]),
                        "regime_bar_counts": reg_counts, "hold_days": HOLD_DAYS, "min_move": MIN_MOVE},
               "features": {}}
    raw_p_for_holm = []   # collect block_p_le0 across the CONTINUOUS-feature top+bottom x 3 regimes sweep

    for feat in avail:
        is_flag = feat in ("liq_capitulation", "liq_short_panic")
        dirs = ("top",) if is_flag else ("top", "bottom")
        results["features"][feat] = {"is_flag": is_flag, "directions": {}}
        for direction in dirs:
            fi, fr, pi, pr = fired_pool_arrays(lab, feat, direction, hold)
            d_out = {"n_fired_total": int(len(fr)), "regimes": {}}
            if len(fr) < 20:
                d_out["note"] = "insufficient fired"
                results["features"][feat]["directions"][direction] = d_out
                continue
            for rg in REGIMES:
                boot = block_bootstrap_edge(fi, fr, pi, pr, reg_arr, rg, n_dates, block_days=21, n_boot=2000, seed=1)
                shu = regime_shuffle(fi, fr, pi, pr, reg_arr, rg, n_shuf=1000, block_days=21, seed=7)
                d_out["regimes"][rg] = {"block_boot": boot, "regime_shuffle": shu}
                if boot is not None and not is_flag:
                    raw_p_for_holm.append((f"{feat}|{direction}|{rg}", boot["block_p_le0"]))
            results["features"][feat]["directions"][direction] = d_out

    # Holm correction across the continuous-feature sweep (best-of-N discipline)
    holm = holm_bonferroni(raw_p_for_holm, alpha=0.05)
    results["holm_bonferroni"] = {"n_tests": len(raw_p_for_holm), "table": holm}

    # ---- console report ----
    print("\n===== PER-FEATURE BATTERY (1d, time-exit, 7d hold) =====", flush=True)
    hdr = f"{'feature':24}{'dir':7}{'reg':5}{'nFire':>7}{'nEff':>6}{'obs_pp':>9}{'blk_p05':>9}{'blk_ple0':>10}{'shuf_p':>9}{'rgcond':>8}"
    print(hdr)
    print("-" * len(hdr))
    for feat, fd in results["features"].items():
        for direction, dd in fd["directions"].items():
            if "note" in dd:
                print(f"{feat:24}{direction:7}{'--':5}  ({dd['note']}, nFired={dd['n_fired_total']})")
                continue
            for rg in REGIMES:
                r = dd["regimes"].get(rg, {})
                b = r.get("block_boot"); s = r.get("regime_shuffle")
                if b is None:
                    print(f"{feat:24}{direction:7}{rg:5}  (insufficient in regime)")
                    continue
                shp = s["p_shuf_ge_real"] if s else None
                rgc = s["regime_conditional"] if s else None
                print(f"{feat:24}{direction:7}{rg:5}{b['n_fired']:>7}{b['n_eff_fired_dates']:>6}"
                      f"{b['obs_edge_pp']:>9.2f}{b['block_p05_pp']:>9.2f}{b['block_p_le0']:>10.3f}"
                      f"{(shp if shp is not None else float('nan')):>9.3f}{str(rgc):>8}", flush=True)

    print("\n===== HOLM-BONFERRONI (continuous features, top+bottom x 3 regimes; bear-positive hunt) =====")
    print(f"  {len(raw_p_for_holm)} tests in the family.")
    survivors = [(k, v) for k, v in holm.items() if v["reject_at_05"]]
    if survivors:
        for k, v in sorted(survivors, key=lambda kv: kv[1]["p"]):
            print(f"  SURVIVES Holm: {k:40} p={v['p']:.4f} p_holm={v['p_holm']:.4f}")
    else:
        print("  NO test survives Holm correction at alpha=0.05 (block_p_le0 family).")

    # ---- DECISIVE verdict scan: bear-positive (block_p_le0<0.05) OR regime-conditional (p_shuf<0.05) ----
    print("\n===== DECISIVE VERDICT SCAN =====")
    bear_pos = []; rg_cond = []
    for feat, fd in results["features"].items():
        for direction, dd in fd["directions"].items():
            if "note" in dd: continue
            for rg in REGIMES:
                r = dd["regimes"].get(rg, {})
                b = r.get("block_boot"); s = r.get("regime_shuffle")
                if b and rg == "bear" and b["block_p_le0"] < 0.05 and b["obs_edge_pp"] > 0:
                    bear_pos.append((feat, direction, b["obs_edge_pp"], b["block_p_le0"]))
                if s and s["regime_conditional"] and b and b["obs_edge_pp"] > 0:
                    rg_cond.append((feat, direction, rg, b["obs_edge_pp"], s["p_shuf_ge_real"]))
    print(f"  BEAR-POSITIVE (raw block_p_le0 < 0.05, edge>0):")
    if bear_pos:
        for feat, d, e, p in bear_pos:
            holm_key = f"{feat}|{d}|bear"
            hk = holm.get(holm_key, {})
            print(f"    {feat} [{d}] bear edge={e:+.2f}pp raw_p={p:.3f} holm_p={hk.get('p_holm','NA')} "
                  f"holm_reject={hk.get('reject_at_05','NA')}")
    else:
        print("    NONE.")
    print(f"  REGIME-CONDITIONAL (p_shuf_ge_real < 0.05, edge>0):")
    if rg_cond:
        for feat, d, rg, e, p in rg_cond:
            print(f"    {feat} [{d}] {rg} edge={e:+.2f}pp p_shuf={p:.3f}")
    else:
        print("    NONE.")

    out = Path(__file__).resolve().parent / "v51_exo_decisive_battery_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
