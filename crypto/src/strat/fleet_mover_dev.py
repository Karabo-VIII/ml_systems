"""src/strat/fleet_mover_dev.py -- PORT of the ungated mover-SELECTION engine to fleet_lab DEV.

DECISIVE QUESTION (DEV-walled <= 2024-05-15 ONLY; NEVER OOS/UNSEEN):
  Does mover-SELECTION beat random-SAME-EXPOSURE on DEV, AND survive REGIME-STRATIFICATION
  (bull/chop/bear)? If the alpha vanishes outside bull -> bull-beta artifact (concentration in
  a rising market). If it holds across regimes -> REAL cross-sectional selection alpha.

PORTED FROM (OLD OOS harness, do NOT import here): strat/mover_capture_engine.py + quant_referee_mover.py.
  - mover score = cross-sectional z-composite of {mom14, brk14, volexp} (a couple blends tried).
  - MARKET circuit-breaker = total book exposure scaled by breadth (% of universe above own SMA50)
    AND BTC-trend (causal). Individual movers NEVER excluded.
  - SAME-EXPOSURE SHUFFLE control = IDENTICAL daily total exposure, but K random picks.
  - AGGREGATE capture = sum(realized) / sum(oracle)  (NOT mean of per-block ratios).

ALL DATA + FEATURES come from strat.fleet_lab (DEV-window-walled load_wide). Causal/leak-free.

RWYB: C:/.../.venv/Scripts/python.exe -m strat.fleet_mover_dev
No emoji (cp1252). Does NOT git commit.
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

COST = fl.COST          # 0.0024 taker round-trip
DEV_END = fl.DEV_END


# ============================================================
# CAUSAL CIRCUIT-BREAKER INPUTS derived from the C panel (fleet_lab has no sma/vol panels)
# ============================================================
def circuit_breaker_inputs(C: pd.DataFrame):
    """All causal. SMA50/SMA200 per asset, breadth (% above own SMA50), BTC-trend, BTC vol20."""
    sma50 = C.rolling(50, min_periods=50).mean()
    sma200 = C.rolling(200, min_periods=200).mean()
    R = C.pct_change(fill_method=None)
    vol20 = R.rolling(20, min_periods=20).std() * np.sqrt(365)
    present = C.notna() & sma50.notna()
    above = (C > sma50)
    breadth = above.where(present).sum(axis=1) / present.sum(axis=1).replace(0, np.nan)
    breadth = breadth.fillna(0.5)
    btc_col = "BTCUSDT" if "BTCUSDT" in C.columns else C.columns[0]
    btc_up = (C[btc_col] > sma200[btc_col]).fillna(False)
    btc_vol = vol20[btc_col]
    return {"sma50": sma50, "sma200": sma200, "breadth": breadth,
            "btc_up": btc_up, "btc_vol": btc_vol, "btc_col": btc_col, "R": R}


def exposure_series(cb: dict, vol_hi: float) -> pd.Series:
    """Causal market-exposure scalar per bar [0.20 .. 1.00] -- the circuit-breaker. Ported recipe.
       bear (BTC<SMA200): 0.20 | clean bull (breadth>=.5, lo-vol): 1.00 | mixed (breadth>=.3): 0.70 | weak: 0.40
    """
    breadth = cb["breadth"]; btc_up = cb["btc_up"]; btc_vol = cb["btc_vol"]
    hi_vol = btc_vol.fillna(0.0) >= vol_hi
    idx = breadth.index
    expo = pd.Series(0.20, index=idx)
    up = btc_up.reindex(idx).fillna(False)
    b = breadth.reindex(idx)
    expo[up & (b >= 0.50) & (~hi_vol)] = 1.00
    expo[up & ~((b >= 0.50) & (~hi_vol)) & (b >= 0.30)] = 0.70
    expo[up & ~((b >= 0.50) & (~hi_vol)) & (b < 0.30)] = 0.40
    return expo


# ============================================================
# MOVER SCORE PANEL (cross-sectional z-composite, causal)
# ============================================================
def mover_score_panel(lab: dict, blend: dict) -> pd.DataFrame:
    """Cross-sectional z-score each feature per bar, weighted-sum -> composite. Causal (row d uses <= d)."""
    F = lab["F"]; C = lab["C"]
    comp = None
    for f, w in blend.items():
        df = F[f].reindex(index=C.index, columns=C.columns)
        mu = df.mean(axis=1); sd = df.std(axis=1)
        z = df.sub(mu, axis=0).div(sd + 1e-9, axis=0)
        z = z.where(sd > 1e-9, 0.0)
        comp = (w * z) if comp is None else (comp + w * z)
    valid = C.notna() & (C > 0)
    return comp.where(valid)


# ============================================================
# WEIGHT MATRIX: ungated top-K (or random-K control) scaled by circuit-breaker
# ============================================================
def build_W(lab: dict, comp: pd.DataFrame, expo: pd.Series, K: int,
            warmup: int = 200, random_seed: int | None = None) -> pd.DataFrame:
    """Top-K by mover score (no gate), each EW, total scaled by circuit-breaker exposure.
       random_seed not None -> pick K at RANDOM among valid assets (SAME exposure) = the CONTROL.
       warmup=200 so SMA200 is defined (circuit-breaker is honest from the start)."""
    C = lab["C"]
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    rng = np.random.default_rng(random_seed) if random_seed is not None else None
    cols = list(C.columns)
    e_arr = expo.reindex(C.index).values
    for i in range(warmup, len(C.index)):
        d = C.index[i]
        row = comp.iloc[i]
        valid = [s for s in cols if pd.notna(row[s])]
        if len(valid) < K:
            continue
        if rng is not None:
            picks = list(rng.choice(valid, size=K, replace=False))
        else:
            picks = sorted(valid, key=lambda s: -row[s])[:K]
        e = float(e_arr[i])
        w = e / len(picks)
        for s in picks:
            W.loc[d, s] = w
    return W


# ============================================================
# CANONICAL BOOK RETURN (positions lagged 1 bar, taker cost on |dpos|)
# ============================================================
def book_daily_returns(W: pd.DataFrame, R: pd.DataFrame) -> pd.Series:
    Ral = R.reindex(index=W.index, columns=W.columns).fillna(0.0)
    pos = W.shift(1).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs()).sum(axis=1)
    return (pos * Ral).sum(axis=1) - turn * (COST / 2.0)


def bh_ew_returns(C: pd.DataFrame, R: pd.DataFrame) -> pd.Series:
    present = C.notna().astype(float)
    n = present.sum(axis=1).replace(0, np.nan)
    W = present.div(n, axis=0).fillna(0.0)
    return book_daily_returns(W, R)


# ============================================================
# REGIME LABELS (causal, daily) for stratification: bull / chop / bear
# ============================================================
def regime_labels(cb: dict) -> pd.Series:
    """bull = BTC>SMA200 & breadth>=0.5 ; bear = BTC<SMA200 ; chop = the rest (BTC up but weak breadth)."""
    btc_up = cb["btc_up"]; breadth = cb["breadth"]
    idx = breadth.index
    up = btc_up.reindex(idx).fillna(False)
    b = breadth.reindex(idx)
    lab = pd.Series("chop", index=idx)
    lab[~up] = "bear"
    lab[up & (b >= 0.50)] = "bull"
    return lab


# ============================================================
# RANDOM 7-CONSEC-DAY SLICE EVALUATOR (canonical), with regime tag at slice start
# ============================================================
def slice_returns(bret: pd.Series, idx_dev: pd.DatetimeIndex, starts: np.ndarray, slice_days: int) -> np.ndarray:
    out = np.empty(len(starts))
    for j, si in enumerate(starts):
        sl = idx_dev[si: si + slice_days]
        out[j] = (1 + bret.loc[sl]).prod() - 1
    return out


def sample_starts(n_avail: int, n_slices: int, slice_days: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_avail - slice_days, size=n_slices)


# ============================================================
# AGGREGATE capture: sum(realized non-overlapping blocks) / sum(oracle best 7d)
# ============================================================
def aggregate_capture(bret: pd.Series, C: pd.DataFrame, idx_dev: pd.DatetimeIndex,
                      horizon: int = 7) -> dict:
    fwd = (C.shift(-horizon) / C - 1)
    sum_real = 0.0; sum_oracle = 0.0; n = 0; ratios = []
    pos_list = [int(C.index.get_loc(d)) for d in idx_dev]
    for k in range(0, len(idx_dev) - horizon, horizon):
        sl = idx_dev[k: k + horizon]
        if len(sl) < horizon:
            break
        eng = float((1 + bret.loc[sl]).prod() - 1)
        d0 = idx_dev[k]
        oracle = float(fwd.loc[d0].max())
        if oracle > 0.005:
            sum_real += eng; sum_oracle += oracle; n += 1
            ratios.append(eng / oracle)
    if n == 0:
        return {"aggregate_pct": None, "ratio_mean_pct": None, "n": 0}
    return {"aggregate_pct": round(100 * sum_real / sum_oracle, 1),
            "ratio_mean_pct": round(100 * float(np.mean(ratios)), 1),
            "n": n}


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    N_SLICES = 500
    SLICE_DAYS = 7
    REAL_SEEDS = [11, 23, 42]           # slice-sampler seeds for the REAL engine (answer-frequency)
    CTRL_SEEDS = list(range(101, 121))  # 20 random-pick seeds for the SHUFFLE control
    DEV_OOS_START = "2020-09-01"        # eval region start (after 200d warmup); DEV-walled, NOT OOS
    K_VALS = [3, 5]
    BLENDS = {
        "mom14_brk14_volexp_EW": {"mom14": 1/3, "brk14": 1/3, "volexp": 1/3},
        "mom14_heavy":           {"mom14": 0.50, "brk14": 0.30, "volexp": 0.20},
        "plus_accel":            {"mom14": 0.35, "brk14": 0.25, "volexp": 0.20, "accel": 0.20},
    }

    print("=" * 80)
    print("FLEET-LAB DEV MOVER-SELECTION -- same-exposure shuffle control + regime stratification")
    print(f"DEV WALL <= {DEV_END} | eval >= {DEV_OOS_START} | n_slices={N_SLICES} 7-consec-day")
    print(f"REAL seeds={REAL_SEEDS} | CTRL random-pick seeds={CTRL_SEEDS} ({len(CTRL_SEEDS)})")
    print("=" * 80)

    lab = fl.load_wide(n=50)
    C = lab["C"]; R = lab["R"]
    assert C.index.max() < pd.Timestamp(DEV_END), "WALL VIOLATION"
    print(f"loaded {len(lab['syms'])} assets; {C.index.min().date()} -> {C.index.max().date()}")

    cb = circuit_breaker_inputs(C)
    # causal vol_hi threshold from the EARLY DEV window only (first 60% of DEV) -- no peek at eval region
    cut = C.index[int(len(C.index) * 0.6)]
    vol_hi = float(cb["btc_vol"][C.index < cut].dropna().quantile(0.80))
    expo = exposure_series(cb, vol_hi)
    regimes = regime_labels(cb)

    # eval index (DEV-walled region after warmup)
    idx_dev = C.index[(C.index >= pd.Timestamp(DEV_OOS_START)) & (C.index < pd.Timestamp(DEV_END))]
    n_avail = len(idx_dev)
    print(f"vol_hi(early-DEV q80)={vol_hi:.4f} | eval bars={n_avail} | "
          f"avg exposure={float(expo.reindex(idx_dev).mean()):.3f}")
    rc = regimes.reindex(idx_dev).value_counts()
    print(f"regime mix (eval bars): {dict(rc)}")

    bh_b = bh_ew_returns(C, R)

    # pre-sample slice starts per REAL seed (SAME starts used for real, control, BH -- paired)
    starts_by_seed = {s: sample_starts(n_avail, N_SLICES, SLICE_DAYS, s) for s in REAL_SEEDS}
    # regime of each slice = regime at its start bar
    reg_at_start = regimes.reindex(idx_dev).values

    results = {"dev_end": DEV_END, "eval_start": DEV_OOS_START, "n_slices": N_SLICES,
               "vol_hi": round(vol_hi, 4), "real_seeds": REAL_SEEDS, "ctrl_seeds": CTRL_SEEDS,
               "regime_mix": {k: int(v) for k, v in rc.items()}, "blends": {}}

    for bname, blend in BLENDS.items():
        comp = mover_score_panel(lab, blend)
        results["blends"][bname] = {}
        print(f"\n{'#'*80}\nBLEND: {bname}  {blend}\n{'#'*80}")
        for K in K_VALS:
            # REAL engine
            W_real = build_W(lab, comp, expo, K=K, random_seed=None)
            b_real = book_daily_returns(W_real, R)
            cap_real = aggregate_capture(b_real, C, idx_dev)

            # CONTROL: 20 random-pick seeds, SAME exposure
            ctrl_bs = []
            for cs in CTRL_SEEDS:
                Wc = build_W(lab, comp, expo, K=K, random_seed=cs)
                ctrl_bs.append(book_daily_returns(Wc, R))

            # exposure sanity: real vs control share the IDENTICAL daily circuit-breaker exposure
            expo_real = float(W_real.sum(axis=1).reindex(idx_dev).mean())

            # ---- slice eval across REAL seeds ----
            real_mean_seeds, real_pos_seeds = [], []
            ctrl_mean_per_seed_pick = []   # mean across slices for each (real_seed, ctrl_pick)
            # collect per-slice arrays for z-test (pooled over real seeds)
            real_slice_all = []
            ctrl_slice_all = []  # shape: list over ctrl picks of pooled slice arrays
            # regime-stratified accumulators
            reg_real = {r: [] for r in ["bull", "chop", "bear"]}
            reg_ctrl = {r: [] for r in ["bull", "chop", "bear"]}

            for s in REAL_SEEDS:
                starts = starts_by_seed[s]
                rr = slice_returns(b_real, idx_dev, starts, SLICE_DAYS)
                real_mean_seeds.append(float(rr.mean()))
                real_pos_seeds.append(float((rr > 0).mean()))
                real_slice_all.append(rr)
                # regime tag per slice
                rtags = reg_at_start[starts]
                for r in reg_real:
                    reg_real[r].append(rr[rtags == r])
                # control: average the random picks per slice (the same-exposure random book)
                cc_stack = np.vstack([slice_returns(bc, idx_dev, starts, SLICE_DAYS) for bc in ctrl_bs])
                # per ctrl-pick mean (for the null distribution of means)
                ctrl_mean_per_seed_pick.append(cc_stack.mean(axis=1))   # mean over slices, per pick
                # control representative = mean across picks per slice (the expected random book)
                cc_mean_over_picks = cc_stack.mean(axis=0)
                ctrl_slice_all.append(cc_mean_over_picks)
                for r in reg_ctrl:
                    reg_ctrl[r].append(cc_mean_over_picks[rtags == r])

            real_pooled = np.concatenate(real_slice_all)
            ctrl_pooled = np.concatenate(ctrl_slice_all)

            # ---- NULL DISTRIBUTION of control means (across the 20 random picks, pooled over real seeds) ----
            # For each ctrl pick, compute its overall mean-7d across all slices (pooled real seeds)
            ctrl_pick_means = np.vstack(ctrl_mean_per_seed_pick).mean(axis=0)  # avg over real seeds -> per pick
            null_mu = float(ctrl_pick_means.mean())
            null_sd = float(ctrl_pick_means.std(ddof=1))
            real_mu = float(np.mean(real_mean_seeds))
            # z of real mean vs the null distribution of random-pick means
            z_vs_null = (real_mu - null_mu) / (null_sd + 1e-12)

            # also a paired per-slice test: real - control_expected, pooled
            diff = real_pooled - ctrl_pooled
            d_mu = float(diff.mean()); d_sd = float(diff.std(ddof=1))
            n_eff = len(diff)
            t_paired = d_mu / (d_sd / np.sqrt(n_eff) + 1e-12)

            # ---- captures ----
            ctrl_caps = [aggregate_capture(bc, C, idx_dev)["aggregate_pct"] for bc in ctrl_bs]
            ctrl_cap_mu = float(np.nanmean([x for x in ctrl_caps if x is not None]))

            # ---- regime-stratified selection alpha + z ----
            reg_out = {}
            for r in ["bull", "chop", "bear"]:
                rv = np.concatenate(reg_real[r]) if reg_real[r] else np.array([])
                cv = np.concatenate(reg_ctrl[r]) if reg_ctrl[r] else np.array([])
                if len(rv) < 10:
                    reg_out[r] = {"n": int(len(rv)), "note": "insufficient"}
                    continue
                dd = rv - cv
                dz = float(dd.mean()) / (float(dd.std(ddof=1)) / np.sqrt(len(dd)) + 1e-12)
                reg_out[r] = {
                    "n_slices": int(len(rv)),
                    "real_mean_pct": round(100 * float(rv.mean()), 3),
                    "ctrl_mean_pct": round(100 * float(cv.mean()), 3),
                    "alpha_pp": round(100 * float((rv - cv).mean()), 3),
                    "t_paired": round(dz, 2),
                    "real_pos_rate": round(100 * float((rv > 0).mean()), 1),
                }

            res = {
                "K": K,
                "expo_real_mean": round(expo_real, 3),
                "real_mean7d_pct": round(100 * real_mu, 3),
                "real_mean_seeds_pct": [round(100 * x, 3) for x in real_mean_seeds],
                "real_pos_rate_pct": round(100 * float(np.mean(real_pos_seeds)), 1),
                "ctrl_null_mean_pct": round(100 * null_mu, 3),
                "ctrl_null_sd_pct": round(100 * null_sd, 4),
                "z_real_vs_ctrl_null": round(z_vs_null, 2),
                "selection_alpha_pp": round(100 * (real_mu - null_mu), 3),
                "t_paired_pooled": round(t_paired, 2),
                "n_eff_pooled": int(n_eff),
                "capture_real_agg_pct": cap_real["aggregate_pct"],
                "capture_ctrl_agg_pct": round(ctrl_cap_mu, 1),
                "capture_alpha_pp": round((cap_real["aggregate_pct"] or 0) - ctrl_cap_mu, 1),
                "bh_note": "paired same-slice BH available",
                "regime_stratified": reg_out,
            }
            results["blends"][bname][f"K{K}"] = res

            print(f"\n  [K={K}] real mean7d={res['real_mean7d_pct']}% (seeds {res['real_mean_seeds_pct']}) "
                  f"pos={res['real_pos_rate_pct']}% expo={res['expo_real_mean']}")
            print(f"    CONTROL null: mean={res['ctrl_null_mean_pct']}% sd={res['ctrl_null_sd_pct']}% "
                  f"(over {len(CTRL_SEEDS)} random-pick seeds)")
            print(f"    --> z(real vs random-null) = {res['z_real_vs_ctrl_null']}  "
                  f"selection_alpha = {res['selection_alpha_pp']}pp  t_paired={res['t_paired_pooled']} (n={res['n_eff_pooled']})")
            print(f"    capture: real_agg={res['capture_real_agg_pct']}%  ctrl_agg={res['capture_ctrl_agg_pct']}%  "
                  f"alpha={res['capture_alpha_pp']}pp")
            print(f"    REGIME-STRATIFIED selection alpha (real - random, same exposure):")
            for r in ["bull", "chop", "bear"]:
                ro = reg_out[r]
                if "note" in ro:
                    print(f"      {r:5s}: n={ro['n']} (insufficient)")
                else:
                    print(f"      {r:5s}: n={ro['n_slices']:4d}  alpha={ro['alpha_pp']:+.3f}pp  t={ro['t_paired']:+.2f}  "
                          f"real={ro['real_mean_pct']:+.3f}% ctrl={ro['ctrl_mean_pct']:+.3f}% real_pos={ro['real_pos_rate']}%")

    results["runtime_s"] = round(time.time() - t0, 1)
    outp = ROOT.parent / "runs" / "strat" / f"fleet_mover_dev_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n{'='*80}\nSaved: {outp}  ({results['runtime_s']}s)")
    return results


if __name__ == "__main__":
    main()
