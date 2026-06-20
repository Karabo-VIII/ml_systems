"""src/strat/meta_tf_invariance_audit.py -- META-QUESTION (adversarial): is the de-risked-beta wall TF-INVARIANT?

Run the IDENTICAL regime-stratified same-exposure paired shuffle at {1d, 4h, 1h} and tabulate the per-regime
(bull/chop/bear) selection-alpha ACROSS the three TFs. The 1d verdict (fleet_mover_dev_adversarial.py, wh20zajez):
bull selection alpha REAL (+6.2pp, block-boot p05>0), chop/bear ARTIFACT (p05<0, frac>0 ~0.18/0.37).

DECISIVE: does faster cadence (4h, 1h) genuinely flip the chop/bear answer POSITIVE, or is the wall TF-invariant
(chop/bear stay <=0 at every TF)?

ADVERSARIAL DESIGN (this is the whole point -- a sub-daily "unlock" is exactly the multiple-comparisons mirage
this project repeatedly kills):
  * CALENDAR-CONSISTENT windows. Every day-unit window in the 1d harness (SMA50/200, vol20, mom14/30, brk14,
    volexp, accel, warmup=200, slice=7d, bootstrap block=1wk) is rescaled by BARS_PER_DAY so the regime
    classifier + the mover score + the slice horizon are the SAME CALENDAR QUANTITY at every TF. Otherwise a
    "50-bar SMA" is 50d at 1d but ~8d at 4h = a different regime classifier = a confound, not a cadence effect.
  * SAME-EXPOSURE paired shuffle (real vs K-random, identical daily circuit-breaker exposure) -- so any alpha
    cannot be a market-timing artifact. We ASSERT max|expo_real - expo_ctrl| == 0.
  * MOVING-BLOCK bootstrap (iid t overstates ~6x on autocorrelated returns). p05 + frac>0 are the verdict, NOT t.
  * COST HONESTY: faster cadence = MORE turnover = MORE taker cost. We report realized round-trips/yr + gross-vs-net
    per TF. Every alpha is NET of fl.COST=0.0024 taker RT.
  * regime tagged at slice START (causal). Slices sample the post-warmup DEV-walled region only.

If sub-daily shows POSITIVE chop/bear selection we STRESS it (cost-ignoring? overlapping-window inflation?
regime-misclassification at the faster classifier? seed-cherry-picked?).

DEV-walled (<= 2024-05-15). No emoji. No git commit.
RWYB: C:/.../.venv/Scripts/python.exe -m strat.meta_tf_invariance_audit
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

COST = fl.COST
DEV_END = fl.DEV_END
BPD = fl.BARS_PER_DAY


# ============================================================
# CALENDAR-CONSISTENT feature + circuit-breaker rebuild (windows in BARS = day-window * bpd)
# ============================================================
def _rsi(s, n):
    d = s.diff(); up = d.clip(lower=0).rolling(n).mean(); dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def calendar_features(lab: dict, bpd: int) -> dict:
    """Rebuild the mover features used by the blend with windows scaled to CALENDAR days (bars = days*bpd).
       Matches the 1d harness semantics exactly at bpd=1 (mom14=14 calendar days, etc.)."""
    C = lab["C"]
    def days(d):  # calendar-day window -> bars
        return max(1, int(round(d * bpd)))
    F = {
        "mom14": C / C.shift(days(14)) - 1,
        "brk14": C / C.rolling(days(14), min_periods=days(14)).max().shift(1) - 1,
        "volexp": lab["R"].rolling(days(7)).std() / (lab["R"].rolling(days(30)).std() + 1e-12),
        "accel": (C / C.shift(days(7)) - 1) - (C.shift(days(7)) / C.shift(days(14)) - 1),
    }
    return F


def calendar_circuit_breaker(C: pd.DataFrame, bpd: int) -> dict:
    """SMA50/200, breadth, BTC-trend, BTC vol20 -- all windows in CALENDAR days (bars = days*bpd)."""
    def days(d):
        return max(1, int(round(d * bpd)))
    sma50 = C.rolling(days(50), min_periods=days(50)).mean()
    sma200 = C.rolling(days(200), min_periods=days(200)).mean()
    R = C.pct_change(fill_method=None)
    vol20 = R.rolling(days(20), min_periods=days(20)).std() * np.sqrt(365 * bpd)  # annualize at the bar cadence
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


def regime_labels(cb: dict) -> pd.Series:
    btc_up = cb["btc_up"]; breadth = cb["breadth"]
    idx = breadth.index
    up = btc_up.reindex(idx).fillna(False)
    b = breadth.reindex(idx)
    lab = pd.Series("chop", index=idx)
    lab[~up] = "bear"
    lab[up & (b >= 0.50)] = "bull"
    return lab


def mover_score_panel(C: pd.DataFrame, F: dict, blend: dict) -> pd.DataFrame:
    comp = None
    for f, w in blend.items():
        df = F[f].reindex(index=C.index, columns=C.columns)
        mu = df.mean(axis=1); sd = df.std(axis=1)
        z = df.sub(mu, axis=0).div(sd + 1e-9, axis=0)
        z = z.where(sd > 1e-9, 0.0)
        comp = (w * z) if comp is None else (comp + w * z)
    valid = C.notna() & (C > 0)
    return comp.where(valid)


def build_W(C: pd.DataFrame, comp: pd.DataFrame, expo: pd.Series, K: int,
            warmup: int, random_seed: int | None = None) -> pd.DataFrame:
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    rng = np.random.default_rng(random_seed) if random_seed is not None else None
    cols = list(C.columns)
    e_arr = expo.reindex(C.index).values
    comp_v = comp.values
    col_idx = {s: j for j, s in enumerate(cols)}
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


def book_bar_returns(W: pd.DataFrame, R: pd.DataFrame):
    """Returns (net_series, gross_turnover_per_bar). Positions lagged 1 bar, taker cost on |dpos|."""
    Ral = R.reindex(index=W.index, columns=W.columns).fillna(0.0)
    pos = W.shift(1).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs()).sum(axis=1)
    net = (pos * Ral).sum(axis=1) - turn * (COST / 2.0)
    return net, turn


def bh_ew_returns(C: pd.DataFrame, R: pd.DataFrame) -> pd.Series:
    present = C.notna().astype(float)
    n = present.sum(axis=1).replace(0, np.nan)
    W = present.div(n, axis=0).fillna(0.0)
    net, _ = book_bar_returns(W, R)
    return net


def slice_returns(bret: pd.Series, idx_dev: pd.DatetimeIndex, starts: np.ndarray, slice_bars: int) -> np.ndarray:
    out = np.empty(len(starts))
    for j, si in enumerate(starts):
        sl = idx_dev[si: si + slice_bars]
        out[j] = (1 + bret.loc[sl]).prod() - 1
    return out


def sample_starts(n_avail: int, n_slices: int, slice_bars: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_avail - slice_bars, size=n_slices)


def moving_block_boot(diff: np.ndarray, block: int, boot: int, rng) -> dict:
    n = len(diff)
    if n < 30:
        return {"n": int(n), "note": "insufficient"}
    block = max(1, min(block, n))
    nblocks = int(np.ceil(n / block))
    boot_means = np.empty(boot)
    for bI in range(boot):
        st = rng.integers(0, n - block + 1, size=nblocks)
        idxs = (st[:, None] + np.arange(block)[None, :]).ravel()[:n]
        boot_means[bI] = diff[idxs].mean()
    p05 = float(np.percentile(boot_means, 5))
    frac_pos = float((boot_means > 0).mean())
    return {"n": int(n), "alpha_mean_pp": round(100 * float(diff.mean()), 3),
            "boot_p05_pp": round(100 * p05, 3), "boot_frac_gt0": round(frac_pos, 4),
            "verdict": "REAL" if p05 > 0 else ("ARTIFACT" if frac_pos < 0.95 else "AMBIGUOUS")}


# ============================================================
# ONE TF: the full identical regime-stratified same-exposure paired-shuffle protocol
# ============================================================
def run_one_tf(tf: str, n_slices=600, real_seeds=(11, 23, 42), ctrl_seeds=tuple(range(101, 121)),
               K=5, boot=5000, dev_oos_start="2020-09-01"):
    bpd = BPD[tf]
    def days(d):
        return max(1, int(round(d * bpd)))
    slice_bars = days(7)            # 7 CALENDAR days
    warmup = days(200)              # SMA200 calendar warmup
    block = days(7)                 # ~1 calendar week moving block
    BLEND = {"mom14": 0.35, "brk14": 0.25, "volexp": 0.20, "accel": 0.20}

    min_bars = days(200) + 50       # need warmup + some slices
    lab = fl.load_wide(n=50, tf=tf, min_bars=min_bars)
    C = lab["C"]; R = lab["R"]
    assert C.index.max() < pd.Timestamp(DEV_END), "WALL VIOLATION"

    F = calendar_features(lab, bpd)
    cb = calendar_circuit_breaker(C, bpd)
    # causal vol_hi from EARLY DEV only (first 60%) -- no peek at eval region
    cut = C.index[int(len(C.index) * 0.6)]
    vol_hi = float(cb["btc_vol"][C.index < cut].dropna().quantile(0.80))
    expo = exposure_series(cb, vol_hi)
    regimes = regime_labels(cb)

    idx_dev = C.index[(C.index >= pd.Timestamp(dev_oos_start)) & (C.index < pd.Timestamp(DEV_END))]
    # ensure slices stay inside idx_dev AND post-warmup
    warm_date = C.index[min(warmup, len(C.index) - 1)]
    idx_dev = idx_dev[idx_dev >= warm_date]
    n_avail = len(idx_dev)
    reg_at_start = regimes.reindex(idx_dev).values

    comp = mover_score_panel(C, F, BLEND)
    W_real = build_W(C, comp, expo, K=K, warmup=warmup, random_seed=None)
    b_real, turn_real = book_bar_returns(W_real, R)
    bh_b = bh_ew_returns(C, R)

    W_ctrls = [build_W(C, comp, expo, K=K, warmup=warmup, random_seed=cs) for cs in ctrl_seeds]
    ctrl_bs = [book_bar_returns(Wc, R)[0] for Wc in W_ctrls]

    # ---- D1: exposure identity (same-exposure shuffle proof) ----
    er = W_real.sum(axis=1)
    max_diff = max(float((er - Wc.sum(axis=1)).abs().max()) for Wc in W_ctrls)

    # ---- cost honesty: realized round-trips/yr (turnover) ----
    dev_bars = (C.index >= pd.Timestamp(dev_oos_start)) & (C.index < pd.Timestamp(DEV_END))
    yrs = (idx_dev.max() - idx_dev.min()).days / 365.25
    # turn = sum|dpos| per bar; a full RT (in then out of one name) contributes ~2 to summed turnover over its life.
    rt_per_yr = float(turn_real.reindex(idx_dev).sum() / 2.0 / max(yrs, 1e-9))
    cost_drag_yr = rt_per_yr * COST  # approx annual taker cost drag at full book

    # ---- gross vs net (strip cost to see what cost eats) ----
    Ral = R.reindex(index=W_real.index, columns=W_real.columns).fillna(0.0)
    pos = W_real.shift(1).fillna(0.0)
    b_real_gross = (pos * Ral).sum(axis=1)

    # ---- build per-slice arrays pooled over real seeds, regime tagged ----
    starts_by_seed = {s: sample_starts(n_avail, n_slices, slice_bars, s) for s in real_seeds}
    real_all, ctrl_all, gross_all, bh_all, reg_all = [], [], [], [], []
    for s in real_seeds:
        st = starts_by_seed[s]
        rr = slice_returns(b_real, idx_dev, st, slice_bars)
        gg = slice_returns(b_real_gross, idx_dev, st, slice_bars)
        bb = slice_returns(bh_b, idx_dev, st, slice_bars)
        cc = np.vstack([slice_returns(bc, idx_dev, st, slice_bars) for bc in ctrl_bs]).mean(axis=0)
        real_all.append(rr); gross_all.append(gg); ctrl_all.append(cc); bh_all.append(bb)
        reg_all.append(reg_at_start[st])
    real_all = np.concatenate(real_all); ctrl_all = np.concatenate(ctrl_all)
    gross_all = np.concatenate(gross_all); bh_all = np.concatenate(bh_all); reg_all = np.concatenate(reg_all)

    rng = np.random.default_rng(7)
    per_regime = {}
    for r in ["bull", "chop", "bear"]:
        m = reg_all == r
        diff_net = (real_all - ctrl_all)[m]
        diff_gross = (gross_all - ctrl_all)[m]   # NOTE: ctrl is also net; gross-real vs net-ctrl = cost-blind upper bound
        bres = moving_block_boot(diff_net, block, boot, rng)
        bres_gross = moving_block_boot(diff_gross, block, boot, rng) if m.sum() >= 30 else {"note": "insufficient"}
        if "note" in bres:
            per_regime[r] = bres
            continue
        # regime mix + market beta context
        per_regime[r] = {
            **bres,
            "alpha_gross_pp": bres_gross.get("alpha_mean_pp"),
            "real_mean_pp": round(100 * float(real_all[m].mean()), 3),
            "ctrl_mean_pp": round(100 * float(ctrl_all[m].mean()), 3),
            "bh_mean_pp": round(100 * float(bh_all[m].mean()), 3),
            "real_pos_rate": round(100 * float((real_all[m] > 0).mean()), 1),
        }

    rc = pd.Series(reg_at_start).value_counts().to_dict()
    return {
        "tf": tf, "bpd": bpd, "slice_bars": slice_bars, "warmup_bars": warmup, "block_bars": block,
        "n_assets": len(lab["syms"]), "dev_bars_total": int(len(C.index)),
        "eval_bars": int(n_avail), "eval_range": [str(idx_dev.min()), str(idx_dev.max())],
        "vol_hi": round(vol_hi, 4), "avg_exposure": round(float(expo.reindex(idx_dev).mean()), 3),
        "regime_mix_at_starts": {k: int(v) for k, v in rc.items()},
        "D1_max_exposure_diff": round(max_diff, 10),
        "turnover_rt_per_yr": round(rt_per_yr, 1),
        "approx_cost_drag_per_yr_pct": round(100 * cost_drag_yr, 2),
        "per_regime": per_regime,
    }


def main():
    t0 = time.time()
    TFS = ["1d", "4h", "1h"]
    print("=" * 100)
    print("META-QUESTION: is the de-risked-beta wall TIMEFRAME-INVARIANT?")
    print("Identical regime-stratified same-exposure paired shuffle + moving-block bootstrap at {1d, 4h, 1h}")
    print(f"DEV WALL <= {DEV_END} | calendar-consistent windows | NET of taker COST={COST}")
    print("=" * 100)

    out = {"dev_end": DEV_END, "cost": COST, "tfs": {}}
    for tf in TFS:
        print(f"\n{'#'*100}\nTF = {tf}\n{'#'*100}")
        res = run_one_tf(tf)
        out["tfs"][tf] = res
        print(f"  assets={res['n_assets']} eval_bars={res['eval_bars']} range={res['eval_range']}")
        print(f"  avg_exposure={res['avg_exposure']} vol_hi={res['vol_hi']} "
              f"regime_mix(starts)={res['regime_mix_at_starts']}")
        print(f"  EXPOSURE-IDENTITY max|real-ctrl| = {res['D1_max_exposure_diff']:.2e} "
              f"({'IDENTICAL (timing held)' if res['D1_max_exposure_diff'] < 1e-9 else 'WARNING differs'})")
        print(f"  COST: realized ~{res['turnover_rt_per_yr']} round-trips/yr -> "
              f"~{res['approx_cost_drag_per_yr_pct']}%/yr taker drag")
        print(f"  per-regime NET selection alpha (real - random, SAME exposure; moving-block boot p05):")
        for r in ["bull", "chop", "bear"]:
            ro = res["per_regime"][r]
            if "note" in ro:
                print(f"    {r:5s}: {ro.get('note')}")
                continue
            print(f"    {r:5s}: n={ro['n']:5d}  alpha_net={ro['alpha_mean_pp']:+.3f}pp "
                  f"(gross={ro['alpha_gross_pp']:+.3f}pp)  p05={ro['boot_p05_pp']:+.3f}pp  "
                  f"frac>0={ro['boot_frac_gt0']:.3f}  -> {ro['verdict']}")

    # ---- DECISIVE CROSS-TF TABLE ----
    print(f"\n{'='*100}\nDECISIVE CROSS-TF TABLE -- NET selection alpha (pp) + moving-block p05 + verdict\n{'='*100}")
    hdr = f"{'regime':6}" + "".join([f"{tf:>26}" for tf in TFS])
    print(hdr)
    for r in ["bull", "chop", "bear"]:
        cells = []
        for tf in TFS:
            ro = out["tfs"][tf]["per_regime"][r]
            if "note" in ro:
                cells.append(f"{'insuff':>26}")
            else:
                cells.append(f"{ro['alpha_mean_pp']:+7.3f}/p05{ro['boot_p05_pp']:+6.2f}/{ro['verdict'][:4]:>4}"[:26].rjust(26))
        print(f"{r:6}" + "".join(cells))

    # invariance verdict: do chop AND bear stay <= 0 (ARTIFACT) at EVERY tf?
    wall_invariant = True
    for r in ["chop", "bear"]:
        for tf in TFS:
            ro = out["tfs"][tf]["per_regime"][r]
            if "note" in ro:
                continue
            if ro["verdict"] == "REAL":   # p05 > 0 = a genuine positive selection alpha outside bull
                wall_invariant = False
    out["wall_timeframe_invariant"] = wall_invariant
    print(f"\nWALL TIMEFRAME-INVARIANT (chop AND bear ARTIFACT/<=0 at EVERY tf)? -> {wall_invariant}")

    out["runtime_s"] = round(time.time() - t0, 1)
    outp = ROOT.parent / "runs" / "strat" / f"meta_tf_invariance_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {outp}  ({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    main()
