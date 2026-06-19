"""src/strat/mover_regime_falsifier.py -- THE FALSIFIER (decisive, DEV-walled).

ORC DECISIVE CYCLE. LANE = adversarial falsifier of the mover-SELECTION-alpha lead.

THE LEAD (NL-C4, proven on the OLD OOS harness, must re-prove on DEV):
  an UNGATED mover engine (rank assets by a mover composite, hold top-K, MARKET-level
  circuit-breaker scaling TOTAL exposure) BEAT random-same-exposure by z=5-6 (p~0) on the
  same-exposure SHUFFLE control = cross-sectional SELECTION ALPHA (not timing).

THE DECISIVE QUESTION (this script answers, on DEV <= 2024-05-15 ONLY):
  Does mover-SELECTION beat random-SAME-EXPOSURE on DEV, AND does it SURVIVE regime
  stratification (bull / chop / bear)? If the alpha vanishes in non-bull regimes -> it is a
  BULL-BETA artifact (in a rising market the biggest movers go up, so picking them beats
  random WITHOUT skill). If it holds across regimes -> REAL cross-sectional selection alpha.

METHOD (all causal, leak-free, ported to fleet_lab DEV; does NOT import referee_harness/OOS):
  1. Mover composite per bar: 0.35 z(mom14) + 0.25 z(rangepos) + 0.20 z(volexp) + 0.20 z(accel).
     (Same recipe as mover_capture_engine; fleet_lab features are all shift-able causal panels.)
  2. Market circuit-breaker (causal): exposure in {0.20,0.40,0.70,1.00} from BTC>SMA200 x breadth(>SMA50) x BTC-vol.
  3. REGIME label per decision-bar di (causal, uses only <= di):
       bull  = BTC>SMA200 AND breadth>=0.50 AND trailing-30d universe-median return > 0
       bear  = BTC<SMA200 AND breadth<0.40
       chop  = everything else
  4. SAME-EXPOSURE SHUFFLE CONTROL: at each di hold the SAME daily total exposure but pick K
     assets at RANDOM (multiple seeds) instead of by mover score. PER-SLICE PAIRED EXCESS =
     real_roi - mean_random_roi. Stratify the excess BY REGIME -> z-score + mean-excess per regime.

VERDICT: selection alpha survives in chop/bear (real skill) vs bull-ONLY (beta artifact).

RWYB: python -m strat.mover_regime_falsifier
No emoji (cp1252). DEV-walled. Does NOT git commit. Does NOT touch OOS/UNSEEN.
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

COST = fl.COST            # taker RT, 0.0024
HOLD = 7
WARMUP = 210              # need 200 bars for SMA200 + slack
N_CTRL_SEEDS = 20         # random-pick control seeds (paired against the real engine each slice)


# ============================================================
# CAUSAL PANELS (re-derived from fleet_lab close; all <= decision bar)
# ============================================================
def build_panels(lab: dict) -> dict:
    """All causal indicator panels on the DEV close grid. shift where a panel must exclude bar di."""
    C = lab["C"]; F = lab["F"]; R = lab["R"]
    # SMA200 / SMA50 use rolling means INCLUDING bar di (close at di is known at decision time di) -- causal.
    sma200 = C.rolling(200, min_periods=200).mean()
    sma50 = C.rolling(50, min_periods=50).mean()
    vol20 = R.rolling(20, min_periods=20).std()
    # mover composite ingredients (fleet_lab F panels are causal: value at row di known at di)
    mom14 = F["mom14"]; volexp = F["volexp"]; accel = F["accel"]; rangepos = F["rangepos"]
    return {"C": C, "R": R, "sma200": sma200, "sma50": sma50, "vol20": vol20,
            "mom14": mom14, "volexp": volexp, "accel": accel, "rangepos": rangepos,
            "F": F, "syms": lab["syms"]}


def _zrow_at(panel: pd.DataFrame, di: int, valid_cols) -> pd.Series:
    """Cross-sectional z-score of panel.iloc[di] over valid_cols (causal, single row)."""
    row = panel.iloc[di][valid_cols]
    mu = row.mean(); sd = row.std()
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(0.0, index=valid_cols)
    return ((row - mu) / (sd + 1e-12)).fillna(0.0)


def mover_score(pan: dict, di: int):
    """Composite mover z-score at bar di. Returns (Series over valid assets, valid_cols list)."""
    C = pan["C"]
    row_c = C.iloc[di]
    # valid = price present and all 4 ingredients present
    valid = []
    for s in C.columns:
        if (pd.notna(row_c[s]) and row_c[s] > 0
                and pd.notna(pan["mom14"].iloc[di][s]) and pd.notna(pan["accel"].iloc[di][s])
                and pd.notna(pan["volexp"].iloc[di][s]) and pd.notna(pan["rangepos"].iloc[di][s])):
            valid.append(s)
    if len(valid) < 8:
        return None, valid
    z_mom = _zrow_at(pan["mom14"], di, valid)
    z_brk = _zrow_at(pan["rangepos"], di, valid)
    z_vex = _zrow_at(pan["volexp"], di, valid)
    z_acc = _zrow_at(pan["accel"], di, valid)
    comp = 0.35 * z_mom + 0.25 * z_brk + 0.20 * z_vex + 0.20 * z_acc
    return comp, valid


# ============================================================
# MARKET CIRCUIT-BREAKER (causal exposure scalar at di)
# ============================================================
def market_exposure(pan: dict, di: int, vol_hi: float) -> float:
    C = pan["C"]; sma200 = pan["sma200"]; sma50 = pan["sma50"]; vol20 = pan["vol20"]
    btc = C.iloc[di].get("BTCUSDT", np.nan)
    s200 = sma200.iloc[di].get("BTCUSDT", np.nan)
    btc_up = pd.notna(s200) and pd.notna(btc) and (btc > s200)
    # breadth above SMA50
    row_c = C.iloc[di]; row_s = sma50.iloc[di]
    pres = row_c.notna() & row_s.notna()
    breadth = float((row_c[pres] > row_s[pres]).mean()) if pres.sum() > 0 else 0.5
    bv = vol20.iloc[di].get("BTCUSDT", np.nan)
    hi_vol = pd.notna(bv) and bv >= vol_hi
    if not btc_up:
        return 0.20
    if breadth >= 0.50 and not hi_vol:
        return 1.00
    if breadth >= 0.30:
        return 0.70
    return 0.40


# ============================================================
# REGIME CLASSIFIER (causal, my lane) -- bull / chop / bear
# ============================================================
def classify_regime(pan: dict, di: int) -> str:
    """Causal regime at decision bar di from BTC-vs-SMA200 x breadth x trailing-30d universe return.
    Uses ONLY data at/<= di. Returns 'bull' / 'chop' / 'bear'."""
    C = pan["C"]; sma200 = pan["sma200"]; sma50 = pan["sma50"]
    btc = C.iloc[di].get("BTCUSDT", np.nan)
    s200 = sma200.iloc[di].get("BTCUSDT", np.nan)
    if pd.isna(btc) or pd.isna(s200):
        return "chop"
    btc_up = btc > s200
    # breadth above SMA50
    row_c = C.iloc[di]; row_s = sma50.iloc[di]
    pres = row_c.notna() & row_s.notna()
    breadth = float((row_c[pres] > row_s[pres]).mean()) if pres.sum() > 0 else 0.5
    # trailing 30d universe-median return (causal: ret over [di-30, di])
    if di < 30:
        uni_ret = 0.0
    else:
        past = C.iloc[di - 30]; now = C.iloc[di]
        rr = []
        for s in C.columns:
            p0 = past.get(s); p1 = now.get(s)
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                rr.append(p1 / p0 - 1)
        uni_ret = float(np.median(rr)) if rr else 0.0
    # classification
    if btc_up and breadth >= 0.50 and uni_ret > 0:
        return "bull"
    if (not btc_up) and breadth < 0.40:
        return "bear"
    return "chop"


# ============================================================
# SLICE ROI: real mover-selection vs random-same-exposure
# ============================================================
def fwd_roi(pan: dict, di: int, picks, exposure: float) -> float:
    """Net forward HOLD-bar ROI of an EW top-K book scaled by `exposure`.
    Long-only spot, taker cost on entry+exit of the deployed fraction. (1-exposure) sits in cash (0 ret)."""
    C = pan["C"]
    if not picks:
        return 0.0
    rets = []
    for s in picks:
        p0 = C.iloc[di].get(s); p1 = C.iloc[di + HOLD].get(s)
        if pd.notna(p0) and pd.notna(p1) and p0 > 0:
            rets.append(p1 / p0 - 1)
    if not rets:
        return 0.0
    gross = float(np.mean(rets))                 # EW across the K held names
    # cost: enter + exit the deployed `exposure` fraction once over the hold window
    net = exposure * gross - exposure * COST
    return net


def run_slice(pan: dict, di: int, K: int, vol_hi: float, ctrl_rng) -> dict | None:
    """One decision bar: real mover top-K vs N_CTRL_SEEDS random-same-exposure books. Causal."""
    C = pan["C"]
    if di + HOLD >= len(C.index):
        return None
    comp, valid = mover_score(pan, di)
    if comp is None or len(valid) < K + 2:
        return None
    exposure = market_exposure(pan, di, vol_hi)
    regime = classify_regime(pan, di)
    # REAL: top-K by mover composite
    real_picks = list(comp.sort_values(ascending=False).index[:K])
    real_roi = fwd_roi(pan, di, real_picks, exposure)
    # CONTROL: same exposure, random K picks among the SAME valid set
    ctrl_rois = []
    for _ in range(N_CTRL_SEEDS):
        rp = list(ctrl_rng.choice(valid, size=K, replace=False))
        ctrl_rois.append(fwd_roi(pan, di, rp, exposure))
    ctrl_mean = float(np.mean(ctrl_rois))
    return {"di": di, "date": str(C.index[di].date()), "regime": regime,
            "exposure": exposure, "real": real_roi, "ctrl_mean": ctrl_mean,
            "excess": real_roi - ctrl_mean}


# ============================================================
# DRIVER
# ============================================================
def main():
    t0 = time.time()
    print("=" * 84)
    print("MOVER-SELECTION FALSIFIER -- regime-stratified same-exposure shuffle (DEV <= %s)" % fl.DEV_END)
    print("Question: does mover SELECTION beat random-SAME-EXPOSURE, and SURVIVE bull/chop/bear?")
    print("=" * 84)

    lab = fl.load_wide(n=50)
    pan = build_panels(lab)
    C = pan["C"]
    print(f"DEV universe: {len(lab['syms'])} assets | {C.index.min().date()} -> {C.index.max().date()} "
          f"({len(C.index)} bars)")
    assert C.index.max() < pd.Timestamp(fl.DEV_END), "WALL VIOLATION"

    # Causal vol_hi threshold: 80th pctile of BTC vol20 over the FIRST HALF of DEV (no peeking forward).
    half = len(C.index) // 2
    btc_vol_early = pan["vol20"]["BTCUSDT"].iloc[:half].dropna()
    vol_hi = float(btc_vol_early.quantile(0.80)) if len(btc_vol_early) else 0.05
    print(f"vol_hi threshold (first-half DEV, 80th pctile BTC vol20): {vol_hi:.4f}\n")

    results = {"dev_end": fl.DEV_END, "n_assets": len(lab["syms"]),
               "date_range": [str(C.index.min().date()), str(C.index.max().date())],
               "vol_hi": round(vol_hi, 5), "hold": HOLD, "cost_rt": COST,
               "n_ctrl_seeds": N_CTRL_SEEDS, "by_K": {}}

    # EVALUATE on EVERY eligible decision bar (full DEV coverage -> max regime sample).
    valid_dis = list(range(WARMUP, len(C.index) - HOLD - 1))
    ctrl_rng = np.random.default_rng(20240515)   # fixed control RNG (reproducible)

    for K in [3, 5]:
        print("-" * 84)
        print(f"[K={K}] real mover-top{K} vs {N_CTRL_SEEDS}x random-same-exposure, per decision bar")
        rows = []
        for di in valid_dis:
            r = run_slice(pan, di, K, vol_hi, ctrl_rng)
            if r is not None:
                rows.append(r)
        df = pd.DataFrame(rows)
        n = len(df)
        print(f"  evaluated {n} decision bars")

        # NON-OVERLAPPING subsample (stride=HOLD) -> autocorrelation-free honest z.
        # The full daily grid has 7d holds => adjacent slices share 6/7 of their forward window,
        # so the pooled paired-z is inflated ~sqrt(HOLD). The honest test uses disjoint windows.
        df_no = df.iloc[::HOLD].reset_index(drop=True)

        # ---- selection-alpha stats (paired real-vs-control excess) ----
        def alpha_stats(sub: pd.DataFrame, sub_no: pd.DataFrame | None = None) -> dict:
            m = len(sub)
            if m < 5:
                return {"n": m, "mean_excess_pp": None, "z": None, "real_mean_pp": None,
                        "ctrl_mean_pp": None, "win_rate_pct": None, "z_honest": None, "n_honest": m}
            ex = sub["excess"].to_numpy()
            mean_ex = float(ex.mean())
            sd = float(ex.std(ddof=1))
            z = mean_ex / (sd / np.sqrt(m)) if sd > 1e-12 else 0.0
            # honest non-overlapping z
            z_h = None; n_h = 0
            if sub_no is not None and len(sub_no) >= 5:
                exh = sub_no["excess"].to_numpy(); n_h = len(exh)
                sdh = float(exh.std(ddof=1))
                z_h = round(float(exh.mean() / (sdh / np.sqrt(n_h))), 2) if sdh > 1e-12 else 0.0
            return {
                "n": m,
                "real_mean_pp": round(100 * float(sub["real"].mean()), 3),
                "ctrl_mean_pp": round(100 * float(sub["ctrl_mean"].mean()), 3),
                "mean_excess_pp": round(100 * mean_ex, 3),
                "z": round(float(z), 2),
                "z_honest": z_h, "n_honest": n_h,
                "win_rate_pct": round(100 * float((ex > 0).mean()), 1),
                "avg_exposure": round(float(sub["exposure"].mean()), 3),
            }

        overall = alpha_stats(df, df_no)
        print(f"  OVERALL (pooled): real={overall['real_mean_pp']}% ctrl={overall['ctrl_mean_pp']}% "
              f"excess={overall['mean_excess_pp']}pp  z={overall['z']} (honest z={overall['z_honest']}, "
              f"n_no={overall['n_honest']})  win={overall['win_rate_pct']}%")

        # ---- STRATIFIED BY REGIME ----
        per_regime = {}
        print(f"  {'regime':8}{'n':>6}{'real%':>9}{'ctrl%':>9}{'excess_pp':>11}{'z':>8}{'z_hon':>8}{'win%':>7}{'expo':>7}")
        for reg in ["bull", "chop", "bear"]:
            sub = df[df["regime"] == reg]
            sub_no = df_no[df_no["regime"] == reg]
            st = alpha_stats(sub, sub_no)
            per_regime[reg] = st
            if st["n"] >= 5:
                print(f"  {reg:8}{st['n']:>6}{st['real_mean_pp']:>9}{st['ctrl_mean_pp']:>9}"
                      f"{st['mean_excess_pp']:>11}{st['z']:>8}{str(st['z_honest']):>8}"
                      f"{st['win_rate_pct']:>7}{st['avg_exposure']:>7}")
            else:
                print(f"  {reg:8}{st['n']:>6}   (insufficient n)")

        # regime date-coverage (to sanity-check the labels are causal/sane)
        reg_counts = df["regime"].value_counts().to_dict()
        results["by_K"][f"K{K}"] = {"overall": overall, "per_regime": per_regime,
                                    "regime_counts": {k: int(v) for k, v in reg_counts.items()}}

    # ============================================================
    # VERDICT
    # ============================================================
    print("\n" + "=" * 84)
    print("VERDICT")
    print("=" * 84)
    verdicts = {}
    for K in [3, 5]:
        pr = results["by_K"][f"K{K}"]["per_regime"]
        # Gate on the HONEST non-overlapping z (z_honest), NOT the autocorrelation-inflated pooled z.
        nonbull = []
        for reg in ["chop", "bear"]:
            st = pr[reg]
            if st["n_honest"] >= 20 and st["z_honest"] is not None:
                nonbull.append((reg, st["z_honest"], st["mean_excess_pp"]))
        bull = pr["bull"]
        survives = [(reg, z, mx) for (reg, z, mx) in nonbull if z is not None and z >= 2.0 and mx > 0]
        bull_real = (bull["z_honest"] is not None and bull["z_honest"] >= 2.0
                     and (bull["mean_excess_pp"] or 0) > 0)
        bull_marg = (bull["z_honest"] is not None and bull["z_honest"] >= 1.5
                     and (bull["mean_excess_pp"] or 0) > 0)
        if survives:
            v = "REAL SELECTION ALPHA -- survives non-bull (honest z): " + ", ".join(
                f"{reg}(z={z}, +{mx}pp)" for reg, z, mx in survives)
        elif bull_real:
            v = "BULL-ONLY ARTIFACT -- alpha in bull (honest z=%s) but absent in chop/bear" % bull["z_honest"]
        elif bull_marg:
            v = ("BULL-ONLY, MARGINAL -- bull only borderline (honest z=%s) once autocorrelation-corrected; "
                 "absent in chop/bear" % bull["z_honest"])
        else:
            v = "NO SELECTION ALPHA on DEV -- not significant in any regime once autocorrelation-corrected"
        verdicts[f"K{K}"] = v
        print(f"  K={K}: {v}")
        line = " | ".join(f"{reg}: z_hon={pr[reg]['z_honest']} (pooled z={pr[reg]['z']}) "
                          f"ex={pr[reg]['mean_excess_pp']}pp n_no={pr[reg]['n_honest']}"
                          for reg in ["bull", "chop", "bear"])
        print(f"        {line}")
    results["verdicts"] = verdicts

    results["runtime_s"] = round(time.time() - t0, 1)
    outp = ROOT.parent / "runs" / "strat" / f"mover_regime_falsifier_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved: {outp}  ({results['runtime_s']}s)")
    return results


if __name__ == "__main__":
    main()
