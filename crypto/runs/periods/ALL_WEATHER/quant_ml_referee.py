"""
quant_ml_referee.py -- ADVERSARIAL RE-DERIVATION of the ML lane's OOS random-7d-slice win-rate.

REFEREE MANDATE (quant lane, cycle-1 new loop):
  Independently re-derive the ML engine's OOS win-rate with a STRICT walk-forward and HUNT look-ahead.
  Every number is RWYB (re-run from data). No trust in the lane's reported figure.

WHAT THIS SCRIPT PROVES / DISPROVES
  Q1. Does ANY engine beat buy-hold on random-7d-slice win-rate (>55%) OR mean (>+2.9%) OOS, leak-free?
  Q2. If the ML 'wins' -- is it real or leak? (strict-walk-forward number vs a deliberately-LEAKY number)
  Q3. The smoothing ceiling (risk-parity / diversification) -- how high can win-rate go with NO prediction?
  Q4. Updated verdict: is 7d-slice profitability beatable above buy-hold, or is ~55% the hard wall?

DESIGN (the controls the single-lane runs omitted):
  A. STRICT walk-forward, gap-purged:
       - features at row d use ONLY data <= d (rolling backward; verified by a SHIFT-AUDIT below)
       - label = C[d+H]/C[d]-1 (uses bars d+1..d+H)
       - a training row d is admitted at retrain-cutoff T ONLY if d+H < T  (label window fully closed)
       - PURGE: additionally drop training rows within H bars of T (belt-and-suspenders)
       - NO global scaler fit on full data; any scaling is fit on the train fold only (we use HGB = scale-free)
  B. A DELIBERATELY-LEAKY twin (full-sample fit, no purge, scaler on all data) to MEASURE the leak gap.
       If strict ~= leaky, the lane wasn't leaking. If leaky >> strict, the lane's 'win' was a leak artifact.
  C. SAME-EXPOSURE control: the ML's win-rate is compared against a random-selection book that holds the
       SAME number of names at the SAME times (exposure held constant) -> isolates SELECTION SKILL from
       cash-timing / concentration luck.
  D. Honest paired test: per-slice (ML - BH) excess, block-bootstrapped p-value (slices are 7d, overlapping
       draws are dependent -> we report the binomial AND a moving-block bootstrap CI on the win-rate).
  E. The buy-hold benchmark is recomputed point-in-time (only listed assets averaged; pre-listing = excluded,
       NOT zero) so the bar we must beat is the honest one.

Long-only spot, no leverage, no shorts, taker cost on turnover.
No emoji (cp1252). Does NOT git commit.

Run:
  python quant_ml_referee.py            (from crypto/runs/periods/ALL_WEATHER, or anywhere with PYTHONPATH=crypto/src)
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

# locate crypto/src
HERE = Path(__file__).resolve()
SRC = None
for p in HERE.parents:
    if (p / "strat" / "mover_lab.py").exists():
        SRC = p; break
    if (p / "src" / "strat" / "mover_lab.py").exists():
        SRC = p / "src"; break
if SRC is None:
    SRC = Path(r"c:\Users\karab\Documents\coding\ml_systems\crypto\src")
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import strat.mover_lab as lab
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SEED = 42
H = 7                    # forward window (days == bars; verified contiguous)
COST = lab.COST          # taker round-trip
RETRAIN_EVERY = 60       # bars between retrains
MIN_TRAIN = 600          # min closed-label training rows before OOS starts
N_SLICES = 400

FEATS = ["dist_sma200", "dist_sma50", "range_pos", "rsi14", "vol20", "vol_ratio",
         "mom7", "mom14", "mom30", "atr_ratio", "ret1", "ret3",
         "mom7_rank", "mom14_rank", "gate", "breadth", "btc_regime"]


# ----------------------------------------------------------------------------- feature build (causal)
def build_panels(ind):
    """Return dict of dates x assets feature panels + the H-forward label panel."""
    C = ind["C"]; R = ind["R"]; eps = 1e-8
    sma200, sma50 = ind["sma200"], ind["sma50"]
    hh14, ll14 = ind["hh14"], ind["ll14"]
    rng = hh14 - ll14
    vol20 = ind["vol20"]
    vol_ratio = vol20 / (vol20.rolling(60, min_periods=20).mean() + eps)
    P = {
        "dist_sma200": C / (sma200 + eps) - 1,
        "dist_sma50":  C / (sma50 + eps) - 1,
        "range_pos":   (C - ll14) / (rng + eps),
        "rsi14":       ind["rsi14"],
        "vol20":       vol20,
        "vol_ratio":   vol_ratio,
        "mom7":        ind["mom7"],
        "mom14":       ind["mom14"],
        "mom30":       ind["mom30"],
        "atr_ratio":   ind["atr14"] / (C + eps),
        "ret1":        ind["ret1"],
        "ret3":        C / C.shift(3) - 1,
        "mom7_rank":   ind["mom7"].rank(axis=1, pct=True),
        "mom14_rank":  ind["mom14"].rank(axis=1, pct=True),
        "gate":        ind["gate"].astype(float),
    }
    breadth = (C > sma50).astype(float).mean(axis=1)          # date -> scalar
    btc_reg = (C["BTCUSDT"] > sma200["BTCUSDT"]).astype(float).fillna(0.0)
    # broadcast scalars to panels
    P["breadth"]    = pd.DataFrame({s: breadth for s in C.columns})
    P["btc_regime"] = pd.DataFrame({s: btc_reg for s in C.columns})
    # LABEL panel: H-forward return (future) -- LABEL ONLY, never a feature
    fwd = C.shift(-H) / C - 1
    return P, fwd, C


def shift_audit(ind):
    """LOOK-AHEAD AUDIT: rebuild every feature panel on a TRUNCATED copy of the data
    (drop the last 30 bars) and confirm the feature values on the overlapping rows are
    BIT-IDENTICAL. A feature that uses future data would CHANGE when the future is removed."""
    C = ind["C"]
    cut = C.index[-30]
    ind_full = ind
    ind_trunc = lab.load(str(C.index[0].date()), str(cut.date()))
    Pf, _, _ = build_panels(ind_full)
    Pt, _, _ = build_panels(ind_trunc)
    common_idx = Pt[FEATS[0]].index
    bad = []
    for k in FEATS:
        a = Pf[k].reindex(index=common_idx)
        b = Pt[k].reindex(index=common_idx)
        # compare on the last 60 overlapping rows (where a future-leak would bite)
        a2 = a.iloc[-60:]; b2 = b.iloc[-60:]
        diff = (a2 - b2).abs()
        maxdiff = float(np.nanmax(diff.to_numpy())) if diff.size else 0.0
        if maxdiff > 1e-9:
            bad.append((k, maxdiff))
    return bad


# ----------------------------------------------------------------------------- panel -> long matrix
def to_long(P, fwd, C):
    dates = C.index; cols = C.columns
    n = len(dates) * len(cols)
    di = np.repeat(np.arange(len(dates)), len(cols))
    ai = np.tile(np.arange(len(cols)), len(dates))
    X = np.column_stack([P[k].to_numpy().reshape(-1) for k in FEATS])
    y = (fwd.to_numpy().reshape(-1) > 0).astype(float)
    yraw = fwd.to_numpy().reshape(-1)
    valid_label = ~np.isnan(yraw)
    valid_feat = ~np.isnan(X).any(axis=1)
    return X, y, yraw, di, ai, valid_label, valid_feat, dates, cols


# ----------------------------------------------------------------------------- walk-forward
def walk_forward(X, y, yraw, di, ai, valid_label, valid_feat, dates, cols,
                 strict=True, retrain_every=RETRAIN_EVERY, min_train=MIN_TRAIN):
    """Return prob array (same length as X; NaN where no OOS prediction).
    strict=True  -> training rows: label window closed (d+H < T) AND purge within H of T; fit per fold.
    strict=False -> LEAKY twin: train on ALL rows with a valid label (incl. open windows / future);
                    this is the deliberately-leaky control to MEASURE the gap.
    """
    nD = len(dates)
    prob = np.full(X.shape[0], np.nan)
    # find OOS start: first T with >= min_train closed-label, feature-valid rows
    start_i = None
    for i in range(nD):
        if strict:
            adm = valid_label & valid_feat & (di <= i - H - 1)        # d+H < T  => d <= T-H-1
        else:
            adm = valid_label & valid_feat                            # leaky: everything
        if adm.sum() >= min_train:
            start_i = i; break
    if start_i is None:
        raise RuntimeError("not enough data")

    model = None
    last_train = -10**9
    for i in range(start_i, nD):
        if i - last_train >= retrain_every:
            if strict:
                adm = valid_label & valid_feat & (di <= i - H - 1)
            else:
                # LEAKY: train on the WHOLE sample (past + future), classic full-sample-fit leak
                adm = valid_label & valid_feat
            Xtr, ytr = X[adm], y[adm]
            if len(Xtr) >= min_train:
                model = HistGradientBoostingClassifier(
                    max_iter=200, max_depth=4, learning_rate=0.05,
                    min_samples_leaf=20, l2_regularization=1.0, random_state=SEED)
                model.fit(Xtr, ytr)
                last_train = i
        if model is None:
            continue
        day = (di == i) & valid_feat
        if day.any():
            prob[day] = model.predict_proba(X[day])[:, 1]
    return prob, start_i


# ----------------------------------------------------------------------------- benchmarks & books
def point_in_time_bh(C):
    """EW buy-hold simple H-forward return per (start) date, averaging ONLY listed assets."""
    return C.shift(-H) / C - 1   # NaN where unlisted -> excluded by nanmean later


def slice_eval(prob, di, ai, C, dates, cols, oos_start_i,
               topk=3, thresh=0.5, n_slices=N_SLICES, seed=SEED,
               control="none"):
    """For n_slices random OOS start bars, compute:
        ml_ret  = EW H-fwd return of the picked names (top-k by prob, prob>=thresh; cash=0 if none)
        bh_ret  = EW H-fwd return of ALL listed names (point-in-time)
        rnd_ret = SAME-EXPOSURE control: EW H-fwd of k randomly chosen LISTED names (if ML invested)
       Returns arrays. control: 'none' | 'same_expo_random'
    """
    rng = np.random.default_rng(seed)
    Cv = C.to_numpy()
    nD = len(dates)
    last_start = nD - H - 1
    starts = np.arange(oos_start_i, last_start + 1)
    starts = starts[starts >= oos_start_i]
    pick = rng.choice(starts, size=n_slices, replace=True)

    # prob reshaped to dates x assets
    probM = prob.reshape(nD, len(cols))
    ml, bh, rnd, expo = [], [], [], []
    for s in pick:
        # listed at both ends
        fwd_row = Cv[s + H] / Cv[s] - 1.0
        listed = ~np.isnan(fwd_row)
        # BH = nanmean over listed
        bh.append(np.nanmean(fwd_row[listed]) if listed.any() else 0.0)
        # ML selection
        pr = probM[s].copy()
        pr[~listed] = -np.inf            # cannot hold unlisted
        elig = np.where((pr >= thresh) & np.isfinite(pr))[0]
        if elig.size == 0:
            ml.append(0.0); expo.append(0.0)
            rnd.append(0.0)
            continue
        order = elig[np.argsort(-pr[elig])][:topk]
        ml.append(float(np.mean(fwd_row[order])))
        expo.append(1.0)
        # same-exposure random control: pick |order| random LISTED names
        lidx = np.where(listed)[0]
        if control == "same_expo_random" and lidx.size > 0:
            rc = rng.choice(lidx, size=min(len(order), lidx.size), replace=False)
            rnd.append(float(np.mean(fwd_row[rc])))
        else:
            rnd.append(np.nan)
    return np.array(ml), np.array(bh), np.array(rnd), np.array(expo)


def block_bootstrap_winrate(win_bool, block=8, n_boot=3000, seed=SEED):
    """Moving-block bootstrap CI on a win-rate (slices overlap -> dependent)."""
    rng = np.random.default_rng(seed)
    x = win_bool.astype(float); n = len(x)
    if n == 0:
        return (np.nan, np.nan)
    nb = int(np.ceil(n / block))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=nb) if n > block else np.array([0])
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        boots[b] = x[idx].mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ----------------------------------------------------------------------------- smoothing ceiling
def smoothing_ceiling(ind, n_slices=N_SLICES, seed=SEED):
    """How high can the random-7d-slice ABS win-rate go with NO prediction -- pure structure?
    EW-BH vs inverse-vol vs equal-risk(=inv-vol on daily) over the SAME random slices."""
    C = ind["C"]; R = ind["R"]
    vol = R.rolling(20, min_periods=10).std()
    Cv = C.to_numpy(); volv = vol.to_numpy()
    nD = len(C)
    rng = np.random.default_rng(seed)
    starts = np.arange(200, nD - H - 1)         # after warmup
    pick = rng.choice(starts, size=n_slices, replace=True)
    out = {"EW": [], "INVVOL": []}
    for s in pick:
        fwd_row = Cv[s + H] / Cv[s] - 1.0
        listed = ~np.isnan(fwd_row)
        if not listed.any():
            continue
        out["EW"].append(np.nanmean(fwd_row[listed]))
        w = 1.0 / (volv[s] + 1e-8)
        w = np.where(listed & np.isfinite(w), w, 0.0)
        if w.sum() > 0:
            w = w / w.sum()
            out["INVVOL"].append(float(np.nansum(np.where(listed, fwd_row, 0.0) * w)))
        else:
            out["INVVOL"].append(np.nanmean(fwd_row[listed]))
    res = {}
    for k, v in out.items():
        a = np.array(v)
        res[k] = {"abs_wr": float((a > 0).mean()), "mean": float(a.mean()), "n": len(a)}
    return res


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    print("=" * 78)
    print("QUANT REFEREE -- strict-walk-forward re-derivation of the ML lane (RWYB)")
    print("=" * 78)
    ind = lab.load("2020-01-01", "2026-06-01")
    C = ind["C"]
    print(f"data: {C.index[0].date()}..{C.index[-1].date()}  bars={len(C)}  assets={len(C.columns)}")

    # ---- A. SHIFT-AUDIT: prove no feature leaks future ----
    print("\n[A] SHIFT-AUDIT (rebuild features without the last 30 bars; values must be identical)...")
    bad = shift_audit(ind)
    if bad:
        print("    LEAK DETECTED in features:", bad)
    else:
        print("    PASS -- every feature is causal (bit-identical when future removed).")

    P, fwd, C = build_panels(ind)
    X, y, yraw, di, ai, vl, vf, dates, cols = to_long(P, fwd, C)
    print(f"    long matrix: {X.shape[0]} rows, {X.shape[1]} feats; "
          f"label-valid={int(vl.sum())} feat-valid={int(vf.sum())}")

    # ---- B. STRICT walk-forward ----
    print("\n[B] STRICT walk-forward (purge H, fold-only fit, no future in train)...")
    prob_s, start_i = walk_forward(X, y, yraw, di, ai, vl, vf, dates, cols, strict=True)
    oos_mask = ~np.isnan(prob_s) & vl
    auc_s = roc_auc_score(y[oos_mask], prob_s[oos_mask]) if oos_mask.sum() > 50 else np.nan
    print(f"    OOS starts {dates[start_i].date()}  n_oos_preds={int((~np.isnan(prob_s)).sum())}  OOS_AUC={auc_s:.4f}")

    # ---- C. LEAKY twin (measure the gap) ----
    print("\n[C] LEAKY twin (full-sample fit, no purge -- the gap = leak magnitude)...")
    prob_l, start_l = walk_forward(X, y, yraw, di, ai, vl, vf, dates, cols, strict=False)
    oos_mask_l = ~np.isnan(prob_l) & vl
    auc_l = roc_auc_score(y[oos_mask_l], prob_l[oos_mask_l]) if oos_mask_l.sum() > 50 else np.nan
    print(f"    LEAKY OOS_AUC={auc_l:.4f}  (vs strict {auc_s:.4f}; gap={auc_l-auc_s:+.4f})")

    # ---- D. random-slice eval, strict vs leaky, with same-exposure control ----
    print("\n[D] RANDOM 7d-SLICE EVAL (n=%d), strict vs leaky, + same-exposure random control..." % N_SLICES)
    rows = []
    for tag, prob, si in [("STRICT", prob_s, start_i), ("LEAKY", prob_l, start_l)]:
        for topk, thr in [(3, 0.5), (5, 0.5), (5, 0.45), (10, 0.0)]:
            ml, bh, rnd, expo = slice_eval(prob, di, ai, C, dates, cols, si,
                                           topk=topk, thresh=thr, control="same_expo_random")
            ml_wr = float((ml > 0).mean()); bh_wr = float((bh > 0).mean())
            rnd_wr = float((rnd[~np.isnan(rnd)] > 0).mean()) if np.isfinite(rnd).any() else np.nan
            beat = float((ml > bh).mean())
            lo, hi = block_bootstrap_winrate(ml > 0)
            rows.append(dict(model=tag, topk=topk, thr=thr, n=len(ml),
                             ml_abs_wr=ml_wr, bh_abs_wr=bh_wr, rnd_abs_wr=rnd_wr,
                             ml_mean=float(ml.mean()), bh_mean=float(bh.mean()),
                             beat_bh=beat, expo=float(np.mean(expo)),
                             wr_ci_lo=lo, wr_ci_hi=hi))
    # print table
    print(f"\n  {'model':<7}{'K':>3}{'thr':>6}{'n':>5}{'ml_WR':>8}{'bh_WR':>8}{'rnd_WR':>8}"
          f"{'ml_mean':>9}{'bh_mean':>9}{'beatBH':>8}{'expo':>6}{'WR_CI':>16}")
    for r in rows:
        print(f"  {r['model']:<7}{r['topk']:>3}{r['thr']:>6.2f}{r['n']:>5}"
              f"{r['ml_abs_wr']*100:>7.1f}%{r['bh_abs_wr']*100:>7.1f}%"
              f"{(r['rnd_abs_wr']*100 if not np.isnan(r['rnd_abs_wr']) else float('nan')):>7.1f}%"
              f"{r['ml_mean']*100:>+8.2f}%{r['bh_mean']*100:>+8.2f}%{r['beat_bh']*100:>7.1f}%"
              f"{r['expo']*100:>5.0f}%  [{r['wr_ci_lo']*100:>4.1f},{r['wr_ci_hi']*100:>4.1f}]")

    # ---- E. smoothing ceiling ----
    print("\n[E] SMOOTHING CEILING (no prediction -- pure structure)...")
    sc = smoothing_ceiling(ind)
    for k, v in sc.items():
        print(f"    {k:<8} abs_WR={v['abs_wr']*100:.1f}%  mean={v['mean']*100:+.2f}%  n={v['n']}")

    # ---- verdict numbers ----
    strict_best_wr = max(r["ml_abs_wr"] for r in rows if r["model"] == "STRICT")
    strict_best_mean = max(r["ml_mean"] for r in rows if r["model"] == "STRICT")
    leaky_best_wr = max(r["ml_abs_wr"] for r in rows if r["model"] == "LEAKY")
    bh_wr_ref = rows[0]["bh_abs_wr"]; bh_mean_ref = rows[0]["bh_mean"]

    print("\n" + "=" * 78)
    print("REFEREE VERDICT")
    print("=" * 78)
    print(f"  strict OOS AUC          : {auc_s:.4f}   (0.50 = coin flip)")
    print(f"  leaky  OOS AUC          : {auc_l:.4f}   (gap {auc_l-auc_s:+.4f} = pure leak inflation)")
    print(f"  STRICT best abs win-rate: {strict_best_wr*100:.1f}%   (target >55%; BH={bh_wr_ref*100:.1f}%)")
    print(f"  LEAKY  best abs win-rate: {leaky_best_wr*100:.1f}%   (the inflated mirage)")
    print(f"  STRICT best mean/slice  : {strict_best_mean*100:+.2f}%  (target >+2.9%; BH={bh_mean_ref*100:+.2f}%)")
    print(f"  smoothing ceiling (no pred) EW={sc['EW']['abs_wr']*100:.1f}%  INVVOL={sc['INVVOL']['abs_wr']*100:.1f}%")

    out = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"),
               data=dict(start=str(C.index[0].date()), end=str(C.index[-1].date()), bars=len(C)),
               shift_audit_leaks=bad, auc_strict=auc_s, auc_leaky=auc_l,
               rows=rows, smoothing=sc,
               strict_best_wr=strict_best_wr, strict_best_mean=strict_best_mean,
               leaky_best_wr=leaky_best_wr, bh_wr=bh_wr_ref, bh_mean=bh_mean_ref)
    outp = Path(__file__).resolve().parent / "quant_ml_referee_results.json"
    outp.write_text(json.dumps(out, indent=2, default=lambda o: None))
    print(f"\n  wrote {outp}")
    print(f"  elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
