"""
experiments/adaptive_ma/sol/oracle_dna_shuffled_falsifier.py

FALSIFIER: shuffled-label control on the ORACLE-DNA classifier (SOL, 4h).

WHAT THIS TESTS
---------------
The oracle-decomposition method (docs/ORACLE_DECOMPOSITION_2026_06_06.md, step 2) builds a
classifier P(oracle-entry at t | causal past-only features) and calls the part that generalizes on
held-out the "MA-DNA" (the realizable signal). This script is the FALSIFIER for that claim. It asks:

    Is the DNA classifier learning a GENUINE feature->oracle-entry map, or is it fitting noise /
    regime-beta that a permutation cannot tell apart from the real thing?

Three controls, all evaluated on HELD-OUT folds the classifier never saw:

  (A) SHUFFLED-LABEL CONTROL  -- permute the capturable/non labels in the FIT data, refit, and
      re-measure held-out AUC / IC / capture-skill. A genuine pipeline MUST collapse to chance
      (AUC ~ 0.5, IC ~ 0, capture-skill ~ 0). A shuffled control that does NOT collapse means the
      apparatus is LEAKING (the held-out labels are predictable without any real feature->label link
      -- e.g. a target leak or a time/regime distribution shift the model rides). [primary check]

  (B) POSITIVE CONTROL        -- plant a SYNTHETIC label that is a known (noisy) function of real
      past-only features, refit, confirm held-out AUC >> 0.5. This proves the pipeline HAS POWER:
      if a genuine learnable DNA existed, it WOULD be found. (Two-sided soundness, per the project's
      "a gate must ACCEPT a real signal, not only reject ghosts" rule.) Its OWN shuffled twin must
      also collapse -- proving the collapse in (A) is a property of permutation, not of weak features.

  (C) REGIME-MATCHED FIREWALL -- the DNA may merely detect "we are in an uptrend" (regime beta).
      Compare the DNA's held-out capture to a cost-matched random-entry null drawn ONLY from the same
      regime (price above its own past-only SMA-200). If the DNA does not beat the regime-matched
      null, its apparent skill is regime beta, not entry TIMING. (Plain all-bars null reported too.)

VERDICTS (kept distinct):
  * APPARATUS-SOUND  iff  shuffled control collapses (A)  AND  positive control has power (B).
                     This is what the task asks to CONFIRM: a non-collapsing shuffled control = leak.
  * DNA-GENUINE      iff  real held-out AUC > shuffled p95 (and > 0.5 by margin)  AND  real capture
                     beats BOTH the plain and the regime-matched random-entry null p95.
  A sound apparatus with a NON-genuine DNA is a valid, honest refutation of the SOL-4h MA-DNA signal
  (it would say: regime beta / noise, not timing) -- NOT a bug in the falsifier.

HARD CONSTRAINTS (inherited): LONG-ONLY, honest taker round-trip cost 0.0024, next-bar-OPEN entry,
open-to-open fixed-H exit (NO intra-bar high look-ahead on the proxy), features are past-only
normalized chimera columns only (target_* / forward columns NEVER enter X).

RWYB:
    .venv/Scripts/python.exe experiments/adaptive_ma/sol/oracle_dna_shuffled_falsifier.py --selftest
    .venv/Scripts/python.exe experiments/adaptive_ma/sol/oracle_dna_shuffled_falsifier.py
    .venv/Scripts/python.exe experiments/adaptive_ma/sol/oracle_dna_shuffled_falsifier.py --asset SOL --cadence 4h --n-shuffle 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "runs" / "research"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from oracle_ceiling_builder import oracle_high_capture, WIN  # noqa: E402  (reuse the audited oracle DP)

COST_RT = 0.0024

__contract__ = {
    "kind": "oracle_dna_shuffled_falsifier",
    "version": "1.0",
    "inputs": ["chimera (asset, cadence) past-only norm_/xd_ features", "oracle-entry labels from oracle_high_capture"],
    "outputs": ["held-out AUC/IC/capture-skill: real vs shuffled-null vs positive-control vs regime-firewall; JSON+verdict"],
    "invariants": [
        "features are past-only normalized chimera cols only; target_*/high/forward cols NEVER in X",
        "fit on TRAIN+VAL; evaluate on HELD-OUT (OOS+UNSEEN) the model never saw; shuffle permutes FIT labels only",
        "capture proxy uses next-bar-OPEN entry + open-to-open fixed-H exit (no intra-bar high look-ahead)",
        "shuffled control must collapse to chance else APPARATUS LEAK; positive control must retain power",
        "regime-matched null draws random entries only from price>SMA200(past) bars -> isolates timing from regime beta",
        "honest cost 0.0024 round-trip subtracted on every proxy move (real, shuffled, null alike)",
    ],
}


# ---------------------------------------------------------------------------
# feature set: the canonical past-only normalized chimera columns (std~=1), plus cross-asset xd_.
# These are explicitly lookahead_safe. target_*/voladj_* (forward) and raw OHLC are EXCLUDED from X.
def _feature_cols(cols):
    feats = [c for c in cols if c.startswith("norm_") or c.startswith("xd_")]
    # guard: never let a forward/target column slip in
    feats = [c for c in feats if "target" not in c and "voladj" not in c]
    return feats


def _window_mask(ts_ms, name):
    lo, hi = WIN[name]
    lo_ms = 0 if lo == "0" else int(pd.Timestamp(lo).value // 1_000_000)
    hi_ms = int(pd.Timestamp(hi).value // 1_000_000)
    return (ts_ms >= lo_ms) & (ts_ms < hi_ms)


def _sma_past(close, w):
    """Strictly past-only SMA: sma[i] uses close[i-w..i-1] (shifted by 1 so bar i is excluded)."""
    s = pd.Series(close).shift(1).rolling(w, min_periods=w).mean().to_numpy()
    return s


def load_asset(asset, cadence):
    from pipeline.chimera_loader import ChimeraLoader
    g = ChimeraLoader().load(asset + "USDT", cadence=cadence)
    cols = list(g.columns)
    ts = g["timestamp"].to_numpy().astype(np.int64)
    op = g["open"].to_numpy().astype(np.float64)
    hi = g["high"].to_numpy().astype(np.float64)
    cl = g["close"].to_numpy().astype(np.float64)
    if not np.all(np.diff(ts) > 0):
        order = np.argsort(ts, kind="stable")
        ts, op, hi, cl = ts[order], op[order], hi[order], cl[order]
        g = g[order]
    feats = _feature_cols(cols)
    X = g.select(feats).to_numpy().astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)  # normalized -> 0 == mean
    return ts, op, hi, cl, X, feats


# ---------------------------------------------------------------------------
def fwd_open_to_open(op, H, cost=COST_RT):
    """Honest forward net return of entering at open[i+1] and exiting at open[i+1+H]. NaN where invalid."""
    n = len(op)
    fwd = np.full(n, np.nan)
    nxt = np.full(n, np.nan)
    nxt[: n - 1] = op[1:]                      # entry fill = next open
    for i in range(n):
        ei = i + 1
        xi = i + 1 + H
        if xi < n and ei < n:
            fwd[i] = op[xi] / op[ei] - 1.0 - cost
    return fwd


def compound(nets):
    nets = np.asarray(nets, float)
    nets = nets[~np.isnan(nets)]
    return float(np.prod(1.0 + nets) - 1.0) if len(nets) else 0.0


def capture_skill(p_pred, fwd, k_frac, eligible):
    """Normalized capture SKILL of a probability ranking, in [chance=0 .. perfect=1], using bounded
    PER-MOVE MEAN net returns (NOT compounds -- a 666-trade compound of a linear `k*mean` chance proxy
    is unbounded below -1 and pollutes the ratio; per-move means are bounded and cost/scale-invariant).

    dna_mean   = mean honest fwd over the top-(k_frac) eligible bars ranked by p_pred
    best_mean  = mean fwd over the top-(k_frac) eligible bars ranked by ACTUAL fwd (perfect selection)
    chance_mean= mean fwd over ALL eligible bars (random selection expectation)
    skill = (dna_mean - chance_mean) / (best_mean - chance_mean)  -> 0 at chance, 1 at perfect selection.
    Also reports the honest compound of the selected entries (proper, for the firewall comparison)."""
    idx = np.where(eligible & ~np.isnan(fwd))[0]
    if len(idx) < 5:
        return {"skill": float("nan"), "dna_compound_pct": float("nan"), "n_sel": 0}
    k = max(1, int(round(k_frac * len(idx))))
    f = fwd[idx]
    pr = p_pred[idx]
    sel = idx[np.argsort(-pr)[:k]]                 # top-k by predicted prob
    dna_mean = float(np.mean(fwd[sel]))
    best_mean = float(np.mean(f[np.argsort(-f)[:k]]))   # top-k by actual fwd = ceiling for this exit policy
    chance_mean = float(np.mean(f))                # random-selection expectation
    denom = best_mean - chance_mean
    skill = float((dna_mean - chance_mean) / denom) if abs(denom) > 1e-12 else float("nan")
    return {"skill": skill, "dna_compound_pct": compound(fwd[sel]) * 100.0, "n_sel": int(k),
            "dna_mean_net_pct": dna_mean * 100.0, "best_mean_net_pct": best_mean * 100.0,
            "chance_mean_net_pct": chance_mean * 100.0}


def fit_predict(Xtr, ytr, Xho, seed=0, shuffle=False, model="logistic"):
    """Fit on train, return held-out P(label=1). model='logistic' (StandardScaler + balanced L2 logistic, the
    linear baseline) or 'gbm' (HistGradientBoostingClassifier -- captures MA-crossover NONLINEARITIES + feature
    interactions / regime conditioning = the STRONGEST fair form of the 'adaptive multi-MA' hypothesis). The SAME
    model is used for the real fit, the shuffled control, and the positive control, so the firewall stays valid."""
    rng = np.random.default_rng(seed)
    y = ytr.copy()
    if shuffle:
        y = y[rng.permutation(len(y))]            # PERMUTE labels (break feature->label link)
    if len(np.unique(y)) < 2:
        return np.full(len(Xho), float(np.mean(y)))
    if model == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.06,
                                             l2_regularization=1.0, early_stopping=True,
                                             validation_fraction=0.15, random_state=seed)
        clf.fit(Xtr, y)                            # tree model -> no scaling needed
        return clf.predict_proba(Xho)[:, 1]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced", solver="lbfgs")
    clf.fit(sc.transform(Xtr), y)
    return clf.predict_proba(sc.transform(Xho))[:, 1]


def eval_block(p_ho, y_ho, fwd_ho, k_frac, elig_all, elig_regime):
    """AUC + IC(rank corr of p vs fwd) + plain & regime-matched capture skill on a held-out block."""
    out = {}
    m = ~np.isnan(y_ho.astype(float))
    out["auc"] = float(roc_auc_score(y_ho[m], p_ho[m])) if len(np.unique(y_ho[m])) == 2 else float("nan")
    mm = (~np.isnan(fwd_ho)) & (~np.isnan(p_ho))
    if mm.sum() > 10:
        ic, _ = spearmanr(p_ho[mm], fwd_ho[mm])
        out["ic_fwd"] = float(ic)
    else:
        out["ic_fwd"] = float("nan")
    out["capture_plain"] = capture_skill(p_ho, fwd_ho, k_frac, elig_all)
    out["capture_regime"] = capture_skill(p_ho, fwd_ho, k_frac, elig_regime)
    return out


def random_entry_null_capture(fwd, eligible, k_frac, n_books, seed):
    """Cost-matched random-entry null: compound of k random eligible bars, n_books draws."""
    rng = np.random.default_rng(seed)
    idx = np.where(eligible & ~np.isnan(fwd))[0]
    if len(idx) < 5:
        return None
    k = max(1, int(round(k_frac * len(idx))))
    comps = np.array([compound(fwd[rng.choice(idx, size=k, replace=True)]) for _ in range(n_books)])
    return {"p50_pct": float(np.percentile(comps, 50) * 100), "p95_pct": float(np.percentile(comps, 95) * 100),
            "mean_pct": float(comps.mean() * 100), "k": int(k), "n_pool": int(len(idx))}


# ---------------------------------------------------------------------------
def run(asset="SOL", cadence="4h", n_shuffle=50, n_books=400, seed=7, verbose=True, exit_h=None,
        min_move_net=0.0, model="logistic"):
    ts, op, hi, cl, X, feats = load_asset(asset, cadence)
    n = len(op)

    # ---- labels: oracle-entry (capturable) vs non, from the audited perfect-foresight DP.
    # min_move_net = per-move net floor: 0.0 = scalp oracle (2-bar wiggles); 0.03-0.05 = SWING oracle (fewer,
    # larger multi-day moves = the unit the project targets, and a FAIR test for trend instruments like MAs).
    f_dp, trades = oracle_high_capture(ts, op, hi, min_move_net=min_move_net)
    y = np.zeros(n, dtype=int)
    ent = np.array([i for i, j in trades], dtype=int)
    y[ent] = 1
    holds = np.array([j - i for i, j in trades], dtype=int)
    H = int(exit_h) if exit_h else int(max(1, np.median(holds)))   # exit horizon (bars); default=oracle median hold

    # ---- honest forward proxy + regime (past-only SMA-200 uptrend)
    fwd = fwd_open_to_open(op, H, COST_RT)
    sma200 = _sma_past(cl, 200)
    uptrend = (cl > sma200)                        # past-only regime: close above its own trailing SMA-200

    # ---- splits: FIT = TRAIN+VAL ; HELD-OUT = OOS, UNSEEN (model never sees them)
    m_tr = _window_mask(ts, "TRAIN") | _window_mask(ts, "VAL")
    m_oos = _window_mask(ts, "OOS")
    m_uns = _window_mask(ts, "UNSEEN")
    m_ho = m_oos | m_uns
    Xtr, ytr = X[m_tr], y[m_tr]
    k_frac = float(ytr.mean())                     # match selection count to the oracle base rate

    # eligibility for the capture proxy (need a valid fwd at horizon H)
    elig_all = ~np.isnan(fwd)
    elig_reg = elig_all & np.nan_to_num(uptrend, nan=False).astype(bool)

    if verbose:
        print("=" * 80)
        print(f"ORACLE-DNA SHUFFLED-LABEL FALSIFIER  --  {asset} {cadence}")
        print("=" * 80)
        print(f"bars={n}  features={len(feats)}  oracle_entries={int(y.sum())} ({y.mean():.3f})  "
              f"exit_H={H} bars  k_frac(base rate)={k_frac:.3f}")
        print(f"FIT(TRAIN+VAL)={int(m_tr.sum())}  HELD-OUT OOS={int(m_oos.sum())}  UNSEEN={int(m_uns.sum())}")

    # ---- (1) REAL fit
    p_real = fit_predict(Xtr, ytr, X, seed=seed, shuffle=False, model=model)

    def block(mask):
        return eval_block(p_real[mask], y[mask], fwd[mask], k_frac,
                          elig_all[mask], elig_reg[mask])

    real = {"OOS": block(m_oos), "UNSEEN": block(m_uns), "HELD_OUT": block(m_ho)}

    # ---- (A) SHUFFLED-LABEL CONTROL: permute FIT labels, refit, re-eval held-out. n_shuffle seeds.
    sh_auc, sh_ic, sh_cap_plain, sh_cap_reg = [], [], [], []
    for s in range(n_shuffle):
        p_s = fit_predict(Xtr, ytr, X, seed=1000 + s, shuffle=True, model=model)
        b = eval_block(p_s[m_ho], y[m_ho], fwd[m_ho], k_frac, elig_all[m_ho], elig_reg[m_ho])
        sh_auc.append(b["auc"]); sh_ic.append(b["ic_fwd"])
        sh_cap_plain.append(b["capture_plain"]["skill"]); sh_cap_reg.append(b["capture_regime"]["skill"])

    def dist(a):
        a = np.array([x for x in a if not (isinstance(x, float) and np.isnan(x))], float)
        if len(a) == 0:
            return {"mean": float("nan"), "p05": float("nan"), "p95": float("nan"), "max": float("nan")}
        return {"mean": float(a.mean()), "p05": float(np.percentile(a, 5)),
                "p95": float(np.percentile(a, 95)), "max": float(a.max())}

    shuffled = {"auc": dist(sh_auc), "ic_fwd": dist(sh_ic),
                "capture_plain_skill": dist(sh_cap_plain), "capture_regime_skill": dist(sh_cap_reg),
                "n_shuffle": n_shuffle}

    # ---- (B) POSITIVE CONTROL: plant a noisy-but-learnable label from real past-only features,
    #          confirm power (held-out AUC >> 0.5); its OWN shuffled twin must also collapse.
    rng = np.random.default_rng(123)
    # signal = linear combo of two real momentum features + noise; label = signal in top half
    def col(name):
        return X[:, feats.index(name)] if name in feats else np.zeros(n)
    signal = 0.8 * col("norm_momentum_accel") + 0.8 * col("norm_return_4") + rng.normal(0, 1.0, n)
    y_pos = (signal > np.median(signal[m_tr])).astype(int)
    p_pos = fit_predict(Xtr, y_pos[m_tr], X, seed=seed, shuffle=False, model=model)
    pos_auc = float(roc_auc_score(y_pos[m_ho], p_pos[m_ho]))
    pos_sh = []
    for s in range(min(n_shuffle, 30)):
        p_ps = fit_predict(Xtr, y_pos[m_tr], X, seed=2000 + s, shuffle=True, model=model)
        pos_sh.append(float(roc_auc_score(y_pos[m_ho], p_ps[m_ho])))
    # the positive-control label lives in feature space, so its shuffled-twin AUC has WIDER variance
    # (a random weight vector can partially align with the planted 2-feature direction) -> judge the
    # collapse by the robust MEAN (~0.5), not the small-n p95 tail.
    positive_control = {"held_out_auc": pos_auc, "shuffled_auc": dist(pos_sh),
                        "has_power": bool(pos_auc > 0.60),
                        "shuffled_collapses": bool(dist(pos_sh)["mean"] < 0.55)}

    # ---- (C) REGIME-MATCHED FIREWALL on the DNA capture (held-out)
    null_plain = random_entry_null_capture(fwd[m_ho], elig_all[m_ho], k_frac, n_books, seed)
    null_regime = random_entry_null_capture(fwd[m_ho], elig_reg[m_ho], k_frac, n_books, seed + 1)
    dna_cap_plain_pct = real["HELD_OUT"]["capture_plain"]["dna_compound_pct"]
    dna_cap_reg_pct = real["HELD_OUT"]["capture_regime"]["dna_compound_pct"]
    firewall = {
        "dna_capture_plain_compound_pct": dna_cap_plain_pct,
        "dna_capture_regime_compound_pct": dna_cap_reg_pct,
        "null_plain": null_plain, "null_regime": null_regime,
        "beats_plain_null_p95": bool(null_plain is not None and dna_cap_plain_pct > null_plain["p95_pct"]),
        "beats_regime_null_p95": bool(null_regime is not None and dna_cap_reg_pct > null_regime["p95_pct"]),
    }

    # ---- VERDICTS
    real_auc = real["HELD_OUT"]["auc"]
    real_ic = real["HELD_OUT"]["ic_fwd"]
    real_cap_skill = real["HELD_OUT"]["capture_plain"]["skill"]
    # SOUNDNESS = "does the permutation destroy the signal ON AVERAGE?" -> a MEAN question, per the selftest's
    # own documented principle ("single permutations are noisy; the MEAN is the statistically correct
    # 'collapses to chance' check"). The previous gate used the shuffle p95 TAIL (<0.55), which at modest
    # n_shuffle straddles the cutoff from sampling noise alone (BTC-1d: mean=0.502 SOUND but p95=0.555 -> false
    # leak alarm; SOL-4h: mean=0.489 p95=0.523). A genuine distribution-shift leak elevates the MEAN, not just
    # the tail, so mean-based detection still catches real leaks. Tail belongs to the GENUINENESS check below.
    shuffled_collapses = bool(shuffled["auc"]["mean"] < 0.54 and abs(shuffled["ic_fwd"]["mean"]) < 0.03
                              and shuffled["capture_plain_skill"]["mean"] < 0.10)
    apparatus_sound = bool(shuffled_collapses and positive_control["has_power"]
                           and positive_control["shuffled_collapses"])
    dna_auc_genuine = bool((real_auc > shuffled["auc"]["p95"]) and (real_auc > 0.53))
    dna_capture_genuine = bool(firewall["beats_plain_null_p95"] and firewall["beats_regime_null_p95"])
    dna_genuine = bool(dna_auc_genuine and dna_capture_genuine)

    result = {
        "asset": asset, "cadence": cadence, "n_bars": n, "n_features": len(feats),
        "model": model, "min_move_net": float(min_move_net),
        "oracle_entries": int(y.sum()), "oracle_base_rate": float(y.mean()), "exit_H_bars": H,
        "cost_rt": COST_RT, "fit_rows": int(m_tr.sum()), "held_out_rows": int(m_ho.sum()),
        "real": real, "shuffled_control": shuffled, "positive_control": positive_control,
        "regime_firewall": firewall,
        "real_held_out_auc": real_auc, "real_held_out_ic_fwd": real_ic,
        "real_held_out_capture_skill": real_cap_skill,
        "VERDICT": {
            "shuffled_control_collapses": shuffled_collapses,
            "positive_control_has_power": positive_control["has_power"],
            "APPARATUS_SOUND": apparatus_sound,
            "dna_auc_beats_shuffled": dna_auc_genuine,
            "dna_capture_beats_firewall": dna_capture_genuine,
            "DNA_GENUINE_SIGNAL": dna_genuine,
        },
    }

    if verbose:
        print("\n" + "-" * 80)
        print("[REAL] held-out (OOS+UNSEEN)")
        for w in ["OOS", "UNSEEN", "HELD_OUT"]:
            b = real[w]
            print(f"  {w:8} AUC={b['auc']:.4f}  IC_fwd={b['ic_fwd']:+.4f}  "
                  f"capture_skill plain={b['capture_plain']['skill']:+.3f} regime={b['capture_regime']['skill']:+.3f}  "
                  f"(dna_compound plain={b['capture_plain']['dna_compound_pct']:.1f}%)")
        print("\n[A] SHUFFLED-LABEL CONTROL (permute FIT labels, refit, re-eval held-out)  "
              f"n_shuffle={n_shuffle}")
        print(f"  AUC          real={real_auc:.4f}   shuffled mean={shuffled['auc']['mean']:.4f} "
              f"p95={shuffled['auc']['p95']:.4f} max={shuffled['auc']['max']:.4f}")
        print(f"  IC_fwd       real={real_ic:+.4f}  shuffled mean={shuffled['ic_fwd']['mean']:+.4f} "
              f"p95={shuffled['ic_fwd']['p95']:+.4f}")
        print(f"  cap_skill    real={real_cap_skill:+.3f}  shuffled mean={shuffled['capture_plain_skill']['mean']:+.3f} "
              f"p95={shuffled['capture_plain_skill']['p95']:+.3f}")
        print(f"  -> shuffled collapses to chance? {shuffled_collapses}")
        print("\n[B] POSITIVE CONTROL (planted noisy-but-learnable label)")
        print(f"  held-out AUC={pos_auc:.4f}  has_power(>0.60)={positive_control['has_power']}  "
              f"its shuffled twin p95={positive_control['shuffled_auc']['p95']:.4f} "
              f"collapses={positive_control['shuffled_collapses']}")
        print("\n[C] REGIME-MATCHED FIREWALL (held-out capture vs random-entry null)")
        print(f"  DNA capture plain ={dna_cap_plain_pct:.1f}%   null_plain  p95="
              f"{(null_plain or {}).get('p95_pct', float('nan')):.1f}%  beats={firewall['beats_plain_null_p95']}")
        print(f"  DNA capture regime={dna_cap_reg_pct:.1f}%   null_regime p95="
              f"{(null_regime or {}).get('p95_pct', float('nan')):.1f}%  beats={firewall['beats_regime_null_p95']}")
        print("\n" + "=" * 80)
        print("VERDICT")
        print("=" * 80)
        print(f"  APPARATUS_SOUND (shuffled collapses AND positive control has power) = {apparatus_sound}")
        print(f"  DNA_GENUINE_SIGNAL (real beats shuffled AND beats plain+regime null)= {dna_genuine}")
        if apparatus_sound and not dna_genuine:
            print("  => Falsifier VALID and the SOL-4h MA-DNA does NOT survive: noise / regime-beta, not timing.")
        elif apparatus_sound and dna_genuine:
            print("  => Falsifier VALID and the SOL-4h MA-DNA SURVIVES the shuffled+firewall controls.")
        else:
            print("  => Apparatus NOT sound (shuffled did not collapse OR positive control lacked power) "
                  "-> the DNA proxy / pipeline is LEAKING. Investigate before trusting any DNA number.")

    return result


# ---------------------------------------------------------------------------
def _selftest():
    """No-market synthetic checks of the falsifier's own machinery."""
    print("=" * 70)
    print("[falsifier selftest]")
    print("=" * 70)
    ok = True
    rng = np.random.default_rng(0)
    n = 4000
    # learnable: y depends on feature 0; shuffled must kill held-out AUC
    Xs = rng.normal(0, 1, (n, 5))
    ys = (Xs[:, 0] + 0.3 * rng.normal(0, 1, n) > 0).astype(int)
    tr = slice(0, 3000); ho = slice(3000, n)
    p_real = fit_predict(Xs[tr], ys[tr], Xs[ho], shuffle=False)
    auc_real = roc_auc_score(ys[ho], p_real)
    # single permutations are noisy; the falsifier always uses a DISTRIBUTION -> average many seeds,
    # whose MEAN is the statistically correct "collapses to chance" check.
    auc_shuf = np.mean([roc_auc_score(ys[ho], fit_predict(Xs[tr], ys[tr], Xs[ho], seed=s, shuffle=True))
                        for s in range(20)])
    print(f"  learnable label: AUC real={auc_real:.3f} (expect >0.8)  shuffled<20 seeds mean>={auc_shuf:.3f} (expect ~0.5)")
    ok &= auc_real > 0.8 and abs(auc_shuf - 0.5) < 0.06
    # capture skill: perfect ranking -> ~1, random ranking -> ~0
    fwd = rng.normal(0.0, 0.05, n)
    elig = np.ones(n, bool)
    perfect = capture_skill(fwd, fwd, 0.2, elig)["skill"]           # rank by the truth
    randp = capture_skill(rng.normal(0, 1, n), fwd, 0.2, elig)["skill"]
    print(f"  capture_skill: perfect_rank={perfect:.3f} (expect ~1)  random_rank={randp:+.3f} (expect ~0)")
    ok &= perfect > 0.9 and abs(randp) < 0.4
    # fwd no look-ahead: fwd[i] uses op[i+1] and op[i+1+H] only
    op = np.cumprod(1 + rng.normal(0, 0.01, 50)) * 100
    f = fwd_open_to_open(op, 3)
    manual = op[1 + 3] / op[1] - 1 - COST_RT
    print(f"  fwd_open_to_open[0]={f[0]:.5f} manual={manual:.5f}")
    ok &= abs(f[0] - manual) < 1e-9 and np.isnan(f[-1])
    print(f"\n[falsifier selftest] {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="SOL")
    ap.add_argument("--cadence", default="4h")
    ap.add_argument("--n-shuffle", type=int, default=50)
    ap.add_argument("--n-books", type=int, default=400)
    ap.add_argument("--exit-h", type=int, default=0, help="override exit horizon in bars (0=oracle median hold)")
    ap.add_argument("--min-move-net", type=float, default=0.0,
                    help="per-move net floor for the oracle: 0.0=scalp (2-bar wiggles), 0.03-0.05=SWING (multi-day)")
    ap.add_argument("--model", default="logistic", choices=["logistic", "gbm"],
                    help="logistic (linear baseline) or gbm (nonlinear: MA-crossover interactions + regime conditioning)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if not _selftest():
        print("SELFTEST FAILED -- aborting before touching market data")
        sys.exit(1)
    print()
    res = run(asset=args.asset, cadence=args.cadence, n_shuffle=args.n_shuffle, n_books=args.n_books,
              exit_h=(args.exit_h or None), min_move_net=args.min_move_net, model=args.model)
    tag = f"_swing{int(args.min_move_net*100)}" if args.min_move_net > 0 else ""
    tag += f"_{args.model}" if args.model != "logistic" else ""
    outp = Path(__file__).resolve().parent / f"oracle_dna_shuffled_falsifier_{args.asset}_{args.cadence}{tag}.json"
    outp.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote {outp}")
