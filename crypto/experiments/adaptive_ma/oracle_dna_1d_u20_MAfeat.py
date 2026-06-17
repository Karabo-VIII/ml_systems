"""
experiments/adaptive_ma/oracle_dna_1d_u20_MAfeat.py

1d ORACLE-DNA on a u20 subsample using a STRICT, FAITHFUL causal MA feature set
(1/2/3-MA distance / slope / gap / cross / ribbon), built from price -- NOT the 40 generic
chimera norm_/xd_ columns the sibling runner (oracle_dna_1d_u20_runner.py) uses.

WHY a second runner: the task asks specifically for
    P(oracle-entry | causal 1/2/3-MA distance/slope/gap/cross/ribbon features).
Chimera carries only ONE MA column (norm_ma_distance + xd_ma_distance). So the sibling runner's
"MA-DNA" is really an all-features DNA (a SUPERSET of the MA instrument). This runner isolates the
MA instrument itself: it constructs the 1/2/3-MA distance/slope/gap/cross/ribbon features from close,
all strictly causal (through bar t close; the realizable proxy still enters at open[t+1]), and runs
the SAME audited falsifier controls on them.

REUSES (does NOT re-implement) the audited primitives:
  * runs/research/oracle_ceiling_builder.py :: oracle_high_capture, summarize, COST_RT  (the DP + labels)
  * experiments/adaptive_ma/sol/oracle_dna_shuffled_falsifier.py ::
        load_asset (price+ts), fit_predict, eval_block, capture_skill,
        random_entry_null_capture, fwd_open_to_open, _sma_past, _window_mask, compound
Only the FEATURE MATRIX X changes (chimera norm/xd  ->  strict MA-DNA features).

GATE (per task): capture-rate (beats firewall null) + positive realizable compound, NOT AUC.

RWYB: .venv/Scripts/python.exe experiments/adaptive_ma/oracle_dna_1d_u20_MAfeat.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "runs" / "research"))
sys.path.insert(0, str(ROOT / "experiments" / "adaptive_ma" / "sol"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from sklearn.metrics import roc_auc_score  # noqa: E402
from oracle_ceiling_builder import oracle_high_capture, summarize, COST_RT  # noqa: E402
import oracle_dna_shuffled_falsifier as fal  # noqa: E402

CADENCE = "1d"
U20 = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ZEC", "TRX", "PEPE", "ADA",
       "LINK", "SUI", "AVAX", "TAO", "FET", "ENJ", "ORDI", "NEAR", "WLD", "ENA"]

N_SHUFFLE = 30
N_BOOKS = 200
MA_WINS = (10, 20, 50)       # the "1 / 2 / 3" MAs (fast / mid / slow)
SLOPE_LAG = 3                 # bars over which each MA slope is measured

__contract__ = {
    "kind": "oracle_dna_1d_u20_MAfeat",
    "version": "1.0",
    "inputs": ["price (open/high/close/ts) per u20 asset 1d", "oracle-entry labels from oracle_high_capture"],
    "outputs": ["held-out AUC + shuffled-label control + capture-rate-vs-firewall-null + realizable ceiling, MA features only"],
    "invariants": [
        "features are STRICTLY CAUSAL MA(1/2/3) distance/slope/gap/cross/ribbon built from close (through bar t)",
        "NO target_/high/forward column ever enters X; realizable proxy enters at open[t+1] (no intra-bar high look-ahead)",
        "fit on TRAIN+VAL; evaluate on HELD-OUT (OOS+UNSEEN); shuffle permutes FIT labels only",
        "honest cost 0.0024 round-trip on every proxy move; gate = capture beats firewall AND positive compound",
    ],
}


# ---------------------------------------------------------------------------
def build_ma_features(cl: np.ndarray):
    """STRICT causal 1/2/3-MA distance/slope/gap/cross/ribbon features from close.

    Causality: every MA is rolling/ewm THROUGH bar t (close[t] is known at bar-t close, before the
    realizable open[t+1] entry). No .shift into the future anywhere. Slope/cross use PAST values only.
    Returns (X[n, k], feat_names). Rows with insufficient history are 0 (StandardScaler-safe; the
    held-out windows are all far past the warm-up)."""
    s = pd.Series(cl, dtype="float64")
    emas = {w: s.ewm(span=w, adjust=False).mean() for w in MA_WINS}
    smas = {w: s.rolling(w, min_periods=w).mean() for w in MA_WINS}

    feats = {}
    # (a) DISTANCE: price vs each MA (EMA and SMA families) -- "1/2/3-MA distance"
    for w in MA_WINS:
        feats[f"ema{w}_dist"] = (s / emas[w] - 1.0)
        feats[f"sma{w}_dist"] = (s / smas[w] - 1.0)
    # (b) SLOPE: each MA's own momentum over SLOPE_LAG bars (past-only by construction)
    for w in MA_WINS:
        feats[f"ema{w}_slope"] = (emas[w] / emas[w].shift(SLOPE_LAG) - 1.0)
    # (c) GAP: pairwise MA gaps (fast-mid, mid-slow, fast-slow)
    feats["gap_fast_mid"] = (emas[MA_WINS[0]] / emas[MA_WINS[1]] - 1.0)
    feats["gap_mid_slow"] = (emas[MA_WINS[1]] / emas[MA_WINS[2]] - 1.0)
    feats["gap_fast_slow"] = (emas[MA_WINS[0]] / emas[MA_WINS[2]] - 1.0)
    # (d) CROSS: fast-vs-slow state + recency of the last sign flip (capped, normalized past-only)
    diff_fs = (emas[MA_WINS[0]] - emas[MA_WINS[2]])
    cross_state = np.sign(diff_fs.to_numpy())
    feats["cross_state"] = pd.Series(cross_state, index=s.index)
    sign = np.sign(diff_fs.to_numpy())
    bars_since = np.zeros(len(sign))
    cnt = 0
    last = 0.0
    for i in range(len(sign)):
        if i == 0 or sign[i] == 0 or sign[i] == last:
            cnt += 1
        else:
            cnt = 0
        if sign[i] != 0:
            last = sign[i]
        bars_since[i] = cnt
    feats["bars_since_cross"] = pd.Series(np.minimum(bars_since, 50.0) / 50.0, index=s.index)
    # (e) RIBBON: normalized spread of the 3 MAs + bullish/bearish stacking alignment
    stack = np.vstack([emas[w].to_numpy() for w in MA_WINS])
    width = (np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0)) / np.where(cl == 0, np.nan, cl)
    feats["ribbon_width"] = pd.Series(width, index=s.index)
    e_f, e_m, e_s = (emas[w].to_numpy() for w in MA_WINS)
    aligned = np.where((e_f > e_m) & (e_m > e_s), 1.0, np.where((e_f < e_m) & (e_m < e_s), -1.0, 0.0))
    feats["ribbon_aligned"] = pd.Series(aligned, index=s.index)

    names = list(feats.keys())
    X = np.column_stack([feats[k].to_numpy().astype(np.float64) for k in names])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names


# ---------------------------------------------------------------------------
def per_asset(asset):
    t0 = time.time()
    # reuse the audited loader for price+ts (ignore its chimera X; we build MA features)
    ts, op, hi, cl, _Xchim, _feats = fal.load_asset(asset, CADENCE)
    n = len(op)

    # MA-DNA feature matrix (the instrument under test)
    X, ma_names = build_ma_features(cl)

    # labels: oracle-entry from the audited perfect-foresight DP
    f_dp, trades = oracle_high_capture(ts, op, hi)
    if not trades:
        return {"asset": asset, "error": "no oracle trades"}
    y = np.zeros(n, dtype=int)
    ent = np.array([i for i, j in trades], dtype=int)
    y[ent] = 1
    holds = np.array([j - i for i, j in trades], dtype=int)
    H = int(max(1, np.median(holds)))
    s_orc = summarize(ts, trades, op, hi)

    # honest forward proxy + past-only SMA-200 regime
    fwd = fal.fwd_open_to_open(op, H, COST_RT)
    sma200 = fal._sma_past(cl, 200)
    uptrend = (cl > sma200)

    # splits: FIT=TRAIN+VAL, HELD-OUT=OOS+UNSEEN
    m_tr = fal._window_mask(ts, "TRAIN") | fal._window_mask(ts, "VAL")
    m_oos = fal._window_mask(ts, "OOS")
    m_uns = fal._window_mask(ts, "UNSEEN")
    m_ho = m_oos | m_uns
    Xtr, ytr = X[m_tr], y[m_tr]
    k_frac = float(ytr.mean()) if ytr.mean() > 0 else float(y.mean())

    elig_all = ~np.isnan(fwd)
    elig_reg = elig_all & np.nan_to_num(uptrend, nan=False).astype(bool)

    # (1) REAL fit on MA features
    p_real = fal.fit_predict(Xtr, ytr, X, seed=7, shuffle=False)
    real_ho = fal.eval_block(p_real[m_ho], y[m_ho], fwd[m_ho], k_frac, elig_all[m_ho], elig_reg[m_ho])
    real_auc = real_ho["auc"]
    real_ic = real_ho["ic_fwd"]
    cap_plain = real_ho["capture_plain"]
    cap_regime = real_ho["capture_regime"]

    # (A) SHUFFLED-LABEL CONTROL
    sh_auc, sh_cap = [], []
    for sd in range(N_SHUFFLE):
        p_s = fal.fit_predict(Xtr, ytr, X, seed=1000 + sd, shuffle=True)
        b = fal.eval_block(p_s[m_ho], y[m_ho], fwd[m_ho], k_frac, elig_all[m_ho], elig_reg[m_ho])
        sh_auc.append(b["auc"]); sh_cap.append(b["capture_plain"]["skill"])

    def dist(a):
        a = np.array([x for x in a if not (isinstance(x, float) and np.isnan(x))], float)
        if len(a) == 0:
            return {"mean": float("nan"), "p05": float("nan"), "p95": float("nan")}
        return {"mean": float(a.mean()), "p05": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95))}

    sh_auc_d = dist(sh_auc)
    sh_cap_d = dist(sh_cap)

    # (B) POSITIVE CONTROL: plant a noisy-but-learnable label from MA features (slope+gap), prove power.
    # z-score the (small-magnitude) MA columns FIRST so the planted signal is not drowned by unit noise
    # (an un-scaled raw distance/slope ~0.01 vs noise ~1.0 would make the planted label ~pure noise and
    # spuriously read as "no power"). Noise kept < signal so a genuine learnable MA label IS findable.
    rng = np.random.default_rng(123)
    def zcol(name):
        v = X[:, ma_names.index(name)] if name in ma_names else np.zeros(n)
        sd = np.std(v[m_tr]) if np.std(v[m_tr]) > 1e-9 else 1.0
        return (v - np.mean(v[m_tr])) / sd
    signal = 1.2 * zcol("ema10_slope") + 1.2 * zcol("gap_fast_slow") + rng.normal(0, 0.7, n)
    y_pos = (signal > np.median(signal[m_tr])).astype(int)
    p_pos = fal.fit_predict(Xtr, y_pos[m_tr], X, seed=7, shuffle=False)
    pos_auc = float(roc_auc_score(y_pos[m_ho], p_pos[m_ho]))
    pos_sh = []
    for sd in range(min(N_SHUFFLE, 30)):
        p_ps = fal.fit_predict(Xtr, y_pos[m_tr], X, seed=2000 + sd, shuffle=True)
        pos_sh.append(float(roc_auc_score(y_pos[m_ho], p_ps[m_ho])))
    pos_sh_d = dist(pos_sh)

    # (C) REGIME-MATCHED FIREWALL null on capture (held-out)
    null_plain = fal.random_entry_null_capture(fwd[m_ho], elig_all[m_ho], k_frac, N_BOOKS, 7)
    null_regime = fal.random_entry_null_capture(fwd[m_ho], elig_reg[m_ho], k_frac, N_BOOKS, 8)
    dna_cap_plain_pct = cap_plain["dna_compound_pct"]
    dna_cap_reg_pct = cap_regime["dna_compound_pct"]
    beats_plain = bool(null_plain is not None and dna_cap_plain_pct > null_plain["p95_pct"])
    beats_regime = bool(null_regime is not None and dna_cap_reg_pct > null_regime["p95_pct"])

    # VERDICTS (gate = capture beats firewall AND positive realizable compound; NOT AUC)
    shuffled_collapses = bool(sh_auc_d["p95"] < 0.55 and sh_cap_d["p95"] < 0.15)
    apparatus_sound_mean = bool(sh_auc_d["mean"] < 0.55 and pos_auc > 0.60 and pos_sh_d["mean"] < 0.55)
    dna_auc_beats_shuffled = bool(real_auc > sh_auc_d["p95"] and real_auc > 0.53)
    capture_gate = bool(beats_plain and beats_regime and dna_cap_plain_pct > 0.0)

    return {
        "asset": asset, "n_bars": n, "n_ma_features": len(ma_names), "ma_features": ma_names,
        "oracle_n_trades": int(len(trades)),
        "oracle_hold_bars_median": float(np.median(holds)),
        "oracle_base_rate": float(y.mean()), "exit_H_bars": H,
        "oracle_compound_pct": s_orc.get("total_capturable_compound_pct"),
        # held-out MA-DNA skill
        "held_out_auc_real": real_auc,
        "held_out_auc_shuffled_mean": sh_auc_d["mean"],
        "held_out_auc_shuffled_p95": sh_auc_d["p95"],
        "held_out_ic_fwd": real_ic,
        # capture-rate + firewall
        "capture_skill_heldout": cap_plain["skill"],
        "capture_skill_regime": cap_regime["skill"],
        "dna_capture_compound_pct": dna_cap_plain_pct,
        "dna_capture_regime_compound_pct": dna_cap_reg_pct,
        "null_plain_p95_pct": (null_plain or {}).get("p95_pct"),
        "null_regime_p95_pct": (null_regime or {}).get("p95_pct"),
        "beats_plain_null_p95": beats_plain, "beats_regime_null_p95": beats_regime,
        # realizable ceiling (perfect selection mean-net per move on held-out)
        "realizable_ceiling_mean_net_pct": cap_plain.get("best_mean_net_pct"),
        "dna_mean_net_pct": cap_plain.get("dna_mean_net_pct"),
        "chance_mean_net_pct": cap_plain.get("chance_mean_net_pct"),
        # positive control (two-sided soundness on MA features)
        "positive_control_auc": pos_auc, "positive_control_has_power": bool(pos_auc > 0.60),
        "positive_control_shuffled_mean": pos_sh_d["mean"],
        "verdict": {
            "apparatus_sound_mean_criterion": apparatus_sound_mean,
            "shuffled_collapses_p95": shuffled_collapses,
            "dna_auc_beats_shuffled": dna_auc_beats_shuffled,
            "CAPTURE_GATE_PASS": capture_gate,                 # the task's gate
            "DNA_GENUINE_SIGNAL": bool(dna_auc_beats_shuffled and capture_gate),
        },
        "secs": round(time.time() - t0, 1),
    }


def _agg(vals):
    a = np.asarray([v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))], float)
    if len(a) == 0:
        return {"n": 0}
    return {"n": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)),
            "min": float(a.min()), "max": float(a.max())}


def main():
    t_start = time.time()
    print(f"[1d ORACLE-DNA u20 -- STRICT MA FEATURES] assets={len(U20)}  MA_wins={MA_WINS}  "
          f"n_shuffle={N_SHUFFLE}  n_books={N_BOOKS}  cost_rt={COST_RT}")
    results = []
    for a in U20:
        try:
            r = per_asset(a)
        except Exception as e:
            r = {"asset": a, "error": repr(e)[:200]}
        results.append(r)
        if "error" in r:
            print(f"  {a:6} ERROR: {r['error']}")
        else:
            print(f"  {a:6} bars={r['n_bars']:5} nMA={r['n_ma_features']:2} "
                  f"AUC real={r['held_out_auc_real']:.3f} shufp95={r['held_out_auc_shuffled_p95']:.3f} "
                  f"cap_skill={r['capture_skill_heldout']:+.3f} "
                  f"capComp={r['dna_capture_compound_pct']:+.1f}% null_p95={r['null_plain_p95_pct']:+.1f}% "
                  f"GATE={r['verdict']['CAPTURE_GATE_PASS']} ({r['secs']}s)")

    ok = [r for r in results if "error" not in r]
    agg = {
        "n_assets_ok": len(ok), "n_assets_err": len(results) - len(ok),
        "held_out_auc_real": _agg([r["held_out_auc_real"] for r in ok]),
        "held_out_auc_shuffled_mean": _agg([r["held_out_auc_shuffled_mean"] for r in ok]),
        "held_out_ic_fwd": _agg([r["held_out_ic_fwd"] for r in ok]),
        "capture_skill_heldout": _agg([r["capture_skill_heldout"] for r in ok]),
        "dna_capture_compound_pct": _agg([r["dna_capture_compound_pct"] for r in ok]),
        "realizable_ceiling_mean_net_pct": _agg([r["realizable_ceiling_mean_net_pct"] for r in ok]),
        "positive_control_auc": _agg([r["positive_control_auc"] for r in ok]),
        "n_capture_compound_positive": int(sum(1 for r in ok if r["dna_capture_compound_pct"] > 0)),
        "n_beats_plain_null_p95": int(sum(1 for r in ok if r["beats_plain_null_p95"])),
        "n_beats_regime_null_p95": int(sum(1 for r in ok if r["beats_regime_null_p95"])),
        "n_capture_gate_pass": int(sum(1 for r in ok if r["verdict"]["CAPTURE_GATE_PASS"])),
        "n_auc_beats_shuffled_p95": int(sum(1 for r in ok if r["verdict"]["dna_auc_beats_shuffled"])),
        "n_dna_genuine": int(sum(1 for r in ok if r["verdict"]["DNA_GENUINE_SIGNAL"])),
        "n_apparatus_sound_mean": int(sum(1 for r in ok if r["verdict"]["apparatus_sound_mean_criterion"])),
        "n_positive_control_power": int(sum(1 for r in ok if r["positive_control_has_power"])),
    }

    blob = {
        "meta": {"cadence": CADENCE, "universe": "u20 (first 20 of u50)", "assets": U20,
                 "feature_set": "STRICT causal MA(1/2/3) distance/slope/gap/cross/ribbon (built from close)",
                 "ma_wins": list(MA_WINS), "slope_lag": SLOPE_LAG,
                 "cost_rt": COST_RT, "n_shuffle": N_SHUFFLE, "n_books": N_BOOKS,
                 "gate": "capture beats plain+regime firewall null AND positive realizable compound (NOT AUC)",
                 "reused": ["oracle_ceiling_builder.oracle_high_capture/summarize",
                            "sol/oracle_dna_shuffled_falsifier.{load_asset,fit_predict,eval_block,"
                            "capture_skill,random_entry_null_capture,fwd_open_to_open}"],
                 "elapsed_secs": round(time.time() - t_start, 1)},
        "aggregate": agg, "per_asset": results,
    }
    outp = ROOT / "experiments" / "adaptive_ma" / "oracle_dna_1d_u20_MAfeat.json"
    outp.write_text(json.dumps(blob, indent=2), encoding="utf-8")

    print("\n" + "=" * 92)
    print("U20 1d AGGREGATE -- STRICT MA FEATURES")
    print("=" * 92)
    print(f"  assets ok={agg['n_assets_ok']}  err={agg['n_assets_err']}  elapsed={blob['meta']['elapsed_secs']}s")
    print(f"  held-out AUC real:   mean={agg['held_out_auc_real'].get('mean'):.4f} "
          f"median={agg['held_out_auc_real'].get('median'):.4f} max={agg['held_out_auc_real'].get('max'):.4f}")
    print(f"  held-out AUC shuf mean (collapse check): {agg['held_out_auc_shuffled_mean'].get('mean'):.4f}")
    print(f"  held-out IC_fwd:     mean={agg['held_out_ic_fwd'].get('mean'):+.4f} "
          f"median={agg['held_out_ic_fwd'].get('median'):+.4f}")
    print(f"  capture_skill:       mean={agg['capture_skill_heldout'].get('mean'):+.4f} "
          f"median={agg['capture_skill_heldout'].get('median'):+.4f} max={agg['capture_skill_heldout'].get('max'):+.4f}")
    print(f"  capture compound %:  mean={agg['dna_capture_compound_pct'].get('mean'):+.1f} "
          f"median={agg['dna_capture_compound_pct'].get('median'):+.1f}  "
          f"positive={agg['n_capture_compound_positive']}/{agg['n_assets_ok']}")
    print(f"  realizable ceiling mean-net/move %: mean={agg['realizable_ceiling_mean_net_pct'].get('mean'):.2f}")
    print(f"  positive-control AUC mean={agg['positive_control_auc'].get('mean'):.3f} "
          f"(power {agg['n_positive_control_power']}/{agg['n_assets_ok']})  "
          f"apparatus_sound_mean={agg['n_apparatus_sound_mean']}/{agg['n_assets_ok']}")
    print(f"  firewall: beats PLAIN null p95={agg['n_beats_plain_null_p95']}/{agg['n_assets_ok']}   "
          f"beats REGIME null p95={agg['n_beats_regime_null_p95']}/{agg['n_assets_ok']}  (regime = binding firewall)")
    print(f"  AUC>shuffled_p95: {agg['n_auc_beats_shuffled_p95']}/{agg['n_assets_ok']}   "
          f"CAPTURE_GATE_PASS: {agg['n_capture_gate_pass']}/{agg['n_assets_ok']}   "
          f"DNA_GENUINE: {agg['n_dna_genuine']}/{agg['n_assets_ok']}")
    print(f"\n[OK] wrote {outp}")
    return blob


if __name__ == "__main__":
    main()
