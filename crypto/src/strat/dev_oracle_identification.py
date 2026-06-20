"""src/strat/dev_oracle_identification.py -- DEV-WALLED oracle ceiling + honest capture + identification AUC.

TASK (ORC decisive cycle 2026-06-20):
  On DEV data (<= 2024-05-15) via fleet_lab.load_wide():
  1. Build the move-capture ORACLE: best holdable top-K per 7d slice = ceiling.
  2. Compute HONEST AGGREGATE capture rate: sum(realized) / sum(oracle_available) -- dollar-weighted.
     NOT mean(eng/oracle) which is broken by small-denominator blocks.
  3. IDENTIFICATION AUC: does the mover composite score (mom7/mom14/mom30/rsi14/brk14/rangepos/volexp/accel +
     chimera vpin/ofi/dev/fdclose/dvol) predict oracle-top-K membership? AUC > 0.52 = identifiable ex-ante.
  4. REGIME STRATIFICATION: bull/chop/bear (BTC vs 200d SMA + breadth) -- does selection alpha vanish in
     non-bull regimes? Vanishing = bull-beta artifact. Holding across regimes = REAL selection alpha.

DATA WALL: DEV_END = 2024-05-15. load_wide() enforces this. No OOS/UNSEEN touched.
Long-only spot, taker 0.0024 RT, causal (shift-1 features). K=5 primary, K=3 secondary.
No emoji (cp1252). Does NOT git commit.

Run: python -m strat.dev_oracle_identification
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRYPTO = ROOT.parent

from strat.fleet_lab import load_wide, slice_dates, DEV_END, COST

HORIZON = 7    # 7-day hold
K_VALS  = [3, 5]
N_SLICES = 300   # random DEV slices for profitability test
SEEDS = [11, 23, 42]
WARMUP = 40   # minimum bars before evaluating


# ============================================================
# MOVER COMPOSITE SCORE (vectorized, all features shift-1 causal)
# ============================================================
def build_mover_composite(lab: dict) -> pd.DataFrame:
    """
    Cross-sectional composite z-score per bar from all available TI + chimera features.
    Subset used: mom7, mom14, brk14, rangepos, volexp, accel (price TIs)
                + vpin, ofi, dev (chimera, if present).
    All features already shift-1 causal from fleet_lab. Additional shift not needed.
    Returns DataFrame [dates x syms] with composite score (higher = more mover-like).
    """
    F = lab["F"]
    C = lab["C"]
    # Signs: positive = mover-favoring
    # mom7/mom14/brk14/rangepos/volexp/accel: high = strong mover
    # vpin: high toxicity = big move likely (positive sign)
    # ofi: high OFI = buy pressure = positive
    # dev: high deviation = stretched = potential mean-rev, sign is ambiguous; use raw (positive = up deviation)
    # fdclose: forward-looking proxy for activity; use positive
    feature_signs = {
        "mom7":     +1,
        "mom14":    +1,
        "mom30":    +1,
        "brk14":    +1,
        "rangepos": +1,
        "volexp":   +1,
        "accel":    +1,
        "vpin":     +1,
        "ofi":      +1,
        "dev":      +1,
        "fdclose":  +1,
        "dvol":     +1,
        "rsi14":    +1,
    }
    parts = []
    used = []
    for feat, sgn in feature_signs.items():
        if feat not in F:
            continue
        df = F[feat].reindex(index=C.index, columns=C.columns)
        # cross-sectional z-score per row
        mu = df.mean(axis=1)
        sd = df.std(axis=1)
        z = df.sub(mu, axis=0).div(sd + 1e-12, axis=0)
        z = z.where(sd > 1e-9, 0.0)
        parts.append(sgn * z)
        used.append(feat)
    if not parts:
        raise RuntimeError("No features available for composite score")
    comp = sum(parts) / len(parts)
    return comp, used


# ============================================================
# REGIME CLASSIFIER (BTC vs 200d SMA + breadth)
# ============================================================
def build_regime(lab: dict) -> pd.Series:
    """
    'bull': BTC > SMA200 + breadth > 0.5
    'chop': BTC > SMA200 + breadth 0.3-0.5, OR mixed BTC
    'bear': BTC < SMA200
    Returns Series indexed by date.
    """
    C = lab["C"]
    if "BTCUSDT" not in C.columns:
        return pd.Series("unknown", index=C.index)
    btc = C["BTCUSDT"]
    sma200 = btc.rolling(200, min_periods=100).mean()
    btc_up = btc > sma200
    sma50 = C.rolling(50, min_periods=25).mean()
    above50 = (C > sma50).sum(axis=1)
    total = C.notna().sum(axis=1)
    breadth = above50 / total.replace(0, np.nan)
    breadth = breadth.fillna(0.5)
    regime = pd.Series("chop", index=C.index)
    regime[btc_up & (breadth >= 0.5)] = "bull"
    regime[~btc_up] = "bear"
    return regime


# ============================================================
# ORACLE: best single-asset 7d forward return per block
# ============================================================
def build_oracle_series(lab: dict, horizon: int = HORIZON) -> tuple[pd.Series, pd.Series]:
    """
    Non-overlapping 7d blocks starting from WARMUP.
    Returns:
      oracle_series: Series indexed by block-start-date, value = best available forward return
      oracle_topK_sets: dict {start_date: set of oracle-top-K symbols}
    """
    C = lab["C"]
    idx = C.index
    fwd = (C.shift(-horizon) / C - 1)
    oracle_best_1 = {}
    oracle_topK = {}   # will be filled for K=3 and K=5
    for si in range(WARMUP, len(idx) - horizon, horizon):
        d = idx[si]
        row = fwd.iloc[si].dropna()
        if len(row) < 3:
            continue
        oracle_best_1[d] = float(row.max())
        oracle_topK[d] = {k: set(row.sort_values(ascending=False).index[:k]) for k in K_VALS}
    return oracle_best_1, oracle_topK


# ============================================================
# ENGINE: top-K by composite score per block-start
# ============================================================
def build_engine_returns(lab: dict, comp: pd.DataFrame, K: int,
                         horizon: int = HORIZON, random_seed: int | None = None) -> dict:
    """
    Non-overlapping 7d blocks. At each block start, pick top-K by composite score (causal).
    If random_seed != None: pick K at random (same-exposure shuffle control).
    Returns dict {start_date: (realized_fwd_ret, [picks])} cost-adjusted.
    """
    C = lab["C"]
    idx = C.index
    rng = np.random.default_rng(random_seed) if random_seed is not None else None
    results = {}
    for si in range(WARMUP, len(idx) - horizon, horizon):
        d = idx[si]
        row_score = comp.iloc[si]
        valid_syms = [s for s in C.columns if pd.notna(row_score[s]) and pd.notna(C[s].iloc[si]) and C[s].iloc[si] > 0]
        if len(valid_syms) < K:
            continue
        if rng is not None:
            picks = list(rng.choice(valid_syms, size=K, replace=False))
        else:
            picks = sorted(valid_syms, key=lambda s: -float(row_score[s]))[:K]
        # Realized: mean of compound returns over horizon bars (hold entry at bar si, exit at si+horizon)
        realized_rets = []
        for s in picks:
            prices = [C[s].iloc[si + j] for j in range(horizon + 1)]
            if any(pd.isna(p) or p <= 0 for p in prices):
                continue
            # Daily compounding
            daily_rets = [(prices[j+1] / prices[j] - 1) for j in range(horizon)]
            # Net of RT cost: taker in + taker out split equally across position
            compound = float(np.prod([1 + r for r in daily_rets]) - 1)
            realized_rets.append(compound - COST)  # net of full RT cost (in+out)
        if not realized_rets:
            continue
        results[d] = (float(np.mean(realized_rets)), picks)
    return results


# ============================================================
# HONEST AGGREGATE CAPTURE RATE
# ============================================================
def aggregate_capture(engine_rets: dict, oracle_best: dict) -> dict:
    """
    Dollar-weighted aggregate: sum(realized) / sum(oracle_available).
    Only include blocks where oracle_best > 0.5% (there IS a move to capture).
    """
    sum_real = 0.0; sum_oracle = 0.0; n = 0
    per_block_ratio = []
    for d, (real, _) in engine_rets.items():
        if d not in oracle_best:
            continue
        o = oracle_best[d]
        if o > 0.005:
            sum_real += real
            sum_oracle += o
            n += 1
            per_block_ratio.append(real / o)
    if n == 0 or sum_oracle == 0:
        return {"aggregate_pct": None, "ratio_mean_pct": None, "n": 0}
    cr = np.array(per_block_ratio)
    return {
        "aggregate_pct": round(100 * sum_real / sum_oracle, 2),     # HONEST dollar-weighted
        "ratio_mean_pct": round(float(cr.mean()) * 100, 2),          # BROKEN (dominated by small denom)
        "ratio_median_pct": round(float(np.median(cr)) * 100, 2),
        "pct_positive_blocks": round(float((cr > 0).mean()) * 100, 1),
        "n_blocks": n,
        "sum_real_pct": round(100 * sum_real, 2),
        "sum_oracle_pct": round(100 * sum_oracle, 2),
    }


# ============================================================
# AUC: mover score predicts oracle-top-K membership
# ============================================================
def compute_identification_auc(lab: dict, comp: pd.DataFrame,
                                oracle_topK_sets: dict, K: int) -> dict:
    """
    For each block-start bar, treat oracle-top-K as positives (label=1).
    The composite score for each asset is the predictor.
    AUC = fraction of (positive, negative) pairs where score[positive] > score[negative].
    Only computed on bars where we have valid oracle-top-K and enough valid scores.
    """
    C = lab["C"]
    idx = C.index
    all_labels = []
    all_scores = []
    for d, topK_dict in oracle_topK_sets.items():
        if K not in topK_dict:
            continue
        positives = topK_dict[K]
        si = idx.get_loc(d)
        row = comp.iloc[si]
        valid = {s: float(row[s]) for s in C.columns if pd.notna(row[s]) and pd.notna(C[s].iloc[si])}
        if len(valid) < K + 2:
            continue
        for s, sc in valid.items():
            label = 1 if s in positives else 0
            all_labels.append(label)
            all_scores.append(sc)
    labels = np.array(all_labels, dtype=float)
    scores = np.array(all_scores, dtype=float)
    if labels.sum() < 5 or (labels == 0).sum() < 5:
        return {"auc": None, "n_pairs": 0}
    # Efficient AUC computation via rank
    from scipy.stats import mannwhitneyu
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    try:
        u_stat, p_val = mannwhitneyu(pos_scores, neg_scores, alternative="greater")
        n_pos = len(pos_scores); n_neg = len(neg_scores)
        auc = float(u_stat) / (n_pos * n_neg)
    except Exception:
        auc = float(np.mean([float(np.mean(neg_scores < ps)) for ps in pos_scores]))
        p_val = float("nan")
    return {
        "auc": round(auc, 4),
        "p_val": round(float(p_val), 6) if not np.isnan(p_val) else None,
        "n_pos_obs": int(labels.sum()),
        "n_neg_obs": int((labels == 0).sum()),
        "n_blocks": len(oracle_topK_sets),
    }


# ============================================================
# SELECTION ALPHA vs SHUFFLE CONTROL (regime-stratified)
# ============================================================
def regime_stratified_alpha(lab: dict, engine_rets: dict, shuffle_rets_list: list,
                             regime: pd.Series) -> dict:
    """
    Split blocks by regime. For each regime:
      engine mean return vs shuffle control mean return -> selection alpha.
    """
    results = {}
    for reg in ["bull", "chop", "bear"]:
        reg_dates = set(d for d in engine_rets.keys() if regime.get(d, "unknown") == reg)
        eng = [engine_rets[d][0] for d in reg_dates if d in engine_rets]
        shuf_vals = []
        for shuf_rets in shuffle_rets_list:
            shuf = [shuf_rets[d][0] for d in reg_dates if d in shuf_rets]
            if shuf:
                shuf_vals.append(float(np.mean(shuf)))
        shuf_mean = float(np.mean(shuf_vals)) if shuf_vals else float("nan")
        eng_mean = float(np.mean(eng)) if eng else float("nan")
        results[reg] = {
            "n_blocks": len(eng),
            "engine_mean_pct": round(100 * eng_mean, 3) if not np.isnan(eng_mean) else None,
            "shuffle_mean_pct": round(100 * shuf_mean, 3) if not np.isnan(shuf_mean) else None,
            "selection_alpha_pp": round(100 * (eng_mean - shuf_mean), 3) if not np.isnan(eng_mean + shuf_mean) else None,
        }
    return results


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    print("=" * 76)
    print("DEV ORACLE + IDENTIFICATION (DEV-WALLED <= 2024-05-15)")
    print(f"  K_VALS={K_VALS} HORIZON={HORIZON}d N_SLICES={N_SLICES} SEEDS={SEEDS}")
    print("=" * 76)

    # 1. Load DEV data
    print("\n[1] Loading DEV data via fleet_lab.load_wide(n=50)...")
    lab = load_wide(n=50)
    C = lab["C"]
    print(f"    {len(lab['syms'])} assets; {C.index.min().date()} -> {C.index.max().date()} (DEV_END={DEV_END})")
    assert C.index.max() < pd.Timestamp(DEV_END), "WALL VIOLATION"
    print(f"    Features: {list(lab['F'].keys())}")

    # 2. Build composite score
    print("\n[2] Building mover composite score...")
    comp, used_feats = build_mover_composite(lab)
    print(f"    Used features: {used_feats}")
    print(f"    comp shape: {comp.shape}, non-null: {comp.notna().sum().sum()}")

    # 3. Regime
    print("\n[3] Building regime series...")
    regime = build_regime(lab)
    reg_counts = regime.value_counts().to_dict()
    print(f"    Regime counts: {reg_counts}")

    # 4. Oracle ceiling
    print("\n[4] Building oracle (best available top-K per 7d block)...")
    oracle_best, oracle_topK_sets = build_oracle_series(lab)
    n_blocks = len(oracle_best)
    orc_arr = np.array(list(oracle_best.values()))
    print(f"    {n_blocks} non-overlapping 7d blocks; oracle best-1 mean={100*orc_arr.mean():.2f}% "
          f"median={100*np.median(orc_arr):.2f}% pos={100*np.mean(orc_arr>0):.1f}%")

    # Regime breakdown of oracle
    print("    Oracle by regime:")
    for reg in ["bull", "chop", "bear"]:
        reg_vals = [oracle_best[d] for d in oracle_best if regime.get(d, "unknown") == reg]
        if reg_vals:
            print(f"      {reg:5s}: n={len(reg_vals)} mean={100*np.mean(reg_vals):.2f}% "
                  f"pos={100*np.mean(np.array(reg_vals)>0):.1f}%")

    results = {
        "dev_end": DEV_END,
        "n_assets": len(lab["syms"]),
        "date_range": [str(C.index.min().date()), str(C.index.max().date())],
        "used_features": used_feats,
        "horizon": HORIZON,
        "regime_counts": {k: int(v) for k, v in reg_counts.items()},
        "oracle": {
            "n_blocks": n_blocks,
            "mean_best_pct": round(100 * float(orc_arr.mean()), 2),
            "median_best_pct": round(100 * float(np.median(orc_arr)), 2),
            "pct_positive": round(100 * float(np.mean(orc_arr > 0)), 1),
        },
        "engine": {},
    }

    for K in K_VALS:
        print(f"\n[5] ENGINE K={K} -- building block returns...")
        eng_rets = build_engine_returns(lab, comp, K)
        print(f"    {len(eng_rets)} blocks with valid picks")

        # Shuffle controls (3 seeds)
        print(f"    Building shuffle controls (same exposure, random K picks)...")
        shuffle_list = []
        for ctrl_seed in [101, 202, 303]:
            shuf = build_engine_returns(lab, comp, K, random_seed=ctrl_seed)
            shuffle_list.append(shuf)

        # --- HONEST AGGREGATE CAPTURE RATE ---
        print(f"\n[5a] Honest aggregate capture rate K={K}:")
        cap = aggregate_capture(eng_rets, oracle_best)
        ctrl_caps = [aggregate_capture(sh, oracle_best)["aggregate_pct"] for sh in shuffle_list]
        ctrl_cap_mean = float(np.mean([c for c in ctrl_caps if c is not None]))
        print(f"     ENGINE aggregate_pct = {cap['aggregate_pct']}%  (honest dollar-weighted)")
        print(f"     ENGINE ratio_mean_pct = {cap['ratio_mean_pct']}%  (broken estimator for reference)")
        print(f"     SHUFFLE ctrl aggregate = {ctrl_caps}  mean={ctrl_cap_mean:.2f}%")
        print(f"     n_blocks_with_move = {cap['n_blocks']} | pct_positive_blocks={cap['pct_positive_blocks']}%")
        print(f"     sum_realized={cap['sum_real_pct']}%  sum_oracle={cap['sum_oracle_pct']}%")

        # --- IDENTIFICATION AUC ---
        print(f"\n[5b] Identification AUC (mover score predicts oracle-top-{K} membership):")
        auc_res = compute_identification_auc(lab, comp, oracle_topK_sets, K)
        print(f"     AUC = {auc_res['auc']}  (p={auc_res['p_val']})  "
              f"n_pos={auc_res['n_pos_obs']} n_neg={auc_res['n_neg_obs']}")
        if auc_res['auc'] is not None:
            if auc_res['auc'] > 0.55:
                print(f"     IDENTIFIABLE: AUC {auc_res['auc']} >> 0.52 -- mover score predicts oracle membership")
            elif auc_res['auc'] > 0.52:
                print(f"     MARGINAL: AUC {auc_res['auc']} slightly above 0.52")
            else:
                print(f"     NOT IDENTIFIABLE: AUC {auc_res['auc']} ~ 0.5 = random on DEV")

        # --- REGIME STRATIFICATION ---
        print(f"\n[5c] Regime-stratified selection alpha K={K}:")
        reg_alpha = regime_stratified_alpha(lab, eng_rets, shuffle_list, regime)
        for reg, rv in reg_alpha.items():
            verdict = ""
            if rv["selection_alpha_pp"] is not None:
                if rv["selection_alpha_pp"] > 0.5:
                    verdict = "[POSITIVE ALPHA]"
                elif rv["selection_alpha_pp"] < -0.5:
                    verdict = "[NEGATIVE ALPHA]"
                else:
                    verdict = "[FLAT]"
            print(f"     {reg:5s}: n={rv['n_blocks']:3d}  eng={rv['engine_mean_pct']}%  "
                  f"shuf={rv['shuffle_mean_pct']}%  alpha={rv['selection_alpha_pp']}pp  {verdict}")

        # Check for bull-beta artifact
        if reg_alpha.get("bull", {}).get("selection_alpha_pp") and reg_alpha.get("bear", {}).get("selection_alpha_pp"):
            bull_alpha = reg_alpha["bull"]["selection_alpha_pp"]
            bear_alpha = reg_alpha["bear"]["selection_alpha_pp"]
            if bull_alpha is not None and bear_alpha is not None:
                if bull_alpha > 0 and bear_alpha <= 0:
                    print(f"     *** BULL-BETA ARTIFACT: alpha present in bull ({bull_alpha:.2f}pp) but vanishes/negative in bear ({bear_alpha:.2f}pp) ***")
                elif bull_alpha > 0 and bear_alpha > 0:
                    print(f"     *** ROBUST SELECTION: alpha present in BOTH bull ({bull_alpha:.2f}pp) AND bear ({bear_alpha:.2f}pp) ***")

        # EW buy-hold baseline (for context)
        print(f"\n[5d] EW buy-hold baseline on same blocks (K={K}):")
        ew_rets = []
        for d, (_, picks) in eng_rets.items():
            # EW over all valid assets at that bar
            si = C.index.get_loc(d)
            row_rets = []
            for s in C.columns:
                if si + HORIZON < len(C.index) and pd.notna(C[s].iloc[si]) and pd.notna(C[s].iloc[si + HORIZON]):
                    raw = C[s].iloc[si + HORIZON] / C[s].iloc[si] - 1
                    row_rets.append(float(raw) - COST)
            if row_rets:
                ew_rets.append(float(np.mean(row_rets)))
        ew_mean = float(np.mean(ew_rets)) if ew_rets else float("nan")
        eng_mean_all = float(np.mean([v[0] for v in eng_rets.values()]))
        print(f"     EW mean = {100*ew_mean:.3f}%  ENGINE mean = {100*eng_mean_all:.3f}%  "
              f"alpha = {100*(eng_mean_all - ew_mean):.3f}pp")

        results["engine"][f"K{K}"] = {
            "n_blocks": len(eng_rets),
            "engine_mean_pct": round(100 * float(np.mean([v[0] for v in eng_rets.values()])), 3),
            "ew_mean_pct": round(100 * ew_mean, 3),
            "selection_alpha_vs_ew_pp": round(100 * (eng_mean_all - ew_mean), 3),
            "capture": {**cap, "shuffle_ctrl_aggregate_pct_seeds": ctrl_caps, "shuffle_ctrl_mean": round(ctrl_cap_mean, 2)},
            "identification_auc": auc_res,
            "regime_stratified_alpha": reg_alpha,
        }

    # --- ORACLE CEILING SUMMARY ---
    print("\n" + "=" * 76)
    print("SUMMARY: DEV oracle + identification")
    print("=" * 76)
    print(f"Oracle ceiling (best single asset per 7d block): mean={results['oracle']['mean_best_pct']}% "
          f"pos={results['oracle']['pct_positive']}% n={results['oracle']['n_blocks']}")
    for K in K_VALS:
        res = results["engine"][f"K{K}"]
        cap = res["capture"]
        auc = res["identification_auc"]
        print(f"\n  K={K}:")
        print(f"    Engine mean 7d:      {res['engine_mean_pct']:.3f}%")
        print(f"    EW baseline mean 7d: {res['ew_mean_pct']:.3f}%")
        print(f"    Selection alpha:     {res['selection_alpha_vs_ew_pp']:.3f}pp vs EW")
        print(f"    Capture (honest agg):{cap['aggregate_pct']}%  vs oracle {cap['sum_oracle_pct']}% available")
        print(f"    Shuffle ctrl cap:    {cap['shuffle_ctrl_mean']}% (selection adds {round(cap['aggregate_pct'] - cap['shuffle_ctrl_mean'], 2) if cap['aggregate_pct'] is not None else 'N/A'}pp)")
        print(f"    Identification AUC:  {auc['auc']} (p={auc['p_val']})")
        for reg, rv in res["regime_stratified_alpha"].items():
            print(f"    Regime {reg}: alpha={rv['selection_alpha_pp']}pp ({rv['n_blocks']} blocks)")

    results["runtime_s"] = round(time.time() - t0, 1)
    outp = CRYPTO / "runs" / "strat" / f"dev_oracle_identification_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved: {outp}  ({results['runtime_s']}s)")
    return results


if __name__ == "__main__":
    main()
