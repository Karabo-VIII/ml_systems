"""src/strat/quant_referee_mover.py -- INDEPENDENT QUANT REFEREE re-derivation.

Adversarial referee for the mover-capture tournament. RE-DERIVES from raw indicators
(does NOT trust any lane's saved W). Hunts look-ahead. Adds the controls the lanes omitted:

  1. Independent ungated-mover W + market circuit-breaker (rebuilt from _mover_scores logic).
  2. Leak-free random-7d-slice profitability (canonical ref.slice_stats) -- K=3 seeds.
  3. TWO capture-rate estimators, because the ratio-of-returns the lanes used is fragile:
       (a) RATIO-MEAN  = mean over blocks of (eng_ret / oracle_best)   [lane's metric]
       (b) AGGREGATE   = sum(realized over blocks) / sum(oracle_best over blocks) [honest pooled]
     The AGGREGATE is the dollar-weighted fraction of available move actually harvested;
     the ratio-mean is dominated by small-denominator blocks and is NOT a capture %.
  4. SAME-EXPOSURE SHUFFLE CONTROL: hold the SAME daily total exposure (circuit-breaker)
     but pick K assets at RANDOM each day instead of by mover score. If the real engine
     does not beat this control, the 'mover selection' adds nothing beyond market timing.
  5. Independent gate-exclusion audit (is the top-7d-mover below its own SMA200?).
  6. 2025-05-15 holdings spot-check, re-derived.

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

import strat.mover_lab as lab
import strat.adaptive_meta_engine as ame
import strat.referee_harness as ref

COST = lab.COST
OOS_START = "2022-01-01"
OOS_END = "2026-06-01"
N = 500
SEEDS = [11, 23, 42]


# ============================================================
# Vectorized mover scores (re-derived, identical recipe to mover_capture_engine)
# ============================================================
def mover_score_panel(ind: dict) -> pd.DataFrame:
    """Cross-sectional z-scored composite per bar. Vectorized. Causal (all inputs <= d)."""
    C = ind["C"]
    mom14 = ind["mom14"]; mom7 = ind["mom7"]
    hh14 = ind["hh14"]; ll14 = ind["ll14"]
    vol20 = ind["vol20"]; vol60 = ind["vol60"]
    rng = (hh14 - ll14)
    breakout = (C - ll14) / (rng + 1e-9)
    breakout = breakout.where(rng > 1e-9, 0.5)
    accel = mom7 - mom14 / 2.0
    vol_exp = vol20 / (vol60 + 1e-9)
    valid = C.notna() & (C > 0) & mom14.notna() & mom7.notna() & hh14.notna() & ll14.notna()

    def zrow(df):
        d = df.where(valid)
        mu = d.mean(axis=1)
        sd = d.std(axis=1)
        z = d.sub(mu, axis=0).div(sd + 1e-9, axis=0)
        return z.where(sd > 1e-9, 0.0)

    comp = 0.35 * zrow(mom14) + 0.25 * zrow(breakout) + 0.20 * zrow(vol_exp) + 0.20 * zrow(accel)
    return comp.where(valid)


def market_exposure_series(ind: dict, vol_hi: float) -> pd.Series:
    """Vectorized causal market-exposure scalar per bar [0.2..1.0]."""
    C = ind["C"]; sma200 = ind["sma200"]; sma50 = ind["sma50"]; vol20 = ind["vol20"]
    btc_up = (C["BTCUSDT"] > sma200["BTCUSDT"])
    above = (C > sma50)
    present = C.notna() & sma50.notna()
    breadth = above.where(present).sum(axis=1) / present.sum(axis=1).replace(0, np.nan)
    breadth = breadth.fillna(0.5)
    hi_vol = vol20["BTCUSDT"].fillna(0.0) >= vol_hi
    expo = pd.Series(0.20, index=C.index)   # default bear
    up = btc_up.fillna(False)
    expo[up & (breadth >= 0.50) & (~hi_vol)] = 1.00
    expo[up & ~((breadth >= 0.50) & (~hi_vol)) & (breadth >= 0.30)] = 0.70
    expo[up & ~((breadth >= 0.50) & (~hi_vol)) & (breadth < 0.30)] = 0.40
    return expo


def ungated_mover_W(ind: dict, vol_hi: float, K: int, warmup: int = 60,
                    random_pick_seed: int | None = None) -> pd.DataFrame:
    """Top-K by mover score (no gate), scaled by circuit-breaker.
    If random_pick_seed is not None: pick K at RANDOM among valid assets (same exposure) -- the CONTROL."""
    C = ind["C"]
    comp = mover_score_panel(ind)
    expo = market_exposure_series(ind, vol_hi)
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    rng = np.random.default_rng(random_pick_seed) if random_pick_seed is not None else None
    cols = list(C.columns)
    for i, d in enumerate(C.index):
        if i < warmup:
            continue
        row = comp.loc[d]
        valid = [s for s in cols if pd.notna(row[s])]
        if not valid:
            continue
        if rng is not None:
            picks = list(rng.choice(valid, size=min(K, len(valid)), replace=False))
        else:
            picks = sorted(valid, key=lambda s: -row[s])[:K]
        e = float(expo.loc[d])
        w = e / len(picks)
        for s in picks:
            W.loc[d, s] = w
    return W


# ============================================================
# Capture-rate: ratio-mean (lane) vs AGGREGATE (honest pooled)
# ============================================================
def capture_rates(W: pd.DataFrame, ind: dict, horizon: int = 7, oos_start: str | None = None) -> dict:
    C = ind["C"]; R = ind["R"]
    pos = W.shift(1).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs()).sum(axis=1)
    R_al = R.reindex(index=W.index, columns=W.columns).fillna(0.0)
    daily = (pos * R_al).sum(axis=1) - turn * (COST / 2.0)
    fwd = (C.shift(-horizon) / C - 1)
    idx = W.index
    if oos_start is not None:
        startpos = int(np.searchsorted(idx, pd.Timestamp(oos_start)))
    else:
        startpos = 0
    ratios = []; sum_real = 0.0; sum_oracle = 0.0; n = 0
    for si in range(startpos, len(idx) - horizon, horizon):
        sl = idx[si: si + horizon]
        if len(sl) < horizon:
            break
        eng = float((1 + daily.loc[sl]).prod() - 1)
        oracle = float(fwd.loc[idx[si]].max())
        if oracle > 0.005:   # only blocks with an available up-move
            ratios.append(eng / oracle)
            sum_real += eng; sum_oracle += oracle; n += 1
    if n == 0:
        return {"ratio_mean": None, "aggregate": None, "n": 0}
    cr = np.array(ratios)
    return {
        "ratio_mean_pct": round(float(cr.mean()) * 100, 1),
        "ratio_median_pct": round(float(np.median(cr)) * 100, 1),
        "aggregate_pct": round(100 * sum_real / sum_oracle, 1),   # honest pooled fraction
        "pct_positive": round(float((cr > 0).mean()) * 100, 1),
        "n": n,
    }


def slice_pack(b, bh_b):
    prs = [ref.slice_stats(b, bh_b, OOS_START, OOS_END, N, 7, s) for s in SEEDS]
    return {
        "pos_rate": round(float(np.mean([x["pos_rate"] for x in prs])), 1),
        "pos_seeds": [x["pos_rate"] for x in prs],
        "mean": round(float(np.mean([x["mean_pct"] for x in prs])), 2),
        "median": round(float(np.mean([x["median_pct"] for x in prs])), 2),
        "p05": round(float(np.mean([x["p05_pct"] for x in prs])), 2),
        "beat_bh": round(float(np.mean([x["beat_bh_pct"] for x in prs])), 1),
    }


def main():
    t0 = time.time()
    print("=" * 78)
    print("QUANT REFEREE -- independent re-derivation of the ungated mover engine")
    print(f"OOS {OOS_START}->{OOS_END} | n={N} | seeds={SEEDS} | 7-consec-trading-day slices")
    print("=" * 78)

    ind = lab.load("2020-01-01", OOS_END)
    ind["vol60"] = ind["R"].rolling(60, min_periods=30).std() * np.sqrt(365)
    C = ind["C"]

    train_mask = C.index < pd.Timestamp(OOS_START)
    vol_hi = float(ind["vol20"]["BTCUSDT"][train_mask].dropna().quantile(ame.VOL_HI_PCTILE))
    print(f"vol_hi_threshold (train-only): {vol_hi:.4f}")

    bh_W = ref.bh_ew_weights(ind)
    bh_b = ref.book_daily_returns(bh_W, ind)
    bh_stats = {"pos_rate": round(float(np.mean([ref.bh_slice_stats(bh_b, OOS_START, OOS_END, N, 7, s)["pos_rate"] for s in SEEDS])), 1),
                "mean": round(float(np.mean([ref.bh_slice_stats(bh_b, OOS_START, OOS_END, N, 7, s)["mean_pct"] for s in SEEDS])), 2)}

    Wr = ame.build_weight_matrix(ind, vol_hi)
    router_b = ref.book_daily_returns(Wr, ind)

    results = {"vol_hi": round(vol_hi, 4), "bh": bh_stats}

    # --- BH + router ---
    print(f"\n[BH]     pos={bh_stats['pos_rate']}% mean={bh_stats['mean']}%")
    rp = slice_pack(router_b, bh_b)
    rcap = capture_rates(Wr, ind, oos_start=OOS_START)
    results["gated_router"] = {**rp, "capture": rcap}
    print(f"[ROUTER] pos={rp['pos_rate']}% mean={rp['mean']}% beat_bh={rp['beat_bh']}% "
          f"p05={rp['p05']}% | cap_agg={rcap['aggregate_pct']}% cap_ratiomean={rcap['ratio_mean_pct']}%")

    # --- ungated mover engines K=1,3,5 + the SHUFFLE CONTROL ---
    for K in [1, 3, 5]:
        W = ungated_mover_W(ind, vol_hi, K=K)
        b = ref.book_daily_returns(W, ind)
        sp = slice_pack(b, bh_b)
        cap = capture_rates(W, ind, oos_start=OOS_START)
        fp = lab.evaluate(W, ind, H=7, label=f"mover_K{K}")
        expo_oos = float(W.sum(axis=1)[C.index >= pd.Timestamp(OOS_START)].mean())
        bear22 = (C.index >= "2022-01-01") & (C.index < "2023-01-01")
        expo_bear = float(W.sum(axis=1)[bear22].mean())

        # SHUFFLE CONTROL: same exposure, random K picks (3 seeds, averaged)
        ctrl_pos, ctrl_mean, ctrl_cap = [], [], []
        for cs in [101, 202, 303]:
            Wc = ungated_mover_W(ind, vol_hi, K=K, random_pick_seed=cs)
            bc = ref.book_daily_returns(Wc, ind)
            scp = slice_pack(bc, bh_b)
            ctrl_pos.append(scp["pos_rate"]); ctrl_mean.append(scp["mean"])
            ctrl_cap.append(capture_rates(Wc, ind, oos_start=OOS_START)["aggregate_pct"])
        ctrl = {"pos_rate": round(float(np.mean(ctrl_pos)), 1),
                "mean": round(float(np.mean(ctrl_mean)), 2),
                "cap_agg": round(float(np.mean(ctrl_cap)), 1)}

        results[f"mover_K{K}"] = {
            **sp, "capture": cap, "expo_oos": round(expo_oos, 3),
            "expo_bear22": round(expo_bear, 3),
            "comp_full": fp["comp_full"], "maxDD": fp["maxDD"], "comp_2022": fp["comp_2022"],
            "shuffle_control": ctrl,
            "selection_alpha_mean": round(sp["mean"] - ctrl["mean"], 2),
            "selection_alpha_cap": round(cap["aggregate_pct"] - ctrl["cap_agg"], 1),
        }
        print(f"\n[MOVER K={K}] pos={sp['pos_rate']}% (seeds {sp['pos_seeds']}) mean={sp['mean']}% "
              f"beat_bh={sp['beat_bh']}% p05={sp['p05']}%")
        print(f"   capture: aggregate={cap['aggregate_pct']}%  ratio_mean={cap['ratio_mean_pct']}%  "
              f"ratio_median={cap['ratio_median_pct']}%  pos_blocks={cap['pct_positive']}%")
        print(f"   expo OOS={expo_oos:.2f} bear22={expo_bear:.2f} | comp_full={fp['comp_full']}% "
              f"maxDD={fp['maxDD']}% comp2022={fp['comp_2022']}%")
        print(f"   SHUFFLE-CONTROL (same expo, random picks): pos={ctrl['pos_rate']}% mean={ctrl['mean']}% "
              f"cap_agg={ctrl['cap_agg']}%")
        print(f"   --> SELECTION ALPHA: mean +{results[f'mover_K{K}']['selection_alpha_mean']}pp  "
              f"capture +{results[f'mover_K{K}']['selection_alpha_cap']}pp")

    # --- ORACLE ceiling (aggregate) ---
    fwd = (C.shift(-7) / C - 1)
    idx = C.index
    startpos = int(np.searchsorted(idx, pd.Timestamp(OOS_START)))
    orc = []
    for si in range(startpos, len(idx) - 7, 7):
        o = float(fwd.iloc[si].max())
        orc.append(o)
    orc = np.array(orc)
    results["oracle"] = {
        "mean_7d_best_pct": round(float(orc.mean()) * 100, 2),
        "pos_rate": round(float((orc > 0).mean()) * 100, 1),
        "n_blocks": len(orc),
    }
    print(f"\n[ORACLE] best-asset mean 7d={results['oracle']['mean_7d_best_pct']}% "
          f"pos={results['oracle']['pos_rate']}% n={results['oracle']['n_blocks']}")

    # --- GATE-EXCLUSION audit (independent) ---
    gate = ind["gate"]
    excl_top1 = 0; excl_anytop3 = 0; btc_is_top = 0; days = 0
    oos_idx = idx[(idx >= pd.Timestamp(OOS_START)) & (idx < pd.Timestamp(OOS_END))]
    for d in oos_idx:
        i = idx.get_loc(d)
        if i + 7 >= len(idx):
            continue
        f = fwd.loc[d].dropna()
        if len(f) < 3:
            continue
        days += 1
        ranked = f.sort_values(ascending=False)
        top1 = ranked.index[0]; top3 = list(ranked.index[:3])
        if not bool(gate.loc[d, top1]):
            excl_top1 += 1
        excl_anytop3 += sum(1 for s in top3 if not bool(gate.loc[d, s]))
        if top1 == "BTCUSDT":
            btc_is_top += 1
    results["gate_audit"] = {
        "top1_below_own_sma200_pct": round(100 * excl_top1 / days, 1),
        "top3_excluded_pct": round(100 * excl_anytop3 / (days * 3), 1),
        "btc_is_top1_pct": round(100 * btc_is_top / days, 1),
        "n_days": days,
    }
    ga = results["gate_audit"]
    print(f"\n[GATE AUDIT OOS] top1<own-SMA200: {ga['top1_below_own_sma200_pct']}% | "
          f"top3 excluded: {ga['top3_excluded_pct']}% | BTC-is-top1: {ga['btc_is_top1_pct']}% | n={ga['n_days']}")

    # --- 2025-05-15 holdings (re-derived) ---
    comp = mover_score_panel(ind)
    expo = market_exposure_series(ind, vol_hi)
    target = pd.Timestamp("2025-05-15")
    d = idx[idx >= target][0]
    row = comp.loc[d].dropna().sort_values(ascending=False)
    top3 = list(row.index[:3])
    f7 = fwd.loc[d]
    sc = {"date": str(d.date()), "exposure": round(float(expo.loc[d]), 2), "K3_holds": top3,
          "ranks": {s: {"rank": int(list(row.index).index(s)) + 1 if s in row.index else None,
                        "fwd7d_pct": round(float(f7[s]) * 100, 1) if pd.notna(f7[s]) else None,
                        "gate": bool(gate.loc[d, s])}
                    for s in ["DOGEUSDT", "AVAXUSDT", "SOLUSDT", "BTCUSDT", "ETHUSDT"]}}
    results["spot_2025_05_15"] = sc
    print(f"\n[SPOT 2025-05-15] exposure={sc['exposure']} K3 holds={top3}")
    for s, info in sc["ranks"].items():
        print(f"   {s:10s} rank={info['rank']} fwd7d={info['fwd7d_pct']}% gate={info['gate']}")

    results["runtime_s"] = round(time.time() - t0, 1)
    outp = ROOT.parent / "runs" / "strat" / "quant_referee_mover_results.json"
    outp.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved: {outp} ({results['runtime_s']}s)")
    return results


if __name__ == "__main__":
    main()
