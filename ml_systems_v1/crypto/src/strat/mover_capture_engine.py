"""src/strat/mover_capture_engine.py -- CORRECTED mover-capture engine.

OBJECTIVE: MOVE-CAPTURE across a multi-day (7d) window with a 14d lookback.
Be IN the assets that MOVE and ride them.

CORRECTION vs prior gated-router:
  - NO per-asset SMA200 gate (that gate excluded 39% of top-7d movers, 38% of top-3)
  - MARKET-LEVEL circuit-breaker only: scale total book exposure by
    f(BTC-vs-SMA200, breadth, vol). Individual movers are NEVER excluded.

MOVER SIGNAL: composite of 4 sub-signals (all causal, 14d lookback):
  1. 14d momentum (returns over 14 days)
  2. Breakout: price within top 10% of 14d range (new-high proximity)
  3. Vol-expansion: current vol20 vs 60d vol_ma (vol spike relative to own baseline)
  4. Momentum-acceleration: 7d mom vs 14d mom (recent acceleration)

MARKET CIRCUIT-BREAKER: total exposure = f(btc_above_sma200, breadth, vol)
  - Clean bull (BTC>SMA200 + breadth>0.5 + low vol): 100% exposure
  - Mixed (BTC>SMA200 + breadth>0.3): 70% exposure
  - Weak (BTC>SMA200 + breadth<0.3 OR hi vol): 40% exposure
  - Bear (BTC<SMA200): 20% exposure
  This scales the WHOLE book, preserving the ranking of individual movers.

JUDGE METRICS:
  (a) CAPTURE-RATE = realized / available-move (vs 7d-oracle) after cost
  (b) Random-7d-slice profitability + mean vs buy-hold AND vs gated router
  (c) Bear-survival via the circuit-breaker (2022 + 2024/2025 drawdowns)

RWYB: python -m strat.mover_capture_engine
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


# ============================================================
# MARKET-LEVEL CIRCUIT BREAKER
# ============================================================
def _market_exposure(ind: dict, i: int, vol_hi_threshold: float) -> float:
    """Causal market exposure scalar [0, 1] at bar i.
    Uses BTC vs SMA200, cross-asset breadth above SMA50, and BTC vol.
    NO individual asset filtering -- this scales the whole book.
    """
    C = ind["C"]; sma200 = ind["sma200"]; sma50 = ind["sma50"]
    vol20 = ind["vol20"]
    d = C.index[i]

    # BTC trend
    btc = C.loc[d, "BTCUSDT"] if "BTCUSDT" in C.columns else float("nan")
    s200 = sma200.loc[d, "BTCUSDT"] if "BTCUSDT" in sma200.columns else float("nan")
    btc_up = (not pd.isna(s200)) and (not pd.isna(btc)) and (btc > s200)

    # Cross-asset breadth: fraction of universe above SMA50
    row_c = C.iloc[i]; row_s50 = sma50.iloc[i]
    above = 0; total = 0
    for sym in C.columns:
        cv = row_c[sym]; sv = row_s50[sym]
        if pd.notna(cv) and pd.notna(sv):
            above += int(cv > sv); total += 1
    breadth = above / total if total > 0 else 0.5

    # BTC vol
    btc_vol = vol20.loc[d, "BTCUSDT"] if "BTCUSDT" in vol20.columns and pd.notna(vol20.loc[d, "BTCUSDT"]) else 0.0
    hi_vol = btc_vol >= vol_hi_threshold

    # Exposure table
    if not btc_up:
        return 0.20   # bear: 20% book scale (survival mode, not cash)
    if breadth >= 0.50 and not hi_vol:
        return 1.00   # clean bull: full exposure
    if breadth >= 0.30:
        return 0.70   # mixed: 70%
    # weak (breadth<0.30 or hi_vol while BTC still above SMA200)
    return 0.40


# ============================================================
# MOVER SIGNAL (composite, causal, 14d lookback)
# ============================================================
def _mover_scores(ind: dict, i: int) -> dict[str, float]:
    """Composite mover score at bar i for all assets.
    Returns {sym: score} for assets with valid prices.
    Sub-signals:
      mom14      = 14d price return (trend)
      breakout   = (C - ll14) / (hh14 - ll14)  range position (0..1)
      vol_exp    = vol20 / vol60  (expansion relative to own history)
      mom_accel  = mom7 - mom14/2  (7d momentum vs half of 14d baseline)
    All z-scored across the universe at each bar then averaged (equal weight).
    """
    C = ind["C"]; d = C.index[i]
    mom14 = ind["mom14"]; mom7 = ind["mom7"]
    hh14 = ind["hh14"]; ll14 = ind["ll14"]
    vol20 = ind["vol20"]

    syms = []
    raw_mom14, raw_breakout, raw_vol_exp, raw_accel = [], [], [], []

    for sym in C.columns:
        cv = C.loc[d, sym]
        m14 = mom14.loc[d, sym]; m7 = mom7.loc[d, sym]
        hh = hh14.loc[d, sym]; ll = ll14.loc[d, sym]
        v20 = vol20.loc[d, sym]

        if pd.isna(cv) or cv <= 0 or pd.isna(m14) or pd.isna(m7):
            continue
        if pd.isna(hh) or pd.isna(ll):
            continue

        rng = hh - ll
        breakout = (cv - ll) / (rng + 1e-9) if rng > 1e-9 else 0.5
        accel = m7 - m14 / 2.0
        # vol_exp relative to 60d vol baseline (stored in ind["vol60"] if present, else use vol20 fallback)
        v60 = ind.get("vol60", ind["vol20"]).loc[d, sym]
        vol_exp = v20 / (v60 + 1e-9) if pd.notna(v60) and v60 > 0 else 1.0

        syms.append(sym)
        raw_mom14.append(m14)
        raw_breakout.append(breakout)
        raw_vol_exp.append(vol_exp)
        raw_accel.append(accel)

    if not syms:
        return {}

    def zscore(arr):
        a = np.array(arr, dtype=float)
        std = a.std()
        return (a - a.mean()) / (std + 1e-9) if std > 1e-9 else np.zeros_like(a)

    z_mom14 = zscore(raw_mom14)
    z_breakout = zscore(raw_breakout)
    z_vol_exp = zscore(raw_vol_exp)
    z_accel = zscore(raw_accel)

    # Composite: weighted average of 4 signals
    composite = 0.35 * z_mom14 + 0.25 * z_breakout + 0.20 * z_vol_exp + 0.20 * z_accel
    return {s: float(c) for s, c in zip(syms, composite)}


# ============================================================
# BUILD WEIGHT MATRIX (mover-capture, market circuit-breaker)
# ============================================================
def build_mover_weight_matrix(ind: dict, vol_hi_threshold: float,
                               K: int = 3, warmup: int = 60) -> pd.DataFrame:
    """Build W (dates x assets) for the mover-capture engine.
    - Rank ALL assets by composite mover score (no individual gate)
    - Hold top-K EW
    - Scale total exposure by market circuit-breaker
    - Rebalance every day (signal changes daily; positions carry via lag in evaluate)
    """
    C = ind["C"]
    W = pd.DataFrame(0.0, index=C.index, columns=C.columns)

    for i, d in enumerate(C.index):
        if i < warmup:
            continue
        exposure = _market_exposure(ind, i, vol_hi_threshold)
        scores = _mover_scores(ind, i)
        if not scores:
            continue
        # Sort all scored assets descending, pick top-K (NO gate filter)
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        picks = ranked[:K]
        if not picks:
            continue
        w_per = exposure / len(picks)
        for sym, _ in picks:
            if sym in W.columns:
                W.loc[d, sym] = w_per

    return W


# ============================================================
# CAPTURE-RATE METRIC (oracle comparison, causal)
# ============================================================
def compute_capture_rate(W: pd.DataFrame, ind: dict, horizon: int = 7) -> dict:
    """CAPTURE-RATE = realized_7d_return / available_7d_move (oracle best asset in window).
    Oracle = maximum 7d forward return available in the universe (in-window oracle, NOT future signal).
    This is a JUDGMENT metric only (oracle uses future prices) -- used to gauge how much of
    the available move we harvest. Cost-adjusted realized returns used.
    """
    C = ind["C"]; R = ind["R"]
    # Realized engine returns (cost-adjusted daily, then compounded per window)
    pos = W.shift(1).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs()).sum(axis=1)
    R_aligned = R.reindex(index=W.index, columns=W.columns).fillna(0.0)
    daily_bret = (pos * R_aligned).sum(axis=1) - turn * (COST / 2.0)

    # Oracle: best single-asset 7d forward return (simple, not cost-adjusted -- the ideal)
    fwd7 = (C.shift(-horizon) / C - 1).fillna(0.0)

    idx = W.index
    capture_ratios = []
    for si in range(0, len(idx) - horizon, horizon):  # non-overlapping blocks
        sl = idx[si: si + horizon]
        if len(sl) < horizon:
            break
        eng_ret = float((1 + daily_bret.loc[sl]).prod() - 1)
        oracle_best = float(fwd7.loc[idx[si]].max())  # best asset available at block start
        if oracle_best > 0.005:  # only count blocks where there IS a move to capture (>0.5%)
            capture_ratios.append(eng_ret / oracle_best)

    if not capture_ratios:
        return {"capture_rate_mean": None, "capture_rate_median": None, "n_blocks": 0}

    cr = np.array(capture_ratios)
    return {
        "capture_rate_mean": round(float(cr.mean()) * 100, 1),   # % of oracle captured
        "capture_rate_median": round(float(np.median(cr)) * 100, 1),
        "capture_rate_p25": round(float(np.percentile(cr, 25)) * 100, 1),
        "capture_rate_p75": round(float(np.percentile(cr, 75)) * 100, 1),
        "n_blocks": len(cr),
        "pct_positive_capture": round(float((cr > 0).mean()) * 100, 1),
    }


# ============================================================
# SPOT CHECK: 2025-05-15 slice (does it hold DOGE/AVAX/SOL?)
# ============================================================
def spot_check_20250515(ind: dict, vol_hi_threshold: float, K: int = 3) -> dict:
    """Check what the mover-capture engine holds on 2025-05-15."""
    C = ind["C"]
    target = pd.Timestamp("2025-05-15")
    # Find nearest available bar
    idx = C.index
    i = None
    for j, d in enumerate(idx):
        if d >= target:
            i = j
            break
    if i is None or i < 60:
        return {"error": "date not in range or insufficient warmup"}

    d = idx[i]
    exposure = _market_exposure(ind, i, vol_hi_threshold)
    scores = _mover_scores(ind, i)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    top10 = ranked[:10]

    # Check what the gated router would hold
    regime = ame._detect_regime(ind, i, vol_hi_threshold)
    if regime == "downtrend":
        gated_holds = {"BTCUSDT": 0.10}
    elif regime == "clean-uptrend":
        gated_holds = ame._weights_uptrend(ind, i)
    elif regime == "recovery-bounce":
        gated_holds = ame._weights_recovery(ind, i)
    else:
        gated_holds = ame._weights_chop(ind, i)

    # Forward 7d returns for context (evaluation only, not used in signal)
    fwd7 = {}
    for sym in C.columns:
        if pd.notna(C.loc[d, sym]):
            end_i = min(i + 7, len(idx) - 1)
            end_d = idx[end_i]
            start_p = C.loc[d, sym]
            end_p = C.loc[end_d, sym]
            if pd.notna(end_p) and pd.notna(start_p) and start_p > 0:
                fwd7[sym] = round(float(end_p / start_p - 1) * 100, 1)

    # Gate status (was the top mover gated out by SMA200?)
    gate_status = {}
    for sym, score in top10:
        gate_status[sym] = bool(ind["gate"].loc[d, sym])

    return {
        "date": str(d.date()),
        "market_exposure": round(exposure, 2),
        "regime_gated_router": regime,
        "top10_by_mover_score": [
            {"sym": sym, "score": round(score, 3),
             "fwd7d_pct": fwd7.get(sym),
             "gate": gate_status[sym]}
            for sym, score in top10
        ],
        "mover_engine_holds_top{K}".replace("{K}", str(K)): [sym for sym, _ in top10[:K]],
        "gated_router_holds": list(gated_holds.keys()),
        "key_assets": {s: {"fwd7d_pct": fwd7.get(s), "score_rank": next((j+1 for j, (sym, _) in enumerate(ranked) if sym == s), None), "gate": bool(ind["gate"].loc[d, s]) if s in C.columns else None}
                      for s in ["DOGEUSDT", "AVAXUSDT", "SOLUSDT", "BTCUSDT", "ETHUSDT"]
                      if s in C.columns}
    }


# ============================================================
# MAIN EVALUATION
# ============================================================
def main():
    t0 = time.time()
    OOS_START = "2022-01-01"
    OOS_END = "2026-06-01"
    N = 500
    SEEDS = [11, 23, 42]
    K_VALS = [3, 5]

    print("=" * 76)
    print("MOVER-CAPTURE ENGINE -- CORRECTED (no per-asset gate, market circuit-breaker)")
    print(f"OOS: {OOS_START} -> {OOS_END} | n_slices={N} | seeds={SEEDS}")
    print("Signal: composite(mom14 35% + breakout 25% + vol_exp 20% + accel 20%)")
    print("Circuit-breaker: 20%/40%/70%/100% total exposure by market regime")
    print("=" * 76)

    # Load data (full range for warmup + full OOS)
    ind = lab.load("2020-01-01", OOS_END)
    C = ind["C"]

    # Add vol60 to ind for vol-expansion signal
    ind["vol60"] = ind["R"].rolling(60, min_periods=30).std() * np.sqrt(365)

    # Causal vol threshold from TRAINING data only
    train_mask = C.index < pd.Timestamp(OOS_START)
    btc_vol_train = ind["vol20"]["BTCUSDT"][train_mask].dropna()
    vol_hi_threshold = float(btc_vol_train.quantile(ame.VOL_HI_PCTILE))
    print(f"\nvol_hi_threshold (train-only): {vol_hi_threshold:.4f}")

    # BH baseline
    bh_W = ref.bh_ew_weights(ind)
    bh_b = ref.book_daily_returns(bh_W, ind)

    # Gated router (baseline to beat)
    Wr = ame.build_weight_matrix(ind, vol_hi_threshold)
    router_b = ref.book_daily_returns(Wr, ind)

    print("\n[BH] EW buy-hold baseline:")
    bh_stats_all = {}
    for s in SEEDS:
        st = ref.bh_slice_stats(bh_b, OOS_START, OOS_END, N, 7, s)
        bh_stats_all[s] = st
    bh_pr = [bh_stats_all[s]["pos_rate"] for s in SEEDS]
    bh_mn = [bh_stats_all[s]["mean_pct"] for s in SEEDS]
    print(f"  pos_rate seeds={bh_pr} mean={round(float(np.mean(bh_pr)),1)}%")
    print(f"  mean_pct seeds={bh_mn} mean={round(float(np.mean(bh_mn)),2)}%")

    print("\n[ROUTER] gated regime-router (baseline to beat on capture+profitability):")
    rpr = [ref.slice_stats(router_b, bh_b, OOS_START, OOS_END, N, 7, s)["pos_rate"] for s in SEEDS]
    rmn = [ref.slice_stats(router_b, bh_b, OOS_START, OOS_END, N, 7, s)["mean_pct"] for s in SEEDS]
    rbw = [ref.slice_stats(router_b, bh_b, OOS_START, OOS_END, N, 7, s)["beat_bh_pct"] for s in SEEDS]
    print(f"  pos_rate seeds={rpr} mean={round(float(np.mean(rpr)),1)}%")
    print(f"  mean_pct seeds={rmn} mean={round(float(np.mean(rmn)),2)}%")
    print(f"  beat_bh seeds={rbw} mean={round(float(np.mean(rbw)),1)}%")

    # Router capture rate
    router_cr = compute_capture_rate(Wr, ind, horizon=7)
    print(f"  capture_rate mean={router_cr['capture_rate_mean']}% median={router_cr['capture_rate_median']}% "
          f"pos={router_cr['pct_positive_capture']}% n={router_cr['n_blocks']}")

    # Full results container
    results = {
        "oos": [OOS_START, OOS_END], "n_slices": N, "seeds": SEEDS,
        "vol_hi_threshold": round(vol_hi_threshold, 4),
        "bh": {"pos_rate": round(float(np.mean(bh_pr)), 1), "pos_rate_seeds": bh_pr,
               "mean_pct": round(float(np.mean(bh_mn)), 2)},
        "gated_router": {
            "pos_rate": round(float(np.mean(rpr)), 1), "pos_rate_seeds": rpr,
            "mean_pct": round(float(np.mean(rmn)), 2),
            "beat_bh": round(float(np.mean(rbw)), 1),
            "capture_rate": router_cr,
        },
        "mover_engine": {},
    }

    # Build and evaluate mover engines
    for K in K_VALS:
        print(f"\n[MOVER-CAPTURE K={K}] Building weight matrix (no per-asset gate)...")
        W = build_mover_weight_matrix(ind, vol_hi_threshold, K=K)

        # Verify exposure on bear bars
        oos_mask = C.index >= pd.Timestamp(OOS_START)
        bear_2022 = (C.index >= "2022-01-01") & (C.index < "2023-01-01")
        avg_expo_oos = float(W.sum(axis=1)[oos_mask].mean())
        avg_expo_bear = float(W.sum(axis=1)[bear_2022].mean())
        print(f"  avg exposure OOS: {round(avg_expo_oos, 3)} | 2022-bear: {round(avg_expo_bear, 3)}")

        # Engine daily returns
        mover_b = ref.book_daily_returns(W, ind)

        # Random-slice evaluation
        print(f"  Running {N}x{len(SEEDS)} slice evaluations...")
        prs = [ref.slice_stats(mover_b, bh_b, OOS_START, OOS_END, N, 7, s) for s in SEEDS]
        pr = [x["pos_rate"] for x in prs]
        mn = [x["mean_pct"] for x in prs]
        bw = [x["beat_bh_pct"] for x in prs]
        p05 = [x["p05_pct"] for x in prs]
        dn_mean = [x["down_wk_eng_mean"] for x in prs]
        dn_pr = [x["down_wk_eng_posrate"] for x in prs]

        print(f"  pos_rate seeds={pr} mean={round(float(np.mean(pr)),1)}% [BH:{round(float(np.mean(bh_pr)),1)}% Router:{round(float(np.mean(rpr)),1)}%]")
        print(f"  mean_pct seeds={mn} mean={round(float(np.mean(mn)),2)}% [BH:{round(float(np.mean(bh_mn)),2)}%]")
        print(f"  beat_bh seeds={bw} mean={round(float(np.mean(bw)),1)}%")
        print(f"  p05 seeds={p05} mean={round(float(np.mean(p05)),2)}%")
        print(f"  down-BH-week eng mean={[round(x,2) for x in dn_mean]} eng_posrate={dn_pr}")

        # Full-period stats
        fp = lab.evaluate(W, ind, H=7, label=f"mover_cap_K{K}")
        print(f"  full-period: comp_full={fp['comp_full']}% maxDD={fp['maxDD']}% "
              f"comp_2022={fp['comp_2022']}% green_all={fp['green_all']}%")

        # Capture rate
        cr = compute_capture_rate(W, ind, horizon=7)
        print(f"  capture_rate mean={cr['capture_rate_mean']}% median={cr['capture_rate_median']}% "
              f"pos={cr['pct_positive_capture']}% n={cr['n_blocks']}")

        results["mover_engine"][f"K{K}"] = {
            "pos_rate": round(float(np.mean(pr)), 1), "pos_rate_seeds": pr,
            "mean_pct": round(float(np.mean(mn)), 2),
            "beat_bh": round(float(np.mean(bw)), 1),
            "p05": round(float(np.mean(p05)), 2),
            "down_wk_eng_mean": round(float(np.mean(dn_mean)), 2),
            "down_wk_eng_posrate": round(float(np.mean(dn_pr)), 1),
            "full_period": fp,
            "avg_expo_oos": round(avg_expo_oos, 3),
            "avg_expo_bear_2022": round(avg_expo_bear, 3),
            "capture_rate": cr,
        }

    # Spot check 2025-05-15
    print("\n[SPOT CHECK] 2025-05-15 -- does mover engine hold DOGE/AVAX/SOL?")
    sc = spot_check_20250515(ind, vol_hi_threshold, K=3)
    print(f"  date: {sc.get('date')} | exposure: {sc.get('market_exposure')} | regime(gated): {sc.get('regime_gated_router')}")
    if "top10_by_mover_score" in sc:
        print("  Top-10 by mover score (no gate):")
        for row in sc["top10_by_mover_score"]:
            gate_str = "GATED-OUT" if not row["gate"] else "gate-pass"
            fwd = row["fwd7d_pct"]
            print(f"    {row['sym']:12s} score={row['score']:6.3f} fwd7d={fwd}% [{gate_str}]")
    if "key_assets" in sc:
        print("  Key assets (DOGE/AVAX/SOL/BTC/ETH):")
        for sym, info in sc["key_assets"].items():
            print(f"    {sym:12s} rank={info['score_rank']} fwd7d={info['fwd7d_pct']}% gate={info['gate']}")
    results["spot_check_20250515"] = sc

    # Print summary table
    print("\n" + "=" * 76)
    print("SUMMARY TABLE -- MOVER CAPTURE vs BASELINES")
    print("=" * 76)
    fmt = "{:<28s} {:>10s} {:>10s} {:>10s} {:>10s}"
    print(fmt.format("Metric", "BH-EW", "Gated-Router", "Mover K=3", "Mover K=5"))
    print("-" * 76)

    def g(key, sub=None):
        r3 = results["mover_engine"].get("K3", {})
        r5 = results["mover_engine"].get("K5", {})
        if sub:
            r3 = r3.get(sub, {}); r5 = r5.get(sub, {})
        return (
            str(results["bh"].get(key, "?")),
            str(results["gated_router"].get(key, "?")),
            str(r3.get(key, "?")),
            str(r5.get(key, "?")),
        )

    def row(label, bh_v, rt_v, m3_v, m5_v):
        print(fmt.format(label, str(bh_v), str(rt_v), str(m3_v), str(m5_v)))

    bh_v = results["bh"]
    rt_v = results["gated_router"]
    m3 = results["mover_engine"].get("K3", {})
    m5 = results["mover_engine"].get("K5", {})
    m3fp = m3.get("full_period", {})
    m5fp = m5.get("full_period", {})

    row("pos_rate (7d slices) %", bh_v["pos_rate"], rt_v["pos_rate"], m3.get("pos_rate"), m5.get("pos_rate"))
    row("mean 7d return %", bh_v["mean_pct"], rt_v["mean_pct"], m3.get("mean_pct"), m5.get("mean_pct"))
    row("beat_bh %", "-", rt_v["beat_bh"], m3.get("beat_bh"), m5.get("beat_bh"))
    row("p05 7d %", "-", "-", m3.get("p05"), m5.get("p05"))
    row("down-wk eng mean %", "-", "-", m3.get("down_wk_eng_mean"), m5.get("down_wk_eng_mean"))
    row("capture_rate mean %", "-", rt_v.get("capture_rate", {}).get("capture_rate_mean"),
        m3.get("capture_rate", {}).get("capture_rate_mean"), m5.get("capture_rate", {}).get("capture_rate_mean"))
    row("capture_rate median %", "-", rt_v.get("capture_rate", {}).get("capture_rate_median"),
        m3.get("capture_rate", {}).get("capture_rate_median"), m5.get("capture_rate", {}).get("capture_rate_median"))
    row("avg expo OOS", "-", "-", m3.get("avg_expo_oos"), m5.get("avg_expo_oos"))
    row("avg expo 2022-bear", "-", "-", m3.get("avg_expo_bear_2022"), m5.get("avg_expo_bear_2022"))
    row("full comp_full %", "-", "-", m3fp.get("comp_full"), m5fp.get("comp_full"))
    row("full maxDD %", "-", "-", m3fp.get("maxDD"), m5fp.get("maxDD"))
    row("2022 comp %", "-", "-", m3fp.get("comp_2022"), m5fp.get("comp_2022"))
    row("green_all %", "-", "-", m3fp.get("green_all"), m5fp.get("green_all"))

    results["runtime_s"] = round(time.time() - t0, 1)
    outp = ROOT.parent / "runs" / "strat" / f"mover_capture_engine_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved: {outp}  ({results['runtime_s']}s)")
    return results


if __name__ == "__main__":
    main()
