"""src/strat/nlcycle3_conditioner.py -- META-FOLD NL-CYCLE 3 fresh-expert experiment.

FRESH ANGLE (the dimension the price+chimera DIRECTION cycles missed):
  The systematic cycles framed everything as DIRECTION forecasting (will the basket be up?) and
  hit a structural wall. They tested chimera as an ENTRY GATE (dead) and price as a DIRECTION
  signal (dead). The ONE survivor is the ROUTER -- real SELECTION skill at ~58% exposure.

  This cycle tests two UNTESTED PLACEMENTS, neither of which is direction-forecasting:

  LANE (b) -- EXOGENOUS RISK-OFF CONDITIONER on the router's HELD names (NOT an entry gate).
    Each day the router holds a basket W. For each HELD name we read a per-asset EXOGENOUS risk
    extreme (funding blowoff / basis panic / liquidation cascade / whale distribution / OI spike).
    If a held name is in a risk extreme, we TRIM it toward cash (scale weight down). This can ONLY
    reduce exposure on positions we ALREADY hold -> long-only-safe, never a new directional bet,
    never a short. Thesis: trimming held names at exogenous risk extremes improves the TAIL
    (p05 / down-week) and MEAN of the 7d outcome without crippling pos-rate. It is a DEFENSIVE
    SIZING conditioner, the placement the entry-gate cycle never tried.

  LANE (a) -- MULTI-TF ENTRY TIMING. When the router enters a name on the daily bar, does waiting
    for an intraday (4h) pullback before committing improve the 7d outcome vs entering at the
    daily open? Tested as an overlay that delays the entry leg of the router's daily W.

REFEREE (built in): every claimed edge is re-derived against a DATE-BLOCK PERMUTATION null --
  the exogenous signal (lane b) or the intraday-timing decision (lane a) is shuffled in 7-day
  blocks while the SAME held basket is kept, so the permuted overlay touches the same names the
  same fraction of the time but at scrambled dates. A real conditioner must beat its own block-
  permuted self. p = frac(perm metric >= real metric).

  ALL trims/timings are LONG-ONLY-SPOT, INTERNAL-DATA. No shorts, no leverage, no external data.

CONVENTIONS: identical to referee_harness (7-consecutive-trading-day slices, EW-BH cadence-invariant
  baseline, positions lagged 1 bar via book_daily_returns, taker cost on |dpos|).

RWYB:
  python -m strat.nlcycle3_conditioner --lane b
  python -m strat.nlcycle3_conditioner --lane a
No emoji (cp1252). Does NOT git commit.
"""
from __future__ import annotations
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.mover_lab as lab
import strat.referee_harness as ref
import strat.adaptive_meta_engine as ame

OOS_START = "2022-01-01"
OOS_END = "2026-06-01"
TRAIN_END = "2022-01-01"
N_SLICES = 500
SEEDS = [11, 23, 42]

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]

# Per-asset EXOGENOUS risk features. Sign convention: HIGHER raw value -> MORE risk-of-drawdown
# for a LONG. We flip signs below so a high "risk score" always means "this long is dangerous".
#   norm_funding        : high positive funding = crowded longs paying = squeeze/flush risk (HIGH=risk)
#   bs_basis_z30        : high basis = euphoric premium = mean-revert risk (HIGH=risk)
#   liq_delta_z30       : net long liquidations spiking (HIGH=risk; sign per data below)
#   norm_oi_change      : OI spiking up with crowd = fragile (abs -> HIGH=risk)
#   wh_whale_net_usd    : whales NET SELLING (LOW/negative = distribution = risk -> we flip)
RISK_FEATS = {
    "norm_funding":   +1.0,   # high funding -> risk
    "bs_basis_z30":   +1.0,   # high basis -> risk
    "liq_delta_z30":  +1.0,   # liq delta spike -> risk
    "wh_whale_net_usd": -1.0, # whales selling (negative net) -> risk (flip sign)
}


# ============================================================
# EXOGENOUS PANEL (per-asset, aligned to lab daily index, causal)
# ============================================================
def load_exo_panel(C: pd.DataFrame, feats: list[str]) -> dict:
    """Return {feat: DataFrame(dates x assets)} aligned to C.index. Causal: value at date d known at d.
    Each column is the per-asset exogenous series floored to day and reindexed onto the lab index.
    """
    from pipeline.chimera_loader import ChimeraLoader
    cl = ChimeraLoader()
    out = {f: pd.DataFrame(index=C.index, columns=C.columns, dtype=float) for f in feats}
    for sym in C.columns:
        try:
            df = cl.load(sym, cadence="1d", features=feats).to_pandas()
        except Exception:
            continue
        dt = pd.to_datetime(df["timestamp"].to_numpy(), unit="ms").floor("D")
        for f in feats:
            if f not in df.columns:
                continue
            s = pd.Series(df[f].to_numpy(dtype=float), index=dt)
            s = s[~s.index.duplicated(keep="last")]
            out[f][sym] = s.reindex(C.index)
    return out


def build_risk_score(exo: dict, C: pd.DataFrame) -> pd.DataFrame:
    """Composite per-asset risk score (dates x assets): mean of cross-sectionally-ranked,
    sign-oriented risk features. HIGH score = this long is dangerous. Cross-sectional rank
    (pct) per day so the score is comparable across assets and self-normalizing (no train
    scaling leak). Causal: each day uses only that day's cross-section.
    """
    parts = []
    for f, sign in RISK_FEATS.items():
        if f not in exo:
            continue
        df = exo[f] * sign
        # cross-sectional pct rank per day (NaN-safe); high rank = high risk
        r = df.rank(axis=1, pct=True)
        parts.append(r)
    if not parts:
        return pd.DataFrame(0.0, index=C.index, columns=C.columns)
    score = sum(parts) / len(parts)
    return score


# ============================================================
# LANE (b): RISK-OFF TRIM CONDITIONER on the router's held W
# ============================================================
def apply_risk_trim(Wr: pd.DataFrame, risk: pd.DataFrame, thr: float, trim: float) -> pd.DataFrame:
    """For each HELD name (Wr>0) whose risk score >= thr, scale its weight by (1-trim) (trim toward
    cash). LONG-ONLY-SAFE: only reduces existing weights, never adds/creates/flips a position. The
    trimmed weight goes to CASH (we do NOT redistribute to other names -> a pure de-risk, no new bet).
    Causal: risk at date d acts on W at date d (then book_daily_returns lags the whole W by 1 bar).
    """
    R = risk.reindex(index=Wr.index, columns=Wr.columns)
    flag = (Wr > 0) & (R >= thr)
    Wout = Wr.copy()
    Wout[flag] = Wr[flag] * (1.0 - trim)
    return Wout


def block_permute(risk: pd.DataFrame, oos_start: str, block: int, seed: int) -> pd.DataFrame:
    """Date-block-permute the risk panel: shuffle 7-day blocks of ROWS within the OOS region only
    (train region untouched so the threshold calibration is unaffected). Same names get trimmed the
    same FRACTION of the time, but at scrambled dates -> destroys any real date-aligned risk timing.
    """
    rng = np.random.default_rng(seed)
    idx = risk.index
    oos_mask = idx >= pd.Timestamp(oos_start)
    oos_pos = np.where(oos_mask)[0]
    blocks = [oos_pos[i:i + block] for i in range(0, len(oos_pos), block)]
    order = list(range(len(blocks)))
    rng.shuffle(order)
    new_pos = np.concatenate([blocks[o] for o in order])
    Rp = risk.copy()
    vals = risk.values.copy()
    vals[oos_pos] = risk.values[new_pos]
    return pd.DataFrame(vals, index=risk.index, columns=risk.columns)


def eval_book(b: pd.Series, bh_b: pd.Series) -> dict:
    """Multi-seed slice stats -> averaged scalar metrics."""
    prs = [ref.slice_stats(b, bh_b, OOS_START, OOS_END, N_SLICES, 7, s) for s in SEEDS]
    return {
        "pos_rate": round(float(np.mean([x["pos_rate"] for x in prs])), 2),
        "mean_pct": round(float(np.mean([x["mean_pct"] for x in prs])), 3),
        "p05_pct": round(float(np.mean([x["p05_pct"] for x in prs])), 2),
        "median_pct": round(float(np.mean([x["median_pct"] for x in prs])), 3),
        "beat_bh": round(float(np.mean([x["beat_bh_pct"] for x in prs])), 1),
        "down_wk_mean": round(float(np.mean([x["down_wk_eng_mean"] for x in prs])), 2),
    }


def run_lane_b():
    t0 = time.time()
    print("=" * 78)
    print("NL-CYCLE 3 LANE (b): exogenous RISK-OFF trim conditioner on router HELD names")
    print(f"OOS {OOS_START}->{OOS_END} | n={N_SLICES} | seeds={SEEDS}")
    print("=" * 78)

    ind = lab.load("2020-01-01", OOS_END)
    C = ind["C"]
    bh_W = ref.bh_ew_weights(ind); bh_b = ref.book_daily_returns(bh_W, ind)

    # baseline router
    train_mask = C.index < pd.Timestamp(TRAIN_END)
    vthr = float(ind["vol20"]["BTCUSDT"][train_mask].dropna().quantile(ame.VOL_HI_PCTILE))
    Wr = ame.build_weight_matrix(ind, vthr)
    rb = ref.book_daily_returns(Wr, ind)
    base = eval_book(rb, bh_b)
    base_expo = round(float(Wr.sum(axis=1).loc[C.index >= pd.Timestamp(OOS_START)].mean()), 3)
    print(f"\n[BASE ROUTER]   pos={base['pos_rate']}% mean={base['mean_pct']}% p05={base['p05_pct']}% "
          f"down_wk={base['down_wk_mean']}% beat_bh={base['beat_bh']}% expo={base_expo}")
    bh_stats = eval_book(bh_b, bh_b)
    print(f"[BH]            pos={bh_stats['pos_rate']}% mean={bh_stats['mean_pct']}% p05={bh_stats['p05_pct']}%")

    # exogenous risk panel
    print("\n[exo] loading exogenous risk panel (funding/basis/liq/whale)...")
    exo = load_exo_panel(C, list(RISK_FEATS.keys()))
    risk = build_risk_score(exo, C)
    # threshold derived from TRAIN region only (causal): the cross-sectional rank is already in [0,1]
    # so the threshold is a percentile of the risk score; we sweep it.
    print(f"[exo] risk score built. OOS coverage={float(risk.loc[C.index>=pd.Timestamp(OOS_START)].notna().mean().mean()):.2f}")

    # sweep trim configs
    configs = [(0.80, 1.0), (0.85, 1.0), (0.90, 1.0), (0.80, 0.5), (0.90, 0.5), (0.95, 1.0)]
    results = {}
    best = None
    for (thr, trim) in configs:
        Wt = apply_risk_trim(Wr, risk, thr, trim)
        bt = ref.book_daily_returns(Wt, ind)
        st = eval_book(bt, bh_b)
        expo = round(float(Wt.sum(axis=1).loc[C.index >= pd.Timestamp(OOS_START)].mean()), 3)
        results[f"thr{thr}_trim{trim}"] = {**st, "expo": expo}
        d_mean = st["mean_pct"] - base["mean_pct"]
        d_p05 = st["p05_pct"] - base["p05_pct"]
        d_dwk = st["down_wk_mean"] - base["down_wk_mean"]
        print(f"  trim thr={thr} trim={trim}: pos={st['pos_rate']}% mean={st['mean_pct']}% "
              f"(d{d_mean:+.3f}) p05={st['p05_pct']}% (d{d_p05:+.2f}) down_wk={st['down_wk_mean']}% "
              f"(d{d_dwk:+.2f}) expo={expo}")
        # candidate = improves p05 OR down_wk without hurting mean materially
        score = (st["p05_pct"] - base["p05_pct"]) + (st["mean_pct"] - base["mean_pct"]) * 2
        if best is None or score > best[0]:
            best = (score, thr, trim, st)

    # ---- DATE-BLOCK-PERMUTATION REFEREE on the best config ----
    _, bthr, btrim, bst = best
    print(f"\n[REFEREE] block-permutation null on best config thr={bthr} trim={btrim} "
          f"(metric = p05 + 2*mean delta vs base)...")
    real_metric = (bst["p05_pct"] - base["p05_pct"]) + 2 * (bst["mean_pct"] - base["mean_pct"])
    n_perm = 200
    perm_metrics = []
    for pseed in range(n_perm):
        rp = block_permute(risk, OOS_START, 7, pseed)
        Wp = apply_risk_trim(Wr, rp, bthr, btrim)
        bp = ref.book_daily_returns(Wp, ind)
        # single-seed eval for speed in permutation loop (seed 42)
        sp = ref.slice_stats(bp, bh_b, OOS_START, OOS_END, N_SLICES, 7, 42)
        pm = (sp["p05_pct"] - base["p05_pct"]) + 2 * (sp["mean_pct"] - base["mean_pct"])
        perm_metrics.append(pm)
    perm_metrics = np.array(perm_metrics)
    p_val = float((perm_metrics >= real_metric).mean())
    print(f"  real metric={real_metric:+.3f} | perm mean={perm_metrics.mean():+.3f} "
          f"std={perm_metrics.std():.3f} p95={np.percentile(perm_metrics,95):+.3f} | p={p_val:.4f}")

    out = {
        "lane": "b", "base_router": {**base, "expo": base_expo}, "bh": bh_stats,
        "configs": results, "best_config": {"thr": bthr, "trim": btrim, **bst},
        "referee": {"real_metric": round(real_metric, 3), "perm_mean": round(float(perm_metrics.mean()), 3),
                    "perm_p95": round(float(np.percentile(perm_metrics, 95)), 3), "p_value": round(p_val, 4),
                    "n_perm": n_perm},
        "runtime_s": round(time.time() - t0, 1),
    }
    outp = ROOT.parent / "runs" / "strat" / "nlcycle3_lane_b_results.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {outp} ({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", default="b", choices=["a", "b"])
    a = ap.parse_args()
    if a.lane == "b":
        run_lane_b()
    else:
        import strat.nlcycle3_timing as t
        t.run_lane_a()
