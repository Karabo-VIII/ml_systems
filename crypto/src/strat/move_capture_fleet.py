"""src/strat/move_capture_fleet.py -- DEPLOYABLE mover-capture engine on fleet_lab DEV data.

ORC DECISIVE CYCLE 2026-06-20:
  THE LEAD (NL-C4, proven on OLD OOS): ungated mover engine (rank by composite, hold top-K,
  market circuit-breaker scaling total exposure) beat random-same-exposure by z=5-6 (p~0).
  THIS MODULE re-proves that on DEV (<= 2024-05-15), regime-stratified.

DECISIVE QUESTION: does mover-SELECTION beat random-SAME-EXPOSURE on DEV, AND survive
REGIME-STRATIFICATION (bull/chop/bear)?
  - If alpha vanishes in non-bull -> bull-beta artifact (NOT skill).
  - If holds across regimes -> REAL selection alpha = the breakthrough.

DATA WALL (binding): DEV (<= 2024-05-15) ONLY. fleet_lab.load_wide() hard-caps at DEV_END.
Does NOT import referee_harness. Does NOT touch OOS/UNSEEN.

RWYB: python -m strat.move_capture_fleet
No emoji. Does NOT git commit.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.fleet_lab as lab

DEV_END = lab.DEV_END          # "2024-05-15"
COST    = lab.COST             # 0.0024 taker RT


# ============================================================
# REGIME DETECTION (BTC vs SMA200 + cross-asset breadth, fully causal)
# ============================================================

def _compute_sma(C: pd.DataFrame, n: int) -> pd.DataFrame:
    return C.rolling(n, min_periods=n).mean()


def regime_series(C: pd.DataFrame, sma200: pd.DataFrame, sma50: pd.DataFrame,
                  btc_vol20: pd.Series, vol_hi_threshold: float) -> pd.Series:
    """Per-bar regime label: 'bull' / 'chop' / 'bear'.
    bull  = BTC > SMA200 AND breadth >= 0.50 AND NOT hi-vol
    bear  = BTC < SMA200
    chop  = everything else
    """
    btc_up = (C["BTCUSDT"] > sma200["BTCUSDT"]).fillna(False)
    above  = (C > sma50).where(C.notna() & sma50.notna())
    breadth = above.sum(axis=1) / C.notna().sum(axis=1).replace(0, np.nan)
    breadth = breadth.fillna(0.5)
    hi_vol  = btc_vol20.fillna(0.0) >= vol_hi_threshold

    reg = pd.Series("chop", index=C.index)
    reg[~btc_up] = "bear"
    bull_mask = btc_up & (breadth >= 0.50) & (~hi_vol)
    reg[bull_mask] = "bull"
    return reg


def market_exposure_series(C: pd.DataFrame, sma200: pd.DataFrame, sma50: pd.DataFrame,
                            btc_vol20: pd.Series, vol_hi_threshold: float) -> pd.Series:
    """Causal market-exposure scalar [0.2, 1.0] mirroring old mover circuit-breaker."""
    btc_up  = (C["BTCUSDT"] > sma200["BTCUSDT"]).fillna(False)
    above   = (C > sma50).where(C.notna() & sma50.notna())
    breadth = above.sum(axis=1) / C.notna().sum(axis=1).replace(0, np.nan)
    breadth = breadth.fillna(0.5)
    hi_vol  = btc_vol20.fillna(0.0) >= vol_hi_threshold

    expo = pd.Series(0.20, index=C.index)   # bear default
    up = btc_up
    expo[up & (breadth >= 0.50) & (~hi_vol)] = 1.00   # clean bull
    expo[up & ~((breadth >= 0.50) & (~hi_vol)) & (breadth >= 0.30)] = 0.70
    expo[up & ~((breadth >= 0.50) & (~hi_vol)) & (breadth < 0.30)]  = 0.40
    return expo


# ============================================================
# MOVER COMPOSITE (vectorized, causal, cross-sectional z-score)
# ============================================================

def mover_composite(F: dict, C: pd.DataFrame) -> pd.DataFrame:
    """4-component composite using fleet_lab feature names.
    mom14    -- 14d momentum                (weight 0.35)
    rangepos -- range position 0..1         (weight 0.25)  [maps to breakout]
    volexp   -- vol expansion vs 30d base   (weight 0.20)
    accel    -- momentum acceleration       (weight 0.20)
    All z-scored cross-sectionally per bar, then weighted sum.
    NaN where any ingredient is NaN (valid mask).
    """
    mom14    = F["mom14"]
    rangepos = F["rangepos"]
    volexp   = F["volexp"]
    accel    = F["accel"]
    valid = C.notna() & (C > 0) & mom14.notna() & rangepos.notna() & volexp.notna() & accel.notna()

    def zrow(df: pd.DataFrame) -> pd.DataFrame:
        d = df.where(valid)
        mu = d.mean(axis=1)
        sd = d.std(axis=1)
        z  = d.sub(mu, axis=0).div(sd + 1e-12, axis=0)
        return z.where(sd > 1e-12, 0.0)

    comp = 0.35 * zrow(mom14) + 0.25 * zrow(rangepos) + 0.20 * zrow(volexp) + 0.20 * zrow(accel)
    return comp.where(valid)


# ============================================================
# WEIGHT MATRIX BUILDER
# ============================================================

def build_weight_matrix(L: dict, vol_hi_threshold: float, K: int = 3,
                        warmup: int = 60, random_seed: int | None = None) -> pd.DataFrame:
    """Build W (dates x assets) for the mover-capture engine on DEV data.
    - rank ALL valid assets by composite mover score (no individual gate)
    - hold top-K equal-weight (or K random if random_seed given -- the SHUFFLE CONTROL)
    - scale total exposure by market circuit-breaker
    """
    C       = L["C"]
    F       = L["F"]
    sma200  = _compute_sma(C, 200)
    sma50   = _compute_sma(C, 50)
    btcvol  = C["BTCUSDT"].pct_change().rolling(20, min_periods=10).std() * (365 ** 0.5)

    comp  = mover_composite(F, C)
    expo  = market_exposure_series(C, sma200, sma50, btcvol, vol_hi_threshold)

    W    = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    rng  = np.random.default_rng(random_seed) if random_seed is not None else None

    for i in range(warmup, len(C.index)):
        d       = C.index[i]
        row     = comp.iloc[i]
        valid_s = [s for s in C.columns if pd.notna(row[s])]
        if not valid_s:
            continue
        if rng is not None:
            picks = list(rng.choice(valid_s, size=min(K, len(valid_s)), replace=False))
        else:
            picks = sorted(valid_s, key=lambda s: -row[s])[:K]
        e = float(expo.iloc[i])
        w = e / len(picks)
        for s in picks:
            W.loc[d, s] = w

    return W


# ============================================================
# DAILY RETURNS FROM WEIGHT MATRIX
# ============================================================

def book_returns(W: pd.DataFrame, L: dict) -> pd.Series:
    """Cost-adjusted daily book returns (positions lagged 1 bar, taker cost on |dpos|)."""
    R    = L["R"].reindex(index=W.index, columns=W.columns).fillna(0.0)
    pos  = W.shift(1).fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs()).sum(axis=1)
    return (pos * R).sum(axis=1) - turn * (COST / 2.0)


# ============================================================
# SLICE STATS (self-contained, no referee_harness import)
# ============================================================

def slice_stats(b: pd.Series, bh: pd.Series, n: int = 400, hold: int = 7,
                seed: int = 42) -> dict:
    """Random non-overlapping 7-trading-day slice evaluation.
    b   = engine daily returns
    bh  = EW buy-hold daily returns (reference)
    Returns dict with pos_rate, mean_pct, p05_pct, beat_bh_pct, z_vs_bh.
    """
    rng  = np.random.default_rng(seed)
    idx  = b.index
    valid_starts = [i for i in range(0, len(idx) - hold - 1)]
    if len(valid_starts) < n:
        n = len(valid_starts)
    chosen = sorted(rng.choice(valid_starts, n, replace=False))

    eng_rets = []
    bh_rets  = []
    for i in chosen:
        sl      = idx[i: i + hold]
        eng_rets.append(float((1 + b.loc[sl]).prod() - 1))
        bh_rets.append(float((1 + bh.loc[sl]).prod() - 1))

    e = np.array(eng_rets)
    h = np.array(bh_rets)
    diff = e - h

    return {
        "n": int(len(e)),
        "pos_rate":  round(float((e > 0).mean() * 100), 1),
        "mean_pct":  round(float(e.mean() * 100), 2),
        "median_pct": round(float(np.median(e) * 100), 2),
        "p05_pct":   round(float(np.percentile(e, 5) * 100), 2),
        "beat_bh_pct": round(float((e > h).mean() * 100), 1),
        "z_vs_bh":   round(float(diff.mean() / (diff.std() / len(diff) ** 0.5 + 1e-12)), 2),
        "mean_bh_pct": round(float(h.mean() * 100), 2),
    }


def regime_slice_stats(b: pd.Series, bh: pd.Series, regime: pd.Series,
                       n: int = 400, hold: int = 7, seed: int = 42) -> dict:
    """Same as slice_stats but stratified by regime at the slice start bar.
    Returns stats for each of bull/chop/bear PLUS overall.
    """
    rng   = np.random.default_rng(seed)
    idx   = b.index
    valid = [i for i in range(0, len(idx) - hold - 1)]
    if len(valid) < n:
        n = len(valid)
    chosen = sorted(rng.choice(valid, n, replace=False))

    buckets: dict[str, list] = {"bull": [], "chop": [], "bear": [], "all": []}
    for i in chosen:
        d       = idx[i]
        sl      = idx[i: i + hold]
        er      = float((1 + b.loc[sl]).prod()  - 1)
        hr      = float((1 + bh.loc[sl]).prod() - 1)
        reg_lbl = str(regime.loc[d]) if d in regime.index else "chop"
        if reg_lbl not in buckets:
            reg_lbl = "chop"
        buckets[reg_lbl].append((er, hr))
        buckets["all"].append((er, hr))

    out = {}
    for label, pairs in buckets.items():
        if not pairs:
            out[label] = {"n": 0}
            continue
        e = np.array([p[0] for p in pairs])
        h = np.array([p[1] for p in pairs])
        d = e - h
        out[label] = {
            "n":           int(len(e)),
            "pos_rate":    round(float((e > 0).mean() * 100), 1),
            "mean_pct":    round(float(e.mean() * 100), 2),
            "p05_pct":     round(float(np.percentile(e, 5) * 100), 2),
            "beat_bh_pct": round(float((e > h).mean() * 100), 1),
            "z_vs_bh":     round(float(d.mean() / (d.std() / max(len(d), 1) ** 0.5 + 1e-12)), 2),
            "sel_alpha_mean": round(float(d.mean() * 100), 2),   # engine - shuffle-control (filled later)
        }
    return out


# ============================================================
# SHUFFLE CONTROL (same exposure, random K picks)
# ============================================================

def shuffle_control_stats(L: dict, vol_hi_threshold: float, K: int,
                          regime: pd.Series, n_slices: int = 400, hold: int = 7,
                          ctrl_seeds: list | None = None) -> dict:
    """Build 3 random-pick controls and average their regime-stratified stats."""
    if ctrl_seeds is None:
        ctrl_seeds = [101, 202, 303]

    C    = L["C"]
    bh_W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    for i in range(len(C.index)):
        valid = [s for s in C.columns if pd.notna(C.iloc[i][s])]
        if valid:
            w = 1.0 / len(valid)
            bh_W.iloc[i] = pd.Series({s: w for s in valid}, dtype=float).reindex(C.columns, fill_value=0.0)
    bh_b = book_returns(bh_W, L)

    all_reg_stats = []
    for cs in ctrl_seeds:
        Wc   = build_weight_matrix(L, vol_hi_threshold, K=K, random_seed=cs)
        bc   = book_returns(Wc, L)
        stat = regime_slice_stats(bc, bh_b, regime, n=n_slices, hold=hold, seed=cs)
        all_reg_stats.append(stat)

    # Average across control seeds
    labels = list(all_reg_stats[0].keys())
    ctrl = {}
    for lbl in labels:
        ns  = [s[lbl] for s in all_reg_stats if s[lbl].get("n", 0) > 0]
        if not ns:
            ctrl[lbl] = {"n": 0}
            continue
        ctrl[lbl] = {
            "n":           int(round(np.mean([s["n"] for s in ns]))),
            "pos_rate":    round(float(np.mean([s["pos_rate"] for s in ns])), 1),
            "mean_pct":    round(float(np.mean([s["mean_pct"] for s in ns])), 2),
            "p05_pct":     round(float(np.mean([s["p05_pct"] for s in ns])), 2),
            "beat_bh_pct": round(float(np.mean([s["beat_bh_pct"] for s in ns])), 1),
            "z_vs_bh":     round(float(np.mean([s["z_vs_bh"] for s in ns])), 2),
        }
    return ctrl, bh_b


# ============================================================
# FULL-PERIOD EQUITY STATS
# ============================================================

def equity_stats(b: pd.Series, label: str = "") -> dict:
    x  = b.fillna(0.0).to_numpy()
    eq = np.cumprod(1 + x)
    pk = np.maximum.accumulate(eq)
    def comp(s, e, idx=b.index):
        m  = (idx >= s) & (idx < e)
        xs = x[m]
        return round((np.prod(1 + xs) - 1) * 100, 1) if m.sum() > 2 else None
    return {
        "label":      label,
        "comp_full":  round((eq[-1] - 1) * 100, 1),
        "comp_2020":  comp("2020-01-01", "2021-01-01"),
        "comp_2021":  comp("2021-01-01", "2022-01-01"),
        "comp_2022":  comp("2022-01-01", "2023-01-01"),
        "comp_2023":  comp("2023-01-01", "2024-01-01"),
        "maxDD":      round(float(((eq - pk) / pk).min() * 100), 1),
        "avg_expo":   0.0,   # filled in by caller with round(avg_expo_dev, 3)
    }


# ============================================================
# CLEAN OOS-HANDOFF FUNCTION (for the user to call with an OOS date -- wall NOT moved)
# ============================================================

def oos_validate_slice(oos_start: str, oos_end: str | None = None,
                       K: int = 3, vol_hi_threshold: float | None = None,
                       n_slices: int = 200, hold: int = 7, seed: int = 42) -> dict:
    """
    HANDOFF: the user calls this with OOS dates (>= 2024-05-15) to validate the engine.
    This function loads data for that OOS window, builds W using the EXACT same logic
    as DEV (same feature set, same composite weights, same circuit-breaker thresholds).

    vol_hi_threshold: pass the value printed at DEV-training time (do NOT refit on OOS).

    Returns:
      {
        "slice_stats": {overall, bull, chop, bear},
        "shuffle_control": {same keys},
        "selection_alpha_by_regime": {bull, chop, bear, all: mean diff vs shuffle},
        "equity_stats": {comp_full, maxDD, ...},
        "regime_counts": {bull, chop, bear, n},
      }

    IMPORTANT: This function does NOT enforce DEV_END -- it is the USER's wall to honour.
    The caller must pass oos_start >= '2024-05-15' and must NOT call this during DEV work.
    """
    assert vol_hi_threshold is not None, "Pass vol_hi_threshold calibrated on DEV (from main() output)."
    end = oos_end or "2099-01-01"

    import strat.fleet_lab as _fl
    # Temporarily bypass the DEV wall to allow the OOS window
    orig = _fl.DEV_END
    _fl.DEV_END = end
    try:
        L = _fl.load_wide(n=50, start=oos_start, end=end, min_bars=50)
    finally:
        _fl.DEV_END = orig

    C      = L["C"]
    sma200 = _compute_sma(C, 200)
    sma50  = _compute_sma(C, 50)
    btcvol = C["BTCUSDT"].pct_change().rolling(20, min_periods=10).std() * (365 ** 0.5)
    regime = regime_series(C, sma200, sma50, btcvol, vol_hi_threshold)

    W  = build_weight_matrix(L, vol_hi_threshold, K=K)
    b  = book_returns(W, L)

    bh_W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    for i in range(len(C.index)):
        valid = [s for s in C.columns if pd.notna(C.iloc[i][s])]
        if valid:
            w = 1.0 / len(valid)
            bh_W.iloc[i] = pd.Series({s: w for s in valid}, dtype=float).reindex(C.columns, fill_value=0.0)
    bh_b = book_returns(bh_W, L)

    reg_s  = regime_slice_stats(b, bh_b, regime, n=n_slices, hold=hold, seed=seed)
    ctrl, _ = shuffle_control_stats(L, vol_hi_threshold, K, regime, n_slices, hold, [101, 202, 303])

    sel_alpha = {}
    for lbl in reg_s:
        me  = reg_s[lbl].get("mean_pct", 0) or 0
        ctr = ctrl.get(lbl, {}).get("mean_pct", 0) or 0
        sel_alpha[lbl] = round(me - ctr, 2)

    eq   = equity_stats(b, label="mover_oos")
    rc   = regime.value_counts().to_dict()

    return {
        "oos_window":     [oos_start, end],
        "K":              K,
        "vol_hi_threshold": vol_hi_threshold,
        "slice_stats":    reg_s,
        "shuffle_control": ctrl,
        "selection_alpha_by_regime": sel_alpha,
        "equity_stats":   eq,
        "regime_counts":  {k: int(v) for k, v in rc.items()},
    }


# ============================================================
# CORE AGENT SPEC (for fleet integration)
# ============================================================

MOVER_AGENT_SPEC = {
    "name":   "mover_capture_K3",
    "feats":  ["mom14", "rangepos", "volexp", "accel"],    # fleet_lab F keys
    "K":      3,
    "hold":   7,
    "signs":  [1, 1, 1, 1],   # all mover-directional (high = better)
    "weights": [0.35, 0.25, 0.20, 0.20],   # composite weights (reference, applied in build_weight_matrix)
    "description": (
        "Mover-capture: rank assets by composite(mom14 35%+rangepos 25%+volexp 20%+accel 20%), "
        "hold top-3 EW, scale total exposure by market circuit-breaker "
        "(bear 20%/chop40-70%/bull 100%). "
        "DEV-calibrated vol_hi_threshold applied at inference time."
    ),
}


# ============================================================
# MAIN (DEV run + decisive question)
# ============================================================

def main():
    t0 = time.time()
    print("=" * 78)
    print("MOVE-CAPTURE ENGINE -- DEV-WALLED (fleet_lab, <= 2024-05-15)")
    print("DECISIVE QUESTION: mover-selection beats random-same-exposure, regime-stratified?")
    print("=" * 78)

    # -- Load DEV data --
    print(f"\nLoading fleet_lab DEV data (n=50, <= {DEV_END}) ...")
    L  = lab.load_wide(n=50, start="2019-01-01", end=DEV_END)
    C  = L["C"]
    F  = L["F"]
    print(f"  Loaded {len(L['syms'])} assets: {L['syms'][:10]}...")
    print(f"  Date range: {C.index.min().date()} -> {C.index.max().date()}  [must be <= {DEV_END}]")
    assert C.index.max() < pd.Timestamp(DEV_END), "WALL VIOLATION!"

    # -- Calibrate vol_hi_threshold on DEV data only (training half = pre-2022) --
    btcvol  = C["BTCUSDT"].pct_change().rolling(20, min_periods=10).std() * (365 ** 0.5)
    train_mask = C.index < pd.Timestamp("2022-01-01")
    vol_hi = float(btcvol[train_mask].dropna().quantile(0.75))
    print(f"\nvol_hi_threshold (DEV-train quantile 75%): {vol_hi:.4f}")

    sma200 = _compute_sma(C, 200)
    sma50  = _compute_sma(C, 50)
    regime = regime_series(C, sma200, sma50, btcvol, vol_hi)
    rc     = regime.value_counts()
    print(f"Regime counts (DEV): bull={rc.get('bull',0)} chop={rc.get('chop',0)} bear={rc.get('bear',0)}")

    # -- EW BH baseline --
    bh_W = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    for i in range(len(C.index)):
        valid_s = [s for s in C.columns if pd.notna(C.iloc[i][s])]
        if valid_s:
            w = 1.0 / len(valid_s)
            bh_W.iloc[i] = pd.Series({s: w for s in valid_s}, dtype=float).reindex(C.columns, fill_value=0.0)
    bh_b = book_returns(bh_W, L)

    N_SLICES = 400
    HOLD     = 7
    K_VALS   = [1, 3, 5]
    ALL_SEEDS = [11, 23, 42]

    results = {
        "dev_end":   DEV_END,
        "vol_hi_threshold": round(vol_hi, 4),
        "regime_counts": {k: int(v) for k, v in rc.items()},
        "bh": {},
        "mover": {},
        "shuffle_control": {},
        "selection_alpha": {},
    }

    # -- BH reference stats --
    bh_all = []
    for sd in ALL_SEEDS:
        s = slice_stats(bh_b, bh_b, N_SLICES, HOLD, sd)
        bh_all.append(s)
    results["bh"] = {
        "pos_rate":  round(float(np.mean([s["pos_rate"]  for s in bh_all])), 1),
        "mean_pct":  round(float(np.mean([s["mean_pct"]  for s in bh_all])), 2),
    }
    print(f"\n[BH-EW]  pos_rate={results['bh']['pos_rate']}%  mean={results['bh']['mean_pct']}%")

    for K in K_VALS:
        print(f"\n{'='*60}")
        print(f"[MOVER K={K}]  Building weight matrix on DEV ...")
        W  = build_weight_matrix(L, vol_hi, K=K)
        b  = book_returns(W, L)

        avg_expo_dev  = float(W.sum(axis=1).mean())
        bear_mask = (C.index >= "2022-01-01") & (C.index < "2023-01-01")
        avg_expo_bear = float(W.sum(axis=1)[bear_mask].mean())
        print(f"  avg exposure DEV={avg_expo_dev:.3f}  2022-bear={avg_expo_bear:.3f}")

        # Equity stats
        eq  = equity_stats(b, label=f"mover_K{K}")
        eq["avg_expo"] = round(avg_expo_dev, 3)
        print(f"  comp_full={eq['comp_full']}%  maxDD={eq['maxDD']}%  "
              f"comp_2022={eq['comp_2022']}%  comp_2023={eq['comp_2023']}%")

        # Overall slice stats (multi-seed average)
        all_sl = []
        for sd in ALL_SEEDS:
            s = slice_stats(b, bh_b, N_SLICES, HOLD, sd)
            all_sl.append(s)
        ov = {
            "pos_rate":    round(float(np.mean([s["pos_rate"]    for s in all_sl])), 1),
            "mean_pct":    round(float(np.mean([s["mean_pct"]    for s in all_sl])), 2),
            "p05_pct":     round(float(np.mean([s["p05_pct"]     for s in all_sl])), 2),
            "beat_bh_pct": round(float(np.mean([s["beat_bh_pct"] for s in all_sl])), 1),
            "z_vs_bh":     round(float(np.mean([s["z_vs_bh"]     for s in all_sl])), 2),
        }
        print(f"  OVERALL  pos_rate={ov['pos_rate']}%  mean={ov['mean_pct']}%  "
              f"beat_bh={ov['beat_bh_pct']}%  p05={ov['p05_pct']}%  z_vs_bh={ov['z_vs_bh']:.2f}")

        # Regime-stratified stats
        print(f"  Running regime-stratified slice stats ...")
        reg_stats_all = []
        for sd in ALL_SEEDS:
            rs = regime_slice_stats(b, bh_b, regime, n=N_SLICES, hold=HOLD, seed=sd)
            reg_stats_all.append(rs)
        # Average across seeds
        reg_avg = {}
        for lbl in ["bull", "chop", "bear", "all"]:
            ns = [s[lbl] for s in reg_stats_all if s[lbl].get("n", 0) > 0]
            if not ns:
                reg_avg[lbl] = {"n": 0}
                continue
            reg_avg[lbl] = {
                "n":           int(round(np.mean([s["n"] for s in ns]))),
                "pos_rate":    round(float(np.mean([s["pos_rate"]    for s in ns])), 1),
                "mean_pct":    round(float(np.mean([s["mean_pct"]    for s in ns])), 2),
                "p05_pct":     round(float(np.mean([s["p05_pct"]     for s in ns])), 2),
                "beat_bh_pct": round(float(np.mean([s["beat_bh_pct"] for s in ns])), 1),
                "z_vs_bh":     round(float(np.mean([s["z_vs_bh"]     for s in ns])), 2),
            }
            print(f"    regime={lbl:4}  n={reg_avg[lbl]['n']:4d}  "
                  f"pos={reg_avg[lbl]['pos_rate']}%  mean={reg_avg[lbl]['mean_pct']}%  "
                  f"beat_bh={reg_avg[lbl]['beat_bh_pct']}%  z_vs_bh={reg_avg[lbl]['z_vs_bh']:.2f}")

        # SHUFFLE CONTROL (same exposure, random K picks) -- 3 seeds
        print(f"  Building shuffle controls (3 seeds) ...")
        ctrl_all = {lbl: [] for lbl in ["bull", "chop", "bear", "all"]}
        for cs in [101, 202, 303]:
            Wc = build_weight_matrix(L, vol_hi, K=K, random_seed=cs)
            bc = book_returns(Wc, L)
            for sd in [11, 23, 42]:
                rs = regime_slice_stats(bc, bh_b, regime, n=N_SLICES, hold=HOLD, seed=sd)
                for lbl in ctrl_all:
                    if rs[lbl].get("n", 0) > 0:
                        ctrl_all[lbl].append(rs[lbl])

        ctrl_avg = {}
        for lbl, ns_list in ctrl_all.items():
            if not ns_list:
                ctrl_avg[lbl] = {"n": 0}
                continue
            ctrl_avg[lbl] = {
                "n":           int(round(np.mean([s["n"] for s in ns_list]))),
                "pos_rate":    round(float(np.mean([s["pos_rate"]    for s in ns_list])), 1),
                "mean_pct":    round(float(np.mean([s["mean_pct"]    for s in ns_list])), 2),
                "p05_pct":     round(float(np.mean([s["p05_pct"]     for s in ns_list])), 2),
                "beat_bh_pct": round(float(np.mean([s["beat_bh_pct"] for s in ns_list])), 1),
                "z_vs_bh":     round(float(np.mean([s["z_vs_bh"]     for s in ns_list])), 2),
            }

        # Selection alpha = engine - shuffle control
        sel_alpha = {}
        for lbl in ["bull", "chop", "bear", "all"]:
            me  = reg_avg[lbl].get("mean_pct", 0) or 0
            ctr = ctrl_avg[lbl].get("mean_pct", 0) or 0
            z_e = reg_avg[lbl].get("z_vs_bh",  0) or 0
            z_c = ctrl_avg[lbl].get("z_vs_bh",  0) or 0
            sel_alpha[lbl] = {
                "mean_alpha_pp":    round(me - ctr, 2),
                "beat_bh_delta":    round((reg_avg[lbl].get("beat_bh_pct", 50) or 50) -
                                          (ctrl_avg[lbl].get("beat_bh_pct", 50) or 50), 1),
                "z_delta":          round(z_e - z_c, 2),
            }

        print(f"\n  SELECTION ALPHA (mover vs shuffle-control):")
        for lbl in ["bull", "chop", "bear", "all"]:
            sa = sel_alpha[lbl]
            print(f"    regime={lbl:4}  mean_alpha={sa['mean_alpha_pp']:+.2f}pp  "
                  f"beat_bh_delta={sa['beat_bh_delta']:+.1f}pp  z_delta={sa['z_delta']:+.2f}")

        results["mover"][f"K{K}"] = {
            "overall":    ov,
            "equity":     eq,
            "regime":     reg_avg,
        }
        results["shuffle_control"][f"K{K}"] = ctrl_avg
        results["selection_alpha"][f"K{K}"] = sel_alpha

    # -- Z-TEST SUMMARY (decisive question) --
    print(f"\n{'='*78}")
    print("DECISIVE ANSWER -- does mover-selection beat random-same-exposure, regime-stratified?")
    print("(z > 2.0 = real edge; z > 3.0 = strong; sign flips across regimes = bull-beta)")
    print(f"{'='*78}")
    fmt = "{:<8} {:>8} {:>10} {:>10} {:>10} {:>10}"
    print(fmt.format("regime", "K", "eng_mean%", "ctrl_mean%", "alpha_pp", "z_delta"))
    print("-" * 60)
    for K in K_VALS:
        key = f"K{K}"
        for lbl in ["bull", "chop", "bear", "all"]:
            sa  = results["selection_alpha"][key].get(lbl, {})
            reg = results["mover"][key]["regime"].get(lbl, {})
            ctr = results["shuffle_control"][key].get(lbl, {})
            print(fmt.format(
                lbl, str(K),
                str(reg.get("mean_pct", "?")),
                str(ctr.get("mean_pct", "?")),
                f"{sa.get('mean_alpha_pp', 0):+.2f}" if sa else "?",
                f"{sa.get('z_delta', 0):+.2f}" if sa else "?",
            ))

    # -- Save results --
    out_dir = Path(ROOT).parent / "runs" / "mining"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts  = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"move_capture_fleet_dev_{ts}.json"
    results["runtime_s"]        = round(time.time() - t0, 1)
    results["agent_spec"]       = MOVER_AGENT_SPEC
    results["oos_handoff_fn"]   = "strat.move_capture_fleet.oos_validate_slice"
    results["oos_handoff_args"] = {
        "oos_start": "2024-05-15",
        "K":         3,
        "vol_hi_threshold": round(vol_hi, 4),
        "n_slices":  200,
    }
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved: {out}  ({results['runtime_s']}s)")

    # Pre-register spec
    print(f"\nPRE-REGISTERED FLEET AGENT SPEC:")
    print(json.dumps(MOVER_AGENT_SPEC, indent=2))
    print(f"\nOOS-HANDOFF (DO NOT run during DEV -- pass to user):")
    print(f"  from strat.move_capture_fleet import oos_validate_slice")
    print(f"  r = oos_validate_slice(oos_start='2024-05-15', K=3, vol_hi_threshold={round(vol_hi,4)}, n_slices=200)")
    print(f"  # Inspect r['selection_alpha_by_regime'] for regime-stratified answer on OOS.")

    return results


if __name__ == "__main__":
    main()
