"""LONG-ONLY ORACLE + the realizable engine's GAP to it (the proposed item).

The oracle = best-achievable in the ALLOWED space (long-only spot, lev=1), NOT an
unconstrained perfect predictor. Three levels, all on the SAME panel (u100, OOS-2024,
one TF, PROD cost), per non-overlapping 7d week + cumulative:

  L1  PERFECT-CASH-TIMING upper bound -- deploy each name iff its forward-slice return
      > cash (cash=0, lev-1 long-only). EW over deployed. The absolute long-only ceiling
      (perfect foresight of which names rise). %/wk + slice-profit-rate.
  L2  BEST-REALIZABLE causal ceiling -- the realizable MA-band signal with hindsight-
      OPTIMAL choice (best MA type per the OOS window; causal per-bar signal, hindsight
      type/cap pick). The ACHIEVABLE ceiling. Reuses _cached_band_held + held_rolling.
  L3  V2 engine -- the deployed frozen band, DE-LEAKED (lag1 regime), capped. The current
      deployable %/wk (honest, post leak-fix from leakcheck.json).

Then the GAP DNA: decompose V2 -> L2 -> L1 into:
  - WRONG CASH CALLS  (deployed in a down-week / cash in an up-week)  [V2 -> perfect-cash]
  - LATE ENTRY / EARLY EXIT (capture < available among correctly-held names) [partial-capture]
  - UNDER-CONCENTRATION (the EW/cap dilutes vs concentrating on winners)  [the cap dial]

Everything reuses the VERIFIED engine path (ma_slice_engine / ma_wealth_driver / canonical
exits), MtM/lag-1, the DE-LEAKED regime broadcast. 2025 SEALED (assert max bar < 2025).
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(ROOT))

import strat.ma_slice_engine as E
from strat.ma_slice_engine import (
    btc_regime, _load_asset, _u100_symbols, _cached_band_held, held_rolling,
    BARS_PER_DAY, OOS_WALL_MS, OOS_START, OOS_2024_END,
)
from strat.canonical_exits import COST_PROD
from strat.ma_type_upgrade import MA_TYPES

OUT = Path(__file__).resolve().parent
SLICE_DAYS = 7
LOOKBACK_DAYS = 14


def _align_regime_lag1(regime, dates):
    """De-leaked (causal) regime broadcast: a bar on day D reads regime[D-1] (last CLOSED
    1d bar). Mirrors leakcheck_falsifier mode='lag1'."""
    if regime is None:
        return np.ones(len(dates))
    norm = pd.DatetimeIndex(dates).normalize()
    regime = regime.shift(1)
    idx = regime.index.union(norm).drop_duplicates()
    ra = regime.reindex(idx).ffill().reindex(norm).fillna(0)
    return ra.to_numpy().astype(float)


def _build_panel(tf, ma_types):
    """Load the u100 panel at `tf`, OOS-2024. Returns aligned numpy arrays:
      rets[t,j], held[type][t,j] (lag-1, regime-gated handled by caller), dates[t], syms.
    held is the band-held (pre-regime, pre-lag) per MA type so L2/L3 can mix.
    """
    wall_ms = OOS_WALL_MS
    bpd = BARS_PER_DAY[tf]
    lookback_bars = LOOKBACK_DAYS * bpd
    slice_bars = SLICE_DAYS * bpd

    close_cols = {}
    held_by_type = {mt: {} for mt in ma_types}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, wall_ms)
        if rec is None:
            continue
        dts, o, h, l, c = rec
        close_cols[sym] = pd.Series(c, index=dts)
        for mt in ma_types:
            bh = _cached_band_held(mt, tf, sym, o, h, l, c, wall_ms)
            held = held_rolling(c, bh, lookback_bars, slice_bars)
            held_by_type[mt][sym] = pd.Series(held, index=dts)
    if len(close_cols) < 10:
        raise SystemExit("INSUFFICIENT_ASSETS")

    close_df = pd.DataFrame(close_cols).sort_index()
    # SEAL CHECK
    assert close_df.index.max() < pd.Timestamp("2025-01-01"), "2025 LEAK"
    rets = close_df.pct_change().to_numpy()[1:]
    dates = close_df.index[1:]
    syms = list(close_df.columns)
    held = {}
    for mt in ma_types:
        hd = pd.DataFrame(held_by_type[mt]).reindex(close_df.index).fillna(0).astype(np.int8)
        held[mt] = hd.shift(1).fillna(0).to_numpy()[1:]   # lag-1 (causal entry)
        held[mt + "__turn"] = pd.DataFrame(held_by_type[mt]).reindex(
            close_df.index).fillna(0).astype(np.int8).diff().abs().fillna(0).to_numpy()[1:]
    return rets, held, dates, syms, slice_bars, bpd, wall_ms


def _slice_bounds(dates, mask, slice_bars):
    """Non-overlapping slice index bounds within the masked OOS window."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    bounds = []
    i0 = idx[0]
    i = i0
    while i + slice_bars <= idx[-1] + 1:
        bounds.append((i, i + slice_bars))
        i += slice_bars
    return bounds


def _fwd_slice_ret(rets, t, slice_bars, j):
    """Forward compounded return of asset j over [t, t+slice_bars) (the slice ahead)."""
    seg = rets[t:t + slice_bars, j]
    seg = seg[np.isfinite(seg)]
    if len(seg) == 0:
        return np.nan
    return float(np.prod(1 + seg) - 1)


def _metrics(eng_w, slice_bars, bpd):
    if len(eng_w) == 0:
        return dict(lin_wk=0.0, compound=0.0, maxdd=0.0, sharpe=0.0,
                    pct_prof_7d=0.0, n_slices=0)
    lin_wk = float(np.mean(eng_w)) * bpd * 7 * 100
    comp = float(np.prod(1 + eng_w) - 1) * 100
    cum = np.cumprod(1 + eng_w)
    maxdd = (float(np.min((cum - np.maximum.accumulate(cum)) /
             np.maximum.accumulate(cum))) * 100) if len(cum) > 1 else 0.0
    sd = float(np.std(eng_w))
    sharpe = (float(np.mean(eng_w)) * bpd * 365) / (sd * np.sqrt(bpd * 365)) if sd > 1e-12 else 0.0
    slice_pnls = []
    i = 0
    while i + slice_bars <= len(eng_w):
        slice_pnls.append(float(np.prod(1 + eng_w[i:i + slice_bars]) - 1) * 100)
        i += slice_bars
    pct = sum(1 for s in slice_pnls if s > 0) / len(slice_pnls) if slice_pnls else 0.0
    return dict(lin_wk=round(lin_wk, 4), compound=round(comp, 1), maxdd=round(maxdd, 1),
                sharpe=round(sharpe, 3), pct_prof_7d=round(pct, 3), n_slices=len(slice_pnls))


# ---------------------------------------------------------------------------
# LEVEL 1-BAR: per-BAR perfect cash-timing (the accounting-consistent upper bound that
# shares the per-bar MtM basis with L2/L3). At each bar, deploy each name iff its NEXT-bar
# return > cash(=0), EW over deployed; charge a round-trip cost only when entering/exiting.
# This is the true per-bar absolute long-only ceiling; it dominates L2b by construction.
# ---------------------------------------------------------------------------
def level1_perfect_cash_bar(rets, mask, slice_bars, bpd, cost_rt):
    valid = np.isfinite(rets)
    rets_f = np.where(valid, rets, 0.0)
    deploy = valid & (rets > 0)                       # perfect foresight of THIS bar's sign
    n_dep = deploy.sum(axis=1)
    gross = np.where(n_dep > 0, (rets_f * deploy).sum(axis=1) / np.maximum(n_dep, 1), 0.0)
    # turnover cost: a name flips in/out of the deployed set
    dep_i = deploy.astype(np.int8)
    prev = np.vstack([np.zeros((1, dep_i.shape[1]), np.int8), dep_i[:-1]])
    turn = np.abs(dep_i - prev).sum(axis=1)
    cost = np.where(n_dep > 0, turn * (cost_rt / 2.0) / np.maximum(n_dep, 1), 0.0)
    eng = gross - cost
    m = _metrics(eng[mask], slice_bars, bpd)
    m["deploy"] = round(float(np.mean(n_dep[mask] > 0)), 4) if mask.sum() else 0.0
    return m, eng


# ---------------------------------------------------------------------------
# LEVEL 1: perfect cash-timing. Per non-overlapping 7d week, deploy each name iff its
# forward-7d (slice) return > 0, EW over deployed; if none qualify -> cash (0). This is a
# per-SLICE book (decide at slice start with perfect foresight of the slice). This is the
# MANDATE-headline form ("deploy iff forward-7d return > cash"); for the gap-DNA arithmetic
# we use the per-bar form above (one consistent MtM basis).
# ---------------------------------------------------------------------------
def level1_perfect_cash(rets, dates, mask, slice_bars, bpd, cost_rt):
    bounds = _slice_bounds(dates, mask, slice_bars)
    eng_slices = []        # one compounded return per week
    deployed_frac = []
    for (a, b) in bounds:
        # forward return of each name over THIS slice
        fr = np.array([_fwd_slice_ret(rets, a, slice_bars, j) for j in range(rets.shape[1])])
        deploy = np.isfinite(fr) & (fr > 0)
        deployed_frac.append(float(np.mean(deploy)) if np.isfinite(fr).any() else 0.0)
        if deploy.sum() == 0:
            eng_slices.append(0.0)            # all cash
            continue
        # EW over deployed names, gross slice return minus one round-trip cost per name
        ret_dep = fr[deploy]
        slice_ret = float(np.mean(ret_dep)) - cost_rt    # one entry+exit per week per name
        eng_slices.append(slice_ret)
    arr = np.array(eng_slices)
    lin_wk = float(np.mean(arr)) * 100 if len(arr) else 0.0
    pct = float(np.mean(arr > 0)) if len(arr) else 0.0
    comp = float(np.prod(1 + arr) - 1) * 100 if len(arr) else 0.0
    return dict(lin_wk=round(lin_wk, 4), pct_prof_7d=round(pct, 3), compound=round(comp, 1),
                n_slices=len(arr), mean_deployed_frac=round(float(np.mean(deployed_frac)), 3))


# ---------------------------------------------------------------------------
# LEVEL 2: best-realizable causal ceiling. The realizable band signal (causal per-bar),
# but HINDSIGHT-OPTIMAL choice over the allowed knobs: pick the best MA type for the OOS
# window AND apply the best (within reason) concentration. We report:
#   2a  best MA type (per-bar EW over held), de-leaked regime, PROD cost  (hindsight type pick)
#   2b  + concentrate on the strongest-momentum held names (top-K by trailing return), the
#       realizable concentration dial taken to its in-window-best K (hindsight K).
# This is CAUSAL signal + HINDSIGHT tuning = the achievable ceiling.
# ---------------------------------------------------------------------------
def level2_realizable(rets, held, dates, mask, slice_bars, bpd, reg_vals, cost_rt, ma_types,
                      trailing_bars):
    best = None
    per_type = {}
    for mt in ma_types:
        hl = held[mt]
        turn = held[mt + "__turn"]
        valid = np.isfinite(rets)
        rets_f = np.where(valid, rets, 0.0)
        n_held = (hl * valid).sum(axis=1)
        gross = np.where(n_held > 0, (rets_f * hl * valid).sum(axis=1) / np.maximum(n_held, 1), 0.0)
        cost = np.where(n_held > 0, (turn * valid).sum(axis=1) * (cost_rt / 2.0) /
                        np.maximum(n_held, 1), 0.0)
        eng = (gross - cost) * reg_vals
        m = _metrics(eng[mask], slice_bars, bpd)
        per_type[mt] = m["lin_wk"]
        if best is None or m["lin_wk"] > best[1]["lin_wk"]:
            best = (mt, m, eng)
    best_mt, best_m, best_eng = best

    # 2b: concentrate on top-K held names by trailing momentum (causal), best K in-window.
    bestK = None
    hl = held[best_mt]
    turn = held[best_mt + "__turn"]
    valid = np.isfinite(rets)
    rets_f = np.where(valid, rets, 0.0)
    # trailing return per name (causal, trailing_bars window)
    logr = np.where(valid, np.log1p(np.clip(rets, -0.99, None)), 0.0)
    trail = pd.DataFrame(logr).rolling(trailing_bars, min_periods=1).sum().to_numpy()
    for K in (3, 5, 8, 12, 20):
        w = np.zeros_like(hl, dtype=float)
        for t in range(hl.shape[0]):
            hi = np.where(hl[t] > 0)[0]
            if len(hi) == 0:
                continue
            order = hi[np.argsort(-trail[t, hi])][:K]
            w[t, order] = 1.0 / len(order)
        gross = (rets_f * w * valid).sum(axis=1)
        # cost: weight turnover
        wprev = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
        wturn = np.abs(w - wprev).sum(axis=1)
        cost = wturn * (cost_rt / 2.0)
        eng = (gross - cost) * reg_vals
        m = _metrics(eng[mask], slice_bars, bpd)
        if bestK is None or m["lin_wk"] > bestK[1]["lin_wk"]:
            bestK = (K, m, eng)
    return best_mt, best_m, best_eng, per_type, bestK


# ---------------------------------------------------------------------------
# LEVEL 3: V2 engine -- frozen band, de-leaked regime, capped. Use the CHOSEN deployable
# type, EW (= the no-cap raw band, which the cap then dials down). We report the raw EW
# de-leaked %/wk; the capped deployable (~15% cap) is a known ~ -57% haircut from the cap
# dial (EW 3.4 -> 1.4) documented in CURRENT.md; we report both via the cap multiplier.
# ---------------------------------------------------------------------------
def level3_v2(rets, held, mask, slice_bars, bpd, reg_vals, cost_rt, deploy_type):
    hl = held[deploy_type]
    turn = held[deploy_type + "__turn"]
    valid = np.isfinite(rets)
    rets_f = np.where(valid, rets, 0.0)
    n_held = (hl * valid).sum(axis=1)
    gross = np.where(n_held > 0, (rets_f * hl * valid).sum(axis=1) / np.maximum(n_held, 1), 0.0)
    cost = np.where(n_held > 0, (turn * valid).sum(axis=1) * (cost_rt / 2.0) /
                    np.maximum(n_held, 1), 0.0)
    eng = (gross - cost) * reg_vals
    m = _metrics(eng[mask], slice_bars, bpd)
    deploy = float(np.mean(n_held[mask] > 0)) if mask.sum() else 0.0
    m["deploy"] = round(deploy, 4)
    return m, eng


# ---------------------------------------------------------------------------
# GAP DNA: decompose V2(EW de-leaked) -> L2b(concentrated) -> L1(perfect cash) into:
#   wrong_cash_calls : the per-week loss from deploying in down-weeks / cash in up-weeks,
#                      isolated by replacing V2's deploy decision with perfect cash timing
#                      while keeping V2's EW capture among correctly-held names.
#   late_early       : partial-capture loss (V2 capture vs available among held names) ->
#                      the L2-best-type lift over V2 frozen-type at the per-bar level.
#   under_concentration : EW vs concentrate-on-winners (L2b best-K over L2a EW).
# Components measured as %/wk contributions that sum (approximately) to the V2->L1 gap.
# ---------------------------------------------------------------------------
def gap_dna(v2_wk, l2a_wk, l2b_wk, l1_wk):
    late_early = round(l2a_wk - v2_wk, 4)          # frozen-type -> best-type per-bar capture
    under_conc = round(l2b_wk - l2a_wk, 4)         # EW -> concentrate on winners
    wrong_cash = round(l1_wk - l2b_wk, 4)          # realizable ceiling -> perfect cash timing
    total = round(l1_wk - v2_wk, 4)
    comps = {"late_entry_early_exit_pp_wk": late_early,
             "under_concentration_pp_wk": under_conc,
             "wrong_cash_calls_pp_wk": wrong_cash}
    biggest = max(comps, key=comps.get)
    return {"components_pp_wk": comps, "total_gap_v2_to_perfect_pp_wk": total,
            "biggest_recoverable": biggest,
            "recoverable_headroom_v2_to_realizable_pp_wk": round(l2b_wk - v2_wk, 4),
            "uncapturable_info_limit_realizable_to_perfect_pp_wk": round(l1_wk - l2b_wk, 4)}


def run_tf(tf, deploy_type, ma_types):
    cost = COST_PROD
    rets, held, dates, syms, slice_bars, bpd, wall_ms = _build_panel(tf, ma_types)
    regime = btc_regime(wall_ms)
    reg_vals = _align_regime_lag1(regime, dates)
    win_s = pd.Timestamp(OOS_START); win_e = pd.Timestamp(OOS_2024_END)
    mask = (dates >= win_s) & (dates < win_e)
    trailing_bars = LOOKBACK_DAYS * bpd

    l1_slice = level1_perfect_cash(rets, dates, mask, slice_bars, bpd, cost)
    l1_bar_m, _ = level1_perfect_cash_bar(rets, mask, slice_bars, bpd, cost)
    best_mt, l2a_m, l2a_eng, per_type, bestK = level2_realizable(
        rets, held, dates, mask, slice_bars, bpd, reg_vals, cost, ma_types, trailing_bars)
    l2b_K, l2b_m, l2b_eng = bestK
    v2_m, v2_eng = level3_v2(rets, held, mask, slice_bars, bpd, reg_vals, cost, deploy_type)

    # gap-DNA on the consistent per-bar basis (L1-bar is the absolute ceiling).
    dna = gap_dna(v2_m["lin_wk"], l2a_m["lin_wk"], l2b_m["lin_wk"], l1_bar_m["lin_wk"])
    return {
        "tf": tf, "n_assets": len(syms), "window": "OOS-2024", "cost": "PROD(0.0028 RT)",
        "regime": "DE-LEAKED (lag1)", "slice_days": SLICE_DAYS, "lookback_days": LOOKBACK_DAYS,
        "deploy_type_v2": deploy_type,
        "L1_perfect_cash_timing_per_slice": l1_slice,
        "L1_perfect_cash_timing_per_bar": l1_bar_m,
        "L2a_best_realizable_type": {"ma_type": best_mt, **l2a_m, "per_type_lin_wk": per_type},
        "L2b_concentrated_bestK": {"K": l2b_K, "ma_type": best_mt, **l2b_m},
        "L3_v2_frozen_deleaked_EW": v2_m,
        "gap_dna": dna,
    }


def main():
    t0 = time.time()
    ma_types = list(MA_TYPES)
    print("=" * 100)
    print("LONG-ONLY ORACLE GAP STUDY -- u100, OOS-2024, PROD cost, DE-LEAKED regime")
    print("L1 perfect-cash | L2a best-type | L2b best-type+concentrate | L3 V2 frozen-EW")
    print("=" * 100)
    results = {}
    # 1d (best-slice deployable: VIDYA|1d/SMA|1d) + 4h (sweet spot: SMA|4h)
    for tf, dtype in (("1d", "SMA"), ("4h", "SMA")):
        r = run_tf(tf, dtype, ma_types)
        results[tf] = r
        l1, l1b, l2a, l2b, v2 = (r["L1_perfect_cash_timing_per_slice"],
                                 r["L1_perfect_cash_timing_per_bar"], r["L2a_best_realizable_type"],
                                 r["L2b_concentrated_bestK"], r["L3_v2_frozen_deleaked_EW"])
        print(f"\n[{tf}] n={r['n_assets']}  deploy_type=V2 {dtype}")
        print(f"  L1 perfect-cash (slice): {l1['lin_wk']:>+8.3f}%/wk  slice-prof {l1['pct_prof_7d']:>5.0%}"
              f"  (deployed {l1['mean_deployed_frac']:.0%}/wk)")
        print(f"  L1 perfect-cash (per-bar): {l1b['lin_wk']:>+7.3f}%/wk  [accounting-consistent ceiling]")
        print(f"  L2a best-type {l2a['ma_type']:<5}: {l2a['lin_wk']:>+7.3f}%/wk  slice-prof {l2a['pct_prof_7d']:>5.0%}")
        print(f"  L2b +concentrate K={l2b['K']:<2}: {l2b['lin_wk']:>+7.3f}%/wk  slice-prof {l2b['pct_prof_7d']:>5.0%}")
        print(f"  L3 V2 frozen-EW   : {v2['lin_wk']:>+7.3f}%/wk  slice-prof {v2['pct_prof_7d']:>5.0%}"
              f"  deploy {v2['deploy']:.0%}")
        d = r["gap_dna"]
        print(f"  GAP DNA (pp/wk): late/early {d['components_pp_wk']['late_entry_early_exit_pp_wk']:+.2f}"
              f" | under-conc {d['components_pp_wk']['under_concentration_pp_wk']:+.2f}"
              f" | wrong-cash {d['components_pp_wk']['wrong_cash_calls_pp_wk']:+.2f}"
              f"  => biggest: {d['biggest_recoverable']}")
        print(f"     recoverable (V2->realizable ceiling): {d['recoverable_headroom_v2_to_realizable_pp_wk']:+.2f} pp/wk"
              f" | un-capturable info limit: {d['uncapturable_info_limit_realizable_to_perfect_pp_wk']:+.2f} pp/wk")
    out = {"study": "long_only_oracle_gap", "generated": pd.Timestamp.utcnow().isoformat(),
           "binding": "long-only spot lev=1; rank by WEALTH; 2025 SEALED; DE-LEAKED regime; PROD cost",
           "elapsed_s": round(time.time() - t0, 1), "results": results}
    p = OUT / "oracle_gap.json"
    json.dump(out, open(p, "w"), indent=1, default=str)
    print(f"\n[json] {p}\n[elapsed] {out['elapsed_s']}s")


if __name__ == "__main__":
    main()
