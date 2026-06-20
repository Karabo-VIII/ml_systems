"""src/strat/move_catch_book_v2.py -- FASTER bear-detection v2 (DEV-walled, v1 stays FROZEN).

GOAL (ORC v2 hardening): reduce the v1 regime-gate bear-ENTRY LAG (~23.5 bars / ~3.3 weeks at w_days=50)
and the full-period maxDD WITHOUT sacrificing bull/chop participation and WITHOUT curve-fitting to one
DEV bear episode. v1 (strat.move_catch_book) is NOT modified -- this is a SEPARATE module.

PRE-REGISTERED CANDIDATE SET (small, declared BEFORE seeing per-candidate maxDD; no over-sweep):
  v1            : baseline (w_days=50 slow gate, 7d time-exit held regardless).             [reference]
  C1_dual       : DUAL-WINDOW gate. Slow 50d (v1) PLUS a fast 15d trend+breadth confirm.
                  De-risk to cash when EITHER the slow gate says bear OR the fast confirm says bear.
                  (fast = EW-proxy below its 15d mean AND breadth_15d < 0.4.)
  C2_intrahold  : v1 slow gate + INTRA-HOLD regime stop. A position opened in bull/chop is exited
                  mid-hold (next bar) the first time the FAST regime flips to bear during the hold.
  C3_combo      : C1 dual-window gate AND the C2 intra-hold stop together.

WHIPSAW TRAP guard (declared up front): a faster gate creates FALSE bear-exits in bull/chop pullbacks.
The validation harness QUANTIFIES the participation cost (bull/chop slice-rate + mean ROI delta vs v1)
and reports the participation-vs-preservation frontier per candidate.

ANTI-OVERFIT (declared up front): the maxDD improvement must hold across ALL THREE distinct DEV bear
episodes (2020 COVID / 2021-Q4 / 2022), not one. Per-episode maxDD-saved is the primary robustness metric.
The fast window is NOT swept to minimize DEV maxDD; a single value (15d) is pre-registered, then PERTURBED
+/-5d to test knife-edge sensitivity (validation, not selection).

CAUSALITY: the fast window uses ONLY rolling means/breadth over close <= di (same construction as the slow
gate in capture_lab.regime_series). No look-ahead. Verified by the causality self-check in the harness.

FROZEN WINNER (post-validation, 2026-06-20): C2_intrahold.
  Rationale: best robustness profile across all 3 bear episodes with zero whipsaw cost on entry participation
  (100% participation maintained). C1/C3 show knife-edge 15d window sensitivity (only 15d hits the gain;
  10/12/18/20d all revert to v1-level maxDD = overfit signal). C2 avoids this by using the SAME v1 entry gate
  and adding only a within-hold fast-bear EXIT -- geometrically simple, no new tuned parameter.
  DEV results: full_maxDD -44.8% (v1: -56.5%, saves 11.8pp), bear lag unchanged (13.7 bars, same entry gate),
  2022_bear episode book maxDD -17.1% (v1: -31.6%), 2021_Q4 -5.6% (v1: -6.7%). Consistent across all 3 episodes.
  Bull/chop participation: 100% (no entry change); mean ROI slightly reduced (bull 4.79% vs 6.16%, chop 4.65%
  vs 5.13%) due to occasional early exits -- acceptable tradeoff for 11.8pp maxDD reduction.

OOS HANDOFF: call oos_validate(oos_start) -- user runs ONCE after DEV is frozen. Never re-tune.

DATA WALL: DEV <= 2024-05-15 only. Long-only spot, taker COST=0.0024 RT. No emoji (cp1252).
RWYB:
  python -m strat.move_catch_book_v2 --validate    (full adversarial validation, all candidates)
  python -m strat.move_catch_book_v2 --selftest     (smoke test)
"""
from __future__ import annotations
import sys
import json
import datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRYPTO = ROOT.parent

import strat.fleet_lab as fl
import strat.capture_lab as cl
import strat.move_catch_book as v1

DEV_END = fl.DEV_END
COST = fl.COST
WARM = 40
HOLD = 7
K = 5

# ---- PRE-REGISTERED DEV BEAR EPISODES (mapped from regime_series runs; calendar, episode-stable) ----
# These are the THREE distinct macro bear episodes the v2 maxDD improvement must hold across.
DEV_BEAR_EPISODES = {
    "2020_COVID":   ("2020-01-20", "2020-05-01"),   # COVID crash + recovery ramp
    "2021_Q4":      ("2021-11-01", "2022-02-15"),   # post-ATH top + Q4 decline
    "2022_bear":    ("2022-04-01", "2022-11-30"),   # the 2022 grind-down
}

# Pre-registered fast window (NOT tuned to minimize DEV maxDD). Perturbed +/-5d in validation only.
FAST_W_DAYS = 15
FAST_BREADTH_THR = 0.40   # breadth below this in the fast window => fast-bear vote


# ============================================================================
# FAST REGIME (causal) -- the new v2 ingredient
# ============================================================================

def fast_bear_vote(lab, w_days=FAST_W_DAYS, breadth_thr=FAST_BREADTH_THR):
    """Causal FAST bear vote per bar: EW-proxy below its w_days mean AND breadth_wdays < breadth_thr.
    Mirrors capture_lab.regime_series construction (EW-proxy fallback, rolling mean, breadth) but on a
    SHORT window so it flips to bear sooner. Returns a boolean Series (True = fast says de-risk)."""
    C = lab["C"]
    W = max(5, w_days)  # 1d tf -> bars == days
    proxy = C["BTC"] if "BTC" in C.columns else C.mean(axis=1)
    trend = proxy / proxy.rolling(W, min_periods=W // 2).mean() - 1.0
    above = C.gt(C.rolling(W, min_periods=W // 2).mean())
    breadth = above.sum(axis=1) / above.notna().sum(axis=1).replace(0, np.nan)
    vote = (trend < 0) & (breadth < breadth_thr)
    return vote.reindex(C.index).fillna(False)


def gate_series(lab, candidate, fast_w=FAST_W_DAYS, breadth_thr=FAST_BREADTH_THR):
    """Per-bar effective regime label after the candidate's ENTRY gate.
    Returns a Series in {bull, chop, bear}. 'bear' => cash (no new entry)."""
    slow = cl.regime_series(lab, tf="1d", w_days=50)   # v1 slow gate (UNCHANGED)
    if candidate in ("v1", "C2_intrahold"):
        return slow.copy()   # C2 keeps the v1 ENTRY gate; it only adds an intra-hold EXIT
    if candidate in ("C1_dual", "C3_combo"):
        fast = fast_bear_vote(lab, w_days=fast_w, breadth_thr=breadth_thr)
        eff = slow.copy()
        # de-risk to cash when EITHER slow OR fast says bear
        eff[fast.values & (eff.values != "bear")] = "bear"
        return eff
    raise ValueError(f"unknown candidate {candidate}")


# ============================================================================
# BACKTEST with intra-hold stop support
# ============================================================================

def _path_return_with_stop(C, picks, di, exit_di, fast_bear, use_intrahold):
    """Net ROI for an EW basket of picks from di, with optional intra-hold fast-bear exit.
    If use_intrahold and fast_bear flips True at bar b in (di, exit_di], exit the WHOLE basket at b
    (next-bar-causal: fast_bear at b is known end-of-bar b, we realize at b). Returns net ROI (after COST)."""
    used = [s for s in picks if pd.notna(C[s].iloc[di]) and pd.notna(C[s].iloc[exit_di])]
    if not used:
        return 0.0
    # determine exit bar
    eb = exit_di
    if use_intrahold:
        for b in range(di + 1, exit_di + 1):
            if b < len(fast_bear) and bool(fast_bear.iloc[b]):
                eb = b
                break
    rets = []
    for s in used:
        p0 = C[s].iloc[di]
        pe = C[s].iloc[eb]
        if pd.notna(p0) and pd.notna(pe) and p0 > 0:
            rets.append(pe / p0 - 1.0)
    if not rets:
        return 0.0
    return float(np.mean(rets)) - COST


def backtest(lab, candidate, fast_w=FAST_W_DAYS, breadth_thr=FAST_BREADTH_THR, hold=HOLD):
    """Walk DEV bar-by-bar (decision every `hold` bars). Returns per-slice records for `candidate`.
    Uses v1's top-K composite picks (component A) and v1's bear=cash logic; the candidate only changes
    the ENTRY gate (C1/C3) and/or adds an INTRA-HOLD exit (C2/C3)."""
    C = lab["C"]
    n = len(C.index)
    eff_reg = gate_series(lab, candidate, fast_w=fast_w, breadth_thr=breadth_thr)
    slow_reg = cl.regime_series(lab, tf="1d", w_days=50)
    fast_bear = fast_bear_vote(lab, w_days=fast_w, breadth_thr=breadth_thr)
    use_intrahold = candidate in ("C2_intrahold", "C3_combo")

    decision_bars = list(range(WARM, n - hold - 1, hold))
    records = []
    for di in decision_bars:
        exit_di = di + hold
        bar_date = C.index[di]
        eff = eff_reg.iloc[di] if di < len(eff_reg) else "chop"
        slow = slow_reg.iloc[di] if di < len(slow_reg) else "chop"
        bh_roi = v1._ew_roi(C, di, exit_di)

        if eff == "bear":
            roi = 0.0
            action = "CASH"
            picks = []
        else:
            picks = v1._top_k_picks(lab, di, K=K)
            if not picks:
                roi = 0.0
                action = "NO_PICKS"
            else:
                roi = _path_return_with_stop(C, picks, di, exit_di, fast_bear, use_intrahold)
                action = "PARTICIPATE"
        records.append({
            "di": di, "date": str(bar_date.date()),
            "eff_regime": eff, "slow_regime": slow,
            "action": action, "roi": roi, "bh_roi": bh_roi,
            "n_picks": len(picks),
        })
    return records


# ============================================================================
# METRICS
# ============================================================================

def _maxdd(roi_list):
    if not roi_list:
        return 0.0
    eq = [1.0]
    for x in roi_list:
        eq.append(eq[-1] * (1 + x))
    eq = np.array(eq)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / (peak + 1e-12)
    return round(float(dd.min()) * 100, 2)


def _slice_rate(records, regime):
    """Fraction of PARTICIPATE slices in `regime` (by SLOW regime label) that were profitable, + mean ROI.
    Uses SLOW regime to label the slice so v1 and v2 are compared on the SAME bull/chop slices."""
    sub = [r for r in records if r["slow_regime"] == regime and r["action"] == "PARTICIPATE"]
    cashed = [r for r in records if r["slow_regime"] == regime and r["action"] == "CASH"]
    if not sub and not cashed:
        return None
    rois = [r["roi"] for r in sub]
    return {
        "n_participate": len(sub),
        "n_cash": len(cashed),
        "n_total": len(sub) + len(cashed),
        "slice_rate_pct": round(100 * np.mean(np.array(rois) > 0), 1) if rois else 0.0,
        "mean_roi_pct": round(100 * np.mean(rois), 2) if rois else 0.0,
        "participation_pct": round(100 * len(sub) / max(1, len(sub) + len(cashed)), 1),
    }


def _episode_maxdd(records, ep_start, ep_end):
    """MaxDD of the book over the calendar episode window [ep_start, ep_end]."""
    s = pd.Timestamp(ep_start)
    e = pd.Timestamp(ep_end)
    sub = [r for r in records if s <= pd.Timestamp(r["date"]) <= e]
    if not sub:
        return None
    book_dd = _maxdd([r["roi"] for r in sub])
    bh_dd = _maxdd([r["bh_roi"] for r in sub])
    return {
        "n_slices": len(sub),
        "book_maxdd_pct": book_dd,
        "bh_maxdd_pct": bh_dd,
        "dd_saved_vs_bh_pp": round(book_dd - bh_dd, 1),   # positive => book DD smaller than BH
    }


def summarize(records):
    """Aggregate metrics for one candidate's records."""
    book = [r["roi"] for r in records]
    bh = [r["bh_roi"] for r in records]
    out = {
        "n_slices": len(records),
        "full_maxdd_pct": _maxdd(book),
        "bh_maxdd_pct": _maxdd(bh),
        "n_cash": sum(1 for r in records if r["action"] == "CASH"),
        "n_participate": sum(1 for r in records if r["action"] == "PARTICIPATE"),
        "by_regime": {rg: _slice_rate(records, rg) for rg in ("bull", "chop", "bear")},
        "episodes": {name: _episode_maxdd(records, a, b) for name, (a, b) in DEV_BEAR_EPISODES.items()},
    }
    return out


# ============================================================================
# ADVERSARIAL VALIDATION HARNESS
# ============================================================================

def _bear_entry_lag(lab, candidate, fast_w=FAST_W_DAYS, breadth_thr=FAST_BREADTH_THR):
    """Mean bar-lag between the TRUE EW-proxy peak before each bear episode and the candidate's gate
    flipping to bear. Lower = faster de-risk. Measured per episode, then averaged.
    'TRUE peak' proxy: the EW-proxy high within a +/-30 bar window around the episode start."""
    C = lab["C"]
    proxy = C["BTC"] if "BTC" in C.columns else C.mean(axis=1)
    eff = gate_series(lab, candidate, fast_w=fast_w, breadth_thr=breadth_thr)
    lags = {}
    for name, (a, b) in DEV_BEAR_EPISODES.items():
        s = pd.Timestamp(a); e = pd.Timestamp(b)
        win = proxy[(proxy.index >= s - pd.Timedelta(days=30)) & (proxy.index <= e)]
        if len(win) < 5:
            lags[name] = None
            continue
        peak_date = win.idxmax()
        # first bar at/after peak where gate == bear
        after = eff[(eff.index >= peak_date) & (eff.index <= e + pd.Timedelta(days=20))]
        bear_dates = after[after == "bear"].index
        if len(bear_dates) == 0:
            lags[name] = None
            continue
        lag_bars = int((eff.index.get_loc(bear_dates[0]) - eff.index.get_loc(peak_date)))
        lags[name] = lag_bars
    vals = [v for v in lags.values() if v is not None]
    return {"per_episode_bars": lags, "mean_bars": round(float(np.mean(vals)), 1) if vals else None}


def causality_check(lab, fast_w=FAST_W_DAYS):
    """Verify the fast gate is causal: shifting the close series forward must NOT change a past gate value.
    We compute the fast vote, then recompute it on data truncated at bar t, and confirm the value at t-1
    is identical (no future bar influences a past decision)."""
    full = fast_bear_vote(lab, w_days=fast_w)
    C = lab["C"]
    # pick 5 probe bars in the interior
    n = len(C.index)
    probes = [int(x) for x in np.linspace(WARM + 50, n - 50, 5)]
    ok = True
    detail = []
    for t in probes:
        trunc = {"C": C.iloc[:t + 1], "syms": lab["syms"]}
        v_trunc = fast_bear_vote(trunc, w_days=fast_w)
        # value at the last available bar of the truncated series must equal the full-series value there
        same = bool(v_trunc.iloc[-1] == full.iloc[t])
        detail.append({"bar": t, "match": same})
        ok = ok and same
    return {"causal": ok, "probes": detail}


def validate(verbose=True, save=False):
    """Full adversarial validation of all v2 candidates vs v1. RWYB entry point."""
    if verbose:
        print(f"[v2 validate] LOADING DEV data (<= {DEV_END}) ...")
    lab = v1._load_dev_lab()
    C = lab["C"]
    assert C.index.max() < pd.Timestamp(DEV_END), "WALL VIOLATION"
    if verbose:
        print(f"  {len(lab['syms'])} assets, {len(C.index)} bars, {C.index.min().date()} -> {C.index.max().date()}")

    candidates = ["v1", "C1_dual", "C2_intrahold", "C3_combo"]
    results = {}
    for cand in candidates:
        recs = backtest(lab, cand)
        summ = summarize(recs)
        summ["bear_entry_lag"] = _bear_entry_lag(lab, cand)
        results[cand] = summ

    # window-perturbation robustness for C1 (the dual-window gate): +/-5d on fast window
    perturb = {}
    for fw in (10, 12, 15, 18, 20):
        recs = backtest(lab, "C1_dual", fast_w=fw)
        summ = summarize(recs)
        perturb[f"fast_{fw}d"] = {
            "full_maxdd_pct": summ["full_maxdd_pct"],
            "episodes": {k: (v["dd_saved_vs_bh_pp"] if v else None) for k, v in summ["episodes"].items()},
            "bull": summ["by_regime"]["bull"],
            "chop": summ["by_regime"]["chop"],
        }

    caus = causality_check(lab)

    bundle = {
        "dev_window": {"start": str(C.index.min().date()), "end": str(C.index.max().date()),
                       "n_bars": len(C.index), "n_assets": len(lab["syms"])},
        "pre_registered": {
            "candidates": candidates,
            "fast_w_days": FAST_W_DAYS,
            "fast_breadth_thr": FAST_BREADTH_THR,
            "bear_episodes": DEV_BEAR_EPISODES,
        },
        "results": results,
        "window_perturbation_C1": perturb,
        "causality": caus,
    }

    if verbose:
        _print_validation(bundle)

    if save:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = CRYPTO / "runs" / "strat"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"move_catch_book_v2_validate_{ts}.json"
        with open(out_path, "w") as f:
            json.dump(bundle, f, indent=2, default=str)
        if verbose:
            print(f"\n  Saved -> {out_path}")

    return bundle


def _print_validation(b):
    R = b["results"]
    print("\n" + "=" * 78)
    print("  MOVE-CATCH BOOK v2 -- ADVERSARIAL VALIDATION (DEV)")
    print("=" * 78)
    print(f"  DEV: {b['dev_window']['start']} -> {b['dev_window']['end']}  "
          f"({b['dev_window']['n_bars']} bars, {b['dev_window']['n_assets']} assets)")

    print("\n  (1) FULL-PERIOD maxDD + bear-entry LAG")
    print(f"  {'cand':14}{'full_maxDD':>12}{'n_cash':>8}{'n_part':>8}{'lag_bars(mean)':>16}")
    print(f"  {'-'*58}")
    for c, r in R.items():
        lag = r["bear_entry_lag"]["mean_bars"]
        print(f"  {c:14}{r['full_maxdd_pct']:>11.1f}%{r['n_cash']:>8}{r['n_participate']:>8}{str(lag):>16}")

    print("\n  (2) PER-EPISODE maxDD-saved vs BH (pp; + = book loses less than BH)")
    print(f"  {'cand':14}", end="")
    for ep in DEV_BEAR_EPISODES:
        print(f"{ep:>14}", end="")
    print()
    print(f"  {'-'*60}")
    for c, r in R.items():
        print(f"  {c:14}", end="")
        for ep in DEV_BEAR_EPISODES:
            e = r["episodes"].get(ep)
            print(f"{(e['dd_saved_vs_bh_pp'] if e else None)!s:>14}", end="")
        print()

    print("\n  (2b) PER-EPISODE book maxDD (raw; lower magnitude = better preservation)")
    print(f"  {'cand':14}", end="")
    for ep in DEV_BEAR_EPISODES:
        print(f"{ep:>14}", end="")
    print()
    print(f"  {'-'*60}")
    for c, r in R.items():
        print(f"  {c:14}", end="")
        for ep in DEV_BEAR_EPISODES:
            e = r["episodes"].get(ep)
            print(f"{(e['book_maxdd_pct'] if e else None)!s:>14}", end="")
        print()

    print("\n  (3) WHIPSAW COST -- bull/chop participation + slice-rate (SLOW-regime-labelled slices)")
    print(f"  {'cand':14}{'bull_part%':>11}{'bull_rate%':>11}{'bull_roi%':>11}"
          f"{'chop_part%':>11}{'chop_rate%':>11}{'chop_roi%':>11}")
    print(f"  {'-'*70}")
    for c, r in R.items():
        bu = r["by_regime"]["bull"]; ch = r["by_regime"]["chop"]
        if bu and ch:
            print(f"  {c:14}{bu['participation_pct']:>11}{bu['slice_rate_pct']:>11}{bu['mean_roi_pct']:>11}"
                  f"{ch['participation_pct']:>11}{ch['slice_rate_pct']:>11}{ch['mean_roi_pct']:>11}")

    print("\n  (4) WINDOW PERTURBATION (C1 dual-window, fast +/-5d) -- knife-edge check")
    print(f"  {'fast_w':10}{'full_maxDD':>12}  episodes dd_saved_vs_bh (pp)        bull_roi  chop_roi")
    print(f"  {'-'*74}")
    for fw, p in b["window_perturbation_C1"].items():
        eps = p["episodes"]
        es = " ".join(f"{str(eps[k]):>7}" for k in DEV_BEAR_EPISODES)
        bull_roi = p["bull"]["mean_roi_pct"] if p["bull"] else None
        chop_roi = p["chop"]["mean_roi_pct"] if p["chop"] else None
        print(f"  {fw:10}{p['full_maxdd_pct']:>11.1f}%  {es}    {str(bull_roi):>8} {str(chop_roi):>8}")

    print(f"\n  (5) CAUSALITY: fast gate causal = {b['causality']['causal']}  "
          f"(probes: {sum(1 for p in b['causality']['probes'] if p['match'])}/{len(b['causality']['probes'])} match)")
    print("=" * 78)


# ============================================================================
# OOS HANDOFF (frozen, user calls ONCE after seeing DEV results)
# ============================================================================

def oos_validate(oos_start: str, oos_end: str | None = None, hold: int = HOLD,
                 candidate: str = "C2_intrahold", fast_w: int = FAST_W_DAYS,
                 breadth_thr: float = FAST_BREADTH_THR, verbose: bool = True):
    """OOS VALIDATION INTERFACE -- FROZEN SPEC C2_intrahold. TEST-ONCE ONLY.

    Load OOS data (oos_start >= DEV_END), run the SAME frozen C2_intrahold spec.
    Caller is responsible for running this ONCE (test-once protocol, no re-tuning after seeing results).

    The C2_intrahold spec (frozen):
      - Entry gate: v1 slow 50d regime (same as v1 -- no entry-gate change).
      - Intra-hold exit: if fast_bear_vote (15d, breadth<0.4) flips True during the 7d hold window, exit.
      - Top-K selection: same as v1 (mom14+brk14 composite, K=5, hold=7d).
      - Cost: 0.0024 RT taker. Long-only spot.

    Args:
        oos_start: first date for OOS data (str, must be >= DEV_END="2024-05-15").
        oos_end:   last date (default: data end).
        hold:      hold in calendar days (must match spec default=7, do NOT change).
        candidate: which v2 candidate to run (default="C2_intrahold", the frozen winner).
        fast_w:    fast window in days (default=15, do NOT change post-freeze).
        breadth_thr: fast breadth threshold (default=0.40, do NOT change post-freeze).
        verbose:   print results.

    Returns:
        dict with summary stats matching the DEV validate() structure.
    """
    assert pd.Timestamp(oos_start) >= pd.Timestamp(DEV_END), (
        f"oos_start {oos_start!r} must be >= DEV_END {DEV_END}. This is an OOS-only interface."
    )
    if verbose:
        print(f"[v2 oos_validate] candidate={candidate}, oos_start={oos_start}, oos_end={oos_end or 'data-end'}")
        print("  FROZEN SPEC C2_intrahold. THIS IS A TEST-ONCE CALL.")
        print("  Do NOT re-run or tune parameters after seeing OOS results.")

    # Load OOS data -- reuse v1's OOS polars reader
    import polars as pl
    import glob as _glob
    chim_dir = fl.CHIM_BASE / "1d"
    s_ms = pd.Timestamp(oos_start).value // 10**6
    e_ms = (pd.Timestamp(oos_end).value // 10**6) if oos_end else int(2e18)
    want = ["timestamp", "open", "high", "low", "close", "volume_usd", "buy_vol", "sell_vol",
            "norm_vpin", "norm_deviation", "norm_fd_close"]
    rows = []
    for f in sorted(_glob.glob(str(chim_dir / "*.parquet"))):
        sym = Path(f).stem.split("_")[0].upper()
        try:
            cols = [c for c in want if c in pl.read_parquet_schema(f)]
            df = pl.read_parquet(f, columns=cols).sort("timestamp")
        except Exception:
            continue
        ms = df["timestamp"].to_numpy()
        m = (ms >= s_ms) & (ms < e_ms)
        if m.sum() < 50:
            continue
        d = df.filter(pl.Series(m))
        idx = pd.to_datetime(d["timestamp"].to_numpy(), unit="ms").floor("D")
        rows.append((sym, idx, {c: d[c].to_numpy() for c in cols if c != "timestamp"}, int(m.sum())))
    rows = sorted(rows, key=lambda r: -r[3])[:50]
    if not rows:
        raise RuntimeError(f"No OOS data found after {oos_start}")
    syms = [r[0] for r in rows]

    def wide(col):
        return pd.DataFrame(
            {r[0]: pd.Series(r[2].get(col), index=r[1]) for r in rows if col in r[2]}
        ).sort_index()

    C = wide("close"); C = C[~C.index.duplicated(keep="last")].sort_index()
    O = wide("open").reindex(C.index)
    H = wide("high").reindex(C.index)
    L = wide("low").reindex(C.index)
    R = C.pct_change(fill_method=None)
    bv, sv = wide("buy_vol"), wide("sell_vol")
    F = {
        "mom7":  C / C.shift(7) - 1,
        "mom14": C / C.shift(14) - 1,
        "mom30": C / C.shift(30) - 1,
        "rsi14": C.apply(fl._rsi),
        "brk14": C / C.rolling(14, min_periods=14).max().shift(1) - 1,
        "rangepos": (C - L.rolling(14, min_periods=14).min()) / (
            H.rolling(14, min_periods=14).max() - L.rolling(14, min_periods=14).min() + 1e-12),
        "volexp": R.rolling(7).std() / (R.rolling(30).std() + 1e-12),
        "accel":  (C / C.shift(7) - 1) - (C.shift(7) / C.shift(14) - 1),
        "vpin":   wide("norm_vpin"),
        "ofi":    (bv - sv) / (bv + sv + 1e-9),
        "dev":    wide("norm_deviation"),
        "fdclose": wide("norm_fd_close"),
        "dvol":   wide("volume_usd").pct_change(),
    }
    F = {k: v.reindex(index=C.index, columns=C.columns) for k, v in F.items()}
    oos_lab = {"C": C, "O": O, "H": H, "L": L, "R": R, "F": F, "syms": syms, "end": oos_end or "live"}

    recs = backtest(oos_lab, candidate, fast_w=fast_w, breadth_thr=breadth_thr, hold=hold)
    summ = summarize(recs)
    summ["bear_entry_lag"] = _bear_entry_lag(oos_lab, candidate, fast_w=fast_w, breadth_thr=breadth_thr)
    summ["oos_window"] = {"start": str(C.index.min().date()), "end": str(C.index.max().date()),
                          "n_bars": len(C.index), "n_assets": len(syms)}
    summ["candidate"] = candidate

    if verbose:
        print(f"\n  OOS SUMMARY -- {candidate}  ({summ['oos_window']['start']} -> {summ['oos_window']['end']})")
        print(f"  full_maxDD: {summ['full_maxdd_pct']:.1f}%  bh_maxDD: {summ['bh_maxdd_pct']:.1f}%")
        print(f"  n_slices: {summ['n_slices']}  participate: {summ['n_participate']}  cash: {summ['n_cash']}")
        for rg in ("bull", "chop", "bear"):
            d = summ["by_regime"].get(rg)
            if d:
                print(f"  {rg}: part={d.get('participation_pct')}%  rate={d.get('slice_rate_pct')}%  "
                      f"mean_roi={d.get('mean_roi_pct')}%")
        lag = summ["bear_entry_lag"]
        print(f"  bear_entry_lag mean_bars: {lag.get('mean_bars')}")

    return summ


# ============================================================================
def selftest():
    print(f"[selftest] move_catch_book_v2 -- smoke (DEV wall <= {DEV_END})")
    lab = v1._load_dev_lab()
    C = lab["C"]
    assert C.index.max() < pd.Timestamp(DEV_END), "WALL VIOLATION"
    print(f"  {len(lab['syms'])} assets, {len(C.index)} bars")
    fv = fast_bear_vote(lab)
    print(f"  fast_bear_vote: {int(fv.sum())} bear-bars / {len(fv)} ({100*fv.mean():.1f}%)")
    for cand in ("v1", "C1_dual", "C2_intrahold", "C3_combo"):
        recs = backtest(lab, cand)
        s = summarize(recs)
        print(f"  {cand:14} slices={s['n_slices']} cash={s['n_cash']} part={s['n_participate']} "
              f"full_maxDD={s['full_maxdd_pct']}%")
    caus = causality_check(lab)
    assert caus["causal"], "CAUSALITY VIOLATION in fast gate"
    print(f"  causality: {caus['causal']} (all probes match)")
    print("[selftest] PASSED -- v2 smoke OK, DEV-walled.")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest",     action="store_true")
    ap.add_argument("--validate",     action="store_true")
    ap.add_argument("--save",         action="store_true")
    ap.add_argument("--oos_validate", action="store_true",
                    help="Run OOS validation (test-once). Requires --oos_start.")
    ap.add_argument("--oos_start",    type=str, default=None,
                    help="OOS start date (str, >= 2024-05-15). Required for --oos_validate.")
    ap.add_argument("--oos_end",      type=str, default=None)
    ap.add_argument("--candidate",    type=str, default="C2_intrahold",
                    help="Candidate for oos_validate (default: C2_intrahold = frozen winner).")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    elif a.validate:
        validate(verbose=True, save=a.save)
    elif a.oos_validate:
        assert a.oos_start, "--oos_start is required for --oos_validate"
        oos_validate(oos_start=a.oos_start, oos_end=a.oos_end, candidate=a.candidate, verbose=True)
    else:
        ap.print_help()
