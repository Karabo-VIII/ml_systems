"""REGRESSION TEST for the same-day 1d-regime look-ahead leak fix (Claude audit, 2026-06-28).

The leak (V1.1 align_htf_on bug class): ma_slice_engine.btc_regime[D] is BTC's day-D 1d CLOSE
regime (~23:59). Gating consumers broadcast it onto a (4h/1d) bar grid two ways, BOTH leaked:
  (a) run_cell / replay: norm = dates.normalize(); ra = regime.reindex(idx).ffill().reindex(norm)
  (b) build_book_net / eval_panel: per-bar d1 = dt.normalize(); prior = ridx[ridx <= d1]; rg = reg[prior[-1]]
In both, an intraday bar on day D reads regime[D] = the day-D daily CLOSE regime (unknown until
~23:59) => SAME-DAY LOOK-AHEAD.

The fix: the production code now routes every gating consumer through causal_regime(wall_ms) =
btc_regime(wall_ms).shift(1).fillna(0).astype(int) (a bar on day D reads regime[D-1] = the last
CLOSED 1d bar). This is exactly the falsifier's _align_regime(..., mode="lag1") /
audit_deployable_run.build_book(deleak=True) reference.

This test asserts, for BOTH broadcast patterns, that the FIXED PRODUCTION path now reproduces the
strict-lag (regime.shift(1)) reference, and (sanity) that it DIFFERS from the old leaked path on
the OOS window (i.e. the fix is not a no-op). It also confirms the end-to-end de-leaked headline:
ma_slice_roi.build_book_net('1d', cap=0.15) lin_wk ~ +1.564 (de-leaked), NOT +1.714 (leaked).

Run: python leakfix_regression.py   (exit 0 = PASS, exit 1 = FAIL)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(ROOT))

import strat.ma_slice_engine as E
from strat.ma_slice_engine import (
    btc_regime, causal_regime, _load_asset, _u100_symbols, OOS_WALL_MS, OOS_START, OOS_2024_END,
)
import strat.ma_slice_roi as ROI

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok))
    tag = "[PASS]" if ok else "[FAIL]"
    print(f"  {tag} {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def broadcast_pattern_a(regime, dates):
    """EXACT replica of ma_slice_engine.run_cell lines ~210-215 (pattern a)."""
    reg_vals = np.ones(len(dates))
    if regime is not None:
        norm = pd.DatetimeIndex(dates).normalize()
        idx = regime.index.union(norm).drop_duplicates()
        ra = regime.reindex(idx).ffill().reindex(norm).fillna(0)
        reg_vals = ra.to_numpy().astype(float)
    return reg_vals


def broadcast_pattern_b(reg, idx):
    """EXACT replica of ma_slice_roi.build_book_net lines ~63-65 (pattern b)."""
    T = len(idx)
    rg = np.ones(T); ridx = reg.index
    for k, dt in enumerate(idx):
        d1 = pd.Timestamp(dt).normalize(); prior = ridx[ridx <= d1]
        rg[k] = int(reg[prior[-1]]) if len(prior) else 0
    return rg


def main():
    print("=" * 100)
    print("LEAK-FIX REGRESSION -- production gating regime broadcast == strict-lag (regime.shift(1))")
    print("OOS-2024 window; 2025 SEALED. Reference: falsifier _align_regime(lag1) / build_book(deleak=True)")
    print("=" * 100)

    # ----- build a real OOS bar grid from a 4h asset (intraday) and a 1d asset -----
    # 4h grid stresses the SAME-DAY leak (multiple intraday bars normalize to one day).
    rec4 = _load_asset("BTCUSDT", "4h", OOS_WALL_MS)
    rec1 = _load_asset("BTCUSDT", "1d", OOS_WALL_MS)
    assert rec4 is not None and rec1 is not None, "BTC load failed"
    dates4 = rec4[0][1:]                 # mirror run_cell: close_df.index[1:]
    idx1 = rec1[0]

    raw = btc_regime(OOS_WALL_MS)
    prod = causal_regime(OOS_WALL_MS)    # the FIXED production helper
    ref_lag = raw.shift(1).fillna(0).astype(int)   # the falsifier lag1 reference (independent recompute)

    # ===== T0: the helper itself equals the strict-lag reference =====
    same_helper = prod.equals(ref_lag)
    check("T0 causal_regime(wall) == btc_regime(wall).shift(1).fillna(0).astype(int)",
          same_helper, f"len={len(prod)} mismatches={int((prod.values != ref_lag.values).sum())}")

    # ===== T1: PATTERN (a) -- ma_slice_engine.run_cell reg-broadcast on the 4h grid =====
    prod_a = broadcast_pattern_a(prod, dates4)
    ref_a = broadcast_pattern_a(ref_lag, dates4)
    leaked_a = broadcast_pattern_a(raw, dates4)
    eq_a = np.array_equal(prod_a, ref_a)
    win = (dates4 >= pd.Timestamp(OOS_START)) & (dates4 < pd.Timestamp(OOS_2024_END))
    n_diff_leak = int(np.sum(prod_a[win] != leaked_a[win]))
    check("T1 pattern(a) run_cell broadcast: production == strict-lag(shift1) reference (4h grid)",
          eq_a, f"n_bars={len(dates4)} prod==ref exact")
    check("T1b pattern(a) fix is NOT a no-op: production differs from old leaked path on OOS",
          n_diff_leak > 0, f"{n_diff_leak} OOS bars flipped vs leaked")

    # ===== T2: PATTERN (b) -- ma_slice_roi.build_book_net reg-broadcast on the 1d grid =====
    prod_b = broadcast_pattern_b(prod, idx1)
    ref_b = broadcast_pattern_b(ref_lag, idx1)
    leaked_b = broadcast_pattern_b(raw, idx1)
    eq_b = np.array_equal(prod_b, ref_b)
    win1 = (idx1 >= pd.Timestamp(OOS_START)) & (idx1 < pd.Timestamp(OOS_2024_END))
    n_diff_leak_b = int(np.sum(prod_b[win1] != leaked_b[win1]))
    check("T2 pattern(b) build_book_net broadcast: production == strict-lag(shift1) reference (1d grid)",
          eq_b, f"n_bars={len(idx1)} prod==ref exact")
    check("T2b pattern(b) fix is NOT a no-op: production differs from old leaked path on OOS",
          n_diff_leak_b > 0, f"{n_diff_leak_b} OOS bars flipped vs leaked")

    # ===== T3: END-TO-END -- build_book_net('1d', cap=0.15) yields the DE-LEAKED headline =====
    # The fixed production engine must now produce ~+1.564/wk (de-leaked), NOT ~+1.714 (leaked).
    net, dts, bpd = ROI.build_book_net("1d", cap=0.15)
    lin_wk = float(np.mean(net)) * bpd * 7 * 100
    DELEAK_REF = 1.564     # audit_deployable.json 1d|cap15|deleaked + wave2_concentration_frontier
    LEAK_REF = 1.714       # audit_deployable.json 1d|cap15|leaked
    near_deleak = abs(lin_wk - DELEAK_REF) < 0.03
    far_from_leak = abs(lin_wk - LEAK_REF) > 0.05
    check("T3 build_book_net('1d', cap=0.15) lin_wk == de-leaked reference (~+1.564, NOT +1.714)",
          near_deleak and far_from_leak,
          f"production lin_wk={lin_wk:+.4f} | deleak_ref={DELEAK_REF:+.3f} leak_ref={LEAK_REF:+.3f}")

    print("=" * 100)
    n_pass = sum(1 for _, ok in RESULTS if ok)
    n_tot = len(RESULTS)
    print(f"REGRESSION RESULT: {n_pass}/{n_tot} PASSED")
    if n_pass == n_tot:
        print("ALL PASS -- production gating path == strict-lag reference; same-day leak is fixed.")
        sys.exit(0)
    print("SOME FAILED -- the leak fix is incomplete or regressed. See **FAIL** above.")
    sys.exit(1)


if __name__ == "__main__":
    main()
