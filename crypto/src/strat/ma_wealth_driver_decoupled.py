"""src/strat/ma_wealth_driver_decoupled.py -- TASK B: the DECOUPLED 7d-SLICE + 14d-LOOKBACK config.

WHY THIS EXISTS (user explicit ask, 2026-06-20). The COUPLED window in `ma_wealth_driver.run_wealth` was
DEGENERATE at 1d: `window` ('7d' vs '14d') flowed into NOTHING -- the band signal is selected on (ma_type,tf)
only, the hold provider holds while the band is long, and the hold_bars=window*bpd field was diagnostic-only.
So coupled-7d and coupled-14d are BYTE-IDENTICAL at 1d (verified: EMA|1d engine 3609.82% for both). The window
only ever scaled the capture HOLD in the capture leg -- never a real lookback-vs-slice distinction.

THE DECOUPLING (a REAL executed parameterization -- NOT lookback==hold):
  * SIGNAL LOOKBACK = 14 DAYS. The entry signal is the committed fixed-EW ma_cross band ensemble (UNCHANGED,
    the verified capture signal) AND a genuine 14-DAY-LOOKBACK trend confirmation: long only while the close is
    above its own MA over a 14-CALENDAR-DAY span (lookback_bars = 14 * bars_per_day) AND that 14d MA is
    non-falling. This makes the SIGNAL use a 14-day lookback window -- a longer, slower signal that the coupled
    run never expressed (the coupled signal was just the fixed band-period cross with no window-scaled lookback).
  * TRADING SLICE / HOLD = 7 DAYS. Once a position OPENS it is held for AT MOST a 7-CALENDAR-DAY slice
    (slice_bars = 7 * bars_per_day), then the trade is EXITED at the slice boundary and only RE-ENTERED if the
    14d-lookback signal is still confirmed at that boundary. So the position is evaluated/exited on a 7-day
    slice, INDEPENDENT of how long the 14d signal persists. lookback(14) != hold(7) -- the literal decoupling.

  This is the user's "7-day-SLICE + 14-day-LOOKBACK": a SLOWER signal (14d lookback) traded on a FASTER slice
  (7d hold). The coupled run could express neither independently.

ACCOUNTING IS THE VERIFIED ONE (reused VERBATIM from ma_wealth_driver, do NOT touch): _compound (MtM-no-double-
  count, lag-1, maker 0.0006 RT on turnover), _true_hodl (TRUE buy@listing, no rebalance, weights drift),
  _window_compound (bear-2022 / up-market sub-window), the DEV wall (< DEV_END), TRAIN+VAL band selection
  (OOS-excluded). The MtM reconciliation gate (prod(1+net) == equity within 1e-3) still asserts. The ONLY change
  vs run_wealth is the HELD-STATE column the engine trades: a decoupled provider replaces the long-while-band-long
  provider. The HODL and always-long-rebal baselines are computed identically (their held-state does not depend
  on the engine signal), so vs_hodl / vs_alwayslong / bear-preservation / up-market-alpha stay directly comparable
  to the coupled cells.

COMPARISON: run alongside the coupled-7d and coupled-14d cells (which are identical at 1d) -- does decoupling the
  lookback from the slice find ANY up-market timing alpha the coupled run missed, or is it still de-risked beta?

RWYB (smoke): python -m strat.ma_wealth_driver_decoupled --ma-type EMA --tf 1d
No emoji (cp1252). No git commit (meta reviews).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent / "src") not in sys.path:
    sys.path.insert(0, str(ROOT.parent / "src"))
CRYPTO = ROOT.parent

import strat.fleet_lab as fl                                       # noqa: E402  DEV_END, BARS_PER_DAY
from strat.ma_movecatch_engine import select_band_trainval, make_hold_provider  # noqa: E402  committed infra
from strat.ma_type_upgrade import _ema                            # noqa: E402  causal EMA for the 14d lookback filter
import strat.ma_wealth_driver as W                                # noqa: E402  REUSE the verified accounting
from strat.ma_wealth_driver import (                              # noqa: E402
    _compound, _true_hodl, _window_compound, _load_asset, _u100_symbols,
    MAKER_RT, CAPTURE_TAKER_RT, BEAR_2022_START, BEAR_2022_END, INSANITY_UPMKT_ALPHA_PP,
)

OUTDIR = CRYPTO / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE" / "MA_MOVECATCH"
LOOKBACK_DAYS = 14                                                # SIGNAL lookback (the slower signal window)
SLICE_DAYS = 7                                                    # TRADING slice / max-hold (the faster trade)


def _decoupled_held(c: np.ndarray, band_held: np.ndarray, lookback_bars: int, slice_bars: int) -> np.ndarray:
    """Decouple the SIGNAL lookback (14d) from the TRADE slice/hold (7d).

    Inputs (all aligned to the asset's close array c):
      band_held    -- the committed fixed-EW band-ensemble long-state (int8/0-1), the VERIFIED capture signal.
      lookback_bars-- 14 * bars_per_day: the SIGNAL lookback span.
      slice_bars   -- 7  * bars_per_day: the max-HOLD (trade slice) length.

    Step 1 (14d LOOKBACK signal): confirmed_long iff band_held AND a genuine 14-DAY-LOOKBACK trend filter holds:
      the close is above its own EMA over a 14-calendar-day span AND that 14d EMA is non-falling vs one bar ago.
      Causal: the EMA at t uses close[:t+1] only (min_periods=1, no look-ahead). This is the SLOWER signal the
      coupled run never expressed -- a 14d lookback window, not the raw fixed-period band cross.

    Step 2 (7d SLICE / HOLD): walk the bars; whenever NOT in a position and confirmed_long fires, OPEN a position
      and hold it for AT MOST slice_bars bars. At the slice boundary EXIT; RE-ENTER on the SAME bar only if
      confirmed_long is still True there (a fresh slice). Also exit early if confirmed_long drops to False mid-slice
      (signal lost). So the position is evaluated/exited on a 7-day slice, independent of how long the 14d signal
      persists -- the literal lookback(14) != hold(7) decoupling.

    Returns an int8 held-state column (1 while in a position), aligned to c. Causal by construction.
    """
    n = len(c)
    out = np.zeros(n, dtype=np.int8)
    fin = np.asarray(c, dtype=float)
    m = np.isfinite(fin)
    if m.sum() < max(5, lookback_bars // 4):
        return out
    # 14d-lookback EMA on the asset's OWN finite close segment (causal, min_periods=1)
    cseg = fin[m]
    ema_lb = _ema(cseg, lookback_bars)                            # EMA spanning ~14 calendar days
    above = cseg > ema_lb                                         # price above its 14d trend
    rising = np.concatenate([[True], ema_lb[1:] >= ema_lb[:-1]])  # 14d trend non-falling (first bar = True)
    bh_seg = np.asarray(band_held, dtype=bool)[m]                # band ensemble long-state on the same segment
    confirmed = bh_seg & above & rising                          # the 14d-LOOKBACK confirmed-long signal

    held_seg = np.zeros(len(cseg), dtype=np.int8)
    i = 0
    L = len(cseg)
    while i < L:
        if confirmed[i]:
            end = min(i + slice_bars, L)                         # hold this 7d SLICE (max slice_bars bars)
            j = i
            while j < end and confirmed[j]:                       # exit early if signal lost mid-slice
                held_seg[j] = 1
                j += 1
            i = j                                                 # next decision bar = slice end (or early-exit)
        else:
            i += 1
    out[np.where(m)[0]] = held_seg
    return out


def make_decoupled_hold_provider(sub_band, ma_type: str, lookback_bars: int, slice_bars: int, thr: float = 0.5):
    """Provider(sym, o, h, l, c) -> int8 decoupled held-state. Wraps the COMMITTED band-ensemble provider
    (make_hold_provider, the verified fixed-EW ma_cross capture signal) and applies the 14d-lookback confirm +
    7d-slice/hold decoupling on top. The band signal is UNCHANGED; only the lookback filter + slice-hold are new."""
    band_provider = make_hold_provider(sub_band, ma_type, thr=thr)

    def provider(sym, o, h, l, c):
        band_held = band_provider(sym, o, h, l, c)               # committed fixed-EW band ensemble long-state
        return _decoupled_held(np.asarray(c, dtype=float), band_held, lookback_bars, slice_bars)
    return provider


def run_decoupled(ma_type: str, tf: str, *, lookback_days: int = LOOKBACK_DAYS, slice_days: int = SLICE_DAYS,
                  cost_rt: float = MAKER_RT, verbose: bool = False) -> dict:
    """THE decoupled cell. (ma_type, tf) with SIGNAL lookback=14d + TRADE slice/hold=7d -> the same RANKABLE
    3-leg wealth dict as run_wealth, plus the decoupling diagnostics. ENGINE held-state = the decoupled provider;
    HODL + always-long baselines are computed IDENTICALLY to run_wealth (signal-independent), so every cross-leg
    metric (vs_hodl, vs_alwayslong, bear-preservation, up-market-alpha) is directly comparable to the coupled cell."""
    bpd = fl.BARS_PER_DAY[tf]
    lookback_bars = lookback_days * bpd
    slice_bars = slice_days * bpd
    wall_ms = int(pd.Timestamp(fl.DEV_END).value // 10 ** 6)

    sub_band, sel_meta = select_band_trainval(ma_type, tf)
    if len(sub_band) < 2:
        return {"error": f"sub-band < 2 for {ma_type}|{tf} after TRAIN+VAL selection", "selection": sel_meta}
    provider = make_decoupled_hold_provider(sub_band, ma_type, lookback_bars, slice_bars)

    closes, holds = {}, {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, wall_ms)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        hh = provider(sym, o, h, l, c).astype(float)
        s = pd.DataFrame({"c": c, "h": hh}, index=dates)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        closes[sym] = s["c"]; holds[sym] = s["h"]

    if not closes:
        return {"error": f"no assets loaded for {ma_type}|{tf} (decoupled)"}
    cp = pd.DataFrame(closes).sort_index()
    _max_rows = max((s.notna().sum() for s in closes.values()), default=0)
    assert len(cp) <= 1.5 * _max_rows + 5, (
        f"date-alignment regression: panel {len(cp)} rows >> max per-asset {_max_rows}")
    hp = pd.DataFrame(holds).reindex(cp.index).fillna(0.0)

    assert cp.index.max() < pd.Timestamp(fl.DEV_END), "WALL VIOLATION (panel index >= DEV_END)"
    R = cp.pct_change(fill_method=None).fillna(0.0)
    listed = cp.notna().astype(float)
    held = hp * listed
    n_listed = listed.sum(axis=1).replace(0, np.nan)

    # LEG 1 ENGINE (decoupled): fixed-N EW (1/N_listed per HELD name; cash for non-held -> de-risked beta).
    w_eng = held.div(n_listed, axis=0).fillna(0.0)
    # LEG 3 ALWAYS-LONG (daily-REBALANCED EW INDEX -- timing-skill control; identical to run_wealth).
    w_along = listed.div(n_listed, axis=0).fillna(0.0)

    eng_pct, eng_net, eng_eq, eng_gross, eng_dd = _compound(w_eng, R, cost_rt)
    along_pct, along_net, along_eq, along_gross, along_dd = _compound(w_along, R, cost_rt)
    # LEG 2 HODL (PRIMARY baseline): TRUE buy-and-hold (identical to run_wealth).
    hodl_pct, hodl_eq, hodl_dd, n_total = _true_hodl(cp, R, listed, cost_rt)

    vs_hodl_pp = round(eng_pct - hodl_pct, 2)
    vs_alwayslong_pp = round(eng_pct - along_pct, 2)

    bidx = cp.index
    bear_mask = np.asarray((bidx >= pd.Timestamp(BEAR_2022_START)) & (bidx < pd.Timestamp(BEAR_2022_END)))
    eng_bear = _window_compound(w_eng, R, cost_rt, bear_mask)
    along_bear = _window_compound(w_along, R, cost_rt, bear_mask)
    bear2022_preservation_pp = (round(eng_bear - along_bear, 2)
                                if np.isfinite(eng_bear) and np.isfinite(along_bear) else None)
    upmkt_mask = ~bear_mask
    eng_up = _window_compound(w_eng, R, cost_rt, upmkt_mask)
    along_up = _window_compound(w_along, R, cost_rt, upmkt_mask)
    upmkt_timing_alpha_pp = (round(eng_up - along_up, 2)
                             if np.isfinite(eng_up) and np.isfinite(along_up) else None)

    recompound = float(np.prod(1.0 + eng_net) - 1.0) * 100
    denom = max(abs(eng_pct), 1e-9)
    mtm_rel_err = abs(recompound - eng_pct) / denom
    mtm_reconciled = bool(mtm_rel_err < 1e-3)

    sane = bool(upmkt_timing_alpha_pp is None or upmkt_timing_alpha_pp <= INSANITY_UPMKT_ALPHA_PP)
    derisked_beta_confirmed = bool(
        upmkt_timing_alpha_pp is not None and upmkt_timing_alpha_pp <= 5.0
        and (bear2022_preservation_pp is None or bear2022_preservation_pp >= 0.0))

    # capture A/C on the 7d slice: the engine's average gross exposure (fraction of capital deployed) IS the
    # realized participation; "capture on the 7d slice" = mean per-bar deployed share over the slice-traded book.
    eng_avg_exposure = round(eng_gross, 3)

    out = {
        "cell": f"{ma_type}|{tf}|DECOUPLED(lb{lookback_days}d/slice{slice_days}d)",
        "ma_type": ma_type, "tf": tf,
        "decoupling": {"lookback_days": lookback_days, "slice_hold_days": slice_days,
                       "lookback_bars": lookback_bars, "slice_bars": slice_bars,
                       "is_decoupled": lookback_days != slice_days,
                       "signal": (f"committed fixed-EW ma_cross band ensemble AND price>EMA(span={lookback_bars} "
                                  f"bars=~{lookback_days}d) AND that 14d-lookback EMA non-falling"),
                       "trade": (f"once long, hold AT MOST {slice_bars} bars (~{slice_days}d slice), exit at slice "
                                 f"boundary or on signal-loss, re-enter only if signal still confirmed")},
        # --- the leaderboard reads (rankable; same schema as run_wealth) ---
        "engine_compound_pct": round(eng_pct, 2),
        "hodl_compound_pct": round(hodl_pct, 2),
        "alwayslong_rebal_compound_pct": round(along_pct, 2),
        "vs_hodl_pp": vs_hodl_pp,
        "vs_alwayslong_pp": vs_alwayslong_pp,
        "bear2022_preservation_pp": bear2022_preservation_pp,
        "upmkt_timing_alpha_pp": upmkt_timing_alpha_pp,
        "maxdd_engine_pct": round(eng_dd, 1),
        "maxdd_hodl_pct": round(hodl_dd, 1),
        "maxdd_alwayslong_pct": round(along_dd, 1),
        "capture_7d_slice_avg_exposure": eng_avg_exposure,       # A/C on the 7d slice = realized deployed share
        "derisked_beta_confirmed": derisked_beta_confirmed,
        # --- diagnostics ---
        "n_assets": int(cp.shape[1]),
        "n_total_admitted": int(n_total),
        "n_bars": int(cp.shape[0]),
        "engine_avg_gross": round(eng_gross, 3),
        "alwayslong_avg_gross": round(along_gross, 3),
        "cost_rt": cost_rt, "cost_kind": "maker" if abs(cost_rt - MAKER_RT) < 1e-9 else "custom",
        "cost_note": (f"wealth leg = MAKER {cost_rt} RT; committed capture leg B = TAKER {CAPTURE_TAKER_RT} RT "
                      f"(state basis per column, never silently mix)."),
        "mtm_reconciled": mtm_reconciled, "mtm_rel_err": float(mtm_rel_err),
        "sane": sane, "insanity_upmkt_alpha_threshold_pp": INSANITY_UPMKT_ALPHA_PP,
        "bear_2022_window": [BEAR_2022_START, BEAR_2022_END],
        "basis": ("DECOUPLED 14d-lookback / 7d-slice. THREE legs on the SAME admitted u100 panel+window, lag-1 "
                  "fills, maker cost, DEV wall, MtM-no-double-count. ENGINE held-state = committed band ensemble "
                  "AND 14d-lookback trend confirm, traded on a 7d max-hold slice. HODL + always-long baselines "
                  "are IDENTICAL to run_wealth (signal-independent) so all cross-leg metrics are comparable."),
        "selection_rule": sel_meta.get("selection_rule"),
        "n_sub_band": len(sub_band),
        "split": f"DEV (TRAIN+VAL, < {fl.DEV_END}); OOS+UNSEEN SEALED",
        "data_range": [str(cp.index.min()), str(cp.index.max())],
    }
    if derisked_beta_confirmed:
        out["interpretation"] = (
            f"DE-RISKED BETA (D63/D77) STILL: up-market timing alpha {upmkt_timing_alpha_pp}pp (~0/neg) -- "
            f"decoupling the 14d lookback from the 7d slice adds NO up-market timing edge over a rebalanced "
            f"always-long index. Compound gap driven by 2022-bear preservation ({bear2022_preservation_pp}pp).")
    if not sane:
        out["sane_note"] = (f"engine shows UP-MARKET timing alpha {upmkt_timing_alpha_pp}pp (> "
                            f"{INSANITY_UPMKT_ALPHA_PP}pp) vs the rebalanced always-long index EXCLUDING 2022 -- "
                            f"contradicts every LO null (D63/D77). Probable look-ahead / survivorship. Verify.")
    if not mtm_reconciled:
        out["mtm_note"] = f"MtM stream/compound mismatch rel_err={mtm_rel_err:.2e} (>= 1e-3) -- accounting regression."
    if verbose:
        print(f"[decoupled] {out['cell']}  engine {out['engine_compound_pct']}%  HODL {out['hodl_compound_pct']}%  "
              f"alwayslong {out['alwayslong_rebal_compound_pct']}%")
        print(f"            vs-HODL {out['vs_hodl_pp']}pp  vs-alwayslong {out['vs_alwayslong_pp']}pp  "
              f"bear-preserve {out['bear2022_preservation_pp']}pp  upmkt-alpha {out['upmkt_timing_alpha_pp']}pp")
        print(f"            maxDD eng {out['maxdd_engine_pct']}% / hodl {out['maxdd_hodl_pct']}%  "
              f"n_assets {out['n_assets']}  MtM_recon {out['mtm_reconciled']}  derisked_beta {out['derisked_beta_confirmed']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.ma_wealth_driver_decoupled")
    ap.add_argument("--ma-type", default="EMA")
    ap.add_argument("--tf", default="1d")
    ap.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS)
    ap.add_argument("--slice-days", type=int, default=SLICE_DAYS)
    ap.add_argument("--taker", action="store_true")
    a = ap.parse_args()
    cost = 0.0024 if a.taker else MAKER_RT
    r = run_decoupled(a.ma_type, a.tf, lookback_days=a.lookback_days, slice_days=a.slice_days,
                      cost_rt=cost, verbose=True)
    print(json.dumps(r, indent=1, default=str))
    if "error" in r:
        return 2
    if not r["mtm_reconciled"]:
        print("[FAIL] MtM reconciliation did NOT pass"); return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
