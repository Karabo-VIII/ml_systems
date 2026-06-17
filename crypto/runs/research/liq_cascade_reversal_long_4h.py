"""runs/research/liq_cascade_reversal_long_4h.py

LIQUIDATION-CASCADE REVERSAL LONG (4h) -- the researcher's highest-ranked next-trigger.

THESIS
------
After a SHORT-LIQUIDATION FLUSH (shorts force-bought -> a violent up-spike / squeeze), enter LONG on a
RECLAIM confirmation, ride the continuation with the proven exit policy. Economic read: a short-flush
clears bearish positioning; if price then reclaims the prior bar's high IN AN UPTREND with a clean
directional ER, the squeeze is demand, not a wick -> a multi-candle MOVE.

REUSES THE PROVEN APPARATUS UNCHANGED (runs/research/minimal_3dof_4h_breakout.py):
  cadence 4h | EMA 8/21 trend conditioning (fast>slow) | Kaufman ER win20 HARD GATE er>0.40
  | ATR-trail (3.0x, win14) + time-stop 42 bars (7d) | canonical WindowSpec | taker 0.0024
  | SetupHarness (next-bar-open fill, intra-bar pessimistic) | firewall.random_entry_null (regime-matched)
Only the ENTRY TRIGGER is swapped: breakout-confirm -> liq short-flush + reclaim.

NEW DEGREES OF FREEDOM (<=3, PRE-REGISTERED before any held-out look):
  DOF1  Z_THR = 2.0      liq_short_z30 short-flush threshold (== pipeline SPIKE_Z_THRESH; principled)
  DOF2  reclaim          close[t] > high[t-1]  (4h bar reclaims prior bar's high; fixed, no numeric knob)
  DOF3  panic OR         OR liq_short_panic == 1 (binary; no threshold)
  (ER_GATE/ATR/time/MA all INHERITED from the proven apparatus -> NOT counted as DOF)

LOOK-AHEAD (the decisive subtlety)
----------------------------------
liq_short_z30 / liq_short_panic are DAILY features forward-filled identically across all 6 of a day's
4h bars (verified: 0/2334 days have intraday variation). So the task's literal `.shift(1)` (one 4h bar)
still lands on the SAME DAY's full-day liquidation total -> it leaks the rest of the day on 5/6 bars.
=> The STRICTLY past-only gate uses the PRIOR COMPLETED DAY's value (known before the current day opens).
We run BOTH:
   gate="literal_shift1" : task's literal .shift(1)  [LEAKY on this daily-filled substrate -- control]
   gate="safe_priorday"  : prior completed day's value [DECISIVE -- this is the believable result]
and render the harness's own leak_guard on each.

FALSIFIER (must pass to be a real edge):
  beat the REGIME-MATCHED cost-matched random-entry null on held-out (OOS & UNSEEN) AND positive there
  AND held-out sample discipline n_trades >= 10.

RWYB:  python runs/research/liq_cascade_reversal_long_4h.py [--assets BTCUSDT,ETHUSDT,...] [--quick]
LONG-ONLY, SPOT, lev 1. Does NOT commit / deploy / move capital.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.chimera_loader import ChimeraLoader            # noqa: E402
from pipeline.universe_loader import UniverseLoader          # noqa: E402
from wealth_bot.harness import WindowSpec, ema_past_only     # noqa: E402
from strat.setup_harness import SetupHarness, ExitPolicy     # noqa: E402
from strat.firewall import random_entry_null                 # noqa: E402
from strat.battery import evaluate as battery_evaluate       # noqa: E402

# ---- INHERITED proven-apparatus constants (UNCHANGED from minimal_3dof_4h_breakout) ---------------
CADENCE   = "4h"
TAKER     = 0.0024
MA_FAST, MA_SLOW = 8, 21          # EMA trend conditioning
ER_WIN    = 20                    # Kaufman ER lookback
ER_GATE   = 0.40                  # HARD GATE er>0.40 (clean directional move)
ATR_WIN   = 14
ATR_MULT  = 3.0
TIME_STOP = 42                    # 7 days = 42 * 4h bars
WIN = WindowSpec(train_end="2024-05-15", val_end="2025-03-15", oos_end="2025-12-31", unseen_end="2026-05-22")
WINDOWS = ["TRAIN", "VAL", "OOS", "UNSEEN"]
HELD = ["OOS", "UNSEEN"]

# ---- NEW pre-registered entry-trigger DOF ---------------------------------------------------------
Z_THR = 2.0                       # DOF1: liq_short_z30 short-flush threshold (== pipeline SPIKE_Z_THRESH)


def _load(loader, sym):
    try:
        g = loader.load(sym, cadence=CADENCE)
    except Exception:
        return None
    d = g.to_dict(as_series=False)
    raw = np.asarray(d["date"])
    dt = pd.to_datetime(raw, unit="ms") if np.issubdtype(raw.dtype, np.number) else pd.to_datetime(raw)
    for c in ("liq_short_z30", "liq_short_panic"):
        if c not in d:
            return None
    return pd.DataFrame({"date": dt,
                         "open": np.asarray(d["open"], float), "high": np.asarray(d["high"], float),
                         "low": np.asarray(d["low"], float), "close": np.asarray(d["close"], float),
                         "liq_short_z30": np.asarray(d["liq_short_z30"], float),
                         "liq_short_panic": np.asarray(d["liq_short_panic"], float)})


def _kaufman_er(close: pd.Series, win: int) -> pd.Series:
    change = (close - close.shift(win)).abs()
    vol_path = close.diff().abs().rolling(win, min_periods=win // 2).sum()
    return (change / vol_path.replace(0.0, np.nan)).clip(0.0, 1.0)


def _prior_day_value(df: pd.DataFrame, col: str) -> pd.Series:
    """Map each 4h bar to the PRIOR completed calendar day's daily value of `col`.
    The feature is constant within a day (verified); we take the per-day value, shift the per-day
    series by 1 day, and broadcast back -> strictly known before the current day opens (no intraday leak)."""
    day = df["date"].dt.floor("D")
    per_day = df.groupby(day)[col].first()       # one value per day (constant within day)
    prior = per_day.shift(1)                       # prior completed day
    return day.map(prior)


def build_entry(df: pd.DataFrame, gate: str) -> pd.DataFrame:
    """Past-only entry: short-flush(gate) AND reclaim AND trend-up AND ER-gate. gate in {literal_shift1, safe_priorday}."""
    out = df.copy().reset_index(drop=True)
    close, high, low = out["close"].astype(float), out["high"].astype(float), out["low"].astype(float)

    fast = ema_past_only(close, length=MA_FAST, shift=0)
    slow = ema_past_only(close, length=MA_SLOW, shift=0)
    er = _kaufman_er(close, ER_WIN).shift(1)
    reclaim = close > high.shift(1)                 # DOF2: reclaim prior bar's high (confirmed at close)

    if gate == "literal_shift1":                    # task's literal prior-4h-bar gate (LEAKY substrate)
        gz = out["liq_short_z30"].shift(1)
        gp = out["liq_short_panic"].shift(1)
    elif gate == "safe_priorday":                   # strictly past-only: prior completed DAY's value
        gz = _prior_day_value(out, "liq_short_z30")
        gp = _prior_day_value(out, "liq_short_panic")
    else:
        raise ValueError(gate)
    short_flush = (gz >= Z_THR) | (gp >= 0.5)       # DOF1 (Z_THR) + DOF3 (panic OR)

    # ATR (Wilder TR, rolling mean); SetupHarness reads atr[j-1] (prior bar) -> leak-safe trail width
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(ATR_WIN, min_periods=ATR_WIN // 2).mean()

    out["entry"] = (short_flush & reclaim & (fast > slow) & (er > ER_GATE)).fillna(False).astype(int)
    out["_short_flush"] = short_flush.fillna(False).astype(int)
    return out


def run_asset(df: pd.DataFrame, gate: str, n_books: int = 300):
    d = build_entry(df, gate)
    policy = ExitPolicy(atr_trail_mult=ATR_MULT, atr_col="atr", max_hold_bars=TIME_STOP)
    h = SetupHarness(d, "entry", policy, WIN, cost_rt=TAKER, regime_match_on_entry=True)
    res = h.run()
    fw = random_entry_null(h, n_books=n_books, seed=7, regime_matched=True)
    lg = h.leak_guard()
    rec = {"n_flush_bars": int(d["_short_flush"].sum()), "n_setups": int(d["entry"].sum())}
    for w in WINDOWS:
        ws = res.window_stats[w]
        nets = [t["net_pnl"] for t in res.trades if t["window"] == w]
        rec[w] = {"comp": round(ws.compound_pct, 2), "n": ws.n_trades, "wr": round(ws.win_rate, 3),
                  "exp_pct": round(float(np.mean(nets) * 100), 4) if nets else 0.0,
                  "dd": round(ws.max_dd_pct, 2),
                  "fw_real": fw["per_window"][w]["real"], "fw_p95": fw["per_window"][w]["null_p95"],
                  "fw_beats": fw["per_window"][w]["beats_null"]}
    rec["fw_beats_held"] = bool(fw["beats_held"])
    rec["fw_pos_held"] = bool(fw["pos_held"])
    rec["fw_verdict"] = fw["verdict"]
    rec["leak_guard"] = lg["verdict"]
    rec["held_n"] = sum(rec[w]["n"] for w in HELD)
    return rec, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=str, default="BTCUSDT")
    ap.add_argument("--quick", action="store_true", help="u100 first 25")
    ap.add_argument("--universe", type=str, default=None, choices=["u10", "u50", "u100"])
    ap.add_argument("--n-books", type=int, default=300)
    args = ap.parse_args()

    loader = ChimeraLoader()
    if args.universe:
        syms = UniverseLoader.load().list(args.universe)
        if args.quick:
            syms = syms[:25]
    else:
        syms = [s.strip().upper() for s in args.assets.split(",")]

    print(f"[LIQ-CASCADE-REVERSAL-LONG 4h] Z_THR={Z_THR} | INHERITED: EMA{MA_FAST}/{MA_SLOW} ER>{ER_GATE}"
          f"(w{ER_WIN}) ATR{ATR_MULT}x(w{ATR_WIN}) time{TIME_STOP}(7d) taker{TAKER} | assets={len(syms)}",
          flush=True)

    blob = {"config": {"cadence": CADENCE, "Z_THR": Z_THR, "er_gate": ER_GATE, "atr_mult": ATR_MULT,
                       "time_stop": TIME_STOP, "ma": [MA_FAST, MA_SLOW], "taker": TAKER,
                       "windows": WIN.__dict__}, "gates": {}}
    pooled = {}
    for gate in ("safe_priorday", "literal_shift1"):
        per_asset = {}
        pool = {w: [] for w in WINDOWS}
        fw_held_cnt = 0
        for s in syms:
            df = _load(loader, s)
            if df is None or len(df) < 600:
                continue
            rec, res = run_asset(df, gate, n_books=args.n_books)
            per_asset[s] = rec
            fw_held_cnt += int(rec["fw_beats_held"] and rec["fw_pos_held"])
            for w in WINDOWS:
                pool[w] += [t["net_pnl"] for t in res.trades if t["window"] == w]
        # pooled held-out view
        held_nets = np.array(pool["OOS"] + pool["UNSEEN"], float)
        pooled_held_comp = float((np.prod(1.0 + held_nets) - 1.0) * 100) if held_nets.size else 0.0
        agg = {"n_assets": len(per_asset),
               "fw_beats+pos_held_assets": fw_held_cnt,
               "pooled_held_n": int(held_nets.size),
               "pooled_held_compound_pct": round(pooled_held_comp, 2),
               "pooled_held_winrate": round(float((held_nets > 0).mean()), 3) if held_nets.size else None,
               "pooled_held_exp_pct": round(float(held_nets.mean() * 100), 4) if held_nets.size else None}
        blob["gates"][gate] = {"per_asset": per_asset, "aggregate": agg}
        pooled[gate] = agg

        print("\n" + "=" * 92)
        print(f"GATE = {gate}   ({'STRICTLY PAST-ONLY -> DECISIVE' if gate=='safe_priorday' else 'LITERAL .shift(1) -> LEAKY substrate, control only'})")
        print("=" * 92)
        for s, r in per_asset.items():
            print(f"  {s:10} flush_bars={r['n_flush_bars']:>4} setups={r['n_setups']:>3} | "
                  f"OOS comp={r['OOS']['comp']:>+8.2f} n={r['OOS']['n']:>3} p95={r['OOS']['fw_p95']} beats={r['OOS']['fw_beats']} | "
                  f"UNS comp={r['UNSEEN']['comp']:>+8.2f} n={r['UNSEEN']['n']:>3} p95={r['UNSEEN']['fw_p95']} beats={r['UNSEEN']['fw_beats']}")
            print(f"             held_n={r['held_n']:>3}  fw_beats_held={r['fw_beats_held']}  fw_pos_held={r['fw_pos_held']}  "
                  f"leak_guard={r['leak_guard'][:38]}")
        print(f"  -> assets beating regime-matched null + positive on held-out: {agg['fw_beats+pos_held_assets']}/{agg['n_assets']}")
        print(f"  -> POOLED held-out: n={agg['pooled_held_n']} compound={agg['pooled_held_compound_pct']}% "
              f"winrate={agg['pooled_held_winrate']} exp/trade={agg['pooled_held_exp_pct']}%")

    outp = ROOT / "runs" / "research" / "liq_cascade_reversal_long_4h_result.json"
    outp.write_text(json.dumps(blob, indent=2, default=str), encoding="utf-8")
    print(f"\n[saved] {outp}", flush=True)

    # ---- FALSIFIER VERDICT (decisive = safe_priorday) ----
    sp = pooled["safe_priorday"]
    print("\n" + "#" * 92)
    print("FALSIFIER VERDICT (decisive gate = safe_priorday, strictly past-only):")
    print(f"  assets beating regime-matched null + positive on held-out: {sp['fw_beats+pos_held_assets']}/{sp['n_assets']}")
    print(f"  pooled held-out n={sp['pooled_held_n']} (sample discipline n>=10: {sp['pooled_held_n']>=10}) "
          f"compound={sp['pooled_held_compound_pct']}%")
    print("#" * 92)
    return blob


if __name__ == "__main__":
    main()
