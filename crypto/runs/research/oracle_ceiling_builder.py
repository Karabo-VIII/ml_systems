"""oracle_ceiling_builder.py -- perfect-foresight LONG-ONLY per-move-capture CEILING map.

WHAT (the realizable ceiling the MA-DNA must approach)
------------------------------------------------------
For each (cadence x asset) this computes the MAXIMUM compound return a clairvoyant
LONG-ONLY single-position trader could realize, given:
  * entry FILL = open[k]                 (honest next-bar-open fill, setup_harness contract)
  * exit       = high[j], j > k          (perfect-foresight: sell at the bar's HIGH)
  * hold-time band: min_hold_hours <= (ts[j]-ts[k]) < 7 days   (expressed per cadence in bars via real ts)
  * NON-OVERLAPPING single position (next entry >= exit bar + 1)
  * net round-trip TAKER cost = 0.0024 subtracted per move
  * objective = maximise COMPOUND return (product of per-move multipliers)

This is solved EXACTLY by a backward DP (max-product longest path):
    f[i] = max( f[i+1],  max_{valid j} (high[j]/open[i] - cost) * f[j+1] )
The DP only ever selects net-positive moves (skip dominates a losing trade), so every
"move" in the oracle is a captured up-leg. f, the trade list, per-window splits, and the
per-move distribution are recovered by backtracking.

This is a CEILING, not a strategy: it is intentionally clairvoyant on WHICH bar to enter
and WHICH future high to exit at. It uses honest fills (open entry) and honest cost (0.0024),
so it is the *realizable* upper bound -- the most a perfect timer could net.

dollar bars (n~4.1M) are too fine for the O(n*H) high-capture DP; for dollar we report the
O(n) unconstrained close-to-close multiplicative oracle as a clearly-labelled, looser
upper bound (perfect foresight on ultra-fine bars is granularity-degenerate -- see caveat).

RWYB:  .venv/Scripts/python.exe runs/research/oracle_ceiling_builder.py --selftest
       .venv/Scripts/python.exe runs/research/oracle_ceiling_builder.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from numba import njit  # noqa: E402

COST_RT = 0.0024
MIN_HOLD_HOURS = 1.0            # "hours" floor -> excludes sub-hour scalps; expressed per cadence in bars via ts
MAX_HOLD_MS = 7 * 24 * 3600 * 1000   # strict < 7 days
MS_PER_HOUR = 3600 * 1000

# setup_harness window split (exact parity for held-out comparability)
WIN = {
    "TRAIN":  ("0",            "2024-05-15"),
    "VAL":    ("2024-05-15",   "2025-03-15"),
    "OOS":    ("2025-03-15",   "2025-12-31"),
    "UNSEEN": ("2025-12-31",   "2100-01-01"),
}


@njit(cache=True)
def _dp_high_capture(open_, high, jlo, jhi, cost, min_mult):
    """Backward DP. Returns (f, argj). f[i]=best multiplier from bar i (in cash); argj[i]=exit bar or -1.
    min_mult = minimum per-move multiplier a move must clear to be eligible (1.0 = any net-positive move =
    scalping oracle; 1.03 = each move must net >=3% after cost = SWING oracle). A move's per-trade multiplier is
    (high[j]/open[i] - cost); only moves with that >= min_mult are candidates, so the floor shapes WHICH oracle
    (scalp vs swing) we decompose without changing the max-compound objective over the eligible set."""
    n = open_.shape[0]
    f = np.ones(n + 1, dtype=np.float64)
    argj = np.full(n, -1, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        best = f[i + 1]            # option: skip bar i
        bj = -1
        inv = 1.0 / open_[i]
        lo = jlo[i]
        hi = jhi[i]
        if lo <= hi:
            for j in range(lo, hi + 1):
                move_mult = high[j] * inv - cost      # per-trade multiplier (net of round-trip cost)
                if move_mult < min_mult:              # below the per-move floor -> not an eligible move
                    continue
                val = move_mult * f[j + 1]
                if val > best:
                    best = val
                    bj = j
        f[i] = best
        argj[i] = bj
    return f, argj


def _recover_trades(argj):
    """Forward walk to recover the chosen non-overlapping (entry_i, exit_j) moves."""
    n = argj.shape[0]
    trades = []
    i = 0
    while i < n:
        j = argj[i]
        if j < 0:
            i += 1
        else:
            trades.append((i, int(j)))
            i = j + 1
    return trades


def _windows_ms(ts_ms):
    import pandas as pd
    out = {}
    for w, (lo, hi) in WIN.items():
        lo_ms = 0 if lo == "0" else int(pd.Timestamp(lo).value // 1_000_000)
        hi_ms = int(pd.Timestamp(hi).value // 1_000_000)
        out[w] = (lo_ms, hi_ms)
    return out


def oracle_high_capture(ts_ms, open_, high, cost=COST_RT, min_hold_hours=MIN_HOLD_HOURS, min_move_net=0.0):
    """Exact perfect-foresight high-capture oracle. ts_ms int64, open_/high float64.
    min_move_net = per-move net-return floor (0.0 = scalping oracle: any net-positive wiggle; 0.03 = SWING oracle:
    each captured move must net >=3% after cost). A higher floor yields fewer/larger multi-day moves -- the unit of
    trading the project actually targets ("a SETUP across a MOVE"), and a FAIR test for trend instruments like MAs."""
    n = len(open_)
    min_hold_ms = int(min_hold_hours * MS_PER_HOUR)
    # window per entry via real timestamps (handles event bars + gaps)
    jlo = np.searchsorted(ts_ms, ts_ms + min_hold_ms, side="left").astype(np.int64)
    jhi = (np.searchsorted(ts_ms, ts_ms + MAX_HOLD_MS, side="left") - 1).astype(np.int64)
    jhi = np.minimum(jhi, n - 1)
    f, argj = _dp_high_capture(open_, high, jlo, jhi, float(cost), 1.0 + float(min_move_net))
    trades = _recover_trades(argj)
    return f, trades


def _onp_unconstrained(close, cost=COST_RT):
    """O(n) unconstrained multiplicative long-only oracle (close-to-close, unlimited round-trips).
    cash[i]=max wealth in cash after bar i; hold[i]=max wealth in units*price after bar i.
    Round-trip cost charged on the SELL leg (equivalent net to splitting). Returns (mult, n_round_trips)."""
    n = len(close)
    cash = 1.0
    hold = -1e18
    rts = 0
    # track moves: we approximate count by counting buy->sell transitions
    in_pos = False
    for i in range(n):
        p = close[i]
        new_cash = cash
        sell = hold * p * (1.0 - cost)
        if sell > new_cash:
            new_cash = sell
            if in_pos:
                pass
        # buy
        new_hold = hold
        buy = cash / p
        if buy > new_hold:
            new_hold = buy
        cash, hold = new_cash, new_hold
    # count round-trips via a second pass greedy reconstruction (close-to-close zigzag net of cost)
    # number of profitable legs: walk troughs->peaks where peak/trough-1 > cost
    rts = _count_legs(close, cost)
    return cash, rts


def _count_legs(close, cost):
    """Count net-profitable monotone up-legs (buy trough, sell peak) exceeding round-trip cost."""
    n = len(close)
    if n < 2:
        return 0
    legs = 0
    i = 0
    while i < n - 1:
        # find trough
        while i < n - 1 and close[i + 1] <= close[i]:
            i += 1
        trough = i
        # find peak
        while i < n - 1 and close[i + 1] >= close[i]:
            i += 1
        peak = i
        if peak > trough and (close[peak] / close[trough] - 1.0) > cost:
            legs += 1
        i = max(peak, trough + 1)
        if peak == trough:
            i += 1
    return legs


def summarize(ts_ms, trades, open_, high):
    """Per-move net %, window splits, compound. net = high[j]/open[i] - 1 - cost."""
    import pandas as pd
    wins = _windows_ms(ts_ms)
    nets, holds_h, ent_ts = [], [], []
    per_win = {w: [] for w in WIN}
    for (i, j) in trades:
        net = high[j] / open_[i] - 1.0 - COST_RT
        nets.append(net)
        hold_ms = ts_ms[j] - ts_ms[i]
        holds_h.append(hold_ms / MS_PER_HOUR)
        ent = ts_ms[i]
        ent_ts.append(ent)
        for w, (lo, hi) in wins.items():
            if lo <= ent < hi:
                per_win[w].append(net)
                break
    nets = np.array(nets) if nets else np.array([0.0])
    comp = float(np.prod(1.0 + nets) - 1.0) if len(trades) else 0.0
    win_out = {}
    for w in WIN:
        wn = np.array(per_win[w]) if per_win[w] else np.array([])
        win_out[w] = {
            "n_moves": int(len(wn)),
            "compound_pct": float((np.prod(1.0 + wn) - 1.0) * 100) if len(wn) else 0.0,
            "mean_net_pct": float(wn.mean() * 100) if len(wn) else 0.0,
        }
    return {
        "n_moves": len(trades),
        "mean_net_per_move_pct": float(nets.mean() * 100) if len(trades) else 0.0,
        "median_net_per_move_pct": float(np.median(nets) * 100) if len(trades) else 0.0,
        "total_capturable_compound_pct": comp * 100,
        "mean_hold_hours": float(np.mean(holds_h)) if holds_h else 0.0,
        "median_hold_hours": float(np.median(holds_h)) if holds_h else 0.0,
        "per_window": win_out,
    }


# ---------------------------------------------------------------------------
def _selftest():
    print("=" * 70)
    print("[oracle selftest] DP correctness on hand-checkable series")
    print("=" * 70)
    ok = True
    # Case 1: single clean up-move. bars hourly. open=[10,?], high captures 12 -> net=12/10-1-cost
    ts = np.array([0, 1, 2, 3], dtype=np.int64) * MS_PER_HOUR
    op = np.array([10.0, 10.0, 11.0, 11.5])
    hi = np.array([10.0, 11.0, 12.0, 11.5])
    f, tr = oracle_high_capture(ts, op, hi, min_hold_hours=1.0)
    # best: enter open[1]=10 (or open[0]=10), exit high[2]=12 -> 12/10-cost=1.1976
    exp = 12.0 / 10.0 - COST_RT
    got = f[0]
    print(f"  case1 single up-move: f[0]={got:.5f} expected~{exp:.5f} trades={tr}")
    ok &= abs(got - exp) < 1e-6
    # Case 2: flat/no edge -> no trade
    op2 = np.array([10.0, 10.0, 10.0, 10.0]); hi2 = np.array([10.0, 10.0, 10.0, 10.0])
    f2, tr2 = oracle_high_capture(ts, op2, hi2, min_hold_hours=1.0)
    print(f"  case2 flat: f[0]={f2[0]:.5f} trades={tr2} (expect 1.0, [])")
    ok &= abs(f2[0] - 1.0) < 1e-9 and len(tr2) == 0
    # Case 3: up-move then PULLBACK then up-move -> re-entering lower beats one long hold.
    # enter open[1]=10 -> exit high[2]=12 (=1.1976); re-enter open[3]=9 -> exit high[4]=14 (=1.5532)
    # two-trade product=1.860 ; single trade open10->high14=1.3976 -> DP must pick the two-trade path.
    ts3 = np.arange(6, dtype=np.int64) * MS_PER_HOUR
    op3 = np.array([10.0, 10.0, 12.0, 9.0, 9.0, 14.0])
    hi3 = np.array([10.0, 12.0, 12.0, 9.0, 14.0, 14.0])
    f3, tr3 = oracle_high_capture(ts3, op3, hi3, min_hold_hours=1.0)
    exp3 = (12.0 / 10.0 - COST_RT) * (14.0 / 9.0 - COST_RT)
    print(f"  case3 pullback two-move: f[0]={f3[0]:.5f} expected~{exp3:.5f} trades={tr3}")
    ok &= len(tr3) == 2 and abs(f3[0] - exp3) < 1e-6
    print(f"\n[oracle selftest] {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    import pandas as pd
    from pipeline.chimera_loader import ChimeraLoader

    ASSETS = ["SOL", "BTC", "ETH", "BNB", "AVAX"]
    EXACT_CADENCES = ["1d", "4h", "1h", "30m", "15m"]
    ONP_CADENCES = ["dollar"]

    L = ChimeraLoader()
    out = {"cost_rt": COST_RT, "min_hold_hours": MIN_HOLD_HOURS, "max_hold_days": 7,
           "method_exact": "perfect-foresight high-capture DP (entry=open[k], exit=high[j], non-overlap, hold<7d)",
           "method_dollar": "O(n) unconstrained close-to-close multiplicative oracle (looser upper bound; no hold cap)",
           "results": {}}

    for cad in EXACT_CADENCES + ONP_CADENCES:
        out["results"][cad] = {}
        for a in ASSETS:
            sym = a + "USDT"
            try:
                g = L.load(sym, cadence=cad)
            except Exception as e:
                out["results"][cad][a] = {"error": f"{type(e).__name__}: {str(e)[:60]}"}
                print(f"  {cad:7} {a:5} LOAD-ERR {str(e)[:50]}")
                continue
            cols = g.columns
            tcol = "timestamp" if "timestamp" in cols else ("date" if "date" in cols else None)
            ts_raw = g[tcol].to_numpy()
            if np.issubdtype(ts_raw.dtype, np.number):
                ts_ms = ts_raw.astype(np.int64)
            else:
                ts_ms = (pd.to_datetime(ts_raw).astype("int64") // 1_000_000).to_numpy().astype(np.int64)
            op = g["open"].to_numpy().astype(np.float64)
            hi = g["high"].to_numpy().astype(np.float64)
            cl = g["close"].to_numpy().astype(np.float64)
            n = len(op)
            # guard: strictly increasing ts for searchsorted; dedupe-sort if needed
            if not np.all(np.diff(ts_ms) > 0):
                order = np.argsort(ts_ms, kind="stable")
                ts_ms, op, hi, cl = ts_ms[order], op[order], hi[order], cl[order]

            if cad in ONP_CADENCES:
                mult, legs = _onp_unconstrained(cl, COST_RT)
                span_days = (ts_ms[-1] - ts_ms[0]) / (24 * 3600 * 1000)
                res = {
                    "n_bars": n,
                    "method": "onp_close_to_close_unconstrained",
                    "total_capturable_compound_pct": float((mult - 1.0) * 100),
                    "n_round_trips": int(legs),
                    "span_days": float(span_days),
                    "CAVEAT": "granularity-degenerate upper bound; not directly comparable to high-capture cadences",
                }
                out["results"][cad][a] = res
                print(f"  {cad:7} {a:5} n={n:>8}  ONP comp={(mult-1)*100:>14.1f}%  legs={legs}")
            else:
                f, trades = oracle_high_capture(ts_ms, op, hi)
                s = summarize(ts_ms, trades, op, hi)
                s["n_bars"] = n
                s["span_days"] = float((ts_ms[-1] - ts_ms[0]) / (24 * 3600 * 1000))
                out["results"][cad][a] = s
                ho = s["per_window"]["OOS"]; hu = s["per_window"]["UNSEEN"]
                print(f"  {cad:7} {a:5} n={n:>8}  moves={s['n_moves']:>6}  "
                      f"mean/med net%={s['mean_net_per_move_pct']:>6.2f}/{s['median_net_per_move_pct']:>5.2f}  "
                      f"FULLcomp={s['total_capturable_compound_pct']:>16.1f}%  "
                      f"OOS={ho['compound_pct']:>10.1f}%(n{ho['n_moves']}) UNSEEN={hu['compound_pct']:>9.1f}%(n{hu['n_moves']})")

    outp = ROOT / "runs" / "research" / "oracle_ceiling_map.json"
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote {outp}")
    return out


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(0 if _selftest() else 1)
    _selftest()
    print()
    main()
