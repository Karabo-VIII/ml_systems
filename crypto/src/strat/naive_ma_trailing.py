"""src/strat/naive_ma_trailing.py -- a NAIVE exercise (user 2026-06-10): long-term MA (81/100/200), buy when price
CLOSES ABOVE the MA (cross-up), exit on a TRAILING STOP. Portfolio sim with explicit cash.

SPEC (verbatim): start capital $10,000; $500 per bet (fixed notional); risk 10% per trade (interpreted as a 10%
TRAILING STOP -> ~$50 max risk per $500 bet); timeframe < 1d; a random 1-month slice from each year.

Notes: event-driven portfolio over the aligned bar grid; MAs computed on the FULL series (warmup), trades fire ONLY
inside the slice; open positions force-closed (MtM) at slice end. Trailing stop on the running HIGH; stop-touch when
the bar LOW <= trail (fills at the trail level, or at the open on a gap-down). Taker cost 0.24% round-trip charged.
Universe = u10 (a small liquid portfolio so the $500 bets can deploy). No emoji (cp1252).
Run: python src/strat/naive_ma_trailing.py --cadence 1h --mas 81,100,200
"""
from __future__ import annotations
import argparse, sys, warnings, json
from pathlib import Path
import numpy as np
if not hasattr(np, "NaN"): np.NaN = np.nan
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from strat.entry_signal_lab import load_ohlc, U10

CAPITAL = 10_000.0
BET = 500.0
TRAIL = 0.10
COST_RT = 0.0024


def random_month_slices(years, seed=42):
    """One random calendar month per year (reproducible). Returns list of (year, month, start, end)."""
    rng = np.random.default_rng(seed)
    slices = []
    for y in years:
        m = int(rng.integers(1, 13))
        start = pd.Timestamp(year=y, month=m, day=1)
        end = (start + pd.offsets.MonthEnd(1)) + pd.Timedelta(days=1)   # inclusive of month-end
        slices.append((y, m, start, end))
    return slices


def load_panel(universe, cadence, mas):
    syms = universe
    O, H, L, C, MA = {}, {}, {}, {}, {n: {} for n in mas}
    for s in syms:
        df = load_ohlc(s, cadence)
        if df is None or len(df) < max(mas) + 50: continue
        df = df.set_index("date")
        O[s], H[s], L[s], C[s] = df["open"], df["high"], df["low"], df["close"]
        for n in mas:
            MA[n][s] = df["close"].rolling(n, min_periods=n // 2).mean()
    return O, H, L, C, MA, list(C.keys())


def run_slice(O, H, L, C, MA, syms, ma_period, start, end):
    """Event-driven portfolio backtest for ONE MA period over ONE slice. Returns stats dict."""
    grid = sorted(set().union(*[set(C[s].loc[(C[s].index >= start) & (C[s].index < end)].index) for s in syms]))
    if len(grid) < 5: return None
    ma = MA[ma_period]
    cash = CAPITAL
    pos = {}            # sym -> {entry, hwm, peak_locked}
    trades = []
    eq_curve = []
    prev = {s: None for s in syms}
    for t in grid:
        # 1) EXITS first (trailing stop)
        for s in list(pos.keys()):
            if t not in C[s].index: continue
            hi, lo, op, cl = H[s].get(t, np.nan), L[s].get(t, np.nan), O[s].get(t, np.nan), C[s].get(t, np.nan)
            if not np.isfinite(cl): continue
            p = pos[s]
            p["hwm"] = max(p["hwm"], hi if np.isfinite(hi) else cl)
            trail_px = p["hwm"] * (1 - TRAIL)
            if np.isfinite(lo) and lo <= trail_px:
                fill = min(trail_px, op) if (np.isfinite(op) and op < trail_px) else trail_px   # gap-down fills worse
                ret = fill / p["entry"] - 1.0
                pnl = BET * ret - BET * COST_RT / 2                        # exit-side cost (entry charged at entry)
                cash += BET + pnl
                trades.append({"sym": s, "ret": ret, "pnl": pnl, "bars": p["bars"]})
                del pos[s]
        # 2) ENTRIES (close crosses above MA), if free cash
        for s in syms:
            if t not in C[s].index: continue
            cl = C[s].get(t, np.nan); m = ma[s].get(t, np.nan)
            pcl, pm = prev[s], None
            if np.isfinite(cl) and np.isfinite(m):
                if s not in pos and cash >= BET and pcl is not None and pcl[0] <= pcl[1] and cl > m:
                    cash -= BET + BET * COST_RT / 2                       # entry-side cost
                    pos[s] = {"entry": cl, "hwm": H[s].get(t, cl), "bars": 0}
                prev[s] = (cl, m)
            for ss in pos: pos[ss]["bars"] += 1 if ss == s else 0
        # 3) mark-to-market equity
        mtm = cash + sum(BET * (C[s].get(t, np.nan) / pos[s]["entry"]) for s in pos if np.isfinite(C[s].get(t, np.nan)))
        eq_curve.append(mtm)
    # force-close at slice end
    last = grid[-1]
    for s in list(pos.keys()):
        cl = C[s].get(last, np.nan)
        if np.isfinite(cl):
            ret = cl / pos[s]["entry"] - 1.0; pnl = BET * ret - BET * COST_RT / 2
            cash += BET + pnl; trades.append({"sym": s, "ret": ret, "pnl": pnl, "bars": pos[s]["bars"], "forced": True})
    eq = np.array(eq_curve) if eq_curve else np.array([CAPITAL])
    final = cash
    wins = [t for t in trades if t["pnl"] > 0]
    dd = float(((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min() * 100) if len(eq) else 0.0
    return {"final": round(final, 0), "ret_pct": round((final / CAPITAL - 1) * 100, 2), "n_trades": len(trades),
            "win_rate": round(len(wins) / max(len(trades), 1), 2),
            "avg_win_pct": round(float(np.mean([t["ret"] for t in wins])) * 100, 2) if wins else 0.0,
            "avg_loss_pct": round(float(np.mean([t["ret"] for t in trades if t["pnl"] <= 0])) * 100, 2) if (len(trades) - len(wins)) else 0.0,
            "maxdd_pct": round(dd, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m strat.naive_ma_trailing")
    ap.add_argument("--cadence", default="1h"); ap.add_argument("--mas", default="81,100,200")
    ap.add_argument("--universe", default="u10"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    mas = [int(x) for x in a.mas.split(",")]
    from strat.tsmom_ensemble import _universe
    syms_u = _universe(a.universe)
    O, H, L, C, MA, syms = load_panel(syms_u, a.cadence, mas)
    years = list(range(2020, 2027))
    slices = random_month_slices(years, a.seed)
    print(f"## NAIVE MA + 10% TRAILING STOP -- portfolio ${CAPITAL:.0f}, ${BET:.0f}/bet, {a.cadence}, {a.universe} ({len(syms)} assets), taker cost")
    print(f"   random month per year (seed {a.seed}): " + ", ".join(f"{y}-{m:02d}" for y, m, _, _ in slices))
    allout = {}
    for ma_period in mas:
        print(f"\n   --- MA({ma_period}) ---")
        print(f"   {'slice':10} {'ret%':>7} {'final$':>9} {'trades':>7} {'win%':>5} {'avgW%':>6} {'avgL%':>6} {'maxDD%':>7}")
        rows = []
        for y, m, start, end in slices:
            r = run_slice(O, H, L, C, MA, syms, ma_period, start, end)
            if not r: print(f"   {y}-{m:02d}     (no data)"); continue
            rows.append(r); allout[f"MA{ma_period}_{y}-{m:02d}"] = r
            print(f"   {f'{y}-{m:02d}':10} {r['ret_pct']:>7} {r['final']:>9.0f} {r['n_trades']:>7} {r['win_rate']*100:>5.0f} {r['avg_win_pct']:>6} {r['avg_loss_pct']:>6} {r['maxdd_pct']:>7}")
        if rows:
            rets = [r["ret_pct"] for r in rows]
            print(f"   {'AVG/slice':10} {np.mean(rets):>7.2f} {'':>9} {np.mean([r['n_trades'] for r in rows]):>7.1f} "
                  f"{np.mean([r['win_rate'] for r in rows])*100:>5.0f} | sum-of-slices ret%={sum(rets):.1f}  +slices={sum(1 for x in rets if x>0)}/{len(rets)}")
    if a.json:
        import subprocess
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        outdir = ROOT.parent / "runs" / "mining"; outdir.mkdir(parents=True, exist_ok=True)
        p = outdir / f"naive_ma_trailing_{a.universe}_{a.cadence}_seed{a.seed}.json"
        json.dump({"repro": {"command": "python " + " ".join(sys.argv), "git_sha": sha,
                             "spec": {"capital": CAPITAL, "bet": BET, "trail": TRAIL, "cost_rt": COST_RT}},
                   "slices": [f"{y}-{m:02d}" for y, m, _, _ in slices], "results": allout},
                  open(p, "w", encoding="utf-8"), indent=2, default=str)
        print(f"\n[persisted] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
