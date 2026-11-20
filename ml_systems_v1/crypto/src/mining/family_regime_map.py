"""INDICATOR-FAMILY x REGIME MAP -- "find the winning family per asset, per regime;
ROI and winners>losers" (user mandate 2026-06-10 evening).

WHAT THIS IS: six CANONICAL indicator families with PREDETERMINED, textbook params
(zero optimization -- the configs are written below before any data is read), each
traded LONG-ONLY at bar clock with next-bar-open fills and taker 24bps RT, each in
two exit modes (signal-exit; signal-exit + 2xATR(14) trailing stop = the user's
"ride the trend, stop out" frame). Every TRADE is tagged with the CAUSAL regime at
entry (daily close vs SMA200, known at the prior day's close): BULL or BEAR.

  TREND families : EMA_20_100 cross, EMA_50_200 cross, DONCHIAN_20_10 breakout,
                   ROC_20 zero-cross
  MR families    : RSI14 (<30 in, >50 out), BOLL_20_2 (close < lower in, mid out)

THE MAP: per (asset, regime) pick the TRAIN-best family (by summed net, n>=10
trades) -> report that family's OOS stats. The full grid is also reported (no
cherry-picking; the map is the deliverable, the grid is the evidence).

METRICS per cell: n trades, WIN RATE, mean/median net per trade, sum net, profit
factor, avg win, avg loss -- so the "winners > losers" question is answered in
COUNT and in MAGNITUDE separately (trend systems classically have win rate < 50%
with positive skew; MR the reverse; the data decides).

DISCIPLINE: splits TRAIN <2024-01-01 / OOS <2025-07-01 / UNSEEN untouched (series
truncated at the OOS end -- UNSEEN bars are never simulated). Selection on TRAIN
only. Cost 24bps RT. No emoji (cp1252).

Run:
  python -m mining.family_regime_map --universe u10 --cadences 1d,4h
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pipeline.chimera_loader import ChimeraLoader  # noqa: E402

OUT = ROOT / "runs" / "mining"
OUT.mkdir(parents=True, exist_ok=True)

__contract__ = {
    "kind": "research",
    "inputs": {"chimera_ohlc": "ChimeraLoader per cadence + 1d regime"},
    "outputs": {"study_json": "runs/mining/family_regime_map_<tag>_<stamp>.json"},
    "invariants": {
        "predetermined_params": "all family params fixed in-source before data; zero sweeps",
        "causal": "signals at bar close; fills next-bar open; regime = prior-day SMA200 state",
        "unseen_untouched": "series truncated at OOS end; UNSEEN bars never simulated",
        "train_only_selection": "the (asset,regime)->family map is chosen on TRAIN only",
    },
}

DAY_MS = 86_400_000
TRAIN_END_MS = int(dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
OOS_END_MS = int(dt.datetime(2025, 7, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
COST_RT = 0.0024
MIN_TRADES_SELECT = 10


def _norm_sym(s: str) -> str:
    s = s.upper()
    return s if s.endswith("USDT") else s + "USDT"


# ----------------------------------------------------------- indicator primitives

def ema(x, n):
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    a = 2.0 / (n + 1)
    out[n - 1] = np.mean(x[:n])
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def sma(x, n):
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        c = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def wilder(x, n):
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    out[n - 1] = np.mean(x[:n])
    for i in range(n, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def rsi14(close):
    d = np.diff(close, prepend=close[0])
    up = wilder(np.maximum(d, 0), 14)
    dn = wilder(np.maximum(-d, 0), 14)
    rs = up / (dn + 1e-12)
    return 100 - 100 / (1 + rs)


def atr14(high, low, close):
    pc = np.roll(close, 1)
    pc[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))
    return wilder(tr, 14)


# ----------------------------------------------------------- family signal defs
# Each returns (enter[t], exit[t]) boolean arrays: signal at bar t close.

def fam_signals(name, o, h, l, c):
    if name == "EMA_20_100":
        f, s = ema(c, 20), ema(c, 100)
        above = f > s
        prev = np.roll(above, 1)
        prev[0] = above[0]
        return (~prev & above), (prev & ~above)
    if name == "EMA_50_200":
        f, s = ema(c, 50), ema(c, 200)
        above = f > s
        prev = np.roll(above, 1)
        prev[0] = above[0]
        return (~prev & above), (prev & ~above)
    if name == "DONCH_20_10":
        hh = np.full(len(c), np.nan)
        ll = np.full(len(c), np.nan)
        for i in range(20, len(c)):
            hh[i] = np.max(h[i - 20:i])
        for i in range(10, len(c)):
            ll[i] = np.min(l[i - 10:i])
        return (c > hh), (c < ll)
    if name == "ROC_20":
        roc = np.full(len(c), np.nan)
        roc[20:] = c[20:] / c[:-20] - 1.0
        pos = roc > 0
        prev = np.roll(pos, 1)
        prev[0] = pos[0]
        return (~prev & pos), (prev & ~pos)
    if name == "RSI14_30_50":
        r = rsi14(c)
        return (r < 30), (r > 50)
    if name == "BOLL_20_2":
        m = sma(c, 20)
        sd = np.full(len(c), np.nan)
        for i in range(19, len(c)):
            sd[i] = np.std(c[i - 19:i + 1])
        return (c < m - 2 * sd), (c >= m)
    raise ValueError(name)


FAMILIES = ["EMA_20_100", "EMA_50_200", "DONCH_20_10", "ROC_20", "RSI14_30_50", "BOLL_20_2"]
FAMILY_CLASS = {"EMA_20_100": "TREND", "EMA_50_200": "TREND", "DONCH_20_10": "TREND",
                "ROC_20": "TREND", "RSI14_30_50": "MR", "BOLL_20_2": "MR"}


def simulate(name, stop: bool, ts, o, h, l, c, regime_bull, split_arr) -> list[dict]:
    """Long-only state machine. Signal at close of t -> fill open of t+1.
    With stop=True: also exit when close < highwater - 2*ATR14 (trailing)."""
    ent, exi = fam_signals(name, o, h, l, c)
    atr = atr14(h, l, c)
    n = len(c)
    trades = []
    in_pos = False
    entry_px = hw = 0.0
    e_regime = e_split = None
    for t in range(1, n - 1):
        if not in_pos:
            if ent[t] and np.isfinite(o[t + 1]) and o[t + 1] > 0 and not np.isnan(atr[t]):
                in_pos = True
                entry_px = o[t + 1]
                hw = c[t]
                e_regime = "BULL" if regime_bull[t] else "BEAR"
                e_split = split_arr[t]
        else:
            hw = max(hw, c[t])
            stop_hit = stop and np.isfinite(atr[t]) and (c[t] < hw - 2.0 * atr[t])
            if exi[t] or stop_hit:
                exit_px = o[t + 1]
                trades.append({"net": exit_px / entry_px - 1.0 - COST_RT,
                               "regime": e_regime, "split": e_split,
                               "stopped": bool(stop_hit and not exi[t])})
                in_pos = False
    if in_pos:  # force-close at the last bar's open (end = OOS boundary)
        trades.append({"net": o[n - 1] / entry_px - 1.0 - COST_RT,
                       "regime": e_regime, "split": e_split, "stopped": False})
    return trades


def cell_stats(trades: list[dict]) -> dict:
    if not trades:
        return {"n": 0}
    nets = np.array([t["net"] for t in trades])
    wins = nets[nets > 0]
    losses = nets[nets <= 0]
    return {
        "n": len(nets), "win_rate": float((nets > 0).mean()),
        "mean": float(nets.mean()), "median": float(np.median(nets)),
        "sum": float(nets.sum()),
        "avg_win": float(wins.mean()) if len(wins) else None,
        "avg_loss": float(losses.mean()) if len(losses) else None,
        "profit_factor": (float(wins.sum() / -losses.sum())
                          if len(losses) and losses.sum() < 0 else None),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="indicator family x regime map")
    ap.add_argument("--universe", default="u10")
    ap.add_argument("--cadences", default="1d,4h")
    args = ap.parse_args()
    spec = yaml.safe_load(open(ROOT / "config" / "universes" / f"{args.universe}.yaml"))
    syms = [a["symbol"] for a in spec["assets"]]
    cadences = args.cadences.split(",")

    report = {}
    for cad in cadences:
        t0 = time.time()
        cells = {}      # (sym, family, stop, regime, split) -> trades
        for sym in syms:
            try:
                df = ChimeraLoader().load(sym, cadence=cad,
                                          features=["open", "high", "low", "close"])
            except Exception as ex:
                print(f"[{cad}][{sym}] SKIP {type(ex).__name__}")
                continue
            ts = df["timestamp"].to_numpy()
            keep = ts < OOS_END_MS          # UNSEEN never simulated
            if keep.sum() < 250:            # late listings: too little pre-UNSEEN history
                print(f"[{cad}][{sym}] SKIP (only {int(keep.sum())} bars before OOS end)")
                continue
            ts = ts[keep]
            o = df["open"].to_numpy().astype(float)[keep]
            h = df["high"].to_numpy().astype(float)[keep]
            l = df["low"].to_numpy().astype(float)[keep]
            c = df["close"].to_numpy().astype(float)[keep]
            # causal daily regime: prior-day close vs prior-day SMA200
            dd = ChimeraLoader().load(sym, cadence="1d", features=["close"])
            dts = dd["timestamp"].to_numpy()
            dc = dd["close"].to_numpy().astype(float)
            dsma = sma(dc, 200)
            bull_by_day = {int(dts[i] // DAY_MS): bool(dc[i] > dsma[i])
                           for i in range(len(dts)) if np.isfinite(dsma[i])}
            day_of_bar = (ts // DAY_MS).astype(int)
            regime_bull = np.array([bull_by_day.get(d - 1, False) for d in day_of_bar])
            split_arr = np.where(ts < TRAIN_END_MS, "TRAIN", "OOS")
            for fam in FAMILIES:
                for stop in (False, True):
                    trades = simulate(fam, stop, ts, o, h, l, c, regime_bull, split_arr)
                    for tr in trades:
                        key = (sym, fam, stop, tr["regime"], tr["split"])
                        cells.setdefault(key, []).append(tr)

        # full grid stats
        grid = {f"{k[0]}|{k[1]}|{'stop' if k[2] else 'sig'}|{k[3]}|{k[4]}": cell_stats(v)
                for k, v in cells.items()}

        # THE MAP: per (asset, regime) TRAIN-best (family, stop-mode) -> OOS validation
        the_map = {}
        for sym in syms:
            for reg in ["BULL", "BEAR"]:
                best, best_sum = None, -1e18
                for fam in FAMILIES:
                    for stop in (False, True):
                        s = cell_stats(cells.get((sym, fam, stop, reg, "TRAIN"), []))
                        if s.get("n", 0) >= MIN_TRADES_SELECT and s["sum"] > best_sum:
                            best, best_sum = (fam, stop), s["sum"]
                if best is None:
                    continue
                oos = cell_stats(cells.get((sym, best[0], best[1], reg, "OOS"), []))
                the_map[f"{sym}|{reg}"] = {
                    "family": best[0], "stop": best[1],
                    "train_sum": best_sum,
                    "train": cell_stats(cells.get((sym, best[0], best[1], reg, "TRAIN"), [])),
                    "oos": oos,
                }
        # map-level OOS aggregation
        oos_nets = []
        for v in the_map.values():
            key = (v["family"], v["stop"])
        for sym in syms:
            for reg in ["BULL", "BEAR"]:
                m = the_map.get(f"{sym}|{reg}")
                if m:
                    for tr in cells.get((sym, m["family"], m["stop"], reg, "OOS"), []):
                        oos_nets.append(tr["net"])
        map_oos = cell_stats([{"net": x} for x in oos_nets])
        pos_assets = len({k.split("|")[0] for k, v in the_map.items()
                          if v["oos"].get("n", 0) > 0 and v["oos"]["sum"] > 0})

        # family-class truth table (pooled, predetermined configs, no selection)
        class_table = {}
        for cls in ["TREND", "MR"]:
            for sp in ["TRAIN", "OOS"]:
                pool = [tr for k, v in cells.items() for tr in v
                        if FAMILY_CLASS[k[1]] == cls and k[4] == sp]
                class_table[f"{cls}|{sp}"] = cell_stats(pool)

        report[cad] = {"map": the_map, "map_oos_pooled": map_oos,
                       "map_oos_assets_positive": pos_assets,
                       "class_table": class_table, "grid": grid}
        print(f"[{cad}] done ({time.time()-t0:.0f}s)")

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=ROOT).stdout.strip()
    except Exception:
        sha = "unknown"
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT / f"family_regime_map_{args.universe}_{stamp}.json"
    out_path.write_text(json.dumps(
        {"tool": "family_regime_map", "git_sha": sha,
         "params": {"families": FAMILIES, "cost_rt": COST_RT,
                    "stop": "2xATR14 trailing variant", "regimes": "daily SMA200 BULL/BEAR (causal)"},
         "report": report}, indent=1), encoding="utf-8")

    # ---------------- console story
    for cad, rep in report.items():
        print("\n" + "=" * 100)
        print(f"[{cad}] THE MAP -- TRAIN-selected (family, exit) per (asset, regime) -> OOS truth")
        print("=" * 100)
        print(f"{'asset|regime':<22} {'family':<13} {'exit':<5} | {'TRAIN n/win%/sum':>20} | "
              f"{'OOS n':>5} {'win%':>5} {'med':>7} {'sum':>8} {'PF':>5}")
        for key in sorted(rep["map"]):
            v = rep["map"][key]
            tr, oo = v["train"], v["oos"]
            print(f"{key:<22} {v['family']:<13} {'stop' if v['stop'] else 'sig':<5} | "
                  f"{tr['n']:>4}/{tr['win_rate']*100:3.0f}%/{tr['sum']*100:+7.1f}% | "
                  f"{oo.get('n',0):>5} {oo.get('win_rate',0)*100 if oo.get('n') else 0:4.0f}% "
                  f"{(oo.get('median') or 0)*100:+6.2f}% {(oo.get('sum') or 0)*100:+7.1f}% "
                  f"{oo.get('profit_factor') if oo.get('profit_factor') is not None else '--'}")
        mo = rep["map_oos_pooled"]
        if mo.get("n"):
            print(f"\nMAP pooled OOS: n={mo['n']} trades, win {mo['win_rate']*100:.0f}%, "
                  f"median {mo['median']*100:+.2f}%, sum {mo['sum']*100:+.1f}%, "
                  f"PF {mo['profit_factor']}; assets with positive OOS sum: "
                  f"{rep['map_oos_assets_positive']}/10")
        print("\nFAMILY-CLASS truth (pooled, no selection) -- the winners-vs-losers anatomy:")
        for k, s in rep["class_table"].items():
            if s.get("n"):
                print(f"  {k:<10} n={s['n']:>5} win {s['win_rate']*100:3.0f}% "
                      f"avg_win {(s['avg_win'] or 0)*100:+.2f}% avg_loss {(s['avg_loss'] or 0)*100:+.2f}% "
                      f"PF {s['profit_factor'] and round(s['profit_factor'],2)} sum {s['sum']*100:+.1f}%")
    print(f"\nJSON -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
