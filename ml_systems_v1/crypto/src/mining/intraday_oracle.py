"""INTRADAY ORACLE-DECOMPOSE -- the honest test of the user's intraday-speculation frame
(2026-06-11): entry on {15m,30m,1h,2h,4h} bars, MAX 3-day hold, EVENT-CLOCK (low-turnover:
one breakout entry, ride, exit -- NOT re-evaluated every bar, which is what whipsaws
bar-clock trend books into the cost wall, finecadence F1 -26 to -54%).

Top-down (oracle-first), NOT bottom-up TI-tuning (which is the dead per-asset config DNA).
Per cadence + per asset (u50):
  CAUSAL trigger : close breaks the prior ~1-day high (Donchian breakout = trend confirm).
  EVENT-CLOCK    : enter next-bar open; ONE position per asset at a time (no re-entry while
                   held); exit at min(3d-in-bars time-stop, chandelier 3xATR trailing).
  ORACLE ceiling : per fired event, the hindsight-best long exit (max high in the 3d window)
                   net of cost -- the realizable ceiling for THIS entry.
  CAPTURE        : causal_net / oracle_ceiling.
  NULL           : random-entry in the same split, same hold-length dist, same cost.
Splits SEL(<2025-03-15)/OOS(<2025-12-31)/UNSEEN(<2026-06-01); UNSEEN is the final litmus.
Cost: taker 0.24% RT headline; maker 0.06% RT sensitivity. Graded via the canonical scorecard.

The question it answers: net of cost, at each intraday cadence, with a 3d-max event-clock hold,
is there a causal trend-entry that (a) realizes a positive fraction of the oracle and (b) beats
random entries on UNSEEN? If no -> the intraday wall is confirmed with the user's parameters; if
a cadence shows a sliver -> exploit precisely that. No emoji (cp1252).

Run: python -m mining.intraday_oracle --universe u50 --cadences 15m,30m,1h,2h,4h
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pipeline.chimera_loader import ChimeraLoader          # noqa: E402
from mining.family_regime_map import atr14, _norm_sym       # noqa: E402
from strat.scorecard import score_trades                    # noqa: E402

OUT = ROOT / "runs" / "mining"
OUT.mkdir(parents=True, exist_ok=True)

TRAIN_END_MS = int(dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
SEL_END_MS = int(dt.datetime(2025, 3, 15, tzinfo=dt.timezone.utc).timestamp() * 1000)
OOS_END_MS = int(dt.datetime(2025, 12, 31, tzinfo=dt.timezone.utc).timestamp() * 1000)
UNS_END_MS = int(dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
TAKER_RT, MAKER_RT = 0.0024, 0.0006

BARS_PER_DAY = {"15m": 96, "30m": 48, "1h": 24, "2h": 12, "4h": 6}


def split_of(ms: int) -> str:
    return "SEL" if ms < SEL_END_MS else ("OOS" if ms < OOS_END_MS else ("UNSEEN" if ms < UNS_END_MS else "POST"))


def _load(sym: str, cad: str) -> pl.DataFrame | None:
    """Load OHLC at cadence; 2h is resampled from 1h."""
    try:
        if cad == "2h":
            df = ChimeraLoader().load(_norm_sym(sym), cadence="1h",
                                      features=["open", "high", "low", "close"])
            df = df.with_columns((pl.col("timestamp") // (2 * 3600_000) * (2 * 3600_000)).alias("g"))
            df = df.group_by("g", maintain_order=True).agg([
                pl.col("open").first(), pl.col("high").max(), pl.col("low").min(),
                pl.col("close").last()]).rename({"g": "timestamp"})
            return df
        return ChimeraLoader().load(_norm_sym(sym), cadence=cad,
                                    features=["open", "high", "low", "close"])
    except Exception:
        return None


def simulate(sym: str, cad: str, rng: np.random.Generator) -> tuple[list, list]:
    """Event-clock breakout trend-ride. Returns (events, null_trades)."""
    df = _load(sym, cad)
    if df is None or len(df) < 600:
        return [], []
    ts = df["timestamp"].to_numpy()
    o = df["open"].to_numpy().astype(float)
    h = df["high"].to_numpy().astype(float)
    l = df["low"].to_numpy().astype(float)
    c = df["close"].to_numpy().astype(float)
    n = len(c)
    bpd = BARS_PER_DAY[cad]
    look = bpd               # ~1-day breakout lookback (trend confirmation)
    maxhold = 3 * bpd        # 3-day max hold
    atr = atr14(h, l, c)

    # prior-`look`-bar high (causal: bars t-look..t-1)
    prior_high = np.full(n, np.nan)
    for t in range(look, n):
        prior_high[t] = np.max(h[t - look:t])

    events, holds = [], []
    t = look + 1
    while t < n - 2:
        if np.isfinite(prior_high[t]) and c[t] > prior_high[t] and np.isfinite(atr[t]) and o[t + 1] > 0:
            entry = o[t + 1]
            f = t + 1
            end = min(f + maxhold, n - 1)
            # causal exit: chandelier 3xATR trail OR time-stop at end
            hw = c[f]
            exit_idx = end
            for x in range(f, end):
                hw = max(hw, c[x])
                if np.isfinite(atr[x]) and c[x] < hw - 3.0 * atr[x]:
                    exit_idx = min(x + 1, end)
                    break
            causal_net = o[exit_idx] / entry - 1.0
            # oracle ceiling: best exit (max high) within [f, end]
            ceil_px = float(np.max(h[f:end + 1]))
            oracle_net = ceil_px / entry - 1.0
            hold = exit_idx - f
            events.append({"sym": sym, "ts": int(ts[f]), "split": split_of(int(ts[f])),
                           "causal_gross": causal_net, "oracle_gross": oracle_net, "hold": hold})
            holds.append(hold)
            t = exit_idx + 1     # event-clock: no overlapping positions
        else:
            t += 1

    # random-entry null: same count, random valid bars, hold sampled from the real dist
    null = []
    if events and holds:
        valid = np.arange(look + 1, n - max(holds) - 2)
        if len(valid) > len(events):
            picks = rng.choice(valid, size=len(events), replace=False)
            for i, p in enumerate(picks):
                hold = holds[i % len(holds)]
                if o[p] > 0 and p + hold < n:
                    null.append({"sym": sym, "ts": int(ts[p]), "split": split_of(int(ts[p])),
                                 "net": o[p + hold] / o[p] - 1.0 - TAKER_RT})
    return events, null


def grade_cadence(sym_events: list, sym_nulls: list, cost_rt: float) -> dict:
    """Per-cadence honest grade via the canonical scorecard."""
    trades = [{"sym": e["sym"], "ts": e["ts"], "split": e["split"],
               "net": e["causal_gross"] - cost_rt} for e in sym_events]
    card = score_trades("intraday_breakout", trades)
    # oracle + capture per split
    out = {"n_events": len(sym_events), "scorecard": {sp: card["per_split"].get(sp, {})
                                                      for sp in ["SEL", "OOS", "UNSEEN"]}}
    for sp in ["SEL", "OOS", "UNSEEN"]:
        ev = [e for e in sym_events if e["split"] == sp]
        if len(ev) >= 5:
            orc = np.array([e["oracle_gross"] - cost_rt for e in ev])
            cau = np.array([e["causal_gross"] - cost_rt for e in ev])
            cap = [c / o for c, o in zip(cau, orc) if o > 0.005]
            nl = np.array([t["net"] for t in sym_nulls if t["split"] == sp])
            out[f"{sp}_oracle_mean_pct"] = round(float(orc.mean() * 100), 2)
            out[f"{sp}_capture_median"] = round(float(np.median(cap)), 3) if cap else None
            out[f"{sp}_causal_vs_null_pp"] = (round(float((cau.mean() - nl.mean()) * 100), 3)
                                              if len(nl) >= 5 else None)
            out[f"{sp}_median_hold_bars"] = int(np.median([e["hold"] for e in ev]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m mining.intraday_oracle")
    ap.add_argument("--universe", default="u50")
    ap.add_argument("--cadences", default="15m,30m,1h,2h,4h")
    ap.add_argument("--maker", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    cost = MAKER_RT if a.maker else TAKER_RT
    spec = yaml.safe_load(open(ROOT / "config" / "universes" / f"{a.universe}.yaml"))
    syms = ([x["symbol"] for x in spec["assets"]] if "assets" in spec
            else [x["symbol"] for x in yaml.safe_load(open(ROOT / "config/universes/u50.yaml"))["assets"]])
    rng = np.random.default_rng(a.seed)

    report = {}
    for cad in a.cadences.split(","):
        if cad not in BARS_PER_DAY:
            continue
        ev_all, null_all = [], []
        for s in syms:
            e, nl = simulate(s, cad, rng)
            ev_all += e
            null_all += nl
        report[cad] = grade_cadence(ev_all, null_all, cost)
        print(f"[{cad}] events={report[cad]['n_events']} graded")

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT / f"intraday_oracle_{a.universe}_{stamp}.json"
    json.dump({"repro": {"command": "python " + " ".join(sys.argv), "git_sha": sha},
               "cost_rt": cost, "report": report}, open(p, "w", encoding="utf-8"), indent=1, default=str)

    print(f"\n## INTRADAY ORACLE-DECOMPOSE -- {a.universe} -- {'maker' if a.maker else 'taker'} -- "
          f"event-clock breakout, 3d max hold")
    print(f"   {'cad':>4} {'events':>7} | {'OOS oracle/capture/causal-vs-null':>34} | "
          f"{'UNSEEN causal mean+-se / vs-null':>34} | hold")
    for cad, r in report.items():
        oc = (f"{r.get('OOS_oracle_mean_pct')}% / {r.get('OOS_capture_median')} / "
              f"{r.get('OOS_causal_vs_null_pp')}pp")
        us = r["scorecard"].get("UNSEEN", {})
        uvn = r.get("UNSEEN_causal_vs_null_pp")
        usc = (f"{us.get('mean_pct')}+-{us.get('se_pct')}% / {uvn}pp jk{us.get('jk_drop_top5pct_mean_pct')}"
               if us.get("n") else "-")
        print(f"   {cad:>4} {r['n_events']:>7} | {oc:>34} | {usc:>34} | {r.get('UNSEEN_median_hold_bars','-')}")
    print(f"\n   READ: a cadence is a SLIVER only if UNSEEN causal mean>0 AND beats null (vs-null>0) AND "
          f"jk-positive. Else the intraday wall holds at the user's frame.\n   JSON -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
