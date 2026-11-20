"""CROSS-INSTRUMENT GENERALIZATION -- which signal generalizes MOST across instruments?

User /orc task (2026-06-11): "find strategies that generalize most across instruments and trade
those" -- the logical payoff of the granularity ladder (POOLED won: one config across all assets is
the robust answer; so the question becomes WHICH signal pooled across the universe generalizes best).
Frame: the other instance's sub-day / <=3-day-hold event-clock (intraday_oracle confirmed the
DIRECTIONAL breakout is walled at every cadence -- oracle +6%/event, causal capture NEGATIVE). This
lab widens that to a LIBRARY of signal families and RANKS them by generalization, INCLUDING the one
channel the fork flagged un-refuted: the MAGNITUDE / VOLATILITY channel (bet move SIZE, not direction).

GENERALIZATION SCORE (per family, pooled across u50, event-clock, <=3d hold, predetermined params so
NO selection leak): on the held-out OOS split, per instrument compute mean per-event net; then
  breadth      = fraction of instruments with OOS mean net > 0  (the core "generalizes" metric)
  breadth_cost = fraction with OOS mean net > a round-trip cost (tradeable breadth)
  magnitude    = pooled OOS mean per-event net (+- se)
  concentration= fair drop-top-5pct of the pooled OOS nets (survives a few big trades?)
  persistence  = does the per-instrument sign agree TRAIN vs OOS (stable, not regime-luck)
A family "generalizes + is tradeable" iff breadth_cost robustly > 0.5 AND concentration stays > 0.
Rank families; "trade these" = the survivors (if any). UNSEEN sealed (not loaded).

FAMILIES (predetermined; directional + the magnitude channel + a regime-gate reference):
  MA20_100  (trend cross) | DONCH20 (breakout) | ROC20 (momentum) | RSI_30_50 (bounce) |
  BOLL_LO (mean-reversion) | VOLEXP (magnitude: vol-expansion + up-filter long) |
  NRBREAK (range-compression breakout) | TRENDGATE (long while close>SMA100 = regime reference)

MECHANIC (reuses the fork's event-clock, <=3d hold, one-position-per-asset, ATR chandelier+time-stop;
next-bar-open fills, taker RT). Cadences: 4h, 1h (sub-day, the user's frame) + 1d reference.
No emoji (cp1252).

Run: python -m strat.cross_instrument_generalize --universe u50 --cadences 4h,1h,1d
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT.parent / "src") not in sys.path:
    sys.path.insert(0, str(ROOT.parent / "src"))

from mining.family_regime_map import sma, ema, atr14, rsi14, _norm_sym  # noqa: E402
from pipeline.chimera_loader import ChimeraLoader  # noqa: E402

OUT = ROOT.parent / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)

TAKER_RT = 0.0024
TRAIN_END_MS = int(dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
OOS_END_MS = int(dt.datetime(2025, 7, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
BARS_PER_DAY = {"1d": 1, "4h": 6, "1h": 24, "30m": 48, "15m": 96}
FAMILIES = ["MA20_100", "DONCH20", "ROC20", "RSI_30_50", "BOLL_LO", "VOLEXP", "NRBREAK", "TRENDGATE"]


def split_of(ms):
    return "TRAIN" if ms < TRAIN_END_MS else ("OOS" if ms < OOS_END_MS else "UNSEEN")


def entry_signal(fam, o, h, l, c):
    """Boolean ENTRY array (causal at bar t). Exit is handled by the event-clock ride."""
    n = len(c)
    if fam == "MA20_100":
        f, s = ema(c, 20), ema(c, 100)
        a = f > s; p = np.roll(a, 1); p[0] = a[0]; return (~p) & a
    if fam == "DONCH20":
        hh = np.full(n, np.nan)
        for i in range(20, n): hh[i] = np.max(h[i - 20:i])
        return c > hh
    if fam == "ROC20":
        roc = np.full(n, np.nan); roc[20:] = c[20:] / c[:-20] - 1
        a = roc > 0; p = np.roll(a, 1); p[0] = a[0]; return (~p) & a
    if fam == "RSI_30_50":
        r = rsi14(c); pr = np.roll(r, 1); pr[0] = r[0]
        return (pr < 30) & (r >= 30)            # cross UP out of oversold
    if fam == "BOLL_LO":
        m = sma(c, 20); sd = np.full(n, np.nan)
        for i in range(19, n): sd[i] = np.std(c[i - 19:i + 1])
        below = c < (m - 2 * sd); pb = np.roll(below, 1); pb[0] = below[0]
        return pb & (~below)                    # reclaim back above the lower band
    if fam == "VOLEXP":                          # MAGNITUDE channel: vol expands + up-filter
        ret = np.zeros(n); ret[1:] = c[1:] / c[:-1] - 1
        rv = np.full(n, np.nan)
        for i in range(20, n): rv[i] = np.std(ret[i - 19:i + 1])
        rvmed = np.full(n, np.nan)
        for i in range(120, n): rvmed[i] = np.median(rv[i - 99:i + 1])
        s100 = sma(c, 100)
        exp = (rv > 1.5 * rvmed); pe = np.roll(exp, 1); pe[0] = exp[0]
        return (~pe) & exp & np.isfinite(s100) & (c > s100)   # new vol-expansion, in uptrend
    if fam == "NRBREAK":                         # range-compression then break up
        rng = h - l
        rmed = np.full(n, np.nan)
        for i in range(20, n): rmed[i] = np.median(rng[i - 19:i + 1])
        narrow = rng < 0.6 * rmed                # compressed bar
        pn = np.roll(narrow, 1); pn[0] = narrow[0]
        hi1 = np.roll(h, 1)
        return pn & (c > hi1)                    # break above prior bar after compression
    if fam == "TRENDGATE":                        # reference: long while above SMA100 (re-arm on flip)
        s100 = sma(c, 100); a = np.isfinite(s100) & (c > s100)
        p = np.roll(a, 1); p[0] = a[0]; return (~p) & a
    raise ValueError(fam)


def ride_events(fam, cad, o, h, l, c, ms):
    """Event-clock: enter on signal -> ride <=3d (time-stop) with a 3xATR chandelier; one position
    at a time. Returns per-event nets tagged with split + asset handled by caller."""
    ent = entry_signal(fam, o, h, l, c)
    atr = atr14(h, l, c)
    n = len(c)
    maxhold = 3 * BARS_PER_DAY[cad]
    events = []
    i = 50
    while i < n - 2:
        if ent[i] and np.isfinite(o[i + 1]) and o[i + 1] > 0 and not np.isnan(atr[i]):
            f = i + 1
            entry = o[f]
            end = min(f + maxhold, n - 1)
            hw = c[f]; exit_idx = end
            for t in range(f, end):
                hw = max(hw, h[t])
                if not np.isnan(atr[t]) and c[t] < hw - 3.0 * atr[t]:
                    exit_idx = min(t + 1, end); break
            net = o[exit_idx] / entry - 1 - TAKER_RT
            events.append({"net": float(net), "split": split_of(int(ms[f])), "hold": exit_idx - f})
            i = exit_idx + 1                      # one position at a time (low turnover)
        else:
            i += 1
    return events


def _stat(nets):
    a = np.asarray(nets)
    if len(a) < 1:
        return {"n": 0}
    k = max(1, int(round(len(a) * 0.05)))
    return {"n": len(a), "mean": float(a.mean()), "se": float(a.std() / np.sqrt(len(a))),
            "win": float((a > 0).mean()), "jk5pct": float(np.sort(a)[:-k].mean()) if len(a) > k else None}


def run(universe, cadences):
    spec = yaml.safe_load(open(ROOT.parent / "config" / "universes" / f"{universe}.yaml"))
    if "assets" in spec:
        syms = [a["symbol"] for a in spec["assets"]]
    else:
        u50 = yaml.safe_load(open(ROOT.parent / "config" / "universes" / "u50.yaml"))
        syms = [a["symbol"] for a in u50["assets"]] + [a["symbol"] for a in spec.get("extra_assets", [])]
        syms = [s for s in dict.fromkeys(syms) if s not in set(spec.get("excluded_assets") or [])]

    report = {}
    for cad in cadences:
        per_asset_events = {}
        for sym in syms:
            try:
                df = ChimeraLoader().load(_norm_sym(sym), cadence=cad,
                                          features=["open", "high", "low", "close"])
            except Exception:
                continue
            ms = df["timestamp"].to_numpy()
            if (ms < OOS_END_MS).sum() < 300:
                continue
            keep = ms < OOS_END_MS                # UNSEEN sealed (never loaded into eval)
            o = df["open"].to_numpy().astype(float)[keep]
            h = df["high"].to_numpy().astype(float)[keep]
            l = df["low"].to_numpy().astype(float)[keep]
            c = df["close"].to_numpy().astype(float)[keep]
            mss = ms[keep]
            per_asset_events[sym] = {fam: ride_events(fam, cad, o, h, l, c, mss) for fam in FAMILIES}

        fam_report = {}
        for fam in FAMILIES:
            # per-instrument OOS mean (the generalization unit)
            per_asset_oos = {}
            pooled_oos, pooled_train = [], []
            for sym, fams in per_asset_events.items():
                ev = fams[fam]
                oos = [e["net"] for e in ev if e["split"] == "OOS"]
                tr = [e["net"] for e in ev if e["split"] == "TRAIN"]
                pooled_oos += oos; pooled_train += tr
                if len(oos) >= 3:
                    per_asset_oos[sym] = float(np.mean(oos))
            if len(per_asset_oos) < 5:
                fam_report[fam] = {"assets": len(per_asset_oos)}
                continue
            vals = np.array(list(per_asset_oos.values()))
            # persistence: per-asset TRAIN-sign vs OOS-sign agreement
            agree = []
            for sym in per_asset_oos:
                tr = [e["net"] for e in per_asset_events[sym][fam] if e["split"] == "TRAIN"]
                if len(tr) >= 3:
                    agree.append((np.mean(tr) > 0) == (per_asset_oos[sym] > 0))
            ps = _stat(pooled_oos)
            fam_report[fam] = {
                "assets": len(per_asset_oos),
                "breadth_pos": float((vals > 0).mean()),
                "breadth_cost": float((vals > TAKER_RT).mean()),
                "median_asset_oos": float(np.median(vals)),
                "pooled_oos_mean": ps["mean"], "pooled_oos_se": ps["se"],
                "pooled_oos_win": ps["win"], "pooled_oos_jk5pct": ps["jk5pct"],
                "pooled_oos_n": ps["n"],
                "persistence": float(np.mean(agree)) if agree else None,
                "median_hold_bars": int(np.median([e["hold"] for s in per_asset_events
                                                   for e in per_asset_events[s][fam]
                                                   if e["split"] == "OOS"]) or 0)
                if pooled_oos else None,
            }
        report[cad] = fam_report
    return {"universe": universe, "cadences": cadences, "n_assets": len(syms), "report": report}


def main():
    ap = argparse.ArgumentParser(prog="python -m strat.cross_instrument_generalize")
    ap.add_argument("--universe", default="u50")
    ap.add_argument("--cadences", default="4h,1h,1d")
    a = ap.parse_args()
    r = run(a.universe, a.cadences.split(","))

    for cad, fr in r["report"].items():
        print(f"\n## CROSS-INSTRUMENT GENERALIZATION -- {a.universe} {cad} -- event-clock <=3d hold "
              f"-- OOS held-out (UNSEEN sealed)")
        print(f"   {'family':11} {'breadth>0':>9} {'breadth>cost':>12} {'pooled mean+-se':>17} "
              f"{'win':>5} {'fair-jk':>9} {'persist':>7} {'assets':>6} {'GENERALIZES':>12}")
        # rank by breadth_cost then jk
        ranked = sorted([f for f in FAMILIES if fr.get(f, {}).get("assets", 0) >= 5],
                        key=lambda f: (fr[f]["breadth_cost"], fr[f].get("pooled_oos_jk5pct") or -9),
                        reverse=True)
        for fam in ranked:
            d = fr[fam]
            jk = d.get("pooled_oos_jk5pct")
            # AUDIT FIX: 'TRADEABLE' requires the jk (concentration-robust) edge to ITSELF
            # clear cost -- a positive raw mean with jk barely>0 is beta/concentration, not
            # a generalizing edge (VOLEXP: +1.88% mean but jk +0.57% = 91% PnL from 2 bull
            # months). Plus multiple-comparison reality across the grid: treat marginal as 'no'.
            robust = (jk is not None and jk > TAKER_RT)   # trimmed edge still beats cost
            gen = ("TRADEABLE" if d["breadth_cost"] > 0.5 and robust and d["pooled_oos_mean"] > 2 * TAKER_RT
                   else ("marginal" if d["breadth_pos"] > 0.5 and (jk or -9) > 0 else "no"))
            print(f"   {fam:11} {d['breadth_pos']*100:7.0f}% {d['breadth_cost']*100:10.0f}% "
                  f"{d['pooled_oos_mean']*100:+8.2f}+-{d['pooled_oos_se']*100:.2f}% "
                  f"{d['pooled_oos_win']*100:4.0f}% {(jk*100 if jk is not None else 0):+8.2f}% "
                  f"{(d['persistence'] or 0)*100:6.0f}% {d['assets']:6d} {gen:>12}")

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT / f"cross_instr_generalize_{a.universe}_{stamp}.json"
    json.dump({"repro": {"command": "python " + " ".join(sys.argv), "git_sha": sha}, "result": r},
              open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    print("READ: GENERALIZES iff breadth>0 majority + jk>0; TRADEABLE adds breadth>cost majority + mean>cost. "
          "OOS held-out; UNSEEN sealed. Predetermined params (no selection leak).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
