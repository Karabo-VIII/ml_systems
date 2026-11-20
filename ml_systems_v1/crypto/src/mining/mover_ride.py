"""MOVER-RIDE study -- 1m event-clock intraday mover riding, ORACLE + CAUSAL side by side.

THE OPEN CELL (user mandate 2026-06-10 + D68's own pointer):
D67 killed bar-level mover-capture ("a causal entry is too late -- by confirmation the
move is mostly done, you bank the give-back") at 1d/4h/1h. The mechanism is
RESOLUTION-BOUND: at 1m, a trigger at +1.5% realized run-up on a 5% mover still has most
of the move ahead. D68 measured the cadence gradient (daily-neg -> 4h +1.5 -> 1h +5.6pp
OOS) and explicitly left sub-hour untested. The 15m/30m "cost wall" killed per-BAR
rotation (many tiny trades), not event-clock riding (1 trade per asset-day, 1.5-3%
gross per true event vs 0.24% RT cost). This tool tests the cell honestly.

FRAMING (user): "we ride the trend, we don't predict it... capture as much of a mover as
possible, with the oracle framework and TI/causal framework side by side."

DESIGN:
  ORACLE side (hindsight, per fired event): meat remaining after the causal trigger
    fill -> day high; the capture CEILING. Plus the meat-after-trigger curve on
    mover days (decomposition: how much of a >=5% day survives each trigger level).
  CAUSAL side (strictly causal, UNCONDITIONAL -- the trigger never knows whether the
    day is a mover): first minute whose CLOSE >= day_open*(1+T) -> fill next-minute
    OPEN; ride a high-water trailing stop (k%), day-bounded exit (no overnight risk);
    one event per asset-day, no re-entry.
  KPI: capture_rate = causal_net / oracle_ceiling per event (the L2 KPI).
  NULL (r2, audit-corrected): K random POST-TRIGGER entry minutes in the same day
    (both arms condition identically on the fired trigger) -> a fair TIMING null.
    The r1 whole-day null was hindsight-conditioned (day membership encodes the
    future fact that the trigger fired; pre-trigger entries have the run-up
    guaranteed ahead) and carried ~0.5pp/event structural bias toward false KILL
    (zero-edge GBM probe). r1 deltas are descriptive only, never a gate.

PRE-REGISTRATION (gates stated before any result was read):
  Grid: T in {1.0, 1.5, 2.5}% x trail in {1, 2, 3}% = 9 cells, ALL reported.
  Selection: ON TRAIN ONLY -- highest portfolio net at 24bps RT with per-asset breadth
    (>=6/10 assets positive per-event mean). Exactly ONE cell advances to OOS.
  OOS gates for "alive": portfolio net > 0 at 24bps, breadth >= 6/10, survives
    drop-top-3-events jackknife, beats the random-entry null (cluster-boot p05 > 0).
  UNSEEN: never touched here (count only). Splits as D71: TRAIN <2024-01-01,
    OOS <2025-07-01, UNSEEN after.
Costs: net = gross - RT at {0, 6, 12, 24}bps; 24bps is the canonical gate.
Fills: decision at close of minute m -> fill at open of m+1 (never same-bar). Final
minute exits fill at that minute's close (market-on-close approximation, identical for
event and null). Seeded; lineage in JSON. No emoji (cp1252).

Run:
  python -m mining.mover_ride --universe u10
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
import polars as pl
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mining.cascade_oracle import load_panel  # noqa: E402  (1m grid + is_real + OI/regime)

OUT = ROOT / "runs" / "mining"
OUT.mkdir(parents=True, exist_ok=True)

__contract__ = {
    "kind": "research",
    "inputs": {"liq_subbar_1m": "data/processed/liq_subbar/<SYM>_1m.parquet (via cascade_oracle.load_panel)"},
    "outputs": {"study_json": "runs/mining/mover_ride_<tag>_<stamp>.json"},
    "invariants": {
        "causal_fills": "decision close of m -> fill open of m+1; trigger is unconditional (never sees day outcome)",
        "one_event_per_asset_day": "no re-entry; day-bounded exits",
        "preregistered_grid": "9 cells all reported; ONE TRAIN-selected cell goes to OOS",
        "unseen_untouched": "UNSEEN events counted, never analyzed here",
        "null_same_day": "random-entry null drawn from the SAME day (cost/regime/vol-matched)",
    },
}

TRAIN_END_MS = int(dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
OOS_END_MS = int(dt.datetime(2025, 7, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
COSTS_RT = [0.0, 0.0006, 0.0012, 0.0024]
GATE_COST = "24bps"
TRIGGERS = [0.010, 0.015, 0.025]
TRAILS = [0.01, 0.02, 0.03]
K_NULLS = 20
MIN_MINUTES_LEFT = 30        # trigger must leave >=30m of day to ride
DAY_MS = 86_400_000
CLUSTER_MS = 7 * DAY_MS      # weekly clusters for the bootstrap (cross-asset + serial)


def _norm_sym(s: str) -> str:
    s = s.upper()
    return s if s.endswith("USDT") else s + "USDT"


def split_of(ms: int) -> str:
    return "TRAIN" if ms < TRAIN_END_MS else ("OOS" if ms < OOS_END_MS else "UNSEEN")


# ----------------------------------------------------------------- per-day mechanics

def trail_exit(opens: np.ndarray, closes: np.ndarray, f: int, end: int, k: float) -> float:
    """Ride from fill row f (entry px opens[f]) to day end with a k% high-water trail.

    Decision at close of minute x in [f, end-1] (close < runmax*(1-k)) -> exit fill
    open[x+1] (x+1 <= end guaranteed). If never breached, exit at closes[end]
    (market-on-close). Vectorized (numpy): O(window) per call in C.
    """
    entry = opens[f]
    c = closes[f:end]
    runmax = np.maximum.accumulate(np.maximum(c, entry))
    breach = c < runmax * (1.0 - k)
    idx = int(np.argmax(breach))
    if breach[idx]:
        return float(opens[f + idx + 1] / entry - 1.0)
    return float(closes[end] / entry - 1.0)


def study_asset(sym: str, rng: np.random.Generator) -> dict:
    t0 = time.time()
    df = load_panel(sym)
    ms = df["minute_ts"].to_numpy()
    opens = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    real = df["is_real"].to_numpy().astype(bool)
    n = len(df)

    day_ids = ms // DAY_MS
    # day boundaries (1m grid is complete, so days are contiguous blocks)
    day_starts = np.flatnonzero(np.diff(day_ids, prepend=day_ids[0] - 1))
    day_ends = np.append(day_starts[1:] - 1, n - 1)

    events = []   # one dict per (day, T) fired -- the causal rider events
    days_meta = []  # per-day decomposition rows
    for ds, de in zip(day_starts, day_ends):
        if de - ds < 1200:  # skip partial days (<20h of grid)
            continue
        # day must be mostly real data
        if real[ds:de + 1].mean() < 0.90:
            continue
        d_open = opens[ds]
        if not np.isfinite(d_open) or d_open <= 0:
            continue
        d_high = float(np.max(highs[ds:de + 1]))
        d_close = closes[de]
        runup = d_high / d_open - 1.0
        day_ret = d_close / d_open - 1.0
        sp = split_of(int(ms[ds]))
        days_meta.append({"sym": sym, "day_ms": int(ms[ds]), "split": sp,
                          "runup": float(runup), "day_ret": float(day_ret)})
        if sp == "UNSEEN":
            continue  # UNSEEN: day counted above, never analyzed
        rel = closes[ds:de + 1] / d_open - 1.0
        for T in TRIGGERS:
            hit = np.flatnonzero(rel >= T)
            if len(hit) == 0:
                continue
            m = ds + int(hit[0])           # decision minute (close crossed T)
            f = m + 1                      # fill row
            if de - f < MIN_MINUTES_LEFT:
                continue
            entry = opens[f]
            if not np.isfinite(entry) or entry <= 0:
                continue
            # ORACLE ceiling from the SAME fill: best exit = max close after fill
            ceil_px = float(np.max(closes[f:de + 1]))
            oracle_ceiling = ceil_px / entry - 1.0
            ev = {
                "sym": sym, "day_ms": int(ms[ds]), "t0_ms": int(ms[m]), "split": sp,
                "T": T, "trigger_minute_of_day": int(m - ds),
                "runup_day": float(runup), "day_ret": float(day_ret),
                "is_mover3": bool(runup >= 0.03), "is_mover5": bool(runup >= 0.05),
                "oracle_ceiling": float(oracle_ceiling),
                "to_close_gross": float(d_close / entry - 1.0),
                "trails": {}, "nulls": {},
            }
            for k in TRAILS:
                g = trail_exit(opens, closes, f, de, k)
                ev["trails"][f"{round(k*100)}pct"] = float(g)
            # fair TIMING null (r2): random POST-trigger entries in the same day --
            # both arms condition identically on the fired trigger
            null_pool = np.arange(f + 1, de - MIN_MINUTES_LEFT)
            if len(null_pool) >= K_NULLS:
                picks = rng.choice(null_pool, size=K_NULLS, replace=False)
                for k in TRAILS:
                    vals = [trail_exit(opens, closes, int(p), de, k) for p in picks]
                    ev["nulls"][f"{round(k*100)}pct"] = float(np.mean(vals))
            events.append(ev)
    out = {"sym": sym, "events": events, "days": days_meta}
    n_d = len(days_meta)
    print(f"[{sym}] days={n_d} events={len(events)} "
          f"movers5(T/O)={sum(1 for d in days_meta if d['runup']>=0.05 and d['split']=='TRAIN')}/"
          f"{sum(1 for d in days_meta if d['runup']>=0.05 and d['split']=='OOS')} "
          f"({time.time()-t0:.0f}s)")
    return out


# ----------------------------------------------------------------- aggregation

def cell_events(events: list[dict], split: str, T: float, trail_key: str) -> list[dict]:
    return [e for e in events if e["split"] == split and e["T"] == T
            and trail_key in e["trails"]]


def portfolio_stats(evs: list[dict], trail_key: str, n_assets: int) -> dict:
    """Per-event and simple-portfolio stats for one (T, trail) cell, all cost scenarios."""
    if not evs:
        return {"n": 0}
    g = np.array([e["trails"][trail_key] for e in evs])
    out = {"n": len(g), "gross_mean": float(g.mean()), "gross_median": float(np.median(g)),
           "win_rate_gross": float((g > 0).mean())}
    # fixed split-span annualization (audit fix: never the cell's own event span)
    sp = evs[0]["split"]
    span_ms = (TRAIN_END_MS - int(dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
               if sp == "TRAIN" else OOS_END_MS - TRAIN_END_MS)
    years = span_ms / (365.25 * DAY_MS)
    for c in COSTS_RT:
        key = f"{round(c*1e4)}bps"
        net = g - c
        out[f"net_mean_{key}"] = float(net.mean())
        # equal-weight 1/n_assets per event, day-bounded, no compounding
        out[f"portfolio_net_per_year_{key}"] = float(net.sum() / n_assets / years)
    # capture rate vs oracle ceiling (only meaningful where ceiling > 0)
    caps = [e["trails"][trail_key] / e["oracle_ceiling"] for e in evs if e["oracle_ceiling"] > 0.005]
    out["capture_rate_median"] = float(np.median(caps)) if caps else None
    # paired null contrast (weekly-cluster bootstrap)
    pairs = [(e["trails"][trail_key] - e["nulls"][trail_key], e["t0_ms"] // CLUSTER_MS)
             for e in evs if trail_key in e.get("nulls", {})]
    if len(pairs) >= 5:
        d = np.array([p[0] for p in pairs])
        cl = np.array([p[1] for p in pairs])
        uniq = np.unique(cl)
        rng = np.random.default_rng(7)
        boots = []
        for _ in range(5_000):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            boots.append(np.concatenate([d[cl == c] for c in pick]).mean())
        boots = np.array(boots)
        out["null_contrast"] = {"n": len(d), "n_clusters": int(len(uniq)), "ci_level": 0.90,
                                "mean_delta": float(d.mean()),
                                "p05": float(np.quantile(boots, 0.05)),
                                "p95": float(np.quantile(boots, 0.95))}
    # breadth + jackknife on net at gate cost. Audit fix: denominator = FULL universe
    # (zero/thin-event assets count as non-positive); >=3 events to count positive.
    per = {}
    for e in evs:
        per.setdefault(e["sym"], []).append(e["trails"][trail_key] - 0.0024)
    rows = {s: float(np.mean(v)) for s, v in per.items() if len(v) >= 3}
    out["breadth_pos"] = sum(1 for v in rows.values() if v > 0)
    out["breadth_tot"] = n_assets
    out["breadth_qualifying"] = len(rows)
    net24 = np.sort(g - 0.0024)
    for K in (1, 3, 5):
        out[f"jk_drop_top{K}_net24_mean"] = float(net24[:-K].mean()) if len(net24) > K else None
    return out


def meat_curve(events: list[dict], split: str) -> dict:
    """Decomposition: on mover days (runup>=5%), what does each trigger level leave?"""
    out = {}
    for T in TRIGGERS:
        evs = [e for e in events if e["split"] == split and e["T"] == T and e["is_mover5"]]
        if not evs:
            out[f"T{T*100:.1f}"] = {"n": 0}
            continue
        ceil_ = np.array([e["oracle_ceiling"] for e in evs])
        out[f"T{T*100:.1f}"] = {
            "n_mover_events": len(evs),
            "oracle_ceiling_mean": float(ceil_.mean()),
            "oracle_ceiling_median": float(np.median(ceil_)),
            "trigger_minute_median": float(np.median([e["trigger_minute_of_day"] for e in evs])),
            "to_close_gross_mean": float(np.mean([e["to_close_gross"] for e in evs])),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="1m event-clock mover-ride study")
    ap.add_argument("--assets", nargs="+", default=None)
    ap.add_argument("--universe", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="u10")
    args = ap.parse_args()
    if args.assets:
        syms = [_norm_sym(a) for a in args.assets]
    elif args.universe:
        spec = yaml.safe_load(open(ROOT / "config" / "universes" / f"{args.universe}.yaml"))
        syms = [a["symbol"] for a in spec["assets"]]
        args.tag = args.universe
    else:
        ap.error("provide --assets or --universe")

    rng = np.random.default_rng(args.seed)
    all_events, all_days = [], []
    for sym in syms:
        try:
            res = study_asset(sym, rng)
        except FileNotFoundError as e:
            print(f"[{sym}] SKIP: {e}")
            continue
        all_events.extend(res["events"])
        all_days.extend(res["days"])
    n_assets = len({e["sym"] for e in all_events}) or 1

    # opportunity surface per split
    surface = {}
    for sp in ["TRAIN", "OOS", "UNSEEN"]:
        ds = [d for d in all_days if d["split"] == sp]
        surface[sp] = {
            "asset_days": len(ds),
            "mover3_days": sum(1 for d in ds if d["runup"] >= 0.03),
            "mover5_days": sum(1 for d in ds if d["runup"] >= 0.05),
        }

    # full pre-registered grid, TRAIN + OOS, all cells reported
    grid = {}
    for sp in ["TRAIN", "OOS"]:
        for T in TRIGGERS:
            for k in TRAILS:
                tk = f"{round(k*100)}pct"
                evs = cell_events(all_events, sp, T, tk)
                grid[f"{sp}|T{T*100:.1f}|trail{tk}"] = portfolio_stats(evs, tk, n_assets)

    # TRAIN selection (pre-registered criterion)
    best_key, best_val = None, -1e18
    for T in TRIGGERS:
        for k in TRAILS:
            tk = f"{round(k*100)}pct"
            s = grid[f"TRAIN|T{T*100:.1f}|trail{tk}"]
            if s.get("n", 0) < 30:
                continue
            if s.get("breadth_tot", 0) and s["breadth_pos"] / s["breadth_tot"] >= 0.6:
                v = s.get(f"portfolio_net_per_year_{GATE_COST}", -1e18)
                if v > best_val:
                    best_val, best_key = v, (T, tk)

    decomposition = {"TRAIN": meat_curve(all_events, "TRAIN"), "OOS": meat_curve(all_events, "OOS")}

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=ROOT).stdout.strip()
    except Exception:
        sha = "unknown"
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "tool": "mover_ride", "git_sha": sha, "seed": args.seed, "params": {
            "triggers": TRIGGERS, "trails": TRAILS, "k_nulls": K_NULLS,
            "min_minutes_left": MIN_MINUTES_LEFT, "gate_cost": GATE_COST,
            "selection": "TRAIN-only: max portfolio_net/yr @24bps with breadth>=0.6 and n>=30",
        },
        "surface": surface, "grid": grid,
        "train_selected_cell": ({"T": best_key[0], "trail": best_key[1],
                                 "train_portfolio_net_per_year_24bps": best_val}
                                if best_key else None),
        "decomposition_meat_curve": decomposition,
        "n_assets": n_assets,
        "caveats": [
            "u10 current membership = survivorship on absolute levels; contrasts are within-asset",
            "exit at final-minute close is a market-on-close approximation (identical for null)",
            "UNSEEN: days counted in surface only; zero events analyzed",
        ],
    }
    out_path = OUT / f"mover_ride_{args.tag}_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=1, default=str))

    # ----------- console story
    print("\n" + "=" * 78)
    print("MOVER-RIDE 1m EVENT-CLOCK -- ORACLE vs CAUSAL, STORY")
    print("=" * 78)
    for sp in ["TRAIN", "OOS"]:
        s = surface[sp]
        print(f"\n[{sp}] {s['asset_days']} asset-days; movers >=3%: {s['mover3_days']} "
              f"({s['mover3_days']/max(1,s['asset_days'])*100:.1f}%), >=5%: {s['mover5_days']} "
              f"({s['mover5_days']/max(1,s['asset_days'])*100:.1f}%)")
        mc = decomposition[sp]
        for tk, v in mc.items():
            if v.get("n_mover_events"):
                print(f"  MEAT after trigger {tk}: oracle ceiling mean {v['oracle_ceiling_mean']*100:+.2f}% "
                      f"median {v['oracle_ceiling_median']*100:+.2f}% (n={v['n_mover_events']} mover-events, "
                      f"trigger at median minute {v['trigger_minute_median']:.0f})")
    print("\nPRE-REGISTERED GRID (portfolio net %/yr @24bps | per-event net mean | breadth | null delta):")
    for sp in ["TRAIN", "OOS"]:
        print(f"  [{sp}]")
        for T in TRIGGERS:
            for k in TRAILS:
                tk = f"{round(k*100)}pct"
                s = grid[f"{sp}|T{T*100:.1f}|trail{tk}"]
                if s.get("n", 0) == 0:
                    continue
                nc = s.get("null_contrast", {})
                ncs = (f"null d {nc['mean_delta']*100:+.3f} [{nc['p05']*100:+.3f},{nc['p95']*100:+.3f}]"
                       if nc else "null --")
                print(f"    T{T*100:.1f} trail{tk}: {s.get(f'portfolio_net_per_year_{GATE_COST}',0)*100:+7.2f}%/yr | "
                      f"ev {s.get(f'net_mean_{GATE_COST}',0)*100:+.3f}% (n={s['n']}, win {s['win_rate_gross']*100:.0f}%) | "
                      f"breadth {s.get('breadth_pos',0)}/{s.get('breadth_tot',0)} | {ncs} | "
                      f"capture_med {s.get('capture_rate_median')}")
    if best_key:
        print(f"\nTRAIN-SELECTED CELL -> OOS: T={best_key[0]*100:.1f}% trail={best_key[1]} "
              f"(TRAIN {best_val*100:+.2f}%/yr @24bps)")
        s = grid[f"OOS|T{best_key[0]*100:.1f}|trail{best_key[1]}"]
        if s.get("n"):
            nc = s.get("null_contrast", {})
            jk3 = s.get("jk_drop_top3_net24_mean")
            jk3s = f"{jk3*100:+.3f}%" if jk3 is not None else "--"
            print(f"  OOS: {s.get(f'portfolio_net_per_year_{GATE_COST}',0)*100:+.2f}%/yr @24bps | "
                  f"ev {s.get(f'net_mean_{GATE_COST}',0)*100:+.3f}% (n={s['n']}) | "
                  f"breadth {s.get('breadth_pos',0)}/{s.get('breadth_tot',0)} | "
                  f"jk K=3 {jk3s} | "
                  + (f"null d {nc['mean_delta']*100:+.3f} [p05 {nc['p05']*100:+.3f}]" if nc else "null --"))
    else:
        print("\nNO TRAIN cell met the selection bar (n>=30, breadth>=0.6) -- grid is the verdict.")
    print(f"\nJSON -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
