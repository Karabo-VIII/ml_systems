"""src/strat/edge_search.py -- the EXHAUSTIVE held-out edge search (user 2026-06-10: "just find me a winning LO strat").
Searches the UNTESTED breadth axes -- momentum CONTINUATION, breakouts, strength, vol-expansion, structural-feature
entries -- over the BROAD universe (u100, incl. long-tail movers), at the 1d/3d/7d hold horizons, on HELD-OUT data,
each vs a RANDOM-entry null. Honest: reports per-setup forward return + whether it BEATS RANDOM on UNSEEN.

The reframe vs everything tried: we tested SELECTION (predict which asset moves) = null. This tests CONTINUATION /
SETUP (enter what is ALREADY breaking/strong, hold the 1-7d follow-through). Different mechanism (momentum
autocorrelation), genuinely under-tested at depth this session.

Per-trade forward return at the entry bar over h bars, NET of taker round-trip. Aggregated per window (TRAIN/OOS/
UNSEEN). A setup has a real edge only if UNSEEN mean forward-return > 0 AND beats its random-entry null. No emoji.
Run: python src/strat/edge_search.py --universe u100 --cadence 1d
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
from strat.entry_signal_lab import load_ohlc, WIN
from strat.tsmom_ensemble import _universe

TAKER_RT = 0.0024
HORIZONS = {"1d": 1, "3d": 3, "7d": 7}     # in DAILY bars (the user's targets)


def setups(df):
    """Boolean ENTRY masks per setup family (all PAST-ONLY at the bar). df indexed by date with OHLCV."""
    c = df["close"]; h = df["high"]; l = df["low"]; v = df.get("volume")
    out = {}
    # CONTINUATION / breakout families ("buy strength")
    out["breakout_20"] = c >= c.rolling(20, min_periods=10).max()                 # new 20-bar high
    out["breakout_50"] = c >= c.rolling(50, min_periods=25).max()
    out["donchian_55"] = c >= h.rolling(55, min_periods=28).max().shift(1)        # close breaks prior 55-bar high
    r5 = c / c.shift(5) - 1.0
    out["mom_up_5"] = (r5 > 0.15) & (c > c.rolling(20, min_periods=10).mean())    # +15% in 5d, in uptrend
    rng = (h - l) / c
    out["volexp_break"] = (rng > rng.rolling(20, min_periods=10).quantile(0.8)) & (c > c.shift(1)) & (c > c.rolling(20, min_periods=10).mean())
    out["accel"] = (c / c.shift(3) - 1 > 0.08) & (c.shift(3) / c.shift(6) - 1 > 0.0)   # accelerating up
    # pullback-in-uptrend (the dynamic-S/R use)
    ma20 = c.rolling(20, min_periods=10).mean(); ma50 = c.rolling(50, min_periods=25).mean()
    out["pullback_up"] = (ma20 > ma50) & (c <= ma20 * 1.02) & (c >= ma20 * 0.98) & (c.shift(1) < c.shift(2))
    if v is not None:
        vz = (v - v.rolling(20, min_periods=10).mean()) / (v.rolling(20, min_periods=10).std() + 1e-9)
        out["vol_surge_up"] = (vz > 2.0) & (c > c.shift(1)) & (c > c.rolling(20, min_periods=10).mean())   # volume spike + up
    return {k: m.fillna(False) for k, m in out.items()}


def fwd_returns(df, mask, h, cost=TAKER_RT):
    """Net forward return over h bars at each entry bar (next-bar fill -> +h)."""
    c = df["close"].to_numpy(float); idx = np.where(mask.to_numpy())[0]
    idx = idx[(idx + 1 + h) < len(c)]
    if len(idx) == 0: return np.array([]), np.array([])
    entry = c[idx + 1]; exit_ = c[idx + 1 + h]
    r = exit_ / entry - 1.0 - cost
    return r, df.index.to_numpy()[idx + 1]


def search(universe, cadence, seed=0, regime_gate=False):
    syms = _universe(universe)
    panels = {}
    for s in syms:
        d = load_ohlc(s, cadence)
        if d is None or len(d) < 120: continue
        # load_ohlc drops volume; re-fetch with volume via chimera
        panels[s] = d.set_index("date")
    # add volume from chimera if available
    from pipeline.chimera_loader import ChimeraLoader
    L = ChimeraLoader()
    oos_lo, oos_hi = pd.Timestamp(WIN.val_end), pd.Timestamp(WIN.oos_end)
    uns_lo, uns_hi = pd.Timestamp(WIN.oos_end), pd.Timestamp(WIN.unseen_end)
    rng = np.random.default_rng(seed)
    agg = {}     # setup -> horizon -> window -> list of returns ; + random null
    for s, df in panels.items():
        try:
            cd = L.load(s, cadence=cadence)
            cd = cd.to_pandas() if hasattr(cd, "to_pandas") else cd
            cd["date"] = pd.to_datetime(cd["date"], unit="ms") if str(cd["date"].dtype).startswith(("int", "float")) else pd.to_datetime(cd["date"])
            vol = cd.sort_values("date").drop_duplicates("date", keep="last").set_index("date")["volume"]
            df = df.copy(); df["volume"] = vol.reindex(df.index)
        except Exception:
            pass
        S = setups(df)
        if regime_gate:
            up = (df['close'] > df['close'].rolling(200, min_periods=100).mean()).fillna(False)
            S = {k: (m & up) for k, m in S.items()}
        for name, mask in S.items():
            for hk, hb in HORIZONS.items():
                r, dts = fwd_returns(df, mask, hb)
                if len(r) == 0: continue
                # random-entry null: same count of random bars
                rmask = pd.Series(False, index=df.index)
                pool = np.arange(len(df) - hb - 2)
                if len(pool) > len(r):
                    pick = rng.choice(pool, size=len(r), replace=False)
                    rmask.iloc[pick] = True
                    rr, rdts = fwd_returns(df, rmask, hb)
                else:
                    rr, rdts = np.array([]), np.array([])
                agg.setdefault(name, {}).setdefault(hk, {"TRAIN": [], "OOS": [], "UNSEEN": [], "NULL_UNSEEN": []})
                for ri, di in zip(r, dts):
                    di = pd.Timestamp(di)
                    w = "OOS" if (oos_lo <= di < oos_hi) else ("UNSEEN" if (uns_lo <= di < uns_hi) else ("TRAIN" if di < oos_lo else None))
                    if w: agg[name][hk][w].append(ri)
                for ri, di in zip(rr, rdts):
                    di = pd.Timestamp(di)
                    if uns_lo <= di < uns_hi: agg[name][hk]["NULL_UNSEEN"].append(ri)
    return agg


def report(agg):
    rows = []
    for name, hd in agg.items():
        for hk, w in hd.items():
            def m(key): return (round(float(np.mean(w[key])) * 100, 2), len(w[key])) if w[key] else (None, 0)
            tr, ntr = m("TRAIN"); oo, noo = m("OOS"); un, nun = m("UNSEEN"); nl, nnl = m("NULL_UNSEEN")
            hit = round(float(np.mean(np.array(w["UNSEEN"]) > 0)), 2) if w["UNSEEN"] else None
            beats_null = (un is not None and nl is not None and un > nl)
            rows.append({"setup": name, "h": hk, "n_unseen": nun, "TRAIN%": tr, "OOS%": oo, "UNSEEN%": un,
                         "UNSEEN_hit": hit, "NULL%": nl, "beats_null": beats_null})
    rows.sort(key=lambda r: (r["UNSEEN%"] if r["UNSEEN%"] is not None else -99), reverse=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m strat.edge_search")
    ap.add_argument("--universe", default="u100"); ap.add_argument("--cadence", default="1d")
    ap.add_argument("--regime-gate", action="store_true", help="only take setups when asset close>200d MA")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-preflight", action="store_true", help="skip the discovery coverage preflight")
    a = ap.parse_args(argv)
    # DISCOVERY PREFLIGHT (stage-03 contract): surface this run's dimensional coverage vs the canonical
    # lattice so omissions (other cadences, strat-based exits) are CONSCIOUS, not silent. Non-fatal.
    if not a.no_preflight:
        try:
            from framework.discovery_contract import preflight, CoverageResult
            pf = preflight({"name": f"edge_search:{a.universe}:{a.cadence}",
                            "axes": {"cadence": [a.cadence], "entry_policy": ["breakout_donchian"],
                                     "exit_mechanism": ["fixed_horizon_time_stop"], "exit_family": ["mechanical"],
                                     "regime": (["trend_sma200"] if a.regime_gate else [])}})
            print("## DISCOVERY PREFLIGHT -- coverage of the canonical lattice (omissions below are NOT tested this run)")
            print(CoverageResult(**pf["coverage"]).summary())
            print(f"   preflight verdict: {pf['verdict']}  (this is a single-cadence, mechanical-exit run BY DESIGN;\n"
                  f"   to silence a gap intentionally, waive it -- the point is no timeframe/exit-family is missed SILENTLY)\n")
        except Exception as e:
            print(f"[preflight skipped: {e}]")
    agg = search(a.universe, a.cadence, regime_gate=a.regime_gate)
    rows = report(agg)
    print(f"## EDGE SEARCH -- {a.universe} {a.cadence} -- per-trade forward return by setup x horizon, NET taker, HELD-OUT")
    print(f"   {'setup':14} {'h':>3} {'n_uns':>6} {'TRAIN%':>7} {'OOS%':>7} {'UNSEEN%':>8} {'hit':>4} {'NULL%':>6} {'beats_null':>10}")
    for r in rows:
        flag = "  <== EDGE" if (r["UNSEEN%"] is not None and r["UNSEEN%"] > 0 and r["beats_null"] and r["n_unseen"] >= 30) else ""
        print(f"   {r['setup']:14} {r['h']:>3} {r['n_unseen']:>6} {str(r['TRAIN%']):>7} {str(r['OOS%']):>7} {str(r['UNSEEN%']):>8} "
              f"{str(r['UNSEEN_hit']):>4} {str(r['NULL%']):>6} {str(r['beats_null']):>10}{flag}")
    if a.json:
        import subprocess
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        outdir = ROOT.parent / "runs" / "mining"; outdir.mkdir(parents=True, exist_ok=True)
        p = outdir / f"edge_search_{a.universe}_{a.cadence}{'_regime' if a.regime_gate else ''}.json"
        json.dump({"repro": {"command": "python " + " ".join(sys.argv), "git_sha": sha}, "results": rows},
                  open(p, "w", encoding="utf-8"), indent=2, default=str)
        print(f"[persisted] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
