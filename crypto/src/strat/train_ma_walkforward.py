"""TRAIN-ONLY WALK-FORWARD MA DISCOVERY -- the PEPE x MA exercise, pooled ACROSS assets.

User mandate (2026-06-11, /orc): per-asset config is FALSE (regime_dna_lab proved it).
Now test the NEXT granularities up -- per-CLUSTER and per-REGIME -- the honest way:
walk-forward WITHIN the training window only. Val / OOS / UNSEEN are NOT TOUCHED (this
is discovery/mining; the holdout stays sealed). Overlay 2-MA and 3-MA configs across
ALL assets in TRAIN.

THE QUESTION: is a config assignment at a given GRANULARITY *stable across time*? I.e.,
does the config you'd pick on TRAIN-fold f still win on TRAIN-fold f+1? A granularity
"earns its degrees of freedom" only if its walk-forward (out-of-fold, within-train)
performance BEATS the simplest granularity (POOLED = one config for everyone). This is
exactly the per-asset reality check, now asking: does CLUSTER or REGIME survive where
per-asset did not?

GRANULARITIES (config chosen per group on the SELECT window, tested on the NEXT fold):
  POOLED          : one config for all assets (the floor -- the "1 robust system" answer).
  CLUSTER         : one config per asset_dna cluster (BLUE / STEADY / VOLATILE).
  REGIME          : one config per causal trend regime (UP / DOWN) at entry.
  CLUSTER_REGIME  : one config per (cluster, regime).

FAMILIES (predetermined grids; the PEPE x MA echo):
  2-MA cross : type {SMA,EMA} x (fast,slow) golden/death cross, fast<slow.
  3-MA align : type {SMA,EMA} x (fast,mid,slow) -- long when fast>mid>slow, exit fast<mid.

WALK-FORWARD (within TRAIN, anchored/expanding): TRAIN bars split into K chronological
folds. For each fold f>=1: SELECT best config per group on folds[0..f-1] (all earlier
train data); TEST it on fold f. Aggregate per-trade economics across the test folds.
Force-close at fold boundaries -> no cross-fold look-ahead. A granularity is VALID iff
its mean out-of-fold per-trade net BEATS POOLED across the test folds (it earns its DoF).

HONESTY (carries the regime_dna_lab audit lessons): per-trade expectancy +- se (NOT
sum/breadth); count-invariant concentration (drop-top-5pct + n_eff); causal regime;
force-close at fold edges; selection strictly on the SELECT window. No emoji (cp1252).

Run:
  python -m strat.train_ma_walkforward --universe u10 --folds 5
  python -m strat.train_ma_walkforward --universe u50 --folds 6
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

from mining.family_regime_map import sma, ema, atr14, COST_RT, _norm_sym  # noqa: E402
from pipeline.chimera_loader import ChimeraLoader  # noqa: E402

OUT = ROOT.parent / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)

# TRAIN = everything strictly before val_end. Val/OOS/Unseen are NEVER loaded into eval.
TRAIN_END_MS = int(dt.datetime(2024, 5, 15, tzinfo=dt.timezone.utc).timestamp() * 1000)
REGIME_SMA = 150
MIN_TRADES_SELECT = 8

# ---- config grids (predetermined) ----
MA_TYPES = ("SMA", "EMA")
PAIRS_2MA = [(5, 20), (5, 50), (10, 50), (10, 100), (20, 100), (20, 200), (50, 100), (50, 200)]
TRIPLES_3MA = [(5, 20, 100), (5, 20, 200), (10, 50, 100), (10, 50, 200), (20, 50, 200), (5, 50, 200)]


def configs(family: str):
    if family == "2MA":
        return [("2MA", t, f, s, None) for t in MA_TYPES for (f, s) in PAIRS_2MA]
    return [("3MA", t, f, m, s) for t in MA_TYPES for (f, m, s) in TRIPLES_3MA]


def _ma(c, n, kind):
    return sma(c, n) if kind == "SMA" else ema(c, n)


def signals(cfg, c):
    """ent/exi boolean arrays. 2MA: golden/death cross. 3MA: fast>mid>slow / fast<mid."""
    fam, kind = cfg[0], cfg[1]
    if fam == "2MA":
        f, s = _ma(c, cfg[2], kind), _ma(c, cfg[3], kind)
        above = f > s
    else:
        f, m, s = _ma(c, cfg[2], kind), _ma(c, cfg[3], kind), _ma(c, cfg[4], kind)
        above = (f > m) & (m > s)            # bullish alignment
        # exit when fast loses mid (looser than full mis-alignment -> rides less give-back)
        below_exit = f < m
        prev = np.roll(above, 1); prev[0] = above[0]
        ent = (~prev) & above
        prevx = np.roll(below_exit, 1); prevx[0] = below_exit[0]
        exi = (~prevx) & below_exit
        return ent, exi
    prev = np.roll(above, 1); prev[0] = above[0]
    return (~prev) & above, prev & (~above)


def simulate(cfg, o, c, fold_arr, regime, atr) -> list[dict]:
    """Long-only; signal close t -> fill open t+1; FORCE-CLOSE at fold boundary
    (no cross-fold leak). Each trade tagged fold/regime at entry."""
    ent, exi = signals(cfg, c)
    n = len(c)
    trades = []
    in_pos = False
    entry_px = 0.0
    e_fold = e_reg = e_idx = None
    for t in range(1, n - 1):
        if not in_pos:
            if ent[t] and np.isfinite(o[t + 1]) and o[t + 1] > 0 and not np.isnan(atr[t]):
                in_pos, entry_px = True, o[t + 1]
                e_fold, e_reg, e_idx = fold_arr[t], regime[t], t + 1
        else:
            if fold_arr[t] != e_fold:         # fold boundary -> force-close at the LAST
                # close of the ENTRY fold (c[t-1]); FIX(audit): never price a force-close
                # at a test-fold bar (o[t]) that then leaks into the entry-fold selection
                trades.append({"net": c[t - 1] / entry_px - 1 - COST_RT, "fold": e_fold,
                               "regime": e_reg})
                in_pos = False
                continue
            if exi[t]:
                trades.append({"net": o[t + 1] / entry_px - 1 - COST_RT, "fold": e_fold,
                               "regime": e_reg})
                in_pos = False
    if in_pos:
        trades.append({"net": o[n - 1] / entry_px - 1 - COST_RT, "fold": e_fold, "regime": e_reg})
    return trades


def _drop_top_pct(a, pct=0.05):
    k = max(1, int(round(len(a) * pct)))
    return float(np.sort(a)[:-k].mean()) if len(a) > k else None


def _n_eff(a):
    w = a[a > 0]
    return float(1.0 / np.sum((w / w.sum()) ** 2)) if len(w) and w.sum() > 0 else 0.0


def _stat(nets):
    if len(nets) < 1:
        return {"n": 0}
    a = np.asarray(nets)
    return {"n": len(a), "mean": float(a.mean()), "se": float(a.std() / np.sqrt(len(a))),
            "win": float((a > 0).mean()), "jk5pct": _drop_top_pct(a), "n_eff": _n_eff(a)}


def run(universe: str, family: str, folds: int, cadence: str = "1d"):
    spec = yaml.safe_load(open(ROOT.parent / "config" / "universes" / f"{universe}.yaml"))
    if "assets" in spec:
        rows = spec["assets"]
    else:
        u50 = yaml.safe_load(open(ROOT.parent / "config" / "universes" / "u50.yaml"))
        rows = u50["assets"] + spec.get("extra_assets", [])
        excl = set(spec.get("excluded_assets") or [])
        rows = [r for r in rows if r["symbol"] not in excl]
    cluster_of = {r["symbol"]: r.get("dna", "NA") for r in rows}
    cfgs = configs(family)

    # per asset: trades for every config, TRAIN only, tagged fold+regime
    book = {}          # sym -> {cfg_id: [trades]}
    for r in rows:
        sym = r["symbol"]
        try:
            df = ChimeraLoader().load(_norm_sym(sym), cadence=cadence,
                                      features=["open", "high", "low", "close"])
        except Exception:
            continue
        ts = df["timestamp"].to_numpy()
        keep = ts < TRAIN_END_MS              # TRAIN ONLY -- holdout never loaded
        if keep.sum() < 300:
            continue
        o = df["open"].to_numpy().astype(float)[keep]
        h = df["high"].to_numpy().astype(float)[keep]
        l = df["low"].to_numpy().astype(float)[keep]
        c = df["close"].to_numpy().astype(float)[keep]
        # chronological fold assignment over TRAIN bars
        nb = len(c)
        fold_arr = np.minimum((np.arange(nb) * folds) // nb, folds - 1)
        sm = sma(c, REGIME_SMA)
        regime = np.where(np.isfinite(sm) & (c > sm), "UP", "DOWN").astype(object)
        atr = atr14(h, l, c)
        book[sym] = {ci: simulate(cfg, o, c, fold_arr, regime, atr) for ci, cfg in enumerate(cfgs)}
    assets = list(book.keys())

    def trades_of(sym, ci, sel_folds=None, test_fold=None, regime=None):
        out = []
        for tr in book[sym][ci]:
            if sel_folds is not None and tr["fold"] not in sel_folds:
                continue
            if test_fold is not None and tr["fold"] != test_fold:
                continue
            if regime is not None and tr["regime"] != regime:
                continue
            out.append(tr["net"])
        return out

    # ---- WALK-FORWARD: select on folds[0..f-1], test on fold f ----
    # FIX(audit): record per-(asset,fold) so the granularity comparison is PAIRED
    # (matched denominators) instead of two-sample means over different-sized books.
    regimes = ["UP", "DOWN"]
    clusters = sorted(set(cluster_of[s] for s in assets))
    GRANS = ["POOLED", "CLUSTER", "REGIME", "CLUSTER_REGIME"]
    gran_af = {g: {} for g in GRANS}       # (asset,fold) -> [nets] under that granularity
    picks_log = {g: {} for g in GRANS}

    def best_cfg(syms, sel_folds, regime=None):
        scored = []
        for ci in range(len(cfgs)):
            nets = [x for s in syms for x in trades_of(s, ci, sel_folds=sel_folds, regime=regime)]
            if len(nets) >= MIN_TRADES_SELECT:
                scored.append((ci, float(np.mean(nets))))
        return max(scored, key=lambda x: x[1])[0] if scored else None

    def _put(g, s, f, v):
        if v:
            gran_af[g].setdefault((s, f), []).extend(v)

    for f in range(1, folds):
        sel = list(range(f))
        ci = best_cfg(assets, sel)                                   # POOLED
        if ci is not None:
            for s in assets:
                _put("POOLED", s, f, trades_of(s, ci, test_fold=f))
            picks_log["POOLED"].setdefault(f, cfgs[ci])
        for cl in clusters:                                          # CLUSTER
            syms = [s for s in assets if cluster_of[s] == cl]
            ci = best_cfg(syms, sel)
            if ci is not None:
                for s in syms:
                    _put("CLUSTER", s, f, trades_of(s, ci, test_fold=f))
                picks_log["CLUSTER"].setdefault(f, {})[cl] = cfgs[ci]
        reg_ci = {rg: best_cfg(assets, sel, regime=rg) for rg in regimes}   # REGIME
        for s in assets:
            for rg in regimes:
                if reg_ci[rg] is not None:
                    _put("REGIME", s, f, trades_of(s, reg_ci[rg], test_fold=f, regime=rg))
        for rg in regimes:
            if reg_ci[rg] is not None:
                picks_log["REGIME"].setdefault(f, {})[rg] = cfgs[reg_ci[rg]]
        for cl in clusters:                                          # CLUSTER_REGIME
            syms = [s for s in assets if cluster_of[s] == cl]
            for rg in regimes:
                ci = best_cfg(syms, sel, regime=rg)
                if ci is not None:
                    for s in syms:
                        _put("CLUSTER_REGIME", s, f, trades_of(s, ci, test_fold=f, regime=rg))
                    picks_log["CLUSTER_REGIME"].setdefault(f, {})[f"{cl}/{rg}"] = cfgs[ci]

    summary = {g: _stat([x for nets in gran_af[g].values() for x in nets]) for g in GRANS}

    # PAIRED verdict (FIX(audit)): per (asset,fold) present in BOTH g and POOLED, take
    # the diff of per-trade means -> matched denominators, market-correlation cancels.
    # Report POWER honestly: MDE at 80% power / one-sided 5%. VALID requires the edge to
    # be both POSITIVE-significant AND the test POWERED enough to mean something.
    Z = 2.4865  # z_.95 + z_.80
    verdict = {}
    for g in ["CLUSTER", "REGIME", "CLUSTER_REGIME"]:
        diffs = [float(np.mean(gran_af[g][k]) - np.mean(gran_af["POOLED"][k]))
                 for k in gran_af[g] if k in gran_af["POOLED"]]
        if len(diffs) >= 5:
            d = np.array(diffs)
            se = float(d.std() / np.sqrt(len(d)))
            mde = Z * se
            verdict[g] = {"paired_n": len(d), "paired_mean_diff": float(d.mean()),
                          "paired_se": se, "t": float(d.mean() / (se + 1e-12)),
                          "frac_pos": float((d > 0).mean()), "MDE_80pct": mde,
                          "POWERED_for_5pp": bool(mde < 0.05),
                          "VALID": bool(d.mean() > 1.645 * se),
                          "verdict": ("VALID" if d.mean() > 1.645 * se
                                      else ("INCONCLUSIVE-underpowered" if mde >= 0.05
                                            else "no-edge (powered)"))}
    return {"universe": universe, "family": family, "folds": folds, "n_assets": len(assets),
            "clusters": {cl: [s for s in assets if cluster_of[s] == cl] for cl in clusters},
            "summary": summary, "verdict_paired": verdict,
            "picks": {g: {str(k): v for k, v in d.items()} for g, d in picks_log.items()}}


def main():
    ap = argparse.ArgumentParser(prog="python -m strat.train_ma_walkforward")
    ap.add_argument("--universe", default="u10")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--family", default="both", choices=["2MA", "3MA", "both"])
    ap.add_argument("--cadence", default="1d")
    a = ap.parse_args()
    fams = ["2MA", "3MA"] if a.family == "both" else [a.family]
    out = {}
    for fam in fams:
        r = run(a.universe, fam, a.folds, a.cadence)
        out[fam] = r
        print(f"\n## WALK-FORWARD (TRAIN only, {a.folds} folds) -- {a.universe} {a.cadence} {fam} "
              f"-- {r['n_assets']} assets; Val/OOS/Unseen UNTOUCHED")
        print(f"   clusters: " + ", ".join(f"{k}:{len(v)}" for k, v in r["clusters"].items()))
        print(f"   {'granularity':16} {'oof mean+-se(descr)':>20} {'win':>5} {'n':>6} | PAIRED vs POOLED (the verdict)")
        for g in ["POOLED", "CLUSTER", "REGIME", "CLUSTER_REGIME"]:
            s = r["summary"][g]
            if not s.get("n"):
                continue
            v = r["verdict_paired"].get(g, {})
            if v:
                vs = (f"diff {v['paired_mean_diff']*100:+.2f}+-{v['paired_se']*100:.2f}pp "
                      f"t={v['t']:+.2f} MDE{v['MDE_80pct']*100:.0f}pp -> {v['verdict']}")
            else:
                vs = "(baseline)"
            print(f"   {g:16} {s['mean']*100:+.2f}+-{s['se']*100:.2f}%".ljust(40) +
                  f" {s['win']*100:4.0f}% {s['n']:6d} | {vs}")
        # show the per-cluster / per-regime DNA from the LAST fold's picks
        cl_pick = r["picks"]["CLUSTER"].get(str(a.folds - 1), {})
        rg_pick = r["picks"]["REGIME"].get(str(a.folds - 1), {})
        if cl_pick:
            print("   last-fold CLUSTER DNA: " + ", ".join(
                f"{k}={v[1]}{v[2]}/{v[3]}" + (f"/{v[4]}" if v[4] else "") for k, v in cl_pick.items()))
        if rg_pick:
            print("   last-fold REGIME DNA:  " + ", ".join(
                f"{k}={v[1]}{v[2]}/{v[3]}" + (f"/{v[4]}" if v[4] else "") for k, v in rg_pick.items()))

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT / f"train_ma_wf_{a.universe}_{a.folds}f_{stamp}.json"
    json.dump({"repro": {"command": "python " + " ".join(sys.argv), "git_sha": sha}, "result": out},
              open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[persisted] {p}")
    print("READ: a granularity is VALID iff its walk-forward out-of-fold mean beats POOLED by >1 se.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
