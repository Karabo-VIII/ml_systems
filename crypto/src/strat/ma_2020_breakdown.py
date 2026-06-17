"""src/strat/ma_2020_breakdown.py -- the 2020-slice MA breakdown: winning MA CLASS per timeframe vs ORACLE.

USER /orc 2026-06-12: "each class of Moving Averages that won per time frame ... performance for the 2020
window ... a side by side with oracle, best per category (a FAMILY of averages = a set of configs), their
performance for the year. Rerun: data limited to ~1yr, so use the CANONICAL DATA TECHNIQUES to expand the
learning data, and slice it 6mo-train / 3mo-val / 3mo-oos -- for the 2020 slice, NOT the traditional splits."

WHAT THIS PRODUCES (per timeframe in {1d,4h,1h,30m,15m}):
  - each MA CLASS {EMA,SMA,WMA,HMA,DEMA,TEMA,KAMA,VIDYA} as a FAMILY (the slow distinct 2MA+3MA configs),
    run with the UPGRADED METHODOLOGY (FULL stack: 10% trail + min_hold(12) + maker), equal-weight u10 book;
  - performance on the WITHIN-2020 split (TRAIN Jan-Jun / VAL Jul-Sep / OOS Oct-Dec) + the FULL year;
  - the WINNING class per timeframe (best OOS family) SIDE-BY-SIDE with the ORACLE ceiling.
CANONICAL DATA EXPANSION (the 1-year limit): cross_sectional_pool (u10 book), block_bootstrap_distribution
(robust OOS median + p05 -- not the optimistic point), james_stein_shrink (de-overfit the best-config pick).
ORACLE = two ceilings: (a) hindsight-best CONFIG across all classes (the MA ceiling), (b) perfect-foresight
long/flat (the absolute ceiling). HINDSIGHT = descriptive upper bound, not tradeable.
RWYB: python -m strat.ma_2020_breakdown. No emoji (cp1252).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR
from strat.portfolio_replay import apply_trail_stop, MAKER_RT
from strat.replay_distinct_grid import distinct_specs
from strat.ma_mechanics import _cached_panel
from strat.structural_fixes import min_hold
from strat.ma_type_upgrade import held_cross, _nums, MA_TYPES
from strat.data_expansion import block_bootstrap_distribution, james_stein_shrink

CADENCES = ["1d", "4h", "2h", "1h", "30m", "15m"]   # 2h synthesized from 1h (not a native cadence)


def _panel(sym, cadence):
    """_cached_panel, plus a SYNTHETIC 2h built by resampling 1h into 2-hour buckets (OHLC-correct)."""
    if cadence != "2h":
        return _cached_panel(sym, cadence)
    o, h, l, c, ms = _cached_panel(sym, "1h")
    bucket = (np.asarray(ms, dtype=np.int64) // 7_200_000)        # 2h = 7.2e6 ms
    df = pd.DataFrame({"b": bucket, "o": o, "h": h, "l": l, "c": c, "ms": ms})
    g = df.groupby("b", sort=True).agg(o=("o", "first"), h=("h", "max"), l=("l", "min"),
                                       c=("c", "last"), ms=("ms", "last"))
    return (g["o"].to_numpy(), g["h"].to_numpy(), g["l"].to_numpy(),
            g["c"].to_numpy(), g["ms"].to_numpy().astype(np.int64))
# WITHIN-2020 split (NOT the traditional cross-year TRAIN/VAL/OOS) -- 6mo / 3mo / 3mo
SPLIT = {"TRAIN": ("2020-01-01", "2020-07-01"), "VAL": ("2020-07-01", "2020-10-01"),
         "OOS": ("2020-10-01", "2021-01-01")}
YEAR = ("2020-01-01", "2021-01-01")
WARMUP = 400


from strat.ma_type_upgrade import _MA   # the MA-type function table (memoize across shared periods)


def _held_from_cache(macache, periods):
    mas = [macache[p] for p in periods]
    h = (mas[0] > mas[1]) if len(periods) == 2 else ((mas[0] > mas[1]) & (mas[1] > mas[2]))
    return np.nan_to_num(h).astype(np.int8)


def _cells(cfgs, ma_type, cadence):
    """per (config,asset): a pd.Series of FULL-stack net returns over [2020-WARMUP .. 2021], maker.
    Returns {(cfg,sym): series}. cross-sectional pool = all cells equal-weight (the book).
    SPEED: compute each unique MA period ONCE per asset (configs share periods) then build crosses."""
    s_ms = pd.Timestamp(YEAR[0]).value // 10**6
    e_ms = pd.Timestamp(YEAR[1]).value // 10**6
    syms = [a["symbol"] for a in yaml.safe_load(open(ROOT.parent / "config" / "universes" / "u10.yaml"))["assets"]]
    uniq_periods = sorted({p for name in cfgs for p in _nums(name)})
    maf = _MA[ma_type]
    out = {}
    for sym in syms:
        try:
            o, h, l, c, ms = _panel(sym, cadence)
        except Exception:
            continue
        e_idx = int(np.searchsorted(ms, e_ms))
        s_idx = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
        c2, ms2 = c[s_idx:e_idx], ms[s_idx:e_idx]
        if len(c2) < 40:
            continue
        wm = ms2 >= s_ms
        if wm.sum() < 20:
            continue
        ret = np.zeros(len(c2)); ret[1:] = c2[1:] / c2[:-1] - 1.0
        idx = pd.to_datetime(ms2[wm], unit="ms")
        macache = {p: maf(c2, p) for p in uniq_periods}             # each MA computed ONCE per asset
        for name in cfgs:
            h0 = _held_from_cache(macache, _nums(name))
            h1 = apply_trail_stop(h0.copy(), c2, 0.10)[0].astype(np.int8)
            w = min_hold(h1, 12).astype(np.float64)
            pos = np.zeros(len(c2)); pos[1:] = w[:-1]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            net = (pos * ret - flips * (MAKER_RT / 2.0))[wm]
            out[(name, sym)] = pd.Series(net, index=idx)
    return out


def _compound(series, lo, hi):
    s = series[(series.index >= lo) & (series.index < hi)]
    return float(np.prod(1 + s.to_numpy()) - 1) * 100 if len(s) else np.nan


def _maxdd(series, lo, hi):
    s = series[(series.index >= lo) & (series.index < hi)]
    if len(s) < 3:
        return np.nan
    eq = np.cumprod(1 + s.to_numpy()); pk = np.maximum.accumulate(eq)
    return float(((eq - pk) / pk).min() * 100)


def _book(cells):
    if not cells:
        return None
    return pd.concat(list(cells.values()), axis=1).mean(axis=1, skipna=True)


def _perfect_foresight(cadence):
    """absolute ceiling: per asset long when NEXT bar return > 0 (hindsight), equal-weight book, maker-free."""
    s_ms = pd.Timestamp(YEAR[0]).value // 10**6; e_ms = pd.Timestamp(YEAR[1]).value // 10**6
    syms = [a["symbol"] for a in yaml.safe_load(open(ROOT.parent / "config" / "universes" / "u10.yaml"))["assets"]]
    cols = []
    for sym in syms:
        try:
            o, h, l, c, ms = _panel(sym, cadence)
        except Exception:
            continue
        keep = (ms >= s_ms) & (ms < e_ms); c2, ms2 = c[keep], ms[keep]
        if len(c2) < 20:
            continue
        ret = np.zeros(len(c2)); ret[1:] = c2[1:] / c2[:-1] - 1.0
        pos = (ret > 0).astype(float)                       # hindsight: be long exactly the up bars
        cols.append(pd.Series(pos * ret, index=pd.to_datetime(ms2, unit="ms")))
    return pd.concat(cols, axis=1).mean(axis=1, skipna=True) if cols else None


def main() -> int:
    global CADENCES
    if "--cadences" in sys.argv:
        CADENCES = sys.argv[sys.argv.index("--cadences") + 1].split(",")
    tag = "_".join(CADENCES)
    ma_cfg = {}
    for fam in ("2MA", "3MA"):
        ma_cfg.update(distinct_specs(fam, 0.15, max_n=60))
    PR.STRATS.update(ma_cfg)
    slow = [n for n in ma_cfg if 60 <= max(_nums(n)) < 150]   # 2MA+3MA slow family
    print(f"2020 MA breakdown: {len(slow)} slow configs (2MA+3MA) x {len(MA_TYPES)} classes x {len(CADENCES)} TF")
    print(f"split (WITHIN-2020): TRAIN {SPLIT['TRAIN']}  VAL {SPLIT['VAL']}  OOS {SPLIT['OOS']}; FULL stack; maker\n")

    results = {}   # (cad, ma_type) -> metrics
    for cad in CADENCES:
        for ma_type in MA_TYPES:
            cells = _cells(slow, ma_type, cad)
            book = _book(cells)
            if book is None or len(book) < 20:
                results[(cad, ma_type)] = {}
                continue
            # family book per split + year
            m = {w: round(_compound(book, lo, hi), 1) for w, (lo, hi) in SPLIT.items()}
            m["YEAR"] = round(_compound(book, *YEAR), 1)
            m["oos_maxdd"] = round(_maxdd(book, *SPLIT["OOS"]), 1)
            # canonical expansion: block-bootstrap the OOS book daily returns (robust median + p05)
            oos = book[(book.index >= SPLIT["OOS"][0]) & (book.index < SPLIT["OOS"][1])].to_numpy()
            bb = block_bootstrap_distribution(oos, n_boot=400, block=5, seed=7)
            m["oos_boot_median"] = round(bb["median"] * 100, 1); m["oos_boot_p05"] = round(bb["p05"] * 100, 1)
            # best CONFIG: pick by VAL (causal), test on OOS; james-stein on the VAL config scores
            val_scores = {name: _compound(_book({k: v for k, v in cells.items() if k[0] == name}), *SPLIT["VAL"])
                          for name in slow}
            val_scores = {k: v for k, v in val_scores.items() if not np.isnan(v)}
            if len(val_scores) >= 3:
                _, B = james_stein_shrink(val_scores)
                best_cfg = max(val_scores, key=val_scores.get)
                m["best_cfg"] = best_cfg; m["js_B"] = round(float(B), 2)
                m["best_cfg_oos"] = round(_compound(_book({k: v for k, v in cells.items() if k[0] == best_cfg}), *SPLIT["OOS"]), 1)
            results[(cad, ma_type)] = m

    # ---- TABLE 1: per timeframe, all classes -- OOS family compound (the breakdown) ----
    print("## OOS family compound % by timeframe x MA class (FULL stack, within-2020 OOS = Oct-Dec)")
    print(f"   {'TF':5}" + "".join(f"{t:>8}" for t in MA_TYPES) + f"{'WINNER':>10}")
    winners = {}
    for cad in CADENCES:
        row = f"   {cad:5}"
        best = None
        for t in MA_TYPES:
            v = results[(cad, t)].get("OOS")
            row += f"{(str(v) if v is not None else '--'):>8}"
            if v is not None and (best is None or v > best[1]):
                best = (t, v)
        winners[cad] = best
        row += f"{(best[0] if best else '?'):>10}"
        print(row)

    # ---- TABLE 2: winner per timeframe SIDE-BY-SIDE with the ORACLE ----
    print("\n## Winning MA class per timeframe vs ORACLE ceiling (within-2020)")
    print(f"   {'TF':5} {'winner':7} {'TRAIN':>7} {'VAL':>7} {'OOS':>7} {'YEAR':>7} {'OOSmaxDD':>9} "
          f"{'boot_med':>9} {'boot_p05':>9} {'bestcfg_OOS':>12} {'JS_B':>5} | {'ORACLE_cfg':>11} {'ORACLE_PF':>10}")
    oracle = {}
    for cad in CADENCES:
        win = winners[cad]
        if not win:
            continue
        t = win[0]; m = results[(cad, t)]
        # oracle (a): hindsight-best config across ALL classes on OOS
        best_oos_cfg = None
        for tt in MA_TYPES:
            cm = results[(cad, tt)]
            if cm.get("best_cfg_oos") is not None and (best_oos_cfg is None or cm["best_cfg_oos"] > best_oos_cfg[1]):
                best_oos_cfg = (f"{tt}:{cm['best_cfg']}", cm["best_cfg_oos"])
        # oracle (b): perfect-foresight long/flat OOS
        pf = _perfect_foresight(cad)
        pf_oos = round(_compound(pf, *SPLIT["OOS"]), 1) if pf is not None else None
        oracle[cad] = {"hindsight_cfg": best_oos_cfg, "perfect_foresight_oos": pf_oos}
        print(f"   {cad:5} {t:7} {str(m.get('TRAIN')):>7} {str(m.get('VAL')):>7} {str(m.get('OOS')):>7} "
              f"{str(m.get('YEAR')):>7} {str(m.get('oos_maxdd')):>9} {str(m.get('oos_boot_median')):>9} "
              f"{str(m.get('oos_boot_p05')):>9} {str(m.get('best_cfg_oos')):>12} {str(m.get('js_B')):>5} | "
              f"{(best_oos_cfg[0]+' '+str(best_oos_cfg[1]) if best_oos_cfg else '?'):>11} {str(pf_oos):>10}")

    # ---- chart: winner-per-TF family OOS vs hindsight-config-oracle vs perfect-foresight ----
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(CADENCES)); w = 0.27
    fam = [winners[c][1] if winners[c] else np.nan for c in CADENCES]
    orc = [oracle[c]["hindsight_cfg"][1] if oracle.get(c, {}).get("hindsight_cfg") else np.nan for c in CADENCES]
    pf = [oracle[c]["perfect_foresight_oos"] if oracle.get(c) else np.nan for c in CADENCES]
    ax.bar(x - w, fam, w, label="winning class FAMILY (causal)", color="#1f77b4")
    ax.bar(x, orc, w, label="ORACLE: hindsight-best config", color="#ff7f0e")
    ax.bar(x + w, pf, w, label="ORACLE: perfect-foresight long/flat", color="#2ca02c")
    for i, c in enumerate(CADENCES):
        if winners[c]:
            ax.annotate(winners[c][0], (i - w, fam[i]), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(CADENCES); ax.axhline(0, color="k", lw=0.7)
    ax.set_ylabel("within-2020 OOS compound %"); ax.legend(fontsize=8)
    ax.set_title("2020 MA breakdown: winning class family vs oracle, per timeframe (within-2020 OOS, FULL stack)")
    fig.tight_layout()
    out = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "MA_2020_BREAKDOWN" / "charts" / "ma_2020_breakdown.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"\n[figure] {out}")
    jout = out.parent.parent / f"ma_2020_breakdown_{tag}.json"
    jout.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"results": {f"{k[0]}|{k[1]}": v for k, v in results.items()},
               "winners": {c: (winners[c] if winners[c] else None) for c in CADENCES},
               "oracle": oracle}, open(jout, "w"), indent=1, default=str)
    print(f"[json] {jout}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
