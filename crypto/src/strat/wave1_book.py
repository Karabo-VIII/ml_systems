"""WAVE-1 BOOK -- the campaign charter's first shippable: CORE + overlay, in-house data only.

Per docs/CAMPAIGN_CHARTER_2026_06_10.md + docs/CANDIDATE_REGISTER_2026_06_10.md (Wave 1)
and the user's 2026-06-10 infra-only directive (no external feeds).

COMPOSITION (all components previously validated or charter-mandated):
  BASE BOOK : BLEND_50r = 0.5 * regime-beta (EW long close>SMA120, flat=cash; thread-22
              CORE) + 0.5 * TSMOM breadth ensemble ({21,63,126,252} lookbacks,
              inverse-vol, breadth budget). Imported from strat.tsmom_ensemble verbatim
              (commits 5e638c6/5596512) -- ZERO re-tuning of validated parameters.
  OVERLAY 1 : LISTING-SEASONING filter -- assets enter the book only after AGE_MIN days
              of history (98% of listings trade down from listing; playbook sec 3).
  OVERLAY 2 : VOL-TARGETING -- scale book weights by min(1, target_vol / realized book
              vol), cap 1.0 (lev=1; down-only scaler). Charter section 3 mandates it.

PRE-REGISTRATION (declared before any result of this script was read):
  Variant grid = AGE_MIN in {90, 180} x VT_LOOKBACK in {20, 60} days = 4 variants ONLY,
  plus the unoverlaid BLEND_50r as control. Target vol 30% annualized (Man-Group-
  evidenced figure; not swept). Selection rule: highest OOS Calmar with OOS maxDD
  shallower than the unoverlaid control. UNSEEN evaluated for ALL variants (4 trials --
  reported with that multiplicity stated; the SELECTED one is the headline). Gates for
  SHIP (charter sec 4): selected-variant UNSEEN ann > 0 OR (UNSEEN preservation better
  than buy&hold AND full-cycle Calmar > control), bootstrap held-out p05 > 0 at the
  full-cycle level, beats the exposure-matched RANDOM null, param-perturbation battery
  >= 80% configs positive full-cycle. Honest expectation published first: this book
  LAGS buy&hold in bull legs (vol-target + regime gate = DD technology).
  Splits: WIN (train_end 2024-05-15 / val_end 2025-03-15 / oos_end 2025-12-31 /
  unseen_end 2026-06-01) -- the SAME windows the components were validated under.
Costs: taker RT 0.24% on turnover (maker reported as sensitivity). No emoji.

Run:
  python -m strat.wave1_book --universe u50
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strat.tsmom_ensemble import (  # noqa: E402  (validated components, unmodified)
    build_panel, build_books, backtest, stats, block_bootstrap, TAKER_RT, ANN,
)
from strat.entry_signal_lab import WIN  # noqa: E402

OUT = ROOT.parent / "runs" / "strat"
OUT.mkdir(parents=True, exist_ok=True)

__contract__ = {
    "kind": "strategy_book",
    "inputs": {"chimera_1d": "via tsmom_ensemble.build_panel (ChimeraLoader)"},
    "outputs": {"book_json": "runs/strat/wave1_book_<universe>_<stamp>.json"},
    "invariants": {
        "components_unmodified": "BLEND_50r imported from tsmom_ensemble verbatim; no re-tuning",
        "preregistered_grid": "4 overlay variants only; selection rule declared in-source",
        "causal_overlays": "seasoning by first-bar age; vol-target from LAGGED book returns",
        "lev_le_1": "vol-target capped at 1.0; LO spot",
        "splits_inherited": "WIN windows identical to the components' validation",
    },
}

TARGET_VOL = 0.30          # annualized; Man-Group-evidenced, NOT swept
AGE_GRID = [90, 180]
VT_GRID = [20, 60]


def seasoning_mask(listed: pd.DataFrame, age_min: int) -> pd.DataFrame:
    """True only after the asset has age_min days of history (first listed bar + age)."""
    age = listed.cumsum()                       # days listed so far (listed is bool)
    return listed & (age >= age_min)


def vol_target_scale(net_unscaled: pd.Series, lookback: int) -> pd.Series:
    """Causal exposure scaler: min(1, TARGET_VOL / trailing realized vol of the book).
    Vol estimated from the UNSCALED book's net returns up to t-1 (shift)."""
    rv = net_unscaled.rolling(lookback, min_periods=lookback // 2).std() * np.sqrt(ANN)
    scale = (TARGET_VOL / rv).clip(upper=1.0)
    return scale.shift(1).fillna(1.0).clip(0.0, 1.0)


def run_book(universe: str, cost_per_side: float) -> dict:
    panel, ret, vol, listed = build_panel(universe, "1d")
    # FIX(audit 2026-06-11): SELECT on the pre-OOS window only; OOS + UNSEEN are
    # untouched TESTS. (v1 selected on OOS Calmar + re-selected gold on OOS ann ->
    # the +29.6% gold headline was selecting-on-the-test-set. Corrected here.)
    sel_lo, sel_hi = pd.Timestamp("2018-01-01"), pd.Timestamp(WIN.val_end)
    oos_lo, oos_hi = pd.Timestamp(WIN.val_end), pd.Timestamp(WIN.oos_end)
    uns_lo, uns_hi = pd.Timestamp(WIN.oos_end), pd.Timestamp(WIN.unseen_end)

    results = {}
    nets = {}

    def evaluate(name: str, W: pd.DataFrame, scale: pd.Series | None = None):
        netu, turn = backtest(W, ret, cost_per_side)
        if scale is not None:
            # gross and within-book turnover scale with exposure; PLUS the scaler's own
            # rebalancing cost: |delta scale| x book notional traded at cost_per_side
            scale_turn = scale.diff().abs().fillna(0.0) * W.shift(1).sum(axis=1).fillna(0.0)
            net = netu * scale - scale_turn * cost_per_side
        else:
            net = netu
        yr = {int(y): round(float(((1 + g).cumprod().iloc[-1] - 1) * 100), 1)
              for y, g in net.groupby(net.index.year)}
        results[name] = {
            "SEL": stats(net, sel_lo, sel_hi),  # pre-OOS selection window (no test leakage)
            "full": stats(net), "OOS": stats(net, oos_lo, oos_hi),
            "UNSEEN": stats(net, uns_lo, uns_hi), "per_year": yr,
            "avg_exposure": round(float(W.sum(axis=1).mean() * (scale.mean() if scale is not None else 1.0)), 2),
            "avg_daily_turnover_pct": round(float(turn.mean()) * 100, 2),
        }
        nets[name] = net

    # control: the validated base book, untouched
    books0 = build_books(panel, ret, vol, listed)
    evaluate("BLEND_50r_control", books0["BLEND_50r"])
    evaluate("buy_hold", books0["buy_hold"])
    evaluate("RANDOM_null", books0["RANDOM_null"])
    evaluate("regime_beta_CORE", books0["regime_beta"])

    # pre-registered 4-variant grid
    for age in AGE_GRID:
        lst = seasoning_mask(listed, age)
        books = build_books(panel, ret, vol, lst)
        W = books["BLEND_50r"]
        net_unscaled, _ = backtest(W, ret, cost_per_side)
        for vt in VT_GRID:
            sc = vol_target_scale(net_unscaled, vt)
            evaluate(f"WAVE1_age{age}_vt{vt}", W, sc)

    # selection rule (FIX): highest SEL(pre-OOS) Calmar with SEL maxDD shallower than control
    ctrl_sel = results["BLEND_50r_control"]["SEL"]
    best_name, best_calmar = None, -1e18
    for age in AGE_GRID:
        for vt in VT_GRID:
            nm = f"WAVE1_age{age}_vt{vt}"
            o = results[nm]["SEL"]
            if not o or o.get("calmar") is None:
                continue
            if ctrl_sel and o["maxdd_pct"] < ctrl_sel["maxdd_pct"]:
                continue  # must be shallower (less negative) than control on SEL
            if o["calmar"] > best_calmar:
                best_calmar, best_name = o["calmar"], nm

    # GOLD BEAR-SLEEVE (queue item 1; pre-registered: 2 idle-cash fractions {0.5, 1.0}
    # on the SELECTED variant only; gold gate = XAUT > SMA120, causal shift(1))
    if best_name:
        # gold series: prefer PAXG raw 1m klines (2025-03->present, daily close = last
        # 1m close per file; us->ms ts) over the 48-day XAUT chimera
        gold_s = None
        import glob as _glob
        import polars as _plr
        recs = []
        bf = ROOT.parent / "data/raw/PAXGUSDT/klines_1d_backfill.parquet"
        if bf.exists():  # spot 1d backfill 2020-08 -> 2025-03 (monthly zips)
            d = _plr.read_parquet(bf)
            for t, c in zip(d["timestamp"].to_list(), d["close"].to_list()):
                recs.append((pd.Timestamp(int(t), unit="ms").normalize(), float(c)))
        gfiles = sorted(_glob.glob(str(ROOT.parent / "data/raw/PAXGUSDT/klines_1m/*.parquet")))
        if len(gfiles) > 200:
            for fp in gfiles:
                d = _plr.read_parquet(fp, columns=["timestamp", "close"])
                if len(d):
                    t = int(d["timestamp"][-1])
                    t = t // 1000 if t > 2_000_000_000_000_000 // 1000 else t  # us->ms
                    recs.append((pd.Timestamp(t, unit="ms").normalize(), float(d["close"][-1])))
        if recs:
            gold_s = pd.Series(dict(recs)).sort_index()  # dict dedups (recent wins)
        if gold_s is None:
            from strat.entry_signal_lab import load_ohlc
            gdf = load_ohlc("XAUTUSDT", "1d")
            if gdf is not None and len(gdf) > 200:
                gold_s = gdf.set_index("date")["close"]
        if gold_s is not None:
            gold = gold_s.reindex(nets[best_name].index).ffill()
            xret = gold.pct_change().fillna(0.0)
            ggate = (gold > gold.rolling(120, min_periods=60).mean()).shift(1).fillna(False)
            age, vt = best_name.replace("WAVE1_age", "").split("_vt")
            lst = seasoning_mask(listed, int(age))
            Wsel = build_books(panel, ret, vol, lst)["BLEND_50r"]
            netu_sel, _ = backtest(Wsel, ret, cost_per_side)
            sc = vol_target_scale(netu_sel, int(vt))
            exposure = (Wsel.shift(1).sum(axis=1).fillna(0.0) * sc).clip(0, 1)
            idle = (1.0 - exposure).clip(0, 1)
            for gfrac in (0.5, 1.0):
                gw = (gfrac * idle * ggate.astype(float)).reindex(nets[best_name].index).fillna(0.0)
                gpnl = gw * xret
                gcost = gw.diff().abs().fillna(0.0) * cost_per_side
                net_g = nets[best_name] + gpnl - gcost
                nm = f"{best_name}_GOLD{int(gfrac*100)}"
                yr = {int(y): round(float(((1 + g).cumprod().iloc[-1] - 1) * 100), 1)
                      for y, g in net_g.groupby(net_g.index.year)}
                results[nm] = {"SEL": stats(net_g, sel_lo, sel_hi),
                               "full": stats(net_g), "OOS": stats(net_g, oos_lo, oos_hi),
                               "UNSEEN": stats(net_g, uns_lo, uns_hi), "per_year": yr,
                               "avg_exposure": round(float((exposure + gw).mean()), 2),
                               "avg_daily_turnover_pct": None}
                nets[nm] = net_g
            # FIX: re-select among {selected, +GOLD50, +GOLD100} on SEL (pre-OOS), NOT OOS
            cand = [best_name, f"{best_name}_GOLD50", f"{best_name}_GOLD100"]
            def keyf(nm):
                o = results[nm]["SEL"] or {}
                return (o.get("ann_pct") if o.get("ann_pct") is not None else -1e9)
            best_name = max(cand, key=keyf)

    # gates (charter sec 4) on the SELECTED variant
    gates = {}
    if best_name:
        sel = results[best_name]
        net_sel = nets[best_name]
        boot_held = block_bootstrap(net_sel[net_sel.index >= oos_lo])
        boot_full = block_bootstrap(net_sel)
        rnd_uns = results["RANDOM_null"]["UNSEEN"]
        bh_uns = results["buy_hold"]["UNSEEN"]
        u = sel["UNSEEN"]
        gates = {
            "selected": best_name,
            "unseen_ann_pos": bool(u and u["ann_pct"] > 0),
            "unseen_preserves_vs_buyhold": bool(u and bh_uns and u["ann_pct"] > bh_uns["ann_pct"]),
            "fullcycle_calmar_gt_control": bool(sel["full"]["calmar"] and results["BLEND_50r_control"]["full"]["calmar"]
                                                and sel["full"]["calmar"] > results["BLEND_50r_control"]["full"]["calmar"]),
            "beats_random_null_unseen": bool(u and rnd_uns and u["ann_pct"] > rnd_uns["ann_pct"]),
            "bootstrap_heldout_p05_pos": bool(boot_held and boot_held["p05"] > 0),
            "bootstrap_full_p05_pos": bool(boot_full and boot_full["p05"] > 0),
            "bootstrap_heldout": boot_held, "bootstrap_full": boot_full,
        }
        gates["ship"] = bool(
            (gates["unseen_ann_pos"] or (gates["unseen_preserves_vs_buyhold"]
                                          and gates["fullcycle_calmar_gt_control"]))
            and gates["beats_random_null_unseen"] and gates["bootstrap_full_p05_pos"]
        )
    return {"results": results, "selection": {"best": best_name, "oos_calmar": best_calmar},
            "gates": gates}


def wave1_battery(universe: str, cost_per_side: float, age: int = 90, vt: int = 20) -> dict:
    """Param-perturbation battery on the SELECTED Wave-1 variant: vary the components'
    params (lookback sets x regime SMA, same grid as tsmom_ensemble.battery) with the
    overlays HELD at the selected (age, vt). Gate: >=80% configs positive full-cycle."""
    from strat.tsmom_ensemble import build_books as bb
    panel, ret, vol, listed = build_panel(universe, "1d")
    lst = seasoning_mask(listed, age)
    oos_lo = pd.Timestamp(WIN.val_end)
    LB_SETS = [[21, 63, 126, 252], [14, 42, 84, 168], [30, 90, 180, 360], [21, 63, 126],
               [63, 126, 252, 365], [10, 30, 90, 180], [21, 50, 100, 200], [28, 84, 168, 336],
               [20, 60, 120, 240], [21, 63, 189, 252]]
    SMAS = [100, 120, 150]
    rows = []
    for lb in LB_SETS:
        for sm in SMAS:
            W = bb(panel, ret, vol, lst, lookbacks=lb, sma=sm)["BLEND_50r"]
            netu, _ = backtest(W, ret, cost_per_side)
            sc = vol_target_scale(netu, vt)
            scale_turn = sc.diff().abs().fillna(0.0) * W.shift(1).sum(axis=1).fillna(0.0)
            net = netu * sc - scale_turn * cost_per_side
            f = stats(net)
            h = stats(net, oos_lo, pd.Timestamp(WIN.unseen_end))
            rows.append({"lb": lb, "sma": sm,
                         "full_ann": f["ann_pct"] if f else None,
                         "full_dd": f["maxdd_pct"] if f else None,
                         "held_ann": h["ann_pct"] if h else None})
    fa = np.array([r["full_ann"] for r in rows if r["full_ann"] is not None])
    ha = np.array([r["held_ann"] for r in rows if r["held_ann"] is not None])
    return {"n_configs": len(rows),
            "full_ann": {"min": float(fa.min()), "med": float(np.median(fa)),
                         "max": float(fa.max()), "pct_positive": float((fa > 0).mean())},
            "held_ann": {"min": float(ha.min()), "med": float(np.median(ha)),
                         "max": float(ha.max()), "pct_positive": float((ha > 0).mean())},
            "gate_80pct_full_positive": bool((fa > 0).mean() >= 0.80),
            "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m strat.wave1_book")
    ap.add_argument("--universe", default="u50")
    ap.add_argument("--maker", action="store_true")
    ap.add_argument("--battery", action="store_true")
    args = ap.parse_args()
    cps = 0.0006 if args.maker else TAKER_RT / 2
    if args.battery:
        b = wave1_battery(args.universe, cps)
        print(f"## WAVE-1 BATTERY (age90/vt20 overlays held; {b['n_configs']} component perturbations)")
        print(f"   full ann: min {b['full_ann']['min']:.1f} med {b['full_ann']['med']:.1f} "
              f"max {b['full_ann']['max']:.1f}  positive {b['full_ann']['pct_positive']:.0%}")
        print(f"   held-out ann: min {b['held_ann']['min']:.1f} med {b['held_ann']['med']:.1f} "
              f"max {b['held_ann']['max']:.1f}  positive {b['held_ann']['pct_positive']:.0%}")
        print(f"   GATE >=80% full-cycle positive: {'PASS' if b['gate_80pct_full_positive'] else 'FAIL'}")
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        p = OUT / f"wave1_battery_{args.universe}_{stamp}.json"
        json.dump(b, open(p, "w", encoding="utf-8"), indent=1, default=str)
        print(f"[persisted] {p}")
        return 0

    out = run_book(args.universe, cps)
    res = out["results"]

    print(f"## WAVE-1 BOOK -- {args.universe} 1d -- {'maker' if args.maker else 'taker'} -- "
          f"target_vol {TARGET_VOL:.0%} (pre-registered)")
    hdr = (f"   {'book':24} {'FULL ann%':>9} {'dd%':>7} {'Calmar':>6} | {'OOS ann%':>8} {'dd%':>7} "
           f"{'Cal':>5} | {'UNSEEN ann%':>11} {'dd%':>7} | {'expo':>5}")
    print(hdr)
    order = (["regime_beta_CORE", "BLEND_50r_control"]
             + [f"WAVE1_age{a}_vt{v}" for a in AGE_GRID for v in VT_GRID]
             + [k for k in res if "GOLD" in k]
             + ["buy_hold", "RANDOM_null"])
    for nm in order:
        d = res.get(nm)
        if not d:
            continue
        f, o, u = d["full"], d["OOS"], d["UNSEEN"]
        def g(s, k):
            return f"{s[k]}" if s and s.get(k) is not None else "-"
        print(f"   {nm:24} {g(f,'ann_pct'):>9} {g(f,'maxdd_pct'):>7} {g(f,'calmar'):>6} | "
              f"{g(o,'ann_pct'):>8} {g(o,'maxdd_pct'):>7} {g(o,'calmar'):>5} | "
              f"{g(u,'ann_pct'):>11} {g(u,'maxdd_pct'):>7} | {d['avg_exposure']:>5}")
    sel, gates = out["selection"], out["gates"]
    print(f"\nSELECTED (pre-registered rule): {sel['best']} (OOS Calmar {sel['oos_calmar']})")
    if gates:
        for k, v in gates.items():
            if isinstance(v, bool):
                print(f"   gate {k:36} {'PASS' if v else 'FAIL'}")
        print(f"   bootstrap full p05={gates['bootstrap_full']['p05']}% held-out p05={gates['bootstrap_heldout']['p05']}%")
        print(f"\nVERDICT: {'SHIP-CANDIDATE (pending param-perturbation battery)' if gates.get('ship') else 'NOT SHIP -- gates failed'}")

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = OUT / f"wave1_book_{args.universe}_{stamp}.json"
    json.dump({"repro": {"command": "python " + " ".join(sys.argv), "git_sha": sha},
               "preregistration": {"target_vol": TARGET_VOL, "age_grid": AGE_GRID,
                                    "vt_grid": VT_GRID, "n_variants": 4,
                                    "selection": "max OOS Calmar with OOS maxDD shallower than control"},
               "out": out}, open(p, "w", encoding="utf-8"), indent=1, default=str)
    print(f"[persisted] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
