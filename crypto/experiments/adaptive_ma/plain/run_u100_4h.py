"""experiments/adaptive_ma/plain/run_u100_4h.py -- ER-gated fixed-MA (4h) over u100, held-out edge.

Runs the strictly-3-DOF ER-gated fixed-MA breakout (ergated_fixed_ma_4h.py) over every u100 asset on
4h bars and emits the HELD-OUT per-trade NET expectancy + trade count PER ASSET (the brief deliverable).

Pipeline per asset:
  ChimeraLoader.load(sym,'4h') -> build_entry (ER-gate AND breakout AND fast>slow, all past-only) ->
  SetupHarness with ExitPolicy(ATR-trail 3xATR14 + time-stop 42 bars) -> per-trade NET (taker 0.0024,
  sourced from src/strat/fill_model) -> per-window expectancy + count.

HELD-OUT = OOS + UNSEEN (every constant was fixed up-front; these windows chose nothing). UNSEEN
(>= 2025-12-31) is the verdict surface. NET = after-cost per-trade return (SetupHarness subtracts cost).

Honest-null posture (RESEARCHER_REPORT_1): this is the MINIMAL config the 6-cell adaptive map must beat.
A negative / null held-out expectancy here is a VALID, valuable refutation -- it says "do not add map
cells, the timing premise itself fails on held-out".

RWYB:
  python experiments/adaptive_ma/plain/run_u100_4h.py            # full u100
  python experiments/adaptive_ma/plain/run_u100_4h.py --quick    # first 15 assets
  python experiments/adaptive_ma/plain/run_u100_4h.py --firewall # + cost-matched random-entry null (slow)

Writes results_u100_4h.json (+ .quick.json) and prints a per-asset table. No emoji (cp1252). No commit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import ergated_fixed_ma_4h as M  # noqa: E402  (the 3-DOF core: build_entry / run_asset / config)
from pipeline.chimera_loader import ChimeraLoader  # noqa: E402

WINDOWS = ["TRAIN", "VAL", "OOS", "UNSEEN"]
HELD = ["OOS", "UNSEEN"]
MIN_BARS = M.SLOW_LEN + M.BREAKOUT_N + M.ER_WIN + 50  # need warmup + something to trade


def _held_nets(trades: list) -> list:
    return [t["net_pnl"] for t in trades if t["window"] in HELD]


def _window_nets(trades: list, w: str) -> list:
    return [t["net_pnl"] for t in trades if t["window"] == w]


def _expectancy_pct(nets: list):
    a = np.asarray(nets, float)
    return round(float(a.mean() * 100), 4) if a.size else None


def _winrate(nets: list):
    a = np.asarray(nets, float)
    return round(float((a > 0).mean()), 3) if a.size else None


def run(quick: bool, do_firewall: bool) -> dict:
    assert M.TAKER == 0.0024, "brief requires taker cost 0.0024"
    loader = ChimeraLoader()
    syms = loader.universes.list("u100")
    if quick:
        syms = syms[:15]
    print(f"[ergated_fixed_ma_4h u100] assets={len(syms)} cadence=4h taker={M.TAKER}", flush=True)
    print(f"  3 DOF: ER>{M.ER_GATE_THR} | SMA({M.FAST_LEN})/{M.SLOW_LEN} | "
          f"trail={M.ATR_TRAIL_MULT}xATR{M.ATR_WIN} + time<={M.TIME_STOP_BARS}bars(7d)", flush=True)
    print(f"  structural consts: ER_WIN={M.ER_WIN} ATR_WIN={M.ATR_WIN} breakout_N={M.BREAKOUT_N}", flush=True)
    print(f"  entry = ER-gate AND close>prior{M.BREAKOUT_N}-high AND fast>slow (all past-only)", flush=True)

    per_asset = {}
    n_ok = 0
    n_skip = 0
    for k, sym in enumerate(syms, 1):
        try:
            df = M.load_ohlc_4h(loader, sym)
        except Exception as e:  # noqa: BLE001
            n_skip += 1
            print(f"  SKIP {sym}: {type(e).__name__}: {e}", flush=True)
            continue
        if len(df) < MIN_BARS:
            n_skip += 1
            print(f"  SKIP {sym}: too few 4h bars ({len(df)})", flush=True)
            continue
        h = M.run_asset(df)
        res = h.run()
        rec = {"n_bars": len(df), "n_setups": int(M.build_entry(df)[M.ENTRY_COL].sum())}
        for w in WINDOWS:
            nets = _window_nets(res.trades, w)
            rec[w] = {"n": len(nets), "exp_pct": _expectancy_pct(nets), "wr": _winrate(nets)}
        held = _held_nets(res.trades)
        rec["HELD"] = {"n": len(held), "exp_pct": _expectancy_pct(held), "wr": _winrate(held)}
        rec["_held_nets"] = [round(float(x), 6) for x in held]  # for pooling
        rec["_unseen_nets"] = [round(float(x), 6) for x in _window_nets(res.trades, "UNSEEN")]

        if do_firewall:
            try:
                from strat.firewall import random_entry_null
                fw = random_entry_null(h, n_books=200, seed=7, regime_matched=True)
                rec["firewall"] = {"beats_held": bool(fw.get("beats_held")),
                                   "verdict": fw.get("verdict")}
            except Exception as e:  # noqa: BLE001
                rec["firewall"] = {"error": repr(e)}
        per_asset[sym] = rec
        n_ok += 1
        if k % 10 == 0:
            print(f"  [run] {k}/{len(syms)} assets processed", flush=True)

    agg = _aggregate(per_asset, do_firewall)
    out = {
        "config": {
            "cadence": "4h", "taker": M.TAKER, "n_assets_run": n_ok, "n_assets_skipped": n_skip,
            "dof": {"ER_GATE_THR": M.ER_GATE_THR, "MA": [M.FAST_LEN, M.SLOW_LEN, "sma"],
                    "exit": {"atr_trail_mult": M.ATR_TRAIL_MULT, "atr_win": M.ATR_WIN,
                             "time_stop_bars": M.TIME_STOP_BARS}},
            "structural_consts": {"ER_WIN": M.ER_WIN, "ATR_WIN": M.ATR_WIN, "BREAKOUT_N": M.BREAKOUT_N},
            "windows": M.WINDOWS.__dict__, "held_out": HELD,
            "entry": "ER>thr AND close>prior_N_high AND fast>slow (past-only)",
        },
        "aggregate": agg, "per_asset": per_asset,
    }
    _print_table(per_asset, agg, do_firewall)
    outpath = HERE / ("results_u100_4h.quick.json" if quick else "results_u100_4h.json")
    outpath.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"[saved] {outpath}", flush=True)
    _write_markdown(per_asset, agg, do_firewall, quick)
    return out


def _aggregate(per_asset: dict, do_firewall: bool) -> dict:
    syms = list(per_asset.keys())
    # pooled held-out per-trade nets across ALL assets (equal-weight per trade)
    pool_held, pool_unseen = [], []
    for s in syms:
        pool_held += per_asset[s]["_held_nets"]
        pool_unseen += per_asset[s]["_unseen_nets"]
    pool_held = np.asarray(pool_held, float)
    pool_unseen = np.asarray(pool_unseen, float)

    # per-asset held-out expectancy distribution (assets with >=1 held trade)
    asset_exp = [per_asset[s]["HELD"]["exp_pct"] for s in syms if per_asset[s]["HELD"]["exp_pct"] is not None]
    asset_exp = np.asarray(asset_exp, float)

    agg = {
        "pooled_held_trade_exp_pct": (round(float(pool_held.mean() * 100), 4) if pool_held.size else None),
        "pooled_held_median_pct": (round(float(np.median(pool_held) * 100), 4) if pool_held.size else None),
        "pooled_held_winrate": (round(float((pool_held > 0).mean()), 3) if pool_held.size else None),
        "pooled_held_n_trades": int(pool_held.size),
        "pooled_unseen_trade_exp_pct": (round(float(pool_unseen.mean() * 100), 4) if pool_unseen.size else None),
        "pooled_unseen_winrate": (round(float((pool_unseen > 0).mean()), 3) if pool_unseen.size else None),
        "pooled_unseen_n_trades": int(pool_unseen.size),
        "n_assets_held_exp_positive": int((asset_exp > 0).sum()),
        "n_assets_with_held_trades": int(asset_exp.size),
        "median_asset_held_exp_pct": (round(float(np.median(asset_exp)), 4) if asset_exp.size else None),
        "mean_asset_held_exp_pct": (round(float(asset_exp.mean()), 4) if asset_exp.size else None),
    }
    if pool_held.size:  # block-bootstrap-free simple p05 of per-trade nets, plus naive 2-sided sign info
        agg["pooled_held_p05_pct"] = round(float(np.percentile(pool_held, 5) * 100), 4)
        agg["pooled_held_p95_pct"] = round(float(np.percentile(pool_held, 95) * 100), 4)
    if do_firewall:
        fb = [per_asset[s].get("firewall", {}).get("beats_held") for s in syms]
        agg["firewall_n_beat_held"] = int(sum(1 for x in fb if x is True))
        agg["firewall_n_evaluated"] = int(sum(1 for x in fb if x is not None))
    return agg


def _print_table(per_asset: dict, agg: dict, do_firewall: bool):
    # sort by held-out expectancy desc (None last)
    def keyf(s):
        e = per_asset[s]["HELD"]["exp_pct"]
        return (e is None, -(e or 0.0))
    syms = sorted(per_asset.keys(), key=keyf)
    print("\n" + "=" * 92)
    print("ER-GATED FIXED-MA (4h, 3 DOF) -- HELD-OUT (OOS+UNSEEN) per-trade NET expectancy per asset")
    print("=" * 92)
    print(f"{'asset':12} {'HELD n':>7} {'HELD exp%':>10} {'HELD wr':>8} "
          f"{'OOS n':>6} {'OOS exp%':>9} {'UNSEEN n':>9} {'UNSEEN exp%':>12}")
    print("-" * 92)
    for s in syms:
        r = per_asset[s]
        he, oo, un = r["HELD"], r["OOS"], r["UNSEEN"]
        print(f"{s:12} {he['n']:>7} {('%+.3f' % he['exp_pct']) if he['exp_pct'] is not None else '   --':>10} "
              f"{(('%.2f' % he['wr']) if he['wr'] is not None else '--'):>8} "
              f"{oo['n']:>6} {('%+.3f' % oo['exp_pct']) if oo['exp_pct'] is not None else '  --':>9} "
              f"{un['n']:>9} {('%+.3f' % un['exp_pct']) if un['exp_pct'] is not None else '   --':>12}")
    print("=" * 92)
    print(f"POOLED held-out per-trade NET expectancy : "
          f"{agg['pooled_held_trade_exp_pct']}%  (n={agg['pooled_held_n_trades']}, "
          f"winrate={agg['pooled_held_winrate']}, median={agg['pooled_held_median_pct']}%)")
    print(f"POOLED UNSEEN-only per-trade NET expectancy: "
          f"{agg['pooled_unseen_trade_exp_pct']}%  (n={agg['pooled_unseen_n_trades']}, "
          f"winrate={agg['pooled_unseen_winrate']})")
    print(f"ASSETS with POSITIVE held-out expectancy : "
          f"{agg['n_assets_held_exp_positive']}/{agg['n_assets_with_held_trades']} "
          f"(median asset held exp={agg['median_asset_held_exp_pct']}%)")
    if do_firewall and "firewall_n_beat_held" in agg:
        print(f"FIREWALL (beats cost-matched random-entry null on held-out): "
              f"{agg['firewall_n_beat_held']}/{agg['firewall_n_evaluated']} assets")
    print("=" * 92 + "\n", flush=True)


def _write_markdown(per_asset: dict, agg: dict, do_firewall: bool, quick: bool):
    def keyf(s):
        e = per_asset[s]["HELD"]["exp_pct"]
        return (e is None, -(e or 0.0))
    syms = sorted(per_asset.keys(), key=keyf)
    lines = []
    lines.append(f"# ER-gated fixed-MA (4h, 3 DOF) -- held-out results{' (QUICK)' if quick else ''}")
    lines.append("")
    lines.append(f"- **3 DOF**: ER hard-gate > {M.ER_GATE_THR} | fixed SMA({M.FAST_LEN})/{M.SLOW_LEN} | "
                 f"ATR-trail {M.ATR_TRAIL_MULT}xATR{M.ATR_WIN} + time-stop {M.TIME_STOP_BARS} bars (7d)")
    lines.append(f"- **Entry**: `ER>{M.ER_GATE_THR} AND close>prior_{M.BREAKOUT_N}_bar_high AND fast>slow` "
                 f"(all past-only; SetupHarness fills at next-bar open)")
    lines.append(f"- **Cost**: taker {M.TAKER} round-trip (src/strat/fill_model). **Held-out = OOS+UNSEEN.** "
                 f"NET = after-cost per-trade return.")
    lines.append(f"- **Pooled held-out per-trade NET expectancy**: **{agg['pooled_held_trade_exp_pct']}%** "
                 f"(n={agg['pooled_held_n_trades']}, winrate={agg['pooled_held_winrate']}, "
                 f"median={agg['pooled_held_median_pct']}%, p05={agg.get('pooled_held_p05_pct')}%, "
                 f"p95={agg.get('pooled_held_p95_pct')}%)")
    lines.append(f"- **Pooled UNSEEN-only per-trade NET expectancy**: **{agg['pooled_unseen_trade_exp_pct']}%** "
                 f"(n={agg['pooled_unseen_n_trades']}, winrate={agg['pooled_unseen_winrate']})")
    lines.append(f"- **Assets with positive held-out expectancy**: "
                 f"{agg['n_assets_held_exp_positive']}/{agg['n_assets_with_held_trades']} "
                 f"(median asset held exp = {agg['median_asset_held_exp_pct']}%)")
    if do_firewall and "firewall_n_beat_held" in agg:
        lines.append(f"- **Firewall** (beats cost-matched random-entry null on held-out): "
                     f"{agg['firewall_n_beat_held']}/{agg['firewall_n_evaluated']} assets")
    lines.append("")
    lines.append("| asset | HELD n | HELD exp% | HELD wr | OOS n | OOS exp% | UNSEEN n | UNSEEN exp% |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in syms:
        r = per_asset[s]
        he, oo, un = r["HELD"], r["OOS"], r["UNSEEN"]
        def f(x):
            return f"{x:+.3f}" if x is not None else "--"
        lines.append(f"| {s} | {he['n']} | {f(he['exp_pct'])} | "
                     f"{he['wr'] if he['wr'] is not None else '--'} | "
                     f"{oo['n']} | {f(oo['exp_pct'])} | {un['n']} | {f(un['exp_pct'])} |")
    mdpath = HERE / ("RESULTS_4h.quick.md" if quick else "RESULTS_4h.md")
    mdpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[saved] {mdpath}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="first 15 assets only")
    ap.add_argument("--firewall", action="store_true", help="run cost-matched random-entry null per asset (slow)")
    args = ap.parse_args()
    run(quick=args.quick, do_firewall=args.firewall)
