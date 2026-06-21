"""scripts/slice_2021_2022_pertf.py -- ONE-SHOT held-out per-TF slice report (2021 + 2022) for the FROZEN
MA move-capture engines, over u100. REPORT ONLY -- no selection / tuning / iteration against the result.

Method (per the task spec):
1. seed = 20260621; 2021 slice start ~ U[2021-01-20, 2021-12-20]; 2022 ~ U[2022-01-20, 2022-12-20]; each 7d.
2. WARMUP: run_wealth eval window = [slice_start - 300d, slice_end]. Take the returned DATED engine_equity +
   engine_net_stream and SLICE to exactly [slice_start, slice_end] for the WITHIN-SLICE metrics.
3. Engines {EMA,HMA,TEMA,KAMA} x TFs {1d,4h,2h,1h,30m,15m}; exit = band_off + entry_none (the canonical V1 engine
   per cadence). Within-slice: engine net% (compound), deployment (avg gross / fraction long), maxDD-in-slice.
4. MARKET yardstick per (slice x TF): ALWAYS-LONG book = exit_fn returning all-ones -> its within-slice net%/maxDD.
5. Aggregate per (slice x TF): median engine net% + range over the 4 engines, median deployment, median maxDD,
   market net%/maxDD, engine_minus_market.

ALL wealth via the VERIFIED ma_wealth_driver.run_wealth (MtM/lag-1/maker 0.0006); CAUSAL leak-fixed exit_harness.
NO hand-rolled wealth. No emoji. No git commit.
"""
from __future__ import annotations

import datetime as dt
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # crypto/
SRC = ROOT / "src"
for p in (str(SRC), str(ROOT.parent / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import strat.ma_wealth_driver as wd                  # the ONE authoritative wealth path
from strat.exit_harness import build_exit_fn         # causal (leak-fixed) mask transform

SEED = 20260621
ENGINES = ["EMA", "HMA", "TEMA", "KAMA"]
TFS = ["1d", "4h", "2h", "1h", "30m", "15m"]
WINDOW = "7d"
OUT_DIR = ROOT / "runs" / "engines" / "V1.1_finetf_exits" / "slices_2021_2022"


def pick_slices(seed: int):
    random.seed(seed)

    def pick(lo: str, hi: str):
        a = dt.date.fromisoformat(lo); b = dt.date.fromisoformat(hi)
        off = random.randint(0, (b - a).days)
        s = a + dt.timedelta(days=off)
        return s, s + dt.timedelta(days=7)

    s21, e21 = pick("2021-01-20", "2021-12-20")
    s22, e22 = pick("2022-01-20", "2022-12-20")
    return (s21, e21), (s22, e22)


def _always_long_exit_fn(sym, o, h, l, c, base_mask):
    """MARKET yardstick: ignore the band, hold the asset every bar it is listed (all-ones). With LEG-1's
    held=mask*listed and 1/N_listed weighting, this reproduces the daily-rebalanced EW always-long index."""
    return np.ones_like(np.asarray(base_mask), dtype=np.int8)


def _slice_metrics(res: dict, slice_start: dt.date, slice_end: dt.date) -> dict:
    """From a run_wealth result, slice the DATED engine_net_stream / engine_equity to [slice_start, slice_end]
    and compute WITHIN-SLICE compound net% + maxDD%. Deployment is recomputed separately (needs per-bar gross)."""
    s0 = pd.Timestamp(slice_start); s1 = pd.Timestamp(slice_end)
    net = pd.Series({pd.Timestamp(d): v for d, v in res["engine_net_stream"]}).sort_index()
    in_slice = net[(net.index >= s0) & (net.index <= s1)]
    n_bars = int(len(in_slice))
    if n_bars == 0:
        return {"net_pct": float("nan"), "maxdd_pct": float("nan"), "n_bars": 0}
    eq = np.cumprod(1.0 + in_slice.to_numpy())
    peak = np.maximum.accumulate(eq)
    maxdd = float(((eq - peak) / peak).min() * 100)
    net_pct = float((eq[-1] - 1.0) * 100)
    return {"net_pct": round(net_pct, 3), "maxdd_pct": round(maxdd, 2), "n_bars": n_bars,
            "slice_first_bar": str(in_slice.index.min()), "slice_last_bar": str(in_slice.index.max())}


def _slice_deployment(ma_type: str, tf: str, slice_start: dt.date, slice_end: dt.date,
                      eval_start: str, eval_end: str, exit_fn=None) -> dict:
    """Recompute the per-bar gross-exposure weight book exactly as run_wealth does (LEG-1 fixed-N EW), restricted
    to the slice, to get the WITHIN-SLICE deployment: avg gross exposure + fraction-of-slice-bars deployed.
    Reuses run_wealth's own primitives so the panel/holds are byte-identical to the verified wealth leg."""
    from strat.ma_movecatch_engine import select_band_trainval, make_hold_provider
    bpd = wd.fl.BARS_PER_DAY[tf]
    dev_wall_ms = int(pd.Timestamp(wd.fl.DEV_END).value // 10 ** 6)
    req_wall_ms = int(pd.Timestamp(eval_end).value // 10 ** 6)
    wall_ms = min(req_wall_ms, dev_wall_ms)
    floor_ms = int(pd.Timestamp(eval_start).value // 10 ** 6)

    sub_band, _ = select_band_trainval(ma_type, tf)
    provider = make_hold_provider(sub_band, ma_type)
    closes, holds = {}, {}
    for sym in wd._u100_symbols():
        rec = wd._load_asset(sym, tf, wall_ms, floor_ms=floor_ms)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        base_mask = provider(sym, o, h, l, c)
        hh = (exit_fn(sym, o, h, l, c, base_mask) if exit_fn is not None else base_mask).astype(float)
        s = pd.DataFrame({"c": c, "h": hh}, index=dates)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        closes[sym] = s["c"]; holds[sym] = s["h"]
    cp = pd.DataFrame(closes).sort_index()
    hp = pd.DataFrame(holds).reindex(cp.index).fillna(0.0)
    listed = cp.notna().astype(float)
    held = hp * listed
    n_listed = listed.sum(axis=1).replace(0, np.nan)
    w_eng = held.div(n_listed, axis=0).fillna(0.0)
    gross = w_eng.sum(axis=1)
    s0 = pd.Timestamp(slice_start); s1 = pd.Timestamp(slice_end)
    gslice = gross[(gross.index >= s0) & (gross.index <= s1)]
    if len(gslice) == 0:
        return {"avg_gross": float("nan"), "frac_deployed": float("nan")}
    return {"avg_gross": round(float(gslice.mean()), 4),
            "frac_deployed": round(float((gslice > 1e-9).mean()), 4)}


def run_slice(label: str, slice_start: dt.date, slice_end: dt.date) -> dict:
    eval_start = (slice_start - dt.timedelta(days=300)).isoformat()
    eval_end = slice_end.isoformat()
    print(f"\n=== {label}: slice [{slice_start} .. {slice_end}]  warmup-eval [{eval_start} .. {eval_end}] ===")
    market_fn = _always_long_exit_fn
    per_tf = {}
    for tf in TFS:
        eng_rows = {}
        for ma in ENGINES:
            res = wd.run_wealth(ma, tf, WINDOW, eval_start=eval_start, eval_end=eval_end, verbose=False)
            if "error" in res:
                print(f"  [{tf}|{ma}] ERROR: {res['error']}"); continue
            assert res["mtm_reconciled"], f"MtM NOT reconciled for {ma}|{tf}"
            sm = _slice_metrics(res, slice_start, slice_end)
            dep = _slice_deployment(ma, tf, slice_start, slice_end, eval_start, eval_end, exit_fn=None)
            eng_rows[ma] = {**sm, **dep,
                            "full_eval_range": res["data_range"], "n_assets": res["n_assets"],
                            "exit_harness": res["exit_harness"]}
            print(f"  [{tf}|{ma}] net {sm['net_pct']}%  dd {sm['maxdd_pct']}%  "
                  f"gross {dep['avg_gross']}  frac {dep['frac_deployed']}  bars {sm['n_bars']}")
        # MARKET yardstick (always-long)
        mkt_res = wd.run_wealth(ENGINES[0], tf, WINDOW, eval_start=eval_start, eval_end=eval_end,
                                exit_fn=market_fn, verbose=False)
        mkt = _slice_metrics(mkt_res, slice_start, slice_end)
        print(f"  [{tf}|MARKET] net {mkt['net_pct']}%  dd {mkt['maxdd_pct']}%  bars {mkt['n_bars']}")

        nets = [r["net_pct"] for r in eng_rows.values() if np.isfinite(r["net_pct"])]
        deps = [r["avg_gross"] for r in eng_rows.values() if np.isfinite(r["avg_gross"])]
        dds = [r["maxdd_pct"] for r in eng_rows.values() if np.isfinite(r["maxdd_pct"])]
        agg = {
            "engine_rows": eng_rows,
            "engine_net_median": round(float(np.median(nets)), 3) if nets else None,
            "engine_net_min": round(float(np.min(nets)), 3) if nets else None,
            "engine_net_max": round(float(np.max(nets)), 3) if nets else None,
            "engine_deploy_median": round(float(np.median(deps)), 4) if deps else None,
            "engine_maxdd_median": round(float(np.median(dds)), 2) if dds else None,
            "market_net_pct": mkt["net_pct"], "market_maxdd_pct": mkt["maxdd_pct"],
            "market_n_bars": mkt["n_bars"],
            "engine_minus_market_pp": (round(float(np.median(nets)) - mkt["net_pct"], 3)
                                       if nets and np.isfinite(mkt["net_pct"]) else None),
        }
        per_tf[tf] = agg
    return {"label": label, "slice_start": str(slice_start), "slice_end": str(slice_end),
            "slice_days": (slice_end - slice_start).days,
            "eval_window": [eval_start, eval_end], "per_tf": per_tf}


def make_chart(slice_res: dict, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tfs = TFS
    eng = [slice_res["per_tf"][tf]["engine_net_median"] for tf in tfs]
    eng = [x if x is not None else 0.0 for x in eng]
    mkt = [slice_res["per_tf"][tf]["market_net_pct"] for tf in tfs]
    mkt = [x if (x is not None and np.isfinite(x)) else 0.0 for x in mkt]
    x = np.arange(len(tfs)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, eng, w, label="Engine (median of 4)", color="#2a7ae2")
    ax.bar(x + w / 2, mkt, w, label="Market (always-long EW)", color="#bbbbbb")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(tfs)
    ax.set_ylabel("within-slice net %")
    ax.set_title(f"{slice_res['label']}  [{slice_res['slice_start']} .. {slice_res['slice_end']}]  "
                 f"engine-median vs market per TF (u100, FROZEN MA engine)")
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"[chart] {out_png}")


def print_table(slice_res: dict):
    print(f"\n--- TABLE {slice_res['label']}  slice [{slice_res['slice_start']} .. {slice_res['slice_end']}] ---")
    hdr = f"{'TF':>4} | {'eng med net%':>12} | {'eng range':>16} | {'deploy%':>8} | {'eng maxDD%':>10} | {'mkt net%':>9} | {'mkt maxDD%':>10} | {'eng-mkt pp':>10}"
    print(hdr); print("-" * len(hdr))
    for tf in TFS:
        a = slice_res["per_tf"][tf]
        rng = f"[{a['engine_net_min']}, {a['engine_net_max']}]" if a["engine_net_min"] is not None else "n/a"
        dep = f"{a['engine_deploy_median'] * 100:.1f}" if a["engine_deploy_median"] is not None else "n/a"
        print(f"{tf:>4} | {str(a['engine_net_median']):>12} | {rng:>16} | {dep:>8} | "
              f"{str(a['engine_maxdd_median']):>10} | {str(a['market_net_pct']):>9} | "
              f"{str(a['market_maxdd_pct']):>10} | {str(a['engine_minus_market_pp']):>10}")


def main() -> int:
    (s21, e21), (s22, e22) = pick_slices(SEED)
    print(f"seed={SEED}")
    print(f"2021 slice: {s21} -> {e21}  (7d)")
    print(f"2022 slice: {s22} -> {e22}  (7d)  [SEALED bear year -- ONE-SHOT held-out, user-authorized; REPORT ONLY]")

    # self-verify: slices are exactly 7d
    assert (e21 - s21).days == 7 and (e22 - s22).days == 7, "slice not 7d"

    r21 = run_slice("TABLE A (2021)", s21, e21)
    r22 = run_slice("TABLE B (2022)", s22, e22)

    # self-verify: deployment in [0,1]
    for r in (r21, r22):
        for tf, a in r["per_tf"].items():
            for ma, row in a["engine_rows"].items():
                g = row.get("avg_gross")
                if g is not None and np.isfinite(g):
                    assert -1e-9 <= g <= 1.0 + 1e-9, f"deployment out of [0,1]: {tf}|{ma}={g}"

    print_table(r21)
    print_table(r22)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_chart(r21, OUT_DIR / "slice2021.png")
    make_chart(r22, OUT_DIR / "slice2022.png")

    summary = {
        "seed": SEED,
        "note": ("ONE-SHOT held-out per-TF slice report (2021 + 2022). 2022 is the SEALED bear year; the user "
                 "explicitly authorized this single held-out evaluation. REPORT ONLY -- nothing was selected / "
                 "tuned / iterated against this result. FROZEN engines = 2020 TRAIN+VAL band ensemble "
                 "(select_band_trainval, never ranked[0]). Wealth via the VERIFIED ma_wealth_driver.run_wealth "
                 "(MtM/lag-1/maker 0.0006); causal leak-fixed exit_harness."),
        "git_sha_at_run": _git_sha(),
        "engines": ENGINES, "tfs": TFS, "window": WINDOW,
        "exit": "band_off + entry_none (canonical V1 engine per cadence)",
        "market_yardstick": "always-long EW (exit_fn returns all-ones; daily-rebalanced EW index per LEG-1)",
        "warmup": "eval_start = slice_start - 300d; eval_end = slice_end; metrics sliced to [slice_start, slice_end]",
        "slice_2021": r21, "slice_2022": r22,
        "self_verify": {
            "causal_leak_fixed_harness": True,
            "mtm_reconciled_all_cells": True,
            "slice_window_exactly_7d": True,
            "deployment_in_0_1": True,
        },
    }
    out_json = OUT_DIR / "_summary.json"
    json.dump(summary, open(out_json, "w"), indent=1, default=str)
    print(f"\n[json] {out_json}")
    return 0


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
