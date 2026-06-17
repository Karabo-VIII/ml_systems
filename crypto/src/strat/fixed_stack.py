"""src/strat/fixed_stack.py -- the ISOLATED keeper stack across ALL cadences (4h/1h/30m/15m).

WHY (user /orc 2026-06-12, continuation of fixed_approach.py): the cumulative ladder showed L1_FIXED +
L3_TRAIL(10%) are the keepers, min-hold is a no-op at coarse cadence, and the SMA200 regime gate is
HARMFUL. This isolates the keepers WITHOUT the harmful regime confound and EXTENDS to the FINE cadences
(15m/30m) where min-hold + maker actually pay (per the building block: 15m taker -2.3% -> maker +5.4%,
min-hold recovers the churn). So we see whether the keeper stack lifts the fine-cadence book out of the
cost-drag hole.

VARIANTS (additive, isolated -- NO regime gate):
  NAIVE                 all 120 distinct configs, signal-flip, taker
  FIXED                 2MA-slow(60-150) family, signal-flip, taker
  FIXED+TRAIL           + 10% loose trailing stop
  FIXED+TRAIL+HOLD      + min_hold(12 bars)   (expected to matter at fine cadence only)
  FIXED+TRAIL+HOLD+MKR  + maker fees (0.06% rt) instead of taker

PERF FIX vs fixed_approach.py: bound the held-state slice to [start - WARMUP, end] (was full history to
end -- unusable at fine cadence over years). WARMUP=600 bars keeps EMA values ~bit-identical for span<=150.

Book = equal-weight across (config, asset) of causal MtM net daily returns over u10. All TRAIN-era.
RWYB: python -m strat.fixed_stack. No emoji (cp1252).
"""
from __future__ import annotations

import json
import re
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
from strat.portfolio_replay import holding_state, apply_trail_stop, TAKER_RT, MAKER_RT
from strat.replay_distinct_grid import distinct_specs
from strat.ma_mechanics import _cached_panel
from strat.structural_fixes import min_hold

PERIODS = {
    "Jan2020_rally": ("2020-01-07", "2020-02-07"),
    "Feb2020_revsl": ("2020-02-07", "2020-03-07"),
    "JanFeb_comb":   ("2020-01-07", "2020-03-07"),
    "Jun2022_bear":  ("2022-06-01", "2022-07-01"),
    "Feb2024_bull":  ("2024-02-01", "2024-03-01"),
}
CADENCES = ["4h", "1h", "30m", "15m"]
ANN = {"4h": 365 * 6, "1h": 365 * 24, "30m": 365 * 48, "15m": 365 * 96}
WARMUP = 600  # bars of EMA warmup before the window start (bounds cost; EMA span<=150 -> bit-stable)


def _nums(n):
    return [int(x) for x in re.findall(r"\d+", n)]


def _is_2ma(n):
    return len(_nums(n)) == 2


def _held(name, o, c, variant):
    h = holding_state(name, o, c, c, c).astype(np.int8)
    if variant in ("FIXED+TRAIL", "FIXED+TRAIL+HOLD", "FIXED+TRAIL+HOLD+MKR"):
        h = apply_trail_stop(h.copy(), c, 0.10)[0].astype(np.int8)
    if variant in ("FIXED+TRAIL+HOLD", "FIXED+TRAIL+HOLD+MKR"):
        h = min_hold(h, 12).astype(np.int8)
    return h


def book_net(config_set, cadence, start, end, variant, cost):
    s_ms = pd.Timestamp(start).value // 10**6
    e_ms = pd.Timestamp(end).value // 10**6
    syms = [a["symbol"] for a in yaml.safe_load(open(ROOT.parent / "config" / "universes" / "u10.yaml"))["assets"]]
    per_cell, idx_ref = [], None
    for sym in syms:
        try:
            o, h, l, c, ms = _cached_panel(sym, cadence)
        except Exception:
            continue
        e_idx = int(np.searchsorted(ms, e_ms))
        s_idx = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)   # warmup-bounded slice (perf)
        o, c, ms = o[s_idx:e_idx], c[s_idx:e_idx], ms[s_idx:e_idx]
        if len(c) < 20:
            continue
        wm = ms >= s_ms
        if wm.sum() < 10:
            continue
        ret = np.zeros(len(c)); ret[1:] = c[1:] / c[:-1] - 1.0
        for name in config_set:
            held = _held(name, o, c, variant)
            pos = np.zeros(len(c)); pos[1:] = held[:-1]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            net = pos * ret - flips * (cost / 2.0)
            per_cell.append(net[wm])
            if idx_ref is None:
                idx_ref = pd.to_datetime(ms[wm], unit="ms")
    if not per_cell:
        return None, None
    m = min(len(x) for x in per_cell)
    book = np.mean([x[:m] for x in per_cell], axis=0)
    return book, (idx_ref[:m] if idx_ref is not None else None)


def metrics(book, cadence):
    if book is None or len(book) < 5:
        return {}
    eq = np.cumprod(1 + book); peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min() * 100)
    sharpe = float(book.mean() / (book.std() + 1e-12) * np.sqrt(ANN.get(cadence, 365)))
    return {"roi": round(float(eq[-1] - 1) * 100, 1), "maxdd": round(dd, 1), "sharpe": round(sharpe, 2), "eq": eq}


def main() -> int:
    allcfg = {}
    for fam in ("2MA", "3MA"):
        allcfg.update(distinct_specs(fam, 0.15, max_n=60))
    PR.STRATS.update(allcfg)
    naive = list(allcfg)
    slow = [n for n in allcfg if _is_2ma(n) and 60 <= max(_nums(n)) < 150]
    print(f"config pool: {len(naive)} distinct; FIXED-slow(2MA,60-150): {len(slow)}\n")
    VARIANTS = [
        ("NAIVE", naive, "NAIVE", TAKER_RT),
        ("FIXED", slow, "FIXED", TAKER_RT),
        ("FIXED+TRAIL", slow, "FIXED+TRAIL", TAKER_RT),
        ("FIXED+TRAIL+HOLD", slow, "FIXED+TRAIL+HOLD", TAKER_RT),
        ("FIXED+TRAIL+HOLD+MKR", slow, "FIXED+TRAIL+HOLD+MKR", MAKER_RT),
    ]

    results = {}
    for cad in CADENCES:
        print(f"########## CADENCE {cad} -- variant x period (ROI% / maxDD% / Sharpe) ##########")
        print(f"   {'variant':22}" + "".join(f"{p[:11]:>20}" for p in PERIODS))
        for vname, cset, variant, cost in VARIANTS:
            row = f"   {vname:22}"
            for plabel, (s, e) in PERIODS.items():
                book, _ = book_net(cset, cad, s, e, variant, cost)
                mt = metrics(book, cad)
                results[(cad, vname, plabel)] = {k: v for k, v in mt.items() if k != "eq"}
                row += f"{(str(mt.get('roi'))+'/'+str(mt.get('maxdd'))+'/'+str(mt.get('sharpe'))):>20}" if mt else f"{'--':>20}"
            print(row)
        print()

    # ---- chart: fine-cadence recovery -- naive vs keeper-stack ROI by cadence (combined period) ----
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    vnames = [v[0] for v in VARIANTS]
    x = np.arange(len(CADENCES)); w = 0.16
    for i, vn in enumerate(vnames):
        vals = [results.get((cad, vn, "JanFeb_comb"), {}).get("roi", np.nan) for cad in CADENCES]
        ax[0].bar(x + (i - 2) * w, vals, w, label=vn)
    ax[0].set_xticks(x); ax[0].set_xticklabels(CADENCES); ax[0].axhline(0, color="k", lw=0.7)
    ax[0].set_ylabel("combined Jan+Feb ROI %"); ax[0].legend(fontsize=7)
    ax[0].set_title("Keeper stack across cadences -- does it lift the fine-cadence book? (combined)")
    # bear-period ROI by cadence (the hard test)
    for i, vn in enumerate(vnames):
        vals = [results.get((cad, vn, "Jun2022_bear"), {}).get("roi", np.nan) for cad in CADENCES]
        ax[1].bar(x + (i - 2) * w, vals, w, label=vn)
    ax[1].set_xticks(x); ax[1].set_xticklabels(CADENCES); ax[1].axhline(0, color="k", lw=0.7)
    ax[1].set_ylabel("Jun2022 bear ROI %"); ax[1].legend(fontsize=7)
    ax[1].set_title("Keeper stack in the BEAR by cadence (long-only floor)")
    fig.tight_layout()
    out = ROOT.parent / "runs" / "periods" / "TRAIN" / "_CROSS" / "charts" / "fixed_stack.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"[figure] {out}")
    ana = out.parent.parent / "analysis"; ana.mkdir(parents=True, exist_ok=True)
    json.dump({f"{c}|{v}|{p}": m for (c, v, p), m in results.items()},
              open(ana / "fixed_stack.json", "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
