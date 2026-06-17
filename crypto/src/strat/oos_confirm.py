"""src/strat/oos_confirm.py -- the OUT-OF-SAMPLE confirmation of the assembled complete stack.

WHY (user /orc 2026-06-12): everything in the fixed-approach arc (ladder -> keeper stack -> regime gate
-> complete stack) is TRAIN-era IN-SAMPLE structural design. The honest litmus: does the assembled stack
TRANSFER to data the design never saw? Run the variants on the full VAL and OOS spans. UNSEEN stays SEALED.

VARIANTS (the build-fix-upgrade arc, end to end):
  NAIVE       all 120 distinct configs, signal-flip, TAKER          (the un-fixed baseline)
  FIXED       2MA-slow(60-150) family, signal-flip, TAKER           (the family fix)
  FULL        FIXED + TRAIL(10%) + min_hold(12) + MAKER             (the full keeper stack)
  FULL_GATE   FULL + BTC100-hysteresis market gate                  (the 4h regime overlay)

Per (variant, cadence in {4h,1h}, span in {TRAIN_ref, VAL, OOS}): book ROI% / maxDD% / Sharpe, plus
%POSITIVE cells (breadth -- the concentration firewall: a high book ROI on few cells is fragile).
Equal-weight u10 book, causal MtM. RWYB: python -m strat.oos_confirm. No emoji (cp1252).
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
from strat.complete_stack import _sma, _hysteresis

# spans: a TRAIN reference (2021 full year, unseen-by-the-2020/2022/2024-design-windows) + VAL + OOS.
SPANS = {
    "TRAINref_2021": ("2021-01-01", "2022-01-01"),
    "VAL":           ("2024-05-15", "2025-03-15"),
    "OOS":           ("2025-03-15", "2025-12-31"),
}
CADENCES = ["4h", "1h"]
ANN = {"4h": 365 * 6, "1h": 365 * 24}
WARMUP = 600
VARIANTS = ["NAIVE", "FIXED", "FULL", "FULL_GATE"]


def _nums(n):
    return [int(x) for x in re.findall(r"\d+", n)]


_BTC_H = {}


def _btc_hyst(cadence):
    if cadence not in _BTC_H:
        o, h, l, c, ms = _cached_panel("BTCUSDT", cadence)
        _BTC_H[cadence] = (ms, _hysteresis(c, 100, 0.03))
    return _BTC_H[cadence]


def _weight(name, o, c, ms, cadence, variant):
    h = holding_state(name, o, c, c, c).astype(np.int8)
    if variant in ("NAIVE", "FIXED"):
        return h.astype(np.float64)
    h = apply_trail_stop(h.copy(), c, 0.10)[0].astype(np.int8)
    h = min_hold(h, 12).astype(np.float64)
    if variant == "FULL":
        return h
    btc_ms, btc_reg = _btc_hyst(cadence)                 # FULL_GATE
    idx = np.clip(np.searchsorted(btc_ms, ms, side="right") - 1, 0, len(btc_reg) - 1)
    return h * btc_reg[idx].astype(np.float64)


def book(config_set, cadence, start, end, variant):
    """returns (book_series, list_of_per_cell_full_span_ROI)."""
    cost = TAKER_RT if variant in ("NAIVE", "FIXED") else MAKER_RT
    s_ms = pd.Timestamp(start).value // 10**6
    e_ms = pd.Timestamp(end).value // 10**6
    syms = [a["symbol"] for a in yaml.safe_load(open(ROOT.parent / "config" / "universes" / "u10.yaml"))["assets"]]
    per_cell, cell_roi = [], []
    for sym in syms:
        try:
            o, h, l, c, ms = _cached_panel(sym, cadence)
        except Exception:
            continue
        e_idx = int(np.searchsorted(ms, e_ms))
        s_idx = max(0, int(np.searchsorted(ms, s_ms)) - WARMUP)
        o, c, ms = o[s_idx:e_idx], c[s_idx:e_idx], ms[s_idx:e_idx]
        if len(c) < 50:
            continue
        wm = ms >= s_ms
        if wm.sum() < 30:
            continue
        ret = np.zeros(len(c)); ret[1:] = c[1:] / c[:-1] - 1.0
        for name in config_set:
            w = _weight(name, o, c, ms, cadence, variant)
            pos = np.zeros(len(c)); pos[1:] = w[:-1]
            flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
            net = (pos * ret - flips * (cost / 2.0))[wm]
            per_cell.append(net)
            cell_roi.append(float(np.cumprod(1 + net)[-1] - 1) * 100)
    if not per_cell:
        return None, []
    m = min(len(x) for x in per_cell)
    return np.mean([x[:m] for x in per_cell], axis=0), cell_roi


def metrics(bk, rois, cadence):
    if bk is None or len(bk) < 10:
        return {}
    eq = np.cumprod(1 + bk); peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min() * 100)
    sharpe = float(bk.mean() / (bk.std() + 1e-12) * np.sqrt(ANN.get(cadence, 365)))
    pos = 100.0 * np.mean(np.array(rois) > 0) if rois else float("nan")
    return {"roi": round(float(eq[-1] - 1) * 100, 1), "maxdd": round(dd, 1), "sharpe": round(sharpe, 2),
            "pos": round(pos, 0), "eq": eq}


def main() -> int:
    allcfg = {}
    for fam in ("2MA", "3MA"):
        allcfg.update(distinct_specs(fam, 0.15, max_n=60))
    PR.STRATS.update(allcfg)
    naive = list(allcfg)
    slow = [n for n in allcfg if len(_nums(n)) == 2 and 60 <= max(_nums(n)) < 150]
    cfgset = {"NAIVE": naive, "FIXED": slow, "FULL": slow, "FULL_GATE": slow}
    print(f"OOS confirm: NAIVE({len(naive)}) FIXED/FULL/FULL_GATE({len(slow)}); spans {list(SPANS)}\n")

    results = {}
    for cad in CADENCES:
        print(f"########## CADENCE {cad} -- variant x span (ROI% / maxDD% / Sharpe / %posCells) ##########")
        print(f"   {'variant':12}" + "".join(f"{s:>26}" for s in SPANS))
        for v in VARIANTS:
            row = f"   {v:12}"
            for span, (s, e) in SPANS.items():
                bk, rois = book(cfgset[v], cad, s, e, v)
                mt = metrics(bk, rois, cad)
                results[(cad, v, span)] = {k: val for k, val in mt.items() if k != "eq"}
                if mt:
                    cell = f"{mt.get('roi')}/{mt.get('maxdd')}/{mt.get('sharpe')}/{mt.get('pos')}%"
                else:
                    cell = "--"
                row += f"{cell:>26}"
            print(row)
        print()

    # transfer view: did FULL beat NAIVE out of sample? did the gate help?
    print("[TRANSFER] out-of-sample (VAL, OOS): FULL vs NAIVE, and FULL_GATE vs FULL (4h+1h mean ROI)")
    print(f"   {'span':6} {'NAIVE':>8} {'FIXED':>8} {'FULL':>8} {'FULL_GATE':>10} {'FULL-NAIVE':>11} {'GATE-FULL':>10}")
    for span in SPANS:
        m = {v: np.mean([results[(c, v, span)].get("roi", np.nan) for c in CADENCES]) for v in VARIANTS}
        print(f"   {span[:6]:6} {m['NAIVE']:>8.1f} {m['FIXED']:>8.1f} {m['FULL']:>8.1f} {m['FULL_GATE']:>10.1f} "
              f"{m['FULL']-m['NAIVE']:>+11.1f} {m['FULL_GATE']-m['FULL']:>+10.1f}")

    # chart: ROI by variant per span (4h) + equity curves (OOS, 4h)
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(SPANS)); w = 0.2
    for i, v in enumerate(VARIANTS):
        ax[0].bar(x + (i - 1.5) * w, [results[("4h", v, s)].get("roi", np.nan) for s in SPANS], w, label=v)
    ax[0].set_xticks(x); ax[0].set_xticklabels(list(SPANS)); ax[0].axhline(0, color="k", lw=0.7)
    ax[0].legend(fontsize=8); ax[0].set_ylabel("book ROI % (4h)")
    ax[0].set_title("Out-of-sample transfer -- variant x span (4h)")
    for v in VARIANTS:
        bk, rois = book(cfgset[v], "4h", *SPANS["OOS"], v)
        mt = metrics(bk, rois, "4h")
        if mt:
            ax[1].plot(mt["eq"], label=f"{v} ({mt['roi']:+.0f}%, DD{mt['maxdd']:.0f})", lw=1.3)
    ax[1].axhline(1.0, color="grey", ls=":", lw=0.6); ax[1].legend(fontsize=8)
    ax[1].set_title("OOS (2025-03..12) book equity by variant (4h)"); ax[1].set_ylabel("$1 book equity")
    fig.tight_layout()
    out = ROOT.parent / "runs" / "periods" / "_OOS_CONFIRM" / "charts" / "oos_confirm.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110); plt.close(fig)
    print(f"\n[figure] {out}")
    json.dump({f"{c}|{v}|{s}": m for (c, v, s), m in results.items()},
              open(out.parent.parent / "oos_confirm.json", "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
