"""src/strat/grade_full_stack.py -- grade the ONE transferable finding with the canonical scorecard.

WHY: the OOS confirmation gave POINT estimates (the FULL keeper stack cuts the OOS book loss -19->-6%,
4h +2%). The North Star demands ROBUST not point-estimate: block-bootstrap p05 > 0, breadth, PBO. So run
the FULL stack (FIXED 2MA-slow + TRAIL10 + min_hold12 + MAKER, 4h) through `strat.scorecard.score_book`
-- the canonical grader used for ALL strategy grading.

CRITICAL UNSEEN GUARD: the scorecard keys its SHIP verdict on UNSEEN (2025-12-31..2026-06-01), which is
SEALED. We build the book series ending at 2025-12-31 (the UNSEEN boundary) so UNSEEN stays UNTOUCHED --
spending the test-once UNSEEN is an irreversible strategic call for the USER, not an autonomous one. This
grades robustness on the SEL(TRAIN+VAL)+OOS span only; ship_read will honestly show no-UNSEEN.

Outputs: per-split compound/maxDD/Sharpe + rolling soft-bench, full + OOS-heldout block-bootstrap p05,
PBO over the 20-config grid, per-asset OOS breadth. RWYB: python -m strat.grade_full_stack. No emoji.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR
from strat.portfolio_replay import holding_state, apply_trail_stop, MAKER_RT
from strat.replay_distinct_grid import distinct_specs
from strat.ma_mechanics import _cached_panel
from strat.structural_fixes import min_hold
from strat.scorecard import score_book

CADENCE = "4h"
END = "2025-12-31"          # UNSEEN boundary -- never cross it here
WARMUP = 600


def _nums(n):
    return [int(x) for x in re.findall(r"\d+", n)]


def _full_net(name, o, c):
    """FULL keeper-stack net per-bar return (maker), full history."""
    h = holding_state(name, o, c, c, c).astype(np.int8)
    h = apply_trail_stop(h.copy(), c, 0.10)[0].astype(np.int8)
    h = min_hold(h, 12).astype(np.float64)
    ret = np.zeros(len(c)); ret[1:] = c[1:] / c[:-1] - 1.0
    pos = np.zeros(len(c)); pos[1:] = h[:-1]
    flips = np.abs(np.diff(np.concatenate([[0.0], pos])))
    return pos * ret - flips * (MAKER_RT / 2.0)


def main() -> int:
    allcfg = {}
    for fam in ("2MA", "3MA"):
        allcfg.update(distinct_specs(fam, 0.15, max_n=60))
    PR.STRATS.update(allcfg)
    slow = [n for n in allcfg if len(_nums(n)) == 2 and 60 <= max(_nums(n)) < 150]
    syms = [a["symbol"] for a in yaml.safe_load(open(ROOT.parent / "config" / "universes" / "u10.yaml"))["assets"]]
    e_ms = pd.Timestamp(END).value // 10**6
    print(f"grade FULL stack: {len(slow)} cfg x {len(syms)} assets, {CADENCE}, span -> {END} (UNSEEN sealed)\n")

    # per (config, asset) 4h net series, date-indexed, capped at END
    cell_series = {}          # (cfg, sym) -> Series
    asset_series = {sym: [] for sym in syms}
    for sym in syms:
        try:
            o, h, l, c, ms = _cached_panel(sym, CADENCE)
        except Exception:
            continue
        keep = ms < e_ms
        o, c, ms = o[keep], c[keep], ms[keep]
        if len(c) < 100:
            continue
        idx = pd.to_datetime(ms, unit="ms")
        for name in slow:
            net = _full_net(name, o, c)
            s = pd.Series(net, index=idx)
            cell_series[(name, sym)] = s
            asset_series[sym].append(s)

    # book (equal-weight across all live cells per 4h bar) -> daily compounded
    book_df = pd.DataFrame(cell_series)
    book_4h = book_df.mean(axis=1, skipna=True)
    daily_net = book_4h.resample("1D").apply(lambda x: float((1 + x).prod() - 1)).dropna()

    # per-config daily book matrix for PBO (each config = mean across its assets)
    cfg_daily = {}
    for name in slow:
        cols = [cell_series[(name, sym)] for sym in syms if (name, sym) in cell_series]
        if not cols:
            continue
        c4 = pd.concat(cols, axis=1).mean(axis=1, skipna=True)
        cfg_daily[name] = c4.resample("1D").apply(lambda x: float((1 + x).prod() - 1))
    grid = pd.DataFrame(cfg_daily).reindex(daily_net.index).fillna(0.0).to_numpy()

    card = score_book("FULL_4h_keeperstack", daily_net, grid_returns=grid)

    # per-asset OOS breadth (the concentration firewall)
    oos = ("2025-03-15", "2025-12-31")
    breadth = {}
    for sym in syms:
        if not asset_series[sym]:
            continue
        a4 = pd.concat(asset_series[sym], axis=1).mean(axis=1, skipna=True)
        ad = a4.resample("1D").apply(lambda x: float((1 + x).prod() - 1)).dropna()
        seg = ad[(ad.index >= oos[0]) & (ad.index < oos[1])]
        if len(seg) > 5:
            breadth[sym] = round(float((1 + seg).prod() - 1) * 100, 1)
    bpos = sum(1 for v in breadth.values() if v > 0)

    # report
    print("=== CANONICAL SCORECARD: FULL_4h_keeperstack (UNSEEN untouched) ===")
    for sp in ("SEL", "OOS", "UNSEEN"):
        p = card["per_split"].get(sp, {})
        if p.get("n", 0) >= 5:
            sb = p["softbench_roi"]
            print(f"  {sp:7} n={p['n']:4} compound {p['compound_pct']:+7.2f}%  ann {p['ann_pct']:+8.2f}%  "
                  f"maxDD {p['maxdd_pct']:+6.2f}%  Sharpe {p['sharpe']:+5.2f}  "
                  f"| 1d med {sb.get('1d',{}).get('median','?')} %pos {sb.get('1d',{}).get('pct_pos','?')}")
        else:
            print(f"  {sp:7} n={p.get('n',0)}  (sealed / insufficient -- UNSEEN intentionally untouched)")
    fb = card["full_block_bootstrap"]; hb = card["heldout_block_bootstrap"]
    print(f"\n  full-cycle (SEL+OOS) block-bootstrap p05 {fb.get('p05')}  p95 {fb.get('p95')}  "
          f"(robust iff p05 > 0)")
    print(f"  OOS-heldout block-bootstrap p05 {hb.get('p05')}  p95 {hb.get('p95')}")
    print(f"  PBO (prob backtest overfit, 20-config grid): {card.get('pbo', {})}")
    print(f"\n  per-asset OOS breadth ({bpos}/{len(breadth)} positive): " +
          ", ".join(f"{k}{v:+.0f}" for k, v in sorted(breadth.items(), key=lambda kv: -kv[1])))
    sr = card["ship_read"]
    print(f"\n  ship_read (UNSEEN-keyed, EXPECTED no-ship -- UNSEEN sealed): {sr}")
    print(f"  ROBUSTNESS VERDICT (TRAIN+VAL+OOS, no UNSEEN): full_p05_pos={sr['full_p05_pos']} "
          f"heldout_p05_pos={sr['heldout_p05_pos']} OOS_breadth={bpos}/{len(breadth)}")

    out = ROOT.parent / "runs" / "periods" / "_OOS_CONFIRM" / "full_stack_scorecard.json"
    card["oos_breadth"] = {"positive": bpos, "total": len(breadth), "per_asset": breadth}
    json.dump(card, open(out, "w"), indent=1, default=str)
    print(f"\n[json] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
