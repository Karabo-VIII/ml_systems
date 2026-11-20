"""src/strat/ma_mechanics.py -- the WHAT-WORKED / WHAT-DIDN'T mechanics engine for the MA building block.

WHY (user /orc 2026-06-12): *"do an analysis on what worked and what did not: overtrading, whipsaws,
etc. I need explainers across the board."* The per-instrument sweep told us WHICH cells won; this
decomposes WHY -- the trade-level economics: gross vs net, the cost drag, whipsaw (1-2 bar flips),
trades-per-day, win rate vs hold. It is the mechanism behind the cadence-decay + holding-time story.

Per (asset, cadence, config, exit) over the oldest month it computes, from the engine's own holding
state (causal, lagged 1 bar, MtM):
  - GROSS compound (no cost) vs NET compound (taker round-trip) -> COST DRAG = gross - net
  - n_trades, trades/day, WHIPSAW fraction (round trips held <= 2 bars), avg hold, win rate
  - the per-bar equity curve (for the chart modules to draw)

extract() is the reusable kernel (also used by ma_equity_grid). main() runs the full grid, prints the
aggregate explainers, writes the 4-panel mechanics figure + a JSON.

RWYB:  python -m strat.ma_mechanics
No emoji (cp1252).
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.portfolio_replay as PR
from strat.portfolio_replay import holding_state, apply_trail_stop, TAKER_RT, MAKER_RT
from strat.portfolio_replay_per_asset import per_asset_trades
from strat.replay_distinct_grid import distinct_specs
from strat.ma_per_instrument import _panel

PLOTS = ROOT.parent / "runs" / "strat" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
START, END = "2020-01-07", "2020-02-07"
TFS = ["4h", "1h", "30m", "15m"]
EXITS = [("signalflip", 0.0), ("trail5", 0.05), ("trail10", 0.10)]
CAD_COLOR = {"4h": "#1b9e77", "1h": "#7570b3", "30m": "#d95f02", "15m": "#e7298a"}
BARS_PER_DAY = {"4h": 6, "1h": 24, "30m": 48, "15m": 96}
_SPECS = {}
_PANEL_CACHE = {}
MAX_N = 2000   # configs per family to inject (2000 -> the FULL distinct set: 306 2MA + 1466 3MA)
COST_RT = TAKER_RT   # round-trip cost (set to MAKER_RT via --maker for the cost-lever comparison)
COST_NAME = "taker"


def _cached_panel(asset, cadence):
    k = (asset, cadence)
    if k not in _PANEL_CACHE:
        _PANEL_CACHE[k] = _panel(asset, cadence)
    return _PANEL_CACHE[k]


def _ensure_specs():
    if not _SPECS:
        for fam in ("2MA", "3MA"):
            _SPECS.update(distinct_specs(fam, 0.15, max_n=MAX_N))
        PR.STRATS.update(_SPECS)
    return _SPECS


def extract(asset, cadence, name, trail, start=START, end=END):
    """Reusable kernel: trade-level + bar-level economics for one (asset, config, exit). Returns None
    if the asset has < 5 window bars. Includes the per-bar equity curve over the window."""
    _ensure_specs()
    s_ms = pd.Timestamp(start).value // 10**6
    e_ms = pd.Timestamp(end).value // 10**6
    o, h, l, c, ms = _cached_panel(asset, cadence)
    keep = ms < e_ms
    o, c, ms = o[keep], c[keep], ms[keep]
    if (ms >= s_ms).sum() < 5:
        return None
    held = holding_state(name, o, c, c, c).astype(np.int8)
    if trail > 0:
        held = apply_trail_stop(held.copy(), c, trail)[0].astype(np.int8)
    # bar-level MtM (causal: position is yesterday's held)
    ret = np.zeros(len(c)); ret[1:] = c[1:] / c[:-1] - 1.0
    pos = np.zeros(len(c)); pos[1:] = held[:-1]
    flips = np.abs(np.diff(np.concatenate([[0.0], pos])))      # |pos_t - pos_{t-1}|
    gross_bar = pos * ret
    cost_bar = flips * (COST_RT / 2.0)                        # half round-trip per side
    net_bar = gross_bar - cost_bar
    eq_gross = np.cumprod(1 + gross_bar)
    eq_net = np.cumprod(1 + net_bar)
    # trade-level
    trades = [t for t in per_asset_trades(o, c, held, ms, COST_RT) if s_ms <= t["entry_ms"] < e_ms]
    holds = np.array([t["hold"] for t in trades]) if trades else np.array([])
    rets = np.array([t["ret"] for t in trades]) if trades else np.array([])
    n = len(trades)
    n_whip = int((holds <= 2).sum()) if n else 0
    days = (e_ms - s_ms) / 86400000.0
    return {
        "asset": asset, "cadence": cadence, "config": name, "trail": trail,
        "net_pct": round(float((eq_net[-1] - 1) * 100), 2),
        "gross_pct": round(float((eq_gross[-1] - 1) * 100), 2),
        "cost_drag_pct": round(float((eq_gross[-1] - eq_net[-1]) * 100), 2),
        "n_trades": n, "trades_per_day": round(n / days, 2),
        "whipsaw_frac": round(n_whip / n, 2) if n else 0.0, "n_whip": n_whip,
        "avg_hold_bars": round(float(holds.mean()), 1) if n else 0.0,
        "win_rate": round(float((rets > 0).mean()), 2) if n else 0.0,
        "_eq_net": eq_net, "_eq_gross": eq_gross, "_dates": pd.to_datetime(ms, unit="ms"),
        "_trades": trades, "_c": c, "_ms": ms,
    }


def run_grid(start=START, end=END):
    cells = []
    for cad in TFS:
        for asset in sorted({a for a in _u10()}):
            for name, (fam, _p) in _ensure_specs().items():
                for ex_name, trail in EXITS:
                    m = extract(asset, cad, name, trail, start, end)
                    if m is None or m["n_trades"] == 0:
                        continue
                    m = {k: v for k, v in m.items() if not k.startswith("_")}
                    m["family"] = fam; m["exit"] = ex_name
                    cells.append(m)
    return cells


def _u10():
    import yaml
    return [a["symbol"] for a in yaml.safe_load(open(ROOT.parent / "config" / "universes" / "u10.yaml"))["assets"]]


def mechanics_figure(cells, out):
    nt = np.array([c["n_trades"] for c in cells]); net = np.array([c["net_pct"] for c in cells])
    gross = np.array([c["gross_pct"] for c in cells]); cad = np.array([c["cadence"] for c in cells])
    whip = np.array([c["whipsaw_frac"] for c in cells]); hold = np.array([c["avg_hold_bars"] for c in cells])
    win = np.array([c["win_rate"] for c in cells]); drag = np.array([c["cost_drag_pct"] for c in cells])
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("MA mechanics -- what worked / what didn't (u10, oldest month 2020-01-07..02-07)",
                 fontsize=13, fontweight="bold")
    # 1. overtrading: trades vs net
    a = ax[0, 0]
    for c in TFS:
        m = cad == c; a.scatter(nt[m], net[m], s=16, alpha=0.45, color=CAD_COLOR[c], label=c)
    a.axhline(0, color="k", lw=0.7); a.set_xscale("symlog")
    a.set_xlabel("# trades (symlog)"); a.set_ylabel("net compound %")
    a.legend(title="cadence", fontsize=8); a.set_title("A. OVERTRADING: more trades -> worse net")
    # 2. whipsaw fraction by cadence
    b = ax[0, 1]
    wf = [whip[cad == c].mean() * 100 for c in TFS]
    b.bar(TFS, wf, color=[CAD_COLOR[c] for c in TFS], alpha=0.85)
    b.set_ylabel("% of trades held <=2 bars"); b.set_title("B. WHIPSAW: finer cadence flips more")
    for i, v in enumerate(wf):
        b.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    # 3. cost bleed: gross vs net by cadence
    cax = ax[1, 0]
    x = np.arange(len(TFS)); w = 0.38
    cax.bar(x - w / 2, [gross[cad == c].mean() for c in TFS], w, label="gross (no cost)", color="#9ecae1")
    cax.bar(x + w / 2, [net[cad == c].mean() for c in TFS], w, label="net (taker)", color="#3182bd")
    cax.set_xticks(x); cax.set_xticklabels(TFS); cax.axhline(0, color="k", lw=0.7)
    cax.set_ylabel("mean compound %"); cax.legend(fontsize=8)
    cax.set_title("C. COST BLEED: gross vs net (gap = cost drag)")
    # 4. win rate vs hold
    d = ax[1, 1]
    for c in TFS:
        m = cad == c; d.scatter(hold[m], win[m] * 100, s=16, alpha=0.45, color=CAD_COLOR[c], label=c)
    d.axhline(50, color="k", lw=0.7, ls="--"); d.set_xscale("symlog")
    d.set_xlabel("avg hold (bars, symlog)"); d.set_ylabel("win rate %")
    d.legend(title="cadence", fontsize=8); d.set_title("D. WIN RATE vs HOLD: hold longer -> win more")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out, dpi=110); plt.close(fig)


def main() -> int:
    global COST_RT, COST_NAME
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--maker", action="store_true", help="use maker round-trip cost (the cost lever)")
    ap.add_argument("--start", default=START); ap.add_argument("--end", default=END)
    a = ap.parse_args()
    if a.maker:
        COST_RT, COST_NAME = MAKER_RT, "maker"
    suffix = "" if COST_NAME == "taker" else f"_{COST_NAME}"
    from strat.period_store import sub as _psub
    charts_dir, raw_dir = _psub(a.start, "charts"), _psub(a.start, "raw")
    cells = run_grid(a.start, a.end)
    out = charts_dir / f"ma_mechanics{suffix}.png"
    mechanics_figure(cells, out)
    # aggregate explainers (printed for the narrative)
    nt = np.array([c["n_trades"] for c in cells]); net = np.array([c["net_pct"] for c in cells])
    gross = np.array([c["gross_pct"] for c in cells]); cad = np.array([c["cadence"] for c in cells])
    whip = np.array([c["whipsaw_frac"] for c in cells]); drag = np.array([c["cost_drag_pct"] for c in cells])
    hold = np.array([c["avg_hold_bars"] for c in cells]); win = np.array([c["win_rate"] for c in cells])
    print(f"## MA MECHANICS [{COST_NAME.upper()}] -- {len(cells)} cells (FULL set x 3 exits x 4 tf x 7 assets)")
    print(f"   {'cad':4} {'net%':>7} {'gross%':>7} {'costDrag%':>9} {'trades':>7} {'tr/day':>7} {'whip%':>6} {'hold(b)':>8} {'win%':>5}")
    for c in TFS:
        m = cad == c
        print(f"   {c:4} {net[m].mean():>7.1f} {gross[m].mean():>7.1f} {drag[m].mean():>9.1f} "
              f"{nt[m].mean():>7.0f} {np.array([x['trades_per_day'] for x in cells if x['cadence']==c]).mean():>7.1f} "
              f"{whip[m].mean()*100:>5.0f}% {hold[m].mean():>8.0f} {win[m].mean()*100:>4.0f}%")
    # winners vs losers mechanics
    top = net >= np.percentile(net, 75); bot = net <= np.percentile(net, 25)
    print(f"\n   WINNERS (top 25% net): trades {nt[top].mean():.0f}  whip {whip[top].mean()*100:.0f}%  "
          f"hold {hold[top].mean():.0f}b  win {win[top].mean()*100:.0f}%  costDrag {drag[top].mean():.1f}%")
    print(f"   LOSERS  (bot 25% net): trades {nt[bot].mean():.0f}  whip {whip[bot].mean()*100:.0f}%  "
          f"hold {hold[bot].mean():.0f}b  win {win[bot].mean()*100:.0f}%  costDrag {drag[bot].mean():.1f}%")
    json.dump(cells, open(raw_dir / f"ma_mechanics{suffix}.json", "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[figure] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
