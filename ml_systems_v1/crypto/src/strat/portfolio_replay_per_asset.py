"""PER-ASSET report + MOVE charts for the portfolio_replay book (u10).

User /orc (2026-06-11): "report per asset in u10; show charts of the moves (buys, exits);
tell what exit mechanism was used."

EXIT MECHANISM (from portfolio_replay.holding_state): SIGNAL FLIP, not a stop.
  - 2MA (ema_50_100): hold while fast>slow; EXIT on the death cross (fast crosses below slow).
  - 3MA (ema_10_50_100): hold while fast>mid>slow; EXIT when the alignment breaks (fast<mid).
  No ATR trailing stop, no time stop in this engine. The book holds an asset while EITHER
  strategy is long, and exits to cash when BOTH are flat.

Outputs:
  - per-asset table: trades, win%, mean/median per-trade net, sum, avg hold, portfolio $ contribution.
  - a 10-panel chart (one per u10 asset): log price + entry (^) / exit (v) markers + shaded held
    spans, for the COMBINED book (either-strategy) signal.
Fills next-bar open; taker RT; signal-flip exits. No emoji (cp1252).

Run: python -m strat.portfolio_replay_per_asset --window ALL
"""
from __future__ import annotations

import argparse
import datetime as dt
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
if str(ROOT.parent / "src") not in sys.path:
    sys.path.insert(0, str(ROOT.parent / "src"))

from strat.portfolio_replay import (run, holding_state, _ma, WIN, TAKER_RT, _mask, apply_trail_stop)  # noqa: E402
from pipeline.chimera_loader import ChimeraLoader                                   # noqa: E402
from mining.family_regime_map import _norm_sym                                      # noqa: E402

OUTDIR = ROOT.parent / "runs" / "strat" / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)
STRATS_2_3 = ["ema_50_100", "ema_10_50_100"]


def per_asset_trades(o, c, held, ms, cost_rt):
    """Round-trip trades from the combined-book holding state. Entry at bar i (held 0->1)
    -> fill open[i+1]; exit at bar j (held 1->0) -> fill open[j+1]. Exit reason = signal flip."""
    n = len(c)
    trades = []
    inpos = False
    e_i = e_px = None
    for i in range(1, n - 1):
        if not inpos and held[i] and not held[i - 1]:
            if o[i + 1] > 0:
                inpos, e_i, e_px = True, i + 1, o[i + 1]
        elif inpos and not held[i] and held[i - 1]:
            x_px = o[i + 1]
            trades.append({"entry_idx": e_i, "exit_idx": i + 1, "entry_px": e_px, "exit_px": x_px,
                           "ret": x_px / e_px - 1 - cost_rt, "hold": (i + 1) - e_i,
                           "entry_ms": int(ms[e_i]), "exit_ms": int(ms[i + 1]), "exit_reason": "signal_flip"})
            inpos = False
    if inpos:                                        # open at end -> mark-to-last
        trades.append({"entry_idx": e_i, "exit_idx": n - 1, "entry_px": e_px, "exit_px": o[n - 1],
                       "ret": o[n - 1] / e_px - 1 - cost_rt, "hold": (n - 1) - e_i,
                       "entry_ms": int(ms[e_i]), "exit_ms": int(ms[n - 1]), "exit_reason": "open_at_end"})
    return trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="u10")
    ap.add_argument("--window", default="ALL")
    ap.add_argument("--start", default=None); ap.add_argument("--end", default=None)
    ap.add_argument("--trail-stop", type=float, default=0.0, help="high-water trailing stop (e.g. 0.05)")
    a = ap.parse_args()
    if a.start or a.end:
        WIN["CUSTOM"] = (a.start, a.end); a.window = "CUSTOM"
    cost = TAKER_RT

    # portfolio run for the per-asset $ contribution (weights x returns)
    r = run(a.universe, "1d", STRATS_2_3, a.window, cost, False, 0.02, 0.15, trail_stop=a.trail_stop)
    W = r["_W_full"]; retp = r["_ret_full"]; wmask = np.array(r["_wmask"])
    contrib = (W.shift(1).fillna(0.0) * retp)[wmask].sum(axis=0)     # per-asset summed weighted return

    spec = yaml.safe_load(open(ROOT.parent / "config" / "universes" / f"{a.universe}.yaml"))
    syms = [x["symbol"] for x in spec["assets"]] if "assets" in spec else []

    rows = []
    panels = []
    for sym in syms:
        try:
            df = ChimeraLoader().load(_norm_sym(sym), cadence="1d", features=["open", "high", "low", "close"])
        except Exception:
            continue
        idx = pd.to_datetime(df["timestamp"].to_numpy(), unit="ms").floor("D")
        o = df["open"].to_numpy().astype(float); h = df["high"].to_numpy().astype(float)
        l = df["low"].to_numpy().astype(float); c = df["close"].to_numpy().astype(float)
        ms = df["timestamp"].to_numpy()
        # combined book holding = either strategy long
        h2 = holding_state("ema_50_100", o, h, l, c)
        h3 = holding_state("ema_10_50_100", o, h, l, c)
        held = ((h2 + h3) > 0).astype(np.int8)
        stop_idx = set()
        if a.trail_stop and a.trail_stop > 0:        # RISK CONTROL: trailing-stop exits
            held, stop_idx = apply_trail_stop(held, c, a.trail_stop)
        # window mask on bars
        wm = _mask(ms, a.window)
        trades_all = per_asset_trades(o, c, held, ms, cost)
        for t in trades_all:                          # label the exit reason
            t["exit_reason"] = "trail_stop" if (t["exit_idx"] - 1) in stop_idx else t["exit_reason"]
        trades = [t for t in trades_all if _mask(np.array([t["entry_ms"]]), a.window)[0]]
        nets = np.array([t["ret"] for t in trades]) if trades else np.array([])
        rows.append({
            "asset": sym[:-4], "n_trades": len(trades),
            "win%": round(float((nets > 0).mean()) * 100, 0) if len(nets) else 0,
            "mean_net%": round(float(nets.mean()) * 100, 2) if len(nets) else 0,
            "median_net%": round(float(np.median(nets)) * 100, 2) if len(nets) else 0,
            "sum_net%": round(float(nets.sum()) * 100, 1) if len(nets) else 0,
            "avg_hold_bars": int(np.mean([t["hold"] for t in trades])) if trades else 0,
            "biggest_win%": round(float(nets.max()) * 100, 1) if len(nets) else 0,
            "biggest_loss%": round(float(nets.min()) * 100, 1) if len(nets) else 0,
            "port_contrib": round(float(contrib.get(sym, 0.0)), 3),
        })
        # chart data: price + MA50/MA100 + held shading + entry/exit markers, within window
        m = wm
        x = idx[m]; cp = c[m]; ema50 = _ma(c, 50, "EMA")[m]; ema100 = _ma(c, 100, "EMA")[m]
        ent = [(idx[t["entry_idx"]], o[t["entry_idx"]]) for t in trades if m[t["entry_idx"]]]
        exi_sig = [(idx[t["exit_idx"]], o[t["exit_idx"]]) for t in trades
                   if t["exit_idx"] < len(idx) and m[t["exit_idx"]] and t["exit_reason"] != "trail_stop"]
        exi_stop = [(idx[t["exit_idx"]], o[t["exit_idx"]]) for t in trades
                    if t["exit_idx"] < len(idx) and m[t["exit_idx"]] and t["exit_reason"] == "trail_stop"]
        spans = [(idx[t["entry_idx"]], idx[min(t["exit_idx"], len(idx) - 1)]) for t in trades]
        panels.append((sym[:-4], x, cp, ema50, ema100, ent, exi_sig, exi_stop, spans))

    # ---- FIGURE: 10 panels ----
    n = len(panels)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(17, 3.2 * ((n + 1) // 2)))
    axes = np.array(axes).flatten()
    for ax, (name, x, cp, e50, e100, ent, exi_sig, exi_stop, spans) in zip(axes, panels):
        ax.plot(x, cp, lw=0.8, color="black", label="close")
        ax.plot(x, e50, lw=0.7, color="tab:blue", alpha=0.7, label="EMA50")
        ax.plot(x, e100, lw=0.7, color="tab:orange", alpha=0.7, label="EMA100")
        for (s0, s1) in spans:
            ax.axvspan(s0, s1, color="green", alpha=0.07)
        if ent:
            ax.scatter([p[0] for p in ent], [p[1] for p in ent], marker="^", s=42, color="green", zorder=5, label="buy")
        if exi_sig:
            ax.scatter([p[0] for p in exi_sig], [p[1] for p in exi_sig], marker="v", s=42, color="red", zorder=5, label="exit (signal flip)")
        if exi_stop:
            ax.scatter([p[0] for p in exi_stop], [p[1] for p in exi_stop], marker="x", s=55, color="darkorange", zorder=6, label="exit (trail stop)")
        ax.set_title(name, fontsize=9); ax.set_yscale("log"); ax.grid(alpha=0.2); ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper left")
    for j in range(len(panels), len(axes)):
        axes[j].axis("off")
    exit_desc = (f"EXIT = 5%-class trailing stop ({a.trail_stop:.0%}) + signal flip (orange x = stop, red v = signal)"
                 if a.trail_stop and a.trail_stop > 0 else "EXIT = signal flip (death cross / alignment break), NO stop")
    fig.suptitle(f"{a.universe} per-asset MOVES -- book = ema_50_100 (2MA) OR ema_10_50_100 (3MA); "
                 f"{exit_desc}. window={a.window}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fpath = OUTDIR / f"portfolio_replay_per_asset_{a.universe}_{a.window}_{stamp}.png"
    fig.savefig(fpath, dpi=110); plt.close(fig)

    # ---- TABLE ----
    print(f"\n## PER-ASSET REPORT -- {a.universe} {a.window} -- book: ema_50_100 (2MA) OR ema_10_50_100 (3MA)")
    _em = (f"trailing stop {a.trail_stop:.0%} + signal flip" if a.trail_stop and a.trail_stop > 0
           else "signal flip (2MA death cross / 3MA alignment break); NO stop")
    print(f"## EXIT MECHANISM: {_em}.")
    hdr = (f"| {'asset':6} | {'trades':>6} | {'win%':>4} | {'mean%':>6} | {'med%':>6} | {'sum%':>7} | "
           f"{'hold':>4} | {'maxW%':>6} | {'maxL%':>6} | {'port$':>6} |")
    print(hdr); print("|" + "-" * (len(hdr) - 2) + "|")
    for x in sorted(rows, key=lambda z: -z["port_contrib"]):
        print(f"| {x['asset']:6} | {x['n_trades']:>6} | {x['win%']:>4.0f} | {x['mean_net%']:>+6.2f} | "
              f"{x['median_net%']:>+6.2f} | {x['sum_net%']:>+7.1f} | {x['avg_hold_bars']:>4} | "
              f"{x['biggest_win%']:>+6.1f} | {x['biggest_loss%']:>+6.1f} | {x['port_contrib']:>+6.3f} |")
    tot_tr = sum(x["n_trades"] for x in rows)
    allnet = [x["mean_net%"] for x in rows]
    print(f"\nTotals: {tot_tr} trades across {len(rows)} assets; "
          f"per-asset win-rate range {min(x['win%'] for x in rows):.0f}-{max(x['win%'] for x in rows):.0f}%; "
          f"book-level the few big winners carry it (trend anatomy).")
    jpath = ROOT.parent / "runs" / "strat" / f"portfolio_replay_per_asset_{a.universe}_{a.window}_{stamp}.json"
    json.dump({"exit_mechanism": "signal_flip (2MA death cross / 3MA alignment break); no stop/trail/time",
               "window": a.window, "rows": rows, "figure": str(fpath)}, open(jpath, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\n[figure] {fpath}\n[json]   {jpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
