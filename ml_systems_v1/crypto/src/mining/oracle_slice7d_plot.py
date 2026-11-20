"""Plot the random-7d price-oracle vs TI-oracle slices from an oracle_slice7d artifact.

Reads the JSON artifact (which stores win_start_idx/bars per cadence, so plots
reproduce the exact run windows), reloads OHLC via ChimeraLoader, recomputes the
WINNING config's fast/slow MAs (full-history warmup, same primitives), replays the
causal trade simulation WITH a trade log (identical semantics to
ti_oracle_anchor.causal_ma_long_return: window-start already-long entry, golden/death
crosses, next-bar-open fills, force-close at window end), and renders:
  per cadence: close price, fast/slow MA of the winning config, price-oracle
  low->high markers, green shading over long spans, stats in the title.
Output: runs/mining/plots/oracle_slice7d_<asset>_<stamp>/{<cad>.png, combined.png}
No emoji (cp1252).

Run:
  python -m mining.oracle_slice7d_plot                 # latest artifact
  python -m mining.oracle_slice7d_plot --artifact runs/mining/oracle_slice7d_BTCUSDT_20260610_201244.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from strat.ti_oracle_anchor import moving_avg  # noqa: E402 (read-only import)
from pipeline.chimera_loader import ChimeraLoader  # noqa: E402

OUT_BASE = ROOT / "runs" / "mining" / "plots"


def simulate_with_log(f: np.ndarray, s: np.ndarray, open_full: np.ndarray,
                      ws: int, we: int) -> list[tuple[int, int]]:
    """Replicates ti_oracle_anchor.causal_ma_long_return but returns the trade log
    [(entry_idx, exit_idx)] instead of the return. Same fills, same edge rules."""
    above = f > s
    n = len(open_full)
    trades: list[tuple[int, int]] = []
    in_pos = False
    entry_idx = -1
    if (ws < n and not np.isnan(f[ws]) and not np.isnan(s[ws])
            and ws - 1 >= 0 and not np.isnan(f[ws - 1]) and not np.isnan(s[ws - 1])
            and above[ws]):
        if open_full[ws] > 0:
            in_pos, entry_idx = True, ws
    t0 = max(1, ws - 1)
    for t in range(t0, we):
        if np.isnan(f[t]) or np.isnan(s[t]) or np.isnan(f[t - 1]) or np.isnan(s[t - 1]):
            continue
        golden = (not above[t - 1]) and above[t]
        death = above[t - 1] and (not above[t])
        fill = t + 1
        if not in_pos and golden:
            if ws <= fill < we and open_full[fill] > 0:
                in_pos, entry_idx = True, fill
        elif in_pos and death:
            exit_idx = fill if (fill < we and fill < n) else we - 1
            trades.append((entry_idx, exit_idx))
            in_pos = False
    if in_pos:
        trades.append((entry_idx, we - 1))
    return trades


def plot_slice(ax, asset: str, sl: dict) -> None:
    cad = sl["cadence"]
    df = ChimeraLoader().load(asset, cadence=cad,
                              features=["open", "high", "low", "close"])
    ts = df["timestamp"].to_numpy()
    o = df["open"].to_numpy().astype(np.float64)
    h = df["high"].to_numpy().astype(np.float64)
    lo = df["low"].to_numpy().astype(np.float64)
    c = df["close"].to_numpy().astype(np.float64)
    ws, nb = int(sl["win_start_idx"]), int(sl["bars"])
    we = ws + nb
    # context margin: 15% of window on the left for visual run-in
    pad = max(2, nb // 7)
    a, b = max(0, ws - pad), we
    x = [dt.datetime.fromtimestamp(int(t) / 1000, dt.timezone.utc) for t in ts[a:b]]

    kind, fast, slow = sl["ti_winning_config"].split("_")
    f_arr = moving_avg(c, int(fast), kind)
    s_arr = moving_avg(c, int(slow), kind)

    ax.plot(x, c[a:b], lw=1.1, color="black", label="close")
    ax.plot(x, f_arr[a:b], lw=0.9, ls="--", color="tab:blue", label=f"{kind}{fast}")
    ax.plot(x, s_arr[a:b], lw=0.9, ls=":", color="tab:orange", label=f"{kind}{slow}")
    # window boundary
    ax.axvline(x[ws - a], color="gray", lw=0.8, alpha=0.7)
    # price-oracle low -> high
    li = int(np.argmin(lo[ws:we])) + ws
    hi_roi = sl["price_oracle_roi"]
    if li + 1 < we:
        hj = int(np.argmax(h[li + 1:we])) + li + 1
        ax.scatter([x[li - a]], [lo[li]], marker="^", s=70, color="green", zorder=5,
                   label="oracle low (entry)")
        ax.scatter([x[hj - a]], [h[hj]], marker="v", s=70, color="red", zorder=5,
                   label="oracle high (exit)")
        ax.plot([x[li - a], x[hj - a]], [lo[li], h[hj]], color="green", lw=0.8, alpha=0.5)
    # long spans of the winning config
    for (ei, xi) in simulate_with_log(f_arr, s_arr, o, ws, we):
        ax.axvspan(x[ei - a], x[xi - a], color="green", alpha=0.12)
    cap = sl["capture_ratio"]
    ax.set_title(f"{cad}: {sl['window_utc'][0]} -> {sl['window_utc'][1]}   "
                 f"price-oracle {hi_roi*100:+.2f}% | TI {sl['ti_oracle_roi']*100:+.2f}% "
                 f"({sl['ti_winning_config']}) | capture {cap if cap is None else round(cap,2)} | "
                 f"B&H {sl['buy_and_hold_window']*100:+.2f}%", fontsize=9)
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.tick_params(labelsize=7)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=None)
    args = ap.parse_args()
    art = args.artifact or sorted(glob.glob(str(ROOT / "runs/mining/oracle_slice7d_*.json")))[-1]
    payload = json.loads(Path(art).read_text(encoding="utf-8"))
    asset = payload["asset"]
    slices = [s for s in payload["slices"] if "error" not in s]

    stamp = Path(art).stem.split("_")[-2] + "_" + Path(art).stem.split("_")[-1]
    out_dir = OUT_BASE / f"oracle_slice7d_{asset}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # combined figure
    fig, axes = plt.subplots(len(slices), 1, figsize=(14, 3.2 * len(slices)))
    if len(slices) == 1:
        axes = [axes]
    for ax, sl in zip(axes, slices):
        plot_slice(ax, asset, sl)
    fig.suptitle(f"{asset} -- random 7d slices: PRICE-ORACLE vs TI-ORACLE "
                 f"(seed {payload['seed']}; green shade = winning config long; "
                 f"causal signals, hindsight config choice)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    combined = out_dir / "combined.png"
    fig.savefig(combined, dpi=110)
    plt.close(fig)

    # per-cadence singles
    for sl in slices:
        fig, ax = plt.subplots(figsize=(14, 4))
        plot_slice(ax, asset, sl)
        fig.tight_layout()
        fig.savefig(out_dir / f"{sl['cadence']}.png", dpi=110)
        plt.close(fig)

    print(f"PLOTS -> {out_dir}")
    print(f"  combined: {combined}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
