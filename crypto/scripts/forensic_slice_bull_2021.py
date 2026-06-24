"""forensic_slice_bull_2021.py -- Per-asset per-trade forensic ledger for the 7-day bull slice 2021-01-02..08.

REUSES verified committed engine internals from the 1st-success MA drawdown-insurance sleeve (efb23a4b).
Imports: strat.ma_movecatch_engine.select_band_trainval, strat.ma_type_upgrade.held_cross/_nums/_MA,
         strat.ma_wealth_driver._load_asset, strat.fleet_lab (DEV_END, BARS_PER_DAY).

No emoji (Windows cp1252). No git commit. Runs synchronously.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -- path setup: script is at crypto/scripts/, src is at crypto/src/
CRYPTO = Path(__file__).resolve().parents[1]
SRC = CRYPTO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# -- verified engine imports (committed infra, do NOT reimplement)
import strat.fleet_lab as fl
from strat.ma_movecatch_engine import select_band_trainval
from strat.ma_type_upgrade import held_cross, _nums
from strat.ma_wealth_driver import _load_asset, MAKER_RT

# -- constants
LEADERBOARD = CRYPTO / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE" / "config_leaderboard.json"
OUT_CSV = (CRYPTO / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE" /
           "MA_MOVECATCH" / "forensic_slice_bull_2021-01.csv")
CHARTS_DIR = (CRYPTO / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE" /
              "MA_MOVECATCH" / "charts" / "slices")

ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XLMUSDT", "HBARUSDT",
          "ADAUSDT", "ENJUSDT", "AVAXUSDT", "DOGEUSDT", "LINKUSDT"]
TFS = ["1d", "4h", "2h", "1h", "30m", "15m"]
TF_ORDER = {t: i for i, t in enumerate(TFS)}

SLICE_START = pd.Timestamp("2021-01-02")
SLICE_END_EXCL = pd.Timestamp("2021-01-09")   # exclusive: covers Jan 02..08 inclusive

MA_TYPE = "EMA"
THR = 0.5   # band-on-fraction threshold (> 0.5 = held)

CHART_ASSETS = ["SOLUSDT", "BTCUSDT"]


def load_published_band(tf: str) -> list:
    """Get the FULL published band for a TF (for band-on-fraction computation)."""
    d = json.load(open(LEADERBOARD))
    key = f"{MA_TYPE}|{tf}"
    node = d["grid"].get(key)
    if node is None:
        raise KeyError(f"no leaderboard cell {key}")
    return list(node["band"]["band_configs"])


def compute_band_on_frac(c_full: np.ndarray, band_configs: list) -> np.ndarray:
    """Compute per-bar band-on-fraction (mean held_cross over all published-band members).
    c_full = FULL DEV-walled close array. Returns float array in [0,1], same length as c_full."""
    if len(c_full) == 0 or len(band_configs) == 0:
        return np.zeros(len(c_full), dtype=np.float64)
    on_sum = np.zeros(len(c_full), dtype=np.float64)
    for name in band_configs:
        periods = _nums(name)
        on_sum += held_cross(c_full, periods, MA_TYPE).astype(np.float64)
    frac = on_sum / len(band_configs)
    assert np.all(frac >= 0.0) and np.all(frac <= 1.0 + 1e-12), "band_on_frac out of [0,1]"
    return frac


def extract_trades(dates: pd.DatetimeIndex, c: np.ndarray, band_frac: np.ndarray) -> list:
    """Extract trades (contiguous held_state==1 runs) that overlap the slice window.
    Returns list of dicts with entry/exit metadata. Lag-1 causal fill for entry/exit prices."""
    n = len(dates)
    held_state = (band_frac > THR).astype(np.int8)

    # find contiguous runs of held_state==1 via diff transitions
    # run boundaries: where held_state changes
    padded = np.concatenate([[0], held_state, [0]])
    starts = np.where(np.diff(padded) == 1)[0]   # index in original array where run begins
    ends = np.where(np.diff(padded) == -1)[0]    # index in original array where run ends (first 0 bar)

    trades = []
    for rs, re in zip(starts, ends):
        # rs = first bar in the run (held_state[rs]==1)
        # re = first bar with held_state==0 after the run (run is rs..re-1 inclusive of held_state==1)
        # check overlap with slice: any bar in [rs, re-1] must be in slice
        # slice mask: dates >= SLICE_START and dates < SLICE_END_EXCL
        run_dates = dates[rs:re]
        overlap = ((run_dates >= SLICE_START) & (run_dates < SLICE_END_EXCL))
        if not overlap.any():
            continue

        # true entry bar (the crossing bar where held_state first became 1)
        entry_bar = rs
        entry_time = dates[entry_bar]

        # lag-1 execution: entry fill at NEXT bar's close
        entry_fill_bar = min(entry_bar + 1, n - 1)
        entry_px = float(c[entry_fill_bar])

        # true exit: first bar where held_state returns to 0 = re
        # if run extends past slice end, clamp exit to last slice bar
        last_slice_bar = int(np.searchsorted(dates, SLICE_END_EXCL, side="left")) - 1
        last_slice_bar = max(0, min(last_slice_bar, n - 1))

        run_end_bar = re - 1   # last bar where held_state==1
        if run_end_bar >= last_slice_bar:
            # run extends past (or to) slice end -> window_end_still_held
            exit_bar = last_slice_bar
            exit_reason = "window_end_still_held"
            # exit fill at next bar's close after the last slice bar
            exit_fill_bar = min(last_slice_bar + 1, n - 1)
        else:
            # run ends cleanly within the series
            exit_bar = run_end_bar
            exit_reason = "signal_cross_down"
            # lag-1: exit fill at NEXT bar's close after the cross-down (re is the first 0 bar)
            exit_fill_bar = min(re + 1, n - 1)

        exit_time = dates[exit_bar]
        exit_px = float(c[exit_fill_bar])

        # hold_bars = number of bars held (from entry_fill_bar to exit_fill_bar inclusive if clamped,
        # but semantically it's the number of held_state==1 bars in the run that are in-scope)
        # keep it simple: bars between entry_fill_bar and exit_fill_bar
        hold_bars = max(0, exit_fill_bar - entry_fill_bar)

        # ROI net
        roi_gross = (exit_px / entry_px - 1.0) * 100 if entry_px > 0 else 0.0
        roi_net = roi_gross - MAKER_RT * 100   # one round-trip maker cost

        # signal quality
        sig_at_entry = float(band_frac[entry_bar])
        max_band_during = float(np.max(band_frac[rs:re])) if re > rs else sig_at_entry

        trades.append({
            "entry_time": entry_time,
            "entry_px": entry_px,
            "exit_time": exit_time,
            "exit_px": exit_px,
            "hold_bars": hold_bars,
            "ROI_pct_net": round(roi_net, 4),
            "signal_band_on_frac_at_entry": round(sig_at_entry, 4),
            "max_band_on_during_hold": round(max_band_during, 4),
            "exit_reason": exit_reason,
            # for chart: the actual held bars in the slice
            "_rs": rs, "_re": re,
        })

    return trades


def run_asset_tf(sym: str, tf: str, published_band: list, wall_ms: int):
    """Run one (asset, tf) cell. Returns (trades_list, slice_band_frac, slice_dates, slice_close)."""
    rec = _load_asset(sym, tf, wall_ms)
    if rec is None:
        return [], np.array([]), pd.DatetimeIndex([]), np.array([])

    dates, o, h, l, c = rec

    # ASSERT: all loaded dates < DEV_END (the _load_asset wall already enforces this, but assert it)
    assert dates.max() < pd.Timestamp(fl.DEV_END), f"WALL VIOLATION: {sym}|{tf} has bars >= DEV_END"

    # compute band-on-frac over FULL series (warm EMAs naturally)
    band_frac = compute_band_on_frac(np.asarray(c, dtype=float), published_band)

    # slice mask for the window 2021-01-02 .. <2021-01-09
    slice_mask = (dates >= SLICE_START) & (dates < SLICE_END_EXCL)

    if slice_mask.sum() == 0:
        return [], np.array([]), pd.DatetimeIndex([]), np.array([])

    # extract trades on FULL series, then filter to slice-overlapping runs
    trades = extract_trades(dates, np.asarray(c, dtype=float), band_frac)

    slice_dates = dates[slice_mask]
    slice_close = np.asarray(c, dtype=float)[slice_mask]
    slice_band_frac = band_frac[slice_mask]

    return trades, slice_band_frac, slice_dates, slice_close


def make_charts(sym: str, tf_data: dict, summary: dict, out_path: Path):
    """6-panel chart for one asset. tf_data[tf] = (trades, slice_band_frac, slice_dates, slice_close)."""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes_flat = axes.flatten()

    for idx, tf in enumerate(TFS):
        ax = axes_flat[idx]
        trades, band_frac, dates, close = tf_data.get(tf, ([], np.array([]), pd.DatetimeIndex([]), np.array([])))

        n_tr = summary.get(sym, {}).get(tf, {}).get("n_trades", 0)
        roi = summary.get(sym, {}).get(tf, {}).get("slice_roi", 0.0)

        ax.set_title(f"{sym} {tf}  n_trades={n_tr}  slice_ROI={roi:+.2f}%", fontsize=9)
        ax.set_xlabel("bar", fontsize=7)
        ax.set_ylabel("price", fontsize=7)
        ax.tick_params(labelsize=6)

        if len(dates) == 0:
            ax.annotate("no data", xy=(0.5, 0.5), xycoords="axes fraction",
                        ha="center", va="center", fontsize=10)
            continue

        x = np.arange(len(dates))
        ax.plot(x, close, color="steelblue", linewidth=0.9, label="close")

        # band-on-fraction shading on twin axis
        ax2 = ax.twinx()
        ax2.set_ylim(-0.1, 1.3)
        ax2.axhline(0.5, color="gray", linewidth=0.4, linestyle="--", alpha=0.6)
        ax2.plot(x, band_frac, color="green", linewidth=0.5, alpha=0.5, label="band_frac")
        ax2.set_ylabel("band_frac", fontsize=6)
        ax2.tick_params(labelsize=5)

        # shade held regions
        if len(band_frac) > 0:
            held_mask = band_frac > THR
            # shade contiguous held spans
            in_held = False
            for xi, hv in enumerate(held_mask):
                if hv and not in_held:
                    span_start = xi; in_held = True
                elif not hv and in_held:
                    ax.axvspan(span_start, xi, alpha=0.12, color="green")
                    in_held = False
            if in_held:
                ax.axvspan(span_start, len(x) - 1, alpha=0.12, color="green")

        # trade markers (using slice-bar index matching)
        date_to_xi = {d: i for i, d in enumerate(dates)}
        for t in trades:
            # entry marker at entry_time (may be before slice start, clamp to first bar)
            et = t["entry_time"]
            xt = date_to_xi.get(et)
            if xt is None:
                # entry before slice: use bar 0
                xt = 0
            if 0 <= xt < len(close):
                ax.plot(xt, close[xt], marker="^", color="green", markersize=6, zorder=5)

            # exit marker
            ex_t = t["exit_time"]
            xx = date_to_xi.get(ex_t)
            if xx is None:
                xx = len(close) - 1
            if 0 <= xx < len(close):
                ax.plot(xx, close[xx], marker="v", color="red", markersize=6, zorder=5)

    fig.suptitle(f"{sym} -- EMA MA-cross band forensic  2021-01-02 to 2021-01-08", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=110)
    plt.close(fig)


def main():
    wall_ms = int(pd.Timestamp(fl.DEV_END).value // 10 ** 6)

    # print published band sizes at startup
    print("[startup] Published band sizes per TF:")
    pub_bands = {}
    for tf in TFS:
        band = load_published_band(tf)
        pub_bands[tf] = band
        print(f"  EMA|{tf}: {len(band)} members")

    rows = []          # trade ledger rows
    summary = {}       # summary[sym][tf] = {n_trades, slice_roi}
    tf_data_by_asset = {sym: {} for sym in ASSETS}   # for charts

    for sym in ASSETS:
        summary[sym] = {}
        for tf in TFS:
            published_band = pub_bands[tf]
            trades, slice_band_frac, slice_dates, slice_close = run_asset_tf(
                sym, tf, published_band, wall_ms)

            tf_data_by_asset[sym][tf] = (trades, slice_band_frac, slice_dates, slice_close)

            if len(slice_dates) == 0:
                print(f"  WARN: {sym}|{tf} has 0 bars in slice -- skipped")
                summary[sym][tf] = {"n_trades": 0, "slice_roi": 0.0}
                continue

            # compound slice ROI across trades
            if trades:
                compound = 1.0
                for t in trades:
                    compound *= (1.0 + t["ROI_pct_net"] / 100.0)
                slice_roi = (compound - 1.0) * 100.0
            else:
                slice_roi = 0.0

            summary[sym][tf] = {"n_trades": len(trades), "slice_roi": round(slice_roi, 3)}

            for t in trades:
                rows.append({
                    "asset": sym,
                    "tf": tf,
                    "entry_time": t["entry_time"],
                    "entry_px": t["entry_px"],
                    "exit_time": t["exit_time"],
                    "exit_px": t["exit_px"],
                    "hold_bars": t["hold_bars"],
                    "ROI_pct_net": t["ROI_pct_net"],
                    "signal_band_on_frac_at_entry": t["signal_band_on_frac_at_entry"],
                    "max_band_on_during_hold": t["max_band_on_during_hold"],
                    "exit_reason": t["exit_reason"],
                })

            print(f"  {sym}|{tf}: {len(slice_dates)} slice bars, {len(trades)} trades,"
                  f" slice_ROI={slice_roi:+.2f}%")

    # --- DELIVERABLE 1: write CSV ---
    if rows:
        df = pd.DataFrame(rows)
        # sort by asset, TF order, entry_time
        df["_tf_ord"] = df["tf"].map(TF_ORDER)
        df = df.sort_values(["asset", "_tf_ord", "entry_time"]).drop(columns=["_tf_ord"])
        df.to_csv(str(OUT_CSV), index=False)
        print(f"\n[CSV] Written: {OUT_CSV} ({len(df)} rows)")
    else:
        # write empty CSV with headers
        df = pd.DataFrame(columns=["asset", "tf", "entry_time", "entry_px", "exit_time", "exit_px",
                                    "hold_bars", "ROI_pct_net", "signal_band_on_frac_at_entry",
                                    "max_band_on_during_hold", "exit_reason"])
        df.to_csv(str(OUT_CSV), index=False)
        print(f"\n[CSV] Written (empty): {OUT_CSV}")

    # --- DELIVERABLE 2: markdown summary grid ---
    print("\n\n=== SUMMARY GRID: n_trades / slice_ROI% ===\n")
    header = "| asset | " + " | ".join(TFS) + " |"
    sep = "|---" * (len(TFS) + 1) + "|"
    print(header)
    print(sep)

    col_n = {tf: [] for tf in TFS}
    col_roi = {tf: [] for tf in TFS}

    for sym in ASSETS:
        cells = []
        for tf in TFS:
            s = summary[sym].get(tf, {"n_trades": 0, "slice_roi": 0.0})
            n = s["n_trades"]; r = s["slice_roi"]
            col_n[tf].append(n); col_roi[tf].append(r)
            sign = "+" if r >= 0 else ""
            cells.append(f"{n} / {sign}{r:.1f}")
        print(f"| {sym} | " + " | ".join(cells) + " |")

    # bottom means row
    mean_cells = []
    for tf in TFS:
        mn = round(sum(col_n[tf]) / len(col_n[tf]), 1)
        mr = round(sum(col_roi[tf]) / len(col_roi[tf]), 1)
        sign = "+" if mr >= 0 else ""
        mean_cells.append(f"{mn} / {sign}{mr:.1f}")
    print(f"| MEAN | " + " | ".join(mean_cells) + " |")

    # --- DELIVERABLE 3: charts for SOLUSDT and BTCUSDT ---
    for sym in CHART_ASSETS:
        if sym == "SOLUSDT":
            out_path = CHARTS_DIR / "forensic_SOLUSDT_alltf_bull_2021-01.png"
        else:
            out_path = CHARTS_DIR / "forensic_BTCUSDT_alltf_bull_2021-01.png"
        make_charts(sym, tf_data_by_asset[sym], summary, out_path)
        exists = os.path.exists(str(out_path))
        sz = os.path.getsize(str(out_path)) if exists else 0
        print(f"\n[chart] {out_path}  exists={exists}  size={sz:,} bytes")

    # --- quick analysis print ---
    print("\n\n=== ANALYSIS (SOL + BTC cadence contrast) ===")
    print("\nSOL entry_time by TF:")
    for tf in TFS:
        trades, _, _, _ = tf_data_by_asset["SOLUSDT"].get(tf, ([], None, None, None))
        if trades:
            et = trades[0]["entry_time"]
            print(f"  {tf}: entry at {et}  (n_trades={len(trades)})")
        else:
            print(f"  {tf}: 0 trades")

    print("\nBTC entry_time by TF:")
    for tf in TFS:
        trades, _, _, _ = tf_data_by_asset["BTCUSDT"].get(tf, ([], None, None, None))
        if trades:
            et = trades[0]["entry_time"]
            print(f"  {tf}: entry at {et}  (n_trades={len(trades)})")
        else:
            print(f"  {tf}: 0 trades")

    print("\nSlice ROI 1d vs 15m across all assets:")
    for sym in ASSETS:
        r1d = summary[sym].get("1d", {}).get("slice_roi", 0.0)
        r15 = summary[sym].get("15m", {}).get("slice_roi", 0.0)
        n1d = summary[sym].get("1d", {}).get("n_trades", 0)
        n15 = summary[sym].get("15m", {}).get("n_trades", 0)
        print(f"  {sym}: 1d={r1d:+.2f}% (n={n1d})  15m={r15:+.2f}% (n={n15})")

    print(f"\n[done] CSV: {OUT_CSV}  rows={len(rows)}")


if __name__ == "__main__":
    main()
