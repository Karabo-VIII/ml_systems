"""Build #2: Intraday +15%-cross entry on Binance 1h klines.

Hypothesis: instead of waiting until t+1 daily close to enter (24h after trigger),
enter when the asset crosses +N% INTRADAY on day t itself. Captures the wick of
the trigger bar rather than buying the next-day close.

Scope: DEGEN + VOLATILE buckets only (where alpha lives per oracle trace + Build #1).

Methodology:
  1. Pull 1h Binance klines for DEGEN/VOLATILE assets, 24Q1-25Q4.
  2. For each (asset, day), compute cumulative-day return = close[h] / close[h=0_of_day] - 1.
  3. Find the FIRST hour where cum_ret_today >= +15% (the intraday cross).
  4. Entry = that hour's close.
  5. Exit at +N day close (default 5d post entry).
  6. Cost 24bps RT. Size 4% NAV. Cap 5 simultaneous.

Compare to t+1-close-entry baseline (Build #1) on same trigger.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Restrict to DEGEN + VOLATILE only (Build #1 confirms BLUE/STEADY don't carry alpha here)
SYMBOLS = [
    # DEGEN
    "BONK", "PEPE", "SHIB", "WIF", "WLD",
    # VOLATILE (canonical)
    "ADA", "ALGO", "ATOM", "AVAX", "DOGE", "DOT", "FET", "FIL",
    "LINK", "LTC", "NEAR", "RENDER", "TAO", "ZEC",
    # Other small caps (default VOLATILE bucket) — most-fired symbols per Build #1
    "ARKM", "AR", "ENA", "FLOKI", "PNUT", "SUI", "SUPER", "SEI", "TIA", "ORDI",
    "ZEN", "PENGU",
]

START_MS = 1704067200000  # 2024-01-01 UTC
END_MS   = 1767225599999  # 2025-12-31 UTC


def fetch_klines_1h(sym):
    """Fetch ALL 1h klines for a symbol across the 24Q1-25Q4 window (~17544 bars).

    Binance limit=1000, so we paginate by 1000h chunks (~42 days each).
    """
    url = "https://api.binance.com/api/v3/klines"
    all_rows = []
    cur = START_MS
    while cur < END_MS:
        for try_sym in [f"{sym}USDT", f"1000{sym}USDT"]:
            params = {
                "symbol": try_sym,
                "interval": "1h",
                "startTime": cur,
                "endTime": END_MS,
                "limit": 1000,
            }
            try:
                r = requests.get(url, params=params, timeout=20)
            except Exception:
                return None, "net-error"
            if r.status_code == 200 and r.json():
                data = r.json()
                for k in data:
                    all_rows.append({
                        "asset": sym,
                        "openTime": k[0],
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                    })
                cur = data[-1][0] + 3600_000  # advance past last bar
                break
        else:
            return (pd.DataFrame(all_rows) if all_rows else None), "no-data"
        time.sleep(0.06)
    if not all_rows:
        return None, "empty"
    return pd.DataFrame(all_rows), "ok"


def main():
    print(f"Pulling 1h klines for {len(SYMBOLS)} symbols ({START_MS} -> {END_MS}) ...")
    frames = []
    skipped = []
    for s in SYMBOLS:
        df, info = fetch_klines_1h(s)
        if df is None or len(df) < 200:
            skipped.append(s)
            print(f"  SKIP {s} ({info})")
            continue
        frames.append(df)
        print(f"  OK   {s}: {len(df)} hourly bars")

    if not frames:
        print("No data pulled — abort.")
        return 1

    panel = pd.concat(frames, ignore_index=True)
    panel["dt"] = pd.to_datetime(panel["openTime"], unit="ms", utc=True)
    panel["date"] = panel["dt"].dt.date
    panel["hour"] = panel["dt"].dt.hour
    panel = panel.sort_values(["asset", "dt"]).reset_index(drop=True)
    print(f"\nPanel: {len(panel)} rows, {panel['asset'].nunique()} assets")

    # Per-(asset, date) day-open close (= first hour of UTC day's close at hour 0)
    # Use the FIRST hour bar of each day as the "day open reference"
    panel["day_open_close"] = (
        panel.groupby(["asset", "date"])["close"].transform("first")
    )
    # Cumulative-day return from the first hour close
    panel["cum_day_ret"] = panel["close"] / panel["day_open_close"] - 1

    # Identify INTRADAY first cross of +15% per (asset, date)
    INTRADAY_TRIGGER = 0.15
    HOLD_DAYS = 5
    COST_RT = 0.0024
    SIZE = 0.04
    MAX_SIMUL = 5

    # First-hour-of-each-day where cum_ret crossed +15%
    first_cross = (panel[panel["cum_day_ret"] >= INTRADAY_TRIGGER]
                    .groupby(["asset", "date"]).first().reset_index())
    first_cross = first_cross.rename(columns={
        "close": "entry_close",
        "dt": "entry_dt",
        "cum_day_ret": "cum_ret_at_cross",
    })
    print(f"\nIntraday +{INTRADAY_TRIGGER*100:.0f}% crosses: {len(first_cross)} events "
          f"across {first_cross['asset'].nunique()} assets")

    # Compute exit close = close N days after the entry date (= entry_date + 5 days at last hour)
    panel_idx = panel.set_index(["asset", "dt"])["close"].sort_index()

    def exit_close_for(row):
        target_dt = row["entry_dt"] + pd.Timedelta(days=HOLD_DAYS)
        sub = panel_idx.loc[row["asset"]]
        # find first bar at or after target_dt
        idx = sub.index.searchsorted(target_dt)
        if idx >= len(sub):
            return None
        return sub.iloc[idx]

    first_cross["exit_close"] = first_cross.apply(exit_close_for, axis=1)
    first_cross = first_cross[first_cross["exit_close"].notna()].copy()
    first_cross["gross_ret"] = first_cross["exit_close"] / first_cross["entry_close"] - 1
    first_cross["net_ret"]   = first_cross["gross_ret"] - COST_RT

    # Apply position cap (FIFO simulation by entry_dt)
    fc = first_cross.sort_values("entry_dt").reset_index(drop=True)
    open_positions = []
    trade_log = []
    skipped_capacity = 0
    for _, ev in fc.iterrows():
        # Close any expired positions
        open_positions = [p for p in open_positions if p["exit_dt"] > ev["entry_dt"]]
        if len(open_positions) >= MAX_SIMUL:
            skipped_capacity += 1
            continue
        open_positions.append({
            "asset": ev["asset"],
            "entry_dt": ev["entry_dt"],
            "exit_dt": ev["entry_dt"] + pd.Timedelta(days=HOLD_DAYS),
        })
        trade_log.append({
            "asset": ev["asset"],
            "trigger_date": ev["date"],
            "entry_dt": ev["entry_dt"],
            "entry_close": ev["entry_close"],
            "exit_close": ev["exit_close"],
            "gross_ret": ev["gross_ret"],
            "net_ret": ev["net_ret"],
            "cum_ret_at_cross": ev["cum_ret_at_cross"],
        })

    tl = pd.DataFrame(trade_log)
    print(f"\nTrades closed: {len(tl)} (skipped due to cap: {skipped_capacity})")
    if len(tl) == 0:
        print("No trades — abort.")
        return 1

    # Summary
    print(f"  mean_net_per_trade = {tl['net_ret'].mean()*100:+.3f}%")
    print(f"  win_rate           = {(tl['net_ret']>0).mean()*100:.1f}%")
    print(f"  median             = {tl['net_ret'].median()*100:+.3f}%")
    print(f"  best               = {tl['net_ret'].max()*100:+.2f}%")
    print(f"  worst              = {tl['net_ret'].min()*100:+.2f}%")
    total_nav = SIZE * tl["net_ret"].sum() * 100
    print(f"  TOTAL NAV 8Q @ {SIZE*100}% size: {total_nav:+.2f}%")

    # Compare to Build #1 (t+1 close entry) — apples-to-apples on the trigger thresh
    print()
    print("Comparison to Build #1 (t+1 close entry, same trigger thresh):")
    print("  Build #1 (+15% gated bucketed 5d): +36.55% NAV 8Q @ 4% size, 264 trades")
    print("  Build #2 (intraday +15% cross, 5d hold, DEGEN+VOLATILE):"
          f" {total_nav:+.2f}% NAV, {len(tl)} trades")

    # Save
    out_path = OUT_DIR / "INTRADAY_ENTRY_BUILD2_2026_05_18.md"
    tl.to_parquet(OUT_DIR / "mover_lane_trades_BUILD2_INTRADAY.parquet")

    lines = []
    def w(s=""):
        lines.append(s)
    w("# Build #2: Intraday +15%-Cross Entry (Real Binance 1h Data)")
    w()
    w("**Date**: 2026-05-18  ")
    w(f"**Window**: 24Q1 → 25Q4  ")
    w(f"**Universe**: {panel['asset'].nunique()} DEGEN/VOLATILE assets  ")
    w(f"**Source**: Binance 1h public klines  ")
    w(f"**Bars pulled**: {len(panel)} total  ")
    w()
    w("## Method")
    w()
    w(f"- Trigger: cumulative day-return (from UTC midnight close ref) crosses +{INTRADAY_TRIGGER*100:.0f}% intraday")
    w(f"- Entry: that hourly bar's close")
    w(f"- Exit: hourly bar at entry_dt + {HOLD_DAYS}d")
    w(f"- Cost: {COST_RT*100*100:.0f}bps RT, Size: {SIZE*100:.0f}% NAV, Max simul: {MAX_SIMUL}")
    w()
    w("## Results")
    w()
    w(f"- Triggers fired: {len(first_cross) + skipped_capacity}  ")
    w(f"- Trades closed: {len(tl)} (skipped at cap: {skipped_capacity})  ")
    w(f"- Mean net per trade: {tl['net_ret'].mean()*100:+.3f}%  ")
    w(f"- Win rate: {(tl['net_ret']>0).mean()*100:.1f}%  ")
    w(f"- Median per trade: {tl['net_ret'].median()*100:+.3f}%  ")
    w(f"- Best / worst trade: {tl['net_ret'].max()*100:+.2f}% / {tl['net_ret'].min()*100:+.2f}%  ")
    w(f"- **TOTAL NAV 8Q @ {SIZE*100:.0f}%: {total_nav:+.2f}%**  ")
    w()
    w("## Comparison vs Build #1 (t+1 close entry)")
    w()
    w("| Variant | Entry | Trades | Mean net/trade | NAV 8Q @4% |")
    w("|---|---|---:|---:|---:|")
    w("| Build #1 (+15% gated bucketed 5d) | t+1 close | 264 | +3.461% | +36.55% |")
    w(f"| **Build #2 (intraday +15% cross 5d)** | intraday cross | {len(tl)} | "
      f"{tl['net_ret'].mean()*100:+.3f}% | **{total_nav:+.2f}%** |")
    w()
    delta = total_nav - 36.55
    w(f"Delta: **{delta:+.2f}pp** (intraday entry vs t+1 close)")
    w()
    if delta > 0:
        w("**Hypothesis CONFIRMED**: capturing the wick of day-t lifts NAV — intraday entry is the better trigger time.")
    else:
        w("**Hypothesis REJECTED at this scope**: t+1 close is competitive with intraday cross — likely because")
        w("the average post-cross intraday continuation is small and the t+1-close-mean-revert pattern is overstated.")
        w("Could re-test with a +N% cross at hour ≤ 12 to skip late-day crosses.")
    w()

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
