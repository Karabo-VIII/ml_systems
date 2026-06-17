"""Funding-rate extreme overlay backtest.

Hypothesis: extreme funding rates (crowded positioning) predict mean reversion.
When funding is deeply positive (|z| > 2σ), longs are paying shorts -- crowd is
long -- mean reversion favors SHORT. When deeply negative, favor LONG.

Data: Binance /fapi/v1/fundingRate for every USDT perpetual, full history.
Each funding rate stamped at 8h intervals (00:00, 08:00, 16:00 UTC).

Test:
  1. For each asset, compute rolling 30-day z-score of funding rate.
  2. When |z| > Z_THRESH, measure forward return over 8h, 24h, 48h, 72h.
  3. Directional: if z > +Z_THRESH, trade SHORT; if z < -Z_THRESH, trade LONG.
  4. Aggregate across all assets, all events.

Ship criterion: any (z_thresh, horizon) cell with
  - n >= 100 events
  - mean_net > 0.5% per event (after 0.16% RT taker)
  - t-stat > 2.0
  - hit rate > 55%

Concede: no edge in funding extremes net of cost.
"""
import pandas as pd
import numpy as np
import requests
import time
import math
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TAKER_RT = 0.0016  # 0.08% per side × 2
ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "funding_extreme_overlay.log"

# Universe: active USDT perps with long history
UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
    "APTUSDT", "ARBUSDT", "ATOMUSDT", "AAVEUSDT", "BCHUSDT",
    "DOTUSDT", "ETCUSDT", "FILUSDT", "HBARUSDT", "ICPUSDT",
    "INJUSDT", "NEARUSDT", "OPUSDT", "RENDERUSDT", "SUIUSDT",
    "TRXUSDT", "UNIUSDT", "WIFUSDT", "WLDUSDT", "PEPEUSDT",
]

FAPI = "https://fapi.binance.com"


def fetch_funding_history(symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict]:
    """Binance returns up to 1000 funding events per call."""
    all_events = []
    cursor = start_ms
    while cursor < end_ms:
        try:
            r = requests.get(f"{FAPI}/fapi/v1/fundingRate",
                             params={"symbol": symbol, "startTime": cursor,
                                     "endTime": end_ms, "limit": limit},
                             timeout=15)
            if r.status_code != 200:
                break
            data = r.json()
        except Exception:
            break
        if not data:
            break
        all_events.extend(data)
        last_time = int(data[-1]["fundingTime"])
        if last_time <= cursor or len(data) < limit:
            break
        cursor = last_time + 1
        time.sleep(0.1)
    return all_events


def fetch_klines(symbol: str, start_ms: int, end_ms: int, interval: str = "1h") -> list[list]:
    """Fetch 1h klines for return measurement."""
    out = []
    cursor = start_ms
    per_bar = 3600 * 1000
    while cursor < end_ms:
        try:
            r = requests.get(f"{FAPI}/fapi/v1/klines",
                             params={"symbol": symbol, "interval": interval,
                                     "startTime": cursor, "endTime": end_ms, "limit": 1500},
                             timeout=15)
            if r.status_code != 200:
                break
            data = r.json()
        except Exception:
            break
        if not data:
            break
        out.extend(data)
        last_open = data[-1][0]
        if last_open + per_bar <= cursor or len(data) < 1500:
            break
        cursor = last_open + per_bar
        time.sleep(0.1)
    return out


def main():
    # Window: 2024-01 to 2026-04
    start_ms = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp("2026-04-15", tz="UTC").timestamp() * 1000)

    all_events = []  # will be (symbol, ts_ms, rate, z_score, fwd_8h, fwd_24h, fwd_48h, fwd_72h)

    for sym in UNIVERSE:
        print(f"[info] fetching {sym}...", flush=True)
        t0 = time.time()
        fundings = fetch_funding_history(sym, start_ms, end_ms)
        if len(fundings) < 200:
            print(f"  [skip] only {len(fundings)} funding events")
            continue

        # Build funding series
        f_df = pd.DataFrame(fundings)
        f_df["ts"] = f_df["fundingTime"].astype("int64")   # Windows default int is int32 -> overflow
        f_df["rate"] = f_df["fundingRate"].astype(float)
        f_df = f_df.sort_values("ts").reset_index(drop=True)

        # Rolling 30-day z-score (30*3=90 events = 30 days of 8h funding)
        f_df["rolling_mean"] = f_df["rate"].rolling(90, min_periods=30).mean()
        f_df["rolling_std"] = f_df["rate"].rolling(90, min_periods=30).std()
        f_df["z"] = (f_df["rate"] - f_df["rolling_mean"]) / f_df["rolling_std"].replace(0, np.nan)
        f_df = f_df.dropna(subset=["z"])

        # Fetch 1h klines for the same window
        klines = fetch_klines(sym, start_ms, end_ms)
        if len(klines) < 500:
            print(f"  [skip] only {len(klines)} klines")
            continue
        k_df = pd.DataFrame(klines, columns=[
            "open_time", "o", "h", "l", "c", "v", "close_time", "qv", "n_trades",
            "tbbv", "tbqv", "ig"
        ])
        k_df["open_time"] = k_df["open_time"].astype("int64")   # Windows default int is int32 -> overflow
        k_df["close"] = k_df["c"].astype(float)
        k_df = k_df.sort_values("open_time").reset_index(drop=True)

        # For each funding event, find close at t, t+8h, t+24h, t+48h, t+72h
        for _, row in f_df.iterrows():
            t_ms = int(row["ts"])
            z = float(row["z"])
            # Index into klines: find kline with open_time >= t_ms
            idx = k_df["open_time"].searchsorted(t_ms, side="right") - 1
            if idx < 0 or idx >= len(k_df):
                continue
            p0 = k_df.iloc[idx]["close"]
            if p0 <= 0:
                continue

            fwd = {}
            for h_lab, h_bars in [("8h", 8), ("24h", 24), ("48h", 48), ("72h", 72)]:
                tgt_idx = idx + h_bars
                if tgt_idx >= len(k_df):
                    fwd[h_lab] = np.nan
                    continue
                p1 = k_df.iloc[tgt_idx]["close"]
                if p1 <= 0:
                    fwd[h_lab] = np.nan
                    continue
                fwd[h_lab] = (p1 / p0) - 1.0

            all_events.append((sym, t_ms, z, fwd["8h"], fwd["24h"], fwd["48h"], fwd["72h"]))
        print(f"  [ok] {len(f_df)} events, elapsed {time.time()-t0:.0f}s")

    if not all_events:
        print("[concede] no data")
        return 1

    ev_df = pd.DataFrame(all_events, columns=["sym", "ts", "z", "r_8h", "r_24h", "r_48h", "r_72h"])
    ev_df.to_csv(ROOT / "logs" / "funding_extreme_events.csv", index=False)
    print(f"\n[info] {len(ev_df)} total events across {ev_df['sym'].nunique()} assets")
    print(f"       z distribution: mean={ev_df['z'].mean():.3f}, std={ev_df['z'].std():.3f}, "
          f"p5={ev_df['z'].quantile(0.05):.2f}, p95={ev_df['z'].quantile(0.95):.2f}")

    # Test: z > Z_THRESH -> short (negative exposure), z < -Z_THRESH -> long
    print("\n" + "=" * 90)
    print(f"{'thresh':>7} {'side':>5} {'horizon':>8} {'n':>6} {'mean_gross':>11} {'mean_net':>11} {'std':>8} {'t':>6} {'hit':>6} verdict")
    print("=" * 90)

    ship = []
    for z_thresh in [1.0, 1.5, 2.0, 2.5, 3.0]:
        for h_col in ["r_8h", "r_24h", "r_48h", "r_72h"]:
            horizon = h_col.replace("r_", "")
            # SHORT on z > +thresh (long-crowded, expect down)
            mask_short = (ev_df["z"] > z_thresh) & ev_df[h_col].notna()
            rets_short = -ev_df.loc[mask_short, h_col].values  # short = negative return
            # LONG on z < -thresh (short-crowded, expect up)
            mask_long = (ev_df["z"] < -z_thresh) & ev_df[h_col].notna()
            rets_long = ev_df.loc[mask_long, h_col].values

            for side_name, rets in [("SHORT", rets_short), ("LONG", rets_long)]:
                if len(rets) < 30:
                    continue
                n = len(rets)
                mean_g = rets.mean()
                mean_n = mean_g - TAKER_RT
                std = rets.std(ddof=1) if n > 1 else 1e-9
                std = max(std, 1e-9)
                t_stat = (mean_n * math.sqrt(n)) / std
                hit = float((rets > TAKER_RT).mean())
                flag = ""
                if n >= 100 and mean_n > 0.005 and t_stat > 2.0 and hit > 0.55:
                    flag = "  <<SHIP"
                    ship.append((z_thresh, side_name, horizon, n, mean_n, t_stat, hit))
                print(f"{z_thresh:>6.1f}  {side_name:>5} {horizon:>7} {n:>6} "
                      f"{mean_g*100:>+9.2f}%  {mean_n*100:>+9.2f}%  {std*100:>6.2f}% "
                      f"{t_stat:>5.2f}  {hit*100:>4.1f}%{flag}")
        print()

    print("=" * 90)
    if ship:
        print(f"[SHIP] {len(ship)} cells pass strict criteria:")
        for z, side, hor, n, m, t, hit in ship:
            print(f"  z>{z} {side} {hor}: n={n} mean_net={m*100:+.2f}% t={t:.2f} hit={hit*100:.1f}%")
        return 0
    else:
        # Find best cell for reporting
        rows = []
        for z_thresh in [1.0, 1.5, 2.0, 2.5, 3.0]:
            for h_col in ["r_8h", "r_24h", "r_48h", "r_72h"]:
                mask_short = (ev_df["z"] > z_thresh) & ev_df[h_col].notna()
                rets_short = -ev_df.loc[mask_short, h_col].values
                mask_long = (ev_df["z"] < -z_thresh) & ev_df[h_col].notna()
                rets_long = ev_df.loc[mask_long, h_col].values
                for side_name, rets in [("SHORT", rets_short), ("LONG", rets_long)]:
                    if len(rets) < 30:
                        continue
                    mean_n = rets.mean() - TAKER_RT
                    std = rets.std(ddof=1)
                    t = (mean_n * math.sqrt(len(rets))) / max(std, 1e-9)
                    rows.append((z_thresh, side_name, h_col, len(rets), mean_n, t))
        if rows:
            rows.sort(key=lambda r: r[5], reverse=True)
            print(f"[CONCEDE] best cell below ship threshold:")
            for z, side, h, n, m, t in rows[:3]:
                print(f"  z>{z} {side} {h}: n={n} mean_net={m*100:+.2f}% t={t:.2f}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
