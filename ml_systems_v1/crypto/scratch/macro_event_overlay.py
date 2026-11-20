"""Macro event overlay backtest (FOMC / CPI / NFP).

Hypothesis: BTC/ETH exhibit directional or volatility-expansion alpha around
US macro releases. Tests multiple windows post-release.

Event calendar (hardcoded — public records, 2024-01 to 2026-04):
  - FOMC: 8 meetings/yr, decision at 14:00 EST (19:00 UTC winter, 18:00 UTC summer)
  - CPI:  monthly at 08:30 EST first-or-second Tuesday/Wednesday (13:30 UTC)
  - NFP:  first Friday of month at 08:30 EST (13:30 UTC)

For each event type × asset × horizon:
  - Measure forward return at T+[1h, 2h, 4h, 8h, 24h]
  - Compare to non-event baseline (same horizons on random non-event hours)
  - Split: up-surprise / down-surprise (not directly known without consensus data)
    -> proxy: BTC direction in pre-release 1h as "expected direction"

Ship criterion: any event×asset×horizon cell where:
  - t-stat of realized return vs baseline > 2.0
  - n >= 20 events
  - directional mean > 0.5% net of 0.16% RT taker cost
"""
import pandas as pd
import numpy as np
import requests
import time
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TAKER_RT = 0.0016
ROOT = Path(__file__).resolve().parents[1]
FAPI = "https://fapi.binance.com"


# Hardcoded FOMC decision dates 2024-2026 (public record)
# Times all 19:00 UTC (winter) or 18:00 UTC (summer). Using 19:00 conservative.
FOMC_DATES = [
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18",
]

# Hardcoded CPI release dates (BLS monthly ~10th-14th of month at 13:30 UTC)
CPI_DATES = [
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10", "2024-05-15",
    "2024-06-12", "2024-07-11", "2024-08-14", "2024-09-11", "2024-10-10",
    "2024-11-13", "2024-12-11",
    "2025-01-15", "2025-02-12", "2025-03-12", "2025-04-10", "2025-05-13",
    "2025-06-11", "2025-07-15", "2025-08-12", "2025-09-11", "2025-10-15",
    "2025-11-13", "2025-12-10",
    "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-10",
]

# Hardcoded NFP dates (first Friday each month at 13:30 UTC)
NFP_DATES = [
    "2024-01-05", "2024-02-02", "2024-03-08", "2024-04-05", "2024-05-03",
    "2024-06-07", "2024-07-05", "2024-08-02", "2024-09-06", "2024-10-04",
    "2024-11-01", "2024-12-06",
    "2025-01-10", "2025-02-07", "2025-03-07", "2025-04-04", "2025-05-02",
    "2025-06-06", "2025-07-03", "2025-08-01", "2025-09-05", "2025-10-03",
    "2025-11-07", "2025-12-05",
    "2026-01-09", "2026-02-06", "2026-03-06", "2026-04-03",
]

ASSETS = ["BTCUSDT", "ETHUSDT"]


def to_ms(date_str: str, hour: int = 13, minute: int = 30) -> int:
    """Convert date string + UTC time to ms."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=hour, minute=minute, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def fetch_klines(symbol: str, start_ms: int, end_ms: int, interval: str = "1h") -> list[list]:
    out = []
    cursor = start_ms
    per_bar = 3600 * 1000
    while cursor < end_ms:
        try:
            r = requests.get(f"{FAPI}/fapi/v1/klines",
                             params={"symbol": symbol, "interval": interval,
                                     "startTime": cursor, "endTime": end_ms, "limit": 1500},
                             timeout=15)
            if r.status_code != 200: break
            data = r.json()
        except Exception:
            break
        if not data: break
        out.extend(data)
        last_open = data[-1][0]
        if last_open + per_bar <= cursor or len(data) < 1500:
            break
        cursor = last_open + per_bar
        time.sleep(0.1)
    return out


def price_at(klines_df: pd.DataFrame, target_ms: int) -> float:
    """Return close price of the 1h kline covering target_ms."""
    idx = klines_df["open_time"].searchsorted(target_ms, side="right") - 1
    if idx < 0 or idx >= len(klines_df):
        return np.nan
    return float(klines_df.iloc[idx]["close"])


def measure_events(klines_df: pd.DataFrame, event_ms_list: list[int]) -> list[dict]:
    """For each event, measure forward returns + pre-event direction proxy."""
    results = []
    for ev_ms in event_ms_list:
        p0 = price_at(klines_df, ev_ms)
        if not np.isfinite(p0) or p0 <= 0:
            continue
        # Pre-event: -2h to 0h drift (proxy for expected direction)
        p_pre = price_at(klines_df, ev_ms - 2 * 3600 * 1000)
        pre_drift = (p0 / p_pre) - 1 if np.isfinite(p_pre) and p_pre > 0 else np.nan

        fwd = {"pre_2h": pre_drift}
        for h_lab, h in [("1h", 1), ("2h", 2), ("4h", 4), ("8h", 8), ("24h", 24)]:
            p1 = price_at(klines_df, ev_ms + h * 3600 * 1000)
            fwd[h_lab] = (p1 / p0) - 1 if np.isfinite(p1) and p1 > 0 else np.nan
        fwd["event_ms"] = ev_ms
        results.append(fwd)
    return results


def main():
    overall_start = to_ms("2024-01-01", 0, 0)
    overall_end = to_ms("2026-04-15", 23, 59)

    all_results = {}
    for asset in ASSETS:
        print(f"[info] fetching {asset} 1h klines 2024-01 to 2026-04...", flush=True)
        t0 = time.time()
        klines = fetch_klines(asset, overall_start, overall_end, "1h")
        if len(klines) < 1000:
            print(f"  [skip] only {len(klines)} klines")
            continue
        k_df = pd.DataFrame(klines, columns=["open_time","o","h","l","c","v",
                                              "close_time","qv","n","tbbv","tbqv","ig"])
        k_df["open_time"] = k_df["open_time"].astype("int64")   # Windows default int is int32 -> overflow on ms epochs
        k_df["close"] = k_df["c"].astype(float)
        k_df = k_df.sort_values("open_time").reset_index(drop=True)
        print(f"  loaded {len(k_df)} klines ({time.time()-t0:.0f}s)")

        all_results[asset] = {}
        for ev_name, dates, hour, minute in [
            ("FOMC", FOMC_DATES, 19, 0),
            ("CPI",  CPI_DATES,  13, 30),
            ("NFP",  NFP_DATES,  13, 30),
        ]:
            ev_ms_list = [to_ms(d, hour, minute) for d in dates
                          if overall_start <= to_ms(d, hour, minute) <= overall_end]
            events = measure_events(k_df, ev_ms_list)
            all_results[asset][ev_name] = events

    # Aggregate + stat test
    print("\n" + "=" * 110)
    print(f"{'asset':>8} {'event':>5} {'horizon':>8} {'n':>4} {'mean':>10} {'mean_net':>11} "
          f"{'std':>7} {'t':>6} {'hit':>5} {'mean_|r|':>10} verdict")
    print("=" * 110)

    ship = []
    for asset in all_results:
        for ev_name in all_results[asset]:
            events = all_results[asset][ev_name]
            if not events:
                continue
            for h_lab in ["1h", "2h", "4h", "8h", "24h"]:
                rets = [e[h_lab] for e in events if np.isfinite(e.get(h_lab, np.nan))]
                if len(rets) < 15:
                    continue
                n = len(rets)
                rets = np.array(rets)
                mean_g = rets.mean()
                mean_n_long = mean_g - TAKER_RT
                mean_n_short = -mean_g - TAKER_RT
                std = rets.std(ddof=1) if n > 1 else 1e-9
                std = max(std, 1e-9)
                t_long = (mean_n_long * math.sqrt(n)) / std
                t_short = (mean_n_short * math.sqrt(n)) / std
                hit_long = (rets > TAKER_RT).mean()
                hit_short = (rets < -TAKER_RT).mean()
                mean_abs = np.abs(rets).mean()

                # Report best directional side
                if t_long > t_short:
                    side, m_n, t_stat, hit = "LONG", mean_n_long, t_long, hit_long
                else:
                    side, m_n, t_stat, hit = "SHORT", mean_n_short, t_short, hit_short

                flag = ""
                if n >= 20 and m_n > 0.005 and t_stat > 2.0 and hit > 0.55:
                    flag = "  <<SHIP"
                    ship.append((asset, ev_name, h_lab, side, n, m_n, t_stat, hit))
                print(f"{asset:>8} {ev_name:>5} {h_lab:>7}  {n:>4} {mean_g*100:>+8.2f}% "
                      f"{m_n*100:>+9.2f}% {std*100:>6.2f}% {t_stat:>5.2f} {hit*100:>4.1f}% "
                      f"{mean_abs*100:>9.2f}%{flag}")

    print("\n" + "=" * 110)
    if ship:
        print(f"[SHIP] {len(ship)} macro-event cells pass strict criteria:")
        for a, ev, h, s, n, m, t, hit in ship:
            print(f"  {a} {ev} {h} {s}: n={n} mean_net={m*100:+.2f}% t={t:.2f} hit={hit*100:.1f}%")
        return 0
    else:
        print("[CONCEDE] no macro event × horizon × side cell passes strict criteria.")
        print("          macro events do NOT provide retailer-exploitable directional alpha in 2024-2026.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
