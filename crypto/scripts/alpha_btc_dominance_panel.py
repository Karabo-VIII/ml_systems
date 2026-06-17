"""Alpha turn-007: BTC-dominance (BTC.D) daily panel builder.

Computes BTC.D proxy = BTC mcap / total mcap, where mcap is approximated via
Binance price × circulating-supply proxy. Since we lack on-chain supply data
cleanly, we build a *relative-strength proxy*: BTC cumulative return vs
equal-weight ALT basket (9 alts of project universe).

BTC.D is then computed as: BTC_return_idx / alt_basket_return_idx, rebased
to 100 at start. A rising series = BTC outperforming = BTC-leadership regime.

Also computes:
  - 30d / 90d SMA + regime label (UP/DOWN/SIDEWAYS)
  - 7d / 30d relative-strength differential

Unblocks Bravo's 30-min BTC.D probe. Output = daily CSV.

NO NEW INFRA — anonymous Binance klines (same cache as cycle_gate).
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "logs" / "frontier" / "btc_dominance"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = ROOT / "logs" / "frontier" / "cycle_gate"  # reuse BTC cache dir
BINANCE_URL = "https://api.binance.com/api/v3/klines"

# Project universe alts (from CLAUDE.md — 10 assets total; BTC is baseline)
ALTS = ["ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
        "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]


def fetch_klines(symbol: str, start: str = "2020-01-01") -> pd.DataFrame:
    cache = CACHE_DIR / f"{symbol.lower()}_daily_klines.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    all_rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        url = (f"{BINANCE_URL}?symbol={symbol}&interval=1d&startTime={cursor}"
               f"&endTime={end_ms}&limit=1000")
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                rows = json.loads(r.read().decode())
        except Exception as e:
            print(f"[WARN] {symbol} fetch err at {cursor}: {e}")
            break
        if not rows:
            break
        all_rows.extend(rows)
        last_ts = rows[-1][0]
        if last_ts <= cursor:
            break
        cursor = last_ts + 86_400_000
        if len(rows) < 1000:
            break
        time.sleep(0.1)
    cols = ["open_ts", "open", "high", "low", "close", "volume",
            "close_ts", "qav", "n_trades", "tb_bav", "tb_qav", "ignore"]
    df = pd.DataFrame(all_rows, columns=cols)
    df["date"] = pd.to_datetime(df["open_ts"], unit="ms").dt.tz_localize(None).dt.normalize()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df[["date", "close"]].drop_duplicates("date").reset_index(drop=True)
    df.to_parquet(cache)
    print(f"[CACHE] wrote {cache} ({len(df)} rows)")
    return df


def main() -> None:
    btc = fetch_klines("BTCUSDT").rename(columns={"close": "btc_close"})
    print(f"[BTC] {len(btc)} days")
    alt_series = []
    for s in ALTS:
        d = fetch_klines(s).rename(columns={"close": s.lower().replace("usdt", "")})
        alt_series.append(d)

    # Align all to BTC dates
    panel = btc.copy()
    for a in alt_series:
        panel = panel.merge(a, on="date", how="left")
    panel = panel.sort_values("date").reset_index(drop=True)

    # Compute daily returns
    alt_cols = [c for c in panel.columns if c not in ("date", "btc_close")]
    panel["btc_ret"] = panel["btc_close"].pct_change().fillna(0.0)
    for c in alt_cols:
        panel[f"{c}_ret"] = panel[c].pct_change().fillna(0.0)
    alt_ret_cols = [f"{c}_ret" for c in alt_cols]
    panel["alt_basket_ret"] = panel[alt_ret_cols].mean(axis=1)

    # Relative strength: BTC return minus alt basket return
    panel["btc_vs_alt_diff"] = panel["btc_ret"] - panel["alt_basket_ret"]
    # Cumulative index for both
    panel["btc_idx"] = (1.0 + panel["btc_ret"]).cumprod() * 100.0
    panel["alt_idx"] = (1.0 + panel["alt_basket_ret"]).cumprod() * 100.0
    panel["btc_d_proxy"] = panel["btc_idx"] / panel["alt_idx"]

    # Trend / regime classification
    panel["btc_d_sma30"] = panel["btc_d_proxy"].rolling(30, min_periods=10).mean()
    panel["btc_d_sma90"] = panel["btc_d_proxy"].rolling(90, min_periods=20).mean()
    panel["btc_d_chg_30d"] = panel["btc_d_proxy"].pct_change(30)
    # BTC-LEADERSHIP = dominance rising over 30d
    # ALT-SEASON = dominance falling over 30d
    panel["regime"] = "SIDEWAYS"
    panel.loc[panel["btc_d_chg_30d"] > 0.03, "regime"] = "BTC_LEADERSHIP"
    panel.loc[panel["btc_d_chg_30d"] < -0.03, "regime"] = "ALT_SEASON"

    # Persist
    out_csv = OUT_DIR / "btc_dominance_daily.csv"
    keep_cols = [
        "date", "btc_close", "btc_ret", "alt_basket_ret", "btc_vs_alt_diff",
        "btc_idx", "alt_idx", "btc_d_proxy", "btc_d_sma30", "btc_d_sma90",
        "btc_d_chg_30d", "regime",
    ]
    panel[keep_cols].to_csv(out_csv, index=False)
    print(f"[SAVE] {out_csv}")
    print()
    print(f"[REGIME] distribution (all {len(panel)} days):")
    print(panel["regime"].value_counts())
    print()
    # Restrict to blend window + print regime per sleeve uplift hint
    window = panel[(panel["date"] >= "2025-01-01") & (panel["date"] <= "2026-04-22")]
    print(f"[BLEND WINDOW] {len(window)} days")
    print(window["regime"].value_counts())
    print()
    print(f"[PREVIEW] last 5 days:")
    print(window[["date", "btc_d_proxy", "btc_d_chg_30d", "regime"]].tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
