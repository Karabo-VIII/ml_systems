"""
coinbase_effect_probe.py -- Wave 7 feasibility probe for the Coinbase listing effect.

Source for listing dates: Coinbase Exchange candle API (earliest 6h candle per product,
walking forward in 75-day chunks from the symbol's Binance first date).

Cross-references with our Binance chimera-1d price panel and runs a chronological event study:
abnormal return in event window [-5, +10] days around the Coinbase listing date.

Long-only spot feasibility: can you buy on Binance the day Coinbase lists a token
and capture upside, net of 20bps cost?
"""

import sys, json, time, requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone

CRYPTO_ROOT = Path(r"C:\Users\karab\Documents\coding\ml_systems\crypto")
DATA_ROOT   = CRYPTO_ROOT / "data"
CHIMERA_1D  = DATA_ROOT / "processed" / "chimera" / "1d"

out_dir = CRYPTO_ROOT / "logs" / "frontier" / "coinbase_effect"
out_dir.mkdir(parents=True, exist_ok=True)

EXCHANGE_BASE = "https://api.exchange.coinbase.com"
GRAN_6H    = 21600   # 6-hour bars -- largest supported by Exchange API
CHUNK_DAYS = 75      # 300 bars x 6h = 75 days per request

# ── 1. LOAD BINANCE DAILY CLOSE PANEL ──────────────────────────────────────────
print("=== Step 1: Loading Binance daily close panel from chimera/1d ===")

close_frames = {}
latest_file_per_sym = {}

for fpath in sorted(CHIMERA_1D.glob("*_v51_chimera_1d_*.parquet")):
    stem      = fpath.stem
    sym_usdt  = stem.split("_v51_")[0].upper()
    sym       = sym_usdt.replace("USDT", "").replace("BUSD", "").replace("1000", "")
    if not sym:
        continue
    file_date = stem.split("_")[-1]
    if sym in latest_file_per_sym and file_date <= latest_file_per_sym[sym]:
        continue
    latest_file_per_sym[sym] = file_date

    try:
        df     = pd.read_parquet(fpath, columns=["timestamp", "close"])
        ts     = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.normalize()
        series = pd.Series(df["close"].values, index=ts, name=sym)
        series = series[~series.index.duplicated(keep="last")]
        if len(series) < 30:
            continue
        close_frames[sym] = series
    except Exception as e:
        print(f"  Warning: {fpath.name}: {e}")

klines = pd.DataFrame(close_frames).sort_index()
print(f"Price panel: {klines.shape[1]} symbols, {klines.shape[0]} days")
print(f"Date range: {klines.index.min().date()} to {klines.index.max().date()}")
binance_syms = set(klines.columns)

# Earliest Binance date per symbol (as UTC datetime, 30 days lookback for search start)
bnb_first = {}
for sym in klines.columns:
    first_ts = klines[sym].dropna().index[0]
    # Convert pandas Timestamp to python datetime
    dt = first_ts.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    bnb_first[sym] = dt - timedelta(days=30)

# ── 2. FETCH COINBASE EXCHANGE PRODUCT LIST ────────────────────────────────────
print("\n=== Step 2: Fetching Coinbase Exchange product list ===")

r = requests.get(f"{EXCHANGE_BASE}/products", timeout=15)
r.raise_for_status()
all_products = r.json()
print(f"Total Exchange products: {len(all_products)}")

# Keep USD-quoted, online, non-disabled; one product per base currency
usd_products = [
    p for p in all_products
    if p.get("quote_currency") in ("USD", "USDT", "USDC")
    and p.get("status") == "online"
    and not p.get("trading_disabled", False)
]
base_to_product = {}
for p in usd_products:
    base     = p["base_currency"].upper()
    priority = {"USD": 0, "USDT": 1, "USDC": 2}.get(p["quote_currency"], 9)
    if base not in base_to_product or priority < base_to_product[base]["_priority"]:
        base_to_product[base] = {**p, "_priority": priority}

# Only keep tokens in our Binance panel
cb_overlap = {k: v for k, v in base_to_product.items() if k in binance_syms}
print(f"Coinbase products overlapping with Binance panel: {len(cb_overlap)}")

# ── 3. FIND FIRST CANDLE DATE PER PRODUCT ─────────────────────────────────────
print("\n=== Step 3: Finding first Coinbase listing date (6h candles, 75d chunks) ===")


def find_first_candle(product_id: str, search_from: datetime) -> datetime | None:
    """Walk forward in 75-day windows of 6h candles to find the earliest available bar."""
    url        = f"{EXCHANGE_BASE}/products/{product_id}/candles"
    search_end = datetime(2026, 9, 1, tzinfo=timezone.utc)
    cursor     = search_from if search_from.tzinfo else search_from.replace(tzinfo=timezone.utc)

    while cursor < search_end:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS), search_end)
        params = {
            "granularity": GRAN_6H,
            "start": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end":   chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        try:
            resp = requests.get(url, params=params, timeout=12)
            if resp.status_code == 200:
                candles = resp.json()
                if candles and isinstance(candles, list):
                    earliest_ts = min(c[0] for c in candles)
                    return datetime.utcfromtimestamp(earliest_ts).replace(tzinfo=timezone.utc)
                cursor = chunk_end   # no candles in this window, advance
            elif resp.status_code == 429:
                time.sleep(3.0)
                continue             # retry same window
            elif resp.status_code == 404:
                return None
            else:
                cursor = chunk_end   # skip on other errors
        except Exception:
            cursor = chunk_end
        time.sleep(0.12)

    return None


listing_records = []
items = sorted(cb_overlap.items())
n     = len(items)

for i, (sym, p) in enumerate(items):
    pid        = p["id"]
    search_from = bnb_first.get(sym, datetime(2020, 1, 1, tzinfo=timezone.utc))
    first_date  = find_first_candle(pid, search_from)
    status      = str(first_date.date()) if first_date else "NOT_FOUND"
    if (i + 1) % 10 == 0 or i < 5:
        print(f"  [{i+1:3d}/{n}] {sym:12s} ({pid:20s}): {status}")
    if first_date:
        listing_records.append({
            "symbol":                sym,
            "product_id":            pid,
            "coinbase_first_candle": first_date,
        })

cb_dates_df = pd.DataFrame(listing_records)
print(f"\nProducts with valid first-candle dates: {len(cb_dates_df)}")
if not cb_dates_df.empty:
    print(cb_dates_df.sort_values("coinbase_first_candle").head(20).to_string(index=False))
cb_dates_df.to_parquet(out_dir / "coinbase_listing_dates.parquet", index=False)
print(f"Saved to {out_dir / 'coinbase_listing_dates.parquet'}")

if cb_dates_df.empty:
    print("ERROR: no listing dates found. Exiting.")
    sys.exit(1)

# ── 4. EVENTS ─────────────────────────────────────────────────────────────────
print("\n=== Step 4: Events ===")

events = (
    cb_dates_df
    .sort_values("coinbase_first_candle")
    .drop_duplicates(subset="symbol", keep="first")
    .reset_index(drop=True)
)
print(f"Events: {len(events)}")
print(events[["symbol", "product_id", "coinbase_first_candle"]].to_string(index=False))

# ── 5. EVENT STUDY ─────────────────────────────────────────────────────────────
print("\n=== Step 5: Event study ===")

COST_RT = 0.0020  # 20bps round-trip taker

results = []

for _, row in events.iterrows():
    sym      = row["symbol"]
    evt_date = pd.Timestamp(row["coinbase_first_candle"]).normalize()

    if sym not in klines.columns:
        continue

    prices    = klines[sym].dropna()
    all_dates = prices.index
    if len(prices) < 20:
        continue

    # Nearest bar <= evt_date
    ei = int(np.searchsorted(all_dates, evt_date, side="right")) - 1
    if ei < 0:
        continue
    # Skip if nearest bar is more than 5 calendar days away
    if abs((all_dates[ei] - evt_date).days) > 5:
        continue
    if ei < 5 or ei + 10 >= len(prices):
        continue

    evt_p = float(prices.iloc[ei])
    if evt_p <= 0:
        continue

    def fwd(n):
        return float(prices.iloc[ei + n]) / evt_p - 1 - COST_RT / 2

    pre_5 = float(evt_p / prices.iloc[ei - 5] - 1)
    pre_1 = float(evt_p / prices.iloc[ei - 1] - 1)

    first_bnb         = prices.index[0]
    days_cb_after_bnb = int((evt_date - first_bnb).days)

    results.append({
        "symbol":                sym,
        "coinbase_listing":      str(evt_date.date()),
        "binance_first":         str(first_bnb.date()),
        "cb_after_binance_days": days_cb_after_bnb,
        "pre_5d":  round(pre_5,   4),
        "pre_1d":  round(pre_1,   4),
        "fwd_1d":  round(fwd(1),  4),
        "fwd_3d":  round(fwd(3),  4),
        "fwd_5d":  round(fwd(5),  4),
        "fwd_7d":  round(fwd(7),  4),
        "fwd_10d": round(fwd(10), 4),
    })

res_df = pd.DataFrame(results)
if res_df.empty:
    print("No events with sufficient price data -- aborting.")
    sys.exit(1)

print(f"\nEvents with sufficient price data: {len(res_df)}")

print(f"\nCoinbase-after-Binance timing:")
print(f"  Median days CB after Binance:      {res_df['cb_after_binance_days'].median():.0f}")
print(f"  % CB before/same as Binance:       {(res_df['cb_after_binance_days'] <= 0).mean()*100:.1f}%")
print(f"  % CB 1-30d after Binance:          {(res_df['cb_after_binance_days'].between(1, 30)).mean()*100:.1f}%")
print(f"  % CB 31-180d after Binance:        {(res_df['cb_after_binance_days'].between(31, 180)).mean()*100:.1f}%")
print(f"  % CB 181d+ after Binance:          {(res_df['cb_after_binance_days'] > 180).mean()*100:.1f}%")

print(f"\n--- Returns around Coinbase listing day ---")
for label, col in [
    ("Pre-5d run-up", "pre_5d"),
    ("Pre-1d run-up", "pre_1d"),
    ("Fwd 1d (net)",  "fwd_1d"),
    ("Fwd 3d (net)",  "fwd_3d"),
    ("Fwd 5d (net)",  "fwd_5d"),
    ("Fwd 7d (net)",  "fwd_7d"),
    ("Fwd 10d (net)", "fwd_10d"),
]:
    s = res_df[col].dropna()
    if len(s) < 2:
        continue
    t = s.mean() / s.std() * np.sqrt(len(s)) if s.std() > 0 else 0
    print(f"  {label:20s}: mean={s.mean()*100:7.1f}%  med={s.median()*100:6.1f}%  "
          f"hit={(s>0).mean()*100:4.0f}%  t={t:5.2f}  n={len(s)}")

# ── 6. BTC-ADJUSTED ABNORMAL RETURN ────────────────────────────────────────────
print("\n=== Step 6: BTC-adjusted abnormal return ===")
if "BTC" in klines.columns:
    btc     = klines["BTC"]
    btc_rows = []
    for _, row in res_df.iterrows():
        evt_date = pd.Timestamp(row["coinbase_listing"], tz="UTC")
        ei       = int(np.searchsorted(klines.index, evt_date, side="right")) - 1
        if ei < 1 or ei + 10 >= len(btc):
            continue
        btc_p = float(btc.iloc[ei])
        if btc_p <= 0:
            continue
        btc_rows.append({
            "symbol":     row["symbol"],
            "btc_fwd_1d":  float(btc.iloc[ei + 1])  / btc_p - 1,
            "btc_fwd_3d":  float(btc.iloc[ei + 3])  / btc_p - 1,
            "btc_fwd_5d":  float(btc.iloc[ei + 5])  / btc_p - 1,
            "btc_fwd_7d":  float(btc.iloc[ei + 7])  / btc_p - 1,
            "btc_fwd_10d": float(btc.iloc[ei + 10]) / btc_p - 1,
        })
    btc_df = pd.DataFrame(btc_rows)
    merged = res_df.merge(btc_df, on="symbol", how="inner")
    print(f"Events with BTC benchmark: {len(merged)}")
    for h in ["1d", "3d", "5d", "7d", "10d"]:
        ab = merged[f"fwd_{h}"] - merged[f"btc_fwd_{h}"]
        t  = ab.mean() / ab.std() * np.sqrt(len(ab)) if ab.std() > 0 else 0
        print(f"  Abnormal fwd_{h}: mean={ab.mean()*100:7.1f}%  med={ab.median()*100:6.1f}%  "
              f"hit={(ab>0).mean()*100:4.0f}%  t={t:5.2f}")
else:
    print("BTC not in panel")

# ── 7. CHRONOLOGICAL SPLIT ─────────────────────────────────────────────────────
print("\n=== Step 7: Chronological split ===")
res_df["listing_dt_parsed"] = pd.to_datetime(res_df["coinbase_listing"])
splits = [
    ("TRAIN (<2022)", res_df[res_df["listing_dt_parsed"] < "2022-01-01"]),
    ("TRAIN (2022)",  res_df[res_df["listing_dt_parsed"].between("2022-01-01", "2022-12-31")]),
    ("VAL (2023)",    res_df[res_df["listing_dt_parsed"].between("2023-01-01", "2023-12-31")]),
    ("OOS (2024+)",   res_df[res_df["listing_dt_parsed"] >= "2024-01-01"]),
]
for era, era_df in splits:
    if len(era_df) < 3:
        print(f"  {era}: n={len(era_df)} (too few)")
        continue
    print(f"  {era}: n={len(era_df)}")
    for h in ["1d", "3d", "7d", "10d"]:
        s = era_df[f"fwd_{h}"].dropna()
        t = s.mean() / s.std() * np.sqrt(len(s)) if s.std() > 0 else 0
        print(f"    fwd_{h}: mean={s.mean()*100:7.1f}%  hit={(s>0).mean()*100:4.0f}%  t={t:5.2f}")

# ── 8. FULL EVENT TABLE ────────────────────────────────────────────────────────
print("\n--- Full event table ---")
cols = ["symbol", "coinbase_listing", "binance_first", "cb_after_binance_days",
        "pre_5d", "pre_1d", "fwd_1d", "fwd_3d", "fwd_7d", "fwd_10d"]
print(res_df.sort_values("coinbase_listing")[cols].to_string(index=False))

# ── 9. SAVE ────────────────────────────────────────────────────────────────────
res_df.to_parquet(out_dir / "event_study_results.parquet", index=False)
res_df.to_csv(out_dir / "event_study_results.csv", index=False)

summary = {
    "n_events":                      int(len(res_df)),
    "n_cb_with_candle_dates":        int(len(cb_dates_df)),
    "median_days_cb_after_binance":  float(res_df["cb_after_binance_days"].median()),
    "pct_cb_before_binance":         float((res_df["cb_after_binance_days"] <= 0).mean()),
    "fwd_1d_mean":  float(res_df["fwd_1d"].mean()),
    "fwd_3d_mean":  float(res_df["fwd_3d"].mean()),
    "fwd_7d_mean":  float(res_df["fwd_7d"].mean()),
    "fwd_10d_mean": float(res_df["fwd_10d"].mean()),
    "fwd_1d_hit":   float((res_df["fwd_1d"] > 0).mean()),
    "fwd_3d_hit":   float((res_df["fwd_3d"] > 0).mean()),
    "fwd_7d_hit":   float((res_df["fwd_7d"] > 0).mean()),
    "fwd_10d_hit":  float((res_df["fwd_10d"] > 0).mean()),
}
with open(out_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n=== PROBE COMPLETE ===")
print(f"Outputs: {out_dir}")
print(json.dumps(summary, indent=2))
