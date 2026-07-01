"""Economic-calendar ingester — high-impact FX events (ForexFactory, keyless).

FX is heavily event-driven (NFP, FOMC, ECB, CPI, rate decisions). This fetches
the ForexFactory weekly JSON feed (impact-rated High/Medium/Low, no API key) and
stores it, plus a helper to flag bars near high-impact events for the pair's
currencies.

(Named econ_calendar, NOT calendar: a module named calendar.py shadows Python's
stdlib `calendar` when run as a script and breaks `requests`/`http.cookiejar`.)

LIMITATION: the free FF feed is CURRENT-WEEK ONLY (there are month/next-week
variants). For a live/forward pipeline this is exactly right -- fetch weekly and
cache (respect the ~2-downloads-per-5-min limit). For HISTORICAL event backfill,
use the FRED releases API (scheduled US macro release dates) instead.

Output: data/processed/calendar/events_<YYYYMMDD>.parquet (idempotent append).

    python -m pipeline.ingest.econ_calendar          # fetch this week + print high-impact
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import requests

_PIPELINE_PARENT = str(Path(__file__).resolve().parents[2])
if _PIPELINE_PARENT not in sys.path:
    sys.path.insert(0, _PIPELINE_PARENT)

from pipeline import layout
from pipeline.parquet_io import atomic_write_parquet, append_parquet

__contract__ = {
    "kind": "pipeline_stage",
    "stage": "fetch_calendar",
    "inputs": {"source": "https://nfs.faireconomy.media/ff_calendar_thisweek.json"},
    "outputs": {"files": "data/processed/calendar/events_<YYYYMMDD>.parquet",
                "columns": ["timestamp", "currency", "title", "impact", "forecast", "previous"]},
    "invariants": {"keyless": True, "current_week_only": True,
                   "impact_rated": True, "atomic_write": True},
    "rationale": "Event flags for bars; FX is event-driven (NFP/FOMC/ECB/CPI).",
}

FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
CAL_DIR = layout.PROCESSED / "calendar"


def fetch_week() -> pl.DataFrame:
    """Fetch + normalize the current-week ForexFactory calendar."""
    r = requests.get(FEED_URL, timeout=30, headers={"User-Agent": "forex-pipeline/1.0"})
    r.raise_for_status()
    events = r.json()
    rows = []
    for e in events:
        # FF uses the currency code in the "country" field (USD, EUR, ...).
        ccy = (e.get("country") or e.get("currency") or "").upper()
        dt_raw = e.get("date") or ""
        try:
            ts = int(datetime.fromisoformat(dt_raw).astimezone(timezone.utc).timestamp() * 1000)
        except (ValueError, TypeError):
            continue
        rows.append({
            "timestamp": ts, "currency": ccy, "title": e.get("title", ""),
            "impact": (e.get("impact") or "").capitalize(),
            "forecast": str(e.get("forecast", "")), "previous": str(e.get("previous", "")),
        })
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort("timestamp").unique(
        subset=["timestamp", "currency", "title"], keep="first")


def event_flags(bars: pl.DataFrame, events: pl.DataFrame, currencies: list[str],
                window_min: int = 60) -> pl.DataFrame:
    """Add is_high_impact_event (within +-window_min of a High event for any of
    `currencies`) and mins_to_next_high_event to a bar frame. The event TIME is a
    KNOWN schedule (not future price), so it is safe as a conditioning feature."""
    hi = events.filter((pl.col("impact") == "High")
                       & (pl.col("currency").is_in([c.upper() for c in currencies])))
    if hi.is_empty():
        return bars.with_columns([pl.lit(0).alias("is_high_impact_event"),
                                  pl.lit(None, dtype=pl.Float64).alias("mins_to_next_high_event")])
    ev_ts = sorted(hi["timestamp"].to_list())
    w = window_min * 60_000
    flags, mins_next = [], []
    for t in bars["timestamp"].to_list():
        flags.append(1 if any(abs(t - et) <= w for et in ev_ts) else 0)
        future = [et for et in ev_ts if et >= t]
        mins_next.append((future[0] - t) / 60_000 if future else None)
    return bars.with_columns([
        pl.Series("is_high_impact_event", flags, dtype=pl.Int8),
        pl.Series("mins_to_next_high_event", mins_next, dtype=pl.Float64),
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch ForexFactory econ calendar (keyless)")
    ap.add_argument("--min-impact", default="High", choices=["High", "Medium", "Low"])
    args = ap.parse_args()
    try:
        df = fetch_week()
    except requests.RequestException as e:
        print(f"[calendar] fetch FAIL: {type(e).__name__}: {e} "
              f"(note: ~2 downloads / 5 min limit)", flush=True)
        sys.exit(1)
    if df.is_empty():
        print("[calendar] no events parsed", flush=True)
        sys.exit(1)
    latest = datetime.fromtimestamp(df["timestamp"].max() / 1000, tz=timezone.utc).date()
    out = CAL_DIR / f"events_{latest.strftime('%Y%m%d')}.parquet"
    if out.exists():
        append_parquet(out, df, date_col="timestamp", sort_col="timestamp")
    else:
        atomic_write_parquet(df, out, required_cols={"timestamp", "currency", "impact"})
    n_high = int((df["impact"] == "High").sum())
    print(f"[calendar] {len(df)} events this week ({n_high} High) -> "
          f"{out.relative_to(layout.PROJECT_ROOT)}", flush=True)
    for r in df.filter(pl.col("impact") == "High").head(12).iter_rows(named=True):
        t = datetime.fromtimestamp(r["timestamp"] / 1000, tz=timezone.utc)
        print(f"    {t.strftime('%a %H:%M UTC')}  {r['currency']:<4} {r['title']}", flush=True)


if __name__ == "__main__":
    main()
