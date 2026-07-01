"""Central-bank policy-rate ingester + carry proxy (FRED, keyless).

FX carry = the interest-rate differential between the two currencies in a pair.
This fetches each currency's short/policy rate from FRED's KEYLESS CSV endpoint
(no API key needed) and builds a forward-filled daily rate panel, from which the
carry of any pair = rate(base) - rate(quote) (annualized %, positive = long the
pair earns carry).

Any series that 404s or returns empty is skipped with a warning (graceful
degradation), so a wrong series ID in config/rates.yaml never crashes the run.

Output: data/processed/rates/rates_daily_<YYYYMMDD>.parquet (date + rate_<CCY>).

    python -m pipeline.ingest.rates            # fetch + build panel + print carry table
"""
from __future__ import annotations

import argparse
import io
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import requests
import yaml

_PIPELINE_PARENT = str(Path(__file__).resolve().parents[2])
if _PIPELINE_PARENT not in sys.path:
    sys.path.insert(0, _PIPELINE_PARENT)

from pipeline import layout
from pipeline.parquet_io import atomic_write_parquet

__contract__ = {
    "kind": "pipeline_stage",
    "stage": "fetch_rates",
    "inputs": {"config": "config/rates.yaml", "source": "FRED keyless CSV"},
    "outputs": {"files": "data/processed/rates/rates_daily_<YYYYMMDD>.parquet",
                "columns": ["date", "rate_<CCY> (annualized %)"]},
    "invariants": {"keyless_fred": True, "missing_series_skipped_not_fatal": True,
                   "forward_fill_daily_no_future_interp": True, "atomic_write": True},
    "rationale": "Carry = rate(base) - rate(quote); a core FX factor input.",
}

CONFIG_PATH = layout.PROJECT_ROOT / "config" / "rates.yaml"
RATES_DIR = layout.PROCESSED / "rates"


def _load_cfg() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_series(series_id: str, url_tmpl: str, session: requests.Session) -> pl.DataFrame | None:
    """Fetch one FRED series -> DataFrame(date, rate). None on failure/empty."""
    url = url_tmpl.format(series=series_id)
    try:
        r = session.get(url, timeout=30)
        if r.status_code != 200 or not r.text.strip():
            return None
        df = pl.read_csv(io.StringIO(r.text))
        # columns: observation_date, <SERIES_ID>. FRED uses "." for missing.
        val_col = [c for c in df.columns if c != "observation_date"]
        if not val_col:
            return None
        vc = val_col[0]
        df = df.select([
            pl.col("observation_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("date"),
            pl.col(vc).cast(pl.Utf8).str.strip_chars()
              .replace(".", None).cast(pl.Float64, strict=False).alias("rate"),
        ]).drop_nulls("date")
        return df.drop_nulls("rate") if len(df) else None
    except (requests.RequestException, Exception):
        return None


def build_rate_panel(start: date | None = None) -> tuple[pl.DataFrame, list[str]]:
    """Fetch every configured currency's rate and build a forward-filled daily panel."""
    cfg = _load_cfg()
    url_tmpl = cfg["source_url"]
    session = requests.Session()
    session.headers.update({"User-Agent": "forex-pipeline/1.0"})
    frames, got = [], []
    for ccy, sid in cfg["series"].items():
        s = fetch_series(sid, url_tmpl, session)
        if s is None or s.is_empty():
            print(f"[rates] SKIP {ccy} ({sid}): no data", flush=True)
            continue
        frames.append(s.rename({"rate": f"rate_{ccy}"}))
        got.append(ccy)
    if not frames:
        return pl.DataFrame(), []
    from functools import reduce
    wide = reduce(lambda a, b: a.join(b, on="date", how="full", coalesce=True), frames).sort("date")
    if start:
        wide = wide.filter(pl.col("date") >= start)
    # forward-fill: each rate steps and holds until the next announcement (causal;
    # never interpolate forward into a future decision).
    wide = wide.with_columns([pl.col(c).forward_fill() for c in wide.columns if c != "date"])
    return wide, got


def carry(pair: str, panel: pl.DataFrame) -> pl.Expr | None:
    """Carry(annualized %) = rate(base) - rate(quote), as a polars expr on `panel`."""
    n = layout.normalize_pair(pair)
    bc, qc = f"rate_{n.base}", f"rate_{n.quote}"
    if bc in panel.columns and qc in panel.columns:
        return (pl.col(bc) - pl.col(qc)).alias(f"carry_{n.flat}")
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch central-bank rates + carry (FRED, keyless)")
    ap.add_argument("--start", default="2015-01-01", help="Panel start date (YYYY-MM-DD).")
    args = ap.parse_args()
    panel, got = build_rate_panel(date.fromisoformat(args.start))
    if panel.is_empty():
        print("[rates] no series fetched", flush=True)
        sys.exit(1)
    latest = panel["date"].max()
    out = RATES_DIR / f"rates_daily_{latest.strftime('%Y%m%d')}.parquet"
    atomic_write_parquet(panel, out, required_cols={"date"})
    print(f"[rates] {len(got)} currencies: {got}", flush=True)
    print(f"[rates] panel {panel.shape} -> {out.relative_to(layout.PROJECT_ROOT)}", flush=True)
    # latest rates + a carry table for the majors
    last = panel.tail(1).to_dicts()[0]
    print("[rates] latest:", {k: last[k] for k in last if k.startswith("rate_")}, flush=True)
    majors = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD"]
    print("[rates] carry (annualized %, base-quote):", flush=True)
    for p in majors:
        e = carry(p, panel)
        if e is not None:
            v = panel.tail(1).select(e).row(0)[0]
            print(f"    {p}: {v:+.2f}%" if v is not None else f"    {p}: n/a", flush=True)


if __name__ == "__main__":
    main()
