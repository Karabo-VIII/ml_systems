"""External daily sources. Each row carries event_time_ms of the UTC day it describes.

The orchestrator joins each source with its publication lag from
``lags.families`` in ``config/pipeline.yaml`` (DefiLlama and Deribit one day,
Farside ETF flows and Wikipedia two days). Parsers accept fixtures and do not
require the network. Fetch functions record an error and return an empty frame
rather than raising on a bad response.

Each source is one file per source, and every fetch *merges* into it keyed by
(asset, day): a run over a short range adds or revises those days and never
drops the rest of the history. The newest fetch wins for a day it covers
(Farside revises recent rows).
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from pipeline.constants import DAY_MS
from pipeline.fetch_crypto import http_get
from pipeline.io import atomic_write_parquet
from pipeline.symbols import canonical_symbol

LAG_MS = DAY_MS

# First title is the current Wikipedia page. Earlier titles are fallbacks:
# the pageviews API 404s a new title for dates before the article was moved.
WIKI_ARTICLES = {
    "BTCUSDT": ("Bitcoin",),
    "ETHUSDT": ("Ethereum",),
    "SOLUSDT": ("Solana_(blockchain_platform)", "Solana"),
    "BNBUSDT": ("Binance",),
    "XRPUSDT": ("Ripple_(payment_protocol)",),
    "DOGEUSDT": ("Dogecoin",),
    "ADAUSDT": ("Cardano_(blockchain_platform)", "Cardano"),
    "AVAXUSDT": ("Avalanche_(blockchain_platform)", "Avalanche"),
    "LINKUSDT": ("Chainlink_(blockchain)", "Chainlink"),
    "LTCUSDT": ("Litecoin",),
}

ETF_URLS = {
    "BTCUSDT": "https://farside.co.uk/bitcoin-etf-flow-all-data/",
    "ETHUSDT": "https://farside.co.uk/ethereum-etf-flow-all-data/",
}

_ISSUER = re.compile(r"^[A-Z][A-Z0-9]{1,6}$")


def _midnight(day: date) -> int:
    dt = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _day_from_ms(ts_ms: int) -> date:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).date()


def _clean_header(cell: str) -> str:
    text = re.sub(r"\s+", " ", cell.replace("\xa0", " ")).strip()
    text = re.sub(r"\s+\d+(\.\d+)?%$", "", text)  # "ETHA 0.25%" -> "ETHA"
    text = text.replace(" ", "_")
    if text.lower().startswith("unnamed") or text.lower() in {"nan", "none"}:
        return ""
    if text.lower() in {"total", "total_usdm", "total_usd_m"}:
        return "total_usdm"
    return text


def _parse_flow_number(cell: str) -> float | None:
    text = cell.strip().replace(",", "").replace("\xa0", "")
    if text in {"", "-", "–", "—"}:
        return None
    neg = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    try:
        value = float(text)
    except ValueError:
        return None
    return -value if neg else value


def parse_farside_html(html: str, asset: str) -> pl.DataFrame:
    """One total column, issuer columns with cleaned names. No 'Unnamed' headers."""
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.I | re.S)
    header: list[str] | None = None
    out: list[dict] = []
    for row in rows:
        cells = [re.sub(r"<[^>]+>", "", c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.I | re.S)]
        cells = [re.sub(r"\s+", " ", c).strip() for c in cells]
        if not cells:
            continue
        if header is None and any(c.lower() == "date" or "total" in c.lower() for c in cells):
            header = [_clean_header(c) for c in cells]
            continue
        if header is None:
            continue
        # Date is the first non-empty cell.
        raw_date = cells[0]
        try:
            day = datetime.strptime(raw_date[:11].strip(), "%d %b %Y").date()
        except ValueError:
            try:
                day = datetime.strptime(raw_date[:10], "%Y-%m-%d").date()
            except ValueError:
                continue
        record: dict = {"asset": canonical_symbol(asset), "event_time_ms": _midnight(day)}
        for name, cell in zip(header, cells):
            if not name or name.lower() == "date":
                continue
            key = f"etf_{name.lower()}"
            record[key] = _parse_flow_number(cell)
        if "etf_total_usdm" not in record:
            record["etf_total_usdm"] = None
        out.append(record)
    if not out:
        return pl.DataFrame(schema={"asset": pl.Utf8, "event_time_ms": pl.Int64, "etf_total_usdm": pl.Float64})
    # Issuer columns mix ints, floats, and nulls across hundreds of rows.
    # Default inference (first 100 rows) then fails when a later cell is 18.0.
    value_cols = sorted({k for rec in out for k in rec if k not in {"asset", "event_time_ms"}})
    schema = {"asset": pl.Utf8, "event_time_ms": pl.Int64, **{c: pl.Float64 for c in value_cols}}
    df = pl.DataFrame(out, schema=schema).sort("event_time_ms")
    return df.unique(subset=["event_time_ms"], keep="last", maintain_order=True)


def parse_defillama_chart(payload: list | dict) -> pl.DataFrame:
    """Aggregate stablecoin circulating USD. Empty or odd payloads stay empty."""
    rows_in = payload if isinstance(payload, list) else payload.get("data", []) if isinstance(payload, dict) else []
    rows = []
    if not isinstance(rows_in, list):
        return pl.DataFrame(schema={"event_time_ms": pl.Int64, "stbl_supply_usd": pl.Float64})
    for item in rows_in:
        if not isinstance(item, dict):
            continue
        try:
            ts = int(item.get("date") or 0)
        except (TypeError, ValueError):
            continue
        if ts <= 0:
            continue
        circ = None
        circulating = item.get("circulating")
        if isinstance(circulating, dict):
            circ = circulating.get("peggedUSD")
        if circ is None and isinstance(item.get("totalCirculatingUSD"), dict):
            circ = item["totalCirculatingUSD"].get("peggedUSD")
        if circ is None:
            continue
        # DeFiLlama dates are unix seconds.
        day = _day_from_ms(ts * 1000 if ts < 10_000_000_000 else ts)
        rows.append({"event_time_ms": _midnight(day), "stbl_supply_usd": float(circ)})
    if not rows:
        return pl.DataFrame(schema={"event_time_ms": pl.Int64, "stbl_supply_usd": pl.Float64})
    df = pl.DataFrame(rows).sort("event_time_ms").unique(subset=["event_time_ms"], keep="last", maintain_order=True)
    df = df.with_columns(pl.col("stbl_supply_usd").diff().alias("stbl_flow_usd"))
    return df


def parse_deribit_dvol(payload: dict, asset: str) -> pl.DataFrame:
    """Deribit rows are [ts_ms, open, high, low, close]."""
    data = []
    if isinstance(payload, dict):
        data = payload.get("result", {}).get("data", []) or []
    rows = []
    sym = canonical_symbol(asset if asset.endswith("USDT") else asset + "USDT")
    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) < 5:
            continue
        try:
            ts = int(item[0])
            o, h, l, c = (float(item[1]), float(item[2]), float(item[3]), float(item[4]))
        except (TypeError, ValueError):
            continue
        day = _day_from_ms(ts)
        rows.append({
            "asset": sym,
            "event_time_ms": _midnight(day),
            "dv_dvol_open": o,
            "dv_dvol_high": h,
            "dv_dvol_low": l,
            "dv_dvol_close": c,
        })
    if not rows:
        return pl.DataFrame(schema={
            "asset": pl.Utf8, "event_time_ms": pl.Int64, "dv_dvol_close": pl.Float64,
        })
    return pl.DataFrame(rows).sort("event_time_ms").unique(subset=["asset", "event_time_ms"], keep="last", maintain_order=True)


def parse_wiki_items(items: list, asset: str) -> pl.DataFrame:
    rows = []
    sym = canonical_symbol(asset)
    for item in items or []:
        if not isinstance(item, dict):
            continue
        stamp = str(item.get("timestamp") or "")
        views = item.get("views")
        if len(stamp) < 8 or views is None:
            continue
        try:
            day = datetime.strptime(stamp[:8], "%Y%m%d").date()
            views_i = int(views)
        except ValueError:
            continue
        rows.append({"asset": sym, "event_time_ms": _midnight(day), "soc_wiki_views": views_i})
    if not rows:
        return pl.DataFrame(schema={"asset": pl.Utf8, "event_time_ms": pl.Int64, "soc_wiki_views": pl.Int64})
    return pl.DataFrame(rows).sort("event_time_ms").unique(subset=["event_time_ms"], keep="last", maintain_order=True)


def _write(df: pl.DataFrame, path: Path, schema_name: str, params: dict) -> None:
    """Merge ``df`` into the file at ``path`` keyed by (asset, event_time_ms); new rows win."""
    if df.is_empty():
        return
    key = ["event_time_ms"] if "asset" not in df.columns else ["asset", "event_time_ms"]
    if path.exists():
        try:
            old = pl.read_parquet(path)
        except Exception:  # noqa: BLE001  an unreadable file is replaced, not merged
            old = pl.DataFrame()
        if not old.is_empty() and all(k in old.columns for k in key):
            df = pl.concat([old, df], how="diagonal_relaxed")
    df = df.unique(subset=key, keep="last", maintain_order=True)
    atomic_write_parquet(df, path, schema_name=schema_name, params=params, sort_cols=key)


def external_dir(root: Path, source: str) -> Path:
    return root / "bronze" / "external" / source


def fetch_defillama(root: Path, *, timeout: int = 60) -> tuple[str, int]:
    url = "https://stablecoins.llama.fi/stablecoincharts/all"
    status, body, err = http_get(url, timeout=timeout)
    if status != 200 or err or not body:
        return err or f"HTTP {status}", 0
    try:
        df = parse_defillama_chart(json.loads(body.decode("utf-8", errors="replace")))
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}", 0
    path = external_dir(root, "defillama") / "stable_supply.parquet"
    _write(df, path, "external_defillama_v1", {"source_url": url})
    return "ok", df.height


DERIBIT_WINDOW_DAYS = 500


def fetch_deribit(root: Path, start: date, end: date, *, timeout: int = 60) -> tuple[str, int]:
    frames = []
    errors = []
    # The API caps the points per call, so the range is requested in windows.
    windows = []
    cur = start
    while cur <= end:
        stop = min(end, cur + timedelta(days=DERIBIT_WINDOW_DAYS - 1))
        windows.append((_midnight(cur), _midnight(stop + timedelta(days=1))))
        cur = stop + timedelta(days=1)
    for currency, asset in (("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")):
        for start_ms, end_ms in windows:
            url = (
                "https://www.deribit.com/api/v2/public/get_volatility_index_data"
                f"?currency={currency}&start_timestamp={start_ms}&end_timestamp={end_ms}&resolution=1D"
            )
            status, body, err = http_get(url, timeout=timeout)
            if status != 200 or err or not body:
                errors.append(f"{currency} {_day_from_ms(start_ms)}: {err or f'HTTP {status}'}")
                continue
            try:
                frames.append(parse_deribit_dvol(json.loads(body.decode("utf-8", errors="replace")), asset))
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
    frames = [f for f in frames if not f.is_empty()]
    if not frames:
        return "; ".join(errors) or "empty", 0
    df = pl.concat(frames, how="diagonal_relaxed")
    path = external_dir(root, "deribit") / "dvol.parquet"
    _write(df, path, "external_dvol_v1", {"source": "deribit"})
    return "ok", df.height


def fetch_farside(root: Path, *, timeout: int = 60) -> tuple[str, int]:
    frames = []
    errors = []
    for asset, url in ETF_URLS.items():
        status, body, err = http_get(url, timeout=timeout)
        if status != 200 or err or not body:
            errors.append(f"{asset}: {err or status}")
            continue
        df = parse_farside_html(body.decode("utf-8", errors="replace"), asset)
        if not df.is_empty():
            frames.append(df)
    if not frames:
        return "; ".join(errors) or "empty", 0
    df = pl.concat(frames, how="diagonal_relaxed")
    path = external_dir(root, "farside") / "etf_flows.parquet"
    _write(df, path, "external_etf_v1", {"source": "farside"})
    return "ok", df.height


def fetch_wikipedia(root: Path, assets: list[str], start: date, end: date, *, timeout: int = 60) -> tuple[str, int]:
    frames = []
    errors = []
    start_s = start.strftime("%Y%m%d")
    end_s = end.strftime("%Y%m%d")
    for asset in assets:
        titles = WIKI_ARTICLES.get(canonical_symbol(asset))
        if not titles:
            continue
        if isinstance(titles, str):
            titles = (titles,)
        last_err = None
        got = None
        for article in titles:
            url = (
                "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
                f"en.wikipedia/all-access/user/{article}/daily/{start_s}/{end_s}"
            )
            status, body, err = http_get(url, timeout=timeout)
            if status != 200 or err or not body:
                last_err = f"{asset}: {err or status} ({article})"
                continue
            try:
                items = json.loads(body.decode("utf-8", errors="replace")).get("items", [])
            except Exception as exc:
                last_err = f"{asset}: {exc} ({article})"
                continue
            df = parse_wiki_items(items, asset)
            if not df.is_empty():
                got = df
                break
            last_err = f"{asset}: empty ({article})"
        if got is not None:
            frames.append(got)
        elif last_err:
            errors.append(last_err)
    if not frames:
        return "; ".join(errors) or "empty", 0
    df = pl.concat(frames, how="diagonal_relaxed")
    path = external_dir(root, "wikipedia") / "pageviews.parquet"
    _write(df, path, "external_wiki_v1", {"source": "wikipedia"})
    return "ok", df.height


def load_external(root: Path, name: str) -> pl.DataFrame:
    directory = external_dir(root, name)
    if not directory.exists():
        return pl.DataFrame()
    files = sorted(directory.glob("*.parquet"))
    if not files:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(p) for p in files], how="diagonal_relaxed")


def fetch_all_external(root: Path, assets: list[str], start: date, end: date, *, timeout: int = 60) -> list[str]:
    """Best-effort. A down host is a recorded line, not an aborted rebuild."""
    notes = []
    for label, fn in (
        ("defillama", lambda: fetch_defillama(root, timeout=timeout)),
        ("deribit", lambda: fetch_deribit(root, start, end, timeout=timeout)),
        ("farside", lambda: fetch_farside(root, timeout=timeout)),
        ("wikipedia", lambda: fetch_wikipedia(root, assets, start, end, timeout=timeout)),
    ):
        try:
            status, n = fn()
        except Exception as exc:
            status, n = f"{type(exc).__name__}: {exc}", 0
        notes.append(f"{label}: {status} rows={n}")
        print(f"[external] {label}: {status} rows={n}")
    return notes
