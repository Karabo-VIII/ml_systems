"""External parsers, as-of lag, and the v2 gold reader."""

from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from pipeline.align import asof_join_knowable
from pipeline.constants import DAY_MS
from pipeline.dataset import load_gold
from pipeline.fetch_external import (
    parse_defillama_chart,
    parse_deribit_dvol,
    parse_farside_html,
    parse_wiki_items,
)
from pipeline.io import atomic_write_parquet


def test_farside_cleans_headers_and_parentheses():
    html = """
    <table><tr><th>Date</th><th>IBIT</th><th>ETHA 0.25%</th><th>Unnamed: 12</th><th>Total</th></tr>
    <tr><td>11 Jan 2024</td><td>100.5</td><td>(12.0)</td><td>9</td><td>88.5</td></tr>
    </table>
    """
    df = parse_farside_html(html, "BTC")
    assert "etf_unnamed:_12" not in df.columns
    assert "etf_etha" in df.columns
    assert df["etf_etha"][0] == -12.0
    assert df["etf_total_usdm"][0] == 88.5
    assert df["etf_ibit"][0] == 100.5


def test_defillama_empty_and_diff_are_safe():
    empty = parse_defillama_chart({})
    assert empty.is_empty()
    parsed = parse_defillama_chart([
        {"date": 1700000000, "totalCirculatingUSD": {"peggedUSD": 100.0}},
        {"date": 1700086400, "totalCirculatingUSD": {"peggedUSD": 110.0}},
    ])
    assert parsed.height == 2
    assert parsed["stbl_flow_usd"][1] == 10.0
    assert parsed["stbl_flow_usd"][0] is None


def test_deribit_and_wiki_parsers():
    dvol = parse_deribit_dvol({"result": {"data": [[1700000000000, 40, 45, 39, 42]]}}, "ETH")
    assert dvol["asset"][0] == "ETHUSDT"
    assert dvol["dv_dvol_close"][0] == 42
    wiki = parse_wiki_items([{"timestamp": "2024011100", "views": 1234}], "sol")
    assert wiki["soc_wiki_views"][0] == 1234
    assert wiki["asset"][0] == "SOLUSDT"


def test_external_day_is_not_visible_until_the_next_midnight():
    day = datetime(2024, 6, 1, tzinfo=timezone.utc)
    midnight = int(day.timestamp() * 1000)
    panel = pl.DataFrame({"event_time_ms": [midnight], "etf_total_usdm": [5.0]})
    bars = pl.DataFrame({
        "ts_ms": [midnight + 3_600_000, midnight + DAY_MS],
        "asset": ["BTCUSDT", "BTCUSDT"],
    })
    joined = asof_join_knowable(
        bars, panel, ["etf_total_usdm"], DAY_MS, feature_time_col="event_time_ms",
    )
    assert joined["etf_total_usdm"].to_list() == [None, 5.0]


def test_gold_loader_reads_the_latest_file(tmp_path):
    folder = tmp_path / "gold" / "time_1h"
    folder.mkdir(parents=True)
    early = folder / "BTCUSDT_20240101.parquet"
    late = folder / "BTCUSDT_20240602.parquet"
    frame = pl.DataFrame({
        "asset": ["BTCUSDT"], "bar_type": ["time_1h"], "ts_ms": [1], "close": [10.0],
    })
    atomic_write_parquet(frame, early, schema_name="dataset_v1", params={"v": 1}, sort_cols=["ts_ms"])
    atomic_write_parquet(
        frame.with_columns(pl.lit(11.0).alias("close")),
        late, schema_name="dataset_v1", params={"v": 2}, sort_cols=["ts_ms"],
    )
    loaded = load_gold("btc", "time_1h", root=tmp_path)
    assert loaded["close"][0] == 11.0


def test_gold_loader_columns_that_a_month_lacks_come_back_null(tmp_path):
    """A series that starts mid-history (Vision metrics, a perpetual listed after the spot pair)
    is absent from the earlier months.

    ``columns=`` returns what the full load would select: null where a month has no such column,
    in time order, without ``ts_ms`` unless it is asked for. A column that no month has raises,
    so a misspelt name is not read as an all-null feature.
    """
    folder = tmp_path / "gold" / "time_1d"
    folder.mkdir(parents=True)
    jan = pl.DataFrame({
        "asset": ["BTCUSDT"] * 2, "bar_type": ["time_1d"] * 2, "ts_ms": [1, 2], "close": [1.0, 2.0],
    })
    sep = jan.with_columns(pl.Series("ts_ms", [3, 4]), pl.Series("close", [3.0, 4.0]),
                           pl.Series("s3_oi_mean", [30.0, 40.0]))
    for name, frame in (("BTCUSDT_2020-01.parquet", jan), ("BTCUSDT_2020-09.parquet", sep)):
        atomic_write_parquet(frame, folder / name, schema_name="dataset_v1", params={"v": 1}, sort_cols=["ts_ms"])
    full = load_gold("btc", "time_1d", root=tmp_path)

    picked = load_gold("btc", "time_1d", root=tmp_path, columns=["ts_ms", "s3_oi_mean"])
    assert picked.equals(full.select(["ts_ms", "s3_oi_mean"]))
    assert picked["s3_oi_mean"].to_list() == [None, None, 30.0, 40.0]

    no_clock = load_gold("btc", "time_1d", root=tmp_path, columns=["s3_oi_mean", "close"])
    assert no_clock.equals(full.select(["s3_oi_mean", "close"]))

    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        load_gold("btc", "time_1d", root=tmp_path, columns=["ts_ms", "s3_oi_mena"])


def test_farside_mixed_ints_and_floats_across_many_rows():
    """Live Farside pages have integer-looking early rows and later floats (e.g. 18.0)."""
    from datetime import date, timedelta

    rows = ["<tr><th>Date</th><th>IBIT</th><th>FBTC</th><th>Total</th></tr>"]
    start = date(2024, 1, 11)
    for i in range(120):
        day = start + timedelta(days=i)
        # First 100 rows look like ints; later rows include a true float.
        ibit = str(i) if i < 110 else "18.0"
        fbtc = "-" if i < 5 else ("0" if i < 110 else "2.5")
        total = f"{i}.0"
        rows.append(f"<tr><td>{day.strftime('%d %b %Y')}</td><td>{ibit}</td><td>{fbtc}</td><td>{total}</td></tr>")
    df = parse_farside_html("<table>" + "".join(rows) + "</table>", "BTC")
    assert df.height == 120
    assert df.schema["etf_ibit"] == pl.Float64
    assert df.schema["etf_fbtc"] == pl.Float64
    assert df.schema["etf_total_usdm"] == pl.Float64
    assert df["etf_ibit"][-1] == 18.0
    assert df["etf_fbtc"][-1] == 2.5
    assert df["etf_fbtc"][0] is None


def test_farside_btc_and_eth_tables_concat():
    btc = parse_farside_html(
        "<table><tr><th>Date</th><th>IBIT</th><th>Total</th></tr>"
        "<tr><td>11 Jan 2024</td><td>10</td><td>10.5</td></tr></table>",
        "BTC",
    )
    eth = parse_farside_html(
        "<table><tr><th>Date</th><th>ETHA 0.25%</th><th>Total</th></tr>"
        "<tr><td>11 Jan 2024</td><td>18.0</td><td>18.0</td></tr></table>",
        "ETH",
    )
    out = pl.concat([btc, eth], how="diagonal_relaxed")
    assert out.height == 2
    assert set(out["asset"].to_list()) == {"BTCUSDT", "ETHUSDT"}
    assert out.schema["etf_ibit"] == pl.Float64
    assert out.schema["etf_etha"] == pl.Float64


def test_gold_loader_accepts_fx_pairs(tmp_path):
    from pipeline.dataset import load_gold, latest_gold_path
    from pipeline.symbols import canonical_instrument

    assert canonical_instrument("eur/usd") == "EURUSD"
    assert canonical_instrument("BTC") == "BTCUSDT"
    folder = tmp_path / "gold" / "time_1h"
    folder.mkdir(parents=True)
    path = folder / "EURUSD_20240115.parquet"
    frame = pl.DataFrame({
        "asset": ["EURUSD"], "bar_type": ["time_1h"], "ts_ms": [1], "close": [1.1],
    })
    atomic_write_parquet(frame, path, schema_name="dataset_v1", params={"v": 1}, sort_cols=["ts_ms"])
    assert latest_gold_path(tmp_path, "EUR/USD", "time_1h") == path
    loaded = load_gold("EURUSD", "time_1h", root=tmp_path)
    assert loaded["close"][0] == 1.1


def test_wikipedia_falls_back_to_earlier_title(monkeypatch, tmp_path):
    """Current Wikipedia titles 404 for dates before a page was moved (e.g. SOL in 2020)."""
    from datetime import date
    import pipeline.fetch_external as fe

    calls = []

    def fake_get(url, timeout=60):
        calls.append(url)
        if "Solana_(blockchain_platform)" in url:
            return 404, b"", "HTTP 404"
        if url.endswith("Solana/daily/20200811/20200813") or "user/Solana/daily/" in url:
            payload = {"items": [{"timestamp": "2020081100", "views": 42}]}
            import json
            return 200, json.dumps(payload).encode(), None
        return 404, b"", "HTTP 404"

    monkeypatch.setattr(fe, "http_get", fake_get)
    status, n = fe.fetch_wikipedia(tmp_path, ["SOLUSDT"], date(2020, 8, 11), date(2020, 8, 13))
    assert status == "ok"
    assert n == 1
    assert any("Solana_(blockchain_platform)" in u for u in calls)
    assert any("/user/Solana/daily/" in u for u in calls)
    df = fe.load_external(tmp_path, "wikipedia")
    assert df.height == 1
    assert int(df["soc_wiki_views"][0]) == 42
